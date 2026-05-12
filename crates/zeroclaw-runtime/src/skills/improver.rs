// Skill self-improvement: atomic writer + history-scanning helpers for the
// background review fork (see `agent::loop_` post-turn hook + `tools::skill_manage`).
//
// This module owns:
// - `SkillImprover` — atomic temp+validate+rename for SKILL.toml plus cooldown
//   tracking (in-memory and durable on-disk via the `updated_at` field).
// - `extract_skill_executions_from_history` / `looks_like_failure` — surface a
//   list of failed skill slugs from history that the review prompt can pass
//   along as a hint ("these skills failed this run"), without those failures
//   *gating* whether the fork runs.

use anyhow::{Context, Result, bail};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;
use zeroclaw_config::schema::SkillImprovementConfig;
use zeroclaw_providers::ChatMessage;

/// Manages skill self-improvement with cooldown tracking.
pub struct SkillImprover {
    workspace_dir: PathBuf,
    config: SkillImprovementConfig,
    cooldowns: HashMap<String, Instant>,
}

impl SkillImprover {
    pub fn new(workspace_dir: PathBuf, config: SkillImprovementConfig) -> Self {
        Self {
            workspace_dir,
            config,
            cooldowns: HashMap::new(),
        }
    }

    /// Check whether a skill is eligible for improvement (enabled + cooldown expired).
    ///
    /// Combines an in-memory cooldown (fast path, per-process) with a durable
    /// on-disk cooldown (`updated_at` field in `SKILL.toml`) so cooldowns survive
    /// process restarts.
    pub fn should_improve_skill(&self, slug: &str) -> bool {
        if !self.config.enabled {
            return false;
        }
        if let Some(last) = self.cooldowns.get(slug) {
            let elapsed = Instant::now().saturating_duration_since(*last);
            if elapsed.as_secs() < self.config.cooldown_secs {
                return false;
            }
        }
        if self.is_on_disk_cooldown(slug) {
            return false;
        }
        true
    }

    // SKILL.toml `updated_at` is bumped on every successful improvement, so its
    // age is a durable proxy for "improved recently."
    fn is_on_disk_cooldown(&self, slug: &str) -> bool {
        let toml_path = self.skills_dir().join(slug).join("SKILL.toml");
        let Ok(content) = std::fs::read_to_string(&toml_path) else {
            return false;
        };
        let Ok(parsed) = content.parse::<toml::Table>() else {
            return false;
        };
        let Some(updated_at) = parsed
            .get("skill")
            .and_then(|v| v.as_table())
            .and_then(|t| t.get("updated_at"))
            .and_then(|v| v.as_str())
        else {
            return false;
        };
        let Ok(ts) = chrono::DateTime::parse_from_rfc3339(updated_at) else {
            return false;
        };
        let elapsed = chrono::Utc::now().signed_duration_since(ts);
        elapsed.num_seconds() < self.config.cooldown_secs as i64
    }

    /// Improve an existing skill file atomically.
    ///
    /// Writes to a temp file first, validates, then renames over the original.
    /// Returns `Ok(Some(slug))` if the skill was improved.
    ///
    /// **Caller-gated:** this does NOT check `should_improve_skill` — callers
    /// must check that themselves before invoking, so they can also skip the
    /// (expensive) LLM call that produces `improved_content`.
    pub async fn improve_skill(
        &mut self,
        slug: &str,
        improved_content: &str,
        improvement_reason: &str,
    ) -> Result<Option<String>> {
        // Validate the improved content before writing.
        validate_skill_content(improved_content)?;

        let skill_dir = self.skills_dir().join(slug);
        let toml_path = skill_dir.join("SKILL.toml");

        if !toml_path.exists() {
            bail!("Skill file not found: {}", toml_path.display());
        }

        // Read existing content to preserve audit trail.
        let existing = tokio::fs::read_to_string(&toml_path)
            .await
            .with_context(|| format!("Failed to read {}", toml_path.display()))?;

        // Build the updated content with audit metadata appended.
        let now = chrono::Utc::now().to_rfc3339();
        let audit_entry = format!(
            "\n# Improvement: {now}\n# Reason: {}\n",
            improvement_reason.replace('\n', " ")
        );

        let updated = append_improvement_metadata(improved_content, &now, improvement_reason);

        // Preserve any existing audit trail from the original file.
        let audit_trail = extract_audit_trail(&existing);
        let final_content = if audit_trail.is_empty() {
            format!("{updated}{audit_entry}")
        } else {
            format!("{updated}\n{audit_trail}{audit_entry}")
        };

        // Atomic write: temp file → validate → rename.
        let temp_path = skill_dir.join(".SKILL.toml.tmp");
        tokio::fs::write(&temp_path, final_content.as_bytes())
            .await
            .with_context(|| format!("Failed to write temp file: {}", temp_path.display()))?;

        // Validate the temp file is readable and valid.
        let written = tokio::fs::read_to_string(&temp_path).await?;
        if let Err(e) = validate_skill_content(&written) {
            // Clean up temp file and abort.
            let _ = tokio::fs::remove_file(&temp_path).await;
            bail!("Validation failed after write: {e}");
        }

        // Rename atomically (same filesystem).
        tokio::fs::rename(&temp_path, &toml_path)
            .await
            .with_context(|| {
                format!(
                    "Failed to rename {} to {}",
                    temp_path.display(),
                    toml_path.display()
                )
            })?;

        // Record cooldown.
        self.cooldowns.insert(slug.to_string(), Instant::now());

        Ok(Some(slug.to_string()))
    }

    fn skills_dir(&self) -> PathBuf {
        self.workspace_dir.join("skills")
    }
}

/// Heuristic: does tool-result content look like a failure?
///
/// Catches the common shapes — explicit error/failure strings, panics,
/// exceptions, "not found", and shell exit-code lines.
fn looks_like_failure(content: &str) -> bool {
    let lower = content.to_lowercase();
    lower.contains("error")
        || lower.contains("failed")
        || lower.contains("panic")
        || lower.contains("exception")
        || lower.contains("not found")
        || lower.starts_with("exit code")
}

/// Extract skill tool executions from conversation history.
///
/// Returns `(skill_slug, succeeded)` pairs, one per dotted tool-result found.
/// Handles two emission formats:
/// - XML: `<tool_result name="slug.tool">…content…</tool_result>` (prompt-guided
///   tool-calling)
/// - Native: a `tool`-role message preceded by an `assistant` message whose
///   content embeds a JSON tool-call with a dotted `"name": "slug.tool"`
///
/// Deduplicates on `(slug, succeeded)` so the same skill can appear twice if
/// it both succeeded and failed within the same window.
pub fn extract_skill_executions_from_history(history: &[ChatMessage]) -> Vec<(String, bool)> {
    let mut results: Vec<(String, bool)> = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for (i, msg) in history.iter().enumerate() {
        let content = &msg.content;

        // Format 1: XML <tool_result name="slug.tool">…</tool_result>.
        let open_marker = "<tool_result name=\"";
        let close_marker = "</tool_result>";
        let mut pos = 0;
        while pos < content.len() {
            let Some(start) = content[pos..].find(open_marker) else {
                break;
            };
            let abs = pos + start + open_marker.len();
            let Some(end) = content[abs..].find('"') else {
                break;
            };
            let name = &content[abs..abs + end];
            if let Some(dot_pos) = name.find('.') {
                let slug = name[..dot_pos].to_string();
                let after_tag = abs + end + 1;
                let body_start = content[after_tag..].find('>').map(|p| after_tag + p + 1);
                let body_end = content[after_tag..].find(close_marker);
                let body = match (body_start, body_end) {
                    (Some(s), Some(e)) if s <= after_tag + e => &content[s..after_tag + e],
                    _ => "",
                };
                let succeeded = !looks_like_failure(body);
                let key = (slug.clone(), succeeded);
                if seen.insert(key) {
                    results.push((slug, succeeded));
                }
            }
            pos = abs + end + 1;
        }

        // Format 2: native tool-role message preceded by an assistant message
        // whose JSON tool-call carries a dotted `"name": "slug.tool"`.
        if msg.role == "tool" && i > 0 {
            let prev = &history[i - 1];
            if prev.role == "assistant" {
                let prev_content = &prev.content;
                let name_marker = "\"name\"";
                let mut pos = 0;
                while pos < prev_content.len() {
                    let Some(start) = prev_content[pos..].find(name_marker) else {
                        break;
                    };
                    let after = pos + start + name_marker.len();
                    let rest = prev_content[after..].trim_start();
                    let Some(rest) = rest.strip_prefix(':') else {
                        pos = after + 1;
                        continue;
                    };
                    let rest = rest.trim_start();
                    let Some(rest) = rest.strip_prefix('"') else {
                        pos = after + 1;
                        continue;
                    };
                    let Some(end) = rest.find('"') else {
                        break;
                    };
                    let name = &rest[..end];
                    if let Some(dot_pos) = name.find('.') {
                        let slug = name[..dot_pos].to_string();
                        let succeeded = !looks_like_failure(content);
                        let key = (slug.clone(), succeeded);
                        if seen.insert(key) {
                            results.push((slug, succeeded));
                        }
                    }
                    // Advance past this name occurrence.
                    let consumed = prev_content.len() - rest.len() + end + 1;
                    pos = consumed;
                }
            }
        }
    }

    results
}

/// Unique skill slugs seen in `history`, regardless of success/failure.
pub fn extract_skill_slugs_from_history(history: &[ChatMessage]) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    extract_skill_executions_from_history(history)
        .into_iter()
        .filter_map(|(slug, _)| {
            if seen.insert(slug.clone()) {
                Some(slug)
            } else {
                None
            }
        })
        .collect()
}

/// Validate skill content: must be non-empty, valid UTF-8 (already a &str),
/// and contain parseable TOML front-matter with a `[skill]` section.
pub fn validate_skill_content(content: &str) -> Result<()> {
    if content.trim().is_empty() {
        bail!("Skill content is empty");
    }

    // Must contain a [skill] section.
    #[derive(serde::Deserialize)]
    struct Partial {
        skill: PartialSkill,
    }
    #[derive(serde::Deserialize)]
    struct PartialSkill {
        name: Option<String>,
    }

    // Try parsing as TOML. Strip trailing comment lines that aren't valid TOML.
    let toml_portion = strip_trailing_comments(content);
    let parsed: Partial = toml::from_str(&toml_portion)
        .with_context(|| "Skill content contains malformed TOML front-matter")?;

    if parsed.skill.name.as_deref().unwrap_or("").is_empty() {
        bail!("Skill TOML missing required 'name' field");
    }

    Ok(())
}

/// Append updated_at and improvement_reason to the [skill] section's front-matter.
fn append_improvement_metadata(content: &str, timestamp: &str, reason: &str) -> String {
    // Find the end of the [skill] section (before the first [[tools]] or end of file).
    let tools_pos = content.find("[[tools]]");
    let (skill_section, rest) = match tools_pos {
        Some(pos) => (&content[..pos], &content[pos..]),
        None => (content, ""),
    };

    // Strip any existing `updated_at` / `improvement_reason` keys to avoid
    // emitting them twice (TOML rejects duplicate keys, so leaving the old
    // lines in place would break parsing on the next read).
    let skill_section = {
        let mut lines: Vec<&str> = skill_section.lines().collect();
        lines.retain(|line| {
            let trimmed = line.trim_start();
            !trimmed.starts_with("updated_at") && !trimmed.starts_with("improvement_reason")
        });
        lines.join("\n") + "\n"
    };

    let escaped_reason = reason.replace('"', "\\\"").replace('\n', " ");
    format!(
        "{skill_section}updated_at = \"{timestamp}\"\nimprovement_reason = \"{escaped_reason}\"\n{rest}"
    )
}

/// Extract existing audit trail comments (lines starting with `# Improvement:` or `# Reason:`).
fn extract_audit_trail(content: &str) -> String {
    content
        .lines()
        .filter(|line| {
            let trimmed = line.trim();
            trimmed.starts_with("# Improvement:") || trimmed.starts_with("# Reason:")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Strip trailing comment-only lines that would break TOML parsing.
fn strip_trailing_comments(content: &str) -> String {
    let lines: Vec<&str> = content.lines().collect();
    let mut end = lines.len();
    while end > 0 {
        let line = lines[end - 1].trim();
        if line.is_empty() || line.starts_with('#') {
            end -= 1;
        } else {
            break;
        }
    }
    lines[..end].join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Validation ──────────────────────────────────────────

    #[test]
    fn validate_empty_content_rejected() {
        assert!(validate_skill_content("").is_err());
        assert!(validate_skill_content("   \n  ").is_err());
    }

    #[test]
    fn validate_malformed_toml_rejected() {
        assert!(validate_skill_content("not valid toml {{").is_err());
    }

    #[test]
    fn validate_missing_name_rejected() {
        let content = r#"
[skill]
description = "no name field"
version = "0.1.0"
"#;
        assert!(validate_skill_content(content).is_err());
    }

    #[test]
    fn validate_valid_content_accepted() {
        let content = r#"
[skill]
name = "test-skill"
description = "A test skill"
version = "0.1.0"
"#;
        assert!(validate_skill_content(content).is_ok());
    }

    // ── Cooldown enforcement ────────────────────────────────

    #[test]
    fn cooldown_allows_first_improvement() {
        let improver = SkillImprover::new(
            PathBuf::from("/tmp/test"),
            SkillImprovementConfig {
                enabled: true,
                cooldown_secs: 3600,
            },
        );
        assert!(improver.should_improve_skill("test-skill"));
    }

    #[test]
    fn cooldown_blocks_recent_improvement() {
        let mut improver = SkillImprover::new(
            PathBuf::from("/tmp/test"),
            SkillImprovementConfig {
                enabled: true,
                cooldown_secs: 3600,
            },
        );
        improver
            .cooldowns
            .insert("test-skill".to_string(), Instant::now());
        assert!(!improver.should_improve_skill("test-skill"));
    }

    #[test]
    fn cooldown_disabled_blocks_all() {
        let improver = SkillImprover::new(
            PathBuf::from("/tmp/test"),
            SkillImprovementConfig {
                enabled: false,
                cooldown_secs: 0,
            },
        );
        assert!(!improver.should_improve_skill("test-skill"));
    }

    // ── Atomic write ────────────────────────────────────────

    #[tokio::test]
    async fn improve_skill_atomic_write() {
        let dir = tempfile::tempdir().unwrap();
        let skill_dir = dir.path().join("skills").join("test-skill");
        tokio::fs::create_dir_all(&skill_dir).await.unwrap();

        let original = r#"[skill]
name = "test-skill"
description = "Original description"
version = "0.1.0"
author = "zeroclaw-auto"
tags = ["auto-generated"]
"#;
        tokio::fs::write(skill_dir.join("SKILL.toml"), original)
            .await
            .unwrap();

        let mut improver = SkillImprover::new(
            dir.path().to_path_buf(),
            SkillImprovementConfig {
                enabled: true,
                cooldown_secs: 0,
            },
        );

        let improved = r#"[skill]
name = "test-skill"
description = "Improved description with better steps"
version = "0.1.1"
author = "zeroclaw-auto"
tags = ["auto-generated", "improved"]
"#;

        let result = improver
            .improve_skill("test-skill", improved, "Added better step descriptions")
            .await
            .unwrap();
        assert_eq!(result, Some("test-skill".to_string()));

        // Verify the file was updated.
        let content = tokio::fs::read_to_string(skill_dir.join("SKILL.toml"))
            .await
            .unwrap();
        assert!(content.contains("Improved description"));
        assert!(content.contains("updated_at"));
        assert!(content.contains("improvement_reason"));

        // Verify temp file was cleaned up.
        assert!(!skill_dir.join(".SKILL.toml.tmp").exists());
    }

    #[tokio::test]
    async fn improve_skill_invalid_content_aborts() {
        let dir = tempfile::tempdir().unwrap();
        let skill_dir = dir.path().join("skills").join("test-skill");
        tokio::fs::create_dir_all(&skill_dir).await.unwrap();

        let original = r#"[skill]
name = "test-skill"
description = "Original"
version = "0.1.0"
"#;
        tokio::fs::write(skill_dir.join("SKILL.toml"), original)
            .await
            .unwrap();

        let mut improver = SkillImprover::new(
            dir.path().to_path_buf(),
            SkillImprovementConfig {
                enabled: true,
                cooldown_secs: 0,
            },
        );

        // Empty content should fail validation.
        let result = improver
            .improve_skill("test-skill", "", "bad improvement")
            .await;
        assert!(result.is_err());

        // Original file should be untouched.
        let content = tokio::fs::read_to_string(skill_dir.join("SKILL.toml"))
            .await
            .unwrap();
        assert!(content.contains("Original"));
    }

    #[tokio::test]
    async fn improve_skill_writes_when_cooldown_not_checked_by_caller() {
        // `improve_skill` is caller-gated: it writes whenever given valid
        // content, even if `should_improve_skill` would return false. This
        // mirrors how the agent loop must check cooldown itself before
        // paying for the LLM call.
        let dir = tempfile::tempdir().unwrap();
        let skill_dir = dir.path().join("skills").join("test-skill");
        tokio::fs::create_dir_all(&skill_dir).await.unwrap();
        tokio::fs::write(
            skill_dir.join("SKILL.toml"),
            "[skill]\nname = \"test-skill\"\n",
        )
        .await
        .unwrap();

        let mut improver = SkillImprover::new(
            dir.path().to_path_buf(),
            SkillImprovementConfig {
                enabled: true,
                cooldown_secs: 9999,
            },
        );
        improver
            .cooldowns
            .insert("test-skill".to_string(), Instant::now());

        let result = improver
            .improve_skill(
                "test-skill",
                "[skill]\nname = \"test-skill\"\ndescription = \"better\"\n",
                "test",
            )
            .await
            .unwrap();
        assert_eq!(result, Some("test-skill".to_string()));
    }

    // ── Metadata appending ──────────────────────────────────

    #[test]
    fn append_metadata_adds_fields() {
        let content = r#"[skill]
name = "test"
description = "A skill"
version = "0.1.0"
"#;
        let result = append_improvement_metadata(content, "2026-01-01T00:00:00Z", "Better steps");
        assert!(result.contains("updated_at = \"2026-01-01T00:00:00Z\""));
        assert!(result.contains("improvement_reason = \"Better steps\""));
    }

    #[test]
    fn append_metadata_preserves_tools() {
        let content = r#"[skill]
name = "test"
description = "A skill"
version = "0.1.0"

[[tools]]
name = "action"
kind = "shell"
command = "echo hello"
"#;
        let result = append_improvement_metadata(content, "2026-01-01T00:00:00Z", "Improved");
        assert!(result.contains("[[tools]]"));
        assert!(result.contains("echo hello"));
    }

    // ── Audit trail extraction ──────────────────────────────

    #[test]
    fn extract_audit_trail_from_content() {
        let content = r#"[skill]
name = "test"
# Improvement: 2026-01-01T00:00:00Z
# Reason: First improvement
# Improvement: 2026-02-01T00:00:00Z
# Reason: Second improvement
"#;
        let trail = extract_audit_trail(content);
        assert!(trail.contains("First improvement"));
        assert!(trail.contains("Second improvement"));
        assert_eq!(trail.lines().count(), 4);
    }

    #[test]
    fn extract_audit_trail_empty_when_none() {
        let content = "[skill]\nname = \"test\"\n";
        let trail = extract_audit_trail(content);
        assert!(trail.is_empty());
    }

    // ── Duplicate-key handling on repeat improvements ───────

    #[test]
    fn append_metadata_replaces_existing_improvement_reason() {
        // A previously-improved skill carries both `updated_at` and
        // `improvement_reason`. Appending again must strip both before
        // emitting new values so the resulting TOML stays valid.
        let already_improved = r#"[skill]
name = "test"
description = "A skill"
version = "0.1.0"
updated_at = "2025-12-01T00:00:00Z"
improvement_reason = "first pass"
"#;
        let result = append_improvement_metadata(
            already_improved,
            "2026-01-01T00:00:00Z",
            "second pass",
        );
        let new_section = result.split("[[tools]]").next().unwrap_or(&result);
        assert_eq!(new_section.matches("updated_at").count(), 1);
        assert_eq!(new_section.matches("improvement_reason").count(), 1);
        assert!(new_section.contains("2026-01-01T00:00:00Z"));
        assert!(new_section.contains("second pass"));
        assert!(!new_section.contains("first pass"));
        // The rewritten section must still parse as TOML.
        let parsed: Result<toml::Table, _> = new_section.parse();
        assert!(parsed.is_ok(), "rewritten section should be valid TOML");
    }

    // ── Failure heuristic ───────────────────────────────────

    #[test]
    fn looks_like_failure_detects_common_shapes() {
        assert!(looks_like_failure("Error: file not found"));
        assert!(looks_like_failure("Command failed with status 1"));
        assert!(looks_like_failure("thread 'main' panicked at ..."));
        assert!(looks_like_failure("Exception in user code"));
        assert!(looks_like_failure("not found"));
        assert!(looks_like_failure("exit code 137"));
    }

    #[test]
    fn looks_like_failure_passes_clean_output() {
        assert!(!looks_like_failure("Done. Wrote 12 lines."));
        assert!(!looks_like_failure("ok"));
        assert!(!looks_like_failure(""));
    }

    // ── History extraction ──────────────────────────────────

    #[test]
    fn extract_executions_xml_marks_failure() {
        let history = vec![
            ChatMessage::user("run my-skill"),
            ChatMessage::assistant(
                "<tool_result name=\"my-skill.run\">Error: command not found</tool_result>",
            ),
        ];
        let executions = extract_skill_executions_from_history(&history);
        assert_eq!(executions, vec![("my-skill".to_string(), false)]);
    }

    #[test]
    fn extract_executions_xml_marks_success() {
        let history = vec![
            ChatMessage::user("run my-skill"),
            ChatMessage::assistant(
                "<tool_result name=\"my-skill.run\">Done. Wrote 3 files.</tool_result>",
            ),
        ];
        let executions = extract_skill_executions_from_history(&history);
        assert_eq!(executions, vec![("my-skill".to_string(), true)]);
    }

    #[test]
    fn extract_executions_native_format() {
        let history = vec![
            ChatMessage::user("run it"),
            ChatMessage::assistant(
                "{\"tool_calls\": [{\"name\": \"deploy.run\", \"args\": {}}]}",
            ),
            ChatMessage {
                role: "tool".into(),
                content: "Error: connection refused".into(),
            },
        ];
        let executions = extract_skill_executions_from_history(&history);
        assert_eq!(executions, vec![("deploy".to_string(), false)]);
    }

    #[test]
    fn extract_slugs_dedupes() {
        let history = vec![
            ChatMessage::user("run my-skill"),
            ChatMessage::assistant(
                "<tool_result name=\"my-skill.run\">ok</tool_result>\
                 <tool_result name=\"my-skill.run\">Error</tool_result>",
            ),
        ];
        let slugs = extract_skill_slugs_from_history(&history);
        assert_eq!(slugs, vec!["my-skill".to_string()]);
    }

    // ── On-disk cooldown ────────────────────────────────────

    #[tokio::test]
    async fn should_improve_blocks_when_updated_at_recent() {
        let dir = tempfile::tempdir().unwrap();
        let skill_dir = dir.path().join("skills").join("test-skill");
        tokio::fs::create_dir_all(&skill_dir).await.unwrap();
        let recent = chrono::Utc::now().to_rfc3339();
        tokio::fs::write(
            skill_dir.join("SKILL.toml"),
            format!("[skill]\nname = \"test-skill\"\nupdated_at = \"{recent}\"\n"),
        )
        .await
        .unwrap();

        let improver = SkillImprover::new(
            dir.path().to_path_buf(),
            SkillImprovementConfig {
                enabled: true,
                cooldown_secs: 9999,
            },
        );
        assert!(!improver.should_improve_skill("test-skill"));
    }

    #[tokio::test]
    async fn should_improve_allows_when_updated_at_stale() {
        let dir = tempfile::tempdir().unwrap();
        let skill_dir = dir.path().join("skills").join("test-skill");
        tokio::fs::create_dir_all(&skill_dir).await.unwrap();
        let stale = (chrono::Utc::now() - chrono::Duration::seconds(10_000)).to_rfc3339();
        tokio::fs::write(
            skill_dir.join("SKILL.toml"),
            format!("[skill]\nname = \"test-skill\"\nupdated_at = \"{stale}\"\n"),
        )
        .await
        .unwrap();

        let improver = SkillImprover::new(
            dir.path().to_path_buf(),
            SkillImprovementConfig {
                enabled: true,
                cooldown_secs: 3600,
            },
        );
        assert!(improver.should_improve_skill("test-skill"));
    }
}
