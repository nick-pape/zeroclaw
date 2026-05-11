# v0.8.0 Excision Pass — Incident Log

Working audit trail for the v0.8.0 excision pass. Each entry records a deletion candidate where the decision wasn't pure-delete: the surrounding test was real, the call-site couldn't be reached safely, or the suppression turned out to mark live code.

Format: site, decision (deleted / kept / kept-with-narrow), reason.

## Phase 1 — Orphaned files

- `v3.toml` (785 lines, repo root) — **deleted**. Zero references in code, docs, tests, scripts, .gitignore, CI. Residue from the scrapped `zeroclaw config generate` (commit 73f906474).
- `release-notes-notes.md` (32 lines, repo root) — **deleted**. Scratch TODOs accidentally committed; bullets belong in the runbook PR.

## Phase 2 — `#[allow(dead_code)]` sweep

### Skipped (test impact would force test edits per Q1 rule)

- `crates/zeroclaw-tools/src/google_workspace.rs:28` `rate_limit_per_minute: u32` — kept. Field is dead in production but constructor is invoked by ~14 legitimate tests of other tool methods. Removing would force test signature edits with no test-semantic gain.
- `crates/zeroclaw-providers/src/azure_openai.rs:14,16` `resource_name`, `deployment_name` — kept. Constructor is called from many tests (4-arg `new()`); a couple of tests assert on these fields specifically (lines 576-577) but most just construct. Removing forces test edits across the file.
- `crates/zeroclaw-runtime/src/agent/agent.rs:49` `allowed_tools: Option<Vec<String>>` — kept. Identically-named field exists in `crates/zeroclaw-gateway/src/api.rs:82` and `crates/zeroclaw-runtime/src/cron/types.rs:152,190`; verifying full disconnection is a multi-crate trace, deferred.
- `crates/zeroclaw-runtime/src/security/audit.rs:225` `buffer: Mutex<Vec<AuditEvent>>` — kept. Buffered batch-flush could be a half-wired feature; flushing a Mutex<Vec> as part of audit chain is load-bearing in a way that needs a deeper trace before deletion.
- `src/service/mod.rs:6`, `src/integrations/mod.rs:7`, `src/hardware/mod.rs:8`, `src/skills/mod.rs:23` — kept. The `handle_command` dispatchers are wired only on certain feature combinations; the `#[allow(dead_code)]` is a wrong-shape suppression but converting to `#[cfg(feature = "X")]` requires per-crate feature audit.
- `crates/zeroclaw-tools/src/browser.rs:66,68,70,2006` — kept. Fields/fns gated to the `browser-native` feature; the suppression marks the cfg-off path, which is a legitimate (if ugly) pattern.
- `crates/zeroclaw-plugins/src/host.rs:25` `verification: VerificationResult` — kept. Plugin trust audit field; deletion needs a security/trust review.
- `tests/integration/channel_matrix.rs:19`, `crates/zeroclaw-runtime/src/agent/tests.rs:63`, `crates/zeroclaw-runtime/src/sop/engine.rs:1002` — kept. Test file / `#[cfg(test)]` block content; user directive: don't touch tests.
- `crates/zeroclaw-channels/src/lark.rs:179` `event_id`, `crates/zeroclaw-channels/src/bluesky.rs:48,62`, `crates/zeroclaw-channels/src/reddit.rs:45` — kept. Deserializer struct fields. Serde reads and discards them; deletion would force a separate "manually skip in serde" change.
- `crates/zeroclaw-providers/src/bedrock.rs:532,555,571` — kept. Same shape as the lark/bluesky case (response-deserialize fields).

### Deleted

(see commits `chore(excision): drop WIP stubs in tools + gateway` and following)

## Phase 3 — Stale comment refs (PR / issue / phase numbers)

(populated)

## Phase 4 — Stale `#[serde(alias)]`

(populated)

## Phase 5 — `channels_except_webhook` + `channels` hand-rolled lists

(populated)

## Phase 6 — FeishuConfig folded into LarkConfig

(populated)
