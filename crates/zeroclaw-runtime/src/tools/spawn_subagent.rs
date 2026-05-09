//! `spawn_subagent` agent-loop tool (#6272 P10c).
//!
//! Lets a parent agent spawn an ephemeral SubAgent that inherits the
//! parent's identity, security policy, and memory allowlist, runs a
//! focused prompt, and returns the response. Cron's `JobType::Agent`
//! dispatch (P10b) is the other SubAgent spawn site; both funnel
//! through [`crate::subagent::SubAgentSpawn`] so permission
//! inheritance, tracing-span shape, and audit attribution stay uniform.
//!
//! v0.8.0 surface accepts only a `prompt`. The narrowing-override path
//! (sub-agents that drop privileges below the parent) is deferred to
//! v0.8.1 along with the `[agents.<alias>].subagent_*` config block;
//! the spawn validator already supports it via
//! [`crate::subagent::SubAgentOverrides`], so adding the surface later
//! is purely additive.

use crate::subagent::{SubAgentOverrides, SubAgentSpawn};
use anyhow::Result;
use async_trait::async_trait;
use serde_json::json;
use std::sync::Arc;
use tracing::Instrument;
use zeroclaw_api::tool::{Tool, ToolResult};
use zeroclaw_config::schema::Config;

/// Spawn an ephemeral SubAgent that inherits the parent agent's
/// identity and runs a focused prompt under the same alias.
pub struct SpawnSubagentTool {
    config: Arc<Config>,
    parent_alias: String,
}

impl SpawnSubagentTool {
    pub fn new(config: Arc<Config>, parent_alias: impl Into<String>) -> Self {
        Self {
            config,
            parent_alias: parent_alias.into(),
        }
    }
}

#[async_trait]
impl Tool for SpawnSubagentTool {
    fn name(&self) -> &str {
        "spawn_subagent"
    }

    fn description(&self) -> &str {
        "Spawn an ephemeral SubAgent that inherits this agent's identity, \
         security policy, and memory allowlist. The SubAgent runs the supplied \
         prompt to completion under the parent's permissions envelope and \
         returns its response. Use for focused subtasks (research lookup, \
         multi-step reasoning, etc.) that should not pollute this agent's main \
         conversation history. Cost-aware: each SubAgent run is a full agent \
         loop and consumes provider tokens."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The task or question for the SubAgent. Be specific and self-contained — the SubAgent does not see this conversation's history."
                }
            },
            "required": ["prompt"]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> Result<ToolResult> {
        let prompt = args
            .get("prompt")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| anyhow::anyhow!("Missing or empty 'prompt' parameter"))?
            .to_string();

        let subagent_ctx = match SubAgentSpawn::for_agent(&self.config, &self.parent_alias)
            .and_then(|spawn| spawn.build(SubAgentOverrides::default()))
        {
            Ok(ctx) => ctx,
            Err(e) => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!("subagent spawn failed: {e:#}")),
                });
            }
        };

        let run_id = uuid::Uuid::new_v4().to_string();
        let span = tracing::info_span!(
            "subagent",
            parent_alias = %subagent_ctx.agent_id,
            run_id = %run_id,
            spawn_site = "tool",
        );

        let temperature = self
            .config
            .providers
            .first_model_provider()
            .and_then(|e| e.temperature)
            .unwrap_or(0.7);
        let session_path = std::path::PathBuf::from(format!("subagent-{run_id}"));

        let run_result = Box::pin(
            crate::agent::run(
                (*self.config).clone(),
                &self.parent_alias,
                Some(prompt),
                None,
                None,
                temperature,
                vec![],
                false,
                Some(session_path),
                None,
            )
            .instrument(span),
        )
        .await;

        match run_result {
            Ok(response) => Ok(ToolResult {
                success: true,
                output: if response.trim().is_empty() {
                    "subagent completed without output".to_string()
                } else {
                    response
                },
                error: None,
            }),
            Err(e) => Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("subagent run failed: {e}")),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zeroclaw_config::schema::{AliasedAgentConfig, Config, RiskProfileConfig};

    fn config_with_agent(alias: &str) -> Config {
        let mut config = Config::default();
        config
            .risk_profiles
            .insert("default".to_string(), RiskProfileConfig::default());
        config.agents.insert(
            alias.to_string(),
            AliasedAgentConfig {
                risk_profile: "default".to_string(),
                ..AliasedAgentConfig::default()
            },
        );
        config
    }

    #[test]
    fn tool_name_and_schema_are_well_formed() {
        let tool = SpawnSubagentTool::new(Arc::new(config_with_agent("alpha")), "alpha");
        assert_eq!(tool.name(), "spawn_subagent");
        let schema = tool.parameters_schema();
        assert!(schema["properties"]["prompt"].is_object());
        assert_eq!(schema["required"][0], "prompt");
    }

    #[tokio::test]
    async fn missing_prompt_is_rejected() {
        let tool = SpawnSubagentTool::new(Arc::new(config_with_agent("alpha")), "alpha");
        let err = tool
            .execute(json!({}))
            .await
            .expect_err("missing prompt must fail");
        assert!(err.to_string().contains("prompt"));
    }

    #[tokio::test]
    async fn empty_prompt_is_rejected() {
        let tool = SpawnSubagentTool::new(Arc::new(config_with_agent("alpha")), "alpha");
        let err = tool
            .execute(json!({ "prompt": "   " }))
            .await
            .expect_err("empty prompt must fail");
        assert!(err.to_string().contains("prompt"));
    }

    #[tokio::test]
    async fn unknown_parent_alias_surfaces_spawn_failure() {
        // Parent alias that is not configured: SubAgentSpawn::for_agent
        // returns Err, the tool reports a structured spawn failure
        // (no panic, no recursion attempt).
        let tool = SpawnSubagentTool::new(Arc::new(Config::default()), "missing-alpha");
        let result = tool
            .execute(json!({ "prompt": "hello" }))
            .await
            .expect("execute returns Ok with structured failure");
        assert!(!result.success);
        assert!(
            result
                .error
                .as_deref()
                .unwrap_or_default()
                .contains("subagent spawn failed"),
            "expected spawn-failure error, got: {:?}",
            result.error
        );
    }
}
