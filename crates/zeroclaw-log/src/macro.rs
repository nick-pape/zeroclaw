//! The `record!` macro — a thin syntactic wrapper around `tracing::event!`
//! that gives every emission site a uniform kwarg API.
//!
//! Architecture: `record!` has NO write path of its own. It expands to a
//! single `tracing::event!` call. The `LogCaptureLayer` (installed once
//! in the daemon's tracing-subscriber chain) is the SOLE consumer that
//! turns tracing events into [`crate::LogEvent`]s and writes them to:
//!
//!   1. JSONL persistence (when enabled via `[observability] log_persistence`)
//!   2. The process-wide broadcast Sender (dashboard SSE)
//!   3. The Observer bridge (Prometheus / OTel typed metrics)
//!
//! Field schema lives in exactly one place: `FieldCollector` in
//! [`crate::layer`]. The macro doesn't enforce a whitelist; unknown
//! fields land in the event's `attributes` map. Adding a new typed
//! field means: add to `LogEvent` struct + add a `FieldCollector` arm.
//! The macro stays untouched.
//!
//! Usage:
//!
//! ```ignore
//! use zeroclaw_log::record;
//!
//! record!(
//!     INFO,
//!     action: "llm_request",
//!     category: "agent",
//!     outcome: "success",
//!     agent: agent_alias,
//!     channel: "discord.clamps",
//!     model_provider: "anthropic.clamps",
//!     model: "claude-sonnet-4-6",
//!     trace_id: turn_id,
//!     duration_ms: 412_u64,
//!     message: "LLM request completed",
//! );
//! ```

/// Emit a structured ZeroClaw log event through `tracing::event!`. The
/// `LogCaptureLayer` picks it up and routes it to JSONL + broadcast +
/// Observer bridge.
#[macro_export]
macro_rules! record {
    ($level:ident, $($key:ident : $value:expr),+ $(,)?) => {{
        // `%` prefix forces tracing to render every value through
        // `Display`, which any `&str`, `String`, integer, `serde_json::Value`,
        // or `anyhow::Error` implements. That sidesteps `tracing::Value`'s
        // restrictive trait set and lets callers pass arbitrary expressions.
        $crate::tracing::event!(
            target: "zeroclaw_log_event",
            $crate::tracing::Level::$level,
            $($key = %($value)),+
        );
    }};
}

/// Wrap a future in a tracing span carrying alias-bound attribution.
/// Field-name set is shared with `record!` (see [`crate::event`]). The
/// trailing `=> <future>` becomes the body.
#[macro_export]
macro_rules! scope {
    ($($key:ident : $value:expr),+ $(,)? => $body:expr) => {{
        use $crate::tracing::Instrument;
        ($body).instrument($crate::tracing::info_span!(
            "zeroclaw_scope",
            $($key = %($value)),+
        ))
    }};
}

#[cfg(test)]
mod tests {
    // The macro emits a tracing event; verifying the full pipeline
    // (event → LogCaptureLayer → writer → JSONL) requires the layer to
    // be installed. The layer is exercised in `crate::layer::tests`
    // which sets up a subscriber and asserts on the writer's output.
    // This test just verifies the macro expands and compiles with the
    // expected kwarg shape.
    #[test]
    fn macro_compiles_with_full_kwarg_set() {
        let agent_alias = "clamps";
        record!(
            INFO,
            action: "test_event",
            category: "agent",
            outcome: "success",
            agent: agent_alias,
            channel: "discord.clamps",
            model_provider: "anthropic.clamps",
            model: "claude-sonnet-4-6",
            tool: "shell",
            session_key: "discord.clamps_user_abc",
            cron_job_id: "uuid-1",
            duration_ms: 412_u64,
            trace_id: "trace-abc",
            span_id: "span-1",
            message: "hello",
        );
    }

    #[test]
    fn macro_compiles_with_minimal_kwarg_set() {
        record!(WARN, action: "minimal", message: "tiny event");
    }
}
