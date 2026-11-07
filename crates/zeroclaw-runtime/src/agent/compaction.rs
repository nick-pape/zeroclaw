//! LLM context compaction: summarise the oldest whole turns into ONE
//! synthetic summary message, keep the newest whole turns verbatim.
//!
//! The provider-facing shape after a compaction is exactly
//!
//! ```text
//! [leading system prompt(s)] + [ONE summary message] + [newest whole turns, verbatim]
//! ```
//!
//! The summary occupies the same synthetic slot the trim breadcrumb uses (the
//! first message after the leading system block), so every existing trim
//! path keeps treating it as owner-tracked bookkeeping: it is never counted
//! as a turn boundary and never dropped by the whole-turn trim fallback. The
//! slot's provenance is carried by two owner flags, never inferred from
//! message text:
//!
//! - `crumb_present` (existing): the slot is occupied by a synthetic message;
//! - `summary_present` (new, [`LeadProvenance::summary_present`]): the
//!   occupant is a compaction summary rather than the plain breadcrumb.
//!
//! A later compaction folds the slot's content into the new summary
//! (summary-of-summary + newer turns). Even when a backend could not record
//! `summary_present` (the slot is then classified as a breadcrumb), the
//! slot's content is still handed to the summariser, so nothing a previous
//! summary carried is silently lost.
//!
//! Token accounting uses the runtime's calibrated estimate: the ~4
//! chars/token history estimator scaled by the provider-reported calibration
//! ratio of the route (`reported_input_tokens / estimated_tokens`), the same
//! machinery the whole-turn trim uses.
//!
//! The summary is never written to the memory backend. A successful
//! compaction calls [`on_compaction_summary`], an intentionally inert seam for
//! future memory extraction.
//!
//! TODO(idle-compaction): compact a thread proactively after it has been idle
//! for a while (so the next inbound message never waits on a summary call).
//! Not implemented; the triggers are the budget seams in `turn/mod.rs` and the
//! manual `/compact` channel command.

use std::fmt::Write as _;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use crate::agent::history::estimate_history_tokens;
use zeroclaw_config::scattered_types::ContextCompressionConfig;
use zeroclaw_config::schema::Config;
use zeroclaw_providers::{ChatMessage, ModelProvider};

/// Built-in instructions for the summary call (its system prompt).
/// Overridable with `context_compression.prompt`.
pub const DEFAULT_COMPACTION_PROMPT: &str = "\
You are compacting the conversation history of an AI assistant so that the \
assistant can keep working after its oldest turns are removed from its \
context window. You are writing a summary that the SAME assistant will read \
as its own memory of those turns, so address it to yourself (\"the user asked \
me...\", \"I ran...\", \"I found...\").

Write a VERY DETAILED summary. Do NOT be concise: completeness and precision \
matter far more than brevity. You may use up to roughly 16,000 tokens. There \
is no limit on the number of bullets or sections. Anything you leave out is \
lost forever.

Preserve exactly, character for character: identifiers, IDs, names, \
amounts, prices, quantities, dates, times, URLs, file paths, commands, error \
messages, and anything the user quoted. Never round numbers or paraphrase an \
identifier.

Tool results: record which tool I called, with its arguments, and what it \
returned in substance (totals, counts, the records the user asked about or \
that open work depends on, and every identifier I may need again). Do NOT \
copy long raw listings (hundreds of rows, whole JSON payloads) into the \
summary: say what was fetched and note that the same call can fetch it again.

The headings in the material you are given (\"PREVIOUS SUMMARY\", \"SUMMARY \
SO FAR\", \"CONVERSATION TURNS TO SUMMARISE\", \"NOTE\") are labels for you \
only: never repeat them in your output.

Organise the summary in these sections, in this order:

1. What the user asked for and why: every request and goal, in the user's \
own words where the wording matters.
2. Decisions, preferences and corrections the user gave: every decision, \
stated preference, constraint and correction, including who said it when it \
matters.
3. Actions taken and their results: every action I (the assistant) took — \
tool calls, searches, lookups, changes made, messages sent — each with its \
concrete result, keeping all identifiers, amounts, dates, IDs, names and \
URLs exactly.
4. Errors hit and how they were resolved: each error, its cause if known, \
and the fix or workaround (or that it is still unresolved).
5. Current state of each piece of work: what is done, what is in progress, \
what is blocked, and the exact state of any artefact (cart, order, booking, \
document, ticket, etc.).
6. Open questions and next steps: what I still have to do, what I promised, \
and what I am waiting on from the user or from a system.
7. Things the user said NOT to do again: every prohibition, complaint or \
\"don't do X\" instruction, verbatim where possible.

If the material contains a previous summary, fold ALL of its content into \
the new summary (it describes even older turns) and then add the newer \
material; never drop facts from the previous summary just because they are \
old. If a section has nothing to report, write \"None.\" for it. Output only \
the summary itself, with no preamble and no closing remarks.";

/// Prepended to the summariser material when a previous summary is carried
/// forward verbatim by code.
const CARRIED_SUMMARY_NOTE: &str = "NOTE: a summary of even older turns already exists and is \
kept word for word separately. Summarise ONLY the conversation turns below; do not restate or \
reconstruct the older summary.\n\n";

/// Separates the carried summary from the summary of the turns after it.
const LATER_TURNS_HEADING: &str = "LATER TURNS (summarised after everything above):";

/// Closes the summary inside the synthetic slot. Loading a stored history
/// merges consecutive user rows (`normalize_cached_channel_turns`), so the
/// next real user message can end up appended to the slot; the marker tells
/// a later compaction where the summary stops and that message starts.
pub const SUMMARY_END_MARKER: &str = "[End of conversation summary]";

/// Labels this module puts in summariser material; a model that echoes one
/// at the top of its output has it stripped.
const MATERIAL_LABELS: &[&str] = &[
    "SUMMARY SO FAR",
    "PREVIOUS SUMMARY",
    "EARLIER-CONTEXT MARKER",
    "CONVERSATION TURNS TO SUMMARISE",
    "NOTE: a summary of even older turns",
    "FOCUS REQUESTED BY THE USER",
    "LATER TURNS",
];

/// Drop material labels the model echoed at the start of its summary.
fn strip_echoed_labels(summary: &str) -> String {
    let mut rest = summary.trim();
    loop {
        let first_line = rest.lines().next().unwrap_or("");
        let bare = first_line.trim_start_matches(['#', '*', ' ', '_']);
        if first_line.is_empty() || !MATERIAL_LABELS.iter().any(|label| bare.starts_with(label)) {
            break;
        }
        rest = rest[first_line.len()..].trim_start();
    }
    rest.to_string()
}

/// True when `text` starts with a channel timestamp prefix such as
/// `[2026-11-07 08:09:33 -06:00] `.
fn starts_with_channel_timestamp(text: &str) -> bool {
    let pattern = b"[dddd-dd-dd dd:dd:dd +dd:dd] ";
    let bytes = text.as_bytes();
    bytes.len() >= pattern.len()
        && pattern.iter().zip(bytes).all(|(&p, &b)| match p {
            b'd' => b.is_ascii_digit(),
            b'+' => b == b'+' || b == b'-',
            _ => p == b,
        })
}

/// Split the previous slot into the carried summary body (header removed; a
/// fresh header is added when the slot is rebuilt) and any user message that
/// history loading merged onto its end. Slots written before the end marker
/// existed fall back to the last timestamped paragraph after the last
/// later-turns heading.
fn split_carried_slot(slot: &str) -> (String, Option<String>) {
    let header = summary_header();
    let body = slot.strip_prefix(header.as_str()).unwrap_or(slot);
    let trailing = |text: &str| Some(text.trim().to_string()).filter(|text| !text.is_empty());
    if let Some(at) = body.rfind(SUMMARY_END_MARKER) {
        return (
            body[..at].trim().to_string(),
            trailing(&body[at + SUMMARY_END_MARKER.len()..]),
        );
    }
    let search_from = body.rfind(LATER_TURNS_HEADING).unwrap_or(0);
    let merged_at = body[search_from..]
        .match_indices("\n\n[")
        .map(|(at, _)| search_from + at)
        .filter(|&at| starts_with_channel_timestamp(&body[at + 2..]))
        .last();
    match merged_at {
        Some(at) => (body[..at].trim().to_string(), trailing(&body[at..])),
        None => (body.trim().to_string(), None),
    }
}

/// Model-visible header that opens the synthetic summary message. Purely
/// informational: classification of the slot is carried by the owner flag,
/// never by matching this text.
pub fn summary_header() -> String {
    crate::i18n::get_required_cli_string("history-compaction-summary-header")
}

/// Resolved compaction settings for one agent (from its runtime profile's
/// `context_compression` section).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactionSettings {
    pub enabled: bool,
    pub summary_max_tokens: usize,
    pub keep_recent_turns: usize,
    pub keep_recent_max_tokens: usize,
    pub timeout_secs: u64,
    pub prompt: Option<String>,
}

impl CompactionSettings {
    #[must_use]
    pub fn from_config(cfg: &ContextCompressionConfig) -> Self {
        Self {
            enabled: cfg.enabled,
            summary_max_tokens: cfg.summary_max_tokens,
            keep_recent_turns: cfg.keep_recent_turns,
            keep_recent_max_tokens: cfg.keep_recent_max_tokens,
            timeout_secs: cfg.timeout_secs,
            prompt: cfg
                .prompt
                .as_deref()
                .map(str::trim)
                .filter(|prompt| !prompt.is_empty())
                .map(str::to_string),
        }
    }

    /// The agent's settings, or `None` when the agent has no runtime profile
    /// (compaction then stays off).
    #[must_use]
    pub fn for_agent(config: &Config, agent_alias: &str) -> Option<Self> {
        config
            .runtime_profile_for_agent(agent_alias)
            .map(|profile| Self::from_config(&profile.context_compression))
    }

    fn system_prompt(&self) -> &str {
        self.prompt.as_deref().unwrap_or(DEFAULT_COMPACTION_PROMPT)
    }
}

impl Default for CompactionSettings {
    fn default() -> Self {
        Self::from_config(&ContextCompressionConfig::default())
    }
}

/// Owner-tracked provenance of the synthetic slot after the leading system
/// messages. `summary_present` implies `crumb_present`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LeadProvenance {
    pub crumb_present: bool,
    pub summary_present: bool,
}

impl LeadProvenance {
    #[must_use]
    pub fn new(crumb_present: bool, summary_present: bool) -> Self {
        Self {
            crumb_present,
            summary_present: crumb_present && summary_present,
        }
    }
}

/// Per-conversation compaction scope, installed by the history owner (the
/// channel orchestrator) around one tool-call loop through
/// [`COMPACTION_SCOPE`]. It carries the owner's `summary_present` record in
/// and out of the loop (the breadcrumb flag travels through
/// `ToolLoop::history_has_trim_breadcrumb` as before) plus the session key
/// for the memory-extraction seam. Without a scope the loop never compacts:
/// a history owner that cannot persist `summary_present` must not create
/// summaries it would later misclassify.
#[derive(Debug, Default)]
pub struct CompactionScope {
    summary_present: Mutex<bool>,
    compactions: AtomicUsize,
    session_key: Option<String>,
}

impl CompactionScope {
    #[must_use]
    pub fn new(summary_present: bool, session_key: Option<String>) -> Arc<Self> {
        Arc::new(Self {
            summary_present: Mutex::new(summary_present),
            compactions: AtomicUsize::new(0),
            session_key,
        })
    }

    #[must_use]
    pub fn summary_present(&self) -> bool {
        *self
            .summary_present
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    pub fn set_summary_present(&self, present: bool) {
        *self
            .summary_present
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = present;
    }

    /// Number of compactions performed under this scope.
    #[must_use]
    pub fn compactions(&self) -> usize {
        self.compactions.load(Ordering::SeqCst)
    }

    fn record_compaction(&self) {
        self.compactions.fetch_add(1, Ordering::SeqCst);
    }

    #[must_use]
    pub fn session_key(&self) -> Option<&str> {
        self.session_key.as_deref()
    }
}

tokio::task_local! {
    /// The compaction scope of the current tool-call loop (see
    /// [`CompactionScope`]).
    pub static COMPACTION_SCOPE: Arc<CompactionScope>;
}

/// The scope installed for the current task, if any.
#[must_use]
pub fn current_scope() -> Option<Arc<CompactionScope>> {
    COMPACTION_SCOPE.try_with(Arc::clone).ok()
}

/// Something that can turn summary material into a summary. Production uses
/// [`ProviderSummarizer`]; tests substitute scripted implementations.
#[async_trait::async_trait]
pub trait Summarizer: Send + Sync {
    async fn summarize(&self, system_prompt: &str, material: &str) -> anyhow::Result<String>;
}

enum ProviderHandle<'a> {
    Owned(Box<dyn ModelProvider>),
    Borrowed(&'a dyn ModelProvider),
}

impl ProviderHandle<'_> {
    fn get(&self) -> &dyn ModelProvider {
        match self {
            Self::Owned(provider) => provider.as_ref(),
            Self::Borrowed(provider) => *provider,
        }
    }
}

/// Summary call through a model provider: one system prompt + one user
/// message, no tools, no native thinking, bounded by `timeout`.
pub struct ProviderSummarizer<'a> {
    provider: ProviderHandle<'a>,
    model: String,
    temperature: Option<f64>,
    timeout: Duration,
}

impl<'a> ProviderSummarizer<'a> {
    #[must_use]
    pub fn owned(
        provider: Box<dyn ModelProvider>,
        model: String,
        temperature: Option<f64>,
        timeout: Duration,
    ) -> Self {
        Self {
            provider: ProviderHandle::Owned(provider),
            model,
            temperature,
            timeout,
        }
    }

    #[must_use]
    pub fn borrowed(
        provider: &'a dyn ModelProvider,
        model: String,
        temperature: Option<f64>,
        timeout: Duration,
    ) -> Self {
        Self {
            provider: ProviderHandle::Borrowed(provider),
            model,
            temperature,
            timeout,
        }
    }

    #[must_use]
    pub fn model(&self) -> &str {
        &self.model
    }
}

#[async_trait::async_trait]
impl Summarizer for ProviderSummarizer<'_> {
    async fn summarize(&self, system_prompt: &str, material: &str) -> anyhow::Result<String> {
        let call = self.provider.get().chat_with_system(
            Some(system_prompt),
            material,
            &self.model,
            self.temperature,
        );
        // Native thinking is a per-turn override the channel scopes around the
        // whole tool loop; the summary call must not inherit it.
        let call = zeroclaw_api::NATIVE_THINKING_OVERRIDE.scope(None, call);
        match tokio::time::timeout(self.timeout, call).await {
            Ok(result) => result,
            Err(_) => anyhow::bail!(
                "compaction summary call timed out after {}s",
                self.timeout.as_secs()
            ),
        }
    }
}

/// Resolve the summary provider for `agent_alias`: the agent-level
/// `summary_provider` override, else the runtime profile's
/// `context_compression.summary_provider`. `Ok(None)` means none is
/// configured and the caller should reuse the agent's own route. The alias's
/// `max_tokens` is overridden with `summary_max_tokens` and its HTTP timeout
/// is raised to at least `timeout_secs` for this call only.
pub fn resolve_summary_provider(
    config: &Config,
    agent_alias: &str,
    settings: &CompactionSettings,
) -> anyhow::Result<Option<ProviderSummarizer<'static>>> {
    let Some(provider_ref) = config.effective_summary_provider(agent_alias) else {
        return Ok(None);
    };
    let name = provider_ref.as_str().trim().to_string();
    let mut scoped = config.clone();
    let mut temperature = None;
    if let Some((family, alias)) = name.split_once('.') {
        for (entry_family, entry_alias, entry) in scoped.providers.models.iter_entries_mut() {
            if entry_family == family && entry_alias == alias {
                entry.max_tokens =
                    Some(u32::try_from(settings.summary_max_tokens).unwrap_or(u32::MAX));
                entry.timeout_secs =
                    Some(entry.timeout_secs.unwrap_or(0).max(settings.timeout_secs));
                temperature = entry.temperature;
            }
        }
    }
    let resolved = zeroclaw_providers::create_model_provider_from_ref_with_model(&scoped, &name)?;
    let model = resolved
        .model
        .filter(|model| !model.trim().is_empty())
        .ok_or_else(|| {
            anyhow::Error::msg(format!(
                "context_compression.summary_provider `{name}` names no model; set `model` on that providers.models entry"
            ))
        })?;
    Ok(Some(ProviderSummarizer::owned(
        resolved.provider,
        model,
        // Low temperature unless the alias pins its own.
        temperature.or(Some(0.2)),
        Duration::from_secs(settings.timeout_secs.max(1)),
    )))
}

/// Why a compaction did not happen. On every error the history is left
/// exactly as it was.
#[derive(Debug)]
pub enum CompactionError {
    /// Nothing older than the protected verbatim tail.
    NothingToCompact,
    /// The budget leaves no room for summary material.
    BudgetTooSmall,
    /// The summary call failed or timed out.
    Summarizer(anyhow::Error),
    /// The summariser returned an empty summary.
    EmptySummary,
}

impl std::fmt::Display for CompactionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NothingToCompact => f.write_str("nothing older than the protected recent turns"),
            Self::BudgetTooSmall => {
                f.write_str("context budget leaves no room for summary material")
            }
            Self::Summarizer(error) => write!(f, "summary call failed: {error}"),
            Self::EmptySummary => f.write_str("summary call returned an empty summary"),
        }
    }
}

impl std::error::Error for CompactionError {}

/// Inputs for one compaction.
#[derive(Debug, Clone, Copy)]
pub struct CompactionRequest<'a> {
    pub settings: &'a CompactionSettings,
    /// Optional focus text (manual `/compact <focus>`).
    pub focus: Option<&'a str>,
    /// The route's context budget in calibrated tokens; sizes how much
    /// material one summary call may carry.
    pub budget_tokens: usize,
    /// Calibration ratio (`reported / estimated`, at least 1.0).
    pub ratio: f64,
    /// Also summarise the earlier tool iterations of the newest turn (the
    /// one still in progress), keeping its user message and its latest
    /// iteration verbatim. Used when that turn alone outgrows the budget.
    pub split_current_turn: bool,
}

/// What a successful compaction did. Token counts are calibrated estimates
/// of the raw history (callers that also carry tool schemas or prepared
/// multimodal payloads measure the provider-facing population themselves).
#[derive(Debug, Clone)]
pub struct CompactionOutcome {
    pub tokens_before: usize,
    pub tokens_after: usize,
    pub turns_summarised: usize,
    pub messages_summarised: usize,
    pub kept_turns: usize,
    pub summary_calls: usize,
    pub summary: String,
    /// Whether a previous summary was folded into this one.
    pub folded_previous_summary: bool,
    /// Earlier tool iterations of the in-progress turn that were summarised
    /// (0 unless the turn was split).
    pub iterations_summarised: usize,
    /// When the in-progress turn was split: its user message's index before
    /// and after compaction (it is kept verbatim but moves).
    pub current_turn_moved: Option<(usize, usize)>,
}

/// Where a compaction cuts the history.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CompactionPlan {
    /// Number of leading system messages.
    pub system_end: usize,
    /// First summarised message (after the synthetic slot, if any).
    pub body_start: usize,
    /// First message of the verbatim tail (a turn boundary).
    pub tail_start: usize,
    pub turns_summarised: usize,
    pub kept_turns: usize,
    /// Split plan only: the in-progress turn's user message, kept verbatim
    /// in front of `tail_start` (which is then its latest tool iteration).
    pub current_turn_head: Option<usize>,
    pub iterations_summarised: usize,
}

fn is_turn_boundary(message: &ChatMessage) -> bool {
    message.role == "user"
        && !message
            .content
            .starts_with(zeroclaw_api::tool_carrier::TOOL_RESULTS_PREFIX)
}

pub(crate) fn calibrated(estimated: usize, ratio: f64) -> usize {
    (estimated as f64 * ratio.max(1.0)).round() as usize
}

fn calibrated_text_tokens(text: &str, ratio: f64) -> usize {
    calibrated(text.len().div_ceil(4) + 4, ratio)
}

/// Choose the verbatim tail: up to `keep_recent_turns` newest whole turns,
/// stopping before an older turn would push the tail above
/// `keep_recent_max_tokens` (0 = no cap). The newest whole turn is always
/// kept. Everything between the synthetic slot and the tail is summarised.
/// When the protected tail would cover every turn (so nothing is left to
/// summarise), the tail shrinks to the newest turn alone and the rest is
/// summarised. Whole turns only, so an assistant tool call is never
/// separated from its tool results. `None` only when the newest turn is the
/// sole turn after the synthetic slot.
///
/// With `split_current_turn`, the newest turn (the one in progress) is also
/// cut between tool iterations: its user message and its latest iteration
/// (an assistant message plus the results after it) stay verbatim, every
/// older turn and its earlier iterations are summarised. Falls back to the
/// whole-turn plan when the turn has fewer than two iterations.
pub(crate) fn plan_compaction(
    history: &[ChatMessage],
    crumb_present: bool,
    keep_recent_turns: usize,
    keep_recent_max_tokens: usize,
    ratio: f64,
    split_current_turn: bool,
) -> Option<CompactionPlan> {
    let system_end = history.iter().take_while(|m| m.role == "system").count();
    let body_start =
        (system_end + usize::from(crumb_present && history.len() > system_end)).min(history.len());
    let boundaries: Vec<usize> = (body_start..history.len())
        .filter(|&index| is_turn_boundary(&history[index]))
        .collect();
    let newest = *boundaries.last()?;
    if split_current_turn {
        let iterations: Vec<usize> = (newest + 1..history.len())
            .filter(|&index| history[index].role == "assistant")
            .collect();
        if iterations.len() >= 2 {
            return Some(CompactionPlan {
                system_end,
                body_start,
                tail_start: *iterations.last()?,
                turns_summarised: boundaries.len() - 1,
                kept_turns: 1,
                current_turn_head: Some(newest),
                iterations_summarised: iterations.len() - 1,
            });
        }
    }
    let mut tail_start = newest;
    let mut kept_turns = 1usize;
    for &candidate in boundaries.iter().rev().skip(1) {
        if kept_turns >= keep_recent_turns.max(1) {
            break;
        }
        if keep_recent_max_tokens > 0 {
            let tail_tokens = calibrated(estimate_history_tokens(&history[candidate..]), ratio);
            if tail_tokens > keep_recent_max_tokens {
                break;
            }
        }
        tail_start = candidate;
        kept_turns += 1;
    }
    // The protected tail swallowed every turn, yet we are over budget (the
    // only reason a plan is asked for). Fold the older protected turns into
    // the summary and keep just the newest whole turn: summarising them
    // beats the whole-turn trim fallback, which would drop them outright.
    if tail_start <= body_start {
        tail_start = newest;
        kept_turns = 1;
    }
    if tail_start <= body_start {
        return None;
    }
    let turns_summarised = boundaries.iter().filter(|&&b| b < tail_start).count();
    Some(CompactionPlan {
        system_end,
        body_start,
        tail_start,
        turns_summarised,
        kept_turns,
        current_turn_head: None,
        iterations_summarised: 0,
    })
}

/// Whether the in-progress (newest) turn should be split between tool
/// iterations: the whole-turn plan has nothing to summarise, or the newest
/// turn plus the leading system messages alone exceeds `room_tokens` (the
/// budget left after tool schemas and the summary's reserve), so no amount
/// of summarising older turns can make the next request fit.
pub(crate) fn current_turn_needs_split(
    history: &[ChatMessage],
    crumb_present: bool,
    settings: &CompactionSettings,
    ratio: f64,
    room_tokens: usize,
) -> bool {
    let Some(plan) = plan_compaction(
        history,
        crumb_present,
        settings.keep_recent_turns,
        settings.keep_recent_max_tokens,
        ratio,
        false,
    ) else {
        return true;
    };
    let newest = (plan.body_start..history.len())
        .rev()
        .find(|&index| is_turn_boundary(&history[index]))
        .unwrap_or(plan.tail_start);
    let floor = calibrated(estimate_history_tokens(&history[..plan.system_end]), ratio)
        + calibrated(estimate_history_tokens(&history[newest..]), ratio);
    floor > room_tokens
}

/// Replace `[IMAGE:data:...]` inline payloads with a placeholder so base64
/// bytes never reach the summariser.
fn strip_inline_image_payloads(content: &str) -> std::borrow::Cow<'_, str> {
    if !content.contains("[IMAGE:data:") {
        return std::borrow::Cow::Borrowed(content);
    }
    let mut out = String::with_capacity(content.len().min(4096));
    let mut cursor = 0usize;
    while let Some(rel) = content[cursor..].find("[IMAGE:data:") {
        let start = cursor + rel;
        out.push_str(&content[cursor..start]);
        match content[start..].find(']') {
            Some(rel_end) => {
                out.push_str("[inline image omitted]");
                cursor = start + rel_end + 1;
            }
            None => {
                out.push_str("[inline image omitted]");
                cursor = content.len();
            }
        }
    }
    out.push_str(&content[cursor..]);
    std::borrow::Cow::Owned(out)
}

fn render_message(message: &ChatMessage) -> String {
    let label = match message.role.as_str() {
        "user"
            if message
                .content
                .starts_with(zeroclaw_api::tool_carrier::TOOL_RESULTS_PREFIX) =>
        {
            "tool results"
        }
        "user" => "user",
        "assistant" => "assistant (me)",
        "tool" => "tool result",
        "system" => "system note",
        other => other,
    };
    format!(
        "### {label}\n{}\n\n",
        strip_inline_image_payloads(&message.content)
    )
}

/// Split rendered units into chunks whose calibrated size each fits
/// `chunk_budget`. A single unit larger than the budget is cut (head + tail)
/// for the summariser input only, with an explicit marker; the history
/// itself is never altered this way.
fn chunk_units(units: Vec<String>, chunk_budget: usize, ratio: f64) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();
    for unit in units {
        let unit = if calibrated_text_tokens(&unit, ratio) > chunk_budget {
            let max_chars = ((chunk_budget as f64 / ratio.max(1.0)) as usize)
                .saturating_sub(16)
                .saturating_mul(4);
            let cut = crate::agent::history::truncate_tool_result(&unit, max_chars.max(256));
            format!(
                "{cut}\n[... this single message was too large for one summary call and was shortened for the summariser only ...]\n\n"
            )
        } else {
            unit
        };
        if !current.is_empty()
            && calibrated_text_tokens(&current, ratio) + calibrated_text_tokens(&unit, ratio)
                > chunk_budget
        {
            chunks.push(std::mem::take(&mut current));
        }
        current.push_str(&unit);
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

fn build_material(
    previous: Option<&str>,
    previous_label: &str,
    chunk: &str,
    part: (usize, usize),
    focus: Option<&str>,
) -> String {
    let mut material = String::new();
    if let Some(focus) = focus.map(str::trim).filter(|focus| !focus.is_empty()) {
        let _ = write!(
            material,
            "FOCUS REQUESTED BY THE USER for this summary (still cover every section, but give \
             this the most detail):\n{focus}\n\n"
        );
    }
    if let Some(previous) = previous {
        let _ = write!(material, "{previous_label}\n{previous}\n\n");
    }
    let _ = write!(
        material,
        "CONVERSATION TURNS TO SUMMARISE (part {} of {}, oldest first):\n\n{chunk}",
        part.0, part.1
    );
    if part.0 < part.1 {
        material.push_str(
            "\n\nMore turns follow in later parts; write the complete summary of everything so \
             far, it will be extended with the next part.",
        );
    }
    material
}

/// Summarise the oldest whole turns of `history` into ONE synthetic summary
/// message placed in the synthetic slot after the leading system messages,
/// keeping the planned verbatim tail. On success `history` and `lead` are
/// rewritten (`crumb_present = summary_present = true`); on any error both
/// are left untouched so the caller's whole-turn trim fallback applies.
pub async fn compact_history(
    history: &mut Vec<ChatMessage>,
    lead: &mut LeadProvenance,
    summarizer: &dyn Summarizer,
    request: &CompactionRequest<'_>,
) -> Result<CompactionOutcome, CompactionError> {
    let ratio = request.ratio.max(1.0);
    let settings = request.settings;
    let plan = plan_compaction(
        history,
        lead.crumb_present,
        settings.keep_recent_turns,
        settings.keep_recent_max_tokens,
        ratio,
        request.split_current_turn,
    )
    .ok_or(CompactionError::NothingToCompact)?;
    let tokens_before = calibrated(estimate_history_tokens(history), ratio);

    // The synthetic slot's content is always folded in: as the previous
    // summary when the owner recorded one, otherwise as an earlier-context
    // marker (a breadcrumb, or a summary a backend could not record).
    let previous_slot = lead
        .crumb_present
        .then(|| history.get(plan.system_end))
        .flatten()
        .map(|message| strip_inline_image_payloads(&message.content).into_owned());
    let previous_label = if lead.summary_present {
        "PREVIOUS SUMMARY of even older turns (fold ALL of it into the new summary):"
    } else {
        "EARLIER-CONTEXT MARKER (some older turns were already removed without a summary; keep \
         any facts it contains):"
    };

    let system_prompt = settings.system_prompt();
    let prompt_tokens = calibrated_text_tokens(system_prompt, ratio) + 512;
    let per_call_budget = request
        .budget_tokens
        .saturating_sub(settings.summary_max_tokens)
        .saturating_sub(prompt_tokens);
    // The running summary can grow to `summary_max_tokens`; reserve it so the
    // running summary plus the next chunk always fit one call.
    let chunk_budget = per_call_budget.saturating_sub(settings.summary_max_tokens);
    if chunk_budget < 1024 {
        return Err(CompactionError::BudgetTooSmall);
    }

    // A previous SUMMARY is carried forward verbatim by code, never by asking
    // the model to copy it: a local summariser told to "fold ALL of it in"
    // was seen replacing a ~10k-token summary with a 500-token one about the
    // newest turn only. The model summarises just the new turns, appended
    // below the carried text, and is asked to consolidate only once the
    // combined summary passes `summary_max_tokens`. A breadcrumb (no summary)
    // is still folded in by the model as before.
    // A user message that history loading merged onto the slot belongs to
    // the oldest turn being summarised, not to the carried summary.
    let (carried, merged_user) = if lead.summary_present {
        match previous_slot.as_deref().map(split_carried_slot) {
            Some((body, merged)) => (Some(body), merged),
            None => (None, None),
        }
    } else {
        (None, None)
    };
    let (mut running, mut label) = if carried.is_some() {
        (None, previous_label)
    } else {
        (previous_slot, previous_label)
    };

    let mut units: Vec<String> = Vec::new();
    if let Some(merged) = &merged_user {
        units.push(render_message(&ChatMessage::user(merged.clone())));
    }
    for (index, message) in history
        .iter()
        .enumerate()
        .take(plan.tail_start)
        .skip(plan.body_start)
    {
        if Some(index) == plan.current_turn_head {
            units.push(format!(
                "### user (the request I am still working on; it stays in my context verbatim, \
                 so record what I have done for it so far)\n{}\n\n",
                strip_inline_image_payloads(&message.content)
            ));
        } else {
            units.push(render_message(message));
        }
    }
    let chunks = chunk_units(units, chunk_budget, ratio);
    let total = chunks.len();
    let mut calls = 0usize;
    for (index, chunk) in chunks.iter().enumerate() {
        let mut material = build_material(
            running.as_deref(),
            label,
            chunk,
            (index + 1, total),
            request.focus,
        );
        if carried.is_some() && running.is_none() {
            material.insert_str(0, CARRIED_SUMMARY_NOTE);
        }
        let summary = summarizer
            .summarize(system_prompt, &material)
            .await
            .map_err(CompactionError::Summarizer)?;
        calls += 1;
        let summary = strip_echoed_labels(&summary);
        if summary.is_empty() {
            return Err(CompactionError::EmptySummary);
        }
        running = Some(summary);
        label = "SUMMARY SO FAR of the older parts (extend it; keep ALL of it):";
    }
    let new_part = running.ok_or(CompactionError::EmptySummary)?;
    let summary = match carried {
        None => new_part,
        Some(carried) => {
            let combined = format!("{carried}\n\n{LATER_TURNS_HEADING}\n\n{new_part}");
            let combined_tokens = calibrated_text_tokens(&combined, ratio);
            if combined_tokens <= settings.summary_max_tokens || combined_tokens > per_call_budget {
                // Under the cap: keep it all, verbatim. Too big for one
                // consolidation call: also keep it verbatim (over the cap
                // beats losing facts; the projection check decides fit).
                combined
            } else {
                let material = format!(
                    "PREVIOUS SUMMARY of even older turns (fold ALL of it into the new summary; \
                     it is over the length limit, so merge duplicates and tighten wording, but \
                     keep every fact, identifier, amount and date):\n{combined}"
                );
                let consolidated = summarizer
                    .summarize(system_prompt, &material)
                    .await
                    .map_err(CompactionError::Summarizer)?;
                calls += 1;
                let consolidated = strip_echoed_labels(&consolidated);
                // A consolidation that kept less than 40% of the text dropped
                // facts, not just duplicates: keep the verbatim combination.
                if consolidated.len() * 10 < combined.len() * 4 {
                    ::zeroclaw_log::record!(
                        WARN,
                        ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Note)
                            .with_category(::zeroclaw_log::EventCategory::Agent)
                            .with_attrs(::serde_json::json!({
                                "event": "compaction_consolidation_rejected",
                                "combined_chars": combined.len(),
                                "consolidated_chars": consolidated.len(),
                            })),
                        "Summary consolidation looked lossy; keeping the verbatim summary"
                    );
                    combined
                } else {
                    consolidated
                }
            }
        }
    };
    let message = ChatMessage::user(format!(
        "{}\n\n{summary}\n\n{SUMMARY_END_MARKER}",
        summary_header()
    ));

    let messages_summarised =
        plan.tail_start - plan.body_start - usize::from(plan.current_turn_head.is_some());
    let folded_previous_summary = lead.summary_present;
    let mut compacted = Vec::with_capacity(plan.system_end + 2 + history.len() - plan.tail_start);
    compacted.extend_from_slice(&history[..plan.system_end]);
    compacted.push(message);
    let current_turn_moved = plan.current_turn_head.map(|head| {
        compacted.push(history[head].clone());
        (head, compacted.len() - 1)
    });
    compacted.extend_from_slice(&history[plan.tail_start..]);
    *history = compacted;
    *lead = LeadProvenance::new(true, true);
    let tokens_after = calibrated(estimate_history_tokens(history), ratio);
    Ok(CompactionOutcome {
        tokens_before,
        tokens_after,
        turns_summarised: plan.turns_summarised,
        messages_summarised,
        kept_turns: plan.kept_turns,
        summary_calls: calls,
        summary,
        folded_previous_summary,
        iterations_summarised: plan.iterations_summarised,
        current_turn_moved,
    })
}

/// Memory-extraction seam: called once after every successful compaction
/// with the new summary and the conversation's session key. Deliberately
/// inert — the summary is NOT written to the memory backend.
///
/// TODO(memory extraction from compaction summaries): pull durable facts out
/// of `summary` into the memory backend here (a separate, later task).
pub fn on_compaction_summary(session_key: Option<&str>, summary: &str) {
    ::zeroclaw_log::record!(
        DEBUG,
        ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Note)
            .with_category(::zeroclaw_log::EventCategory::Agent)
            .with_attrs(::serde_json::json!({
                "event": "compaction_summary_available",
                "session_key": session_key,
                "summary_chars": summary.chars().count(),
            })),
        "compaction summary available for memory extraction (not implemented)"
    );
}

/// Record the start / success / failure of a compaction.
pub fn record_compaction_event(
    trigger: &str,
    session_key: Option<&str>,
    result: Result<&CompactionOutcome, &CompactionError>,
    tokens_before: usize,
    tokens_after: Option<usize>,
    budget_tokens: usize,
) {
    match result {
        Ok(outcome) => ::zeroclaw_log::record!(
            INFO,
            ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Complete)
                .with_category(::zeroclaw_log::EventCategory::Agent)
                .with_outcome(::zeroclaw_log::EventOutcome::Success)
                .with_attrs(::serde_json::json!({
                    "event": "context_compaction_succeeded",
                    "trigger": trigger,
                    "session_key": session_key,
                    "tokens_before": tokens_before,
                    "tokens_after": tokens_after.unwrap_or(outcome.tokens_after),
                    "budget_tokens": budget_tokens,
                    "turns_summarised": outcome.turns_summarised,
                    "messages_summarised": outcome.messages_summarised,
                    "kept_turns": outcome.kept_turns,
                    "summary_calls": outcome.summary_calls,
                    "summary_chars": outcome.summary.chars().count(),
                    "folded_previous_summary": outcome.folded_previous_summary,
                    "iterations_summarised": outcome.iterations_summarised,
                })),
            format!(
                "Context compacted ({trigger}): summarised {} turn(s), kept {}, {} -> {} tok",
                outcome.turns_summarised,
                outcome.kept_turns,
                tokens_before,
                tokens_after.unwrap_or(outcome.tokens_after)
            )
        ),
        Err(error) => ::zeroclaw_log::record!(
            WARN,
            ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Fail)
                .with_category(::zeroclaw_log::EventCategory::Agent)
                .with_outcome(::zeroclaw_log::EventOutcome::Failure)
                .with_attrs(::serde_json::json!({
                    "event": "context_compaction_failed",
                    "trigger": trigger,
                    "session_key": session_key,
                    "tokens_before": tokens_before,
                    "budget_tokens": budget_tokens,
                    "error": error.to_string(),
                })),
            format!("Context compaction ({trigger}) did not run: {error}")
        ),
    }
}

/// Record that a compaction is starting.
pub fn record_compaction_start(
    trigger: &str,
    session_key: Option<&str>,
    tokens_before: usize,
    budget_tokens: usize,
) {
    ::zeroclaw_log::record!(
        INFO,
        ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Start)
            .with_category(::zeroclaw_log::EventCategory::Agent)
            .with_attrs(::serde_json::json!({
                "event": "context_compaction_started",
                "trigger": trigger,
                "session_key": session_key,
                "tokens_before": tokens_before,
                "budget_tokens": budget_tokens,
            })),
        format!(
            "Context compaction ({trigger}) starting at {tokens_before} tok (budget {budget_tokens})"
        )
    );
}

/// Run one compaction for `agent_alias` with its configured summary provider
/// (or `fallback` = the agent's own provider/model when none is configured),
/// logging start/success/failure and calling [`on_compaction_summary`] on
/// success. Shared by the in-loop budget seams and the channel `/compact`.
#[allow(clippy::too_many_arguments)]
pub async fn run_compaction(
    trigger: &str,
    config: Option<&Config>,
    agent_alias: Option<&str>,
    fallback: Option<(&dyn ModelProvider, &str, Option<f64>)>,
    settings: &CompactionSettings,
    session_key: Option<&str>,
    history: &mut Vec<ChatMessage>,
    lead: &mut LeadProvenance,
    focus: Option<&str>,
    budget_tokens: usize,
    ratio: f64,
    split_current_turn: bool,
) -> Result<CompactionOutcome, CompactionError> {
    let tokens_before = calibrated(estimate_history_tokens(history), ratio);
    if plan_compaction(
        history,
        lead.crumb_present,
        settings.keep_recent_turns,
        settings.keep_recent_max_tokens,
        ratio,
        split_current_turn,
    )
    .is_none()
    {
        let error = CompactionError::NothingToCompact;
        record_compaction_event(
            trigger,
            session_key,
            Err(&error),
            tokens_before,
            None,
            budget_tokens,
        );
        return Err(error);
    }
    record_compaction_start(trigger, session_key, tokens_before, budget_tokens);
    let configured = match config.zip(agent_alias) {
        Some((config, alias)) => match resolve_summary_provider(config, alias, settings) {
            Ok(summarizer) => summarizer,
            Err(error) => {
                let error = CompactionError::Summarizer(error);
                record_compaction_event(
                    trigger,
                    session_key,
                    Err(&error),
                    tokens_before,
                    None,
                    budget_tokens,
                );
                return Err(error);
            }
        },
        None => None,
    };
    let fallback_summarizer;
    let summarizer: &dyn Summarizer = match (&configured, fallback) {
        (Some(summarizer), _) => summarizer,
        (None, Some((provider, model, temperature))) => {
            fallback_summarizer = ProviderSummarizer::borrowed(
                provider,
                model.to_string(),
                temperature,
                Duration::from_secs(settings.timeout_secs.max(1)),
            );
            &fallback_summarizer
        }
        (None, None) => {
            let error = CompactionError::Summarizer(anyhow::Error::msg(
                "no summary provider configured and no agent route available",
            ));
            record_compaction_event(
                trigger,
                session_key,
                Err(&error),
                tokens_before,
                None,
                budget_tokens,
            );
            return Err(error);
        }
    };
    let request = CompactionRequest {
        settings,
        focus,
        budget_tokens,
        ratio,
        split_current_turn,
    };
    let result = compact_history(history, lead, summarizer, &request).await;
    record_compaction_event(
        trigger,
        session_key,
        result.as_ref(),
        tokens_before,
        None,
        budget_tokens,
    );
    if let Ok(outcome) = &result {
        on_compaction_summary(session_key, &outcome.summary);
        if let Some(scope) = current_scope() {
            scope.record_compaction();
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    /// Scripted summariser: records every material it receives and replies
    /// with `summary-<n>`, or fails when `fail` is set.
    struct ScriptedSummarizer {
        seen: StdMutex<Vec<String>>,
        fail: bool,
    }

    impl ScriptedSummarizer {
        fn ok() -> Self {
            Self {
                seen: StdMutex::new(Vec::new()),
                fail: false,
            }
        }
        fn failing() -> Self {
            Self {
                seen: StdMutex::new(Vec::new()),
                fail: true,
            }
        }
        fn seen(&self) -> Vec<String> {
            self.seen.lock().unwrap().clone()
        }
    }

    #[async_trait::async_trait]
    impl Summarizer for ScriptedSummarizer {
        async fn summarize(&self, _system: &str, material: &str) -> anyhow::Result<String> {
            let mut seen = self.seen.lock().unwrap();
            seen.push(material.to_string());
            if self.fail {
                anyhow::bail!("provider down");
            }
            Ok(format!("summary-{}", seen.len()))
        }
    }

    fn rows(history: &[ChatMessage]) -> Vec<(String, String)> {
        history
            .iter()
            .map(|m| (m.role.clone(), m.content.clone()))
            .collect()
    }

    fn settings(keep_turns: usize, keep_max_tokens: usize) -> CompactionSettings {
        CompactionSettings {
            enabled: true,
            summary_max_tokens: 2_000,
            keep_recent_turns: keep_turns,
            keep_recent_max_tokens: keep_max_tokens,
            timeout_secs: 5,
            prompt: None,
        }
    }

    fn request(settings: &CompactionSettings) -> CompactionRequest<'_> {
        CompactionRequest {
            settings,
            focus: None,
            budget_tokens: 100_000,
            ratio: 1.0,
            split_current_turn: false,
        }
    }

    fn native_tool_turn(history: &mut Vec<ChatMessage>, turn: usize) {
        history.push(ChatMessage::user(format!("request {turn}")));
        history.push(ChatMessage::assistant(format!(
            r#"{{"content":null,"tool_calls":[{{"id":"call-{turn}","name":"shell","arguments":"{{}}"}}]}}"#
        )));
        history.push(ChatMessage::tool(format!(
            r#"{{"tool_call_id":"call-{turn}","content":"result {turn}"}}"#
        )));
        history.push(ChatMessage::assistant(format!("answer {turn}")));
    }

    fn conversation(turns: usize) -> Vec<ChatMessage> {
        let mut history = vec![ChatMessage::system("system prompt")];
        for turn in 0..turns {
            native_tool_turn(&mut history, turn);
        }
        history
    }

    #[tokio::test]
    async fn summary_replaces_oldest_turns_and_keeps_the_tail_verbatim() {
        let mut history = conversation(6);
        let original = history.clone();
        let mut lead = LeadProvenance::default();
        let summarizer = ScriptedSummarizer::ok();
        let settings = settings(4, 0);

        let outcome = compact_history(&mut history, &mut lead, &summarizer, &request(&settings))
            .await
            .expect("compaction succeeds");

        assert_eq!(outcome.turns_summarised, 2);
        assert_eq!(outcome.kept_turns, 4);
        assert_eq!(outcome.messages_summarised, 8);
        assert_eq!(lead, LeadProvenance::new(true, true));
        // Ordering: system -> ONE summary -> the 4 newest turns verbatim.
        assert_eq!(rows(&history[..1]), rows(&original[..1]));
        assert_eq!(history[1].role, "user");
        assert!(history[1].content.ends_with(&format!(
            "summary-1

{SUMMARY_END_MARKER}"
        )));
        assert_eq!(rows(&history[2..]), rows(&original[9..]));
        assert_eq!(history[2].content, "request 2");
        // The summariser saw the two oldest turns, tool exchange included.
        let seen = summarizer.seen();
        assert_eq!(seen.len(), 1);
        assert!(seen[0].contains("request 0") && seen[0].contains("result 1"));
        assert!(!seen[0].contains("request 2"));
    }

    #[tokio::test]
    async fn tool_call_and_results_are_never_split() {
        let mut history = vec![ChatMessage::system("s")];
        history.push(ChatMessage::user("old"));
        history.push(ChatMessage::assistant("old answer"));
        history.push(ChatMessage::user("newest request"));
        for index in 0..5 {
            history.push(ChatMessage::assistant(format!(
                r#"{{"content":null,"tool_calls":[{{"id":"c{index}","name":"shell","arguments":"{{}}"}}]}}"#
            )));
            history.push(ChatMessage::tool(format!(
                r#"{{"tool_call_id":"c{index}","content":"r{index}"}}"#
            )));
            // Prompt-mode tool results are user-role rows but never turn
            // boundaries.
            history.push(ChatMessage::user(format!(
                "{} r{index}",
                zeroclaw_api::tool_carrier::TOOL_RESULTS_PREFIX
            )));
        }
        let original = history.clone();
        let mut lead = LeadProvenance::default();
        let summarizer = ScriptedSummarizer::ok();
        let settings = settings(1, 0);
        compact_history(&mut history, &mut lead, &summarizer, &request(&settings))
            .await
            .expect("compaction succeeds");
        // The whole newest turn (user + every call/result row) survives.
        assert_eq!(rows(&history[2..]), rows(&original[3..]));
        for (index, message) in history.iter().enumerate() {
            if message.role == "tool" {
                assert_eq!(history[index - 1].role, "assistant");
            }
        }
    }

    #[test]
    fn tail_respects_token_cap_but_always_keeps_newest_turn() {
        let mut history = vec![ChatMessage::system("s")];
        for turn in 0..4 {
            history.push(ChatMessage::user(format!("turn {turn}")));
            history.push(ChatMessage::assistant("x".repeat(4_000)));
        }
        // Each turn is ~1,000+ tokens; a 2,500-token cap admits two turns.
        let plan = plan_compaction(&history, false, 4, 2_500, 1.0, false).expect("plan");
        assert_eq!(plan.kept_turns, 2);
        assert_eq!(history[plan.tail_start].content, "turn 2");
        // A cap below one turn still keeps the newest whole turn.
        let plan = plan_compaction(&history, false, 4, 10, 1.0, false).expect("plan");
        assert_eq!(plan.kept_turns, 1);
        assert_eq!(history[plan.tail_start].content, "turn 3");
        // The calibration ratio scales the cap check.
        let plan = plan_compaction(&history, false, 4, 2_500, 2.0, false).expect("plan");
        assert_eq!(plan.kept_turns, 1);
        // Nothing older than the newest turn: nothing to compact.
        assert!(plan_compaction(&history[..3], false, 4, 0, 1.0, false).is_none());
    }

    #[test]
    fn protected_tail_covering_every_turn_shrinks_to_newest_turn() {
        // Summary slot + 4 turns, keep_recent_turns = 4: the tail would cover
        // every turn. Over budget, the older three are summarised instead of
        // being left for the whole-turn trim to drop.
        let mut history = vec![ChatMessage::system("s"), ChatMessage::user("[summary]")];
        for turn in 0..4 {
            history.push(ChatMessage::user(format!("turn {turn}")));
            history.push(ChatMessage::assistant(format!("reply {turn}")));
        }
        let plan = plan_compaction(&history, true, 4, 0, 1.0, false).expect("plan");
        assert_eq!(plan.body_start, 2);
        assert_eq!(plan.kept_turns, 1);
        assert_eq!(plan.turns_summarised, 3);
        assert_eq!(history[plan.tail_start].content, "turn 3");
        // Slot + only the newest turn: still nothing to compact.
        let only_newest = [
            ChatMessage::system("s"),
            ChatMessage::user("[summary]"),
            ChatMessage::user("turn 0"),
            ChatMessage::assistant("reply 0"),
        ];
        assert!(plan_compaction(&only_newest, true, 4, 0, 1.0, false).is_none());
    }

    #[tokio::test]
    async fn summarizer_failure_leaves_history_untouched() {
        let mut history = conversation(6);
        let original = history.clone();
        let mut lead = LeadProvenance::new(true, false);
        history.insert(1, ChatMessage::user("[crumb]"));
        let original_with_crumb = history.clone();
        let summarizer = ScriptedSummarizer::failing();
        let settings = settings(2, 0);
        let error = compact_history(&mut history, &mut lead, &summarizer, &request(&settings))
            .await
            .expect_err("compaction fails");
        assert!(matches!(error, CompactionError::Summarizer(_)));
        assert_eq!(rows(&history), rows(&original_with_crumb));
        assert_eq!(lead, LeadProvenance::new(true, false));
        assert_eq!(history.len(), original.len() + 1);
    }

    #[tokio::test]
    async fn second_compaction_carries_the_previous_summary_verbatim() {
        let mut history = conversation(6);
        let mut lead = LeadProvenance::default();
        let summarizer = ScriptedSummarizer::ok();
        let settings = settings(2, 0);
        compact_history(&mut history, &mut lead, &summarizer, &request(&settings))
            .await
            .expect("first compaction");
        for turn in 6..9 {
            native_tool_turn(&mut history, turn);
        }
        let outcome = compact_history(&mut history, &mut lead, &summarizer, &request(&settings))
            .await
            .expect("second compaction");
        assert!(outcome.folded_previous_summary);
        let seen = summarizer.seen();
        assert_eq!(seen.len(), 2);
        // The model summarises only the new turns; it is never handed the
        // previous summary to copy forward.
        assert!(seen[1].starts_with(CARRIED_SUMMARY_NOTE));
        assert!(!seen[1].contains("summary-1"));
        assert!(seen[1].contains("request 4"));
        // One summary message: the carried text verbatim, then the new part.
        assert_eq!(history[1].role, "user");
        let slot = &history[1].content;
        assert!(slot.starts_with(&summary_header()));
        let carried_at = slot.find("summary-1").expect("carried summary kept");
        let heading_at = slot.find(LATER_TURNS_HEADING).expect("later-turns heading");
        assert!(carried_at < heading_at);
        assert!(slot.ends_with(&format!(
            "summary-2

{SUMMARY_END_MARKER}"
        )));
        assert_eq!(slot.matches(SUMMARY_END_MARKER).count(), 1);
        assert_eq!(slot.matches(&summary_header()).count(), 1);
        assert_eq!(
            history
                .iter()
                .filter(|m| m.content.contains("summary-"))
                .count(),
            1
        );
        assert_eq!(history[2].content, "request 7");
    }

    /// Returns `body_chars` of text for turn material and `consolidate_pct`
    /// percent of the input length for a consolidation call.
    struct SizedSummarizer {
        body_chars: usize,
        consolidate_pct: usize,
        seen: std::sync::Mutex<Vec<String>>,
    }

    #[async_trait::async_trait]
    impl Summarizer for SizedSummarizer {
        async fn summarize(&self, _system: &str, material: &str) -> anyhow::Result<String> {
            self.seen.lock().unwrap().push(material.to_string());
            if material.starts_with("PREVIOUS SUMMARY") {
                return Ok("c".repeat(material.len() * self.consolidate_pct / 100));
            }
            Ok("b".repeat(self.body_chars))
        }
    }

    async fn compact_twice_with(consolidate_pct: usize) -> (Vec<ChatMessage>, usize, Vec<String>) {
        let mut history = conversation(6);
        let mut lead = LeadProvenance::default();
        // ~1,500 tokens per summary; two together pass the 2,000-token cap.
        let summarizer = SizedSummarizer {
            body_chars: 6_000,
            consolidate_pct,
            seen: std::sync::Mutex::new(Vec::new()),
        };
        let settings = settings(2, 0);
        compact_history(&mut history, &mut lead, &summarizer, &request(&settings))
            .await
            .expect("first compaction");
        for turn in 6..9 {
            native_tool_turn(&mut history, turn);
        }
        let outcome = compact_history(&mut history, &mut lead, &summarizer, &request(&settings))
            .await
            .expect("second compaction");
        let seen = summarizer.seen.lock().unwrap().clone();
        (history, outcome.summary_calls, seen)
    }

    #[tokio::test]
    async fn summary_over_the_cap_is_consolidated() {
        let (history, calls, seen) = compact_twice_with(60).await;
        assert_eq!(calls, 2, "new-turns call + consolidation call");
        assert_eq!(seen.len(), 3);
        assert!(seen[2].starts_with("PREVIOUS SUMMARY"));
        assert!(seen[2].contains(LATER_TURNS_HEADING));
        assert!(history[1].content.contains("ccc"));
        assert!(!history[1].content.contains(LATER_TURNS_HEADING));
    }

    #[tokio::test]
    async fn lossy_consolidation_is_rejected_and_the_verbatim_summary_kept() {
        let (history, _calls, seen) = compact_twice_with(10).await;
        assert_eq!(seen.len(), 3);
        let slot = &history[1].content;
        assert!(slot.contains(LATER_TURNS_HEADING));
        assert!(!slot.contains("ccc"));
        assert_eq!(slot.matches(&"b".repeat(6_000)).count(), 2);
    }

    #[tokio::test]
    async fn oversized_material_is_summarised_in_chunks() {
        let mut history = vec![ChatMessage::system("s")];
        for turn in 0..10 {
            history.push(ChatMessage::user(format!("turn {turn}")));
            history.push(ChatMessage::assistant("y".repeat(8_000)));
        }
        let mut lead = LeadProvenance::default();
        let summarizer = ScriptedSummarizer::ok();
        let settings = CompactionSettings {
            summary_max_tokens: 1_000,
            ..settings(1, 0)
        };
        let request = CompactionRequest {
            settings: &settings,
            focus: Some("the y payloads"),
            // Room for ~2 turns of material per call.
            budget_tokens: 2_000
                + 4_500
                + calibrated_text_tokens(DEFAULT_COMPACTION_PROMPT, 1.0)
                + 512,
            ratio: 1.0,
            split_current_turn: false,
        };
        let outcome = compact_history(&mut history, &mut lead, &summarizer, &request)
            .await
            .expect("chunked compaction");
        assert!(outcome.summary_calls > 1, "{outcome:?}");
        let seen = summarizer.seen();
        assert!(seen[0].contains("FOCUS REQUESTED BY THE USER"));
        assert!(seen[0].contains("the y payloads"));
        assert!(seen[1].contains("SUMMARY SO FAR"));
        assert!(history[1].content.ends_with(&format!(
            "summary-{}

{SUMMARY_END_MARKER}",
            seen.len()
        )));
        assert_eq!(history.len(), 4);
    }

    #[test]
    fn inline_image_payloads_never_reach_the_summariser() {
        let rendered = render_message(&ChatMessage::user(
            "look [IMAGE:data:image/png;base64,AAAABBBB] here",
        ));
        assert!(rendered.contains("[inline image omitted]"));
        assert!(!rendered.contains("AAAABBBB"));
    }

    #[tokio::test]
    async fn user_message_merged_onto_the_slot_is_summarised_not_carried() {
        for slot_body in [
            format!(
                "old facts

{SUMMARY_END_MARKER}

[2026-11-07 08:09:33 -06:00] merged q"
            ),
            // Written before the end marker existed.
            "old facts

[2026-11-07 08:09:33 -06:00] merged q"
                .to_string(),
        ] {
            let mut history = vec![
                ChatMessage::system("s"),
                ChatMessage::user(format!(
                    "{}

{slot_body}",
                    summary_header()
                )),
                ChatMessage::assistant("merged answer"),
            ];
            native_tool_turn(&mut history, 1);
            let mut lead = LeadProvenance::new(true, true);
            let summarizer = ScriptedSummarizer::ok();
            let settings = settings(1, 0);
            compact_history(&mut history, &mut lead, &summarizer, &request(&settings))
                .await
                .expect("compaction succeeds");
            let seen = summarizer.seen();
            assert!(seen[0].contains("merged q") && seen[0].contains("merged answer"));
            assert!(!seen[0].contains("old facts"));
            let slot = &history[1].content;
            let carried_at = slot.find("old facts").expect("carried");
            let heading_at = slot.find(LATER_TURNS_HEADING).expect("heading");
            assert!(carried_at < heading_at);
            assert!(!slot.contains("merged q"), "{slot}");
            assert_eq!(slot.matches(SUMMARY_END_MARKER).count(), 1);
        }
    }

    #[test]
    fn echoed_material_labels_are_stripped() {
        assert_eq!(
            strip_echoed_labels(
                "SUMMARY SO FAR of the older parts (extend it; keep ALL of it):

## LATER TURNS (x):
1. real"
            ),
            "1. real"
        );
        assert_eq!(
            strip_echoed_labels(
                "  1. real
SUMMARY SO FAR"
            ),
            "1. real
SUMMARY SO FAR"
        );
    }

    fn in_progress_turn() -> Vec<ChatMessage> {
        let mut history = vec![
            ChatMessage::system("s"),
            ChatMessage::user("old"),
            ChatMessage::assistant("old answer"),
            ChatMessage::user("current request"),
        ];
        for index in 0..3 {
            history.push(ChatMessage::assistant(format!(
                r#"{{"content":null,"tool_calls":[{{"id":"c{index}","name":"shell","arguments":"{{}}"}}]}}"#
            )));
            history.push(ChatMessage::tool(format!(
                r#"{{"tool_call_id":"c{index}","content":"r{index}"}}"#
            )));
        }
        history
    }

    #[tokio::test]
    async fn split_summarises_earlier_iterations_of_the_current_turn() {
        let mut history = in_progress_turn();
        let original = history.clone();
        let mut lead = LeadProvenance::default();
        let summarizer = ScriptedSummarizer::ok();
        let settings = settings(4, 0);
        let request = CompactionRequest {
            split_current_turn: true,
            ..request(&settings)
        };
        let outcome = compact_history(&mut history, &mut lead, &summarizer, &request)
            .await
            .expect("split compaction");
        assert_eq!(outcome.iterations_summarised, 2);
        assert_eq!(outcome.turns_summarised, 1);
        assert_eq!(outcome.current_turn_moved, Some((3, 2)));
        // system, summary, the request verbatim, then its latest iteration.
        assert_eq!(history.len(), 5);
        assert_eq!(history[2].content, "current request");
        assert_eq!(rows(&history[3..]), rows(&original[8..]));
        let seen = summarizer.seen();
        assert!(seen[0].contains("old answer") && seen[0].contains("r0") && seen[0].contains("r1"));
        assert!(seen[0].contains("still working on"));
        assert!(!seen[0].contains("r2"));
        for (index, message) in history.iter().enumerate() {
            if message.role == "tool" {
                assert_eq!(history[index - 1].role, "assistant");
            }
        }
    }

    #[test]
    fn split_needs_two_iterations_and_is_needed_when_nothing_else_compacts() {
        let history = in_progress_turn();
        let settings = settings(4, 0);
        // Only one iteration: the split plan falls back to the whole-turn plan.
        let one_iteration = &history[..6];
        let plan = plan_compaction(one_iteration, false, 4, 0, 1.0, true).expect("plan");
        assert_eq!(plan.current_turn_head, None);
        // Older turns exist and the newest fits: no split.
        assert!(!current_turn_needs_split(
            &history, false, &settings, 1.0, 100_000
        ));
        // The newest turn alone overflows the room: split.
        assert!(current_turn_needs_split(
            &history, false, &settings, 1.0, 10
        ));
        // Summary slot + only the in-progress turn: nothing else to compact.
        let mut slot_only = vec![ChatMessage::system("s"), ChatMessage::user("[summary]")];
        slot_only.extend_from_slice(&history[3..]);
        assert!(current_turn_needs_split(
            &slot_only, true, &settings, 1.0, 100_000
        ));
        let plan = plan_compaction(&slot_only, true, 4, 0, 1.0, true).expect("split plan");
        assert_eq!(plan.current_turn_head, Some(2));
        assert_eq!(plan.turns_summarised, 0);
    }
}
