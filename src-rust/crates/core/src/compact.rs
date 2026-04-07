// cc-core: Context Compaction Module
//
// Implements automatic context compaction for managing the 200K token context window:
// - Auto-compact when approaching limit (90% threshold)
// - Micro-compact for proactive management (75% threshold)
// - Full conversation summarization
// - Tool result truncation
// - Context collapse for extreme cases
//
// Mirrors the TypeScript compact services (autoCompact.ts, compact.ts, microCompact.ts)

use crate::types::{ContentBlock, Message, MessageContent, Role, ToolResultContent};

use crate::error::{ClaudeError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default context window size (200K tokens).
pub const DEFAULT_CONTEXT_WINDOW: u64 = 200_000;

/// Target buffer to keep free after compaction.
pub const AUTOCOMPACT_BUFFER_TOKENS: u64 = 13_000;

/// Fraction of context window at which auto-compact triggers.
pub const AUTOCOMPACT_TRIGGER_FRACTION: f64 = 0.90;

/// Fraction for proactive micro-compact.
pub const MICROCOMPACT_TRIGGER_FRACTION: f64 = 0.75;

/// Warning threshold at 80%.
pub const WARNING_THRESHOLD_FRACTION: f64 = 0.80;

/// Critical threshold at 95%.
pub const CRITICAL_THRESHOLD_FRACTION: f64 = 0.95;

/// How many recent messages to preserve during compaction.
pub const KEEP_RECENT_MESSAGES: usize = 10;

/// Max consecutive auto-compact failures before disabling.
pub const MAX_CONSECUTIVE_FAILURES: u32 = 3;

/// Default summary target tokens.
pub const DEFAULT_SUMMARY_TARGET_TOKENS: usize = 2048;

/// Message indicating truncated tool results.
pub const TIME_BASED_MC_CLEARED_MESSAGE: &str = "[Old tool result content cleared]";

/// Preamble to prevent tool use during compaction.
pub const NO_TOOLS_PREAMBLE: &str = r#"You are being asked to summarize a conversation history.
DO NOT invoke any tools. Only produce a text summary.
"#;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Tracks auto-compact state across the session.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct AutoCompactState {
    /// Total compactions performed this session.
    pub compaction_count: u32,
    /// Consecutive failures (reset on success).
    pub consecutive_failures: u32,
    /// Whether the circuit breaker is open.
    pub disabled: bool,
    /// Last compaction timestamp.
    pub last_compaction_at: Option<chrono::DateTime<chrono::Utc>>,
    /// Last compaction reason.
    pub last_reason: Option<CompactionReason>,
}

impl AutoCompactState {
    /// Create a new auto-compact state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a successful compaction.
    pub fn on_success(&mut self, reason: CompactionReason) {
        self.compaction_count += 1;
        self.consecutive_failures = 0;
        self.last_compaction_at = Some(chrono::Utc::now());
        self.last_reason = Some(reason);
    }

    /// Record a failed compaction.
    pub fn on_failure(&mut self) {
        self.consecutive_failures += 1;
        if self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES {
            tracing::warn!(
                failures = self.consecutive_failures,
                "Auto-compact circuit breaker opened - disabling for this session"
            );
            self.disabled = true;
        }
    }

    /// Check if auto-compact is currently enabled.
    pub fn is_enabled(&self) -> bool {
        !self.disabled
    }

    /// Get time since last compaction.
    pub fn time_since_last(&self) -> Option<chrono::Duration> {
        self.last_compaction_at.map(|last| chrono::Utc::now() - last)
    }
}

/// Reasons for compaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompactionReason {
    /// Automatic compaction at threshold.
    AutoCompactThreshold,
    /// Context window would be exceeded.
    ContextWindowExceeded,
    /// Reactive compaction after error.
    ReactiveCompact,
    /// User explicitly requested compaction.
    UserRequested,
    /// Time-based micro-compact (idle gap).
    TimeBased,
    /// API-native cache edits.
    CacheEdits,
}

impl std::fmt::Display for CompactionReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompactionReason::AutoCompactThreshold => write!(f, "auto-compact threshold"),
            CompactionReason::ContextWindowExceeded => write!(f, "context window exceeded"),
            CompactionReason::ReactiveCompact => write!(f, "reactive compact"),
            CompactionReason::UserRequested => write!(f, "user requested"),
            CompactionReason::TimeBased => write!(f, "time-based"),
            CompactionReason::CacheEdits => write!(f, "cache edits"),
        }
    }
}

/// Result of a compaction operation.
#[derive(Debug, Clone)]
pub struct CompactResult {
    /// Messages after compaction.
    pub messages: Vec<Message>,
    /// Summary message that was added.
    pub summary_message: Option<Message>,
    /// Number of messages removed.
    pub messages_removed: usize,
    /// Number of messages preserved.
    pub messages_preserved: usize,
    /// Estimated tokens saved.
    pub tokens_saved: u64,
    /// Reason for compaction.
    pub reason: CompactionReason,
}

/// A group of messages forming a semantic unit (one API round).
#[derive(Debug, Clone)]
pub struct MessageGroup {
    /// Messages in this group.
    pub messages: Vec<Message>,
    /// Topic hint extracted from the group.
    pub topic_hint: Option<String>,
    /// Estimated token count.
    pub token_estimate: usize,
}

impl MessageGroup {
    /// Create a message group from messages.
    pub fn from_messages(messages: Vec<Message>) -> Self {
        let topic_hint = extract_topic_hint(&messages);
        let token_estimate = estimate_tokens_for_messages(&messages);
        Self { messages, topic_hint, token_estimate }
    }
}

/// Configuration for micro-compaction.
#[derive(Debug, Clone, Copy)]
pub struct MicroCompactConfig {
    /// Trigger threshold fraction.
    pub trigger_threshold: f64,
    /// Keep this many recent messages.
    pub keep_recent_messages: usize,
    /// Target token count for summary.
    pub summary_target_tokens: usize,
}

impl Default for MicroCompactConfig {
    fn default() -> Self {
        Self {
            trigger_threshold: MICROCOMPACT_TRIGGER_FRACTION,
            keep_recent_messages: KEEP_RECENT_MESSAGES,
            summary_target_tokens: DEFAULT_SUMMARY_TARGET_TOKENS,
        }
    }
}

/// Configuration for full compaction.
#[derive(Debug, Clone, Copy)]
pub struct CompactConfig {
    /// Trigger threshold fraction.
    pub trigger_threshold: f64,
    /// Keep this many recent messages.
    pub keep_recent_messages: usize,
    /// Target buffer after compaction.
    pub target_buffer_tokens: u64,
}

impl Default for CompactConfig {
    fn default() -> Self {
        Self {
            trigger_threshold: AUTOCOMPACT_TRIGGER_FRACTION,
            keep_recent_messages: KEEP_RECENT_MESSAGES,
            target_buffer_tokens: AUTOCOMPACT_BUFFER_TOKENS,
        }
    }
}

/// Token warning state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenWarningState {
    /// Below 80% - no warning.
    Ok,
    /// 80-95% - yellow warning.
    Warning,
    /// Above 95% - red critical.
    Critical,
}

impl TokenWarningState {
    /// Check if compaction is strongly recommended.
    pub fn should_compact(&self) -> bool {
        matches!(self, TokenWarningState::Critical)
    }

    /// Get display color hint.
    pub fn color_hint(&self) -> &'static str {
        match self {
            TokenWarningState::Ok => "green",
            TokenWarningState::Warning => "yellow",
            TokenWarningState::Critical => "red",
        }
    }
}

/// Triggers for compaction.
#[derive(Debug, Clone)]
pub struct CompactTrigger {
    /// Whether compaction should trigger.
    pub should_compact: bool,
    /// Current token usage fraction.
    pub fraction_used: f64,
    /// Estimated tokens.
    pub estimated_tokens: u64,
    /// Context window size.
    pub context_window: u64,
    /// Recommended strategy.
    pub recommended_strategy: CompactionStrategy,
}

/// Compaction strategies.
/// Compaction strategies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactionStrategy {
    /// No compaction needed.
    None,
    /// Micro-compact (tool results only).
    MicroCompact,
    /// Partial compaction (keep recent).
    Partial { keep_recent: usize },
    /// Tool results only.
    ToolResultsOnly,
    /// Full compaction (summarize all).
    Full,
}

/// Result of token analysis.
#[derive(Debug, Clone)]
pub struct TokenAnalysis {
    /// Total estimated tokens.
    pub total_tokens: u64,
    /// Tokens by message role.
    pub by_role: HashMap<Role, u64>,
    /// Tokens by content type.
    pub by_content_type: HashMap<ContentType, u64>,
    /// Fraction of context window used.
    pub fraction_used: f64,
    /// Warning state.
    pub warning_state: TokenWarningState,
}

/// Content types for token accounting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContentType {
    Text,
    ToolUse,
    ToolResult,
    Thinking,
    Image,
    Document,
}

/// Context collapse configuration.
#[derive(Debug, Clone, Copy)]
pub struct ContextCollapseConfig {
    /// Trigger at 97% usage.
    pub trigger_fraction: f64,
    /// Aggressive mode - collapse more.
    pub aggressive: bool,
}

impl Default for ContextCollapseConfig {
    fn default() -> Self {
        Self {
            trigger_fraction: 0.97,
            aggressive: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Token Estimation
// ---------------------------------------------------------------------------

/// Rough token estimation: chars / 4 * 4/3 padding.
pub fn estimate_tokens_for_messages(messages: &[Message]) -> usize {
    let chars: usize = messages.iter().map(|m| estimate_message_chars(m)).sum();
    (chars / 4) * 4 / 3
}

/// Estimate characters in a message.
pub fn estimate_message_chars(message: &Message) -> usize {
    match &message.content {
        MessageContent::Text(t) => t.len(),
        MessageContent::Blocks(blocks) => blocks.iter().map(estimate_block_chars).sum(),
    }
}

fn estimate_block_chars(block: &ContentBlock) -> usize {
    match block {
        ContentBlock::Text { text } => text.len(),
        ContentBlock::ToolUse { name, input, .. } => {
            name.len() + input.to_string().len()
        }
        ContentBlock::ToolResult { content, .. } => match content {
            ToolResultContent::Text(t) => t.len(),
            ToolResultContent::Blocks(blocks) => blocks.iter().map(estimate_block_chars).sum(),
        },
        ContentBlock::Thinking { thinking, .. } => thinking.len(),
        ContentBlock::RedactedThinking { data } => data.len(),
        ContentBlock::Image { .. } => 2000,
        ContentBlock::Document { .. } => 2000,
        _ => 100,
    }
}

/// Get context window size for a model.
pub fn context_window_for_model(model: &str) -> u64 {
    let model_lower = model.to_lowercase();
    
    // Claude 4 family
    if model_lower.contains("claude-opus-4")
        || model_lower.contains("claude-sonnet-4")
        || model_lower.contains("claude-haiku-4")
    {
        return 200_000;
    }
    
    // Claude 3.5 family
    if model_lower.contains("claude-3-5") {
        return 200_000;
    }
    
    // Claude 3 family
    if model_lower.contains("claude-3-opus")
        || model_lower.contains("claude-3-sonnet")
        || model_lower.contains("claude-3-haiku")
    {
        return 200_000;
    }
    
    // Default to 200K
    DEFAULT_CONTEXT_WINDOW
}

// ---------------------------------------------------------------------------
// Message Grouping
// ---------------------------------------------------------------------------

/// Group messages by API round (assistant message boundaries).
pub fn group_messages_by_api_round(messages: &[Message]) -> Vec<MessageGroup> {
    let mut groups: Vec<MessageGroup> = Vec::new();
    let mut current: Vec<Message> = Vec::new();

    for msg in messages {
        if msg.role == Role::Assistant {
            current.push(msg.clone());
            groups.push(MessageGroup::from_messages(current.clone()));
            current.clear();
        } else {
            current.push(msg.clone());
        }
    }

    // Trailing non-assistant messages (shouldn't happen in practice)
    if !current.is_empty() {
        groups.push(MessageGroup::from_messages(current));
    }

    groups
}

/// Extract topic hint from a group of messages.
pub fn extract_topic_hint(messages: &[Message]) -> Option<String> {
    for msg in messages {
        let blocks = match &msg.content {
            MessageContent::Blocks(b) => b,
            _ => continue,
        };
        
        for block in blocks {
            match block {
                ContentBlock::ToolUse { name, input, .. } => {
                    // Try file_path first
                    if let Some(fp) = input.get("file_path").and_then(|v| v.as_str()) {
                        return Some(fp.to_string());
                    }
                    // Then command
                    if let Some(cmd) = input.get("command").and_then(|v| v.as_str()) {
                        let first_word = cmd.split_whitespace().next().unwrap_or(cmd);
                        return Some(first_word.to_string());
                    }
                    return Some(name.clone());
                }
                ContentBlock::Text { text } if text.len() < 200 => {
                    return Some(text.chars().take(50).collect());
                }
                _ => {}
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Compaction Logic
// ---------------------------------------------------------------------------

/// Check if auto-compact should trigger.
pub fn should_auto_compact(
    estimated_tokens: u64,
    context_window: u64,
    config: &CompactConfig,
) -> bool {
    let fraction = estimated_tokens as f64 / context_window as f64;
    fraction >= config.trigger_threshold
}

/// Check if compaction is recommended based on token usage.
pub fn should_compact(
    estimated_tokens: u64,
    context_window: u64,
) -> CompactTrigger {
    let fraction = if context_window == 0 {
        0.0
    } else {
        estimated_tokens as f64 / context_window as f64
    };

    let strategy = if fraction >= AUTOCOMPACT_TRIGGER_FRACTION {
        CompactionStrategy::Partial { keep_recent: KEEP_RECENT_MESSAGES }
    } else if fraction >= MICROCOMPACT_TRIGGER_FRACTION {
        CompactionStrategy::MicroCompact
    } else {
        CompactionStrategy::None
    };

    CompactTrigger {
        should_compact: fraction >= AUTOCOMPACT_TRIGGER_FRACTION,
        fraction_used: fraction,
        estimated_tokens,
        context_window,
        recommended_strategy: strategy,
    }
}

/// Calculate token warning state.
pub fn calculate_token_warning_state(
    estimated_tokens: u64,
    context_window: u64,
) -> TokenWarningState {
    let fraction = if context_window == 0 {
        0.0
    } else {
        estimated_tokens as f64 / context_window as f64
    };

    if fraction >= CRITICAL_THRESHOLD_FRACTION {
        TokenWarningState::Critical
    } else if fraction >= WARNING_THRESHOLD_FRACTION {
        TokenWarningState::Warning
    } else {
        TokenWarningState::Ok
    }
}

/// Calculate index at which to split messages for compaction.
pub fn calculate_messages_to_keep_index(
    messages: &[Message],
    last_summarized_index: Option<usize>,
    min_tokens: u64,
    max_tokens: u64,
) -> usize {
    if messages.len() < KEEP_RECENT_MESSAGES * 2 {
        return 0;
    }

    let start_from = last_summarized_index.unwrap_or(0);
    let total_tokens: u64 = messages[start_from..]
        .iter()
        .map(|m| estimate_message_chars(m) as u64 / 4)
        .sum();

    if total_tokens < min_tokens {
        return 0;
    }

    // Binary search for the right split point
    let mut low = start_from;
    let mut high = messages.len().saturating_sub(KEEP_RECENT_MESSAGES);
    let mut best = low;

    while low <= high {
        let mid = (low + high) / 2;
        let tokens: u64 = messages[start_from..mid]
            .iter()
            .map(|m| estimate_message_chars(m) as u64 / 4)
            .sum();

        if tokens >= min_tokens && tokens <= max_tokens {
            best = mid;
            break;
        } else if tokens < min_tokens {
            low = mid + 1;
            best = mid;
        } else {
            if mid > 0 {
                high = mid - 1;
            } else {
                break;
            }
        }
    }

    // Ensure we don't break API invariants (orphaned tool_use without tool_result)
    adjust_index_to_preserve_invariants(messages, best)
}

/// Adjust index to preserve API invariants (no orphaned tool_use).
pub fn adjust_index_to_preserve_invariants(messages: &[Message], mut index: usize) -> usize {
    // Walk back to find a safe cut point
    while index > 0 && index < messages.len() {
        let prev_msg = &messages[index - 1];
        let curr_msg = &messages[index];

        // Don't cut between tool_use and tool_result
        let prev_has_tool_use = prev_msg.get_tool_use_blocks().len() > 0;
        let curr_has_tool_result = curr_msg.get_tool_result_blocks().len() > 0;

        if prev_has_tool_use && !curr_has_tool_result {
            // Would orphan tool_use, move back
            index -= 1;
        } else {
            break;
        }
    }

    index
}

/// Generate a prompt for compact summarization.
pub fn get_compact_prompt(
    groups: &[MessageGroup],
    is_partial: bool,
) -> String {
    let base = if is_partial {
        "Summarize the following conversation history (partial):\n\n"
    } else {
        "Summarize the following conversation history:\n\n"
    };

    let mut prompt = NO_TOOLS_PREAMBLE.to_string();
    prompt.push_str(base);

    for (i, group) in groups.iter().enumerate() {
        prompt.push_str(&format!("Round {}:\n", i + 1));
        
        if let Some(ref hint) = group.topic_hint {
            prompt.push_str(&format!("  Topic: {}\n", hint));
        }
        
        // Summarize each message in the group
        for msg in &group.messages {
            match msg.role {
                Role::User => {
                    if let Some(text) = msg.get_text() {
                        let preview: String = text.chars().take(200).collect();
                        prompt.push_str(&format!("  User: {}...\n", preview));
                    }
                }
                Role::Assistant => {
                    let tool_count = msg.get_tool_use_blocks().len();
                    if tool_count > 0 {
                        prompt.push_str(&format!("  Assistant: used {} tool(s)\n", tool_count));
                    } else if let Some(text) = msg.get_text() {
                        let preview: String = text.chars().take(200).collect();
                        prompt.push_str(&format!("  Assistant: {}...\n", preview));
                    }
                }
            }
        }
        prompt.push('\n');
    }

    prompt.push_str("\nProvide a concise summary of the key points, decisions, and current state.");
    prompt
}

/// Format a compact summary as a message.
pub fn format_compact_summary(summary: &str) -> Message {
    let content = format!(
        "<compact_summary>\n{}\n</compact_summary>",
        summary
    );
    Message::user(content)
}

// ---------------------------------------------------------------------------
// Micro-compact (tool result only)
// ---------------------------------------------------------------------------

/// Clear tool results from older messages to save tokens.
pub fn micro_compact_messages(
    messages: &mut Vec<Message>,
    keep_recent: usize,
) -> usize {
    if messages.len() <= keep_recent {
        return 0;
    }

    let split_at = messages.len().saturating_sub(keep_recent);
    let mut cleared = 0usize;

    for msg in messages.iter_mut().take(split_at) {
        if msg.role != Role::User {
            continue;
        }
        
        let blocks = match &mut msg.content {
            MessageContent::Blocks(b) => b,
            _ => continue,
        };

        for block in blocks.iter_mut() {
            if let ContentBlock::ToolResult { content, .. } = block {
                *content = ToolResultContent::Text(TIME_BASED_MC_CLEARED_MESSAGE.to_string());
                cleared += 1;
            }
        }
    }

    cleared
}

/// Evaluate if time-based micro-compact should trigger.
pub fn evaluate_time_based_trigger(
    messages: &[Message],
    last_activity: Option<std::time::Instant>,
    gap_threshold_minutes: u64,
) -> bool {
    let Some(last) = last_activity else {
        return false;
    };
    
    let elapsed = last.elapsed().as_secs() / 60;
    elapsed >= gap_threshold_minutes && messages.len() > KEEP_RECENT_MESSAGES
}

// ---------------------------------------------------------------------------
// Full Compaction
// ---------------------------------------------------------------------------

/// Perform full conversation compaction.
/// 
/// This is a placeholder that would normally call an LLM to generate a summary.
/// In the real implementation, this would make an API call.
pub fn compact_conversation(
    messages: &[Message],
    keep_recent: usize,
) -> Result<CompactResult> {
    if messages.len() <= keep_recent {
        return Err(ClaudeError::Other(
            "Not enough messages to compact".to_string()
        ));
    }

    let groups = group_messages_by_api_round(messages);
    let split_at = groups.len().saturating_sub(keep_recent);
    
    // Calculate tokens in groups to be compacted
    let tokens_to_compact: usize = groups[..split_at]
        .iter()
        .map(|g| g.token_estimate)
        .sum();

    // Build summary (placeholder)
    let summary = build_summary_from_groups(&groups[..split_at]);
    let summary_message = format_compact_summary(&summary);

    // Build new message list
    let mut new_messages = vec![summary_message.clone()];
    
    // Add preserved messages
    for group in &groups[split_at..] {
        new_messages.extend(group.messages.clone());
    }

    let removed = messages.len() - new_messages.len() + 1; // +1 for summary

    Ok(CompactResult {
        messages: new_messages,
        summary_message: Some(summary_message),
        messages_removed: removed,
        messages_preserved: messages.len() - removed,
        tokens_saved: tokens_to_compact as u64,
        reason: CompactionReason::AutoCompactThreshold,
    })
}

/// Build a summary from message groups (placeholder implementation).
fn build_summary_from_groups(groups: &[MessageGroup]) -> String {
    let mut summary = String::new();
    summary.push_str("Previous conversation summary:\n\n");

    // Count statistics
    let user_count = groups.iter().map(|g| g.messages.iter().filter(|m| m.role == Role::User).count()).sum::<usize>();
    let assistant_count = groups.iter().map(|g| g.messages.iter().filter(|m| m.role == Role::Assistant).count()).sum::<usize>();
    let tool_uses: usize = groups.iter().map(|g| g.messages.iter().map(|m| m.get_tool_use_blocks().len()).sum::<usize>()).sum();

    summary.push_str(&format!("- {} user messages\n", user_count));
    summary.push_str(&format!("- {} assistant responses\n", assistant_count));
    if tool_uses > 0 {
        summary.push_str(&format!("- {} tool invocations\n", tool_uses));
    }

    // Extract key topics
    summary.push_str("\nKey topics:\n");
    let mut seen_topics = std::collections::HashSet::new();
    for (i, group) in groups.iter().enumerate().take(5) {
        if let Some(ref hint) = group.topic_hint {
            if seen_topics.insert(hint.clone()) {
                summary.push_str(&format!("{}. {}\n", i + 1, hint));
            }
        }
    }

    summary
}

// ---------------------------------------------------------------------------
// Context Collapse
// ---------------------------------------------------------------------------

/// Perform context collapse - extreme compaction for near-limit scenarios.
pub fn context_collapse(
    messages: &mut Vec<Message>,
    config: &ContextCollapseConfig,
) -> Result<usize> {
    let estimated = estimate_tokens_for_messages(messages);
    let limit = (DEFAULT_CONTEXT_WINDOW as f64 * config.trigger_fraction) as usize;
    
    if estimated < limit {
        return Ok(0);
    }

    // Aggressive mode: clear all tool results except most recent
    let keep_recent = if config.aggressive { 3 } else { KEEP_RECENT_MESSAGES };
    let cleared = micro_compact_messages(messages, keep_recent);

    // If still over limit, remove thinking blocks
    if config.aggressive {
        for msg in messages.iter_mut() {
            if let MessageContent::Blocks(blocks) = &mut msg.content {
                blocks.retain(|b| !matches!(b, ContentBlock::Thinking { .. }));
            }
        }
    }

    Ok(cleared)
}

/// Check if context collapse should trigger.
pub fn should_context_collapse(
    estimated_tokens: u64,
    context_window: u64,
) -> bool {
    let fraction = estimated_tokens as f64 / context_window as f64;
    fraction >= 0.97
}

// ---------------------------------------------------------------------------
// Snip Compact (history snip)
// ---------------------------------------------------------------------------

/// Snip messages from a specific point (for resume/rewind).
pub fn snip_compact(
    messages: &mut Vec<Message>,
    snip_after_uuid: &str,
) -> usize {
    if let Some(pos) = messages.iter().position(|m| {
        m.uuid.as_ref().map(|u| u == snip_after_uuid).unwrap_or(false)
    }) {
        let removed = messages.len() - pos - 1;
        messages.truncate(pos + 1);
        removed
    } else {
        0
    }
}

// ---------------------------------------------------------------------------
// Auto-compact orchestration
// ---------------------------------------------------------------------------

/// Check and perform auto-compact if needed.
pub fn auto_compact_if_needed(
    messages: &mut Vec<Message>,
    model: &str,
    state: &mut AutoCompactState,
    config: &CompactConfig,
) -> Result<Option<CompactResult>> {
    if !state.is_enabled() {
        return Ok(None);
    }

    let estimated = estimate_tokens_for_messages(messages) as u64;
    let window = context_window_for_model(model);

    if !should_auto_compact(estimated, window, config) {
        return Ok(None);
    }

    match compact_conversation(messages, config.keep_recent_messages) {
        Ok(result) => {
            *messages = result.messages.clone();
            state.on_success(CompactionReason::AutoCompactThreshold);
            Ok(Some(result))
        }
        Err(e) => {
            state.on_failure();
            Err(e)
        }
    }
}

/// Reactive compact - called after a context window error.
pub fn reactive_compact(
    messages: &mut Vec<Message>,
) -> Result<CompactResult> {
    // More aggressive - keep fewer messages
    let keep_recent = KEEP_RECENT_MESSAGES / 2;
    
    let mut result = compact_conversation(messages, keep_recent)?;
    result.reason = CompactionReason::ReactiveCompact;
    
    *messages = result.messages.clone();
    Ok(result)
}

// ---------------------------------------------------------------------------
// Token Analysis
// ---------------------------------------------------------------------------

/// Analyze token usage breakdown.
pub fn analyze_tokens(messages: &[Message]) -> TokenAnalysis {
    let mut by_role: HashMap<Role, u64> = HashMap::new();
    let mut by_content_type: HashMap<ContentType, u64> = HashMap::new();
    let mut total = 0u64;

    for msg in messages {
        let msg_tokens = estimate_message_chars(msg) as u64 / 4;
        total += msg_tokens;
        
        *by_role.entry(msg.role.clone()).or_insert(0) += msg_tokens;

        if let MessageContent::Blocks(blocks) = &msg.content {
            for block in blocks {
                let block_tokens = (estimate_block_chars(block) / 4) as u64;
                let content_type = match block {
                    ContentBlock::Text { .. } => ContentType::Text,
                    ContentBlock::ToolUse { .. } => ContentType::ToolUse,
                    ContentBlock::ToolResult { .. } => ContentType::ToolResult,
                    ContentBlock::Thinking { .. } => ContentType::Thinking,
                    ContentBlock::Image { .. } => ContentType::Image,
                    ContentBlock::Document { .. } => ContentType::Document,
                    _ => ContentType::Text,
                };
                *by_content_type.entry(content_type).or_insert(0) += block_tokens;
            }
        }
    }

    let fraction = total as f64 / DEFAULT_CONTEXT_WINDOW as f64;
    let warning_state = calculate_token_warning_state(total, DEFAULT_CONTEXT_WINDOW);

    TokenAnalysis {
        total_tokens: total,
        by_role,
        by_content_type,
        fraction_used: fraction,
        warning_state,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_estimation() {
        let msg = Message::user("Hello, world!");
        let chars = estimate_message_chars(&msg);
        assert!(chars > 0);
        
        let tokens = estimate_tokens_for_messages(&[msg]);
        assert!(tokens > 0);
    }

    #[test]
    fn test_context_window_for_model() {
        assert_eq!(context_window_for_model("claude-sonnet-4-6"), 200_000);
        assert_eq!(context_window_for_model("claude-3-5-sonnet"), 200_000);
        assert_eq!(context_window_for_model("claude-3-opus"), 200_000);
        assert_eq!(context_window_for_model("unknown"), DEFAULT_CONTEXT_WINDOW);
    }

    #[test]
    fn test_message_grouping() {
        let messages = vec![
            Message::user("Hello"),
            Message::assistant("Hi there"),
            Message::user("How are you?"),
            Message::assistant("I'm well"),
        ];
        
        let groups = group_messages_by_api_round(&messages);
        assert_eq!(groups.len(), 2);
    }

    #[test]
    fn test_should_auto_compact() {
        let config = CompactConfig::default();
        
        assert!(should_auto_compact(190_000, 200_000, &config));
        assert!(!should_auto_compact(100_000, 200_000, &config));
    }

    #[test]
    fn test_token_warning_state() {
        assert_eq!(
            calculate_token_warning_state(100_000, 200_000),
            TokenWarningState::Ok
        );
        assert_eq!(
            calculate_token_warning_state(170_000, 200_000),
            TokenWarningState::Warning
        );
        assert_eq!(
            calculate_token_warning_state(195_000, 200_000),
            TokenWarningState::Critical
        );
    }

    #[test]
    fn test_micro_compact() {
        let mut messages = vec![
            Message::user("Hello"),
            Message::user_blocks(vec![ContentBlock::ToolResult {
                tool_use_id: "1".to_string(),
                content: ToolResultContent::Text("Result".to_string()),
                is_error: Some(false),
            }]),
            Message::user("More"),
            Message::assistant("Response"),
        ];
        
        let cleared = micro_compact_messages(&mut messages, 2);
        assert_eq!(cleared, 1);
    }

    #[test]
    fn test_calculate_keep_index() {
        let messages: Vec<Message> = (0..20)
            .map(|i| Message::user(format!("Message {}", i).repeat(50)))
            .collect();
        
        let index = calculate_messages_to_keep_index(&messages, None, 1000, 5000);
        // Should keep some messages
        assert!(index < 20);
    }

    #[test]
    fn test_compact_result_serialization() {
        let state = AutoCompactState {
            compaction_count: 5,
            consecutive_failures: 0,
            disabled: false,
            last_compaction_at: Some(chrono::Utc::now()),
            last_reason: Some(CompactionReason::AutoCompactThreshold),
        };
        
        let json = serde_json::to_string(&state).unwrap();
        assert!(json.contains("compaction_count"));
    }

    #[test]
    fn test_snip_compact() {
        let mut messages = vec![
            Message { role: Role::User, content: MessageContent::Text("First".into()), uuid: Some("1".into()), cost: None },
            Message { role: Role::User, content: MessageContent::Text("Second".into()), uuid: Some("2".into()), cost: None },
            Message { role: Role::User, content: MessageContent::Text("Third".into()), uuid: Some("3".into()), cost: None },
        ];
        
        let removed = snip_compact(&mut messages, "1");
        assert_eq!(removed, 2); // Removes messages after uuid "1"
        assert_eq!(messages.len(), 1);
    }

    #[test]
    fn test_analyze_tokens() {
        let messages = vec![
            Message::user("Hello world"),
            Message::assistant("Response"),
        ];
        
        let analysis = analyze_tokens(&messages);
        assert!(analysis.total_tokens > 0);
        assert!(analysis.by_role.contains_key(&Role::User));
        assert!(analysis.by_role.contains_key(&Role::Assistant));
    }

    #[test]
    fn test_compact_circuit_breaker() {
        let mut state = AutoCompactState::new();
        
        // Simulate failures
        for _ in 0..MAX_CONSECUTIVE_FAILURES {
            state.on_failure();
        }
        
        assert!(state.disabled);
        assert!(!state.is_enabled());
    }
}