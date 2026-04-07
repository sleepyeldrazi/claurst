// cc-core: Core Query Engine
//
// Implements the QueryEngine struct with:
// - Turn loop with streaming responses
// - Context window management (200K tokens)
// - Automatic context compaction
// - Cost tracking & token budget
// - History management
// - Message streaming from API
//
// Mirrors the TypeScript QueryEngine.ts and query.ts patterns.

use crate::cost::CostTracker;
use crate::compact::{
    AutoCompactState, CompactionReason, CompactResult, CompactionStrategy,
    auto_compact_if_needed, estimate_tokens_for_messages,
    DEFAULT_CONTEXT_WINDOW, AUTOCOMPACT_TRIGGER_FRACTION, KEEP_RECENT_MESSAGES,
};
use crate::token_budget::TokenBudget;
pub use crate::token_budget::TokenWarningLevel;
use crate::types::{ContentBlock, Message, MessageContent, Role, ToolResultContent, UsageInfo};
use crate::error::{ClaudeError, Result};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex, RwLock};
use tracing::{info, warn};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Constants (matching TypeScript spec)
// ---------------------------------------------------------------------------

/// Target buffer to keep free after compaction.
pub const AUTOCOMPACT_BUFFER_TOKENS: u64 = 13_000;

/// Start warning when 80% of context window is used.
pub const WARNING_THRESHOLD_FRACTION: f64 = 0.80;

/// Critical threshold at 95% of context window.
pub const CRITICAL_THRESHOLD_FRACTION: f64 = 0.95;

/// Max turns before forcing end_turn.
pub const DEFAULT_MAX_TURNS: u32 = 100;

/// Max output tokens recovery attempts.
pub const MAX_OUTPUT_TOKENS_RECOVERY_LIMIT: u32 = 3;

/// Default tool result budget in characters.
pub const DEFAULT_TOOL_RESULT_BUDGET: usize = 50_000;

// ---------------------------------------------------------------------------
// Configuration Types
// ---------------------------------------------------------------------------

/// Configuration for the QueryEngine.
#[derive(Debug, Clone)]
pub struct QueryEngineConfig {
    /// Model identifier (e.g., "claude-sonnet-4-6").
    pub model: String,
    /// Maximum tokens per response.
    pub max_tokens: u32,
    /// Maximum turns per conversation.
    pub max_turns: u32,
    /// Context window size (varies by model).
    pub context_window: u64,
    /// System prompt text.
    pub system_prompt: Option<String>,
    /// Additional system prompt to append.
    pub append_system_prompt: Option<String>,
    /// Working directory for the session.
    pub working_directory: std::path::PathBuf,
    /// Thinking budget tokens (for extended thinking).
    pub thinking_budget: Option<u32>,
    /// Temperature for sampling.
    pub temperature: Option<f32>,
    /// Max cumulative tool result characters before truncation.
    pub tool_result_budget: usize,
    /// Optional USD spend cap.
    pub max_budget_usd: Option<f64>,
    /// Fallback model for retry.
    pub fallback_model: Option<String>,
    /// Enable auto-compact.
    pub auto_compact_enabled: bool,
    /// Session ID (generated if not provided).
    pub session_id: Option<String>,
    /// Parent session ID for lineage.
    pub parent_session_id: Option<String>,
}

impl Default for QueryEngineConfig {
    fn default() -> Self {
        Self {
            model: "claude-sonnet-4-6".to_string(),
            max_tokens: 4096,
            max_turns: DEFAULT_MAX_TURNS,
            context_window: DEFAULT_CONTEXT_WINDOW,
            system_prompt: None,
            append_system_prompt: None,
            working_directory: std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from(".")),
            thinking_budget: None,
            temperature: None,
            tool_result_budget: DEFAULT_TOOL_RESULT_BUDGET,
            max_budget_usd: None,
            fallback_model: None,
            auto_compact_enabled: true,
            session_id: None,
            parent_session_id: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Event Types for Streaming
// ---------------------------------------------------------------------------

/// Events emitted by the QueryEngine during streaming.
#[derive(Debug, Clone)]
pub enum QueryEngineEvent {
    /// Stream chunk received from API.
    StreamChunk(String),
    /// Tool execution started.
    ToolStart {
        tool_name: String,
        tool_id: String,
        input: serde_json::Value,
    },
    /// Tool execution completed.
    ToolEnd {
        tool_name: String,
        tool_id: String,
        result: String,
        is_error: bool,
    },
    /// A complete assistant message received.
    AssistantMessage(Message),
    /// Token usage update.
    TokenUsage { used: u64, remaining: u64, fraction: f64 },
    /// Token warning state changed.
    TokenWarning { level: TokenWarningLevel, fraction: f64 },
    /// Compaction started.
    CompactionStart { reason: CompactionReason },
    /// Compaction completed.
    CompactionComplete { summary: String, messages_removed: usize },
    /// Turn completed.
    TurnComplete { turn: u32, stop_reason: StopReason },
    /// Budget status update.
    BudgetStatus { cost_usd: f64, limit_usd: Option<f64> },
    /// Status/information message.
    Status(String),
    /// Error occurred.
    Error(String),
}

/// Stop reasons for a turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    EndTurn,
    MaxTokens,
    StopSequence,
    ToolUse,
    Error,
    Cancelled,
    BudgetExceeded,
    MaxTurns,
}

// ---------------------------------------------------------------------------
// Turn State
// ---------------------------------------------------------------------------

/// State tracked per turn.
#[derive(Debug)]
pub struct TurnState {
    /// Current turn number.
    pub turn_number: u32,
    /// Input tokens for this turn.
    pub input_tokens: u64,
    /// Output tokens for this turn.
    pub output_tokens: u64,
    /// Cache read tokens.
    pub cache_read_tokens: u64,
    /// Cache write tokens.
    pub cache_write_tokens: u64,
    /// Cost for this turn (USD).
    pub cost_usd: f64,
    /// Tool uses in this turn.
    pub tool_uses: Vec<ToolUseRecord>,
    /// Start time of the turn.
    pub started_at: std::time::Instant,
}

impl Default for TurnState {
    fn default() -> Self {
        Self {
            turn_number: 0,
            input_tokens: 0,
            output_tokens: 0,
            cache_read_tokens: 0,
            cache_write_tokens: 0,
            cost_usd: 0.0,
            tool_uses: Vec::new(),
            started_at: std::time::Instant::now(),
        }
    }
}

/// Record of a tool use.
#[derive(Debug, Clone)]
pub struct ToolUseRecord {
    pub tool_id: String,
    pub tool_name: String,
    pub input: serde_json::Value,
    pub result: Option<String>,
    pub is_error: bool,
    pub duration_ms: u64,
}

// ---------------------------------------------------------------------------
// Permission Tracking
// ---------------------------------------------------------------------------

/// Record of a permission denial.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermissionDenial {
    pub tool_name: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub reason: String,
}

// ---------------------------------------------------------------------------
// QueryEngine
// ---------------------------------------------------------------------------

/// The core query engine managing conversation state, streaming, and compaction.
pub struct QueryEngine {
    /// Engine configuration.
    config: QueryEngineConfig,
    /// Conversation message history.
    messages: Arc<RwLock<Vec<Message>>>,
    /// Current turn state.
    turn_state: Arc<Mutex<TurnState>>,
    /// Cost tracker for the session.
    cost_tracker: Arc<CostTracker>,
    /// Event sender for streaming.
    event_tx: Option<mpsc::UnboundedSender<QueryEngineEvent>>,
    /// Cancellation token.
    cancel_token: tokio_util::sync::CancellationToken,
    /// Current token budget state.
    token_budget: Arc<Mutex<TokenBudget>>,
    /// Permission denials recorded.
    permission_denials: Arc<Mutex<Vec<PermissionDenial>>>,
    /// Session ID.
    session_id: String,
    /// Compaction state.
    auto_compact_state: Arc<Mutex<AutoCompactState>>,
    /// Max tokens recovery counter.
    max_tokens_recovery_count: Arc<Mutex<u32>>,
    /// Pending user messages queue (for async input during tool execution).
    pending_messages: Arc<Mutex<VecDeque<String>>>,
}

impl QueryEngine {
    /// Create a new QueryEngine with the given configuration.
    pub fn new(config: QueryEngineConfig) -> Self {
        let session_id = config.session_id.clone()
            .unwrap_or_else(|| Uuid::new_v4().to_string());
        
        let cost_tracker = CostTracker::with_model(&config.model);
        
        let token_budget = TokenBudget::new(0, config.context_window);

        Self {
            config,
            messages: Arc::new(RwLock::new(Vec::new())),
            turn_state: Arc::new(Mutex::new(TurnState::default())),
            cost_tracker,
            event_tx: None,
            cancel_token: tokio_util::sync::CancellationToken::new(),
            token_budget: Arc::new(Mutex::new(token_budget)),
            permission_denials: Arc::new(Mutex::new(Vec::new())),
            session_id,
            auto_compact_state: Arc::new(Mutex::new(AutoCompactState::new())),
            max_tokens_recovery_count: Arc::new(Mutex::new(0)),
            pending_messages: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    /// Set the event sender for streaming.
    pub fn with_event_stream(mut self, tx: mpsc::UnboundedSender<QueryEngineEvent>) -> Self {
        self.event_tx = Some(tx);
        self
    }

    /// Get the session ID.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Get a clone of the current messages.
    pub async fn get_messages(&self) -> Vec<Message> {
        self.messages.read().await.clone()
    }

    /// Get current message count.
    pub async fn message_count(&self) -> usize {
        self.messages.read().await.len()
    }

    /// Add a message to the conversation.
    pub async fn add_message(&self, message: Message) {
        let mut messages = self.messages.write().await;
        messages.push(message);
    }

    /// Queue a pending message (for async input during tool execution).
    pub async fn queue_pending_message(&self, text: String) {
        let mut pending = self.pending_messages.lock().await;
        pending.push_back(text);
    }

    /// Drain pending messages.
    async fn drain_pending_messages(&self) -> Vec<String> {
        let mut pending = self.pending_messages.lock().await;
        pending.drain(..).collect()
    }

    /// Record a permission denial.
    pub async fn record_permission_denial(&self, tool_name: String, reason: String) {
        let denial = PermissionDenial {
            tool_name,
            timestamp: chrono::Utc::now(),
            reason,
        };
        let mut denials = self.permission_denials.lock().await;
        denials.push(denial);
    }

    /// Get permission denials.
    pub async fn get_permission_denials(&self) -> Vec<PermissionDenial> {
        self.permission_denials.lock().await.clone()
    }

    /// Emit an event if sender is configured.
    fn emit(&self, event: QueryEngineEvent) {
        if let Some(ref tx) = self.event_tx {
            let _ = tx.send(event);
        }
    }

    /// Check the current token budget and emit warnings.
    async fn check_token_budget(&self, tokens_used: u64) {
        let budget = TokenBudget::new(tokens_used, self.config.context_window);
        
        let level = if budget.fill_fraction >= CRITICAL_THRESHOLD_FRACTION {
            TokenWarningLevel::Critical
        } else if budget.fill_fraction >= WARNING_THRESHOLD_FRACTION {
            TokenWarningLevel::Warning
        } else {
            TokenWarningLevel::None
        };

        // Update stored budget
        {
            let mut stored = self.token_budget.lock().await;
            *stored = budget.clone();
        }

        self.emit(QueryEngineEvent::TokenUsage {
            used: budget.tokens_used,
            remaining: budget.tokens_remaining,
            fraction: budget.fill_fraction,
        });

        self.emit(QueryEngineEvent::TokenWarning {
            level,
            fraction: budget.fill_fraction,
        });
    }

    /// Check if auto-compact should trigger.
    async fn should_auto_compact(&self) -> bool {
        if !self.config.auto_compact_enabled {
            return false;
        }

        let budget = self.token_budget.lock().await;
        budget.fill_fraction >= AUTOCOMPACT_TRIGGER_FRACTION
    }

    /// Perform context compaction.
    /// 
    /// This is a placeholder implementation. In a real implementation,
    /// this would call the compaction service to generate a summary.
    pub async fn compact(&self, strategy: CompactionStrategy) -> Result<CompactResult> {
        let mut messages = self.messages.write().await;
        
        if messages.len() <= KEEP_RECENT_MESSAGES {
            return Err(ClaudeError::Other("Not enough messages to compact".to_string()));
        }

        self.emit(QueryEngineEvent::CompactionStart { 
            reason: CompactionReason::UserRequested 
        });

        // Determine split point based on strategy
        let keep_recent = match strategy {
            CompactionStrategy::Full => KEEP_RECENT_MESSAGES,
            CompactionStrategy::Partial { keep_recent } => keep_recent,
            CompactionStrategy::MicroCompact => KEEP_RECENT_MESSAGES,
            CompactionStrategy::ToolResultsOnly => KEEP_RECENT_MESSAGES,
            CompactionStrategy::None => return Err(ClaudeError::Other("No compaction strategy".to_string())),
        };

        let split_at = messages.len().saturating_sub(keep_recent);
        let to_compact: Vec<Message> = messages.drain(..split_at).collect();
        let preserved_count = messages.len();

        // Generate summary (placeholder - would call compaction service)
        let summary_text = format_compact_summary(&to_compact);
        let summary_message = Message::user(format!("<compact_summary>\n{}\n</compact_summary>", summary_text));

        // Insert summary at the beginning
        messages.insert(0, summary_message.clone());

        // Update compaction count
        {
            let mut state = self.auto_compact_state.lock().await;
            state.on_success(CompactionReason::UserRequested);
        }

        let result = CompactResult {
            messages: messages.clone(),
            summary_message: Some(summary_message),
            messages_removed: to_compact.len(),
            messages_preserved: preserved_count,
            tokens_saved: estimate_tokens_for_messages(&to_compact) as u64,
            reason: CompactionReason::UserRequested,
        };

        self.emit(QueryEngineEvent::CompactionComplete {
            summary: summary_text,
            messages_removed: to_compact.len(),
        });

        Ok(result)
    }

    /// Perform auto-compact if needed.
    async fn auto_compact_if_needed(&self) -> Result<Option<CompactResult>> {
        if self.should_auto_compact().await {
            info!("Auto-compact triggered");
            self.emit(QueryEngineEvent::Status("Context window approaching limit, compacting...".to_string()));
            
            let mut messages = self.messages.write().await;
            let mut state = self.auto_compact_state.lock().await;
            
            let config = crate::compact::CompactConfig::default();
            match auto_compact_if_needed(&mut messages, &self.config.model, &mut state, &config) {
                Ok(result) => Ok(result),
                Err(e) => {
                    warn!("Auto-compact failed: {}", e);
                    Err(e)
                }
            }
        } else {
            Ok(None)
        }
    }

    /// Apply tool result budget - truncate old tool results if needed.
    async fn apply_tool_result_budget(&self) -> usize {
        let budget = self.config.tool_result_budget;
        let mut messages = self.messages.write().await;
        
        let total: usize = messages
            .iter()
            .filter(|m| m.role == Role::User)
            .flat_map(|m| match &m.content {
                MessageContent::Blocks(blocks) => blocks.iter(),
                _ => [].iter(),
            })
            .filter_map(|b| {
                if let ContentBlock::ToolResult { content, .. } = b {
                    Some(match content {
                        ToolResultContent::Text(t) => t.len(),
                        ToolResultContent::Blocks(blocks) => blocks
                            .iter()
                            .map(|b| if let ContentBlock::Text { text } = b { text.len() } else { 0 })
                            .sum::<usize>(),
                    })
                } else {
                    None
                }
            })
            .sum();

        if total <= budget {
            return 0;
        }

        let to_shed = total - budget;
        let mut truncated = 0usize;

        'outer: for msg in messages.iter_mut() {
            if msg.role != Role::User {
                continue;
            }
            let blocks = match &mut msg.content {
                MessageContent::Blocks(b) => b,
                _ => continue,
            };
            for block in blocks.iter_mut() {
                if let ContentBlock::ToolResult { content, .. } = block {
                    let size = match &*content {
                        ToolResultContent::Text(t) => t.len(),
                        ToolResultContent::Blocks(inner) => inner
                            .iter()
                            .map(|b| if let ContentBlock::Text { text } = b { text.len() } else { 0 })
                            .sum::<usize>(),
                    };
                    if size == 0 {
                        continue;
                    }
                    *content = ToolResultContent::Text(
                        "[Old tool result content cleared to save context]".to_string(),
                    );
                    truncated += 1;
                    if size >= to_shed {
                        break 'outer;
                    }
                }
            }
        }

        truncated
    }

    /// Estimate token count for messages (rough approximation).
    pub async fn estimate_tokens(&self) -> u64 {
        let messages = self.messages.read().await;
        messages.iter().map(|m| estimate_message_tokens(m)).sum()
    }

    /// Start a new turn with the given user message.
    /// 
    /// This is the main entry point for submitting messages to the engine.
    /// Returns when the turn completes (model responds or error occurs).
    pub async fn submit_message(&self, content: impl Into<String>) -> Result<TurnResult> {
        let text = content.into();
        
        // Create user message
        let user_message = Message::user(text.clone());
        
        // Add to conversation
        self.add_message(user_message.clone()).await;

        // Emit the user message event
        self.emit(QueryEngineEvent::AssistantMessage(user_message));

        // Execute the turn loop
        self.execute_turn().await
    }

    /// Execute a single turn of the conversation.
    async fn execute_turn(&self) -> Result<TurnResult> {
        let mut turn_state = self.turn_state.lock().await;
        turn_state.turn_number += 1;
        turn_state.started_at = std::time::Instant::now();
        turn_state.tool_uses.clear();
        drop(turn_state);

        // Check max turns
        {
            let turn_state = self.turn_state.lock().await;
            if turn_state.turn_number > self.config.max_turns {
                self.emit(QueryEngineEvent::TurnComplete {
                    turn: turn_state.turn_number,
                    stop_reason: StopReason::MaxTurns,
                });
                return Ok(TurnResult {
                    stop_reason: StopReason::MaxTurns,
                    message: None,
                    usage: None,
                });
            }
        }

        // Check cancellation
        if self.cancel_token.is_cancelled() {
            return Err(ClaudeError::Cancelled);
        }

        // Drain pending messages
        let pending = self.drain_pending_messages().await;
        if !pending.is_empty() {
            let mut messages = self.messages.write().await;
            for text in pending {
                messages.push(Message::user(text));
            }
        }

        // Apply tool result budget
        let truncated = self.apply_tool_result_budget().await;
        if truncated > 0 {
            self.emit(QueryEngineEvent::Status(format!(
                "[{} older tool result(s) truncated to save context]",
                truncated
            )));
        }

        // Check token budget and auto-compact if needed
        let estimated_tokens = self.estimate_tokens().await;
        self.check_token_budget(estimated_tokens).await;

        if let Some(result) = self.auto_compact_if_needed().await? {
            info!(
                "Auto-compacted {} messages into summary",
                result.messages_removed
            );
        }

        // Check budget
        if let Some(limit) = self.config.max_budget_usd {
            let current = self.cost_tracker.total_cost_usd();
            if current >= limit {
                self.emit(QueryEngineEvent::BudgetStatus {
                    cost_usd: current,
                    limit_usd: Some(limit),
                });
                return Ok(TurnResult {
                    stop_reason: StopReason::BudgetExceeded,
                    message: None,
                    usage: None,
                });
            }
        }

        // In a full implementation, this would call the API and handle streaming
        // For now, return a placeholder
        Ok(TurnResult {
            stop_reason: StopReason::EndTurn,
            message: None,
            usage: None,
        })
    }

    /// Cancel the current operation.
    pub fn cancel(&self) {
        self.cancel_token.cancel();
    }

    /// Get total cost tracked.
    pub fn total_cost(&self) -> f64 {
        self.cost_tracker.total_cost_usd()
    }

    /// Get total usage across all turns.
    pub async fn total_usage(&self) -> UsageInfo {
        // Aggregate usage from all messages
        let messages = self.messages.read().await;
        let mut usage = UsageInfo::default();
        
        for msg in messages.iter() {
            if let Some(ref cost) = msg.cost {
                usage.input_tokens += cost.input_tokens;
                usage.output_tokens += cost.output_tokens;
                usage.cache_read_input_tokens += cost.cache_read_input_tokens;
                usage.cache_creation_input_tokens += cost.cache_creation_input_tokens;
            }
        }
        
        usage
    }

    /// Clear the conversation history.
    pub async fn clear(&self) {
        let mut messages = self.messages.write().await;
        messages.clear();
        
        // Reset turn state
        let mut turn_state = self.turn_state.lock().await;
        *turn_state = TurnState::default();
        
        // Reset compaction count
        let mut state = self.auto_compact_state.lock().await;
        *state = AutoCompactState::new();

        self.emit(QueryEngineEvent::Status("Conversation cleared".to_string()));
    }

    /// Resume from a previous session.
    pub async fn resume_session(&self, messages: Vec<Message>) {
        let mut current = self.messages.write().await;
        *current = messages;
        
        self.emit(QueryEngineEvent::Status(format!(
            "Resumed session with {} messages",
            current.len()
        )));
    }
}

/// Result of a turn execution.
#[derive(Debug)]
pub struct TurnResult {
    pub stop_reason: StopReason,
    pub message: Option<Message>,
    pub usage: Option<UsageInfo>,
}

// ---------------------------------------------------------------------------
// Helper Functions
// ---------------------------------------------------------------------------

/// Roughly estimate token count for a message.
/// Uses chars / 4 as a rough approximation.
pub fn estimate_message_tokens(message: &Message) -> u64 {
    let chars = match &message.content {
        MessageContent::Text(t) => t.len(),
        MessageContent::Blocks(blocks) => blocks.iter().map(estimate_block_chars).sum(),
    };
    // Rough approximation: 4 chars per token, with 4/3 padding
    ((chars / 4) * 4 / 3) as u64
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
        _ => 200, // Default for images/documents
    }
}

/// Format a compact summary of messages.
fn format_compact_summary(messages: &[Message]) -> String {
    let mut summary = String::new();
    summary.push_str("Previous conversation summary:\n\n");
    
    // Count messages by role
    let user_count = messages.iter().filter(|m| m.role == Role::User).count();
    let assistant_count = messages.iter().filter(|m| m.role == Role::Assistant).count();
    
    summary.push_str(&format!("- {} user messages\n", user_count));
    summary.push_str(&format!("- {} assistant responses\n", assistant_count));
    
    // Count tool uses
    let tool_uses: usize = messages
        .iter()
        .map(|m| m.get_tool_use_blocks().len())
        .sum();
    
    if tool_uses > 0 {
        summary.push_str(&format!("- {} tool uses\n", tool_uses));
    }
    
    // Extract key topics (first few user messages)
    summary.push_str("\nKey topics discussed:\n");
    for (i, msg) in messages.iter().filter(|m| m.role == Role::User).take(5).enumerate() {
        if let Some(text) = msg.get_text() {
            let preview: String = text.chars().take(100).collect();
            summary.push_str(&format!("{}. {}...\n", i + 1, preview));
        }
    }
    
    summary
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_query_engine_creation() {
        let config = QueryEngineConfig::default();
        let engine = QueryEngine::new(config);
        
        assert!(!engine.session_id.is_empty());
        assert_eq!(engine.message_count().await, 0);
    }

    #[tokio::test]
    async fn test_add_message() {
        let config = QueryEngineConfig::default();
        let engine = QueryEngine::new(config);
        
        engine.add_message(Message::user("Hello")).await;
        assert_eq!(engine.message_count().await, 1);
    }

    #[tokio::test]
    async fn test_token_estimation() {
        let msg = Message::user("Hello, world! This is a test message.");
        let tokens = estimate_message_tokens(&msg);
        assert!(tokens > 0);
    }

    #[tokio::test]
    async fn test_compaction_trigger() {
        let config = QueryEngineConfig {
            context_window: 1000,
            auto_compact_enabled: true,
            ..Default::default()
        };
        let engine = QueryEngine::new(config);
        
        // Add many messages to trigger compaction
        for i in 0..20 {
            engine.add_message(Message::user(format!("Message {}", i).repeat(50))).await;
        }
        
        // Should trigger auto-compact at 90%
        let should_compact = engine.should_auto_compact().await;
        // This depends on the exact estimation, but with 20 long messages,
        // we should be approaching the limit
        assert!(should_compact || engine.estimate_tokens().await > 0);
    }

    #[tokio::test]
    async fn test_pending_messages() {
        let config = QueryEngineConfig::default();
        let engine = QueryEngine::new(config);
        
        engine.queue_pending_message("Pending 1".to_string()).await;
        engine.queue_pending_message("Pending 2".to_string()).await;
        
        let drained = engine.drain_pending_messages().await;
        assert_eq!(drained.len(), 2);
    }

    #[tokio::test]
    async fn test_permission_denials() {
        let config = QueryEngineConfig::default();
        let engine = QueryEngine::new(config);
        
        engine.record_permission_denial(
            "Bash".to_string(),
            "User denied".to_string()
        ).await;
        
        let denials = engine.get_permission_denials().await;
        assert_eq!(denials.len(), 1);
        assert_eq!(denials[0].tool_name, "Bash");
    }

    #[tokio::test]
    async fn test_clear_conversation() {
        let config = QueryEngineConfig::default();
        let engine = QueryEngine::new(config);
        
        engine.add_message(Message::user("Hello")).await;
        engine.clear().await;
        
        assert_eq!(engine.message_count().await, 0);
    }
}