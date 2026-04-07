// Bridge Transport Layer
//
// Implements WebSocket and SSE transports for remote session communication.
// Mirrors the TypeScript transport layer from the spec.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use bytes::Bytes;
use futures::{SinkExt, StreamExt};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::net::TcpStream;
use tokio::sync::{mpsc, oneshot};
use tokio::time::{interval, timeout};
use tokio_tungstenite::{
    connect_async, tungstenite::client::IntoClientRequest, tungstenite::http, tungstenite::Message as WsMessage,
    MaybeTlsStream, WebSocketStream,
};
use tracing::{debug, error, info, trace, warn};
use url::Url;

// ---------------------------------------------------------------------------
// Constants (mirroring TypeScript spec)
// ---------------------------------------------------------------------------

const DEFAULT_MAX_BUFFER_SIZE: usize = 1000;
const DEFAULT_BASE_RECONNECT_DELAY_MS: u64 = 1000;
const DEFAULT_MAX_RECONNECT_DELAY_MS: u64 = 30_000;
const DEFAULT_RECONNECT_GIVE_UP_MS: u64 = 600_000; // 10 minutes
const DEFAULT_PING_INTERVAL_MS: u64 = 10_000;
const DEFAULT_KEEPALIVE_INTERVAL_MS: u64 = 300_000; // 5 minutes
const SLEEP_DETECTION_THRESHOLD_MS: u64 = 60_000;
const KEEP_ALIVE_FRAME: &str = r#"{"type":"keep_alive"}"#;

const BATCH_FLUSH_INTERVAL_MS: u64 = 100;
const POST_TIMEOUT_MS: u64 = 15_000;
const CLOSE_GRACE_MS: u64 = 3000;

const SSE_RECONNECT_BASE_DELAY_MS: u64 = 1000;
const SSE_RECONNECT_MAX_DELAY_MS: u64 = 30_000;
const SSE_RECONNECT_GIVE_UP_MS: u64 = 600_000;
const SSE_LIVENESS_TIMEOUT_MS: u64 = 45_000;

// Close codes that are permanent (no retry)
const PERMANENT_CLOSE_CODES: &[u16] = &[1002, 4001, 4003];

// ---------------------------------------------------------------------------
// Error Types
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum TransportError {
    #[error("WebSocket error: {0}")]
    WebSocket(String),
    #[error("Connection closed (code: {code}, reason: {reason})")]
    Closed { code: u16, reason: String },
    #[error("Authentication failed")]
    AuthFailed,
    #[error("Session expired or not found")]
    SessionExpired,
    #[error("Max reconnection attempts exceeded")]
    MaxReconnectsExceeded,
    #[error("Give up after {0}ms")]
    GiveUp(u64),
    #[error("HTTP error: {status}")]
    Http { status: u16 },
    #[error("Timeout")]
    Timeout,
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

// ---------------------------------------------------------------------------
// Message Types
// ---------------------------------------------------------------------------

/// Messages flowing from server to client (read direction)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InboundMessage {
    /// User message from web UI
    UserMessage {
        content: String,
        session_id: String,
        message_id: String,
        #[serde(default)]
        file_attachments: Vec<FileAttachment>,
    },
    /// Control request from server
    ControlRequest {
        request_id: String,
        request: ControlRequestDetails,
    },
    /// Control response from server
    ControlResponse { response: ControlResponseDetails },
    /// Cancel request
    ControlCancelRequest { request_id: String },
    /// Keepalive ping
    Ping,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileAttachment {
    pub file_uuid: String,
    pub file_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "subtype", rename_all = "snake_case")]
pub enum ControlRequestDetails {
    Initialize,
    SetModel {
        model: Option<String>,
    },
    SetMaxThinkingTokens {
        max_tokens: Option<u32>,
    },
    SetPermissionMode {
        mode: String,
    },
    Interrupt,
    CanUseTool {
        tool_name: String,
        input: serde_json::Value,
        tool_use_id: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlResponseDetails {
    pub subtype: String,
    pub request_id: String,
    #[serde(flatten)]
    pub payload: serde_json::Value,
}

/// Messages flowing from client to server (write direction)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OutboundMessage {
    /// Text delta for streaming
    TextDelta {
        text: String,
        message_id: String,
        index: Option<usize>,
    },
    /// Tool execution started
    ToolStart {
        tool_name: String,
        tool_id: String,
        input_preview: Option<String>,
    },
    /// Tool execution completed
    ToolEnd {
        tool_name: String,
        tool_id: String,
        result: String,
        is_error: bool,
    },
    /// Permission request
    PermissionRequest {
        request_id: String,
        tool_use_id: String,
        tool_name: String,
        description: String,
        options: Vec<String>,
    },
    /// Turn complete
    TurnComplete {
        message_id: String,
        stop_reason: String,
        usage: Option<Usage>,
    },
    /// Error
    Error {
        message: String,
        code: Option<String>,
    },
    /// Pong response
    Pong { server_time: Option<u64> },
    /// Session state
    SessionState {
        session_id: String,
        state: SessionState,
    },
    /// Control response
    ControlResponse {
        session_id: String,
        response: serde_json::Value,
    },
    /// Keepalive
    KeepAlive,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionState {
    Connecting,
    Connected,
    Idle,
    Processing,
    Disconnected,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub cost_usd: Option<f64>,
}

// NDJSON serialization helper
fn to_ndjson<T: Serialize>(msg: &T) -> Result<String> {
    let mut json = serde_json::to_string(msg)?;
    // Escape U+2028 and U+2029 for NDJSON safety
    json = json
        .replace('\u{2028}', "\\u2028")
        .replace('\u{2029}', "\\u2029");
    Ok(json)
}

// ---------------------------------------------------------------------------
// Transport Trait
// ---------------------------------------------------------------------------

#[async_trait::async_trait]
pub trait Transport: Send + Sync {
    /// Write a single message
    async fn write(&self, msg: OutboundMessage) -> Result<()>;
    /// Write a batch of messages
    async fn write_batch(&self, msgs: Vec<OutboundMessage>) -> Result<()>;
    /// Close the transport
    async fn close(&self) -> Result<()>;
    /// Check if connected
    fn is_connected(&self) -> bool;
    /// Get current state label
    fn state_label(&self) -> &'static str;
    /// Get last sequence number (for SSE resume)
    fn last_sequence_num(&self) -> u64;
    /// Set callback for received data
    fn set_on_data(&self, callback: Box<dyn Fn(InboundMessage) + Send + Sync>);
    /// Set callback for close events
    fn set_on_close(&self, callback: Box<dyn Fn(Option<u16>) + Send + Sync>);
    /// Set callback for connect events
    fn set_on_connect(&self, callback: Box<dyn Fn() + Send + Sync>);
    /// Connect the transport
    async fn connect(&self) -> Result<()>;
    /// Report session state (V2 only)
    fn report_state(&self, state: SessionState);
    /// Report metadata (V2 only)
    fn report_metadata(&self, metadata: serde_json::Value);
    /// Report delivery status (V2 only)
    fn report_delivery(&self, event_id: &str, status: &str);
    /// Flush pending writes (V2 only)
    async fn flush(&self) -> Result<()>;
}

// ---------------------------------------------------------------------------
// WebSocket Transport
// ---------------------------------------------------------------------------

/// WebSocket transport implementation
pub struct WebSocketTransport {
    url: String,
    headers: Vec<(String, String)>,
    inner: Arc<RwLock<WsTransportInner>>,
    message_tx: mpsc::UnboundedSender<OutboundMessage>,
    message_rx: Arc<RwLock<Option<mpsc::UnboundedReceiver<OutboundMessage>>>>,
}

struct WsTransportInner {
    connected: bool,
    state: TransportState,
    reconnect_attempts: u32,
    last_sequence_num: u64,
    on_data: Option<Box<dyn Fn(InboundMessage) + Send + Sync>>,
    on_close: Option<Box<dyn Fn(Option<u16>) + Send + Sync>>,
    on_connect: Option<Box<dyn Fn() + Send + Sync>>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum TransportState {
    Idle,
    Connected,
    Reconnecting,
    Closing,
    Closed,
}

impl WebSocketTransport {
    pub fn new(url: impl Into<String>, headers: Vec<(String, String)>) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        Self {
            url: url.into(),
            headers,
            inner: Arc::new(RwLock::new(WsTransportInner {
                connected: false,
                state: TransportState::Idle,
                reconnect_attempts: 0,
                last_sequence_num: 0,
                on_data: None,
                on_close: None,
                on_connect: None,
            })),
            message_tx: tx,
            message_rx: Arc::new(RwLock::new(Some(rx))),
        }
    }

    fn reset_state(&self) {
        let mut inner = self.inner.write();
        inner.connected = false;
        inner.state = TransportState::Idle;
        inner.reconnect_attempts = 0;
    }

    async fn run_connection_loop(&self, cancel: tokio_util::sync::CancellationToken) {
        let mut base_delay = Duration::from_millis(DEFAULT_BASE_RECONNECT_DELAY_MS);
        let max_delay = Duration::from_millis(DEFAULT_MAX_RECONNECT_DELAY_MS);
        let give_up = Duration::from_millis(DEFAULT_RECONNECT_GIVE_UP_MS);
        let start_time = std::time::Instant::now();

        loop {
            if cancel.is_cancelled() {
                info!("WebSocket connection loop cancelled");
                break;
            }

            // Check give up
            if start_time.elapsed() > give_up {
                error!("WebSocket giving up after {}ms", give_up.as_millis());
                self.trigger_close(Some(0));
                break;
            }

            match self.connect_and_run().await {
                Ok(()) => {
                    // Normal close
                    break;
                }
                Err(TransportError::Closed { code, .. })
                    if PERMANENT_CLOSE_CODES.contains(&code) =>
                {
                    error!("WebSocket permanent close code {}, not retrying", code);
                    self.trigger_close(Some(code));
                    break;
                }
                Err(TransportError::AuthFailed) => {
                    error!("WebSocket auth failed, not retrying");
                    self.trigger_close(Some(4003));
                    break;
                }
                Err(e) => {
                    warn!("WebSocket error, will retry: {}", e);

                    // Increment reconnect attempts
                    {
                        let mut inner = self.inner.write();
                        inner.reconnect_attempts += 1;
                        inner.state = TransportState::Reconnecting;
                    }

                    // Calculate backoff
                    let attempt = self.inner.read().reconnect_attempts;
                    let delay = std::cmp::min(
                        base_delay * 2u32.pow(attempt.saturating_sub(1).min(5)),
                        max_delay,
                    );

                    // Check for sleep detection
                    if delay > Duration::from_millis(SLEEP_DETECTION_THRESHOLD_MS) {
                        info!("WebSocket sleep detected, resetting reconnect budget");
                        self.inner.write().reconnect_attempts = 0;
                    }

                    tokio::select! {
                        _ = tokio::time::sleep(delay) => {}
                        _ = cancel.cancelled() => break,
                    }
                }
            }
        }
    }

    async fn connect_and_run(&self) -> std::result::Result<(), TransportError> {
        let url = Url::parse(&self.url).map_err(|e| TransportError::Other(e.into()))?;

        info!("WebSocket connecting to {}", self.url);

        // Build request with headers
        let mut request = url
            .as_str()
            .into_client_request()
            .map_err(|e| TransportError::WebSocket(e.to_string()))?;
        for (key, value) in &self.headers {
            request.headers_mut().insert(
                key.parse::<http::header::HeaderName>()
                    .map_err(|_| TransportError::WebSocket("Invalid header name".into()))?,
                value
                    .parse::<http::header::HeaderValue>()
                    .map_err(|_| TransportError::WebSocket("Invalid header value".into()))?,
            );
        }

        let (ws_stream, _) = connect_async(request)
            .await
            .map_err(|e| TransportError::WebSocket(e.to_string()))?;

        info!("WebSocket connected");

        // Update state
        {
            let mut inner = self.inner.write();
            inner.connected = true;
            inner.state = TransportState::Connected;
            inner.reconnect_attempts = 0;
        }

        // Trigger connect callback
        if let Some(ref cb) = self.inner.read().on_connect {
            cb();
        }

        // Run the WebSocket
        self.run_ws(ws_stream).await
    }

    async fn run_ws(
        &self,
        ws_stream: WebSocketStream<MaybeTlsStream<TcpStream>>,
    ) -> std::result::Result<(), TransportError> {
        let (mut write, mut read) = ws_stream.split();

        // Take the message receiver
        let mut msg_rx = self.message_rx.write().take().unwrap();

        // Ping interval
        let mut ping_interval = interval(Duration::from_millis(DEFAULT_PING_INTERVAL_MS));
        let mut keepalive_interval = interval(Duration::from_millis(DEFAULT_KEEPALIVE_INTERVAL_MS));

        let result = loop {
            tokio::select! {
                // Handle incoming messages from server
                msg = read.next() => {
                    match msg {
                        Some(Ok(WsMessage::Text(text))) => {
                            trace!("WebSocket received: {}", text);
                            self.handle_inbound_text(&text);
                        }
                        Some(Ok(WsMessage::Binary(bin))) => {
                            if let Ok(text) = String::from_utf8(bin) {
                                self.handle_inbound_text(&text);
                            }
                        }
                        Some(Ok(WsMessage::Close(cf))) => {
                            let code = cf.as_ref().map(|c| c.code.into());
                            break Err(TransportError::Closed {
                                code: code.unwrap_or(1000),
                                reason: cf.map(|c| c.reason.to_string()).unwrap_or_default(),
                            });
                        }
                        Some(Ok(WsMessage::Frame(_))) => {}
                        Some(Ok(WsMessage::Ping(_))) => {}
                        Some(Ok(WsMessage::Pong(_))) => {}
                        Some(Err(e)) => {
                            break Err(TransportError::WebSocket(e.to_string()));
                        }
                        None => {
                            break Err(TransportError::Closed { code: 1001, reason: "Stream ended".into() });
                        }
                    }
                }

                // Handle outgoing messages
                Some(msg) = msg_rx.recv() => {
                    let json = match to_ndjson(&msg) {
                        Ok(j) => j,
                        Err(e) => {
                            error!("Failed to serialize message: {}", e);
                            continue;
                        }
                    };
                    trace!("WebSocket sending: {}", json);
                    if let Err(e) = write.send(WsMessage::Text(json)).await {
                        break Err(TransportError::WebSocket(e.to_string()));
                    }
                }

                // Send keepalive
                _ = keepalive_interval.tick() => {
                    trace!("WebSocket sending keepalive");
                    if let Err(e) = write.send(WsMessage::Text(KEEP_ALIVE_FRAME.to_string())).await {
                        break Err(TransportError::WebSocket(e.to_string()));
                    }
                }

                // Send ping
                _ = ping_interval.tick() => {
                    trace!("WebSocket sending ping");
                    if let Err(e) = write.send(WsMessage::Ping(Vec::new())).await {
                        break Err(TransportError::WebSocket(e.to_string()));
                    }
                }
            }
        };

        // Restore the receiver for reconnection
        let (tx, rx) = mpsc::unbounded_channel();
        *self.message_rx.write() = Some(rx);
        // Note: we lose any pending messages in the old channel - in production
        // we'd want to drain and re-queue them

        result
    }

    fn handle_inbound_text(&self, text: &str) {
        // Try to parse as InboundMessage
        match serde_json::from_str::<InboundMessage>(text) {
            Ok(msg) => {
                if let Some(ref cb) = self.inner.read().on_data {
                    cb(msg);
                }
            }
            Err(e) => {
                // Might be a raw SDK message - try parsing as generic JSON
                trace!("Failed to parse inbound message: {}", e);
            }
        }
    }

    fn trigger_close(&self, code: Option<u16>) {
        self.inner.write().connected = false;
        if let Some(ref cb) = self.inner.read().on_close {
            cb(code);
        }
    }
}

#[async_trait::async_trait]
impl Transport for WebSocketTransport {
    async fn write(&self, msg: OutboundMessage) -> Result<()> {
        self.message_tx
            .send(msg)
            .map_err(|_| anyhow::anyhow!("WebSocket message channel closed"))?;
        Ok(())
    }

    async fn write_batch(&self, msgs: Vec<OutboundMessage>) -> Result<()> {
        for msg in msgs {
            self.write(msg).await?;
        }
        Ok(())
    }

    async fn close(&self) -> Result<()> {
        self.inner.write().state = TransportState::Closing;
        self.message_tx.send(OutboundMessage::KeepAlive).ok(); // Wake up the loop
        Ok(())
    }

    fn is_connected(&self) -> bool {
        self.inner.read().connected
    }

    fn state_label(&self) -> &'static str {
        match self.inner.read().state {
            TransportState::Idle => "idle",
            TransportState::Connected => "connected",
            TransportState::Reconnecting => "reconnecting",
            TransportState::Closing => "closing",
            TransportState::Closed => "closed",
        }
    }

    fn last_sequence_num(&self) -> u64 {
        self.inner.read().last_sequence_num
    }

    fn set_on_data(&self, callback: Box<dyn Fn(InboundMessage) + Send + Sync>) {
        self.inner.write().on_data = Some(callback);
    }

    fn set_on_close(&self, callback: Box<dyn Fn(Option<u16>) + Send + Sync>) {
        self.inner.write().on_close = Some(callback);
    }

    fn set_on_connect(&self, callback: Box<dyn Fn() + Send + Sync>) {
        self.inner.write().on_connect = Some(callback);
    }

    async fn connect(&self) -> Result<()> {
        self.reset_state();
        let cancel = tokio_util::sync::CancellationToken::new();
        self.run_connection_loop(cancel).await;
        Ok(())
    }

    fn report_state(&self, _state: SessionState) {
        // No-op for WebSocket transport
    }

    fn report_metadata(&self, _metadata: serde_json::Value) {
        // No-op for WebSocket transport
    }

    fn report_delivery(&self, _event_id: &str, _status: &str) {
        // No-op for WebSocket transport
    }

    async fn flush(&self) -> Result<()> {
        // No-op for WebSocket transport
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// SSE Transport
// ---------------------------------------------------------------------------

/// SSE frame as parsed from the stream
#[derive(Debug, Clone)]
struct SseFrame {
    event: Option<String>,
    id: Option<String>,
    data: Option<String>,
}

/// SSE transport implementation
pub struct SseTransport {
    session_url: String,
    auth_token: Arc<RwLock<String>>,
    inner: Arc<RwLock<SseTransportInner>>,
    message_tx: mpsc::UnboundedSender<OutboundMessage>,
    message_rx: Arc<RwLock<Option<mpsc::UnboundedReceiver<OutboundMessage>>>>,
    http_client: reqwest::Client,
}

struct SseTransportInner {
    connected: bool,
    state: TransportState,
    reconnect_attempts: u32,
    last_sequence_num: u64,
    on_data: Option<Box<dyn Fn(InboundMessage) + Send + Sync>>,
    on_close: Option<Box<dyn Fn(Option<u16>) + Send + Sync>>,
    on_connect: Option<Box<dyn Fn() + Send + Sync>>,
}

impl SseTransport {
    pub fn new(session_url: impl Into<String>, auth_token: impl Into<String>) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(35))
            .build()
            .expect("Failed to build HTTP client");

        Self {
            session_url: session_url.into(),
            auth_token: Arc::new(RwLock::new(auth_token.into())),
            inner: Arc::new(RwLock::new(SseTransportInner {
                connected: false,
                state: TransportState::Idle,
                reconnect_attempts: 0,
                last_sequence_num: 0,
                on_data: None,
                on_close: None,
                on_connect: None,
            })),
            message_tx: tx,
            message_rx: Arc::new(RwLock::new(Some(rx))),
            http_client,
        }
    }

    pub fn update_auth_token(&self, token: impl Into<String>) {
        *self.auth_token.write() = token.into();
    }

    fn reset_state(&self) {
        let mut inner = self.inner.write();
        inner.connected = false;
        inner.state = TransportState::Idle;
        inner.reconnect_attempts = 0;
    }

    async fn run_connection_loop(&self, cancel: tokio_util::sync::CancellationToken) {
        let mut base_delay = Duration::from_millis(SSE_RECONNECT_BASE_DELAY_MS);
        let max_delay = Duration::from_millis(SSE_RECONNECT_MAX_DELAY_MS);
        let give_up = Duration::from_millis(SSE_RECONNECT_GIVE_UP_MS);
        let start_time = std::time::Instant::now();

        loop {
            if cancel.is_cancelled() {
                info!("SSE connection loop cancelled");
                break;
            }

            // Check give up
            if start_time.elapsed() > give_up {
                error!("SSE giving up after {}ms", give_up.as_millis());
                self.trigger_close(Some(0));
                break;
            }

            match self.connect_and_run().await {
                Ok(()) => {
                    // Normal close
                    break;
                }
                Err(TransportError::Http { status: 401 })
                | Err(TransportError::Http { status: 403 }) => {
                    error!("SSE auth failed (HTTP 401/403), not retrying");
                    self.trigger_close(Some(4003));
                    break;
                }
                Err(TransportError::Http { status: 404 }) => {
                    error!("SSE session not found (HTTP 404), not retrying");
                    self.trigger_close(Some(4001));
                    break;
                }
                Err(e) => {
                    warn!("SSE error, will retry: {}", e);

                    // Increment reconnect attempts
                    {
                        let mut inner = self.inner.write();
                        inner.reconnect_attempts += 1;
                        inner.state = TransportState::Reconnecting;
                    }

                    // Calculate backoff
                    let attempt = self.inner.read().reconnect_attempts;
                    let delay = std::cmp::min(
                        base_delay * 2u32.pow(attempt.saturating_sub(1).min(5)),
                        max_delay,
                    );

                    tokio::select! {
                        _ = tokio::time::sleep(delay) => {}
                        _ = cancel.cancelled() => break,
                    }
                }
            }
        }
    }

    async fn connect_and_run(&self) -> std::result::Result<(), TransportError> {
        let sse_url = format!("{}/worker/events/stream", self.session_url);
        let last_seq = self.inner.read().last_sequence_num;

        info!("SSE connecting to {}", sse_url);

        let mut request = self
            .http_client
            .get(&sse_url)
            .bearer_auth(self.auth_token.read().clone())
            .header("accept", "text/event-stream");

        // Resume from last sequence number
        if last_seq > 0 {
            request = request.query(&[("from_sequence_num", last_seq.to_string())]);
        }

        let resp = request
            .send()
            .await
            .map_err(|e| TransportError::Other(e.into()))?;

        let status = resp.status();
        if !status.is_success() {
            return Err(TransportError::Http {
                status: status.as_u16(),
            });
        }

        info!("SSE connected");

        // Update state
        {
            let mut inner = self.inner.write();
            inner.connected = true;
            inner.state = TransportState::Connected;
            inner.reconnect_attempts = 0;
        }

        // Trigger connect callback
        if let Some(ref cb) = self.inner.read().on_connect {
            cb();
        }

        // Run SSE stream
        self.run_sse_stream(resp).await
    }

    async fn run_sse_stream(
        &self,
        resp: reqwest::Response,
    ) -> std::result::Result<(), TransportError> {
        let mut stream = resp.bytes_stream();
        let mut buffer = String::new();
        let mut last_activity = std::time::Instant::now();

        // Take the message receiver for outgoing POSTs
        let mut msg_rx = self.message_rx.write().take().unwrap();

        let result = loop {
            tokio::select! {
                // Read SSE data
                chunk = stream.next() => {
                    match chunk {
                        Some(Ok(bytes)) => {
                            last_activity = std::time::Instant::now();
                            if let Ok(text) = String::from_utf8(bytes.to_vec()) {
                                buffer.push_str(&text);
                                self.process_sse_buffer(&mut buffer);
                            }
                        }
                        Some(Err(e)) => {
                            break Err(TransportError::Other(e.into()));
                        }
                        None => {
                            break Err(TransportError::Closed { code: 1001, reason: "Stream ended".into() });
                        }
                    }
                }

                // Handle outgoing messages via POST
                Some(msg) = msg_rx.recv() => {
                    if let Err(e) = self.post_message(msg).await {
                        warn!("Failed to POST message: {}", e);
                    }
                }

                // Liveness timeout check
                _ = tokio::time::sleep(Duration::from_millis(SSE_LIVENESS_TIMEOUT_MS)) => {
                    if last_activity.elapsed() > Duration::from_millis(SSE_LIVENESS_TIMEOUT_MS) {
                        warn!("SSE liveness timeout, reconnecting");
                        break Err(TransportError::Timeout);
                    }
                }
            }
        };

        // Restore the receiver
        let (tx, rx) = mpsc::unbounded_channel();
        *self.message_rx.write() = Some(rx);

        result
    }

    fn process_sse_buffer(&self, buffer: &mut String) {
        // Parse SSE frames from buffer
        let (frames, remaining) = parse_sse_frames(buffer);
        *buffer = remaining;

        for frame in frames {
            // Update sequence number
            if let Some(ref id) = frame.id {
                if let Ok(seq) = id.parse::<u64>() {
                    self.inner.write().last_sequence_num = seq;
                }
            }

            // Process data
            if let Some(data) = frame.data {
                // Try to parse as SDK event
                if let Ok(event) = serde_json::from_str::<serde_json::Value>(&data) {
                    if let Some(payload) = event.get("payload") {
                        // Convert payload to InboundMessage
                        if let Ok(msg) = serde_json::from_value::<InboundMessage>(payload.clone()) {
                            if let Some(ref cb) = self.inner.read().on_data {
                                cb(msg);
                            }
                        }
                    }
                }
            }
        }
    }

    async fn post_message(&self, msg: OutboundMessage) -> Result<()> {
        let url = format!("{}/worker/events", self.session_url);
        let json = to_ndjson(&msg)?;

        let body = serde_json::json!({
            "event": serde_json::from_str::<serde_json::Value>(&json)?,
            "ts": chrono::Utc::now().timestamp_millis(),
        });

        let auth_token = self.auth_token.read().clone();
        let resp = self
            .http_client
            .post(&url)
            .bearer_auth(auth_token)
            .json(&body)
            .timeout(Duration::from_secs(5))
            .send()
            .await
            .context("POST event failed")?;

        if !resp.status().is_success() {
            anyhow::bail!("POST returned {}", resp.status());
        }

        Ok(())
    }

    fn trigger_close(&self, code: Option<u16>) {
        self.inner.write().connected = false;
        if let Some(ref cb) = self.inner.read().on_close {
            cb(code);
        }
    }
}

#[async_trait::async_trait]
impl Transport for SseTransport {
    async fn write(&self, msg: OutboundMessage) -> Result<()> {
        self.message_tx
            .send(msg)
            .map_err(|_| anyhow::anyhow!("SSE message channel closed"))?;
        Ok(())
    }

    async fn write_batch(&self, msgs: Vec<OutboundMessage>) -> Result<()> {
        for msg in msgs {
            self.write(msg).await?;
        }
        Ok(())
    }

    async fn close(&self) -> Result<()> {
        self.inner.write().state = TransportState::Closing;
        Ok(())
    }

    fn is_connected(&self) -> bool {
        self.inner.read().connected
    }

    fn state_label(&self) -> &'static str {
        match self.inner.read().state {
            TransportState::Idle => "idle",
            TransportState::Connected => "connected",
            TransportState::Reconnecting => "reconnecting",
            TransportState::Closing => "closing",
            TransportState::Closed => "closed",
        }
    }

    fn last_sequence_num(&self) -> u64 {
        self.inner.read().last_sequence_num
    }

    fn set_on_data(&self, callback: Box<dyn Fn(InboundMessage) + Send + Sync>) {
        self.inner.write().on_data = Some(callback);
    }

    fn set_on_close(&self, callback: Box<dyn Fn(Option<u16>) + Send + Sync>) {
        self.inner.write().on_close = Some(callback);
    }

    fn set_on_connect(&self, callback: Box<dyn Fn() + Send + Sync>) {
        self.inner.write().on_connect = Some(callback);
    }

    async fn connect(&self) -> Result<()> {
        self.reset_state();
        let cancel = tokio_util::sync::CancellationToken::new();
        self.run_connection_loop(cancel).await;
        Ok(())
    }

    fn report_state(&self, _state: SessionState) {
        // Would POST to /worker with state update
        debug!("Report state not implemented for SSE transport");
    }

    fn report_metadata(&self, _metadata: serde_json::Value) {
        debug!("Report metadata not implemented for SSE transport");
    }

    fn report_delivery(&self, event_id: &str, status: &str) {
        // Would POST to /worker/events/{event_id}/delivery
        debug!("Report delivery {} -> {} not implemented", event_id, status);
    }

    async fn flush(&self) -> Result<()> {
        // Wait for any pending POSTs to complete
        Ok(())
    }
}

/// Parse SSE frames from buffer
fn parse_sse_frames(buffer: &str) -> (Vec<SseFrame>, String) {
    let mut frames = Vec::new();
    let mut lines = buffer.lines().peekable();
    let mut remaining = String::new();
    let mut current_frame = SseFrame {
        event: None,
        id: None,
        data: None,
    };

    while let Some(line) = lines.next() {
        // Check if this might be incomplete (last line without newline)
        let is_last = lines.peek().is_none();
        if is_last && !buffer.ends_with('\n') {
            remaining = line.to_string();
            break;
        }

        if line.is_empty() {
            // End of frame
            if current_frame.event.is_some()
                || current_frame.id.is_some()
                || current_frame.data.is_some()
            {
                frames.push(current_frame);
                current_frame = SseFrame {
                    event: None,
                    id: None,
                    data: None,
                };
            }
            continue;
        }

        if line.starts_with(':') {
            // Comment/keepalive - ignore
            continue;
        }

        if let Some((key, value)) = line.split_once(':') {
            let value = value.strip_prefix(' ').unwrap_or(value);
            match key {
                "event" => current_frame.event = Some(value.to_string()),
                "id" => current_frame.id = Some(value.to_string()),
                "data" => {
                    if let Some(ref mut data) = current_frame.data {
                        data.push('\n');
                        data.push_str(value);
                    } else {
                        current_frame.data = Some(value.to_string());
                    }
                }
                _ => {}
            }
        }
    }

    // Don't forget the last frame if buffer ends with newline
    if current_frame.event.is_some() || current_frame.id.is_some() || current_frame.data.is_some() {
        frames.push(current_frame);
    }

    (frames, remaining)
}

// ---------------------------------------------------------------------------
// Hybrid Transport (WebSocket reads + HTTP POST writes)
// ---------------------------------------------------------------------------

/// Hybrid transport: WebSocket for reads, HTTP POST for writes
pub struct HybridTransport {
    ws_transport: WebSocketTransport,
    session_url: String,
    auth_token: Arc<RwLock<String>>,
    http_client: reqwest::Client,
    dropped_count: AtomicUsize,
}

impl HybridTransport {
    pub fn new(
        ws_url: impl Into<String>,
        session_url: impl Into<String>,
        auth_token: impl Into<String>,
        headers: Vec<(String, String)>,
    ) -> Self {
        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to build HTTP client");

        Self {
            ws_transport: WebSocketTransport::new(ws_url, headers),
            session_url: session_url.into(),
            auth_token: Arc::new(RwLock::new(auth_token.into())),
            http_client,
            dropped_count: AtomicUsize::new(0),
        }
    }

    pub fn ws_url_to_post_url(ws_url: &str) -> Result<String> {
        let url = Url::parse(ws_url)?;
        let scheme = if url.scheme() == "wss" {
            "https"
        } else {
            "http"
        };
        let host = url.host_str().context("Missing host")?;
        let port = url.port();
        let path = url.path();

        let base = if let Some(port) = port {
            format!("{}://{}:{}", scheme, host, port)
        } else {
            format!("{}://{}", scheme, host)
        };

        // Convert /v1/session_ingress/ws/{id} to /v1/session_ingress/events/{id}
        let post_path = path.replace("/ws/", "/events/");

        Ok(base + &post_path)
    }

    async fn post_message(&self, msg: OutboundMessage) -> Result<()> {
        let url = Self::ws_url_to_post_url(&format!("{}/events", self.session_url))?;

        let auth_token = self.auth_token.read().clone();
        let resp = self
            .http_client
            .post(&url)
            .bearer_auth(auth_token)
            .json(&msg)
            .timeout(Duration::from_millis(POST_TIMEOUT_MS))
            .send()
            .await?;

        if !resp.status().is_success() {
            anyhow::bail!("POST returned {}", resp.status());
        }

        Ok(())
    }
}

#[async_trait::async_trait]
impl Transport for HybridTransport {
    async fn write(&self, msg: OutboundMessage) -> Result<()> {
        // Use HTTP POST instead of WebSocket
        if let Err(e) = self.post_message(msg).await {
            warn!("Hybrid transport POST failed: {}", e);
            self.dropped_count.fetch_add(1, Ordering::SeqCst);
            return Err(e);
        }
        Ok(())
    }

    async fn write_batch(&self, msgs: Vec<OutboundMessage>) -> Result<()> {
        for msg in msgs {
            self.write(msg).await?;
        }
        Ok(())
    }

    async fn close(&self) -> Result<()> {
        self.ws_transport.close().await
    }

    fn is_connected(&self) -> bool {
        self.ws_transport.is_connected()
    }

    fn state_label(&self) -> &'static str {
        self.ws_transport.state_label()
    }

    fn last_sequence_num(&self) -> u64 {
        self.ws_transport.last_sequence_num()
    }

    fn set_on_data(&self, callback: Box<dyn Fn(InboundMessage) + Send + Sync>) {
        self.ws_transport.set_on_data(callback);
    }

    fn set_on_close(&self, callback: Box<dyn Fn(Option<u16>) + Send + Sync>) {
        self.ws_transport.set_on_close(callback);
    }

    fn set_on_connect(&self, callback: Box<dyn Fn() + Send + Sync>) {
        self.ws_transport.set_on_connect(callback);
    }

    async fn connect(&self) -> Result<()> {
        self.ws_transport.connect().await
    }

    fn report_state(&self, _state: SessionState) {
        // No-op for hybrid
    }

    fn report_metadata(&self, _metadata: serde_json::Value) {
        // No-op for hybrid
    }

    fn report_delivery(&self, _event_id: &str, _status: &str) {
        // No-op for hybrid
    }

    async fn flush(&self) -> Result<()> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Transport Factory
// ---------------------------------------------------------------------------

/// Transport type selection
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TransportType {
    /// Pure WebSocket
    WebSocket,
    /// Server-Sent Events (V2)
    Sse,
    /// Hybrid WebSocket reads + HTTP POST writes
    Hybrid,
}

/// Create a transport based on URL and configuration
pub fn create_transport(
    transport_type: TransportType,
    url: impl Into<String>,
    auth_token: impl Into<String>,
    headers: Vec<(String, String)>,
) -> Box<dyn Transport> {
    match transport_type {
        TransportType::WebSocket => Box::new(WebSocketTransport::new(url, headers)),
        TransportType::Sse => Box::new(SseTransport::new(url, auth_token)),
        TransportType::Hybrid => {
            let url_str = url.into();
            // For hybrid, url is WebSocket URL, session_url is derived
            Box::new(HybridTransport::new(
                url_str.clone(),
                url_str,
                auth_token,
                headers,
            ))
        }
    }
}

/// Select transport type based on environment/URL
pub fn select_transport_type(url: &str) -> TransportType {
    // Check environment variables per spec
    if std::env::var("CLAUDE_CODE_USE_CCR_V2").is_ok() {
        return TransportType::Sse;
    }

    if std::env::var("CLAUDE_CODE_POST_FOR_SESSION_INGRESS_V2").is_ok() && url.starts_with("ws") {
        return TransportType::Hybrid;
    }

    if url.starts_with("ws://") || url.starts_with("wss://") {
        TransportType::WebSocket
    } else if url.starts_with("http://") || url.starts_with("https://") {
        TransportType::Sse
    } else {
        TransportType::WebSocket
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_sse_frames() {
        let buffer = "event: sdk_event\nid: 42\ndata: {\"hello\":\"world\"}\n\n";
        let (frames, remaining) = parse_sse_frames(buffer);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].event, Some("sdk_event".to_string()));
        assert_eq!(frames[0].id, Some("42".to_string()));
        assert_eq!(frames[0].data, Some(r#"{"hello":"world"}"#.to_string()));
        assert!(remaining.is_empty());
    }

    #[test]
    fn test_parse_sse_keepalive() {
        let buffer = ":keepalive\n\n";
        let (frames, _) = parse_sse_frames(buffer);
        assert!(frames.is_empty());
    }

    #[test]
    fn test_parse_sse_multiline_data() {
        let buffer = "event: test\ndata: line1\ndata: line2\n\n";
        let (frames, _) = parse_sse_frames(buffer);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].data, Some("line1\nline2".to_string()));
    }

    #[test]
    fn test_ndjson_escaping() {
        let msg = OutboundMessage::TextDelta {
            text: "line\u{2028}sep\u{2029}para".to_string(),
            message_id: "test".to_string(),
            index: None,
        };
        let json = to_ndjson(&msg).unwrap();
        assert!(json.contains("\\u2028"));
        assert!(json.contains("\\u2029"));
    }

    #[test]
    fn test_ws_url_to_post_url() {
        let post = HybridTransport::ws_url_to_post_url(
            "wss://api.example.com/v1/session_ingress/ws/session_123",
        )
        .unwrap();
        assert_eq!(
            post,
            "https://api.example.com/v1/session_ingress/events/session_123"
        );

        let post =
            HybridTransport::ws_url_to_post_url("ws://localhost:8080/v2/session_ingress/ws/abc")
                .unwrap();
        assert_eq!(post, "http://localhost:8080/v2/session_ingress/events/abc");
    }
}
