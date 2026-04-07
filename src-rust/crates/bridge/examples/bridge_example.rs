//! Bridge Protocol Usage Example
//!
//! This example demonstrates how to use the bridge protocol
//! for remote session management.

use claurst_bridge::{
    BridgeConfig, BridgeSessionInfo, start_bridge_session, poll_bridge_messages, post_bridge_response,
    transport::{create_transport, select_transport_type, Transport, TransportType, InboundMessage, OutboundMessage},
    RemoteSessionManager, TokenRefreshScheduler, JwtClaims,
    device_fingerprint, to_compat_session_id,
};
use tokio_util::sync::CancellationToken;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Example 1: Start a bridge session (HTTP polling mode)
    println!("=== Example 1: Bridge Session ===");
    
    // Set token for testing
    std::env::set_var("CLAURST_BRIDGE_TOKEN", "test-token");
    
    match start_bridge_session(None).await {
        Ok(info) => {
            println!("Session URL: {}", info.session_url);
            println!("Session ID: {}", info.session_id);
        }
        Err(e) => {
            println!("Expected error (no valid token): {}", e);
        }
    }

    // Example 2: Create a transport (WebSocket)
    println!("\n=== Example 2: WebSocket Transport ===");
    
    let ws_url = "wss://api.anthropic.com/v1/session_ingress/ws/session_123";
    let transport = create_transport(
        TransportType::WebSocket,
        ws_url,
        "test-token",
        vec![
            ("anthropic-version".to_string(), "2023-06-01".to_string()),
            ("x-environment-runner-version".to_string(), "0.0.1".to_string()),
        ],
    );
    
    println!("Transport created: {}", transport.state_label());

    // Example 3: Create SSE transport
    println!("\n=== Example 3: SSE Transport ===");
    
    let sse_url = "https://api.anthropic.com/v1/code/sessions/session_123";
    let sse_transport = create_transport(
        TransportType::Sse,
        sse_url,
        "worker-jwt-token",
        vec![],
    );
    
    println!("SSE Transport created: {}", sse_transport.state_label());

    // Example 4: Transport type selection
    println!("\n=== Example 4: Transport Type Selection ===");
    
    let ws_type = select_transport_type("wss://example.com/ws");
    println!("WS URL -> {:?}", ws_type);
    
    let http_type = select_transport_type("https://example.com/api");
    println!("HTTP URL -> {:?}", http_type);

    // Example 5: JWT handling
    println!("\n=== Example 5: JWT Handling ===");
    
    // Example JWT (not valid, just for structure demo)
    let example_jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyMTIzIiwiZXhwIjoxNzAwMDAwMDAwLCJpYXQiOjE2OTk5OTk5OTksInNlc3Npb25faWQiOiJzZXNzX2FiYyJ9.test";
    
    match JwtClaims::decode(example_jwt) {
        Ok(claims) => {
            println!("Decoded JWT claims: {:?}", claims);
            println!("Is expired: {}", claims.is_expired());
        }
        Err(e) => {
            println!("JWT decode error (expected): {}", e);
        }
    }

    // Example 6: Device fingerprint
    println!("\n=== Example 6: Device Fingerprint ===");
    let fp = device_fingerprint();
    println!("Device fingerprint: {}", fp);

    // Example 7: Session ID compatibility
    println!("\n=== Example 7: Session ID Compatibility ===");
    
    let cse_id = "cse_abc123def456";
    let compat_id = to_compat_session_id(cse_id);
    println!("{} -> {}", cse_id, compat_id);
    
    let session_id = "session_xyz789";
    let infra_id = claurst_bridge::to_infra_session_id(session_id);
    println!("{} -> {}", session_id, infra_id);

    // Example 8: Token refresh scheduler
    println!("\n=== Example 8: Token Refresh Scheduler ===");
    
    let scheduler = TokenRefreshScheduler::new();
    
    // Schedule a refresh (would need valid exp claim)
    // scheduler.schedule("session_123", token, |session_id| {
    //     println!("Refreshing token for {}", session_id);
    // });
    
    println!("Scheduler created");

    // Example 9: Remote Session Manager
    println!("\n=== Example 9: Remote Session Manager ===");
    
    let manager = RemoteSessionManager::new(
        "https://api.anthropic.com",
        "org-uuid-123",
    );
    
    println!("RemoteSessionManager created");
    println!("(API calls would require valid auth tokens)");

    // Example 10: Message types
    println!("\n=== Example 10: Message Types ===");
    
    let user_msg = InboundMessage::UserMessage {
        content: "Hello, Claude!".to_string(),
        session_id: "session_123".to_string(),
        message_id: "msg_456".to_string(),
        file_attachments: vec![],
    };
    println!("Inbound: {:?}", user_msg);
    
    let text_delta = OutboundMessage::TextDelta {
        text: "Hello back!".to_string(),
        message_id: "msg_789".to_string(),
        index: Some(0),
    };
    println!("Outbound: {:?}", text_delta);

    println!("\n=== Examples complete ===");
    Ok(())
}
