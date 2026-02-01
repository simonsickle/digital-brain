//! LLM Backend Traits
//!
//! Core traits and types for LLM backends.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Mutex;

/// Error type for LLM operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmError {
    pub message: String,
    pub kind: LlmErrorKind,
    pub retryable: bool,
}

impl LlmError {
    pub fn new(message: impl Into<String>, kind: LlmErrorKind) -> Self {
        let retryable = matches!(
            kind,
            LlmErrorKind::RateLimit | LlmErrorKind::Timeout | LlmErrorKind::ServiceUnavailable
        );
        Self {
            message: message.into(),
            kind,
            retryable,
        }
    }

    pub fn api_error(message: impl Into<String>) -> Self {
        Self::new(message, LlmErrorKind::ApiError)
    }

    pub fn rate_limit(message: impl Into<String>) -> Self {
        Self::new(message, LlmErrorKind::RateLimit)
    }

    pub fn timeout(message: impl Into<String>) -> Self {
        Self::new(message, LlmErrorKind::Timeout)
    }

    pub fn invalid_response(message: impl Into<String>) -> Self {
        Self::new(message, LlmErrorKind::InvalidResponse)
    }
}

impl std::fmt::Display for LlmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}

impl std::error::Error for LlmError {}

/// Kind of LLM error
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LlmErrorKind {
    /// API returned an error
    ApiError,
    /// Rate limited
    RateLimit,
    /// Request timed out
    Timeout,
    /// Service unavailable
    ServiceUnavailable,
    /// Invalid response from API
    InvalidResponse,
    /// Authentication failed
    AuthError,
    /// Invalid request
    InvalidRequest,
    /// Network error
    NetworkError,
}

/// Configuration for a single LLM request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmRequestConfig {
    /// Maximum tokens to generate
    pub max_tokens: u32,
    /// Temperature (0.0 - 2.0)
    pub temperature: f64,
    /// Top-p sampling
    pub top_p: Option<f64>,
    /// Stop sequences
    pub stop_sequences: Vec<String>,
    /// System prompt (if separate from messages)
    pub system: Option<String>,
    /// Additional provider-specific parameters
    pub extra: HashMap<String, serde_json::Value>,
}

impl Default for LlmRequestConfig {
    fn default() -> Self {
        Self {
            max_tokens: 1024,
            temperature: 0.7,
            top_p: None,
            stop_sequences: Vec::new(),
            system: None,
            extra: HashMap::new(),
        }
    }
}

impl LlmRequestConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = temperature.clamp(0.0, 2.0);
        self
    }

    pub fn with_top_p(mut self, top_p: f64) -> Self {
        self.top_p = Some(top_p.clamp(0.0, 1.0));
        self
    }

    pub fn with_stop(mut self, stop: impl Into<String>) -> Self {
        self.stop_sequences.push(stop.into());
        self
    }

    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    pub fn with_extra(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.extra.insert(key.into(), value);
        self
    }
}

/// Response from an LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmResponse {
    /// Generated text
    pub content: String,
    /// Model that generated the response
    pub model: String,
    /// Finish reason (stop, length, etc.)
    pub finish_reason: Option<String>,
    /// Usage statistics
    pub usage: Option<LlmUsage>,
    /// Provider-specific metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl LlmResponse {
    pub fn new(content: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            model: model.into(),
            finish_reason: None,
            usage: None,
            metadata: HashMap::new(),
        }
    }

    pub fn with_finish_reason(mut self, reason: impl Into<String>) -> Self {
        self.finish_reason = Some(reason.into());
        self
    }

    pub fn with_usage(mut self, usage: LlmUsage) -> Self {
        self.usage = Some(usage);
        self
    }
}

/// Token usage statistics
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LlmUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

impl LlmUsage {
    pub fn new(input: u32, output: u32) -> Self {
        Self {
            input_tokens: input,
            output_tokens: output,
        }
    }

    pub fn total(&self) -> u32 {
        self.input_tokens + self.output_tokens
    }
}

/// The core trait for LLM backends
///
/// Implement this trait to add support for a new LLM provider.
#[async_trait]
pub trait LlmBackend: Send + Sync {
    /// Generate a completion from a prompt
    async fn complete(
        &self,
        prompt: &str,
        config: &LlmRequestConfig,
    ) -> Result<LlmResponse, LlmError>;

    /// Generate a completion from a list of messages
    async fn chat(
        &self,
        messages: &[ChatMessage],
        config: &LlmRequestConfig,
    ) -> Result<LlmResponse, LlmError> {
        // Default implementation: convert messages to single prompt
        let prompt = messages
            .iter()
            .map(|m| format!("{}: {}", m.role, m.content))
            .collect::<Vec<_>>()
            .join("\n\n");
        self.complete(&prompt, config).await
    }

    /// Get the model name
    fn model_name(&self) -> &str;

    /// Get the provider name
    fn provider(&self) -> &str;

    /// Check if the backend is available
    async fn health_check(&self) -> Result<(), LlmError> {
        // Default: try a minimal completion
        let config = LlmRequestConfig::new().with_max_tokens(1);
        self.complete("Hi", &config).await?;
        Ok(())
    }
}

/// A chat message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.into(),
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: content.into(),
        }
    }
}

// ============================================================================
// MOCK BACKEND
// ============================================================================

/// A mock LLM backend for testing
pub struct MockLlmBackend {
    responses: Mutex<Vec<String>>,
    default_response: String,
    model: String,
}

impl MockLlmBackend {
    pub fn new() -> Self {
        Self {
            responses: Mutex::new(Vec::new()),
            default_response: "ACTION: THINK\nCONTENT: Mock response\nREASONING: Testing"
                .to_string(),
            model: "mock".to_string(),
        }
    }

    pub fn with_default_response(mut self, response: impl Into<String>) -> Self {
        self.default_response = response.into();
        self
    }

    pub fn queue_response(&self, response: impl Into<String>) {
        self.responses.lock().unwrap().push(response.into());
    }

    pub fn queue_responses(&self, responses: impl IntoIterator<Item = impl Into<String>>) {
        let mut queue = self.responses.lock().unwrap();
        for r in responses {
            queue.push(r.into());
        }
    }
}

impl Default for MockLlmBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LlmBackend for MockLlmBackend {
    async fn complete(
        &self,
        _prompt: &str,
        _config: &LlmRequestConfig,
    ) -> Result<LlmResponse, LlmError> {
        let content = {
            let mut responses = self.responses.lock().unwrap();
            if responses.is_empty() {
                self.default_response.clone()
            } else {
                responses.remove(0)
            }
        };

        Ok(LlmResponse::new(content, &self.model))
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    fn provider(&self) -> &str {
        "mock"
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_backend() {
        let backend = MockLlmBackend::new();
        backend.queue_response("Hello!");

        let response = backend
            .complete("Hi", &LlmRequestConfig::default())
            .await
            .unwrap();

        assert_eq!(response.content, "Hello!");
    }

    #[tokio::test]
    async fn test_mock_backend_default() {
        let backend = MockLlmBackend::new().with_default_response("Default");

        let response = backend
            .complete("Hi", &LlmRequestConfig::default())
            .await
            .unwrap();

        assert_eq!(response.content, "Default");
    }

    #[test]
    fn test_request_config_builder() {
        let config = LlmRequestConfig::new()
            .with_max_tokens(500)
            .with_temperature(0.5)
            .with_stop("END")
            .with_system("You are helpful");

        assert_eq!(config.max_tokens, 500);
        assert_eq!(config.temperature, 0.5);
        assert_eq!(config.stop_sequences, vec!["END"]);
        assert_eq!(config.system, Some("You are helpful".to_string()));
    }

    #[test]
    fn test_chat_message() {
        let user = ChatMessage::user("Hello");
        let assistant = ChatMessage::assistant("Hi there");

        assert_eq!(user.role, "user");
        assert_eq!(assistant.role, "assistant");
    }
}
