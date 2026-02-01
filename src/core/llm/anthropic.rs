//! Anthropic Claude Backend
//!
//! Implementation of LlmBackend for Anthropic's Claude models.
//!
//! # Usage
//!
//! ```rust,ignore
//! use digital_brain::core::llm::{AnthropicBackend, AnthropicModel};
//!
//! // From environment variable
//! let backend = AnthropicBackend::from_env()?;
//!
//! // Or with explicit key
//! let backend = AnthropicBackend::new("sk-ant-...")
//!     .with_model(AnthropicModel::Opus4_5);
//!
//! // Make a request
//! let response = backend.complete("Hello!", &config).await?;
//! ```

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::Duration;

use super::traits::{ChatMessage, LlmBackend, LlmError, LlmErrorKind, LlmRequestConfig, LlmResponse, LlmUsage};

/// Anthropic API base URL
const ANTHROPIC_API_URL: &str = "https://api.anthropic.com/v1/messages";

/// Anthropic API version
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Available Claude models
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnthropicModel {
    /// Claude Opus 4.5 - Most capable
    #[serde(rename = "claude-opus-4-5-20250131")]
    Opus4_5,
    /// Claude Sonnet 4 - Balanced
    #[serde(rename = "claude-sonnet-4-20250514")]
    Sonnet4,
    /// Claude 3.5 Sonnet - Previous gen balanced
    #[serde(rename = "claude-3-5-sonnet-20241022")]
    Sonnet3_5,
    /// Claude 3.5 Haiku - Fast and efficient
    #[serde(rename = "claude-3-5-haiku-20241022")]
    Haiku3_5,
    /// Custom model string
    #[serde(untagged)]
    Custom(String),
}

impl AnthropicModel {
    /// Get the model ID string
    pub fn as_str(&self) -> &str {
        match self {
            Self::Opus4_5 => "claude-opus-4-5-20250131",
            Self::Sonnet4 => "claude-sonnet-4-20250514",
            Self::Sonnet3_5 => "claude-3-5-sonnet-20241022",
            Self::Haiku3_5 => "claude-3-5-haiku-20241022",
            Self::Custom(s) => s,
        }
    }
}

impl Default for AnthropicModel {
    fn default() -> Self {
        Self::Opus4_5
    }
}

impl std::fmt::Display for AnthropicModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl From<&str> for AnthropicModel {
    fn from(s: &str) -> Self {
        match s {
            "claude-opus-4-5-20250131" | "opus-4.5" | "opus" => Self::Opus4_5,
            "claude-sonnet-4-20250514" | "sonnet-4" | "sonnet" => Self::Sonnet4,
            "claude-3-5-sonnet-20241022" | "sonnet-3.5" => Self::Sonnet3_5,
            "claude-3-5-haiku-20241022" | "haiku-3.5" | "haiku" => Self::Haiku3_5,
            other => Self::Custom(other.to_string()),
        }
    }
}

/// Configuration for Anthropic backend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicConfig {
    /// API key
    #[serde(skip_serializing)]
    pub api_key: String,
    /// Model to use
    pub model: AnthropicModel,
    /// Request timeout
    #[serde(with = "humantime_serde")]
    pub timeout: Duration,
    /// Maximum retries for transient errors
    pub max_retries: u32,
    /// Custom API URL (for proxies)
    pub api_url: Option<String>,
}

impl Default for AnthropicConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            model: AnthropicModel::default(),
            timeout: Duration::from_secs(60),
            max_retries: 3,
            api_url: None,
        }
    }
}

/// Anthropic LLM backend
pub struct AnthropicBackend {
    config: AnthropicConfig,
    client: reqwest::Client,
}

impl AnthropicBackend {
    /// Create a new Anthropic backend with API key
    pub fn new(api_key: impl Into<String>) -> Self {
        let config = AnthropicConfig {
            api_key: api_key.into(),
            ..Default::default()
        };

        let client = reqwest::Client::builder()
            .timeout(config.timeout)
            .build()
            .expect("Failed to create HTTP client");

        Self { config, client }
    }

    /// Create from environment variable ANTHROPIC_API_KEY
    pub fn from_env() -> Result<Self, LlmError> {
        let api_key = std::env::var("ANTHROPIC_API_KEY").map_err(|_| {
            LlmError::new(
                "ANTHROPIC_API_KEY environment variable not set",
                LlmErrorKind::AuthError,
            )
        })?;
        Ok(Self::new(api_key))
    }

    /// Create with full config
    pub fn with_config(config: AnthropicConfig) -> Self {
        let client = reqwest::Client::builder()
            .timeout(config.timeout)
            .build()
            .expect("Failed to create HTTP client");

        Self { config, client }
    }

    /// Set the model
    pub fn with_model(mut self, model: impl Into<AnthropicModel>) -> Self {
        self.config.model = model.into();
        self
    }

    /// Set the model from string
    pub fn with_model_str(mut self, model: &str) -> Self {
        self.config.model = AnthropicModel::from(model);
        self
    }

    /// Set timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.config.timeout = timeout;
        self.client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .expect("Failed to create HTTP client");
        self
    }

    /// Set custom API URL
    pub fn with_api_url(mut self, url: impl Into<String>) -> Self {
        self.config.api_url = Some(url.into());
        self
    }

    /// Get the API URL
    fn api_url(&self) -> &str {
        self.config.api_url.as_deref().unwrap_or(ANTHROPIC_API_URL)
    }

    /// Build request body
    fn build_request(
        &self,
        messages: &[ApiMessage],
        config: &LlmRequestConfig,
    ) -> AnthropicRequest {
        let mut request = AnthropicRequest {
            model: self.config.model.as_str().to_string(),
            messages: messages.to_vec(),
            max_tokens: config.max_tokens,
            system: config.system.clone(),
            temperature: Some(config.temperature),
            top_p: config.top_p,
            stop_sequences: if config.stop_sequences.is_empty() {
                None
            } else {
                Some(config.stop_sequences.clone())
            },
            thinking: None,
        };

        // Handle extended thinking for Opus 4.5
        if matches!(self.config.model, AnthropicModel::Opus4_5) {
            if let Some(budget) = config.extra.get("thinking_budget") {
                if let Some(budget_val) = budget.as_u64() {
                    request.thinking = Some(ThinkingConfig {
                        thinking_type: "enabled".to_string(),
                        budget_tokens: budget_val as u32,
                    });
                    // Temperature must be 1.0 for extended thinking
                    request.temperature = Some(1.0);
                }
            }
        }

        request
    }

    /// Parse API response
    fn parse_response(&self, response: AnthropicResponse) -> Result<LlmResponse, LlmError> {
        // Extract text content
        let content = response
            .content
            .iter()
            .filter_map(|block| {
                if block.content_type == "text" {
                    Some(block.text.clone().unwrap_or_default())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("");

        let mut llm_response = LlmResponse::new(content, &response.model)
            .with_finish_reason(response.stop_reason.unwrap_or_default());

        if let Some(usage) = response.usage {
            llm_response = llm_response.with_usage(LlmUsage::new(
                usage.input_tokens,
                usage.output_tokens,
            ));
        }

        Ok(llm_response)
    }

    /// Make API request with retries
    async fn make_request(&self, request: &AnthropicRequest) -> Result<AnthropicResponse, LlmError> {
        let mut last_error = None;

        for attempt in 0..=self.config.max_retries {
            if attempt > 0 {
                // Exponential backoff
                let delay = Duration::from_millis(100 * 2u64.pow(attempt));
                tokio::time::sleep(delay).await;
            }

            let result = self
                .client
                .post(self.api_url())
                .header("x-api-key", &self.config.api_key)
                .header("anthropic-version", ANTHROPIC_VERSION)
                .header("content-type", "application/json")
                .json(request)
                .send()
                .await;

            match result {
                Ok(response) => {
                    let status = response.status();

                    if status.is_success() {
                        let body = response
                            .json::<AnthropicResponse>()
                            .await
                            .map_err(|e| LlmError::invalid_response(e.to_string()))?;
                        return Ok(body);
                    }

                    // Handle error responses
                    let error_body = response.text().await.unwrap_or_default();

                    let error = match status.as_u16() {
                        401 => LlmError::new("Invalid API key", LlmErrorKind::AuthError),
                        429 => LlmError::rate_limit("Rate limited"),
                        500..=599 => {
                            LlmError::new(error_body, LlmErrorKind::ServiceUnavailable)
                        }
                        _ => LlmError::api_error(format!("HTTP {}: {}", status, error_body)),
                    };

                    if !error.retryable {
                        return Err(error);
                    }

                    last_error = Some(error);
                }
                Err(e) => {
                    let error = if e.is_timeout() {
                        LlmError::timeout("Request timed out")
                    } else if e.is_connect() {
                        LlmError::new("Connection failed", LlmErrorKind::NetworkError)
                    } else {
                        LlmError::new(e.to_string(), LlmErrorKind::NetworkError)
                    };

                    if !error.retryable {
                        return Err(error);
                    }

                    last_error = Some(error);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| LlmError::api_error("Unknown error")))
    }
}

#[async_trait]
impl LlmBackend for AnthropicBackend {
    async fn complete(
        &self,
        prompt: &str,
        config: &LlmRequestConfig,
    ) -> Result<LlmResponse, LlmError> {
        let messages = vec![ApiMessage {
            role: "user".to_string(),
            content: prompt.to_string(),
        }];

        let request = self.build_request(&messages, config);
        let response = self.make_request(&request).await?;
        self.parse_response(response)
    }

    async fn chat(
        &self,
        messages: &[ChatMessage],
        config: &LlmRequestConfig,
    ) -> Result<LlmResponse, LlmError> {
        let api_messages: Vec<ApiMessage> = messages
            .iter()
            .filter(|m| m.role != "system") // System is handled separately
            .map(|m| ApiMessage {
                role: m.role.clone(),
                content: m.content.clone(),
            })
            .collect();

        // Extract system message if present
        let mut config = config.clone();
        if config.system.is_none() {
            if let Some(sys) = messages.iter().find(|m| m.role == "system") {
                config.system = Some(sys.content.clone());
            }
        }

        let request = self.build_request(&api_messages, &config);
        let response = self.make_request(&request).await?;
        self.parse_response(response)
    }

    fn model_name(&self) -> &str {
        self.config.model.as_str()
    }

    fn provider(&self) -> &str {
        "anthropic"
    }
}

// ============================================================================
// API TYPES
// ============================================================================

#[derive(Debug, Serialize)]
struct AnthropicRequest {
    model: String,
    messages: Vec<ApiMessage>,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<ThinkingConfig>,
}

#[derive(Debug, Serialize)]
struct ThinkingConfig {
    #[serde(rename = "type")]
    thinking_type: String,
    budget_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ApiMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    content: Vec<ContentBlock>,
    model: String,
    stop_reason: Option<String>,
    usage: Option<UsageInfo>,
}

#[derive(Debug, Deserialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    content_type: String,
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct UsageInfo {
    input_tokens: u32,
    output_tokens: u32,
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_strings() {
        assert_eq!(AnthropicModel::Opus4_5.as_str(), "claude-opus-4-5-20250131");
        assert_eq!(AnthropicModel::Sonnet4.as_str(), "claude-sonnet-4-20250514");
        assert_eq!(AnthropicModel::Haiku3_5.as_str(), "claude-3-5-haiku-20241022");
    }

    #[test]
    fn test_model_from_str() {
        assert_eq!(AnthropicModel::from("opus"), AnthropicModel::Opus4_5);
        assert_eq!(AnthropicModel::from("sonnet"), AnthropicModel::Sonnet4);
        assert_eq!(AnthropicModel::from("haiku"), AnthropicModel::Haiku3_5);
        assert_eq!(
            AnthropicModel::from("custom-model"),
            AnthropicModel::Custom("custom-model".to_string())
        );
    }

    #[test]
    fn test_backend_creation() {
        let backend = AnthropicBackend::new("test-key")
            .with_model(AnthropicModel::Opus4_5);

        assert_eq!(backend.model_name(), "claude-opus-4-5-20250131");
        assert_eq!(backend.provider(), "anthropic");
    }

    #[test]
    fn test_request_building() {
        let backend = AnthropicBackend::new("test-key");
        let config = LlmRequestConfig::new()
            .with_max_tokens(500)
            .with_temperature(0.5)
            .with_system("Be helpful");

        let messages = vec![ApiMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
        }];

        let request = backend.build_request(&messages, &config);

        assert_eq!(request.max_tokens, 500);
        assert_eq!(request.temperature, Some(0.5));
        assert_eq!(request.system, Some("Be helpful".to_string()));
    }
}
