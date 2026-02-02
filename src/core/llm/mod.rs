//! LLM Backend Implementations
//!
//! This module provides implementations for various LLM providers.
//! The design is extensible - implement `LlmBackend` trait for any provider.
//!
//! # Supported Backends
//!
//! - **Anthropic** - Claude models (Opus, Sonnet, Haiku)
//! - **OpenAI** - GPT models (coming soon)
//! - **Ollama** - Local models (coming soon)
//! - **Mock** - For testing
//!
//! # Usage
//!
//! ```rust,ignore
//! use digital_brain::core::llm::{AnthropicBackend, LlmBackend};
//!
//! let backend = AnthropicBackend::new("your-api-key")
//!     .with_model("claude-opus-4-5-20250131");
//!
//! let response = backend.complete("Hello!", &config).await?;
//! ```

mod anthropic;
mod traits;

pub use anthropic::{AnthropicBackend, AnthropicConfig, AnthropicModel};
pub use traits::{
    ChatMessage, LlmBackend, LlmError, LlmErrorKind, LlmRequestConfig, LlmResponse, LlmUsage,
    MockLlmBackend,
};

/// Available LLM providers
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum LlmProvider {
    Anthropic,
    OpenAI,
    Ollama,
    Mock,
}

impl std::fmt::Display for LlmProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Anthropic => write!(f, "anthropic"),
            Self::OpenAI => write!(f, "openai"),
            Self::Ollama => write!(f, "ollama"),
            Self::Mock => write!(f, "mock"),
        }
    }
}
