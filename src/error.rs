//! Error types for the digital brain.

use thiserror::Error;

/// Result type for brain operations.
pub type Result<T> = std::result::Result<T, BrainError>;

/// Errors that can occur in the digital brain.
#[derive(Error, Debug)]
pub enum BrainError {
    /// Memory not found
    #[error("Memory not found: {0}")]
    MemoryNotFound(String),

    /// Database error
    #[error("Database error: {0}")]
    Database(#[from] rusqlite::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Signal routing error
    #[error("Signal routing error: {0}")]
    Routing(String),

    /// Module not found
    #[error("Module not found: {0}")]
    ModuleNotFound(String),

    /// Capacity exceeded
    #[error("Capacity exceeded: {0}")]
    CapacityExceeded(String),

    /// Invalid state
    #[error("Invalid state: {0}")]
    InvalidState(String),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),
}
