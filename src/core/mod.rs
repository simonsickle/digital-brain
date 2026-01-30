//! Core systems - cross-cutting functionality.

pub mod prediction;
pub mod workspace;

pub use prediction::{PredictionEngine, Prediction, PredictionError, PredictionStats};
pub use workspace::{GlobalWorkspace, WorkspaceConfig, Broadcast};
