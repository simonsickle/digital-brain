//! Core systems - cross-cutting functionality.

pub mod prediction;
pub mod workspace;

pub use prediction::{Prediction, PredictionEngine, PredictionError, PredictionStats};
pub use workspace::{Broadcast, GlobalWorkspace, WorkspaceConfig};
