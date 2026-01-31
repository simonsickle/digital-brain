//! Core systems - cross-cutting functionality.

pub mod neuromodulators;
pub mod prediction;
pub mod workspace;

pub use neuromodulators::{
    AcetylcholineSystem, DopamineSystem, ModulatorLevel, NeuromodulatorState,
    NeuromodulatorySystem, NorepinephrineSystem, RewardCategory, RewardQuality, RewardResult,
    SerotoninSystem, ToleranceTracker,
};
pub use prediction::{Prediction, PredictionEngine, PredictionError, PredictionStats};
pub use workspace::{Broadcast, GlobalWorkspace, WorkspaceConfig};
