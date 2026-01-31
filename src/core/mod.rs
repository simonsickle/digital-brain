//! Core systems - cross-cutting functionality.

pub mod neuromodulators;
pub mod prediction;
pub mod workspace;

pub use neuromodulators::{
    AcetylcholineSystem, CortisolSystem, DopamineSystem, GabaSystem, InhibitionResult,
    ModulatorLevel, NeuromodulatorState, NeuromodulatorySystem, NorepinephrineSystem,
    OxytocinSystem, RewardCategory, RewardQuality, RewardResult, SerotoninSystem, ToleranceTracker,
    TrustLevel,
};
pub use prediction::{Prediction, PredictionEngine, PredictionError, PredictionStats};
pub use workspace::{Broadcast, GlobalWorkspace, WorkspaceConfig};
