//! Core systems - cross-cutting functionality.

pub mod nervous_system;
pub mod neuromodulators;
pub mod prediction;
pub mod workspace;

pub use nervous_system::{
    BrainRegion, NervousSystem, NervousSystemConfig, NervousSystemStats, Pathway, SignalTrace,
};
pub use neuromodulators::{
    AcetylcholineSystem, CortisolSystem, DopamineSystem, GabaSystem, InhibitionResult,
    ModulatorLevel, NeuromodulatorState, NeuromodulatorySystem, NorepinephrineSystem,
    OxytocinSystem, RewardCategory, RewardQuality, RewardResult, SerotoninSystem, ToleranceTracker,
    TrustLevel,
};
pub use prediction::{Prediction, PredictionEngine, PredictionError, PredictionStats};
pub use workspace::{Broadcast, GlobalWorkspace, WorkspaceConfig};
