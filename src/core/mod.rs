//! Core systems - cross-cutting functionality.

pub mod action;
pub mod attention;
pub mod curiosity;
pub mod goals;
pub mod nervous_system;
pub mod neuromodulators;
pub mod prediction;
pub mod workspace;
pub mod world_model;

pub use action::{
    ActionCategory, ActionDecision, ActionId, ActionSelector, ActionStats, ActionTemplate,
    Condition, ExpectedOutcome, Outcome,
};
pub use attention::{
    AttentionBudget, AttentionStats, COMPLEXITY_KEYWORDS, TaskComplexity, estimate_complexity,
};
pub use curiosity::{Competence, CuriosityStats, CuriositySystem, Domain, ExplorationEvent};
pub use goals::{
    Criterion, Goal, GoalEvent, GoalId, GoalManager, GoalStats, GoalStatus, Priority, TimeHorizon,
};
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
pub use world_model::{
    Entity, EntityId, PredictionId, PropertyValue, RelationType, Relationship, WorldModel,
    WorldModelStats, WorldPrediction,
};
