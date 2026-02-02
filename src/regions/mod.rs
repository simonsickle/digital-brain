//! Brain regions - specialized processing modules.
//!
//! Each region handles a specific aspect of cognition:
//!
//! - **hippocampus**: Episodic memory, encoding, retrieval
//! - **amygdala**: Emotional processing, valence assignment
//! - **prefrontal**: Working memory, executive control
//! - **thalamus**: Attention routing, signal gating
//! - **dmn**: Self-model, identity, default mode processing
//! - **schema**: Pattern recognition, abstraction
//! - **basal_ganglia**: Action selection, habit formation
//! - **acc**: Error detection, conflict monitoring
//! - **cerebellum**: Procedural memory, timing, motor learning
//! - **stn**: Response inhibition, task watchdog, emergency brake
//! - **insula**: Interoception, body awareness, empathy, disgust
//! - **hypothalamus**: Drives, homeostasis, circadian rhythms, stress response
//! - **language_cortex**: Code syntax/semantics (Broca's/Wernicke's analog)
//! - **parietal_cortex**: Spatial/structural reasoning, code architecture
//! - **mirror_neurons**: Learning by observation, pattern imitation

pub mod acc;
pub mod amygdala;
pub mod basal_ganglia;
pub mod cerebellum;
pub mod dmn;
pub mod hippocampus;
pub mod hypothalamus;
pub mod insula;
pub mod language_cortex;
pub mod mirror_neurons;
pub mod parietal_cortex;
pub mod prefrontal;
pub mod schema;
pub mod stn;
pub mod thalamus;

// Re-export key types
pub use acc::{ACC, Conflict, ControlSignal, Error, ErrorType};
pub use basal_ganglia::{ActionPattern, BasalGanglia, GateDecision, SelectionResult};
pub use cerebellum::{Cerebellum, Procedure, ProcedureStep, TimingPrediction};
pub use hypothalamus::{
    CircadianPhase, CircadianRhythm, DriveState, DriveType, Hypothalamus, HypothalamusConfig,
    HypothalamusStats, MotivationSummary, StressResponse,
};
pub use insula::{
    BodyState, DisgustResponse, DisgustType, EmpathicResponse, Insula, InsulaConfig, InsulaStats,
    RiskAnticipation, SubjectiveFeeling,
};
pub use language_cortex::{
    CodeIssue, CodePattern, EntityKind, IssueCategory, IssueSeverity, LanguageCortex,
    LanguageCortexConfig, LanguageCortexStats, PatternCategory as LangPatternCategory,
    SemanticEntity, SyntaxElement,
};
pub use mirror_neurons::{
    ActionContext, ActionSource, ActionType, InferredIntent, IntentGoal, LearnedPattern,
    MirrorNeuronConfig, MirrorNeuronStats, MirrorNeuronSystem, ObservedAction,
    PatternCategory as MirrorPatternCategory, Simulation,
};
pub use parietal_cortex::{
    AttentionRegion, DependencyEdge, DependencyKind, Hotspot, Landmark, NavigationPath,
    ParietalCortex, ParietalCortexConfig, ParietalCortexStats, StructureKind, StructureNode,
};
pub use stn::{MonitoredTask, STN, StopReason, StopSignal, TaskConfig, TaskId, TaskState};
