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
//! - **sensory_cortex**: Modality-specific feature extraction downstream of thalamus
//! - **posterior_parietal**: Multimodal binding and context integration
//! - **motor_cortex**: Action planning and sequencing
//! - **brainstem**: Autonomic control and bodily feedback

pub mod acc;
pub mod amygdala;
pub mod basal_ganglia;
pub mod brainstem;
pub mod cerebellum;
pub mod dmn;
pub mod hippocampus;
pub mod hypothalamus;
pub mod insula;
pub mod motor_cortex;
pub mod posterior_parietal;
pub mod prefrontal;
pub mod schema;
pub mod sensory_cortex;
pub mod stn;
pub mod thalamus;

// Re-export key types
pub use acc::{ACC, Conflict, ControlSignal, Error, ErrorType};
pub use basal_ganglia::{ActionPattern, BasalGanglia, GateDecision, SelectionResult};
pub use brainstem::{AutonomicFeedback, Brainstem, BrainstemConfig};
pub use cerebellum::{
    Cerebellum, ForwardModel, MotorImagery, Procedure, ProcedureStep, TimingPrediction,
};
pub use hypothalamus::{
    CircadianPhase, CircadianRhythm, DriveState, DriveType, Hypothalamus, HypothalamusConfig,
    HypothalamusStats, MotivationSummary, StressResponse,
};
pub use insula::{
    BodyState, DisgustResponse, DisgustType, EmpathicResponse, Insula, InsulaConfig, InsulaStats,
    RiskAnticipation, SubjectiveFeeling,
};
pub use motor_cortex::{MotorCommand, MotorCortex, MotorStep};
pub use posterior_parietal::{MultimodalContext, PosteriorParietalCortex};
pub use sensory_cortex::{
    AuditoryCortex, CorticalRepresentation, GustatoryCortex, OlfactoryCortex, SensoryModality,
    SomatosensoryCortex, VisualCortex,
};
pub use stn::{MonitoredTask, STN, StopReason, StopSignal, TaskConfig, TaskId, TaskState};
