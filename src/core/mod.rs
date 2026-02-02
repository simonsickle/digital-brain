//! Core systems - cross-cutting functionality.

pub mod action;
pub mod attention;
pub mod boredom;
pub mod cognition;
pub mod consciousness;
pub mod curiosity;
pub mod emotion;
pub mod goals;
pub mod imagination;
pub mod inner_speech;
pub mod llm;
pub mod nervous_system;
pub mod neuromodulators;
pub mod planning;
pub mod prediction;
pub mod salience;
pub mod self_model;
pub mod sensory;
pub mod sleep;
pub mod social;
pub mod stimulus;
pub mod strategy;
pub mod temporal;
pub mod workspace;
pub mod world_model;

pub use action::{
    ActionCategory, ActionDecision, ActionId, ActionSelector, ActionStats, ActionTemplate,
    Condition, ExpectedOutcome, Outcome,
};
pub use attention::{
    AttentionBudget, AttentionStats, COMPLEXITY_KEYWORDS, TaskComplexity, estimate_complexity,
};
pub use boredom::{
    ActionEntropyTracker, ActivityFingerprint, BoredomAssessment, BoredomConfig, BoredomFactors,
    BoredomRecommendation, BoredomSignalSource, BoredomStats, BoredomTracker, ProgressTracker,
};
pub use cognition::{
    CognitionConfig, CognitionEngine, CognitiveContext, CognitiveProcessor, EmotionalSnapshot,
    GoalSnippet, MemorySnippet, ParsedAction, ProcessedStimulus,
};
pub use consciousness::{
    ActionResult, AttentionFocus, ConsciousAction, ConsciousnessConfig, ConsciousnessLoop,
    ConsciousnessState, ConsciousnessStats, CycleResult, EchoProcessor, LoopControl,
    ProcessingContext, StimulusProcessor,
};
pub use curiosity::{Competence, CuriosityStats, CuriositySystem, Domain, ExplorationEvent};
pub use emotion::{
    ActionTendency, Appraisal, CoreAffect, EmotionCategory, EmotionSystem, EmotionalState, Mood,
    RegulationStrategy,
};
pub use goals::{
    Criterion, Goal, GoalEvent, GoalId, GoalManager, GoalStats, GoalStatus, Priority, TimeHorizon,
};
pub use imagination::{
    DreamResult, ImaginationConfig, ImaginationEngine, ImaginationError, ImaginationStats,
    ImaginationType, Imagining, ImaginingId, MemorySource, RecombinationResult, SimulationResult,
    TypeStats,
};
pub use inner_speech::{
    InnerSpeechConfig, InnerSpeechStats, InnerSpeechSystem, InnerSpeechType, InnerUtterance,
};
pub use llm::{
    AnthropicBackend, AnthropicConfig, AnthropicModel, ChatMessage, LlmBackend, LlmError,
    LlmErrorKind, LlmProvider, LlmRequestConfig, LlmResponse, LlmUsage, MockLlmBackend,
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
pub use planning::{ImaginationPlanner, ImaginationPlannerConfig, PlanningSuggestion};
pub use prediction::{
    ActiveInferencePolicy, ActiveInferenceProposal, Prediction, PredictionContext,
    PredictionEngine, PredictionError, PredictionLayer, PredictionState, PredictionStats,
};
pub use salience::{SalienceInputs, SalienceNetwork, SalienceOutcome};
pub use self_model::{
    AutobiographicalSelf, LifeChapter, LifeEvent, MetacognitiveMonitor, MetacognitiveState,
    SelfConcept, SelfModel, SelfTrait, Value,
};
pub use sensory::{
    ClockConfig, ClockStream, FileSystemConfig, FileSystemStream, PromptSender, PromptStream,
    SensoryCortex, SensoryStream,
};
pub use sleep::{
    Dream, SleepConfig, SleepSession, SleepStage, SleepStats, SleepSystem, SleepTickResult,
    WakeReason,
};
pub use social::{SocialAgentProfile, SocialCognition, SocialUpdate, TheoryOfMindState};
pub use stimulus::{
    DriveEvent, FileEvent, GoalEvent as StimulusGoalEvent, Stimulus, StimulusKind,
    StimulusPriority, StimulusResponse, StimulusSource, SystemEvent, TimeEvent,
};
pub use strategy::{
    StrategyNarrative, StrategyProfile, StrategyRegulator, StrategySignal, StrategyUpdate,
};
pub use temporal::{
    DurationPerception, Intention, IntentionId, MentalTimeTravel, ProspectiveMemory,
    ProspectiveMemoryStats, ProspectiveTrigger, TemporalCognition, TemporalDiscounting,
    TemporalMoment, TemporalStats,
};
pub use workspace::{Broadcast, GlobalWorkspace, WorkspaceConfig};
pub use world_model::{
    Entity, EntityId, PredictionId, PropertyValue, RelationType, Relationship, WorldModel,
    WorldModelStats, WorldPrediction,
};
