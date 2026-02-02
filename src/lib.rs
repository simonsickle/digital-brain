//! # Digital Brain
//!
//! An elegant, modular simulation of consciousness.
//!
//! This crate provides the foundational types and traits for building
//! a digital brain architecture. The design philosophy:
//!
//! - **Modularity**: Each brain region is independent
//! - **Safety**: Rust's type system encodes invariants
//! - **Concurrency**: Modules process signals in parallel
//! - **Elegance**: Capture principles, not neurons
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                     CONSCIOUSNESS LAYER                         │
//! │            (Global Workspace / Attention Routing)               │
//! ├─────────────┬─────────────┬─────────────┬─────────────┬────────┤
//! │  PREFRONTAL │  HIPPOCAMPUS│   AMYGDALA  │  THALAMUS   │  DMN   │
//! │   Working   │   Memory    │  Emotional  │  Attention  │  Self  │
//! │   Memory    │ Consolidate │   Valence   │   Router    │ Model  │
//! └─────────────┴─────────────┴─────────────┴─────────────┴────────┘
//! ```
//!
//! ## Core Components
//!
//! - **Signal Protocol**: Universal inter-module communication
//! - **Hippocampus**: Long-term memory with valence-weighted retrieval
//! - **Prediction Engine**: Dopamine-like surprise and learning modulation
//! - **Global Workspace**: Consciousness layer with attention competition

pub mod agent;
pub mod brain;
pub mod core;
pub mod error;
pub mod regions;
pub mod signal;

pub use agent::{AgentConfig, AgentCycleResult, AgentLoop, AgentState, Percept, PerceptType};
pub use brain::{Brain, BrainConfig, BrainStats, ProcessingResult, SleepReport};
pub use core::{
    ActiveInferencePolicy, ActiveInferenceProposal, BrainRegion, Broadcast, GabaSystem,
    GlobalWorkspace, InhibitionResult, NervousSystem, NervousSystemConfig, NervousSystemStats,
    NeuromodulatorState, NeuromodulatorySystem, OxytocinSystem, Pathway, Prediction,
    PredictionContext, PredictionEngine, PredictionError, PredictionLayer, PredictionState,
    RewardCategory, RewardQuality, SignalTrace, TrustLevel,
};
pub use error::{BrainError, Result};
pub use signal::{Arousal, BrainSignal, MemoryTrace, Salience, SignalType, Valence};

/// Prelude for convenient imports
pub mod prelude {
    pub use crate::core::{
        ActiveInferencePolicy, ActiveInferenceProposal, Broadcast, GabaSystem, GlobalWorkspace,
        InhibitionResult, NeuromodulatorState, NeuromodulatorySystem, OxytocinSystem, Prediction,
        PredictionContext, PredictionEngine, PredictionLayer, PredictionState, RewardCategory,
        RewardQuality, TrustLevel,
    };
    pub use crate::error::*;
    pub use crate::regions::hippocampus::HippocampusStore;
    pub use crate::signal::*;
}
