//! Agent module - the complete autonomous agent built on the digital brain.
//!
//! This module provides the perception-action loop that integrates:
//! - Brain (processing, memory, neuromodulators)
//! - ActionSelector (deciding what to do)
//! - GoalManager (tracking objectives)
//! - WorldModel (representing external state)
//! - CommunicationSystem (structured output generation)
//! - MultiAgentSystem (inter-agent communication and theory of mind)

pub mod agent_loop;
pub mod communication;
pub mod multi_agent;

pub use agent_loop::{
    AgentConfig, AgentCycleResult, AgentLoop, AgentState, Percept, PerceptType,
};
pub use communication::{
    CommunicationIntent, CommunicationStats, CommunicationStyle, CommunicationSystem,
    ConversationMessage, IntentType,
};
pub use multi_agent::{
    AgentId, AgentMessage, AgentModel, AgentProfile, MessageType, MultiAgentStats,
    MultiAgentSystem,
};
