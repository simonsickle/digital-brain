//! Agent module - the complete autonomous agent built on the digital brain.
//!
//! This module provides the perception-action loop that integrates:
//! - Brain (processing, memory, neuromodulators)
//! - ActionSelector (deciding what to do)
//! - GoalManager (tracking objectives)
//! - WorldModel (representing external state)

pub mod agent_loop;

pub use agent_loop::{
    AgentConfig, AgentCycleResult, AgentLoop, AgentState, Percept, PerceptType,
};
