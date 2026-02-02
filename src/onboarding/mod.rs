//! Agent Onboarding System
//!
//! Interactive terminal wizard for configuring and initializing a digital brain agent.
//! Like a parent nurturing a new mind, this module guides users through:
//!
//! - Model provider selection (Anthropic, OpenAI, etc.)
//! - API key configuration
//! - Agent naming and identity
//! - Core values and directives
//! - Initial beliefs and knowledge
//! - Memory and system configuration
//!
//! # Usage
//!
//! ```rust,ignore
//! use digital_brain::onboarding::OnboardingWizard;
//!
//! // Run the interactive wizard
//! let agent_setup = OnboardingWizard::new().run()?;
//!
//! // Use the setup to create a brain
//! let brain = agent_setup.create_brain()?;
//! ```

mod config;
mod prompts;
mod wizard;

pub use config::{
    AgentPersonality, AgentSetup, CoreDirective, InitialKnowledge, ModelProviderConfig,
    OnboardingConfig,
};
pub use prompts::{TerminalStyle, TerminalUI};
pub use wizard::OnboardingWizard;
