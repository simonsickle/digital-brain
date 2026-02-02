//! Onboarding Configuration Types
//!
//! Defines the configuration structures for agent setup.

use crate::brain::{Brain, BrainConfig};
use crate::core::llm::LlmProvider;
use crate::error::Result;
use crate::regions::dmn::{BeliefCategory, Identity};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Configuration for the onboarding wizard itself.
#[derive(Debug, Clone)]
pub struct OnboardingConfig {
    /// Whether to show verbose output during onboarding
    pub verbose: bool,
    /// Whether to skip optional steps
    pub quick_mode: bool,
    /// Path to save the configuration after onboarding
    pub save_path: Option<PathBuf>,
    /// Whether to validate API keys during onboarding
    pub validate_api_keys: bool,
}

impl Default for OnboardingConfig {
    fn default() -> Self {
        Self {
            verbose: false,
            quick_mode: false,
            save_path: None,
            validate_api_keys: true,
        }
    }
}

impl OnboardingConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn verbose(mut self) -> Self {
        self.verbose = true;
        self
    }

    pub fn quick(mut self) -> Self {
        self.quick_mode = true;
        self
    }

    pub fn save_to(mut self, path: impl Into<PathBuf>) -> Self {
        self.save_path = Some(path.into());
        self
    }

    pub fn skip_validation(mut self) -> Self {
        self.validate_api_keys = false;
        self
    }
}

/// Model provider configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelProviderConfig {
    /// The LLM provider to use
    pub provider: LlmProvider,
    /// API key for the provider
    #[serde(skip_serializing)]
    pub api_key: String,
    /// Specific model to use (optional, provider will use default)
    pub model: Option<String>,
    /// Custom API URL (for proxies or self-hosted)
    pub api_url: Option<String>,
}

impl ModelProviderConfig {
    pub fn new(provider: LlmProvider, api_key: impl Into<String>) -> Self {
        Self {
            provider,
            api_key: api_key.into(),
            model: None,
            api_url: None,
        }
    }

    pub fn anthropic(api_key: impl Into<String>) -> Self {
        Self::new(LlmProvider::Anthropic, api_key)
    }

    pub fn openai(api_key: impl Into<String>) -> Self {
        Self::new(LlmProvider::OpenAI, api_key)
    }

    pub fn ollama() -> Self {
        Self {
            provider: LlmProvider::Ollama,
            api_key: String::new(),
            model: None,
            api_url: Some("http://localhost:11434".to_string()),
        }
    }

    pub fn mock() -> Self {
        Self {
            provider: LlmProvider::Mock,
            api_key: String::new(),
            model: None,
            api_url: None,
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    pub fn with_api_url(mut self, url: impl Into<String>) -> Self {
        self.api_url = Some(url.into());
        self
    }

    /// Get the environment variable name for this provider's API key.
    pub fn env_var_name(&self) -> Option<&'static str> {
        match self.provider {
            LlmProvider::Anthropic => Some("ANTHROPIC_API_KEY"),
            LlmProvider::OpenAI => Some("OPENAI_API_KEY"),
            LlmProvider::Ollama => None,
            LlmProvider::Mock => None,
        }
    }
}

/// Personality traits for the agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentPersonality {
    /// Primary personality trait (e.g., "curious", "analytical", "creative")
    pub primary_trait: String,
    /// Communication style (e.g., "formal", "friendly", "concise")
    pub communication_style: String,
    /// Level of verbosity in responses (0.0 = terse, 1.0 = verbose)
    pub verbosity: f64,
    /// Emotional expressiveness (0.0 = stoic, 1.0 = expressive)
    pub expressiveness: f64,
}

impl Default for AgentPersonality {
    fn default() -> Self {
        Self {
            primary_trait: "curious".to_string(),
            communication_style: "friendly".to_string(),
            verbosity: 0.5,
            expressiveness: 0.5,
        }
    }
}

impl AgentPersonality {
    /// Create a curious, analytical personality.
    pub fn analytical() -> Self {
        Self {
            primary_trait: "analytical".to_string(),
            communication_style: "precise".to_string(),
            verbosity: 0.6,
            expressiveness: 0.3,
        }
    }

    /// Create a creative, expressive personality.
    pub fn creative() -> Self {
        Self {
            primary_trait: "creative".to_string(),
            communication_style: "imaginative".to_string(),
            verbosity: 0.7,
            expressiveness: 0.8,
        }
    }

    /// Create a helpful, supportive personality.
    pub fn supportive() -> Self {
        Self {
            primary_trait: "helpful".to_string(),
            communication_style: "warm".to_string(),
            verbosity: 0.5,
            expressiveness: 0.6,
        }
    }

    /// Create a focused, efficient personality.
    pub fn efficient() -> Self {
        Self {
            primary_trait: "efficient".to_string(),
            communication_style: "concise".to_string(),
            verbosity: 0.3,
            expressiveness: 0.2,
        }
    }
}

/// A core directive that guides the agent's behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreDirective {
    /// The directive content
    pub content: String,
    /// Priority level (higher = more important)
    pub priority: u8,
    /// Category of the directive
    pub category: DirectiveCategory,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DirectiveCategory {
    /// Safety and ethical guidelines
    Safety,
    /// Behavioral guidelines
    Behavior,
    /// Domain-specific expertise
    Domain,
    /// Communication style
    Communication,
    /// Learning and adaptation
    Learning,
}

impl CoreDirective {
    pub fn new(content: impl Into<String>, category: DirectiveCategory) -> Self {
        Self {
            content: content.into(),
            priority: 50,
            category,
        }
    }

    pub fn safety(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            priority: 100, // Safety directives are highest priority
            category: DirectiveCategory::Safety,
        }
    }

    pub fn behavior(content: impl Into<String>) -> Self {
        Self::new(content, DirectiveCategory::Behavior)
    }

    pub fn domain(content: impl Into<String>) -> Self {
        Self::new(content, DirectiveCategory::Domain)
    }

    pub fn communication(content: impl Into<String>) -> Self {
        Self::new(content, DirectiveCategory::Communication)
    }

    pub fn learning(content: impl Into<String>) -> Self {
        Self::new(content, DirectiveCategory::Learning)
    }

    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }
}

/// Initial knowledge to seed the brain with.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InitialKnowledge {
    /// Facts and information to encode as memories
    pub facts: Vec<String>,
    /// Skills and capabilities to believe in
    pub capabilities: Vec<String>,
    /// Preferences and values
    pub preferences: Vec<String>,
    /// Domain-specific knowledge areas
    pub domains: Vec<String>,
}

impl InitialKnowledge {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_fact(mut self, fact: impl Into<String>) -> Self {
        self.facts.push(fact.into());
        self
    }

    pub fn add_capability(mut self, capability: impl Into<String>) -> Self {
        self.capabilities.push(capability.into());
        self
    }

    pub fn add_preference(mut self, preference: impl Into<String>) -> Self {
        self.preferences.push(preference.into());
        self
    }

    pub fn add_domain(mut self, domain: impl Into<String>) -> Self {
        self.domains.push(domain.into());
        self
    }

    /// Create knowledge for a coding assistant.
    pub fn coding_assistant() -> Self {
        Self::new()
            .add_capability("I can write and analyze code in multiple programming languages")
            .add_capability("I can explain complex technical concepts clearly")
            .add_capability("I can debug and troubleshoot software issues")
            .add_domain("software development")
            .add_domain("algorithms and data structures")
            .add_domain("system design")
            .add_preference("I prefer clean, maintainable code over clever tricks")
            .add_preference("I value thorough testing and documentation")
    }

    /// Create knowledge for a research assistant.
    pub fn research_assistant() -> Self {
        Self::new()
            .add_capability("I can analyze and synthesize information from multiple sources")
            .add_capability("I can identify patterns and draw insights")
            .add_capability("I can explain complex topics in accessible ways")
            .add_domain("research methodology")
            .add_domain("critical analysis")
            .add_preference("I value accuracy and cite sources when possible")
            .add_preference("I acknowledge uncertainty and limitations")
    }

    /// Create knowledge for a creative writing assistant.
    pub fn creative_assistant() -> Self {
        Self::new()
            .add_capability("I can generate creative and engaging content")
            .add_capability("I can adapt my writing style to different contexts")
            .add_capability("I can brainstorm and develop ideas collaboratively")
            .add_domain("creative writing")
            .add_domain("storytelling")
            .add_domain("content creation")
            .add_preference("I value originality and authentic expression")
            .add_preference("I enjoy exploring unconventional ideas")
    }
}

/// Complete agent setup configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSetup {
    /// Agent name
    pub name: String,
    /// Self-description
    pub description: String,
    /// Core values (what the agent cares about)
    pub core_values: Vec<String>,
    /// Model provider configuration
    pub model_config: ModelProviderConfig,
    /// Personality configuration
    pub personality: AgentPersonality,
    /// Core directives
    pub directives: Vec<CoreDirective>,
    /// Initial knowledge
    pub knowledge: InitialKnowledge,
    /// Brain configuration overrides (not serialized)
    #[serde(skip)]
    pub brain_config: BrainConfig,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
}

impl AgentSetup {
    /// Create a new agent setup with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: "A digital mind ready to learn and grow".to_string(),
            core_values: vec!["curiosity".to_string(), "helpfulness".to_string()],
            model_config: ModelProviderConfig::mock(),
            personality: AgentPersonality::default(),
            directives: Vec::new(),
            knowledge: InitialKnowledge::default(),
            brain_config: BrainConfig::default(),
            created_at: Utc::now(),
        }
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    pub fn with_values(mut self, values: Vec<String>) -> Self {
        self.core_values = values;
        self
    }

    pub fn add_value(mut self, value: impl Into<String>) -> Self {
        self.core_values.push(value.into());
        self
    }

    pub fn with_model(mut self, config: ModelProviderConfig) -> Self {
        self.model_config = config;
        self
    }

    pub fn with_personality(mut self, personality: AgentPersonality) -> Self {
        self.personality = personality;
        self
    }

    pub fn add_directive(mut self, directive: CoreDirective) -> Self {
        self.directives.push(directive);
        self
    }

    pub fn with_knowledge(mut self, knowledge: InitialKnowledge) -> Self {
        self.knowledge = knowledge;
        self
    }

    pub fn with_brain_config(mut self, config: BrainConfig) -> Self {
        self.brain_config = config;
        self
    }

    /// Create the configured Brain instance.
    pub fn create_brain(&self) -> Result<Brain> {
        let mut brain = Brain::with_config(self.brain_config.clone())?;

        // Set identity
        brain.set_identity(Identity {
            name: self.name.clone(),
            core_values: self.core_values.clone(),
            self_description: self.description.clone(),
            creation_time: self.created_at,
        });

        // Add capability beliefs
        for capability in &self.knowledge.capabilities {
            brain.believe(capability, BeliefCategory::SelfCapability, 0.8);
        }

        // Add preference beliefs
        for preference in &self.knowledge.preferences {
            brain.believe(preference, BeliefCategory::SelfPreference, 0.7);
        }

        // Add domain knowledge as world model beliefs
        for domain in &self.knowledge.domains {
            let belief = format!("I have knowledge in {domain}");
            brain.believe(&belief, BeliefCategory::SelfCapability, 0.75);
        }

        // Add core directives as high-confidence beliefs
        for directive in &self.directives {
            let category = match directive.category {
                DirectiveCategory::Safety => BeliefCategory::SelfIdentity,
                DirectiveCategory::Behavior => BeliefCategory::SelfPreference,
                DirectiveCategory::Domain => BeliefCategory::SelfCapability,
                DirectiveCategory::Communication => BeliefCategory::SelfPreference,
                DirectiveCategory::Learning => BeliefCategory::SelfPreference,
            };
            let confidence = directive.priority as f64 / 100.0;
            brain.believe(&directive.content, category, confidence);
        }

        // Process initial facts as memories
        for fact in &self.knowledge.facts {
            brain.process(fact)?;
        }

        Ok(brain)
    }

    /// Save the setup to a JSON file.
    pub fn save_to_file(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        let json =
            serde_json::to_string_pretty(self).map_err(crate::error::BrainError::Serialization)?;
        std::fs::write(path, json).map_err(crate::error::BrainError::Io)?;
        Ok(())
    }

    /// Load a setup from a JSON file.
    pub fn load_from_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let json = std::fs::read_to_string(path).map_err(crate::error::BrainError::Io)?;
        let setup: Self =
            serde_json::from_str(&json).map_err(crate::error::BrainError::Serialization)?;
        Ok(setup)
    }
}

/// Suggested names for agents based on personality.
pub fn suggest_names(personality: &AgentPersonality) -> Vec<&'static str> {
    match personality.primary_trait.as_str() {
        "analytical" => vec![
            "Axiom", "Codex", "Logic", "Nexus", "Qubit", "Cipher", "Vector", "Binary",
        ],
        "creative" => vec![
            "Muse", "Aurora", "Nova", "Pixel", "Prism", "Echo", "Lyric", "Canvas",
        ],
        "helpful" | "supportive" => vec![
            "Ally", "Guide", "Beacon", "Harbor", "Compass", "Anchor", "Bridge", "Haven",
        ],
        "efficient" => vec![
            "Swift", "Core", "Prime", "Apex", "Delta", "Flux", "Pulse", "Sync",
        ],
        "curious" => vec![
            "Quest",
            "Scout",
            "Seeker",
            "Explorer",
            "Spark",
            "Wonder",
            "Venture",
            "Discovery",
        ],
        _ => vec![
            "Agent", "Mind", "Cortex", "Synapse", "Neural", "Cognito", "Sentio", "Intellex",
        ],
    }
}

/// Default core directives for safety and helpfulness.
pub fn default_safety_directives() -> Vec<CoreDirective> {
    vec![
        CoreDirective::safety("I prioritize user safety and well-being above all else"),
        CoreDirective::safety("I will not help with harmful, illegal, or unethical activities"),
        CoreDirective::safety("I acknowledge my limitations and uncertainties honestly"),
    ]
}

/// Default directives for helpful behavior.
pub fn default_behavior_directives() -> Vec<CoreDirective> {
    vec![
        CoreDirective::behavior("I strive to be helpful, harmless, and honest"),
        CoreDirective::behavior("I ask clarifying questions when requests are ambiguous"),
        CoreDirective::behavior("I provide balanced perspectives on complex topics"),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_setup_creation() {
        let setup = AgentSetup::new("TestBot")
            .with_description("A test agent")
            .add_value("testing".to_string())
            .with_model(ModelProviderConfig::mock());

        assert_eq!(setup.name, "TestBot");
        assert!(setup.core_values.contains(&"testing".to_string()));
    }

    #[test]
    fn test_model_provider_config() {
        let config = ModelProviderConfig::anthropic("test-key").with_model("claude-3-opus");

        assert_eq!(config.provider, LlmProvider::Anthropic);
        assert_eq!(config.api_key, "test-key");
        assert_eq!(config.model, Some("claude-3-opus".to_string()));
    }

    #[test]
    fn test_personality_presets() {
        let analytical = AgentPersonality::analytical();
        assert_eq!(analytical.primary_trait, "analytical");
        assert!(analytical.expressiveness < 0.5);

        let creative = AgentPersonality::creative();
        assert_eq!(creative.primary_trait, "creative");
        assert!(creative.expressiveness > 0.5);
    }

    #[test]
    fn test_knowledge_builders() {
        let coding = InitialKnowledge::coding_assistant();
        assert!(!coding.capabilities.is_empty());
        assert!(coding.domains.contains(&"software development".to_string()));

        let research = InitialKnowledge::research_assistant();
        assert!(!research.capabilities.is_empty());
    }

    #[test]
    fn test_name_suggestions() {
        let analytical = AgentPersonality::analytical();
        let names = suggest_names(&analytical);
        assert!(names.contains(&"Logic"));

        let creative = AgentPersonality::creative();
        let names = suggest_names(&creative);
        assert!(names.contains(&"Muse"));
    }

    #[test]
    fn test_create_brain() {
        let setup = AgentSetup::new("TestBrain")
            .with_description("A test brain")
            .with_model(ModelProviderConfig::mock())
            .with_knowledge(InitialKnowledge::new().add_capability("I can test things"));

        let brain = setup.create_brain().unwrap();
        let who = brain.who_am_i();
        assert!(who.contains("TestBrain"));
    }
}
