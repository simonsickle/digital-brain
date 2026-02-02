//! Interactive Onboarding Wizard
//!
//! Guides users through the process of setting up a new digital brain agent.

use std::io;

use super::config::{
    AgentPersonality, AgentSetup, CoreDirective, DirectiveCategory, InitialKnowledge,
    ModelProviderConfig, OnboardingConfig, default_behavior_directives, default_safety_directives,
    suggest_names,
};
use super::prompts::TerminalUI;
use crate::brain::BrainConfig;
use crate::error::Result;

/// Interactive wizard for agent onboarding.
pub struct OnboardingWizard {
    config: OnboardingConfig,
    ui: TerminalUI,
}

impl OnboardingWizard {
    /// Create a new onboarding wizard with default settings.
    pub fn new() -> Self {
        Self {
            config: OnboardingConfig::default(),
            ui: TerminalUI::new(),
        }
    }

    /// Create a wizard with custom configuration.
    pub fn with_config(config: OnboardingConfig) -> Self {
        Self {
            config,
            ui: TerminalUI::new(),
        }
    }

    /// Run the full onboarding wizard.
    pub fn run(&self) -> Result<AgentSetup> {
        self.print_welcome();

        // Step 1: Choose personality
        let personality = self.choose_personality()?;

        // Step 2: Choose name
        let name = self.choose_name(&personality)?;

        // Step 3: Configure model provider
        let model_config = self.configure_model_provider()?;

        // Step 4: Set core values
        let core_values = self.set_core_values()?;

        // Step 5: Add directives
        let directives = self.configure_directives()?;

        // Step 6: Initial knowledge (optional in quick mode)
        let knowledge = if self.config.quick_mode {
            self.quick_knowledge(&personality)?
        } else {
            self.configure_knowledge()?
        };

        // Step 7: Set description
        let description = self.set_description(&name, &personality)?;

        // Step 8: Brain configuration (optional)
        let brain_config = if self.config.quick_mode {
            BrainConfig::default()
        } else {
            self.configure_brain()?
        };

        // Build the setup
        let setup = AgentSetup::new(&name)
            .with_description(description)
            .with_values(core_values)
            .with_model(model_config)
            .with_personality(personality)
            .with_knowledge(knowledge);

        // Add directives
        let mut setup = setup;
        for directive in directives {
            setup = setup.add_directive(directive);
        }
        setup = setup.with_brain_config(brain_config);

        // Save if configured
        if let Some(ref path) = self.config.save_path {
            setup.save_to_file(path)?;
            self.ui
                .success(&format!("Configuration saved to {:?}", path));
        }

        self.print_summary(&setup);

        Ok(setup)
    }

    fn print_welcome(&self) {
        self.ui.header("DIGITAL BRAIN ONBOARDING");
        self.ui
            .info("Welcome! Let's create a new digital mind together.");
        self.ui
            .info("Like a parent nurturing a child, you'll help shape");
        self.ui
            .info("this agent's identity, values, and initial knowledge.");
        self.ui.blank();
        self.ui
            .hint("Press Enter to accept default values shown in [brackets]");
        self.ui.blank();
    }

    fn choose_personality(&self) -> io::Result<AgentPersonality> {
        self.ui.section("Step 1: Personality");
        self.ui
            .info("What kind of personality should your agent have?");
        self.ui.blank();

        let options = &[
            "Curious - Eager to learn and explore new ideas",
            "Analytical - Precise, logical, detail-oriented",
            "Creative - Imaginative, expressive, innovative",
            "Supportive - Warm, helpful, empathetic",
            "Efficient - Concise, focused, task-oriented",
            "Custom - Define your own personality",
        ];

        let choice = self.ui.select("Choose a personality type:", options)?;

        let personality = match choice {
            0 => AgentPersonality::default(), // Curious
            1 => AgentPersonality::analytical(),
            2 => AgentPersonality::creative(),
            3 => AgentPersonality::supportive(),
            4 => AgentPersonality::efficient(),
            5 => self.custom_personality()?,
            _ => AgentPersonality::default(),
        };

        self.ui.success(&format!(
            "Personality set: {} with {} communication style",
            personality.primary_trait, personality.communication_style
        ));

        Ok(personality)
    }

    fn custom_personality(&self) -> io::Result<AgentPersonality> {
        self.ui.blank();
        self.ui.info("Let's define a custom personality:");
        self.ui.blank();

        let primary_trait = self.ui.prompt_default("Primary trait", "curious")?;
        let communication_style = self.ui.prompt_default("Communication style", "friendly")?;

        self.ui.info("Rate verbosity (0 = terse, 10 = verbose):");
        let verbosity_str = self.ui.prompt_default("Verbosity", "5")?;
        let verbosity = verbosity_str.parse::<f64>().unwrap_or(5.0) / 10.0;

        self.ui
            .info("Rate expressiveness (0 = stoic, 10 = expressive):");
        let expressiveness_str = self.ui.prompt_default("Expressiveness", "5")?;
        let expressiveness = expressiveness_str.parse::<f64>().unwrap_or(5.0) / 10.0;

        Ok(AgentPersonality {
            primary_trait,
            communication_style,
            verbosity: verbosity.clamp(0.0, 1.0),
            expressiveness: expressiveness.clamp(0.0, 1.0),
        })
    }

    fn choose_name(&self, personality: &AgentPersonality) -> io::Result<String> {
        self.ui.section("Step 2: Name Your Agent");

        let suggestions = suggest_names(personality);
        self.ui
            .info("Here are some name suggestions based on the personality:");
        self.ui.blank();

        for (i, name) in suggestions.iter().enumerate() {
            if i < 4 {
                self.ui.bullet(name);
            }
        }
        self.ui.blank();

        let default_name = suggestions.first().copied().unwrap_or("Agent");
        let name = self.ui.prompt_default("Agent name", default_name)?;

        self.ui.success(&format!("Agent will be named: {}", name));

        Ok(name)
    }

    fn configure_model_provider(&self) -> io::Result<ModelProviderConfig> {
        self.ui.section("Step 3: Model Provider");
        self.ui.info("Choose which AI model provider to use:");
        self.ui.blank();

        let options = &[
            "Anthropic (Claude) - Recommended",
            "OpenAI (GPT models)",
            "Ollama (Local models)",
            "Mock (For testing, no API needed)",
        ];

        let choice = self.ui.select("Select provider:", options)?;

        let config = match choice {
            0 => self.configure_anthropic()?,
            1 => self.configure_openai()?,
            2 => self.configure_ollama()?,
            3 => {
                self.ui.info("Using mock provider - no API key required");
                ModelProviderConfig::mock()
            }
            _ => ModelProviderConfig::mock(),
        };

        self.ui
            .success(&format!("Model provider configured: {}", config.provider));

        Ok(config)
    }

    fn configure_anthropic(&self) -> io::Result<ModelProviderConfig> {
        self.ui.blank();
        self.ui.info("Anthropic Claude Configuration");
        self.ui
            .hint("Get your API key at: https://console.anthropic.com/");
        self.ui.blank();

        // Check for environment variable first
        if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
            let masked = format!("{}...{}", &key[..8], &key[key.len() - 4..]);
            self.ui.info(&format!(
                "Found ANTHROPIC_API_KEY in environment: {}",
                masked
            ));

            if self.ui.confirm("Use this API key?", true)? {
                return self.select_anthropic_model(key);
            }
        }

        let api_key = self.ui.prompt_secret("Enter Anthropic API key:")?;

        if api_key.is_empty() {
            self.ui.warning("No API key provided - using mock provider");
            return Ok(ModelProviderConfig::mock());
        }

        self.select_anthropic_model(api_key)
    }

    fn select_anthropic_model(&self, api_key: String) -> io::Result<ModelProviderConfig> {
        self.ui.blank();
        self.ui.info("Select a Claude model:");
        self.ui.blank();

        let options = &[
            "Claude Opus 4.5 - Most capable, best for complex tasks",
            "Claude Sonnet 4 - Balanced performance and speed",
            "Claude Sonnet 3.5 - Previous generation, reliable",
            "Claude Haiku 3.5 - Fast and efficient",
        ];

        let choice = self.ui.select("Choose model:", options)?;

        let model = match choice {
            0 => "claude-opus-4-5-20250131",
            1 => "claude-sonnet-4-20250514",
            2 => "claude-3-5-sonnet-20241022",
            3 => "claude-3-5-haiku-20241022",
            _ => "claude-sonnet-4-20250514",
        };

        Ok(ModelProviderConfig::anthropic(api_key).with_model(model))
    }

    fn configure_openai(&self) -> io::Result<ModelProviderConfig> {
        self.ui.blank();
        self.ui.info("OpenAI Configuration");
        self.ui
            .hint("Get your API key at: https://platform.openai.com/");
        self.ui.blank();

        // Check environment variable
        if let Ok(key) = std::env::var("OPENAI_API_KEY") {
            let masked = if key.len() > 12 {
                format!("{}...{}", &key[..8], &key[key.len() - 4..])
            } else {
                "***".to_string()
            };
            self.ui
                .info(&format!("Found OPENAI_API_KEY in environment: {}", masked));

            if self.ui.confirm("Use this API key?", true)? {
                return Ok(ModelProviderConfig::openai(key).with_model("gpt-4"));
            }
        }

        let api_key = self.ui.prompt_secret("Enter OpenAI API key:")?;

        if api_key.is_empty() {
            self.ui.warning("No API key provided - using mock provider");
            return Ok(ModelProviderConfig::mock());
        }

        self.ui.blank();
        let model = self.ui.prompt_default("Model name", "gpt-4")?;

        Ok(ModelProviderConfig::openai(api_key).with_model(model))
    }

    fn configure_ollama(&self) -> io::Result<ModelProviderConfig> {
        self.ui.blank();
        self.ui.info("Ollama Local Configuration");
        self.ui.hint("Make sure Ollama is running locally");
        self.ui.blank();

        let url = self
            .ui
            .prompt_default("Ollama URL", "http://localhost:11434")?;
        let model = self.ui.prompt_default("Model name", "llama2")?;

        Ok(ModelProviderConfig::ollama()
            .with_api_url(url)
            .with_model(model))
    }

    fn set_core_values(&self) -> io::Result<Vec<String>> {
        self.ui.section("Step 4: Core Values");
        self.ui
            .info("What values should guide your agent's behavior?");
        self.ui.blank();

        let options = &[
            "Curiosity - Always eager to learn and explore",
            "Helpfulness - Focused on being useful to users",
            "Honesty - Truthful and transparent",
            "Creativity - Innovative and imaginative",
            "Precision - Accurate and detail-oriented",
            "Empathy - Understanding and compassionate",
            "Efficiency - Practical and results-focused",
            "Wisdom - Thoughtful and considered",
        ];

        let choices = self
            .ui
            .multi_select("Select core values (choose 2-4):", options)?;

        let values: Vec<String> = choices
            .iter()
            .map(|&i| {
                options[i]
                    .split(" - ")
                    .next()
                    .unwrap_or("value")
                    .to_lowercase()
            })
            .collect();

        if values.is_empty() {
            self.ui.warning("No values selected, using defaults");
            return Ok(vec!["curiosity".to_string(), "helpfulness".to_string()]);
        }

        // Allow custom values
        if self.ui.confirm("Add custom values?", false)? {
            let custom = self
                .ui
                .prompt_multiline("Enter custom values (one per line):", "done")?;
            let mut all_values = values;
            all_values.extend(custom.into_iter().map(|s| s.to_lowercase()));
            self.ui
                .success(&format!("Core values set: {:?}", all_values));
            return Ok(all_values);
        }

        self.ui.success(&format!("Core values set: {:?}", values));
        Ok(values)
    }

    fn configure_directives(&self) -> io::Result<Vec<CoreDirective>> {
        self.ui.section("Step 5: Core Directives");
        self.ui
            .info("Directives are rules that guide the agent's behavior.");
        self.ui.blank();

        let mut directives = Vec::new();

        // Always include safety directives
        if self
            .ui
            .confirm("Include standard safety directives?", true)?
        {
            directives.extend(default_safety_directives());
            self.ui.success("Added safety directives");
        }

        // Behavioral directives
        if self
            .ui
            .confirm("Include helpful behavior directives?", true)?
        {
            directives.extend(default_behavior_directives());
            self.ui.success("Added behavior directives");
        }

        // Custom directives
        if !self.config.quick_mode && self.ui.confirm("Add custom directives?", false)? {
            self.ui
                .info("Enter custom directives (they guide your agent's behavior):");
            let custom = self
                .ui
                .prompt_multiline("Enter directives (one per line):", "done")?;

            for directive_text in custom {
                self.ui.blank();
                self.ui.info(&format!("Directive: {}", directive_text));
                let category_options = &[
                    "Safety - Ethical guidelines",
                    "Behavior - How to act",
                    "Domain - Area of expertise",
                    "Communication - How to communicate",
                    "Learning - How to learn and adapt",
                ];
                let cat_choice = self
                    .ui
                    .select("What category is this directive?", category_options)?;

                let category = match cat_choice {
                    0 => DirectiveCategory::Safety,
                    1 => DirectiveCategory::Behavior,
                    2 => DirectiveCategory::Domain,
                    3 => DirectiveCategory::Communication,
                    4 => DirectiveCategory::Learning,
                    _ => DirectiveCategory::Behavior,
                };

                directives.push(CoreDirective::new(directive_text, category));
            }
        }

        self.ui
            .success(&format!("Configured {} directives", directives.len()));

        Ok(directives)
    }

    fn configure_knowledge(&self) -> io::Result<InitialKnowledge> {
        self.ui.section("Step 6: Initial Knowledge");
        self.ui
            .info("Let's give your agent some initial knowledge and capabilities.");
        self.ui.blank();

        let preset_options = &[
            "Start blank - No preset knowledge",
            "Coding Assistant - Programming and development",
            "Research Assistant - Analysis and synthesis",
            "Creative Assistant - Writing and ideation",
            "Custom - Define your own",
        ];

        let choice = self
            .ui
            .select("Choose a knowledge preset:", preset_options)?;

        let mut knowledge = match choice {
            0 => InitialKnowledge::new(),
            1 => InitialKnowledge::coding_assistant(),
            2 => InitialKnowledge::research_assistant(),
            3 => InitialKnowledge::creative_assistant(),
            4 => self.custom_knowledge()?,
            _ => InitialKnowledge::new(),
        };

        // Allow adding more knowledge
        if self.ui.confirm("Add additional knowledge?", false)? {
            self.ui.blank();

            // Capabilities
            self.ui.info("Add capabilities (what can this agent do?):");
            let capabilities = self.ui.prompt_multiline("Enter capabilities:", "done")?;
            for cap in capabilities {
                knowledge.capabilities.push(cap);
            }

            // Domains
            self.ui.info("Add knowledge domains:");
            let domains = self.ui.prompt_multiline("Enter domains:", "done")?;
            for domain in domains {
                knowledge.domains.push(domain);
            }

            // Initial facts
            if self.ui.confirm("Add initial facts to remember?", false)? {
                self.ui.info("Enter facts for the agent to remember:");
                let facts = self.ui.prompt_multiline("Enter facts:", "done")?;
                for fact in facts {
                    knowledge.facts.push(fact);
                }
            }
        }

        self.ui.success(&format!(
            "Knowledge configured: {} capabilities, {} domains",
            knowledge.capabilities.len(),
            knowledge.domains.len()
        ));

        Ok(knowledge)
    }

    fn custom_knowledge(&self) -> io::Result<InitialKnowledge> {
        self.ui.blank();
        self.ui.info("Let's define custom knowledge:");
        self.ui.blank();

        self.ui.info("What capabilities should the agent have?");
        let capabilities = self
            .ui
            .prompt_multiline("Enter capabilities (things the agent can do):", "done")?;

        self.ui.blank();
        self.ui.info("What domains should the agent know about?");
        let domains = self
            .ui
            .prompt_multiline("Enter knowledge domains:", "done")?;

        self.ui.blank();
        self.ui.info("What preferences should guide the agent?");
        let preferences = self.ui.prompt_multiline("Enter preferences:", "done")?;

        Ok(InitialKnowledge {
            facts: Vec::new(),
            capabilities,
            preferences,
            domains,
        })
    }

    fn quick_knowledge(&self, personality: &AgentPersonality) -> io::Result<InitialKnowledge> {
        // In quick mode, select a preset based on personality
        let knowledge = match personality.primary_trait.as_str() {
            "analytical" => InitialKnowledge::coding_assistant(),
            "creative" => InitialKnowledge::creative_assistant(),
            _ => InitialKnowledge::research_assistant(),
        };

        self.ui.info(&format!(
            "Auto-selected knowledge preset based on {} personality",
            personality.primary_trait
        ));

        Ok(knowledge)
    }

    fn set_description(&self, name: &str, personality: &AgentPersonality) -> io::Result<String> {
        self.ui.section("Step 7: Self-Description");
        self.ui.info("How should the agent describe itself?");
        self.ui.blank();

        let default = format!(
            "A {} digital mind with a {} approach",
            personality.primary_trait, personality.communication_style
        );

        let description = self
            .ui
            .prompt_default(&format!("Description for {}", name), &default)?;

        self.ui.success("Description set");

        Ok(description)
    }

    fn configure_brain(&self) -> io::Result<BrainConfig> {
        self.ui.section("Step 8: Brain Configuration");
        self.ui
            .info("Advanced brain settings (press Enter for defaults):");
        self.ui.blank();

        let working_memory_str = self.ui.prompt_default("Working memory capacity", "7")?;
        let working_memory = working_memory_str.parse().unwrap_or(7);

        let consciousness_str = self.ui.prompt_default("Consciousness capacity", "5")?;
        let consciousness = consciousness_str.parse().unwrap_or(5);

        let verbose = self.ui.confirm("Enable verbose logging?", false)?;

        // Memory persistence
        let memory_path = if self.ui.confirm("Use persistent memory?", false)? {
            let path = self.ui.prompt("Memory database path:")?;
            if path.is_empty() { None } else { Some(path) }
        } else {
            None
        };

        Ok(BrainConfig {
            memory_path,
            working_memory_capacity: working_memory,
            consciousness_capacity: consciousness,
            verbose,
        })
    }

    fn print_summary(&self, setup: &AgentSetup) {
        self.ui.header("ONBOARDING COMPLETE");

        self.ui.info(&format!("Name: {}", setup.name));
        self.ui.info(&format!("Description: {}", setup.description));
        self.ui
            .info(&format!("Provider: {}", setup.model_config.provider));
        self.ui.info(&format!(
            "Personality: {} ({})",
            setup.personality.primary_trait, setup.personality.communication_style
        ));
        self.ui
            .info(&format!("Core Values: {:?}", setup.core_values));
        self.ui
            .info(&format!("Directives: {}", setup.directives.len()));
        self.ui.info(&format!(
            "Capabilities: {}",
            setup.knowledge.capabilities.len()
        ));
        self.ui
            .info(&format!("Domains: {}", setup.knowledge.domains.len()));

        self.ui.blank();
        self.ui.divider();
        self.ui.blank();
        self.ui
            .success(&format!("{} is ready to come to life!", setup.name));
        self.ui
            .hint("Use setup.create_brain() to instantiate the brain");
        self.ui.blank();
    }
}

impl Default for OnboardingWizard {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wizard_creation() {
        let wizard = OnboardingWizard::new();
        assert!(!wizard.config.quick_mode);
        assert!(wizard.config.validate_api_keys);
    }

    #[test]
    fn test_wizard_with_config() {
        let config = OnboardingConfig::new().quick().skip_validation();
        let wizard = OnboardingWizard::with_config(config);
        assert!(wizard.config.quick_mode);
        assert!(!wizard.config.validate_api_keys);
    }
}
