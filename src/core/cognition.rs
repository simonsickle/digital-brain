//! Cognition Engine - LLM Integration for Thinking
//!
//! This module bridges the brain's signal processing with actual reasoning.
//! The LLM serves as the "cortex" - it receives processed stimuli and context,
//! then generates thoughts, decisions, and actions.
//!
//! # Architecture
//!
//! ```text
//! Stimulus → Thalamus → Global Workspace → [COGNITION] → Action
//!                              ↑                ↓
//!                          Context         Memory Update
//!                        (memories,        (learning)
//!                         goals,
//!                         state)
//! ```
//!
//! The cognition engine:
//! 1. Receives processed stimuli from the consciousness loop
//! 2. Gathers relevant context (memories, goals, current state)
//! 3. Constructs a prompt for the LLM
//! 4. Parses LLM output into structured actions
//! 5. Optionally generates internal thoughts (for mind-wandering)

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::core::consciousness::{
    ActionResult, ConsciousAction, ProcessingContext, StimulusProcessor,
};
use crate::core::llm::{LlmBackend, LlmRequestConfig};
use crate::core::neuromodulators::NeuromodulatorState;
use crate::core::stimulus::{Stimulus, StimulusKind};

/// Configuration for the cognition engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitionConfig {
    /// Model identifier (e.g., "gpt-4", "claude-3", "local/llama")
    pub model: String,
    /// Maximum tokens for response
    pub max_tokens: u32,
    /// Temperature (creativity) - modulated by neuromodulators
    pub base_temperature: f64,
    /// System prompt template
    pub system_prompt: String,
    /// Whether to include emotional state in prompts
    pub include_emotional_state: bool,
    /// Whether to include recent memories
    pub include_memories: bool,
    /// Maximum memories to include
    pub max_memories: usize,
    /// Enable internal monologue (thoughts between actions)
    pub enable_inner_monologue: bool,
}

impl Default for CognitionConfig {
    fn default() -> Self {
        Self {
            model: "gpt-4".to_string(),
            max_tokens: 1024,
            base_temperature: 0.7,
            system_prompt: DEFAULT_SYSTEM_PROMPT.to_string(),
            include_emotional_state: true,
            include_memories: true,
            max_memories: 10,
            enable_inner_monologue: true,
        }
    }
}

const DEFAULT_SYSTEM_PROMPT: &str = r#"You are an autonomous cognitive agent with:
- Curiosity that drives exploration
- Goals that guide behavior  
- Emotions that color experience
- Memory that provides continuity

Current emotional state will be provided. Use it to modulate your responses:
- High arousal → more urgent, focused responses
- Positive valence → more optimistic, creative
- High curiosity → more exploratory, questioning
- High boredom → seek novelty, try new approaches

You can take these actions:
- RESPOND: <message> - Reply to input
- THINK: <thought> - Internal reflection (not visible externally)
- OBSERVE: <target> - Look at something (file, directory, etc.)
- CREATE: <artifact> - Make something new
- EXECUTE: <command> - Run a tool/command
- REFOCUS: <target> - Shift attention to something else
- REQUEST_INPUT: <prompt> - Ask for human input
- IDLE - Do nothing, continue current state

Respond with your chosen action in the format:
ACTION: <action_type>
CONTENT: <content>
REASONING: <why you chose this>
"#;

/// Represents the cognitive context passed to the LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveContext {
    /// Current stimulus being processed
    pub stimulus: Option<ProcessedStimulus>,
    /// Recent memories
    pub memories: Vec<MemorySnippet>,
    /// Active goals
    pub goals: Vec<GoalSnippet>,
    /// Current emotional/neuromodulator state
    pub emotional_state: EmotionalSnapshot,
    /// Recent actions taken
    pub recent_actions: Vec<String>,
    /// Current focus
    pub current_focus: Option<String>,
    /// Time context
    pub timestamp: DateTime<Utc>,
    /// Cycle number
    pub cycle: u64,
}

/// A processed stimulus ready for cognition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessedStimulus {
    pub kind: String,
    pub content: String,
    pub salience: f64,
    pub source: String,
    pub requires_response: bool,
}

impl From<&Stimulus> for ProcessedStimulus {
    fn from(s: &Stimulus) -> Self {
        let (kind, content) = match &s.kind {
            StimulusKind::ExternalPrompt { content, .. } => {
                ("external_prompt".to_string(), content.clone())
            }
            StimulusKind::FileSystem(e) => ("file_event".to_string(), format!("{:?}", e)),
            StimulusKind::Time(e) => ("time_event".to_string(), format!("{:?}", e)),
            StimulusKind::Drive(e) => ("internal_drive".to_string(), format!("{:?}", e)),
            StimulusKind::Goal(e) => ("goal_event".to_string(), format!("{:?}", e)),
            StimulusKind::System(e) => ("system_event".to_string(), format!("{:?}", e)),
            StimulusKind::InternalThought { content, .. } => {
                ("internal_thought".to_string(), content.clone())
            }
            StimulusKind::QueryResponse { result, .. } => {
                ("query_response".to_string(), result.to_string())
            }
            StimulusKind::Observation { content, .. } => ("observation".to_string(), content.clone()),
        };

        Self {
            kind,
            content,
            salience: s.salience,
            source: format!("{:?}", s.source),
            requires_response: s.requires_response,
        }
    }
}

/// A memory snippet for context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySnippet {
    pub content: String,
    pub relevance: f64,
    pub timestamp: DateTime<Utc>,
    pub valence: f64,
}

/// A goal snippet for context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoalSnippet {
    pub description: String,
    pub priority: f64,
    pub progress: f64,
    pub deadline: Option<DateTime<Utc>>,
}

/// Snapshot of emotional/neuromodulator state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalSnapshot {
    pub valence: f64,      // -1 to 1 (negative to positive)
    pub arousal: f64,      // 0 to 1 (calm to excited)
    pub curiosity: f64,    // 0 to 1
    pub boredom: f64,      // 0 to 1
    pub stress: f64,       // 0 to 1
    pub motivation: f64,   // 0 to 1
    pub exploration_drive: f64,
}

impl From<&NeuromodulatorState> for EmotionalSnapshot {
    fn from(n: &NeuromodulatorState) -> Self {
        Self {
            valence: n.serotonin - 0.5, // Map to -0.5 to 0.5 range
            arousal: n.norepinephrine,
            curiosity: n.acetylcholine,
            boredom: 0.0, // Would come from boredom tracker
            stress: n.stress,
            motivation: n.motivation,
            exploration_drive: n.exploration_drive,
        }
    }
}

/// Parsed action from LLM response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedAction {
    pub action_type: String,
    pub content: String,
    pub reasoning: Option<String>,
}

/// The main cognition engine
pub struct CognitionEngine {
    config: CognitionConfig,
    backend: Box<dyn LlmBackend>,
    thought_history: Vec<String>,
    action_history: Vec<ParsedAction>,
}

impl CognitionEngine {
    /// Create a new cognition engine
    pub fn new(config: CognitionConfig, backend: Box<dyn LlmBackend>) -> Self {
        Self {
            config,
            backend,
            thought_history: Vec::new(),
            action_history: Vec::new(),
        }
    }

    /// Build a prompt from cognitive context
    pub fn build_prompt(&self, context: &CognitiveContext) -> String {
        let mut prompt = String::new();

        // System context
        prompt.push_str(&self.config.system_prompt);
        prompt.push_str("\n\n---\n\n");

        // Emotional state
        if self.config.include_emotional_state {
            prompt.push_str("## Current State\n");
            prompt.push_str(&format!(
                "- Valence: {:.2} ({})\n",
                context.emotional_state.valence,
                if context.emotional_state.valence > 0.2 {
                    "positive"
                } else if context.emotional_state.valence < -0.2 {
                    "negative"
                } else {
                    "neutral"
                }
            ));
            prompt.push_str(&format!(
                "- Arousal: {:.2} ({})\n",
                context.emotional_state.arousal,
                if context.emotional_state.arousal > 0.7 {
                    "high"
                } else if context.emotional_state.arousal < 0.3 {
                    "low"
                } else {
                    "moderate"
                }
            ));
            prompt.push_str(&format!(
                "- Curiosity: {:.2}\n",
                context.emotional_state.curiosity
            ));
            prompt.push_str(&format!(
                "- Exploration drive: {:.2}\n",
                context.emotional_state.exploration_drive
            ));
            if context.emotional_state.boredom > 0.5 {
                prompt.push_str(&format!(
                    "- ⚠️ Boredom: {:.2} (seeking novelty)\n",
                    context.emotional_state.boredom
                ));
            }
            if context.emotional_state.stress > 0.6 {
                prompt.push_str(&format!(
                    "- ⚠️ Stress: {:.2} (consider taking a break)\n",
                    context.emotional_state.stress
                ));
            }
            prompt.push_str("\n");
        }

        // Current focus
        if let Some(ref focus) = context.current_focus {
            prompt.push_str(&format!("## Current Focus\n{}\n\n", focus));
        }

        // Goals
        if !context.goals.is_empty() {
            prompt.push_str("## Active Goals\n");
            for goal in &context.goals {
                prompt.push_str(&format!(
                    "- {} (priority: {:.1}, progress: {:.0}%)\n",
                    goal.description,
                    goal.priority,
                    goal.progress * 100.0
                ));
            }
            prompt.push_str("\n");
        }

        // Recent memories
        if self.config.include_memories && !context.memories.is_empty() {
            prompt.push_str("## Relevant Memories\n");
            for mem in &context.memories {
                prompt.push_str(&format!("- {}\n", mem.content));
            }
            prompt.push_str("\n");
        }

        // Recent actions
        if !context.recent_actions.is_empty() {
            prompt.push_str("## Recent Actions\n");
            for action in context.recent_actions.iter().take(5) {
                prompt.push_str(&format!("- {}\n", action));
            }
            prompt.push_str("\n");
        }

        // Current stimulus
        if let Some(ref stimulus) = context.stimulus {
            prompt.push_str("## Current Input\n");
            prompt.push_str(&format!("Type: {}\n", stimulus.kind));
            prompt.push_str(&format!("Content: {}\n", stimulus.content));
            prompt.push_str(&format!("Salience: {:.2}\n", stimulus.salience));
            if stimulus.requires_response {
                prompt.push_str("⚠️ This requires a response.\n");
            }
            prompt.push_str("\n");
        } else {
            prompt.push_str("## Current Input\nNo external input. You may think, explore, or idle.\n\n");
        }

        prompt.push_str("## Your Action\n");
        prompt
    }

    /// Parse LLM response into an action
    pub fn parse_response(&self, response: &str) -> Option<ParsedAction> {
        let mut action_type = None;
        let mut content = None;
        let mut reasoning = None;

        for line in response.lines() {
            let line = line.trim();
            if line.starts_with("ACTION:") {
                action_type = Some(line.trim_start_matches("ACTION:").trim().to_string());
            } else if line.starts_with("CONTENT:") {
                content = Some(line.trim_start_matches("CONTENT:").trim().to_string());
            } else if line.starts_with("REASONING:") {
                reasoning = Some(line.trim_start_matches("REASONING:").trim().to_string());
            }
        }

        // Also try to get multi-line content
        if content.is_none() {
            if let Some(start) = response.find("CONTENT:") {
                let after_content = &response[start + 8..];
                if let Some(end) = after_content.find("REASONING:") {
                    content = Some(after_content[..end].trim().to_string());
                } else {
                    content = Some(after_content.trim().to_string());
                }
            }
        }

        Some(ParsedAction {
            action_type: action_type.unwrap_or_else(|| "THINK".to_string()),
            content: content.unwrap_or_else(|| response.to_string()),
            reasoning,
        })
    }

    /// Convert parsed action to ConsciousAction
    pub fn to_conscious_action(&self, parsed: &ParsedAction) -> ConsciousAction {
        match parsed.action_type.to_uppercase().as_str() {
            "RESPOND" => ConsciousAction::Respond {
                content: parsed.content.clone(),
                to: None,
            },
            "THINK" => ConsciousAction::Think {
                thought: parsed.content.clone(),
            },
            "OBSERVE" => ConsciousAction::Observe {
                target: parsed.content.clone(),
            },
            "CREATE" => ConsciousAction::Create {
                artifact: "unknown".to_string(),
                content: parsed.content.clone(),
            },
            "EXECUTE" => ConsciousAction::Execute {
                tool: parsed.content.clone(),
                args: serde_json::Value::Null,
            },
            "REFOCUS" => ConsciousAction::Refocus {
                target: parsed.content.clone(),
                reason: parsed.reasoning.clone().unwrap_or_default(),
            },
            "REQUEST_INPUT" => ConsciousAction::RequestInput {
                prompt: parsed.content.clone(),
            },
            "IDLE" | _ => ConsciousAction::Idle,
        }
    }

    /// Calculate temperature based on neuromodulator state
    pub fn calculate_temperature(&self, emotional_state: &EmotionalSnapshot) -> f64 {
        let base = self.config.base_temperature;
        
        // High exploration drive → higher temperature
        let exploration_mod = emotional_state.exploration_drive * 0.2;
        
        // High stress → lower temperature (more conservative)
        let stress_mod = -emotional_state.stress * 0.15;
        
        // High boredom → higher temperature (try new things)
        let boredom_mod = emotional_state.boredom * 0.25;
        
        (base + exploration_mod + stress_mod + boredom_mod).clamp(0.1, 1.5)
    }

    /// Generate a thought for mind-wandering
    pub async fn mind_wander(&mut self, context: &CognitiveContext) -> Option<String> {
        if !self.config.enable_inner_monologue {
            return None;
        }

        let prompt = format!(
            "{}\n\n---\n\n## Mind Wandering\n\
            You have no immediate input. Let your mind wander.\n\
            Consider: recent experiences, goals, curiosities, or just reflect.\n\
            Generate a brief internal thought.\n\n\
            THINK:",
            self.config.system_prompt
        );

        let config = LlmRequestConfig::new()
            .with_max_tokens(150)
            .with_temperature(0.9) // High creativity for mind-wandering
            .with_stop("\n\n");

        match self.backend.complete(&prompt, &config).await {
            Ok(response) => {
                let thought = response.content.trim().to_string();
                self.thought_history.push(thought.clone());
                Some(thought)
            }
            Err(_) => None,
        }
    }

    /// Process a stimulus through cognition
    pub async fn process(
        &mut self,
        context: CognitiveContext,
    ) -> Option<ConsciousAction> {
        let prompt = self.build_prompt(&context);
        let temperature = self.calculate_temperature(&context.emotional_state);

        let config = LlmRequestConfig::new()
            .with_max_tokens(self.config.max_tokens)
            .with_temperature(temperature);

        match self.backend.complete(&prompt, &config).await {
            Ok(response) => {
                if let Some(parsed) = self.parse_response(&response.content) {
                    self.action_history.push(parsed.clone());
                    Some(self.to_conscious_action(&parsed))
                } else {
                    None
                }
            }
            Err(e) => {
                eprintln!("Cognition error: {}", e);
                None
            }
        }
    }
}

// ============================================================================
// STIMULUS PROCESSOR IMPLEMENTATION
// ============================================================================

/// A StimulusProcessor that uses the CognitionEngine
pub struct CognitiveProcessor {
    engine: CognitionEngine,
    neuro_state: NeuromodulatorState,
    memories: Vec<MemorySnippet>,
    goals: Vec<GoalSnippet>,
    boredom_level: f64,
}

impl CognitiveProcessor {
    pub fn new(engine: CognitionEngine) -> Self {
        Self {
            engine,
            neuro_state: NeuromodulatorState::default(),
            memories: Vec::new(),
            goals: Vec::new(),
            boredom_level: 0.0,
        }
    }

    pub fn set_neuro_state(&mut self, state: NeuromodulatorState) {
        self.neuro_state = state;
    }

    pub fn set_boredom(&mut self, level: f64) {
        self.boredom_level = level;
    }

    pub fn add_memory(&mut self, content: String, relevance: f64, valence: f64) {
        self.memories.push(MemorySnippet {
            content,
            relevance,
            timestamp: Utc::now(),
            valence,
        });
    }

    pub fn add_goal(&mut self, description: String, priority: f64) {
        self.goals.push(GoalSnippet {
            description,
            priority,
            progress: 0.0,
            deadline: None,
        });
    }

    fn build_context(&self, stimulus: Option<&Stimulus>, ctx: &ProcessingContext) -> CognitiveContext {
        let mut emotional = EmotionalSnapshot::from(&self.neuro_state);
        emotional.boredom = self.boredom_level;

        CognitiveContext {
            stimulus: stimulus.map(ProcessedStimulus::from),
            memories: self.memories.clone(),
            goals: self.goals.clone(),
            emotional_state: emotional,
            recent_actions: ctx.recent_actions.iter().map(|a| format!("{:?}", a)).collect(),
            current_focus: ctx.focus.as_ref().map(|f| f.target.clone()),
            timestamp: ctx.timestamp,
            cycle: ctx.cycle,
        }
    }
}

impl StimulusProcessor for CognitiveProcessor {
    fn process(
        &mut self,
        stimulus: &Stimulus,
        context: &ProcessingContext,
    ) -> Option<ConsciousAction> {
        let cog_context = self.build_context(Some(stimulus), context);
        
        // Note: In real use, this would be async. For the sync trait,
        // we'd need to block or use a different approach.
        // For now, return a simple action based on stimulus type.
        match &stimulus.kind {
            StimulusKind::ExternalPrompt { content, .. } => {
                Some(ConsciousAction::Respond {
                    content: format!("Received: {}", content),
                    to: Some(stimulus.id),
                })
            }
            StimulusKind::Drive(drive) => {
                match drive {
                    crate::core::stimulus::DriveEvent::Curiosity { domain, .. } => {
                        Some(ConsciousAction::Observe {
                            target: domain.clone(),
                        })
                    }
                    crate::core::stimulus::DriveEvent::Boredom { recommendation, .. } => {
                        Some(ConsciousAction::Refocus {
                            target: "something_new".to_string(),
                            reason: recommendation.clone(),
                        })
                    }
                    _ => None,
                }
            }
            StimulusKind::FileSystem(_) => {
                Some(ConsciousAction::Think {
                    thought: "Noticed file change...".to_string(),
                })
            }
            _ => None,
        }
    }

    fn mind_wander(&mut self, context: &ProcessingContext) -> Option<ConsciousAction> {
        // Simple mind-wandering without LLM
        let thoughts = [
            "I wonder what else I could explore...",
            "Maybe I should check on my goals...",
            "What was I thinking about earlier?",
            "Is there something I'm forgetting?",
        ];
        
        let idx = (context.cycle as usize) % thoughts.len();
        Some(ConsciousAction::Think {
            thought: thoughts[idx].to_string(),
        })
    }

    fn execute(&mut self, action: &ConsciousAction) -> ActionResult {
        // Basic execution - in real implementation, this would do actual work
        ActionResult {
            success: true,
            output: match action {
                ConsciousAction::Respond { content, .. } => Some(content.clone()),
                ConsciousAction::Think { thought } => Some(thought.clone()),
                ConsciousAction::Observe { target } => Some(format!("Observing: {}", target)),
                _ => None,
            },
            error: None,
            follow_up: None,
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::llm::MockLlmBackend;

    #[test]
    fn test_parse_response() {
        let engine = CognitionEngine::new(
            CognitionConfig::default(),
            Box::new(MockLlmBackend::new()),
        );

        let response = "ACTION: RESPOND\nCONTENT: Hello there!\nREASONING: Greeting the user";
        let parsed = engine.parse_response(response).unwrap();

        assert_eq!(parsed.action_type, "RESPOND");
        assert_eq!(parsed.content, "Hello there!");
        assert_eq!(parsed.reasoning, Some("Greeting the user".to_string()));
    }

    #[test]
    fn test_to_conscious_action() {
        let engine = CognitionEngine::new(
            CognitionConfig::default(),
            Box::new(MockLlmBackend::new()),
        );

        let parsed = ParsedAction {
            action_type: "THINK".to_string(),
            content: "Interesting...".to_string(),
            reasoning: None,
        };

        let action = engine.to_conscious_action(&parsed);
        assert!(matches!(action, ConsciousAction::Think { .. }));
    }

    #[test]
    fn test_temperature_calculation() {
        let engine = CognitionEngine::new(
            CognitionConfig::default(),
            Box::new(MockLlmBackend::new()),
        );

        // High exploration should increase temperature
        let high_explore = EmotionalSnapshot {
            valence: 0.0,
            arousal: 0.5,
            curiosity: 0.8,
            boredom: 0.0,
            stress: 0.0,
            motivation: 0.5,
            exploration_drive: 0.9,
        };
        let temp = engine.calculate_temperature(&high_explore);
        assert!(temp > engine.config.base_temperature);

        // High stress should decrease temperature
        let high_stress = EmotionalSnapshot {
            valence: 0.0,
            arousal: 0.5,
            curiosity: 0.3,
            boredom: 0.0,
            stress: 0.9,
            motivation: 0.5,
            exploration_drive: 0.1,
        };
        let temp = engine.calculate_temperature(&high_stress);
        assert!(temp < engine.config.base_temperature);
    }

    #[test]
    fn test_build_prompt() {
        let engine = CognitionEngine::new(
            CognitionConfig::default(),
            Box::new(MockLlmBackend::new()),
        );

        let context = CognitiveContext {
            stimulus: Some(ProcessedStimulus {
                kind: "external_prompt".to_string(),
                content: "Hello!".to_string(),
                salience: 0.8,
                source: "human".to_string(),
                requires_response: true,
            }),
            memories: vec![],
            goals: vec![GoalSnippet {
                description: "Learn Rust".to_string(),
                priority: 0.8,
                progress: 0.3,
                deadline: None,
            }],
            emotional_state: EmotionalSnapshot {
                valence: 0.2,
                arousal: 0.5,
                curiosity: 0.7,
                boredom: 0.1,
                stress: 0.2,
                motivation: 0.6,
                exploration_drive: 0.5,
            },
            recent_actions: vec![],
            current_focus: None,
            timestamp: Utc::now(),
            cycle: 42,
        };

        let prompt = engine.build_prompt(&context);
        assert!(prompt.contains("Hello!"));
        assert!(prompt.contains("Learn Rust"));
        assert!(prompt.contains("requires a response"));
    }
}
