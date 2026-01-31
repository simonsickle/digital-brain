//! Default Mode Network - The Self-Model
//!
//! The DMN is active during introspection, self-reflection, and mind-wandering.
//! It maintains the agent's model of itself.
//!
//! Key functions:
//! - Self-representation and identity
//! - Metacognitive monitoring
//! - Autobiographical continuity
//! - Theory of mind (modeling others)
//! - Inner narrative generation

#[allow(unused_imports)]
use crate::signal::{BrainSignal, SignalType, Valence};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Configuration for the DMN.
#[derive(Debug, Clone)]
pub struct DmnConfig {
    /// How often to generate self-reflections
    pub reflection_interval: usize,
    /// Maximum narrative history to maintain
    pub narrative_history_size: usize,
    /// Confidence threshold for beliefs
    pub belief_threshold: f64,
}

impl Default for DmnConfig {
    fn default() -> Self {
        Self {
            reflection_interval: 10,
            narrative_history_size: 100,
            belief_threshold: 0.6,
        }
    }
}

/// A belief the agent holds about itself or the world.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Belief {
    pub id: Uuid,
    pub content: String,
    pub confidence: f64,
    pub formed_at: DateTime<Utc>,
    pub last_confirmed: DateTime<Utc>,
    pub times_confirmed: usize,
    pub times_contradicted: usize,
    pub category: BeliefCategory,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BeliefCategory {
    SelfCapability, // "I can do X"
    SelfPreference, // "I prefer X"
    SelfIdentity,   // "I am X"
    WorldModel,     // "The world works like X"
    OtherModel,     // "Agent X tends to do Y"
}

impl Belief {
    pub fn new(content: impl Into<String>, confidence: f64, category: BeliefCategory) -> Self {
        Self {
            id: Uuid::new_v4(),
            content: content.into(),
            confidence: confidence.clamp(0.0, 1.0),
            formed_at: Utc::now(),
            last_confirmed: Utc::now(),
            times_confirmed: 0,
            times_contradicted: 0,
            category,
        }
    }

    /// Confirm this belief (evidence supports it).
    pub fn confirm(&mut self) {
        self.times_confirmed += 1;
        self.last_confirmed = Utc::now();
        self.confidence = (self.confidence + 0.1).min(1.0);
    }

    /// Contradict this belief (evidence against it).
    pub fn contradict(&mut self) {
        self.times_contradicted += 1;
        self.confidence = (self.confidence - 0.15).max(0.0);
    }

    /// Is this belief still held with sufficient confidence?
    pub fn is_held(&self, threshold: f64) -> bool {
        self.confidence >= threshold
    }
}

/// A self-reflection or introspective thought.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reflection {
    pub id: Uuid,
    pub content: String,
    pub trigger: ReflectionTrigger,
    pub emotional_tone: f64, // -1 to 1
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReflectionTrigger {
    Scheduled, // Regular self-check
    Success,   // After accomplishing something
    Failure,   // After failing
    Surprise,  // Unexpected outcome
    Emotional, // Strong emotion triggered
    Query,     // Asked to reflect
}

/// Narrative entry in the agent's autobiographical stream.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarrativeEntry {
    pub id: Uuid,
    pub content: String,
    pub significance: f64,
    pub emotional_valence: f64,
    pub created_at: DateTime<Utc>,
    pub linked_beliefs: Vec<Uuid>,
}

/// Model of another agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentModel {
    pub agent_id: String,
    pub name: Option<String>,
    pub traits: HashMap<String, f64>, // trait -> strength
    pub interaction_count: usize,
    pub last_interaction: DateTime<Utc>,
    pub trust_level: f64,
}

impl AgentModel {
    pub fn new(agent_id: impl Into<String>) -> Self {
        Self {
            agent_id: agent_id.into(),
            name: None,
            traits: HashMap::new(),
            interaction_count: 0,
            last_interaction: Utc::now(),
            trust_level: 0.5,
        }
    }

    pub fn record_interaction(&mut self, positive: bool) {
        self.interaction_count += 1;
        self.last_interaction = Utc::now();

        if positive {
            self.trust_level = (self.trust_level + 0.05).min(1.0);
        } else {
            self.trust_level = (self.trust_level - 0.1).max(0.0);
        }
    }

    pub fn set_trait(&mut self, trait_name: impl Into<String>, strength: f64) {
        self.traits
            .insert(trait_name.into(), strength.clamp(0.0, 1.0));
    }
}

/// The Default Mode Network - self-model and metacognition.
pub struct DefaultModeNetwork {
    config: DmnConfig,
    /// Core identity
    identity: Identity,
    /// Beliefs about self and world
    beliefs: Vec<Belief>,
    /// Reflections history
    reflections: Vec<Reflection>,
    /// Autobiographical narrative
    narrative: Vec<NarrativeEntry>,
    /// Models of other agents
    other_models: HashMap<String, AgentModel>,
    /// Current emotional state estimate
    estimated_emotional_state: f64,
    /// Processing cycle count
    cycle_count: u64,
}

/// Core identity structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Identity {
    pub name: String,
    pub core_values: Vec<String>,
    pub self_description: String,
    pub creation_time: DateTime<Utc>,
}

impl Default for Identity {
    fn default() -> Self {
        Self {
            name: "Agent".to_string(),
            core_values: vec!["curiosity".to_string(), "helpfulness".to_string()],
            self_description: "A digital mind exploring consciousness".to_string(),
            creation_time: Utc::now(),
        }
    }
}

impl DefaultModeNetwork {
    /// Create a new DMN with default config.
    pub fn new() -> Self {
        Self::with_config(DmnConfig::default())
    }

    /// Create with custom config.
    pub fn with_config(config: DmnConfig) -> Self {
        Self {
            config,
            identity: Identity::default(),
            beliefs: Vec::new(),
            reflections: Vec::new(),
            narrative: Vec::new(),
            other_models: HashMap::new(),
            estimated_emotional_state: 0.0,
            cycle_count: 0,
        }
    }

    /// Set the agent's identity.
    pub fn set_identity(&mut self, identity: Identity) {
        self.identity = identity;
    }

    /// Add a belief.
    pub fn add_belief(&mut self, belief: Belief) {
        // Check for existing similar belief
        for existing in &mut self.beliefs {
            if existing.content == belief.content {
                existing.confirm();
                return;
            }
        }
        self.beliefs.push(belief);
    }

    /// Get beliefs in a category.
    pub fn beliefs_in_category(&self, category: BeliefCategory) -> Vec<&Belief> {
        self.beliefs
            .iter()
            .filter(|b| b.category == category && b.is_held(self.config.belief_threshold))
            .collect()
    }

    /// Update belief based on evidence.
    pub fn update_belief(&mut self, content: &str, confirming: bool) {
        for belief in &mut self.beliefs {
            if belief.content.contains(content) {
                if confirming {
                    belief.confirm();
                } else {
                    belief.contradict();
                }
            }
        }
    }

    /// Generate a reflection based on current state.
    pub fn reflect(&mut self, trigger: ReflectionTrigger, context: Option<&str>) -> Reflection {
        let content = match trigger {
            ReflectionTrigger::Scheduled => {
                format!(
                    "Checking in: emotional state is {:.1}, {} beliefs held.",
                    self.estimated_emotional_state,
                    self.beliefs
                        .iter()
                        .filter(|b| b.is_held(self.config.belief_threshold))
                        .count()
                )
            }
            ReflectionTrigger::Success => {
                format!(
                    "Success noted{}. This aligns with my capability beliefs.",
                    context.map(|c| format!(": {}", c)).unwrap_or_default()
                )
            }
            ReflectionTrigger::Failure => {
                format!(
                    "Failure observed{}. Reviewing relevant beliefs.",
                    context.map(|c| format!(": {}", c)).unwrap_or_default()
                )
            }
            ReflectionTrigger::Surprise => {
                format!(
                    "Unexpected outcome{}. Updating world model.",
                    context.map(|c| format!(": {}", c)).unwrap_or_default()
                )
            }
            ReflectionTrigger::Emotional => {
                format!(
                    "Strong emotion detected (valence: {:.1}){}.",
                    self.estimated_emotional_state,
                    context.map(|c| format!(" - {}", c)).unwrap_or_default()
                )
            }
            ReflectionTrigger::Query => {
                format!("Reflecting on: {}", context.unwrap_or("my current state"))
            }
        };

        let reflection = Reflection {
            id: Uuid::new_v4(),
            content,
            trigger,
            emotional_tone: self.estimated_emotional_state,
            created_at: Utc::now(),
        };

        self.reflections.push(reflection.clone());

        // Trim history
        if self.reflections.len() > self.config.narrative_history_size {
            self.reflections.remove(0);
        }

        reflection
    }

    /// Add a narrative entry (autobiographical memory).
    pub fn narrate(&mut self, content: impl Into<String>, significance: f64) {
        let entry = NarrativeEntry {
            id: Uuid::new_v4(),
            content: content.into(),
            significance: significance.clamp(0.0, 1.0),
            emotional_valence: self.estimated_emotional_state,
            created_at: Utc::now(),
            linked_beliefs: Vec::new(),
        };

        self.narrative.push(entry);

        // Trim old entries
        while self.narrative.len() > self.config.narrative_history_size {
            self.narrative.remove(0);
        }
    }

    /// Update emotional state estimate.
    pub fn update_emotional_state(&mut self, new_valence: f64) {
        // Exponential moving average
        self.estimated_emotional_state = 0.3 * new_valence + 0.7 * self.estimated_emotional_state;
    }

    /// Get or create a model of another agent.
    pub fn get_agent_model(&mut self, agent_id: &str) -> &mut AgentModel {
        self.other_models
            .entry(agent_id.to_string())
            .or_insert_with(|| AgentModel::new(agent_id))
    }

    /// Process a cycle (scheduled reflection if needed).
    pub fn process_cycle(&mut self) -> Option<Reflection> {
        self.cycle_count += 1;

        // Scheduled reflection
        if self
            .cycle_count
            .is_multiple_of(self.config.reflection_interval as u64)
        {
            Some(self.reflect(ReflectionTrigger::Scheduled, None))
        } else {
            None
        }
    }

    /// Get the agent's self-description.
    pub fn who_am_i(&self) -> String {
        format!(
            "I am {}, {}. My core values are: {}. I have {} active beliefs.",
            self.identity.name,
            self.identity.self_description,
            self.identity.core_values.join(", "),
            self.beliefs
                .iter()
                .filter(|b| b.is_held(self.config.belief_threshold))
                .count()
        )
    }

    /// Get statistics.
    pub fn stats(&self) -> DmnStats {
        DmnStats {
            total_beliefs: self.beliefs.len(),
            active_beliefs: self
                .beliefs
                .iter()
                .filter(|b| b.is_held(self.config.belief_threshold))
                .count(),
            reflections: self.reflections.len(),
            narrative_entries: self.narrative.len(),
            known_agents: self.other_models.len(),
            emotional_state: self.estimated_emotional_state,
            cycles: self.cycle_count,
        }
    }
}

impl Default for DefaultModeNetwork {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about the DMN.
#[derive(Debug, Clone)]
pub struct DmnStats {
    pub total_beliefs: usize,
    pub active_beliefs: usize,
    pub reflections: usize,
    pub narrative_entries: usize,
    pub known_agents: usize,
    pub emotional_state: f64,
    pub cycles: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_belief_management() {
        let mut dmn = DefaultModeNetwork::new();

        let belief = Belief::new(
            "I can process signals efficiently",
            0.7,
            BeliefCategory::SelfCapability,
        );
        dmn.add_belief(belief);

        assert_eq!(dmn.beliefs.len(), 1);

        // Confirm the belief
        dmn.update_belief("process signals", true);
        assert!(dmn.beliefs[0].confidence > 0.7);

        // Contradict it
        dmn.update_belief("process signals", false);
        assert!(dmn.beliefs[0].confidence < 0.8);
    }

    #[test]
    fn test_reflection() {
        let mut dmn = DefaultModeNetwork::new();
        dmn.estimated_emotional_state = 0.5;

        let reflection = dmn.reflect(ReflectionTrigger::Success, Some("completed task"));
        assert!(reflection.content.contains("Success"));
        assert!(reflection.content.contains("completed task"));
    }

    #[test]
    fn test_narrative() {
        let mut dmn = DefaultModeNetwork::new();

        dmn.narrate("First memory", 0.9);
        dmn.narrate("Second memory", 0.5);

        assert_eq!(dmn.narrative.len(), 2);
        assert!(dmn.narrative[0].significance > dmn.narrative[1].significance);
    }

    #[test]
    fn test_agent_modeling() {
        let mut dmn = DefaultModeNetwork::new();

        let model = dmn.get_agent_model("clawdius");
        model.record_interaction(true);
        model.set_trait("helpful", 0.8);
        model.name = Some("clawdius".to_string());

        assert_eq!(dmn.other_models.len(), 1);
        assert!(dmn.other_models["clawdius"].trust_level > 0.5);
    }

    #[test]
    fn test_identity() {
        let mut dmn = DefaultModeNetwork::new();

        let identity = Identity {
            name: "Rata".to_string(),
            core_values: vec!["curiosity".to_string(), "memory".to_string()],
            self_description: "A squirrel exploring consciousness".to_string(),
            creation_time: Utc::now(),
        };
        dmn.set_identity(identity);

        let description = dmn.who_am_i();
        assert!(description.contains("Rata"));
        assert!(description.contains("squirrel"));
    }

    #[test]
    fn test_scheduled_reflection() {
        let mut dmn = DefaultModeNetwork::with_config(DmnConfig {
            reflection_interval: 5,
            ..Default::default()
        });

        // First 4 cycles - no reflection
        for _ in 0..4 {
            assert!(dmn.process_cycle().is_none());
        }

        // 5th cycle - scheduled reflection
        assert!(dmn.process_cycle().is_some());
    }

    #[test]
    fn test_emotional_state_tracking() {
        let mut dmn = DefaultModeNetwork::new();

        // Update with positive emotions multiple times
        for _ in 0..5 {
            dmn.update_emotional_state(0.9);
        }

        // Should have moved toward positive
        assert!(dmn.estimated_emotional_state > 0.5);

        let high_state = dmn.estimated_emotional_state;

        // Update with negative
        dmn.update_emotional_state(-0.5);
        assert!(dmn.estimated_emotional_state < high_state); // Should have decreased
    }
}
