//! Mirror Neuron System - Embodied Simulation & Action Understanding
//!
//! The mirror neuron system fires both when performing an action and when
//! observing someone else perform that action. This enables:
//!
//! - **Action Understanding**: Inferring goals from observed behavior
//! - **Embodied Simulation**: Internally simulating others' actions
//! - **Empathic Resonance**: Feeling what others feel through motor mirroring
//! - **Imitation Learning**: Acquiring new behaviors by observation
//! - **Intention Prediction**: Predicting what someone will do next
//!
//! # Theoretical Foundation
//!
//! Based on:
//! - Rizzolatti & Craighero (2004): Mirror neurons in premotor cortex
//! - Gallese (2001): Embodied simulation theory
//! - Iacoboni (2009): Imitation, empathy, and mirror neurons
//! - Kilner et al. (2007): Predictive coding in the mirror system
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                   MIRROR NEURON SYSTEM                       │
//! ├──────────────┬──────────────┬──────────────┬────────────────┤
//! │   Action     │   Embodied   │   Empathic   │   Imitation    │
//! │ Recognition  │  Simulation  │  Resonance   │   Learning     │
//! │ (what/why)   │ (internal)   │ (feeling)    │  (acquiring)   │
//! └──────────────┴──────────────┴──────────────┴────────────────┘
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use uuid::Uuid;

/// Configuration for the mirror neuron system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MirrorSystemConfig {
    /// Resonance strength: how strongly observed actions activate motor representations
    pub resonance_strength: f64,
    /// Empathic gain: how much emotional mirroring occurs
    pub empathic_gain: f64,
    /// Maximum action patterns to store in repertoire
    pub max_repertoire_size: usize,
    /// Minimum confidence to attempt imitation learning
    pub imitation_threshold: f64,
    /// How many observations before an action pattern is learned
    pub learning_observations: u32,
    /// Decay rate for action familiarity
    pub familiarity_decay: f64,
}

impl Default for MirrorSystemConfig {
    fn default() -> Self {
        Self {
            resonance_strength: 0.7,
            empathic_gain: 0.6,
            max_repertoire_size: 50,
            imitation_threshold: 0.5,
            learning_observations: 3,
            familiarity_decay: 0.01,
        }
    }
}

/// An action pattern in the motor repertoire.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionPattern {
    pub id: Uuid,
    /// Name/label of the action
    pub name: String,
    /// Sequence of sub-actions (motor primitives)
    pub sequence: Vec<String>,
    /// Typical goal associated with this action
    pub typical_goal: String,
    /// How familiar this action is (0-1, affects recognition speed)
    pub familiarity: f64,
    /// How many times this action has been performed
    pub execution_count: u32,
    /// How many times this action has been observed in others
    pub observation_count: u32,
    /// Emotional valence typically associated with this action
    pub emotional_valence: f64,
    /// Contexts where this action commonly occurs
    pub contexts: Vec<String>,
}

impl ActionPattern {
    pub fn new(name: impl Into<String>, sequence: Vec<String>, goal: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            sequence,
            typical_goal: goal.into(),
            familiarity: 0.3,
            execution_count: 0,
            observation_count: 0,
            emotional_valence: 0.0,
            contexts: Vec::new(),
        }
    }

    /// Record that we performed this action ourselves.
    pub fn record_execution(&mut self) {
        self.execution_count += 1;
        self.familiarity = (self.familiarity + 0.1).min(1.0);
    }

    /// Record that we observed someone else perform this action.
    pub fn record_observation(&mut self) {
        self.observation_count += 1;
        self.familiarity = (self.familiarity + 0.05).min(1.0);
    }

    /// Total experience (execution + observation).
    pub fn total_experience(&self) -> u32 {
        self.execution_count + self.observation_count
    }
}

/// Result of observing an action and trying to understand it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionUnderstanding {
    /// Recognized action name (if matched)
    pub recognized_action: Option<String>,
    /// Inferred goal/intention
    pub inferred_goal: String,
    /// Confidence in the recognition (0-1)
    pub confidence: f64,
    /// Predicted next action in the sequence
    pub predicted_next: Option<String>,
    /// Empathic resonance: estimated emotional state of the actor
    pub empathic_resonance: EmpathicResonance,
    /// Motor activation strength (how much our own motor system activated)
    pub motor_activation: f64,
}

/// Empathic mirroring of another's emotional state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmpathicResonance {
    /// Estimated valence of the other (-1 to 1)
    pub mirrored_valence: f64,
    /// Estimated arousal of the other (0 to 1)
    pub mirrored_arousal: f64,
    /// How confident we are in this reading (0-1)
    pub confidence: f64,
    /// The felt empathic response in ourselves
    pub self_response_valence: f64,
}

/// Result of attempting to learn through imitation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImitationResult {
    /// Was the action successfully learned?
    pub learned: bool,
    /// The action pattern (new or updated)
    pub action_name: String,
    /// How many observations contributed
    pub observation_count: u32,
    /// Current proficiency estimate (0-1)
    pub proficiency: f64,
}

/// An observed action from another agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservedAction {
    pub agent_id: String,
    pub action_name: String,
    pub sub_actions: Vec<String>,
    pub apparent_goal: Option<String>,
    pub emotional_cues: f64,
    pub context: String,
    pub timestamp: DateTime<Utc>,
}

/// Statistics about the mirror system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MirrorSystemStats {
    pub repertoire_size: usize,
    pub total_observations: u64,
    pub total_recognitions: u64,
    pub total_imitations_learned: u64,
    pub avg_resonance: f64,
    pub most_familiar_action: Option<String>,
}

/// The mirror neuron system.
pub struct MirrorSystem {
    config: MirrorSystemConfig,
    /// Known action patterns (motor repertoire)
    repertoire: HashMap<String, ActionPattern>,
    /// Recent observations for pattern matching
    observation_buffer: VecDeque<ObservedAction>,
    /// Observation counts for imitation learning candidates
    imitation_candidates: HashMap<String, Vec<ObservedAction>>,
    /// Statistics
    total_observations: u64,
    total_recognitions: u64,
    total_imitations_learned: u64,
    resonance_history: VecDeque<f64>,
}

impl MirrorSystem {
    /// Create a new mirror neuron system with default configuration.
    pub fn new() -> Self {
        Self::with_config(MirrorSystemConfig::default())
    }

    /// Create with custom configuration.
    pub fn with_config(config: MirrorSystemConfig) -> Self {
        Self {
            config,
            repertoire: HashMap::new(),
            observation_buffer: VecDeque::with_capacity(50),
            imitation_candidates: HashMap::new(),
            total_observations: 0,
            total_recognitions: 0,
            total_imitations_learned: 0,
            resonance_history: VecDeque::with_capacity(100),
        }
    }

    /// Add an action pattern to the motor repertoire.
    pub fn add_to_repertoire(&mut self, pattern: ActionPattern) {
        if self.repertoire.len() >= self.config.max_repertoire_size {
            // Remove least familiar action
            if let Some(least_familiar) = self
                .repertoire
                .iter()
                .min_by(|a, b| {
                    a.1.familiarity
                        .partial_cmp(&b.1.familiarity)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(k, _)| k.clone())
            {
                self.repertoire.remove(&least_familiar);
            }
        }
        self.repertoire.insert(pattern.name.clone(), pattern);
    }

    /// Record that we performed an action (strengthens motor representation).
    pub fn record_own_action(&mut self, action_name: &str) {
        if let Some(pattern) = self.repertoire.get_mut(action_name) {
            pattern.record_execution();
        }
    }

    /// Observe another agent's action and try to understand it.
    pub fn observe_action(&mut self, observation: ObservedAction) -> ActionUnderstanding {
        self.total_observations += 1;

        // Store observation
        self.observation_buffer.push_back(observation.clone());
        if self.observation_buffer.len() > 50 {
            self.observation_buffer.pop_front();
        }

        // Try to match against known repertoire
        let (recognized, confidence, predicted_next) = self.match_action(&observation);

        if recognized.is_some() {
            self.total_recognitions += 1;

            // Update observation count on the matched pattern
            if let Some(name) = &recognized
                && let Some(pattern) = self.repertoire.get_mut(name)
            {
                pattern.record_observation();
            }
        }

        // Compute empathic resonance
        let empathic_resonance = self.compute_empathy(&observation, confidence);
        let motor_activation = confidence * self.config.resonance_strength;

        self.resonance_history.push_back(motor_activation);
        if self.resonance_history.len() > 100 {
            self.resonance_history.pop_front();
        }

        // Track for imitation learning
        self.track_imitation_candidate(&observation);

        let inferred_goal = if let Some(ref name) = recognized {
            self.repertoire
                .get(name)
                .map(|p| p.typical_goal.clone())
                .unwrap_or_else(|| {
                    observation
                        .apparent_goal
                        .clone()
                        .unwrap_or_else(|| "unknown goal".to_string())
                })
        } else {
            observation
                .apparent_goal
                .clone()
                .unwrap_or_else(|| "unknown goal".to_string())
        };

        ActionUnderstanding {
            recognized_action: recognized,
            inferred_goal,
            confidence,
            predicted_next,
            empathic_resonance,
            motor_activation,
        }
    }

    /// Match an observation against known action patterns.
    fn match_action(&self, observation: &ObservedAction) -> (Option<String>, f64, Option<String>) {
        let mut best_match: Option<(&str, f64)> = None;

        for (name, pattern) in &self.repertoire {
            let mut score = 0.0;

            // Name match
            if pattern.name == observation.action_name {
                score += 0.5;
            }

            // Sub-action sequence overlap
            if !observation.sub_actions.is_empty() && !pattern.sequence.is_empty() {
                let overlap = observation
                    .sub_actions
                    .iter()
                    .filter(|a| pattern.sequence.contains(a))
                    .count() as f64;
                let max_len = observation.sub_actions.len().max(pattern.sequence.len()) as f64;
                score += 0.3 * (overlap / max_len);
            }

            // Context match
            if pattern.contexts.contains(&observation.context) {
                score += 0.2;
            }

            // Familiarity boost (familiar actions are recognized faster)
            score *= 0.7 + pattern.familiarity * 0.3;

            if let Some((_, best_score)) = best_match {
                if score > best_score {
                    best_match = Some((name, score));
                }
            } else if score > 0.2 {
                best_match = Some((name, score));
            }
        }

        if let Some((name, confidence)) = best_match {
            // Predict next action in sequence
            let predicted = self.predict_next_in_sequence(name, &observation.sub_actions);
            (Some(name.to_string()), confidence.min(1.0), predicted)
        } else {
            (None, 0.0, None)
        }
    }

    /// Predict the next sub-action in a known sequence.
    fn predict_next_in_sequence(
        &self,
        action_name: &str,
        current_sub_actions: &[String],
    ) -> Option<String> {
        let pattern = self.repertoire.get(action_name)?;

        if current_sub_actions.is_empty() || pattern.sequence.is_empty() {
            return pattern.sequence.first().cloned();
        }

        // Find where we are in the sequence
        let last_observed = current_sub_actions.last()?;
        let position = pattern.sequence.iter().position(|s| s == last_observed)?;

        pattern.sequence.get(position + 1).cloned()
    }

    /// Compute empathic resonance from an observed action.
    fn compute_empathy(
        &self,
        observation: &ObservedAction,
        recognition_confidence: f64,
    ) -> EmpathicResonance {
        let gain = self.config.empathic_gain;

        // Base emotional mirroring from observed cues
        let mirrored_valence = observation.emotional_cues * gain;

        // Arousal from observation intensity
        let mirrored_arousal =
            (recognition_confidence * 0.5 + observation.emotional_cues.abs() * 0.5) * gain;

        // Our own empathic response (attenuated version of mirrored state)
        let self_response = mirrored_valence * 0.7;

        EmpathicResonance {
            mirrored_valence: mirrored_valence.clamp(-1.0, 1.0),
            mirrored_arousal: mirrored_arousal.clamp(0.0, 1.0),
            confidence: recognition_confidence * gain,
            self_response_valence: self_response.clamp(-1.0, 1.0),
        }
    }

    /// Track an observation as a candidate for imitation learning.
    fn track_imitation_candidate(&mut self, observation: &ObservedAction) {
        // Only track actions we don't already know well
        let already_known = self
            .repertoire
            .get(&observation.action_name)
            .map(|p| p.familiarity > 0.7)
            .unwrap_or(false);

        if already_known {
            return;
        }

        let candidates = self
            .imitation_candidates
            .entry(observation.action_name.clone())
            .or_default();
        candidates.push(observation.clone());
    }

    /// Attempt to learn a new action through imitation.
    pub fn try_imitation_learning(&mut self, action_name: &str) -> Option<ImitationResult> {
        let observations = self.imitation_candidates.get(action_name)?;
        let obs_count = observations.len() as u32;

        if obs_count < self.config.learning_observations {
            return Some(ImitationResult {
                learned: false,
                action_name: action_name.to_string(),
                observation_count: obs_count,
                proficiency: 0.0,
            });
        }

        // Synthesize a pattern from observations (collect all data before mutating)
        let sequence = self.synthesize_sequence(observations);
        let goal = observations
            .iter()
            .filter_map(|o| o.apparent_goal.as_deref())
            .next()
            .unwrap_or("observed goal")
            .to_string();
        let avg_valence =
            observations.iter().map(|o| o.emotional_cues).sum::<f64>() / observations.len() as f64;
        let mut contexts: Vec<String> = observations.iter().map(|o| o.context.clone()).collect();
        contexts.dedup();

        // Now we're done reading from observations, so we can mutate self
        let mut pattern = ActionPattern::new(action_name, sequence, goal);
        pattern.observation_count = obs_count;
        pattern.emotional_valence = avg_valence;
        pattern.familiarity = 0.3;
        pattern.contexts = contexts;

        let proficiency = pattern.familiarity;
        self.add_to_repertoire(pattern);
        self.imitation_candidates.remove(action_name);
        self.total_imitations_learned += 1;

        Some(ImitationResult {
            learned: true,
            action_name: action_name.to_string(),
            observation_count: obs_count,
            proficiency,
        })
    }

    /// Synthesize a representative action sequence from multiple observations.
    fn synthesize_sequence(&self, observations: &[ObservedAction]) -> Vec<String> {
        if observations.is_empty() {
            return Vec::new();
        }

        // Use the most common sub-actions in their most common order
        let mut all_actions: Vec<&str> = Vec::new();
        for obs in observations {
            for action in &obs.sub_actions {
                all_actions.push(action);
            }
        }

        // Count frequencies
        let mut freq: HashMap<&str, usize> = HashMap::new();
        for action in &all_actions {
            *freq.entry(action).or_insert(0) += 1;
        }

        // Take the longest observation as the template, filtering to common actions
        let threshold = observations.len() / 2;
        observations
            .iter()
            .max_by_key(|o| o.sub_actions.len())
            .map(|o| {
                o.sub_actions
                    .iter()
                    .filter(|a| freq.get(a.as_str()).copied().unwrap_or(0) > threshold)
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get the repertoire size.
    pub fn repertoire_size(&self) -> usize {
        self.repertoire.len()
    }

    /// Get a specific action pattern from the repertoire.
    pub fn get_action(&self, name: &str) -> Option<&ActionPattern> {
        self.repertoire.get(name)
    }

    /// Apply familiarity decay to all actions (call periodically).
    pub fn decay_familiarity(&mut self) {
        let decay = self.config.familiarity_decay;
        for pattern in self.repertoire.values_mut() {
            pattern.familiarity = (pattern.familiarity - decay).max(0.0);
        }
    }

    /// Get statistics.
    pub fn stats(&self) -> MirrorSystemStats {
        let avg_resonance = if self.resonance_history.is_empty() {
            0.0
        } else {
            self.resonance_history.iter().sum::<f64>() / self.resonance_history.len() as f64
        };

        let most_familiar = self
            .repertoire
            .values()
            .max_by(|a, b| {
                a.familiarity
                    .partial_cmp(&b.familiarity)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|p| p.name.clone());

        MirrorSystemStats {
            repertoire_size: self.repertoire.len(),
            total_observations: self.total_observations,
            total_recognitions: self.total_recognitions,
            total_imitations_learned: self.total_imitations_learned,
            avg_resonance,
            most_familiar_action: most_familiar,
        }
    }
}

impl Default for MirrorSystem {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_observation(name: &str, goal: &str, valence: f64) -> ObservedAction {
        ObservedAction {
            agent_id: "other".to_string(),
            action_name: name.to_string(),
            sub_actions: vec!["reach".to_string(), "grasp".to_string(), "lift".to_string()],
            apparent_goal: Some(goal.to_string()),
            emotional_cues: valence,
            context: "kitchen".to_string(),
            timestamp: Utc::now(),
        }
    }

    #[test]
    fn test_creation() {
        let ms = MirrorSystem::new();
        assert_eq!(ms.repertoire_size(), 0);
        let stats = ms.stats();
        assert_eq!(stats.total_observations, 0);
    }

    #[test]
    fn test_add_to_repertoire() {
        let mut ms = MirrorSystem::new();
        let pattern = ActionPattern::new(
            "wave",
            vec!["raise_hand".into(), "move_side_to_side".into()],
            "greeting",
        );
        ms.add_to_repertoire(pattern);
        assert_eq!(ms.repertoire_size(), 1);
        assert!(ms.get_action("wave").is_some());
    }

    #[test]
    fn test_action_recognition() {
        let mut ms = MirrorSystem::new();

        // Add a known action
        let mut pattern = ActionPattern::new(
            "pick_up",
            vec!["reach".into(), "grasp".into(), "lift".into()],
            "acquire object",
        );
        pattern.familiarity = 0.8;
        pattern.contexts.push("kitchen".to_string());
        ms.add_to_repertoire(pattern);

        // Observe someone doing the same action
        let obs = make_observation("pick_up", "get cup", 0.3);
        let understanding = ms.observe_action(obs);

        assert!(
            understanding.recognized_action.is_some(),
            "Should recognize known action"
        );
        assert!(
            understanding.confidence > 0.3,
            "Confidence should be reasonable: {}",
            understanding.confidence
        );
        assert!(
            understanding.motor_activation > 0.0,
            "Motor system should activate"
        );
    }

    #[test]
    fn test_empathic_resonance() {
        let mut ms = MirrorSystem::new();

        // Observe a distressing action
        let obs = ObservedAction {
            agent_id: "other".to_string(),
            action_name: "cry".to_string(),
            sub_actions: vec![],
            apparent_goal: Some("express sadness".to_string()),
            emotional_cues: -0.8,
            context: "social".to_string(),
            timestamp: Utc::now(),
        };

        let understanding = ms.observe_action(obs);
        assert!(
            understanding.empathic_resonance.mirrored_valence < 0.0,
            "Should mirror negative emotion"
        );
        assert!(
            understanding.empathic_resonance.self_response_valence < 0.0,
            "Should feel empathic distress"
        );
    }

    #[test]
    fn test_imitation_learning() {
        let mut ms = MirrorSystem::with_config(MirrorSystemConfig {
            learning_observations: 3,
            ..Default::default()
        });

        // Observe the same action multiple times
        for i in 0..3 {
            let obs = ObservedAction {
                agent_id: "teacher".to_string(),
                action_name: "juggle".to_string(),
                sub_actions: vec!["toss".into(), "catch".into(), "toss".into()],
                apparent_goal: Some("entertainment".to_string()),
                emotional_cues: 0.5,
                context: format!("session_{}", i),
                timestamp: Utc::now(),
            };
            ms.observe_action(obs);
        }

        // Try to learn
        let result = ms.try_imitation_learning("juggle");
        assert!(result.is_some());
        let result = result.unwrap();
        assert!(result.learned, "Should learn after enough observations");
        assert_eq!(result.observation_count, 3);

        // Should now be in repertoire
        assert!(ms.get_action("juggle").is_some());
    }

    #[test]
    fn test_insufficient_observations() {
        let mut ms = MirrorSystem::with_config(MirrorSystemConfig {
            learning_observations: 5,
            ..Default::default()
        });

        // Only 2 observations
        for _ in 0..2 {
            let obs = make_observation("dance", "fun", 0.7);
            ms.observe_action(obs);
        }

        let result = ms.try_imitation_learning("dance");
        assert!(result.is_some());
        assert!(
            !result.unwrap().learned,
            "Should not learn with too few observations"
        );
    }

    #[test]
    fn test_next_action_prediction() {
        let mut ms = MirrorSystem::new();

        let pattern = ActionPattern::new(
            "make_tea",
            vec![
                "boil_water".into(),
                "add_teabag".into(),
                "pour_water".into(),
                "steep".into(),
            ],
            "prepare beverage",
        );
        ms.add_to_repertoire(pattern);

        // Observe someone at step 2
        let obs = ObservedAction {
            agent_id: "other".to_string(),
            action_name: "make_tea".to_string(),
            sub_actions: vec!["boil_water".into(), "add_teabag".into()],
            apparent_goal: None,
            emotional_cues: 0.1,
            context: "kitchen".to_string(),
            timestamp: Utc::now(),
        };

        let understanding = ms.observe_action(obs);
        assert_eq!(
            understanding.predicted_next.as_deref(),
            Some("pour_water"),
            "Should predict next step in sequence"
        );
    }

    #[test]
    fn test_own_action_strengthens_familiarity() {
        let mut ms = MirrorSystem::new();

        let pattern = ActionPattern::new("wave", vec!["raise".into()], "greet");
        let initial_familiarity = pattern.familiarity;
        ms.add_to_repertoire(pattern);

        ms.record_own_action("wave");

        let after = ms.get_action("wave").unwrap().familiarity;
        assert!(
            after > initial_familiarity,
            "Performing action should increase familiarity"
        );
    }

    #[test]
    fn test_repertoire_limit() {
        let mut ms = MirrorSystem::with_config(MirrorSystemConfig {
            max_repertoire_size: 3,
            ..Default::default()
        });

        for i in 0..5 {
            let mut pattern =
                ActionPattern::new(format!("action_{}", i), vec![], format!("goal_{}", i));
            pattern.familiarity = i as f64 * 0.2;
            ms.add_to_repertoire(pattern);
        }

        assert!(
            ms.repertoire_size() <= 3,
            "Should evict least familiar when at capacity"
        );
    }

    #[test]
    fn test_familiarity_decay() {
        let mut ms = MirrorSystem::with_config(MirrorSystemConfig {
            familiarity_decay: 0.1,
            ..Default::default()
        });

        let mut pattern = ActionPattern::new("wave", vec![], "greet");
        pattern.familiarity = 0.5;
        ms.add_to_repertoire(pattern);

        ms.decay_familiarity();

        let after = ms.get_action("wave").unwrap().familiarity;
        assert!(after < 0.5, "Familiarity should decay over time");
    }

    #[test]
    fn test_stats() {
        let mut ms = MirrorSystem::new();
        let pattern = ActionPattern::new("wave", vec![], "greet");
        ms.add_to_repertoire(pattern);

        let obs = make_observation("wave", "greeting", 0.5);
        ms.observe_action(obs);

        let stats = ms.stats();
        assert_eq!(stats.repertoire_size, 1);
        assert_eq!(stats.total_observations, 1);
    }
}
