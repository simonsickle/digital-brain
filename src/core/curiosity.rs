//! Curiosity/Exploration Drive
//!
//! Intrinsic motivation to explore and learn, independent of external rewards.
//! This module provides curiosity-driven exploration based on:
//! - Information gain (reducing uncertainty)
//! - Novelty detection (new patterns)
//! - Competence/mastery progress
//!
//! Integration with neuromodulators:
//! - High dopamine → exploit (seek known rewards)
//! - Low dopamine + high ACh → explore (seek information)
//! - Curiosity rewards are intrinsic (not tolerance-building)

use std::collections::HashMap;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::core::neuromodulators::NeuromodulatorState;
use crate::core::action::ActionTemplate;

/// Domain of knowledge/skill
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Domain(pub String);

impl Domain {
    pub fn new(name: &str) -> Self {
        Self(name.to_string())
    }
}

impl From<&str> for Domain {
    fn from(s: &str) -> Self {
        Domain::new(s)
    }
}

/// Record of an exploration event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplorationEvent {
    /// Unique identifier
    pub id: Uuid,
    /// Domain explored
    pub domain: Domain,
    /// What was discovered/learned
    pub discovery: String,
    /// Information gain (bits of uncertainty reduced)
    pub info_gain: f64,
    /// Novelty score (0-1)
    pub novelty: f64,
    /// Competence gain (0-1)
    pub competence_gain: f64,
    /// When it happened
    pub timestamp: DateTime<Utc>,
}

impl ExplorationEvent {
    pub fn new(domain: Domain, discovery: &str) -> Self {
        Self {
            id: Uuid::new_v4(),
            domain,
            discovery: discovery.to_string(),
            info_gain: 0.0,
            novelty: 0.0,
            competence_gain: 0.0,
            timestamp: Utc::now(),
        }
    }

    pub fn with_info_gain(mut self, gain: f64) -> Self {
        self.info_gain = gain.max(0.0);
        self
    }

    pub fn with_novelty(mut self, novelty: f64) -> Self {
        self.novelty = novelty.clamp(0.0, 1.0);
        self
    }

    pub fn with_competence_gain(mut self, gain: f64) -> Self {
        self.competence_gain = gain.clamp(0.0, 1.0);
        self
    }

    /// Total curiosity reward from this event
    pub fn curiosity_reward(&self) -> f64 {
        // Weight: info_gain (40%), novelty (30%), competence (30%)
        self.info_gain * 0.4 + self.novelty * 0.3 + self.competence_gain * 0.3
    }
}

/// Competence level in a domain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Competence {
    /// Current skill level (0-1)
    pub level: f64,
    /// Confidence in the level estimate
    pub confidence: f64,
    /// Number of practice attempts
    pub attempts: u64,
    /// Number of successes
    pub successes: u64,
    /// Last updated
    pub updated_at: DateTime<Utc>,
}

impl Default for Competence {
    fn default() -> Self {
        Self {
            level: 0.0,
            confidence: 0.0,
            attempts: 0,
            successes: 0,
            updated_at: Utc::now(),
        }
    }
}

impl Competence {
    /// Update competence based on outcome
    pub fn update(&mut self, success: bool, difficulty: f64) {
        self.attempts += 1;
        if success {
            self.successes += 1;
        }

        // Update level based on success rate and difficulty
        let success_rate = self.successes as f64 / self.attempts as f64;
        let weighted_success = if success {
            difficulty * 0.1
        } else {
            -difficulty * 0.05
        };

        self.level = (self.level + weighted_success).clamp(0.0, 1.0);
        self.confidence = (self.attempts as f64 / (self.attempts as f64 + 10.0)).min(0.95);
        self.updated_at = Utc::now();
    }

    /// Get learning opportunity (high when near edge of competence)
    pub fn learning_opportunity(&self, task_difficulty: f64) -> f64 {
        // Optimal learning when task is slightly above current level
        let difficulty_match = 1.0 - (task_difficulty - self.level - 0.1).abs();
        difficulty_match.max(0.0)
    }
}

/// Statistics about curiosity system
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CuriosityStats {
    pub total_explorations: u64,
    pub total_info_gain: f64,
    pub average_novelty: f64,
    pub domains_explored: usize,
    pub highest_competence_domain: Option<Domain>,
    pub most_uncertain_domain: Option<Domain>,
}

/// The curiosity system
#[derive(Debug)]
pub struct CuriositySystem {
    /// Uncertainty in each domain (higher = more to learn)
    uncertainty_map: HashMap<Domain, f64>,
    /// Competence in each domain
    competence_map: HashMap<Domain, Competence>,
    /// Recent exploration history
    exploration_history: Vec<ExplorationEvent>,
    /// Maximum history to keep
    max_history: usize,
    /// Threshold for "interesting" information gain
    info_gain_threshold: f64,
    /// Novelty decay rate (how quickly things become familiar)
    novelty_decay: f64,
    /// Pattern memory (for detecting novelty)
    seen_patterns: HashMap<String, u64>,
    /// Statistics
    stats: CuriosityStats,
}

impl Default for CuriositySystem {
    fn default() -> Self {
        Self::new()
    }
}

impl CuriositySystem {
    /// Create a new curiosity system
    pub fn new() -> Self {
        Self {
            uncertainty_map: HashMap::new(),
            competence_map: HashMap::new(),
            exploration_history: Vec::new(),
            max_history: 1000,
            info_gain_threshold: 0.1,
            novelty_decay: 0.95,
            seen_patterns: HashMap::new(),
            stats: CuriosityStats::default(),
        }
    }

    /// Register a domain with initial uncertainty
    pub fn register_domain(&mut self, domain: Domain, initial_uncertainty: f64) {
        self.uncertainty_map.insert(domain.clone(), initial_uncertainty.max(0.0));
        self.competence_map.entry(domain).or_default();
    }

    /// Get uncertainty in a domain
    pub fn uncertainty(&self, domain: &Domain) -> f64 {
        self.uncertainty_map.get(domain).copied().unwrap_or(1.0) // Unknown = high uncertainty
    }

    /// Get competence in a domain
    pub fn competence(&self, domain: &Domain) -> &Competence {
        static DEFAULT_COMPETENCE: Competence = Competence {
            level: 0.0,
            confidence: 0.0,
            attempts: 0,
            successes: 0,
            updated_at: DateTime::<Utc>::MIN_UTC,
        };
        self.competence_map.get(domain).unwrap_or(&DEFAULT_COMPETENCE)
    }

    /// Calculate novelty of a pattern
    pub fn novelty(&self, pattern: &str) -> f64 {
        let seen_count = self.seen_patterns.get(pattern).copied().unwrap_or(0);
        if seen_count == 0 {
            1.0 // Completely novel
        } else {
            // Exponential decay of novelty with exposure
            self.novelty_decay.powi(seen_count as i32)
        }
    }

    /// Record seeing a pattern (reduces future novelty)
    pub fn observe_pattern(&mut self, pattern: &str) {
        *self.seen_patterns.entry(pattern.to_string()).or_insert(0) += 1;
    }

    /// Estimate information gain from an action
    pub fn estimate_info_gain(&self, action: &ActionTemplate) -> f64 {
        // Extract domain from action tags or category
        let domain = Domain::new(&action.category.to_string());
        let uncertainty = self.uncertainty(&domain);
        
        // Higher uncertainty = more potential info gain
        // But also consider expected outcomes
        let outcome_entropy: f64 = action.expected_outcomes.iter()
            .map(|o| {
                let p = o.probability;
                if p > 0.0 && p < 1.0 {
                    -p * p.log2()
                } else {
                    0.0
                }
            })
            .sum();

        uncertainty * 0.5 + outcome_entropy * 0.5
    }

    /// Calculate curiosity value for a potential action
    pub fn curiosity_value(&self, action: &ActionTemplate, state: &NeuromodulatorState) -> f64 {
        let expected_info_gain = self.estimate_info_gain(action);
        let novelty = self.action_novelty(action);
        let competence_progress = self.competence_progress(action);

        // Base curiosity calculation
        let base_curiosity = expected_info_gain * 0.4 + novelty * 0.3 + competence_progress * 0.3;

        // Modulate by neuromodulator state
        // High ACh (learning depth) boosts curiosity
        let ach_boost = state.learning_depth * 0.3;
        
        // Low dopamine increases exploration drive
        let dopamine_boost = (1.0 - state.dopamine) * 0.2;
        
        // Exploration drive from neuromodulators
        let exploration_boost = state.exploration_drive * 0.2;

        base_curiosity + ach_boost + dopamine_boost + exploration_boost
    }

    /// Get novelty of an action
    fn action_novelty(&self, action: &ActionTemplate) -> f64 {
        // Use action name + description as pattern
        let pattern = format!("{}:{}", action.name, action.category.to_string());
        self.novelty(&pattern)
    }

    /// Estimate competence progress from an action
    fn competence_progress(&self, action: &ActionTemplate) -> f64 {
        let domain = Domain::new(&action.category.to_string());
        let competence = self.competence(&domain);
        
        // Estimate task difficulty from effort cost
        let difficulty = action.effort_cost;
        
        competence.learning_opportunity(difficulty)
    }

    /// Should we explore vs exploit?
    pub fn explore_vs_exploit(&self, state: &NeuromodulatorState) -> f64 {
        // Returns exploration probability (0 = exploit, 1 = explore)
        
        // High dopamine → exploit (seek known rewards)
        let exploit_bias = state.motivation * 0.5 + state.dopamine * 0.3;
        
        // Low dopamine + high ACh → explore (seek information)
        let explore_bias = state.learning_depth * (1.0 - state.dopamine) * 0.4
            + state.exploration_drive * 0.3
            + (1.0 - state.motivation) * 0.2;

        // Return exploration probability
        let total = explore_bias + exploit_bias;
        if total > 0.0 {
            explore_bias / total
        } else {
            0.5 // Default to balanced
        }
    }

    /// Record an exploration event
    pub fn record_exploration(&mut self, event: ExplorationEvent) {
        // Update uncertainty
        let uncertainty = self.uncertainty_map.entry(event.domain.clone()).or_insert(1.0);
        *uncertainty = (*uncertainty - event.info_gain * 0.1).max(0.0);

        // Update pattern novelty
        self.observe_pattern(&event.discovery);

        // Update stats
        self.stats.total_explorations += 1;
        self.stats.total_info_gain += event.info_gain;
        self.stats.average_novelty = self.stats.average_novelty * 0.9 + event.novelty * 0.1;
        self.stats.domains_explored = self.uncertainty_map.len();

        // Store in history
        self.exploration_history.push(event);
        while self.exploration_history.len() > self.max_history {
            self.exploration_history.remove(0);
        }

        // Update domain stats
        self.update_domain_stats();
    }

    /// Update competence after attempting something
    pub fn update_competence(&mut self, domain: &Domain, success: bool, difficulty: f64) {
        self.competence_map
            .entry(domain.clone())
            .or_default()
            .update(success, difficulty);
        
        self.update_domain_stats();
    }

    /// Find most uncertain domain (highest learning potential)
    pub fn most_uncertain_domain(&self) -> Option<&Domain> {
        self.uncertainty_map
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(d, _)| d)
    }

    /// Find domain with highest competence
    pub fn highest_competence_domain(&self) -> Option<&Domain> {
        self.competence_map
            .iter()
            .max_by(|a, b| {
                a.1.level
                    .partial_cmp(&b.1.level)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(d, _)| d)
    }

    /// Get domains sorted by learning opportunity
    pub fn domains_by_opportunity(&self) -> Vec<(&Domain, f64)> {
        let mut domains: Vec<_> = self.uncertainty_map
            .keys()
            .map(|d| {
                let uncertainty = self.uncertainty(d);
                let competence = self.competence(d);
                // Balance uncertainty with competence for optimal learning zone
                let opportunity = uncertainty * 0.6 + (1.0 - competence.level) * competence.confidence * 0.4;
                (d, opportunity)
            })
            .collect();

        domains.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        domains
    }

    /// Decay novelty over time (make familiar things less novel)
    pub fn decay_novelty(&mut self, factor: f64) {
        for count in self.seen_patterns.values_mut() {
            *count = ((*count as f64) * factor) as u64;
        }
        // Remove patterns seen 0 times (reset to novel)
        self.seen_patterns.retain(|_, v| *v > 0);
    }

    /// Get statistics
    pub fn stats(&self) -> &CuriosityStats {
        &self.stats
    }

    /// Get exploration history
    pub fn history(&self) -> &[ExplorationEvent] {
        &self.exploration_history
    }

    /// Update domain stats
    fn update_domain_stats(&mut self) {
        self.stats.highest_competence_domain = self.highest_competence_domain().cloned();
        self.stats.most_uncertain_domain = self.most_uncertain_domain().cloned();
    }

    /// Clear history but keep learned knowledge
    pub fn clear_history(&mut self) {
        self.exploration_history.clear();
    }

    /// Get all domains
    pub fn domains(&self) -> impl Iterator<Item = &Domain> {
        self.uncertainty_map.keys()
    }

    /// Set info gain threshold
    pub fn set_info_gain_threshold(&mut self, threshold: f64) {
        self.info_gain_threshold = threshold.max(0.0);
    }

    /// Set novelty decay rate
    pub fn set_novelty_decay(&mut self, decay: f64) {
        self.novelty_decay = decay.clamp(0.0, 1.0);
    }
}

// Implement display for ActionCategory (needed for curiosity)
impl std::fmt::Display for crate::core::action::ActionCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Exploration => write!(f, "exploration"),
            Self::Exploitation => write!(f, "exploitation"),
            Self::Communication => write!(f, "communication"),
            Self::Maintenance => write!(f, "maintenance"),
            Self::Rest => write!(f, "rest"),
            Self::Learning => write!(f, "learning"),
            Self::Social => write!(f, "social"),
            Self::Creative => write!(f, "creative"),
            Self::Defensive => write!(f, "defensive"),
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::action::{ActionCategory, ExpectedOutcome, Outcome};

    fn make_test_state() -> NeuromodulatorState {
        NeuromodulatorState {
            dopamine: 0.5,
            serotonin: 0.5,
            norepinephrine: 0.5,
            acetylcholine: 0.5,
            cortisol: 0.3,
            gaba: 0.5,
            oxytocin: 0.5,
            motivation: 0.5,
            patience: 0.5,
            stress: 0.3,
            learning_depth: 0.5,
            frustration: 0.2,
            exploration_drive: 0.3,
            impulse_control: 0.5,
            cooperativeness: 0.5,
            is_satiated: false,
            is_stressed: false,
            is_burned_out: false,
            is_deliberating: false,
            should_pivot: false,
            should_seek_help: false,
            prefer_cooperation: false,
            mood_stability: 0.7,
        }
    }

    fn make_test_action(name: &str) -> ActionTemplate {
        ActionTemplate {
            id: Uuid::new_v4(),
            name: name.to_string(),
            description: format!("Test action: {}", name),
            preconditions: vec![],
            expected_outcomes: vec![ExpectedOutcome {
                outcome: Outcome::success("Success", 0.5),
                probability: 0.8,
            }],
            effort_cost: 0.3,
            time_cost: 1,
            category: ActionCategory::Exploration,
            tags: vec![],
        }
    }

    #[test]
    fn test_curiosity_creation() {
        let curiosity = CuriositySystem::new();
        assert_eq!(curiosity.stats().total_explorations, 0);
    }

    #[test]
    fn test_domain_registration() {
        let mut curiosity = CuriositySystem::new();
        let domain = Domain::new("coding");

        curiosity.register_domain(domain.clone(), 0.8);

        assert_eq!(curiosity.uncertainty(&domain), 0.8);
    }

    #[test]
    fn test_novelty_decay() {
        let mut curiosity = CuriositySystem::new();

        // First observation is completely novel
        assert_eq!(curiosity.novelty("new_pattern"), 1.0);

        // After observing, novelty decreases
        curiosity.observe_pattern("new_pattern");
        let after_one = curiosity.novelty("new_pattern");
        assert!(after_one < 1.0, "Novelty should decrease after first observation");

        // Multiple observations decrease further
        curiosity.observe_pattern("new_pattern");
        let after_two = curiosity.novelty("new_pattern");
        assert!(after_two < after_one, "Novelty should decrease with more observations");

        // Many observations make it very familiar
        for _ in 0..20 {
            curiosity.observe_pattern("new_pattern");
        }
        let after_many = curiosity.novelty("new_pattern");
        // With decay rate 0.95, after 22 observations: 0.95^22 ≈ 0.32
        assert!(after_many < 0.4, "Novelty should be low after many observations: {}", after_many);
    }

    #[test]
    fn test_explore_vs_exploit() {
        let curiosity = CuriositySystem::new();

        // High dopamine + high motivation → exploit
        let mut exploit_state = make_test_state();
        exploit_state.dopamine = 0.9;
        exploit_state.motivation = 0.9;
        exploit_state.learning_depth = 0.2;

        let explore_prob = curiosity.explore_vs_exploit(&exploit_state);
        assert!(explore_prob < 0.5, "High dopamine should favor exploitation");

        // Low dopamine + high ACh → explore
        let mut explore_state = make_test_state();
        explore_state.dopamine = 0.2;
        explore_state.motivation = 0.2;
        explore_state.learning_depth = 0.9;
        explore_state.exploration_drive = 0.8;

        let explore_prob = curiosity.explore_vs_exploit(&explore_state);
        assert!(explore_prob > 0.5, "Low dopamine + high ACh should favor exploration");
    }

    #[test]
    fn test_competence_update() {
        let mut curiosity = CuriositySystem::new();
        let domain = Domain::new("rust");
        curiosity.register_domain(domain.clone(), 0.8);

        // Initial competence is 0
        assert_eq!(curiosity.competence(&domain).level, 0.0);

        // Success increases competence
        curiosity.update_competence(&domain, true, 0.5);
        assert!(curiosity.competence(&domain).level > 0.0);

        // Multiple successes increase further
        for _ in 0..10 {
            curiosity.update_competence(&domain, true, 0.5);
        }
        assert!(curiosity.competence(&domain).level > 0.3);
        assert!(curiosity.competence(&domain).confidence > 0.5);
    }

    #[test]
    fn test_exploration_event() {
        let mut curiosity = CuriositySystem::new();
        let domain = Domain::new("ml");
        curiosity.register_domain(domain.clone(), 1.0);

        let event = ExplorationEvent::new(domain.clone(), "learned about transformers")
            .with_info_gain(0.3)
            .with_novelty(0.8)
            .with_competence_gain(0.1);

        let initial_uncertainty = curiosity.uncertainty(&domain);

        curiosity.record_exploration(event);

        // Uncertainty should decrease
        assert!(curiosity.uncertainty(&domain) < initial_uncertainty);
        assert_eq!(curiosity.stats().total_explorations, 1);
    }

    #[test]
    fn test_curiosity_value() {
        let mut curiosity = CuriositySystem::new();
        curiosity.register_domain(Domain::new("exploration"), 0.9);

        let action = make_test_action("explore_new_area");
        let state = make_test_state();

        let value = curiosity.curiosity_value(&action, &state);
        assert!(value > 0.0);
    }

    #[test]
    fn test_domains_by_opportunity() {
        let mut curiosity = CuriositySystem::new();

        curiosity.register_domain(Domain::new("easy"), 0.2);
        curiosity.register_domain(Domain::new("hard"), 0.9);
        curiosity.register_domain(Domain::new("medium"), 0.5);

        let ranked = curiosity.domains_by_opportunity();

        // High uncertainty should rank higher
        assert!(ranked.len() == 3);
        assert!(ranked[0].1 >= ranked[1].1);
        assert!(ranked[1].1 >= ranked[2].1);
    }

    #[test]
    fn test_learning_opportunity() {
        let mut competence = Competence::default();
        competence.level = 0.5;
        competence.confidence = 0.8;

        // Optimal task is slightly above level (0.6)
        let optimal = competence.learning_opportunity(0.6);
        let too_easy = competence.learning_opportunity(0.2);
        let too_hard = competence.learning_opportunity(0.9);

        assert!(optimal > too_easy);
        assert!(optimal > too_hard);
    }

    #[test]
    fn test_most_uncertain_domain() {
        let mut curiosity = CuriositySystem::new();

        curiosity.register_domain(Domain::new("known"), 0.1);
        curiosity.register_domain(Domain::new("unknown"), 0.9);

        let most_uncertain = curiosity.most_uncertain_domain().unwrap();
        assert_eq!(most_uncertain.0, "unknown");
    }

    #[test]
    fn test_exploration_event_reward() {
        let event = ExplorationEvent::new(Domain::new("test"), "discovery")
            .with_info_gain(1.0)
            .with_novelty(1.0)
            .with_competence_gain(1.0);

        // Maximum reward should be 1.0 (0.4 + 0.3 + 0.3)
        assert!((event.curiosity_reward() - 1.0).abs() < 0.01);
    }
}
