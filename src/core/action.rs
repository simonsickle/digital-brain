//! Action Selection System
//!
//! The brain can process inputs, but needs a way to decide on outputs/actions.
//! This module provides action selection based on neuromodulator state, goals,
//! and learned action values.
//!
//! Inspired by:
//! - Basal ganglia action selection (go/no-go pathways)
//! - Actor-critic reinforcement learning
//! - Neuromodulatory influences on decision-making (Doya)

use std::collections::HashMap;
use uuid::Uuid;
use serde::{Deserialize, Serialize};

use crate::core::neuromodulators::NeuromodulatorState;
// Note: Valence imported but unused - kept for future expansion
#[allow(unused_imports)]
use crate::signal::Valence;

/// Unique identifier for an action
pub type ActionId = Uuid;

/// Condition that must be met for an action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Condition {
    /// A state must be true
    StateIs { key: String, value: String },
    /// A resource must be available
    ResourceAvailable { resource: String, amount: f64 },
    /// Time-based condition
    TimeElapsed { since: String, duration_secs: u64 },
    /// Goal must be active
    GoalActive { goal_description: String },
    /// Neuromodulator threshold
    NeuromodulatorAbove { modulator: String, threshold: f64 },
    /// Custom predicate (for extensibility)
    Custom { name: String, params: HashMap<String, String> },
}

/// Expected outcome of an action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Outcome {
    /// Description of what happened
    pub description: String,
    /// Value/reward of the outcome (-1 to +1)
    pub value: f64,
    /// Related goal (if any)
    pub related_goal: Option<String>,
    /// Progress made toward goal (0 to 1)
    pub progress: f64,
    /// Was this outcome surprising?
    pub surprise: f64,
    /// Side effects
    pub side_effects: Vec<String>,
}

impl Outcome {
    pub fn success(description: &str, value: f64) -> Self {
        Self {
            description: description.to_string(),
            value,
            related_goal: None,
            progress: 0.0,
            surprise: 0.0,
            side_effects: Vec::new(),
        }
    }

    pub fn failure(description: &str) -> Self {
        Self {
            description: description.to_string(),
            value: -0.3,
            related_goal: None,
            progress: 0.0,
            surprise: 0.3,
            side_effects: Vec::new(),
        }
    }

    pub fn with_goal(mut self, goal: &str, progress: f64) -> Self {
        self.related_goal = Some(goal.to_string());
        self.progress = progress;
        self
    }
}

/// Template for an available action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionTemplate {
    /// Unique identifier
    pub id: ActionId,
    /// Human-readable name
    pub name: String,
    /// Description of what this action does
    pub description: String,
    /// Conditions that must be met to execute
    pub preconditions: Vec<Condition>,
    /// Expected outcomes (probabilistic)
    pub expected_outcomes: Vec<ExpectedOutcome>,
    /// Effort cost (0 to 1, affects fatigue)
    pub effort_cost: f64,
    /// Time cost in arbitrary units
    pub time_cost: u32,
    /// Action category for tolerance tracking
    pub category: ActionCategory,
    /// Tags for filtering/selection
    pub tags: Vec<String>,
}

/// Expected outcome with probability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedOutcome {
    pub outcome: Outcome,
    pub probability: f64,
}

/// Categories of actions (for tolerance and preference tracking)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ActionCategory {
    /// Information seeking (reading, searching)
    Exploration,
    /// Goal-directed work
    Exploitation,
    /// Communication (sending messages, responding)
    Communication,
    /// Maintenance (organizing, cleaning up)
    Maintenance,
    /// Rest/recovery
    Rest,
    /// Learning/skill acquisition
    Learning,
    /// Social interaction
    Social,
    /// Creative generation
    Creative,
    /// Defensive/protective
    Defensive,
}

/// Decision made by the action selector
#[derive(Debug, Clone)]
pub enum ActionDecision {
    /// Execute this specific action
    Execute(ActionId),
    /// Wait before acting
    Wait {
        reason: String,
        until: Option<Condition>,
    },
    /// Need more deliberation between options
    Deliberate {
        options: Vec<ActionId>,
        conflict_reason: String,
    },
    /// Curiosity-driven exploration
    Explore {
        domain: String,
        curiosity_level: f64,
    },
    /// No valid action available
    NoAction {
        reason: String,
    },
}

/// Statistics about action selection
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ActionStats {
    pub total_decisions: u64,
    pub executions: u64,
    pub waits: u64,
    pub deliberations: u64,
    pub explorations: u64,
    pub no_actions: u64,
    pub average_decision_confidence: f64,
    pub category_counts: HashMap<ActionCategory, u64>,
}

/// The action selector - decides what to do based on brain state
#[derive(Debug)]
pub struct ActionSelector {
    /// Available action templates
    available_actions: Vec<ActionTemplate>,
    /// Learned action values (Q-values)
    action_values: HashMap<ActionId, f64>,
    /// Action inhibition map (action A inhibits actions B, C, ...)
    inhibition_map: HashMap<ActionId, Vec<ActionId>>,
    /// Recent action history (for pattern detection)
    action_history: Vec<ActionHistoryEntry>,
    /// Maximum history length
    max_history: usize,
    /// Exploration rate (epsilon in epsilon-greedy)
    exploration_rate: f64,
    /// Learning rate for value updates
    learning_rate: f64,
    /// Statistics
    stats: ActionStats,
}

#[derive(Debug, Clone)]
struct ActionHistoryEntry {
    action_id: ActionId,
    timestamp: std::time::Instant,
    outcome_value: Option<f64>,
}

impl Default for ActionSelector {
    fn default() -> Self {
        Self::new()
    }
}

impl ActionSelector {
    /// Create a new action selector
    pub fn new() -> Self {
        Self {
            available_actions: Vec::new(),
            action_values: HashMap::new(),
            inhibition_map: HashMap::new(),
            action_history: Vec::new(),
            max_history: 100,
            exploration_rate: 0.1,
            learning_rate: 0.1,
            stats: ActionStats::default(),
        }
    }

    /// Register an available action
    pub fn register_action(&mut self, template: ActionTemplate) {
        // Initialize value if not present
        if !self.action_values.contains_key(&template.id) {
            // Initialize based on expected value
            let expected_value: f64 = template
                .expected_outcomes
                .iter()
                .map(|eo| eo.outcome.value * eo.probability)
                .sum();
            self.action_values.insert(template.id, expected_value);
        }
        self.available_actions.push(template);
    }

    /// Add inhibition relationship (action A inhibits action B)
    pub fn add_inhibition(&mut self, inhibitor: ActionId, inhibited: ActionId) {
        self.inhibition_map
            .entry(inhibitor)
            .or_default()
            .push(inhibited);
    }

    /// Select an action based on current state
    pub fn select(
        &mut self,
        neuro_state: &NeuromodulatorState,
        active_goals: &[String],
        current_state: &HashMap<String, String>,
    ) -> ActionDecision {
        self.stats.total_decisions += 1;

        // Filter to valid actions (preconditions met)
        let valid_actions: Vec<_> = self
            .available_actions
            .iter()
            .filter(|a| self.check_preconditions(&a.preconditions, current_state, neuro_state))
            .collect();

        if valid_actions.is_empty() {
            self.stats.no_actions += 1;
            return ActionDecision::NoAction {
                reason: "No actions have satisfied preconditions".to_string(),
            };
        }

        // Check if we should wait (patience-gated)
        if self.should_wait(neuro_state) {
            self.stats.waits += 1;
            return ActionDecision::Wait {
                reason: "High patience suggests waiting for better opportunity".to_string(),
                until: None,
            };
        }

        // Score each action
        let mut scored_actions: Vec<_> = valid_actions
            .iter()
            .map(|a| {
                let score = self.score_action(a, neuro_state, active_goals);
                (a, score)
            })
            .collect();

        // Sort by score (highest first)
        scored_actions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Check for exploration (epsilon-greedy with neuromodulator influence)
        let explore_chance = self.exploration_rate * (1.0 - neuro_state.motivation);
        if rand_float() < explore_chance && scored_actions.len() > 1 {
            self.stats.explorations += 1;
            return ActionDecision::Explore {
                domain: scored_actions
                    .last()
                    .map(|(a, _)| a.name.clone())
                    .unwrap_or_default(),
                curiosity_level: explore_chance,
            };
        }

        // Check for decision conflict (top actions too close in score)
        if scored_actions.len() >= 2 {
            let top_score = scored_actions[0].1;
            let second_score = scored_actions[1].1;
            let score_diff = (top_score - second_score).abs();

            // If scores are very close and we're not highly motivated, deliberate
            if score_diff < 0.1 && neuro_state.motivation < 0.7 {
                self.stats.deliberations += 1;
                return ActionDecision::Deliberate {
                    options: scored_actions
                        .iter()
                        .take(3)
                        .map(|(a, _)| a.id)
                        .collect(),
                    conflict_reason: format!(
                        "Top actions have similar scores ({:.2} vs {:.2})",
                        top_score, second_score
                    ),
                };
            }
        }

        // Execute top action
        if let Some((action, _score)) = scored_actions.first() {
            self.stats.executions += 1;
            *self.stats.category_counts.entry(action.category).or_insert(0) += 1;

            // Record in history
            self.action_history.push(ActionHistoryEntry {
                action_id: action.id,
                timestamp: std::time::Instant::now(),
                outcome_value: None,
            });
            if self.action_history.len() > self.max_history {
                self.action_history.remove(0);
            }

            return ActionDecision::Execute(action.id);
        }

        self.stats.no_actions += 1;
        ActionDecision::NoAction {
            reason: "No valid action found after scoring".to_string(),
        }
    }

    /// Score an action based on multiple factors
    fn score_action(
        &self,
        action: &ActionTemplate,
        neuro_state: &NeuromodulatorState,
        active_goals: &[String],
    ) -> f64 {
        // Base value from learning
        let base_value = self.action_values.get(&action.id).copied().unwrap_or(0.0);

        // Goal alignment bonus
        let goal_bonus = self.goal_alignment(action, active_goals);

        // Effort cost penalty (modulated by motivation)
        let effort_penalty = action.effort_cost * (1.0 - neuro_state.motivation * 0.5);

        // Category preference based on neuromodulator state
        let category_modifier = self.category_preference(action.category, neuro_state);

        // Recency penalty (avoid repeating same action)
        let recency_penalty = self.recency_penalty(action.id);

        // Inhibition from recent actions
        let inhibition = self.inhibition_penalty(action.id);

        // Combine factors
        let score = (base_value + goal_bonus + category_modifier)
            - effort_penalty
            - recency_penalty
            - inhibition;

        score.max(0.0) // Floor at 0
    }

    /// Calculate goal alignment bonus
    fn goal_alignment(&self, action: &ActionTemplate, active_goals: &[String]) -> f64 {
        let mut bonus = 0.0;
        for outcome in &action.expected_outcomes {
            if let Some(ref goal) = outcome.outcome.related_goal {
                if active_goals.iter().any(|g| g.contains(goal) || goal.contains(g)) {
                    bonus += outcome.probability * outcome.outcome.progress * 0.5;
                }
            }
        }
        bonus
    }

    /// Get category preference based on neuromodulator state
    fn category_preference(&self, category: ActionCategory, state: &NeuromodulatorState) -> f64 {
        match category {
            ActionCategory::Exploration => {
                // High ACh, low motivation → explore
                state.learning_depth * (1.0 - state.motivation) * 0.3
            }
            ActionCategory::Exploitation => {
                // High motivation → exploit
                state.motivation * 0.4
            }
            ActionCategory::Defensive => {
                // High stress → defensive
                state.stress * 0.5
            }
            ActionCategory::Rest => {
                // Low dopamine, low norepinephrine (arousal) → rest
                (1.0 - state.dopamine) * (1.0 - state.norepinephrine) * 0.3
            }
            ActionCategory::Communication => {
                // Moderate preference, boosted by social contexts
                0.1
            }
            ActionCategory::Learning => {
                // High ACh → learning
                state.learning_depth * 0.3
            }
            ActionCategory::Social => {
                // Moderate baseline
                0.1 + (1.0 - state.stress) * 0.1
            }
            ActionCategory::Creative => {
                // Low stress, moderate dopamine
                (1.0 - state.stress) * state.dopamine * 0.2
            }
            ActionCategory::Maintenance => {
                // Low priority unless specifically needed
                0.05
            }
        }
    }

    /// Calculate recency penalty (avoid repeating same action)
    fn recency_penalty(&self, action_id: ActionId) -> f64 {
        for (i, entry) in self.action_history.iter().rev().enumerate() {
            if entry.action_id == action_id {
                // More recent = higher penalty, decays with distance
                return 0.3 * (1.0 / (i as f64 + 1.0));
            }
        }
        0.0
    }

    /// Calculate inhibition from recent actions
    fn inhibition_penalty(&self, action_id: ActionId) -> f64 {
        let mut penalty = 0.0;
        for entry in self.action_history.iter().rev().take(5) {
            if let Some(inhibited) = self.inhibition_map.get(&entry.action_id) {
                if inhibited.contains(&action_id) {
                    penalty += 0.2;
                }
            }
        }
        penalty
    }

    /// Check if patience suggests waiting
    fn should_wait(&self, state: &NeuromodulatorState) -> bool {
        // High patience + low urgency → wait
        state.patience > 0.8 && state.stress < 0.3 && state.motivation < 0.5
    }

    /// Check if preconditions are met
    fn check_preconditions(
        &self,
        conditions: &[Condition],
        current_state: &HashMap<String, String>,
        neuro_state: &NeuromodulatorState,
    ) -> bool {
        for condition in conditions {
            let met = match condition {
                Condition::StateIs { key, value } => {
                    current_state.get(key).map(|v| v == value).unwrap_or(false)
                }
                Condition::ResourceAvailable { .. } => true, // Assume available for now
                Condition::TimeElapsed { .. } => true,       // Assume elapsed for now
                Condition::GoalActive { .. } => true,        // Assume active for now
                Condition::NeuromodulatorAbove { modulator, threshold } => {
                    let value = match modulator.as_str() {
                        "dopamine" => neuro_state.dopamine,
                        "serotonin" => neuro_state.serotonin,
                        "norepinephrine" | "arousal" => neuro_state.norepinephrine,
                        "acetylcholine" => neuro_state.acetylcholine,
                        "cortisol" => neuro_state.cortisol,
                        "gaba" => neuro_state.gaba,
                        "oxytocin" => neuro_state.oxytocin,
                        "motivation" => neuro_state.motivation,
                        "patience" => neuro_state.patience,
                        "stress" => neuro_state.stress,
                        "learning_depth" => neuro_state.learning_depth,
                        "frustration" => neuro_state.frustration,
                        "exploration_drive" => neuro_state.exploration_drive,
                        "impulse_control" => neuro_state.impulse_control,
                        "cooperativeness" => neuro_state.cooperativeness,
                        _ => 0.0,
                    };
                    value > *threshold
                }
                Condition::Custom { .. } => true, // Assume met for extensibility
            };
            if !met {
                return false;
            }
        }
        true
    }

    /// Update action value based on outcome (TD learning)
    pub fn update_from_outcome(&mut self, action_id: ActionId, outcome: &Outcome) {
        // Update history with outcome
        for entry in self.action_history.iter_mut().rev() {
            if entry.action_id == action_id && entry.outcome_value.is_none() {
                entry.outcome_value = Some(outcome.value);
                break;
            }
        }

        // TD update: V(s) = V(s) + α * (r - V(s))
        let current_value = self.action_values.get(&action_id).copied().unwrap_or(0.0);
        let td_error = outcome.value - current_value;
        let new_value = current_value + self.learning_rate * td_error;

        self.action_values.insert(action_id, new_value);

        // Update confidence
        self.stats.average_decision_confidence =
            self.stats.average_decision_confidence * 0.99 + (1.0 - td_error.abs()) * 0.01;
    }

    /// Get action by ID
    pub fn get_action(&self, id: ActionId) -> Option<&ActionTemplate> {
        self.available_actions.iter().find(|a| a.id == id)
    }

    /// Get statistics
    pub fn stats(&self) -> &ActionStats {
        &self.stats
    }

    /// Set exploration rate
    pub fn set_exploration_rate(&mut self, rate: f64) {
        self.exploration_rate = rate.clamp(0.0, 1.0);
    }

    /// Set learning rate
    pub fn set_learning_rate(&mut self, rate: f64) {
        self.learning_rate = rate.clamp(0.0, 1.0);
    }

    /// Get all registered actions
    pub fn actions(&self) -> &[ActionTemplate] {
        &self.available_actions
    }

    /// Clear action history
    pub fn clear_history(&mut self) {
        self.action_history.clear();
    }
}

/// Simple random float (0.0 to 1.0) - replace with proper RNG in production
fn rand_float() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    (nanos % 1000) as f64 / 1000.0
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_neuro_state() -> NeuromodulatorState {
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

    fn make_test_action(name: &str, category: ActionCategory) -> ActionTemplate {
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
            category,
            tags: vec![],
        }
    }

    #[test]
    fn test_action_registration() {
        let mut selector = ActionSelector::new();
        let action = make_test_action("test", ActionCategory::Exploration);
        let id = action.id;

        selector.register_action(action);

        assert_eq!(selector.actions().len(), 1);
        assert!(selector.get_action(id).is_some());
    }

    #[test]
    fn test_action_selection_basic() {
        let mut selector = ActionSelector::new();
        selector.register_action(make_test_action("explore", ActionCategory::Exploration));
        selector.register_action(make_test_action("exploit", ActionCategory::Exploitation));

        let state = make_test_neuro_state();
        let goals = vec!["complete task".to_string()];
        let current_state = HashMap::new();

        let decision = selector.select(&state, &goals, &current_state);

        // Should make some decision
        assert!(matches!(
            decision,
            ActionDecision::Execute(_) | ActionDecision::Deliberate { .. }
        ));
    }

    #[test]
    fn test_high_motivation_prefers_exploitation() {
        let mut selector = ActionSelector::new();
        
        let explore = make_test_action("explore", ActionCategory::Exploration);
        let exploit = make_test_action("exploit", ActionCategory::Exploitation);
        let exploit_id = exploit.id;
        
        selector.register_action(explore);
        selector.register_action(exploit);
        selector.set_exploration_rate(0.0); // Disable random exploration

        let mut state = make_test_neuro_state();
        state.motivation = 0.9; // High motivation
        state.learning_depth = 0.1; // Low learning drive

        let decision = selector.select(&state, &[], &HashMap::new());

        // Should prefer exploitation
        if let ActionDecision::Execute(id) = decision {
            assert_eq!(id, exploit_id);
        }
    }

    #[test]
    fn test_high_stress_prefers_defensive() {
        let mut selector = ActionSelector::new();
        
        let normal = make_test_action("normal", ActionCategory::Exploitation);
        let defensive = make_test_action("defensive", ActionCategory::Defensive);
        let defensive_id = defensive.id;
        
        selector.register_action(normal);
        selector.register_action(defensive);
        selector.set_exploration_rate(0.0);

        let mut state = make_test_neuro_state();
        state.stress = 0.9; // High stress
        state.motivation = 0.3;

        let decision = selector.select(&state, &[], &HashMap::new());

        // Should prefer defensive
        if let ActionDecision::Execute(id) = decision {
            assert_eq!(id, defensive_id);
        }
    }

    #[test]
    fn test_outcome_updates_value() {
        let mut selector = ActionSelector::new();
        let action = make_test_action("test", ActionCategory::Exploration);
        let id = action.id;
        selector.register_action(action);

        // Execute action
        selector.action_history.push(ActionHistoryEntry {
            action_id: id,
            timestamp: std::time::Instant::now(),
            outcome_value: None,
        });

        let initial_value = *selector.action_values.get(&id).unwrap();

        // Good outcome
        selector.update_from_outcome(id, &Outcome::success("Great!", 1.0));

        let new_value = *selector.action_values.get(&id).unwrap();
        assert!(new_value > initial_value);
    }

    #[test]
    fn test_recency_penalty() {
        let mut selector = ActionSelector::new();
        let action = make_test_action("test", ActionCategory::Exploration);
        let id = action.id;
        selector.register_action(action);

        // No recent execution
        assert_eq!(selector.recency_penalty(id), 0.0);

        // Execute action
        selector.action_history.push(ActionHistoryEntry {
            action_id: id,
            timestamp: std::time::Instant::now(),
            outcome_value: None,
        });

        // Should have recency penalty now
        assert!(selector.recency_penalty(id) > 0.0);
    }

    #[test]
    fn test_patience_causes_wait() {
        let mut selector = ActionSelector::new();
        selector.register_action(make_test_action("test", ActionCategory::Exploration));

        let mut state = make_test_neuro_state();
        state.patience = 0.95; // Very high patience
        state.stress = 0.1;    // Low stress
        state.motivation = 0.2; // Low motivation

        let decision = selector.select(&state, &[], &HashMap::new());

        assert!(matches!(decision, ActionDecision::Wait { .. }));
    }

    #[test]
    fn test_no_valid_actions() {
        let mut selector = ActionSelector::new();
        
        // Action with impossible precondition
        let mut action = make_test_action("impossible", ActionCategory::Exploration);
        action.preconditions = vec![Condition::StateIs {
            key: "impossible".to_string(),
            value: "true".to_string(),
        }];
        selector.register_action(action);

        let state = make_test_neuro_state();
        let decision = selector.select(&state, &[], &HashMap::new());

        assert!(matches!(decision, ActionDecision::NoAction { .. }));
    }

    #[test]
    fn test_stats_tracking() {
        let mut selector = ActionSelector::new();
        selector.register_action(make_test_action("test", ActionCategory::Exploration));

        let state = make_test_neuro_state();

        // Make several decisions
        for _ in 0..5 {
            selector.select(&state, &[], &HashMap::new());
        }

        assert_eq!(selector.stats().total_decisions, 5);
    }
}
