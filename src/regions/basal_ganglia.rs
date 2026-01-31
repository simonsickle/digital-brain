//! Basal Ganglia - Action Selection and Habit Formation
//!
//! The basal ganglia is the brain's action selection system. It decides which
//! actions to execute and which to suppress, based on learned values and current
//! context. Key functions:
//!
//! - **Action gating**: Go/no-go decisions for competing actions
//! - **Habit formation**: Frequently rewarded action sequences become automatic
//! - **Reward prediction**: Learning expected values of actions
//! - **Motor programs**: Chunking action sequences into units
//!
//! # Computational Model
//!
//! Based on actor-critic architecture:
//! - **Striatum** (actor): Selects actions based on learned values
//! - **Pallidum**: Inhibits competing actions
//! - **Substantia Nigra**: Provides dopamine reward signals
//!
//! ```text
//!     Cortex (action candidates)
//!            │
//!            ▼
//!     ┌──────────────┐
//!     │   STRIATUM   │◀──── Dopamine (reward signal)
//!     │   (Select)   │
//!     └──────┬───────┘
//!            │
//!     ┌──────▼───────┐
//!     │   PALLIDUM   │
//!     │  (Inhibit)   │
//!     └──────┬───────┘
//!            │
//!            ▼
//!     Thalamus → Motor Output
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Unique identifier for an action pattern
pub type ActionPatternId = Uuid;

/// An action pattern that can become habitual
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionPattern {
    pub id: ActionPatternId,
    /// Name/description of the action
    pub name: String,
    /// Context in which this action is appropriate
    pub context_triggers: Vec<String>,
    /// Learned value of this action (Q-value)
    pub value: f64,
    /// How automatic this action has become (0 = deliberate, 1 = fully habitual)
    pub automaticity: f64,
    /// Number of times executed
    pub execution_count: u64,
    /// Number of times rewarded
    pub reward_count: u64,
    /// Average reward received
    pub average_reward: f64,
    /// Last execution time
    pub last_executed: Option<DateTime<Utc>>,
    /// Sequence of sub-actions (for motor programs)
    pub sub_actions: Vec<String>,
    /// Is this currently inhibited?
    pub inhibited: bool,
}

impl ActionPattern {
    pub fn new(name: &str) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.to_string(),
            context_triggers: Vec::new(),
            value: 0.0,
            automaticity: 0.0,
            execution_count: 0,
            reward_count: 0,
            average_reward: 0.0,
            last_executed: None,
            sub_actions: Vec::new(),
            inhibited: false,
        }
    }

    pub fn with_context(mut self, trigger: &str) -> Self {
        self.context_triggers.push(trigger.to_string());
        self
    }

    pub fn with_sub_actions(mut self, actions: Vec<&str>) -> Self {
        self.sub_actions = actions.into_iter().map(|s| s.to_string()).collect();
        self
    }

    /// Update after execution with reward
    pub fn update(&mut self, reward: f64, learning_rate: f64) {
        self.execution_count += 1;
        self.last_executed = Some(Utc::now());

        if reward > 0.0 {
            self.reward_count += 1;
        }

        // Update average reward (exponential moving average)
        self.average_reward = self.average_reward * (1.0 - learning_rate) + reward * learning_rate;

        // Update value (TD learning)
        let prediction_error = reward - self.value;
        self.value += learning_rate * prediction_error;

        // Increase automaticity with successful executions
        if reward > 0.0 {
            self.automaticity = (self.automaticity + 0.01).min(1.0);
        } else if reward < 0.0 {
            // Negative rewards decrease automaticity (forces deliberation)
            self.automaticity = (self.automaticity - 0.05).max(0.0);
        }
    }

    /// Check if this action is triggered by given context
    pub fn triggered_by(&self, context: &str) -> bool {
        let context_lower = context.to_lowercase();
        self.context_triggers
            .iter()
            .any(|t| context_lower.contains(&t.to_lowercase()))
    }

    /// Get success rate
    pub fn success_rate(&self) -> f64 {
        if self.execution_count == 0 {
            0.0
        } else {
            self.reward_count as f64 / self.execution_count as f64
        }
    }
}

/// Go/No-Go decision for an action
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateDecision {
    /// Execute this action
    Go,
    /// Suppress this action
    NoGo,
    /// Need more deliberation (competing high-value actions)
    Deliberate,
}

/// Result of action selection
#[derive(Debug, Clone)]
pub struct SelectionResult {
    /// Selected action (if any)
    pub selected: Option<ActionPatternId>,
    /// Gate decision
    pub decision: GateDecision,
    /// Competing actions that were suppressed
    pub suppressed: Vec<ActionPatternId>,
    /// Was this selection habitual or deliberate?
    pub habitual: bool,
    /// Confidence in selection (0-1)
    pub confidence: f64,
    /// Prediction error from last reward
    pub prediction_error: Option<f64>,
}

/// Statistics for the basal ganglia
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BasalGangliaStats {
    pub total_selections: u64,
    pub habitual_selections: u64,
    pub deliberate_selections: u64,
    pub no_go_decisions: u64,
    pub total_reward: f64,
    pub average_prediction_error: f64,
    pub habits_formed: u64,
}

/// The Basal Ganglia system
#[derive(Debug)]
pub struct BasalGanglia {
    /// Known action patterns
    actions: HashMap<ActionPatternId, ActionPattern>,
    /// Actions indexed by context trigger
    context_index: HashMap<String, Vec<ActionPatternId>>,
    /// Currently active action (being executed)
    active_action: Option<ActionPatternId>,
    /// Learning rate for value updates
    learning_rate: f64,
    /// Threshold for habitual execution (automaticity above this = no deliberation)
    habit_threshold: f64,
    /// Dopamine level (modulates learning and action selection)
    dopamine: f64,
    /// Statistics
    stats: BasalGangliaStats,
    /// Recent prediction errors (for tracking)
    recent_errors: Vec<f64>,
}

impl BasalGanglia {
    pub fn new() -> Self {
        Self {
            actions: HashMap::new(),
            context_index: HashMap::new(),
            active_action: None,
            learning_rate: 0.1,
            habit_threshold: 0.7,
            dopamine: 0.5,
            stats: BasalGangliaStats::default(),
            recent_errors: Vec::new(),
        }
    }

    /// Register a new action pattern
    pub fn register_action(&mut self, action: ActionPattern) {
        let id = action.id;

        // Index by context triggers
        for trigger in &action.context_triggers {
            self.context_index
                .entry(trigger.to_lowercase())
                .or_default()
                .push(id);
        }

        self.actions.insert(id, action);
    }

    /// Set dopamine level (from neuromodulator system)
    pub fn set_dopamine(&mut self, level: f64) {
        self.dopamine = level.clamp(0.0, 1.0);
    }

    /// Select an action given the current context
    pub fn select(&mut self, context: &str) -> SelectionResult {
        self.stats.total_selections += 1;

        // Find all actions triggered by this context
        let candidates = self.find_candidates(context);

        if candidates.is_empty() {
            return SelectionResult {
                selected: None,
                decision: GateDecision::NoGo,
                suppressed: vec![],
                habitual: false,
                confidence: 0.0,
                prediction_error: None,
            };
        }

        // Sort by value (highest first)
        let mut scored: Vec<(ActionPatternId, f64, f64)> = candidates
            .iter()
            .filter_map(|id| {
                self.actions.get(id).map(|a| {
                    // Score = value + dopamine bonus + automaticity bonus
                    let score = a.value + self.dopamine * 0.2 + a.automaticity * 0.1;
                    (*id, score, a.automaticity)
                })
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Check if top action is habitual
        if let Some((top_id, top_score, top_auto)) = scored.first() {
            let is_habitual = *top_auto >= self.habit_threshold;

            // Check for competing actions (close in value)
            let competition = if scored.len() > 1 {
                let second_score = scored[1].1;
                (top_score - second_score).abs() < 0.1
            } else {
                false
            };

            // Go/NoGo/Deliberate decision
            let decision = if competition && !is_habitual {
                // Close competition and not habitual = deliberate
                self.stats.deliberate_selections += 1;
                GateDecision::Deliberate
            } else if *top_score < 0.0 && !is_habitual {
                // Negative value and not habitual = suppress
                self.stats.no_go_decisions += 1;
                GateDecision::NoGo
            } else {
                // Execute
                if is_habitual {
                    self.stats.habitual_selections += 1;
                } else {
                    self.stats.deliberate_selections += 1;
                }
                GateDecision::Go
            };

            let suppressed: Vec<ActionPatternId> =
                scored.iter().skip(1).map(|(id, _, _)| *id).collect();

            // Calculate confidence based on margin
            let confidence = if scored.len() > 1 {
                let margin = top_score - scored[1].1;
                (margin / 2.0 + 0.5).clamp(0.0, 1.0)
            } else {
                0.8
            };

            SelectionResult {
                selected: if decision == GateDecision::Go {
                    self.active_action = Some(*top_id);
                    Some(*top_id)
                } else {
                    None
                },
                decision,
                suppressed,
                habitual: is_habitual,
                confidence,
                prediction_error: None,
            }
        } else {
            SelectionResult {
                selected: None,
                decision: GateDecision::NoGo,
                suppressed: vec![],
                habitual: false,
                confidence: 0.0,
                prediction_error: None,
            }
        }
    }

    /// Provide reward feedback for the active action
    pub fn reward(&mut self, amount: f64) -> Option<f64> {
        if let Some(action_id) = self.active_action
            && let Some(action) = self.actions.get_mut(&action_id)
        {
            let old_value = action.value;
            action.update(amount, self.learning_rate * (1.0 + self.dopamine));

            let prediction_error = amount - old_value;
            self.recent_errors.push(prediction_error);
            if self.recent_errors.len() > 100 {
                self.recent_errors.remove(0);
            }

            // Update stats
            self.stats.total_reward += amount;
            self.stats.average_prediction_error =
                self.recent_errors.iter().sum::<f64>() / self.recent_errors.len() as f64;

            // Check if this action just became a habit
            if action.automaticity >= self.habit_threshold
                && action.execution_count == 1 + (self.habit_threshold / 0.01) as u64
            {
                self.stats.habits_formed += 1;
            }

            self.active_action = None;
            return Some(prediction_error);
        }
        None
    }

    /// Complete current action without explicit reward (neutral outcome)
    pub fn complete(&mut self) {
        self.active_action = None;
    }

    /// Find candidate actions for a context
    fn find_candidates(&self, context: &str) -> Vec<ActionPatternId> {
        let mut candidates = Vec::new();
        let context_lower = context.to_lowercase();

        // Direct context matches
        for (trigger, ids) in &self.context_index {
            if context_lower.contains(trigger) {
                candidates.extend(ids.iter().cloned());
            }
        }

        // Also check actions directly
        for (id, action) in &self.actions {
            if action.triggered_by(&context_lower) && !candidates.contains(id) {
                candidates.push(*id);
            }
        }

        // Remove inhibited actions
        candidates.retain(|id| self.actions.get(id).map(|a| !a.inhibited).unwrap_or(false));

        candidates
    }

    /// Inhibit an action (prevent it from being selected)
    pub fn inhibit(&mut self, action_id: ActionPatternId) {
        if let Some(action) = self.actions.get_mut(&action_id) {
            action.inhibited = true;
        }
    }

    /// Release inhibition on an action
    pub fn release(&mut self, action_id: ActionPatternId) {
        if let Some(action) = self.actions.get_mut(&action_id) {
            action.inhibited = false;
        }
    }

    /// Get an action by ID
    pub fn get_action(&self, id: ActionPatternId) -> Option<&ActionPattern> {
        self.actions.get(&id)
    }

    /// Get all habits (actions above automaticity threshold)
    pub fn habits(&self) -> Vec<&ActionPattern> {
        self.actions
            .values()
            .filter(|a| a.automaticity >= self.habit_threshold)
            .collect()
    }

    /// Get statistics
    pub fn stats(&self) -> &BasalGangliaStats {
        &self.stats
    }

    /// Decay automaticity for unused actions (use it or lose it)
    pub fn decay_unused(&mut self, hours_threshold: f64, decay_amount: f64) {
        let now = Utc::now();
        let threshold = chrono::Duration::hours(hours_threshold as i64);

        for action in self.actions.values_mut() {
            if let Some(last) = action.last_executed
                && now - last > threshold
            {
                action.automaticity = (action.automaticity - decay_amount).max(0.0);
            }
        }
    }

    /// Create a motor program (sequence of actions)
    pub fn create_motor_program(
        &mut self,
        name: &str,
        sequence: Vec<&str>,
        context: &str,
    ) -> ActionPatternId {
        let program = ActionPattern::new(name)
            .with_context(context)
            .with_sub_actions(sequence);
        let id = program.id;
        self.register_action(program);
        id
    }

    /// Get actions sorted by value
    pub fn actions_by_value(&self) -> Vec<&ActionPattern> {
        let mut actions: Vec<&ActionPattern> = self.actions.values().collect();
        actions.sort_by(|a, b| {
            b.value
                .partial_cmp(&a.value)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        actions
    }
}

impl Default for BasalGanglia {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_registration_and_selection() {
        let mut bg = BasalGanglia::new();

        let action = ActionPattern::new("greet").with_context("hello");
        let id = action.id;
        bg.register_action(action);

        let result = bg.select("hello there");
        assert_eq!(result.selected, Some(id));
        assert_eq!(result.decision, GateDecision::Go);
    }

    #[test]
    fn test_reward_learning() {
        let mut bg = BasalGanglia::new();

        let action = ActionPattern::new("test").with_context("test");
        let id = action.id;
        bg.register_action(action);

        // Select and reward
        bg.select("test context");
        let error = bg.reward(1.0);

        assert!(error.is_some());
        let action = bg.get_action(id).unwrap();
        assert!(action.value > 0.0);
        assert_eq!(action.execution_count, 1);
    }

    #[test]
    fn test_habit_formation() {
        let mut bg = BasalGanglia::new();
        bg.learning_rate = 0.5; // Fast learning for test

        let action = ActionPattern::new("habit_test").with_context("habit");
        let id = action.id;
        bg.register_action(action);

        // Repeatedly reward to form habit
        for _ in 0..100 {
            bg.select("habit context");
            bg.reward(1.0);
        }

        let action = bg.get_action(id).unwrap();
        assert!(
            action.automaticity >= bg.habit_threshold,
            "Action should become habitual"
        );
    }

    #[test]
    fn test_inhibition() {
        let mut bg = BasalGanglia::new();

        let action = ActionPattern::new("inhibit_test").with_context("inhibit");
        let id = action.id;
        bg.register_action(action);

        bg.inhibit(id);

        let result = bg.select("inhibit context");
        assert!(
            result.selected.is_none(),
            "Inhibited action should not be selected"
        );
    }

    #[test]
    fn test_competition_and_deliberation() {
        let mut bg = BasalGanglia::new();

        // Two actions with same context, similar values
        let mut action1 = ActionPattern::new("option_a").with_context("choice");
        action1.value = 0.5;
        let mut action2 = ActionPattern::new("option_b").with_context("choice");
        action2.value = 0.48;

        bg.register_action(action1);
        bg.register_action(action2);

        let result = bg.select("make a choice");
        assert_eq!(
            result.decision,
            GateDecision::Deliberate,
            "Close competition should trigger deliberation"
        );
    }

    #[test]
    fn test_motor_program() {
        let mut bg = BasalGanglia::new();

        let id = bg.create_motor_program(
            "morning_routine",
            vec!["wake", "coffee", "email"],
            "morning",
        );

        let action = bg.get_action(id).unwrap();
        assert_eq!(action.sub_actions.len(), 3);
        assert!(action.triggered_by("morning time"));
    }
}
