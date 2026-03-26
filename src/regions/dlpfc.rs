//! Dorsolateral Prefrontal Cortex (dlPFC) - Cognitive Flexibility
//!
//! The dlPFC is the seat of executive control beyond simple working memory:
//!
//! - **Task Switching**: Shifting between different cognitive tasks/rules
//! - **Rule Learning**: Discovering and applying abstract rules from experience
//! - **Set Shifting**: Changing behavioral strategies when conditions change
//! - **Inhibitory Control**: Suppressing prepotent but incorrect responses
//! - **Cognitive Load Tracking**: Monitoring processing demands
//!
//! # Theoretical Foundation
//!
//! Based on:
//! - Miller & Cohen (2001): Prefrontal cortex and cognitive control
//! - Miyake et al. (2000): Three executive functions (shifting, updating, inhibiting)
//! - Wisconsin Card Sorting Task: Measuring set shifting ability
//! - Monsell (2003): Task-set reconfiguration and switch costs
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │              DORSOLATERAL PREFRONTAL CORTEX                  │
//! ├───────────────┬──────────────┬───────────────┬──────────────┤
//! │     Task      │    Rule      │     Set       │  Inhibitory  │
//! │   Switching   │   Learning   │   Shifting    │   Control    │
//! │ (reconfigure) │  (abstract)  │  (flexible)   │  (suppress)  │
//! └───────────────┴──────────────┴───────────────┴──────────────┘
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use uuid::Uuid;

/// Configuration for cognitive flexibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DlpfcConfig {
    /// Base cost of switching tasks (0-1, affects performance briefly)
    pub switch_cost: f64,
    /// How fast switch cost decays per cycle
    pub switch_recovery_rate: f64,
    /// Maximum number of active rules
    pub max_active_rules: usize,
    /// Minimum evidence to learn a new rule
    pub rule_learning_threshold: u32,
    /// How many recent outcomes to track for set shifting
    pub outcome_history_size: usize,
    /// Consecutive failures before triggering a set shift
    pub perseveration_threshold: u32,
    /// Inhibition strength (how well prepotent responses are suppressed)
    pub inhibition_strength: f64,
}

impl Default for DlpfcConfig {
    fn default() -> Self {
        Self {
            switch_cost: 0.3,
            switch_recovery_rate: 0.1,
            max_active_rules: 5,
            rule_learning_threshold: 3,
            outcome_history_size: 20,
            perseveration_threshold: 3,
            inhibition_strength: 0.7,
        }
    }
}

/// A cognitive task set: the current "mode" of processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskSet {
    pub id: Uuid,
    pub name: String,
    /// The rules active under this task set
    pub active_rules: Vec<RuleId>,
    /// When this task set was activated
    pub activated_at: DateTime<Utc>,
    /// Total cycles spent in this task set
    pub total_cycles: u64,
    /// Successful outcomes while in this set
    pub successes: u32,
    /// Failed outcomes while in this set
    pub failures: u32,
}

impl TaskSet {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            active_rules: Vec::new(),
            activated_at: Utc::now(),
            total_cycles: 0,
            successes: 0,
            failures: 0,
        }
    }

    /// Success rate for this task set.
    pub fn success_rate(&self) -> f64 {
        let total = self.successes + self.failures;
        if total == 0 {
            0.5 // No data yet
        } else {
            self.successes as f64 / total as f64
        }
    }
}

/// Unique identifier for a learned rule.
pub type RuleId = Uuid;

/// An abstract rule learned from experience.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveRule {
    pub id: RuleId,
    /// Human-readable description of the rule
    pub description: String,
    /// Condition pattern (what triggers this rule)
    pub condition: String,
    /// Action/response pattern
    pub action: String,
    /// How many times this rule has been confirmed
    pub confirmations: u32,
    /// How many times this rule has been violated
    pub violations: u32,
    /// Confidence in this rule (0-1)
    pub confidence: f64,
    /// When this rule was first learned
    pub learned_at: DateTime<Utc>,
    /// Context tags (when this rule applies)
    pub context_tags: Vec<String>,
}

impl CognitiveRule {
    pub fn new(
        description: impl Into<String>,
        condition: impl Into<String>,
        action: impl Into<String>,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            description: description.into(),
            condition: condition.into(),
            action: action.into(),
            confirmations: 1,
            violations: 0,
            confidence: 0.3,
            learned_at: Utc::now(),
            context_tags: Vec::new(),
        }
    }

    /// Confirm this rule (increase confidence).
    pub fn confirm(&mut self) {
        self.confirmations += 1;
        self.confidence = (self.confidence + 0.1).min(1.0);
    }

    /// Violate this rule (decrease confidence).
    pub fn violate(&mut self) {
        self.violations += 1;
        self.confidence = (self.confidence - 0.15).max(0.0);
    }

    /// Reliability: confirmations vs total observations.
    pub fn reliability(&self) -> f64 {
        let total = self.confirmations + self.violations;
        if total == 0 {
            0.0
        } else {
            self.confirmations as f64 / total as f64
        }
    }

    pub fn with_context(mut self, tag: impl Into<String>) -> Self {
        self.context_tags.push(tag.into());
        self
    }
}

/// An observed outcome for tracking performance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Outcome {
    pub success: bool,
    pub task_set_id: Uuid,
    pub rule_applied: Option<RuleId>,
    pub timestamp: DateTime<Utc>,
    pub context: String,
}

/// Result of a set shift evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SetShiftResult {
    /// Whether a shift was recommended
    pub should_shift: bool,
    /// Consecutive failures that triggered the evaluation
    pub consecutive_failures: u32,
    /// Current task set success rate
    pub current_success_rate: f64,
    /// Suggested new strategy (if shifting)
    pub suggestion: Option<String>,
}

/// Result of attempting to inhibit a response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InhibitionOutcome {
    /// Was the response successfully inhibited?
    pub inhibited: bool,
    /// The response that was (or wasn't) suppressed
    pub response: String,
    /// Why inhibition was attempted
    pub reason: String,
    /// Cognitive load from the inhibition attempt
    pub cognitive_cost: f64,
}

/// Statistics about cognitive flexibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DlpfcStats {
    pub total_task_switches: u64,
    pub total_rules_learned: usize,
    pub total_set_shifts: u64,
    pub total_inhibitions: u64,
    pub successful_inhibitions: u64,
    pub current_task_set: Option<String>,
    pub current_switch_cost: f64,
    pub cognitive_load: f64,
    pub perseveration_score: f64,
}

/// The Dorsolateral Prefrontal Cortex module.
pub struct DLPFC {
    config: DlpfcConfig,
    /// Current active task set
    current_task_set: Option<TaskSet>,
    /// Previously used task sets (for switching back)
    task_set_history: VecDeque<TaskSet>,
    /// All learned rules
    rules: HashMap<RuleId, CognitiveRule>,
    /// Recent outcomes for set shifting detection
    outcome_history: VecDeque<Outcome>,
    /// Current switch cost (decays over time)
    current_switch_cost: f64,
    /// Current cognitive load (0-1)
    cognitive_load: f64,
    /// Pending rule candidates (condition -> observations before becoming a rule)
    rule_candidates: HashMap<String, Vec<(String, String)>>,
    /// Statistics
    total_switches: u64,
    total_set_shifts: u64,
    total_inhibitions: u64,
    successful_inhibitions: u64,
}

impl DLPFC {
    /// Create a new dlPFC with default configuration.
    pub fn new() -> Self {
        Self::with_config(DlpfcConfig::default())
    }

    /// Create with custom configuration.
    pub fn with_config(config: DlpfcConfig) -> Self {
        Self {
            config,
            current_task_set: None,
            task_set_history: VecDeque::with_capacity(10),
            rules: HashMap::new(),
            outcome_history: VecDeque::with_capacity(20),
            current_switch_cost: 0.0,
            cognitive_load: 0.0,
            rule_candidates: HashMap::new(),
            total_switches: 0,
            total_set_shifts: 0,
            total_inhibitions: 0,
            successful_inhibitions: 0,
        }
    }

    /// Set the current task set (cognitive mode).
    pub fn set_task(&mut self, name: impl Into<String>) -> f64 {
        let name = name.into();

        // Check if we're switching from an existing task
        if let Some(current) = self.current_task_set.take() {
            if current.name != name {
                // Task switch! Incur switch cost
                self.current_switch_cost = self.config.switch_cost;
                self.cognitive_load = (self.cognitive_load + 0.2).min(1.0);
                self.total_switches += 1;
                self.task_set_history.push_back(current);
                if self.task_set_history.len() > 10 {
                    self.task_set_history.pop_front();
                }
            } else {
                // Same task, restore it
                self.current_task_set = Some(current);
                return 0.0; // No switch cost
            }
        }

        self.current_task_set = Some(TaskSet::new(name));
        self.current_switch_cost
    }

    /// Process one cycle: decay switch cost, update cognitive load.
    pub fn cycle(&mut self) {
        // Decay switch cost
        self.current_switch_cost =
            (self.current_switch_cost - self.config.switch_recovery_rate).max(0.0);

        // Decay cognitive load
        self.cognitive_load = (self.cognitive_load - 0.02).max(0.0);

        // Update current task set cycle count
        if let Some(task) = &mut self.current_task_set {
            task.total_cycles += 1;
        }
    }

    /// Record an observation that might lead to rule learning.
    pub fn observe_pattern(
        &mut self,
        condition: impl Into<String>,
        action: impl Into<String>,
        context: impl Into<String>,
    ) -> Option<CognitiveRule> {
        let condition = condition.into();
        let action = action.into();
        let context = context.into();

        let observations = self.rule_candidates.entry(condition.clone()).or_default();
        observations.push((action.clone(), context.clone()));

        // Check if we have enough consistent observations to form a rule
        if observations.len() >= self.config.rule_learning_threshold as usize {
            // Check consistency: are the actions mostly the same?
            let most_common_action = most_common(observations.iter().map(|(a, _)| a.as_str()));
            let consistency = observations
                .iter()
                .filter(|(a, _)| a == &most_common_action)
                .count() as f64
                / observations.len() as f64;

            if consistency >= 0.6 {
                let rule = CognitiveRule::new(
                    format!("When {}, then {}", condition, most_common_action),
                    condition.clone(),
                    most_common_action.clone(),
                )
                .with_context(context);

                let id = rule.id;
                self.rules.insert(id, rule.clone());

                // Add to current task set
                if let Some(task) = &mut self.current_task_set
                    && task.active_rules.len() < self.config.max_active_rules
                {
                    task.active_rules.push(id);
                }

                // Clear candidates for this condition
                self.rule_candidates.remove(&condition);

                return Some(rule);
            }
        }
        None
    }

    /// Record an outcome and check if set shifting is needed.
    pub fn record_outcome(
        &mut self,
        success: bool,
        context: impl Into<String>,
        rule_applied: Option<RuleId>,
    ) -> SetShiftResult {
        let task_set_id = self
            .current_task_set
            .as_ref()
            .map(|t| t.id)
            .unwrap_or_else(Uuid::new_v4);

        let outcome = Outcome {
            success,
            task_set_id,
            rule_applied,
            timestamp: Utc::now(),
            context: context.into(),
        };
        self.outcome_history.push_back(outcome);
        if self.outcome_history.len() > self.config.outcome_history_size {
            self.outcome_history.pop_front();
        }

        // Update task set stats
        if let Some(task) = &mut self.current_task_set {
            if success {
                task.successes += 1;
            } else {
                task.failures += 1;
            }
        }

        // Update rule confidence
        if let Some(rule_id) = rule_applied
            && let Some(rule) = self.rules.get_mut(&rule_id)
        {
            if success {
                rule.confirm();
            } else {
                rule.violate();
            }
        }

        // Check for perseveration (repeated failures without adaptation)
        self.evaluate_set_shift()
    }

    /// Evaluate whether the current strategy should be abandoned.
    fn evaluate_set_shift(&mut self) -> SetShiftResult {
        let consecutive_failures = self
            .outcome_history
            .iter()
            .rev()
            .take_while(|o| !o.success)
            .count() as u32;

        let current_success_rate = self
            .current_task_set
            .as_ref()
            .map(|t| t.success_rate())
            .unwrap_or(0.5);

        let should_shift = consecutive_failures >= self.config.perseveration_threshold;

        let suggestion = if should_shift {
            self.total_set_shifts += 1;
            self.cognitive_load = (self.cognitive_load + 0.3).min(1.0);

            // Find a previously successful strategy
            let best_previous = self
                .task_set_history
                .iter()
                .max_by(|a, b| {
                    a.success_rate()
                        .partial_cmp(&b.success_rate())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|t| t.name.clone());

            Some(best_previous.unwrap_or_else(|| "Try a novel approach".to_string()))
        } else {
            None
        };

        SetShiftResult {
            should_shift,
            consecutive_failures,
            current_success_rate,
            suggestion,
        }
    }

    /// Attempt to inhibit a prepotent response.
    pub fn inhibit_response(
        &mut self,
        response: impl Into<String>,
        reason: impl Into<String>,
        urgency: f64,
    ) -> InhibitionOutcome {
        let response = response.into();
        let reason = reason.into();
        self.total_inhibitions += 1;

        // Inhibition success depends on:
        // 1. Base inhibition strength
        // 2. Current cognitive load (harder when loaded)
        // 3. Urgency of the response (harder to inhibit urgent responses)
        let effective_strength = self.config.inhibition_strength
            * (1.0 - self.cognitive_load * 0.3)
            * (1.0 - urgency.clamp(0.0, 1.0) * 0.4);

        let inhibited = effective_strength > 0.4;

        if inhibited {
            self.successful_inhibitions += 1;
        }

        let cognitive_cost = if inhibited { 0.15 } else { 0.05 };
        self.cognitive_load = (self.cognitive_load + cognitive_cost).min(1.0);

        InhibitionOutcome {
            inhibited,
            response,
            reason,
            cognitive_cost,
        }
    }

    /// Find applicable rules for a given condition.
    pub fn find_rules(&self, condition: &str) -> Vec<&CognitiveRule> {
        self.rules
            .values()
            .filter(|r| r.condition == condition && r.confidence > 0.3)
            .collect()
    }

    /// Get the current switch cost (0 when fully recovered).
    pub fn switch_cost(&self) -> f64 {
        self.current_switch_cost
    }

    /// Get the current cognitive load.
    pub fn cognitive_load(&self) -> f64 {
        self.cognitive_load
    }

    /// Get the current task set name.
    pub fn current_task(&self) -> Option<&str> {
        self.current_task_set.as_ref().map(|t| t.name.as_str())
    }

    /// Perseveration score: tendency to stick with failing strategies (0=flexible, 1=rigid).
    pub fn perseveration_score(&self) -> f64 {
        let recent: Vec<_> = self.outcome_history.iter().rev().take(10).collect();
        if recent.is_empty() {
            return 0.0;
        }

        let failure_streak = recent.iter().take_while(|o| !o.success).count();
        let all_same_task = recent
            .iter()
            .all(|o| o.task_set_id == recent[0].task_set_id);

        let streak_score = failure_streak as f64 / recent.len() as f64;
        let rigidity_bonus = if all_same_task && failure_streak > 2 {
            0.3
        } else {
            0.0
        };

        (streak_score + rigidity_bonus).min(1.0)
    }

    /// Get statistics.
    pub fn stats(&self) -> DlpfcStats {
        DlpfcStats {
            total_task_switches: self.total_switches,
            total_rules_learned: self.rules.len(),
            total_set_shifts: self.total_set_shifts,
            total_inhibitions: self.total_inhibitions,
            successful_inhibitions: self.successful_inhibitions,
            current_task_set: self.current_task_set.as_ref().map(|t| t.name.clone()),
            current_switch_cost: self.current_switch_cost,
            cognitive_load: self.cognitive_load,
            perseveration_score: self.perseveration_score(),
        }
    }
}

impl Default for DLPFC {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper: find the most common string in an iterator.
fn most_common<'a>(items: impl Iterator<Item = &'a str>) -> String {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for item in items {
        *counts.entry(item).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .map(|(item, _)| item.to_string())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_creation() {
        let dlpfc = DLPFC::new();
        assert!(dlpfc.current_task().is_none());
        assert_eq!(dlpfc.cognitive_load(), 0.0);
    }

    #[test]
    fn test_task_switching_cost() {
        let mut dlpfc = DLPFC::new();

        // Set first task (no cost)
        let cost = dlpfc.set_task("reading");
        assert_eq!(cost, 0.0);

        // Switch task (should incur cost)
        let cost = dlpfc.set_task("writing");
        assert!(cost > 0.0, "Task switch should incur cost");
        assert_eq!(dlpfc.current_task(), Some("writing"));
    }

    #[test]
    fn test_same_task_no_cost() {
        let mut dlpfc = DLPFC::new();
        dlpfc.set_task("reading");
        let cost = dlpfc.set_task("reading");
        assert_eq!(cost, 0.0, "Same task should have no switch cost");
    }

    #[test]
    fn test_switch_cost_recovery() {
        let mut dlpfc = DLPFC::new();
        dlpfc.set_task("reading");
        dlpfc.set_task("writing");

        assert!(dlpfc.switch_cost() > 0.0);

        // Run cycles to recover
        for _ in 0..10 {
            dlpfc.cycle();
        }

        assert_eq!(dlpfc.switch_cost(), 0.0, "Switch cost should fully recover");
    }

    #[test]
    fn test_rule_learning() {
        let mut dlpfc = DLPFC::with_config(DlpfcConfig {
            rule_learning_threshold: 3,
            ..Default::default()
        });
        dlpfc.set_task("sorting");

        // Observe consistent pattern
        assert!(
            dlpfc
                .observe_pattern("red card", "place left", "round 1")
                .is_none()
        );
        assert!(
            dlpfc
                .observe_pattern("red card", "place left", "round 2")
                .is_none()
        );
        let rule = dlpfc.observe_pattern("red card", "place left", "round 3");

        assert!(
            rule.is_some(),
            "Should learn rule after threshold observations"
        );
        let rule = rule.unwrap();
        assert!(rule.description.contains("red card"));
        assert!(rule.description.contains("place left"));
    }

    #[test]
    fn test_rule_confirmation_and_violation() {
        let mut dlpfc = DLPFC::new();
        dlpfc.set_task("test");

        // Create a rule manually
        let rule = CognitiveRule::new("test rule", "condition", "action");
        let rule_id = rule.id;
        dlpfc.rules.insert(rule_id, rule);

        // Confirm the rule
        dlpfc.record_outcome(true, "context", Some(rule_id));
        assert!(dlpfc.rules[&rule_id].confidence > 0.3);

        // Violate the rule
        dlpfc.record_outcome(false, "context", Some(rule_id));
        let after_violation = dlpfc.rules[&rule_id].confidence;
        // Confidence should have decreased from the violation
        assert!(after_violation < 0.5);
    }

    #[test]
    fn test_set_shifting() {
        let mut dlpfc = DLPFC::with_config(DlpfcConfig {
            perseveration_threshold: 3,
            ..Default::default()
        });
        dlpfc.set_task("strategy_a");

        // Record consecutive failures
        let result = dlpfc.record_outcome(false, "trial 1", None);
        assert!(!result.should_shift);

        let result = dlpfc.record_outcome(false, "trial 2", None);
        assert!(!result.should_shift);

        let result = dlpfc.record_outcome(false, "trial 3", None);
        assert!(
            result.should_shift,
            "Should recommend shift after {} consecutive failures",
            result.consecutive_failures
        );
        assert!(result.suggestion.is_some());
    }

    #[test]
    fn test_inhibitory_control() {
        let mut dlpfc = DLPFC::new();

        // Inhibit with low urgency (should succeed)
        let result = dlpfc.inhibit_response("blurt out answer", "not my turn", 0.2);
        assert!(result.inhibited, "Should inhibit low-urgency response");

        // Inhibit with high urgency while cognitively loaded
        dlpfc.cognitive_load = 0.9;
        let result = dlpfc.inhibit_response("scream", "inappropriate", 0.9);
        assert!(
            !result.inhibited,
            "Should fail to inhibit high-urgency response under load"
        );
    }

    #[test]
    fn test_perseveration_score() {
        let mut dlpfc = DLPFC::new();
        dlpfc.set_task("rigid_strategy");

        // All failures in same task set = high perseveration
        for _ in 0..5 {
            dlpfc.record_outcome(false, "trial", None);
        }

        let score = dlpfc.perseveration_score();
        assert!(
            score > 0.5,
            "Perseveration score should be high after repeated failures: {}",
            score
        );
    }

    #[test]
    fn test_find_rules() {
        let mut dlpfc = DLPFC::new();

        let rule = CognitiveRule::new("test", "if_raining", "take_umbrella");
        let mut confirmed_rule = rule.clone();
        confirmed_rule.confidence = 0.8;
        dlpfc.rules.insert(confirmed_rule.id, confirmed_rule);

        let mut weak_rule = CognitiveRule::new("test2", "if_raining", "wear_boots");
        weak_rule.confidence = 0.1; // Below threshold
        dlpfc.rules.insert(weak_rule.id, weak_rule);

        let found = dlpfc.find_rules("if_raining");
        assert_eq!(found.len(), 1, "Should only find confident rules");
    }

    #[test]
    fn test_stats() {
        let mut dlpfc = DLPFC::new();
        dlpfc.set_task("a");
        dlpfc.set_task("b");
        dlpfc.inhibit_response("impulse", "reason", 0.3);

        let stats = dlpfc.stats();
        assert_eq!(stats.total_task_switches, 1);
        assert_eq!(stats.total_inhibitions, 1);
        assert_eq!(stats.current_task_set, Some("b".to_string()));
    }
}
