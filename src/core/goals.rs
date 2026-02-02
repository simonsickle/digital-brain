//! Goal Management System
//!
//! The prefrontal cortex has a simple goal stack, but proper agents need:
//! - Goal decomposition into subgoals
//! - Priority-based selection
//! - Progress tracking
//! - Deadline awareness
//! - Conflict detection between goals
//!
//! This module integrates with neuromodulators:
//! - Goal completion triggers Achievement reward
//! - Blocked goals increase stress/cortisol
//! - Progress triggers anticipation (dopamine)
//! - Long-term goals require high patience (serotonin)

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use crate::core::neuromodulators::NeuromodulatorState;
use crate::core::strategy::StrategyProfile;

/// Unique identifier for a goal
pub type GoalId = Uuid;

/// Status of a goal
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GoalStatus {
    /// Goal is active and being pursued
    Active,
    /// Goal is blocked by some condition
    Blocked { reason: String },
    /// Goal has been achieved
    Completed { completed_at: DateTime<Utc> },
    /// Goal was abandoned
    Abandoned {
        reason: String,
        abandoned_at: DateTime<Utc>,
    },
    /// Goal is paused (not actively pursued but not abandoned)
    Paused { reason: String },
}

impl GoalStatus {
    pub fn is_active(&self) -> bool {
        matches!(self, GoalStatus::Active)
    }

    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            GoalStatus::Completed { .. } | GoalStatus::Abandoned { .. }
        )
    }
}

/// Priority level for goals
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Serialize, Deserialize)]
pub enum Priority {
    /// Background/nice-to-have
    Low = 1,
    /// Normal priority
    #[default]
    Medium = 2,
    /// Important
    High = 3,
    /// Urgent/critical
    Critical = 4,
}

impl Priority {
    pub fn as_f64(&self) -> f64 {
        match self {
            Priority::Low => 0.25,
            Priority::Medium => 0.5,
            Priority::High => 0.75,
            Priority::Critical => 1.0,
        }
    }
}

/// Criterion for success
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Criterion {
    /// Description of what needs to be true
    pub description: String,
    /// Is this criterion satisfied?
    pub satisfied: bool,
    /// Weight of this criterion (0 to 1)
    pub weight: f64,
}

impl Criterion {
    pub fn new(description: &str) -> Self {
        Self {
            description: description.to_string(),
            satisfied: false,
            weight: 1.0,
        }
    }

    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight.clamp(0.0, 1.0);
        self
    }

    pub fn satisfy(&mut self) {
        self.satisfied = true;
    }
}

/// A goal with full metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Goal {
    /// Unique identifier
    pub id: GoalId,
    /// Human-readable description
    pub description: String,
    /// Priority level
    pub priority: Priority,
    /// Optional deadline
    pub deadline: Option<DateTime<Utc>>,
    /// Criteria for success
    pub success_criteria: Vec<Criterion>,
    /// Parent goal (for hierarchical decomposition)
    pub parent: Option<GoalId>,
    /// Current status
    pub status: GoalStatus,
    /// Progress (0.0 to 1.0)
    pub progress: f64,
    /// When the goal was created
    pub created_at: DateTime<Utc>,
    /// Tags for categorization
    pub tags: Vec<String>,
    /// Estimated effort (0 to 1, where 1 is maximum effort)
    pub estimated_effort: f64,
    /// Time horizon (how long-term is this goal?)
    pub time_horizon: TimeHorizon,
    /// Notes/context
    pub notes: String,
}

/// Time horizon for goal planning
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum TimeHorizon {
    /// Immediate (< 1 hour)
    Immediate,
    /// Short-term (hours to days)
    #[default]
    ShortTerm,
    /// Medium-term (weeks)
    MediumTerm,
    /// Long-term (months+)
    LongTerm,
}

impl TimeHorizon {
    /// Minimum patience required to pursue this time horizon
    pub fn required_patience(&self) -> f64 {
        match self {
            TimeHorizon::Immediate => 0.0,
            TimeHorizon::ShortTerm => 0.3,
            TimeHorizon::MediumTerm => 0.5,
            TimeHorizon::LongTerm => 0.7,
        }
    }
}

impl Default for Goal {
    fn default() -> Self {
        Self {
            id: Uuid::new_v4(),
            description: String::new(),
            priority: Priority::default(),
            deadline: None,
            success_criteria: Vec::new(),
            parent: None,
            status: GoalStatus::Active,
            progress: 0.0,
            created_at: Utc::now(),
            tags: Vec::new(),
            estimated_effort: 0.5,
            time_horizon: TimeHorizon::default(),
            notes: String::new(),
        }
    }
}

impl Goal {
    /// Create a new goal with description
    pub fn new(description: &str) -> Self {
        Self {
            description: description.to_string(),
            ..Default::default()
        }
    }

    /// Set priority
    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }

    /// Set deadline
    pub fn with_deadline(mut self, deadline: DateTime<Utc>) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// Add success criterion
    pub fn with_criterion(mut self, criterion: Criterion) -> Self {
        self.success_criteria.push(criterion);
        self
    }

    /// Set parent goal
    pub fn with_parent(mut self, parent: GoalId) -> Self {
        self.parent = Some(parent);
        self
    }

    /// Set time horizon
    pub fn with_horizon(mut self, horizon: TimeHorizon) -> Self {
        self.time_horizon = horizon;
        self
    }

    /// Set estimated effort
    pub fn with_effort(mut self, effort: f64) -> Self {
        self.estimated_effort = effort.clamp(0.0, 1.0);
        self
    }

    /// Add tag
    pub fn with_tag(mut self, tag: &str) -> Self {
        self.tags.push(tag.to_string());
        self
    }

    /// Calculate progress from criteria
    pub fn calculate_progress(&self) -> f64 {
        if self.success_criteria.is_empty() {
            return self.progress;
        }

        let total_weight: f64 = self.success_criteria.iter().map(|c| c.weight).sum();
        if total_weight == 0.0 {
            return 0.0;
        }

        let satisfied_weight: f64 = self
            .success_criteria
            .iter()
            .filter(|c| c.satisfied)
            .map(|c| c.weight)
            .sum();

        satisfied_weight / total_weight
    }

    /// Check if all criteria are satisfied
    pub fn is_complete(&self) -> bool {
        if self.success_criteria.is_empty() {
            return self.progress >= 1.0;
        }
        self.success_criteria.iter().all(|c| c.satisfied)
    }

    /// Check if deadline has passed
    pub fn is_overdue(&self) -> bool {
        self.deadline.map(|d| Utc::now() > d).unwrap_or(false)
    }

    /// Get urgency based on deadline proximity
    pub fn urgency(&self) -> f64 {
        match self.deadline {
            None => self.priority.as_f64(),
            Some(deadline) => {
                let now = Utc::now();
                if now >= deadline {
                    1.0 // Overdue = maximum urgency
                } else {
                    let time_remaining = (deadline - now).num_seconds() as f64;
                    let time_since_creation = (now - self.created_at).num_seconds() as f64;
                    let total_time = time_since_creation + time_remaining;

                    if total_time <= 0.0 {
                        return self.priority.as_f64();
                    }

                    // Urgency increases as deadline approaches
                    let time_pressure = 1.0 - (time_remaining / total_time);
                    (self.priority.as_f64() + time_pressure) / 2.0
                }
            }
        }
    }
}

/// Statistics about goal management
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GoalStats {
    pub total_goals: u64,
    pub active_goals: u64,
    pub completed_goals: u64,
    pub abandoned_goals: u64,
    pub blocked_goals: u64,
    pub average_completion_time_secs: f64,
    pub completion_rate: f64,
}

/// Event emitted when goal state changes
#[derive(Debug, Clone)]
pub enum GoalEvent {
    Created {
        goal_id: GoalId,
    },
    ProgressUpdated {
        goal_id: GoalId,
        old_progress: f64,
        new_progress: f64,
    },
    Completed {
        goal_id: GoalId,
    },
    Blocked {
        goal_id: GoalId,
        reason: String,
    },
    Unblocked {
        goal_id: GoalId,
    },
    Abandoned {
        goal_id: GoalId,
        reason: String,
    },
    DeadlineApproaching {
        goal_id: GoalId,
        hours_remaining: f64,
    },
}

/// The goal manager
#[derive(Debug)]
pub struct GoalManager {
    /// All goals
    goals: HashMap<GoalId, Goal>,
    /// Parent → children mapping
    goal_hierarchy: HashMap<GoalId, Vec<GoalId>>,
    /// Recent events
    events: Vec<GoalEvent>,
    /// Maximum events to keep
    max_events: usize,
    /// Statistics
    stats: GoalStats,
    /// Current strategic bias (long-horizon vs recovery)
    strategy_profile: StrategyProfile,
}

impl Default for GoalManager {
    fn default() -> Self {
        Self::new()
    }
}

impl GoalManager {
    /// Create a new goal manager
    pub fn new() -> Self {
        Self {
            goals: HashMap::new(),
            goal_hierarchy: HashMap::new(),
            events: Vec::new(),
            max_events: 100,
            stats: GoalStats::default(),
            strategy_profile: StrategyProfile::default(),
        }
    }

    /// Add a goal
    pub fn add(&mut self, goal: Goal) -> GoalId {
        let id = goal.id;

        // Track hierarchy
        if let Some(parent) = goal.parent {
            self.goal_hierarchy.entry(parent).or_default().push(id);
        }

        self.goals.insert(id, goal);
        self.stats.total_goals += 1;
        self.stats.active_goals += 1;

        self.emit_event(GoalEvent::Created { goal_id: id });

        id
    }

    /// Get a goal by ID
    pub fn get(&self, id: GoalId) -> Option<&Goal> {
        self.goals.get(&id)
    }

    /// Get a mutable reference to a goal
    pub fn get_mut(&mut self, id: GoalId) -> Option<&mut Goal> {
        self.goals.get_mut(&id)
    }

    /// Get all active goals
    pub fn active_goals(&self) -> Vec<&Goal> {
        self.goals
            .values()
            .filter(|g| g.status.is_active())
            .collect()
    }

    /// Get highest priority actionable goal based on neuromodulator state
    pub fn get_active_goal(&self, state: &NeuromodulatorState) -> Option<&Goal> {
        let mut candidates: Vec<_> = self
            .goals
            .values()
            .filter(|g| g.status.is_active())
            .filter(|g| !self.should_defer(g.id, state))
            .collect();

        if candidates.is_empty() {
            return None;
        }

        // Sort by effective priority (urgency + priority + stress boost for blocked subgoals)
        candidates.sort_by(|a, b| {
            let score_a = self.score_goal(a, state);
            let score_b = self.score_goal(b, state);
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        candidates.first().copied()
    }

    /// Score a goal for selection
    fn score_goal(&self, goal: &Goal, state: &NeuromodulatorState) -> f64 {
        let base_score = goal.urgency();

        // Motivation boost
        let motivation_bonus = state.motivation * 0.2;

        // Stress penalty for low-priority goals
        let stress_penalty = if goal.priority < Priority::High {
            state.stress * 0.1
        } else {
            0.0
        };

        // Progress momentum (near-complete goals get bonus)
        let momentum = if goal.progress > 0.7 { 0.15 } else { 0.0 };

        // Subgoal bonus (if parent is high priority)
        let subgoal_bonus = goal
            .parent
            .and_then(|p| self.goals.get(&p))
            .map(|parent| parent.priority.as_f64() * 0.1)
            .unwrap_or(0.0);

        let horizon_bias = match goal.time_horizon {
            TimeHorizon::Immediate => -0.2,
            TimeHorizon::ShortTerm => -0.1,
            TimeHorizon::MediumTerm => 0.1,
            TimeHorizon::LongTerm => 0.2,
        };
        let long_horizon_adjust = horizon_bias * self.strategy_profile.long_horizon_bias;

        let recovery_penalty =
            self.strategy_profile.recovery_priority * goal.estimated_effort * 0.3;

        base_score + motivation_bonus + momentum + subgoal_bonus + long_horizon_adjust
            - stress_penalty
            - recovery_penalty
    }

    /// Check if a goal should be deferred based on patience
    pub fn should_defer(&self, goal_id: GoalId, state: &NeuromodulatorState) -> bool {
        let Some(goal) = self.goals.get(&goal_id) else {
            return false;
        };

        // Check if patience is sufficient for this time horizon
        if state.patience < goal.time_horizon.required_patience() {
            return true;
        }

        // High effort + low motivation = defer
        if goal.estimated_effort > 0.7 && state.motivation < 0.3 {
            return true;
        }

        // Recovery priority defers long-horizon, high-effort goals
        if self.strategy_profile.recovery_priority > 0.6
            && (goal.time_horizon == TimeHorizon::LongTerm || goal.estimated_effort > 0.6)
        {
            return true;
        }

        false
    }

    /// Apply a new strategy profile for goal scoring.
    pub fn apply_strategy_profile(&mut self, profile: StrategyProfile) {
        self.strategy_profile = profile;
    }

    /// Get current strategy profile.
    pub fn strategy_profile(&self) -> &StrategyProfile {
        &self.strategy_profile
    }

    /// Decompose a goal into subgoals
    pub fn decompose(&mut self, goal_id: GoalId, subgoals: Vec<Goal>) -> Vec<GoalId> {
        let mut ids = Vec::new();

        for mut subgoal in subgoals {
            subgoal.parent = Some(goal_id);
            let id = self.add(subgoal);
            ids.push(id);
        }

        ids
    }

    /// Update progress on a goal
    pub fn update_progress(&mut self, goal_id: GoalId, progress: f64) -> Option<GoalEvent> {
        let (old_progress, new_progress, should_complete) = {
            let goal = self.goals.get_mut(&goal_id)?;
            let old = goal.progress;
            goal.progress = progress.clamp(0.0, 1.0);
            let complete = goal.is_complete() && goal.status.is_active();
            (old, goal.progress, complete)
        };

        let event = GoalEvent::ProgressUpdated {
            goal_id,
            old_progress,
            new_progress,
        };
        self.emit_event(event.clone());

        // Check for completion
        if should_complete {
            self.complete_goal(goal_id);
        }

        Some(event)
    }

    /// Mark a criterion as satisfied
    pub fn satisfy_criterion(&mut self, goal_id: GoalId, criterion_index: usize) -> bool {
        let Some(goal) = self.goals.get_mut(&goal_id) else {
            return false;
        };

        if criterion_index >= goal.success_criteria.len() {
            return false;
        }

        goal.success_criteria[criterion_index].satisfy();
        goal.progress = goal.calculate_progress();

        // Check for completion
        if goal.is_complete() && goal.status.is_active() {
            self.complete_goal(goal_id);
        }

        true
    }

    /// Complete a goal
    pub fn complete_goal(&mut self, goal_id: GoalId) -> bool {
        let parent_id = {
            let Some(goal) = self.goals.get_mut(&goal_id) else {
                return false;
            };

            if goal.status.is_terminal() {
                return false;
            }

            goal.status = GoalStatus::Completed {
                completed_at: Utc::now(),
            };
            goal.progress = 1.0;

            goal.parent
        };

        self.stats.active_goals = self.stats.active_goals.saturating_sub(1);
        self.stats.completed_goals += 1;

        // Update completion rate
        let total_terminal = self.stats.completed_goals + self.stats.abandoned_goals;
        if total_terminal > 0 {
            self.stats.completion_rate = self.stats.completed_goals as f64 / total_terminal as f64;
        }

        self.emit_event(GoalEvent::Completed { goal_id });

        // Update parent progress if exists
        if let Some(parent_id) = parent_id {
            self.update_parent_progress(parent_id);
        }

        true
    }

    /// Block a goal
    pub fn block_goal(&mut self, goal_id: GoalId, reason: &str) -> bool {
        let Some(goal) = self.goals.get_mut(&goal_id) else {
            return false;
        };

        if !goal.status.is_active() {
            return false;
        }

        goal.status = GoalStatus::Blocked {
            reason: reason.to_string(),
        };

        self.stats.active_goals = self.stats.active_goals.saturating_sub(1);
        self.stats.blocked_goals += 1;

        self.emit_event(GoalEvent::Blocked {
            goal_id,
            reason: reason.to_string(),
        });

        true
    }

    /// Unblock a goal
    pub fn unblock_goal(&mut self, goal_id: GoalId) -> bool {
        let Some(goal) = self.goals.get_mut(&goal_id) else {
            return false;
        };

        if !matches!(goal.status, GoalStatus::Blocked { .. }) {
            return false;
        }

        goal.status = GoalStatus::Active;

        self.stats.blocked_goals = self.stats.blocked_goals.saturating_sub(1);
        self.stats.active_goals += 1;

        self.emit_event(GoalEvent::Unblocked { goal_id });

        true
    }

    /// Abandon a goal
    pub fn abandon_goal(&mut self, goal_id: GoalId, reason: &str) -> bool {
        let Some(goal) = self.goals.get_mut(&goal_id) else {
            return false;
        };

        if goal.status.is_terminal() {
            return false;
        }

        let was_active = goal.status.is_active();
        goal.status = GoalStatus::Abandoned {
            reason: reason.to_string(),
            abandoned_at: Utc::now(),
        };

        if was_active {
            self.stats.active_goals = self.stats.active_goals.saturating_sub(1);
        }
        self.stats.abandoned_goals += 1;

        // Update completion rate
        let total_terminal = self.stats.completed_goals + self.stats.abandoned_goals;
        if total_terminal > 0 {
            self.stats.completion_rate = self.stats.completed_goals as f64 / total_terminal as f64;
        }

        self.emit_event(GoalEvent::Abandoned {
            goal_id,
            reason: reason.to_string(),
        });

        // Also abandon subgoals
        if let Some(children) = self.goal_hierarchy.get(&goal_id).cloned() {
            for child_id in children {
                self.abandon_goal(child_id, "Parent goal abandoned");
            }
        }

        true
    }

    /// Update parent goal progress based on children
    fn update_parent_progress(&mut self, parent_id: GoalId) {
        let children = match self.goal_hierarchy.get(&parent_id) {
            Some(c) => c.clone(),
            None => return,
        };

        if children.is_empty() {
            return;
        }

        let total_progress: f64 = children
            .iter()
            .filter_map(|id| self.goals.get(id))
            .map(|g| g.progress)
            .sum();

        let avg_progress = total_progress / children.len() as f64;

        let should_complete = {
            if let Some(parent) = self.goals.get_mut(&parent_id) {
                parent.progress = avg_progress;
                parent.is_complete() && parent.status.is_active()
            } else {
                false
            }
        };

        if should_complete {
            // Will recursively update parent's parent
            self.complete_goal(parent_id);
        }
    }

    /// Check for approaching deadlines
    pub fn check_deadlines(&mut self) -> Vec<GoalEvent> {
        let mut events = Vec::new();
        let now = Utc::now();

        for goal in self.goals.values() {
            if !goal.status.is_active() {
                continue;
            }

            if let Some(deadline) = goal.deadline {
                let hours_remaining = (deadline - now).num_hours() as f64;

                // Alert for goals due within 24 hours
                if hours_remaining > 0.0 && hours_remaining <= 24.0 {
                    events.push(GoalEvent::DeadlineApproaching {
                        goal_id: goal.id,
                        hours_remaining,
                    });
                }
            }
        }

        for event in &events {
            self.emit_event(event.clone());
        }

        events
    }

    /// Get subgoals of a goal
    pub fn subgoals(&self, goal_id: GoalId) -> Vec<&Goal> {
        self.goal_hierarchy
            .get(&goal_id)
            .map(|children| {
                children
                    .iter()
                    .filter_map(|id| self.goals.get(id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get goals by tag
    pub fn goals_by_tag(&self, tag: &str) -> Vec<&Goal> {
        self.goals
            .values()
            .filter(|g| g.tags.iter().any(|t| t == tag))
            .collect()
    }

    /// Get goals by status
    pub fn goals_by_status(&self, status_check: impl Fn(&GoalStatus) -> bool) -> Vec<&Goal> {
        self.goals
            .values()
            .filter(|g| status_check(&g.status))
            .collect()
    }

    /// Get statistics
    pub fn stats(&self) -> &GoalStats {
        &self.stats
    }

    /// Get recent events
    pub fn events(&self) -> &[GoalEvent] {
        &self.events
    }

    /// Clear events
    pub fn clear_events(&mut self) {
        self.events.clear();
    }

    /// Emit an event
    fn emit_event(&mut self, event: GoalEvent) {
        self.events.push(event);
        if self.events.len() > self.max_events {
            self.events.remove(0);
        }
    }

    /// Get all goals
    pub fn all_goals(&self) -> impl Iterator<Item = &Goal> {
        self.goals.values()
    }

    /// Remove completed goals older than a duration
    pub fn cleanup_old_goals(&mut self, max_age_days: i64) {
        let cutoff = Utc::now() - chrono::Duration::days(max_age_days);

        let to_remove: Vec<GoalId> = self
            .goals
            .iter()
            .filter(|(_, g)| match &g.status {
                GoalStatus::Completed { completed_at } => *completed_at < cutoff,
                GoalStatus::Abandoned { abandoned_at, .. } => *abandoned_at < cutoff,
                _ => false,
            })
            .map(|(id, _)| *id)
            .collect();

        for id in to_remove {
            self.goals.remove(&id);
            // Also clean up hierarchy references
            for children in self.goal_hierarchy.values_mut() {
                children.retain(|c| *c != id);
            }
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn test_goal_creation() {
        let mut manager = GoalManager::new();
        let goal = Goal::new("Test goal").with_priority(Priority::High);
        let id = manager.add(goal);

        assert!(manager.get(id).is_some());
        assert_eq!(manager.stats().total_goals, 1);
        assert_eq!(manager.stats().active_goals, 1);
    }

    #[test]
    fn test_goal_completion() {
        let mut manager = GoalManager::new();
        let goal = Goal::new("Test goal");
        let id = manager.add(goal);

        manager.update_progress(id, 1.0);

        let goal = manager.get(id).unwrap();
        assert!(matches!(goal.status, GoalStatus::Completed { .. }));
        assert_eq!(manager.stats().completed_goals, 1);
    }

    #[test]
    fn test_criterion_based_completion() {
        let mut manager = GoalManager::new();
        let goal = Goal::new("Test goal")
            .with_criterion(Criterion::new("Step 1"))
            .with_criterion(Criterion::new("Step 2"));
        let id = manager.add(goal);

        // Complete first criterion
        manager.satisfy_criterion(id, 0);
        let goal = manager.get(id).unwrap();
        assert_eq!(goal.progress, 0.5);
        assert!(goal.status.is_active());

        // Complete second criterion
        manager.satisfy_criterion(id, 1);
        let goal = manager.get(id).unwrap();
        assert_eq!(goal.progress, 1.0);
        assert!(matches!(goal.status, GoalStatus::Completed { .. }));
    }

    #[test]
    fn test_goal_hierarchy() {
        let mut manager = GoalManager::new();
        let parent = Goal::new("Parent goal");
        let parent_id = manager.add(parent);

        let child1 = Goal::new("Child 1").with_parent(parent_id);
        let child2 = Goal::new("Child 2").with_parent(parent_id);
        manager.add(child1);
        manager.add(child2);

        let subgoals = manager.subgoals(parent_id);
        assert_eq!(subgoals.len(), 2);
    }

    #[test]
    fn test_decompose() {
        let mut manager = GoalManager::new();
        let parent = Goal::new("Parent");
        let parent_id = manager.add(parent);

        let subgoals = vec![Goal::new("Sub 1"), Goal::new("Sub 2"), Goal::new("Sub 3")];

        let ids = manager.decompose(parent_id, subgoals);
        assert_eq!(ids.len(), 3);

        for id in ids {
            let goal = manager.get(id).unwrap();
            assert_eq!(goal.parent, Some(parent_id));
        }
    }

    #[test]
    fn test_parent_progress_from_children() {
        let mut manager = GoalManager::new();
        let parent = Goal::new("Parent");
        let parent_id = manager.add(parent);

        let subgoals = vec![Goal::new("Sub 1"), Goal::new("Sub 2")];
        let ids = manager.decompose(parent_id, subgoals);

        // Complete first child
        manager.complete_goal(ids[0]);

        let parent = manager.get(parent_id).unwrap();
        assert_eq!(parent.progress, 0.5);

        // Complete second child
        manager.complete_goal(ids[1]);

        let parent = manager.get(parent_id).unwrap();
        assert!(matches!(parent.status, GoalStatus::Completed { .. }));
    }

    #[test]
    fn test_blocking() {
        let mut manager = GoalManager::new();
        let goal = Goal::new("Test");
        let id = manager.add(goal);

        manager.block_goal(id, "Waiting for dependency");

        let goal = manager.get(id).unwrap();
        assert!(matches!(goal.status, GoalStatus::Blocked { .. }));
        assert_eq!(manager.stats().blocked_goals, 1);

        manager.unblock_goal(id);

        let goal = manager.get(id).unwrap();
        assert!(goal.status.is_active());
        assert_eq!(manager.stats().blocked_goals, 0);
    }

    #[test]
    fn test_abandonment_cascades() {
        let mut manager = GoalManager::new();
        let parent = Goal::new("Parent");
        let parent_id = manager.add(parent);

        let subgoals = vec![Goal::new("Sub 1"), Goal::new("Sub 2")];
        let ids = manager.decompose(parent_id, subgoals);

        // Abandon parent
        manager.abandon_goal(parent_id, "No longer relevant");

        // Children should also be abandoned
        for id in ids {
            let goal = manager.get(id).unwrap();
            assert!(matches!(goal.status, GoalStatus::Abandoned { .. }));
        }
    }

    #[test]
    fn test_get_active_goal() {
        let mut manager = GoalManager::new();
        let state = make_test_state();

        let low = Goal::new("Low priority").with_priority(Priority::Low);
        let high = Goal::new("High priority").with_priority(Priority::High);

        manager.add(low);
        let high_id = manager.add(high);

        let active = manager.get_active_goal(&state).unwrap();
        assert_eq!(active.id, high_id);
    }

    #[test]
    fn test_strategy_profile_biases_long_horizon() {
        let mut manager = GoalManager::new();
        let mut state = make_test_state();
        state.patience = 0.9;

        let long = Goal::new("Long horizon goal")
            .with_priority(Priority::Medium)
            .with_horizon(TimeHorizon::LongTerm);
        let short = Goal::new("Immediate goal")
            .with_priority(Priority::Medium)
            .with_horizon(TimeHorizon::Immediate);

        let long_id = manager.add(long);
        manager.add(short);

        manager.apply_strategy_profile(StrategyProfile {
            long_horizon_bias: 0.8,
            recovery_priority: 0.0,
            sleep_quality: 0.8,
            mood_stability: 0.8,
            stress: 0.1,
        });

        let active = manager.get_active_goal(&state).unwrap();
        assert_eq!(active.id, long_id);
    }

    #[test]
    fn test_defer_long_term_with_low_patience() {
        let mut manager = GoalManager::new();
        let mut state = make_test_state();
        state.patience = 0.2; // Low patience

        let long_term = Goal::new("Long term goal").with_horizon(TimeHorizon::LongTerm);
        let id = manager.add(long_term);

        assert!(manager.should_defer(id, &state));
    }

    #[test]
    fn test_urgency_increases_near_deadline() {
        // Create goals with different deadlines
        let goal_no_deadline = Goal::new("No deadline");

        // Far deadline: created now, deadline in 30 days
        let mut goal_far_deadline = Goal::new("Far deadline");
        goal_far_deadline.deadline = Some(Utc::now() + chrono::Duration::days(30));
        goal_far_deadline.created_at = Utc::now() - chrono::Duration::hours(1); // Created 1 hour ago

        // Near deadline: created yesterday, deadline in 2 hours
        let mut goal_near_deadline = Goal::new("Near deadline");
        goal_near_deadline.deadline = Some(Utc::now() + chrono::Duration::hours(2));
        goal_near_deadline.created_at = Utc::now() - chrono::Duration::days(1); // Created 1 day ago

        let near_urgency = goal_near_deadline.urgency();
        let far_urgency = goal_far_deadline.urgency();
        let no_deadline_urgency = goal_no_deadline.urgency();

        // Near deadline should have higher urgency
        assert!(
            near_urgency > far_urgency,
            "Near urgency ({}) should be > far urgency ({})",
            near_urgency,
            far_urgency
        );
        assert!(
            near_urgency > no_deadline_urgency,
            "Near urgency ({}) should be > no deadline urgency ({})",
            near_urgency,
            no_deadline_urgency
        );
    }

    #[test]
    fn test_events_emitted() {
        let mut manager = GoalManager::new();
        let goal = Goal::new("Test");
        let id = manager.add(goal);

        assert_eq!(manager.events().len(), 1);
        assert!(matches!(manager.events()[0], GoalEvent::Created { .. }));

        manager.update_progress(id, 0.5);
        assert_eq!(manager.events().len(), 2);
        assert!(matches!(
            manager.events()[1],
            GoalEvent::ProgressUpdated { .. }
        ));
    }

    #[test]
    fn test_completion_rate() {
        let mut manager = GoalManager::new();

        let g1 = manager.add(Goal::new("Goal 1"));
        let g2 = manager.add(Goal::new("Goal 2"));
        let g3 = manager.add(Goal::new("Goal 3"));

        manager.complete_goal(g1);
        manager.complete_goal(g2);
        manager.abandon_goal(g3, "Not needed");

        // 2 completed out of 3 terminal
        assert!((manager.stats().completion_rate - 0.666).abs() < 0.01);
    }
}
