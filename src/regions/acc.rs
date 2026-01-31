//! Anterior Cingulate Cortex (ACC) - Error Detection and Conflict Monitoring
//!
//! The ACC is the brain's alarm system. It detects when things go wrong, monitors
//! for conflicts between competing processes, and signals when more control is needed.
//! Key functions:
//!
//! - **Error detection**: Recognizing when outcomes differ from expectations
//! - **Conflict monitoring**: Detecting competing responses or goals
//! - **Cognitive control**: Signaling when to engage more deliberate processing
//! - **Effort allocation**: Determining how much mental effort to invest
//!
//! # Computational Model
//!
//! The ACC maintains running estimates of:
//! - Expected vs actual outcomes (prediction errors)
//! - Response conflict (multiple active responses)
//! - Cognitive demand (need for control)
//!
//! When thresholds are exceeded, it triggers control signals to prefrontal cortex.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Types of errors the ACC can detect
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ErrorType {
    /// Outcome different from prediction
    PredictionError {
        expected: String,
        actual: String,
        magnitude: f64,
    },
    /// Action produced negative result
    ActionError { action: String, consequence: String },
    /// Goal cannot be achieved
    GoalFailure { goal: String, reason: String },
    /// Resource exhausted unexpectedly
    ResourceError {
        resource: String,
        expected: f64,
        actual: f64,
    },
    /// Communication/social error
    SocialError { context: String, violation: String },
    /// Internal inconsistency
    ConsistencyError { belief_a: String, belief_b: String },
}

/// A detected error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Error {
    pub id: uuid::Uuid,
    pub error_type: ErrorType,
    pub severity: f64, // 0-1, how serious
    pub detected_at: DateTime<Utc>,
    pub resolved: bool,
    pub resolution: Option<String>,
}

impl Error {
    pub fn new(error_type: ErrorType, severity: f64) -> Self {
        Self {
            id: uuid::Uuid::new_v4(),
            error_type,
            severity: severity.clamp(0.0, 1.0),
            detected_at: Utc::now(),
            resolved: false,
            resolution: None,
        }
    }

    pub fn resolve(&mut self, resolution: &str) {
        self.resolved = true;
        self.resolution = Some(resolution.to_string());
    }
}

/// Conflict between competing options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conflict {
    pub id: uuid::Uuid,
    /// What's competing
    pub options: Vec<String>,
    /// Strength of each option (should be similar for true conflict)
    pub strengths: Vec<f64>,
    /// How unresolved this conflict is
    pub intensity: f64,
    pub detected_at: DateTime<Utc>,
    pub resolved: bool,
}

impl Conflict {
    pub fn new(options: Vec<String>, strengths: Vec<f64>) -> Self {
        // Calculate conflict intensity based on similarity of strengths
        let max = strengths.iter().cloned().fold(0.0_f64, f64::max);
        let min = strengths.iter().cloned().fold(1.0_f64, f64::min);
        let intensity = 1.0 - (max - min); // High when strengths are similar

        Self {
            id: uuid::Uuid::new_v4(),
            options,
            strengths,
            intensity,
            detected_at: Utc::now(),
            resolved: false,
        }
    }

    pub fn resolve(&mut self) {
        self.resolved = true;
    }
}

/// Control signal sent to other brain regions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ControlSignal {
    /// Increase attention/focus
    IncreaseAttention { target: String, amount: f64 },
    /// Slow down, be more careful
    SlowDown { reason: String },
    /// Switch strategy
    SwitchStrategy { from: String, to: String },
    /// Allocate more effort
    IncreaseEffort { domain: String, amount: f64 },
    /// Disengage from current task
    Disengage { reason: String },
    /// Alert: something's wrong
    Alert { message: String, urgency: f64 },
}

/// Statistics for the ACC
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ACCStats {
    pub total_errors_detected: u64,
    pub total_conflicts_detected: u64,
    pub errors_resolved: u64,
    pub conflicts_resolved: u64,
    pub control_signals_sent: u64,
    pub average_error_severity: f64,
    pub average_conflict_intensity: f64,
    pub current_cognitive_load: f64,
}

/// The Anterior Cingulate Cortex
#[derive(Debug)]
pub struct ACC {
    /// Recent errors
    errors: VecDeque<Error>,
    /// Active conflicts
    conflicts: VecDeque<Conflict>,
    /// Pending control signals
    control_queue: VecDeque<ControlSignal>,
    /// Running estimate of prediction error rate
    error_rate: f64,
    /// Running estimate of conflict level
    conflict_level: f64,
    /// Current cognitive load estimate
    cognitive_load: f64,
    /// Threshold for sending control signals
    alert_threshold: f64,
    /// Statistics
    stats: ACCStats,
    /// Error history for learning (keeps last N)
    error_history: VecDeque<(ErrorType, bool)>, // (error, was_resolved)
}

impl ACC {
    pub fn new() -> Self {
        Self {
            errors: VecDeque::new(),
            conflicts: VecDeque::new(),
            control_queue: VecDeque::new(),
            error_rate: 0.0,
            conflict_level: 0.0,
            cognitive_load: 0.3,
            alert_threshold: 0.6,
            stats: ACCStats::default(),
            error_history: VecDeque::new(),
        }
    }

    /// Detect a prediction error
    pub fn prediction_error(&mut self, expected: &str, actual: &str, magnitude: f64) -> &Error {
        let error = Error::new(
            ErrorType::PredictionError {
                expected: expected.to_string(),
                actual: actual.to_string(),
                magnitude,
            },
            magnitude,
        );

        self.register_error(error)
    }

    /// Detect an action error
    pub fn action_error(&mut self, action: &str, consequence: &str, severity: f64) -> &Error {
        let error = Error::new(
            ErrorType::ActionError {
                action: action.to_string(),
                consequence: consequence.to_string(),
            },
            severity,
        );

        self.register_error(error)
    }

    /// Detect a goal failure
    pub fn goal_failure(&mut self, goal: &str, reason: &str, severity: f64) -> &Error {
        let error = Error::new(
            ErrorType::GoalFailure {
                goal: goal.to_string(),
                reason: reason.to_string(),
            },
            severity,
        );

        self.register_error(error)
    }

    /// Register an error and potentially trigger control signals
    fn register_error(&mut self, error: Error) -> &Error {
        self.stats.total_errors_detected += 1;

        // Update error rate (exponential moving average)
        self.error_rate = self.error_rate * 0.9 + error.severity * 0.1;

        // Update cognitive load
        self.cognitive_load = (self.cognitive_load + error.severity * 0.2).min(1.0);

        // Check if we should alert
        if error.severity > self.alert_threshold {
            self.control_queue.push_back(ControlSignal::Alert {
                message: format!("High-severity error: {:?}", error.error_type),
                urgency: error.severity,
            });
            self.stats.control_signals_sent += 1;
        }

        // Maybe slow down
        if self.error_rate > 0.5 {
            self.control_queue.push_back(ControlSignal::SlowDown {
                reason: "High error rate detected".to_string(),
            });
            self.stats.control_signals_sent += 1;
        }

        // Track in history
        self.error_history
            .push_back((error.error_type.clone(), false));
        if self.error_history.len() > 100 {
            self.error_history.pop_front();
        }

        // Update stats
        let total_severity: f64 = self.errors.iter().map(|e| e.severity).sum();
        self.stats.average_error_severity = if self.errors.is_empty() {
            error.severity
        } else {
            (total_severity + error.severity) / (self.errors.len() + 1) as f64
        };

        self.errors.push_back(error);

        // Keep bounded
        if self.errors.len() > 50 {
            self.errors.pop_front();
        }

        self.errors.back().unwrap()
    }

    /// Detect conflict between options
    pub fn detect_conflict(&mut self, options: Vec<String>, strengths: Vec<f64>) -> &Conflict {
        let conflict = Conflict::new(options, strengths);

        self.stats.total_conflicts_detected += 1;

        // Update conflict level
        self.conflict_level = self.conflict_level * 0.8 + conflict.intensity * 0.2;

        // Update cognitive load
        self.cognitive_load = (self.cognitive_load + conflict.intensity * 0.15).min(1.0);

        // High conflict triggers more attention
        if conflict.intensity > 0.7 {
            self.control_queue
                .push_back(ControlSignal::IncreaseAttention {
                    target: "decision".to_string(),
                    amount: conflict.intensity,
                });
            self.stats.control_signals_sent += 1;
        }

        // Update stats
        let total_intensity: f64 = self.conflicts.iter().map(|c| c.intensity).sum();
        self.stats.average_conflict_intensity = if self.conflicts.is_empty() {
            conflict.intensity
        } else {
            (total_intensity + conflict.intensity) / (self.conflicts.len() + 1) as f64
        };

        self.conflicts.push_back(conflict);

        if self.conflicts.len() > 20 {
            self.conflicts.pop_front();
        }

        self.conflicts.back().unwrap()
    }

    /// Resolve an error
    pub fn resolve_error(&mut self, error_id: uuid::Uuid, resolution: &str) {
        if let Some(error) = self.errors.iter_mut().find(|e| e.id == error_id) {
            error.resolve(resolution);
            self.stats.errors_resolved += 1;

            // Mark in history
            if let Some((_, resolved)) = self.error_history.back_mut() {
                *resolved = true;
            }

            // Decrease cognitive load
            self.cognitive_load = (self.cognitive_load - 0.1).max(0.0);
        }
    }

    /// Resolve a conflict
    pub fn resolve_conflict(&mut self, conflict_id: uuid::Uuid) {
        if let Some(conflict) = self.conflicts.iter_mut().find(|c| c.id == conflict_id) {
            conflict.resolve();
            self.stats.conflicts_resolved += 1;

            // Decrease conflict level
            self.conflict_level = (self.conflict_level - 0.2).max(0.0);
            self.cognitive_load = (self.cognitive_load - 0.1).max(0.0);
        }
    }

    /// Get next control signal (if any)
    pub fn next_control_signal(&mut self) -> Option<ControlSignal> {
        self.control_queue.pop_front()
    }

    /// Peek at pending control signals
    pub fn pending_signals(&self) -> &VecDeque<ControlSignal> {
        &self.control_queue
    }

    /// Get current cognitive load
    pub fn cognitive_load(&self) -> f64 {
        self.cognitive_load
    }

    /// Should we engage more effortful processing?
    pub fn needs_control(&self) -> bool {
        self.cognitive_load > 0.5 || self.error_rate > 0.3 || self.conflict_level > 0.5
    }

    /// Get effort recommendation for current state
    pub fn effort_recommendation(&self) -> f64 {
        // More load/errors/conflicts = more effort needed
        (self.cognitive_load * 0.4 + self.error_rate * 0.3 + self.conflict_level * 0.3).min(1.0)
    }

    /// Signal that things are going well (reduces load)
    pub fn success_signal(&mut self) {
        self.cognitive_load = (self.cognitive_load - 0.05).max(0.1);
        self.error_rate = (self.error_rate - 0.02).max(0.0);
    }

    /// Get unresolved errors
    pub fn unresolved_errors(&self) -> Vec<&Error> {
        self.errors.iter().filter(|e| !e.resolved).collect()
    }

    /// Get active conflicts
    pub fn active_conflicts(&self) -> Vec<&Conflict> {
        self.conflicts.iter().filter(|c| !c.resolved).collect()
    }

    /// Decay cognitive load over time (call periodically)
    pub fn decay(&mut self) {
        self.cognitive_load = (self.cognitive_load * 0.95).max(0.1);
        self.error_rate *= 0.98;
        self.conflict_level *= 0.98;
    }

    /// Get statistics
    pub fn stats(&self) -> ACCStats {
        let mut stats = self.stats.clone();
        stats.current_cognitive_load = self.cognitive_load;
        stats
    }

    /// Check for patterns in errors (same type recurring)
    pub fn recurring_errors(&self) -> Vec<(String, usize)> {
        let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

        for error in &self.errors {
            let key = match &error.error_type {
                ErrorType::PredictionError { .. } => "prediction",
                ErrorType::ActionError { action, .. } => action.as_str(),
                ErrorType::GoalFailure { goal, .. } => goal.as_str(),
                ErrorType::ResourceError { resource, .. } => resource.as_str(),
                ErrorType::SocialError { .. } => "social",
                ErrorType::ConsistencyError { .. } => "consistency",
            };
            *counts.entry(key.to_string()).or_insert(0) += 1;
        }

        let mut recurring: Vec<(String, usize)> =
            counts.into_iter().filter(|(_, c)| *c > 2).collect();
        recurring.sort_by(|a, b| b.1.cmp(&a.1));
        recurring
    }
}

impl Default for ACC {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_detection() {
        let mut acc = ACC::new();

        let error = acc.prediction_error("sunny", "rainy", 0.7);
        assert!(!error.resolved);
        assert!(error.severity > 0.5);
        assert_eq!(acc.stats().total_errors_detected, 1);
    }

    #[test]
    fn test_control_signals_on_high_severity() {
        let mut acc = ACC::new();
        acc.alert_threshold = 0.5;

        acc.action_error("deploy", "system crashed", 0.9);

        // Should have generated an alert
        let signal = acc.next_control_signal();
        assert!(matches!(signal, Some(ControlSignal::Alert { .. })));
    }

    #[test]
    fn test_conflict_detection() {
        let mut acc = ACC::new();

        // Two options with similar strength = high conflict
        let conflict = acc.detect_conflict(
            vec!["option_a".to_string(), "option_b".to_string()],
            vec![0.7, 0.68],
        );

        assert!(
            conflict.intensity > 0.9,
            "Similar strengths should create high conflict"
        );
    }

    #[test]
    fn test_cognitive_load_management() {
        let mut acc = ACC::new();
        let initial = acc.cognitive_load();

        // Errors increase load
        acc.action_error("test", "failed", 0.5);
        assert!(acc.cognitive_load() > initial);

        // Success decreases load
        let after_error = acc.cognitive_load();
        acc.success_signal();
        assert!(acc.cognitive_load() < after_error);
    }

    #[test]
    fn test_error_resolution() {
        let mut acc = ACC::new();

        let error = acc.action_error("test", "failed", 0.5);
        let error_id = error.id;

        acc.resolve_error(error_id, "fixed the bug");

        let unresolved = acc.unresolved_errors();
        assert!(unresolved.is_empty());
        assert_eq!(acc.stats().errors_resolved, 1);
    }

    #[test]
    fn test_needs_control() {
        let mut acc = ACC::new();
        assert!(!acc.needs_control(), "Fresh ACC shouldn't need control");

        // Add several errors
        for _ in 0..10 {
            acc.action_error("test", "failed", 0.6);
        }

        assert!(
            acc.needs_control(),
            "High error rate should trigger control need"
        );
    }
}
