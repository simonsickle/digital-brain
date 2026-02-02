//! Cerebellum - Procedural Memory, Timing, and Motor Learning
//!
//! The cerebellum handles automatic, learned skills and precise timing. Unlike
//! declarative memory (hippocampus), procedural memory is implicit - you can do
//! it without consciously knowing how. Key functions:
//!
//! - **Procedural memory**: Storing learned skills and procedures
//! - **Timing**: Precise temporal predictions and sequencing
//! - **Error correction**: Fine-tuning actions based on feedback
//! - **Automatization**: Making learned skills effortless
//!
//! # Computational Model
//!
//! Based on adaptive filter theory:
//! - Learns input-output mappings through practice
//! - Predicts outcomes and adjusts based on errors
//! - Gradually removes need for conscious control

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use crate::core::imagination::{ImaginationType, Imagining};
use crate::regions::motor_cortex::MotorCommand;

/// A learned procedure/skill
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Procedure {
    pub id: Uuid,
    /// Name of the skill
    pub name: String,
    /// Steps in the procedure
    pub steps: Vec<ProcedureStep>,
    /// How well-learned this procedure is (0 = novice, 1 = automatic)
    pub skill_level: f64,
    /// Total practice time (arbitrary units)
    pub practice_amount: f64,
    /// Error rate during execution
    pub error_rate: f64,
    /// Average execution time
    pub avg_execution_time: f64,
    /// When this was last practiced
    pub last_practiced: Option<DateTime<Utc>>,
    /// How many times executed successfully
    pub success_count: u64,
    /// How many times executed with errors
    pub error_count: u64,
}

/// A single step in a procedure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcedureStep {
    /// What to do
    pub action: String,
    /// Expected duration (for timing)
    pub expected_duration_ms: u64,
    /// Conditions that must be met
    pub preconditions: Vec<String>,
    /// What this step should produce
    pub expected_outcome: Option<String>,
    /// Error margin allowed
    pub tolerance: f64,
}

impl Procedure {
    pub fn new(name: &str) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.to_string(),
            steps: Vec::new(),
            skill_level: 0.0,
            practice_amount: 0.0,
            error_rate: 1.0, // Start with high error rate
            avg_execution_time: 0.0,
            last_practiced: None,
            success_count: 0,
            error_count: 0,
        }
    }

    pub fn add_step(&mut self, action: &str, duration_ms: u64) -> &mut Self {
        self.steps.push(ProcedureStep {
            action: action.to_string(),
            expected_duration_ms: duration_ms,
            preconditions: Vec::new(),
            expected_outcome: None,
            tolerance: 0.1,
        });
        self
    }

    pub fn with_steps(mut self, steps: Vec<(&str, u64)>) -> Self {
        for (action, duration) in steps {
            self.add_step(action, duration);
        }
        self
    }

    /// Update skill after practice
    pub fn practice(&mut self, success: bool, execution_time: f64) {
        self.last_practiced = Some(Utc::now());
        self.practice_amount += 1.0;

        if success {
            self.success_count += 1;
            // Skill increases with successful practice (diminishing returns)
            let learning_rate = 0.1 * (1.0 - self.skill_level);
            self.skill_level = (self.skill_level + learning_rate).min(1.0);
        } else {
            self.error_count += 1;
            // Errors decrease skill slightly
            self.skill_level = (self.skill_level - 0.02).max(0.0);
        }

        // Update error rate
        self.error_rate = self.error_count as f64 / (self.success_count + self.error_count) as f64;

        // Update average execution time
        self.avg_execution_time = self.avg_execution_time * 0.9 + execution_time * 0.1;
    }

    /// Is this procedure well-learned enough to be automatic?
    pub fn is_automatic(&self) -> bool {
        self.skill_level > 0.8 && self.error_rate < 0.1
    }

    /// How much more practice is needed?
    pub fn practice_needed(&self) -> f64 {
        (1.0 - self.skill_level) * 100.0 // Rough estimate of practice units needed
    }

    /// Decay skill if not practiced (use it or lose it)
    pub fn decay(&mut self, days_since_practice: f64) {
        // Slower decay for well-learned skills
        let decay_rate = 0.01 * (1.0 - self.skill_level * 0.5);
        let decay = decay_rate * days_since_practice;
        self.skill_level = (self.skill_level - decay).max(0.0);
    }

    /// Total expected duration
    pub fn total_duration(&self) -> u64 {
        self.steps.iter().map(|s| s.expected_duration_ms).sum()
    }
}

/// A timing prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingPrediction {
    pub event: String,
    pub predicted_time: DateTime<Utc>,
    pub confidence: f64,
    pub actual_time: Option<DateTime<Utc>>,
}

impl TimingPrediction {
    pub fn new(event: &str, predicted_time: DateTime<Utc>, confidence: f64) -> Self {
        Self {
            event: event.to_string(),
            predicted_time,
            confidence,
            actual_time: None,
        }
    }

    pub fn error(&self) -> Option<i64> {
        self.actual_time
            .map(|actual| (actual - self.predicted_time).num_milliseconds())
    }
}

/// Forward model for predicting outcomes of a procedure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForwardModel {
    pub procedure_id: Uuid,
    pub expected_outcomes: Vec<String>,
    pub expected_sensations: Vec<String>,
    pub confidence: f64,
    pub last_updated: DateTime<Utc>,
}

/// Motor imagery produced by cerebellar simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotorImagery {
    pub command_id: Uuid,
    pub action_name: String,
    pub predicted_outcomes: Vec<String>,
    pub predicted_sensations: Vec<String>,
    pub timing_prediction: TimingPrediction,
    pub success_probability: f64,
    pub confidence: f64,
}

impl MotorImagery {
    pub fn to_imagining(&self) -> Imagining {
        let content = format!(
            "Simulate action '{}': outcomes [{}], sensations [{}]",
            self.action_name,
            self.predicted_outcomes.join(", "),
            self.predicted_sensations.join(", ")
        );

        let mut imagining = Imagining::new(ImaginationType::Simulation, content, Vec::new());
        imagining.confidence = self.confidence;
        imagining.novelty = 0.4;
        imagining.utility = self.success_probability;
        imagining
    }
}

/// Statistics for the cerebellum
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CerebellumStats {
    pub procedures_learned: u64,
    pub automatic_skills: u64,
    pub total_practice: f64,
    pub average_skill_level: f64,
    pub average_error_rate: f64,
    pub timing_predictions: u64,
    pub timing_accuracy: f64,
    pub forward_models: u64,
    pub imagery_simulations: u64,
}

/// The Cerebellum
#[derive(Debug)]
pub struct Cerebellum {
    /// Learned procedures
    procedures: HashMap<Uuid, Procedure>,
    /// Procedures indexed by name
    name_index: HashMap<String, Uuid>,
    /// Forward models for procedures
    forward_models: HashMap<Uuid, ForwardModel>,
    /// Active timing predictions
    timing_predictions: Vec<TimingPrediction>,
    /// Timing error history (for calibration)
    timing_errors: Vec<i64>,
    /// Internal clock rate (for timing calibration)
    clock_rate: f64,
    /// Motor imagery history
    imagery_history: Vec<MotorImagery>,
    /// Maximum motor imagery history
    imagery_history_limit: usize,
    /// Statistics
    stats: CerebellumStats,
}

impl Cerebellum {
    pub fn new() -> Self {
        Self {
            procedures: HashMap::new(),
            name_index: HashMap::new(),
            forward_models: HashMap::new(),
            timing_predictions: Vec::new(),
            timing_errors: Vec::new(),
            clock_rate: 1.0,
            imagery_history: Vec::new(),
            imagery_history_limit: 32,
            stats: CerebellumStats::default(),
        }
    }

    /// Learn a new procedure
    pub fn learn(&mut self, procedure: Procedure) -> Uuid {
        let id = procedure.id;
        self.name_index.insert(procedure.name.clone(), id);
        self.procedures.insert(id, procedure);
        self.stats.procedures_learned += 1;
        self.update_stats();
        id
    }

    /// Get a procedure by ID
    pub fn get(&self, id: Uuid) -> Option<&Procedure> {
        self.procedures.get(&id)
    }

    /// Get a procedure by name
    pub fn get_by_name(&self, name: &str) -> Option<&Procedure> {
        self.name_index
            .get(name)
            .and_then(|id| self.procedures.get(id))
    }

    /// Practice a procedure
    pub fn practice(&mut self, id: Uuid, success: bool, execution_time: f64) {
        if let Some(procedure) = self.procedures.get_mut(&id) {
            procedure.practice(success, execution_time);
            self.stats.total_practice += 1.0;
            self.update_stats();
        }
    }

    /// Execute a procedure (returns steps to perform)
    pub fn execute(&self, id: Uuid) -> Option<Vec<&ProcedureStep>> {
        self.procedures.get(&id).map(|p| p.steps.iter().collect())
    }

    /// Check if a procedure is automatic (can run without conscious control)
    pub fn is_automatic(&self, id: Uuid) -> bool {
        self.procedures
            .get(&id)
            .map(|p| p.is_automatic())
            .unwrap_or(false)
    }

    /// Get all automatic procedures
    pub fn automatic_procedures(&self) -> Vec<&Procedure> {
        self.procedures
            .values()
            .filter(|p| p.is_automatic())
            .collect()
    }

    /// Create a timing prediction
    pub fn predict_timing(&mut self, event: &str, duration_ms: i64) -> TimingPrediction {
        let predicted = Utc::now()
            + chrono::Duration::milliseconds((duration_ms as f64 * self.clock_rate) as i64);

        // Confidence based on timing history accuracy
        let confidence = if self.timing_errors.is_empty() {
            0.5
        } else {
            let avg_error: f64 = self
                .timing_errors
                .iter()
                .map(|e| e.abs() as f64)
                .sum::<f64>()
                / self.timing_errors.len() as f64;
            (1.0 - avg_error / 1000.0).clamp(0.1, 0.95) // Normalize to confidence
        };

        let prediction = TimingPrediction::new(event, predicted, confidence);
        self.timing_predictions.push(prediction.clone());
        self.stats.timing_predictions += 1;
        prediction
    }

    /// Update or create a forward model for a procedure.
    pub fn update_forward_model(
        &mut self,
        procedure_id: Uuid,
        outcomes: Vec<String>,
        sensations: Vec<String>,
        confidence: f64,
    ) {
        let model = ForwardModel {
            procedure_id,
            expected_outcomes: outcomes,
            expected_sensations: sensations,
            confidence: confidence.clamp(0.0, 1.0),
            last_updated: Utc::now(),
        };

        self.forward_models.insert(procedure_id, model);
        self.stats.forward_models = self.forward_models.len() as u64;
    }

    /// Simulate the outcome of a motor command (motor imagery).
    pub fn simulate_command(&mut self, command: &MotorCommand) -> MotorImagery {
        let procedure_id = self.name_index.get(&command.name).copied();
        let (model_outcomes, model_sensations, model_confidence) = procedure_id
            .and_then(|id| self.forward_models.get(&id))
            .map(|model| {
                (
                    Some(model.expected_outcomes.clone()),
                    Some(model.expected_sensations.clone()),
                    Some(model.confidence),
                )
            })
            .unwrap_or((None, None, None));
        let success_probability = procedure_id
            .and_then(|id| self.procedures.get(&id))
            .map(|proc| (1.0 - proc.error_rate).clamp(0.05, 0.99))
            .unwrap_or(0.5);

        let predicted_outcomes = model_outcomes.unwrap_or_else(|| {
            if command.steps.is_empty() {
                vec![format!("complete:{}", command.name)]
            } else {
                command
                    .steps
                    .iter()
                    .map(|step| format!("complete:{}", step.description))
                    .collect()
            }
        });

        let predicted_sensations = model_sensations.unwrap_or_else(|| {
            command
                .steps
                .iter()
                .map(|step| format!("sensation:{}", step.description))
                .collect()
        });

        let timing_prediction = self.predict_timing(
            &format!("motor:{}", command.name),
            command.estimated_duration_ms as i64,
        );

        let confidence = (timing_prediction.confidence * 0.5
            + model_confidence.unwrap_or(0.4) * 0.3
            + success_probability * 0.2)
            .clamp(0.0, 1.0);

        let imagery = MotorImagery {
            command_id: command.id,
            action_name: command.name.clone(),
            predicted_outcomes,
            predicted_sensations,
            timing_prediction,
            success_probability,
            confidence,
        };

        self.record_imagery(imagery.clone());
        imagery
    }

    /// Record actual timing (for learning)
    pub fn record_timing(&mut self, event: &str, actual_time: DateTime<Utc>) {
        // Find matching prediction
        if let Some(prediction) = self
            .timing_predictions
            .iter_mut()
            .find(|p| p.event == event && p.actual_time.is_none())
        {
            prediction.actual_time = Some(actual_time);

            // Calculate and store error
            if let Some(error) = prediction.error() {
                self.timing_errors.push(error);
                if self.timing_errors.len() > 100 {
                    self.timing_errors.remove(0);
                }

                // Adjust clock rate based on systematic errors
                let avg_error: f64 = self.timing_errors.iter().map(|e| *e as f64).sum::<f64>()
                    / self.timing_errors.len() as f64;

                // If we're consistently early (negative error), slow down
                // If we're consistently late (positive error), speed up
                self.clock_rate = (self.clock_rate - avg_error / 10000.0).clamp(0.8, 1.2);
            }
        }

        // Update timing accuracy stat
        if !self.timing_errors.is_empty() {
            let accurate_count = self
                .timing_errors
                .iter()
                .filter(|e| e.abs() < 100) // Within 100ms
                .count();
            self.stats.timing_accuracy = accurate_count as f64 / self.timing_errors.len() as f64;
        }
    }

    /// Decay skills that haven't been practiced
    pub fn decay_all(&mut self, days: f64) {
        for procedure in self.procedures.values_mut() {
            if let Some(last) = procedure.last_practiced {
                let days_since = (Utc::now() - last).num_hours() as f64 / 24.0;
                if days_since > days {
                    procedure.decay(days_since - days);
                }
            }
        }
        self.update_stats();
    }

    /// Combine two procedures into a larger one (chunking)
    pub fn chunk(&mut self, name: &str, procedure_ids: Vec<Uuid>) -> Option<Uuid> {
        let mut combined_steps = Vec::new();

        for id in &procedure_ids {
            if let Some(proc) = self.procedures.get(id) {
                combined_steps.extend(proc.steps.clone());
            } else {
                return None;
            }
        }

        let mut combined = Procedure::new(name);
        combined.steps = combined_steps;

        // Inherit some skill from component procedures
        let avg_skill: f64 = procedure_ids
            .iter()
            .filter_map(|id| self.procedures.get(id))
            .map(|p| p.skill_level)
            .sum::<f64>()
            / procedure_ids.len() as f64;
        combined.skill_level = avg_skill * 0.5; // Start at half the average

        Some(self.learn(combined))
    }

    /// Get procedures that need practice (skill below threshold)
    pub fn needs_practice(&self, threshold: f64) -> Vec<&Procedure> {
        self.procedures
            .values()
            .filter(|p| p.skill_level < threshold)
            .collect()
    }

    /// Update statistics
    fn update_stats(&mut self) {
        if self.procedures.is_empty() {
            return;
        }

        self.stats.automatic_skills = self
            .procedures
            .values()
            .filter(|p| p.is_automatic())
            .count() as u64;

        self.stats.average_skill_level =
            self.procedures.values().map(|p| p.skill_level).sum::<f64>()
                / self.procedures.len() as f64;

        self.stats.average_error_rate = self.procedures.values().map(|p| p.error_rate).sum::<f64>()
            / self.procedures.len() as f64;
    }

    fn record_imagery(&mut self, imagery: MotorImagery) {
        if self.imagery_history.len() == self.imagery_history_limit {
            self.imagery_history.remove(0);
        }
        self.imagery_history.push(imagery);
        self.stats.imagery_simulations += 1;
    }

    /// Get statistics
    pub fn stats(&self) -> &CerebellumStats {
        &self.stats
    }

    /// Find procedures matching a pattern
    pub fn find(&self, pattern: &str) -> Vec<&Procedure> {
        let pattern_lower = pattern.to_lowercase();
        self.procedures
            .values()
            .filter(|p| p.name.to_lowercase().contains(&pattern_lower))
            .collect()
    }
}

impl Default for Cerebellum {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_procedure_learning() {
        let mut cerebellum = Cerebellum::new();

        let procedure = Procedure::new("typing").with_steps(vec![
            ("position_hands", 100),
            ("press_key", 50),
            ("release_key", 50),
        ]);

        let id = cerebellum.learn(procedure);

        assert!(cerebellum.get(id).is_some());
        assert!(!cerebellum.is_automatic(id));
    }

    #[test]
    fn test_practice_improves_skill() {
        let mut cerebellum = Cerebellum::new();

        let procedure = Procedure::new("test_skill");
        let id = cerebellum.learn(procedure);

        let initial = cerebellum.get(id).unwrap().skill_level;

        // Practice successfully many times
        for _ in 0..50 {
            cerebellum.practice(id, true, 100.0);
        }

        let final_skill = cerebellum.get(id).unwrap().skill_level;
        assert!(final_skill > initial, "Skill should improve with practice");
    }

    #[test]
    fn test_automaticity() {
        let mut cerebellum = Cerebellum::new();

        let mut procedure = Procedure::new("automatic_test");
        procedure.skill_level = 0.9;
        procedure.error_rate = 0.05;
        let id = cerebellum.learn(procedure);

        assert!(cerebellum.is_automatic(id));
    }

    #[test]
    fn test_timing_prediction() {
        let mut cerebellum = Cerebellum::new();

        let prediction = cerebellum.predict_timing("event", 1000);

        assert!(prediction.confidence > 0.0);
        assert!(prediction.actual_time.is_none());
    }

    #[test]
    fn test_chunking() {
        let mut cerebellum = Cerebellum::new();

        let proc1 = Procedure::new("step1").with_steps(vec![("a", 100)]);
        let proc2 = Procedure::new("step2").with_steps(vec![("b", 100)]);

        let id1 = cerebellum.learn(proc1);
        let id2 = cerebellum.learn(proc2);

        let combined_id = cerebellum.chunk("combined", vec![id1, id2]);
        assert!(combined_id.is_some());

        let combined = cerebellum.get(combined_id.unwrap()).unwrap();
        assert_eq!(combined.steps.len(), 2);
    }

    #[test]
    fn test_skill_decay() {
        let mut cerebellum = Cerebellum::new();

        let mut procedure = Procedure::new("decay_test");
        procedure.skill_level = 0.5;
        procedure.last_practiced = Some(Utc::now() - chrono::Duration::days(30));
        let id = cerebellum.learn(procedure);

        cerebellum.decay_all(7.0); // Decay if not practiced in 7 days

        let skill = cerebellum.get(id).unwrap().skill_level;
        assert!(skill < 0.5, "Skill should decay without practice");
    }

    #[test]
    fn test_motor_imagery_from_command() {
        let mut cerebellum = Cerebellum::new();
        let mut motor_cortex = crate::regions::motor_cortex::MotorCortex::new();

        let procedure = Procedure::new("wave").with_steps(vec![("raise", 400), ("wave", 600)]);
        let procedure_id = cerebellum.learn(procedure);
        cerebellum.update_forward_model(
            procedure_id,
            vec!["greeting acknowledged".to_string()],
            vec!["arm movement".to_string()],
            0.7,
        );

        let action = crate::regions::basal_ganglia::ActionPattern::new("wave")
            .with_sub_actions(vec!["raise", "wave"]);
        let command = motor_cortex.prepare_command(&action, "greet", 0.6, false);

        let imagery = cerebellum.simulate_command(&command);
        assert!(
            imagery
                .predicted_outcomes
                .iter()
                .any(|o| o.contains("greeting"))
        );
        assert!(imagery.success_probability > 0.0);

        let imagining = imagery.to_imagining();
        assert_eq!(imagining.imagination_type, ImaginationType::Simulation);
    }
}
