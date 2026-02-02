//! Motor Cortex - Action Planning and Execution Sequencing
//!
//! Translates basal ganglia action selections into structured motor commands.
//! Provides sequencing, timing estimates, and execution readiness.

use crate::regions::basal_ganglia::ActionPattern;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use uuid::Uuid;

/// Individual step in a motor command.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotorStep {
    pub description: String,
    pub estimated_duration_ms: u64,
}

/// Structured motor command produced by motor cortex.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotorCommand {
    pub id: Uuid,
    pub action_id: Uuid,
    pub name: String,
    pub context: String,
    pub steps: Vec<MotorStep>,
    pub confidence: f64,
    pub urgency: f64,
    pub estimated_duration_ms: u64,
    pub habitual: bool,
    pub prepared_at: DateTime<Utc>,
}

/// Motor cortex module.
pub struct MotorCortex {
    command_history: VecDeque<MotorCommand>,
    history_limit: usize,
}

impl Default for MotorCortex {
    fn default() -> Self {
        Self::new()
    }
}

impl MotorCortex {
    pub fn new() -> Self {
        Self {
            command_history: VecDeque::with_capacity(16),
            history_limit: 16,
        }
    }

    /// Prepare a motor command from an action pattern.
    pub fn prepare_command(
        &mut self,
        action: &ActionPattern,
        context: &str,
        selection_confidence: f64,
        habitual: bool,
    ) -> MotorCommand {
        let mut steps = Vec::new();
        if action.sub_actions.is_empty() {
            steps.push(MotorStep {
                description: action.name.clone(),
                estimated_duration_ms: 900,
            });
        } else {
            for step in &action.sub_actions {
                steps.push(MotorStep {
                    description: step.clone(),
                    estimated_duration_ms: 700,
                });
            }
        }

        let automaticity_bonus = action.automaticity * 0.2;
        let confidence = (selection_confidence * 0.7 + automaticity_bonus).clamp(0.0, 1.0);
        let urgency = ((action.value + 1.0) / 2.0).clamp(0.0, 1.0);

        let base_duration: u64 = steps.iter().map(|step| step.estimated_duration_ms).sum();
        let duration_reduction = (automaticity_bonus * 0.25).min(0.2);
        let estimated_duration_ms = (base_duration as f64 * (1.0 - duration_reduction)) as u64;

        let command = MotorCommand {
            id: Uuid::new_v4(),
            action_id: action.id,
            name: action.name.clone(),
            context: context.to_string(),
            steps,
            confidence,
            urgency,
            estimated_duration_ms,
            habitual,
            prepared_at: Utc::now(),
        };

        self.record_command(command.clone());
        command
    }

    /// Get the most recent motor command.
    pub fn last_command(&self) -> Option<&MotorCommand> {
        self.command_history.back()
    }

    /// Recent command history (most recent last).
    pub fn history(&self) -> Vec<&MotorCommand> {
        self.command_history.iter().collect()
    }

    fn record_command(&mut self, command: MotorCommand) {
        if self.command_history.len() == self.history_limit {
            self.command_history.pop_front();
        }
        self.command_history.push_back(command);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepares_command_from_sub_actions() {
        let mut cortex = MotorCortex::new();
        let action = ActionPattern::new("greet")
            .with_sub_actions(vec!["raise hand", "wave"])
            .with_context("hello");

        let command = cortex.prepare_command(&action, "hello", 0.7, false);

        assert_eq!(command.steps.len(), 2);
        assert!(
            command
                .steps
                .iter()
                .any(|step| step.description == "raise hand")
        );
    }

    #[test]
    fn defaults_to_single_step() {
        let mut cortex = MotorCortex::new();
        let action = ActionPattern::new("nod").with_context("agree");

        let command = cortex.prepare_command(&action, "agree", 0.6, true);

        assert_eq!(command.steps.len(), 1);
        assert_eq!(command.steps[0].description, "nod");
        assert!(command.habitual);
    }
}
