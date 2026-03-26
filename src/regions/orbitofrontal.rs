//! Orbitofrontal Cortex - Value-Based Decision Making
//!
//! The orbitofrontal cortex (OFC) is critical for adaptive behavior through:
//!
//! - **Outcome evaluation**: Assesses the value of outcomes after they occur
//! - **Expected value computation**: Predicts how good/bad an option will be
//! - **Reversal learning**: Updates preferences when contingencies change
//! - **Counterfactual reasoning**: "What would have happened if I chose differently?"
//! - **Credit assignment**: Links actions to their consequences across time
//!
//! The OFC integrates information from sensory cortices, amygdala, and
//! hippocampus to maintain a running model of "what's valuable right now."

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use crate::signal::{BrainSignal, SignalType};

/// A valued option being tracked by the OFC.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValuedOption {
    pub id: Uuid,
    /// Description of the option/action
    pub description: String,
    /// Expected value based on experience (-1.0 to 1.0)
    pub expected_value: f64,
    /// Confidence in the expected value (0-1)
    pub confidence: f64,
    /// Number of times this option was chosen
    pub choice_count: u64,
    /// Running average of actual outcomes
    pub actual_outcome_avg: f64,
    /// Volatility of outcomes (standard deviation estimate)
    pub volatility: f64,
    /// Last time this option was evaluated
    pub last_evaluated: DateTime<Utc>,
    /// Associated context tags
    pub context: Vec<String>,
}

impl ValuedOption {
    pub fn new(description: &str) -> Self {
        Self {
            id: Uuid::new_v4(),
            description: description.to_string(),
            expected_value: 0.0,
            confidence: 0.2,
            choice_count: 0,
            actual_outcome_avg: 0.0,
            volatility: 0.5,
            last_evaluated: Utc::now(),
            context: Vec::new(),
        }
    }

    pub fn with_context(mut self, tags: Vec<String>) -> Self {
        self.context = tags;
        self
    }

    /// Update expected value based on actual outcome (Rescorla-Wagner learning).
    pub fn update_from_outcome(&mut self, actual: f64, learning_rate: f64) {
        let prediction_error = actual - self.expected_value;
        self.expected_value += learning_rate * prediction_error;
        self.expected_value = self.expected_value.clamp(-1.0, 1.0);

        // Update outcome average
        let n = self.choice_count as f64;
        self.actual_outcome_avg = (self.actual_outcome_avg * n + actual) / (n + 1.0);

        // Update volatility (running variance estimate)
        let deviation = (actual - self.actual_outcome_avg).abs();
        self.volatility = self.volatility * 0.8 + deviation * 0.2;

        // Confidence increases with experience, decreases with volatility
        self.confidence = ((self.choice_count as f64 / (self.choice_count as f64 + 5.0))
            - self.volatility * 0.3)
            .clamp(0.1, 0.95);

        self.choice_count += 1;
        self.last_evaluated = Utc::now();
    }
}

/// Result of an evaluation by the OFC.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evaluation {
    /// The option evaluated
    pub option_id: Uuid,
    /// Computed expected value
    pub expected_value: f64,
    /// Risk-adjusted value (accounts for volatility)
    pub risk_adjusted_value: f64,
    /// Confidence in this evaluation
    pub confidence: f64,
    /// Counterfactual regret from last choice (if applicable)
    pub counterfactual_regret: f64,
    /// Should current strategy be reversed?
    pub suggest_reversal: bool,
}

/// Outcome feedback for credit assignment.
#[derive(Debug, Clone)]
pub struct OutcomeFeedback {
    /// Which option led to this outcome
    pub option_description: String,
    /// The actual outcome value (-1.0 to 1.0)
    pub outcome_value: f64,
    /// Context in which this occurred
    pub context: Vec<String>,
}

/// Configuration for the OFC.
#[derive(Debug, Clone)]
pub struct OrbitofrontalConfig {
    /// Learning rate for value updates
    pub learning_rate: f64,
    /// Risk aversion factor (higher = more risk-averse)
    pub risk_aversion: f64,
    /// Threshold for suggesting strategy reversal
    pub reversal_threshold: f64,
    /// Maximum options to track
    pub max_options: usize,
}

impl Default for OrbitofrontalConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.15,
            risk_aversion: 0.3,
            reversal_threshold: 0.4,
            max_options: 64,
        }
    }
}

/// Statistics for the OFC.
#[derive(Debug, Clone, Default)]
pub struct OrbitofrontalStats {
    pub total_evaluations: u64,
    pub total_outcomes: u64,
    pub reversals_suggested: u64,
    pub tracked_options: usize,
    pub avg_confidence: f64,
    pub avg_prediction_error: f64,
}

/// Orbitofrontal Cortex - value-based decision maker.
pub struct OrbitofrontalCortex {
    config: OrbitofrontalConfig,
    options: HashMap<String, ValuedOption>,
    /// Last chosen option (for counterfactual reasoning)
    last_choice: Option<String>,
    /// Last unchosen option values (for regret computation)
    last_alternatives: Vec<(String, f64)>,
    /// Running prediction error average
    avg_prediction_error: f64,
    total_evaluations: u64,
    total_outcomes: u64,
    reversals_suggested: u64,
}

impl OrbitofrontalCortex {
    pub fn new() -> Self {
        Self::with_config(OrbitofrontalConfig::default())
    }

    pub fn with_config(config: OrbitofrontalConfig) -> Self {
        Self {
            config,
            options: HashMap::new(),
            last_choice: None,
            last_alternatives: Vec::new(),
            avg_prediction_error: 0.0,
            total_evaluations: 0,
            total_outcomes: 0,
            reversals_suggested: 0,
        }
    }

    /// Register or update an option.
    pub fn register_option(&mut self, description: &str, context: Vec<String>) -> Uuid {
        let option = self
            .options
            .entry(description.to_string())
            .or_insert_with(|| ValuedOption::new(description).with_context(context));
        option.id
    }

    /// Evaluate a set of options and recommend the best one.
    pub fn evaluate(&mut self, option_descriptions: &[&str]) -> Vec<Evaluation> {
        // Ensure all options exist first
        for desc in option_descriptions {
            self.options
                .entry(desc.to_string())
                .or_insert_with(|| ValuedOption::new(desc));
        }

        // Snapshot last choice info for counterfactual reasoning
        let last_choice_info = self.last_choice.as_ref().and_then(|last| {
            self.options
                .get(last)
                .map(|o| (last.clone(), o.actual_outcome_avg, o.choice_count))
        });

        let mut evaluations: Vec<Evaluation> = Vec::new();

        for desc in option_descriptions {
            let option = self.options.get(*desc).unwrap();

            let risk_penalty = option.volatility * self.config.risk_aversion;
            let risk_adjusted = option.expected_value - risk_penalty;

            // Counterfactual regret
            let regret = if let Some((ref last, last_avg, _)) = last_choice_info {
                if last != *desc {
                    (option.expected_value - last_avg).max(0.0)
                } else {
                    0.0
                }
            } else {
                0.0
            };

            // Suggest reversal if current strategy is persistently worse
            let suggest_reversal = if let Some((_, last_avg, last_count)) = last_choice_info {
                last_count > 3 && last_avg < option.expected_value - self.config.reversal_threshold
            } else {
                false
            };

            if suggest_reversal {
                self.reversals_suggested += 1;
            }

            evaluations.push(Evaluation {
                option_id: option.id,
                expected_value: option.expected_value,
                risk_adjusted_value: risk_adjusted,
                confidence: option.confidence,
                counterfactual_regret: regret,
                suggest_reversal,
            });

            self.total_evaluations += 1;
        }

        // Sort by risk-adjusted value (best first)
        evaluations.sort_by(|a, b| {
            b.risk_adjusted_value
                .partial_cmp(&a.risk_adjusted_value)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Record choice and alternatives for counterfactual reasoning
        if !evaluations.is_empty() {
            if let Some(desc) = option_descriptions.first() {
                self.last_choice = Some(desc.to_string());
            }
            self.last_alternatives = evaluations
                .iter()
                .skip(1)
                .zip(option_descriptions.iter().skip(1))
                .map(|(e, d)| (d.to_string(), e.expected_value))
                .collect();
        }

        evaluations
    }

    /// Report an outcome for credit assignment.
    pub fn report_outcome(&mut self, feedback: &OutcomeFeedback) {
        let learning_rate = self.config.learning_rate;

        if let Some(option) = self.options.get_mut(&feedback.option_description) {
            let prediction_error = (feedback.outcome_value - option.expected_value).abs();
            self.avg_prediction_error = self.avg_prediction_error * 0.9 + prediction_error * 0.1;

            option.update_from_outcome(feedback.outcome_value, learning_rate);
            self.total_outcomes += 1;
        } else {
            // New option encountered through outcome
            let mut option = ValuedOption::new(&feedback.option_description)
                .with_context(feedback.context.clone());
            option.update_from_outcome(feedback.outcome_value, learning_rate);
            self.options
                .insert(feedback.option_description.clone(), option);
            self.total_outcomes += 1;
        }
    }

    /// Generate an evaluation signal for broadcasting.
    pub fn to_signal(&self) -> BrainSignal {
        let avg_value = if self.options.is_empty() {
            0.0
        } else {
            self.options.values().map(|o| o.expected_value).sum::<f64>() / self.options.len() as f64
        };

        BrainSignal::new(
            "orbitofrontal_cortex",
            SignalType::Evaluation,
            serde_json::json!({
                "avg_expected_value": avg_value,
                "tracked_options": self.options.len(),
                "avg_prediction_error": self.avg_prediction_error,
            }),
        )
        .with_valence(avg_value)
        .with_salience((self.avg_prediction_error * 0.5 + 0.3).clamp(0.0, 1.0))
        .with_metadata("reversals_suggested", self.reversals_suggested)
    }

    /// Get a specific option by description.
    pub fn get_option(&self, description: &str) -> Option<&ValuedOption> {
        self.options.get(description)
    }

    /// Get all tracked options sorted by expected value.
    pub fn ranked_options(&self) -> Vec<&ValuedOption> {
        let mut options: Vec<&ValuedOption> = self.options.values().collect();
        options.sort_by(|a, b| {
            b.expected_value
                .partial_cmp(&a.expected_value)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        options
    }

    /// Get statistics.
    pub fn stats(&self) -> OrbitofrontalStats {
        let avg_confidence = if self.options.is_empty() {
            0.0
        } else {
            self.options.values().map(|o| o.confidence).sum::<f64>() / self.options.len() as f64
        };

        OrbitofrontalStats {
            total_evaluations: self.total_evaluations,
            total_outcomes: self.total_outcomes,
            reversals_suggested: self.reversals_suggested,
            tracked_options: self.options.len(),
            avg_confidence,
            avg_prediction_error: self.avg_prediction_error,
        }
    }
}

impl Default for OrbitofrontalCortex {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learns_from_outcomes() {
        let mut ofc = OrbitofrontalCortex::new();
        ofc.register_option("approach", vec![]);

        // Positive outcomes should increase expected value
        for _ in 0..5 {
            ofc.report_outcome(&OutcomeFeedback {
                option_description: "approach".to_string(),
                outcome_value: 0.8,
                context: vec![],
            });
        }

        let option = ofc.get_option("approach").unwrap();
        assert!(option.expected_value > 0.3);
        assert!(option.confidence > 0.3);
    }

    #[test]
    fn evaluates_and_ranks_options() {
        let mut ofc = OrbitofrontalCortex::new();

        // Give "approach" positive history
        for _ in 0..5 {
            ofc.report_outcome(&OutcomeFeedback {
                option_description: "approach".to_string(),
                outcome_value: 0.7,
                context: vec![],
            });
        }

        // Give "avoid" negative history
        for _ in 0..5 {
            ofc.report_outcome(&OutcomeFeedback {
                option_description: "avoid".to_string(),
                outcome_value: -0.3,
                context: vec![],
            });
        }

        let evaluations = ofc.evaluate(&["approach", "avoid"]);
        assert!(evaluations[0].risk_adjusted_value > evaluations[1].risk_adjusted_value);
    }

    #[test]
    fn reversal_learning() {
        let mut ofc = OrbitofrontalCortex::new();

        // First, "option_a" is good
        for _ in 0..5 {
            ofc.report_outcome(&OutcomeFeedback {
                option_description: "option_a".to_string(),
                outcome_value: 0.8,
                context: vec![],
            });
        }

        // Choose option_a
        ofc.evaluate(&["option_a", "option_b"]);

        // Now option_a becomes bad
        for _ in 0..5 {
            ofc.report_outcome(&OutcomeFeedback {
                option_description: "option_a".to_string(),
                outcome_value: -0.5,
                context: vec![],
            });
        }
        // And option_b becomes good
        for _ in 0..5 {
            ofc.report_outcome(&OutcomeFeedback {
                option_description: "option_b".to_string(),
                outcome_value: 0.8,
                context: vec![],
            });
        }

        let evaluations = ofc.evaluate(&["option_a", "option_b"]);
        // option_b should now be ranked higher
        let option_b_eval = evaluations
            .iter()
            .find(|e| {
                ofc.get_option("option_b")
                    .is_some_and(|o| o.id == e.option_id)
            })
            .unwrap();
        assert!(option_b_eval.risk_adjusted_value > 0.0);
    }

    #[test]
    fn volatility_reduces_confidence() {
        let mut ofc = OrbitofrontalCortex::new();

        // Volatile outcomes
        let outcomes = [0.9, -0.8, 0.7, -0.6, 0.5, -0.4];
        for val in outcomes {
            ofc.report_outcome(&OutcomeFeedback {
                option_description: "volatile".to_string(),
                outcome_value: val,
                context: vec![],
            });
        }

        let volatile_confidence = ofc.get_option("volatile").unwrap().confidence;

        // Stable outcomes
        for _ in 0..6 {
            ofc.report_outcome(&OutcomeFeedback {
                option_description: "stable".to_string(),
                outcome_value: 0.5,
                context: vec![],
            });
        }

        let stable_confidence = ofc.get_option("stable").unwrap().confidence;
        assert!(
            stable_confidence > volatile_confidence,
            "Stable option should have higher confidence than volatile"
        );
    }

    #[test]
    fn generates_evaluation_signal() {
        let ofc = OrbitofrontalCortex::new();
        let signal = ofc.to_signal();
        assert_eq!(signal.signal_type, SignalType::Evaluation);
    }
}
