//! Salience Network - Dorsal/Ventral Attention Coordination
//!
//! Combines goal-directed (dorsal) and stimulus-driven (ventral) attention
//! into a unified salience signal used to bias attention and routing.

use crate::core::attention::estimate_complexity;
use crate::signal::BrainSignal;
use serde::{Deserialize, Serialize};

/// Inputs needed to compute salience network state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SalienceInputs {
    pub cognitive_load: f64,
    pub interoceptive_alert: bool,
    pub stress_level: f64,
    pub mood_stability: f64,
    pub social_affinity: f64,
}

/// Output of the salience network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SalienceOutcome {
    pub focus: Option<String>,
    pub salience_boost: f64,
    pub priority_boost: i32,
    pub dorsal_weight: f64,
    pub ventral_weight: f64,
    pub reason: String,
}

/// Dorsal attention network (goal-directed).
#[derive(Debug, Default)]
struct DorsalAttentionNetwork;

impl DorsalAttentionNetwork {
    fn weight(&self, signal: &BrainSignal, cognitive_load: f64) -> f64 {
        let content = signal
            .content
            .as_str()
            .map(|s| s.to_string())
            .unwrap_or_else(|| signal.content.to_string());
        let complexity = estimate_complexity(&content).suggested_budget();
        (complexity * 0.8 + cognitive_load * 0.2).clamp(0.0, 1.0)
    }
}

/// Ventral attention network (stimulus-driven).
#[derive(Debug, Default)]
struct VentralAttentionNetwork;

impl VentralAttentionNetwork {
    fn weight(&self, signal: &BrainSignal, interoceptive_alert: bool, stress_level: f64) -> f64 {
        let mut weight = signal.salience.value() * 0.5;
        if signal.is_surprising() {
            weight += 0.25;
        }
        if interoceptive_alert {
            weight += 0.15;
        }
        weight += stress_level * 0.1;
        weight.clamp(0.0, 1.0)
    }
}

/// Salience network orchestrator.
#[derive(Debug, Default)]
pub struct SalienceNetwork {
    dorsal: DorsalAttentionNetwork,
    ventral: VentralAttentionNetwork,
    last_outcome: Option<SalienceOutcome>,
}

impl SalienceNetwork {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn update(&mut self, signal: &BrainSignal, inputs: SalienceInputs) -> SalienceOutcome {
        let mut dorsal_weight = self.dorsal.weight(signal, inputs.cognitive_load);
        let mut ventral_weight =
            self.ventral
                .weight(signal, inputs.interoceptive_alert, inputs.stress_level);

        dorsal_weight =
            (dorsal_weight + inputs.mood_stability * 0.1 + inputs.social_affinity * 0.1)
                .clamp(0.0, 1.0);
        ventral_weight = (ventral_weight
            + (1.0 - inputs.mood_stability) * 0.1
            + (1.0 - inputs.social_affinity) * 0.05)
            .clamp(0.0, 1.0);

        let balance = ventral_weight - dorsal_weight;
        let salience_boost = (balance * 0.2).clamp(-0.1, 0.2);
        let priority_boost = if ventral_weight > 0.7 { 1 } else { 0 };

        let focus = if dorsal_weight > ventral_weight {
            extract_focus(signal)
        } else {
            None
        };

        let reason = if ventral_weight > dorsal_weight {
            "ventral_capture"
        } else if dorsal_weight > ventral_weight {
            "dorsal_focus"
        } else {
            "balanced"
        }
        .to_string();

        let outcome = SalienceOutcome {
            focus,
            salience_boost,
            priority_boost,
            dorsal_weight,
            ventral_weight,
            reason,
        };

        self.last_outcome = Some(outcome.clone());
        outcome
    }

    pub fn last_outcome(&self) -> Option<&SalienceOutcome> {
        self.last_outcome.as_ref()
    }
}

fn extract_focus(signal: &BrainSignal) -> Option<String> {
    let content = signal
        .content
        .as_str()
        .map(|s| s.to_string())
        .unwrap_or_else(|| signal.content.to_string());
    let cleaned = content.replace(|c: char| !c.is_alphanumeric() && !c.is_whitespace(), " ");
    let focus = cleaned
        .split_whitespace()
        .take(3)
        .collect::<Vec<_>>()
        .join(" ");
    if focus.is_empty() { None } else { Some(focus) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal::{BrainSignal, SignalType};

    #[test]
    fn ventral_network_boosts_surprising_signal() {
        let mut network = SalienceNetwork::new();
        let signal = BrainSignal::new("test", SignalType::Sensory, "loud crash")
            .with_salience(0.9)
            .with_arousal(0.9);

        let outcome = network.update(
            &signal,
            SalienceInputs {
                cognitive_load: 0.2,
                interoceptive_alert: true,
                stress_level: 0.3,
                mood_stability: 0.2,
                social_affinity: 0.3,
            },
        );

        assert!(outcome.ventral_weight > outcome.dorsal_weight);
        assert!(outcome.salience_boost > 0.0);
    }

    #[test]
    fn dorsal_network_focuses_on_complex_signal() {
        let mut network = SalienceNetwork::new();
        let signal = BrainSignal::new(
            "test",
            SignalType::Query,
            "Analyze the multi-step system behavior and explain why it fails",
        )
        .with_salience(0.4);

        let outcome = network.update(
            &signal,
            SalienceInputs {
                cognitive_load: 0.6,
                interoceptive_alert: false,
                stress_level: 0.1,
                mood_stability: 0.8,
                social_affinity: 0.7,
            },
        );

        assert!(outcome.dorsal_weight >= outcome.ventral_weight);
        assert!(outcome.focus.is_some());
    }
}
