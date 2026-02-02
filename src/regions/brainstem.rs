//! Brainstem - Autonomic Control & Bodily Feedback
//!
//! Simulates autonomic centers that translate hypothalamic drives and
//! neuromodulatory tone into real-time bodily feedback for the insula
//! and hypothalamus.

use crate::core::neuromodulators::NeuromodulatorState;
use crate::regions::hypothalamus::{DriveType, Hypothalamus};
use crate::regions::insula::BodyState;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Configuration for brainstem autonomic responses.
#[derive(Debug, Clone)]
pub struct BrainstemConfig {
    pub adjustment_rate: f64,
    pub stress_sensitivity: f64,
    pub drive_sensitivity: f64,
}

impl Default for BrainstemConfig {
    fn default() -> Self {
        Self {
            adjustment_rate: 0.35,
            stress_sensitivity: 0.8,
            drive_sensitivity: 0.6,
        }
    }
}

/// Autonomic feedback produced by the brainstem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutonomicFeedback {
    pub body_state: BodyState,
    pub stress_signal: f64,
    pub drive_pressure: f64,
    pub notes: Vec<String>,
    pub timestamp: DateTime<Utc>,
}

/// Brainstem autonomic controller.
pub struct Brainstem {
    config: BrainstemConfig,
    state: BodyState,
}

impl Brainstem {
    pub fn new() -> Self {
        Self::with_config(BrainstemConfig::default())
    }

    pub fn with_config(config: BrainstemConfig) -> Self {
        Self {
            config,
            state: BodyState::new(),
        }
    }

    /// Update autonomic state and return feedback.
    pub fn update(
        &mut self,
        hypothalamus: &Hypothalamus,
        neuromodulators: &NeuromodulatorState,
    ) -> AutonomicFeedback {
        let hunger = hypothalamus
            .get_drive(DriveType::Hunger)
            .map(|d| d.level)
            .unwrap_or(0.0);
        let thirst = hypothalamus
            .get_drive(DriveType::Thirst)
            .map(|d| d.level)
            .unwrap_or(0.0);
        let fatigue = hypothalamus
            .get_drive(DriveType::Fatigue)
            .map(|d| d.level)
            .unwrap_or(0.0);
        let safety = hypothalamus
            .get_drive(DriveType::Safety)
            .map(|d| d.level)
            .unwrap_or(0.0);
        let stress = hypothalamus.stress.level() * self.config.stress_sensitivity;
        let arousal = neuromodulators.norepinephrine;

        let drive_pressure =
            (hunger + thirst + fatigue + safety) * 0.25 * self.config.drive_sensitivity;

        let heart_rate = (1.0 + stress * 0.4 + arousal * 0.2 - fatigue * 0.1).clamp(0.6, 1.6);
        let breathing = (0.3 + stress * 0.5 + arousal * 0.2).clamp(0.1, 1.0);
        let gut = (-hunger * 0.4 - stress * 0.2 + (neuromodulators.dopamine - 0.5) * 0.2)
            .clamp(-1.0, 1.0);
        let tension = (stress * 0.6 + arousal * 0.2 + safety * 0.2).clamp(0.0, 1.0);
        let energy = (0.7 - fatigue * 0.6 + neuromodulators.dopamine * 0.1).clamp(0.0, 1.0);
        let temperature = (stress * 0.2 - fatigue * 0.1).clamp(-0.3, 0.6);
        let pain = (safety * 0.2 + stress * 0.1).clamp(0.0, 1.0);

        let target = BodyState {
            heart_rate,
            breathing,
            gut,
            tension,
            energy,
            temperature,
            pain,
            timestamp: Utc::now(),
        };

        self.state = self.state.blend(&target, self.config.adjustment_rate);

        let mut notes = Vec::new();
        if hunger > 0.6 {
            notes.push("hunger_signal".to_string());
        }
        if thirst > 0.6 {
            notes.push("thirst_signal".to_string());
        }
        if fatigue > 0.6 {
            notes.push("fatigue_signal".to_string());
        }
        if stress > 0.6 {
            notes.push("stress_signal".to_string());
        }

        AutonomicFeedback {
            body_state: self.state.clone(),
            stress_signal: stress.clamp(0.0, 1.0),
            drive_pressure: drive_pressure.clamp(0.0, 1.0),
            notes,
            timestamp: Utc::now(),
        }
    }

    /// Get the current autonomic state.
    pub fn state(&self) -> &BodyState {
        &self.state
    }
}

impl Default for Brainstem {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feedback_reflects_drive_pressure() {
        let mut brainstem = Brainstem::new();
        let mut hypothalamus = Hypothalamus::new();
        let neuromodulators = NeuromodulatorState::default();

        hypothalamus.satisfy_drive(DriveType::Hunger, -0.5);
        hypothalamus.trigger_stress(0.6);

        let feedback = brainstem.update(&hypothalamus, &neuromodulators);

        assert!(feedback.drive_pressure >= 0.0);
        assert!(feedback.body_state.heart_rate >= 0.8);
    }
}
