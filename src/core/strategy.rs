//! Long-Horizon Strategy Regulation
//!
//! Tracks mood and sleep signals to adapt long-horizon planning and narratives.

use std::collections::VecDeque;

/// Inputs for strategy regulation.
#[derive(Debug, Clone, Copy)]
pub struct StrategySignal {
    pub sleep_quality: f64,
    pub mood_stability: f64,
    pub stress: f64,
}

impl StrategySignal {
    pub fn new(sleep_quality: f64, mood_stability: f64, stress: f64) -> Self {
        Self {
            sleep_quality: sleep_quality.clamp(0.0, 1.0),
            mood_stability: mood_stability.clamp(0.0, 1.0),
            stress: stress.clamp(0.0, 1.0),
        }
    }
}

/// Strategic profile derived from recent signals.
#[derive(Debug, Clone, Copy)]
pub struct StrategyProfile {
    /// Bias toward long-horizon goals (-1.0 = short-term, +1.0 = long-term).
    pub long_horizon_bias: f64,
    /// Need for recovery/rest (0.0 to 1.0).
    pub recovery_priority: f64,
    /// Rolling sleep quality estimate.
    pub sleep_quality: f64,
    /// Rolling mood stability estimate.
    pub mood_stability: f64,
    /// Rolling stress estimate.
    pub stress: f64,
}

impl Default for StrategyProfile {
    fn default() -> Self {
        Self {
            long_horizon_bias: 0.0,
            recovery_priority: 0.0,
            sleep_quality: 0.5,
            mood_stability: 0.5,
            stress: 0.0,
        }
    }
}

/// Narrative update when strategic posture shifts.
#[derive(Debug, Clone)]
pub struct StrategyNarrative {
    pub content: String,
    pub significance: f64,
}

/// Update payload from the strategy regulator.
#[derive(Debug, Clone)]
pub struct StrategyUpdate {
    pub profile: StrategyProfile,
    pub narrative: Option<StrategyNarrative>,
}

/// Regulator that smooths signals and emits strategic adjustments.
pub struct StrategyRegulator {
    history: VecDeque<StrategySignal>,
    max_history: usize,
    profile: StrategyProfile,
}

impl StrategyRegulator {
    pub fn new() -> Self {
        Self {
            history: VecDeque::with_capacity(12),
            max_history: 12,
            profile: StrategyProfile::default(),
        }
    }

    pub fn update(&mut self, signal: StrategySignal) -> StrategyUpdate {
        if self.history.len() >= self.max_history {
            self.history.pop_front();
        }
        self.history.push_back(signal);

        let (avg_sleep, avg_mood, avg_stress) = self.averages();
        let long_horizon_bias =
            ((avg_sleep - 0.5) * 0.7 + (avg_mood - 0.5) * 0.7 - avg_stress * 0.4).clamp(-1.0, 1.0);
        let recovery_priority = ((1.0 - avg_sleep) * 0.7 + avg_stress * 0.3).clamp(0.0, 1.0);

        let new_profile = StrategyProfile {
            long_horizon_bias,
            recovery_priority,
            sleep_quality: avg_sleep,
            mood_stability: avg_mood,
            stress: avg_stress,
        };

        let change = (new_profile.long_horizon_bias - self.profile.long_horizon_bias).abs()
            + (new_profile.recovery_priority - self.profile.recovery_priority).abs();

        let narrative = if change > 0.25 {
            Some(StrategyNarrative {
                content: strategy_note(&new_profile),
                significance: change.clamp(0.2, 1.0),
            })
        } else {
            None
        };

        self.profile = new_profile;

        StrategyUpdate {
            profile: new_profile,
            narrative,
        }
    }

    pub fn profile(&self) -> StrategyProfile {
        self.profile
    }

    fn averages(&self) -> (f64, f64, f64) {
        if self.history.is_empty() {
            return (
                self.profile.sleep_quality,
                self.profile.mood_stability,
                self.profile.stress,
            );
        }

        let count = self.history.len() as f64;
        let mut sleep_sum = 0.0;
        let mut mood_sum = 0.0;
        let mut stress_sum = 0.0;

        for signal in &self.history {
            sleep_sum += signal.sleep_quality;
            mood_sum += signal.mood_stability;
            stress_sum += signal.stress;
        }

        (
            (sleep_sum / count).clamp(0.0, 1.0),
            (mood_sum / count).clamp(0.0, 1.0),
            (stress_sum / count).clamp(0.0, 1.0),
        )
    }
}

impl Default for StrategyRegulator {
    fn default() -> Self {
        Self::new()
    }
}

fn strategy_note(profile: &StrategyProfile) -> String {
    let note = if profile.recovery_priority > 0.6 {
        "Recovery prioritized; focus on short-term stability"
    } else if profile.long_horizon_bias > 0.3 {
        "Stable mood and rest support long-horizon goals"
    } else if profile.long_horizon_bias < -0.3 {
        "Volatility suggests near-term focus"
    } else {
        "Balanced strategic posture"
    };

    format!(
        "{} (sleep {:.2}, mood {:.2}, stress {:.2})",
        note, profile.sleep_quality, profile.mood_stability, profile.stress
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strategy_regulator_biases_toward_recovery_when_sleep_low() {
        let mut regulator = StrategyRegulator::new();
        let update = regulator.update(StrategySignal::new(0.2, 0.3, 0.7));
        assert!(update.profile.recovery_priority > 0.5);
        assert!(update.profile.long_horizon_bias < 0.0);
    }

    #[test]
    fn strategy_regulator_emits_narrative_on_large_shift() {
        let mut regulator = StrategyRegulator::new();
        regulator.update(StrategySignal::new(0.9, 0.9, 0.1));
        let update = regulator.update(StrategySignal::new(0.2, 0.2, 0.8));
        assert!(update.narrative.is_some());
    }
}
