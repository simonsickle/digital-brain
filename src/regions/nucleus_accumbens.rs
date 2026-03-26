//! Nucleus Accumbens - Reward Processing Hub
//!
//! The nucleus accumbens is the brain's central reward processing station,
//! sitting at the intersection of dopaminergic motivation circuits and
//! limbic emotional circuits. It provides:
//!
//! - **Reward valuation**: Computes subjective value of outcomes
//! - **Incentive salience**: Transforms "liking" into "wanting"
//! - **Reward prediction errors**: Drives learning by comparing expected vs actual reward
//! - **Hedonic adaptation**: Adjusts baseline pleasure levels over time
//! - **Effort-reward tradeoff**: Gates whether effort is worth the expected payoff

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use uuid::Uuid;

use crate::signal::{BrainSignal, SignalType};

/// A reward event as processed by the nucleus accumbens.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardEvent {
    pub id: Uuid,
    /// What triggered this reward
    pub source: String,
    /// Raw reward magnitude (-1.0 to 1.0, negative = aversive)
    pub magnitude: f64,
    /// Expected reward (from predictions)
    pub expected: f64,
    /// Prediction error (actual - expected)
    pub prediction_error: f64,
    /// Subjective value after hedonic adaptation
    pub subjective_value: f64,
    /// Whether this exceeded the hedonic baseline
    pub above_baseline: bool,
    pub timestamp: DateTime<Utc>,
}

/// Current motivational state computed by the nucleus accumbens.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotivationalState {
    /// Current "wanting" level (incentive salience, 0-1)
    pub wanting: f64,
    /// Current "liking" level (hedonic response, 0-1)
    pub liking: f64,
    /// Hedonic baseline (shifts with adaptation)
    pub hedonic_baseline: f64,
    /// Whether the system is in a reward-seeking state
    pub reward_seeking: bool,
    /// Current effort willingness (0-1)
    pub effort_willingness: f64,
    /// Anhedonia risk (prolonged low reward, 0-1)
    pub anhedonia_risk: f64,
}

impl Default for MotivationalState {
    fn default() -> Self {
        Self {
            wanting: 0.5,
            liking: 0.5,
            hedonic_baseline: 0.0,
            reward_seeking: false,
            effort_willingness: 0.6,
            anhedonia_risk: 0.0,
        }
    }
}

/// Configuration for the nucleus accumbens.
#[derive(Debug, Clone)]
pub struct NucleusAccumbensConfig {
    /// Rate of hedonic adaptation (how fast baseline shifts)
    pub adaptation_rate: f64,
    /// How much dopamine level influences wanting
    pub dopamine_sensitivity: f64,
    /// Effort discount rate (how much effort reduces subjective value)
    pub effort_discount: f64,
    /// Maximum reward history to retain
    pub history_capacity: usize,
}

impl Default for NucleusAccumbensConfig {
    fn default() -> Self {
        Self {
            adaptation_rate: 0.02,
            dopamine_sensitivity: 0.8,
            effort_discount: 0.3,
            history_capacity: 32,
        }
    }
}

/// Statistics for the nucleus accumbens.
#[derive(Debug, Clone, Default)]
pub struct NucleusAccumbensStats {
    pub total_rewards_processed: u64,
    pub positive_rewards: u64,
    pub negative_rewards: u64,
    pub avg_prediction_error: f64,
    pub current_hedonic_baseline: f64,
    pub current_wanting: f64,
    pub current_liking: f64,
}

/// Nucleus Accumbens - the reward hub.
pub struct NucleusAccumbens {
    config: NucleusAccumbensConfig,
    state: MotivationalState,
    reward_history: VecDeque<RewardEvent>,
    /// Current dopamine level (set by neuromodulatory system)
    dopamine_level: f64,
    /// Running average of prediction errors
    avg_prediction_error: f64,
    total_rewards: u64,
    positive_count: u64,
    negative_count: u64,
}

impl NucleusAccumbens {
    pub fn new() -> Self {
        Self::with_config(NucleusAccumbensConfig::default())
    }

    pub fn with_config(config: NucleusAccumbensConfig) -> Self {
        Self {
            config,
            state: MotivationalState::default(),
            reward_history: VecDeque::with_capacity(32),
            dopamine_level: 0.5,
            avg_prediction_error: 0.0,
            total_rewards: 0,
            positive_count: 0,
            negative_count: 0,
        }
    }

    /// Process a reward event and update motivational state.
    pub fn process_reward(&mut self, source: &str, magnitude: f64, expected: f64) -> RewardEvent {
        let magnitude = magnitude.clamp(-1.0, 1.0);
        let expected = expected.clamp(-1.0, 1.0);
        let prediction_error = magnitude - expected;

        // Subjective value accounts for hedonic adaptation
        let subjective_value = magnitude - self.state.hedonic_baseline;
        let above_baseline = subjective_value > 0.0;

        let event = RewardEvent {
            id: Uuid::new_v4(),
            source: source.to_string(),
            magnitude,
            expected,
            prediction_error,
            subjective_value,
            above_baseline,
            timestamp: Utc::now(),
        };

        // Update hedonic baseline (adaptation)
        self.state.hedonic_baseline += magnitude * self.config.adaptation_rate;
        self.state.hedonic_baseline = self.state.hedonic_baseline.clamp(-0.5, 0.5);

        // Update prediction error running average
        self.avg_prediction_error = self.avg_prediction_error * 0.9 + prediction_error.abs() * 0.1;

        // Update wanting (incentive salience)
        // Positive prediction errors increase wanting; negative decrease
        let wanting_delta = prediction_error * self.config.dopamine_sensitivity * 0.1;
        self.state.wanting = (self.state.wanting + wanting_delta).clamp(0.0, 1.0);

        // Update liking (hedonic impact)
        if subjective_value > 0.0 {
            self.state.liking = (self.state.liking + subjective_value * 0.15).min(1.0);
        } else {
            self.state.liking = (self.state.liking + subjective_value * 0.1).max(0.0);
        }

        // Reward seeking when wanting exceeds liking
        self.state.reward_seeking = self.state.wanting > self.state.liking + 0.1;

        // Effort willingness tracks dopamine and recent rewards
        self.state.effort_willingness =
            (self.dopamine_level * 0.5 + self.state.wanting * 0.3 + self.state.liking * 0.2)
                .clamp(0.0, 1.0);

        // Anhedonia risk: prolonged low liking
        if self.state.liking < 0.3 {
            self.state.anhedonia_risk = (self.state.anhedonia_risk + 0.02).min(1.0);
        } else {
            self.state.anhedonia_risk = (self.state.anhedonia_risk - 0.05).max(0.0);
        }

        // Track counts
        self.total_rewards += 1;
        if magnitude > 0.0 {
            self.positive_count += 1;
        } else if magnitude < 0.0 {
            self.negative_count += 1;
        }

        // Record history
        if self.reward_history.len() >= self.config.history_capacity {
            self.reward_history.pop_front();
        }
        self.reward_history.push_back(event.clone());

        event
    }

    /// Evaluate whether an action is worth the effort.
    /// Returns the effort-discounted value (positive = worth it).
    pub fn evaluate_effort_tradeoff(&self, expected_reward: f64, required_effort: f64) -> f64 {
        let discounted_reward = expected_reward - required_effort * self.config.effort_discount;
        let dopamine_bonus = (self.dopamine_level - 0.5) * 0.2;
        (discounted_reward + dopamine_bonus) * self.state.effort_willingness
    }

    /// Set dopamine level from neuromodulatory system.
    pub fn set_dopamine(&mut self, level: f64) {
        self.dopamine_level = level.clamp(0.0, 1.0);
    }

    /// Get current motivational state.
    pub fn state(&self) -> &MotivationalState {
        &self.state
    }

    /// Generate a reward signal for broadcasting.
    pub fn to_signal(&self) -> BrainSignal {
        BrainSignal::new("nucleus_accumbens", SignalType::Reward, self.state.clone())
            .with_valence(self.state.liking * 2.0 - 1.0)
            .with_arousal(self.state.wanting)
            .with_salience(
                (self.state.wanting * 0.5 + self.avg_prediction_error * 0.5).clamp(0.0, 1.0),
            )
            .with_metadata("hedonic_baseline", self.state.hedonic_baseline)
            .with_metadata("effort_willingness", self.state.effort_willingness)
            .with_metadata("anhedonia_risk", self.state.anhedonia_risk)
    }

    /// Decay motivational state toward baseline (called during rest/sleep).
    pub fn rest(&mut self) {
        self.state.wanting = self.state.wanting * 0.95 + 0.5 * 0.05;
        self.state.liking = self.state.liking * 0.95 + 0.5 * 0.05;
        self.state.hedonic_baseline *= 0.9; // Baseline drifts toward zero
        self.state.anhedonia_risk = (self.state.anhedonia_risk - 0.03).max(0.0);
        self.state.effort_willingness =
            (self.state.effort_willingness * 0.9 + 0.6 * 0.1).clamp(0.0, 1.0);
    }

    /// Get recent reward history.
    pub fn recent_rewards(&self, count: usize) -> Vec<&RewardEvent> {
        self.reward_history.iter().rev().take(count).collect()
    }

    /// Get statistics.
    pub fn stats(&self) -> NucleusAccumbensStats {
        NucleusAccumbensStats {
            total_rewards_processed: self.total_rewards,
            positive_rewards: self.positive_count,
            negative_rewards: self.negative_count,
            avg_prediction_error: self.avg_prediction_error,
            current_hedonic_baseline: self.state.hedonic_baseline,
            current_wanting: self.state.wanting,
            current_liking: self.state.liking,
        }
    }
}

impl Default for NucleusAccumbens {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positive_reward_increases_liking() {
        let mut nac = NucleusAccumbens::new();
        let initial_liking = nac.state().liking;

        nac.process_reward("test", 0.8, 0.3);

        assert!(nac.state().liking > initial_liking);
    }

    #[test]
    fn prediction_error_drives_wanting() {
        let mut nac = NucleusAccumbens::new();
        let initial_wanting = nac.state().wanting;

        // Large positive prediction error (got more than expected)
        nac.process_reward("surprise", 0.9, 0.1);

        assert!(nac.state().wanting > initial_wanting);
    }

    #[test]
    fn hedonic_adaptation_shifts_baseline() {
        let mut nac = NucleusAccumbens::new();
        let initial_baseline = nac.state().hedonic_baseline;

        // Repeated positive rewards
        for _ in 0..10 {
            nac.process_reward("repeated", 0.7, 0.5);
        }

        assert!(nac.state().hedonic_baseline > initial_baseline);
    }

    #[test]
    fn effort_tradeoff_evaluation() {
        let mut nac = NucleusAccumbens::new();
        nac.set_dopamine(0.8); // High dopamine = more willing

        let high_reward = nac.evaluate_effort_tradeoff(0.9, 0.3);
        let low_reward = nac.evaluate_effort_tradeoff(0.2, 0.8);

        assert!(high_reward > low_reward);
        assert!(high_reward > 0.0);
    }

    #[test]
    fn rest_restores_toward_baseline() {
        let mut nac = NucleusAccumbens::new();

        // Drive state to extreme
        for _ in 0..10 {
            nac.process_reward("excess", 0.9, 0.1);
        }

        let pre_rest_wanting = nac.state().wanting;
        nac.rest();

        assert!(nac.state().wanting < pre_rest_wanting || nac.state().wanting <= 0.5);
    }

    #[test]
    fn generates_reward_signal() {
        let nac = NucleusAccumbens::new();
        let signal = nac.to_signal();
        assert_eq!(signal.signal_type, SignalType::Reward);
    }
}
