//! Neuroplasticity - Learning Through Connection Strength Changes
//!
//! Real brains learn by modifying the strength of connections between neurons.
//! This module implements the core plasticity mechanisms:
//!
//! - **Hebbian Learning**: "Neurons that fire together wire together"
//! - **Long-Term Potentiation (LTP)**: Strengthening of frequently co-activated pathways
//! - **Long-Term Depression (LTD)**: Weakening of infrequently used pathways
//! - **Homeostatic Plasticity**: Maintaining overall network stability
//! - **Spike-Timing Dependent Plasticity (STDP)**: Order-sensitive learning
//!
//! # Theoretical Foundation
//!
//! Based on:
//! - Hebb (1949): Cell assemblies and synaptic learning rules
//! - Bliss & Lomo (1973): Long-term potentiation in hippocampus
//! - Turrigiano (2008): Homeostatic synaptic plasticity
//! - Bi & Poo (1998): STDP temporal asymmetry
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │                 NEUROPLASTICITY ENGINE                   │
//! ├──────────────┬──────────────┬──────────────┬────────────┤
//! │   Hebbian    │     LTP      │  Homeostatic │    STDP    │
//! │  Learning    │    / LTD     │  Regulation  │  Temporal  │
//! │ (co-firing)  │ (frequency)  │ (stability)  │  (order)   │
//! └──────────────┴──────────────┴──────────────┴────────────┘
//! ```

use crate::core::nervous_system::BrainRegion;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

/// Configuration for the plasticity engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlasticityConfig {
    /// Hebbian learning rate (how fast co-activation strengthens connections)
    pub hebbian_rate: f64,
    /// LTD rate (how fast disuse weakens connections)
    pub depression_rate: f64,
    /// Homeostatic target: desired average pathway strength
    pub homeostatic_target: f64,
    /// Homeostatic correction rate
    pub homeostatic_rate: f64,
    /// STDP temporal window in milliseconds
    pub stdp_window_ms: i64,
    /// Maximum pathway strength (prevents runaway potentiation)
    pub max_strength: f64,
    /// Minimum pathway strength (prevents total disconnection)
    pub min_strength: f64,
    /// How many activation events to remember for STDP
    pub activation_history_size: usize,
    /// Minimum co-activations before Hebbian learning kicks in
    pub hebbian_threshold: u32,
    /// Enable consolidation (periodic strengthening of stable changes)
    pub enable_consolidation: bool,
}

impl Default for PlasticityConfig {
    fn default() -> Self {
        Self {
            hebbian_rate: 0.01,
            depression_rate: 0.002,
            homeostatic_target: 0.6,
            homeostatic_rate: 0.001,
            stdp_window_ms: 50,
            max_strength: 1.0,
            min_strength: 0.05,
            activation_history_size: 100,
            hebbian_threshold: 3,
            enable_consolidation: true,
        }
    }
}

/// Record of a region activation (for STDP and co-activation tracking).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationEvent {
    pub region: BrainRegion,
    pub timestamp: DateTime<Utc>,
    pub strength: f64,
    pub signal_source: String,
}

/// A proposed change to a pathway's strength.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlasticityUpdate {
    pub from: BrainRegion,
    pub to: BrainRegion,
    /// Signed delta: positive = potentiation, negative = depression
    pub delta: f64,
    pub mechanism: PlasticityMechanism,
    pub timestamp: DateTime<Utc>,
}

/// Which plasticity mechanism produced a change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlasticityMechanism {
    /// Co-activation strengthening
    Hebbian,
    /// Frequency-dependent potentiation
    LTP,
    /// Disuse weakening
    LTD,
    /// Network stability correction
    Homeostatic,
    /// Timing-dependent modification
    STDP,
    /// Periodic consolidation of stable changes
    Consolidation,
}

/// Co-activation tracking between two regions.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CoActivationTracker {
    count: u32,
    last_co_activation: DateTime<Utc>,
    /// Running average of time between co-activations
    avg_interval_ms: f64,
}

impl CoActivationTracker {
    fn new() -> Self {
        Self {
            count: 0,
            last_co_activation: Utc::now(),
            avg_interval_ms: 1000.0,
        }
    }

    fn record(&mut self, now: DateTime<Utc>) {
        let interval = (now - self.last_co_activation).num_milliseconds() as f64;
        self.avg_interval_ms = self.avg_interval_ms * 0.8 + interval * 0.2;
        self.count += 1;
        self.last_co_activation = now;
    }
}

/// Statistics about plasticity activity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlasticityStats {
    pub total_updates: u64,
    pub ltp_count: u64,
    pub ltd_count: u64,
    pub hebbian_count: u64,
    pub stdp_count: u64,
    pub homeostatic_count: u64,
    pub consolidation_count: u64,
    pub avg_network_strength: f64,
    pub strongest_pathway: Option<(String, String, f64)>,
    pub weakest_pathway: Option<(String, String, f64)>,
}

/// The neuroplasticity engine.
///
/// Monitors brain region activations and produces pathway strength
/// modifications based on multiple learning rules.
pub struct PlasticityEngine {
    config: PlasticityConfig,
    /// Recent activation events per region
    activation_history: HashMap<BrainRegion, VecDeque<ActivationEvent>>,
    /// Co-activation counts between region pairs
    co_activations: HashMap<(BrainRegion, BrainRegion), CoActivationTracker>,
    /// Pending updates to apply to the nervous system
    pending_updates: Vec<PlasticityUpdate>,
    /// Current pathway strengths (mirror of nervous system for delta computation)
    pathway_strengths: HashMap<(BrainRegion, BrainRegion), f64>,
    /// Cumulative changes since last consolidation
    cumulative_changes: HashMap<(BrainRegion, BrainRegion), f64>,
    /// Cycles since last consolidation
    cycles_since_consolidation: u64,
    /// Statistics
    stats: PlasticityStatsInternal,
}

#[derive(Debug, Default)]
struct PlasticityStatsInternal {
    total_updates: u64,
    ltp_count: u64,
    ltd_count: u64,
    hebbian_count: u64,
    stdp_count: u64,
    homeostatic_count: u64,
    consolidation_count: u64,
}

impl PlasticityEngine {
    /// Create a new plasticity engine with default configuration.
    pub fn new() -> Self {
        Self::with_config(PlasticityConfig::default())
    }

    /// Create with custom configuration.
    pub fn with_config(config: PlasticityConfig) -> Self {
        Self {
            config,
            activation_history: HashMap::new(),
            co_activations: HashMap::new(),
            pending_updates: Vec::new(),
            pathway_strengths: HashMap::new(),
            cumulative_changes: HashMap::new(),
            cycles_since_consolidation: 0,
            stats: PlasticityStatsInternal::default(),
        }
    }

    /// Register a pathway so the engine can track its strength.
    pub fn register_pathway(&mut self, from: BrainRegion, to: BrainRegion, strength: f64) {
        self.pathway_strengths.insert((from, to), strength);
    }

    /// Record that a brain region was activated.
    ///
    /// This is the primary input to the plasticity system. Each time a region
    /// processes a signal, it should call this to enable learning.
    pub fn record_activation(&mut self, region: BrainRegion, strength: f64, signal_source: &str) {
        let event = ActivationEvent {
            region,
            timestamp: Utc::now(),
            strength: strength.clamp(0.0, 1.0),
            signal_source: signal_source.to_string(),
        };

        let history = self
            .activation_history
            .entry(region)
            .or_insert_with(|| VecDeque::with_capacity(self.config.activation_history_size));

        if history.len() >= self.config.activation_history_size {
            history.pop_front();
        }
        history.push_back(event);

        // Check for co-activations with recently active regions
        self.detect_co_activations(region);
    }

    /// Detect co-activations: when two regions fire within the STDP window.
    fn detect_co_activations(&mut self, just_activated: BrainRegion) {
        let now = Utc::now();
        let window = self.config.stdp_window_ms;

        // Collect recently active regions (within STDP window)
        let mut recent_partners: Vec<(BrainRegion, DateTime<Utc>)> = Vec::new();

        for (region, history) in &self.activation_history {
            if *region == just_activated {
                continue;
            }
            if let Some(last) = history.back() {
                let elapsed = (now - last.timestamp).num_milliseconds();
                if elapsed <= window {
                    recent_partners.push((*region, last.timestamp));
                }
            }
        }

        // Record co-activations
        for (partner, _partner_time) in recent_partners {
            let key = normalize_pair(just_activated, partner);
            let tracker = self
                .co_activations
                .entry(key)
                .or_insert_with(CoActivationTracker::new);
            tracker.record(now);
        }
    }

    /// Run one plasticity cycle: compute all learning updates.
    ///
    /// Call this periodically (e.g., every N brain processing cycles).
    /// Returns the updates that should be applied to the nervous system.
    pub fn compute_updates(&mut self) -> Vec<PlasticityUpdate> {
        self.pending_updates.clear();
        self.cycles_since_consolidation += 1;

        self.compute_hebbian_updates();
        self.compute_ltd_updates();
        self.compute_homeostatic_updates();
        self.compute_stdp_updates();

        if self.config.enable_consolidation && self.cycles_since_consolidation >= 100 {
            self.compute_consolidation_updates();
            self.cycles_since_consolidation = 0;
        }

        self.pending_updates.clone()
    }

    /// Hebbian learning: strengthen pathways between co-activated regions.
    fn compute_hebbian_updates(&mut self) {
        let threshold = self.config.hebbian_threshold;
        let rate = self.config.hebbian_rate;
        let max = self.config.max_strength;

        let co_activations: Vec<_> = self
            .co_activations
            .iter()
            .filter(|(_, tracker)| tracker.count >= threshold)
            .map(|(key, tracker)| (*key, tracker.count, tracker.avg_interval_ms))
            .collect();

        for ((from, to), count, avg_interval) in co_activations {
            // Stronger learning for more frequent co-activations
            let frequency_factor = (count as f64 / 10.0).min(2.0);
            // Faster co-activations = stronger learning
            let speed_factor = (1000.0 / avg_interval.max(1.0)).min(2.0);

            let delta = rate * frequency_factor * speed_factor;

            let current = self
                .pathway_strengths
                .get(&(from, to))
                .copied()
                .unwrap_or(0.5);

            if current + delta <= max {
                self.pending_updates.push(PlasticityUpdate {
                    from,
                    to,
                    delta,
                    mechanism: PlasticityMechanism::Hebbian,
                    timestamp: Utc::now(),
                });
                *self.cumulative_changes.entry((from, to)).or_insert(0.0) += delta;
                self.stats.hebbian_count += 1;
                self.stats.total_updates += 1;
            }
        }
    }

    /// LTD: weaken pathways that haven't been used recently.
    fn compute_ltd_updates(&mut self) {
        let rate = self.config.depression_rate;
        let min = self.config.min_strength;

        let pathways: Vec<_> = self
            .pathway_strengths
            .iter()
            .map(|(k, v)| (*k, *v))
            .collect();

        for ((from, to), strength) in pathways {
            // Check if this pathway was recently active
            let recently_active = self
                .co_activations
                .get(&normalize_pair(from, to))
                .map(|t| {
                    let elapsed = (Utc::now() - t.last_co_activation).num_seconds();
                    elapsed < 60
                })
                .unwrap_or(false);

            if !recently_active && strength > min {
                let delta = -rate;
                self.pending_updates.push(PlasticityUpdate {
                    from,
                    to,
                    delta,
                    mechanism: PlasticityMechanism::LTD,
                    timestamp: Utc::now(),
                });
                *self.cumulative_changes.entry((from, to)).or_insert(0.0) += delta;
                self.stats.ltd_count += 1;
                self.stats.total_updates += 1;
            }
        }
    }

    /// Homeostatic plasticity: keep network average strength near target.
    fn compute_homeostatic_updates(&mut self) {
        if self.pathway_strengths.is_empty() {
            return;
        }

        let avg_strength: f64 =
            self.pathway_strengths.values().sum::<f64>() / self.pathway_strengths.len() as f64;
        let deviation = self.config.homeostatic_target - avg_strength;

        // Only apply if deviation is significant
        if deviation.abs() < 0.01 {
            return;
        }

        let correction = deviation * self.config.homeostatic_rate;

        let pathways: Vec<_> = self.pathway_strengths.keys().copied().collect();
        for (from, to) in pathways {
            self.pending_updates.push(PlasticityUpdate {
                from,
                to,
                delta: correction,
                mechanism: PlasticityMechanism::Homeostatic,
                timestamp: Utc::now(),
            });
            self.stats.homeostatic_count += 1;
            self.stats.total_updates += 1;
        }
    }

    /// STDP: strengthen if pre fires before post, weaken if post fires first.
    fn compute_stdp_updates(&mut self) {
        let window = self.config.stdp_window_ms;
        let rate = self.config.hebbian_rate * 0.5; // STDP is more subtle

        let pathway_keys: Vec<_> = self.pathway_strengths.keys().copied().collect();

        for (from, to) in pathway_keys {
            let from_time = self
                .activation_history
                .get(&from)
                .and_then(|h| h.back())
                .map(|e| e.timestamp);
            let to_time = self
                .activation_history
                .get(&to)
                .and_then(|h| h.back())
                .map(|e| e.timestamp);

            if let (Some(ft), Some(tt)) = (from_time, to_time) {
                let dt = (tt - ft).num_milliseconds();

                // Within STDP window
                if dt.abs() <= window {
                    let delta = if dt > 0 {
                        // Pre before post: potentiation (causal)
                        rate * (1.0 - dt.abs() as f64 / window as f64)
                    } else {
                        // Post before pre: depression (anti-causal)
                        -rate * 0.5 * (1.0 - dt.abs() as f64 / window as f64)
                    };

                    if delta.abs() > 0.0001 {
                        self.pending_updates.push(PlasticityUpdate {
                            from,
                            to,
                            delta,
                            mechanism: PlasticityMechanism::STDP,
                            timestamp: Utc::now(),
                        });
                        self.stats.stdp_count += 1;
                        self.stats.total_updates += 1;
                    }
                }
            }
        }
    }

    /// Consolidation: reinforce changes that have been consistently applied.
    fn compute_consolidation_updates(&mut self) {
        let changes: Vec<_> = self
            .cumulative_changes
            .iter()
            .filter(|(_, delta)| delta.abs() > 0.02) // Only consolidate significant changes
            .map(|(key, delta)| (*key, *delta))
            .collect();

        for ((from, to), cumulative_delta) in changes {
            // Consolidation makes a fraction of the cumulative change permanent
            let consolidation_delta = cumulative_delta * 0.3;
            self.pending_updates.push(PlasticityUpdate {
                from,
                to,
                delta: consolidation_delta,
                mechanism: PlasticityMechanism::Consolidation,
                timestamp: Utc::now(),
            });
            self.stats.consolidation_count += 1;
            self.stats.total_updates += 1;
        }

        // Reset cumulative tracking after consolidation
        self.cumulative_changes.clear();
    }

    /// Apply a plasticity update to the internal strength mirror.
    /// Call this after applying the update to the actual nervous system.
    pub fn apply_update(&mut self, update: &PlasticityUpdate) {
        let key = (update.from, update.to);
        if let Some(strength) = self.pathway_strengths.get_mut(&key) {
            *strength = (*strength + update.delta)
                .clamp(self.config.min_strength, self.config.max_strength);
        }
    }

    /// Trigger LTP for a specific pathway (e.g., after a reward signal).
    pub fn potentiate(&mut self, from: BrainRegion, to: BrainRegion, magnitude: f64) {
        let delta = self.config.hebbian_rate * magnitude.clamp(0.0, 5.0);
        let update = PlasticityUpdate {
            from,
            to,
            delta,
            mechanism: PlasticityMechanism::LTP,
            timestamp: Utc::now(),
        };
        self.apply_update(&update);
        self.pending_updates.push(update);
        self.stats.ltp_count += 1;
        self.stats.total_updates += 1;
    }

    /// Trigger LTD for a specific pathway (e.g., after a negative prediction error).
    pub fn depress(&mut self, from: BrainRegion, to: BrainRegion, magnitude: f64) {
        let delta = -self.config.depression_rate * magnitude.clamp(0.0, 5.0);
        let update = PlasticityUpdate {
            from,
            to,
            delta,
            mechanism: PlasticityMechanism::LTD,
            timestamp: Utc::now(),
        };
        self.apply_update(&update);
        self.pending_updates.push(update);
        self.stats.ltd_count += 1;
        self.stats.total_updates += 1;
    }

    /// Get statistics about plasticity activity.
    pub fn stats(&self) -> PlasticityStats {
        let avg_strength = if self.pathway_strengths.is_empty() {
            0.0
        } else {
            self.pathway_strengths.values().sum::<f64>() / self.pathway_strengths.len() as f64
        };

        let strongest = self
            .pathway_strengths
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|((f, t), s)| (f.name().to_string(), t.name().to_string(), *s));

        let weakest = self
            .pathway_strengths
            .iter()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|((f, t), s)| (f.name().to_string(), t.name().to_string(), *s));

        PlasticityStats {
            total_updates: self.stats.total_updates,
            ltp_count: self.stats.ltp_count,
            ltd_count: self.stats.ltd_count,
            hebbian_count: self.stats.hebbian_count,
            stdp_count: self.stats.stdp_count,
            homeostatic_count: self.stats.homeostatic_count,
            consolidation_count: self.stats.consolidation_count,
            avg_network_strength: avg_strength,
            strongest_pathway: strongest,
            weakest_pathway: weakest,
        }
    }

    /// Get the current tracked strength of a pathway.
    pub fn pathway_strength(&self, from: BrainRegion, to: BrainRegion) -> Option<f64> {
        self.pathway_strengths.get(&(from, to)).copied()
    }

    /// Get co-activation count between two regions.
    pub fn co_activation_count(&self, a: BrainRegion, b: BrainRegion) -> u32 {
        self.co_activations
            .get(&normalize_pair(a, b))
            .map(|t| t.count)
            .unwrap_or(0)
    }
}

impl Default for PlasticityEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Normalize a region pair so (A, B) and (B, A) map to the same key.
fn normalize_pair(a: BrainRegion, b: BrainRegion) -> (BrainRegion, BrainRegion) {
    if (a as u8) <= (b as u8) {
        (a, b)
    } else {
        (b, a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_engine() -> PlasticityEngine {
        let mut engine = PlasticityEngine::new();
        engine.register_pathway(BrainRegion::Thalamus, BrainRegion::Amygdala, 0.5);
        engine.register_pathway(BrainRegion::Amygdala, BrainRegion::Hippocampus, 0.5);
        engine.register_pathway(BrainRegion::Hippocampus, BrainRegion::Prefrontal, 0.5);
        engine
    }

    #[test]
    fn test_creation() {
        let engine = PlasticityEngine::new();
        let stats = engine.stats();
        assert_eq!(stats.total_updates, 0);
        assert_eq!(stats.avg_network_strength, 0.0);
    }

    #[test]
    fn test_activation_recording() {
        let mut engine = setup_engine();
        engine.record_activation(BrainRegion::Thalamus, 0.8, "test");
        engine.record_activation(BrainRegion::Amygdala, 0.7, "test");

        let count = engine.co_activation_count(BrainRegion::Thalamus, BrainRegion::Amygdala);
        assert!(count > 0, "Co-activation should be detected");
    }

    #[test]
    fn test_hebbian_learning() {
        let mut engine = PlasticityEngine::with_config(PlasticityConfig {
            hebbian_threshold: 2,
            ..Default::default()
        });
        engine.register_pathway(BrainRegion::Thalamus, BrainRegion::Amygdala, 0.5);

        // Fire both regions multiple times
        for _ in 0..5 {
            engine.record_activation(BrainRegion::Thalamus, 0.8, "test");
            engine.record_activation(BrainRegion::Amygdala, 0.7, "test");
        }

        let updates = engine.compute_updates();
        let hebbian_updates: Vec<_> = updates
            .iter()
            .filter(|u| u.mechanism == PlasticityMechanism::Hebbian)
            .collect();

        assert!(
            !hebbian_updates.is_empty(),
            "Should produce Hebbian updates after co-activation"
        );
        assert!(
            hebbian_updates.iter().all(|u| u.delta > 0.0),
            "Hebbian updates should be positive (strengthening)"
        );
    }

    #[test]
    fn test_ltp_potentiation() {
        let mut engine = setup_engine();
        let initial = engine
            .pathway_strength(BrainRegion::Thalamus, BrainRegion::Amygdala)
            .unwrap();

        engine.potentiate(BrainRegion::Thalamus, BrainRegion::Amygdala, 2.0);

        let after = engine
            .pathway_strength(BrainRegion::Thalamus, BrainRegion::Amygdala)
            .unwrap();
        assert!(
            after > initial,
            "LTP should increase pathway strength: {} > {}",
            after,
            initial
        );
    }

    #[test]
    fn test_ltd_depression() {
        let mut engine = setup_engine();
        let initial = engine
            .pathway_strength(BrainRegion::Thalamus, BrainRegion::Amygdala)
            .unwrap();

        engine.depress(BrainRegion::Thalamus, BrainRegion::Amygdala, 2.0);

        let after = engine
            .pathway_strength(BrainRegion::Thalamus, BrainRegion::Amygdala)
            .unwrap();
        assert!(
            after < initial,
            "LTD should decrease pathway strength: {} < {}",
            after,
            initial
        );
    }

    #[test]
    fn test_strength_bounds() {
        let mut engine = PlasticityEngine::with_config(PlasticityConfig {
            max_strength: 0.9,
            min_strength: 0.1,
            ..Default::default()
        });
        engine.register_pathway(BrainRegion::Thalamus, BrainRegion::Amygdala, 0.5);

        // Try to potentiate beyond max
        for _ in 0..100 {
            engine.potentiate(BrainRegion::Thalamus, BrainRegion::Amygdala, 5.0);
        }
        let strength = engine
            .pathway_strength(BrainRegion::Thalamus, BrainRegion::Amygdala)
            .unwrap();
        assert!(
            strength <= 0.9,
            "Strength should not exceed max: {}",
            strength
        );

        // Try to depress below min
        for _ in 0..100 {
            engine.depress(BrainRegion::Thalamus, BrainRegion::Amygdala, 5.0);
        }
        let strength = engine
            .pathway_strength(BrainRegion::Thalamus, BrainRegion::Amygdala)
            .unwrap();
        assert!(
            strength >= 0.1,
            "Strength should not go below min: {}",
            strength
        );
    }

    #[test]
    fn test_homeostatic_regulation() {
        let mut engine = PlasticityEngine::with_config(PlasticityConfig {
            homeostatic_target: 0.5,
            homeostatic_rate: 0.1, // High rate for testing
            ..Default::default()
        });

        // Register pathways that are too strong on average
        engine.register_pathway(BrainRegion::Thalamus, BrainRegion::Amygdala, 0.9);
        engine.register_pathway(BrainRegion::Amygdala, BrainRegion::Hippocampus, 0.9);

        let updates = engine.compute_updates();
        let homeostatic_updates: Vec<_> = updates
            .iter()
            .filter(|u| u.mechanism == PlasticityMechanism::Homeostatic)
            .collect();

        assert!(
            !homeostatic_updates.is_empty(),
            "Should produce homeostatic corrections"
        );
        assert!(
            homeostatic_updates.iter().all(|u| u.delta < 0.0),
            "Corrections should be negative when avg is above target"
        );
    }

    #[test]
    fn test_stats() {
        let mut engine = setup_engine();
        engine.potentiate(BrainRegion::Thalamus, BrainRegion::Amygdala, 1.0);
        engine.depress(BrainRegion::Amygdala, BrainRegion::Hippocampus, 1.0);

        let stats = engine.stats();
        assert_eq!(stats.ltp_count, 1);
        assert_eq!(stats.ltd_count, 1);
        assert_eq!(stats.total_updates, 2);
        assert!(stats.strongest_pathway.is_some());
        assert!(stats.weakest_pathway.is_some());
    }

    #[test]
    fn test_co_activation_symmetry() {
        let mut engine = setup_engine();
        engine.record_activation(BrainRegion::Thalamus, 0.8, "test");
        engine.record_activation(BrainRegion::Amygdala, 0.7, "test");

        // Both orderings should give the same count
        let count_ab = engine.co_activation_count(BrainRegion::Thalamus, BrainRegion::Amygdala);
        let count_ba = engine.co_activation_count(BrainRegion::Amygdala, BrainRegion::Thalamus);
        assert_eq!(count_ab, count_ba);
    }
}
