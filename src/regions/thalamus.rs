//! Thalamus - Sensory Gateway & Attention Router
//!
//! The thalamus is the relay station of the brain. Almost all sensory
//! information passes through it before reaching the cortex.
//!
//! Key functions:
//! - Sensory gating (filter irrelevant input)
//! - Attention routing (direct signals to appropriate modules)
//! - Arousal modulation
//! - Binding disparate signals

#[allow(unused_imports)]
use crate::signal::{Arousal, BrainSignal, Salience, SignalType};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

/// Configuration for thalamic processing.
#[derive(Debug, Clone)]
pub struct ThalamusConfig {
    /// Minimum salience to pass the gate
    pub gate_threshold: f64,
    /// Maximum signals to process per cycle
    pub max_throughput: usize,
    /// How much emotional signals are boosted
    pub emotional_boost: f64,
    /// Buffer size for incoming signals
    pub buffer_size: usize,
    /// Habituation rate (repeated signals become less salient)
    pub habituation_rate: f64,
}

impl Default for ThalamusConfig {
    fn default() -> Self {
        Self {
            gate_threshold: 0.2,
            max_throughput: 20,
            emotional_boost: 0.3,
            buffer_size: 100,
            habituation_rate: 0.1,
        }
    }
}

/// Result of gating a signal.
#[derive(Debug, Clone)]
pub enum GateResult {
    /// Signal passed the gate
    Passed(BrainSignal),
    /// Signal was filtered out
    Filtered { reason: FilterReason, salience: f64 },
}

/// Reasons a signal might be filtered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterReason {
    /// Below salience threshold
    BelowThreshold,
    /// Habituated (seen too many similar signals)
    Habituated,
    /// Buffer overflow
    BufferFull,
    /// Explicitly suppressed
    Suppressed,
}

/// Routing destination for signals.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Destination {
    /// Send to hippocampus for memory encoding
    Hippocampus,
    /// Send to amygdala for emotional processing
    Amygdala,
    /// Send to prefrontal for working memory
    Prefrontal,
    /// Send to global workspace for broadcast
    Workspace,
    /// Send to prediction engine
    Prediction,
    /// Custom module
    Custom(String),
}

/// A routed signal with destination.
#[derive(Debug, Clone)]
pub struct RoutedSignal {
    pub signal: BrainSignal,
    pub destinations: Vec<Destination>,
    pub priority: i32,
    pub routed_at: DateTime<Utc>,
}

/// Habituation tracker for repeated stimuli.
#[derive(Debug, Clone, Default)]
struct HabituationTracker {
    /// Content hash -> exposure count
    exposures: HashMap<u64, usize>,
    /// Content hash -> last seen time
    last_seen: HashMap<u64, DateTime<Utc>>,
}

impl HabituationTracker {
    fn hash_content(content: &serde_json::Value) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        content.to_string().hash(&mut hasher);
        hasher.finish()
    }

    fn record_exposure(&mut self, content: &serde_json::Value) {
        let hash = Self::hash_content(content);
        *self.exposures.entry(hash).or_insert(0) += 1;
        self.last_seen.insert(hash, Utc::now());
    }

    fn get_habituation(&self, content: &serde_json::Value) -> f64 {
        let hash = Self::hash_content(content);
        let exposures = self.exposures.get(&hash).copied().unwrap_or(0);

        // Habituation increases with exposures, caps at 0.8
        (exposures as f64 * 0.1).min(0.8)
    }

    fn decay(&mut self, rate: f64) {
        // Reduce exposure counts over time
        for count in self.exposures.values_mut() {
            *count = (*count as f64 * (1.0 - rate)).floor() as usize;
        }

        // Remove zero-exposure entries
        self.exposures.retain(|_, v| *v > 0);
    }
}

/// The thalamus - sensory gateway and attention router.
pub struct Thalamus {
    config: ThalamusConfig,
    /// Input buffer
    input_buffer: VecDeque<BrainSignal>,
    /// Habituation state
    habituation: HabituationTracker,
    /// Suppressed signal types
    suppressed_types: Vec<SignalType>,
    /// Current attention focus (boosts matching signals)
    attention_focus: Option<String>,
    /// Processing statistics
    stats: ThalamusStats,
}

#[derive(Debug, Clone, Default)]
struct ThalamusStats {
    signals_received: u64,
    signals_passed: u64,
    signals_filtered: u64,
    cycles_processed: u64,
}

impl Thalamus {
    /// Create a new thalamus with default config.
    pub fn new() -> Self {
        Self::with_config(ThalamusConfig::default())
    }

    /// Create with custom config.
    pub fn with_config(config: ThalamusConfig) -> Self {
        Self {
            config,
            input_buffer: VecDeque::new(),
            habituation: HabituationTracker::default(),
            suppressed_types: Vec::new(),
            attention_focus: None,
            stats: ThalamusStats::default(),
        }
    }

    /// Receive an incoming signal into the buffer.
    pub fn receive(&mut self, signal: BrainSignal) -> bool {
        self.stats.signals_received += 1;

        if self.input_buffer.len() >= self.config.buffer_size {
            return false;
        }

        self.input_buffer.push_back(signal);
        true
    }

    /// Gate a single signal (decide if it passes).
    pub fn gate(&mut self, signal: BrainSignal) -> GateResult {
        // Check suppression
        if self.suppressed_types.contains(&signal.signal_type) {
            self.stats.signals_filtered += 1;
            return GateResult::Filtered {
                reason: FilterReason::Suppressed,
                salience: signal.salience.value(),
            };
        }

        // Calculate effective salience
        let mut effective_salience = signal.salience.value();

        // Emotional boost
        effective_salience += signal.emotional_intensity() * self.config.emotional_boost;

        // Attention focus boost
        if let Some(ref focus) = self.attention_focus
            && signal.content.to_string().contains(focus)
        {
            effective_salience += 0.2;
        }

        // Habituation penalty
        let habituation = self.habituation.get_habituation(&signal.content);
        effective_salience -= habituation;

        // Check threshold
        if effective_salience < self.config.gate_threshold {
            self.stats.signals_filtered += 1;

            let reason = if habituation > 0.3 {
                FilterReason::Habituated
            } else {
                FilterReason::BelowThreshold
            };

            return GateResult::Filtered {
                reason,
                salience: effective_salience,
            };
        }

        // Record exposure for habituation
        self.habituation.record_exposure(&signal.content);

        // Boost salience on output
        let mut passed_signal = signal;
        passed_signal.salience = Salience::new(effective_salience);

        self.stats.signals_passed += 1;
        GateResult::Passed(passed_signal)
    }

    /// Route a signal to appropriate destinations.
    pub fn route(&self, signal: &BrainSignal) -> RoutedSignal {
        let mut destinations = Vec::new();

        // Route based on signal type
        match signal.signal_type {
            SignalType::Sensory => {
                destinations.push(Destination::Amygdala); // Emotional tagging first
                if signal.salience.is_high() {
                    destinations.push(Destination::Workspace);
                }
            }
            SignalType::Memory => {
                destinations.push(Destination::Hippocampus);
                destinations.push(Destination::Prefrontal);
            }
            SignalType::Prediction | SignalType::Error => {
                destinations.push(Destination::Prediction);
                if signal.is_surprising() {
                    destinations.push(Destination::Workspace);
                    destinations.push(Destination::Hippocampus);
                }
            }
            SignalType::Emotion => {
                destinations.push(Destination::Prefrontal);
                destinations.push(Destination::Hippocampus);
            }
            SignalType::Attention | SignalType::Broadcast => {
                destinations.push(Destination::Workspace);
                destinations.push(Destination::Prefrontal);
            }
            SignalType::Query => {
                destinations.push(Destination::Hippocampus);
                destinations.push(Destination::Prediction);
            }
            SignalType::Motor => {
                // Motor signals typically don't need internal routing
                destinations.push(Destination::Prefrontal);
            }
        }

        // High emotional content always goes to hippocampus
        if signal.emotional_intensity() > 0.5 && !destinations.contains(&Destination::Hippocampus) {
            destinations.push(Destination::Hippocampus);
        }

        RoutedSignal {
            signal: signal.clone(),
            destinations,
            priority: signal.priority,
            routed_at: Utc::now(),
        }
    }

    /// Process one cycle: gate and route buffered signals.
    pub fn process_cycle(&mut self) -> Vec<RoutedSignal> {
        self.stats.cycles_processed += 1;
        let mut routed = Vec::new();

        // Process up to max_throughput signals
        let to_process: Vec<_> = self
            .input_buffer
            .drain(..self.config.max_throughput.min(self.input_buffer.len()))
            .collect();

        for signal in to_process {
            if let GateResult::Passed(passed) = self.gate(signal) {
                routed.push(self.route(&passed));
            }
        }

        // Decay habituation
        self.habituation.decay(self.config.habituation_rate);

        routed
    }

    /// Set attention focus (boosts matching signals).
    pub fn focus_attention(&mut self, focus: impl Into<String>) {
        self.attention_focus = Some(focus.into());
    }

    /// Clear attention focus.
    pub fn clear_focus(&mut self) {
        self.attention_focus = None;
    }

    /// Suppress a signal type (filter all of this type).
    pub fn suppress(&mut self, signal_type: SignalType) {
        if !self.suppressed_types.contains(&signal_type) {
            self.suppressed_types.push(signal_type);
        }
    }

    /// Unsuppress a signal type.
    pub fn unsuppress(&mut self, signal_type: SignalType) {
        self.suppressed_types.retain(|t| *t != signal_type);
    }

    /// Get statistics.
    pub fn stats(&self) -> (u64, u64, u64, u64) {
        (
            self.stats.signals_received,
            self.stats.signals_passed,
            self.stats.signals_filtered,
            self.stats.cycles_processed,
        )
    }

    /// Get buffer size.
    pub fn buffer_len(&self) -> usize {
        self.input_buffer.len()
    }
}

impl Default for Thalamus {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gate_threshold() {
        let mut thalamus = Thalamus::new();

        // Low salience should be filtered
        let low = BrainSignal::new("test", SignalType::Sensory, "low").with_salience(0.1);
        assert!(matches!(thalamus.gate(low), GateResult::Filtered { .. }));

        // High salience should pass
        let high = BrainSignal::new("test", SignalType::Sensory, "high").with_salience(0.8);
        assert!(matches!(thalamus.gate(high), GateResult::Passed(_)));
    }

    #[test]
    fn test_emotional_boost() {
        let mut thalamus = Thalamus::new();

        // Borderline salience without emotion
        let neutral = BrainSignal::new("test", SignalType::Sensory, "neutral")
            .with_salience(0.15)
            .with_valence(0.0);
        assert!(matches!(
            thalamus.gate(neutral),
            GateResult::Filtered { .. }
        ));

        // Same salience but with emotion should pass
        let emotional = BrainSignal::new("test", SignalType::Sensory, "emotional")
            .with_salience(0.15)
            .with_valence(0.9)
            .with_arousal(0.8);
        assert!(matches!(thalamus.gate(emotional), GateResult::Passed(_)));
    }

    #[test]
    fn test_habituation() {
        let mut thalamus = Thalamus::new();

        // First exposure passes easily
        let signal =
            BrainSignal::new("test", SignalType::Sensory, "repeated content").with_salience(0.5);
        assert!(matches!(
            thalamus.gate(signal.clone()),
            GateResult::Passed(_)
        ));

        // Multiple exposures increase habituation
        for _ in 0..10 {
            thalamus.gate(signal.clone());
        }

        // Now it might be filtered due to habituation
        let result = thalamus.gate(signal.clone());
        // Salience + habituation penalty might drop below threshold
        if let GateResult::Filtered { reason, .. } = result {
            assert_eq!(reason, FilterReason::Habituated);
        }
    }

    #[test]
    fn test_attention_focus() {
        let mut thalamus = Thalamus::new();

        // Set focus on "important"
        thalamus.focus_attention("important");

        // Borderline signal without focus keyword
        let _unfocused =
            BrainSignal::new("test", SignalType::Sensory, "regular content").with_salience(0.15);

        // Borderline signal with focus keyword
        let focused =
            BrainSignal::new("test", SignalType::Sensory, "important content").with_salience(0.15);

        // Focused signal should get boosted and pass
        assert!(matches!(thalamus.gate(focused), GateResult::Passed(_)));
    }

    #[test]
    fn test_suppression() {
        let mut thalamus = Thalamus::new();

        // Suppress motor signals
        thalamus.suppress(SignalType::Motor);

        let motor = BrainSignal::new("test", SignalType::Motor, "move").with_salience(0.9);

        if let GateResult::Filtered { reason, .. } = thalamus.gate(motor) {
            assert_eq!(reason, FilterReason::Suppressed);
        } else {
            panic!("Motor signal should have been suppressed");
        }
    }

    #[test]
    fn test_routing() {
        let thalamus = Thalamus::new();

        // Sensory signal should go to amygdala
        let sensory = BrainSignal::new("test", SignalType::Sensory, "input").with_salience(0.5);
        let routed = thalamus.route(&sensory);
        assert!(routed.destinations.contains(&Destination::Amygdala));

        // Surprising error should go to workspace and hippocampus
        let error = BrainSignal::new("test", SignalType::Error, "surprise!")
            .with_salience(0.9)
            .with_arousal(0.9);
        let routed = thalamus.route(&error);
        assert!(routed.destinations.contains(&Destination::Workspace));
        assert!(routed.destinations.contains(&Destination::Hippocampus));
    }

    #[test]
    fn test_process_cycle() {
        let mut thalamus = Thalamus::new();

        // Add signals to buffer
        for i in 0..5 {
            let signal = BrainSignal::new("test", SignalType::Sensory, format!("signal_{}", i))
                .with_salience(0.5);
            thalamus.receive(signal);
        }

        assert_eq!(thalamus.buffer_len(), 5);

        // Process cycle
        let routed = thalamus.process_cycle();

        // Should have processed all signals
        assert_eq!(routed.len(), 5);
        assert_eq!(thalamus.buffer_len(), 0);
    }
}
