//! Brain Signal Protocol
//!
//! The universal message format for inter-module communication.
//! All brain regions speak this language.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Valence: emotional coloring from negative to positive.
/// Constrained to [-1.0, 1.0] at the type level.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Valence(f64);

impl Valence {
    /// Create a new Valence, clamping to valid range.
    pub fn new(value: f64) -> Self {
        Self(value.clamp(-1.0, 1.0))
    }

    /// Get the raw value.
    pub fn value(&self) -> f64 {
        self.0
    }

    /// Absolute emotional intensity (ignoring direction).
    pub fn intensity(&self) -> f64 {
        self.0.abs()
    }

    /// Is this positively valenced?
    pub fn is_positive(&self) -> bool {
        self.0 > 0.0
    }

    /// Is this negatively valenced?
    pub fn is_negative(&self) -> bool {
        self.0 < 0.0
    }

    /// Neutral valence.
    pub fn neutral() -> Self {
        Self(0.0)
    }
}

impl Default for Valence {
    fn default() -> Self {
        Self::neutral()
    }
}

/// Salience: how attention-grabbing a signal is.
/// Constrained to [0.0, 1.0].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Salience(f64);

impl Salience {
    /// Create a new Salience, clamping to valid range.
    pub fn new(value: f64) -> Self {
        Self(value.clamp(0.0, 1.0))
    }

    /// Get the raw value.
    pub fn value(&self) -> f64 {
        self.0
    }

    /// Is this highly salient (> 0.7)?
    pub fn is_high(&self) -> bool {
        self.0 > 0.7
    }

    /// Boost salience (for attention competition).
    pub fn boost(&self, amount: f64) -> Self {
        Self::new(self.0 + amount)
    }

    /// Default medium salience.
    pub fn medium() -> Self {
        Self(0.5)
    }
}

impl Default for Salience {
    fn default() -> Self {
        Self::medium()
    }
}

/// Arousal: activation level from calm to excited.
/// Constrained to [0.0, 1.0].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Arousal(f64);

impl Arousal {
    /// Create a new Arousal, clamping to valid range.
    pub fn new(value: f64) -> Self {
        Self(value.clamp(0.0, 1.0))
    }

    /// Get the raw value.
    pub fn value(&self) -> f64 {
        self.0
    }

    /// Is this high arousal (> 0.7)?
    pub fn is_high(&self) -> bool {
        self.0 > 0.7
    }

    /// Default medium arousal.
    pub fn medium() -> Self {
        Self(0.5)
    }
}

impl Default for Arousal {
    fn default() -> Self {
        Self::medium()
    }
}

/// Categories of brain signals.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SignalType {
    /// Raw input from environment
    Sensory,
    /// Retrieved or encoded memory
    Memory,
    /// Expected state
    Prediction,
    /// Prediction error (surprise)
    Error,
    /// Valence/arousal tag
    Emotion,
    /// Salience marker
    Attention,
    /// Global workspace broadcast
    Broadcast,
    /// Request for information
    Query,
    /// Action intention
    Motor,
}

/// Universal signal format for brain module communication.
///
/// Design principles:
/// - Self-describing: carries its own metadata
/// - Valenced: emotional coloring is first-class
/// - Salient: importance is explicit, enables competition
/// - Traceable: source and timestamp for debugging/analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainSignal {
    /// Unique identifier for this signal
    pub id: Uuid,
    /// Module ID that generated this signal
    pub source: String,
    /// What kind of signal this is
    pub signal_type: SignalType,
    /// The actual payload (JSON-serializable)
    pub content: serde_json::Value,
    /// How attention-grabbing (0-1)
    pub salience: Salience,
    /// Emotional coloring (-1 to +1)
    pub valence: Valence,
    /// Activation level (0-1)
    pub arousal: Arousal,
    /// When this signal was created
    pub timestamp: DateTime<Utc>,
    /// Confidence in this signal (0-1)
    pub confidence: f64,
    /// Module-specific metadata
    pub metadata: HashMap<String, serde_json::Value>,
    /// Specific target module, or None for broadcast
    pub target: Option<String>,
    /// Priority for processing (higher = first)
    pub priority: i32,
}

impl BrainSignal {
    /// Create a new brain signal.
    pub fn new(source: impl Into<String>, signal_type: SignalType, content: impl Serialize) -> Self {
        Self {
            id: Uuid::new_v4(),
            source: source.into(),
            signal_type,
            content: serde_json::to_value(content).unwrap_or(serde_json::Value::Null),
            salience: Salience::default(),
            valence: Valence::default(),
            arousal: Arousal::default(),
            timestamp: Utc::now(),
            confidence: 1.0,
            metadata: HashMap::new(),
            target: None,
            priority: 0,
        }
    }

    /// Set salience.
    pub fn with_salience(mut self, salience: f64) -> Self {
        self.salience = Salience::new(salience);
        self
    }

    /// Set valence.
    pub fn with_valence(mut self, valence: f64) -> Self {
        self.valence = Valence::new(valence);
        self
    }

    /// Set arousal.
    pub fn with_arousal(mut self, arousal: f64) -> Self {
        self.arousal = Arousal::new(arousal);
        self
    }

    /// Set target module.
    pub fn with_target(mut self, target: impl Into<String>) -> Self {
        self.target = Some(target.into());
        self
    }

    /// Add metadata.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Serialize) -> Self {
        if let Ok(v) = serde_json::to_value(value) {
            self.metadata.insert(key.into(), v);
        }
        self
    }

    /// Is this signal surprising? (high salience + high arousal)
    pub fn is_surprising(&self) -> bool {
        self.salience.is_high() && self.arousal.is_high()
    }

    /// Combined emotional intensity.
    pub fn emotional_intensity(&self) -> f64 {
        self.valence.intensity() * self.arousal.value()
    }

    /// Return a copy with boosted salience (for attention competition).
    pub fn escalate(&self, boost: f64) -> Self {
        let mut escalated = self.clone();
        escalated.salience = self.salience.boost(boost);
        escalated.arousal = Arousal::new(self.arousal.value() + boost / 2.0);
        escalated.priority += 1;
        escalated.metadata.insert(
            "escalated".to_string(),
            serde_json::Value::Bool(true),
        );
        escalated
    }
}

/// A memory trace as stored in the hippocampus.
///
/// Memories are not static recordings — they're patterns
/// that get reconstructed, consolidated, and can decay.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryTrace {
    /// Unique identifier
    pub id: Uuid,
    /// The memory content
    pub content: serde_json::Value,
    /// Emotional weight
    pub valence: Valence,
    /// How important when formed
    pub salience: Salience,
    /// Prediction error when encoded
    pub surprise: f64,
    /// When this memory was created
    pub created_at: DateTime<Utc>,
    /// When last accessed
    pub last_accessed: DateTime<Utc>,
    /// Number of times retrieved
    pub access_count: u32,
    /// Has this been through consolidation?
    pub consolidated: bool,
    /// How fast this fades (valence-weighted)
    pub decay_rate: f64,
    /// Current retrieval strength
    pub strength: f64,
    /// Linked memory IDs
    pub associations: Vec<Uuid>,
    /// Retrieval cues
    pub context_tags: Vec<String>,
}

impl MemoryTrace {
    /// Create a new memory trace from a brain signal.
    pub fn from_signal(signal: &BrainSignal) -> Self {
        let surprise = signal
            .metadata
            .get("prediction_error")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        let mut decay_rate = 0.1;
        // High surprise = lower decay rate (surprising things stick)
        if surprise > 0.5 {
            decay_rate *= 1.0 - surprise * 0.5;
        }

        Self {
            id: Uuid::new_v4(),
            content: signal.content.clone(),
            valence: signal.valence,
            salience: signal.salience,
            surprise,
            created_at: Utc::now(),
            last_accessed: Utc::now(),
            access_count: 0,
            consolidated: false,
            decay_rate,
            strength: 1.0,
            associations: Vec::new(),
            context_tags: Vec::new(),
        }
    }

    /// Record an access, strengthening the memory.
    pub fn access(&mut self) {
        self.last_accessed = Utc::now();
        self.access_count += 1;
        self.strength = (self.strength + 0.1).min(1.0);
    }

    /// Apply time-based decay.
    ///
    /// Valence-weighted: emotional memories decay slower.
    pub fn decay(&mut self, hours_passed: f64) {
        let effective_rate = self.decay_rate * (1.0 - self.valence.intensity() * 0.5);
        let decay_amount = effective_rate * (hours_passed / 24.0);
        self.strength = (self.strength - decay_amount).max(0.0);
    }

    /// Calculate retrieval score.
    ///
    /// Combines: strength, valence, recency, access frequency
    pub fn retrieval_score(&self) -> f64 {
        let days_since_access = (Utc::now() - self.last_accessed).num_hours() as f64 / 24.0;
        let recency_boost = 1.0 / (1.0 + days_since_access);
        let frequency_boost = (self.access_count as f64 / 10.0).min(1.0);
        let valence_boost = self.valence.intensity();

        self.strength * 0.4 + recency_boost * 0.2 + frequency_boost * 0.2 + valence_boost * 0.2
    }

    /// Is this memory still retrievable?
    pub fn is_retrievable(&self) -> bool {
        self.strength > 0.01
    }

    /// Add a context tag for retrieval.
    pub fn tag(&mut self, tag: impl Into<String>) {
        self.context_tags.push(tag.into());
    }

    /// Link to another memory.
    pub fn associate(&mut self, other_id: Uuid) {
        if !self.associations.contains(&other_id) {
            self.associations.push(other_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valence_clamping() {
        assert_eq!(Valence::new(1.5).value(), 1.0);
        assert_eq!(Valence::new(-1.5).value(), -1.0);
        assert_eq!(Valence::new(0.5).value(), 0.5);
    }

    #[test]
    fn test_signal_creation() {
        let signal = BrainSignal::new("test", SignalType::Sensory, "hello world")
            .with_valence(0.8)
            .with_salience(0.9);

        assert_eq!(signal.source, "test");
        assert_eq!(signal.valence.value(), 0.8);
        assert_eq!(signal.salience.value(), 0.9);
    }

    #[test]
    fn test_memory_decay() {
        let signal = BrainSignal::new("test", SignalType::Memory, "test memory")
            .with_valence(0.0); // Neutral valence = normal decay

        let mut memory = MemoryTrace::from_signal(&signal);
        assert_eq!(memory.strength, 1.0);

        memory.decay(24.0); // One day
        assert!(memory.strength < 1.0);
        assert!(memory.strength > 0.0);
    }

    #[test]
    fn test_emotional_memory_decay_slower() {
        let neutral_signal = BrainSignal::new("test", SignalType::Memory, "neutral")
            .with_valence(0.0);
        let emotional_signal = BrainSignal::new("test", SignalType::Memory, "emotional")
            .with_valence(0.9);

        let mut neutral_memory = MemoryTrace::from_signal(&neutral_signal);
        let mut emotional_memory = MemoryTrace::from_signal(&emotional_signal);

        neutral_memory.decay(48.0);
        emotional_memory.decay(48.0);

        // Emotional memory should retain more strength
        assert!(emotional_memory.strength > neutral_memory.strength);
    }
}
