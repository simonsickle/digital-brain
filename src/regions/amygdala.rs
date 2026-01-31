//! Amygdala - The Emotional System
//!
//! Computes emotional valence and arousal from raw signals.
//! The amygdala tags incoming information with emotional significance.
//!
//! Key functions:
//! - Rapid threat/reward detection
//! - Valence computation (positive/negative)
//! - Arousal modulation
//! - Emotional memory tagging

#[allow(unused_imports)]
use crate::signal::{Arousal, BrainSignal, SignalType, Valence};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for emotional processing.
#[derive(Debug, Clone)]
pub struct AmygdalaConfig {
    /// How quickly emotional state decays
    pub emotional_decay_rate: f64,
    /// Threshold for significant emotional response
    pub significance_threshold: f64,
    /// Weight for threat detection vs reward detection
    pub threat_bias: f64,
}

impl Default for AmygdalaConfig {
    fn default() -> Self {
        Self {
            emotional_decay_rate: 0.1,
            significance_threshold: 0.3,
            threat_bias: 1.2, // Slight negativity bias (like humans)
        }
    }
}

/// An emotional appraisal of a signal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalAppraisal {
    /// Computed valence (-1 to 1)
    pub valence: Valence,
    /// Computed arousal (0 to 1)
    pub arousal: Arousal,
    /// Is this emotionally significant?
    pub is_significant: bool,
    /// Primary emotion detected (if any)
    pub primary_emotion: Option<Emotion>,
    /// Confidence in this appraisal
    pub confidence: f64,
}

/// Basic emotion categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Emotion {
    Joy,
    Sadness,
    Fear,
    Anger,
    Surprise,
    Disgust,
    Trust,
    Anticipation,
}

impl Emotion {
    /// Get typical valence for this emotion.
    pub fn typical_valence(&self) -> f64 {
        match self {
            Emotion::Joy => 0.8,
            Emotion::Trust => 0.6,
            Emotion::Anticipation => 0.4,
            Emotion::Surprise => 0.0, // Neutral until context known
            Emotion::Sadness => -0.6,
            Emotion::Fear => -0.7,
            Emotion::Anger => -0.5,
            Emotion::Disgust => -0.8,
        }
    }

    /// Get typical arousal for this emotion.
    pub fn typical_arousal(&self) -> f64 {
        match self {
            Emotion::Joy => 0.7,
            Emotion::Trust => 0.4,
            Emotion::Anticipation => 0.6,
            Emotion::Surprise => 0.9,
            Emotion::Sadness => 0.3,
            Emotion::Fear => 0.9,
            Emotion::Anger => 0.8,
            Emotion::Disgust => 0.6,
        }
    }
}

/// The amygdala - emotional processing center.
pub struct Amygdala {
    config: AmygdalaConfig,
    /// Current emotional state (running average)
    current_valence: f64,
    current_arousal: f64,
    /// Learned emotional associations (keyword -> valence)
    learned_associations: HashMap<String, f64>,
    /// Count of processed signals
    processed_count: u64,
}

impl Amygdala {
    /// Create a new amygdala with default config.
    pub fn new() -> Self {
        Self::with_config(AmygdalaConfig::default())
    }

    /// Create a new amygdala with custom config.
    pub fn with_config(config: AmygdalaConfig) -> Self {
        Self {
            config,
            current_valence: 0.0,
            current_arousal: 0.5,
            learned_associations: Self::default_associations(),
            processed_count: 0,
        }
    }

    /// Default emotional associations (can be learned/modified).
    fn default_associations() -> HashMap<String, f64> {
        let mut map = HashMap::new();

        // Positive associations
        map.insert("success".to_string(), 0.8);
        map.insert("win".to_string(), 0.7);
        map.insert("happy".to_string(), 0.8);
        map.insert("love".to_string(), 0.9);
        map.insert("good".to_string(), 0.5);
        map.insert("great".to_string(), 0.7);
        map.insert("excellent".to_string(), 0.8);
        map.insert("beautiful".to_string(), 0.6);

        // Negative associations
        map.insert("fail".to_string(), -0.7);
        map.insert("error".to_string(), -0.5);
        map.insert("bad".to_string(), -0.5);
        map.insert("terrible".to_string(), -0.8);
        map.insert("danger".to_string(), -0.8);
        map.insert("threat".to_string(), -0.9);
        map.insert("pain".to_string(), -0.7);
        map.insert("loss".to_string(), -0.6);

        map
    }

    /// Process a signal and compute emotional appraisal.
    pub fn appraise(&mut self, signal: &BrainSignal) -> EmotionalAppraisal {
        self.processed_count += 1;

        // Extract content for analysis
        let content_str = signal.content.to_string().to_lowercase();

        // Compute valence from content
        let content_valence = self.compute_content_valence(&content_str);

        // Compute arousal from signal properties
        let arousal_value = self.compute_arousal(signal);

        // Combine with any existing valence on the signal
        let combined_valence = if signal.valence.value().abs() > 0.1 {
            // Signal already has valence - weight it heavily
            signal.valence.value() * 0.7 + content_valence * 0.3
        } else {
            content_valence
        };

        // Apply threat bias (negative emotions weighted more)
        let biased_valence = if combined_valence < 0.0 {
            combined_valence * self.config.threat_bias
        } else {
            combined_valence
        };

        let valence = Valence::new(biased_valence);
        let arousal = Arousal::new(arousal_value);

        // Update running emotional state
        self.update_emotional_state(valence.value(), arousal.value());

        // Determine if significant
        let intensity = valence.intensity() * arousal.value();
        let is_significant = intensity > self.config.significance_threshold;

        // Detect primary emotion
        let primary_emotion = self.detect_emotion(valence.value(), arousal.value());

        EmotionalAppraisal {
            valence,
            arousal,
            is_significant,
            primary_emotion,
            confidence: 0.7, // Could be made more sophisticated
        }
    }

    /// Compute valence from content analysis.
    fn compute_content_valence(&self, content: &str) -> f64 {
        let mut valence_sum = 0.0;
        let mut match_count = 0;

        for (keyword, valence) in &self.learned_associations {
            if content.contains(keyword) {
                valence_sum += valence;
                match_count += 1;
            }
        }

        if match_count > 0 {
            (valence_sum / match_count as f64).clamp(-1.0, 1.0)
        } else {
            0.0
        }
    }

    /// Compute arousal from signal properties.
    fn compute_arousal(&self, signal: &BrainSignal) -> f64 {
        let mut arousal = signal.arousal.value();

        // High salience increases arousal
        if signal.salience.is_high() {
            arousal += 0.2;
        }

        // Surprising signals increase arousal
        if signal.is_surprising() {
            arousal += 0.3;
        }

        // High priority increases arousal
        if signal.priority > 0 {
            arousal += 0.1 * (signal.priority as f64).min(3.0);
        }

        arousal.clamp(0.0, 1.0)
    }

    /// Update running emotional state.
    fn update_emotional_state(&mut self, new_valence: f64, new_arousal: f64) {
        // Exponential moving average
        let alpha = 0.3;
        self.current_valence = alpha * new_valence + (1.0 - alpha) * self.current_valence;
        self.current_arousal = alpha * new_arousal + (1.0 - alpha) * self.current_arousal;
    }

    /// Detect primary emotion from valence and arousal.
    fn detect_emotion(&self, valence: f64, arousal: f64) -> Option<Emotion> {
        // Circumplex model of emotion
        if valence.abs() < 0.2 && arousal < 0.3 {
            return None; // Neutral
        }

        Some(match (valence >= 0.0, arousal > 0.5) {
            (true, true) => {
                if valence > 0.6 {
                    Emotion::Joy
                } else {
                    Emotion::Anticipation
                }
            }
            (true, false) => {
                if valence > 0.4 {
                    Emotion::Trust
                } else {
                    Emotion::Anticipation
                }
            }
            (false, true) => {
                if valence < -0.6 {
                    Emotion::Fear
                } else {
                    Emotion::Anger
                }
            }
            (false, false) => Emotion::Sadness,
        })
    }

    /// Learn a new emotional association.
    pub fn learn_association(&mut self, keyword: impl Into<String>, valence: f64) {
        self.learned_associations
            .insert(keyword.into(), valence.clamp(-1.0, 1.0));
    }

    /// Get current emotional state.
    pub fn current_state(&self) -> (Valence, Arousal) {
        (
            Valence::new(self.current_valence),
            Arousal::new(self.current_arousal),
        )
    }

    /// Apply emotional decay (call periodically).
    pub fn decay(&mut self) {
        self.current_valence *= 1.0 - self.config.emotional_decay_rate;
        self.current_arousal = self.current_arousal * (1.0 - self.config.emotional_decay_rate)
            + 0.5 * self.config.emotional_decay_rate; // Decay toward baseline
    }

    /// Tag a signal with emotional appraisal.
    pub fn tag_signal(&mut self, signal: BrainSignal) -> BrainSignal {
        let appraisal = self.appraise(&signal);

        signal
            .with_valence(appraisal.valence.value())
            .with_arousal(appraisal.arousal.value())
            .with_metadata("emotional_significance", appraisal.is_significant)
            .with_metadata(
                "primary_emotion",
                appraisal.primary_emotion.map(|e| format!("{:?}", e)),
            )
    }

    /// Get statistics about emotional processing.
    pub fn stats(&self) -> AmygdalaStats {
        AmygdalaStats {
            processed_signals: self.processed_count,
            current_valence: self.current_valence,
            current_arousal: self.current_arousal,
            learned_associations: self.learned_associations.len(),
        }
    }
}

impl Default for Amygdala {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about emotional processing.
#[derive(Debug, Clone)]
pub struct AmygdalaStats {
    pub processed_signals: u64,
    pub current_valence: f64,
    pub current_arousal: f64,
    pub learned_associations: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_positive_content_detection() {
        let mut amygdala = Amygdala::new();

        let signal = BrainSignal::new("test", SignalType::Sensory, "This is a great success!");
        let appraisal = amygdala.appraise(&signal);

        assert!(appraisal.valence.is_positive());
    }

    #[test]
    fn test_negative_content_detection() {
        let mut amygdala = Amygdala::new();

        let signal = BrainSignal::new(
            "test",
            SignalType::Sensory,
            "Terrible failure, danger ahead",
        );
        let appraisal = amygdala.appraise(&signal);

        assert!(appraisal.valence.is_negative());
    }

    #[test]
    fn test_threat_bias() {
        let mut amygdala = Amygdala::new();

        // Process positive and negative signals
        let positive = BrainSignal::new("test", SignalType::Sensory, "good things");
        let negative = BrainSignal::new("test", SignalType::Sensory, "bad things");

        let pos_appraisal = amygdala.appraise(&positive);
        let neg_appraisal = amygdala.appraise(&negative);

        // Negative should have stronger absolute value due to threat bias
        assert!(neg_appraisal.valence.intensity() >= pos_appraisal.valence.intensity() * 0.9);
    }

    #[test]
    fn test_emotion_detection() {
        let mut amygdala = Amygdala::new();

        // High positive valence + high arousal = joy
        let signal =
            BrainSignal::new("test", SignalType::Sensory, "love and happiness!").with_arousal(0.8);
        let appraisal = amygdala.appraise(&signal);

        assert_eq!(appraisal.primary_emotion, Some(Emotion::Joy));
    }

    #[test]
    fn test_learned_association() {
        let mut amygdala = Amygdala::new();

        // Learn a new association
        amygdala.learn_association("rust", 0.9);

        let signal = BrainSignal::new("test", SignalType::Sensory, "I love rust programming");
        let appraisal = amygdala.appraise(&signal);

        assert!(appraisal.valence.is_positive());
    }

    #[test]
    fn test_signal_tagging() {
        let mut amygdala = Amygdala::new();

        let signal = BrainSignal::new("test", SignalType::Sensory, "great news!");
        let tagged = amygdala.tag_signal(signal);

        assert!(tagged.valence.is_positive());
        assert!(tagged.metadata.contains_key("emotional_significance"));
    }

    #[test]
    fn test_emotional_decay() {
        let mut amygdala = Amygdala::new();

        // Process high-valence signal
        let signal = BrainSignal::new("test", SignalType::Sensory, "amazing success!");
        amygdala.appraise(&signal);

        let (initial_v, _) = amygdala.current_state();

        // Apply decay
        for _ in 0..10 {
            amygdala.decay();
        }

        let (decayed_v, _) = amygdala.current_state();

        // Valence should have decayed toward neutral
        assert!(decayed_v.intensity() < initial_v.intensity());
    }
}
