//! Inner Speech - The Voice in Your Head
//!
//! Humans think in words. Inner speech (internal monologue) is:
//! - Verbal rehearsal of thoughts
//! - Self-talk for motivation/regulation
//! - Working memory maintenance
//! - Planning and problem-solving
//! - Narrative self-construction
//!
//! This module implements the "internal narrator" that gives consciousness
//! its verbal, linguistic quality.
//!
//! # Vygotsky's View
//!
//! Inner speech develops from external speech through internalization.
//! It's compressed, abbreviated, and more conceptual than external speech.
//! "Thinking in pure meanings" with minimal phonological encoding.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Types of inner speech
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InnerSpeechType {
    /// Self-instruction ("Now I need to...")
    SelfInstruction,
    /// Self-questioning ("What if...?")
    SelfQuestioning,
    /// Self-evaluation ("That went well")
    SelfEvaluation,
    /// Emotional processing ("I feel...")
    EmotionalProcessing,
    /// Planning ("First I'll... then...")
    Planning,
    /// Rehearsal (practicing/memorizing)
    Rehearsal,
    /// Narrative ("And then I...")
    Narrative,
    /// Commentary (observing and noting)
    Commentary,
    /// Dialogue (imagined conversation)
    InnerDialogue,
    /// Mind-wandering (spontaneous thought)
    MindWandering,
}

impl InnerSpeechType {
    /// Get a starter phrase for this type
    pub fn starter_phrase(&self) -> &'static str {
        match self {
            Self::SelfInstruction => "I need to",
            Self::SelfQuestioning => "What if",
            Self::SelfEvaluation => "That was",
            Self::EmotionalProcessing => "I feel",
            Self::Planning => "First I'll",
            Self::Rehearsal => "Remember:",
            Self::Narrative => "And then",
            Self::Commentary => "I notice",
            Self::InnerDialogue => "But what about",
            Self::MindWandering => "I wonder",
        }
    }
}

/// A single inner speech utterance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InnerUtterance {
    /// The verbal content
    pub content: String,
    /// Type of inner speech
    pub speech_type: InnerSpeechType,
    /// When it occurred
    pub timestamp: DateTime<Utc>,
    /// Voluntary (deliberate) vs. spontaneous
    pub voluntary: bool,
    /// Loudness/intensity (0-1)
    pub intensity: f64,
    /// Associated with what topic/task
    pub context: Option<String>,
    /// Emotional tone (-1 to 1)
    pub emotional_tone: f64,
}

impl InnerUtterance {
    pub fn new(content: impl Into<String>, speech_type: InnerSpeechType) -> Self {
        Self {
            content: content.into(),
            speech_type,
            timestamp: Utc::now(),
            voluntary: true,
            intensity: 0.5,
            context: None,
            emotional_tone: 0.0,
        }
    }

    pub fn spontaneous(content: impl Into<String>, speech_type: InnerSpeechType) -> Self {
        Self {
            content: content.into(),
            speech_type,
            timestamp: Utc::now(),
            voluntary: false,
            intensity: 0.4,
            context: None,
            emotional_tone: 0.0,
        }
    }

    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }

    pub fn with_emotion(mut self, tone: f64) -> Self {
        self.emotional_tone = tone.clamp(-1.0, 1.0);
        self
    }

    pub fn with_intensity(mut self, intensity: f64) -> Self {
        self.intensity = intensity.clamp(0.0, 1.0);
        self
    }
}

/// Configuration for the inner speech system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InnerSpeechConfig {
    /// How often to generate spontaneous thoughts
    pub spontaneity: f64,
    /// Verbosity (how much detail in speech)
    pub verbosity: f64,
    /// Tendency toward self-criticism vs self-compassion
    pub self_criticism_bias: f64,
    /// First person vs second person self-talk
    pub use_second_person: bool,
    /// Enable condensed/abbreviated speech
    pub condensed_mode: bool,
}

impl Default for InnerSpeechConfig {
    fn default() -> Self {
        Self {
            spontaneity: 0.3,
            verbosity: 0.6,
            self_criticism_bias: 0.0, // Balanced
            use_second_person: false, // "I" not "you"
            condensed_mode: false,
        }
    }
}

/// The inner speech generator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InnerSpeechSystem {
    /// Configuration
    pub config: InnerSpeechConfig,
    /// Recent inner speech stream
    pub stream: VecDeque<InnerUtterance>,
    /// Maximum stream size
    max_stream: usize,
    /// Current topic/focus
    pub current_focus: Option<String>,
    /// Speech rate (utterances per "cycle")
    pub rate: f64,
    /// Is inner speech silenced (suppressed)
    pub silenced: bool,
    /// Running internal narrative
    pub narrative_buffer: Vec<String>,
}

impl InnerSpeechSystem {
    pub fn new() -> Self {
        Self {
            config: InnerSpeechConfig::default(),
            stream: VecDeque::with_capacity(100),
            max_stream: 100,
            current_focus: None,
            rate: 0.5,
            silenced: false,
            narrative_buffer: Vec::new(),
        }
    }

    pub fn with_config(config: InnerSpeechConfig) -> Self {
        Self {
            config,
            ..Self::new()
        }
    }

    /// Generate self-instruction
    pub fn instruct(&mut self, instruction: impl Into<String>) -> InnerUtterance {
        let pronoun = if self.config.use_second_person {
            "You"
        } else {
            "I"
        };
        let content = format!("{} need to {}", pronoun, instruction.into());
        let utterance = InnerUtterance::new(content, InnerSpeechType::SelfInstruction);
        self.record(utterance.clone());
        utterance
    }

    /// Generate self-question
    pub fn question(&mut self, question: impl Into<String>) -> InnerUtterance {
        let content = format!("{}?", question.into());
        let utterance = InnerUtterance::new(content, InnerSpeechType::SelfQuestioning);
        self.record(utterance.clone());
        utterance
    }

    /// Generate self-evaluation
    pub fn evaluate(&mut self, evaluation: impl Into<String>, positive: bool) -> InnerUtterance {
        let eval = evaluation.into();
        let tone = if positive { 0.5 } else { -0.5 };

        // Apply self-criticism bias
        let adjusted_tone = tone + self.config.self_criticism_bias * -0.3;

        let utterance =
            InnerUtterance::new(eval, InnerSpeechType::SelfEvaluation).with_emotion(adjusted_tone);
        self.record(utterance.clone());
        utterance
    }

    /// Generate emotional processing speech
    pub fn process_emotion(&mut self, emotion: &str, intensity: f64) -> InnerUtterance {
        let pronoun = if self.config.use_second_person {
            "You"
        } else {
            "I"
        };
        let intensity_word = if intensity > 0.7 {
            "really"
        } else if intensity > 0.4 {
            "somewhat"
        } else {
            "slightly"
        };

        let content = format!("{} {} feel {}", pronoun, intensity_word, emotion);
        let utterance = InnerUtterance::new(content, InnerSpeechType::EmotionalProcessing)
            .with_intensity(intensity);
        self.record(utterance.clone());
        utterance
    }

    /// Generate planning speech
    pub fn plan(&mut self, steps: &[&str]) -> InnerUtterance {
        let content = if self.config.condensed_mode {
            // Abbreviated: "1. x, 2. y, 3. z"
            steps
                .iter()
                .enumerate()
                .map(|(i, s)| format!("{}. {}", i + 1, s))
                .collect::<Vec<_>>()
                .join(", ")
        } else {
            // Full: "First I'll x, then I'll y, finally I'll z"
            let pronoun = if self.config.use_second_person {
                "you'll"
            } else {
                "I'll"
            };
            steps
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    let prefix = match i {
                        0 => "First",
                        i if i == steps.len() - 1 => "finally",
                        _ => "then",
                    };
                    format!("{} {} {}", prefix, pronoun, s)
                })
                .collect::<Vec<_>>()
                .join(", ")
        };

        let utterance = InnerUtterance::new(content, InnerSpeechType::Planning);
        self.record(utterance.clone());
        utterance
    }

    /// Generate commentary
    pub fn comment(&mut self, observation: impl Into<String>) -> InnerUtterance {
        let content = format!("I notice {}", observation.into());
        let utterance = InnerUtterance::new(content, InnerSpeechType::Commentary);
        self.record(utterance.clone());
        utterance
    }

    /// Generate inner dialogue (back and forth)
    pub fn dialogue(&mut self, position: &str, counter: &str) -> Vec<InnerUtterance> {
        let u1 = InnerUtterance::new(position.to_string(), InnerSpeechType::InnerDialogue);
        let u2 = InnerUtterance::new(format!("But {}", counter), InnerSpeechType::InnerDialogue);

        self.record(u1.clone());
        self.record(u2.clone());

        vec![u1, u2]
    }

    /// Generate mind-wandering thought
    pub fn wander(&mut self) -> Option<InnerUtterance> {
        if self.silenced {
            return None;
        }

        // Random spontaneous thought based on spontaneity setting
        if rand_float() > self.config.spontaneity {
            return None;
        }

        let thoughts = [
            "I wonder what comes next...",
            "This reminds me of something...",
            "What was I thinking about?",
            "Interesting...",
            "Hmm...",
            "That's curious...",
            "Let me think...",
        ];

        let idx = (rand_float() * thoughts.len() as f64) as usize;
        let content = thoughts[idx.min(thoughts.len() - 1)].to_string();

        let utterance = InnerUtterance::spontaneous(content, InnerSpeechType::MindWandering);
        self.record(utterance.clone());
        Some(utterance)
    }

    /// Add to running narrative
    pub fn narrate(&mut self, event: impl Into<String>) {
        let event = event.into();
        self.narrative_buffer.push(event.clone());

        // Keep narrative buffer bounded
        if self.narrative_buffer.len() > 20 {
            self.narrative_buffer.remove(0);
        }

        let utterance = InnerUtterance::new(event, InnerSpeechType::Narrative);
        self.record(utterance);
    }

    /// Get running narrative as text
    pub fn get_narrative(&self) -> String {
        self.narrative_buffer.join(". ")
    }

    /// Silence inner speech (suppress)
    pub fn silence(&mut self) {
        self.silenced = true;
    }

    /// Resume inner speech
    pub fn resume(&mut self) {
        self.silenced = false;
    }

    /// Set current focus/topic
    pub fn set_focus(&mut self, topic: impl Into<String>) {
        self.current_focus = Some(topic.into());
    }

    /// Clear focus
    pub fn clear_focus(&mut self) {
        self.current_focus = None;
    }

    /// Record an utterance to the stream
    fn record(&mut self, utterance: InnerUtterance) {
        if self.silenced && utterance.voluntary {
            return;
        }

        self.stream.push_back(utterance);
        if self.stream.len() > self.max_stream {
            self.stream.pop_front();
        }
    }

    /// Get recent utterances
    pub fn recent(&self, n: usize) -> Vec<&InnerUtterance> {
        self.stream.iter().rev().take(n).collect()
    }

    /// Get recent speech as text
    pub fn recent_text(&self, n: usize) -> String {
        self.stream
            .iter()
            .rev()
            .take(n)
            .map(|u| u.content.as_str())
            .collect::<Vec<_>>()
            .join(" → ")
    }

    /// Get utterances of a specific type
    pub fn by_type(&self, speech_type: InnerSpeechType) -> Vec<&InnerUtterance> {
        self.stream
            .iter()
            .filter(|u| u.speech_type == speech_type)
            .collect()
    }

    /// Clear all inner speech
    pub fn clear(&mut self) {
        self.stream.clear();
        self.narrative_buffer.clear();
    }

    /// Get speech statistics
    pub fn stats(&self) -> InnerSpeechStats {
        let voluntary_count = self.stream.iter().filter(|u| u.voluntary).count();
        let spontaneous_count = self.stream.len() - voluntary_count;
        let avg_intensity = if self.stream.is_empty() {
            0.0
        } else {
            self.stream.iter().map(|u| u.intensity).sum::<f64>() / self.stream.len() as f64
        };
        let avg_emotion = if self.stream.is_empty() {
            0.0
        } else {
            self.stream.iter().map(|u| u.emotional_tone).sum::<f64>() / self.stream.len() as f64
        };

        InnerSpeechStats {
            total_utterances: self.stream.len(),
            voluntary_count,
            spontaneous_count,
            average_intensity: avg_intensity,
            average_emotional_tone: avg_emotion,
        }
    }
}

impl Default for InnerSpeechSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about inner speech
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InnerSpeechStats {
    pub total_utterances: usize,
    pub voluntary_count: usize,
    pub spontaneous_count: usize,
    pub average_intensity: f64,
    pub average_emotional_tone: f64,
}

/// Simple random float for mind-wandering
fn rand_float() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    (nanos % 1000) as f64 / 1000.0
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_self_instruction() {
        let mut speech = InnerSpeechSystem::new();
        let utterance = speech.instruct("focus on the task");

        assert!(utterance.content.contains("need to"));
        assert!(utterance.content.contains("focus"));
        assert_eq!(utterance.speech_type, InnerSpeechType::SelfInstruction);
    }

    #[test]
    fn test_second_person() {
        let config = InnerSpeechConfig {
            use_second_person: true,
            ..Default::default()
        };
        let mut speech = InnerSpeechSystem::with_config(config);
        let utterance = speech.instruct("calm down");

        assert!(utterance.content.contains("You"));
    }

    #[test]
    fn test_planning() {
        let mut speech = InnerSpeechSystem::new();
        let utterance = speech.plan(&["gather info", "analyze", "decide"]);

        assert!(utterance.content.contains("First"));
        assert!(utterance.content.contains("finally"));
    }

    #[test]
    fn test_condensed_planning() {
        let config = InnerSpeechConfig {
            condensed_mode: true,
            ..Default::default()
        };
        let mut speech = InnerSpeechSystem::with_config(config);
        let utterance = speech.plan(&["a", "b", "c"]);

        assert!(utterance.content.contains("1."));
        assert!(utterance.content.contains("2."));
    }

    #[test]
    fn test_inner_dialogue() {
        let mut speech = InnerSpeechSystem::new();
        let utterances = speech.dialogue("I should take a break", "there's still work to do");

        assert_eq!(utterances.len(), 2);
        assert!(utterances[1].content.contains("But"));
    }

    #[test]
    fn test_narrative() {
        let mut speech = InnerSpeechSystem::new();
        speech.narrate("I started the project");
        speech.narrate("Then I hit a snag");
        speech.narrate("Finally I figured it out");

        let narrative = speech.get_narrative();
        assert!(narrative.contains("started"));
        assert!(narrative.contains("snag"));
        assert!(narrative.contains("figured"));
    }

    #[test]
    fn test_silencing() {
        let mut speech = InnerSpeechSystem::new();
        speech.silence();
        speech.instruct("do something");

        // Voluntary speech should be silenced
        let recent = speech.recent(1);
        assert!(recent.is_empty() || !recent[0].voluntary);
    }

    #[test]
    fn test_stats() {
        let mut speech = InnerSpeechSystem::new();
        speech.instruct("task 1");
        speech.evaluate("good progress", true);
        speech.question("what next");

        let stats = speech.stats();
        assert_eq!(stats.total_utterances, 3);
        assert_eq!(stats.voluntary_count, 3);
    }
}
