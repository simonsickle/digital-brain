//! Language Cortex - Verbal Comprehension & Inner Speech Coupling
//!
//! This region grounds linguistic inputs (external and internal) into
//! compact semantic representations that can be routed into working
//! memory and the global workspace.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::core::inner_speech::InnerUtterance;
use crate::signal::{Arousal, BrainSignal, Salience, SignalType};

/// Primary intent detected in language.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LanguageIntent {
    Statement,
    Question,
    Command,
    Exclamation,
    Reflection,
}

impl std::fmt::Display for LanguageIntent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let label = match self {
            LanguageIntent::Statement => "statement",
            LanguageIntent::Question => "question",
            LanguageIntent::Command => "command",
            LanguageIntent::Exclamation => "exclamation",
            LanguageIntent::Reflection => "reflection",
        };
        write!(f, "{}", label)
    }
}

/// Origin of a language signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LanguageOrigin {
    External,
    InnerSpeech,
}

impl std::fmt::Display for LanguageOrigin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let label = match self {
            LanguageOrigin::External => "external",
            LanguageOrigin::InnerSpeech => "inner_speech",
        };
        write!(f, "{}", label)
    }
}

/// Compact semantic representation derived from language.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageRepresentation {
    pub content: String,
    pub tokens: Vec<String>,
    pub salient_terms: Vec<String>,
    pub key_phrases: Vec<String>,
    pub intent: LanguageIntent,
    pub sentiment: f64,
    pub confidence: f64,
    pub salience: f64,
    pub arousal: f64,
    pub origin: LanguageOrigin,
    pub inner_speech_type: Option<String>,
    pub created_at: DateTime<Utc>,
}

impl LanguageRepresentation {
    /// Convert the representation into a brain signal for routing.
    pub fn to_signal(&self) -> BrainSignal {
        let signal_type = match self.origin {
            LanguageOrigin::External => SignalType::Sensory,
            LanguageOrigin::InnerSpeech => SignalType::Memory,
        };
        let mut signal = BrainSignal::new("language_cortex", signal_type, self.clone())
            .with_salience(self.salience)
            .with_valence(self.sentiment)
            .with_arousal(self.arousal);
        signal.metadata.insert(
            "language_intent".to_string(),
            serde_json::Value::String(self.intent.to_string()),
        );
        signal.metadata.insert(
            "language_origin".to_string(),
            serde_json::Value::String(self.origin.to_string()),
        );
        if let Some(ref speech_type) = self.inner_speech_type {
            signal.metadata.insert(
                "inner_speech_type".to_string(),
                serde_json::Value::String(speech_type.clone()),
            );
        }
        signal
    }

    /// Should this language representation compete for workspace access?
    pub fn should_broadcast(&self) -> bool {
        self.salience >= 0.6
            || matches!(
                self.intent,
                LanguageIntent::Question | LanguageIntent::Command
            )
    }
}

/// Configuration for language processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageCortexConfig {
    pub max_tokens: usize,
    pub max_terms: usize,
    pub max_phrases: usize,
}

impl Default for LanguageCortexConfig {
    fn default() -> Self {
        Self {
            max_tokens: 64,
            max_terms: 8,
            max_phrases: 4,
        }
    }
}

/// Statistics about language processing.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LanguageStats {
    pub total_inputs: usize,
    pub total_inner_utterances: usize,
    pub average_sentiment: f64,
}

/// The language cortex - extracts semantics from text and inner speech.
pub struct LanguageCortex {
    config: LanguageCortexConfig,
    stats: LanguageStats,
}

impl LanguageCortex {
    pub fn new() -> Self {
        Self::with_config(LanguageCortexConfig::default())
    }

    pub fn with_config(config: LanguageCortexConfig) -> Self {
        Self {
            config,
            stats: LanguageStats::default(),
        }
    }

    /// Process a brain signal (external language input).
    pub fn process_signal(&mut self, signal: &BrainSignal) -> Option<LanguageRepresentation> {
        let text = extract_text_from_signal(signal)?;
        Some(self.analyze_text(&text, LanguageOrigin::External, None))
    }

    /// Process inner speech utterances.
    pub fn process_inner_speech(&mut self, utterance: &InnerUtterance) -> LanguageRepresentation {
        let speech_type = Some(format!("{:?}", utterance.speech_type));
        let mut representation =
            self.analyze_text(&utterance.content, LanguageOrigin::InnerSpeech, speech_type);
        representation.sentiment =
            (representation.sentiment + utterance.emotional_tone * 0.5).clamp(-1.0, 1.0);
        representation.salience =
            (representation.salience + utterance.intensity * 0.2).clamp(0.0, 1.0);
        representation.arousal =
            (representation.arousal + utterance.emotional_tone.abs() * 0.3).clamp(0.0, 1.0);
        representation
    }

    pub fn stats(&self) -> &LanguageStats {
        &self.stats
    }

    fn analyze_text(
        &mut self,
        text: &str,
        origin: LanguageOrigin,
        inner_speech_type: Option<String>,
    ) -> LanguageRepresentation {
        let tokens = tokenize(text, self.config.max_tokens);
        let intent = detect_intent(text, &tokens);
        let sentiment = compute_sentiment(&tokens);
        let salient_terms = extract_salient_terms(&tokens, self.config.max_terms);
        let key_phrases = extract_key_phrases(&tokens, self.config.max_phrases);
        let confidence = ((tokens.len() as f64 / 10.0) + 0.2).clamp(0.2, 1.0);
        let salience = compute_salience(intent, sentiment, text);
        let arousal = compute_arousal(intent, sentiment, text);

        if origin == LanguageOrigin::InnerSpeech {
            self.stats.total_inner_utterances += 1;
        } else {
            self.stats.total_inputs += 1;
        }
        let total = (self.stats.total_inputs + self.stats.total_inner_utterances) as f64;
        if total > 0.0 {
            self.stats.average_sentiment =
                (self.stats.average_sentiment * (total - 1.0) + sentiment) / total;
        }

        LanguageRepresentation {
            content: text.to_string(),
            tokens,
            salient_terms,
            key_phrases,
            intent,
            sentiment,
            confidence,
            salience,
            arousal,
            origin,
            inner_speech_type,
            created_at: Utc::now(),
        }
    }
}

impl Default for LanguageCortex {
    fn default() -> Self {
        Self::new()
    }
}

fn extract_text_from_signal(signal: &BrainSignal) -> Option<String> {
    if let Some(text) = signal.content.as_str() {
        return Some(text.to_string());
    }
    if let Some(text) = signal
        .metadata
        .get("source_content")
        .and_then(|value| value.as_str())
    {
        return Some(text.to_string());
    }
    None
}

fn tokenize(text: &str, max_tokens: usize) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .take(max_tokens)
        .map(|s| s.to_lowercase())
        .collect()
}

fn detect_intent(text: &str, tokens: &[String]) -> LanguageIntent {
    let lowercase = text.to_lowercase();
    if lowercase.trim_end().ends_with('?')
        || tokens.first().is_some_and(|t| {
            matches!(
                t.as_str(),
                "why" | "what" | "how" | "when" | "where" | "who"
            )
        })
    {
        return LanguageIntent::Question;
    }
    if lowercase.contains('!') {
        return LanguageIntent::Exclamation;
    }
    if tokens.first().is_some_and(|t| {
        matches!(
            t.as_str(),
            "please" | "do" | "try" | "remember" | "consider" | "let" | "let's" | "make"
        )
    }) {
        return LanguageIntent::Command;
    }
    if lowercase.contains("i think")
        || lowercase.contains("i feel")
        || lowercase.contains("i notice")
        || lowercase.contains("i wonder")
    {
        return LanguageIntent::Reflection;
    }
    LanguageIntent::Statement
}

fn compute_sentiment(tokens: &[String]) -> f64 {
    let mut score: f64 = 0.0;
    let mut total: f64 = 0.0;
    for token in tokens {
        if POSITIVE_WORDS.contains(&token.as_str()) {
            score += 1.0;
            total += 1.0;
        } else if NEGATIVE_WORDS.contains(&token.as_str()) {
            score -= 1.0;
            total += 1.0;
        }
    }
    if total == 0.0 {
        0.0
    } else {
        (score / total).clamp(-1.0, 1.0)
    }
}

fn extract_salient_terms(tokens: &[String], max_terms: usize) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut terms = Vec::new();
    for token in tokens {
        if STOPWORDS.contains(&token.as_str()) || token.len() < 3 {
            continue;
        }
        if seen.insert(token.clone()) {
            terms.push(token.clone());
        }
        if terms.len() >= max_terms {
            break;
        }
    }
    terms
}

fn extract_key_phrases(tokens: &[String], max_phrases: usize) -> Vec<String> {
    let mut phrases = Vec::new();
    for window in tokens.windows(2) {
        if window.iter().any(|t| STOPWORDS.contains(&t.as_str())) {
            continue;
        }
        phrases.push(format!("{} {}", window[0], window[1]));
        if phrases.len() >= max_phrases {
            break;
        }
    }
    phrases
}

fn compute_salience(intent: LanguageIntent, sentiment: f64, text: &str) -> f64 {
    let mut salience = 0.3 + sentiment.abs() * 0.2;
    if matches!(
        intent,
        LanguageIntent::Question | LanguageIntent::Command | LanguageIntent::Exclamation
    ) {
        salience += 0.2;
    }
    if text.len() > 80 {
        salience += 0.1;
    }
    Salience::new(salience).value()
}

fn compute_arousal(intent: LanguageIntent, sentiment: f64, text: &str) -> f64 {
    let mut arousal = 0.2 + sentiment.abs() * 0.4;
    if matches!(intent, LanguageIntent::Exclamation) {
        arousal += 0.2;
    }
    if text.contains('!') {
        arousal += 0.1;
    }
    Arousal::new(arousal).value()
}

const STOPWORDS: [&str; 18] = [
    "the", "and", "for", "with", "that", "this", "from", "have", "has", "was", "were", "are",
    "but", "not", "you", "your", "about", "into",
];

const POSITIVE_WORDS: [&str; 12] = [
    "good",
    "great",
    "love",
    "happy",
    "success",
    "calm",
    "trust",
    "improve",
    "progress",
    "excellent",
    "wonderful",
    "win",
];

const NEGATIVE_WORDS: [&str; 12] = [
    "bad", "sad", "fear", "angry", "stress", "anxious", "worry", "fail", "problem", "error",
    "pain", "tired",
];

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::InnerSpeechType;

    #[test]
    fn test_language_intent_question() {
        let mut cortex = LanguageCortex::new();
        let signal = BrainSignal::new("test", SignalType::Sensory, "Why is this happening?");
        let rep = cortex.process_signal(&signal).expect("representation");
        assert_eq!(rep.intent, LanguageIntent::Question);
    }

    #[test]
    fn test_language_sentiment_positive() {
        let mut cortex = LanguageCortex::new();
        let signal = BrainSignal::new(
            "test",
            SignalType::Sensory,
            "This is a great success and I love it",
        );
        let rep = cortex.process_signal(&signal).expect("representation");
        assert!(rep.sentiment > 0.0);
    }

    #[test]
    fn test_language_inner_speech_origin() {
        let mut cortex = LanguageCortex::new();
        let utterance = InnerUtterance::new("I need to focus", InnerSpeechType::SelfInstruction);
        let rep = cortex.process_inner_speech(&utterance);
        assert_eq!(rep.origin, LanguageOrigin::InnerSpeech);
        assert!(rep.inner_speech_type.is_some());
    }

    #[test]
    fn test_language_representation_signal() {
        let mut cortex = LanguageCortex::new();
        let signal = BrainSignal::new("test", SignalType::Sensory, "Please remember this");
        let rep = cortex.process_signal(&signal).expect("representation");
        let language_signal = rep.to_signal();
        assert_eq!(language_signal.source, "language_cortex");
    }
}
