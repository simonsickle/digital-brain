//! Broca Area - Language Production & Speech Planning
//!
//! Converts inner speech and semantic insights into planned utterances
//! for external communication.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::core::inner_speech::{InnerSpeechType, InnerUtterance};
use crate::regions::temporal_cortex::SemanticInsight;
use crate::signal::{BrainSignal, SignalType};

/// High-level intent for spoken output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeechIntent {
    Inform,
    Question,
    Request,
    Command,
    Reflection,
}

impl SpeechIntent {
    pub fn as_intent_label(&self) -> &'static str {
        match self {
            SpeechIntent::Inform => "inform",
            SpeechIntent::Question => "question",
            SpeechIntent::Request => "request",
            SpeechIntent::Command => "command",
            SpeechIntent::Reflection => "reflection",
        }
    }
}

/// Planned utterance produced by Broca area.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechPlan {
    pub content: String,
    pub intent: SpeechIntent,
    pub urgency: f64,
    pub emotional_tone: f64,
    pub source: String,
    pub created_at: DateTime<Utc>,
}

impl SpeechPlan {
    pub fn to_signal(&self) -> BrainSignal {
        let mut signal = BrainSignal::new("broca", SignalType::Motor, self.clone())
            .with_salience(self.urgency)
            .with_valence(self.emotional_tone);
        signal.metadata.insert(
            "speech_intent".to_string(),
            serde_json::Value::String(self.intent.as_intent_label().to_string()),
        );
        signal.metadata.insert(
            "speech_source".to_string(),
            serde_json::Value::String(self.source.clone()),
        );
        signal
    }
}

/// Configuration for Broca area.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrocaConfig {
    pub min_intensity: f64,
    pub min_salience: f64,
    pub min_confidence: f64,
}

impl Default for BrocaConfig {
    fn default() -> Self {
        Self {
            min_intensity: 0.55,
            min_salience: 0.6,
            min_confidence: 0.6,
        }
    }
}

/// Production stats for Broca area.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BrocaStats {
    pub total_plans: usize,
    pub last_plan_at: Option<DateTime<Utc>>,
}

/// Broca area - language production.
pub struct BrocaArea {
    config: BrocaConfig,
    stats: BrocaStats,
}

impl BrocaArea {
    pub fn new() -> Self {
        Self::with_config(BrocaConfig::default())
    }

    pub fn with_config(config: BrocaConfig) -> Self {
        Self {
            config,
            stats: BrocaStats::default(),
        }
    }

    pub fn plan_from_inner_speech(
        &mut self,
        utterance: &InnerUtterance,
        salience: f64,
    ) -> Option<SpeechPlan> {
        let should_speak = utterance.voluntary
            && (utterance.intensity >= self.config.min_intensity
                || salience >= self.config.min_salience);

        if !should_speak {
            return None;
        }

        let intent = match utterance.speech_type {
            InnerSpeechType::SelfQuestioning => SpeechIntent::Question,
            InnerSpeechType::SelfInstruction => SpeechIntent::Command,
            InnerSpeechType::Planning => SpeechIntent::Request,
            InnerSpeechType::Narrative | InnerSpeechType::Commentary => SpeechIntent::Inform,
            InnerSpeechType::EmotionalProcessing => SpeechIntent::Reflection,
            _ => SpeechIntent::Inform,
        };

        Some(self.record_plan(SpeechPlan {
            content: utterance.content.clone(),
            intent,
            urgency: salience.clamp(0.0, 1.0),
            emotional_tone: utterance.emotional_tone,
            source: "inner_speech".to_string(),
            created_at: Utc::now(),
        }))
    }

    pub fn plan_from_insight(&mut self, insight: &SemanticInsight) -> Option<SpeechPlan> {
        if insight.confidence < self.config.min_confidence {
            return None;
        }

        Some(self.record_plan(SpeechPlan {
            content: format!("Insight: {}", insight.summary),
            intent: SpeechIntent::Inform,
            urgency: insight.confidence.clamp(0.0, 1.0),
            emotional_tone: 0.1,
            source: "semantic_insight".to_string(),
            created_at: Utc::now(),
        }))
    }

    pub fn stats(&self) -> &BrocaStats {
        &self.stats
    }

    fn record_plan(&mut self, plan: SpeechPlan) -> SpeechPlan {
        self.stats.total_plans += 1;
        self.stats.last_plan_at = Some(plan.created_at);
        plan
    }
}

impl Default for BrocaArea {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broca_plans_from_inner_speech() {
        let mut broca = BrocaArea::new();
        let utterance = InnerUtterance::new("We should act now", InnerSpeechType::SelfInstruction)
            .with_intensity(0.9);
        let plan = broca.plan_from_inner_speech(&utterance, 0.7);
        assert!(plan.is_some());
        assert_eq!(broca.stats().total_plans, 1);
    }

    #[test]
    fn test_broca_filters_low_intensity() {
        let mut broca = BrocaArea::new();
        let utterance =
            InnerUtterance::new("maybe later", InnerSpeechType::Commentary).with_intensity(0.1);
        let plan = broca.plan_from_inner_speech(&utterance, 0.1);
        assert!(plan.is_none());
    }
}
