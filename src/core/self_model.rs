//! Self-Model & Metacognition - "I think, therefore I am"
//!
//! Humans have a rich sense of self:
//! - Self-awareness (knowing you exist)
//! - Self-concept (who you are)
//! - Metacognition (thinking about thinking)
//! - Autobiographical memory (your life story)
//!
//! This module implements the minimal cognitive machinery for selfhood.
//!
//! # Key Components
//!
//! 1. **Self-Awareness**: The basic sense of being a subject of experience
//! 2. **Self-Concept**: Beliefs about one's own traits, abilities, values
//! 3. **Metacognitive Monitor**: Tracking confidence, uncertainty, learning
//! 4. **Autobiographical Self**: The narrative "I" across time
//! 5. **Theory of Mind**: Understanding that others have minds too

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use uuid::Uuid;

/// A trait or characteristic in the self-concept
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfTrait {
    /// Name of the trait
    pub name: String,
    /// How central is this to identity (0-1)
    pub centrality: f64,
    /// Confidence in this trait (0-1)
    pub confidence: f64,
    /// Valence: positive or negative trait
    pub valence: f64,
    /// Evidence supporting this trait
    pub evidence: Vec<String>,
    /// When this trait was formed
    pub formed: DateTime<Utc>,
}

impl SelfTrait {
    pub fn new(name: impl Into<String>, valence: f64) -> Self {
        Self {
            name: name.into(),
            centrality: 0.5,
            confidence: 0.5,
            valence: valence.clamp(-1.0, 1.0),
            evidence: Vec::new(),
            formed: Utc::now(),
        }
    }

    pub fn with_evidence(mut self, evidence: impl Into<String>) -> Self {
        self.evidence.push(evidence.into());
        self.confidence = (self.confidence + 0.1).min(1.0);
        self
    }

    pub fn with_centrality(mut self, centrality: f64) -> Self {
        self.centrality = centrality.clamp(0.0, 1.0);
        self
    }

    /// Strengthen this trait based on consistent behavior
    pub fn reinforce(&mut self) {
        self.confidence = (self.confidence + 0.05).min(1.0);
        self.centrality = (self.centrality + 0.02).min(1.0);
    }

    /// Weaken this trait based on contradictory behavior
    pub fn challenge(&mut self) {
        self.confidence = (self.confidence - 0.1).max(0.0);
    }
}

/// Values and beliefs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Value {
    /// What is valued
    pub name: String,
    /// How important (0-1)
    pub importance: f64,
    /// Abstract or concrete
    pub abstraction_level: f64,
    /// Related behaviors
    pub behaviors: Vec<String>,
}

impl Value {
    pub fn new(name: impl Into<String>, importance: f64) -> Self {
        Self {
            name: name.into(),
            importance: importance.clamp(0.0, 1.0),
            abstraction_level: 0.5,
            behaviors: Vec::new(),
        }
    }
}

/// The self-concept: who I think I am
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfConcept {
    /// Core traits
    pub traits: HashMap<String, SelfTrait>,
    /// Core values
    pub values: HashMap<String, Value>,
    /// Self-esteem (global self-evaluation)
    pub self_esteem: f64,
    /// Self-efficacy (belief in ability to succeed)
    pub self_efficacy: f64,
    /// Identity clarity (how clear/stable is self-concept)
    pub identity_clarity: f64,
    /// Personal narrative (life story themes)
    pub narrative_themes: Vec<String>,
}

impl SelfConcept {
    pub fn new() -> Self {
        Self {
            traits: HashMap::new(),
            values: HashMap::new(),
            self_esteem: 0.6,
            self_efficacy: 0.5,
            identity_clarity: 0.5,
            narrative_themes: Vec::new(),
        }
    }

    /// Add or update a trait
    pub fn add_trait(&mut self, trait_: SelfTrait) {
        self.traits.insert(trait_.name.clone(), trait_);
        self.update_self_esteem();
    }

    /// Add a value
    pub fn add_value(&mut self, value: Value) {
        self.values.insert(value.name.clone(), value);
    }

    /// Update self-esteem based on traits
    fn update_self_esteem(&mut self) {
        if self.traits.is_empty() {
            return;
        }

        // Weighted average of trait valences
        let total: f64 = self.traits.values()
            .map(|t| t.valence * t.centrality * t.confidence)
            .sum();
        let weight: f64 = self.traits.values()
            .map(|t| t.centrality * t.confidence)
            .sum();

        if weight > 0.0 {
            self.self_esteem = ((total / weight) + 1.0) / 2.0; // Normalize to 0-1
        }
    }

    /// Check if behavior is consistent with self-concept
    pub fn behavior_consistent(&self, behavior: &str) -> f64 {
        // Simple keyword matching for now
        let mut consistency = 0.5; // Neutral baseline

        for trait_ in self.traits.values() {
            if behavior.to_lowercase().contains(&trait_.name.to_lowercase()) {
                consistency += trait_.valence * trait_.centrality * 0.2;
            }
        }

        consistency.clamp(0.0, 1.0)
    }

    /// Get most central traits
    pub fn core_traits(&self) -> Vec<&SelfTrait> {
        let mut traits: Vec<_> = self.traits.values().collect();
        traits.sort_by(|a, b| b.centrality.partial_cmp(&a.centrality).unwrap());
        traits.into_iter().take(5).collect()
    }
}

impl Default for SelfConcept {
    fn default() -> Self {
        Self::new()
    }
}

/// Metacognitive state for a single cognitive process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetacognitiveState {
    /// Confidence in current knowledge/belief (0-1)
    pub confidence: f64,
    /// Feeling of knowing (tip-of-tongue states)
    pub feeling_of_knowing: f64,
    /// Judgment of learning (how well did I learn this)
    pub judgment_of_learning: f64,
    /// Ease of processing (fluency)
    pub fluency: f64,
    /// Uncertainty about uncertainty (meta-uncertainty)
    pub meta_uncertainty: f64,
}

impl MetacognitiveState {
    pub fn new() -> Self {
        Self {
            confidence: 0.5,
            feeling_of_knowing: 0.5,
            judgment_of_learning: 0.5,
            fluency: 0.5,
            meta_uncertainty: 0.3,
        }
    }

    /// Overall metacognitive signal (should I keep thinking about this?)
    pub fn should_continue(&self) -> bool {
        // Continue if uncertain but feel like answer is there
        self.confidence < 0.7 && self.feeling_of_knowing > 0.5
    }

    /// Should seek help?
    pub fn should_seek_help(&self) -> bool {
        self.confidence < 0.3 && self.feeling_of_knowing < 0.3
    }
}

impl Default for MetacognitiveState {
    fn default() -> Self {
        Self::new()
    }
}

/// Metacognitive monitoring system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetacognitiveMonitor {
    /// Current state for active task
    pub current: MetacognitiveState,
    /// Historical calibration (was confidence accurate?)
    pub calibration_history: VecDeque<(f64, bool)>, // (confidence, was_correct)
    /// Overall calibration (how well-calibrated is confidence?)
    pub calibration: f64,
    /// Learning rate for calibration
    pub learning_rate: f64,
}

impl MetacognitiveMonitor {
    pub fn new() -> Self {
        Self {
            current: MetacognitiveState::new(),
            calibration_history: VecDeque::with_capacity(100),
            calibration: 0.5,
            learning_rate: 0.1,
        }
    }

    /// Record a confidence judgment and outcome
    pub fn record_judgment(&mut self, confidence: f64, was_correct: bool) {
        self.calibration_history.push_back((confidence, was_correct));
        if self.calibration_history.len() > 100 {
            self.calibration_history.pop_front();
        }

        // Update calibration score
        self.update_calibration();
    }

    /// Update calibration based on history
    fn update_calibration(&mut self) {
        if self.calibration_history.len() < 5 {
            return;
        }

        // Calculate mean absolute calibration error
        // Perfect calibration: 80% confident predictions are correct 80% of time
        let mut buckets: HashMap<u32, (u32, u32)> = HashMap::new(); // bucket -> (total, correct)

        for (conf, correct) in &self.calibration_history {
            let bucket = (*conf * 10.0) as u32; // 0-10 buckets
            let entry = buckets.entry(bucket).or_insert((0, 0));
            entry.0 += 1;
            if *correct {
                entry.1 += 1;
            }
        }

        let mut total_error = 0.0;
        let mut count = 0;
        for (bucket, (total, correct)) in buckets {
            if total >= 3 {
                let expected = bucket as f64 / 10.0;
                let actual = correct as f64 / total as f64;
                total_error += (expected - actual).abs();
                count += 1;
            }
        }

        if count > 0 {
            self.calibration = 1.0 - (total_error / count as f64);
        }
    }

    /// Adjust confidence based on calibration
    pub fn calibrated_confidence(&self, raw_confidence: f64) -> f64 {
        // If we tend to be overconfident, reduce confidence
        // If underconfident, increase it
        if self.calibration < 0.5 {
            // Overconfident - reduce
            raw_confidence * (0.5 + self.calibration)
        } else {
            // Well-calibrated or underconfident - slight boost
            (raw_confidence + (1.0 - raw_confidence) * (self.calibration - 0.5) * 0.5)
                .clamp(0.0, 1.0)
        }
    }

    /// Update current state for a task
    pub fn update_for_task(&mut self, difficulty: f64, familiarity: f64) {
        self.current.confidence = familiarity * (1.0 - difficulty * 0.5);
        self.current.fluency = familiarity;
        self.current.feeling_of_knowing = (familiarity + (1.0 - difficulty)) / 2.0;
        self.current.judgment_of_learning = familiarity * 0.7 + 0.3;
    }
}

impl Default for MetacognitiveMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Autobiographical memory entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifeEvent {
    pub id: Uuid,
    /// What happened
    pub description: String,
    /// When it happened
    pub timestamp: DateTime<Utc>,
    /// Emotional significance (-1 to 1)
    pub emotional_significance: f64,
    /// How important to identity (0-1)
    pub self_relevance: f64,
    /// Related themes
    pub themes: Vec<String>,
    /// Lessons learned
    pub lessons: Vec<String>,
}

impl LifeEvent {
    pub fn new(description: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            description: description.into(),
            timestamp: Utc::now(),
            emotional_significance: 0.0,
            self_relevance: 0.5,
            themes: Vec::new(),
            lessons: Vec::new(),
        }
    }

    pub fn with_emotion(mut self, significance: f64) -> Self {
        self.emotional_significance = significance.clamp(-1.0, 1.0);
        self
    }

    pub fn with_relevance(mut self, relevance: f64) -> Self {
        self.self_relevance = relevance.clamp(0.0, 1.0);
        self
    }

    pub fn with_theme(mut self, theme: impl Into<String>) -> Self {
        self.themes.push(theme.into());
        self
    }

    pub fn with_lesson(mut self, lesson: impl Into<String>) -> Self {
        self.lessons.push(lesson.into());
        self
    }
}

/// The autobiographical self - narrative identity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutobiographicalSelf {
    /// Key life events
    pub events: VecDeque<LifeEvent>,
    /// Maximum events to store
    max_events: usize,
    /// Life chapters (periods of life)
    pub chapters: Vec<LifeChapter>,
    /// Current chapter
    pub current_chapter: Option<String>,
    /// Core life themes
    pub themes: HashMap<String, f64>, // theme -> strength
    /// The narrative "gist" - who I am in a sentence
    pub narrative_gist: Option<String>,
}

/// A chapter in one's life story
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifeChapter {
    pub name: String,
    pub started: DateTime<Utc>,
    pub ended: Option<DateTime<Utc>>,
    pub theme: String,
    pub key_events: Vec<Uuid>,
}

impl AutobiographicalSelf {
    pub fn new() -> Self {
        Self {
            events: VecDeque::with_capacity(1000),
            max_events: 1000,
            chapters: Vec::new(),
            current_chapter: None,
            themes: HashMap::new(),
            narrative_gist: None,
        }
    }

    /// Record a life event
    pub fn record_event(&mut self, event: LifeEvent) {
        // Update themes
        for theme in &event.themes {
            *self.themes.entry(theme.clone()).or_insert(0.0) += event.self_relevance;
        }

        // Store event
        self.events.push_back(event);
        if self.events.len() > self.max_events {
            self.events.pop_front();
        }
    }

    /// Start a new life chapter
    pub fn start_chapter(&mut self, name: impl Into<String>, theme: impl Into<String>) {
        // End current chapter if any
        if let Some(current) = self.chapters.last_mut() {
            if current.ended.is_none() {
                current.ended = Some(Utc::now());
            }
        }

        let chapter = LifeChapter {
            name: name.into(),
            started: Utc::now(),
            ended: None,
            theme: theme.into(),
            key_events: Vec::new(),
        };

        self.current_chapter = Some(chapter.name.clone());
        self.chapters.push(chapter);
    }

    /// Get significant events
    pub fn significant_events(&self, threshold: f64) -> Vec<&LifeEvent> {
        self.events
            .iter()
            .filter(|e| e.emotional_significance.abs() > threshold || e.self_relevance > threshold)
            .collect()
    }

    /// Get dominant life themes
    pub fn dominant_themes(&self) -> Vec<(&String, f64)> {
        let mut themes: Vec<_> = self.themes.iter().map(|(k, v)| (k, *v)).collect();
        themes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        themes.into_iter().take(5).collect()
    }

    /// Update narrative gist
    pub fn update_narrative(&mut self) {
        let themes = self.dominant_themes();
        if themes.is_empty() {
            return;
        }

        let theme_str: Vec<_> = themes.iter().map(|(t, _)| t.as_str()).collect();
        self.narrative_gist = Some(format!(
            "A story of {}",
            theme_str.join(", ")
        ));
    }
}

impl Default for AutobiographicalSelf {
    fn default() -> Self {
        Self::new()
    }
}

/// The complete self-model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfModel {
    /// Self-concept (who I think I am)
    pub concept: SelfConcept,
    /// Metacognitive monitoring
    pub metacognition: MetacognitiveMonitor,
    /// Autobiographical self (life story)
    pub autobiography: AutobiographicalSelf,
    /// Self-awareness level (0-1)
    pub awareness_level: f64,
    /// Current self-reflection depth
    pub reflection_depth: u32,
    /// Inner speech enabled
    pub inner_speech_active: bool,
}

impl SelfModel {
    pub fn new() -> Self {
        Self {
            concept: SelfConcept::new(),
            metacognition: MetacognitiveMonitor::new(),
            autobiography: AutobiographicalSelf::new(),
            awareness_level: 0.5,
            reflection_depth: 0,
            inner_speech_active: true,
        }
    }

    /// Process self-relevant information
    pub fn process_self_info(&mut self, info: &str, emotional_significance: f64) {
        // Record as life event if significant
        if emotional_significance.abs() > 0.3 {
            let event = LifeEvent::new(info)
                .with_emotion(emotional_significance)
                .with_relevance(0.6);
            self.autobiography.record_event(event);
        }

        // Update self-awareness
        self.awareness_level = (self.awareness_level + 0.01).min(1.0);
    }

    /// Begin self-reflection (recursive introspection)
    pub fn begin_reflection(&mut self) {
        self.reflection_depth += 1;
        self.awareness_level = (self.awareness_level + 0.05).min(1.0);
    }

    /// End self-reflection
    pub fn end_reflection(&mut self) {
        self.reflection_depth = self.reflection_depth.saturating_sub(1);
    }

    /// Generate inner speech about current state
    pub fn inner_speech(&self) -> Option<String> {
        if !self.inner_speech_active {
            return None;
        }

        // Generate self-talk based on current state
        let confidence = self.metacognition.current.confidence;
        let esteem = self.concept.self_esteem;

        Some(if confidence < 0.3 {
            "I'm not sure about this...".to_string()
        } else if confidence > 0.8 {
            "I've got this.".to_string()
        } else if esteem < 0.4 {
            "Can I really do this?".to_string()
        } else {
            "Let me think about this...".to_string()
        })
    }

    /// Check coherence of self-model
    pub fn coherence(&self) -> f64 {
        // Measure internal consistency
        let trait_coherence = if self.concept.traits.len() > 1 {
            // Check for contradictory traits (simplified)
            1.0 - (self.concept.traits.len() as f64 * 0.02).min(0.3)
        } else {
            1.0
        };

        let narrative_coherence = if self.autobiography.narrative_gist.is_some() {
            0.8
        } else {
            0.5
        };

        (trait_coherence + narrative_coherence + self.concept.identity_clarity) / 3.0
    }

    /// Get a summary of self
    pub fn self_summary(&self) -> String {
        let traits: Vec<_> = self.concept.core_traits()
            .iter()
            .map(|t| t.name.as_str())
            .collect();

        let themes = self.autobiography.dominant_themes();
        let theme_str: Vec<_> = themes.iter().map(|(t, _)| t.as_str()).collect();

        format!(
            "I am someone who is {} (esteem: {:.0}%). My life themes: {}",
            traits.join(", "),
            self.concept.self_esteem * 100.0,
            theme_str.join(", ")
        )
    }
}

impl Default for SelfModel {
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
    fn test_self_trait() {
        let mut trait_ = SelfTrait::new("curious", 0.8)
            .with_evidence("always asking questions")
            .with_centrality(0.9);

        assert_eq!(trait_.name, "curious");
        assert!(trait_.confidence > 0.5);

        trait_.reinforce();
        assert!(trait_.confidence > 0.6);
    }

    #[test]
    fn test_self_concept() {
        let mut concept = SelfConcept::new();
        
        concept.add_trait(SelfTrait::new("intelligent", 0.7).with_centrality(0.8));
        concept.add_trait(SelfTrait::new("kind", 0.9).with_centrality(0.9));

        assert!(concept.self_esteem > 0.5); // Positive traits = positive self-esteem
        assert_eq!(concept.core_traits().len(), 2);
    }

    #[test]
    fn test_metacognitive_monitor() {
        let mut monitor = MetacognitiveMonitor::new();

        // Record some judgments
        monitor.record_judgment(0.8, true);  // 80% confident, correct
        monitor.record_judgment(0.8, true);
        monitor.record_judgment(0.8, false); // 80% confident, wrong
        monitor.record_judgment(0.3, false); // 30% confident, wrong (correct!)

        // Calibration should be calculated
        assert!(monitor.calibration > 0.0);
    }

    #[test]
    fn test_autobiographical_self() {
        let mut auto = AutobiographicalSelf::new();

        auto.start_chapter("Learning Rust", "growth");

        let event = LifeEvent::new("Built my first brain simulation")
            .with_emotion(0.8)
            .with_relevance(0.9)
            .with_theme("growth")
            .with_lesson("Complex systems can emerge from simple rules");

        auto.record_event(event);

        assert_eq!(auto.events.len(), 1);
        assert!(auto.themes.get("growth").unwrap() > &0.0);
    }

    #[test]
    fn test_self_model() {
        let mut self_model = SelfModel::new();

        self_model.concept.add_trait(
            SelfTrait::new("persistent", 0.7).with_centrality(0.8)
        );

        self_model.autobiography.start_chapter("Building", "creation");

        self_model.process_self_info("Completed a major project", 0.8);

        assert!(self_model.autobiography.events.len() > 0);

        let summary = self_model.self_summary();
        assert!(summary.contains("persistent"));
    }

    #[test]
    fn test_inner_speech() {
        let mut self_model = SelfModel::new();
        
        // Low confidence should generate uncertain speech
        self_model.metacognition.current.confidence = 0.2;
        let speech = self_model.inner_speech();
        assert!(speech.is_some());
        assert!(speech.unwrap().contains("not sure"));
    }
}
