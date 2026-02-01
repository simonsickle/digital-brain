//! Emotion System - Rich Affective Experience
//!
//! Humans don't just have valence/arousal - they experience discrete emotions
//! with specific action tendencies, physiological signatures, and cognitive effects.
//!
//! # Theoretical Foundation
//!
//! Combines multiple emotion theories:
//! - **Basic Emotions** (Ekman): Universal discrete emotions
//! - **Circumplex Model** (Russell): Valence × Arousal space
//! - **Appraisal Theory** (Lazarus): Emotions from event evaluation
//! - **Constructionist** (Barrett): Emotions constructed from core affect + context
//!
//! # Architecture
//!
//! ```text
//! Event → Appraisal → Core Affect → Emotion Category → Action Tendency
//!              ↓            ↓              ↓
//!         Relevance    Valence/       Discrete
//!         Coping      Arousal        Label
//!         Agency
//! ```

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Core affect: the basic building block of emotional experience
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CoreAffect {
    /// Valence: unpleasant (-1) to pleasant (+1)
    pub valence: f64,
    /// Arousal: deactivated (0) to activated (1)
    pub arousal: f64,
}

impl CoreAffect {
    pub fn new(valence: f64, arousal: f64) -> Self {
        Self {
            valence: valence.clamp(-1.0, 1.0),
            arousal: arousal.clamp(0.0, 1.0),
        }
    }

    pub fn neutral() -> Self {
        Self::new(0.0, 0.5)
    }

    /// Blend two affects
    pub fn blend(&self, other: &CoreAffect, weight: f64) -> Self {
        let w = weight.clamp(0.0, 1.0);
        Self::new(
            self.valence * (1.0 - w) + other.valence * w,
            self.arousal * (1.0 - w) + other.arousal * w,
        )
    }

    /// Distance from another affect in circumplex space
    pub fn distance(&self, other: &CoreAffect) -> f64 {
        let dv = self.valence - other.valence;
        let da = self.arousal - other.arousal;
        (dv * dv + da * da).sqrt()
    }
}

impl Default for CoreAffect {
    fn default() -> Self {
        Self::neutral()
    }
}

/// Discrete emotion categories (Ekman's basic emotions + extensions)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EmotionCategory {
    // Primary emotions
    Joy,
    Sadness,
    Fear,
    Anger,
    Surprise,
    Disgust,
    
    // Secondary/complex emotions
    Anticipation,
    Trust,
    Contempt,
    
    // Cognitive emotions
    Interest,
    Confusion,
    Boredom,
    
    // Self-conscious emotions
    Pride,
    Shame,
    Guilt,
    Embarrassment,
    
    // Social emotions
    Love,
    Gratitude,
    Envy,
    Jealousy,
    
    // Blended/neutral
    Neutral,
    Mixed,
}

impl EmotionCategory {
    /// Get the prototypical core affect for this emotion
    pub fn prototype_affect(&self) -> CoreAffect {
        match self {
            // High arousal, positive valence
            Self::Joy => CoreAffect::new(0.8, 0.7),
            Self::Anticipation => CoreAffect::new(0.5, 0.8),
            Self::Interest => CoreAffect::new(0.4, 0.6),
            Self::Surprise => CoreAffect::new(0.2, 0.9),
            Self::Pride => CoreAffect::new(0.7, 0.6),
            Self::Love => CoreAffect::new(0.9, 0.5),
            Self::Gratitude => CoreAffect::new(0.7, 0.4),
            
            // High arousal, negative valence
            Self::Fear => CoreAffect::new(-0.7, 0.9),
            Self::Anger => CoreAffect::new(-0.6, 0.8),
            Self::Disgust => CoreAffect::new(-0.6, 0.5),
            Self::Contempt => CoreAffect::new(-0.4, 0.3),
            Self::Envy => CoreAffect::new(-0.5, 0.6),
            Self::Jealousy => CoreAffect::new(-0.6, 0.7),
            
            // Low arousal, negative valence
            Self::Sadness => CoreAffect::new(-0.7, 0.3),
            Self::Shame => CoreAffect::new(-0.6, 0.4),
            Self::Guilt => CoreAffect::new(-0.5, 0.4),
            Self::Embarrassment => CoreAffect::new(-0.4, 0.6),
            Self::Boredom => CoreAffect::new(-0.3, 0.2),
            
            // Neutral/mixed
            Self::Neutral => CoreAffect::new(0.0, 0.3),
            Self::Confusion => CoreAffect::new(-0.2, 0.5),
            Self::Mixed => CoreAffect::new(0.0, 0.5),
            Self::Trust => CoreAffect::new(0.5, 0.3),
        }
    }

    /// Get the action tendency for this emotion
    pub fn action_tendency(&self) -> ActionTendency {
        match self {
            Self::Joy => ActionTendency::Approach,
            Self::Sadness => ActionTendency::Withdraw,
            Self::Fear => ActionTendency::Escape,
            Self::Anger => ActionTendency::Attack,
            Self::Surprise => ActionTendency::Attend,
            Self::Disgust => ActionTendency::Reject,
            Self::Anticipation => ActionTendency::Approach,
            Self::Trust => ActionTendency::Approach,
            Self::Contempt => ActionTendency::Reject,
            Self::Interest => ActionTendency::Explore,
            Self::Confusion => ActionTendency::Attend,
            Self::Boredom => ActionTendency::Explore,
            Self::Pride => ActionTendency::Display,
            Self::Shame => ActionTendency::Hide,
            Self::Guilt => ActionTendency::Repair,
            Self::Embarrassment => ActionTendency::Hide,
            Self::Love => ActionTendency::Connect,
            Self::Gratitude => ActionTendency::Reciprocate,
            Self::Envy => ActionTendency::Acquire,
            Self::Jealousy => ActionTendency::Protect,
            Self::Neutral => ActionTendency::Monitor,
            Self::Mixed => ActionTendency::Monitor,
        }
    }

    /// Classify a core affect into the nearest emotion category
    pub fn from_affect(affect: &CoreAffect) -> Self {
        let categories = [
            Self::Joy, Self::Sadness, Self::Fear, Self::Anger,
            Self::Surprise, Self::Disgust, Self::Interest, Self::Boredom,
            Self::Anticipation, Self::Trust, Self::Neutral,
        ];

        let mut closest = Self::Neutral;
        let mut min_distance = f64::MAX;

        for cat in categories {
            let distance = affect.distance(&cat.prototype_affect());
            if distance < min_distance {
                min_distance = distance;
                closest = cat;
            }
        }

        closest
    }
}

/// Action tendencies - what emotions make us want to do
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActionTendency {
    Approach,   // Move toward
    Withdraw,   // Move away, disengage
    Escape,     // Flee, avoid
    Attack,     // Confront, aggress
    Attend,     // Focus, orient
    Reject,     // Push away, expel
    Explore,    // Investigate, seek novelty
    Display,    // Show off, broadcast
    Hide,       // Conceal, become invisible
    Repair,     // Fix, make amends
    Connect,    // Bond, affiliate
    Reciprocate,// Return the favor
    Acquire,    // Obtain, possess
    Protect,    // Guard, defend
    Monitor,    // Watch, wait
}

/// Appraisal dimensions - how we evaluate events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Appraisal {
    /// Is this relevant to my goals? (-1 to 1)
    pub relevance: f64,
    /// Is this good or bad for me? (-1 to 1)
    pub valence: f64,
    /// Was this expected? (0 to 1)
    pub expectedness: f64,
    /// Can I cope with this? (0 to 1)
    pub coping_potential: f64,
    /// Who/what caused this? (self=1, other=-1, circumstance=0)
    pub agency: f64,
    /// Is this certain or uncertain? (0 to 1)
    pub certainty: f64,
    /// How much attention does this demand? (0 to 1)
    pub attention_demand: f64,
}

impl Appraisal {
    pub fn new() -> Self {
        Self {
            relevance: 0.0,
            valence: 0.0,
            expectedness: 0.5,
            coping_potential: 0.5,
            agency: 0.0,
            certainty: 0.5,
            attention_demand: 0.5,
        }
    }

    /// Determine emotion from appraisal pattern
    pub fn to_emotion(&self) -> EmotionCategory {
        // High relevance required for strong emotion
        if self.relevance.abs() < 0.2 {
            return EmotionCategory::Neutral;
        }

        // Unexpected = surprise
        if self.expectedness < 0.2 {
            return EmotionCategory::Surprise;
        }

        // Positive valence emotions
        if self.valence > 0.3 {
            if self.agency > 0.5 {
                return EmotionCategory::Pride;
            }
            if self.certainty > 0.7 {
                return EmotionCategory::Joy;
            }
            return EmotionCategory::Anticipation;
        }

        // Negative valence emotions
        if self.valence < -0.3 {
            // Low coping = fear, high coping + other agency = anger
            if self.coping_potential < 0.3 {
                return EmotionCategory::Fear;
            }
            if self.coping_potential > 0.6 && self.agency < -0.3 {
                return EmotionCategory::Anger;
            }
            if self.agency > 0.5 {
                // Self-caused negative = guilt/shame
                if self.certainty > 0.6 {
                    return EmotionCategory::Guilt;
                }
                return EmotionCategory::Shame;
            }
            // Certain loss = sadness
            if self.certainty > 0.6 {
                return EmotionCategory::Sadness;
            }
        }

        // Default based on pure valence
        if self.valence > 0.0 {
            EmotionCategory::Interest
        } else {
            EmotionCategory::Confusion
        }
    }
}

impl Default for Appraisal {
    fn default() -> Self {
        Self::new()
    }
}

/// A complete emotional state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalState {
    /// Current core affect
    pub affect: CoreAffect,
    /// Dominant discrete emotion
    pub dominant_emotion: EmotionCategory,
    /// Intensity of dominant emotion (0 to 1)
    pub intensity: f64,
    /// Secondary emotions present
    pub secondary_emotions: Vec<(EmotionCategory, f64)>,
    /// When this state began
    pub onset: DateTime<Utc>,
    /// What triggered this state
    pub trigger: Option<String>,
}

impl EmotionalState {
    pub fn new(emotion: EmotionCategory, intensity: f64) -> Self {
        Self {
            affect: emotion.prototype_affect(),
            dominant_emotion: emotion,
            intensity: intensity.clamp(0.0, 1.0),
            secondary_emotions: Vec::new(),
            onset: Utc::now(),
            trigger: None,
        }
    }

    pub fn neutral() -> Self {
        Self::new(EmotionCategory::Neutral, 0.3)
    }

    pub fn with_trigger(mut self, trigger: impl Into<String>) -> Self {
        self.trigger = Some(trigger.into());
        self
    }

    pub fn add_secondary(&mut self, emotion: EmotionCategory, intensity: f64) {
        self.secondary_emotions.push((emotion, intensity.clamp(0.0, 1.0)));
    }

    /// Duration of current state
    pub fn duration(&self) -> Duration {
        Utc::now() - self.onset
    }

    /// Get the primary action tendency
    pub fn action_tendency(&self) -> ActionTendency {
        self.dominant_emotion.action_tendency()
    }
}

impl Default for EmotionalState {
    fn default() -> Self {
        Self::neutral()
    }
}

/// Mood: longer-term emotional backdrop
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mood {
    /// Baseline affect
    pub baseline: CoreAffect,
    /// Current mood deviation from baseline
    pub current: CoreAffect,
    /// Mood stability (how easily it changes)
    pub stability: f64,
    /// Recent emotional events affecting mood
    pub recent_emotions: VecDeque<(EmotionalState, DateTime<Utc>)>,
}

impl Mood {
    pub fn new() -> Self {
        Self {
            baseline: CoreAffect::new(0.1, 0.4), // Slightly positive, moderate arousal
            current: CoreAffect::new(0.1, 0.4),
            stability: 0.7,
            recent_emotions: VecDeque::with_capacity(20),
        }
    }

    /// Update mood based on emotional event
    pub fn process_emotion(&mut self, emotion: &EmotionalState) {
        // Record the emotion
        self.recent_emotions.push_back((emotion.clone(), Utc::now()));
        if self.recent_emotions.len() > 20 {
            self.recent_emotions.pop_front();
        }

        // Mood shifts toward emotion, modulated by intensity and stability
        let shift_amount = emotion.intensity * (1.0 - self.stability) * 0.1;
        self.current = self.current.blend(&emotion.affect, shift_amount);
    }

    /// Regulate mood back toward baseline
    pub fn regulate(&mut self) {
        let regulation_rate = 0.02;
        self.current = self.current.blend(&self.baseline, regulation_rate);
    }

    /// Get overall mood valence
    pub fn valence(&self) -> f64 {
        self.current.valence
    }

    /// Is mood positive?
    pub fn is_positive(&self) -> bool {
        self.current.valence > 0.1
    }

    /// Is mood negative?
    pub fn is_negative(&self) -> bool {
        self.current.valence < -0.1
    }
}

impl Default for Mood {
    fn default() -> Self {
        Self::new()
    }
}

/// Emotion regulation strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegulationStrategy {
    /// Change the situation
    SituationModification,
    /// Change attention focus
    AttentionalDeployment,
    /// Reframe/reinterpret
    CognitiveReappraisal,
    /// Suppress expression
    ExpressionSuppression,
    /// Accept and observe
    Acceptance,
    /// Problem-focused coping
    ProblemSolving,
    /// Seek social support
    SocialSupport,
    /// Distraction
    Distraction,
}

impl RegulationStrategy {
    /// Get appropriate strategy for an emotion
    pub fn for_emotion(emotion: EmotionCategory, intensity: f64) -> Self {
        match emotion {
            EmotionCategory::Fear if intensity > 0.7 => Self::SituationModification,
            EmotionCategory::Fear => Self::CognitiveReappraisal,
            EmotionCategory::Anger if intensity > 0.8 => Self::AttentionalDeployment,
            EmotionCategory::Anger => Self::CognitiveReappraisal,
            EmotionCategory::Sadness if intensity > 0.6 => Self::SocialSupport,
            EmotionCategory::Sadness => Self::Acceptance,
            EmotionCategory::Shame | EmotionCategory::Guilt => Self::CognitiveReappraisal,
            EmotionCategory::Boredom => Self::SituationModification,
            EmotionCategory::Confusion => Self::ProblemSolving,
            _ => Self::Acceptance,
        }
    }

    /// Effectiveness of strategy (0-1)
    pub fn effectiveness(&self) -> f64 {
        match self {
            Self::CognitiveReappraisal => 0.8,
            Self::ProblemSolving => 0.75,
            Self::Acceptance => 0.7,
            Self::SocialSupport => 0.7,
            Self::AttentionalDeployment => 0.6,
            Self::SituationModification => 0.65,
            Self::Distraction => 0.5,
            Self::ExpressionSuppression => 0.3, // Least effective long-term
        }
    }
}

/// The complete emotion system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionSystem {
    /// Current emotional state
    pub state: EmotionalState,
    /// Current mood
    pub mood: Mood,
    /// Emotion history
    pub history: VecDeque<EmotionalState>,
    /// Maximum history size
    max_history: usize,
    /// Emotional reactivity (how strongly events affect emotions)
    pub reactivity: f64,
    /// Emotion regulation ability
    pub regulation_ability: f64,
}

impl EmotionSystem {
    pub fn new() -> Self {
        Self {
            state: EmotionalState::neutral(),
            mood: Mood::new(),
            history: VecDeque::with_capacity(100),
            max_history: 100,
            reactivity: 0.6,
            regulation_ability: 0.5,
        }
    }

    /// Process an event and generate emotional response
    pub fn process_event(&mut self, appraisal: &Appraisal, trigger: Option<&str>) {
        // Determine emotion from appraisal
        let emotion_category = appraisal.to_emotion();
        
        // Calculate intensity based on relevance and reactivity
        // Use max of relevance and attention_demand, boosted by the other
        let base_intensity = appraisal.relevance.abs().max(appraisal.attention_demand)
            * (1.0 + appraisal.relevance.abs().min(appraisal.attention_demand) * 0.5);
        let intensity = (base_intensity * self.reactivity).clamp(0.0, 1.0);

        // Mood influences emotional response
        let mood_influence = if self.mood.is_negative() && appraisal.valence < 0.0 {
            1.2 // Negative mood amplifies negative emotions
        } else if self.mood.is_positive() && appraisal.valence > 0.0 {
            1.1 // Positive mood slightly amplifies positive emotions
        } else {
            1.0
        };

        let final_intensity = (intensity * mood_influence).clamp(0.0, 1.0);

        // Create new emotional state
        let mut new_state = EmotionalState::new(emotion_category, final_intensity);
        if let Some(t) = trigger {
            new_state = new_state.with_trigger(t);
        }

        // Update affect based on appraisal
        new_state.affect = CoreAffect::new(
            appraisal.valence,
            (appraisal.relevance.abs() + appraisal.attention_demand) / 2.0,
        );

        // Store old state in history
        if self.state.intensity > 0.2 {
            self.history.push_back(self.state.clone());
            if self.history.len() > self.max_history {
                self.history.pop_front();
            }
        }

        // Update state
        self.state = new_state.clone();

        // Update mood
        self.mood.process_emotion(&new_state);
    }

    /// Attempt to regulate current emotion
    pub fn regulate(&mut self) -> Option<RegulationStrategy> {
        // Only regulate negative high-intensity emotions
        if self.state.affect.valence > -0.2 || self.state.intensity < 0.5 {
            return None;
        }

        let strategy = RegulationStrategy::for_emotion(
            self.state.dominant_emotion,
            self.state.intensity,
        );

        // Apply regulation based on ability and strategy effectiveness
        let reduction = strategy.effectiveness() * self.regulation_ability * 0.3;
        self.state.intensity = (self.state.intensity - reduction).max(0.1);

        // Slightly improve affect
        self.state.affect.valence = (self.state.affect.valence + reduction * 0.5).min(0.0);

        Some(strategy)
    }

    /// Natural decay of emotional intensity
    pub fn decay(&mut self) {
        // Emotions naturally fade
        let decay_rate = 0.05;
        self.state.intensity = (self.state.intensity - decay_rate).max(0.1);

        // If intensity drops enough, shift toward neutral
        if self.state.intensity < 0.2 {
            self.state.dominant_emotion = EmotionCategory::Neutral;
            self.state.affect = self.state.affect.blend(&CoreAffect::neutral(), 0.1);
        }

        // Mood regulation
        self.mood.regulate();
    }

    /// Get current valence
    pub fn valence(&self) -> f64 {
        self.state.affect.valence
    }

    /// Get current arousal
    pub fn arousal(&self) -> f64 {
        self.state.affect.arousal
    }

    /// Is currently experiencing strong emotion?
    pub fn is_emotional(&self) -> bool {
        self.state.intensity > 0.5
    }

    /// Get dominant action tendency
    pub fn action_tendency(&self) -> ActionTendency {
        self.state.action_tendency()
    }
}

impl Default for EmotionSystem {
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
    fn test_core_affect() {
        let happy = CoreAffect::new(0.8, 0.7);
        let sad = CoreAffect::new(-0.7, 0.3);

        assert!(happy.valence > 0.0);
        assert!(sad.valence < 0.0);

        let blended = happy.blend(&sad, 0.5);
        assert!(blended.valence.abs() < 0.2); // Should be near neutral
    }

    #[test]
    fn test_emotion_from_affect() {
        let happy_affect = CoreAffect::new(0.8, 0.7);
        let emotion = EmotionCategory::from_affect(&happy_affect);
        assert_eq!(emotion, EmotionCategory::Joy);

        let fearful_affect = CoreAffect::new(-0.7, 0.9);
        let emotion = EmotionCategory::from_affect(&fearful_affect);
        assert_eq!(emotion, EmotionCategory::Fear);
    }

    #[test]
    fn test_appraisal_to_emotion() {
        // High relevance, negative valence, low coping = fear
        let fear_appraisal = Appraisal {
            relevance: 0.9,
            valence: -0.8,
            expectedness: 0.6,
            coping_potential: 0.2,
            agency: -0.5,
            certainty: 0.7,
            attention_demand: 0.9,
        };
        assert_eq!(fear_appraisal.to_emotion(), EmotionCategory::Fear);

        // Unexpected event = surprise
        let surprise_appraisal = Appraisal {
            relevance: 0.8,
            valence: 0.5,
            expectedness: 0.1,
            coping_potential: 0.5,
            agency: 0.0,
            certainty: 0.3,
            attention_demand: 0.8,
        };
        assert_eq!(surprise_appraisal.to_emotion(), EmotionCategory::Surprise);
    }

    #[test]
    fn test_emotion_system() {
        let mut system = EmotionSystem::new();

        // Process a threatening event
        let threat = Appraisal {
            relevance: 0.9,
            valence: -0.8,
            expectedness: 0.3,
            coping_potential: 0.2,
            agency: -0.8,
            certainty: 0.8,
            attention_demand: 0.9,
        };

        system.process_event(&threat, Some("danger"));

        assert!(system.state.intensity > 0.5);
        assert!(system.valence() < 0.0);
        assert!(system.is_emotional());
    }

    #[test]
    fn test_emotion_regulation() {
        let mut system = EmotionSystem::new();
        system.state = EmotionalState::new(EmotionCategory::Anger, 0.8);
        system.state.affect = CoreAffect::new(-0.6, 0.8);

        let initial_intensity = system.state.intensity;
        system.regulate();

        assert!(system.state.intensity < initial_intensity);
    }

    #[test]
    fn test_mood_influence() {
        let mut system = EmotionSystem::new();
        
        // Set negative mood
        system.mood.current = CoreAffect::new(-0.5, 0.4);

        // Process another negative event
        let negative_event = Appraisal {
            relevance: 0.6,
            valence: -0.5,
            ..Default::default()
        };

        system.process_event(&negative_event, None);

        // Mood should amplify the negative emotion
        assert!(system.state.intensity > 0.3);
    }
}
