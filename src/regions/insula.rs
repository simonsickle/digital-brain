//! Insula - The Interoceptive Hub
//!
//! The insula is crucial for:
//! - **Interoception**: Sensing internal body states (heartbeat, breathing, gut)
//! - **Emotional awareness**: Feeling emotions, not just processing them
//! - **Empathy**: Mirroring others' internal states
//! - **Risk/uncertainty**: Anticipating aversive outcomes
//! - **Disgust**: Both physical and moral
//!
//! # Theoretical Foundation
//!
//! Based on:
//! - Craig's interoceptive model: Insula as seat of subjective feelings
//! - Damasio's somatic marker hypothesis: Body states guide decisions
//! - Singer's empathy research: Insula mirrors others' pain/emotions
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────┐
//! │                         INSULA                               │
//! ├────────────────┬────────────────┬────────────────┬──────────┤
//! │   Posterior    │    Mid        │    Anterior    │  Fronto- │
//! │   (body sense) │   (integrate) │   (awareness)  │  Insular │
//! │   heartbeat    │   body+emotion│   subjective   │  (social)│
//! │   breathing    │   integration │   feeling      │  empathy │
//! └────────────────┴────────────────┴────────────────┴──────────┘
//! ```
//!
//! The posterior insula receives raw body signals, mid-insula integrates
//! with emotional context, anterior insula generates subjective awareness,
//! and fronto-insular cortex handles social/empathic processing.

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

// ============================================================================
// BODY STATE (Interoception)
// ============================================================================

/// Internal body state representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BodyState {
    /// Heart rate relative to baseline (0.5 = slow, 1.0 = normal, 1.5 = fast)
    pub heart_rate: f64,
    /// Breathing rate (0 = slow/calm, 1 = rapid/anxious)
    pub breathing: f64,
    /// Gut feelings / visceral state (-1 = nausea/dread, 0 = neutral, 1 = butterflies/excitement)
    pub gut: f64,
    /// Muscle tension (0 = relaxed, 1 = tense)
    pub tension: f64,
    /// Energy level (0 = exhausted, 1 = energized)
    pub energy: f64,
    /// Temperature sensation (-1 = cold, 0 = comfortable, 1 = hot)
    pub temperature: f64,
    /// Pain level (0 = none, 1 = severe)
    pub pain: f64,
    /// When this state was recorded
    pub timestamp: DateTime<Utc>,
}

impl BodyState {
    pub fn new() -> Self {
        Self {
            heart_rate: 1.0,
            breathing: 0.3,
            gut: 0.0,
            tension: 0.2,
            energy: 0.7,
            temperature: 0.0,
            pain: 0.0,
            timestamp: Utc::now(),
        }
    }

    /// Create a stressed body state
    pub fn stressed() -> Self {
        Self {
            heart_rate: 1.3,
            breathing: 0.7,
            gut: -0.3,
            tension: 0.7,
            energy: 0.5,
            temperature: 0.2,
            pain: 0.0,
            timestamp: Utc::now(),
        }
    }

    /// Create a relaxed body state
    pub fn relaxed() -> Self {
        Self {
            heart_rate: 0.8,
            breathing: 0.2,
            gut: 0.1,
            tension: 0.1,
            energy: 0.6,
            temperature: 0.0,
            pain: 0.0,
            timestamp: Utc::now(),
        }
    }

    /// Overall arousal level derived from body state
    pub fn arousal(&self) -> f64 {
        let arousal = (self.heart_rate - 0.5) * 0.3
            + self.breathing * 0.25
            + self.tension * 0.25
            + (1.0 - self.energy) * 0.1
            + self.pain * 0.1;
        arousal.clamp(0.0, 1.0)
    }

    /// Overall valence derived from body state (somatic marker)
    pub fn valence(&self) -> f64 {
        let valence = self.gut * 0.4 + self.energy * 0.3
            - self.tension * 0.2
            - self.pain * 0.3
            - (self.heart_rate - 1.0).abs() * 0.1;
        valence.clamp(-1.0, 1.0)
    }

    /// Is the body in an alert state?
    pub fn is_alert(&self) -> bool {
        self.heart_rate > 1.1 || self.breathing > 0.5 || self.tension > 0.5
    }

    /// Is the body in a calm state?
    pub fn is_calm(&self) -> bool {
        self.heart_rate < 1.0 && self.breathing < 0.4 && self.tension < 0.3
    }

    /// Blend toward another state (for gradual transitions)
    pub fn blend(&self, target: &BodyState, rate: f64) -> BodyState {
        let r = rate.clamp(0.0, 1.0);
        BodyState {
            heart_rate: self.heart_rate * (1.0 - r) + target.heart_rate * r,
            breathing: self.breathing * (1.0 - r) + target.breathing * r,
            gut: self.gut * (1.0 - r) + target.gut * r,
            tension: self.tension * (1.0 - r) + target.tension * r,
            energy: self.energy * (1.0 - r) + target.energy * r,
            temperature: self.temperature * (1.0 - r) + target.temperature * r,
            pain: self.pain * (1.0 - r) + target.pain * r,
            timestamp: Utc::now(),
        }
    }
}

impl Default for BodyState {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// SUBJECTIVE FEELINGS
// ============================================================================

/// A subjective feeling - the "what it's like" of experience
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubjectiveFeeling {
    /// Label for the feeling
    pub label: String,
    /// Intensity (0-1)
    pub intensity: f64,
    /// Pleasant vs unpleasant (-1 to 1)
    pub valence: f64,
    /// Bodily location (if any)
    pub body_location: Option<String>,
    /// Associated thought/trigger
    pub trigger: Option<String>,
    /// When this feeling arose
    pub onset: DateTime<Utc>,
}

impl SubjectiveFeeling {
    pub fn new(label: impl Into<String>, intensity: f64, valence: f64) -> Self {
        Self {
            label: label.into(),
            intensity: intensity.clamp(0.0, 1.0),
            valence: valence.clamp(-1.0, 1.0),
            body_location: None,
            trigger: None,
            onset: Utc::now(),
        }
    }

    pub fn with_location(mut self, location: impl Into<String>) -> Self {
        self.body_location = Some(location.into());
        self
    }

    pub fn with_trigger(mut self, trigger: impl Into<String>) -> Self {
        self.trigger = Some(trigger.into());
        self
    }

    /// Duration since feeling arose
    pub fn duration(&self) -> Duration {
        Utc::now() - self.onset
    }

    /// Is this a strong feeling?
    pub fn is_strong(&self) -> bool {
        self.intensity > 0.6
    }
}

// ============================================================================
// DISGUST SYSTEM
// ============================================================================

/// Types of disgust (insula is central to disgust processing)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DisgustType {
    /// Core disgust - food, bodily fluids, decay
    Core,
    /// Animal-reminder - mortality salience, bodily envelope
    AnimalReminder,
    /// Interpersonal - contact with undesirable people
    Interpersonal,
    /// Moral - violations of purity, fairness
    Moral,
}

impl DisgustType {
    /// Get the typical intensity for this type
    pub fn base_intensity(&self) -> f64 {
        match self {
            Self::Core => 0.8,
            Self::AnimalReminder => 0.6,
            Self::Interpersonal => 0.5,
            Self::Moral => 0.7,
        }
    }
}

/// A disgust response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisgustResponse {
    pub disgust_type: DisgustType,
    pub intensity: f64,
    pub trigger: String,
    pub body_response: BodyState,
    pub timestamp: DateTime<Utc>,
}

impl DisgustResponse {
    pub fn new(disgust_type: DisgustType, trigger: impl Into<String>, intensity: f64) -> Self {
        // Disgust triggers specific body state
        let mut body = BodyState::new();
        body.gut = -0.5 - intensity * 0.3; // Nausea
        body.tension = 0.3 + intensity * 0.2;
        body.breathing = 0.2; // Breath holding

        Self {
            disgust_type,
            intensity: intensity.clamp(0.0, 1.0),
            trigger: trigger.into(),
            body_response: body,
            timestamp: Utc::now(),
        }
    }
}

// ============================================================================
// EMPATHY
// ============================================================================

/// Empathic response to another's state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmpathicResponse {
    /// Who we're empathizing with
    pub target: String,
    /// What state we're mirroring
    pub mirrored_state: String,
    /// How strongly we feel it (0-1)
    pub resonance: f64,
    /// Our resulting body state change
    pub body_effect: BodyState,
    /// When this started
    pub onset: DateTime<Utc>,
}

impl EmpathicResponse {
    pub fn new(target: impl Into<String>, state: impl Into<String>, resonance: f64) -> Self {
        Self {
            target: target.into(),
            mirrored_state: state.into(),
            resonance: resonance.clamp(0.0, 1.0),
            body_effect: BodyState::new(),
            onset: Utc::now(),
        }
    }

    /// Create empathic pain response
    pub fn pain(target: impl Into<String>, intensity: f64) -> Self {
        let mut body = BodyState::new();
        body.tension = 0.3 + intensity * 0.4;
        body.gut = -0.2 * intensity;

        Self {
            target: target.into(),
            mirrored_state: "pain".to_string(),
            resonance: intensity * 0.7, // We don't feel full intensity
            body_effect: body,
            onset: Utc::now(),
        }
    }

    /// Create empathic joy response
    pub fn joy(target: impl Into<String>, intensity: f64) -> Self {
        let mut body = BodyState::new();
        body.energy = 0.6 + intensity * 0.3;
        body.gut = 0.2 * intensity;
        body.tension = (0.2 - intensity * 0.1).max(0.0);

        Self {
            target: target.into(),
            mirrored_state: "joy".to_string(),
            resonance: intensity * 0.6,
            body_effect: body,
            onset: Utc::now(),
        }
    }
}

// ============================================================================
// RISK / UNCERTAINTY ANTICIPATION
// ============================================================================

/// Anticipated risk/uncertainty (insula activates for uncertain outcomes)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAnticipation {
    /// What we're anticipating
    pub situation: String,
    /// Probability of bad outcome (0-1)
    pub risk_level: f64,
    /// How uncertain we are (0 = certain, 1 = very uncertain)
    pub uncertainty: f64,
    /// Potential magnitude of loss
    pub potential_loss: f64,
    /// Body state generated by anticipation
    pub body_response: BodyState,
    /// When this anticipation started
    pub onset: DateTime<Utc>,
}

impl RiskAnticipation {
    pub fn new(situation: impl Into<String>, risk: f64, uncertainty: f64) -> Self {
        let mut body = BodyState::new();

        // Risk and uncertainty both trigger arousal
        let arousal_factor = (risk * 0.5 + uncertainty * 0.5).clamp(0.0, 1.0);
        body.heart_rate = 1.0 + arousal_factor * 0.4;
        body.breathing = 0.3 + arousal_factor * 0.4;
        body.tension = 0.2 + arousal_factor * 0.5;
        body.gut = -arousal_factor * 0.4; // Dread

        Self {
            situation: situation.into(),
            risk_level: risk.clamp(0.0, 1.0),
            uncertainty: uncertainty.clamp(0.0, 1.0),
            potential_loss: 0.5,
            body_response: body,
            onset: Utc::now(),
        }
    }

    pub fn with_loss(mut self, loss: f64) -> Self {
        self.potential_loss = loss.clamp(0.0, 1.0);
        // Higher stakes = stronger body response
        let stakes_factor = self.potential_loss;
        self.body_response.tension += stakes_factor * 0.2;
        self.body_response.gut -= stakes_factor * 0.2;
        self
    }

    /// Overall threat level
    pub fn threat_level(&self) -> f64 {
        (self.risk_level * self.potential_loss * (1.0 + self.uncertainty * 0.5)).clamp(0.0, 1.0)
    }
}

// ============================================================================
// INSULA SYSTEM
// ============================================================================

/// Configuration for the insula
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InsulaConfig {
    /// How sensitive to body signals (interoceptive accuracy)
    pub interoceptive_sensitivity: f64,
    /// How strongly we empathize
    pub empathy_strength: f64,
    /// Disgust sensitivity
    pub disgust_sensitivity: f64,
    /// Risk sensitivity
    pub risk_sensitivity: f64,
    /// How quickly body state returns to baseline
    pub homeostatic_rate: f64,
}

impl Default for InsulaConfig {
    fn default() -> Self {
        Self {
            interoceptive_sensitivity: 0.6,
            empathy_strength: 0.5,
            disgust_sensitivity: 0.5,
            risk_sensitivity: 0.5,
            homeostatic_rate: 0.1,
        }
    }
}

/// The Insula - interoceptive and emotional awareness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Insula {
    /// Configuration
    config: InsulaConfig,
    /// Current body state
    pub body_state: BodyState,
    /// Baseline body state (homeostatic target)
    baseline: BodyState,
    /// Current subjective feelings
    pub feelings: Vec<SubjectiveFeeling>,
    /// Recent empathic responses
    pub empathic_responses: VecDeque<EmpathicResponse>,
    /// Active risk anticipations
    pub risk_anticipations: Vec<RiskAnticipation>,
    /// Recent disgust responses
    pub disgust_history: VecDeque<DisgustResponse>,
    /// Body state history for awareness
    body_history: VecDeque<BodyState>,
    /// Maximum history size
    max_history: usize,
}

impl Insula {
    pub fn new() -> Self {
        Self::with_config(InsulaConfig::default())
    }

    pub fn with_config(config: InsulaConfig) -> Self {
        Self {
            config,
            body_state: BodyState::new(),
            baseline: BodyState::new(),
            feelings: Vec::new(),
            empathic_responses: VecDeque::with_capacity(20),
            risk_anticipations: Vec::new(),
            disgust_history: VecDeque::with_capacity(50),
            body_history: VecDeque::with_capacity(100),
            max_history: 100,
        }
    }

    /// Update body state based on emotional input
    pub fn process_emotion(&mut self, valence: f64, arousal: f64, label: Option<&str>) {
        // Emotions affect body state
        let target = BodyState {
            heart_rate: 1.0 + arousal * 0.3,
            breathing: 0.3 + arousal * 0.4,
            gut: valence * 0.4,
            tension: 0.2 + arousal * 0.3 - valence * 0.1,
            energy: self.body_state.energy, // Energy changes slower
            temperature: arousal * 0.2,
            pain: self.body_state.pain,
            timestamp: Utc::now(),
        };

        // Blend toward target based on sensitivity
        self.body_state = self
            .body_state
            .blend(&target, self.config.interoceptive_sensitivity * 0.5);

        // Generate subjective feeling
        if let Some(lbl) = label {
            let feeling = SubjectiveFeeling::new(lbl, arousal, valence);
            self.add_feeling(feeling);
        }
    }

    /// Process a threat/stressor
    pub fn process_threat(&mut self, description: &str, threat_level: f64) {
        // Immediate body response
        let stress_state = BodyState {
            heart_rate: 1.0 + threat_level * 0.5,
            breathing: 0.3 + threat_level * 0.5,
            gut: -threat_level * 0.5,
            tension: 0.3 + threat_level * 0.5,
            energy: self.body_state.energy,
            temperature: threat_level * 0.3,
            pain: self.body_state.pain,
            timestamp: Utc::now(),
        };

        self.body_state = self.body_state.blend(&stress_state, 0.4);

        // Create risk anticipation
        let risk = RiskAnticipation::new(description, threat_level, 0.5);
        self.risk_anticipations.push(risk);

        // Add feeling
        let feeling = SubjectiveFeeling::new("anxiety", threat_level, -threat_level * 0.7)
            .with_trigger(description);
        self.add_feeling(feeling);
    }

    /// Process disgust trigger
    pub fn process_disgust(&mut self, trigger: &str, disgust_type: DisgustType) {
        let intensity = disgust_type.base_intensity() * self.config.disgust_sensitivity;
        let response = DisgustResponse::new(disgust_type, trigger, intensity);

        // Update body state
        self.body_state = self.body_state.blend(&response.body_response, 0.5);

        // Add to history
        self.disgust_history.push_back(response);
        if self.disgust_history.len() > 50 {
            self.disgust_history.pop_front();
        }

        // Add feeling
        let feeling = SubjectiveFeeling::new("disgust", intensity, -0.8)
            .with_location("stomach")
            .with_trigger(trigger);
        self.add_feeling(feeling);
    }

    /// Integrate autonomic feedback from brainstem (bodily state updates).
    pub fn integrate_autonomic_feedback(&mut self, body_state: BodyState) {
        let blended = self
            .body_state
            .blend(&body_state, self.config.homeostatic_rate);
        self.body_state = blended.clone();
        self.body_history.push_back(blended);

        if self.body_history.len() > self.max_history {
            self.body_history.pop_front();
        }

        if self.body_state.is_alert() {
            let feeling = SubjectiveFeeling::new("physiological_alert", 0.6, -0.2);
            self.add_feeling(feeling);
        } else if self.body_state.is_calm() {
            let feeling = SubjectiveFeeling::new("physiological_calm", 0.4, 0.2);
            self.add_feeling(feeling);
        }
    }

    /// Empathize with another agent's state
    pub fn empathize(
        &mut self,
        target: &str,
        their_state: &str,
        their_valence: f64,
        their_arousal: f64,
    ) {
        let resonance = self.config.empathy_strength * their_arousal;

        let response = if their_valence > 0.3 {
            EmpathicResponse::joy(target, their_arousal)
        } else if their_valence < -0.3 {
            EmpathicResponse::pain(target, their_arousal)
        } else {
            EmpathicResponse::new(target, their_state, resonance)
        };

        // Mirror their state partially in our body
        self.body_state = self
            .body_state
            .blend(&response.body_effect, resonance * 0.3);

        // Record empathic response
        self.empathic_responses.push_back(response);
        if self.empathic_responses.len() > 20 {
            self.empathic_responses.pop_front();
        }
    }

    /// Add a subjective feeling
    pub fn add_feeling(&mut self, feeling: SubjectiveFeeling) {
        // Remove weak/old feelings first
        self.feelings
            .retain(|f| f.intensity > 0.2 && f.duration() < Duration::minutes(30));
        self.feelings.push(feeling);
    }

    /// Get the dominant current feeling
    pub fn dominant_feeling(&self) -> Option<&SubjectiveFeeling> {
        self.feelings
            .iter()
            .max_by(|a, b| a.intensity.partial_cmp(&b.intensity).unwrap())
    }

    /// Get current interoceptive awareness summary
    pub fn awareness_summary(&self) -> String {
        let arousal = self.body_state.arousal();
        let valence = self.body_state.valence();

        let arousal_desc = if arousal > 0.7 {
            "highly activated"
        } else if arousal > 0.4 {
            "moderately aroused"
        } else {
            "calm"
        };

        let valence_desc = if valence > 0.3 {
            "pleasant"
        } else if valence < -0.3 {
            "uncomfortable"
        } else {
            "neutral"
        };

        let feeling_desc = self
            .dominant_feeling()
            .map(|f| format!(", feeling {}", f.label))
            .unwrap_or_default();

        format!(
            "Body state: {}, {} overall{}",
            arousal_desc, valence_desc, feeling_desc
        )
    }

    /// Update - homeostatic regulation and decay
    pub fn update(&mut self) {
        // Record history
        self.body_history.push_back(self.body_state.clone());
        if self.body_history.len() > self.max_history {
            self.body_history.pop_front();
        }

        // Gradually return to baseline (homeostasis)
        self.body_state = self
            .body_state
            .blend(&self.baseline, self.config.homeostatic_rate);

        // Decay feelings
        for feeling in &mut self.feelings {
            feeling.intensity *= 0.95;
        }
        self.feelings.retain(|f| f.intensity > 0.1);

        // Clear old risk anticipations
        self.risk_anticipations
            .retain(|r| r.onset + Duration::hours(1) > Utc::now());
    }

    /// Get somatic marker for a decision
    /// Returns body-based intuition about a choice
    pub fn somatic_marker(&self, choice_description: &str) -> f64 {
        // Check if we have learned associations with this type of choice
        let choice_lower = choice_description.to_lowercase();

        // Check disgust history
        let disgust_signal: f64 = self
            .disgust_history
            .iter()
            .filter(|d| choice_lower.contains(&d.trigger.to_lowercase()))
            .map(|d| -d.intensity)
            .sum::<f64>()
            .clamp(-1.0, 0.0);

        // Check risk anticipations
        let risk_signal: f64 = self
            .risk_anticipations
            .iter()
            .filter(|r| choice_lower.contains(&r.situation.to_lowercase()))
            .map(|r| -r.threat_level())
            .sum::<f64>()
            .clamp(-1.0, 0.0);

        // Current body state contributes
        let body_signal = self.body_state.valence() * 0.3;

        // Combine signals (gut feeling)
        (disgust_signal + risk_signal + body_signal).clamp(-1.0, 1.0)
    }

    /// Statistics
    pub fn stats(&self) -> InsulaStats {
        InsulaStats {
            current_arousal: self.body_state.arousal(),
            current_valence: self.body_state.valence(),
            active_feelings: self.feelings.len(),
            dominant_feeling: self.dominant_feeling().map(|f| f.label.clone()),
            recent_empathic: self.empathic_responses.len(),
            active_risks: self.risk_anticipations.len(),
            disgust_count: self.disgust_history.len(),
            is_calm: self.body_state.is_calm(),
            is_alert: self.body_state.is_alert(),
        }
    }
}

impl Default for Insula {
    fn default() -> Self {
        Self::new()
    }
}

/// Insula statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InsulaStats {
    pub current_arousal: f64,
    pub current_valence: f64,
    pub active_feelings: usize,
    pub dominant_feeling: Option<String>,
    pub recent_empathic: usize,
    pub active_risks: usize,
    pub disgust_count: usize,
    pub is_calm: bool,
    pub is_alert: bool,
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_body_state_arousal() {
        let calm = BodyState::relaxed();
        let stressed = BodyState::stressed();

        assert!(calm.arousal() < stressed.arousal());
        assert!(calm.is_calm());
        assert!(stressed.is_alert());
    }

    #[test]
    fn test_body_state_valence() {
        let mut good = BodyState::new();
        good.gut = 0.5;
        good.energy = 0.9;

        let mut bad = BodyState::new();
        bad.gut = -0.5;
        bad.pain = 0.5;

        assert!(good.valence() > bad.valence());
    }

    #[test]
    fn test_body_state_blend() {
        let calm = BodyState::relaxed();
        let stressed = BodyState::stressed();
        let mid = calm.blend(&stressed, 0.5);

        assert!(mid.heart_rate > calm.heart_rate);
        assert!(mid.heart_rate < stressed.heart_rate);
    }

    #[test]
    fn test_subjective_feeling() {
        let feeling = SubjectiveFeeling::new("joy", 0.8, 0.9)
            .with_location("chest")
            .with_trigger("good news");

        assert!(feeling.is_strong());
        assert_eq!(feeling.body_location, Some("chest".to_string()));
    }

    #[test]
    fn test_disgust_response() {
        let disgust = DisgustResponse::new(DisgustType::Moral, "unfair treatment", 0.7);

        assert!(disgust.body_response.gut < 0.0);
        assert_eq!(disgust.disgust_type, DisgustType::Moral);
    }

    #[test]
    fn test_empathic_response() {
        let pain_empathy = EmpathicResponse::pain("friend", 0.8);
        let joy_empathy = EmpathicResponse::joy("friend", 0.8);

        assert!(pain_empathy.body_effect.tension > joy_empathy.body_effect.tension);
        assert!(joy_empathy.body_effect.energy > pain_empathy.body_effect.energy);
    }

    #[test]
    fn test_risk_anticipation() {
        let low_risk = RiskAnticipation::new("minor issue", 0.2, 0.3);
        let high_risk = RiskAnticipation::new("major threat", 0.8, 0.7).with_loss(0.9);

        assert!(high_risk.threat_level() > low_risk.threat_level());
        assert!(high_risk.body_response.tension > low_risk.body_response.tension);
    }

    #[test]
    fn test_insula_emotion_processing() {
        let mut insula = Insula::new();
        let initial_hr = insula.body_state.heart_rate;

        // Process high-arousal negative emotion
        insula.process_emotion(-0.7, 0.9, Some("fear"));

        // Body should respond
        assert!(insula.body_state.heart_rate > initial_hr);
        assert!(insula.body_state.tension > 0.3);

        // Should have a feeling
        assert!(!insula.feelings.is_empty());
    }

    #[test]
    fn test_insula_empathy() {
        let mut insula = Insula::new();

        // Empathize with someone in pain
        insula.empathize("friend", "pain", -0.8, 0.7);

        // Should have empathic response recorded
        assert!(!insula.empathic_responses.is_empty());
    }

    #[test]
    fn test_insula_somatic_marker() {
        let mut insula = Insula::new();

        // Create disgust association
        insula.process_disgust("unfair deal", DisgustType::Moral);

        // Somatic marker should be negative for similar choice
        let marker = insula.somatic_marker("considering an unfair deal");
        assert!(marker < 0.0);
    }

    #[test]
    fn test_insula_homeostasis() {
        let mut insula = Insula::new();

        // Stress the system
        insula.process_threat("danger", 0.9);
        let stressed_hr = insula.body_state.heart_rate;

        // Multiple updates should return toward baseline
        for _ in 0..20 {
            insula.update();
        }

        assert!(insula.body_state.heart_rate < stressed_hr);
    }

    #[test]
    fn test_insula_stats() {
        let mut insula = Insula::new();
        insula.process_emotion(0.5, 0.3, Some("contentment"));

        let stats = insula.stats();
        assert_eq!(stats.active_feelings, 1);
        assert_eq!(stats.dominant_feeling, Some("contentment".to_string()));
    }
}
