//! Hypothalamus - The Drive and Homeostasis Center
//!
//! The hypothalamus regulates:
//! - **Homeostasis**: Maintaining internal balance (temperature, energy)
//! - **Basic drives**: Hunger, thirst, fatigue, social needs
//! - **Circadian rhythms**: Sleep-wake cycles
//! - **Stress response**: HPA axis activation
//! - **Motivated behavior**: Drive states that push toward action
//!
//! # Theoretical Foundation
//!
//! Based on:
//! - Hull's Drive Reduction Theory: Behavior motivated by drive states
//! - Homeostatic regulation: Set points and error correction
//! - Allostasis: Predictive regulation, anticipating needs
//!
//! # Architecture
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────────┐
//! │                      HYPOTHALAMUS                              │
//! ├──────────────┬──────────────┬──────────────┬──────────────────┤
//! │   Lateral    │  Ventromedial│   SCN       │   Paraventricular │
//! │   (hunger)   │   (satiety)  │  (clock)    │   (stress/HPA)    │
//! └──────────────┴──────────────┴──────────────┴──────────────────┘
//! ```

use chrono::{DateTime, Duration, Timelike, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// DRIVE STATES
// ============================================================================

/// Types of basic drives
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DriveType {
    /// Need for energy/food
    Hunger,
    /// Need for water
    Thirst,
    /// Need for rest
    Fatigue,
    /// Need for social connection
    Social,
    /// Need for safety
    Safety,
    /// Need for novelty/stimulation (boredom)
    Stimulation,
    /// Need for achievement
    Achievement,
    /// Need for autonomy
    Autonomy,
}

impl DriveType {
    /// Get the typical decay rate (how fast this drive builds up)
    pub fn decay_rate(&self) -> f64 {
        match self {
            Self::Hunger => 0.01,      // Builds slowly
            Self::Thirst => 0.015,     // Builds faster than hunger
            Self::Fatigue => 0.008,    // Builds very slowly
            Self::Social => 0.005,     // Builds slowly
            Self::Safety => 0.0,       // Event-driven, not time-based
            Self::Stimulation => 0.02, // Builds quickly (boredom)
            Self::Achievement => 0.003,// Builds very slowly
            Self::Autonomy => 0.002,   // Builds very slowly
        }
    }

    /// Get the urgency multiplier (how much this drive demands attention)
    pub fn urgency(&self) -> f64 {
        match self {
            Self::Hunger => 1.2,
            Self::Thirst => 1.5, // More urgent
            Self::Fatigue => 1.0,
            Self::Social => 0.7,
            Self::Safety => 2.0, // Very urgent
            Self::Stimulation => 0.5,
            Self::Achievement => 0.6,
            Self::Autonomy => 0.5,
        }
    }

    /// All drive types
    pub fn all() -> Vec<Self> {
        vec![
            Self::Hunger,
            Self::Thirst,
            Self::Fatigue,
            Self::Social,
            Self::Safety,
            Self::Stimulation,
            Self::Achievement,
            Self::Autonomy,
        ]
    }
}

/// A drive state - internal need level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriveState {
    /// Type of drive
    pub drive_type: DriveType,
    /// Current level (0 = satisfied, 1 = desperate)
    pub level: f64,
    /// Set point (target level, usually 0)
    pub set_point: f64,
    /// Last time this drive was satisfied
    pub last_satisfied: DateTime<Utc>,
    /// How many times this drive has been satisfied
    pub satisfaction_count: u64,
}

impl DriveState {
    pub fn new(drive_type: DriveType) -> Self {
        Self {
            drive_type,
            level: 0.1, // Start slightly above zero
            set_point: 0.0,
            last_satisfied: Utc::now(),
            satisfaction_count: 0,
        }
    }

    /// Get the drive error (how far from set point)
    pub fn error(&self) -> f64 {
        (self.level - self.set_point).max(0.0)
    }

    /// Get the motivational strength of this drive
    pub fn motivation(&self) -> f64 {
        self.error() * self.drive_type.urgency()
    }

    /// Satisfy this drive (reduce level)
    pub fn satisfy(&mut self, amount: f64) {
        self.level = (self.level - amount).max(0.0);
        self.last_satisfied = Utc::now();
        self.satisfaction_count += 1;
    }

    /// Increase drive level
    pub fn increase(&mut self, amount: f64) {
        self.level = (self.level + amount).min(1.0);
    }

    /// Time since last satisfied
    pub fn time_since_satisfied(&self) -> Duration {
        Utc::now() - self.last_satisfied
    }

    /// Is this drive urgent (needs attention)?
    pub fn is_urgent(&self) -> bool {
        self.level > 0.7
    }

    /// Is this drive critical?
    pub fn is_critical(&self) -> bool {
        self.level > 0.9
    }
}

// ============================================================================
// CIRCADIAN RHYTHM
// ============================================================================

/// Circadian phase
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CircadianPhase {
    /// Early morning (6-9)
    Morning,
    /// Late morning / early afternoon (9-14)
    Day,
    /// Afternoon (14-18)
    Afternoon,
    /// Evening (18-22)
    Evening,
    /// Night (22-6)
    Night,
}

impl CircadianPhase {
    /// Get phase from hour (0-23)
    pub fn from_hour(hour: u32) -> Self {
        match hour {
            6..=8 => Self::Morning,
            9..=13 => Self::Day,
            14..=17 => Self::Afternoon,
            18..=21 => Self::Evening,
            _ => Self::Night,
        }
    }

    /// Get alertness modifier for this phase
    pub fn alertness_modifier(&self) -> f64 {
        match self {
            Self::Morning => 0.7,    // Waking up
            Self::Day => 1.0,        // Peak alertness
            Self::Afternoon => 0.8,  // Post-lunch dip
            Self::Evening => 0.6,    // Winding down
            Self::Night => 0.3,      // Should be sleeping
        }
    }

    /// Is this a good time to sleep?
    pub fn should_sleep(&self) -> bool {
        matches!(self, Self::Night)
    }
}

/// Circadian rhythm tracker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircadianRhythm {
    /// Current phase
    pub phase: CircadianPhase,
    /// Sleep debt (accumulated missed sleep)
    pub sleep_debt: f64,
    /// Time of last sleep
    pub last_sleep: DateTime<Utc>,
    /// Hours slept in last sleep session
    pub last_sleep_duration: f64,
    /// Preferred sleep time (hour)
    pub sleep_preference: u32,
    /// Preferred wake time (hour)
    pub wake_preference: u32,
}

impl CircadianRhythm {
    pub fn new() -> Self {
        Self {
            phase: CircadianPhase::from_hour(Utc::now().hour()),
            sleep_debt: 0.0,
            last_sleep: Utc::now(),
            last_sleep_duration: 8.0,
            sleep_preference: 23,
            wake_preference: 7,
        }
    }

    /// Update phase based on current time
    pub fn update(&mut self) {
        let hour = Utc::now().hour();
        self.phase = CircadianPhase::from_hour(hour);

        // Accumulate sleep debt if awake during night
        if self.phase == CircadianPhase::Night {
            self.sleep_debt += 0.1;
        }
    }

    /// Record a sleep session
    pub fn record_sleep(&mut self, hours: f64) {
        self.last_sleep = Utc::now();
        self.last_sleep_duration = hours;
        // Reduce sleep debt
        self.sleep_debt = (self.sleep_debt - hours * 0.1).max(0.0);
    }

    /// Get current alertness (0-1)
    pub fn alertness(&self) -> f64 {
        let base = self.phase.alertness_modifier();
        let debt_penalty = self.sleep_debt * 0.1;
        (base - debt_penalty).clamp(0.1, 1.0)
    }

    /// Hours since last sleep
    pub fn hours_awake(&self) -> f64 {
        (Utc::now() - self.last_sleep).num_minutes() as f64 / 60.0
    }

    /// Should we sleep now?
    pub fn should_sleep(&self) -> bool {
        self.phase.should_sleep() || self.hours_awake() > 16.0 || self.sleep_debt > 0.5
    }
}

impl Default for CircadianRhythm {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// STRESS RESPONSE (HPA AXIS)
// ============================================================================

/// Stress response state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressResponse {
    /// Current cortisol level (0 = none, 1 = maximum)
    pub cortisol: f64,
    /// Baseline cortisol
    pub baseline: f64,
    /// Time since stress onset
    pub stress_onset: Option<DateTime<Utc>>,
    /// Chronic stress accumulator
    pub chronic_stress: f64,
    /// Recovery rate
    pub recovery_rate: f64,
}

impl StressResponse {
    pub fn new() -> Self {
        Self {
            cortisol: 0.2, // Normal baseline
            baseline: 0.2,
            stress_onset: None,
            chronic_stress: 0.0,
            recovery_rate: 0.05,
        }
    }

    /// Trigger acute stress response
    pub fn trigger(&mut self, intensity: f64) {
        self.cortisol = (self.cortisol + intensity * 0.5).min(1.0);
        if self.stress_onset.is_none() {
            self.stress_onset = Some(Utc::now());
        }
    }

    /// Update stress (recovery toward baseline)
    pub fn update(&mut self) {
        // Recover toward baseline
        if self.cortisol > self.baseline {
            self.cortisol = (self.cortisol - self.recovery_rate).max(self.baseline);
        }

        // If recovered, clear onset
        if self.cortisol <= self.baseline + 0.05 {
            self.stress_onset = None;
        }

        // Prolonged stress becomes chronic
        if let Some(onset) = self.stress_onset {
            let duration = (Utc::now() - onset).num_hours() as f64;
            if duration > 1.0 {
                self.chronic_stress = (self.chronic_stress + 0.01).min(1.0);
            }
        }

        // Chronic stress slowly recovers
        self.chronic_stress = (self.chronic_stress - 0.001).max(0.0);
    }

    /// Is currently stressed?
    pub fn is_stressed(&self) -> bool {
        self.cortisol > 0.4
    }

    /// Is chronically stressed?
    pub fn is_chronic(&self) -> bool {
        self.chronic_stress > 0.3
    }

    /// Get stress level (combines acute and chronic)
    pub fn level(&self) -> f64 {
        (self.cortisol + self.chronic_stress * 0.5).min(1.0)
    }
}

impl Default for StressResponse {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// HYPOTHALAMUS
// ============================================================================

/// Configuration for the hypothalamus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HypothalamusConfig {
    /// How sensitive to drive states
    pub drive_sensitivity: f64,
    /// How quickly drives build up
    pub drive_buildup_rate: f64,
    /// How easily stressed
    pub stress_reactivity: f64,
}

impl Default for HypothalamusConfig {
    fn default() -> Self {
        Self {
            drive_sensitivity: 0.5,
            drive_buildup_rate: 1.0,
            stress_reactivity: 0.5,
        }
    }
}

/// The Hypothalamus - drives, homeostasis, and circadian regulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hypothalamus {
    /// Configuration
    config: HypothalamusConfig,
    /// Drive states
    pub drives: HashMap<DriveType, DriveState>,
    /// Circadian rhythm
    pub circadian: CircadianRhythm,
    /// Stress response system
    pub stress: StressResponse,
    /// Last update time
    last_update: DateTime<Utc>,
}

impl Hypothalamus {
    pub fn new() -> Self {
        Self::with_config(HypothalamusConfig::default())
    }

    pub fn with_config(config: HypothalamusConfig) -> Self {
        let mut drives = HashMap::new();
        for drive_type in DriveType::all() {
            drives.insert(drive_type, DriveState::new(drive_type));
        }

        Self {
            config,
            drives,
            circadian: CircadianRhythm::new(),
            stress: StressResponse::new(),
            last_update: Utc::now(),
        }
    }

    /// Update all systems (call periodically)
    pub fn update(&mut self) {
        let now = Utc::now();
        let elapsed_minutes = (now - self.last_update).num_minutes() as f64;
        self.last_update = now;

        // Update drives (they build up over time)
        for (drive_type, state) in &mut self.drives {
            let increase = drive_type.decay_rate() * elapsed_minutes * self.config.drive_buildup_rate;
            state.increase(increase);
        }

        // Update circadian
        self.circadian.update();

        // Fatigue increases based on hours awake
        if let Some(fatigue) = self.drives.get_mut(&DriveType::Fatigue) {
            let hours_awake = self.circadian.hours_awake();
            if hours_awake > 8.0 {
                fatigue.increase(0.01 * (hours_awake - 8.0));
            }
        }

        // Update stress
        self.stress.update();
    }

    /// Satisfy a drive
    pub fn satisfy_drive(&mut self, drive_type: DriveType, amount: f64) {
        if let Some(state) = self.drives.get_mut(&drive_type) {
            state.satisfy(amount);
        }
    }

    /// Get a specific drive state
    pub fn get_drive(&self, drive_type: DriveType) -> Option<&DriveState> {
        self.drives.get(&drive_type)
    }

    /// Get the most urgent drive
    pub fn most_urgent_drive(&self) -> Option<(&DriveType, &DriveState)> {
        self.drives
            .iter()
            .max_by(|a, b| a.1.motivation().partial_cmp(&b.1.motivation()).unwrap())
    }

    /// Get all urgent drives
    pub fn urgent_drives(&self) -> Vec<(&DriveType, &DriveState)> {
        self.drives
            .iter()
            .filter(|(_, s)| s.is_urgent())
            .collect()
    }

    /// Trigger stress response
    pub fn trigger_stress(&mut self, intensity: f64) {
        let adjusted = intensity * self.config.stress_reactivity;
        self.stress.trigger(adjusted);

        // Stress affects safety drive
        if let Some(safety) = self.drives.get_mut(&DriveType::Safety) {
            safety.increase(adjusted * 0.3);
        }
    }

    /// Record sleep
    pub fn sleep(&mut self, hours: f64) {
        self.circadian.record_sleep(hours);
        self.satisfy_drive(DriveType::Fatigue, hours * 0.1);
    }

    /// Get overall motivational state
    pub fn motivation_summary(&self) -> MotivationSummary {
        let total_drive: f64 = self.drives.values().map(|d| d.motivation()).sum();
        let urgent: Vec<String> = self.urgent_drives()
            .iter()
            .map(|(dt, _)| format!("{:?}", dt))
            .collect();

        MotivationSummary {
            total_drive_pressure: total_drive,
            urgent_drives: urgent,
            alertness: self.circadian.alertness(),
            stress_level: self.stress.level(),
            should_rest: self.circadian.should_sleep(),
        }
    }

    /// Statistics
    pub fn stats(&self) -> HypothalamusStats {
        let mut drive_levels = HashMap::new();
        for (dt, state) in &self.drives {
            drive_levels.insert(format!("{:?}", dt), state.level);
        }

        HypothalamusStats {
            drive_levels,
            circadian_phase: format!("{:?}", self.circadian.phase),
            alertness: self.circadian.alertness(),
            hours_awake: self.circadian.hours_awake(),
            sleep_debt: self.circadian.sleep_debt,
            cortisol: self.stress.cortisol,
            chronic_stress: self.stress.chronic_stress,
            is_stressed: self.stress.is_stressed(),
        }
    }
}

impl Default for Hypothalamus {
    fn default() -> Self {
        Self::new()
    }
}

/// Motivation summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotivationSummary {
    pub total_drive_pressure: f64,
    pub urgent_drives: Vec<String>,
    pub alertness: f64,
    pub stress_level: f64,
    pub should_rest: bool,
}

/// Hypothalamus statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HypothalamusStats {
    pub drive_levels: HashMap<String, f64>,
    pub circadian_phase: String,
    pub alertness: f64,
    pub hours_awake: f64,
    pub sleep_debt: f64,
    pub cortisol: f64,
    pub chronic_stress: f64,
    pub is_stressed: bool,
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drive_state() {
        let mut hunger = DriveState::new(DriveType::Hunger);
        assert!(hunger.level < 0.5);

        hunger.increase(0.7);
        assert!(hunger.level > 0.5);
        assert!(hunger.is_urgent()); // level > 0.7

        hunger.satisfy(0.3);
        assert!(hunger.level < 0.6);
    }

    #[test]
    fn test_drive_motivation() {
        let mut thirst = DriveState::new(DriveType::Thirst);
        thirst.level = 0.8;

        let mut hunger = DriveState::new(DriveType::Hunger);
        hunger.level = 0.8;

        // Thirst should be more urgent at same level
        assert!(thirst.motivation() > hunger.motivation());
    }

    #[test]
    fn test_circadian_phase() {
        assert_eq!(CircadianPhase::from_hour(7), CircadianPhase::Morning);
        assert_eq!(CircadianPhase::from_hour(12), CircadianPhase::Day);
        assert_eq!(CircadianPhase::from_hour(15), CircadianPhase::Afternoon);
        assert_eq!(CircadianPhase::from_hour(20), CircadianPhase::Evening);
        assert_eq!(CircadianPhase::from_hour(2), CircadianPhase::Night);
    }

    #[test]
    fn test_circadian_alertness() {
        let mut circadian = CircadianRhythm::new();
        circadian.phase = CircadianPhase::Day;
        let day_alert = circadian.alertness();

        circadian.phase = CircadianPhase::Night;
        let night_alert = circadian.alertness();

        assert!(day_alert > night_alert);
    }

    #[test]
    fn test_stress_response() {
        let mut stress = StressResponse::new();
        let baseline = stress.cortisol;

        stress.trigger(0.8);
        assert!(stress.cortisol > baseline);
        assert!(stress.is_stressed());

        // Recover
        for _ in 0..20 {
            stress.update();
        }
        assert!(stress.cortisol < 0.5);
    }

    #[test]
    fn test_hypothalamus_drives() {
        let mut hypo = Hypothalamus::new();

        // Increase hunger
        hypo.drives.get_mut(&DriveType::Hunger).unwrap().increase(0.8);

        let urgent = hypo.urgent_drives();
        assert!(!urgent.is_empty());

        let most = hypo.most_urgent_drive();
        assert!(most.is_some());
    }

    #[test]
    fn test_hypothalamus_satisfy() {
        let mut hypo = Hypothalamus::new();
        hypo.drives.get_mut(&DriveType::Hunger).unwrap().level = 0.9;

        hypo.satisfy_drive(DriveType::Hunger, 0.5);

        let hunger = hypo.get_drive(DriveType::Hunger).unwrap();
        assert!(hunger.level < 0.5);
    }

    #[test]
    fn test_hypothalamus_stress() {
        let mut hypo = Hypothalamus::new();

        hypo.trigger_stress(1.0); // High intensity to exceed threshold after reactivity scaling

        assert!(hypo.stress.is_stressed());
        let safety = hypo.get_drive(DriveType::Safety).unwrap();
        assert!(safety.level > 0.1);
    }

    #[test]
    fn test_hypothalamus_sleep() {
        let mut hypo = Hypothalamus::new();
        hypo.drives.get_mut(&DriveType::Fatigue).unwrap().level = 0.8;
        hypo.circadian.sleep_debt = 0.5;

        hypo.sleep(8.0);

        let fatigue = hypo.get_drive(DriveType::Fatigue).unwrap();
        assert!(fatigue.level < 0.8);
        assert!(hypo.circadian.sleep_debt < 0.5);
    }

    #[test]
    fn test_hypothalamus_stats() {
        let hypo = Hypothalamus::new();
        let stats = hypo.stats();

        assert!(!stats.drive_levels.is_empty());
        assert!(stats.alertness > 0.0);
    }
}
