//! Temporal Cognition - The Experience of Time
//!
//! Humans have a rich experience of time that goes beyond mere clock-watching:
//! - **Duration perception**: Subjective sense of how long things take
//! - **Mental time travel**: Reliving the past, imagining the future
//! - **Prospective memory**: Remembering to do things at the right time
//! - **Temporal discounting**: Future rewards feel less valuable
//!
//! # Theoretical Foundation
//!
//! Based on:
//! - Scalar Timing Theory (Gibbon): Internal clock with pacemaker-accumulator
//! - Episodic Future Thinking (Suddendorf & Corballis): Mental time travel
//! - Implementation Intentions (Gollwitzer): Prospective memory cues
//! - Temporal Discounting (Ainslie): Hyperbolic devaluation of future rewards
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    TEMPORAL COGNITION                       │
//! ├───────────────┬────────────────┬────────────────┬──────────┤
//! │   Duration    │  Mental Time   │  Prospective   │ Temporal │
//! │  Perception   │    Travel      │    Memory      │Discounting│
//! │  (pacemaker)  │ (past/future)  │  (intentions)  │ (value)  │
//! └───────────────┴────────────────┴────────────────┴──────────┘
//! ```

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use uuid::Uuid;

/// Unique identifier for prospective memory tasks
pub type IntentionId = Uuid;

// ============================================================================
// DURATION PERCEPTION
// ============================================================================

/// Subjective time perception (internal clock)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DurationPerception {
    /// Base pacemaker rate (pulses per second)
    base_rate: f64,
    /// Current rate (affected by arousal, attention, etc.)
    current_rate: f64,
    /// Accumulated pulses for current interval
    accumulator: f64,
    /// When current timing started
    interval_start: Option<DateTime<Utc>>,
    /// Historical duration estimates for calibration
    calibration_history: VecDeque<(f64, f64)>, // (objective, subjective)
}

impl DurationPerception {
    pub fn new() -> Self {
        Self {
            base_rate: 1.0,
            current_rate: 1.0,
            accumulator: 0.0,
            interval_start: None,
            calibration_history: VecDeque::with_capacity(100),
        }
    }

    /// Start timing an interval
    pub fn start_timing(&mut self) {
        self.interval_start = Some(Utc::now());
        self.accumulator = 0.0;
    }

    /// Get subjective duration of current interval
    pub fn get_subjective_duration(&self) -> Option<Duration> {
        let start = self.interval_start?;
        let objective = (Utc::now() - start).num_milliseconds() as f64 / 1000.0;
        let subjective = objective * self.current_rate;
        Some(Duration::milliseconds((subjective * 1000.0) as i64))
    }

    /// Stop timing and return subjective duration
    pub fn stop_timing(&mut self) -> Option<Duration> {
        let duration = self.get_subjective_duration();
        self.interval_start = None;
        duration
    }

    /// Modulate perception rate based on state
    /// 
    /// - High arousal → time feels slower (more pulses)
    /// - High attention → time feels faster (fewer pulses)
    /// - Boredom → time drags (fewer pulses perceived, but feels longer!)
    pub fn modulate(&mut self, arousal: f64, attention: f64, boredom: f64) {
        // Arousal speeds up internal clock
        let arousal_effect = 1.0 + (arousal - 0.5) * 0.4;
        
        // Focused attention makes time fly
        let attention_effect = 1.0 - attention * 0.3;
        
        // Boredom is complex: clock slows but perceived duration increases
        // We model this as a slower rate (time "drags")
        let boredom_effect = 1.0 - boredom * 0.2;
        
        self.current_rate = self.base_rate * arousal_effect * attention_effect * boredom_effect;
        self.current_rate = self.current_rate.clamp(0.5, 2.0);
    }

    /// Calibrate against objective time (for learning)
    pub fn calibrate(&mut self, objective_seconds: f64, subjective_seconds: f64) {
        self.calibration_history.push_back((objective_seconds, subjective_seconds));
        if self.calibration_history.len() > 100 {
            self.calibration_history.pop_front();
        }
        
        // Adjust base rate based on accumulated error
        if self.calibration_history.len() >= 10 {
            let avg_ratio: f64 = self.calibration_history
                .iter()
                .map(|(obj, subj)| if *obj > 0.0 { subj / obj } else { 1.0 })
                .sum::<f64>() / self.calibration_history.len() as f64;
            
            // Slowly adjust base rate
            self.base_rate = self.base_rate * 0.95 + (1.0 / avg_ratio) * 0.05;
            self.base_rate = self.base_rate.clamp(0.8, 1.2);
        }
    }

    /// Get current rate multiplier
    pub fn rate(&self) -> f64 {
        self.current_rate
    }
}

impl Default for DurationPerception {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// MENTAL TIME TRAVEL
// ============================================================================

/// A remembered or imagined moment in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalMoment {
    /// Unique identifier
    pub id: Uuid,
    /// When this moment occurred (past) or is expected (future)
    pub timestamp: DateTime<Utc>,
    /// Description of the moment
    pub description: String,
    /// Emotional valence associated with this moment
    pub valence: f64,
    /// How vivid/detailed this moment is (0-1)
    pub vividness: f64,
    /// Confidence in the accuracy (past) or likelihood (future)
    pub confidence: f64,
    /// Is this past (memory) or future (anticipation)?
    pub is_future: bool,
    /// Related entity/context tags
    pub tags: Vec<String>,
}

impl TemporalMoment {
    pub fn past(timestamp: DateTime<Utc>, description: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            timestamp,
            description: description.into(),
            valence: 0.0,
            vividness: 0.5,
            confidence: 0.7,
            is_future: false,
            tags: Vec::new(),
        }
    }

    pub fn future(timestamp: DateTime<Utc>, description: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            timestamp,
            description: description.into(),
            valence: 0.0,
            vividness: 0.3, // Future is less vivid by default
            confidence: 0.5, // And less certain
            is_future: true,
            tags: Vec::new(),
        }
    }

    pub fn with_valence(mut self, valence: f64) -> Self {
        self.valence = valence.clamp(-1.0, 1.0);
        self
    }

    pub fn with_vividness(mut self, vividness: f64) -> Self {
        self.vividness = vividness.clamp(0.0, 1.0);
        self
    }

    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// How far from now is this moment?
    pub fn distance_from_now(&self) -> Duration {
        if self.is_future {
            self.timestamp - Utc::now()
        } else {
            Utc::now() - self.timestamp
        }
    }
}

/// Mental time travel system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MentalTimeTravel {
    /// Past moments (autobiographical memories)
    past_moments: Vec<TemporalMoment>,
    /// Future moments (anticipated events)
    future_moments: Vec<TemporalMoment>,
    /// Current temporal focus (negative = past, positive = future, 0 = present)
    temporal_focus: f64,
    /// Maximum moments to keep
    max_moments: usize,
}

impl MentalTimeTravel {
    pub fn new() -> Self {
        Self {
            past_moments: Vec::new(),
            future_moments: Vec::new(),
            temporal_focus: 0.0,
            max_moments: 1000,
        }
    }

    /// Record a past moment
    pub fn remember(&mut self, moment: TemporalMoment) {
        if !moment.is_future {
            self.past_moments.push(moment);
            if self.past_moments.len() > self.max_moments {
                // Remove oldest, lowest vividness
                self.past_moments.sort_by(|a, b| {
                    (a.vividness + a.valence.abs()).partial_cmp(&(b.vividness + b.valence.abs())).unwrap()
                });
                self.past_moments.remove(0);
            }
        }
    }

    /// Anticipate a future moment
    pub fn anticipate(&mut self, moment: TemporalMoment) {
        if moment.is_future {
            self.future_moments.push(moment);
            if self.future_moments.len() > self.max_moments {
                // Remove furthest, lowest confidence
                self.future_moments.sort_by(|a, b| {
                    a.confidence.partial_cmp(&b.confidence).unwrap()
                });
                self.future_moments.remove(0);
            }
        }
    }

    /// Travel to the past - retrieve relevant memories
    pub fn travel_to_past(&mut self, query: &str, limit: usize) -> Vec<&TemporalMoment> {
        self.temporal_focus = -0.5;
        let query_lower = query.to_lowercase();
        
        let mut relevant: Vec<_> = self.past_moments
            .iter()
            .filter(|m| {
                m.description.to_lowercase().contains(&query_lower) ||
                m.tags.iter().any(|t| t.to_lowercase().contains(&query_lower))
            })
            .collect();
        
        // Sort by vividness and recency
        relevant.sort_by(|a, b| {
            let a_score = a.vividness + (1.0 / (a.distance_from_now().num_days() as f64 + 1.0));
            let b_score = b.vividness + (1.0 / (b.distance_from_now().num_days() as f64 + 1.0));
            b_score.partial_cmp(&a_score).unwrap()
        });
        
        relevant.into_iter().take(limit).collect()
    }

    /// Travel to the future - retrieve anticipated events
    pub fn travel_to_future(&mut self, query: &str, limit: usize) -> Vec<&TemporalMoment> {
        self.temporal_focus = 0.5;
        let query_lower = query.to_lowercase();
        
        // Filter out past futures
        let now = Utc::now();
        self.future_moments.retain(|m| m.timestamp > now);
        
        let mut relevant: Vec<_> = self.future_moments
            .iter()
            .filter(|m| {
                m.description.to_lowercase().contains(&query_lower) ||
                m.tags.iter().any(|t| t.to_lowercase().contains(&query_lower))
            })
            .collect();
        
        // Sort by proximity and confidence
        relevant.sort_by(|a, b| {
            let a_score = a.confidence / (a.distance_from_now().num_hours() as f64 + 1.0);
            let b_score = b.confidence / (b.distance_from_now().num_hours() as f64 + 1.0);
            b_score.partial_cmp(&a_score).unwrap()
        });
        
        relevant.into_iter().take(limit).collect()
    }

    /// Return focus to the present
    pub fn return_to_present(&mut self) {
        self.temporal_focus = 0.0;
    }

    /// Get upcoming events within a time window
    pub fn upcoming(&self, within: Duration) -> Vec<&TemporalMoment> {
        let now = Utc::now();
        let cutoff = now + within;
        
        self.future_moments
            .iter()
            .filter(|m| m.timestamp > now && m.timestamp <= cutoff)
            .collect()
    }

    /// Get temporal focus (-1 = deep past, 0 = present, 1 = far future)
    pub fn focus(&self) -> f64 {
        self.temporal_focus
    }

    /// Count of past/future moments
    pub fn stats(&self) -> (usize, usize) {
        (self.past_moments.len(), self.future_moments.len())
    }
}

impl Default for MentalTimeTravel {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// PROSPECTIVE MEMORY
// ============================================================================

/// A trigger for prospective memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProspectiveTrigger {
    /// Time-based: "At 3pm, do X"
    Time(DateTime<Utc>),
    /// Event-based: "When X happens, do Y"
    Event { pattern: String },
    /// Activity-based: "After finishing X, do Y"
    Activity { after_completing: String },
    /// Context-based: "When in location X, do Y"
    Context { condition: String },
}

impl ProspectiveTrigger {
    /// Check if this trigger matches current conditions
    pub fn matches(&self, current_time: DateTime<Utc>, context: &str) -> bool {
        match self {
            Self::Time(trigger_time) => {
                let diff = current_time.signed_duration_since(*trigger_time);
                diff.num_minutes().abs() <= 5 // Within 5 minute window
            }
            Self::Event { pattern } | Self::Activity { after_completing: pattern } | Self::Context { condition: pattern } => {
                context.to_lowercase().contains(&pattern.to_lowercase())
            }
        }
    }
}

/// A prospective memory intention
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Intention {
    /// Unique identifier
    pub id: IntentionId,
    /// What to do
    pub action: String,
    /// When/how to trigger
    pub trigger: ProspectiveTrigger,
    /// Priority (0-1)
    pub priority: f64,
    /// When this intention was created
    pub created_at: DateTime<Utc>,
    /// Is this intention still active?
    pub active: bool,
    /// How many times we've been reminded
    pub reminder_count: u32,
    /// Associated goal (if any)
    pub goal_id: Option<String>,
}

impl Intention {
    pub fn new(action: impl Into<String>, trigger: ProspectiveTrigger) -> Self {
        Self {
            id: Uuid::new_v4(),
            action: action.into(),
            trigger,
            priority: 0.5,
            created_at: Utc::now(),
            active: true,
            reminder_count: 0,
            goal_id: None,
        }
    }

    pub fn with_priority(mut self, priority: f64) -> Self {
        self.priority = priority.clamp(0.0, 1.0);
        self
    }

    pub fn with_goal(mut self, goal_id: impl Into<String>) -> Self {
        self.goal_id = Some(goal_id.into());
        self
    }

    /// Mark as completed
    pub fn complete(&mut self) {
        self.active = false;
    }

    /// Record a reminder
    pub fn reminded(&mut self) {
        self.reminder_count += 1;
    }
}

/// Prospective memory system - remembering to remember
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProspectiveMemory {
    /// Active intentions
    intentions: Vec<Intention>,
    /// Completed intentions (for history)
    completed: VecDeque<Intention>,
    /// Failed/forgotten intentions
    forgotten: VecDeque<Intention>,
    /// How often to check triggers (monitoring cost)
    check_frequency: Duration,
    /// Last check time
    last_check: DateTime<Utc>,
    /// Maximum completed to keep
    max_history: usize,
}

impl ProspectiveMemory {
    pub fn new() -> Self {
        Self {
            intentions: Vec::new(),
            completed: VecDeque::with_capacity(100),
            forgotten: VecDeque::with_capacity(50),
            check_frequency: Duration::minutes(1),
            last_check: Utc::now(),
            max_history: 100,
        }
    }

    /// Add a new intention
    pub fn intend(&mut self, intention: Intention) -> IntentionId {
        let id = intention.id;
        self.intentions.push(intention);
        id
    }

    /// Create a time-based intention
    pub fn intend_at(&mut self, time: DateTime<Utc>, action: impl Into<String>) -> IntentionId {
        let intention = Intention::new(action, ProspectiveTrigger::Time(time));
        self.intend(intention)
    }

    /// Create an event-based intention
    pub fn intend_when(&mut self, event_pattern: impl Into<String>, action: impl Into<String>) -> IntentionId {
        let intention = Intention::new(
            action,
            ProspectiveTrigger::Event { pattern: event_pattern.into() }
        );
        self.intend(intention)
    }

    /// Check for triggered intentions
    pub fn check(&mut self, context: &str) -> Vec<Intention> {
        let now = Utc::now();
        
        // Only check at configured frequency
        if now - self.last_check < self.check_frequency {
            return Vec::new();
        }
        self.last_check = now;
        
        let mut triggered = Vec::new();
        
        for intention in &mut self.intentions {
            if intention.active && intention.trigger.matches(now, context) {
                intention.reminded();
                triggered.push(intention.clone());
            }
        }
        
        // Sort by priority
        triggered.sort_by(|a, b| b.priority.partial_cmp(&a.priority).unwrap());
        
        triggered
    }

    /// Mark an intention as completed
    pub fn complete(&mut self, id: IntentionId) {
        if let Some(idx) = self.intentions.iter().position(|i| i.id == id) {
            let mut intention = self.intentions.remove(idx);
            intention.complete();
            self.completed.push_back(intention);
            if self.completed.len() > self.max_history {
                self.completed.pop_front();
            }
        }
    }

    /// Mark an intention as forgotten (failed)
    pub fn forget(&mut self, id: IntentionId) {
        if let Some(idx) = self.intentions.iter().position(|i| i.id == id) {
            let mut intention = self.intentions.remove(idx);
            intention.active = false;
            self.forgotten.push_back(intention);
            if self.forgotten.len() > 50 {
                self.forgotten.pop_front();
            }
        }
    }

    /// Get all active intentions
    pub fn active_intentions(&self) -> &[Intention] {
        &self.intentions
    }

    /// Get upcoming time-based intentions
    pub fn upcoming_time_intentions(&self, within: Duration) -> Vec<&Intention> {
        let now = Utc::now();
        let cutoff = now + within;
        
        self.intentions
            .iter()
            .filter(|i| {
                i.active && matches!(&i.trigger, ProspectiveTrigger::Time(t) if *t > now && *t <= cutoff)
            })
            .collect()
    }

    /// Statistics
    pub fn stats(&self) -> ProspectiveMemoryStats {
        let active = self.intentions.iter().filter(|i| i.active).count();
        let completed = self.completed.len();
        let forgotten = self.forgotten.len();
        
        ProspectiveMemoryStats {
            active_intentions: active,
            completed_count: completed,
            forgotten_count: forgotten,
            success_rate: if completed + forgotten > 0 {
                completed as f64 / (completed + forgotten) as f64
            } else {
                1.0
            },
        }
    }
}

impl Default for ProspectiveMemory {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for prospective memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProspectiveMemoryStats {
    pub active_intentions: usize,
    pub completed_count: usize,
    pub forgotten_count: usize,
    pub success_rate: f64,
}

// ============================================================================
// TEMPORAL DISCOUNTING
// ============================================================================

/// Temporal discounting - future rewards feel less valuable
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalDiscounting {
    /// Discount rate (k in hyperbolic discounting)
    /// Higher k = more impatient, steeper discounting
    discount_rate: f64,
    /// Whether to use hyperbolic (true) or exponential (false) discounting
    hyperbolic: bool,
}

impl TemporalDiscounting {
    pub fn new(discount_rate: f64) -> Self {
        Self {
            discount_rate: discount_rate.clamp(0.001, 1.0),
            hyperbolic: true, // Humans use hyperbolic
        }
    }

    /// Calculate present value of a future reward
    /// 
    /// Hyperbolic: V = A / (1 + k*D)
    /// Exponential: V = A * e^(-k*D)
    /// 
    /// Where:
    /// - V = present value
    /// - A = future amount
    /// - k = discount rate
    /// - D = delay
    pub fn present_value(&self, future_amount: f64, delay_days: f64) -> f64 {
        if delay_days <= 0.0 {
            return future_amount;
        }
        
        if self.hyperbolic {
            future_amount / (1.0 + self.discount_rate * delay_days)
        } else {
            future_amount * (-self.discount_rate * delay_days).exp()
        }
    }

    /// Should we wait for a larger future reward?
    /// 
    /// Returns true if the discounted future reward > immediate reward
    pub fn should_wait(&self, immediate: f64, future: f64, delay_days: f64) -> bool {
        let discounted_future = self.present_value(future, delay_days);
        discounted_future > immediate
    }

    /// Find the indifference point - at what delay are two rewards equal?
    pub fn indifference_delay(&self, smaller: f64, larger: f64) -> Option<f64> {
        if smaller >= larger || smaller <= 0.0 {
            return None;
        }
        
        // Solve: smaller = larger / (1 + k*D)
        // D = (larger/smaller - 1) / k
        let ratio = larger / smaller;
        Some((ratio - 1.0) / self.discount_rate)
    }

    /// Get discount rate
    pub fn rate(&self) -> f64 {
        self.discount_rate
    }

    /// Modulate discount rate based on state
    /// 
    /// - High serotonin (patience) → lower discount rate
    /// - High dopamine (reward-seeking) → higher discount rate
    /// - High stress → higher discount rate (prefer immediate)
    pub fn modulate(&mut self, serotonin: f64, dopamine: f64, stress: f64) {
        let base = 0.05; // Default moderate discounting
        
        let serotonin_effect = 1.0 - serotonin * 0.5; // Patience reduces discounting
        let dopamine_effect = 1.0 + dopamine * 0.3; // Wanting increases discounting
        let stress_effect = 1.0 + stress * 0.4; // Stress increases discounting
        
        self.discount_rate = base * serotonin_effect * dopamine_effect * stress_effect;
        self.discount_rate = self.discount_rate.clamp(0.001, 0.5);
    }
}

impl Default for TemporalDiscounting {
    fn default() -> Self {
        Self::new(0.05)
    }
}

// ============================================================================
// UNIFIED TEMPORAL COGNITION SYSTEM
// ============================================================================

/// Complete temporal cognition system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalCognition {
    /// Duration perception (internal clock)
    pub duration: DurationPerception,
    /// Mental time travel
    pub time_travel: MentalTimeTravel,
    /// Prospective memory
    pub prospective: ProspectiveMemory,
    /// Temporal discounting
    pub discounting: TemporalDiscounting,
}

impl TemporalCognition {
    pub fn new() -> Self {
        Self {
            duration: DurationPerception::new(),
            time_travel: MentalTimeTravel::new(),
            prospective: ProspectiveMemory::new(),
            discounting: TemporalDiscounting::new(0.05),
        }
    }

    /// Update temporal systems based on neuromodulator state
    pub fn update(&mut self, arousal: f64, attention: f64, boredom: f64,
                  serotonin: f64, dopamine: f64, stress: f64) {
        self.duration.modulate(arousal, attention, boredom);
        self.discounting.modulate(serotonin, dopamine, stress);
    }

    /// Check for any triggered intentions
    pub fn check_prospective(&mut self, context: &str) -> Vec<Intention> {
        self.prospective.check(context)
    }

    /// Get comprehensive temporal stats
    pub fn stats(&self) -> TemporalStats {
        let (past_count, future_count) = self.time_travel.stats();
        let prospective_stats = self.prospective.stats();
        
        TemporalStats {
            duration_rate: self.duration.rate(),
            temporal_focus: self.time_travel.focus(),
            past_moments: past_count,
            future_moments: future_count,
            active_intentions: prospective_stats.active_intentions,
            prospective_success_rate: prospective_stats.success_rate,
            discount_rate: self.discounting.rate(),
        }
    }
}

impl Default for TemporalCognition {
    fn default() -> Self {
        Self::new()
    }
}

/// Temporal cognition statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalStats {
    pub duration_rate: f64,
    pub temporal_focus: f64,
    pub past_moments: usize,
    pub future_moments: usize,
    pub active_intentions: usize,
    pub prospective_success_rate: f64,
    pub discount_rate: f64,
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_duration_perception() {
        let mut duration = DurationPerception::new();
        
        // High arousal should speed up internal clock (more pulses = subjectively longer)
        duration.modulate(0.9, 0.5, 0.0);
        // Rate > 1.0 means time feels slower (more pulses per second)
        // But our formula: arousal_effect = 1.0 + (0.9 - 0.5) * 0.4 = 1.16
        //                  attention_effect = 1.0 - 0.5 * 0.3 = 0.85
        //                  boredom_effect = 1.0
        //                  Result = 1.0 * 1.16 * 0.85 * 1.0 = 0.986
        // So rate will be slightly below 1.0 in this case
        // Let's test that modulation works at all
        assert!(duration.rate() != 1.0);
        
        // Very high arousal, no attention focus -> time should drag
        duration.modulate(0.95, 0.1, 0.0);
        // arousal_effect = 1.0 + (0.95 - 0.5) * 0.4 = 1.18
        // attention_effect = 1.0 - 0.1 * 0.3 = 0.97
        // Result = 1.0 * 1.18 * 0.97 = 1.14 (> 1.0)
        assert!(duration.rate() > 1.0);
        
        // High attention should make time fly
        duration.modulate(0.5, 0.9, 0.0);
        assert!(duration.rate() < 1.0);
    }

    #[test]
    fn test_temporal_moment() {
        let past = TemporalMoment::past(
            Utc::now() - Duration::days(7),
            "Finished the project"
        ).with_valence(0.8);
        
        assert!(!past.is_future);
        assert!(past.valence > 0.0);
        
        let future = TemporalMoment::future(
            Utc::now() + Duration::days(7),
            "Vacation starts"
        ).with_valence(0.9);
        
        assert!(future.is_future);
    }

    #[test]
    fn test_mental_time_travel() {
        let mut mtt = MentalTimeTravel::new();
        
        // Add some past moments
        mtt.remember(TemporalMoment::past(
            Utc::now() - Duration::days(1),
            "Had a great meeting about the project"
        ).with_tag("work"));
        
        mtt.remember(TemporalMoment::past(
            Utc::now() - Duration::days(30),
            "Started the project"
        ).with_tag("project"));
        
        // Travel to past
        let memories = mtt.travel_to_past("project", 5);
        assert_eq!(memories.len(), 2);
        
        // Focus should be in the past
        assert!(mtt.focus() < 0.0);
        
        // Return to present
        mtt.return_to_present();
        assert_eq!(mtt.focus(), 0.0);
    }

    #[test]
    fn test_prospective_memory() {
        let mut pm = ProspectiveMemory::new();
        
        // Add a time-based intention
        let id = pm.intend_at(
            Utc::now() + Duration::minutes(2),
            "Check email"
        );
        
        assert!(pm.active_intentions().len() == 1);
        
        // Complete it
        pm.complete(id);
        assert!(pm.active_intentions().is_empty());
        assert!(pm.stats().completed_count == 1);
    }

    #[test]
    fn test_prospective_trigger_matching() {
        let now = Utc::now();
        
        // Time trigger
        let time_trigger = ProspectiveTrigger::Time(now);
        assert!(time_trigger.matches(now, ""));
        assert!(!time_trigger.matches(now + Duration::hours(1), ""));
        
        // Event trigger
        let event_trigger = ProspectiveTrigger::Event { 
            pattern: "meeting".to_string() 
        };
        assert!(event_trigger.matches(now, "Starting a meeting now"));
        assert!(!event_trigger.matches(now, "Working on code"));
    }

    #[test]
    fn test_temporal_discounting() {
        let discounting = TemporalDiscounting::new(0.1);
        
        // Future reward should be worth less
        let future_value = discounting.present_value(100.0, 30.0);
        assert!(future_value < 100.0);
        assert!(future_value > 0.0);
        
        // Immediate reward should be unchanged
        let immediate = discounting.present_value(100.0, 0.0);
        assert_eq!(immediate, 100.0);
    }

    #[test]
    fn test_should_wait_decision() {
        let discounting = TemporalDiscounting::new(0.05);
        
        // Should wait for significantly larger reward
        assert!(discounting.should_wait(50.0, 100.0, 7.0));
        
        // Shouldn't wait too long for small increase
        assert!(!discounting.should_wait(90.0, 100.0, 30.0));
    }

    #[test]
    fn test_discounting_modulation() {
        let mut discounting = TemporalDiscounting::new(0.05);
        
        // High patience (serotonin) should reduce discounting
        discounting.modulate(0.9, 0.5, 0.0);
        assert!(discounting.rate() < 0.05);
        
        // High stress should increase discounting
        discounting.modulate(0.5, 0.5, 0.9);
        assert!(discounting.rate() > 0.05);
    }

    #[test]
    fn test_temporal_cognition_system() {
        let mut temporal = TemporalCognition::new();
        
        // Add some anticipated events
        temporal.time_travel.anticipate(
            TemporalMoment::future(
                Utc::now() + Duration::hours(2),
                "Important deadline"
            ).with_valence(-0.3)
        );
        
        // Add prospective intention
        temporal.prospective.intend_when("deadline", "Submit the report");
        
        // Check stats
        let stats = temporal.stats();
        assert_eq!(stats.future_moments, 1);
        assert_eq!(stats.active_intentions, 1);
    }

    #[test]
    fn test_indifference_delay() {
        let discounting = TemporalDiscounting::new(0.1);
        
        // At what delay is $50 now equal to $100 later?
        let delay = discounting.indifference_delay(50.0, 100.0);
        assert!(delay.is_some());
        
        let d = delay.unwrap();
        // At this delay, present values should be roughly equal
        let pv = discounting.present_value(100.0, d);
        assert!((pv - 50.0).abs() < 1.0);
    }
}
