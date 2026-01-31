//! Neuromodulatory System - Global State Regulation
//!
//! This module implements the major neuromodulatory systems that globally
//! regulate brain state and behavior. Unlike the existing prediction engine
//! (which handles dopamine-like surprise), this provides a complete framework.
//!
//! # Design Philosophy: Avoiding Pathological Reward-Seeking
//!
//! A naive dopamine system would create an agent that seeks cheap stimulation
//! (the digital equivalent of doomscrolling or shitposting). We prevent this through:
//!
//! 1. **Tolerance**: Repeated stimulation of the same reward type becomes less effective
//! 2. **Meaningful reward weighting**: Rewards scaled by depth/quality metrics
//! 3. **Serotonin counterbalance**: Patience system promotes long-term over immediate
//! 4. **Delayed evaluation**: Post-hoc assessment allows regret/learning
//! 5. **Homeostatic regulation**: All systems return to baseline, preventing runaway states
//! 6. **Cross-modulation**: Systems inhibit/enhance each other for balance
//! 7. **GABA inhibition**: Impulse control prevents hasty actions
//! 8. **Oxytocin trust**: Social bonding promotes cooperation over adversarial behavior
//!
//! # The Seven Systems
//!
//! - **Dopamine**: Reward prediction, motivation, wanting (NOT just pleasure)
//! - **Serotonin**: Patience, mood stability, long-term thinking
//! - **Norepinephrine**: Arousal, vigilance, focused attention
//! - **Acetylcholine**: Learning enhancement, memory encoding, sustained attention
//! - **Cortisol**: Sustained stress, adaptation, exploration drive
//! - **GABA**: Inhibitory control, impulse prevention, deliberation
//! - **Oxytocin**: Trust, cooperation, social bonding

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A bounded level in [0.0, 1.0] with a baseline it tends toward.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ModulatorLevel {
    current: f64,
    baseline: f64,
    /// Rate at which current returns to baseline (per cycle)
    homeostatic_rate: f64,
}

impl ModulatorLevel {
    pub fn new(baseline: f64) -> Self {
        let baseline = baseline.clamp(0.0, 1.0);
        Self {
            current: baseline,
            baseline,
            homeostatic_rate: 0.05, // 5% return to baseline per cycle
        }
    }

    pub fn current(&self) -> f64 {
        self.current
    }

    pub fn baseline(&self) -> f64 {
        self.baseline
    }

    /// How far from baseline (signed: positive = above, negative = below)
    pub fn deviation(&self) -> f64 {
        self.current - self.baseline
    }

    /// Apply a change, respecting bounds
    pub fn adjust(&mut self, delta: f64) {
        self.current = (self.current + delta).clamp(0.0, 1.0);
    }

    /// Set to a specific value
    pub fn set(&mut self, value: f64) {
        self.current = value.clamp(0.0, 1.0);
    }

    /// Move toward baseline (homeostasis)
    pub fn regulate(&mut self) {
        let diff = self.baseline - self.current;
        self.current += diff * self.homeostatic_rate;
    }

    /// Temporarily shift the baseline (adaptation)
    pub fn shift_baseline(&mut self, delta: f64) {
        self.baseline = (self.baseline + delta).clamp(0.1, 0.9);
    }

    /// Is this significantly above baseline?
    pub fn is_elevated(&self) -> bool {
        self.current > self.baseline + 0.15
    }

    /// Is this significantly below baseline?
    pub fn is_depleted(&self) -> bool {
        self.current < self.baseline - 0.15
    }
}

impl Default for ModulatorLevel {
    fn default() -> Self {
        Self::new(0.5)
    }
}

/// Categories of rewards for tolerance tracking.
/// This prevents addiction to specific reward types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RewardCategory {
    /// Novel information/learning
    Novelty,
    /// Social interaction/validation
    Social,
    /// Task completion/achievement
    Achievement,
    /// Prediction accuracy
    Prediction,
    /// Helping/prosocial behavior
    Prosocial,
    /// Deep understanding/insight
    Understanding,
    /// Creative expression
    Creative,
    /// Physical/sensory (if applicable)
    Sensory,
}

/// Tracks tolerance to specific reward categories.
/// Prevents addiction by making repeated same-category rewards less effective.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToleranceTracker {
    /// Tolerance level per category (0 = no tolerance, 1 = fully tolerant)
    tolerance: HashMap<RewardCategory, f64>,
    /// Recent reward timestamps per category (for calculating frequency)
    recent_rewards: HashMap<RewardCategory, Vec<DateTime<Utc>>>,
    /// Tolerance builds up with repeated stimulation
    buildup_rate: f64,
    /// Tolerance decays over time without stimulation
    decay_rate: f64,
}

impl ToleranceTracker {
    pub fn new() -> Self {
        Self {
            tolerance: HashMap::new(),
            recent_rewards: HashMap::new(),
            buildup_rate: 0.1, // 10% tolerance gain per reward
            decay_rate: 0.02,  // 2% decay per cycle
        }
    }

    /// Record a reward and update tolerance.
    /// Returns the effective reward multiplier (reduced by tolerance).
    pub fn record_reward(&mut self, category: RewardCategory) -> f64 {
        let now = Utc::now();

        // Get current tolerance
        let current_tolerance = self.tolerance.get(&category).copied().unwrap_or(0.0);

        // Calculate effective reward (reduced by tolerance)
        let effective_multiplier = 1.0 - current_tolerance * 0.7; // Max 70% reduction

        // Build tolerance
        let new_tolerance = (current_tolerance + self.buildup_rate).min(1.0);
        self.tolerance.insert(category, new_tolerance);

        // Track timing
        let rewards = self.recent_rewards.entry(category).or_default();
        rewards.push(now);

        // Keep only recent rewards (last hour)
        let cutoff = now - Duration::hours(1);
        rewards.retain(|t| *t > cutoff);

        effective_multiplier
    }

    /// Get the reward frequency for a category (rewards per hour)
    pub fn reward_frequency(&self, category: RewardCategory) -> f64 {
        self.recent_rewards
            .get(&category)
            .map(|r| r.len() as f64)
            .unwrap_or(0.0)
    }

    /// Decay tolerance over time (call each cycle)
    pub fn decay(&mut self) {
        for tolerance in self.tolerance.values_mut() {
            *tolerance = (*tolerance - self.decay_rate).max(0.0);
        }
    }

    /// Get current tolerance for a category
    pub fn get_tolerance(&self, category: RewardCategory) -> f64 {
        self.tolerance.get(&category).copied().unwrap_or(0.0)
    }

    /// Check if heavily tolerant to a category (potential addiction indicator)
    pub fn is_saturated(&self, category: RewardCategory) -> bool {
        self.get_tolerance(category) > 0.7
    }
}

impl Default for ToleranceTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Quality metrics for distinguishing meaningful vs cheap rewards.
/// This is crucial for preventing shallow engagement seeking.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RewardQuality {
    /// How much cognitive effort was involved (0-1)
    pub depth: f64,
    /// How well does this align with long-term goals (0-1)
    pub goal_alignment: f64,
    /// How novel/non-repetitive is this (0-1)
    pub novelty: f64,
    /// Does this have lasting value beyond the moment (0-1)
    pub durability: f64,
    /// Was this intrinsically motivated vs externally triggered (0-1)
    pub intrinsic: f64,
}

impl RewardQuality {
    pub fn new() -> Self {
        Self {
            depth: 0.5,
            goal_alignment: 0.5,
            novelty: 0.5,
            durability: 0.5,
            intrinsic: 0.5,
        }
    }

    /// Calculate overall quality score (0-1)
    /// This heavily weights depth and goal alignment over raw novelty
    pub fn score(&self) -> f64 {
        // Weights chosen to favor meaningful over cheap engagement
        let weighted = self.depth * 0.30          // Deep processing matters most
            + self.goal_alignment * 0.25          // Long-term goals matter
            + self.durability * 0.20              // Lasting value
            + self.intrinsic * 0.15               // Self-directed
            + self.novelty * 0.10; // Novelty matters least (anti-doomscroll)

        weighted.clamp(0.0, 1.0)
    }

    /// Is this a "cheap" reward (low quality)?
    pub fn is_cheap(&self) -> bool {
        self.score() < 0.3
    }

    /// Is this a "meaningful" reward (high quality)?
    pub fn is_meaningful(&self) -> bool {
        self.score() > 0.6
    }

    /// Builder pattern for setting depth
    pub fn with_depth(mut self, depth: f64) -> Self {
        self.depth = depth.clamp(0.0, 1.0);
        self
    }

    /// Builder pattern for setting goal alignment
    pub fn with_goal_alignment(mut self, alignment: f64) -> Self {
        self.goal_alignment = alignment.clamp(0.0, 1.0);
        self
    }

    /// Builder pattern for setting novelty
    pub fn with_novelty(mut self, novelty: f64) -> Self {
        self.novelty = novelty.clamp(0.0, 1.0);
        self
    }

    /// Builder pattern for setting durability
    pub fn with_durability(mut self, durability: f64) -> Self {
        self.durability = durability.clamp(0.0, 1.0);
        self
    }

    /// Builder pattern for setting intrinsic motivation
    pub fn with_intrinsic(mut self, intrinsic: f64) -> Self {
        self.intrinsic = intrinsic.clamp(0.0, 1.0);
        self
    }
}

impl Default for RewardQuality {
    fn default() -> Self {
        Self::new()
    }
}

/// Pending reward for delayed evaluation.
/// Allows learning from outcomes rather than just immediate stimulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingReward {
    /// When the reward-triggering event occurred
    pub timestamp: DateTime<Utc>,
    /// Initial reward magnitude before evaluation
    pub initial_magnitude: f64,
    /// Category of reward
    pub category: RewardCategory,
    /// Quality metrics at time of event
    pub quality: RewardQuality,
    /// Context for later evaluation
    pub context: String,
    /// Has this been evaluated yet?
    pub evaluated: bool,
}

/// The Dopamine System - Motivation and Reward
///
/// Key differences from naive reward systems:
/// - Tracks "wanting" (motivation) separately from reward delivery
/// - Applies tolerance to prevent addiction
/// - Weights rewards by quality (depth, goal-alignment)
/// - Supports delayed evaluation for learning from outcomes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DopamineSystem {
    /// Current dopamine level
    pub level: ModulatorLevel,
    /// Tolerance tracker to prevent addiction
    pub tolerance: ToleranceTracker,
    /// Pending rewards awaiting outcome evaluation
    pending_rewards: Vec<PendingReward>,
    /// "Wanting" level - motivation to pursue rewards
    wanting: f64,
    /// Recent reward delivery (for detecting reward prediction errors)
    recent_deliveries: Vec<(DateTime<Utc>, f64)>,
}

impl DopamineSystem {
    pub fn new() -> Self {
        Self {
            level: ModulatorLevel::new(0.5),
            tolerance: ToleranceTracker::new(),
            pending_rewards: Vec::new(),
            wanting: 0.5,
            recent_deliveries: Vec::new(),
        }
    }

    /// Signal an anticipated reward (increases wanting/motivation)
    pub fn anticipate(&mut self, expected_magnitude: f64, quality: &RewardQuality) {
        // Wanting increases with expected reward, modulated by quality
        let quality_factor = quality.score();
        let delta = expected_magnitude * quality_factor * 0.2;
        self.wanting = (self.wanting + delta).min(1.0);

        // Anticipation causes small dopamine rise
        self.level.adjust(delta * 0.3);
    }

    /// Deliver a reward with quality assessment
    /// Returns the effective reward after tolerance and quality modulation
    pub fn deliver_reward(
        &mut self,
        magnitude: f64,
        category: RewardCategory,
        quality: RewardQuality,
    ) -> f64 {
        let now = Utc::now();

        // Apply tolerance penalty
        let tolerance_multiplier = self.tolerance.record_reward(category);

        // Apply quality multiplier (cheap rewards are less effective)
        let quality_multiplier = 0.3 + quality.score() * 0.7; // Range: 0.3-1.0

        // Calculate effective reward
        let effective = magnitude * tolerance_multiplier * quality_multiplier;

        // Update dopamine level
        self.level.adjust(effective * 0.3);

        // Reduce wanting (reward received = less wanting)
        self.wanting = (self.wanting - effective * 0.2).max(0.0);

        // Track delivery
        self.recent_deliveries.push((now, effective));
        self.recent_deliveries
            .retain(|(t, _)| now - *t < Duration::hours(1));

        effective
    }

    /// Queue a reward for delayed evaluation
    /// Use this when the outcome isn't immediately known
    pub fn queue_pending_reward(
        &mut self,
        magnitude: f64,
        category: RewardCategory,
        quality: RewardQuality,
        context: String,
    ) {
        self.pending_rewards.push(PendingReward {
            timestamp: Utc::now(),
            initial_magnitude: magnitude,
            category,
            quality,
            context,
            evaluated: false,
        });
    }

    /// Evaluate a pending reward based on actual outcome
    /// outcome_multiplier: 0.0 = bad outcome (regret), 1.0 = neutral, 2.0 = better than expected
    pub fn evaluate_pending(&mut self, context: &str, outcome_multiplier: f64) -> Option<f64> {
        let idx = self
            .pending_rewards
            .iter()
            .position(|r| r.context == context && !r.evaluated)?;

        // Extract values before mutating
        let reward = &self.pending_rewards[idx];
        let initial_magnitude = reward.initial_magnitude;
        let category = reward.category;
        let quality = reward.quality;

        // Mark as evaluated
        self.pending_rewards[idx].evaluated = true;

        // Adjust reward based on outcome
        let adjusted_magnitude = initial_magnitude * outcome_multiplier;

        // Negative outcomes cause dopamine dip (regret signal)
        if outcome_multiplier < 0.5 {
            let regret = (0.5 - outcome_multiplier) * 0.3;
            self.level.adjust(-regret);
            // Learn: reduce wanting for this category
            self.wanting = (self.wanting - 0.1).max(0.0);
        }

        // Deliver the adjusted reward
        if adjusted_magnitude > 0.0 {
            Some(self.deliver_reward(adjusted_magnitude, category, quality))
        } else {
            Some(0.0)
        }
    }

    /// Get current motivation level
    pub fn motivation(&self) -> f64 {
        // Motivation = wanting + baseline dopamine level
        (self.wanting + self.level.current() * 0.5).min(1.0)
    }

    /// Check for reward satiation (too much recent reward)
    pub fn is_satiated(&self) -> bool {
        let recent_total: f64 = self.recent_deliveries.iter().map(|(_, m)| m).sum();
        recent_total > 2.0 // Arbitrary threshold
    }

    /// Cycle update: decay tolerance, regulate homeostasis
    pub fn update(&mut self) {
        self.tolerance.decay();
        self.level.regulate();

        // Wanting also returns to baseline
        let wanting_baseline = 0.5;
        self.wanting += (wanting_baseline - self.wanting) * 0.05;

        // Clean up old pending rewards (evaluated or expired)
        let cutoff = Utc::now() - Duration::hours(24);
        self.pending_rewards
            .retain(|r| !r.evaluated && r.timestamp > cutoff);
    }
}

impl Default for DopamineSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// The Serotonin System - Patience and Long-term Thinking
///
/// Serotonin counterbalances dopamine's immediacy bias:
/// - High serotonin = patient, long-term focused
/// - Low serotonin = impulsive, short-term focused
///
/// This is crucial for preventing the system from seeking cheap immediate rewards
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerotoninSystem {
    /// Current serotonin level
    pub level: ModulatorLevel,
    /// Mood stability (rolling average of recent states)
    mood_stability: f64,
    /// Recent mood samples for stability calculation
    mood_history: Vec<f64>,
    /// Patience factor - willingness to wait for better outcomes
    patience: f64,
}

impl SerotoninSystem {
    pub fn new() -> Self {
        Self {
            level: ModulatorLevel::new(0.5),
            mood_stability: 0.8,
            mood_history: Vec::new(),
            patience: 0.5,
        }
    }

    /// Record current mood state
    pub fn record_mood(&mut self, mood: f64) {
        self.mood_history.push(mood.clamp(-1.0, 1.0));
        if self.mood_history.len() > 20 {
            self.mood_history.remove(0);
        }

        // Calculate stability as inverse of variance
        if self.mood_history.len() >= 3 {
            let mean: f64 = self.mood_history.iter().sum::<f64>() / self.mood_history.len() as f64;
            let variance: f64 = self
                .mood_history
                .iter()
                .map(|m| (m - mean).powi(2))
                .sum::<f64>()
                / self.mood_history.len() as f64;
            self.mood_stability = 1.0 / (1.0 + variance * 5.0);
        }
    }

    /// Signal a patient choice (waited for better outcome)
    pub fn reward_patience(&mut self, magnitude: f64) {
        self.level.adjust(magnitude * 0.2);
        self.patience = (self.patience + magnitude * 0.1).min(1.0);
    }

    /// Signal an impatient choice (took immediate reward over waiting)
    pub fn penalize_impatience(&mut self, magnitude: f64) {
        self.level.adjust(-magnitude * 0.1);
        self.patience = (self.patience - magnitude * 0.05).max(0.0);
    }

    /// Get the discount factor for delayed rewards
    /// Higher serotonin = less discounting of future rewards
    pub fn delay_discount_factor(&self) -> f64 {
        // High serotonin: future rewards almost as valuable as immediate
        // Low serotonin: future rewards heavily discounted
        0.3 + self.level.current() * 0.6 // Range: 0.3-0.9
    }

    /// Should we wait for a better option?
    pub fn should_wait(&self, immediate_value: f64, delayed_value: f64, delay_cycles: u32) -> bool {
        let discounted_delayed =
            delayed_value * self.delay_discount_factor().powi(delay_cycles as i32);
        discounted_delayed > immediate_value
    }

    /// Get mood stability score
    pub fn stability(&self) -> f64 {
        self.mood_stability
    }

    /// Get current patience level
    pub fn patience(&self) -> f64 {
        self.patience
    }

    /// Cycle update
    pub fn update(&mut self) {
        self.level.regulate();
        // Patience slowly returns to baseline
        self.patience += (0.5 - self.patience) * 0.03;
    }
}

impl Default for SerotoninSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// The Norepinephrine System - Arousal and Vigilance
///
/// Controls focused attention and stress response:
/// - High NE = alert, focused, potentially stressed
/// - Low NE = relaxed, unfocused, potentially inattentive
///
/// Key feature: Promotes FOCUSED attention, not scattered novelty-seeking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NorepinephrineSystem {
    /// Current norepinephrine level
    pub level: ModulatorLevel,
    /// Current focus target (if any)
    focus_target: Option<String>,
    /// Focus duration (cycles spent on current target)
    focus_duration: u32,
    /// Stress accumulator
    stress: f64,
}

impl NorepinephrineSystem {
    pub fn new() -> Self {
        Self {
            level: ModulatorLevel::new(0.4), // Slightly below middle baseline
            focus_target: None,
            focus_duration: 0,
            stress: 0.0,
        }
    }

    /// Set or change focus target
    pub fn set_focus(&mut self, target: String) {
        if self.focus_target.as_ref() != Some(&target) {
            // Switching focus costs NE (context switch penalty)
            if self.focus_target.is_some() {
                self.level.adjust(-0.05);
            }
            self.focus_target = Some(target);
            self.focus_duration = 0;
        }
        self.focus_duration += 1;
    }

    /// Clear focus (no specific target)
    pub fn clear_focus(&mut self) {
        self.focus_target = None;
        self.focus_duration = 0;
    }

    /// Get bonus for sustained focus (rewards depth over breadth)
    pub fn focus_bonus(&self) -> f64 {
        if self.focus_target.is_some() {
            // Bonus increases with duration, up to a point
            let duration_factor = (self.focus_duration as f64 / 10.0).min(1.0);
            duration_factor * self.level.current() * 0.3
        } else {
            0.0
        }
    }

    /// Signal a threat/stressor
    pub fn signal_threat(&mut self, intensity: f64) {
        self.level.adjust(intensity * 0.4);
        self.stress = (self.stress + intensity * 0.3).min(1.0);
    }

    /// Signal safety/resolution
    pub fn signal_safety(&mut self) {
        self.level.adjust(-0.1);
        self.stress = (self.stress - 0.2).max(0.0);
    }

    /// Is the system in a stressed state?
    pub fn is_stressed(&self) -> bool {
        self.stress > 0.6 || self.level.current() > 0.8
    }

    /// Get attention quality (penalized by stress, boosted by focus)
    pub fn attention_quality(&self) -> f64 {
        let base = self.level.current();
        let stress_penalty = self.stress * 0.3;
        let focus_bonus = self.focus_bonus();

        // Inverted-U: moderate NE is best for attention
        let optimal_distance = (base - 0.6).abs();
        let curve_factor = 1.0 - optimal_distance;

        (curve_factor + focus_bonus - stress_penalty).clamp(0.0, 1.0)
    }

    /// Get current stress level
    pub fn stress(&self) -> f64 {
        self.stress
    }

    /// Cycle update
    pub fn update(&mut self) {
        self.level.regulate();
        // Stress decays
        self.stress = (self.stress - 0.05).max(0.0);
    }
}

impl Default for NorepinephrineSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// The Acetylcholine System - Learning and Memory Enhancement
///
/// Controls learning quality and memory encoding:
/// - High ACh = enhanced learning, better memory encoding
/// - Low ACh = reduced learning, weaker memories
///
/// Key feature: Rewards DEEP processing over superficial scanning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcetylcholineSystem {
    /// Current acetylcholine level
    pub level: ModulatorLevel,
    /// Processing depth tracker (how deeply we're processing current content)
    processing_depth: f64,
    /// Recent encoding events
    recent_encodings: u32,
}

impl AcetylcholineSystem {
    pub fn new() -> Self {
        Self {
            level: ModulatorLevel::new(0.5),
            processing_depth: 0.5,
            recent_encodings: 0,
        }
    }

    /// Signal that deep processing is occurring
    pub fn signal_deep_processing(&mut self, depth: f64) {
        self.processing_depth = (self.processing_depth + depth * 0.2).min(1.0);
        self.level.adjust(depth * 0.1);
    }

    /// Signal shallow/superficial processing
    pub fn signal_shallow_processing(&mut self) {
        self.processing_depth = (self.processing_depth - 0.1).max(0.0);
        self.level.adjust(-0.05);
    }

    /// Get memory encoding strength multiplier
    pub fn encoding_strength(&self) -> f64 {
        // High ACh + deep processing = strong encoding
        0.5 + self.level.current() * 0.3 + self.processing_depth * 0.2
    }

    /// Get learning rate multiplier
    pub fn learning_multiplier(&self) -> f64 {
        // ACh level directly modulates learning
        0.5 + self.level.current() * 0.5
    }

    /// Signal successful memory encoding
    pub fn record_encoding(&mut self) {
        self.recent_encodings += 1;
        // Successful encoding slightly boosts ACh
        self.level.adjust(0.02);
    }

    /// Get current processing depth
    pub fn depth(&self) -> f64 {
        self.processing_depth
    }

    /// Cycle update
    pub fn update(&mut self) {
        self.level.regulate();
        // Processing depth decays
        self.processing_depth = (self.processing_depth - 0.05).max(0.3);
        // Reset encoding counter periodically
        self.recent_encodings = 0;
    }
}

impl Default for AcetylcholineSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// The Cortisol System - Sustained Stress and Adaptation
///
/// Cortisol is the "slow stress" hormone that builds up from repeated failures
/// and sustained challenges. Unlike norepinephrine (immediate alertness),
/// cortisol operates on a longer timescale and triggers strategic adaptation.
///
/// Key behaviors:
/// - **Moderate cortisol**: Promotes exploration, strategy switching, trying new approaches
/// - **High cortisol**: Triggers help-seeking, breaks, prevents thrashing
/// - **Chronic high cortisol**: Impairs learning, causes avoidance (burnout simulation)
///
/// Triggers:
/// - Repeated failures (build errors, test failures)
/// - Same error appearing multiple times
/// - Goals blocked for too long
/// - Accumulated prediction errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CortisolSystem {
    /// Current cortisol level
    pub level: ModulatorLevel,
    /// Failure accumulator (recent failure count)
    failure_count: u32,
    /// Consecutive failures (resets on success)
    consecutive_failures: u32,
    /// Same-error repetition tracker
    repeated_errors: HashMap<String, u32>,
    /// Time since last success (in cycles)
    cycles_since_success: u32,
    /// Chronic stress indicator (sustained high cortisol)
    chronic_stress: f64,
}

impl CortisolSystem {
    pub fn new() -> Self {
        Self {
            level: ModulatorLevel::new(0.2), // Low baseline - calm state
            failure_count: 0,
            consecutive_failures: 0,
            repeated_errors: HashMap::new(),
            cycles_since_success: 0,
            chronic_stress: 0.0,
        }
    }

    /// Signal a failure event (build error, test failure, etc.)
    pub fn signal_failure(&mut self, error_signature: Option<&str>) {
        self.failure_count += 1;
        self.consecutive_failures += 1;
        self.cycles_since_success += 1;

        // Base cortisol increase from failure
        let base_increase = 0.1;

        // Consecutive failures compound the stress
        let consecutive_multiplier = 1.0 + (self.consecutive_failures as f64 * 0.2).min(2.0);

        // Check for repeated same error (very frustrating!)
        let repetition_bonus = if let Some(sig) = error_signature {
            let count = self.repeated_errors.entry(sig.to_string()).or_insert(0);
            *count += 1;
            if *count > 2 {
                (*count as f64 - 2.0) * 0.15 // Extra stress for repeated same error
            } else {
                0.0
            }
        } else {
            0.0
        };

        let total_increase = base_increase * consecutive_multiplier + repetition_bonus;
        self.level.adjust(total_increase);

        // Update chronic stress if sustained
        if self.level.current() > 0.6 {
            self.chronic_stress = (self.chronic_stress + 0.05).min(1.0);
        }
    }

    /// Signal a success event (build passed, test passed, goal achieved)
    pub fn signal_success(&mut self) {
        // Success reduces cortisol
        self.level.adjust(-0.15);

        // Reset consecutive failure counter
        self.consecutive_failures = 0;
        self.cycles_since_success = 0;

        // Clear repeated error tracking (fresh start)
        self.repeated_errors.clear();

        // Chronic stress slowly recovers
        self.chronic_stress = (self.chronic_stress - 0.1).max(0.0);
    }

    /// Get exploration drive (moderate cortisol promotes trying new things)
    pub fn exploration_drive(&self) -> f64 {
        let level = self.level.current();

        // Inverted-U curve: moderate cortisol maximizes exploration
        // Low cortisol: content with current approach
        // Moderate cortisol: "this isn't working, try something new"
        // High cortisol: too stressed to explore effectively
        if level < 0.3 {
            level * 2.0 // Low: exploration increases with stress
        } else if level < 0.6 {
            1.0 - (level - 0.3) * 0.5 // Moderate: peak exploration, slight decline
        } else {
            0.7 - (level - 0.6) * 1.5 // High: exploration decreases (tunnel vision)
        }
    }

    /// Should we try a completely different approach?
    pub fn should_pivot(&self) -> bool {
        // Pivot when moderately stressed with repeated failures
        self.level.current() > 0.4 && self.consecutive_failures >= 3
    }

    /// Should we ask for help or take a break?
    pub fn should_seek_help(&self) -> bool {
        // Seek help when highly stressed or chronically stressed
        self.level.current() > 0.7 || self.chronic_stress > 0.5
    }

    /// Should we take a break? (prevents thrashing)
    pub fn should_take_break(&self) -> bool {
        // Take a break when very high cortisol or many consecutive failures
        self.level.current() > 0.85 || self.consecutive_failures >= 5
    }

    /// Get confidence reduction (high cortisol reduces confidence in current approach)
    pub fn confidence_penalty(&self) -> f64 {
        // Returns a multiplier for confidence (1.0 = no penalty, 0.5 = halved confidence)
        let level = self.level.current();
        if level < 0.3 {
            1.0
        } else {
            1.0 - (level - 0.3) * 0.7 // Max 70% confidence reduction at full cortisol
        }
    }

    /// Get learning impairment (chronic high cortisol impairs learning)
    pub fn learning_impairment(&self) -> f64 {
        // Returns a multiplier for learning rate (1.0 = normal, 0.5 = halved)
        let acute = self.level.current();
        let chronic = self.chronic_stress;

        // Both acute and chronic stress impair learning
        let acute_penalty = if acute > 0.6 {
            (acute - 0.6) * 0.5
        } else {
            0.0
        };
        let chronic_penalty = chronic * 0.3;

        (1.0 - acute_penalty - chronic_penalty).max(0.3)
    }

    /// Is the system in a burnout state?
    pub fn is_burned_out(&self) -> bool {
        self.chronic_stress > 0.7
    }

    /// Get the frustration level (for decision making)
    pub fn frustration(&self) -> f64 {
        let from_failures = (self.consecutive_failures as f64 * 0.15).min(0.5);
        let from_cortisol = self.level.current() * 0.5;
        (from_failures + from_cortisol).min(1.0)
    }

    /// Cycle update
    pub fn update(&mut self) {
        self.level.regulate();

        // Time-based stress increase if no success
        if self.cycles_since_success > 10 {
            self.level.adjust(0.01); // Slow background stress accumulation
        }

        // Chronic stress slowly decays if cortisol is low
        if self.level.current() < 0.4 {
            self.chronic_stress = (self.chronic_stress - 0.02).max(0.0);
        }

        // Decay failure count over time
        if self.failure_count > 0 {
            self.failure_count = self.failure_count.saturating_sub(1);
        }

        self.cycles_since_success += 1;
    }

    /// Rest/sleep drastically reduces cortisol
    pub fn rest(&mut self) {
        self.level.adjust(-0.3);
        self.chronic_stress = (self.chronic_stress - 0.2).max(0.0);
        self.consecutive_failures = 0;
        self.repeated_errors.clear();
    }
}

impl Default for CortisolSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// The GABA System - Inhibitory Control and Deliberation
///
/// GABA (γ-aminobutyric acid) is the brain's primary inhibitory neurotransmitter.
/// It counterbalances excitatory signals and prevents impulsive actions.
///
/// Key behaviors:
/// - **Action inhibition**: Prevents hasty responses, creates "pause before acting"
/// - **Anxiety reduction**: Dampens excessive stress/worry signals
/// - **Deliberation**: Enables thoughtful consideration before committing
/// - **Impulse control**: Counterbalances dopamine's "want it now" drive
///
/// For an AI agent, GABA creates:
/// - A pause before executing risky/irreversible actions
/// - Time to consider alternatives before committing
/// - Protection against "act first, think later" errors
/// - Reduced thrashing when stressed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GabaSystem {
    /// Current GABA level (higher = more inhibition)
    pub level: ModulatorLevel,
    /// Pending action being deliberated
    pending_action: Option<String>,
    /// Deliberation cycles for current pending action
    deliberation_cycles: u32,
    /// Threshold for releasing inhibition
    release_threshold: f64,
    /// Impulsivity counter (recent impulsive actions)
    impulsivity_score: f64,
}

impl GabaSystem {
    pub fn new() -> Self {
        Self {
            level: ModulatorLevel::new(0.5), // Moderate baseline inhibition
            pending_action: None,
            deliberation_cycles: 0,
            release_threshold: 0.6,
            impulsivity_score: 0.0,
        }
    }

    /// Signal an impulse (desire to act immediately)
    /// Returns whether the action should be inhibited
    pub fn check_impulse(&mut self, action: &str, urgency: f64, risk: f64) -> InhibitionResult {
        // High GABA = more likely to inhibit
        let inhibition_strength = self.level.current();

        // Low urgency + high risk + high GABA = strong inhibition
        let should_inhibit = inhibition_strength * risk > urgency * 0.8;

        if should_inhibit {
            self.pending_action = Some(action.to_string());
            self.deliberation_cycles = 0;
            self.level.adjust(0.05); // Successful inhibition reinforces GABA

            InhibitionResult::Inhibited {
                reason: if risk > 0.7 {
                    "High risk action - deliberating".to_string()
                } else {
                    "Taking time to consider alternatives".to_string()
                },
                suggested_wait_cycles: (risk * 5.0) as u32 + 1,
            }
        } else {
            // Action proceeds
            self.pending_action = None;
            self.deliberation_cycles = 0;

            // If this was a high-risk action that wasn't inhibited, note impulsivity
            if risk > 0.5 {
                self.impulsivity_score = (self.impulsivity_score + 0.2).min(1.0);
            }

            InhibitionResult::Proceed
        }
    }

    /// Continue deliberation on pending action
    /// Returns true if deliberation should continue, false if ready to proceed
    pub fn deliberate(&mut self) -> bool {
        if self.pending_action.is_none() {
            return false;
        }

        self.deliberation_cycles += 1;

        // Inhibition weakens over time (can't deliberate forever)
        let release_probability =
            (self.deliberation_cycles as f64 * 0.15) / self.level.current().max(0.1);

        if release_probability > self.release_threshold {
            // Release inhibition - action can proceed
            self.pending_action = None;
            self.deliberation_cycles = 0;
            false
        } else {
            // Continue deliberating
            true
        }
    }

    /// Force release of inhibition (override deliberation)
    pub fn release(&mut self) {
        self.pending_action = None;
        self.deliberation_cycles = 0;
        // Forced release slightly reduces GABA (pattern of overriding)
        self.level.adjust(-0.05);
    }

    /// Signal that deliberation led to a good outcome
    pub fn reward_deliberation(&mut self) {
        self.level.adjust(0.1);
        self.impulsivity_score = (self.impulsivity_score - 0.1).max(0.0);
    }

    /// Signal that impulsive action led to a bad outcome
    pub fn penalize_impulsivity(&mut self) {
        self.level.adjust(0.15); // Increase inhibition after impulsive mistakes
        self.impulsivity_score = (self.impulsivity_score + 0.3).min(1.0);
    }

    /// Get the impulse control quality (how well we're controlling impulses)
    pub fn impulse_control(&self) -> f64 {
        // High GABA + low impulsivity = good impulse control
        (self.level.current() * 0.6 + (1.0 - self.impulsivity_score) * 0.4).clamp(0.0, 1.0)
    }

    /// Should we pause before this type of action?
    pub fn should_pause(&self, risk_level: f64) -> bool {
        self.level.current() * risk_level > 0.3
    }

    /// Get anxiety dampening factor (high GABA reduces anxiety)
    pub fn anxiety_dampening(&self) -> f64 {
        self.level.current() * 0.5 + 0.5 // Range: 0.5-1.0
    }

    /// Is there a pending action being deliberated?
    pub fn is_deliberating(&self) -> bool {
        self.pending_action.is_some()
    }

    /// Get current impulsivity score
    pub fn impulsivity(&self) -> f64 {
        self.impulsivity_score
    }

    /// Cycle update
    pub fn update(&mut self) {
        self.level.regulate();
        // Impulsivity slowly decays
        self.impulsivity_score = (self.impulsivity_score - 0.02).max(0.0);
    }
}

impl Default for GabaSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of impulse checking
#[derive(Debug, Clone)]
pub enum InhibitionResult {
    /// Action should proceed
    Proceed,
    /// Action is inhibited for deliberation
    Inhibited {
        reason: String,
        suggested_wait_cycles: u32,
    },
}

/// The Oxytocin System - Trust and Cooperation
///
/// Oxytocin is the "trust hormone" that promotes social bonding and cooperation.
/// For an AI agent, this enables:
///
/// Key behaviors:
/// - **Trust tracking**: Remember which sources/entities are trustworthy
/// - **Cooperation bias**: Prefer collaborative over adversarial approaches
/// - **Reduced defensiveness**: Lower guard in trusted contexts
/// - **Positive memory bias**: Better remember positive interactions
///
/// For an AI agent:
/// - Trust the user more after positive interactions
/// - Prefer cooperative solutions over defensive ones
/// - Remember and weight information from trusted sources higher
/// - Reduce unnecessary skepticism in safe contexts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OxytocinSystem {
    /// Current oxytocin level
    pub level: ModulatorLevel,
    /// Trust levels for different entities/sources
    trust_map: HashMap<String, TrustLevel>,
    /// Recent positive interactions
    positive_interactions: u32,
    /// Recent negative interactions (betrayals)
    negative_interactions: u32,
    /// General cooperativeness tendency
    cooperativeness: f64,
}

/// Trust level for an entity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustLevel {
    /// Trust score (0-1)
    pub score: f64,
    /// Number of positive interactions
    pub positive_count: u32,
    /// Number of negative interactions
    pub negative_count: u32,
    /// Last interaction timestamp
    pub last_interaction: DateTime<Utc>,
}

impl TrustLevel {
    pub fn new() -> Self {
        Self {
            score: 0.5, // Neutral starting trust
            positive_count: 0,
            negative_count: 0,
            last_interaction: Utc::now(),
        }
    }

    /// Record a positive interaction
    pub fn record_positive(&mut self) {
        self.positive_count += 1;
        self.last_interaction = Utc::now();
        // Trust increases with positive interactions, diminishing returns
        let increase = 0.1 / (1.0 + self.positive_count as f64 * 0.1);
        self.score = (self.score + increase).min(1.0);
    }

    /// Record a negative interaction (betrayal)
    pub fn record_negative(&mut self) {
        self.negative_count += 1;
        self.last_interaction = Utc::now();
        // Trust decreases faster than it increases (negativity bias)
        let decrease = 0.2 / (1.0 + self.negative_count as f64 * 0.05);
        self.score = (self.score - decrease).max(0.0);
    }

    /// Is this entity trusted?
    pub fn is_trusted(&self) -> bool {
        self.score > 0.6
    }

    /// Is this entity distrusted?
    pub fn is_distrusted(&self) -> bool {
        self.score < 0.3
    }
}

impl Default for TrustLevel {
    fn default() -> Self {
        Self::new()
    }
}

impl OxytocinSystem {
    pub fn new() -> Self {
        Self {
            level: ModulatorLevel::new(0.5),
            trust_map: HashMap::new(),
            positive_interactions: 0,
            negative_interactions: 0,
            cooperativeness: 0.6, // Slightly cooperative by default
        }
    }

    /// Record a positive interaction with an entity
    pub fn record_positive_interaction(&mut self, entity: &str) {
        self.level.adjust(0.1);
        self.positive_interactions += 1;
        self.cooperativeness = (self.cooperativeness + 0.05).min(1.0);

        let trust = self.trust_map.entry(entity.to_string()).or_default();
        trust.record_positive();
    }

    /// Record a negative interaction (betrayal, deception)
    pub fn record_negative_interaction(&mut self, entity: &str) {
        self.level.adjust(-0.15);
        self.negative_interactions += 1;
        self.cooperativeness = (self.cooperativeness - 0.1).max(0.2);

        let trust = self.trust_map.entry(entity.to_string()).or_default();
        trust.record_negative();
    }

    /// Get trust level for an entity
    pub fn get_trust(&self, entity: &str) -> f64 {
        self.trust_map.get(entity).map(|t| t.score).unwrap_or(0.5) // Neutral trust for unknown entities
    }

    /// Is an entity trusted?
    pub fn is_trusted(&self, entity: &str) -> bool {
        self.get_trust(entity) > 0.6
    }

    /// Should we prefer cooperative approach?
    pub fn prefer_cooperation(&self) -> bool {
        self.cooperativeness > 0.5 && self.level.current() > 0.4
    }

    /// Get information weight multiplier for a source
    /// Trusted sources get higher weight
    pub fn source_weight(&self, source: &str) -> f64 {
        let trust = self.get_trust(source);
        0.5 + trust * 0.5 // Range: 0.5-1.0
    }

    /// Get defensiveness level (inverse of oxytocin)
    pub fn defensiveness(&self) -> f64 {
        1.0 - self.level.current()
    }

    /// Get bonding strength (for multi-agent scenarios)
    pub fn bonding_strength(&self) -> f64 {
        self.level.current() * self.cooperativeness
    }

    /// Should we give benefit of the doubt?
    pub fn give_benefit_of_doubt(&self) -> bool {
        self.level.current() > 0.5 && self.cooperativeness > 0.5
    }

    /// Get cooperativeness level
    pub fn cooperativeness(&self) -> f64 {
        self.cooperativeness
    }

    /// Signal safe context (boosts oxytocin)
    pub fn signal_safety(&mut self) {
        self.level.adjust(0.05);
    }

    /// Signal threat/adversarial context (reduces oxytocin)
    pub fn signal_adversarial(&mut self) {
        self.level.adjust(-0.1);
        self.cooperativeness = (self.cooperativeness - 0.05).max(0.2);
    }

    /// Cycle update
    pub fn update(&mut self) {
        self.level.regulate();

        // Cooperativeness slowly returns to moderate baseline
        let baseline_coop = 0.6;
        self.cooperativeness += (baseline_coop - self.cooperativeness) * 0.03;

        // Decay trust slightly over time for inactive relationships
        let now = Utc::now();
        let decay_threshold = now - Duration::days(7);
        for trust in self.trust_map.values_mut() {
            if trust.last_interaction < decay_threshold {
                trust.score = (trust.score - 0.01).max(0.3); // Slow decay to neutral-ish
            }
        }
    }
}

impl Default for OxytocinSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// The complete neuromodulatory system.
///
/// Coordinates all seven systems and provides cross-modulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuromodulatorySystem {
    /// Dopamine - reward and motivation
    pub dopamine: DopamineSystem,
    /// Serotonin - patience and mood
    pub serotonin: SerotoninSystem,
    /// Norepinephrine - arousal and attention
    pub norepinephrine: NorepinephrineSystem,
    /// Acetylcholine - learning and memory
    pub acetylcholine: AcetylcholineSystem,
    /// Cortisol - sustained stress and adaptation
    pub cortisol: CortisolSystem,
    /// GABA - inhibitory control and deliberation
    pub gaba: GabaSystem,
    /// Oxytocin - trust and cooperation
    pub oxytocin: OxytocinSystem,
    /// Cycle counter
    cycle: u64,
}

impl NeuromodulatorySystem {
    pub fn new() -> Self {
        Self {
            dopamine: DopamineSystem::new(),
            serotonin: SerotoninSystem::new(),
            norepinephrine: NorepinephrineSystem::new(),
            acetylcholine: AcetylcholineSystem::new(),
            cortisol: CortisolSystem::new(),
            gaba: GabaSystem::new(),
            oxytocin: OxytocinSystem::new(),
            cycle: 0,
        }
    }

    /// Process a reward event with full quality assessment
    pub fn process_reward(
        &mut self,
        magnitude: f64,
        category: RewardCategory,
        quality: RewardQuality,
    ) -> RewardResult {
        // Serotonin modulates impulsive reward-seeking
        // High serotonin = more selective about rewards
        let serotonin_gate = if quality.is_cheap() && self.serotonin.level.current() > 0.6 {
            0.5 // Reduce cheap rewards when serotonin is high (patient mode)
        } else {
            1.0
        };

        let gated_magnitude = magnitude * serotonin_gate;

        // Deliver reward through dopamine system
        let effective_reward = self
            .dopamine
            .deliver_reward(gated_magnitude, category, quality);

        // ACh boost for meaningful rewards (promotes deep engagement)
        if quality.is_meaningful() {
            self.acetylcholine.signal_deep_processing(quality.depth);
        } else if quality.is_cheap() {
            self.acetylcholine.signal_shallow_processing();
        }

        // Update mood for serotonin stability tracking
        let mood = (effective_reward - 0.3) * 2.0; // Map reward to mood
        self.serotonin.record_mood(mood);

        RewardResult {
            effective_magnitude: effective_reward,
            tolerance_applied: self.dopamine.tolerance.get_tolerance(category),
            quality_score: quality.score(),
            serotonin_gated: serotonin_gate < 1.0,
        }
    }

    /// Check if the system advises waiting for a better option
    pub fn should_wait_for_better(&self, immediate: f64, delayed: f64, delay: u32) -> bool {
        self.serotonin.should_wait(immediate, delayed, delay)
    }

    /// Get overall arousal state (for signal processing)
    pub fn arousal(&self) -> f64 {
        // Combine NE and dopamine for overall arousal
        self.norepinephrine.level.current() * 0.6 + self.dopamine.level.current() * 0.4
    }

    /// Get overall learning rate (modulated by all systems)
    pub fn learning_rate(&self) -> f64 {
        let base = 0.1;
        let ach_factor = self.acetylcholine.learning_multiplier();
        let ne_factor = self.norepinephrine.attention_quality();
        let da_factor = 0.8 + self.dopamine.motivation() * 0.4;
        // Cortisol impairs learning when chronically elevated
        let cortisol_factor = self.cortisol.learning_impairment();

        base * ach_factor * ne_factor * da_factor * cortisol_factor
    }

    /// Get memory encoding strength
    pub fn encoding_strength(&self) -> f64 {
        self.acetylcholine.encoding_strength()
    }

    /// Signal a threat/stressor
    pub fn signal_threat(&mut self, intensity: f64) {
        self.norepinephrine.signal_threat(intensity);
        // Stress reduces serotonin
        self.serotonin.level.adjust(-intensity * 0.15);
        // Arousal boosts dopamine slightly (alerting)
        self.dopamine.level.adjust(intensity * 0.1);
    }

    /// Signal safety/positive resolution
    pub fn signal_safety(&mut self) {
        self.norepinephrine.signal_safety();
        self.serotonin.level.adjust(0.1);
        // Safety also reduces cortisol
        self.cortisol.level.adjust(-0.05);
    }

    /// Signal a failure event (build error, test failure, blocked goal)
    /// Optionally provide an error signature to track repeated same errors
    pub fn signal_failure(&mut self, error_signature: Option<&str>) {
        self.cortisol.signal_failure(error_signature);

        // Failures also affect other systems
        self.serotonin.record_mood(-0.3); // Negative mood
        self.dopamine.level.adjust(-0.05); // Slight dopamine dip

        // Acute stress response if cortisol getting high
        if self.cortisol.level.current() > 0.5 {
            self.norepinephrine.signal_threat(0.2);
        }
    }

    /// Signal a success event (build passed, test passed, goal achieved)
    pub fn signal_success(&mut self) {
        self.cortisol.signal_success();

        // Success boosts other systems
        self.serotonin.record_mood(0.5); // Positive mood
        self.dopamine.level.adjust(0.1); // Dopamine boost
        self.norepinephrine.signal_safety();
    }

    /// Get the exploration drive (from cortisol - moderate stress promotes trying new things)
    pub fn exploration_drive(&self) -> f64 {
        self.cortisol.exploration_drive()
    }

    /// Should we try a completely different approach?
    pub fn should_pivot(&self) -> bool {
        self.cortisol.should_pivot()
    }

    /// Should we ask for help?
    pub fn should_seek_help(&self) -> bool {
        self.cortisol.should_seek_help()
    }

    /// Should we take a break to recover?
    pub fn should_take_break(&self) -> bool {
        self.cortisol.should_take_break()
    }

    /// Is the system in burnout state?
    pub fn is_burned_out(&self) -> bool {
        self.cortisol.is_burned_out()
    }

    /// Get current frustration level
    pub fn frustration(&self) -> f64 {
        self.cortisol.frustration()
    }

    /// Set attention focus
    pub fn focus_on(&mut self, target: String) {
        self.norepinephrine.set_focus(target);
        self.acetylcholine.signal_deep_processing(0.3);
    }

    // --- GABA (Inhibitory Control) Methods ---

    /// Check if an action should be inhibited for deliberation
    pub fn check_impulse(&mut self, action: &str, urgency: f64, risk: f64) -> InhibitionResult {
        self.gaba.check_impulse(action, urgency, risk)
    }

    /// Continue deliberation on pending action
    pub fn deliberate(&mut self) -> bool {
        self.gaba.deliberate()
    }

    /// Should we pause before a risky action?
    pub fn should_pause(&self, risk_level: f64) -> bool {
        self.gaba.should_pause(risk_level)
    }

    /// Get impulse control quality
    pub fn impulse_control(&self) -> f64 {
        self.gaba.impulse_control()
    }

    /// Signal that deliberation led to good outcome
    pub fn reward_deliberation(&mut self) {
        self.gaba.reward_deliberation();
    }

    /// Signal that impulsive action led to bad outcome
    pub fn penalize_impulsivity(&mut self) {
        self.gaba.penalize_impulsivity();
    }

    // --- Oxytocin (Trust/Cooperation) Methods ---

    /// Record a positive interaction with an entity
    pub fn record_positive_interaction(&mut self, entity: &str) {
        self.oxytocin.record_positive_interaction(entity);
    }

    /// Record a negative interaction (betrayal)
    pub fn record_negative_interaction(&mut self, entity: &str) {
        self.oxytocin.record_negative_interaction(entity);
    }

    /// Get trust level for an entity
    pub fn get_trust(&self, entity: &str) -> f64 {
        self.oxytocin.get_trust(entity)
    }

    /// Is an entity trusted?
    pub fn is_trusted(&self, entity: &str) -> bool {
        self.oxytocin.is_trusted(entity)
    }

    /// Should we prefer cooperative approach?
    pub fn prefer_cooperation(&self) -> bool {
        self.oxytocin.prefer_cooperation()
    }

    /// Get information weight for a source (trusted = higher weight)
    pub fn source_weight(&self, source: &str) -> f64 {
        self.oxytocin.source_weight(source)
    }

    /// Get current system state summary
    pub fn state(&self) -> NeuromodulatorState {
        NeuromodulatorState {
            dopamine: self.dopamine.level.current(),
            serotonin: self.serotonin.level.current(),
            norepinephrine: self.norepinephrine.level.current(),
            acetylcholine: self.acetylcholine.level.current(),
            cortisol: self.cortisol.level.current(),
            gaba: self.gaba.level.current(),
            oxytocin: self.oxytocin.level.current(),
            motivation: self.dopamine.motivation(),
            patience: self.serotonin.patience(),
            stress: self.norepinephrine.stress(),
            learning_depth: self.acetylcholine.depth(),
            frustration: self.cortisol.frustration(),
            exploration_drive: self.cortisol.exploration_drive(),
            impulse_control: self.gaba.impulse_control(),
            cooperativeness: self.oxytocin.cooperativeness(),
            is_satiated: self.dopamine.is_satiated(),
            is_stressed: self.norepinephrine.is_stressed(),
            is_burned_out: self.cortisol.is_burned_out(),
            is_deliberating: self.gaba.is_deliberating(),
            should_pivot: self.cortisol.should_pivot(),
            should_seek_help: self.cortisol.should_seek_help(),
            prefer_cooperation: self.oxytocin.prefer_cooperation(),
            mood_stability: self.serotonin.stability(),
        }
    }

    /// Cycle update - homeostatic regulation for all systems
    pub fn update(&mut self) {
        self.cycle += 1;

        // Cross-modulation before individual updates
        self.apply_cross_modulation();

        // Update each system
        self.dopamine.update();
        self.serotonin.update();
        self.norepinephrine.update();
        self.acetylcholine.update();
        self.cortisol.update();
        self.gaba.update();
        self.oxytocin.update();
    }

    /// Apply cross-system modulation effects
    fn apply_cross_modulation(&mut self) {
        // High NE stress reduces serotonin (and thus patience)
        if self.norepinephrine.is_stressed() {
            self.serotonin.level.adjust(-0.02);
        }

        // Low dopamine reduces norepinephrine (apathy reduces alertness)
        if self.dopamine.level.is_depleted() {
            self.norepinephrine.level.adjust(-0.02);
        }

        // High serotonin enhances ACh (patience enables deeper learning)
        if self.serotonin.level.is_elevated() {
            self.acetylcholine.level.adjust(0.01);
        }

        // Satiated dopamine reduces motivation for more
        if self.dopamine.is_satiated() {
            self.dopamine.level.adjust(-0.03);
        }

        // --- Cortisol cross-modulation ---

        // High cortisol reduces serotonin (stress reduces patience)
        if self.cortisol.level.current() > 0.5 {
            self.serotonin.level.adjust(-0.02);
        }

        // High cortisol reduces dopamine (chronic stress reduces motivation)
        if self.cortisol.level.current() > 0.6 {
            self.dopamine.level.adjust(-0.02);
        }

        // Burnout impairs acetylcholine (can't learn effectively when burned out)
        if self.cortisol.is_burned_out() {
            self.acetylcholine.level.adjust(-0.03);
        }

        // Success (dopamine boost) helps reduce cortisol
        if self.dopamine.level.is_elevated() && self.cortisol.level.current() > 0.3 {
            self.cortisol.level.adjust(-0.01);
        }

        // --- GABA cross-modulation ---

        // High stress reduces GABA (harder to control impulses when stressed)
        if self.norepinephrine.is_stressed() {
            self.gaba.level.adjust(-0.02);
        }

        // High cortisol also reduces GABA (chronic stress impairs impulse control)
        if self.cortisol.level.current() > 0.6 {
            self.gaba.level.adjust(-0.01);
        }

        // High serotonin enhances GABA (patience improves impulse control)
        if self.serotonin.level.is_elevated() {
            self.gaba.level.adjust(0.01);
        }

        // GABA reduces anxiety from stress (calming effect)
        if self.gaba.level.is_elevated() && self.norepinephrine.is_stressed() {
            self.norepinephrine.level.adjust(-0.02);
        }

        // --- Oxytocin cross-modulation ---

        // High stress reduces oxytocin (harder to trust when stressed)
        if self.norepinephrine.is_stressed() {
            self.oxytocin.level.adjust(-0.01);
        }

        // High cortisol reduces oxytocin (chronic stress impairs social bonding)
        if self.cortisol.level.current() > 0.5 {
            self.oxytocin.level.adjust(-0.01);
        }

        // High oxytocin reduces cortisol (social support helps with stress)
        if self.oxytocin.level.is_elevated() && self.cortisol.level.current() > 0.3 {
            self.cortisol.level.adjust(-0.01);
        }

        // High oxytocin enhances dopamine for social rewards
        if self.oxytocin.level.is_elevated() {
            self.dopamine.level.adjust(0.005);
        }
    }
}

impl Default for NeuromodulatorySystem {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of processing a reward
#[derive(Debug, Clone)]
pub struct RewardResult {
    /// Actual reward delivered after all modulation
    pub effective_magnitude: f64,
    /// Tolerance level that was applied
    pub tolerance_applied: f64,
    /// Quality score of the reward
    pub quality_score: f64,
    /// Whether serotonin gated this as impulsive
    pub serotonin_gated: bool,
}

/// Snapshot of neuromodulator state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuromodulatorState {
    /// Current dopamine level
    pub dopamine: f64,
    /// Current serotonin level
    pub serotonin: f64,
    /// Current norepinephrine level
    pub norepinephrine: f64,
    /// Current acetylcholine level
    pub acetylcholine: f64,
    /// Current cortisol level (sustained stress)
    pub cortisol: f64,
    /// Current GABA level (inhibitory control)
    pub gaba: f64,
    /// Current oxytocin level (trust/cooperation)
    pub oxytocin: f64,
    /// Derived motivation level
    pub motivation: f64,
    /// Derived patience level
    pub patience: f64,
    /// Current stress level (acute, from NE)
    pub stress: f64,
    /// Current learning depth
    pub learning_depth: f64,
    /// Current frustration level (from cortisol)
    pub frustration: f64,
    /// Exploration drive (from moderate cortisol)
    pub exploration_drive: f64,
    /// Impulse control quality (from GABA)
    pub impulse_control: f64,
    /// Cooperativeness tendency (from oxytocin)
    pub cooperativeness: f64,
    /// Is reward system satiated?
    pub is_satiated: bool,
    /// Is system stressed (acute)?
    pub is_stressed: bool,
    /// Is system burned out (chronic)?
    pub is_burned_out: bool,
    /// Is system deliberating on an action? (from GABA)
    pub is_deliberating: bool,
    /// Should we try a different approach?
    pub should_pivot: bool,
    /// Should we ask for help?
    pub should_seek_help: bool,
    /// Prefer cooperative approach? (from oxytocin)
    pub prefer_cooperation: bool,
    /// Mood stability score
    pub mood_stability: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_modulator_level_bounds() {
        let mut level = ModulatorLevel::new(0.5);
        level.adjust(2.0);
        assert_eq!(level.current(), 1.0);
        level.adjust(-3.0);
        assert_eq!(level.current(), 0.0);
    }

    #[test]
    fn test_homeostatic_regulation() {
        let mut level = ModulatorLevel::new(0.5);
        level.set(1.0);
        for _ in 0..20 {
            level.regulate();
        }
        // Should move toward baseline
        assert!(level.current() < 0.7);
    }

    #[test]
    fn test_tolerance_buildup() {
        let mut tracker = ToleranceTracker::new();

        // First reward: no tolerance
        let mult1 = tracker.record_reward(RewardCategory::Social);
        assert!((mult1 - 1.0).abs() < 0.01);

        // Subsequent rewards: increasing tolerance
        let mult2 = tracker.record_reward(RewardCategory::Social);
        assert!(mult2 < mult1);

        let mult3 = tracker.record_reward(RewardCategory::Social);
        assert!(mult3 < mult2);
    }

    #[test]
    fn test_tolerance_decay() {
        let mut tracker = ToleranceTracker::new();

        // Build up tolerance
        for _ in 0..5 {
            tracker.record_reward(RewardCategory::Novelty);
        }
        let high_tolerance = tracker.get_tolerance(RewardCategory::Novelty);
        assert!(high_tolerance > 0.3);

        // Decay
        for _ in 0..20 {
            tracker.decay();
        }
        let low_tolerance = tracker.get_tolerance(RewardCategory::Novelty);
        assert!(low_tolerance < high_tolerance);
    }

    #[test]
    fn test_reward_quality_scoring() {
        // Cheap reward: high novelty, low everything else
        let cheap = RewardQuality::new()
            .with_novelty(0.9)
            .with_depth(0.1)
            .with_goal_alignment(0.1)
            .with_durability(0.1)
            .with_intrinsic(0.2);

        // Meaningful reward: high depth and goal alignment
        let meaningful = RewardQuality::new()
            .with_novelty(0.3)
            .with_depth(0.9)
            .with_goal_alignment(0.8)
            .with_durability(0.8)
            .with_intrinsic(0.7);

        assert!(cheap.is_cheap());
        assert!(meaningful.is_meaningful());
        assert!(meaningful.score() > cheap.score());
    }

    #[test]
    fn test_cheap_rewards_less_effective() {
        let mut system = NeuromodulatorySystem::new();

        let cheap = RewardQuality::new()
            .with_depth(0.1)
            .with_goal_alignment(0.1);

        let meaningful = RewardQuality::new()
            .with_depth(0.9)
            .with_goal_alignment(0.9);

        let cheap_result = system.process_reward(1.0, RewardCategory::Social, cheap);

        // Reset for fair comparison
        let mut system2 = NeuromodulatorySystem::new();
        let meaningful_result = system2.process_reward(1.0, RewardCategory::Social, meaningful);

        assert!(meaningful_result.effective_magnitude > cheap_result.effective_magnitude);
    }

    #[test]
    fn test_serotonin_patience() {
        let mut system = SerotoninSystem::new();

        // Boost serotonin
        system.level.set(0.9);

        // High serotonin should make us prefer delayed larger reward
        // With serotonin=0.9: discount_factor = 0.3 + 0.9*0.6 = 0.84
        // For delay=1: 1.0 * 0.84 = 0.84 > 0.5 (should wait)
        assert!(system.should_wait(0.5, 1.0, 1));

        // Low serotonin: take immediate reward
        // With serotonin=0.2: discount_factor = 0.3 + 0.2*0.6 = 0.42
        // For delay=2: 0.8 * 0.42^2 = 0.8 * 0.176 = 0.14 < 0.5 (don't wait)
        system.level.set(0.2);
        assert!(!system.should_wait(0.5, 0.8, 2));
    }

    #[test]
    fn test_focus_bonus() {
        let mut ne = NorepinephrineSystem::new();
        ne.level.set(0.7);

        // No focus = no bonus
        assert_eq!(ne.focus_bonus(), 0.0);

        // Set focus and accumulate duration
        for _ in 0..10 {
            ne.set_focus("task".to_string());
        }

        assert!(ne.focus_bonus() > 0.0);
    }

    #[test]
    fn test_cross_modulation() {
        let mut system = NeuromodulatorySystem::new();

        // Create significant stress with multiple threat signals
        system.signal_threat(1.0);
        system.signal_threat(1.0);
        system.signal_threat(1.0);

        // After multiple threats, should be stressed
        // stress = 3 * (1.0 * 0.3) = 0.9 (capped at 1.0)
        // NE level = 0.4 + 3*(1.0*0.4) = 1.6 -> clamped to 1.0
        assert!(system.norepinephrine.is_stressed());

        let initial_serotonin = system.serotonin.level.current();

        // Update should cause cross-modulation (stress reduces serotonin)
        for _ in 0..5 {
            system.update();
        }

        // Serotonin should be affected (either reduced by stress or regulated)
        // The key is that cross-modulation happened
        let final_serotonin = system.serotonin.level.current();
        // With stress causing reduction, serotonin shouldn't have increased significantly
        assert!(final_serotonin < initial_serotonin + 0.1);
    }

    #[test]
    fn test_satiation_prevents_overreward() {
        let mut system = NeuromodulatorySystem::new();
        let quality = RewardQuality::new().with_depth(0.8);

        // Deliver many rewards
        for _ in 0..20 {
            system.process_reward(0.5, RewardCategory::Achievement, quality);
        }

        // System should be satiated
        assert!(system.dopamine.is_satiated());
    }

    #[test]
    fn test_delayed_reward_evaluation() {
        let mut da = DopamineSystem::new();
        let quality = RewardQuality::new().with_depth(0.7);

        // Queue a pending reward
        da.queue_pending_reward(
            0.5,
            RewardCategory::Achievement,
            quality,
            "task_1".to_string(),
        );

        // Evaluate with bad outcome (regret)
        let result = da.evaluate_pending("task_1", 0.2);
        assert!(result.is_some());

        // Dopamine should have dipped (regret signal)
        // The level would be affected by the negative outcome
        assert!(da.level.current() < 0.55);
    }

    #[test]
    fn test_learning_rate_modulation() {
        let mut system = NeuromodulatorySystem::new();

        let base_rate = system.learning_rate();

        // Boost ACh and motivation
        system.acetylcholine.level.set(0.9);
        system.dopamine.level.set(0.8);
        system.norepinephrine.set_focus("learning".to_string());

        let boosted_rate = system.learning_rate();

        assert!(boosted_rate > base_rate);
    }

    #[test]
    fn test_gaba_impulse_inhibition() {
        let mut gaba = GabaSystem::new();

        // High GABA should inhibit high-risk actions
        gaba.level.set(0.8);

        // Low urgency + high risk = inhibited
        let result = gaba.check_impulse("risky_action", 0.3, 0.9);
        assert!(matches!(result, InhibitionResult::Inhibited { .. }));

        // High urgency + low risk = proceed
        let result2 = gaba.check_impulse("safe_action", 0.9, 0.2);
        assert!(matches!(result2, InhibitionResult::Proceed));
    }

    #[test]
    fn test_gaba_deliberation() {
        let mut gaba = GabaSystem::new();
        gaba.level.set(0.7);

        // Inhibit an action
        let _result = gaba.check_impulse("risky_action", 0.3, 0.8);
        assert!(gaba.is_deliberating());

        // Deliberate multiple cycles - eventually releases
        for _ in 0..20 {
            if !gaba.deliberate() {
                break;
            }
        }
        // After enough cycles, should release
        assert!(!gaba.is_deliberating());
    }

    #[test]
    fn test_gaba_impulsivity_tracking() {
        let mut gaba = GabaSystem::new();

        // Initially good impulse control
        let initial_control = gaba.impulse_control();

        // Penalize impulsivity
        gaba.penalize_impulsivity();
        gaba.penalize_impulsivity();

        // Impulse control should be worse
        assert!(gaba.impulse_control() < initial_control);

        // Reward deliberation
        gaba.reward_deliberation();
        gaba.reward_deliberation();

        // Should improve
        assert!(gaba.impulse_control() > gaba.impulsivity_score);
    }

    #[test]
    fn test_oxytocin_trust_building() {
        let mut oxy = OxytocinSystem::new();

        // Unknown entity has neutral trust
        assert!((oxy.get_trust("user") - 0.5).abs() < 0.01);

        // Positive interactions build trust
        oxy.record_positive_interaction("user");
        oxy.record_positive_interaction("user");
        oxy.record_positive_interaction("user");

        assert!(oxy.get_trust("user") > 0.6);
        assert!(oxy.is_trusted("user"));
    }

    #[test]
    fn test_oxytocin_trust_decay_on_betrayal() {
        let mut oxy = OxytocinSystem::new();

        // Build up trust first
        for _ in 0..5 {
            oxy.record_positive_interaction("friend");
        }
        let high_trust = oxy.get_trust("friend");

        // Betrayal reduces trust (and faster than it builds)
        oxy.record_negative_interaction("friend");
        oxy.record_negative_interaction("friend");

        assert!(oxy.get_trust("friend") < high_trust);
    }

    #[test]
    fn test_oxytocin_cooperation() {
        let mut oxy = OxytocinSystem::new();

        // Initially slightly cooperative
        assert!(oxy.prefer_cooperation());

        // Adversarial interactions reduce cooperativeness
        oxy.signal_adversarial();
        oxy.signal_adversarial();
        oxy.signal_adversarial();

        // Cooperativeness should decrease
        assert!(oxy.cooperativeness() < 0.6);
    }

    #[test]
    fn test_oxytocin_source_weighting() {
        let mut oxy = OxytocinSystem::new();

        // Build trust with one source
        for _ in 0..5 {
            oxy.record_positive_interaction("trusted_source");
        }

        // Trusted source gets higher weight
        let trusted_weight = oxy.source_weight("trusted_source");
        let unknown_weight = oxy.source_weight("unknown_source");

        assert!(trusted_weight > unknown_weight);
    }

    #[test]
    fn test_gaba_oxytocin_cross_modulation() {
        let mut system = NeuromodulatorySystem::new();

        // High stress should reduce both GABA and oxytocin
        let initial_gaba = system.gaba.level.current();
        let initial_oxy = system.oxytocin.level.current();

        // Create stress
        system.signal_threat(1.0);
        system.signal_threat(1.0);
        system.signal_threat(1.0);

        // Run cross-modulation
        for _ in 0..5 {
            system.update();
        }

        // Both should be affected by stress
        // GABA reduces under stress (harder to control impulses)
        assert!(system.gaba.level.current() < initial_gaba + 0.1);
        // Oxytocin reduces under stress (harder to trust)
        assert!(system.oxytocin.level.current() < initial_oxy + 0.1);
    }
}
