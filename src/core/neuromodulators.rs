//! Neuromodulatory System - Global State Regulation
//!
//! This module implements the four major neuromodulatory systems that globally
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
//!
//! # The Four Systems
//!
//! - **Dopamine**: Reward prediction, motivation, wanting (NOT just pleasure)
//! - **Serotonin**: Patience, mood stability, long-term thinking
//! - **Norepinephrine**: Arousal, vigilance, focused attention
//! - **Acetylcholine**: Learning enhancement, memory encoding, sustained attention

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

/// The complete neuromodulatory system.
///
/// Coordinates all four systems and provides cross-modulation.
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

        base * ach_factor * ne_factor * da_factor
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
    }

    /// Set attention focus
    pub fn focus_on(&mut self, target: String) {
        self.norepinephrine.set_focus(target);
        self.acetylcholine.signal_deep_processing(0.3);
    }

    /// Get current system state summary
    pub fn state(&self) -> NeuromodulatorState {
        NeuromodulatorState {
            dopamine: self.dopamine.level.current(),
            serotonin: self.serotonin.level.current(),
            norepinephrine: self.norepinephrine.level.current(),
            acetylcholine: self.acetylcholine.level.current(),
            motivation: self.dopamine.motivation(),
            patience: self.serotonin.patience(),
            stress: self.norepinephrine.stress(),
            learning_depth: self.acetylcholine.depth(),
            is_satiated: self.dopamine.is_satiated(),
            is_stressed: self.norepinephrine.is_stressed(),
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
    }

    /// Apply cross-system modulation effects
    fn apply_cross_modulation(&mut self) {
        // High stress reduces serotonin (and thus patience)
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
    /// Derived motivation level
    pub motivation: f64,
    /// Derived patience level
    pub patience: f64,
    /// Current stress level
    pub stress: f64,
    /// Current learning depth
    pub learning_depth: f64,
    /// Is reward system satiated?
    pub is_satiated: bool,
    /// Is system stressed?
    pub is_stressed: bool,
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
}
