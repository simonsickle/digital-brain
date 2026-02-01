//! Boredom Detection System - Loop & Stagnation Detection
//!
//! Boredom is a metacognitive signal indicating that current activity isn't
//! producing meaningful progress. Unlike frustration (failures) or fatigue
//! (resource depletion), boredom signals unproductive repetition.
//!
//! # Biological Inspiration
//!
//! In biological systems, boredom serves as a motivational signal:
//! - "Current activity isn't yielding sufficient reward/novelty"
//! - "Seek new stimuli or change approach"
//! - Prevents getting stuck in local optima
//!
//! # Detection Signals
//!
//! 1. **Output Similarity**: Recent outputs are semantically similar
//! 2. **Action Entropy Collapse**: Repeatedly taking the same actions
//! 3. **State Stagnation**: Memory/world state not changing
//! 4. **Reward Flatline**: Dopamine staying flat (no surprises)
//! 5. **Goal Distance Plateau**: Not getting closer to goals
//!
//! # Triggered Behaviors
//!
//! When boredom exceeds threshold:
//! - Increase exploration (raise action diversity)
//! - Suggest strategy switch
//! - Request fresh context or input
//! - Escalate/ask for help
//! - Trigger "context refresh" (prune repetitive context)

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

/// Configuration for the boredom detector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoredomConfig {
    /// Window size for tracking recent outputs/actions
    pub window_size: usize,
    /// Similarity threshold for considering outputs "repetitive" (0-1)
    pub similarity_threshold: f64,
    /// How many cycles before boredom starts accumulating
    pub grace_period_cycles: u32,
    /// Rate at which boredom accumulates per repetitive cycle
    pub accumulation_rate: f64,
    /// Rate at which boredom decays when progress is made
    pub decay_rate: f64,
    /// Threshold for triggering boredom response
    pub trigger_threshold: f64,
    /// Maximum boredom level
    pub max_level: f64,
}

impl Default for BoredomConfig {
    fn default() -> Self {
        Self {
            window_size: 10,
            similarity_threshold: 0.75,
            grace_period_cycles: 3,
            accumulation_rate: 0.15,
            decay_rate: 0.1,
            trigger_threshold: 0.6,
            max_level: 1.0,
        }
    }
}

/// Represents a fingerprint of an action or output for comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivityFingerprint {
    /// Unique identifier for this activity
    pub id: String,
    /// Category/type of activity
    pub category: String,
    /// Hash or embedding for similarity comparison
    pub signature: Vec<f64>,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Optional: semantic content summary
    pub content_hash: Option<u64>,
}

impl ActivityFingerprint {
    pub fn new(id: &str, category: &str) -> Self {
        Self {
            id: id.to_string(),
            category: category.to_string(),
            signature: Vec::new(),
            timestamp: Utc::now(),
            content_hash: None,
        }
    }

    pub fn with_signature(mut self, signature: Vec<f64>) -> Self {
        self.signature = signature;
        self
    }

    pub fn with_content_hash(mut self, hash: u64) -> Self {
        self.content_hash = Some(hash);
        self
    }

    /// Calculate similarity to another fingerprint (0-1)
    pub fn similarity(&self, other: &ActivityFingerprint) -> f64 {
        // First check content hash if available (exact match)
        if let (Some(h1), Some(h2)) = (self.content_hash, other.content_hash) {
            if h1 == h2 {
                return 1.0; // Exact match
            }
        }

        // Category match contributes to similarity
        let category_sim = if self.category == other.category {
            0.3
        } else {
            0.0
        };

        // Signature similarity (cosine similarity if available)
        let sig_sim = if !self.signature.is_empty() && !other.signature.is_empty() {
            cosine_similarity(&self.signature, &other.signature)
        } else {
            0.0
        };

        category_sim + sig_sim * 0.7
    }
}

/// Calculate cosine similarity between two vectors
fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    (dot / (norm_a * norm_b)).clamp(0.0, 1.0)
}

/// Tracks action diversity (entropy)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ActionEntropyTracker {
    /// Count of each action category in recent window
    action_counts: HashMap<String, u32>,
    /// Total actions in window
    total_actions: u32,
}

impl ActionEntropyTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record an action
    pub fn record(&mut self, action_category: &str) {
        *self.action_counts.entry(action_category.to_string()).or_insert(0) += 1;
        self.total_actions += 1;
    }

    /// Remove an action (when sliding window moves)
    pub fn remove(&mut self, action_category: &str) {
        if let Some(count) = self.action_counts.get_mut(action_category) {
            if *count > 0 {
                *count -= 1;
                self.total_actions = self.total_actions.saturating_sub(1);
            }
            if *count == 0 {
                self.action_counts.remove(action_category);
            }
        }
    }

    /// Calculate Shannon entropy of action distribution (higher = more diverse)
    pub fn entropy(&self) -> f64 {
        if self.total_actions == 0 {
            return 0.0;
        }

        let mut entropy = 0.0;
        for count in self.action_counts.values() {
            if *count > 0 {
                let p = *count as f64 / self.total_actions as f64;
                entropy -= p * p.log2();
            }
        }
        entropy
    }

    /// Normalized entropy (0-1, where 1 = maximum diversity)
    pub fn normalized_entropy(&self) -> f64 {
        let n = self.action_counts.len();
        if n <= 1 {
            return 0.0;
        }

        let max_entropy = (n as f64).log2();
        if max_entropy == 0.0 {
            return 0.0;
        }

        (self.entropy() / max_entropy).clamp(0.0, 1.0)
    }

    /// Is the action distribution collapsed (low diversity)?
    pub fn is_collapsed(&self, threshold: f64) -> bool {
        self.normalized_entropy() < threshold
    }

    /// Clear the tracker
    pub fn clear(&mut self) {
        self.action_counts.clear();
        self.total_actions = 0;
    }
}

/// Tracks goal progress over time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressTracker {
    /// Recent goal distance measurements
    distances: VecDeque<f64>,
    /// Maximum window size
    max_size: usize,
    /// Threshold for "no progress" (slope)
    stagnation_threshold: f64,
}

impl ProgressTracker {
    pub fn new(max_size: usize) -> Self {
        Self {
            distances: VecDeque::with_capacity(max_size),
            max_size,
            stagnation_threshold: 0.01,
        }
    }

    /// Record current distance to goal (lower = closer)
    pub fn record_distance(&mut self, distance: f64) {
        if self.distances.len() >= self.max_size {
            self.distances.pop_front();
        }
        self.distances.push_back(distance);
    }

    /// Calculate progress rate (negative = getting closer, positive = getting farther)
    pub fn progress_rate(&self) -> f64 {
        if self.distances.len() < 2 {
            return 0.0;
        }

        // Simple linear regression slope
        let n = self.distances.len() as f64;
        let sum_x: f64 = (0..self.distances.len()).map(|i| i as f64).sum();
        let sum_y: f64 = self.distances.iter().sum();
        let sum_xy: f64 = self.distances.iter().enumerate()
            .map(|(i, y)| i as f64 * y)
            .sum();
        let sum_x2: f64 = (0..self.distances.len()).map(|i| (i as f64).powi(2)).sum();

        let denominator = n * sum_x2 - sum_x.powi(2);
        if denominator.abs() < 1e-10 {
            return 0.0;
        }

        (n * sum_xy - sum_x * sum_y) / denominator
    }

    /// Is progress stagnant?
    pub fn is_stagnant(&self) -> bool {
        let rate = self.progress_rate();
        // Stagnant if not making progress (rate >= 0 or very small negative)
        rate >= -self.stagnation_threshold
    }

    /// Clear history
    pub fn clear(&mut self) {
        self.distances.clear();
    }
}

/// Result of boredom assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoredomAssessment {
    /// Current boredom level (0-1)
    pub level: f64,
    /// Is boredom triggered?
    pub triggered: bool,
    /// Contributing factors
    pub factors: BoredomFactors,
    /// Recommended action
    pub recommendation: BoredomRecommendation,
}

/// Factors contributing to boredom
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BoredomFactors {
    /// Output similarity score (higher = more repetitive)
    pub output_similarity: f64,
    /// Action entropy (lower = less diverse)
    pub action_entropy: f64,
    /// Progress rate (0 or positive = stagnant)
    pub progress_rate: f64,
    /// Reward variance (lower = flatter dopamine)
    pub reward_variance: f64,
    /// Cycles since last novel event
    pub cycles_since_novelty: u32,
}

/// Recommended action when bored
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum BoredomRecommendation {
    /// Continue current activity
    Continue,
    /// Increase exploration/randomness
    IncreaseExploration,
    /// Try a different approach/strategy
    SwitchStrategy,
    /// Request new input or context
    RequestFreshInput,
    /// Ask for help or guidance
    SeekHelp,
    /// Take a break / context reset
    ContextReset,
}

/// The main boredom detection system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoredomTracker {
    /// Configuration
    config: BoredomConfig,
    /// Current boredom level (0 to max_level)
    level: f64,
    /// Recent activity fingerprints
    recent_activities: VecDeque<ActivityFingerprint>,
    /// Action entropy tracker
    entropy_tracker: ActionEntropyTracker,
    /// Goal progress tracker
    progress_tracker: ProgressTracker,
    /// Recent reward values (for variance calculation)
    recent_rewards: VecDeque<f64>,
    /// Cycles since last novel/interesting event
    cycles_since_novelty: u32,
    /// Total cycles processed
    total_cycles: u64,
    /// Last assessment result (cached)
    last_assessment: Option<BoredomAssessment>,
    /// Timestamp of last activity
    last_activity: Option<DateTime<Utc>>,
}

impl BoredomTracker {
    /// Create a new boredom tracker with default config
    pub fn new() -> Self {
        Self::with_config(BoredomConfig::default())
    }

    /// Create with custom config
    pub fn with_config(config: BoredomConfig) -> Self {
        Self {
            recent_activities: VecDeque::with_capacity(config.window_size),
            entropy_tracker: ActionEntropyTracker::new(),
            progress_tracker: ProgressTracker::new(config.window_size),
            recent_rewards: VecDeque::with_capacity(config.window_size),
            level: 0.0,
            cycles_since_novelty: 0,
            total_cycles: 0,
            last_assessment: None,
            last_activity: None,
            config,
        }
    }

    /// Record an activity (action or output)
    pub fn record_activity(&mut self, fingerprint: ActivityFingerprint) {
        // Check similarity to recent activities
        let max_similarity = self.recent_activities.iter()
            .map(|a| fingerprint.similarity(a))
            .fold(0.0_f64, |a, b| a.max(b));

        // Update entropy tracker
        self.entropy_tracker.record(&fingerprint.category);

        // If window is full, remove oldest
        if self.recent_activities.len() >= self.config.window_size {
            if let Some(old) = self.recent_activities.pop_front() {
                self.entropy_tracker.remove(&old.category);
            }
        }

        // Add new activity
        self.last_activity = Some(fingerprint.timestamp);
        self.recent_activities.push_back(fingerprint);

        // Update boredom based on similarity
        if max_similarity > self.config.similarity_threshold {
            self.cycles_since_novelty += 1;
            if self.cycles_since_novelty > self.config.grace_period_cycles {
                // Accumulate boredom for repetitive activity
                self.level = (self.level + self.config.accumulation_rate)
                    .min(self.config.max_level);
            }
        } else {
            // Novel activity - decay boredom
            self.signal_novelty();
        }

        self.total_cycles += 1;
    }

    /// Record a reward signal (for variance tracking)
    pub fn record_reward(&mut self, reward: f64) {
        if self.recent_rewards.len() >= self.config.window_size {
            self.recent_rewards.pop_front();
        }
        self.recent_rewards.push_back(reward);
    }

    /// Record goal progress (distance to goal, lower = better)
    pub fn record_goal_distance(&mut self, distance: f64) {
        self.progress_tracker.record_distance(distance);
    }

    /// Signal that something novel/interesting happened
    pub fn signal_novelty(&mut self) {
        self.cycles_since_novelty = 0;
        self.level = (self.level - self.config.decay_rate * 2.0).max(0.0);
    }

    /// Signal meaningful progress was made
    pub fn signal_progress(&mut self) {
        self.level = (self.level - self.config.decay_rate).max(0.0);
        self.cycles_since_novelty = 0;
    }

    /// Get current boredom level (0-1)
    pub fn level(&self) -> f64 {
        self.level / self.config.max_level
    }

    /// Is boredom triggered (above threshold)?
    pub fn is_triggered(&self) -> bool {
        self.level() >= self.config.trigger_threshold
    }

    /// Perform full boredom assessment
    pub fn assess(&mut self) -> BoredomAssessment {
        let factors = self.calculate_factors();
        let level = self.level();
        let triggered = level >= self.config.trigger_threshold;
        
        let recommendation = if !triggered {
            BoredomRecommendation::Continue
        } else {
            self.determine_recommendation(&factors)
        };

        let assessment = BoredomAssessment {
            level,
            triggered,
            factors,
            recommendation,
        };

        self.last_assessment = Some(assessment.clone());
        assessment
    }

    /// Calculate contributing factors
    fn calculate_factors(&self) -> BoredomFactors {
        // Output similarity: average similarity of recent consecutive outputs
        let output_similarity = self.calculate_output_similarity();

        // Action entropy (inverted - low entropy = high boredom contribution)
        let action_entropy = self.entropy_tracker.normalized_entropy();

        // Progress rate
        let progress_rate = self.progress_tracker.progress_rate();

        // Reward variance
        let reward_variance = self.calculate_reward_variance();

        BoredomFactors {
            output_similarity,
            action_entropy,
            progress_rate,
            reward_variance,
            cycles_since_novelty: self.cycles_since_novelty,
        }
    }

    /// Calculate average similarity between consecutive activities
    fn calculate_output_similarity(&self) -> f64 {
        if self.recent_activities.len() < 2 {
            return 0.0;
        }

        let activities: Vec<_> = self.recent_activities.iter().collect();
        let mut total_sim = 0.0;
        let mut count = 0;

        for i in 1..activities.len() {
            total_sim += activities[i].similarity(activities[i - 1]);
            count += 1;
        }

        if count > 0 {
            total_sim / count as f64
        } else {
            0.0
        }
    }

    /// Calculate variance in recent rewards
    fn calculate_reward_variance(&self) -> f64 {
        if self.recent_rewards.len() < 2 {
            return 0.0;
        }

        let mean: f64 = self.recent_rewards.iter().sum::<f64>() / self.recent_rewards.len() as f64;
        let variance: f64 = self.recent_rewards.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / self.recent_rewards.len() as f64;

        variance
    }

    /// Determine recommendation based on factors
    fn determine_recommendation(&self, factors: &BoredomFactors) -> BoredomRecommendation {
        let level = self.level();

        // Severe boredom - drastic action needed
        if level > 0.9 {
            return BoredomRecommendation::ContextReset;
        }

        // High boredom + long time since novelty - seek external help
        if level > 0.8 && factors.cycles_since_novelty > self.config.window_size as u32 * 2 {
            return BoredomRecommendation::SeekHelp;
        }

        // Moderate-high boredom + stagnant progress - try different strategy
        if level > 0.7 && factors.progress_rate >= 0.0 {
            return BoredomRecommendation::SwitchStrategy;
        }

        // Moderate boredom + low action diversity - increase exploration
        if level > 0.6 && factors.action_entropy < 0.4 {
            return BoredomRecommendation::IncreaseExploration;
        }

        // Default for triggered boredom - request fresh input
        BoredomRecommendation::RequestFreshInput
    }

    /// Get exploration boost (higher when bored)
    /// Use this to increase temperature/noise in action selection
    pub fn exploration_boost(&self) -> f64 {
        // Sigmoid-like curve: low boost until threshold, then increases
        let level = self.level();
        if level < self.config.trigger_threshold * 0.5 {
            0.0
        } else {
            ((level - self.config.trigger_threshold * 0.5) * 2.0).min(1.0) * 0.5
        }
    }

    /// Reset boredom (e.g., after context change)
    pub fn reset(&mut self) {
        self.level = 0.0;
        self.cycles_since_novelty = 0;
        self.recent_activities.clear();
        self.recent_rewards.clear();
        self.entropy_tracker.clear();
        self.progress_tracker.clear();
        self.last_assessment = None;
    }

    /// Homeostatic update (call each cycle)
    pub fn update(&mut self) {
        // Slow natural decay of boredom
        self.level = (self.level - self.config.decay_rate * 0.2).max(0.0);

        // Increment novelty counter if no recent activity
        if let Some(last) = self.last_activity {
            if Utc::now() - last > Duration::seconds(60) {
                self.cycles_since_novelty += 1;
            }
        }
    }

    /// Get last cached assessment
    pub fn last_assessment(&self) -> Option<&BoredomAssessment> {
        self.last_assessment.as_ref()
    }

    /// Get statistics
    pub fn stats(&self) -> BoredomStats {
        BoredomStats {
            total_cycles: self.total_cycles,
            current_level: self.level(),
            action_entropy: self.entropy_tracker.normalized_entropy(),
            cycles_since_novelty: self.cycles_since_novelty,
            is_triggered: self.is_triggered(),
            recent_activity_count: self.recent_activities.len(),
        }
    }
}

impl Default for BoredomTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about the boredom tracker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoredomStats {
    pub total_cycles: u64,
    pub current_level: f64,
    pub action_entropy: f64,
    pub cycles_since_novelty: u32,
    pub is_triggered: bool,
    pub recent_activity_count: usize,
}

// ============================================================================
// INTEGRATION TRAIT
// ============================================================================

/// Trait for types that can provide boredom signals
pub trait BoredomSignalSource {
    /// Get a fingerprint of the current output/activity
    fn activity_fingerprint(&self) -> ActivityFingerprint;
    
    /// Get current distance to active goal (if any)
    fn goal_distance(&self) -> Option<f64>;
    
    /// Get current reward/value signal
    fn reward_signal(&self) -> Option<f64>;
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_fingerprint(id: &str, category: &str) -> ActivityFingerprint {
        ActivityFingerprint::new(id, category)
    }

    fn make_fingerprint_with_hash(id: &str, category: &str, hash: u64) -> ActivityFingerprint {
        ActivityFingerprint::new(id, category).with_content_hash(hash)
    }

    #[test]
    fn test_boredom_tracker_creation() {
        let tracker = BoredomTracker::new();
        assert_eq!(tracker.level(), 0.0);
        assert!(!tracker.is_triggered());
    }

    #[test]
    fn test_novel_activities_no_boredom() {
        let mut tracker = BoredomTracker::new();

        // Record diverse activities
        for i in 0..10 {
            let fp = make_fingerprint(&format!("action_{}", i), &format!("category_{}", i % 5));
            tracker.record_activity(fp);
        }

        // Should have low boredom due to diversity
        assert!(tracker.level() < 0.3, "Diverse activities should not cause boredom");
    }

    #[test]
    fn test_repetitive_activities_cause_boredom() {
        let mut tracker = BoredomTracker::new();

        // Record same activity repeatedly (exact hash match)
        for i in 0..20 {
            let fp = make_fingerprint_with_hash(&format!("action_{}", i), "same_category", 12345);
            tracker.record_activity(fp);
        }

        // Should have high boredom
        assert!(tracker.level() > 0.5, "Repetitive activities should cause boredom: {}", tracker.level());
    }

    #[test]
    fn test_novelty_resets_boredom() {
        let mut tracker = BoredomTracker::new();

        // Build up boredom
        for i in 0..15 {
            let fp = make_fingerprint_with_hash(&format!("action_{}", i), "same", 12345);
            tracker.record_activity(fp);
        }

        let boredom_before = tracker.level();
        assert!(boredom_before > 0.3);

        // Signal novelty
        tracker.signal_novelty();

        assert!(tracker.level() < boredom_before, "Novelty should reduce boredom");
    }

    #[test]
    fn test_action_entropy_tracking() {
        let mut tracker = ActionEntropyTracker::new();

        // Single action type = zero entropy
        for _ in 0..10 {
            tracker.record("action_a");
        }
        assert!(tracker.normalized_entropy() < 0.1);

        // Add diversity
        tracker.clear();
        for category in &["a", "b", "c", "d", "e"] {
            for _ in 0..2 {
                tracker.record(category);
            }
        }
        assert!(tracker.normalized_entropy() > 0.9, "Uniform distribution should have high entropy");
    }

    #[test]
    fn test_progress_tracker() {
        let mut tracker = ProgressTracker::new(10);

        // Making progress (distance decreasing)
        for i in (0..10).rev() {
            tracker.record_distance(i as f64);
        }
        assert!(tracker.progress_rate() < 0.0, "Decreasing distance = negative rate (progress)");
        assert!(!tracker.is_stagnant());

        // Stagnant (distance not changing)
        let mut stagnant_tracker = ProgressTracker::new(10);
        for _ in 0..10 {
            stagnant_tracker.record_distance(5.0);
        }
        assert!(stagnant_tracker.is_stagnant());
    }

    #[test]
    fn test_boredom_assessment() {
        let mut tracker = BoredomTracker::new();

        // Build up boredom
        for i in 0..20 {
            let fp = make_fingerprint_with_hash(&format!("action_{}", i), "repetitive", 99999);
            tracker.record_activity(fp);
            tracker.record_reward(0.5); // Flat rewards
        }

        let assessment = tracker.assess();
        
        // Should be triggered after repetitive activity
        if assessment.triggered {
            assert_ne!(assessment.recommendation, BoredomRecommendation::Continue);
        }
    }

    #[test]
    fn test_exploration_boost() {
        let mut tracker = BoredomTracker::new();

        // Low boredom = no boost
        assert_eq!(tracker.exploration_boost(), 0.0);

        // Build up boredom
        for i in 0..25 {
            let fp = make_fingerprint_with_hash(&format!("action_{}", i), "same", 11111);
            tracker.record_activity(fp);
        }

        // High boredom = exploration boost
        assert!(tracker.exploration_boost() > 0.0, "Bored system should boost exploration");
    }

    #[test]
    fn test_fingerprint_similarity() {
        let fp1 = make_fingerprint("a", "coding").with_content_hash(123);
        let fp2 = make_fingerprint("b", "coding").with_content_hash(123);
        let fp3 = make_fingerprint("c", "coding").with_content_hash(456);
        let fp4 = make_fingerprint("d", "different");

        // Same hash = identical
        assert_eq!(fp1.similarity(&fp2), 1.0);

        // Same category, different hash
        let sim_same_cat = fp1.similarity(&fp3);
        assert!(sim_same_cat > 0.0 && sim_same_cat < 1.0);

        // Different category
        let sim_diff_cat = fp1.similarity(&fp4);
        assert!(sim_diff_cat < sim_same_cat);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let c = vec![0.0, 1.0, 0.0];

        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);
        assert!((cosine_similarity(&a, &c) - 0.0).abs() < 0.001);
    }
}
