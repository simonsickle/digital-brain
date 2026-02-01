//! Sleep & Dream System
//!
//! Implements sleep states and dream processing for memory consolidation.
//! Inspired by biological sleep cycles (NREM/REM) and their role in learning.
//!
//! # Sleep Stages
//!
//! ```text
//! AWAKE → DROWSY → LIGHT_SLEEP → DEEP_SLEEP → REM_DREAM → cycle...
//!                                                  ↓
//!                                           (imagination.dream())
//! ```
//!
//! # What Happens During Sleep
//!
//! - **Light Sleep**: Basic maintenance, clear working memory buffers
//! - **Deep Sleep**: Memory consolidation, schema formation, decay/pruning
//! - **REM/Dreams**: Creative recombination via imagination engine
//!
//! # Integration with Consciousness
//!
//! Sleep can be triggered by:
//! - Extended idle periods
//! - Explicit sleep command
//! - High cognitive load (need to consolidate)
//! - Time-based (circadian-like rhythm)

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

use crate::core::imagination::{
    DreamResult, Imagining, ImaginationEngine, ImaginationType, MemorySource,
};
use crate::core::llm::LlmBackend;

/// Sleep stages
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SleepStage {
    /// Fully awake and processing
    Awake,
    /// Transitioning to sleep, reduced responsiveness
    Drowsy,
    /// Light sleep, basic maintenance
    LightSleep,
    /// Deep sleep, memory consolidation
    DeepSleep,
    /// REM sleep, dreaming and creative processing
    RemDream,
    /// Waking up, transitioning back to awake
    Waking,
}

impl SleepStage {
    /// Whether external stimuli should be processed
    pub fn is_responsive(&self) -> bool {
        matches!(self, Self::Awake | Self::Drowsy | Self::Waking)
    }

    /// Whether dreams can occur
    pub fn can_dream(&self) -> bool {
        matches!(self, Self::RemDream)
    }

    /// Whether consolidation should happen
    pub fn should_consolidate(&self) -> bool {
        matches!(self, Self::DeepSleep)
    }
}

/// Configuration for the sleep system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SleepConfig {
    /// Idle cycles before getting drowsy
    pub drowsy_threshold_cycles: u64,
    /// Cycles in drowsy before light sleep
    pub light_sleep_onset_cycles: u64,
    /// Duration of light sleep (cycles)
    pub light_sleep_duration: u64,
    /// Duration of deep sleep (cycles)
    pub deep_sleep_duration: u64,
    /// Duration of REM/dream period (cycles)
    pub rem_duration: u64,
    /// Number of dream associations per REM period
    pub dream_associations: usize,
    /// Priority threshold to wake from sleep
    pub wake_threshold: f64,
    /// Enable automatic sleep
    pub auto_sleep: bool,
    /// Minimum awake time before auto-sleep (cycles)
    pub min_awake_cycles: u64,
}

impl Default for SleepConfig {
    fn default() -> Self {
        Self {
            drowsy_threshold_cycles: 200,
            light_sleep_onset_cycles: 50,
            light_sleep_duration: 30,
            deep_sleep_duration: 50,
            rem_duration: 40,
            dream_associations: 5,
            wake_threshold: 0.8,
            auto_sleep: true,
            min_awake_cycles: 500,
        }
    }
}

/// A single dream from a sleep cycle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dream {
    /// When the dream occurred
    pub timestamp: DateTime<Utc>,
    /// The dream content (chain of associations)
    pub sequence: Vec<Imagining>,
    /// What triggered/seeded the dream
    pub seed: Option<String>,
    /// Sleep cycle number
    pub cycle_number: u32,
    /// How vivid/intense (0.0-1.0)
    pub vividness: f64,
    /// Emotional valence (-1.0 to 1.0)
    pub valence: f64,
}

/// Record of a complete sleep session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SleepSession {
    /// When sleep started
    pub started_at: DateTime<Utc>,
    /// When sleep ended
    pub ended_at: Option<DateTime<Utc>>,
    /// Dreams during this session
    pub dreams: Vec<Dream>,
    /// Number of full sleep cycles completed
    pub cycles_completed: u32,
    /// Memories consolidated
    pub memories_consolidated: u32,
    /// What caused waking
    pub wake_reason: Option<WakeReason>,
}

/// Reasons for waking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WakeReason {
    /// Natural wake after full rest
    Rested,
    /// High-priority stimulus
    Interrupt { stimulus: String, priority: f64 },
    /// Explicit wake command
    Command,
    /// Cycle limit reached
    CycleLimit,
    /// Error during sleep
    Error(String),
}

/// Statistics about sleep
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SleepStats {
    pub total_sleep_sessions: u64,
    pub total_sleep_cycles: u64,
    pub total_dreams: u64,
    pub total_consolidated: u64,
    pub average_dream_vividness: f64,
    pub time_in_stage: std::collections::HashMap<String, u64>,
}

/// The sleep/dream system
pub struct SleepSystem<L: LlmBackend> {
    config: SleepConfig,
    stage: SleepStage,
    stage_cycles: u64,
    idle_cycles: u64,
    awake_cycles: u64,
    current_session: Option<SleepSession>,
    sleep_cycle_count: u32,
    recent_dreams: VecDeque<Dream>,
    imagination: ImaginationEngine<L>,
    stats: SleepStats,
}

impl<L: LlmBackend> SleepSystem<L> {
    /// Create a new sleep system
    pub fn new(config: SleepConfig, imagination: ImaginationEngine<L>) -> Self {
        Self {
            config,
            stage: SleepStage::Awake,
            stage_cycles: 0,
            idle_cycles: 0,
            awake_cycles: 0,
            current_session: None,
            sleep_cycle_count: 0,
            recent_dreams: VecDeque::with_capacity(20),
            imagination,
            stats: SleepStats::default(),
        }
    }

    /// Get current sleep stage
    pub fn stage(&self) -> SleepStage {
        self.stage
    }

    /// Check if currently sleeping
    pub fn is_sleeping(&self) -> bool {
        !matches!(self.stage, SleepStage::Awake)
    }

    /// Check if responsive to stimuli
    pub fn is_responsive(&self) -> bool {
        self.stage.is_responsive()
    }

    /// Get recent dreams
    pub fn recent_dreams(&self) -> &VecDeque<Dream> {
        &self.recent_dreams
    }

    /// Get statistics
    pub fn stats(&self) -> &SleepStats {
        &self.stats
    }

    /// Record idle cycle (called by consciousness loop)
    pub fn record_idle(&mut self) {
        self.idle_cycles += 1;
    }

    /// Record active cycle (called by consciousness loop)
    pub fn record_active(&mut self) {
        self.idle_cycles = 0;
        self.awake_cycles += 1;
    }

    /// Check if we should start sleeping
    pub fn should_sleep(&self) -> bool {
        if !self.config.auto_sleep {
            return false;
        }

        if self.awake_cycles < self.config.min_awake_cycles {
            return false;
        }

        self.idle_cycles >= self.config.drowsy_threshold_cycles
    }

    /// Initiate sleep
    pub fn initiate_sleep(&mut self) {
        if self.stage != SleepStage::Awake {
            return;
        }

        self.stage = SleepStage::Drowsy;
        self.stage_cycles = 0;
        self.sleep_cycle_count = 0;

        self.current_session = Some(SleepSession {
            started_at: Utc::now(),
            ended_at: None,
            dreams: Vec::new(),
            cycles_completed: 0,
            memories_consolidated: 0,
            wake_reason: None,
        });

        self.stats.total_sleep_sessions += 1;
    }

    /// Force immediate wake
    pub fn wake(&mut self, reason: WakeReason) {
        if !self.is_sleeping() {
            return;
        }

        self.stage = SleepStage::Waking;

        if let Some(ref mut session) = self.current_session {
            session.ended_at = Some(Utc::now());
            session.wake_reason = Some(reason);
        }

        // Reset counters
        self.stage_cycles = 0;
        self.awake_cycles = 0;
        self.current_session = None;
    }

    /// Process one sleep cycle (called by consciousness loop)
    pub async fn tick(&mut self, memories: Vec<MemorySource>) -> SleepTickResult {
        self.stage_cycles += 1;

        // Track time in stage
        let stage_key = format!("{:?}", self.stage);
        *self.stats.time_in_stage.entry(stage_key).or_insert(0) += 1;

        match self.stage {
            SleepStage::Awake => SleepTickResult::Awake,

            SleepStage::Drowsy => {
                if self.stage_cycles >= self.config.light_sleep_onset_cycles {
                    self.transition_to(SleepStage::LightSleep);
                }
                SleepTickResult::Transitioning
            }

            SleepStage::LightSleep => {
                // Basic maintenance happens here
                if self.stage_cycles >= self.config.light_sleep_duration {
                    self.transition_to(SleepStage::DeepSleep);
                }
                SleepTickResult::Maintenance
            }

            SleepStage::DeepSleep => {
                // Memory consolidation
                let consolidated = self.consolidate_memories(&memories);

                if self.stage_cycles >= self.config.deep_sleep_duration {
                    self.transition_to(SleepStage::RemDream);
                }

                SleepTickResult::Consolidated { count: consolidated }
            }

            SleepStage::RemDream => {
                // Dream!
                let dream_result = self.dream(memories).await;

                if self.stage_cycles >= self.config.rem_duration {
                    self.sleep_cycle_count += 1;
                    self.stats.total_sleep_cycles += 1;

                    if let Some(ref mut session) = self.current_session {
                        session.cycles_completed = self.sleep_cycle_count;
                    }

                    // Start another cycle or wake naturally
                    if self.sleep_cycle_count >= 3 {
                        // After 3 cycles, wake naturally
                        self.wake(WakeReason::Rested);
                        return SleepTickResult::Waking;
                    } else {
                        // Start another cycle
                        self.transition_to(SleepStage::LightSleep);
                    }
                }

                match dream_result {
                    Some(dream) => SleepTickResult::Dreamed { dream },
                    None => SleepTickResult::Dreaming,
                }
            }

            SleepStage::Waking => {
                self.stage = SleepStage::Awake;
                SleepTickResult::Awake
            }
        }
    }

    /// Check if a stimulus should wake us
    pub fn check_wake_stimulus(&mut self, priority: f64, description: &str) -> bool {
        if !self.is_sleeping() {
            return false;
        }

        if priority >= self.config.wake_threshold {
            self.wake(WakeReason::Interrupt {
                stimulus: description.to_string(),
                priority,
            });
            return true;
        }

        false
    }

    /// Transition to a new sleep stage
    fn transition_to(&mut self, stage: SleepStage) {
        self.stage = stage;
        self.stage_cycles = 0;
    }

    /// Consolidate memories during deep sleep
    fn consolidate_memories(&mut self, memories: &[MemorySource]) -> u32 {
        // In a full implementation, this would:
        // - Decay old, low-importance memories
        // - Strengthen frequently accessed memories
        // - Merge similar memories
        // - Update schemas
        //
        // For now, we just count how many we "processed"
        let consolidated = memories.len().min(10) as u32;

        if let Some(ref mut session) = self.current_session {
            session.memories_consolidated += consolidated;
        }

        self.stats.total_consolidated += consolidated as u64;
        consolidated
    }

    /// Generate a dream during REM
    async fn dream(&mut self, memories: Vec<MemorySource>) -> Option<Dream> {
        // Only dream once per REM period (at the start)
        if self.stage_cycles > 1 {
            return None;
        }

        // Pick a seed from recent memories
        let seed = memories.first().cloned();

        // Run imagination dream sequence
        let result = self
            .imagination
            .dream(seed.clone(), Some(self.config.dream_associations))
            .await;

        match result {
            Ok(dream_result) => {
                // Calculate dream properties
                let vividness = dream_result
                    .sequence
                    .iter()
                    .map(|i| i.novelty)
                    .sum::<f64>()
                    / dream_result.sequence.len().max(1) as f64;

                let valence = dream_result
                    .sequence
                    .iter()
                    .map(|i| i.confidence - 0.5) // Center around 0
                    .sum::<f64>()
                    / dream_result.sequence.len().max(1) as f64;

                let dream = Dream {
                    timestamp: Utc::now(),
                    sequence: dream_result.sequence,
                    seed: seed.map(|s| s.content),
                    cycle_number: self.sleep_cycle_count,
                    vividness,
                    valence,
                };

                // Store dream
                self.recent_dreams.push_back(dream.clone());
                if self.recent_dreams.len() > 20 {
                    self.recent_dreams.pop_front();
                }

                if let Some(ref mut session) = self.current_session {
                    session.dreams.push(dream.clone());
                }

                self.stats.total_dreams += 1;
                self.stats.average_dream_vividness = (self.stats.average_dream_vividness
                    * (self.stats.total_dreams - 1) as f64
                    + vividness)
                    / self.stats.total_dreams as f64;

                Some(dream)
            }
            Err(e) => {
                tracing::warn!("Dream generation failed: {}", e);
                None
            }
        }
    }
}

/// Result of a sleep tick
#[derive(Debug, Clone)]
pub enum SleepTickResult {
    /// Fully awake
    Awake,
    /// Transitioning between stages
    Transitioning,
    /// Doing maintenance (light sleep)
    Maintenance,
    /// Consolidated memories
    Consolidated { count: u32 },
    /// Currently dreaming (no dream completed yet)
    Dreaming,
    /// Completed a dream
    Dreamed { dream: Dream },
    /// Waking up
    Waking,
}

impl SleepTickResult {
    /// Whether we're actively sleeping
    pub fn is_sleeping(&self) -> bool {
        !matches!(self, Self::Awake)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sleep_stages() {
        assert!(SleepStage::Awake.is_responsive());
        assert!(SleepStage::Drowsy.is_responsive());
        assert!(!SleepStage::DeepSleep.is_responsive());

        assert!(SleepStage::RemDream.can_dream());
        assert!(!SleepStage::DeepSleep.can_dream());

        assert!(SleepStage::DeepSleep.should_consolidate());
        assert!(!SleepStage::RemDream.should_consolidate());
    }

    #[test]
    fn test_sleep_config_defaults() {
        let config = SleepConfig::default();
        assert!(config.auto_sleep);
        assert!(config.drowsy_threshold_cycles > 0);
        assert!(config.dream_associations > 0);
    }
}
