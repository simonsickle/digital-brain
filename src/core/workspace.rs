//! Global Workspace - The Consciousness Layer
//!
//! This is where consciousness emerges (theoretically).
//! Based on Global Workspace Theory (Baars, Dehaene).
//!
//! Key concepts:
//! - Signals compete for access to the workspace
//! - Winners get "broadcast" to all modules
//! - Limited capacity creates the bottleneck we experience as attention
//! - Broadcast = conscious access

use crate::signal::{BrainSignal, SignalType};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{BinaryHeap, VecDeque};
use std::cmp::Ordering;
use uuid::Uuid;

/// Configuration for the global workspace.
#[derive(Debug, Clone)]
pub struct WorkspaceConfig {
    /// Maximum items in conscious awareness (Miller's 7±2)
    pub capacity: usize,
    /// Minimum salience to enter competition
    pub salience_threshold: f64,
    /// How long a broadcast stays active (in processing cycles)
    pub broadcast_duration: usize,
    /// Maximum signals to consider per cycle
    pub max_candidates_per_cycle: usize,
}

impl Default for WorkspaceConfig {
    fn default() -> Self {
        Self {
            capacity: 7,  // Miller's magic number
            salience_threshold: 0.3,
            broadcast_duration: 5,
            max_candidates_per_cycle: 50,
        }
    }
}

/// A signal competing for workspace access.
#[derive(Debug, Clone)]
struct Competitor {
    signal: BrainSignal,
    competition_score: f64,
}

impl PartialEq for Competitor {
    fn eq(&self, other: &Self) -> bool {
        self.competition_score == other.competition_score
    }
}

impl Eq for Competitor {}

impl PartialOrd for Competitor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Competitor {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher score = higher priority (max-heap behavior)
        self.competition_score
            .partial_cmp(&other.competition_score)
            .unwrap_or(Ordering::Equal)
    }
}

/// A broadcast event - a signal that won workspace access.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Broadcast {
    /// Unique broadcast ID
    pub id: Uuid,
    /// The signal being broadcast
    pub signal: BrainSignal,
    /// When this broadcast started
    pub started_at: DateTime<Utc>,
    /// How many cycles remaining
    pub cycles_remaining: usize,
    /// Competition score that won
    pub winning_score: f64,
}

impl Broadcast {
    fn new(signal: BrainSignal, score: f64, duration: usize) -> Self {
        Self {
            id: Uuid::new_v4(),
            signal,
            started_at: Utc::now(),
            cycles_remaining: duration,
            winning_score: score,
        }
    }

    /// Tick down the broadcast duration.
    fn tick(&mut self) {
        self.cycles_remaining = self.cycles_remaining.saturating_sub(1);
    }

    /// Is this broadcast still active?
    pub fn is_active(&self) -> bool {
        self.cycles_remaining > 0
    }
}

/// The global workspace - the stage of consciousness.
pub struct GlobalWorkspace {
    config: WorkspaceConfig,
    /// Currently active broadcasts (conscious contents)
    active_broadcasts: VecDeque<Broadcast>,
    /// Signals competing for access
    competition_queue: BinaryHeap<Competitor>,
    /// History of all broadcasts (for analysis)
    broadcast_history: Vec<Broadcast>,
    /// Registered modules that receive broadcasts
    registered_modules: Vec<String>,
    /// Current processing cycle
    cycle_count: u64,
}

impl GlobalWorkspace {
    /// Create a new global workspace with default config.
    pub fn new() -> Self {
        Self::with_config(WorkspaceConfig::default())
    }

    /// Create a new global workspace with custom config.
    pub fn with_config(config: WorkspaceConfig) -> Self {
        Self {
            config,
            active_broadcasts: VecDeque::new(),
            competition_queue: BinaryHeap::new(),
            broadcast_history: Vec::new(),
            registered_modules: Vec::new(),
            cycle_count: 0,
        }
    }

    /// Register a module to receive broadcasts.
    pub fn register_module(&mut self, module_id: impl Into<String>) {
        let id = module_id.into();
        if !self.registered_modules.contains(&id) {
            self.registered_modules.push(id);
        }
    }

    /// Submit a signal for competition.
    pub fn submit(&mut self, signal: BrainSignal) -> bool {
        // Check salience threshold
        if signal.salience.value() < self.config.salience_threshold {
            return false;
        }

        // Calculate competition score
        let score = self.calculate_competition_score(&signal);

        self.competition_queue.push(Competitor {
            signal,
            competition_score: score,
        });

        true
    }

    /// Calculate how competitive a signal is.
    fn calculate_competition_score(&self, signal: &BrainSignal) -> f64 {
        // Base score from salience
        let mut score = signal.salience.value();

        // Boost for emotional content
        score += signal.emotional_intensity() * 0.3;

        // Boost for high arousal
        if signal.arousal.is_high() {
            score += 0.2;
        }

        // Boost for priority
        score += (signal.priority as f64) * 0.1;

        // Slight randomness to prevent deterministic lock-out
        score += (self.cycle_count as f64 % 100.0) * 0.001;

        score.clamp(0.0, 2.0)
    }

    /// Run one processing cycle.
    ///
    /// This is where the magic happens:
    /// 1. Expire old broadcasts
    /// 2. Select winners from competition queue
    /// 3. Broadcast winners to all modules
    ///
    /// Returns the new broadcasts from this cycle.
    pub fn process_cycle(&mut self) -> Vec<Broadcast> {
        self.cycle_count += 1;
        let mut new_broadcasts = Vec::new();

        // 1. Tick existing broadcasts and remove expired
        for broadcast in &mut self.active_broadcasts {
            broadcast.tick();
        }
        self.active_broadcasts.retain(|b| b.is_active());

        // 2. Calculate available capacity
        let available = self.config.capacity.saturating_sub(self.active_broadcasts.len());

        // 3. Select winners from competition queue
        let mut selected = 0;
        while selected < available && !self.competition_queue.is_empty() {
            if let Some(competitor) = self.competition_queue.pop() {
                let broadcast = Broadcast::new(
                    competitor.signal,
                    competitor.competition_score,
                    self.config.broadcast_duration,
                );

                self.broadcast_history.push(broadcast.clone());
                new_broadcasts.push(broadcast.clone());
                self.active_broadcasts.push_back(broadcast);
                selected += 1;
            }
        }

        // 4. Clear remaining competitors (they lost this cycle)
        self.competition_queue.clear();

        // 5. Trim history
        if self.broadcast_history.len() > 10000 {
            self.broadcast_history.drain(0..5000);
        }

        new_broadcasts
    }

    /// Get currently active broadcasts (conscious contents).
    pub fn conscious_contents(&self) -> Vec<&Broadcast> {
        self.active_broadcasts.iter().collect()
    }

    /// Is a particular signal type currently in consciousness?
    pub fn is_conscious(&self, signal_type: SignalType) -> bool {
        self.active_broadcasts
            .iter()
            .any(|b| b.signal.signal_type == signal_type)
    }

    /// Get statistics about workspace activity.
    pub fn stats(&self) -> WorkspaceStats {
        let recent_broadcasts: Vec<_> = self.broadcast_history
            .iter()
            .rev()
            .take(100)
            .collect();

        let avg_winning_score = if recent_broadcasts.is_empty() {
            0.0
        } else {
            recent_broadcasts.iter().map(|b| b.winning_score).sum::<f64>()
                / recent_broadcasts.len() as f64
        };

        let emotional_broadcast_rate = if recent_broadcasts.is_empty() {
            0.0
        } else {
            let emotional_count = recent_broadcasts
                .iter()
                .filter(|b| b.signal.emotional_intensity() > 0.5)
                .count();
            emotional_count as f64 / recent_broadcasts.len() as f64
        };

        WorkspaceStats {
            current_capacity_used: self.active_broadcasts.len(),
            max_capacity: self.config.capacity,
            total_broadcasts: self.broadcast_history.len(),
            current_cycle: self.cycle_count,
            avg_winning_score,
            emotional_broadcast_rate,
            registered_modules: self.registered_modules.len(),
            pending_competitors: self.competition_queue.len(),
        }
    }

    /// Get the current processing cycle number.
    pub fn current_cycle(&self) -> u64 {
        self.cycle_count
    }
}

impl Default for GlobalWorkspace {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about workspace activity.
#[derive(Debug, Clone)]
pub struct WorkspaceStats {
    pub current_capacity_used: usize,
    pub max_capacity: usize,
    pub total_broadcasts: usize,
    pub current_cycle: u64,
    pub avg_winning_score: f64,
    pub emotional_broadcast_rate: f64,
    pub registered_modules: usize,
    pub pending_competitors: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workspace_creation() {
        let ws = GlobalWorkspace::new();
        assert_eq!(ws.config.capacity, 7);
        assert!(ws.active_broadcasts.is_empty());
    }

    #[test]
    fn test_salience_threshold() {
        let mut ws = GlobalWorkspace::new();
        
        // Low salience signal should be rejected
        let low = BrainSignal::new("test", SignalType::Sensory, "low")
            .with_salience(0.1);
        assert!(!ws.submit(low));
        
        // High salience signal should be accepted
        let high = BrainSignal::new("test", SignalType::Sensory, "high")
            .with_salience(0.8);
        assert!(ws.submit(high));
    }

    #[test]
    fn test_competition() {
        let mut ws = GlobalWorkspace::with_config(WorkspaceConfig {
            capacity: 2,
            ..Default::default()
        });
        
        // Submit 3 signals with different salience
        ws.submit(BrainSignal::new("test", SignalType::Sensory, "low").with_salience(0.4));
        ws.submit(BrainSignal::new("test", SignalType::Sensory, "high").with_salience(0.9));
        ws.submit(BrainSignal::new("test", SignalType::Sensory, "medium").with_salience(0.6));
        
        // Process cycle
        let broadcasts = ws.process_cycle();
        
        // Only 2 should win (capacity limit)
        assert_eq!(broadcasts.len(), 2);
        
        // Highest salience should win
        assert!(broadcasts[0].winning_score > broadcasts[1].winning_score);
    }

    #[test]
    fn test_emotional_boost() {
        let mut ws = GlobalWorkspace::with_config(WorkspaceConfig {
            capacity: 1,
            ..Default::default()
        });
        
        // Submit neutral and emotional signals with same base salience
        ws.submit(
            BrainSignal::new("test", SignalType::Sensory, "neutral")
                .with_salience(0.7)
                .with_valence(0.0)
        );
        ws.submit(
            BrainSignal::new("test", SignalType::Sensory, "emotional")
                .with_salience(0.7)
                .with_valence(0.9)
                .with_arousal(0.8)
        );
        
        let broadcasts = ws.process_cycle();
        
        // Emotional signal should win due to boost
        let content: String = serde_json::from_value(broadcasts[0].signal.content.clone())
            .unwrap_or_default();
        assert_eq!(content, "emotional");
    }

    #[test]
    fn test_broadcast_expiration() {
        let mut ws = GlobalWorkspace::with_config(WorkspaceConfig {
            capacity: 7,
            broadcast_duration: 2,
            ..Default::default()
        });
        
        ws.submit(BrainSignal::new("test", SignalType::Sensory, "test").with_salience(0.8));
        ws.process_cycle();
        
        assert_eq!(ws.conscious_contents().len(), 1);
        
        // Cycle 2 - still active
        ws.process_cycle();
        assert_eq!(ws.conscious_contents().len(), 1);
        
        // Cycle 3 - should expire
        ws.process_cycle();
        assert_eq!(ws.conscious_contents().len(), 0);
    }

    #[test]
    fn test_capacity_limit() {
        let mut ws = GlobalWorkspace::with_config(WorkspaceConfig {
            capacity: 3,
            broadcast_duration: 10,
            ..Default::default()
        });
        
        // Submit 5 high-salience signals
        for i in 0..5 {
            ws.submit(
                BrainSignal::new("test", SignalType::Sensory, format!("signal{}", i))
                    .with_salience(0.8)
            );
        }
        
        ws.process_cycle();
        
        // Only 3 should be in consciousness
        assert_eq!(ws.conscious_contents().len(), 3);
    }
}
