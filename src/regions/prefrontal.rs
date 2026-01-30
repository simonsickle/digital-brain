//! Prefrontal Cortex - Working Memory System
//!
//! The limited-capacity workspace where conscious thought happens.
//! This is the "context window" of the digital brain.
//!
//! Key functions:
//! - Maintain active representations (7±2 items)
//! - Goal maintenance and task switching
//! - Chunking for efficiency
//! - Interference management

#[allow(unused_imports)]
use crate::signal::{BrainSignal, SignalType, Salience};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use uuid::Uuid;

/// Configuration for working memory.
#[derive(Debug, Clone)]
pub struct PrefrontalConfig {
    /// Maximum items in working memory (Miller's 7±2)
    pub capacity: usize,
    /// How long items persist without rehearsal (cycles)
    pub decay_cycles: usize,
    /// Maximum chunk size
    pub max_chunk_size: usize,
    /// Interference threshold (similar items compete)
    pub interference_threshold: f64,
}

impl Default for PrefrontalConfig {
    fn default() -> Self {
        Self {
            capacity: 7,
            decay_cycles: 10,
            max_chunk_size: 4,
            interference_threshold: 0.7,
        }
    }
}

/// An item held in working memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkingMemoryItem {
    /// Unique identifier
    pub id: Uuid,
    /// The content being held
    pub content: serde_json::Value,
    /// Source module
    pub source: String,
    /// When this entered working memory
    pub entered_at: DateTime<Utc>,
    /// Cycles until decay (refreshed by rehearsal)
    pub cycles_remaining: usize,
    /// How important this item is (affects displacement)
    pub priority: f64,
    /// Is this part of a chunk?
    pub chunk_id: Option<Uuid>,
    /// Tags for retrieval/association
    pub tags: Vec<String>,
}

impl WorkingMemoryItem {
    /// Create a new working memory item from a signal.
    pub fn from_signal(signal: &BrainSignal, decay_cycles: usize) -> Self {
        Self {
            id: Uuid::new_v4(),
            content: signal.content.clone(),
            source: signal.source.clone(),
            entered_at: Utc::now(),
            cycles_remaining: decay_cycles,
            priority: signal.salience.value() + signal.emotional_intensity() * 0.5,
            chunk_id: None,
            tags: Vec::new(),
        }
    }

    /// Rehearse this item (refresh decay counter).
    pub fn rehearse(&mut self, decay_cycles: usize) {
        self.cycles_remaining = decay_cycles;
    }

    /// Tick down the decay counter.
    pub fn tick(&mut self) {
        self.cycles_remaining = self.cycles_remaining.saturating_sub(1);
    }

    /// Is this item still active?
    pub fn is_active(&self) -> bool {
        self.cycles_remaining > 0
    }

    /// Add a tag.
    pub fn tag(&mut self, tag: impl Into<String>) {
        self.tags.push(tag.into());
    }
}

/// A chunk - multiple items grouped as one unit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub id: Uuid,
    pub items: Vec<Uuid>,
    pub label: String,
    pub created_at: DateTime<Utc>,
}

impl Chunk {
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            items: Vec::new(),
            label: label.into(),
            created_at: Utc::now(),
        }
    }

    pub fn add(&mut self, item_id: Uuid) {
        self.items.push(item_id);
    }

    pub fn size(&self) -> usize {
        self.items.len()
    }
}

/// The current goal being pursued.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Goal {
    pub id: Uuid,
    pub description: String,
    pub priority: f64,
    pub created_at: DateTime<Utc>,
    pub subgoals: Vec<Uuid>,
    pub completed: bool,
}

impl Goal {
    pub fn new(description: impl Into<String>, priority: f64) -> Self {
        Self {
            id: Uuid::new_v4(),
            description: description.into(),
            priority: priority.clamp(0.0, 1.0),
            created_at: Utc::now(),
            subgoals: Vec::new(),
            completed: false,
        }
    }
}

/// The prefrontal cortex - working memory manager.
pub struct PrefrontalCortex {
    config: PrefrontalConfig,
    /// Items currently in working memory
    items: VecDeque<WorkingMemoryItem>,
    /// Active chunks
    chunks: Vec<Chunk>,
    /// Current goal stack
    goals: Vec<Goal>,
    /// Processing cycle count
    cycle_count: u64,
    /// Items displaced due to capacity
    displacement_count: u64,
}

impl PrefrontalCortex {
    /// Create a new prefrontal cortex with default config.
    pub fn new() -> Self {
        Self::with_config(PrefrontalConfig::default())
    }

    /// Create with custom config.
    pub fn with_config(config: PrefrontalConfig) -> Self {
        Self {
            config,
            items: VecDeque::new(),
            chunks: Vec::new(),
            goals: Vec::new(),
            cycle_count: 0,
            displacement_count: 0,
        }
    }

    /// Load an item into working memory.
    /// Returns displaced item if at capacity.
    pub fn load(&mut self, signal: &BrainSignal) -> Option<WorkingMemoryItem> {
        let item = WorkingMemoryItem::from_signal(signal, self.config.decay_cycles);
        self.load_item(item)
    }

    /// Load a pre-constructed item.
    fn load_item(&mut self, item: WorkingMemoryItem) -> Option<WorkingMemoryItem> {
        // Check capacity (chunks count as 1 item each)
        let effective_count = self.effective_item_count();
        
        let displaced = if effective_count >= self.config.capacity {
            // Displace lowest priority item
            self.displace_lowest_priority()
        } else {
            None
        };

        self.items.push_back(item);
        displaced
    }

    /// Count items, treating chunks as single items.
    fn effective_item_count(&self) -> usize {
        let chunked_items: usize = self.chunks.iter().map(|c| c.size()).sum();
        let standalone_items = self.items.iter()
            .filter(|i| i.chunk_id.is_none())
            .count();
        
        standalone_items + self.chunks.len()
    }

    /// Displace the lowest priority item.
    fn displace_lowest_priority(&mut self) -> Option<WorkingMemoryItem> {
        if self.items.is_empty() {
            return None;
        }

        // Find lowest priority standalone item
        let mut min_priority = f64::MAX;
        let mut min_idx = 0;

        for (idx, item) in self.items.iter().enumerate() {
            if item.chunk_id.is_none() && item.priority < min_priority {
                min_priority = item.priority;
                min_idx = idx;
            }
        }

        self.displacement_count += 1;
        self.items.remove(min_idx)
    }

    /// Rehearse an item (keep it active).
    pub fn rehearse(&mut self, item_id: Uuid) -> bool {
        for item in &mut self.items {
            if item.id == item_id {
                item.rehearse(self.config.decay_cycles);
                return true;
            }
        }
        false
    }

    /// Rehearse all items (costly but prevents decay).
    pub fn rehearse_all(&mut self) {
        for item in &mut self.items {
            item.rehearse(self.config.decay_cycles);
        }
    }

    /// Create a chunk from multiple items.
    pub fn chunk(&mut self, item_ids: &[Uuid], label: impl Into<String>) -> Option<Uuid> {
        if item_ids.len() > self.config.max_chunk_size {
            return None;
        }

        let mut chunk = Chunk::new(label);
        
        for id in item_ids {
            // Verify item exists and mark it as chunked
            for item in &mut self.items {
                if item.id == *id && item.chunk_id.is_none() {
                    item.chunk_id = Some(chunk.id);
                    chunk.add(*id);
                    break;
                }
            }
        }

        if chunk.items.is_empty() {
            return None;
        }

        let chunk_id = chunk.id;
        self.chunks.push(chunk);
        Some(chunk_id)
    }

    /// Process one cycle (decay, cleanup).
    pub fn process_cycle(&mut self) -> CycleResult {
        self.cycle_count += 1;
        let mut decayed = Vec::new();

        // Tick all items
        for item in &mut self.items {
            item.tick();
        }

        // Remove decayed items
        let before = self.items.len();
        self.items.retain(|item| {
            if !item.is_active() {
                decayed.push(item.id);
                false
            } else {
                true
            }
        });

        // Clean up empty chunks
        self.chunks.retain(|chunk| {
            chunk.items.iter().any(|id| {
                self.items.iter().any(|item| item.id == *id)
            })
        });

        CycleResult {
            cycle: self.cycle_count,
            items_decayed: before - self.items.len(),
            decayed_ids: decayed,
            current_load: self.effective_item_count(),
            capacity: self.config.capacity,
        }
    }

    /// Set the current goal.
    pub fn set_goal(&mut self, goal: Goal) {
        self.goals.push(goal);
    }

    /// Get the current top goal.
    pub fn current_goal(&self) -> Option<&Goal> {
        self.goals.last()
    }

    /// Complete the current goal.
    pub fn complete_goal(&mut self) -> Option<Goal> {
        if let Some(mut goal) = self.goals.pop() {
            goal.completed = true;
            Some(goal)
        } else {
            None
        }
    }

    /// Get all items currently in working memory.
    pub fn contents(&self) -> Vec<&WorkingMemoryItem> {
        self.items.iter().collect()
    }

    /// Find items by tag.
    pub fn find_by_tag(&self, tag: &str) -> Vec<&WorkingMemoryItem> {
        self.items.iter()
            .filter(|item| item.tags.contains(&tag.to_string()))
            .collect()
    }

    /// Check if at capacity.
    pub fn is_full(&self) -> bool {
        self.effective_item_count() >= self.config.capacity
    }

    /// Get available capacity.
    pub fn available_capacity(&self) -> usize {
        self.config.capacity.saturating_sub(self.effective_item_count())
    }

    /// Get statistics.
    pub fn stats(&self) -> PrefrontalStats {
        PrefrontalStats {
            current_items: self.items.len(),
            effective_load: self.effective_item_count(),
            capacity: self.config.capacity,
            chunks: self.chunks.len(),
            active_goals: self.goals.len(),
            total_cycles: self.cycle_count,
            total_displacements: self.displacement_count,
            utilization: self.effective_item_count() as f64 / self.config.capacity as f64,
        }
    }
}

impl Default for PrefrontalCortex {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of a processing cycle.
#[derive(Debug, Clone)]
pub struct CycleResult {
    pub cycle: u64,
    pub items_decayed: usize,
    pub decayed_ids: Vec<Uuid>,
    pub current_load: usize,
    pub capacity: usize,
}

/// Statistics about working memory.
#[derive(Debug, Clone)]
pub struct PrefrontalStats {
    pub current_items: usize,
    pub effective_load: usize,
    pub capacity: usize,
    pub chunks: usize,
    pub active_goals: usize,
    pub total_cycles: u64,
    pub total_displacements: u64,
    pub utilization: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capacity_limit() {
        let mut pfc = PrefrontalCortex::with_config(PrefrontalConfig {
            capacity: 3,
            ..Default::default()
        });

        // Load 4 items
        for i in 0..4 {
            let signal = BrainSignal::new("test", SignalType::Memory, format!("item_{}", i))
                .with_salience(0.5);
            let displaced = pfc.load(&signal);
            
            if i < 3 {
                assert!(displaced.is_none());
            } else {
                assert!(displaced.is_some()); // 4th item displaces one
            }
        }

        assert_eq!(pfc.effective_item_count(), 3);
    }

    #[test]
    fn test_priority_displacement() {
        let mut pfc = PrefrontalCortex::with_config(PrefrontalConfig {
            capacity: 2,
            ..Default::default()
        });

        // Load low priority item
        let low = BrainSignal::new("test", SignalType::Memory, "low priority")
            .with_salience(0.2);
        pfc.load(&low);

        // Load high priority item
        let high = BrainSignal::new("test", SignalType::Memory, "high priority")
            .with_salience(0.9);
        pfc.load(&high);

        // Load another high priority - should displace low
        let high2 = BrainSignal::new("test", SignalType::Memory, "high priority 2")
            .with_salience(0.8);
        let displaced = pfc.load(&high2);

        assert!(displaced.is_some());
        let displaced_content = displaced.unwrap().content.to_string();
        assert!(displaced_content.contains("low"));
    }

    #[test]
    fn test_decay() {
        let mut pfc = PrefrontalCortex::with_config(PrefrontalConfig {
            capacity: 5,
            decay_cycles: 3,
            ..Default::default()
        });

        let signal = BrainSignal::new("test", SignalType::Memory, "will decay");
        pfc.load(&signal);

        assert_eq!(pfc.contents().len(), 1);

        // Process cycles until decay
        for _ in 0..3 {
            pfc.process_cycle();
        }

        assert_eq!(pfc.contents().len(), 0);
    }

    #[test]
    fn test_rehearsal_prevents_decay() {
        let mut pfc = PrefrontalCortex::with_config(PrefrontalConfig {
            capacity: 5,
            decay_cycles: 2,
            ..Default::default()
        });

        let signal = BrainSignal::new("test", SignalType::Memory, "rehearsed");
        pfc.load(&signal);
        let item_id = pfc.contents()[0].id;

        // Process one cycle
        pfc.process_cycle();
        assert_eq!(pfc.contents().len(), 1);

        // Rehearse
        pfc.rehearse(item_id);

        // Should survive another 2 cycles now
        pfc.process_cycle();
        assert_eq!(pfc.contents().len(), 1);
        pfc.process_cycle();
        assert_eq!(pfc.contents().len(), 0);
    }

    #[test]
    fn test_chunking() {
        let mut pfc = PrefrontalCortex::with_config(PrefrontalConfig {
            capacity: 4,
            max_chunk_size: 3,
            ..Default::default()
        });

        // Load 3 items
        let mut ids = Vec::new();
        for i in 0..3 {
            let signal = BrainSignal::new("test", SignalType::Memory, format!("chunk_item_{}", i));
            pfc.load(&signal);
            ids.push(pfc.contents().last().unwrap().id);
        }

        // Chunk them
        let chunk_id = pfc.chunk(&ids, "phone_number");
        assert!(chunk_id.is_some());

        // Effective count should be 1 (the chunk)
        assert_eq!(pfc.effective_item_count(), 1);

        // Can now add 3 more items
        for i in 0..3 {
            let signal = BrainSignal::new("test", SignalType::Memory, format!("new_item_{}", i));
            pfc.load(&signal);
        }

        assert_eq!(pfc.effective_item_count(), 4);
    }

    #[test]
    fn test_goals() {
        let mut pfc = PrefrontalCortex::new();

        let goal = Goal::new("Complete the task", 0.9);
        pfc.set_goal(goal);

        assert!(pfc.current_goal().is_some());
        assert!(!pfc.current_goal().unwrap().completed);

        let completed = pfc.complete_goal();
        assert!(completed.is_some());
        assert!(completed.unwrap().completed);
        assert!(pfc.current_goal().is_none());
    }

    #[test]
    fn test_stats() {
        let mut pfc = PrefrontalCortex::with_config(PrefrontalConfig {
            capacity: 5,
            ..Default::default()
        });

        for i in 0..3 {
            let signal = BrainSignal::new("test", SignalType::Memory, format!("item_{}", i));
            pfc.load(&signal);
        }

        let stats = pfc.stats();
        assert_eq!(stats.current_items, 3);
        assert_eq!(stats.capacity, 5);
        assert!((stats.utilization - 0.6).abs() < 0.01);
    }
}
