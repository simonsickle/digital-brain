//! Schema System - Pattern Abstraction from Episodes
//!
//! Schemas are generalized knowledge structures abstracted from specific episodes.
//! They represent "when X happens, Y usually follows" patterns.
//!
//! Based on Moltbook Research Paper 10: Schema Formation
//!
//! # The Abstraction Ladder
//!
//! ```text
//! PRINCIPLES    "Always validate before executing"
//!     ↑
//! PROCEDURES    "To deploy: test → build → push → verify"
//!     ↑
//! PATTERNS      "Tests failing after refactor → check imports"
//!     ↑
//! EPISODES      "2024-01-15: Tests failed, was missing import"
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A schema represents abstracted knowledge from multiple episodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Schema {
    /// Unique identifier
    pub id: u64,
    /// The generalization (e.g., "When tests fail after refactor, check imports")
    pub pattern: String,
    /// Category of schema
    pub category: SchemaCategory,
    /// Confidence in this schema (0.0 to 1.0)
    pub confidence: f64,
    /// Number of supporting episodes
    pub support_count: usize,
    /// Number of contradicting episodes
    pub contradiction_count: usize,
    /// IDs of supporting episode memories
    pub supporting_episodes: Vec<u64>,
    /// IDs of contradicting episodes
    pub contradicting_episodes: Vec<u64>,
    /// When this schema was formed
    pub created_at: DateTime<Utc>,
    /// When this schema was last updated
    pub updated_at: DateTime<Utc>,
    /// How many times this schema has been retrieved/used
    pub retrieval_count: u64,
    /// Keywords/triggers that activate this schema
    pub triggers: Vec<String>,
    /// Invalidation conditions - when should this schema be revised?
    pub invalidation_conditions: Vec<String>,
    /// Level of abstraction (1=pattern, 2=procedure, 3=principle)
    pub abstraction_level: u8,
}

impl Schema {
    pub fn new(pattern: &str, category: SchemaCategory) -> Self {
        Self {
            id: 0,
            pattern: pattern.to_string(),
            category,
            confidence: 0.5, // Start with moderate confidence
            support_count: 0,
            contradiction_count: 0,
            supporting_episodes: Vec::new(),
            contradicting_episodes: Vec::new(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            retrieval_count: 0,
            triggers: Vec::new(),
            invalidation_conditions: Vec::new(),
            abstraction_level: 1,
        }
    }

    /// Add a supporting episode to this schema
    pub fn add_support(&mut self, episode_id: u64) {
        self.supporting_episodes.push(episode_id);
        self.support_count += 1;
        self.update_confidence();
        self.updated_at = Utc::now();
    }

    /// Add a contradicting episode to this schema
    pub fn add_contradiction(&mut self, episode_id: u64) {
        self.contradicting_episodes.push(episode_id);
        self.contradiction_count += 1;
        self.update_confidence();
        self.updated_at = Utc::now();
    }

    /// Update confidence based on support/contradiction ratio
    fn update_confidence(&mut self) {
        let total = self.support_count + self.contradiction_count;
        if total > 0 {
            self.confidence = self.support_count as f64 / total as f64;
        }
    }

    /// Check if this schema should be revised (too many contradictions)
    pub fn needs_revision(&self) -> bool {
        // Needs revision if contradiction rate exceeds 30%
        let total = self.support_count + self.contradiction_count;
        if total >= 5 {
            let contradiction_rate = self.contradiction_count as f64 / total as f64;
            contradiction_rate > 0.3
        } else {
            false
        }
    }

    /// Add a trigger keyword
    pub fn add_trigger(&mut self, trigger: &str) {
        if !self.triggers.contains(&trigger.to_string()) {
            self.triggers.push(trigger.to_string());
        }
    }

    /// Add an invalidation condition
    pub fn add_invalidation_condition(&mut self, condition: &str) {
        if !self
            .invalidation_conditions
            .contains(&condition.to_string())
        {
            self.invalidation_conditions.push(condition.to_string());
        }
    }

    /// Record a retrieval of this schema
    pub fn record_retrieval(&mut self) {
        self.retrieval_count += 1;
    }
}

/// Categories of schemas
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SchemaCategory {
    /// Causal patterns: X causes Y
    Causal,
    /// Temporal patterns: X happens before Y
    Temporal,
    /// Conditional patterns: If X then Y
    Conditional,
    /// Procedural: How to do something
    Procedural,
    /// Social: How interactions work
    Social,
    /// Technical: Domain-specific patterns
    Technical,
    /// Emotional: Emotional response patterns
    Emotional,
    /// Self: Patterns about self
    SelfModel,
}

/// The schema store manages all learned schemas
pub struct SchemaStore {
    schemas: HashMap<u64, Schema>,
    next_id: u64,
    /// Index of triggers to schema IDs
    trigger_index: HashMap<String, Vec<u64>>,
    /// Recently activated schemas (for competition/priming)
    recent_activations: Vec<u64>,
    max_recent: usize,
}

impl SchemaStore {
    pub fn new() -> Self {
        Self {
            schemas: HashMap::new(),
            next_id: 1,
            trigger_index: HashMap::new(),
            recent_activations: Vec::new(),
            max_recent: 10,
        }
    }

    /// Create a new schema
    pub fn create(&mut self, pattern: &str, category: SchemaCategory) -> u64 {
        let mut schema = Schema::new(pattern, category);
        schema.id = self.next_id;
        self.next_id += 1;

        let id = schema.id;
        self.schemas.insert(id, schema);
        id
    }

    /// Create a schema with initial support from an episode
    pub fn create_from_episode(
        &mut self,
        pattern: &str,
        category: SchemaCategory,
        episode_id: u64,
        triggers: Vec<&str>,
    ) -> u64 {
        let id = self.create(pattern, category);
        let mut triggers_to_index = Vec::new();
        if let Some(schema) = self.schemas.get_mut(&id) {
            schema.add_support(episode_id);
            for trigger in triggers {
                schema.add_trigger(trigger);
                triggers_to_index.push(trigger.to_string());
            }
        }
        for trigger in triggers_to_index {
            self.index_trigger(&trigger, id);
        }
        id
    }

    /// Find a schema ID by exact pattern match.
    pub fn find_id_by_pattern(&self, pattern: &str) -> Option<u64> {
        self.schemas
            .iter()
            .find_map(|(id, schema)| (schema.pattern == pattern).then_some(*id))
    }

    /// Upsert a schema from a pattern and triggers.
    pub fn upsert_pattern(
        &mut self,
        pattern: &str,
        category: SchemaCategory,
        episode_id: u64,
        triggers: Vec<String>,
    ) -> u64 {
        if let Some(id) = self.find_id_by_pattern(pattern) {
            let mut triggers_to_index = Vec::new();
            if let Some(schema) = self.schemas.get_mut(&id) {
                schema.add_support(episode_id);
                for trigger in triggers {
                    let was_present = schema.triggers.contains(&trigger);
                    schema.add_trigger(&trigger);
                    if !was_present {
                        triggers_to_index.push(trigger);
                    }
                }
            }
            for trigger in triggers_to_index {
                self.index_trigger(&trigger, id);
            }
            return id;
        }

        let id = self.create(pattern, category);
        let mut triggers_to_index = Vec::new();
        if let Some(schema) = self.schemas.get_mut(&id) {
            schema.add_support(episode_id);
            for trigger in triggers {
                schema.add_trigger(&trigger);
                triggers_to_index.push(trigger);
            }
        }
        for trigger in triggers_to_index {
            self.index_trigger(&trigger, id);
        }
        id
    }

    /// Find schemas matching a query
    pub fn find(&mut self, query: &str) -> Vec<&Schema> {
        let query_lower = query.to_lowercase();
        let words: Vec<&str> = query_lower.split_whitespace().collect();

        let mut matching_ids: Vec<u64> = Vec::new();

        // Find schemas with matching triggers
        for word in &words {
            if let Some(ids) = self.trigger_index.get(*word) {
                for id in ids {
                    if !matching_ids.contains(id) {
                        matching_ids.push(*id);
                    }
                }
            }
        }

        // Also search schema patterns directly
        for (id, schema) in &self.schemas {
            if schema.pattern.to_lowercase().contains(&query_lower) && !matching_ids.contains(id) {
                matching_ids.push(*id);
            }
        }

        // Record activations and return schemas
        for id in &matching_ids {
            self.record_activation(*id);
        }

        matching_ids
            .iter()
            .filter_map(|id| self.schemas.get(id))
            .collect()
    }

    /// Find high-confidence schemas
    pub fn find_confident(&self, min_confidence: f64) -> Vec<&Schema> {
        self.schemas
            .values()
            .filter(|s| s.confidence >= min_confidence)
            .collect()
    }

    /// Find schemas that need revision
    pub fn find_needs_revision(&self) -> Vec<&Schema> {
        self.schemas
            .values()
            .filter(|s| s.needs_revision())
            .collect()
    }

    /// Get a schema by ID
    pub fn get(&self, id: u64) -> Option<&Schema> {
        self.schemas.get(&id)
    }

    /// Get a mutable schema by ID
    pub fn get_mut(&mut self, id: u64) -> Option<&mut Schema> {
        self.schemas.get_mut(&id)
    }

    /// Record that a schema was activated (for priming)
    fn record_activation(&mut self, id: u64) {
        // Remove if already present
        self.recent_activations.retain(|&x| x != id);
        // Add to front
        self.recent_activations.insert(0, id);
        // Trim to max
        if self.recent_activations.len() > self.max_recent {
            self.recent_activations.pop();
        }
        // Record retrieval
        if let Some(schema) = self.schemas.get_mut(&id) {
            schema.record_retrieval();
        }
    }

    fn index_trigger(&mut self, trigger: &str, id: u64) {
        let entry = self
            .trigger_index
            .entry(trigger.to_lowercase())
            .or_default();
        if !entry.contains(&id) {
            entry.push(id);
        }
    }

    /// Get recently activated schemas (priming effect)
    pub fn recently_activated(&self) -> Vec<&Schema> {
        self.recent_activations
            .iter()
            .filter_map(|id| self.schemas.get(id))
            .collect()
    }

    /// Add support to a schema
    pub fn add_support(&mut self, schema_id: u64, episode_id: u64) {
        if let Some(schema) = self.schemas.get_mut(&schema_id) {
            schema.add_support(episode_id);
        }
    }

    /// Add contradiction to a schema
    pub fn add_contradiction(&mut self, schema_id: u64, episode_id: u64) {
        if let Some(schema) = self.schemas.get_mut(&schema_id) {
            schema.add_contradiction(episode_id);
        }
    }

    /// Get statistics about the schema store
    pub fn stats(&self) -> SchemaStats {
        let total = self.schemas.len();
        let high_confidence = self
            .schemas
            .values()
            .filter(|s| s.confidence >= 0.8)
            .count();
        let needs_revision = self.schemas.values().filter(|s| s.needs_revision()).count();
        let avg_confidence = if total > 0 {
            self.schemas.values().map(|s| s.confidence).sum::<f64>() / total as f64
        } else {
            0.0
        };

        let by_category: HashMap<SchemaCategory, usize> = {
            let mut map = HashMap::new();
            for schema in self.schemas.values() {
                *map.entry(schema.category).or_insert(0) += 1;
            }
            map
        };

        SchemaStats {
            total_schemas: total,
            high_confidence_count: high_confidence,
            needs_revision_count: needs_revision,
            average_confidence: avg_confidence,
            by_category,
        }
    }
}

impl Default for SchemaStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about the schema store
#[derive(Debug, Clone)]
pub struct SchemaStats {
    pub total_schemas: usize,
    pub high_confidence_count: usize,
    pub needs_revision_count: usize,
    pub average_confidence: f64,
    pub by_category: HashMap<SchemaCategory, usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_creation() {
        let schema = Schema::new("When tests fail, check imports", SchemaCategory::Causal);
        assert_eq!(schema.confidence, 0.5);
        assert_eq!(schema.support_count, 0);
    }

    #[test]
    fn test_support_increases_confidence() {
        let mut schema = Schema::new("Pattern", SchemaCategory::Causal);
        schema.add_support(1);
        schema.add_support(2);
        schema.add_support(3);
        assert!(schema.confidence > 0.5);
    }

    #[test]
    fn test_contradiction_decreases_confidence() {
        let mut schema = Schema::new("Pattern", SchemaCategory::Causal);
        schema.add_support(1);
        schema.add_contradiction(2);
        assert_eq!(schema.confidence, 0.5);
    }

    #[test]
    fn test_needs_revision() {
        let mut schema = Schema::new("Pattern", SchemaCategory::Causal);
        for i in 0..3 {
            schema.add_support(i);
        }
        for i in 3..6 {
            schema.add_contradiction(i);
        }
        assert!(schema.needs_revision());
    }

    #[test]
    fn test_schema_store() {
        let mut store = SchemaStore::new();
        let id = store.create_from_episode(
            "When tests fail after refactor, check imports",
            SchemaCategory::Causal,
            1,
            vec!["tests", "fail", "refactor", "imports"],
        );

        let results = store.find("tests fail");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, id);
    }
}
