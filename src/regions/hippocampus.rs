//! Hippocampus - The Memory System
//!
//! Responsible for:
//! - Encoding new memories with emotional context
//! - Retrieval via pattern matching
//! - Consolidation during "sleep" cycles
//! - Strategic forgetting (valence-weighted decay)

use crate::error::{BrainError, Result};
#[allow(unused_imports)]
use crate::signal::{BrainSignal, MemoryTrace, SignalType};
use chrono::{DateTime, Utc};
use rusqlite::{Connection, params};
use serde_json;
use uuid::Uuid;

/// The hippocampus: where memories live.
///
/// Key features:
/// - Valence-weighted storage (emotional memories are prioritized)
/// - Associative retrieval (pattern completion)
/// - Decay with protection for high-valence memories
/// - Consolidation interface for sleep cycles
pub struct HippocampusStore {
    conn: Connection,
}

impl HippocampusStore {
    /// Create a new hippocampus with in-memory storage.
    pub fn new_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        let store = Self { conn };
        store.init_schema()?;
        Ok(store)
    }

    /// Create a new hippocampus with persistent storage.
    pub fn new(db_path: &str) -> Result<Self> {
        let conn = Connection::open(db_path)?;
        let store = Self { conn };
        store.init_schema()?;
        Ok(store)
    }

    /// Initialize the database schema.
    fn init_schema(&self) -> Result<()> {
        self.conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                valence REAL NOT NULL DEFAULT 0.0,
                salience REAL NOT NULL DEFAULT 0.5,
                surprise REAL NOT NULL DEFAULT 0.0,
                created_at TEXT NOT NULL,
                last_accessed TEXT NOT NULL,
                access_count INTEGER NOT NULL DEFAULT 0,
                consolidated INTEGER NOT NULL DEFAULT 0,
                decay_rate REAL NOT NULL DEFAULT 0.1,
                strength REAL NOT NULL DEFAULT 1.0,
                associations TEXT NOT NULL DEFAULT '[]',
                context_tags TEXT NOT NULL DEFAULT '[]'
            );

            CREATE INDEX IF NOT EXISTS idx_memories_valence ON memories(valence);
            CREATE INDEX IF NOT EXISTS idx_memories_strength ON memories(strength);
            CREATE INDEX IF NOT EXISTS idx_memories_consolidated ON memories(consolidated);
            CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
            ",
        )?;
        Ok(())
    }

    /// Encode a new memory from an incoming signal.
    pub fn encode(&self, signal: &BrainSignal) -> Result<MemoryTrace> {
        let memory = MemoryTrace::from_signal(signal);
        self.store(&memory)?;
        Ok(memory)
    }

    /// Store a memory trace.
    fn store(&self, memory: &MemoryTrace) -> Result<()> {
        self.conn.execute(
            "INSERT INTO memories (
                id, content, valence, salience, surprise,
                created_at, last_accessed, access_count,
                consolidated, decay_rate, strength,
                associations, context_tags
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
            params![
                memory.id.to_string(),
                serde_json::to_string(&memory.content)?,
                memory.valence.value(),
                memory.salience.value(),
                memory.surprise,
                memory.created_at.to_rfc3339(),
                memory.last_accessed.to_rfc3339(),
                memory.access_count,
                memory.consolidated as i32,
                memory.decay_rate,
                memory.strength,
                serde_json::to_string(&memory.associations)?,
                serde_json::to_string(&memory.context_tags)?,
            ],
        )?;
        Ok(())
    }

    /// Retrieve memories, optionally boosting emotional ones.
    pub fn retrieve(&self, limit: usize, valence_boost: bool) -> Result<Vec<MemoryTrace>> {
        let sql = if valence_boost {
            "SELECT * FROM memories 
             WHERE strength > 0.1
             ORDER BY ABS(valence) DESC, strength DESC, last_accessed DESC
             LIMIT ?1"
        } else {
            "SELECT * FROM memories 
             WHERE strength > 0.1
             ORDER BY strength DESC, last_accessed DESC
             LIMIT ?1"
        };

        let mut stmt = self.conn.prepare(sql)?;
        let memories = stmt
            .query_map(params![limit as i64], |row| self.row_to_memory(row))?
            .filter_map(|r| r.ok())
            .collect::<Vec<_>>();

        // Record access for each retrieved memory
        for memory in &memories {
            self.update_access(memory)?;
        }

        Ok(memories)
    }

    /// Retrieve memories by valence range.
    pub fn retrieve_by_valence(
        &self,
        min_valence: f64,
        max_valence: f64,
        limit: usize,
    ) -> Result<Vec<MemoryTrace>> {
        let mut stmt = self.conn.prepare(
            "SELECT * FROM memories
             WHERE valence BETWEEN ?1 AND ?2
             AND strength > 0.1
             ORDER BY ABS(valence) DESC, strength DESC
             LIMIT ?3",
        )?;

        let memories = stmt
            .query_map(params![min_valence, max_valence, limit as i64], |row| {
                self.row_to_memory(row)
            })?
            .filter_map(|r| r.ok())
            .collect();

        Ok(memories)
    }

    /// Retrieve memories by context tag.
    pub fn retrieve_by_tag(&self, tag: &str, limit: usize) -> Result<Vec<MemoryTrace>> {
        let pattern = format!("%\"{}\",%", tag);
        let mut stmt = self.conn.prepare(
            "SELECT * FROM memories
             WHERE context_tags LIKE ?1
             AND strength > 0.1
             ORDER BY strength DESC
             LIMIT ?2",
        )?;

        let memories = stmt
            .query_map(params![pattern, limit as i64], |row| {
                self.row_to_memory(row)
            })?
            .filter_map(|r| r.ok())
            .collect();

        Ok(memories)
    }

    /// Get a specific memory by ID.
    pub fn get(&self, id: Uuid) -> Result<MemoryTrace> {
        let mut stmt = self.conn.prepare("SELECT * FROM memories WHERE id = ?1")?;

        stmt.query_row(params![id.to_string()], |row| self.row_to_memory(row))
            .map_err(|_| BrainError::MemoryNotFound(id.to_string()))
    }

    /// Get unconsolidated memories (for sleep cycle).
    pub fn get_unconsolidated(&self, limit: usize) -> Result<Vec<MemoryTrace>> {
        let mut stmt = self.conn.prepare(
            "SELECT * FROM memories
             WHERE consolidated = 0
             ORDER BY salience DESC, created_at DESC
             LIMIT ?1",
        )?;

        let memories = stmt
            .query_map(params![limit as i64], |row| self.row_to_memory(row))?
            .filter_map(|r| r.ok())
            .collect();

        Ok(memories)
    }

    /// Mark memories as consolidated after sleep.
    pub fn mark_consolidated(&self, memory_ids: &[Uuid]) -> Result<()> {
        for id in memory_ids {
            self.conn.execute(
                "UPDATE memories SET consolidated = 1 WHERE id = ?1",
                params![id.to_string()],
            )?;
        }
        Ok(())
    }

    /// Apply decay to all memories.
    ///
    /// Called periodically to simulate forgetting.
    /// High-valence memories decay slower.
    pub fn decay_all(&self, hours_passed: f64) -> Result<usize> {
        // Get all memories
        let mut stmt = self
            .conn
            .prepare("SELECT * FROM memories WHERE strength > 0")?;

        let memories: Vec<MemoryTrace> = stmt
            .query_map([], |row| self.row_to_memory(row))?
            .filter_map(|r| r.ok())
            .collect();

        let mut decayed_count = 0;

        for mut memory in memories {
            memory.decay(hours_passed);

            if memory.is_retrievable() {
                self.conn.execute(
                    "UPDATE memories SET strength = ?1 WHERE id = ?2",
                    params![memory.strength, memory.id.to_string()],
                )?;
            } else {
                // Remove truly forgotten memories
                self.conn.execute(
                    "DELETE FROM memories WHERE id = ?1",
                    params![memory.id.to_string()],
                )?;
                decayed_count += 1;
            }
        }

        Ok(decayed_count)
    }

    /// Update access metadata after retrieval.
    fn update_access(&self, memory: &MemoryTrace) -> Result<()> {
        self.conn.execute(
            "UPDATE memories 
             SET last_accessed = ?1, access_count = access_count + 1, strength = MIN(1.0, strength + 0.1)
             WHERE id = ?2",
            params![Utc::now().to_rfc3339(), memory.id.to_string()],
        )?;
        Ok(())
    }

    /// Convert a database row to a MemoryTrace.
    fn row_to_memory(&self, row: &rusqlite::Row) -> rusqlite::Result<MemoryTrace> {
        let id_str: String = row.get(0)?;
        let content_str: String = row.get(1)?;
        let created_str: String = row.get(5)?;
        let accessed_str: String = row.get(6)?;
        let associations_str: String = row.get(11)?;
        let tags_str: String = row.get(12)?;

        Ok(MemoryTrace {
            id: Uuid::parse_str(&id_str).unwrap_or_else(|_| Uuid::new_v4()),
            content: serde_json::from_str(&content_str).unwrap_or(serde_json::Value::Null),
            valence: crate::signal::Valence::new(row.get(2)?),
            salience: crate::signal::Salience::new(row.get(3)?),
            surprise: row.get(4)?,
            created_at: DateTime::parse_from_rfc3339(&created_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
            last_accessed: DateTime::parse_from_rfc3339(&accessed_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
            access_count: row.get(7)?,
            consolidated: row.get::<_, i32>(8)? != 0,
            decay_rate: row.get(9)?,
            strength: row.get(10)?,
            associations: serde_json::from_str(&associations_str).unwrap_or_default(),
            context_tags: serde_json::from_str(&tags_str).unwrap_or_default(),
            // Epistemic fields - defaults for backward compatibility
            confidence: 0.7, // Default moderate confidence for existing memories
            source: crate::signal::MemorySource::Unknown,
            verified: false,
            contradictions: Vec::new(),
        })
    }

    /// Get statistics about the memory system.
    pub fn stats(&self) -> Result<MemoryStats> {
        let mut stmt = self.conn.prepare(
            "SELECT 
                COUNT(*) as total,
                AVG(valence) as avg_valence,
                AVG(strength) as avg_strength,
                SUM(CASE WHEN consolidated = 1 THEN 1 ELSE 0 END) as consolidated,
                SUM(CASE WHEN valence > 0.5 THEN 1 ELSE 0 END) as positive,
                SUM(CASE WHEN valence < -0.5 THEN 1 ELSE 0 END) as negative
            FROM memories",
        )?;

        let stats = stmt.query_row([], |row| {
            Ok(MemoryStats {
                total_memories: row.get(0)?,
                avg_valence: row.get::<_, Option<f64>>(1)?.unwrap_or(0.0),
                avg_strength: row.get::<_, Option<f64>>(2)?.unwrap_or(0.0),
                consolidated: row.get(3)?,
                positive_memories: row.get(4)?,
                negative_memories: row.get(5)?,
            })
        })?;

        Ok(stats)
    }
}

/// Statistics about the memory system.
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub total_memories: i64,
    pub avg_valence: f64,
    pub avg_strength: f64,
    pub consolidated: i64,
    pub positive_memories: i64,
    pub negative_memories: i64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_and_retrieve() -> Result<()> {
        let store = HippocampusStore::new_in_memory()?;

        // Encode a positive memory
        let signal = BrainSignal::new("test", SignalType::Sensory, "happy memory")
            .with_valence(0.8)
            .with_salience(0.9);

        let memory = store.encode(&signal)?;
        assert!(memory.strength > 0.9);

        // Retrieve it
        let memories = store.retrieve(10, true)?;
        assert_eq!(memories.len(), 1);
        assert_eq!(memories[0].id, memory.id);

        Ok(())
    }

    #[test]
    fn test_emotional_memories_surface_first() -> Result<()> {
        let store = HippocampusStore::new_in_memory()?;

        // Encode neutral memory first
        let neutral = BrainSignal::new("test", SignalType::Memory, "neutral").with_valence(0.0);
        store.encode(&neutral)?;

        // Encode emotional memory second
        let emotional = BrainSignal::new("test", SignalType::Memory, "emotional").with_valence(0.9);
        let emotional_memory = store.encode(&emotional)?;

        // Retrieve with valence boost
        let memories = store.retrieve(10, true)?;
        assert_eq!(memories[0].id, emotional_memory.id);

        Ok(())
    }

    #[test]
    fn test_stats() -> Result<()> {
        let store = HippocampusStore::new_in_memory()?;

        for i in 0..5 {
            let valence = if i % 2 == 0 { 0.8 } else { -0.8 };
            let signal = BrainSignal::new("test", SignalType::Memory, format!("memory {}", i))
                .with_valence(valence);
            store.encode(&signal)?;
        }

        let stats = store.stats()?;
        assert_eq!(stats.total_memories, 5);
        assert_eq!(stats.positive_memories, 3);
        assert_eq!(stats.negative_memories, 2);

        Ok(())
    }
}
