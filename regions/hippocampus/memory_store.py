"""
Hippocampus Memory Store

Core memory storage with valence-weighted retrieval.
"""

import sqlite3
from datetime import datetime
from typing import List, Optional, Any
import json
import uuid
import sys
from pathlib import Path

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from interfaces.signal import MemoryTrace, BrainSignal, SignalType


class HippocampusStore:
    """
    The hippocampus: where memories live.
    
    Key features:
    - Valence-weighted storage (emotional memories are prioritized)
    - Associative retrieval (pattern completion)
    - Decay with protection for high-valence memories
    - Consolidation interface for sleep cycles
    """
    
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._init_schema()
    
    def _init_schema(self):
        """Initialize the memory database."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                valence REAL DEFAULT 0.0,
                salience REAL DEFAULT 0.5,
                surprise REAL DEFAULT 0.0,
                created_at TEXT NOT NULL,
                last_accessed TEXT NOT NULL,
                access_count INTEGER DEFAULT 0,
                consolidated INTEGER DEFAULT 0,
                decay_rate REAL DEFAULT 0.1,
                strength REAL DEFAULT 1.0,
                associations TEXT DEFAULT '[]',
                context_tags TEXT DEFAULT '[]'
            );
            
            CREATE INDEX IF NOT EXISTS idx_valence ON memories(valence);
            CREATE INDEX IF NOT EXISTS idx_strength ON memories(strength);
            CREATE INDEX IF NOT EXISTS idx_consolidated ON memories(consolidated);
        """)
        self.conn.commit()
    
    def encode(self, signal: BrainSignal) -> MemoryTrace:
        """
        Encode a new memory from incoming signal.
        
        The amygdala's valence tag determines how strongly
        this memory will resist decay.
        """
        memory = MemoryTrace(
            id=str(uuid.uuid4()),
            content=signal.content,
            valence=signal.valence,
            salience=signal.salience,
            surprise=signal.metadata.get('prediction_error', 0.0),
            created_at=datetime.utcnow(),
            last_accessed=datetime.utcnow(),
        )
        
        # High surprise = lower decay rate (surprising things stick)
        if memory.surprise > 0.5:
            memory.decay_rate *= (1 - memory.surprise * 0.5)
        
        self._store(memory)
        return memory
    
    def retrieve(
        self, 
        query: Any = None,
        limit: int = 10,
        valence_boost: bool = True
    ) -> List[MemoryTrace]:
        """
        Retrieve memories matching query.
        
        If valence_boost is True, emotional memories are
        more likely to surface (like human memory).
        """
        cursor = self.conn.execute("""
            SELECT * FROM memories 
            WHERE strength > 0.1
            ORDER BY 
                CASE WHEN ? THEN ABS(valence) ELSE 0 END DESC,
                strength DESC,
                last_accessed DESC
            LIMIT ?
        """, (valence_boost, limit))
        
        memories = []
        for row in cursor:
            memory = self._row_to_memory(row)
            memory.access()  # Strengthen on retrieval
            self._update_access(memory)
            memories.append(memory)
        
        return memories
    
    def retrieve_by_valence(
        self,
        min_valence: float = -1.0,
        max_valence: float = 1.0,
        limit: int = 10
    ) -> List[MemoryTrace]:
        """Retrieve memories within a valence range."""
        cursor = self.conn.execute("""
            SELECT * FROM memories
            WHERE valence BETWEEN ? AND ?
            AND strength > 0.1
            ORDER BY ABS(valence) DESC, strength DESC
            LIMIT ?
        """, (min_valence, max_valence, limit))
        
        return [self._row_to_memory(row) for row in cursor]
    
    def get_unconsolidated(self, limit: int = 100) -> List[MemoryTrace]:
        """Get memories that need consolidation (for sleep cycle)."""
        cursor = self.conn.execute("""
            SELECT * FROM memories
            WHERE consolidated = 0
            ORDER BY salience DESC, created_at DESC
            LIMIT ?
        """, (limit,))
        
        return [self._row_to_memory(row) for row in cursor]
    
    def mark_consolidated(self, memory_ids: List[str]):
        """Mark memories as consolidated after sleep."""
        self.conn.executemany(
            "UPDATE memories SET consolidated = 1 WHERE id = ?",
            [(mid,) for mid in memory_ids]
        )
        self.conn.commit()
    
    def decay_all(self, hours_passed: float = 24):
        """
        Apply decay to all memories.
        
        Called periodically to simulate forgetting.
        High-valence memories decay slower.
        """
        cursor = self.conn.execute("SELECT * FROM memories WHERE strength > 0")
        
        for row in cursor:
            memory = self._row_to_memory(row)
            memory.decay(hours_passed)
            
            if memory.strength > 0.01:  # Don't bother updating near-zero
                self.conn.execute(
                    "UPDATE memories SET strength = ? WHERE id = ?",
                    (memory.strength, memory.id)
                )
        
        # Clean up truly forgotten memories
        self.conn.execute("DELETE FROM memories WHERE strength < 0.01")
        self.conn.commit()
    
    def _store(self, memory: MemoryTrace):
        """Store a memory trace."""
        self.conn.execute("""
            INSERT INTO memories (
                id, content, valence, salience, surprise,
                created_at, last_accessed, access_count,
                consolidated, decay_rate, strength,
                associations, context_tags
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            memory.id,
            json.dumps(memory.content) if not isinstance(memory.content, str) else memory.content,
            memory.valence,
            memory.salience,
            memory.surprise,
            memory.created_at.isoformat(),
            memory.last_accessed.isoformat(),
            memory.access_count,
            int(memory.consolidated),
            memory.decay_rate,
            memory.strength,
            json.dumps(memory.associations),
            json.dumps(memory.context_tags)
        ))
        self.conn.commit()
    
    def _update_access(self, memory: MemoryTrace):
        """Update access metadata."""
        self.conn.execute("""
            UPDATE memories 
            SET last_accessed = ?, access_count = ?, strength = ?
            WHERE id = ?
        """, (
            memory.last_accessed.isoformat(),
            memory.access_count,
            memory.strength,
            memory.id
        ))
        self.conn.commit()
    
    def _row_to_memory(self, row) -> MemoryTrace:
        """Convert database row to MemoryTrace."""
        return MemoryTrace(
            id=row[0],
            content=row[1],
            valence=row[2],
            salience=row[3],
            surprise=row[4],
            created_at=datetime.fromisoformat(row[5]),
            last_accessed=datetime.fromisoformat(row[6]),
            access_count=row[7],
            consolidated=bool(row[8]),
            decay_rate=row[9],
            strength=row[10],
            associations=json.loads(row[11]),
            context_tags=json.loads(row[12])
        )
    
    def stats(self) -> dict:
        """Get memory system statistics."""
        cursor = self.conn.execute("""
            SELECT 
                COUNT(*) as total,
                AVG(valence) as avg_valence,
                AVG(strength) as avg_strength,
                SUM(CASE WHEN consolidated = 1 THEN 1 ELSE 0 END) as consolidated,
                SUM(CASE WHEN valence > 0.5 THEN 1 ELSE 0 END) as positive,
                SUM(CASE WHEN valence < -0.5 THEN 1 ELSE 0 END) as negative
            FROM memories
        """)
        row = cursor.fetchone()
        return {
            'total_memories': row[0],
            'avg_valence': row[1],
            'avg_strength': row[2],
            'consolidated': row[3],
            'positive_memories': row[4],
            'negative_memories': row[5]
        }


if __name__ == "__main__":
    # Quick test
    store = HippocampusStore()
    
    # Encode a positive memory
    signal = BrainSignal(
        source="test",
        signal_type=SignalType.SENSORY,
        content="First successful memory encoding!",
        valence=0.8,
        salience=0.9,
        metadata={'prediction_error': 0.7}
    )
    
    memory = store.encode(signal)
    print(f"Encoded: {memory.id}")
    print(f"Stats: {store.stats()}")
    
    # Retrieve
    memories = store.retrieve(limit=5)
    print(f"Retrieved {len(memories)} memories")
