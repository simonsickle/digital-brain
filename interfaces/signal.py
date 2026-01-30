"""
Brain Signal Protocol

The universal message format for inter-module communication.
All brain regions speak this language.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from enum import Enum


class SignalType(Enum):
    """Categories of brain signals."""
    SENSORY = "sensory"           # Raw input from environment
    MEMORY = "memory"             # Retrieved or encoded memory
    PREDICTION = "prediction"     # Expected state
    ERROR = "error"               # Prediction error (surprise)
    EMOTION = "emotion"           # Valence/arousal tag
    ATTENTION = "attention"       # Salience marker
    BROADCAST = "broadcast"       # Global workspace broadcast
    QUERY = "query"               # Request for information
    MOTOR = "motor"               # Action intention


@dataclass
class BrainSignal:
    """
    Universal signal format for brain module communication.
    
    Design principles:
    - Self-describing: carries its own metadata
    - Valenced: emotional coloring is first-class
    - Salient: importance is explicit, enables competition
    - Traceable: source and timestamp for debugging/analysis
    """
    
    # Required fields
    source: str                   # Module ID that generated this
    signal_type: SignalType       # What kind of signal
    content: Any                  # The actual payload
    
    # Emotional/attention markers
    salience: float = 0.5         # 0-1, how attention-grabbing
    valence: float = 0.0          # -1 to 1, negative to positive
    arousal: float = 0.5          # 0-1, calm to excited
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    confidence: float = 1.0       # 0-1, how certain
    metadata: dict = field(default_factory=dict)
    
    # Routing
    target: Optional[str] = None  # Specific target, or None for broadcast
    priority: int = 0             # Higher = process first
    
    def __post_init__(self):
        """Validate signal values."""
        assert 0 <= self.salience <= 1, "Salience must be 0-1"
        assert -1 <= self.valence <= 1, "Valence must be -1 to 1"
        assert 0 <= self.arousal <= 1, "Arousal must be 0-1"
        assert 0 <= self.confidence <= 1, "Confidence must be 0-1"
    
    @property
    def is_surprising(self) -> bool:
        """High salience + high arousal = surprising."""
        return self.salience > 0.7 and self.arousal > 0.7
    
    @property
    def emotional_intensity(self) -> float:
        """Combined emotional weight."""
        return abs(self.valence) * self.arousal
    
    def escalate(self, boost: float = 0.2) -> 'BrainSignal':
        """Return copy with increased salience (for attention competition)."""
        return BrainSignal(
            source=self.source,
            signal_type=self.signal_type,
            content=self.content,
            salience=min(1.0, self.salience + boost),
            valence=self.valence,
            arousal=min(1.0, self.arousal + boost/2),
            timestamp=self.timestamp,
            confidence=self.confidence,
            metadata={**self.metadata, 'escalated': True},
            target=self.target,
            priority=self.priority + 1
        )


@dataclass  
class MemoryTrace:
    """
    A memory as stored in the hippocampus.
    
    Memories are not static recordings — they're patterns
    that get reconstructed, consolidated, and can decay.
    """
    
    id: str                       # Unique identifier
    content: Any                  # The memory content
    
    # Valence scoring (Rata's research)
    valence: float = 0.0          # Emotional weight
    salience: float = 0.5         # How important when formed
    surprise: float = 0.0         # Prediction error when encoded
    
    # Temporal
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    
    # Consolidation
    consolidated: bool = False    # Has this been through "sleep"?
    decay_rate: float = 0.1       # How fast this fades (valence-weighted)
    strength: float = 1.0         # Current retrieval strength
    
    # Connections
    associations: list = field(default_factory=list)  # Linked memory IDs
    context_tags: list = field(default_factory=list)  # Retrieval cues
    
    def access(self) -> 'MemoryTrace':
        """Record an access, strengthening the memory."""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1
        self.strength = min(1.0, self.strength + 0.1)
        return self
    
    def decay(self, time_delta_hours: float) -> 'MemoryTrace':
        """
        Apply time-based decay.
        
        Valence-weighted: emotional memories decay slower.
        """
        effective_rate = self.decay_rate * (1 - abs(self.valence) * 0.5)
        decay_amount = effective_rate * (time_delta_hours / 24)
        self.strength = max(0.0, self.strength - decay_amount)
        return self
    
    @property
    def retrieval_score(self) -> float:
        """
        How likely this memory is to be retrieved.
        
        Combines: strength, valence, recency, access frequency
        """
        recency_boost = 1.0 / (1 + (datetime.utcnow() - self.last_accessed).days)
        frequency_boost = min(1.0, self.access_count / 10)
        valence_boost = abs(self.valence)
        
        return (
            self.strength * 0.4 +
            recency_boost * 0.2 +
            frequency_boost * 0.2 +
            valence_boost * 0.2
        )
