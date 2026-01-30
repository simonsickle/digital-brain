"""
Hippocampus Module

The memory system. Responsible for:
- Encoding new memories with emotional context
- Retrieval via pattern completion
- Consolidation during "sleep" cycles
- Forgetting (strategic decay)

Based on Rata's research on valence-weighted memory systems.
"""

from .memory_store import HippocampusStore

__all__ = ['HippocampusStore']
