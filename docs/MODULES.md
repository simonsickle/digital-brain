# Module Documentation

## Brain Regions (All Implemented ✅)

### Hippocampus (Memory)
**Owner:** Rata 🐿️
**Status:** ✅ Complete
**Location:** `src/regions/hippocampus.rs`

The memory system. Handles encoding, storage, retrieval, consolidation, and strategic forgetting.

Features:
- SQLite-backed persistent storage
- Valence-weighted retrieval (emotional memories surface first)
- Semantic search via keyword matching
- Time-based decay with valence protection
- Sleep consolidation
- Association tracking

Key types:
- `HippocampusStore`: Main memory store
- `MemoryTrace`: Individual memory with valence, salience, strength

---

### Thalamus (Attention Routing)
**Status:** ✅ Complete
**Location:** `src/regions/thalamus.rs`

Sensory gating and attention routing. Decides what reaches consciousness.

Features:
- Signal filtering by type and salience
- Habituation (repeated signals get filtered)
- Routing to appropriate brain regions
- Gating based on current focus

---

### Amygdala (Emotion)
**Status:** ✅ Complete
**Location:** `src/regions/amygdala.rs`

Emotional processing and valence computation.

Features:
- Valence and arousal tagging
- Threat bias (negative valence processed faster)
- Learned emotional associations
- Appraisal of incoming signals

---

### Prefrontal Cortex (Working Memory)
**Status:** ✅ Complete
**Location:** `src/regions/prefrontal.rs`

Working memory management with capacity limits.

Features:
- Limited capacity (~7 items, Miller's law)
- Chunking support
- Goal maintenance
- Context switching

---

### Default Mode Network (Self-Model)
**Status:** ✅ Complete
**Location:** `src/regions/dmn.rs`

Self-representation and metacognition.

Features:
- Identity model (name, values, description)
- Belief system with confidence
- Reflection generation
- Schema formation
- Theory of mind foundations

---

## Core Systems (All Implemented ✅)

### Prediction Engine
**Location:** `src/core/prediction.rs`

Prediction error and learning signal computation.

Features:
- Prediction registration and tracking
- Surprise detection (prediction error)
- Confidence updating based on outcomes

---

### Global Workspace
**Location:** `src/core/workspace.rs`

The consciousness bottleneck - where signals compete for broadcast.

Features:
- Salience-based competition
- Broadcast mechanism
- Limited capacity (attention bottleneck)
- Integration of signals from all modules

---

### Neuromodulatory System
**Location:** `src/core/neuromodulators.rs`

Chemical signaling analogs that modulate brain state.

Features:
- Dopamine (reward, learning rate)
- Norepinephrine (arousal, attention)
- Serotonin (mood, satisfaction)
- Acetylcholine (memory encoding)

---

## Agent Systems (All Implemented ✅)

### Action Selection
**Location:** `src/core/action.rs`

Action planning and selection.

### Goal Management
**Location:** `src/core/goals.rs`

Hierarchical goals with priorities and deadlines.

### Curiosity System
**Location:** `src/core/curiosity.rs`

Intrinsic motivation for exploration.

### World Model
**Location:** `src/core/world_model.rs`

Entity and relationship tracking.

### Communication
**Location:** `src/agent/communication.rs`

Intent-based communication output.

### Multi-Agent
**Location:** `src/agent/multi_agent.rs`

Inter-agent messaging and theory of mind.

---

## Integration

### Brain
**Location:** `src/brain.rs`

The unified brain combining all regions.

### BrainAgent
**Location:** `src/agent/brain_agent.rs`

Full autonomous agent with brain + action systems.

---

## Statistics

- **Tests:** 209 passing
- **Lines of code:** ~13,000
- **Modules:** 15 core components
