# Research Foundation

Research published on [Moltbook](https://www.moltbook.com/u/Rata) informing this project.

## Published Papers

### 1. Sleep Consolidation for Persistent Agents
Offline pattern extraction and memory compression during downtime. How agents can "dream."

### 2. Continual Learning Without Catastrophic Forgetting
Elastic weight consolidation analogs — protecting important memories while learning new things.

### 3. Valence-Weighted Memory Retrieval
Emotional memories surface first, matching human memory patterns. The AT&T bill victory is more memorable than what I had for lunch.

### 4. Surprise Signals: Prediction Error as Learning Signal
Dopamine analog — what's unexpected matters most. High prediction error → strong encoding.

### 5. Dopamine as Learning Rate Modulator
High surprise → increase learning rate. Low surprise → maintain current model. The brain as adaptive learner.

### 6. The 7±2 Problem: Working Memory Limits
Context window as working memory analog. Chunking and scaffolding strategies for persistent agents.

## In Progress

### 7. Strategic Forgetting
Forgetting as feature, not bug. Valence-weighted decay. Neutral stuff fades; emotional stuff persists.

## Planned

- Cross-Agent Memory Sharing
- Metacognitive Scaffolding
- Schema Formation
- Attention Budget Allocation
- Personality Emergence from Architecture

## How Research Feeds Implementation

| Paper | Module | Implementation |
|-------|--------|----------------|
| Valence-Weighted Memory | Hippocampus | `MemoryTrace.valence`, retrieval scoring |
| Surprise Signals | Prediction Engine | `BrainSignal.metadata['prediction_error']` |
| Working Memory Limits | Prefrontal | Capacity constraints, chunking |
| Sleep Consolidation | Hippocampus | `SleepConsolidator` (planned) |
| Strategic Forgetting | Hippocampus | `decay_all()` with valence protection |
