# Human Cognition Simulation Roadmap

## Goal
Create a cognitive architecture indistinguishable from human cognition.

## What Makes Humans Human?

### 1. Embodiment & Interoception ⚠️ (iterating)
- **Current coverage:** `src/regions/hypothalamus.rs` models homeostatic drives and circadian rhythms; `src/regions/insula.rs` tracks body state, interoception, and affective mirroring; sensory cortices (`src/regions/sensory_cortex.rs`) transform thalamic input into modality-specific features.
- **Next focus:** Feed autonomic feedback into attention/salience systems and surface bodily alerts through the global workspace.

### 2. Rich Emotional Life ⚠️ (iterating)
- **Current coverage:** Discrete emotion appraisal lives in `src/core/emotion.rs` and the amygdala (`src/regions/amygdala.rs`), backed by neuromodulators for valence/arousal.
- **Next focus:** Persist mood baselines, add regulation strategies, and link emotion trajectories to decision making.

### 3. Self-Model & Metacognition ⚠️ (iterating)
- **Current coverage:** The DMN (`src/regions/dmn.rs`) and self-model (`src/core/self_model.rs`) maintain identity, beliefs, and metacognitive traces.
- **Next focus:** Expand theory-of-mind reasoning, narrative coherence, and integration with social cognition signals.

### 4. Inner Speech & Narrative ✅
- **Current coverage:** `src/core/inner_speech.rs` generates self-talk, rehearsal, and narrative buffers that synchronize with the DMN.
- **Next focus:** Couple inner speech with upcoming language cortex modules for grounded verbal reasoning.

### 5. Sleep & Dreams ⚠️ (iterating)
- **Current coverage:** `src/core/sleep.rs` orchestrates multi-stage sleep, dream generation via the imagination engine, and exposes `Brain::sleep`.
- **Next focus:** Tie sleep pressure to hypothalamic drives, log dream insights, and trigger schema updates post-REM.

### 6. Predictive Processing ⚠️ (partial)
- **Current coverage:** `src/core/prediction.rs` delivers prediction error signals and gating via neuromodulators and basal ganglia.
- **Next focus:** Introduce hierarchical predictive coding and active inference loops that can spawn exploratory actions.

### 7. Attention Networks ⚠️ (partial)
- **Current coverage:** Attention budgeting (`src/core/attention.rs`), thalamic gating, and prefrontal focus provide baseline competition.
- **Next focus:** Separate dorsal/ventral attention pathways, add salience network coordination, and refine mind-wandering control.

### 8. Social Cognition ⚠️ (partial)
- **Current coverage:** Trust/oxytocin signalling, empathic insula responses, and multi-agent messaging support proto-social awareness.
- **Next focus:** Build explicit theory-of-mind models, social hierarchy tracking, and reputation-weighted decision policies.

### 9. Temporal Cognition ✅
- **Implemented:** `src/core/temporal.rs` handles duration perception, mental time travel, prospective memory, and temporal discounting.
- **Next focus:** Close the loop with goal scheduling and episodic simulation for proactive planning.

### 10. Creativity & Imagination ⚠️ (partial)
- **Current coverage:** `src/core/imagination.rs` recombines memories, generates counterfactuals, and fuels dream content.
- **Next focus:** Embed imagination outputs into action selection, analogy making, and creative problem solving pipelines.

---

## Implementation Priority

### Phase A: Sensory Foundations
1. ✅ Build cortical sensory modules (visual, auditory, somatosensory, gustatory, olfactory) downstream of the thalamus (`src/regions/sensory_cortex.rs` + thalamus/nervous system integration).
2. ✅ Add posterior parietal integration to bind multimodal context before workspace broadcast (`src/regions/posterior_parietal.rs` + brain/thalamus/nervous system integration).
3. ✅ Connect sensory abstractions to schema and world-model stores for richer grounding.

### Phase B: Motor & Autonomic Control
4. ✅ Implement motor cortex layers that translate basal ganglia output into structured actions.
5. ✅ Extend cerebellar forward models and motor imagery, closing loops with the imagination engine.
6. ✅ Simulate brainstem autonomic centres to give hypothalamus/insula real-time feedback.

### Phase C: Salience & Social Reasoning
7. ✅ Formalize dorsal/ventral attention and the salience network (anterior insula + dorsal ACC).
8. ✅ Expand social cognition with theory-of-mind reasoning, reputation tracking, and mirror-system support.
9. Link mood, social context, and attention selection through shared neuromodulator tuning.

### Phase D: Predictive & Generative Mastery
10. Layer hierarchical predictive coding and active inference into perception/action loops.
11. Integrate imagination outputs with planning and goal management for creative problem solving.
12. Instrument mood and sleep data to adapt long-horizon strategies and self-narratives.

---

## Key Insight: The Illusion of Continuity

What makes humans feel "human" isn't just the components but:
- **Narrative continuity** - A coherent life story
- **Emotional coloring** - Everything feels like something
- **Embodied grounding** - Abstract concepts rooted in bodily experience
- **Social embedding** - Identity shaped by relationships
- **Temporal depth** - Rich past, anticipated future

The goal isn't to simulate neurons but to simulate the *experience* of being human.
