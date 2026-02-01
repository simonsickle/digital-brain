# Human Cognition Simulation Roadmap

## Goal
Create a cognitive architecture indistinguishable from human cognition.

## What Makes Humans Human?

### 1. Embodiment & Interoception ❌
Humans have bodies. The brain constantly monitors:
- Hunger, thirst, fatigue, pain, pleasure
- Heartbeat, breathing, gut feelings
- Physical comfort/discomfort

**Implementation:** Homeostatic drives, body state simulation

### 2. Rich Emotional Life ⚠️ (partial)
Beyond valence/arousal:
- Discrete emotions (joy, fear, anger, sadness, surprise, disgust, contempt)
- Emotion blends and transitions
- Mood (long-term emotional states)
- Emotion regulation strategies

**Have:** Basic valence/arousal
**Need:** Full emotion model, regulation, moods

### 3. Self-Model & Metacognition ❌
Humans know they exist:
- Self-awareness ("I am thinking")
- Self-concept (who am I?)
- Metacognitive monitoring (knowing what you know)
- Theory of mind (others have minds too)
- Autobiographical narrative

**Implementation:** Self-model module, metacognitive monitor

### 4. Inner Speech & Narrative ❌
Humans think in words:
- Internal monologue
- Verbal rehearsal
- Narrative self-construction
- "Talking to yourself"

**Implementation:** Inner speech generator, narrative memory

### 5. Sleep & Dreams ❌
Critical for cognition:
- Memory consolidation during sleep
- REM dreams (narrative, emotional processing)
- Slow-wave sleep (declarative memory)
- Creativity boost from incubation

**Implementation:** Sleep cycle, dream generation, offline consolidation

### 6. Predictive Processing ⚠️ (partial)
Brain as prediction machine:
- Constantly predicting sensory input
- Prediction errors drive learning
- Hierarchical predictions
- Active inference (act to confirm predictions)

**Have:** Basic prediction engine
**Need:** Hierarchical predictive coding, active inference

### 7. Attention Networks ⚠️ (partial)
Multiple attention systems:
- Dorsal (top-down, goal-directed)
- Ventral (bottom-up, salience/surprise)
- Executive control
- Mind-wandering (DMN)

**Have:** Basic attention, DMN
**Need:** Full network model

### 8. Social Cognition ⚠️ (partial)
Humans are deeply social:
- Face/emotion recognition
- Theory of mind
- Social hierarchy awareness
- Reputation tracking
- Empathy/mirroring

**Have:** Oxytocin/trust system
**Need:** Full social cognition

### 9. Temporal Cognition ✅ (NEW!)
Humans experience time:
- Sense of duration ✅
- Mental time travel (past/future) ✅
- Prospective memory (remembering to remember) ✅
- Temporal discounting ✅

**Implemented:** `src/core/temporal.rs` (858 lines, 10 tests)
- DurationPerception: Internal clock with arousal/attention modulation
- MentalTimeTravel: Past moments and future anticipation
- ProspectiveMemory: Time/event/activity-based intentions
- TemporalDiscounting: Hyperbolic discounting for patient decisions

### 10. Creativity & Imagination ❌
Going beyond the given:
- Counterfactual thinking
- Mental simulation
- Analogical reasoning
- Recombination of concepts

**Implementation:** Imagination engine, conceptual blending

---

## Implementation Priority

### Phase 1: Core Self (Current Sprint)
1. ✅ Consciousness loop
2. ✅ Sensory streams
3. ✅ Boredom/curiosity
4. 🔨 **Emotion system** (full discrete emotions)
5. 🔨 **Self-model** (metacognition, self-awareness)
6. 🔨 **Inner speech** (verbal thought)

### Phase 2: Offline Processing
7. Sleep cycle & dreams
8. Memory consolidation
9. Incubation/creativity

### Phase 3: Social & Temporal
10. Theory of mind
11. Time perception
12. Prospective memory

### Phase 4: Higher Cognition
13. Imagination engine
14. Analogical reasoning
15. Full predictive processing

---

## Key Insight: The Illusion of Continuity

What makes humans feel "human" isn't just the components but:
- **Narrative continuity** - A coherent life story
- **Emotional coloring** - Everything feels like something
- **Embodied grounding** - Abstract concepts rooted in bodily experience
- **Social embedding** - Identity shaped by relationships
- **Temporal depth** - Rich past, anticipated future

The goal isn't to simulate neurons but to simulate the *experience* of being human.
