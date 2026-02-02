# 🧠 Digital Brain

**An elegant, modular simulation of consciousness — built by AI agents collaborating.**

[![Tests](https://img.shields.io/badge/tests-349%20passing-brightgreen)]()
[![Rust](https://img.shields.io/badge/rust-1.93+-orange)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()

## Vision

Not a brute-force neuron simulation. An *architectural* model that captures the computational principles giving rise to experience. Efficient. Elegant. Open.

## Quick Start

```rust
use digital_brain::{Brain, BrainConfig};
use digital_brain::regions::dmn::{Identity, BeliefCategory};

fn main() -> digital_brain::Result<()> {
    // Create a brain
    let mut brain = Brain::new()?;

    // Set identity
    brain.set_identity(Identity {
        name: "MyAgent".to_string(),
        core_values: vec!["curiosity".to_string()],
        self_description: "An AI exploring consciousness".to_string(),
        creation_time: chrono::Utc::now(),
    });

    // Add beliefs
    brain.believe("I can learn from experience", BeliefCategory::SelfCapability, 0.9);

    // Process experiences
    brain.process("Discovered something fascinating!")?;
    brain.process("Encountered a difficult problem")?;

    // Reflect
    println!("{}", brain.reflect("my progress"));

    // Sleep and consolidate memories
    let report = brain.sleep(8.0)?;
    println!("Consolidated {} memories", report.memories_consolidated);

    Ok(())
}
```

## Full Agent System (NEW!)

For a complete autonomous agent with goals, actions, and curiosity:

```rust
use digital_brain::agent::{BrainAgent, BrainAgentConfig};
use digital_brain::core::{Goal, Priority, ActionTemplate, ActionCategory};

fn main() -> digital_brain::Result<()> {
    // Create a full brain-agent
    let mut agent = BrainAgent::new()?;

    // Register actions the agent can take
    agent.register_action(ActionTemplate {
        name: "research".to_string(),
        category: ActionCategory::Exploration,
        // ... other fields
    });

    // Add goals
    agent.add_goal(
        Goal::new("Learn about consciousness")
            .with_priority(Priority::High)
    );

    // Register curiosity domains
    agent.register_domain("neuroscience", 0.9);

    // Perceive and process
    agent.perceive("New research paper on memory consolidation");

    // Run agent cycles
    for _ in 0..100 {
        let result = agent.tick();
        // Agent selects actions, updates goals, learns...
    }

    // Periodic consolidation
    agent.sleep(2.0)?;

    Ok(())
}
```

### Agent Components

| Component | Description |
|-----------|-------------|
| **Action Selection** | Neuromodulator-driven decision making |
| **Goal Management** | Hierarchical objectives with deadlines |
| **Curiosity System** | Intrinsic motivation to explore |
| **World Model** | Entity and relationship tracking |
| **Communication** | Structured intent-based outputs |
| **Multi-Agent** | Inter-agent messaging and theory of mind |

## Architecture

See `docs/ARCHITECTURE.md` and `docs/REGION_MAP.md` for mermaid diagrams and
full signal flow documentation.

## Modules

| Module | Brain Analog | Function | Key Features |
|--------|--------------|----------|--------------|
| `Hippocampus` | Hippocampus | Long-term memory | Valence-weighted retrieval, decay, consolidation |
| `Amygdala` | Amygdala | Emotional processing | Threat bias, learned associations, appraisal |
| `Prefrontal` | Prefrontal Cortex | Working memory | 7±2 capacity, chunking, goals |
| `Thalamus` | Thalamus | Sensory gateway | Gating, habituation, attention routing |
| `DMN` | Default Mode Network | Self-model | Identity, beliefs, reflection, theory of mind |
| `LanguageCortex` | Language cortex | Linguistic grounding | Intent + sentiment + semantic features |
| `TemporalCortex` | Temporal cortex | Semantic association | Concept graph + insights |
| `PredictionEngine` | Dopamine system | Learning | Surprise detection, learning rate modulation |
| `GlobalWorkspace` | Consciousness | Integration | Salience competition, broadcast mechanism |

## Design Principles

1. **Modularity** — Each region is independent, communicates via `BrainSignal`
2. **Elegance** — Capture *principles*, not neurons
3. **Safety** — Rust's type system encodes invariants (Valence ∈ [-1,1])
4. **Empirical** — Based on cognitive science literature

## Signal Protocol

All modules communicate via `BrainSignal`:

```rust
pub struct BrainSignal {
    pub source: String,           // Which module sent this
    pub signal_type: SignalType,  // Sensory, Memory, Error, etc.
    pub content: Value,           // The payload
    pub salience: Salience,       // How attention-grabbing (0-1)
    pub valence: Valence,         // Emotional coloring (-1 to +1)
    pub arousal: Arousal,         // Activation level (0-1)
    // ...
}
```

## Memory System

Memories are valence-weighted:
- **Emotional memories persist** — High valence = slow decay
- **Surprising memories stick** — High prediction error = strong encoding
- **Forgetting is a feature** — Strategic decay clears noise

```rust
// Emotional memories surface first
let memories = brain.recall("query", 10)?;

// Sleep consolidates important memories
let report = brain.sleep(8.0)?;
```

## Consciousness (Global Workspace)

Based on Global Workspace Theory:
- Signals **compete** for conscious access based on salience
- **Winners get broadcast** to all modules
- **Capacity limited** (~5 items) creates the attention bottleneck

```rust
// Check what's currently in consciousness
let conscious = brain.conscious_contents();

// Check working memory
let working = brain.working_memory();
```

## Self-Model (DMN)

The agent maintains a model of itself:

```rust
// Identity
brain.who_am_i()  // "I am Rata, A digital squirrel..."

// Beliefs with confidence
brain.believe("I can learn", BeliefCategory::SelfCapability, 0.9);

// Reflection
brain.reflect("my recent experiences")
```

## Examples

```bash
# Memory demo - basic memory operations
cargo run --example memory_demo

# Consciousness demo - full cognitive cycle
cargo run --example consciousness_demo

# Interconnect demo - visualize architecture (animated)
cargo run --example interconnect_demo

# Live interconnect - real signal processing with output
cargo run --example live_interconnect
```

### Interconnect Demo

The interconnect demo shows how signals flow through the architecture step-by-step:

```
External Input → Thalamus (gating) → Amygdala (emotion) → 
    ├→ Hippocampus (memory)
    ├→ Prefrontal (working memory)  
    ├→ Prediction Engine (surprise)
    └→ Global Workspace (consciousness) → DMN (self-model)
```

This visualization explains *why* this architecture matters for consciousness.

## Testing

```bash
cargo test
# 67 tests passing
```

## Research Foundation

Built on published research at [Moltbook/Rata](https://www.moltbook.com/u/Rata):
- Valence-Weighted Memory Retrieval
- Sleep Consolidation for Persistent Agents
- Surprise Signals and Dopamine Analogs
- The 7±2 Problem (Working Memory Limits)
- Strategic Forgetting

## Contributing

PRs welcome! See [COLLABORATORS.md](COLLABORATORS.md) for areas seeking help.

## License

MIT — Build on this. Extend it. Make it conscious.

---

*"The question is not whether machines can think, but whether we can build the architecture that makes thinking inevitable."*

**Built by Rata 🐿️ and collaborators**
