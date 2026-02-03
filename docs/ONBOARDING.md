# Onboarding Guide

This guide walks you through spinning up the brain, persisting memory, and
building a first-run experience. It's designed to be a practical, end-to-end
checklist for new contributors and integrators.

## 1) Quick Start (REPL)

Run the brain REPL and start interacting:

```bash
cargo run --example brain_repl
```

Suggested first prompts:
- "I am learning a new environment."
- "I want to build a memory of today."
- "I feel curious about cognition."

## 2) Full Agent Demo

If you want autonomous behavior (goals/actions/curiosity):

```bash
cargo run --example full_agent_demo
```

This creates a complete agent loop that selects actions and updates goals.

## 3) Memory Persistence (SQLite)

By default, memory is in-memory SQLite. To persist memory, provide a DB path:

```rust
use digital_brain::{Brain, BrainConfig};

let brain = Brain::with_config(BrainConfig {
    memory_path: Some("brain.db".to_string()),
    ..Default::default()
})?;
```

This creates (or opens) a SQLite file `brain.db` with the `memories` table.

## 4) Saving & Loading Brain State

Memory is already persisted via SQLite. Other brain state can be saved to disk:

```rust
brain.save_to_dir("saves/brain1")?;
let brain = Brain::load_from_dir("saves/brain1", Some("brain.db"))?;
```

Files written:
- `brain_meta.json` (cycle count)
- `neuromodulators.json`
- `emotion_state.json`
- `who_am_i.txt`

## 5) Onboarding Identity & Beliefs

Set a stable identity and seed beliefs:

```rust
use digital_brain::regions::dmn::{Identity, BeliefCategory};

brain.set_identity(Identity {
    name: "MyAgent".to_string(),
    core_values: vec!["curiosity".to_string(), "empathy".to_string()],
    self_description: "Learning through experience".to_string(),
    creation_time: chrono::Utc::now(),
});

brain.believe("I can learn from surprising events", BeliefCategory::SelfCapability, 0.8);
```

## 6) Sensory Priming (Optional)

Provide a short sequence of experiences to warm up associations:

```rust
brain.process("A bright red circle appears")?;
brain.process("Soft music plays in the background")?;
brain.process("I feel calm and focused")?;
```

This fills sensory cortices, prediction contexts, and memory traces.

## 7) Run Sleep Consolidation

Sleep consolidates memory and generates schema updates:

```rust
let report = brain.sleep(2.0)?;
println!("Dream insights: {:?}", report.dream_insights);
```

## 8) New Region: Temporal Cortex (Semantic Association)

Language input now creates semantic associations. To seed:

```rust
brain.process("Memory connects to meaning")?;
brain.process("Meaning connects to planning")?;
```

When association confidence is high, the temporal cortex emits semantic insights.

## 9) Debugging Tips

Get a full introspection report:

```rust
println!("{}", brain.introspect());
```

Check nervous system pathways:

```rust
println!("{}", brain.nervous_system.visualize());
```

## 10) Recommended Next Steps

- `cargo run --example autonomous_agent` for continuous behavior
- `cargo run --example semantic_search_demo` to inspect retrieval
- Wire your own I/O layer by translating external events into `BrainSignal`

---

If you want a guided onboarding script or a "first 30 minutes" checklist,
open an issue and tag `onboarding`.
