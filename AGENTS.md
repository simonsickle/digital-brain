# Digital Brain - Agent Guidelines

This document provides essential information for AI coding agents working on the digital-brain project.

## Project Overview

Digital Brain is a Rust-based modular simulation of consciousness. It implements computational principles of cognition rather than brute-force neuron simulation.

## Tech Stack

- **Language**: Rust (Edition 2024)
- **Package Manager**: Cargo
- **Test Framework**: Built-in Rust test framework (`#[test]`)
- **Key Dependencies**: tokio (async), serde (serialization), rusqlite (memory persistence), chrono (time)

## Essential Commands

```bash
# Build
cargo build
cargo build --release

# Test (67 tests)
cargo test

# Lint (warnings as errors in CI)
cargo clippy --all-targets -- -D warnings

# Format check
cargo fmt --all --check

# Format fix
cargo fmt --all

# Run examples
cargo run --example memory_demo
cargo run --example consciousness_demo
cargo run --example interconnect_demo
cargo run --example live_interconnect
```

## Project Structure

```
src/
├── lib.rs              # Library root, re-exports
├── brain.rs            # Main Brain orchestrator
├── signal.rs           # BrainSignal protocol (inter-module communication)
├── error.rs            # Error types
├── regions/            # Brain region modules
│   ├── hippocampus.rs  # Long-term memory (valence-weighted, decay, consolidation)
│   ├── amygdala.rs     # Emotional processing (threat bias, learned associations)
│   ├── prefrontal.rs   # Working memory (7±2 capacity, chunking)
│   ├── thalamus.rs     # Sensory gateway (gating, habituation, routing)
│   ├── dmn.rs          # Default Mode Network (identity, beliefs, reflection)
│   └── mod.rs
└── core/               # Core systems
    ├── workspace.rs    # Global Workspace (consciousness, salience competition)
    ├── prediction.rs   # Prediction engine (surprise, dopamine-like learning)
    └── mod.rs

tests/                  # Integration tests
examples/               # Runnable demos
```

## Architecture Principles

1. **Modularity**: Each brain region is independent, communicates via `BrainSignal`
2. **Type Safety**: Rust's type system encodes invariants (e.g., Valence in [-1,1])
3. **Signal Protocol**: All modules communicate through `BrainSignal` structs

## Key Types

- `BrainSignal`: Core message type between modules (source, signal_type, content, salience, valence, arousal)
- `SignalType`: Sensory, Memory, Prediction, Error, Emotion, Attention, Broadcast, Query, Motor
- `Valence`: Emotional coloring (-1 to +1)
- `Salience`: Attention-grabbing level (0-1)
- `Arousal`: Activation level (0-1)

## CI Pipeline

The project uses GitHub Actions with:
- `cargo check` - Fast compilation check
- `cargo test` - All unit and integration tests
- `cargo fmt --check` - Code formatting
- `cargo clippy -D warnings` - Linting with warnings as errors
- `cargo build --release` - Release build verification

All checks must pass for PRs and merge queue.

## Code Style

- Run `cargo fmt` before committing
- Ensure `cargo clippy -- -D warnings` passes
- Prefix unused variables with `_` (e.g., `_unused_var`)
- Collapse nested if statements when possible (clippy::collapsible_if)
- Use `.is_multiple_of()` instead of `% n == 0`

## Testing

- Unit tests are in `#[cfg(test)]` modules within source files
- Integration tests are in `tests/` directory
- All tests use in-memory SQLite (no external database needed)
- Run `cargo test` to execute all 67 tests
