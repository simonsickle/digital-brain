# Repository Guidelines

## Project Structure & Module Organization
Digital Brain core lives under `src/`. `lib.rs` exposes modules, while `brain.rs` handles orchestration. `regions/` groups cognition modules (hippocampus, amygdala, prefrontal, thalamus, dmn); `core/` holds shared systems like the global workspace and prediction engine. Integration tests sit in `tests/`, and runnable demos live in `examples/` (e.g., `memory_demo`, `consciousness_demo`). Persistence is in-memory via rusqlite; no external assets are versioned.

## Build, Test, and Development Commands
- `cargo build` — compile debug artifacts for iterative checks.
- `cargo build --release` — produce optimized binaries prior to distribution.
- `cargo test` — run all 67 unit and integration tests; must pass before merges.
- `cargo clippy --all-targets -- -D warnings` — lint with warnings treated as errors.
- `cargo fmt --all` / `cargo fmt --all --check` — format code or verify formatting.
- `cargo run --example <name>` — execute demos such as `memory_demo` or `interconnect_demo`.

## Coding Style & Naming Conventions
Target Rust 2024 with four-space indentation. Prefer expressive `snake_case` for functions/variables and `CamelCase` for types and enums. Prefix intentionally unused variables with `_`. Collapse nested `if` expressions when clippy recommends it. Encode invariants (e.g., valence ∈ [-1, 1]) via newtypes or validation guards, and annotate intricate flows with brief comments when necessary.

## Testing Guidelines
Use Rust’s built-in `#[test]` framework. Place tight-scoped tests inside `#[cfg(test)]` modules next to the code; broader behavior goes in `tests/`. Name tests for observable behavior (`handles_low_valence_input`). Favor deterministic inputs, leveraging Tokio test utilities for async contexts. Every behavioral change should ship with at least one new or updated test.

## Commit & Pull Request Guidelines
Write commit subjects in the imperative present (“Add thalamus gating metrics”) and keep them under ~72 characters. Group related changes together, avoiding drive-by refactors within feature commits. Pull requests should summarize the architectural impact, list verification steps (tests, clippy, fmt), and link tracking issues. Attach logs or screenshots when behavior is user-visible.

## Architecture Notes
Maintain module independence by routing communication exclusively through `BrainSignal`. Extending signal types requires updating `SignalType`, `brain.rs`, and affected region handlers. Keep async work non-blocking under Tokio; prefer channels or tasks over thread sleeps to preserve scheduler health.
