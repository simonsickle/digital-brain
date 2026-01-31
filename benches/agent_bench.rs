//! Agent System Benchmarks
//!
//! Run with: cargo bench

use digital_brain::agent::{AgentConfig, AgentLoop, BrainAgent, BrainAgentConfig};
use digital_brain::core::{
    ActionCategory, ActionTemplate, CuriositySystem, Domain, Entity, ExpectedOutcome, Goal,
    Outcome, Priority, WorldModel,
};
use std::time::{Duration, Instant};
use uuid::Uuid;

fn make_action(name: &str, category: ActionCategory) -> ActionTemplate {
    ActionTemplate {
        id: Uuid::new_v4(),
        name: name.to_string(),
        description: format!("Benchmark action: {}", name),
        preconditions: vec![],
        expected_outcomes: vec![ExpectedOutcome {
            outcome: Outcome::success("Success", 0.5),
            probability: 0.8,
        }],
        effort_cost: 0.3,
        time_cost: 5,
        category,
        tags: vec![],
    }
}

fn bench<F: FnMut()>(name: &str, iterations: u32, mut f: F) {
    // Warmup
    for _ in 0..10 {
        f();
    }

    let start = Instant::now();
    for _ in 0..iterations {
        f();
    }
    let elapsed = start.elapsed();

    let per_iter = elapsed / iterations;
    let ops_per_sec = 1_000_000_000.0 / per_iter.as_nanos() as f64;

    println!(
        "{:<40} {:>10.2} µs/iter  {:>12.0} ops/sec",
        name,
        per_iter.as_nanos() as f64 / 1000.0,
        ops_per_sec
    );
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║              DIGITAL BRAIN AGENT BENCHMARKS                  ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Agent Loop Benchmarks
    println!("─── Agent Loop ───────────────────────────────────────────────");

    let config = AgentConfig {
        min_tick_interval: Duration::from_millis(0),
        exploration_rate: 0.0,
        ..Default::default()
    };

    let mut agent = AgentLoop::new(config.clone());
    agent.register_action(make_action("action1", ActionCategory::Exploitation));
    agent.register_action(make_action("action2", ActionCategory::Exploration));
    agent.register_action(make_action("action3", ActionCategory::Rest));

    bench("Agent tick (no percepts)", 10_000, || {
        agent.tick();
    });

    let mut agent = AgentLoop::new(config.clone());
    agent.register_action(make_action("action1", ActionCategory::Exploitation));
    agent.add_goal(Goal::new("Test goal").with_priority(Priority::High));

    bench("Agent tick (with goal)", 10_000, || {
        agent.tick();
    });

    let mut agent = AgentLoop::new(config);
    for i in 0..10 {
        agent.register_action(make_action(
            &format!("action{}", i),
            ActionCategory::Exploitation,
        ));
    }

    bench("Agent tick (10 actions)", 10_000, || {
        agent.tick();
    });

    // Curiosity System Benchmarks
    println!("\n─── Curiosity System ─────────────────────────────────────────");

    let mut curiosity = CuriositySystem::new();
    for i in 0..10 {
        curiosity.register_domain(Domain::new(&format!("domain{}", i)), 0.5);
    }

    let state = digital_brain::core::NeuromodulatorState::default();

    bench("Curiosity explore_vs_exploit", 100_000, || {
        let _ = curiosity.explore_vs_exploit(&state);
    });

    bench("Curiosity novelty check", 100_000, || {
        let _ = curiosity.novelty("test_pattern");
    });

    let mut curiosity = CuriositySystem::new();
    curiosity.register_domain(Domain::new("test"), 0.5);
    let action = make_action("test", ActionCategory::Exploration);

    bench("Curiosity value calculation", 100_000, || {
        let _ = curiosity.curiosity_value(&action, &state);
    });

    // World Model Benchmarks
    println!("\n─── World Model ──────────────────────────────────────────────");

    let mut world = WorldModel::new();

    bench("World add entity", 10_000, || {
        world.add_entity(Entity::new("test", "TestEntity"));
    });

    let mut world = WorldModel::new();
    let ids: Vec<_> = (0..100)
        .map(|i| world.add_entity(Entity::new("test", &format!("Entity{}", i))))
        .collect();

    bench("World find by type (100 entities)", 10_000, || {
        let _ = world.find_by_type("test");
    });

    let id = ids[50];
    bench("World get entity by ID", 100_000, || {
        let _ = world.get_entity(id);
    });

    // Goal Manager Benchmarks
    println!("\n─── Goal Manager ─────────────────────────────────────────────");

    let mut agent = AgentLoop::new(AgentConfig {
        min_tick_interval: Duration::from_millis(0),
        ..Default::default()
    });

    bench("Goal add", 10_000, || {
        agent.add_goal(Goal::new("Test goal"));
    });

    let mut agent = AgentLoop::new(AgentConfig {
        min_tick_interval: Duration::from_millis(0),
        ..Default::default()
    });
    for i in 0..50 {
        agent.add_goal(Goal::new(&format!("Goal {}", i)));
    }

    bench("Goal selection (50 goals)", 10_000, || {
        let _ = agent.goals().get_active_goal(&state);
    });

    // Memory Benchmarks
    println!("\n─── Memory (Hippocampus) ─────────────────────────────────────");

    use digital_brain::prelude::*;
    use digital_brain::regions::hippocampus::HippocampusStore;

    let hippocampus = HippocampusStore::new_in_memory().unwrap();

    // Encode some memories first for retrieval benchmarks
    for i in 0..100 {
        let signal = BrainSignal::new(
            "bench",
            SignalType::Memory,
            format!("Memory content {} with Tesla API and various keywords", i),
        )
        .with_valence(if i % 2 == 0 { 0.5 } else { -0.3 })
        .with_salience(0.5);
        hippocampus.encode(&signal).unwrap();
    }

    bench("Memory encode", 1_000, || {
        let signal = BrainSignal::new(
            "bench",
            SignalType::Memory,
            "Benchmark memory encoding test",
        )
        .with_valence(0.5)
        .with_salience(0.6);
        let _ = hippocampus.encode(&signal);
    });

    bench("Memory retrieve (valence boost)", 1_000, || {
        let _ = hippocampus.retrieve(10, true);
    });

    bench("Memory semantic search", 1_000, || {
        let _ = hippocampus.retrieve_by_query("Tesla API keywords", 10);
    });

    bench("Memory decay (simulated 1h)", 100, || {
        let _ = hippocampus.decay_all(1.0);
    });

    // BrainAgent Benchmarks (if enabled)
    println!("\n─── BrainAgent ───────────────────────────────────────────────");

    let config = BrainAgentConfig {
        agent: AgentConfig {
            min_tick_interval: Duration::from_millis(0),
            ..Default::default()
        },
        enable_curiosity: false,
        enable_world_model: false,
        enable_communication: false,
        enable_multi_agent: false,
        sleep_interval: 100_000, // Don't sleep during bench
        ..Default::default()
    };

    match BrainAgent::with_config(config) {
        Ok(mut brain_agent) => {
            brain_agent.register_action(make_action("action", ActionCategory::Exploitation));

            bench("BrainAgent tick (minimal)", 1_000, || {
                brain_agent.tick();
            });
        }
        Err(e) => {
            println!("  (BrainAgent benchmarks skipped: {})", e);
        }
    }

    let config = BrainAgentConfig {
        agent: AgentConfig {
            min_tick_interval: Duration::from_millis(0),
            ..Default::default()
        },
        enable_curiosity: true,
        enable_world_model: true,
        enable_communication: true,
        enable_multi_agent: false,
        sleep_interval: 100_000,
        ..Default::default()
    };

    match BrainAgent::with_config(config) {
        Ok(mut brain_agent) => {
            brain_agent.register_action(make_action("action", ActionCategory::Exploitation));
            brain_agent.add_goal(Goal::new("Test").with_priority(Priority::High));

            bench("BrainAgent tick (full)", 1_000, || {
                brain_agent.tick();
            });
        }
        Err(e) => {
            println!("  (Full BrainAgent benchmarks skipped: {})", e);
        }
    }

    // Summary
    println!("\n══════════════════════════════════════════════════════════════");
    println!("Benchmarks complete!");
}
