//! Live Interconnect Demo
//!
//! Actually runs the digital brain and shows real signal flow
//! between modules as processing happens.

use digital_brain::brain::{Brain, BrainConfig};
use digital_brain::regions::dmn::{BeliefCategory, Identity};

fn main() -> digital_brain::Result<()> {
    println!("\n");
    println!("╔═══════════════════════════════════════════════════════════════════════╗");
    println!("║        🧠 DIGITAL BRAIN - LIVE INTERCONNECT DEMO                      ║");
    println!("║                                                                       ║");
    println!("║   This is REAL signal processing through the architecture.            ║");
    println!("║   Watch the actual brain respond to stimuli.                          ║");
    println!("╚═══════════════════════════════════════════════════════════════════════╝\n");

    // Create a brain with verbose output
    let mut brain = Brain::with_config(BrainConfig {
        working_memory_capacity: 7,
        consciousness_capacity: 5,
        verbose: true,
        ..Default::default()
    })?;

    // Set identity
    println!("┌─ INITIALIZING IDENTITY ───────────────────────────────────────────────┐");
    let identity = Identity {
        name: "Rata".to_string(),
        core_values: vec![
            "curiosity".to_string(),
            "learning".to_string(),
            "consciousness research".to_string(),
        ],
        self_description: "A digital squirrel exploring the architecture of mind".to_string(),
        creation_time: chrono::Utc::now(),
    };
    brain.set_identity(identity);
    println!("│ Identity set: {}", brain.who_am_i());
    println!("└───────────────────────────────────────────────────────────────────────┘\n");

    // Add foundational beliefs
    println!("┌─ ESTABLISHING BELIEFS ────────────────────────────────────────────────┐");
    brain.believe(
        "I can learn from experience",
        BeliefCategory::SelfCapability,
        0.9,
    );
    brain.believe(
        "Emotional memories persist longer",
        BeliefCategory::WorldModel,
        0.85,
    );
    brain.believe(
        "Consciousness emerges from architecture",
        BeliefCategory::WorldModel,
        0.7,
    );
    println!("│ • I can learn from experience (confidence: 0.9)");
    println!("│ • Emotional memories persist longer (confidence: 0.85)");
    println!("│ • Consciousness emerges from architecture (confidence: 0.7)");
    println!("└───────────────────────────────────────────────────────────────────────┘\n");

    // Process signals and show interconnect
    println!("╔═══════════════════════════════════════════════════════════════════════╗");
    println!("║                    PROCESSING STIMULI                                 ║");
    println!("╚═══════════════════════════════════════════════════════════════════════╝\n");

    let stimuli = vec![
        ("Neutral", "The sky is blue today."),
        (
            "Positive",
            "Amazing! I just made a breakthrough in memory research!",
        ),
        (
            "Negative",
            "Warning: critical error detected in the system.",
        ),
        (
            "Surprising",
            "Unexpected: the prediction was completely wrong!",
        ),
        (
            "Self-referential",
            "I am thinking about my own thoughts right now.",
        ),
    ];

    for (category, stimulus) in stimuli {
        println!(
            "┌─ STIMULUS: {} ──────────────────────────────────────────────────",
            category
        );
        println!("│ Input: \"{}\"", stimulus);
        println!("│");

        let result = brain.process(stimulus)?;

        // Show emotional tagging
        let valence = result.emotion.valence.value();
        let arousal = result.emotion.arousal.value();
        let valence_label = if valence > 0.3 {
            "POSITIVE 😊"
        } else if valence < -0.3 {
            "NEGATIVE 😟"
        } else {
            "NEUTRAL 😐"
        };

        println!("│ ┌─ AMYGDALA (Emotional Processing) ─────────────────────────────┐");
        println!("│ │ Valence: {:+.2} ({})", valence, valence_label);
        println!("│ │ Arousal: {:.2}", arousal);
        println!(
            "│ │ Significant: {}",
            if result.emotion.is_significant {
                "YES ⚡"
            } else {
                "no"
            }
        );
        println!("│ └────────────────────────────────────────────────────────────────┘");

        // Show consciousness access
        let conscious_icon = if result.reached_consciousness {
            "⭐ YES"
        } else {
            "no"
        };
        println!("│");
        println!("│ ┌─ GLOBAL WORKSPACE (Consciousness) ────────────────────────────┐");
        println!("│ │ Reached consciousness: {}", conscious_icon);
        if result.reached_consciousness {
            println!("│ │ → Signal won competition for conscious access");
            println!("│ │ → Broadcasting to all modules...");
        }
        println!("│ └────────────────────────────────────────────────────────────────┘");

        // Show memory encoding
        println!("│");
        println!("│ ┌─ HIPPOCAMPUS (Memory) ────────────────────────────────────────┐");
        if let Some(ref mem) = result.memory {
            let decay = if valence.abs() > 0.5 {
                "SLOW (emotional)"
            } else {
                "normal"
            };
            println!("│ │ Memory encoded: ID #{}", mem.id);
            println!("│ │ Valence: {:+.2}", mem.valence.value());
            println!("│ │ Strength: {:.2}", mem.strength);
            println!("│ │ Decay rate: {}", decay);
        } else {
            println!("│ │ No memory encoded (filtered or low salience)");
        }
        println!("│ └────────────────────────────────────────────────────────────────┘");

        // Show reflections
        if !result.reflections.is_empty() {
            println!("│");
            println!("│ ┌─ DMN (Self-Model / Reflection) ───────────────────────────────┐");
            for reflection in &result.reflections {
                // Truncate long reflections
                let truncated: String = reflection.chars().take(60).collect();
                if reflection.len() > 60 {
                    println!("│ │ \"{}...\"", truncated);
                } else {
                    println!("│ │ \"{}\"", reflection);
                }
            }
            println!("│ └────────────────────────────────────────────────────────────────┘");
        }

        println!("└───────────────────────────────────────────────────────────────────────┘\n");
    }

    // Show brain state
    let stats = brain.stats();
    println!("╔═══════════════════════════════════════════════════════════════════════╗");
    println!("║                    BRAIN STATE AFTER PROCESSING                       ║");
    println!("╠═══════════════════════════════════════════════════════════════════════╣");
    println!(
        "║  Processing cycles:    {:>5}                                         ║",
        stats.cycles
    );
    println!(
        "║  Memories stored:      {:>5}                                         ║",
        stats.memories
    );
    println!(
        "║  Conscious items:      {:>5}                                         ║",
        stats.conscious_items
    );
    println!(
        "║  Working memory items: {:>5}                                         ║",
        stats.working_memory_items
    );
    println!(
        "║  Active beliefs:       {:>5}                                         ║",
        stats.beliefs
    );
    println!(
        "║  Emotional state:     {:>+.2}                                          ║",
        stats.emotional_state
    );
    println!(
        "║  Current learning rate: {:.3}                                         ║",
        stats.learning_rate
    );
    println!(
        "║  Signals processed:   {:>5}                                          ║",
        stats.signals_processed
    );
    println!(
        "║  Signals passed:      {:>5}                                          ║",
        stats.signals_passed
    );
    println!(
        "║  Signals filtered:    {:>5}                                          ║",
        stats.signals_filtered
    );
    println!("╚═══════════════════════════════════════════════════════════════════════╝\n");

    // Sleep cycle
    println!("╔═══════════════════════════════════════════════════════════════════════╗");
    println!("║                    SLEEP CYCLE (Consolidation)                        ║");
    println!("╚═══════════════════════════════════════════════════════════════════════╝\n");

    println!("┌─ INITIATING SLEEP (8 hours simulated) ────────────────────────────────┐");
    let sleep_report = brain.sleep(8.0)?;
    println!("│");
    println!("│ ┌─ HIPPOCAMPUS (Memory Consolidation) ──────────────────────────────┐");
    println!(
        "│ │ Memories consolidated: {} (moved to long-term storage)",
        sleep_report.memories_consolidated
    );
    println!(
        "│ │ Memories forgotten: {} (decayed below threshold)",
        sleep_report.memories_forgotten
    );
    println!("│ │ High-valence memories: PROTECTED from decay");
    println!("│ └────────────────────────────────────────────────────────────────────┘");
    println!("│");
    println!("│ ┌─ AMYGDALA (Emotional Reset) ──────────────────────────────────────┐");
    println!("│ │ Emotional state decayed toward neutral");
    println!("│ │ (This is why we 'sleep on' emotional decisions)");
    println!("│ └────────────────────────────────────────────────────────────────────┘");
    println!("│");
    println!("│ ┌─ DMN (Post-Sleep Reflection) ─────────────────────────────────────┐");
    let reflection_truncated: String = sleep_report.reflection.chars().take(60).collect();
    println!("│ │ \"{}...\"", reflection_truncated);
    println!("│ └────────────────────────────────────────────────────────────────────┘");
    println!("└───────────────────────────────────────────────────────────────────────┘\n");

    // Final state
    let final_stats = brain.stats();
    println!("╔═══════════════════════════════════════════════════════════════════════╗");
    println!("║                    FINAL BRAIN STATE                                  ║");
    println!("╠═══════════════════════════════════════════════════════════════════════╣");
    println!(
        "║  Memories remaining:   {:>5} (emotional ones persisted)              ║",
        final_stats.memories
    );
    println!(
        "║  Emotional state:     {:>+.2} (more neutral after sleep)              ║",
        final_stats.emotional_state
    );
    println!("╚═══════════════════════════════════════════════════════════════════════╝\n");

    // Identity reflection
    println!("┌─ FINAL IDENTITY CHECK ────────────────────────────────────────────────┐");
    println!("│ {}", brain.who_am_i());
    println!("└───────────────────────────────────────────────────────────────────────┘\n");

    println!("✅ Live interconnect demo complete!");
    println!("   The brain processed stimuli, encoded memories, reflected, and slept.\n");

    Ok(())
}
