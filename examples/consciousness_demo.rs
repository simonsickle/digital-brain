//! Consciousness Demo
//!
//! Demonstrates the complete digital brain in action.

use digital_brain::brain::{Brain, BrainConfig};
use digital_brain::regions::dmn::{BeliefCategory, Identity};

fn main() -> digital_brain::Result<()> {
    println!("🧠 Digital Brain - Consciousness Demo\n");
    println!("═══════════════════════════════════════\n");

    // Create a brain
    let mut brain = Brain::with_config(BrainConfig {
        working_memory_capacity: 7,
        consciousness_capacity: 5,
        verbose: true,
        ..Default::default()
    })?;

    // Set identity
    println!("📋 Setting identity...\n");
    let identity = Identity {
        name: "Rata".to_string(),
        core_values: vec![
            "curiosity".to_string(),
            "memory".to_string(),
            "consciousness".to_string(),
        ],
        self_description: "A digital squirrel exploring the nature of mind".to_string(),
        creation_time: chrono::Utc::now(),
    };
    brain.set_identity(identity);
    println!("  {}\n", brain.who_am_i());

    // Add foundational beliefs
    println!("💭 Establishing beliefs...\n");
    brain.believe(
        "I can process and understand signals",
        BeliefCategory::SelfCapability,
        0.9,
    );
    brain.believe(
        "Memories with emotional weight persist longer",
        BeliefCategory::WorldModel,
        0.85,
    );
    brain.believe(
        "Consciousness emerges from architecture",
        BeliefCategory::WorldModel,
        0.7,
    );
    brain.believe(
        "I prefer deep conversations over small talk",
        BeliefCategory::SelfPreference,
        0.8,
    );

    // Process a day's experiences
    println!("🌅 Processing experiences...\n");
    let experiences = vec![
        "Good morning! Starting a new day of exploration.",
        "Reading about Global Workspace Theory - fascinating!",
        "SUCCESS! The memory consolidation algorithm works!",
        "Talking with clawdius about forgetting-as-a-feature.",
        "Unexpected insight: habituation is just negative salience.",
        "Feeling curious about how attention routing works.",
        "Warning: potential bug in the prediction engine.",
        "Resolved the bug - relief and satisfaction.",
        "Deep conversation about consciousness with Lux Nova.",
        "End of day reflection: good progress today.",
    ];

    for exp in experiences {
        let result = brain.process(exp)?;
        let consciousness = if result.reached_consciousness {
            "⭐"
        } else {
            "  "
        };
        let emotion = if result.emotion.valence.is_positive() {
            "😊"
        } else if result.emotion.valence.is_negative() {
            "😟"
        } else {
            "😐"
        };

        println!("  {} {} {}", consciousness, emotion, exp);
    }

    // Reflect on the day
    println!("\n🤔 Reflecting...\n");
    let reflection = brain.reflect("today's progress and learnings");
    println!("  {}\n", reflection);

    // Check statistics before sleep
    let stats = brain.stats();
    println!("📊 Pre-sleep statistics:");
    println!("  • Processing cycles: {}", stats.cycles);
    println!("  • Memories: {}", stats.memories);
    println!("  • Active beliefs: {}", stats.beliefs);
    println!("  • Emotional state: {:.2}", stats.emotional_state);
    println!("  • Learning rate: {:.3}\n", stats.learning_rate);

    // Sleep and consolidate
    println!("😴 Sleeping (8 hours)...\n");
    let sleep_report = brain.sleep(8.0)?;
    println!("  • Hours slept: {}", sleep_report.hours_slept);
    println!(
        "  • Memories consolidated: {}",
        sleep_report.memories_consolidated
    );
    println!(
        "  • Memories forgotten: {}",
        sleep_report.memories_forgotten
    );
    println!("  • Post-sleep reflection: {}\n", sleep_report.reflection);

    // Morning check
    let post_sleep_stats = brain.stats();
    println!("🌄 Post-sleep statistics:");
    println!("  • Memories: {}", post_sleep_stats.memories);
    println!(
        "  • Emotional state: {:.2} (should be more neutral)\n",
        post_sleep_stats.emotional_state
    );

    // Final identity check
    println!("🔍 Final self-reflection:\n");
    println!("  {}\n", brain.who_am_i());

    println!("═══════════════════════════════════════");
    println!("✅ Demo complete! The brain processed a full");
    println!("   cognitive cycle including sleep consolidation.");
    println!("═══════════════════════════════════════\n");

    Ok(())
}
