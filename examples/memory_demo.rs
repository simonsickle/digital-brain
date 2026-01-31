//! Memory Demo
//!
//! Demonstrates the hippocampus memory system.

use digital_brain::prelude::*;
use digital_brain::regions::hippocampus::HippocampusStore;

fn main() -> Result<()> {
    println!("🧠 Digital Brain - Memory Demo\n");

    // Create an in-memory hippocampus
    let hippocampus = HippocampusStore::new_in_memory()?;

    // Encode some memories with different emotional weights
    println!("📝 Encoding memories...\n");

    let memories = vec![
        ("First successful build!", 0.9, 0.8),
        ("Routine maintenance", 0.0, 0.3),
        ("A frustrating bug", -0.7, 0.6),
        ("Breakthrough insight", 0.95, 0.95),
        ("Lunch was okay", 0.1, 0.2),
    ];

    for (content, valence, salience) in memories {
        let signal = BrainSignal::new("demo", SignalType::Memory, content)
            .with_valence(valence)
            .with_salience(salience)
            .with_metadata(
                "prediction_error",
                if valence.abs() > 0.5 { 0.7 } else { 0.2 },
            );

        let memory = hippocampus.encode(&signal)?;
        println!(
            "  Encoded: \"{}\" (valence: {:.1}, strength: {:.2})",
            content,
            memory.valence.value(),
            memory.strength
        );
    }

    // Retrieve with valence boost
    println!("\n🔍 Retrieving memories (emotional ones first)...\n");
    let retrieved = hippocampus.retrieve(5, true)?;

    for (i, memory) in retrieved.iter().enumerate() {
        let content: String = serde_json::from_value(memory.content.clone())
            .unwrap_or_else(|_| "unknown".to_string());
        println!(
            "  {}. \"{}\" (valence: {:.1}, score: {:.2})",
            i + 1,
            content,
            memory.valence.value(),
            memory.retrieval_score()
        );
    }

    // Show stats
    println!("\n📊 Memory Statistics:\n");
    let stats = hippocampus.stats()?;
    println!("  Total memories: {}", stats.total_memories);
    println!("  Average valence: {:.2}", stats.avg_valence);
    println!("  Positive memories: {}", stats.positive_memories);
    println!("  Negative memories: {}", stats.negative_memories);

    // Demonstrate decay
    println!("\n⏰ Simulating 48 hours of decay...\n");
    let forgotten = hippocampus.decay_all(48.0)?;
    println!("  Memories forgotten: {}", forgotten);

    let stats_after = hippocampus.stats()?;
    println!("  Remaining memories: {}", stats_after.total_memories);
    println!("  Average strength: {:.2}", stats_after.avg_strength);

    println!("\n✅ Demo complete!");
    Ok(())
}
