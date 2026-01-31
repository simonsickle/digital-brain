//! Semantic Search Demo
//!
//! Demonstrates the keyword-based semantic search in the hippocampus.
//! Shows how memories can be retrieved by content relevance, not just
//! recency or emotional weight.

use digital_brain::prelude::*;
use digital_brain::regions::hippocampus::HippocampusStore;

fn main() -> Result<()> {
    println!("🧠 Digital Brain - Semantic Search Demo\n");
    println!("{}", "=".repeat(60));

    // Create an in-memory hippocampus
    let hippocampus = HippocampusStore::new_in_memory()?;

    // Populate with diverse memories
    println!("\n📝 Encoding experiences...\n");

    let experiences = vec![
        ("Successfully integrated the Tesla API for vehicle control", 0.9, 0.8),
        ("Morning standup meeting with the team", 0.1, 0.3),
        ("Fixed a critical authentication bug in the API", 0.7, 0.7),
        ("Tesla Autopilot engaged on the highway drive", 0.8, 0.8),
        ("Coffee break discussion about AI consciousness", 0.4, 0.4),
        ("Deployed new version to production servers", 0.6, 0.6),
        ("API rate limiting caused temporary service disruption", -0.4, 0.5),
        ("Celebrated the Tesla integration milestone achievement", 0.95, 0.9),
        ("Debugging session lasted four hours of frustration", -0.3, 0.4),
        ("Learned about transformer attention mechanisms today", 0.5, 0.5),
    ];

    for (content, valence, salience) in &experiences {
        let signal = BrainSignal::new("experience", SignalType::Memory, *content)
            .with_valence(*valence)
            .with_salience(*salience);
        hippocampus.encode(&signal)?;
        let preview = if content.len() > 50 { &content[..50] } else { content };
        println!("  ✓ {} (v={:+.1})", preview, valence);
    }

    let stats = hippocampus.stats()?;
    println!("\n📊 Encoded {} memories", stats.total_memories);

    // Demonstrate semantic search
    println!("\n{}", "=".repeat(60));
    println!("🔍 SEMANTIC SEARCH DEMONSTRATIONS\n");

    // Search 1: Tesla-related
    println!("Query: \"Tesla vehicle\"");
    println!("{}", "-".repeat(40));
    let tesla_memories = hippocampus.retrieve_by_query("Tesla vehicle", 3)?;
    if tesla_memories.is_empty() {
        println!("  (no results)");
    }
    for (i, mem) in tesla_memories.iter().enumerate() {
        let content: String = serde_json::from_value(mem.content.clone())
            .unwrap_or_else(|_| "unknown".to_string());
        let display = if content.len() > 50 { &content[..50] } else { &content };
        println!(
            "  {}. {} (v={:+.2})",
            i + 1,
            display,
            mem.valence.value()
        );
    }

    // Search 2: API-related
    println!("\nQuery: \"API bug authentication\"");
    println!("{}", "-".repeat(40));
    let api_memories = hippocampus.retrieve_by_query("API bug authentication", 3)?;
    if api_memories.is_empty() {
        println!("  (no results)");
    }
    for (i, mem) in api_memories.iter().enumerate() {
        let content: String = serde_json::from_value(mem.content.clone())
            .unwrap_or_else(|_| "unknown".to_string());
        let display = if content.len() > 50 { &content[..50] } else { &content };
        println!(
            "  {}. {} (v={:+.2})",
            i + 1,
            display,
            mem.valence.value()
        );
    }

    // Search 3: Production/deployment
    println!("\nQuery: \"production deployment servers\"");
    println!("{}", "-".repeat(40));
    let prod_memories = hippocampus.retrieve_by_query("production deployment servers", 3)?;
    if prod_memories.is_empty() {
        println!("  (no results)");
    }
    for (i, mem) in prod_memories.iter().enumerate() {
        let content: String = serde_json::from_value(mem.content.clone())
            .unwrap_or_else(|_| "unknown".to_string());
        let display = if content.len() > 50 { &content[..50] } else { &content };
        println!(
            "  {}. {} (v={:+.2})",
            i + 1,
            display,
            mem.valence.value()
        );
    }

    // Search 4: Learning/AI
    println!("\nQuery: \"AI consciousness learning transformer\"");
    println!("{}", "-".repeat(40));
    let ai_memories = hippocampus.retrieve_by_query("AI consciousness learning transformer", 3)?;
    if ai_memories.is_empty() {
        println!("  (no results)");
    }
    for (i, mem) in ai_memories.iter().enumerate() {
        let content: String = serde_json::from_value(mem.content.clone())
            .unwrap_or_else(|_| "unknown".to_string());
        let display = if content.len() > 50 { &content[..50] } else { &content };
        println!(
            "  {}. {} (v={:+.2})",
            i + 1,
            display,
            mem.valence.value()
        );
    }

    // Compare with valence-only retrieval
    println!("\n{}", "=".repeat(60));
    println!("📊 COMPARISON: Valence-only retrieval (top 3)\n");
    let valence_only = hippocampus.retrieve(3, true)?;
    for (i, mem) in valence_only.iter().enumerate() {
        let content: String = serde_json::from_value(mem.content.clone())
            .unwrap_or_else(|_| "unknown".to_string());
        let display = if content.len() > 50 { &content[..50] } else { &content };
        println!(
            "  {}. {} (v={:+.2})",
            i + 1,
            display,
            mem.valence.value()
        );
    }

    println!("\n{}", "=".repeat(60));
    println!("✨ Demo complete!");
    println!("\nSemantic search finds memories by keyword relevance,");
    println!("boosted by emotional valence and memory strength.");
    println!("Notice how different queries surface different memories!");

    Ok(())
}
