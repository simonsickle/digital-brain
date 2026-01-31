//! Nervous System Demo
//!
//! Demonstrates signal routing through the brain's nervous system.

use digital_brain::core::nervous_system::{BrainRegion, NervousSystem};
use digital_brain::signal::{BrainSignal, SignalType};

fn main() {
    println!("\n");
    println!("╔═══════════════════════════════════════════════════════════════════════╗");
    println!("║            🧠 DIGITAL BRAIN - NERVOUS SYSTEM DEMO                     ║");
    println!("║                                                                       ║");
    println!("║   Watch signals flow through neural pathways between brain regions.   ║");
    println!("╚═══════════════════════════════════════════════════════════════════════╝\n");

    // Create nervous system
    let mut ns = NervousSystem::new();

    // Show the pathway structure
    println!("📊 NERVOUS SYSTEM PATHWAYS:\n");
    print!("{}", ns.visualize());

    // Demonstrate signal flow
    println!("\n🔬 SIGNAL ROUTING DEMONSTRATION\n");

    // 1. External sensory input enters through Thalamus
    println!("┌─ STEP 1: Sensory Input ────────────────────────────────────────────────┐");
    println!("│ A sensory signal arrives from the external world...");
    let sensory_signal =
        BrainSignal::new("external", SignalType::Sensory, "I see a red apple").with_salience(0.7);

    let success = ns.transmit(BrainRegion::External, BrainRegion::Thalamus, sensory_signal);
    println!(
        "│ External → Thalamus: {}",
        if success {
            "✓ Transmitted"
        } else {
            "✗ Blocked"
        }
    );

    // Get signal at thalamus
    let thalamus_signals = ns.get_signals(BrainRegion::Thalamus);
    if let Some(sig) = thalamus_signals.first() {
        println!(
            "│ Thalamus received: \"{}\" (salience: {:.2})",
            sig.content,
            sig.salience.value()
        );
    }
    println!("└───────────────────────────────────────────────────────────────────────┘\n");

    // 2. Thalamus routes to Amygdala (fast emotional path)
    println!("┌─ STEP 2: Emotional Processing (Fast Path) ────────────────────────────┐");
    let emotional_signal =
        BrainSignal::new("thalamus", SignalType::Sensory, "Red apple - food!").with_salience(0.7);

    let success = ns.transmit(
        BrainRegion::Thalamus,
        BrainRegion::Amygdala,
        emotional_signal,
    );
    println!(
        "│ Thalamus → Amygdala: {}",
        if success {
            "✓ Transmitted"
        } else {
            "✗ Blocked"
        }
    );

    let amygdala_signals = ns.get_signals(BrainRegion::Amygdala);
    if let Some(sig) = amygdala_signals.first() {
        println!("│ Amygdala processing: \"{}\"", sig.content);
        println!("│ (Emotional tagging will be applied)");
    }
    println!("└───────────────────────────────────────────────────────────────────────┘\n");

    // 3. Parallel route to Hippocampus for memory encoding
    println!("┌─ STEP 3: Memory Encoding (Parallel Path) ─────────────────────────────┐");
    let memory_signal = BrainSignal::new("thalamus", SignalType::Sensory, "Red apple in kitchen")
        .with_salience(0.6);

    let success = ns.transmit(
        BrainRegion::Thalamus,
        BrainRegion::Hippocampus,
        memory_signal,
    );
    println!(
        "│ Thalamus → Hippocampus: {}",
        if success {
            "✓ Transmitted"
        } else {
            "✗ Blocked"
        }
    );

    let hippo_signals = ns.get_signals(BrainRegion::Hippocampus);
    if let Some(sig) = hippo_signals.first() {
        println!("│ Hippocampus encoding: \"{}\"", sig.content);
    }
    println!("└───────────────────────────────────────────────────────────────────────┘\n");

    // 4. Working memory loads relevant info
    println!("┌─ STEP 4: Working Memory Loading ───────────────────────────────────────┐");
    let wm_signal =
        BrainSignal::new("thalamus", SignalType::Sensory, "Apple → edible").with_salience(0.65);

    let success = ns.transmit(BrainRegion::Thalamus, BrainRegion::Prefrontal, wm_signal);
    println!(
        "│ Thalamus → Prefrontal: {}",
        if success {
            "✓ Transmitted"
        } else {
            "✗ Blocked"
        }
    );
    println!("│ Working memory now holds: \"Apple → edible\"");
    println!("└───────────────────────────────────────────────────────────────────────┘\n");

    // 5. Signal competes for consciousness
    println!("┌─ STEP 5: Competition for Consciousness ────────────────────────────────┐");
    let conscious_signal =
        BrainSignal::new("prefrontal", SignalType::Attention, "Focus: red apple")
            .with_salience(0.8);

    let success = ns.transmit(
        BrainRegion::Prefrontal,
        BrainRegion::Workspace,
        conscious_signal,
    );
    println!(
        "│ Prefrontal → Workspace: {}",
        if success {
            "✓ Transmitted"
        } else {
            "✗ Blocked"
        }
    );
    println!("│ High salience signal wins competition!");
    println!("│ Content enters consciousness: \"Focus: red apple\"");
    println!("└───────────────────────────────────────────────────────────────────────┘\n");

    // 6. Conscious broadcast
    println!("┌─ STEP 6: Global Broadcast ─────────────────────────────────────────────┐");
    let broadcast = BrainSignal::new(
        "workspace",
        SignalType::Broadcast,
        "CONSCIOUS: red apple observed",
    )
    .with_salience(1.0);

    ns.broadcast(broadcast);
    println!("│ Workspace broadcasts to all regions:");
    println!("│   → Amygdala   (emotional response)");
    println!("│   → Hippocampus (strengthen memory)");
    println!("│   → Prefrontal  (update working memory)");
    println!("│   → DMN         (update self-model)");
    println!("│   → PredictionEngine (check predictions)");
    println!("└───────────────────────────────────────────────────────────────────────┘\n");

    // Show statistics
    let stats = ns.stats();
    println!("╔═══════════════════════════════════════════════════════════════════════╗");
    println!("║                    NERVOUS SYSTEM STATISTICS                          ║");
    println!("╠═══════════════════════════════════════════════════════════════════════╣");
    println!(
        "║  Total pathways:       {:>4}                                          ║",
        stats.total_pathways
    );
    println!(
        "║  Signals routed:       {:>4}                                          ║",
        stats.total_signals_routed
    );
    println!(
        "║  Signals in queues:    {:>4}                                          ║",
        stats.queued_signals
    );
    println!("╚═══════════════════════════════════════════════════════════════════════╝\n");

    // Show pathway modulation
    println!("🔧 NEUROMODULATOR EFFECTS:\n");
    println!("┌─ Pathway Modulation Example ───────────────────────────────────────────┐");

    // Get original strength
    if let Some(pathway) = ns.get_pathway(BrainRegion::Thalamus, BrainRegion::Amygdala) {
        println!(
            "│ Thalamus → Amygdala original strength: {:.2}",
            pathway.effective_strength
        );
    }

    // Simulate norepinephrine increasing this pathway (heightened vigilance)
    println!("│");
    println!("│ Simulating norepinephrine release (heightened vigilance)...");
    ns.apply_modulation(BrainRegion::Thalamus, BrainRegion::Amygdala, 1.3);

    if let Some(pathway) = ns.get_pathway(BrainRegion::Thalamus, BrainRegion::Amygdala) {
        println!(
            "│ Thalamus → Amygdala modulated strength: {:.2}",
            pathway.effective_strength
        );
    }

    println!("│ (Emotional processing pathway is now MORE sensitive)");
    println!("└───────────────────────────────────────────────────────────────────────┘\n");

    println!("✅ Nervous system demo complete!");
    println!("   The nervous system routes signals between brain regions,");
    println!("   enabling the modular architecture to function as a whole.\n");
}
