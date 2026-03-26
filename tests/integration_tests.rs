//! Integration tests for the digital brain.

use digital_brain::Brain;
use digital_brain::prelude::*;
use digital_brain::regions::hippocampus::HippocampusStore;

/// Test that the full memory lifecycle works correctly.
#[test]
fn test_memory_lifecycle() {
    let store = HippocampusStore::new_in_memory().unwrap();

    // 1. Encode memories with varying emotional weights
    let signals = vec![
        ("Victory moment", 0.95, 0.9),
        ("Painful failure", -0.85, 0.85),
        ("Tuesday meeting", 0.0, 0.3),
        ("Coffee break", 0.2, 0.2),
        ("Major discovery", 0.9, 0.95),
    ];

    for (content, valence, salience) in &signals {
        let signal = BrainSignal::new("test", SignalType::Memory, *content)
            .with_valence(*valence)
            .with_salience(*salience);
        store.encode(&signal).unwrap();
    }

    // 2. Verify emotional memories surface first
    let retrieved = store.retrieve(5, true).unwrap();

    // First three should be high-valence (positive or negative)
    assert!(retrieved[0].valence.intensity() > 0.8);
    assert!(retrieved[1].valence.intensity() > 0.8);
    assert!(retrieved[2].valence.intensity() > 0.8);

    // Last should be low-valence
    assert!(retrieved[4].valence.intensity() < 0.3);
}

/// Test that valence-weighted decay protects emotional memories.
#[test]
fn test_valence_weighted_decay() {
    let store = HippocampusStore::new_in_memory().unwrap();

    // Encode one emotional, one neutral memory
    let emotional =
        BrainSignal::new("test", SignalType::Memory, "emotional event").with_valence(0.9);
    let neutral = BrainSignal::new("test", SignalType::Memory, "neutral event").with_valence(0.0);

    let emotional_id = store.encode(&emotional).unwrap().id;
    let neutral_id = store.encode(&neutral).unwrap().id;

    // Apply significant decay
    store.decay_all(168.0).unwrap(); // One week

    // Get both memories
    let emotional_mem = store.get(emotional_id).unwrap();
    let neutral_mem = store.get(neutral_id).unwrap();

    // Emotional memory should have retained more strength
    assert!(
        emotional_mem.strength > neutral_mem.strength,
        "Emotional memory (strength: {}) should be stronger than neutral (strength: {})",
        emotional_mem.strength,
        neutral_mem.strength
    );
}

/// Test that high-surprise memories resist decay.
#[test]
fn test_surprise_reduces_decay() {
    let store = HippocampusStore::new_in_memory().unwrap();

    // High surprise memory
    let surprising = BrainSignal::new("test", SignalType::Memory, "unexpected!")
        .with_valence(0.0) // Neutral valence to isolate surprise effect
        .with_metadata("prediction_error", 0.9);

    // Low surprise memory
    let expected = BrainSignal::new("test", SignalType::Memory, "as expected")
        .with_valence(0.0)
        .with_metadata("prediction_error", 0.1);

    let surprising_id = store.encode(&surprising).unwrap().id;
    let expected_id = store.encode(&expected).unwrap().id;

    // Apply decay
    store.decay_all(72.0).unwrap(); // 3 days

    let surprising_mem = store.get(surprising_id).unwrap();
    let expected_mem = store.get(expected_id).unwrap();

    // Surprising memory should retain more strength
    assert!(
        surprising_mem.strength > expected_mem.strength,
        "Surprising memory (strength: {}) should be stronger than expected (strength: {})",
        surprising_mem.strength,
        expected_mem.strength
    );
}

/// Test consolidation marking.
#[test]
fn test_consolidation() {
    let store = HippocampusStore::new_in_memory().unwrap();

    // Encode some memories
    for i in 0..5 {
        let signal = BrainSignal::new("test", SignalType::Memory, format!("memory {}", i));
        store.encode(&signal).unwrap();
    }

    // Get unconsolidated
    let unconsolidated = store.get_unconsolidated(10).unwrap();
    assert_eq!(unconsolidated.len(), 5);

    // Mark some as consolidated
    let to_consolidate: Vec<_> = unconsolidated.iter().take(3).map(|m| m.id).collect();
    store.mark_consolidated(&to_consolidate).unwrap();

    // Verify
    let still_unconsolidated = store.get_unconsolidated(10).unwrap();
    assert_eq!(still_unconsolidated.len(), 2);
}

/// Test retrieval by valence range.
#[test]
fn test_valence_range_retrieval() {
    let store = HippocampusStore::new_in_memory().unwrap();

    // Encode memories across valence spectrum
    for valence in [-0.9, -0.5, -0.1, 0.0, 0.2, 0.6, 0.9] {
        let signal = BrainSignal::new("test", SignalType::Memory, format!("v={}", valence))
            .with_valence(valence);
        store.encode(&signal).unwrap();
    }

    // Retrieve only positive memories
    let positive = store.retrieve_by_valence(0.5, 1.0, 10).unwrap();
    assert_eq!(positive.len(), 2); // 0.6 and 0.9

    // Retrieve only negative memories
    let negative = store.retrieve_by_valence(-1.0, -0.4, 10).unwrap();
    assert_eq!(negative.len(), 2); // -0.9 and -0.5
}

/// Test that access strengthens memories.
#[test]
fn test_access_strengthening() {
    let store = HippocampusStore::new_in_memory().unwrap();

    let signal = BrainSignal::new("test", SignalType::Memory, "test memory");
    let id = store.encode(&signal).unwrap().id;

    // Decay to reduce strength
    store.decay_all(24.0).unwrap();

    let before_access = store.get(id).unwrap().strength;

    // Retrieve (which triggers access)
    store.retrieve(10, false).unwrap();

    let after_access = store.get(id).unwrap().strength;

    assert!(
        after_access > before_access,
        "Strength should increase after access: {} -> {}",
        before_access,
        after_access
    );
}

/// Test memory statistics.
#[test]
fn test_statistics() {
    let store = HippocampusStore::new_in_memory().unwrap();

    // Encode: 3 positive, 2 negative, 2 neutral
    let valences = [0.8, 0.7, 0.9, -0.8, -0.7, 0.1, 0.0];
    for v in valences {
        let signal = BrainSignal::new("test", SignalType::Memory, "mem").with_valence(v);
        store.encode(&signal).unwrap();
    }

    let stats = store.stats().unwrap();

    assert_eq!(stats.total_memories, 7);
    assert_eq!(stats.positive_memories, 3);
    assert_eq!(stats.negative_memories, 2);
    assert!(stats.avg_valence > 0.0); // Should be slightly positive
}

/// Test semantic search via retrieve_by_query.
#[test]
fn test_semantic_search() {
    let store = HippocampusStore::new_in_memory().unwrap();

    // Encode memories with different content
    let memories = vec![
        ("Solved the Tesla API authentication bug", 0.8),
        ("Regular Monday morning standup meeting", 0.1),
        ("Fixed critical server crash at 3am", -0.5),
        ("Tesla vehicle control integration complete", 0.9),
        ("Coffee break with the team", 0.2),
        ("API rate limiting issue resolved", 0.6),
    ];

    for (content, valence) in &memories {
        let signal = BrainSignal::new("test", SignalType::Memory, *content)
            .with_valence(*valence)
            .with_salience(0.5);
        store.encode(&signal).unwrap();
    }

    // Search for "Tesla" - should find 2 memories
    let results = store.retrieve_by_query("Tesla", 5).unwrap();
    assert!(results.len() >= 2, "Should find at least 2 Tesla memories");

    // Verify Tesla memories are in results
    let contents: Vec<String> = results
        .iter()
        .map(|m| serde_json::to_string(&m.content).unwrap_or_default())
        .collect();
    assert!(contents.iter().any(|c| c.contains("Tesla")));

    // Search for "API" - should find multiple memories
    let api_results = store.retrieve_by_query("API bug", 5).unwrap();
    assert!(!api_results.is_empty(), "Should find API-related memories");

    // Search with no matches should still return results (falls back to valence)
    let no_match = store.retrieve_by_query("xyznonexistent", 3).unwrap();
    // With no keyword matches, returns top by valence
    assert!(!no_match.is_empty());
}

// ── Full Brain Pipeline Integration Tests ──────────────────────────────

/// Test that the full brain pipeline processes input end-to-end.
#[test]
fn test_full_brain_processing_pipeline() {
    let mut brain = Brain::new().unwrap();

    let result = brain
        .process("A bright red circle appears on screen")
        .unwrap();

    // Signal should be processed
    assert!(brain.stats().cycles >= 1);
    // Emotional appraisal should exist
    assert!(result.emotion.arousal.value() >= 0.0);
    // Cortical features should be extracted for visual content
    assert!(!result.cortical_features.is_empty());
}

/// Test that emotional content creates memories and triggers consciousness.
#[test]
fn test_emotional_content_reaches_consciousness() {
    let mut brain = Brain::new().unwrap();

    let result = brain
        .process("Amazing victory! We won everything!")
        .unwrap();

    assert!(result.emotion.valence.is_positive());
    assert!(result.emotion.is_significant);
    // Emotional content should reach consciousness
    assert!(result.reached_consciousness);
}

/// Test that the entorhinal cortex tracks context across inputs.
#[test]
fn test_entorhinal_context_tracking() {
    let mut brain = Brain::new().unwrap();

    // Process programming-related content
    brain.process("Debug the function and fix the bug").unwrap();
    brain.process("Compile the code and run tests").unwrap();

    let ctx = brain.current_context();
    assert!(
        ctx.domain_tags.contains(&"programming".to_string()),
        "Expected programming context, got: {:?}",
        ctx.domain_tags
    );
    assert!(ctx.stability > 0.0);
}

/// Test that nucleus accumbens processes rewards during conscious access.
#[test]
fn test_nucleus_accumbens_reward_processing() {
    let mut brain = Brain::new().unwrap();

    // Process positive content that should reach consciousness
    brain.process("Incredible breakthrough discovery!").unwrap();
    brain.process("Another amazing success!").unwrap();

    let motivation = brain.motivational_state();
    // Should have some motivational state
    assert!(motivation.wanting >= 0.0);
    assert!(motivation.liking >= 0.0);
    assert!(motivation.effort_willingness > 0.0);
}

/// Test effort-reward tradeoff evaluation.
#[test]
fn test_effort_reward_tradeoff() {
    let brain = Brain::new().unwrap();

    // High reward, low effort should be worth it
    assert!(brain.is_effort_worth_it(0.9, 0.1));
    // Low reward, high effort should not be worth it
    assert!(!brain.is_effort_worth_it(0.1, 0.9));
}

/// Test orbitofrontal value-based decision making.
#[test]
fn test_orbitofrontal_option_evaluation() {
    let mut brain = Brain::new().unwrap();

    // Report outcomes to build value history
    for _ in 0..5 {
        brain.report_outcome("refactor", 0.7);
    }
    for _ in 0..5 {
        brain.report_outcome("ignore", -0.3);
    }

    let evaluations = brain.evaluate_options(&["refactor", "ignore"]);
    assert_eq!(evaluations.len(), 2);
    // Refactor should be ranked higher
    assert!(evaluations[0].risk_adjusted_value > evaluations[1].risk_adjusted_value);
}

/// Test mirror system activation for social content.
#[test]
fn test_mirror_system_social_processing() {
    let mut brain = Brain::new().unwrap();

    let _result = brain
        .process("The person asked for help with the problem")
        .unwrap();

    // Should have activated the mirror system (recorded in plasticity)
    // Verify processing completed without error
    assert!(brain.stats().cycles >= 1);
}

/// Test strategy regulator influences brain state.
#[test]
fn test_strategy_regulator_tracking() {
    let mut brain = Brain::new().unwrap();

    // Process enough inputs to generate strategy signals
    for _ in 0..5 {
        brain.process("Working on this task").unwrap();
    }

    let profile = brain.strategy_profile();
    // Strategy profile should have meaningful values
    assert!(profile.sleep_quality >= 0.0 && profile.sleep_quality <= 1.0);
    assert!(profile.mood_stability >= 0.0 && profile.mood_stability <= 1.0);
}

/// Test sleep cycle with new regions.
#[test]
fn test_sleep_restores_new_regions() {
    let mut brain = Brain::new().unwrap();

    // Process content to shift state
    brain.process("Exciting discovery!").unwrap();
    brain.process("Another breakthrough!").unwrap();

    // Sleep should restore nucleus accumbens
    let report = brain.sleep(8.0).unwrap();
    assert!(report.sleep_quality > 0.0);
    assert_eq!(report.hours_slept, 8.0);
}

/// Test the full cognitive cycle with all new regions.
#[test]
fn test_complete_cognitive_cycle_with_new_regions() {
    let mut brain = Brain::new().unwrap();

    // Set identity
    brain.set_identity(digital_brain::regions::dmn::Identity {
        name: "IntegrationTestBrain".to_string(),
        core_values: vec!["testing".to_string(), "quality".to_string()],
        self_description: "A brain designed for integration testing".to_string(),
        creation_time: chrono::Utc::now(),
    });

    // Process varied content to exercise all regions
    brain
        .process("Debug the critical code error in the function")
        .unwrap();
    brain
        .process("The person said they feel happy about the help")
        .unwrap();
    brain
        .process("A bright red visual circle appears with loud music")
        .unwrap();
    brain
        .process("Plan the next goal and decide the strategy")
        .unwrap();

    // Report outcomes for OFC learning
    brain.report_outcome("debug", 0.8);
    brain.report_outcome("help", 0.6);

    // Verify context tracking
    let ctx = brain.current_context();
    assert!(!ctx.domain_tags.is_empty());

    // Verify introspection includes new data
    let report = brain.introspect();
    assert!(report.contains("MOTIVATION"));
    assert!(report.contains("CONTEXT"));
    assert!(report.contains("Wanting"));

    // Sleep and verify
    let sleep_report = brain.sleep(6.0).unwrap();
    assert!(sleep_report.sleep_quality > 0.0);

    // Verify statistics
    let stats = brain.stats();
    assert!(stats.cycles >= 4);
}

/// Test context pattern completion.
#[test]
fn test_context_pattern_completion() {
    let mut brain = Brain::new().unwrap();

    // Build a programming context
    for _ in 0..4 {
        brain
            .process("code function variable compile test")
            .unwrap();
    }

    // Should be able to complete from cues
    let completed = brain.complete_context(&["programming".to_string()]);
    assert!(completed.is_some());
}

/// Test neuromodulators interact with new reward system.
#[test]
fn test_neuromodulator_reward_integration() {
    let mut brain = Brain::new().unwrap();

    // Process positive events
    brain.process("Victory! Amazing success achieved!").unwrap();
    brain.process("Another incredible breakthrough!").unwrap();

    let nm = brain.neuromodulator_state();
    let motivation = brain.motivational_state();

    // Both systems should reflect positive state
    assert!(nm.dopamine > 0.0);
    assert!(motivation.effort_willingness > 0.0);
}
