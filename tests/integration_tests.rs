//! Integration tests for the digital brain.

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
    let contents: Vec<String> = results.iter()
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
