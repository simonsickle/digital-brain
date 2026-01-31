//! Full brain integration tests.
//!
//! Tests the complete flow: signal → amygdala → workspace → hippocampus

use digital_brain::core::prediction::{Prediction, PredictionEngine};
use digital_brain::core::workspace::{GlobalWorkspace, WorkspaceConfig};
use digital_brain::prelude::*;
use digital_brain::regions::amygdala::Amygdala;
use digital_brain::regions::hippocampus::HippocampusStore;

/// Test the complete cognitive cycle.
#[test]
fn test_complete_cognitive_cycle() {
    // Initialize brain components
    let mut amygdala = Amygdala::new();
    let mut workspace = GlobalWorkspace::with_config(WorkspaceConfig {
        capacity: 5,
        broadcast_duration: 3,
        ..Default::default()
    });
    let hippocampus = HippocampusStore::new_in_memory().unwrap();
    let _prediction_engine = PredictionEngine::new();

    // Register modules with workspace
    workspace.register_module("hippocampus");
    workspace.register_module("amygdala");
    workspace.register_module("prediction");

    // Simulate incoming sensory signals
    let signals = vec![
        "Great news! The project succeeded!",
        "Minor update: temperature is normal",
        "DANGER: System failure detected!",
        "Coffee break time",
        "Unexpected discovery in the data!",
    ];

    let mut broadcasts = Vec::new();

    for content in signals {
        // 1. Create sensory signal
        let signal = BrainSignal::new("sensory", SignalType::Sensory, content);

        // 2. Emotional tagging by amygdala
        let tagged_signal = amygdala.tag_signal(signal);

        // 3. Submit to workspace for competition
        workspace.submit(tagged_signal.clone());

        // 4. Store in hippocampus (all signals, not just broadcast)
        hippocampus.encode(&tagged_signal).unwrap();
    }

    // 5. Process workspace cycle
    let cycle_broadcasts = workspace.process_cycle();
    broadcasts.extend(cycle_broadcasts);

    // Verify emotional signals won the competition
    assert!(!broadcasts.is_empty());

    // The danger signal should have high salience
    let danger_broadcast = broadcasts
        .iter()
        .find(|b| b.signal.content.to_string().contains("DANGER"));
    assert!(danger_broadcast.is_some());

    // Verify hippocampus has all memories
    let stats = hippocampus.stats().unwrap();
    assert_eq!(stats.total_memories, 5);

    // Verify emotional memories have higher retrieval scores
    let memories = hippocampus.retrieve(5, true).unwrap();
    assert!(memories[0].valence.intensity() > 0.3); // First should be emotional
}

/// Test prediction and surprise flow.
#[test]
fn test_prediction_surprise_flow() {
    let mut prediction_engine = PredictionEngine::new();
    let mut workspace = GlobalWorkspace::new();
    let hippocampus = HippocampusStore::new_in_memory().unwrap();

    // Make a confident prediction
    let prediction = Prediction::new("weather_module", "temperature", 72.0, 0.9);
    let pred_id = prediction_engine.predict(prediction);

    // Actual outcome differs significantly
    let error = prediction_engine.evaluate(pred_id, 95.0, 0.8).unwrap();

    // High confidence + high error = surprising
    assert!(error.is_surprising());

    // Convert to signal and broadcast
    let error_signal = error.to_signal("prediction_engine");
    workspace.submit(error_signal.clone());

    let broadcasts = workspace.process_cycle();
    assert!(!broadcasts.is_empty());

    // Store surprising event in memory
    hippocampus.encode(&error_signal).unwrap();

    // Verify memory has high salience
    let memories = hippocampus.retrieve(1, false).unwrap();
    assert!(memories[0].salience.value() > 0.5);
}

/// Test emotional learning affects future processing.
#[test]
fn test_emotional_learning() {
    let mut amygdala = Amygdala::new();

    // Initially, "clippy" has no emotional association
    let signal1 = BrainSignal::new("test", SignalType::Sensory, "clippy lint warnings");
    let appraisal1 = amygdala.appraise(&signal1);
    assert!(appraisal1.valence.intensity() < 0.3); // Neutral

    // Learn that "clippy" is annoying
    amygdala.learn_association("clippy", -0.6);

    // Now it should trigger negative emotion
    let signal2 = BrainSignal::new("test", SignalType::Sensory, "clippy found issues");
    let appraisal2 = amygdala.appraise(&signal2);
    assert!(appraisal2.valence.is_negative());
}

/// Test workspace capacity limits create attentional bottleneck.
#[test]
fn test_attention_bottleneck() {
    let mut workspace = GlobalWorkspace::with_config(WorkspaceConfig {
        capacity: 3,
        broadcast_duration: 2,
        ..Default::default()
    });

    // Submit many signals with varying salience
    for i in 0..10 {
        let salience = 0.4 + (i as f64 * 0.05); // 0.4 to 0.85
        let signal = BrainSignal::new("test", SignalType::Sensory, format!("signal_{}", i))
            .with_salience(salience);
        workspace.submit(signal);
    }

    // Only top 3 should broadcast
    let broadcasts = workspace.process_cycle();
    assert_eq!(broadcasts.len(), 3);

    // They should be the highest salience ones
    for broadcast in &broadcasts {
        assert!(broadcast.winning_score > 0.7);
    }
}

/// Test emotional decay over time.
#[test]
fn test_emotional_state_dynamics() {
    let mut amygdala = Amygdala::new();

    // Process highly emotional signal
    let signal = BrainSignal::new("test", SignalType::Sensory, "amazing incredible success!")
        .with_arousal(0.9);
    amygdala.appraise(&signal);

    let (v1, a1) = amygdala.current_state();
    assert!(v1.is_positive());
    assert!(a1.value() > 0.6);

    // Apply decay cycles
    for _ in 0..20 {
        amygdala.decay();
    }

    let (v2, a2) = amygdala.current_state();

    // Should have decayed toward neutral
    assert!(v2.intensity() < v1.intensity());
    assert!((a2.value() - 0.5).abs() < (a1.value() - 0.5).abs());
}

/// Test memory consolidation flow.
#[test]
fn test_consolidation_cycle() {
    let hippocampus = HippocampusStore::new_in_memory().unwrap();

    // Encode memories during "wake" period
    for i in 0..10 {
        let valence = if i % 2 == 0 { 0.7 } else { -0.3 };
        let signal = BrainSignal::new("test", SignalType::Memory, format!("memory_{}", i))
            .with_valence(valence)
            .with_salience(0.5 + (i as f64 * 0.03));
        hippocampus.encode(&signal).unwrap();
    }

    // Get unconsolidated memories
    let unconsolidated = hippocampus.get_unconsolidated(100).unwrap();
    assert_eq!(unconsolidated.len(), 10);

    // Simulate sleep consolidation - consolidate high-salience memories
    let to_consolidate: Vec<_> = unconsolidated
        .iter()
        .filter(|m| m.salience.value() > 0.6)
        .map(|m| m.id)
        .collect();

    hippocampus.mark_consolidated(&to_consolidate).unwrap();

    // Verify partial consolidation
    let still_unconsolidated = hippocampus.get_unconsolidated(100).unwrap();
    assert!(still_unconsolidated.len() < 10);
}

/// Test the interplay between prediction error and memory encoding.
#[test]
fn test_surprise_strengthens_memory() {
    let hippocampus = HippocampusStore::new_in_memory().unwrap();

    // High surprise memory
    let surprising = BrainSignal::new("test", SignalType::Memory, "unexpected result")
        .with_metadata("prediction_error", 0.9);
    let surprising_id = hippocampus.encode(&surprising).unwrap().id;

    // Low surprise memory
    let expected = BrainSignal::new("test", SignalType::Memory, "expected result")
        .with_metadata("prediction_error", 0.1);
    let expected_id = hippocampus.encode(&expected).unwrap().id;

    // Apply decay
    hippocampus.decay_all(72.0).unwrap();

    // Surprising memory should retain more strength
    let surprising_mem = hippocampus.get(surprising_id).unwrap();
    let expected_mem = hippocampus.get(expected_id).unwrap();

    assert!(
        surprising_mem.strength > expected_mem.strength,
        "Surprising: {}, Expected: {}",
        surprising_mem.strength,
        expected_mem.strength
    );
}
