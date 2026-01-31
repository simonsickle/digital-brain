//! Agent Integration Tests
//!
//! Tests that verify all the new agent systems work together correctly.

use digital_brain::agent::{
    AgentConfig, AgentLoop, CommunicationSystem, IntentType, Percept,
};
use digital_brain::core::{
    ActionCategory, ActionDecision, ActionTemplate, CuriositySystem, Domain, Entity,
    ExpectedOutcome, Goal, Outcome, Priority, TimeHorizon, WorldModel,
};
use std::time::Duration;
use uuid::Uuid;

/// Helper to create a test action
fn make_action(name: &str, category: ActionCategory) -> ActionTemplate {
    ActionTemplate {
        id: Uuid::new_v4(),
        name: name.to_string(),
        description: format!("Test action: {}", name),
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

/// Helper to create a fast agent (no tick delays)
fn fast_agent() -> AgentLoop {
    AgentLoop::new(AgentConfig {
        min_tick_interval: Duration::from_millis(0),
        exploration_rate: 0.0, // Deterministic for tests
        learning_rate: 0.1,
        debug: false,
        ..Default::default()
    })
}

// ============================================================================
// Agent Loop Integration Tests
// ============================================================================

#[test]
fn test_agent_full_cycle() {
    let mut agent = fast_agent();

    // Register actions
    agent.register_action(make_action("work", ActionCategory::Exploitation));
    agent.register_action(make_action("rest", ActionCategory::Rest));

    // Add goal
    agent.add_goal(Goal::new("Complete work").with_priority(Priority::High));

    // Perceive input
    agent.perceive(Percept::text("Start working"));

    // Run cycle
    let result = agent.tick();

    // Should have processed percepts and made a decision
    assert_eq!(result.percepts_processed, 1);
    assert!(matches!(
        result.decision,
        ActionDecision::Execute(_) | ActionDecision::Deliberate { .. }
    ));
}

#[test]
fn test_agent_goal_tracking() {
    let mut agent = fast_agent();

    agent.register_action(make_action("task", ActionCategory::Exploitation));

    let goal_id = agent.add_goal(
        Goal::new("Test goal")
            .with_priority(Priority::High)
            .with_horizon(TimeHorizon::Immediate),
    );

    // Goal should be active
    assert_eq!(agent.goals().stats().active_goals, 1);
    assert!(agent
        .state()
        .active_goals
        .iter()
        .any(|g| g.contains("Test goal")));

    // Complete the goal
    agent.complete_goal(goal_id);

    // Should be completed
    assert_eq!(agent.goals().stats().completed_goals, 1);
}

#[test]
fn test_agent_feedback_affects_neuromodulators() {
    let mut agent = fast_agent();
    agent.register_action(make_action("action", ActionCategory::Exploitation));

    let initial_motivation = agent.neuromodulators().motivation;

    // Positive feedback
    agent.feedback(true, "Great job!");
    agent.tick(); // Process the feedback percept

    // Motivation should increase
    assert!(
        agent.neuromodulators().motivation >= initial_motivation,
        "Motivation should not decrease with positive feedback"
    );
}

#[test]
fn test_agent_error_increases_stress() {
    let mut agent = fast_agent();

    let initial_stress = agent.neuromodulators().stress;

    // Error percept
    agent.perceive(Percept::error("Something went wrong"));
    agent.tick();

    // Stress should increase
    assert!(
        agent.neuromodulators().stress > initial_stress,
        "Stress should increase on error"
    );
}

#[test]
fn test_agent_idle_detection() {
    let mut agent = fast_agent();
    // No actions registered = will become idle

    for _ in 0..15 {
        agent.tick();
    }

    assert!(agent.is_idle(), "Agent should be idle after many cycles with no actions");

    agent.reset_idle();
    assert!(!agent.is_idle(), "Idle should reset");
}

// ============================================================================
// Curiosity System Integration Tests
// ============================================================================

#[test]
fn test_curiosity_explore_vs_exploit() {
    let mut curiosity = CuriositySystem::new();
    curiosity.register_domain(Domain::new("testing"), 0.8);

    // Low dopamine + high learning = explore
    let mut explore_state = digital_brain::core::NeuromodulatorState::default();
    explore_state.dopamine = 0.2;
    explore_state.motivation = 0.2;
    explore_state.learning_depth = 0.9;
    explore_state.exploration_drive = 0.8;

    let explore_prob = curiosity.explore_vs_exploit(&explore_state);
    assert!(
        explore_prob > 0.5,
        "Should favor exploration with low dopamine and high learning"
    );

    // High dopamine + high motivation = exploit
    let mut exploit_state = digital_brain::core::NeuromodulatorState::default();
    exploit_state.dopamine = 0.9;
    exploit_state.motivation = 0.9;
    exploit_state.learning_depth = 0.2;

    let exploit_prob = curiosity.explore_vs_exploit(&exploit_state);
    assert!(
        exploit_prob < 0.5,
        "Should favor exploitation with high dopamine and motivation"
    );
}

#[test]
fn test_curiosity_tracks_competence() {
    let mut curiosity = CuriositySystem::new();
    let domain = Domain::new("rust");
    curiosity.register_domain(domain.clone(), 0.8);

    // Initial competence
    assert_eq!(curiosity.competence(&domain).level, 0.0);

    // Learn through success
    for _ in 0..10 {
        curiosity.update_competence(&domain, true, 0.5);
    }

    assert!(
        curiosity.competence(&domain).level > 0.3,
        "Competence should increase with practice"
    );
    assert!(
        curiosity.competence(&domain).confidence > 0.4,
        "Confidence should increase with attempts"
    );
}

// ============================================================================
// World Model Integration Tests
// ============================================================================

#[test]
fn test_world_model_entity_lifecycle() {
    let mut world = WorldModel::new();

    // Add entity
    let id = world.add_entity(Entity::new("test", "TestEntity").with_property("value", 42.0));

    assert!(world.get_entity(id).is_some());

    // Update property
    world.update_entity_property(id, "value", 100.0);
    assert_eq!(
        world.get_entity(id).unwrap().get_property("value"),
        Some(&digital_brain::core::PropertyValue::Number(100.0))
    );

    // Remove entity
    world.remove_entity(id);
    assert!(world.get_entity(id).is_none());
}

#[test]
fn test_world_model_prediction() {
    let mut world = WorldModel::new();

    let entity = Entity::new("stock", "AAPL").with_property("price", 150.0);
    let id = world.add_entity(entity);

    // Make prediction
    let prediction = digital_brain::core::WorldPrediction::new(
        "Price will rise",
        digital_brain::core::PropertyValue::Number(160.0),
    )
    .for_entity(id, "price")
    .with_confidence(0.7);

    world.predict(prediction);
    assert_eq!(world.pending_predictions().len(), 1);
}

// ============================================================================
// Communication System Integration Tests
// ============================================================================

#[test]
fn test_communication_priority_queue() {
    let mut comm = CommunicationSystem::new();

    // Queue messages with different priorities
    comm.inform("Low priority");
    comm.warn("High priority"); // Warns have higher base urgency

    // Next should be the warning (higher urgency)
    let next = comm.next_to_send().unwrap();
    assert_eq!(next.intent_type, IntentType::Warn);
}

#[test]
fn test_communication_style_modulation() {
    let comm = CommunicationSystem::new();

    // High stress state
    let mut state = digital_brain::core::NeuromodulatorState::default();
    state.stress = 0.9;
    state.cooperativeness = 0.9;

    let modulated = comm.modulated_style(&state);

    // Should be more direct and warmer
    assert!(
        modulated.directness > comm.style().directness,
        "High stress should increase directness"
    );
    assert!(
        modulated.warmth > comm.style().warmth,
        "High cooperativeness should increase warmth"
    );
}

#[test]
fn test_communication_conversation_tracking() {
    let mut comm = CommunicationSystem::new();

    // Receive message
    comm.receive_message("Hello", "user");

    // Send response
    comm.inform("Hi there!");
    comm.next_to_send();

    // Should have 2 messages in history
    assert_eq!(comm.history().count(), 2);
    assert_eq!(comm.stats().total_messages_received, 1);
    assert_eq!(comm.stats().total_messages_sent, 1);
}

// ============================================================================
// Cross-System Integration Tests
// ============================================================================

#[test]
fn test_percept_affects_agent_state() {
    let mut agent = fast_agent();
    agent.register_action(make_action("action", ActionCategory::Exploitation));

    // Negative feedback
    agent.perceive(Percept::feedback("That was wrong", -0.7).with_salience(0.9));

    let initial_dopamine = agent.neuromodulators().dopamine;
    agent.tick();

    // Negative valence should decrease dopamine
    assert!(
        agent.neuromodulators().dopamine <= initial_dopamine,
        "Negative feedback should not increase dopamine"
    );
}

#[test]
fn test_goal_decomposition() {
    let mut agent = fast_agent();

    let parent = Goal::new("Big goal").with_priority(Priority::High);
    let parent_id = agent.add_goal(parent);

    let subgoals = vec![Goal::new("Sub 1"), Goal::new("Sub 2"), Goal::new("Sub 3")];

    let sub_ids = agent.decompose_goal(parent_id, subgoals);

    assert_eq!(sub_ids.len(), 3);
    assert_eq!(agent.goals().stats().total_goals, 4); // 1 parent + 3 children
}

#[test]
fn test_multiple_ticks() {
    let mut agent = fast_agent();
    agent.register_action(make_action("action1", ActionCategory::Exploitation));
    agent.register_action(make_action("action2", ActionCategory::Exploration));

    let results = agent.run_ticks(20);

    assert_eq!(results.len(), 20);
    assert_eq!(agent.stats().total_cycles, 20);
}

#[test]
fn test_world_state_tracking() {
    let mut agent = fast_agent();

    agent.update_world("location", "office");
    agent.update_world("time_of_day", "morning");

    assert_eq!(agent.world_state().len(), 2);
    assert_eq!(
        agent.world_state().get("location"),
        Some(&"office".to_string())
    );
}
