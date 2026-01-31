//! Full Agent Demo
//!
//! Demonstrates all the new agent systems working together:
//! - Agent Loop (perception-action cycle)
//! - Action Selection (neuromodulator-driven decisions)
//! - Goal Management (hierarchical objectives)
//! - Curiosity System (intrinsic motivation)
//! - World Model (entity tracking)
//! - Communication (structured outputs)

use digital_brain::agent::{
    AgentConfig, AgentLoop, CommunicationSystem, IntentType, Percept,
};
use digital_brain::core::{
    ActionCategory, ActionTemplate, CuriositySystem, Domain, Entity, ExpectedOutcome,
    Goal, Outcome, Priority, Relationship, RelationType, TimeHorizon, WorldModel,
};
use std::time::Duration;
use uuid::Uuid;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║           DIGITAL BRAIN - FULL AGENT DEMO                    ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Create the agent with custom config
    let config = AgentConfig {
        min_tick_interval: Duration::from_millis(0), // Fast for demo
        exploration_rate: 0.1,
        learning_rate: 0.15,
        debug: true,
        ..Default::default()
    };
    let mut agent = AgentLoop::new(config);

    // Create supporting systems
    let mut curiosity = CuriositySystem::new();
    let mut world = WorldModel::new();
    let mut comm = CommunicationSystem::new();

    println!("1️⃣  SETTING UP WORLD MODEL");
    println!("─────────────────────────────────────────");
    setup_world_model(&mut world);
    println!();

    println!("2️⃣  REGISTERING ACTIONS");
    println!("─────────────────────────────────────────");
    register_actions(&mut agent);
    println!();

    println!("3️⃣  SETTING UP GOALS");
    println!("─────────────────────────────────────────");
    setup_goals(&mut agent);
    println!();

    println!("4️⃣  INITIALIZING CURIOSITY DOMAINS");
    println!("─────────────────────────────────────────");
    setup_curiosity(&mut curiosity);
    println!();

    println!("5️⃣  RUNNING AGENT LOOP");
    println!("─────────────────────────────────────────");
    run_agent_loop(&mut agent, &mut curiosity, &mut world, &mut comm);
    println!();

    println!("6️⃣  FINAL STATUS");
    println!("─────────────────────────────────────────");
    print_final_status(&agent, &curiosity, &world, &comm);
}

fn setup_world_model(world: &mut WorldModel) {
    // Add some entities
    let user = world.add_entity(
        Entity::new("person", "User")
            .with_property("role", "human")
            .with_property("mood", "neutral")
            .with_valence(0.5)
            .with_tag("primary"),
    );
    println!("  Added entity: User (person)");

    let project = world.add_entity(
        Entity::new("project", "DigitalBrain")
            .with_property("status", "active")
            .with_property("progress", 0.3)
            .with_valence(0.7)
            .with_tag("work"),
    );
    println!("  Added entity: DigitalBrain (project)");

    let task = world.add_entity(
        Entity::new("task", "WriteDocumentation")
            .with_property("priority", "high")
            .with_property("estimated_hours", 4.0)
            .with_valence(0.2),
    );
    println!("  Added entity: WriteDocumentation (task)");

    // Add relationships
    world.add_relationship(Relationship::new(user, project, RelationType::Owns));
    world.add_relationship(Relationship::new(project, task, RelationType::Contains));
    println!("  Added relationships: User owns Project, Project contains Task");

    println!(
        "  World model: {} entities, {} relationships",
        world.stats().total_entities,
        world.stats().total_relationships
    );
}

fn register_actions(agent: &mut AgentLoop) {
    // Code action
    let code_action = ActionTemplate {
        id: Uuid::new_v4(),
        name: "write_code".to_string(),
        description: "Write code for the project".to_string(),
        preconditions: vec![],
        expected_outcomes: vec![ExpectedOutcome {
            outcome: Outcome::success("Code written", 0.7)
                .with_goal("complete project", 0.1),
            probability: 0.85,
        }],
        effort_cost: 0.6,
        time_cost: 30,
        category: ActionCategory::Exploitation,
        tags: vec!["productive".to_string()],
    };
    agent.register_action(code_action);
    println!("  Registered: write_code (Exploitation)");

    // Research action
    let research_action = ActionTemplate {
        id: Uuid::new_v4(),
        name: "research_topic".to_string(),
        description: "Research a new topic".to_string(),
        preconditions: vec![],
        expected_outcomes: vec![ExpectedOutcome {
            outcome: Outcome::success("Learned something new", 0.5),
            probability: 0.9,
        }],
        effort_cost: 0.3,
        time_cost: 15,
        category: ActionCategory::Exploration,
        tags: vec!["learning".to_string()],
    };
    agent.register_action(research_action);
    println!("  Registered: research_topic (Exploration)");

    // Rest action
    let rest_action = ActionTemplate {
        id: Uuid::new_v4(),
        name: "take_break".to_string(),
        description: "Take a short break".to_string(),
        preconditions: vec![],
        expected_outcomes: vec![ExpectedOutcome {
            outcome: Outcome::success("Refreshed", 0.3),
            probability: 0.95,
        }],
        effort_cost: 0.1,
        time_cost: 10,
        category: ActionCategory::Rest,
        tags: vec!["recovery".to_string()],
    };
    agent.register_action(rest_action);
    println!("  Registered: take_break (Rest)");

    // Communicate action
    let comm_action = ActionTemplate {
        id: Uuid::new_v4(),
        name: "send_update".to_string(),
        description: "Send a progress update".to_string(),
        preconditions: vec![],
        expected_outcomes: vec![ExpectedOutcome {
            outcome: Outcome::success("Update sent", 0.4),
            probability: 0.95,
        }],
        effort_cost: 0.2,
        time_cost: 5,
        category: ActionCategory::Communication,
        tags: vec!["social".to_string()],
    };
    agent.register_action(comm_action);
    println!("  Registered: send_update (Communication)");

    println!("  Total actions: {}", agent.actions().actions().len());
}

fn setup_goals(agent: &mut AgentLoop) {
    // Main goal
    let main_goal = Goal::new("Complete the digital brain project")
        .with_priority(Priority::High)
        .with_horizon(TimeHorizon::MediumTerm)
        .with_effort(0.8);
    let main_id = agent.add_goal(main_goal);
    println!("  Added goal: Complete digital brain (High priority)");

    // Subgoals
    let subgoals = vec![
        Goal::new("Implement core modules")
            .with_priority(Priority::High)
            .with_horizon(TimeHorizon::ShortTerm),
        Goal::new("Write documentation")
            .with_priority(Priority::Medium)
            .with_horizon(TimeHorizon::ShortTerm),
        Goal::new("Create examples")
            .with_priority(Priority::Medium)
            .with_horizon(TimeHorizon::ShortTerm),
    ];

    let sub_ids = agent.decompose_goal(main_id, subgoals);
    println!("  Decomposed into {} subgoals", sub_ids.len());

    println!(
        "  Total goals: {} active",
        agent.goals().stats().active_goals
    );
}

fn setup_curiosity(curiosity: &mut CuriositySystem) {
    // Register domains of interest
    curiosity.register_domain(Domain::new("rust_programming"), 0.6);
    println!("  Domain: rust_programming (uncertainty: 0.6)");

    curiosity.register_domain(Domain::new("neural_architectures"), 0.9);
    println!("  Domain: neural_architectures (uncertainty: 0.9)");

    curiosity.register_domain(Domain::new("consciousness"), 0.95);
    println!("  Domain: consciousness (uncertainty: 0.95)");

    curiosity.register_domain(Domain::new("testing"), 0.3);
    println!("  Domain: testing (uncertainty: 0.3)");

    // Show most uncertain
    if let Some(domain) = curiosity.most_uncertain_domain() {
        println!("  Most uncertain domain: {}", domain.0);
    }
}

fn run_agent_loop(
    agent: &mut AgentLoop,
    curiosity: &mut CuriositySystem,
    world: &mut WorldModel,
    comm: &mut CommunicationSystem,
) {
    // Simulate some input
    agent.perceive(Percept::text("Start working on the project"));
    agent.perceive(
        Percept::feedback("Good progress so far!", 0.6)
            .with_salience(0.7),
    );

    println!("  Received 2 percepts");

    // Run several ticks
    println!("\n  Running 10 agent ticks...\n");

    for i in 1..=10 {
        let result = agent.tick();

        // Print decision
        let decision_str = match &result.decision {
            digital_brain::core::ActionDecision::Execute(id) => {
                let name = agent
                    .actions()
                    .get_action(*id)
                    .map(|a| a.name.as_str())
                    .unwrap_or("unknown");
                format!("Execute: {}", name)
            }
            digital_brain::core::ActionDecision::Wait { reason, .. } => {
                format!("Wait: {}", reason)
            }
            digital_brain::core::ActionDecision::Deliberate { options, .. } => {
                format!("Deliberate: {} options", options.len())
            }
            digital_brain::core::ActionDecision::Explore { domain, .. } => {
                format!("Explore: {}", domain)
            }
            digital_brain::core::ActionDecision::NoAction { reason } => {
                format!("NoAction: {}", reason)
            }
        };

        println!("  Tick {}: {} ({:?})", i, decision_str, result.cycle_duration);

        // Simulate some learning
        if i % 3 == 0 {
            curiosity.update_competence(&Domain::new("rust_programming"), true, 0.5);
        }

        // Update world model periodically
        if i == 5 {
            world.update_entity_property(
                world.find_by_name("DigitalBrain").first().unwrap().id,
                "progress",
                0.5,
            );
            println!("  [World updated: project progress -> 0.5]");
        }

        // Generate communication
        if i == 7 {
            let intent = comm.generate_response(
                "Making good progress on the implementation",
                IntentType::Inform,
                agent.neuromodulators(),
            );
            comm.queue(intent);
            println!("  [Queued communication: Inform]");
        }
    }

    // Send pending communication
    if let Some(msg) = comm.next_to_send() {
        println!("\n  📤 Sent: \"{}\" ({:?})", msg.content, msg.intent_type);
    }
}

fn print_final_status(
    agent: &AgentLoop,
    curiosity: &CuriositySystem,
    world: &WorldModel,
    comm: &CommunicationSystem,
) {
    println!("  📊 Agent Loop Stats:");
    let stats = agent.stats();
    println!("     Total cycles: {}", stats.total_cycles);
    println!("     Total actions: {}", stats.total_actions);
    println!("     Explorations: {}", stats.total_explorations);
    println!("     Average cycle: {:.2}ms", stats.average_cycle_ms);

    println!("\n  🎯 Goal Stats:");
    let goal_stats = agent.goals().stats();
    println!("     Active: {}", goal_stats.active_goals);
    println!("     Completed: {}", goal_stats.completed_goals);

    println!("\n  🧠 Neuromodulator State:");
    let neuro = agent.neuromodulators();
    println!("     Dopamine: {:.2}", neuro.dopamine);
    println!("     Motivation: {:.2}", neuro.motivation);
    println!("     Stress: {:.2}", neuro.stress);
    println!("     Patience: {:.2}", neuro.patience);

    println!("\n  🔍 Curiosity Stats:");
    let cur_stats = curiosity.stats();
    println!("     Domains explored: {}", cur_stats.domains_explored);
    println!("     Total explorations: {}", cur_stats.total_explorations);

    println!("\n  🌍 World Model Stats:");
    let world_stats = world.stats();
    println!("     Entities: {}", world_stats.total_entities);
    println!("     Relationships: {}", world_stats.total_relationships);

    println!("\n  💬 Communication Stats:");
    let comm_stats = comm.stats();
    println!("     Messages sent: {}", comm_stats.total_messages_sent);

    println!("\n✅ Demo complete!");
}
