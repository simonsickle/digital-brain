//! Brain-Agent Integration
//!
//! Unifies the Brain (memory, processing, neuromodulators) with the
//! Agent systems (actions, goals, curiosity, world model, communication).
//!
//! This is the "full stack" — a complete autonomous agent with:
//! - Persistent memory (hippocampus)
//! - Emotional processing (amygdala)
//! - Working memory (prefrontal)
//! - Attention routing (thalamus)
//! - Self-model (DMN)
//! - Prediction engine
//! - Neuromodulatory system
//! - Action selection
//! - Goal management
//! - Curiosity-driven exploration
//! - World modeling
//! - Communication interface

use std::time::Duration;
use uuid::Uuid;

use crate::brain::{Brain, BrainConfig, ProcessingResult, SleepReport};
use crate::agent::{
    AgentConfig, AgentLoop, CommunicationSystem, IntentType, Percept,
    MultiAgentSystem, AgentProfile,
};
use crate::core::{
    ActionTemplate, CuriositySystem, Domain, Goal, GoalId, WorldModel, Entity,
    NeuromodulatorState,
};
use crate::error::Result;

/// Configuration for the brain-agent
#[derive(Debug, Clone)]
pub struct BrainAgentConfig {
    /// Brain configuration
    pub brain: BrainConfig,
    /// Agent loop configuration
    pub agent: AgentConfig,
    /// Enable curiosity system
    pub enable_curiosity: bool,
    /// Enable world model
    pub enable_world_model: bool,
    /// Enable communication
    pub enable_communication: bool,
    /// Enable multi-agent
    pub enable_multi_agent: bool,
    /// Sleep interval (cycles between consolidation)
    pub sleep_interval: u64,
    /// Sleep duration (hours equivalent)
    pub sleep_duration: f64,
}

impl Default for BrainAgentConfig {
    fn default() -> Self {
        Self {
            brain: BrainConfig::default(),
            agent: AgentConfig::default(),
            enable_curiosity: true,
            enable_world_model: true,
            enable_communication: true,
            enable_multi_agent: false,
            sleep_interval: 1000,
            sleep_duration: 0.5,
        }
    }
}

/// Result of a brain-agent cycle
#[derive(Debug)]
pub struct BrainAgentCycleResult {
    /// Processing result from brain
    pub processing: Option<ProcessingResult>,
    /// Agent cycle result
    pub agent_cycle: crate::agent::AgentCycleResult,
    /// Any sleep report (if consolidation occurred)
    pub sleep_report: Option<SleepReport>,
    /// Outputs to emit
    pub outputs: Vec<String>,
}

/// Statistics for the brain-agent
#[derive(Debug, Clone, Default)]
pub struct BrainAgentStats {
    pub total_cycles: u64,
    pub total_inputs_processed: u64,
    pub total_actions_taken: u64,
    pub total_goals_completed: u64,
    pub total_sleep_cycles: u64,
    pub total_memories_consolidated: u64,
}

/// A complete brain-agent system
pub struct BrainAgent {
    /// The brain (memory, processing, neuromodulators)
    brain: Brain,
    /// The agent loop (actions, goals)
    agent: AgentLoop,
    /// Curiosity system (optional)
    curiosity: Option<CuriositySystem>,
    /// World model (optional)
    world: Option<WorldModel>,
    /// Communication system (optional)
    comm: Option<CommunicationSystem>,
    /// Multi-agent system (optional)
    multi_agent: Option<MultiAgentSystem>,
    /// Configuration
    config: BrainAgentConfig,
    /// Statistics
    stats: BrainAgentStats,
    /// Cycles since last sleep
    cycles_since_sleep: u64,
}

impl BrainAgent {
    /// Create a new brain-agent with default configuration
    pub fn new() -> Result<Self> {
        Self::with_config(BrainAgentConfig::default())
    }

    /// Create a new brain-agent with custom configuration
    pub fn with_config(config: BrainAgentConfig) -> Result<Self> {
        let brain = Brain::with_config(config.brain.clone())?;
        let agent = AgentLoop::new(config.agent.clone());

        let curiosity = if config.enable_curiosity {
            Some(CuriositySystem::new())
        } else {
            None
        };

        let world = if config.enable_world_model {
            Some(WorldModel::new())
        } else {
            None
        };

        let comm = if config.enable_communication {
            Some(CommunicationSystem::new())
        } else {
            None
        };

        let multi_agent = if config.enable_multi_agent {
            let profile = AgentProfile::new("BrainAgent")
                .with_description("An autonomous agent powered by a digital brain")
                .with_capability("reasoning")
                .with_capability("memory")
                .with_capability("learning");
            Some(MultiAgentSystem::new(profile))
        } else {
            None
        };

        Ok(Self {
            brain,
            agent,
            curiosity,
            world,
            comm,
            multi_agent,
            config,
            stats: BrainAgentStats::default(),
            cycles_since_sleep: 0,
        })
    }

    /// Get the brain
    pub fn brain(&self) -> &Brain {
        &self.brain
    }

    /// Get mutable brain
    pub fn brain_mut(&mut self) -> &mut Brain {
        &mut self.brain
    }

    /// Get the agent loop
    pub fn agent(&self) -> &AgentLoop {
        &self.agent
    }

    /// Get mutable agent loop
    pub fn agent_mut(&mut self) -> &mut AgentLoop {
        &mut self.agent
    }

    /// Get curiosity system
    pub fn curiosity(&self) -> Option<&CuriositySystem> {
        self.curiosity.as_ref()
    }

    /// Get mutable curiosity system
    pub fn curiosity_mut(&mut self) -> Option<&mut CuriositySystem> {
        self.curiosity.as_mut()
    }

    /// Get world model
    pub fn world(&self) -> Option<&WorldModel> {
        self.world.as_ref()
    }

    /// Get mutable world model
    pub fn world_mut(&mut self) -> Option<&mut WorldModel> {
        self.world.as_mut()
    }

    /// Get communication system
    pub fn comm(&self) -> Option<&CommunicationSystem> {
        self.comm.as_ref()
    }

    /// Get mutable communication system
    pub fn comm_mut(&mut self) -> Option<&mut CommunicationSystem> {
        self.comm.as_mut()
    }

    /// Get multi-agent system
    pub fn multi_agent(&self) -> Option<&MultiAgentSystem> {
        self.multi_agent.as_ref()
    }

    /// Get mutable multi-agent system
    pub fn multi_agent_mut(&mut self) -> Option<&mut MultiAgentSystem> {
        self.multi_agent.as_mut()
    }

    /// Register an action
    pub fn register_action(&mut self, action: ActionTemplate) {
        self.agent.register_action(action);
    }

    /// Add a goal
    pub fn add_goal(&mut self, goal: Goal) -> GoalId {
        self.agent.add_goal(goal)
    }

    /// Register a curiosity domain
    pub fn register_domain(&mut self, name: &str, uncertainty: f64) {
        if let Some(curiosity) = &mut self.curiosity {
            curiosity.register_domain(Domain::new(name), uncertainty);
        }
    }

    /// Add an entity to the world model
    pub fn add_entity(&mut self, entity: Entity) -> Option<Uuid> {
        self.world.as_mut().map(|w| w.add_entity(entity))
    }

    /// Process an input through the brain
    pub fn process(&mut self, input: &str) -> Result<ProcessingResult> {
        let result = self.brain.process(input)?;
        self.stats.total_inputs_processed += 1;
        Ok(result)
    }

    /// Run a single cycle of the brain-agent
    pub fn tick(&mut self) -> BrainAgentCycleResult {
        self.stats.total_cycles += 1;
        self.cycles_since_sleep += 1;

        let mut outputs = Vec::new();
        let mut processing = None;
        let mut sleep_report = None;

        // Sync neuromodulator state from brain to agent
        // (In a full implementation, the brain's neuromodulatory system
        // would be the source of truth)
        // For now, we use the agent's state

        // Run agent cycle
        let agent_result = self.agent.tick();

        // Track actions
        if agent_result.executed_action.is_some() {
            self.stats.total_actions_taken += 1;
        }

        // Track goal completions from events
        for event in &agent_result.goal_events {
            if matches!(event, crate::core::GoalEvent::Completed { .. }) {
                self.stats.total_goals_completed += 1;
            }
        }

        // Generate outputs from agent
        for output in &agent_result.outputs {
            outputs.push(output.content.clone());
        }

        // Check for communication outputs
        if let Some(comm) = &mut self.comm {
            while let Some(intent) = comm.next_to_send() {
                outputs.push(format!("[{}] {}", format!("{:?}", intent.intent_type), intent.content));
            }
        }

        // Periodic sleep/consolidation
        if self.cycles_since_sleep >= self.config.sleep_interval {
            match self.brain.sleep(self.config.sleep_duration) {
                Ok(report) => {
                    self.stats.total_sleep_cycles += 1;
                    self.stats.total_memories_consolidated += report.memories_consolidated as u64;
                    sleep_report = Some(report);
                }
                Err(_) => {}
            }
            self.cycles_since_sleep = 0;
        }

        BrainAgentCycleResult {
            processing,
            agent_cycle: agent_result,
            sleep_report,
            outputs,
        }
    }

    /// Perceive input (adds to agent's percept buffer)
    pub fn perceive(&mut self, input: &str) {
        self.agent.perceive(Percept::text(input));

        // Also process through brain (ignore errors)
        let _ = self.brain.process(input).ok();
        self.stats.total_inputs_processed += 1;
    }

    /// Perceive with feedback valence
    pub fn feedback(&mut self, positive: bool, message: &str) {
        self.agent.feedback(positive, message);
    }

    /// Run multiple cycles
    pub fn run(&mut self, cycles: u64) -> Vec<BrainAgentCycleResult> {
        (0..cycles).map(|_| self.tick()).collect()
    }

    /// Sleep and consolidate memories
    pub fn sleep(&mut self, hours: f64) -> Result<SleepReport> {
        let report = self.brain.sleep(hours)?;
        self.stats.total_sleep_cycles += 1;
        self.stats.total_memories_consolidated += report.memories_consolidated as u64;
        self.cycles_since_sleep = 0;
        Ok(report)
    }

    /// Get statistics
    pub fn stats(&self) -> &BrainAgentStats {
        &self.stats
    }

    /// Get brain statistics
    pub fn brain_stats(&self) -> crate::brain::BrainStats {
        self.brain.stats()
    }

    /// Focus attention on a topic
    pub fn focus(&mut self, topic: &str) {
        self.brain.focus(topic);
    }

    /// Add a belief
    pub fn believe(&mut self, content: &str, confidence: f64) {
        self.brain.believe(
            content,
            crate::regions::dmn::BeliefCategory::WorldModel,
            confidence,
        );
    }

    /// Reflect on a topic
    pub fn reflect(&mut self, topic: &str) -> String {
        self.brain.reflect(topic)
    }

    /// Queue a communication
    pub fn say(&mut self, content: &str, intent: IntentType) {
        if let Some(comm) = &mut self.comm {
            let intent_obj = crate::agent::CommunicationIntent::new(content, intent);
            comm.queue(intent_obj);
        }
    }

    /// Inform (shorthand)
    pub fn inform(&mut self, content: &str) {
        self.say(content, IntentType::Inform);
    }

    /// Request (shorthand)
    pub fn request(&mut self, content: &str) {
        self.say(content, IntentType::Request);
    }

    /// Check if idle
    pub fn is_idle(&self) -> bool {
        self.agent.is_idle()
    }
}

impl Default for BrainAgent {
    fn default() -> Self {
        Self::new().expect("Failed to create default BrainAgent")
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{ActionCategory, ExpectedOutcome, Outcome, Priority};

    fn make_test_action(name: &str) -> ActionTemplate {
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
            category: ActionCategory::Exploitation,
            tags: vec![],
        }
    }

    #[test]
    fn test_brain_agent_creation() {
        let agent = BrainAgent::new().unwrap();
        assert_eq!(agent.stats().total_cycles, 0);
    }

    #[test]
    fn test_brain_agent_with_config() {
        let config = BrainAgentConfig {
            enable_curiosity: true,
            enable_world_model: true,
            enable_communication: true,
            enable_multi_agent: true,
            ..Default::default()
        };

        let agent = BrainAgent::with_config(config).unwrap();

        assert!(agent.curiosity().is_some());
        assert!(agent.world().is_some());
        assert!(agent.comm().is_some());
        assert!(agent.multi_agent().is_some());
    }

    #[test]
    fn test_brain_agent_tick() {
        let config = BrainAgentConfig {
            agent: AgentConfig {
                min_tick_interval: Duration::from_millis(0),
                ..Default::default()
            },
            ..Default::default()
        };

        let mut agent = BrainAgent::with_config(config).unwrap();
        agent.register_action(make_test_action("test"));

        let result = agent.tick();

        assert_eq!(agent.stats().total_cycles, 1);
        assert!(result.agent_cycle.cycle_duration.as_nanos() > 0);
    }

    #[test]
    fn test_brain_agent_perceive() {
        let mut agent = BrainAgent::new().unwrap();
        agent.register_action(make_test_action("test"));

        agent.perceive("Hello world");

        assert!(agent.stats().total_inputs_processed > 0);
    }

    #[test]
    fn test_brain_agent_goals() {
        let mut agent = BrainAgent::new().unwrap();

        let goal = Goal::new("Test goal").with_priority(Priority::High);
        let id = agent.add_goal(goal);

        assert!(agent.agent().goals().get(id).is_some());
    }

    #[test]
    fn test_brain_agent_world() {
        let mut agent = BrainAgent::new().unwrap();

        let entity = Entity::new("test", "TestEntity");
        let id = agent.add_entity(entity);

        assert!(id.is_some());
        assert!(agent.world().unwrap().get_entity(id.unwrap()).is_some());
    }

    #[test]
    fn test_brain_agent_curiosity() {
        let mut agent = BrainAgent::new().unwrap();

        agent.register_domain("rust", 0.7);

        let domain = Domain::new("rust");
        assert_eq!(agent.curiosity().unwrap().uncertainty(&domain), 0.7);
    }

    #[test]
    fn test_brain_agent_communication() {
        let mut agent = BrainAgent::new().unwrap();

        agent.inform("Hello!");
        agent.request("Can you help?");

        assert_eq!(agent.comm().unwrap().pending().len(), 2);
    }

    #[test]
    fn test_brain_agent_run() {
        let config = BrainAgentConfig {
            agent: AgentConfig {
                min_tick_interval: Duration::from_millis(0),
                ..Default::default()
            },
            sleep_interval: 1000, // Don't sleep during short run
            ..Default::default()
        };

        let mut agent = BrainAgent::with_config(config).unwrap();
        agent.register_action(make_test_action("test"));

        let results = agent.run(10);

        assert_eq!(results.len(), 10);
        assert_eq!(agent.stats().total_cycles, 10);
    }

    #[test]
    fn test_brain_agent_reflection() {
        let mut agent = BrainAgent::new().unwrap();

        agent.believe("Rust is a great language", 0.9);

        let reflection = agent.reflect("programming");
        assert!(!reflection.is_empty());
    }

    #[test]
    fn test_brain_agent_focus() {
        let mut agent = BrainAgent::new().unwrap();

        // Should not panic
        agent.focus("important topic");
    }
}
