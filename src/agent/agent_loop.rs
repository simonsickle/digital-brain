//! Perception-Action Loop
//!
//! The main agent cycle that connects perception, brain processing,
//! goal management, action selection, and execution.
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                     AGENT LOOP                               │
//! │                                                              │
//! │  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
//! │  │ Perceive │───▶│ Process  │───▶│  Decide  │              │
//! │  └──────────┘    │  (Brain) │    │ (Action) │              │
//! │       ▲          └──────────┘    └────┬─────┘              │
//! │       │                               │                     │
//! │       │          ┌──────────┐         │                     │
//! │       └──────────│  Learn   │◀────────┘                     │
//! │                  │(Outcome) │                               │
//! │                  └──────────┘                               │
//! └─────────────────────────────────────────────────────────────┘
//! ```

use std::collections::HashMap;
use std::time::{Duration, Instant};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::core::action::{ActionDecision, ActionId, ActionSelector, ActionTemplate, Outcome};
use crate::core::goals::{Goal, GoalEvent, GoalId, GoalManager};
use crate::core::neuromodulators::NeuromodulatorState;
use crate::signal::Valence;

/// Types of percepts the agent can receive
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerceptType {
    /// Textual input (messages, commands)
    Text,
    /// Sensory observation of environment
    Observation,
    /// Feedback on previous action
    Feedback,
    /// Internal signal (from brain regions)
    Internal,
    /// Time-based trigger
    Temporal,
    /// Goal-related event
    GoalEvent,
    /// Error or exception
    Error,
}

/// A single percept (input to the agent)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Percept {
    /// Unique identifier
    pub id: Uuid,
    /// Type of percept
    pub percept_type: PerceptType,
    /// Content/payload
    pub content: String,
    /// Salience (attention-grabbing level, 0-1)
    pub salience: f64,
    /// Emotional valence (-1 to +1)
    pub valence: Valence,
    /// Source of the percept
    pub source: String,
    /// Timestamp (for serialization)
    pub created_at: DateTime<Utc>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl Percept {
    /// Create a new text percept
    pub fn text(content: &str) -> Self {
        Self {
            id: Uuid::new_v4(),
            percept_type: PerceptType::Text,
            content: content.to_string(),
            salience: 0.5,
            valence: Valence::new(0.0),
            source: "external".to_string(),
            created_at: Utc::now(),
            metadata: HashMap::new(),
        }
    }

    /// Create a feedback percept
    pub fn feedback(content: &str, valence: f64) -> Self {
        Self {
            id: Uuid::new_v4(),
            percept_type: PerceptType::Feedback,
            content: content.to_string(),
            salience: 0.7,
            valence: Valence::new(valence),
            source: "feedback".to_string(),
            created_at: Utc::now(),
            metadata: HashMap::new(),
        }
    }

    /// Create an internal percept
    pub fn internal(content: &str) -> Self {
        Self {
            id: Uuid::new_v4(),
            percept_type: PerceptType::Internal,
            content: content.to_string(),
            salience: 0.3,
            valence: Valence::new(0.0),
            source: "internal".to_string(),
            created_at: Utc::now(),
            metadata: HashMap::new(),
        }
    }

    /// Create an error percept
    pub fn error(content: &str) -> Self {
        Self {
            id: Uuid::new_v4(),
            percept_type: PerceptType::Error,
            content: content.to_string(),
            salience: 0.9,
            valence: Valence::new(-0.5),
            source: "error".to_string(),
            created_at: Utc::now(),
            metadata: HashMap::new(),
        }
    }

    /// Set salience
    pub fn with_salience(mut self, salience: f64) -> Self {
        self.salience = salience.clamp(0.0, 1.0);
        self
    }

    /// Set valence
    pub fn with_valence(mut self, valence: f64) -> Self {
        self.valence = Valence::new(valence);
        self
    }

    /// Set source
    pub fn with_source(mut self, source: &str) -> Self {
        self.source = source.to_string();
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
}

/// Configuration for the agent
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Maximum percepts to buffer
    pub max_percept_buffer: usize,
    /// Minimum interval between ticks (rate limiting)
    pub min_tick_interval: Duration,
    /// Whether to auto-update neuromodulators each tick
    pub auto_neuromodulator_update: bool,
    /// Exploration rate for action selection
    pub exploration_rate: f64,
    /// Learning rate for action value updates
    pub learning_rate: f64,
    /// Enable debug logging
    pub debug: bool,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_percept_buffer: 100,
            min_tick_interval: Duration::from_millis(100),
            auto_neuromodulator_update: true,
            exploration_rate: 0.1,
            learning_rate: 0.1,
            debug: false,
        }
    }
}

/// Current state of the agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentState {
    /// Current neuromodulator state
    pub neuromodulators: NeuromodulatorState,
    /// Current world state (key-value)
    pub world_state: HashMap<String, String>,
    /// Last action taken
    pub last_action: Option<ActionId>,
    /// Last action's outcome
    pub last_outcome: Option<Outcome>,
    /// Active goal descriptions
    pub active_goals: Vec<String>,
    /// Total cycles completed
    pub total_cycles: u64,
    /// Cycles since last action
    pub idle_cycles: u64,
}

impl Default for AgentState {
    fn default() -> Self {
        Self {
            neuromodulators: NeuromodulatorState::default(),
            world_state: HashMap::new(),
            last_action: None,
            last_outcome: None,
            active_goals: Vec::new(),
            total_cycles: 0,
            idle_cycles: 0,
        }
    }
}

/// Result of a single agent cycle
#[derive(Debug, Clone)]
pub struct AgentCycleResult {
    /// The decision made
    pub decision: ActionDecision,
    /// Action that was executed (if any)
    pub executed_action: Option<ActionId>,
    /// Outcome of the action (if completed)
    pub outcome: Option<Outcome>,
    /// Goal events that occurred
    pub goal_events: Vec<GoalEvent>,
    /// Percepts that were processed
    pub percepts_processed: usize,
    /// Duration of the cycle
    pub cycle_duration: Duration,
    /// Any messages/outputs to emit
    pub outputs: Vec<AgentOutput>,
}

/// Output from the agent (communication, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentOutput {
    /// Type of output
    pub output_type: OutputType,
    /// Content
    pub content: String,
    /// Target (if applicable)
    pub target: Option<String>,
    /// Urgency (0-1)
    pub urgency: f64,
    /// Emotional tone
    pub valence: Valence,
}

/// Types of agent output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutputType {
    /// Response to input
    Response,
    /// Proactive communication
    Proactive,
    /// Status update
    Status,
    /// Request for information
    Query,
    /// Expression of emotion/state
    Expression,
}

/// Statistics about the agent loop
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentLoopStats {
    pub total_cycles: u64,
    pub total_actions: u64,
    pub total_waits: u64,
    pub total_explorations: u64,
    pub total_deliberations: u64,
    pub total_percepts: u64,
    pub average_cycle_ms: f64,
    pub goals_completed: u64,
    pub goals_abandoned: u64,
}

/// The main agent loop
pub struct AgentLoop {
    /// Action selector
    action_selector: ActionSelector,
    /// Goal manager
    goal_manager: GoalManager,
    /// Current agent state
    state: AgentState,
    /// Percept buffer
    percept_buffer: Vec<Percept>,
    /// Configuration
    config: AgentConfig,
    /// Statistics
    stats: AgentLoopStats,
    /// Last tick time
    last_tick: Option<Instant>,
    /// Pending action (for multi-step execution)
    _pending_action: Option<ActionId>,
    /// Action execution callbacks
    action_handlers: HashMap<ActionId, Box<dyn Fn(&ActionTemplate) -> Outcome + Send + Sync>>,
}

impl AgentLoop {
    /// Create a new agent loop
    pub fn new(config: AgentConfig) -> Self {
        let mut action_selector = ActionSelector::new();
        action_selector.set_exploration_rate(config.exploration_rate);
        action_selector.set_learning_rate(config.learning_rate);

        Self {
            action_selector,
            goal_manager: GoalManager::new(),
            state: AgentState::default(),
            percept_buffer: Vec::new(),
            config,
            stats: AgentLoopStats::default(),
            last_tick: None,
            _pending_action: None,
            action_handlers: HashMap::new(),
        }
    }

    /// Create with default config
    pub fn default_agent() -> Self {
        Self::new(AgentConfig::default())
    }

    /// Add a percept to the buffer
    pub fn perceive(&mut self, percept: Percept) {
        self.percept_buffer.push(percept);
        self.stats.total_percepts += 1;

        // Trim buffer if too large
        while self.percept_buffer.len() > self.config.max_percept_buffer {
            self.percept_buffer.remove(0);
        }
    }

    /// Add multiple percepts
    pub fn perceive_all(&mut self, percepts: impl IntoIterator<Item = Percept>) {
        for percept in percepts {
            self.perceive(percept);
        }
    }

    /// Register an action
    pub fn register_action(&mut self, template: ActionTemplate) {
        self.action_selector.register_action(template);
    }

    /// Register an action with a handler
    pub fn register_action_with_handler<F>(
        &mut self,
        template: ActionTemplate,
        handler: F,
    ) where
        F: Fn(&ActionTemplate) -> Outcome + Send + Sync + 'static,
    {
        let id = template.id;
        self.action_selector.register_action(template);
        self.action_handlers.insert(id, Box::new(handler));
    }

    /// Add a goal
    pub fn add_goal(&mut self, goal: Goal) -> GoalId {
        let description = goal.description.clone();
        let id = self.goal_manager.add(goal);
        self.state.active_goals.push(description);
        id
    }

    /// Decompose a goal into subgoals
    pub fn decompose_goal(&mut self, goal_id: GoalId, subgoals: Vec<Goal>) -> Vec<GoalId> {
        self.goal_manager.decompose(goal_id, subgoals)
    }

    /// Update world state
    pub fn update_world(&mut self, key: &str, value: &str) {
        self.state.world_state.insert(key.to_string(), value.to_string());
    }

    /// Get world state
    pub fn world_state(&self) -> &HashMap<String, String> {
        &self.state.world_state
    }

    /// Set neuromodulator state (for external control)
    pub fn set_neuromodulators(&mut self, state: NeuromodulatorState) {
        self.state.neuromodulators = state;
    }

    /// Get current neuromodulator state
    pub fn neuromodulators(&self) -> &NeuromodulatorState {
        &self.state.neuromodulators
    }

    /// Get mutable neuromodulator state
    pub fn neuromodulators_mut(&mut self) -> &mut NeuromodulatorState {
        &mut self.state.neuromodulators
    }

    /// Run a single tick of the agent loop
    pub fn tick(&mut self) -> AgentCycleResult {
        let tick_start = Instant::now();
        self.state.total_cycles += 1;
        self.stats.total_cycles += 1;

        // Rate limiting
        if let Some(last) = self.last_tick {
            let elapsed = tick_start.duration_since(last);
            if elapsed < self.config.min_tick_interval {
                std::thread::sleep(self.config.min_tick_interval - elapsed);
            }
        }
        self.last_tick = Some(Instant::now());

        let mut outputs = Vec::new();
        let mut goal_events = Vec::new();

        // 1. Process percepts
        let percepts_processed = self.process_percepts(&mut outputs);

        // 2. Check goals
        self.update_active_goals();
        goal_events.extend(self.goal_manager.events().iter().cloned());
        self.goal_manager.clear_events();

        // Update stats from goal events
        for event in &goal_events {
            match event {
                GoalEvent::Completed { .. } => self.stats.goals_completed += 1,
                GoalEvent::Abandoned { .. } => self.stats.goals_abandoned += 1,
                _ => {}
            }
        }

        // 3. Select action
        let decision = self.action_selector.select(
            &self.state.neuromodulators,
            &self.state.active_goals,
            &self.state.world_state,
        );

        // 4. Execute or handle decision
        let (executed_action, outcome) = self.handle_decision(&decision, &mut outputs);

        // 5. Learn from outcome
        if let (Some(action_id), Some(outcome)) = (executed_action, &outcome) {
            self.action_selector.update_from_outcome(action_id, outcome);
            self.state.last_action = Some(action_id);
            self.state.last_outcome = Some(outcome.clone());

            // Update goal progress if relevant
            if let Some(ref goal_desc) = outcome.related_goal {
                self.update_goal_from_outcome(goal_desc, outcome);
            }
        }

        // 6. Update statistics
        match &decision {
            ActionDecision::Execute(_) => {
                self.stats.total_actions += 1;
                self.state.idle_cycles = 0;
            }
            ActionDecision::Wait { .. } => {
                self.stats.total_waits += 1;
                self.state.idle_cycles += 1;
            }
            ActionDecision::Explore { .. } => {
                self.stats.total_explorations += 1;
                self.state.idle_cycles = 0;
            }
            ActionDecision::Deliberate { .. } => {
                self.stats.total_deliberations += 1;
                self.state.idle_cycles += 1;
            }
            ActionDecision::NoAction { .. } => {
                self.state.idle_cycles += 1;
            }
        }

        let cycle_duration = tick_start.elapsed();
        self.stats.average_cycle_ms = self.stats.average_cycle_ms * 0.99
            + cycle_duration.as_secs_f64() * 1000.0 * 0.01;

        AgentCycleResult {
            decision,
            executed_action,
            outcome,
            goal_events,
            percepts_processed,
            cycle_duration,
            outputs,
        }
    }

    /// Process buffered percepts
    fn process_percepts(&mut self, outputs: &mut Vec<AgentOutput>) -> usize {
        let percepts: Vec<_> = self.percept_buffer.drain(..).collect();
        let count = percepts.len();

        for percept in percepts {
            // Update emotional state based on valence
            let valence_impact = percept.valence.value() * percept.salience * 0.1;
            self.state.neuromodulators.dopamine =
                (self.state.neuromodulators.dopamine + valence_impact).clamp(0.0, 1.0);

            // Handle specific percept types
            match percept.percept_type {
                PerceptType::Feedback => {
                    // Feedback affects motivation
                    if percept.valence.value() > 0.0 {
                        self.state.neuromodulators.motivation =
                            (self.state.neuromodulators.motivation + 0.1).min(1.0);
                    } else if percept.valence.value() < -0.3 {
                        self.state.neuromodulators.stress =
                            (self.state.neuromodulators.stress + 0.1).min(1.0);
                    }
                }
                PerceptType::Error => {
                    // Errors increase stress
                    self.state.neuromodulators.stress =
                        (self.state.neuromodulators.stress + 0.2).min(1.0);

                    // Generate acknowledgment output
                    outputs.push(AgentOutput {
                        output_type: OutputType::Status,
                        content: format!("Acknowledged error: {}", percept.content),
                        target: None,
                        urgency: percept.salience,
                        valence: Valence::new(-0.3),
                    });
                }
                PerceptType::GoalEvent => {
                    // Goal events might require response
                    if percept.content.contains("deadline") {
                        outputs.push(AgentOutput {
                            output_type: OutputType::Status,
                            content: format!("Deadline alert: {}", percept.content),
                            target: None,
                            urgency: 0.8,
                            valence: Valence::new(-0.2),
                        });
                    }
                }
                _ => {}
            }
        }

        count
    }

    /// Update the list of active goal descriptions
    fn update_active_goals(&mut self) {
        self.state.active_goals = self
            .goal_manager
            .active_goals()
            .iter()
            .map(|g| g.description.clone())
            .collect();
    }

    /// Handle an action decision
    fn handle_decision(
        &mut self,
        decision: &ActionDecision,
        outputs: &mut Vec<AgentOutput>,
    ) -> (Option<ActionId>, Option<Outcome>) {
        match decision {
            ActionDecision::Execute(action_id) => {
                let outcome = self.execute_action(*action_id);
                (Some(*action_id), outcome)
            }
            ActionDecision::Wait { reason, .. } => {
                if self.config.debug {
                    outputs.push(AgentOutput {
                        output_type: OutputType::Status,
                        content: format!("Waiting: {}", reason),
                        target: None,
                        urgency: 0.1,
                        valence: Valence::new(0.0),
                    });
                }
                (None, None)
            }
            ActionDecision::Deliberate { options, conflict_reason } => {
                if self.config.debug {
                    outputs.push(AgentOutput {
                        output_type: OutputType::Status,
                        content: format!(
                            "Deliberating between {} options: {}",
                            options.len(),
                            conflict_reason
                        ),
                        target: None,
                        urgency: 0.3,
                        valence: Valence::new(0.0),
                    });
                }
                // For now, just pick the first option
                if let Some(action_id) = options.first() {
                    let outcome = self.execute_action(*action_id);
                    (Some(*action_id), outcome)
                } else {
                    (None, None)
                }
            }
            ActionDecision::Explore { domain, curiosity_level } => {
                if self.config.debug {
                    outputs.push(AgentOutput {
                        output_type: OutputType::Status,
                        content: format!(
                            "Exploring {} (curiosity: {:.2})",
                            domain, curiosity_level
                        ),
                        target: None,
                        urgency: 0.2,
                        valence: Valence::new(0.2),
                    });
                }
                // Exploration bonus to learning
                self.state.neuromodulators.learning_depth =
                    (self.state.neuromodulators.learning_depth + 0.1).min(1.0);
                (None, None)
            }
            ActionDecision::NoAction { reason } => {
                if self.config.debug && self.state.idle_cycles % 10 == 0 {
                    outputs.push(AgentOutput {
                        output_type: OutputType::Status,
                        content: format!("No action available: {}", reason),
                        target: None,
                        urgency: 0.1,
                        valence: Valence::new(-0.1),
                    });
                }
                (None, None)
            }
        }
    }

    /// Execute an action
    fn execute_action(&mut self, action_id: ActionId) -> Option<Outcome> {
        // Check for registered handler
        if let Some(handler) = self.action_handlers.get(&action_id) {
            if let Some(template) = self.action_selector.get_action(action_id) {
                let template = template.clone();
                return Some(handler(&template));
            }
        }

        // Default: return expected outcome based on probability
        if let Some(template) = self.action_selector.get_action(action_id) {
            // Simple simulation: pick most likely outcome
            if let Some(expected) = template.expected_outcomes.first() {
                return Some(expected.outcome.clone());
            }
        }

        None
    }

    /// Update goal progress based on action outcome
    fn update_goal_from_outcome(&mut self, goal_desc: &str, outcome: &Outcome) {
        // Find matching goal (collect to avoid borrow issue)
        let matching_goal: Option<(GoalId, f64)> = self
            .goal_manager
            .all_goals()
            .find(|g| g.description.contains(goal_desc) || goal_desc.contains(&g.description))
            .map(|g| (g.id, g.progress));

        if let Some((goal_id, current_progress)) = matching_goal {
            let new_progress = (current_progress + outcome.progress).min(1.0);
            self.goal_manager.update_progress(goal_id, new_progress);
        }
    }

    /// Get statistics
    pub fn stats(&self) -> &AgentLoopStats {
        &self.stats
    }

    /// Get agent state
    pub fn state(&self) -> &AgentState {
        &self.state
    }

    /// Get goal manager (read-only)
    pub fn goals(&self) -> &GoalManager {
        &self.goal_manager
    }

    /// Get goal manager (mutable)
    pub fn goals_mut(&mut self) -> &mut GoalManager {
        &mut self.goal_manager
    }

    /// Get action selector (read-only)
    pub fn actions(&self) -> &ActionSelector {
        &self.action_selector
    }

    /// Get action selector (mutable)
    pub fn actions_mut(&mut self) -> &mut ActionSelector {
        &mut self.action_selector
    }

    /// Check if the agent is idle (no recent actions)
    pub fn is_idle(&self) -> bool {
        self.state.idle_cycles > 10
    }

    /// Reset idle counter
    pub fn reset_idle(&mut self) {
        self.state.idle_cycles = 0;
    }

    /// Run the agent for a number of ticks
    pub fn run_ticks(&mut self, n: usize) -> Vec<AgentCycleResult> {
        (0..n).map(|_| self.tick()).collect()
    }

    /// Complete a goal by ID
    pub fn complete_goal(&mut self, goal_id: GoalId) -> bool {
        let result = self.goal_manager.complete_goal(goal_id);
        if result {
            // Reward for goal completion
            self.state.neuromodulators.dopamine =
                (self.state.neuromodulators.dopamine + 0.3).min(1.0);
            self.state.neuromodulators.motivation =
                (self.state.neuromodulators.motivation + 0.2).min(1.0);
        }
        result
    }

    /// Provide feedback on last action
    pub fn feedback(&mut self, positive: bool, message: &str) {
        let valence = if positive { 0.5 } else { -0.5 };
        self.perceive(Percept::feedback(message, valence));

        // Update last action value
        if let Some(action_id) = self.state.last_action {
            let outcome_value = if positive { 0.8 } else { -0.3 };
            self.action_selector.update_from_outcome(
                action_id,
                &Outcome::success(message, outcome_value),
            );
        }
    }
}

// Default neuromodulator state
impl Default for NeuromodulatorState {
    fn default() -> Self {
        Self {
            dopamine: 0.5,
            serotonin: 0.5,
            norepinephrine: 0.5,
            acetylcholine: 0.5,
            cortisol: 0.2,
            gaba: 0.5,
            oxytocin: 0.5,
            motivation: 0.5,
            patience: 0.5,
            stress: 0.2,
            learning_depth: 0.5,
            frustration: 0.1,
            exploration_drive: 0.3,
            impulse_control: 0.5,
            cooperativeness: 0.5,
            is_satiated: false,
            is_stressed: false,
            is_burned_out: false,
            is_deliberating: false,
            should_pivot: false,
            should_seek_help: false,
            prefer_cooperation: false,
            mood_stability: 0.7,
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::action::{ActionCategory, ExpectedOutcome};
    use crate::core::goals::Priority;

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
            time_cost: 1,
            category: ActionCategory::Exploration,
            tags: vec![],
        }
    }

    #[test]
    fn test_agent_creation() {
        let agent = AgentLoop::default_agent();
        assert_eq!(agent.stats().total_cycles, 0);
    }

    #[test]
    fn test_perceive() {
        let mut agent = AgentLoop::default_agent();

        agent.perceive(Percept::text("Hello"));
        agent.perceive(Percept::text("World"));

        assert_eq!(agent.stats().total_percepts, 2);
    }

    #[test]
    fn test_register_action() {
        let mut agent = AgentLoop::default_agent();
        let action = make_test_action("test");

        agent.register_action(action);

        assert_eq!(agent.actions().actions().len(), 1);
    }

    #[test]
    fn test_add_goal() {
        let mut agent = AgentLoop::default_agent();

        let goal = Goal::new("Test goal").with_priority(Priority::High);
        agent.add_goal(goal);

        assert_eq!(agent.goals().stats().total_goals, 1);
        assert_eq!(agent.state().active_goals.len(), 1);
    }

    #[test]
    fn test_tick_basic() {
        let mut agent = AgentLoop::default_agent();
        agent.register_action(make_test_action("action1"));

        let result = agent.tick();

        assert_eq!(agent.stats().total_cycles, 1);
        assert!(result.cycle_duration.as_nanos() > 0);
    }

    #[test]
    fn test_tick_executes_action() {
        let mut agent = AgentLoop::new(AgentConfig {
            min_tick_interval: Duration::from_millis(0),
            ..Default::default()
        });

        agent.register_action(make_test_action("action1"));

        // Run tick - should execute the action
        let result = agent.tick();

        assert!(matches!(
            result.decision,
            ActionDecision::Execute(_) | ActionDecision::Wait { .. }
        ));
    }

    #[test]
    fn test_feedback_updates_state() {
        let mut agent = AgentLoop::default_agent();
        let action = make_test_action("action1");
        let action_id = action.id;
        agent.register_action(action);

        // Execute an action
        agent.state.last_action = Some(action_id);

        // Positive feedback
        let initial_motivation = agent.neuromodulators().motivation;
        agent.feedback(true, "Good job!");

        // Should increase motivation via percept processing
        agent.tick();
        assert!(agent.neuromodulators().motivation >= initial_motivation);
    }

    #[test]
    fn test_world_state() {
        let mut agent = AgentLoop::default_agent();

        agent.update_world("location", "home");
        agent.update_world("time", "morning");

        assert_eq!(agent.world_state().get("location"), Some(&"home".to_string()));
        assert_eq!(agent.world_state().get("time"), Some(&"morning".to_string()));
    }

    #[test]
    fn test_goal_completion_rewards() {
        let mut agent = AgentLoop::default_agent();

        let goal = Goal::new("Test goal");
        let goal_id = agent.add_goal(goal);

        let initial_dopamine = agent.neuromodulators().dopamine;

        agent.complete_goal(goal_id);

        // Should get dopamine boost
        assert!(agent.neuromodulators().dopamine > initial_dopamine);
    }

    #[test]
    fn test_idle_detection() {
        let mut agent = AgentLoop::new(AgentConfig {
            min_tick_interval: Duration::from_millis(0),
            ..Default::default()
        });

        // No actions registered = will be idle
        for _ in 0..15 {
            agent.tick();
        }

        assert!(agent.is_idle());

        agent.reset_idle();
        assert!(!agent.is_idle());
    }

    #[test]
    fn test_run_ticks() {
        let mut agent = AgentLoop::new(AgentConfig {
            min_tick_interval: Duration::from_millis(0),
            ..Default::default()
        });

        agent.register_action(make_test_action("action1"));

        let results = agent.run_ticks(5);

        assert_eq!(results.len(), 5);
        assert_eq!(agent.stats().total_cycles, 5);
    }

    #[test]
    fn test_error_percept_increases_stress() {
        let mut agent = AgentLoop::new(AgentConfig {
            min_tick_interval: Duration::from_millis(0),
            ..Default::default()
        });

        let initial_stress = agent.neuromodulators().stress;

        agent.perceive(Percept::error("Something went wrong"));
        agent.tick();

        assert!(agent.neuromodulators().stress > initial_stress);
    }

    #[test]
    fn test_action_with_handler() {
        let mut agent = AgentLoop::new(AgentConfig {
            min_tick_interval: Duration::from_millis(0),
            ..Default::default()
        });

        let mut action = make_test_action("custom");
        action.preconditions = vec![]; // Ensure it can execute
        let action_id = action.id;

        agent.register_action_with_handler(action, |template| {
            Outcome::success(&format!("Executed {}", template.name), 1.0)
        });

        // Force execution of this action
        let outcome = agent.execute_action(action_id);

        assert!(outcome.is_some());
        assert_eq!(outcome.unwrap().value, 1.0);
    }
}
