# Agent Integration Roadmap

This document outlines the current state of the digital brain and what needs to be implemented for full AI agent integration.

## Current State (as of January 2026)

### Implemented Brain Regions

| Module | File | Purpose | Status |
|--------|------|---------|--------|
| Thalamus | `src/regions/thalamus.rs` | Sensory gateway, attention routing, habituation | Complete |
| Amygdala | `src/regions/amygdala.rs` | Emotional processing, threat detection, valence tagging | Complete |
| Hippocampus | `src/regions/hippocampus.rs` | Long-term memory, consolidation, decay | Complete |
| Prefrontal | `src/regions/prefrontal.rs` | Working memory (7±2), chunking, goal stack | Complete |
| DMN | `src/regions/dmn.rs` | Self-model, beliefs, reflection, identity | Complete |
| Global Workspace | `src/core/workspace.rs` | Consciousness, attention bottleneck, broadcasting | Complete |
| Prediction Engine | `src/core/prediction.rs` | Surprise computation, learning rate modulation | Complete |
| **Neuromodulators** | `src/core/neuromodulators.rs` | Dopamine, serotonin, norepinephrine, acetylcholine | **NEW** |

### Neuromodulatory System Features

The new neuromodulatory system (`src/core/neuromodulators.rs`) provides:

#### Dopamine System
- Reward delivery with tolerance tracking
- Quality-weighted rewards (depth, goal-alignment, durability, intrinsic, novelty)
- Delayed evaluation with regret signals
- Motivation/wanting tracking
- Satiation detection

#### Serotonin System
- Patience and long-term thinking
- Delay discounting for decision-making
- Mood stability tracking
- Counterbalances impulsive reward-seeking

#### Norepinephrine System
- Focused attention (rewards sustained focus)
- Stress/threat response
- Attention quality (inverted-U curve)
- Context switch penalties

#### Acetylcholine System
- Learning rate modulation
- Memory encoding strength
- Deep vs shallow processing tracking

#### Anti-Addiction Safeguards
- **Tolerance**: Repeated same-category rewards become less effective
- **Quality scoring**: Novelty weighted least (10%), depth weighted most (30%)
- **Serotonin gating**: High patience reduces cheap reward effectiveness by 50%
- **Homeostatic regulation**: All levels return to baseline
- **Cross-modulation**: Systems influence each other (stress reduces patience, etc.)

### Current Integration Points

```rust
// Brain API for agents
brain.process(input)              // Process sensory input
brain.focus(topic)                // Direct attention
brain.believe(content, category, confidence)  // Add beliefs
brain.goal_achieved(description)  // Signal achievement (reward)
brain.reward(magnitude, category, quality)    // Custom reward
brain.should_wait(immediate, delayed, delay)  // Patience check
brain.neuromodulator_state()      // Get current state
brain.sleep(hours)                // Consolidation cycle
brain.stats()                     // Full statistics including neuromodulators
```

---

## Missing Components for Full Agent Integration

### 1. Action Selection System (HIGH PRIORITY)

**Problem**: The brain can process inputs but cannot decide on outputs/actions.

**Proposed Implementation**: `src/core/action.rs`

```rust
pub struct ActionSelector {
    available_actions: Vec<ActionTemplate>,
    action_values: HashMap<ActionId, f64>,
    inhibition_map: HashMap<ActionId, Vec<ActionId>>,
}

pub struct ActionTemplate {
    id: ActionId,
    name: String,
    preconditions: Vec<Condition>,
    expected_outcomes: Vec<Outcome>,
    effort_cost: f64,
    time_cost: u32,
}

pub enum ActionDecision {
    Execute(ActionId),
    Wait { reason: String, until: Option<Condition> },
    Deliberate { options: Vec<ActionId> },
    Explore { curiosity_driven: bool },
}

impl ActionSelector {
    /// Select action based on neuromodulator state
    pub fn select(&self, brain_state: &NeuromodulatorState, goals: &[Goal]) -> ActionDecision;

    /// Update action values based on outcome
    pub fn update_from_outcome(&mut self, action: ActionId, outcome: &Outcome);
}
```

**Integration with neuromodulators**:
- High motivation → bias toward goal-directed actions
- High stress → bias toward defensive/safe actions
- Low patience → bias toward immediate actions
- High acetylcholine → more deliberation before acting

---

### 2. Goal Management System (HIGH PRIORITY)

**Problem**: Prefrontal has a simple goal stack, but no proper goal decomposition, prioritization, or progress tracking.

**Proposed Implementation**: `src/core/goals.rs`

```rust
pub struct GoalManager {
    goals: Vec<Goal>,
    goal_hierarchy: HashMap<GoalId, Vec<GoalId>>,  // Parent → children
    progress: HashMap<GoalId, f64>,
}

pub struct Goal {
    id: GoalId,
    description: String,
    priority: f64,
    deadline: Option<DateTime<Utc>>,
    success_criteria: Vec<Criterion>,
    parent: Option<GoalId>,
    status: GoalStatus,
}

pub enum GoalStatus {
    Active,
    Blocked { reason: String },
    Completed,
    Abandoned { reason: String },
}

impl GoalManager {
    /// Decompose a goal into subgoals
    pub fn decompose(&mut self, goal_id: GoalId, subgoals: Vec<Goal>);

    /// Get highest priority actionable goal
    pub fn get_active_goal(&self, state: &NeuromodulatorState) -> Option<&Goal>;

    /// Update progress and potentially complete goals
    pub fn update_progress(&mut self, goal_id: GoalId, progress: f64);

    /// Check if patience advises waiting on this goal
    pub fn should_defer(&self, goal_id: GoalId, brain: &Brain) -> bool;
}
```

**Integration with neuromodulators**:
- Goal completion triggers `Achievement` reward
- Blocked goals increase stress
- Progress triggers anticipation (dopamine)
- Long-term goals require high serotonin (patience)

---

### 3. Perception-Action Loop (MEDIUM PRIORITY)

**Problem**: No closed loop between perception, processing, action, and outcome evaluation.

**Proposed Implementation**: `src/agent/loop.rs`

```rust
pub struct AgentLoop {
    brain: Brain,
    action_selector: ActionSelector,
    goal_manager: GoalManager,
    world_model: WorldModel,
}

impl AgentLoop {
    /// Main agent cycle
    pub async fn tick(&mut self) -> AgentCycleResult {
        // 1. Perceive
        let percepts = self.perceive();

        // 2. Process through brain
        let results: Vec<_> = percepts
            .iter()
            .map(|p| self.brain.process(p))
            .collect();

        // 3. Update world model
        self.world_model.update(&results);

        // 4. Check goals
        let active_goal = self.goal_manager.get_active_goal(
            &self.brain.neuromodulator_state()
        );

        // 5. Select action
        let decision = self.action_selector.select(
            &self.brain.neuromodulator_state(),
            &self.goal_manager.goals,
        );

        // 6. Execute or deliberate
        match decision {
            ActionDecision::Execute(action) => {
                let outcome = self.execute(action).await;
                self.learn_from_outcome(action, &outcome);
            }
            ActionDecision::Wait { .. } => {
                // Do nothing, let neuromodulators update
            }
            ActionDecision::Deliberate { options } => {
                // Run mental simulation
                self.simulate_options(&options);
            }
            ActionDecision::Explore { .. } => {
                // Curiosity-driven action
                self.explore();
            }
        }

        // 7. Homeostatic update
        self.brain.neuromodulators.update();

        AgentCycleResult { ... }
    }

    /// Learn from action outcome
    fn learn_from_outcome(&mut self, action: ActionId, outcome: &Outcome) {
        // Evaluate pending rewards
        self.brain.neuromodulators.dopamine.evaluate_pending(
            &action.to_string(),
            outcome.value,
        );

        // Update action values
        self.action_selector.update_from_outcome(action, outcome);

        // Update goal progress
        if let Some(goal) = outcome.related_goal {
            self.goal_manager.update_progress(goal, outcome.progress);
        }
    }
}
```

---

### 4. Curiosity/Exploration Drive (MEDIUM PRIORITY)

**Problem**: No intrinsic motivation to explore or learn.

**Proposed Implementation**: `src/core/curiosity.rs`

```rust
pub struct CuriositySystem {
    uncertainty_map: HashMap<Domain, f64>,
    exploration_history: Vec<ExplorationEvent>,
    information_gain_threshold: f64,
}

impl CuriositySystem {
    /// Calculate curiosity reward for a potential action
    pub fn curiosity_value(&self, action: &ActionTemplate) -> f64 {
        let expected_info_gain = self.estimate_info_gain(action);
        let novelty = self.novelty_bonus(action);
        let competence_progress = self.competence_gain(action);

        // Weight by neuromodulator state
        expected_info_gain * 0.4 + novelty * 0.3 + competence_progress * 0.3
    }

    /// Should we explore vs exploit?
    pub fn explore_vs_exploit(&self, state: &NeuromodulatorState) -> f64 {
        // High dopamine → exploit (seek known rewards)
        // Low dopamine + high ACh → explore (seek information)
        let exploit_bias = state.motivation;
        let explore_bias = state.learning_depth * (1.0 - state.dopamine);

        explore_bias / (explore_bias + exploit_bias)
    }
}
```

---

### 5. World Model (LOWER PRIORITY)

**Problem**: No representation of the external world state.

**Proposed Implementation**: `src/core/world_model.rs`

```rust
pub struct WorldModel {
    entities: HashMap<EntityId, Entity>,
    relationships: Vec<Relationship>,
    predictions: Vec<WorldPrediction>,
    last_updated: DateTime<Utc>,
}

impl WorldModel {
    /// Update model from processing results
    pub fn update(&mut self, results: &[ProcessingResult]);

    /// Predict future state
    pub fn predict(&self, horizon: u32) -> WorldPrediction;

    /// Check prediction against actual outcome
    pub fn evaluate_prediction(&mut self, prediction_id: Uuid, actual: &WorldState);
}
```

---

### 6. Communication Interface (LOWER PRIORITY)

**Problem**: No structured way to generate outputs/responses.

**Proposed Implementation**: `src/agent/communication.rs`

```rust
pub struct CommunicationSystem {
    output_buffer: Vec<CommunicationIntent>,
    style: CommunicationStyle,
}

pub struct CommunicationIntent {
    content: String,
    intent_type: IntentType,
    urgency: f64,
    emotional_tone: Valence,
}

pub enum IntentType {
    Inform,
    Request,
    Confirm,
    Clarify,
    Express,  // Emotional expression
}

impl CommunicationSystem {
    /// Generate response based on brain state
    pub fn generate_response(&self, brain: &Brain, context: &str) -> CommunicationIntent;
}
```

---

## Implementation Priority

### Phase 1: Core Agent Loop (Next PR)
1. Action Selection System
2. Goal Management System
3. Basic Perception-Action Loop

### Phase 2: Autonomy
4. Curiosity/Exploration Drive
5. World Model

### Phase 3: Communication
6. Communication Interface
7. Multi-agent interaction

---

## Integration Example (Target State)

```rust
use digital_brain::agent::{AgentLoop, AgentConfig};
use digital_brain::brain::Brain;

#[tokio::main]
async fn main() -> Result<()> {
    // Create agent with brain
    let mut agent = AgentLoop::new(AgentConfig {
        brain: Brain::new()?,
        ..Default::default()
    });

    // Set identity
    agent.brain.set_identity(Identity {
        name: "Assistant".to_string(),
        core_values: vec!["helpfulness".to_string(), "honesty".to_string()],
        self_description: "An AI assistant powered by a digital brain".to_string(),
        creation_time: Utc::now(),
    });

    // Add goals
    agent.goal_manager.add(Goal {
        description: "Help users effectively".to_string(),
        priority: 1.0,
        ..Default::default()
    });

    // Main loop
    loop {
        let result = agent.tick().await;

        // Check for outputs
        if let Some(response) = result.communication {
            println!("Agent: {}", response.content);
        }

        // Periodic sleep for consolidation
        if agent.brain.stats().cycles % 1000 == 0 {
            agent.brain.sleep(0.5)?;  // Mini consolidation
        }
    }
}
```

---

## Testing Strategy

Each new component should have:

1. **Unit tests** for individual functions
2. **Integration tests** with the Brain
3. **Behavioral tests** for agent-level behavior:
   - Does it pursue goals?
   - Does it learn from outcomes?
   - Does patience prevent impulsive actions?
   - Does tolerance prevent addiction loops?
   - Does stress trigger appropriate responses?

---

## Open Questions

1. **How should the agent interface with external systems?** (APIs, tools, etc.)
2. **Should there be a "dreaming" mode** for offline consolidation and planning?
3. **How to handle multi-agent scenarios?** (Theory of mind is partially in DMN)
4. **What's the right abstraction for "actions"** in an LLM context?
5. **Should curiosity be a separate neuromodulator** (like some theories suggest)?

---

## References

- Global Workspace Theory (Baars)
- Predictive Processing (Clark, Friston)
- Neuromodulation and decision-making (Doya)
- Temporal difference learning (Sutton & Barto)
- Free Energy Principle (Friston)
