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

use uuid::Uuid;

use serde_json::Value;

use crate::agent::{
    AgentConfig, AgentLoop, AgentProfile, CommunicationSystem, IntentType, MultiAgentSystem,
    Percept,
};
use crate::brain::{Brain, BrainConfig, ProcessingResult, SleepReport};
use crate::core::{
    ActionTemplate, CuriositySystem, Domain, Entity, EntityId, Goal, GoalId, Imagining,
    PlanningSuggestion, PropertyValue, RelationType, Relationship, StrategyRegulator,
    StrategySignal, WorldModel,
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
    /// Strategy regulator (mood/sleep → long-horizon bias)
    strategy_regulator: StrategyRegulator,
    /// Last observed sleep quality (0-1)
    last_sleep_quality: f64,
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
            strategy_regulator: StrategyRegulator::default(),
            last_sleep_quality: 0.5,
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
        self.update_world_model_from_processing(input, &result);
        self.sync_neuromodulators_from_brain();
        Ok(result)
    }

    fn update_world_model_from_processing(&mut self, input: &str, result: &ProcessingResult) {
        let Some(world) = self.world.as_mut() else {
            return;
        };

        let (modalities, features, anchors, binding_strength, novelty, detail_level, confidence) =
            if let Some(context) = &result.multimodal_context {
                (
                    context.modalities.iter().map(|m| m.to_string()).collect(),
                    context.integrated_features.clone(),
                    context.anchors.clone(),
                    context.binding_strength,
                    context.novelty,
                    context.detail_level,
                    context.confidence,
                )
            } else {
                let modalities: Vec<String> = result
                    .cortical_features
                    .iter()
                    .map(|rep| rep.modality.to_string())
                    .collect();
                let mut features = Vec::new();
                let mut anchors = Vec::new();
                let mut confidence_sum = 0.0;
                let mut novelty_sum = 0.0;
                let mut detail_sum = 0.0;

                for rep in &result.cortical_features {
                    for feature in rep.detected_features.iter().take(3) {
                        features.push(format!("{}:{}", rep.modality, feature));
                    }
                    if let Some(primary) = rep.primary_focus.clone() {
                        anchors.push(primary);
                    }
                    confidence_sum += rep.confidence;
                    novelty_sum += rep.novelty;
                    detail_sum += rep.detail_level;
                }

                let count = result.cortical_features.len().max(1) as f64;
                (
                    modalities,
                    features,
                    anchors,
                    0.0,
                    novelty_sum / count,
                    detail_sum / count,
                    confidence_sum / count,
                )
            };

        if modalities.is_empty() && features.is_empty() {
            return;
        }

        let scene_name = "current_scene";
        let existing = world.find_by_name(scene_name);
        let entity_id = if let Some(entity) = existing.first() {
            entity.id
        } else {
            let mut entity = Entity::new("percept", scene_name)
                .with_property("source", "sensory")
                .with_confidence(confidence)
                .with_valence(result.tagged_signal.valence.value());
            if result.multimodal_context.is_some() {
                entity = entity.with_tag("multimodal");
            } else {
                entity = entity.with_tag("sensory");
            }
            for modality in &modalities {
                entity = entity.with_tag(modality);
            }
            world.add_entity(entity)
        };

        let modalities = unique_strings(modalities);
        let features = unique_strings(features);
        let anchors = unique_strings(anchors);

        world.update_entity_property(entity_id, "modalities", property_list(&modalities));
        world.update_entity_property(entity_id, "features", property_list(&features));
        world.update_entity_property(entity_id, "anchors", property_list(&anchors));
        world.update_entity_property(entity_id, "binding_strength", binding_strength);
        world.update_entity_property(entity_id, "novelty", novelty);
        world.update_entity_property(entity_id, "detail_level", detail_level);
        world.update_entity_property(entity_id, "confidence", confidence);
        world.update_entity_property(entity_id, "last_input", input);
        world.update_entity_property(entity_id, "salience", result.tagged_signal.salience.value());
        world.update_entity_property(entity_id, "arousal", result.tagged_signal.arousal.value());
        world.update_entity_property(entity_id, "valence", result.tagged_signal.valence.value());

        if let Some(entity) = world.get_entity_mut(entity_id) {
            entity.valence = result.tagged_signal.valence;
            entity.confidence = confidence;
            entity.tags = modalities;
            entity.tags.push(if result.multimodal_context.is_some() {
                "multimodal".to_string()
            } else {
                "sensory".to_string()
            });
        }

        if !result.errors.is_empty() {
            let avg_error = result.errors.iter().map(|e| e.error_magnitude).sum::<f64>()
                / result.errors.len() as f64;
            world.update_entity_property(entity_id, "prediction_error", avg_error);

            let domain = result
                .errors
                .first()
                .map(|e| e.domain.as_str())
                .unwrap_or("unknown");
            world.record_prediction_error(
                &format!("prediction_error:{}", domain),
                avg_error,
                confidence,
            );

            let adjustment = (avg_error - 0.3) * 0.15;
            if adjustment.abs() > 0.01 {
                self.agent.actions_mut().adjust_exploration_rate(adjustment);
            }
        }
    }

    fn update_world_model_from_imagining(&mut self, imagining: &Imagining) {
        let Some(world) = self.world.as_mut() else {
            return;
        };

        let imagination_name = format!("imagination:{}", imagining.id);
        let imagination_id = find_entity_id(world, &imagination_name, "imagination")
            .unwrap_or_else(|| {
                let mut entity = Entity::new("imagination", &imagination_name)
                    .with_property("content", imagining.content.clone())
                    .with_property("imagination_type", imagining.imagination_type.to_string())
                    .with_property("created_at", imagining.created_at.to_rfc3339())
                    .with_property("confidence", imagining.confidence)
                    .with_property("novelty", imagining.novelty)
                    .with_property("utility", imagining.utility)
                    .with_confidence(imagining.confidence)
                    .with_tag("imagination");

                let type_tag = imagining.imagination_type.to_string();
                entity = entity.with_tag(&type_tag);

                if !imagining.source_memory_ids.is_empty() {
                    entity = entity.with_property(
                        "source_memories",
                        property_list(&imagining.source_memory_ids),
                    );
                }

                world.add_entity(entity)
            });

        world.update_entity_property(imagination_id, "content", imagining.content.clone());
        world.update_entity_property(
            imagination_id,
            "imagination_type",
            imagining.imagination_type.to_string(),
        );
        world.update_entity_property(imagination_id, "confidence", imagining.confidence);
        world.update_entity_property(imagination_id, "novelty", imagining.novelty);
        world.update_entity_property(imagination_id, "utility", imagining.utility);
        world.update_entity_property(
            imagination_id,
            "created_at",
            imagining.created_at.to_rfc3339(),
        );
        if !imagining.source_memory_ids.is_empty() {
            world.update_entity_property(
                imagination_id,
                "source_memories",
                property_list(&imagining.source_memory_ids),
            );
        }

        for (key, value) in &imagining.metadata {
            if let Some(property) = json_to_property(value) {
                world.update_entity_property(imagination_id, &format!("meta_{}", key), property);
            }
        }

        if let Some(entity) = world.get_entity_mut(imagination_id) {
            entity.confidence = imagining.confidence;
            ensure_tag(&mut entity.tags, "imagination");
            ensure_tag(&mut entity.tags, &imagining.imagination_type.to_string());
        }

        let analogies = extract_analogy_terms(imagining);
        if analogies.is_empty() {
            return;
        }

        let strength = (imagining.novelty * 0.6 + imagining.confidence * 0.4).clamp(0.1, 1.0);
        for (term, origin) in analogies {
            let concept_id = ensure_concept_entity(world, &term, imagining.confidence);
            let relationship =
                Relationship::new(imagination_id, concept_id, RelationType::SimilarTo)
                    .with_strength(strength)
                    .with_metadata("origin", origin)
                    .with_metadata("source", "imagination");
            add_relationship_if_missing(world, relationship);
        }
    }

    fn update_world_model_from_planning_suggestions(&mut self, suggestions: &[PlanningSuggestion]) {
        if suggestions.is_empty() {
            return;
        }

        for suggestion in suggestions {
            self.update_world_model_from_planning_suggestion(suggestion);
        }
    }

    fn update_world_model_from_planning_suggestion(&mut self, suggestion: &PlanningSuggestion) {
        if suggestion.is_empty() {
            return;
        }

        let Some(world) = self.world.as_mut() else {
            return;
        };

        let imagination_name = format!("imagination:{}", suggestion.imagining_id);
        let imagination_id = find_entity_id(world, &imagination_name, "imagination")
            .unwrap_or_else(|| {
                let mut entity = Entity::new("imagination", &imagination_name)
                    .with_property("imagination_type", suggestion.imagination_type.to_string())
                    .with_confidence((suggestion.score * 0.8 + 0.2).clamp(0.0, 1.0))
                    .with_tag("imagination");
                let type_tag = suggestion.imagination_type.to_string();
                entity = entity.with_tag(&type_tag);
                world.add_entity(entity)
            });

        let plan_name = format!("plan:{}", suggestion.imagining_id);
        let plan_id = find_entity_id(world, &plan_name, "plan").unwrap_or_else(|| {
            let mut entity = Entity::new("plan", &plan_name)
                .with_property("imagination_id", suggestion.imagining_id.to_string())
                .with_property("imagination_type", suggestion.imagination_type.to_string())
                .with_property("rationale", suggestion.rationale.clone())
                .with_property("score", suggestion.score)
                .with_property("goal_count", suggestion.goals.len() as f64)
                .with_property("action_count", suggestion.actions.len() as f64)
                .with_confidence(suggestion.score)
                .with_tag("plan")
                .with_tag("imagination");
            let type_tag = suggestion.imagination_type.to_string();
            entity = entity.with_tag(&type_tag);
            world.add_entity(entity)
        });

        world.update_entity_property(plan_id, "rationale", suggestion.rationale.clone());
        world.update_entity_property(plan_id, "score", suggestion.score);
        world.update_entity_property(plan_id, "goal_count", suggestion.goals.len() as f64);
        world.update_entity_property(plan_id, "action_count", suggestion.actions.len() as f64);

        let plan_link = Relationship::new(imagination_id, plan_id, RelationType::Causes)
            .with_strength(suggestion.score)
            .with_metadata("source", "imagination");
        add_relationship_if_missing(world, plan_link);

        for goal in &suggestion.goals {
            let goal_name = format!("plan:{}:goal:{}", suggestion.imagining_id, goal.id);
            let goal_id = find_entity_id(world, &goal_name, "goal").unwrap_or_else(|| {
                let mut entity = Entity::new("goal", &goal_name)
                    .with_property("description", goal.description.clone())
                    .with_property("priority", goal.priority.as_f64())
                    .with_property("priority_label", format!("{:?}", goal.priority))
                    .with_property("time_horizon", format!("{:?}", goal.time_horizon))
                    .with_property("effort", goal.estimated_effort)
                    .with_property("notes", goal.notes.clone())
                    .with_confidence(goal.priority.as_f64())
                    .with_tag("goal")
                    .with_tag("imagination");

                if !goal.tags.is_empty() {
                    entity = entity.with_property("tags", property_list(&goal.tags));
                }

                if let Some(deadline) = goal.deadline {
                    entity = entity.with_property("deadline", deadline.to_rfc3339());
                }

                world.add_entity(entity)
            });

            world.update_entity_property(goal_id, "description", goal.description.clone());
            world.update_entity_property(goal_id, "priority", goal.priority.as_f64());
            world.update_entity_property(goal_id, "notes", goal.notes.clone());
            if !goal.tags.is_empty() {
                world.update_entity_property(goal_id, "tags", property_list(&goal.tags));
            }

            let relationship = Relationship::new(plan_id, goal_id, RelationType::Contains)
                .with_strength(goal.priority.as_f64())
                .with_metadata("source", "imagination_plan");
            add_relationship_if_missing(world, relationship);
        }

        for action in &suggestion.actions {
            let action_name = format!("plan:{}:action:{}", suggestion.imagining_id, action.id);
            let action_id = find_entity_id(world, &action_name, "action").unwrap_or_else(|| {
                let mut entity = Entity::new("action", &action_name)
                    .with_property("name", action.name.clone())
                    .with_property("description", action.description.clone())
                    .with_property("category", format!("{:?}", action.category))
                    .with_property("effort_cost", action.effort_cost)
                    .with_property("time_cost", action.time_cost as f64)
                    .with_confidence(suggestion.score)
                    .with_tag("action")
                    .with_tag("imagination");

                if !action.tags.is_empty() {
                    entity = entity.with_property("tags", property_list(&action.tags));
                }

                world.add_entity(entity)
            });

            world.update_entity_property(action_id, "description", action.description.clone());
            world.update_entity_property(action_id, "effort_cost", action.effort_cost);
            world.update_entity_property(action_id, "time_cost", action.time_cost as f64);
            if !action.tags.is_empty() {
                world.update_entity_property(action_id, "tags", property_list(&action.tags));
            }

            let relationship = Relationship::new(plan_id, action_id, RelationType::Contains)
                .with_strength((1.0 - action.effort_cost).clamp(0.0, 1.0))
                .with_metadata("source", "imagination_plan");
            add_relationship_if_missing(world, relationship);
        }
    }

    /// Run a single cycle of the brain-agent
    pub fn tick(&mut self) -> BrainAgentCycleResult {
        self.stats.total_cycles += 1;
        self.cycles_since_sleep += 1;

        let mut outputs = Vec::new();
        let processing = None;
        let mut sleep_report = None;

        self.update_strategy_from_state("cycle");
        self.sync_neuromodulators_from_brain();

        // Sync neuromodulator state from brain to agent
        // (In a full implementation, the brain's neuromodulatory system
        // would be the source of truth)
        // For now, we use the agent's state

        // Run agent cycle
        let agent_result = self.agent.tick();
        self.update_world_model_from_planning_suggestions(&agent_result.planning_suggestions);

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
                outputs.push(format!("[{:?}] {}", intent.intent_type, intent.content));
            }
        }

        // Periodic sleep/consolidation
        if self.cycles_since_sleep >= self.config.sleep_interval {
            if let Ok(report) = self.brain.sleep(self.config.sleep_duration) {
                self.stats.total_sleep_cycles += 1;
                self.stats.total_memories_consolidated += report.memories_consolidated as u64;
                sleep_report = Some(report);
            }
            self.cycles_since_sleep = 0;
        }

        if let Some(report) = &sleep_report {
            self.last_sleep_quality = report.sleep_quality;
            self.update_strategy_from_state("sleep");
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
        if let Ok(result) = self.brain.process(input) {
            self.update_world_model_from_processing(input, &result);
            self.sync_neuromodulators_from_brain();
        }
        self.stats.total_inputs_processed += 1;
        self.update_strategy_from_state("perception");
    }

    /// Perceive with feedback valence
    pub fn feedback(&mut self, positive: bool, message: &str) {
        self.agent.feedback(positive, message);
    }

    /// Submit an imagination for planning integration
    pub fn submit_imagining(&mut self, imagining: Imagining) {
        self.update_world_model_from_imagining(&imagining);
        self.agent.submit_imagining(imagining);
    }

    /// Submit multiple imaginations for planning integration
    pub fn submit_imaginations(&mut self, imaginings: impl IntoIterator<Item = Imagining>) {
        let imaginings: Vec<Imagining> = imaginings.into_iter().collect();
        for imagining in &imaginings {
            self.update_world_model_from_imagining(imagining);
        }
        self.agent.submit_imaginations(imaginings);
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
        self.last_sleep_quality = report.sleep_quality;
        self.update_strategy_from_state("sleep");
        Ok(report)
    }

    fn update_strategy_from_state(&mut self, context: &str) {
        let state = self.brain.neuromodulators.state();
        let update = self.strategy_regulator.update(StrategySignal::new(
            self.last_sleep_quality,
            state.mood_stability,
            state.stress,
        ));
        self.agent
            .goals_mut()
            .apply_strategy_profile(update.profile);

        if let Some(narrative) = update.narrative {
            self.brain.dmn.narrate(
                format!("Strategy update ({}): {}", context, narrative.content),
                narrative.significance,
            );
        }
    }

    fn sync_neuromodulators_from_brain(&mut self) {
        let state = self.brain.neuromodulators.state();
        self.agent.set_neuromodulators(state);
        self.sync_social_world_state();
    }

    fn sync_social_world_state(&mut self) {
        for (agent_id, _) in self.brain.social_hierarchy() {
            if let Some(reputation) = self.brain.social_reputation(&agent_id) {
                self.agent.update_world(
                    &format!("reputation:{}", agent_id),
                    &format!("{:.2}", reputation),
                );
            }
        }
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

fn property_list(values: &[String]) -> PropertyValue {
    PropertyValue::List(values.iter().cloned().map(PropertyValue::String).collect())
}

fn unique_strings(values: Vec<String>) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut deduped = Vec::new();
    for value in values {
        if seen.insert(value.clone()) {
            deduped.push(value);
        }
    }
    deduped
}

fn json_to_property(value: &Value) -> Option<PropertyValue> {
    match value {
        Value::Null => None,
        Value::Bool(b) => Some(PropertyValue::Boolean(*b)),
        Value::Number(n) => n.as_f64().map(PropertyValue::Number),
        Value::String(s) => Some(PropertyValue::String(s.clone())),
        Value::Array(values) => {
            let list: Vec<PropertyValue> = values.iter().filter_map(json_to_property).collect();
            Some(PropertyValue::List(list))
        }
        Value::Object(_) => Some(PropertyValue::String(value.to_string())),
    }
}

fn extract_analogy_terms(imagining: &Imagining) -> Vec<(String, &'static str)> {
    let mut terms = Vec::new();
    collect_terms(&imagining.metadata, "patterns", "pattern", &mut terms);
    collect_terms(&imagining.metadata, "topics", "topic", &mut terms);
    collect_terms(&imagining.metadata, "key_factors", "factor", &mut terms);
    collect_terms(&imagining.metadata, "domain", "domain", &mut terms);
    collect_terms(
        &imagining.metadata,
        "implications",
        "implication",
        &mut terms,
    );

    let mut seen = std::collections::HashSet::new();
    let mut deduped = Vec::new();
    for (term, origin) in terms {
        if let Some(clean) = sanitize_term(&term) {
            let key = clean.to_lowercase();
            if seen.insert(key) {
                deduped.push((clean, origin));
            }
        }
        if deduped.len() >= 8 {
            break;
        }
    }
    deduped
}

fn collect_terms(
    metadata: &std::collections::HashMap<String, Value>,
    key: &str,
    origin: &'static str,
    terms: &mut Vec<(String, &'static str)>,
) {
    let Some(value) = metadata.get(key) else {
        return;
    };

    match value {
        Value::String(s) => terms.push((s.clone(), origin)),
        Value::Array(items) => {
            for item in items {
                if let Value::String(s) = item {
                    terms.push((s.clone(), origin));
                }
            }
        }
        _ => {}
    }
}

fn sanitize_term(term: &str) -> Option<String> {
    let trimmed = term.trim();
    if trimmed.is_empty() {
        return None;
    }
    if trimmed.len() > 64 {
        return None;
    }
    Some(trimmed.to_string())
}

fn find_entity_id(world: &WorldModel, name: &str, entity_type: &str) -> Option<EntityId> {
    world
        .find_by_name(name)
        .into_iter()
        .find(|entity| entity.name == name && entity.entity_type == entity_type)
        .map(|entity| entity.id)
}

fn ensure_concept_entity(world: &mut WorldModel, term: &str, confidence: f64) -> EntityId {
    let name = format!("concept:{}", term.to_lowercase());
    if let Some(id) = find_entity_id(world, &name, "concept") {
        world.update_entity_property(id, "label", term.to_string());
        world.update_entity_property(id, "confidence", confidence);
        if let Some(entity) = world.get_entity_mut(id) {
            entity.confidence = confidence;
            ensure_tag(&mut entity.tags, "concept");
            ensure_tag(&mut entity.tags, "analogy");
        }
        return id;
    }

    let mut entity = Entity::new("concept", &name)
        .with_property("label", term.to_string())
        .with_property("confidence", confidence)
        .with_confidence(confidence)
        .with_tag("concept")
        .with_tag("analogy");
    entity = entity.with_tag(term);
    world.add_entity(entity)
}

fn add_relationship_if_missing(world: &mut WorldModel, relationship: Relationship) {
    let exists = world
        .get_outgoing(relationship.source)
        .iter()
        .any(|existing| {
            existing.target == relationship.target
                && existing.relation_type == relationship.relation_type
        });
    if !exists {
        world.add_relationship(relationship);
    }
}

fn ensure_tag(tags: &mut Vec<String>, tag: &str) {
    if !tags.iter().any(|existing| existing == tag) {
        tags.push(tag.to_string());
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::prediction::{PredictionError, PredictionLayer};
    use crate::core::{ActionCategory, ExpectedOutcome, ImaginationType, Outcome, Priority};
    use chrono::Utc;
    use serde_json::json;
    use std::time::Duration;

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
    fn test_brain_agent_world_model_updates_from_sensory() {
        let mut agent = BrainAgent::new().unwrap();

        agent
            .process("A bright red circle appears while soft music plays")
            .unwrap();

        let world = agent.world().expect("world model should be enabled");
        let scenes = world.find_by_name("current_scene");
        assert!(!scenes.is_empty());
        let scene = scenes[0];
        assert!(scene.get_property("modalities").is_some());
        assert!(scene.get_property("features").is_some());
    }

    #[test]
    fn test_brain_agent_imagination_updates_world_model() {
        let config = BrainAgentConfig {
            agent: AgentConfig {
                min_tick_interval: Duration::from_millis(0),
                ..Default::default()
            },
            ..Default::default()
        };
        let mut agent = BrainAgent::with_config(config).unwrap();

        let imagining = Imagining::new(
            ImaginationType::Synthesis,
            "Blend feedback loops with narrative recall".to_string(),
            vec!["mem-creative".to_string()],
        )
        .with_confidence(0.8)
        .with_utility(0.9)
        .with_novelty(0.7)
        .with_metadata("patterns", json!(["feedback loop", "resonance"]))
        .with_metadata("domain", json!("systems"));

        let imagining_id = imagining.id;
        agent.submit_imagining(imagining);

        let world = agent.world().expect("world model should be enabled");
        let imagination_name = format!("imagination:{}", imagining_id);
        assert!(!world.find_by_name(&imagination_name).is_empty());
        assert!(!world.find_by_type("concept").is_empty());

        agent.tick();

        let world = agent.world().expect("world model should be enabled");
        let plan_name = format!("plan:{}", imagining_id);
        assert!(!world.find_by_name(&plan_name).is_empty());
        assert!(!world.find_by_type("goal").is_empty());
        assert!(!world.find_by_type("action").is_empty());
    }

    #[test]
    fn test_prediction_errors_adjust_exploration() {
        let mut agent = BrainAgent::new().unwrap();
        agent.register_action(make_test_action("test"));

        let mut result = agent.process("A sudden unexpected flash").unwrap();
        let base_rate = agent.agent().actions().exploration_rate();

        let error = PredictionError {
            prediction_id: Uuid::new_v4(),
            layer: PredictionLayer::Conceptual,
            domain: "percept".to_string(),
            source: "test".to_string(),
            expected: serde_json::Value::String("dark".to_string()),
            actual: serde_json::Value::String("flash".to_string()),
            error_magnitude: 0.8,
            error_direction: 0.8,
            surprise: 0.8,
            computed_at: Utc::now(),
        };

        result.errors = vec![error];
        agent.update_world_model_from_processing("A sudden unexpected flash", &result);

        let new_rate = agent.agent().actions().exploration_rate();
        assert!(new_rate > base_rate);
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
