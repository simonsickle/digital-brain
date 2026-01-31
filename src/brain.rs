//! Digital Brain - Complete Integration
//!
//! This module wires all brain regions together into a functioning whole.
//! The Brain struct coordinates processing cycles across all components.

use crate::core::nervous_system::{BrainRegion, NervousSystem, NervousSystemStats};
use crate::core::neuromodulators::{
    NeuromodulatorState, NeuromodulatorySystem, RewardCategory, RewardQuality,
};
use crate::core::prediction::{Prediction, PredictionEngine, PredictionError};
use crate::core::workspace::{Broadcast, GlobalWorkspace, WorkspaceConfig};
use crate::error::Result;
use crate::regions::amygdala::{Amygdala, EmotionalAppraisal};
use crate::regions::dmn::{
    Belief, BeliefCategory, DefaultModeNetwork, Identity, ReflectionTrigger,
};
use crate::regions::hippocampus::HippocampusStore;
use crate::regions::prefrontal::{PrefrontalConfig, PrefrontalCortex};
use crate::regions::schema::{Schema, SchemaCategory, SchemaStats, SchemaStore};
use crate::regions::thalamus::{Destination, Thalamus};
use crate::signal::{BrainSignal, MemoryTrace, Salience, SignalType};

/// Configuration for the complete brain.
#[derive(Debug, Clone)]
pub struct BrainConfig {
    /// Path for persistent memory (None = in-memory)
    pub memory_path: Option<String>,
    /// Working memory capacity
    pub working_memory_capacity: usize,
    /// Global workspace capacity
    pub consciousness_capacity: usize,
    /// Enable detailed logging
    pub verbose: bool,
}

impl Default for BrainConfig {
    fn default() -> Self {
        Self {
            memory_path: None,
            working_memory_capacity: 7,
            consciousness_capacity: 5,
            verbose: false,
        }
    }
}

/// Result of processing a single input.
#[derive(Debug)]
pub struct ProcessingResult {
    /// Original signal after emotional tagging
    pub tagged_signal: BrainSignal,
    /// Emotional appraisal
    pub emotion: EmotionalAppraisal,
    /// Whether signal reached consciousness
    pub reached_consciousness: bool,
    /// Memory trace if encoded
    pub memory: Option<MemoryTrace>,
    /// Any predictions generated
    pub predictions: Vec<Prediction>,
    /// Any prediction errors detected
    pub errors: Vec<PredictionError>,
    /// Reflections triggered
    pub reflections: Vec<String>,
}

/// The complete digital brain.
pub struct Brain {
    /// Sensory gateway
    pub thalamus: Thalamus,
    /// Emotional processing
    pub amygdala: Amygdala,
    /// Long-term memory
    pub hippocampus: HippocampusStore,
    /// Working memory
    pub prefrontal: PrefrontalCortex,
    /// Consciousness/attention
    pub workspace: GlobalWorkspace,
    /// Learning/prediction
    pub prediction: PredictionEngine,
    /// Self-model
    pub dmn: DefaultModeNetwork,
    /// Neuromodulatory system (dopamine, serotonin, norepinephrine, acetylcholine)
    pub neuromodulators: NeuromodulatorySystem,
    /// Nervous system (inter-module signal routing)
    pub nervous_system: NervousSystem,
    /// Schema store (abstracted patterns from episodes)
    pub schemas: SchemaStore,
    /// Processing cycle count
    cycle_count: u64,
    /// Configuration
    #[allow(dead_code)]
    config: BrainConfig,
}

impl Brain {
    /// Create a new brain with default configuration.
    pub fn new() -> Result<Self> {
        Self::with_config(BrainConfig::default())
    }

    /// Create a brain with custom configuration.
    pub fn with_config(config: BrainConfig) -> Result<Self> {
        let hippocampus = match &config.memory_path {
            Some(path) => HippocampusStore::new(path)?,
            None => HippocampusStore::new_in_memory()?,
        };

        let workspace = GlobalWorkspace::with_config(WorkspaceConfig {
            capacity: config.consciousness_capacity,
            ..Default::default()
        });

        let prefrontal = PrefrontalCortex::with_config(PrefrontalConfig {
            capacity: config.working_memory_capacity,
            ..Default::default()
        });

        Ok(Self {
            thalamus: Thalamus::new(),
            amygdala: Amygdala::new(),
            hippocampus,
            prefrontal,
            workspace,
            prediction: PredictionEngine::new(),
            dmn: DefaultModeNetwork::new(),
            neuromodulators: NeuromodulatorySystem::new(),
            nervous_system: NervousSystem::new(),
            schemas: SchemaStore::new(),
            cycle_count: 0,
            config,
        })
    }

    /// Set the brain's identity.
    pub fn set_identity(&mut self, identity: Identity) {
        self.dmn.set_identity(identity);
    }

    /// Process a single input through the complete brain.
    pub fn process(&mut self, input: impl Into<String>) -> Result<ProcessingResult> {
        let content = input.into();

        // 1. Create sensory signal
        let mut signal = BrainSignal::new("external", SignalType::Sensory, &content);

        // 1.5 Neuromodulatory influence on incoming signal
        // Arousal state affects salience perception
        let arousal_boost = (self.neuromodulators.arousal() - 0.5) * 0.2;
        signal.salience = Salience::new(signal.salience.value() + arousal_boost);

        // 1.6 Record signal entering through nervous system
        self.nervous_system
            .transmit(BrainRegion::External, BrainRegion::Thalamus, signal.clone());

        // 2. Gate through thalamus
        self.thalamus.receive(signal);
        let routed_signals = self.thalamus.process_cycle();

        if routed_signals.is_empty() {
            // Signal was filtered - update neuromodulators anyway
            self.neuromodulators.update();
            self.cycle_count += 1;

            return Ok(ProcessingResult {
                tagged_signal: BrainSignal::new("external", SignalType::Sensory, &content),
                emotion: self.amygdala.appraise(&BrainSignal::new(
                    "external",
                    SignalType::Sensory,
                    &content,
                )),
                reached_consciousness: false,
                memory: None,
                predictions: Vec::new(),
                errors: Vec::new(),
                reflections: Vec::new(),
            });
        }

        let routed = &routed_signals[0];

        // 3. Emotional tagging by amygdala
        // Record signal flowing to amygdala via nervous system
        self.nervous_system.transmit(
            BrainRegion::Thalamus,
            BrainRegion::Amygdala,
            routed.signal.clone(),
        );
        let tagged_signal = self.amygdala.tag_signal(routed.signal.clone());
        let emotion = self.amygdala.appraise(&tagged_signal);

        // 3.5 Neuromodulatory response to emotional content
        if emotion.is_significant {
            // Emotional content affects norepinephrine (arousal)
            self.neuromodulators
                .norepinephrine
                .level
                .adjust(emotion.arousal.value() * 0.1);

            // Threat detection affects stress
            if emotion.valence.is_negative() && emotion.arousal.is_high() {
                self.neuromodulators
                    .signal_threat(emotion.arousal.value() * 0.5);
            }
        }

        // 4. Update DMN emotional state
        self.dmn
            .update_emotional_state(tagged_signal.valence.value());

        // Update serotonin mood tracking
        self.neuromodulators
            .serotonin
            .record_mood(tagged_signal.valence.value());

        // 5. Route to appropriate modules
        let mut reached_consciousness = false;
        let mut memory = None;
        let mut reflections = Vec::new();

        for dest in &routed.destinations {
            match dest {
                Destination::Workspace => {
                    // Track signal to workspace via nervous system
                    self.nervous_system.transmit(
                        BrainRegion::Amygdala,
                        BrainRegion::Workspace,
                        tagged_signal.clone(),
                    );
                    self.workspace.submit(tagged_signal.clone());
                }
                Destination::Hippocampus => {
                    // Track signal to hippocampus
                    self.nervous_system.transmit(
                        BrainRegion::Thalamus,
                        BrainRegion::Hippocampus,
                        tagged_signal.clone(),
                    );
                    // Also track emotional tagging pathway
                    self.nervous_system.transmit(
                        BrainRegion::Amygdala,
                        BrainRegion::Hippocampus,
                        tagged_signal.clone(),
                    );

                    // Acetylcholine modulates memory encoding strength
                    self.neuromodulators.acetylcholine.record_encoding();
                    let encoding_strength = self.neuromodulators.encoding_strength();

                    let mut trace = self.hippocampus.encode(&tagged_signal)?;

                    // Strengthen memory based on acetylcholine level
                    trace.strength *= encoding_strength;
                    memory = Some(trace);
                }
                Destination::Prefrontal => {
                    // Track signal to prefrontal
                    self.nervous_system.transmit(
                        BrainRegion::Thalamus,
                        BrainRegion::Prefrontal,
                        tagged_signal.clone(),
                    );

                    // Focus attention through norepinephrine
                    self.neuromodulators
                        .focus_on(format!("processing:{}", &content[..content.len().min(20)]));
                    self.prefrontal.load(&tagged_signal);
                }
                _ => {}
            }
        }

        // 6. Process workspace cycle
        let broadcasts = self.workspace.process_cycle();
        if !broadcasts.is_empty() {
            reached_consciousness = true;

            // Broadcast conscious content through nervous system to all regions
            let broadcast_signal =
                BrainSignal::new("workspace", SignalType::Broadcast, &content).with_salience(1.0);
            self.nervous_system.broadcast(broadcast_signal);

            // Conscious access is a meaningful event - process as reward
            let quality = RewardQuality::new()
                .with_depth(self.neuromodulators.acetylcholine.depth())
                .with_novelty(if emotion.is_significant { 0.7 } else { 0.3 })
                .with_goal_alignment(0.5) // Default; could be computed from goals
                .with_intrinsic(0.6)
                .with_durability(if memory.is_some() { 0.7 } else { 0.3 });

            self.neuromodulators.process_reward(
                0.3 * emotion.arousal.value(), // Magnitude based on arousal
                RewardCategory::Understanding,
                quality,
            );

            // Strong conscious experience triggers reflection
            if emotion.is_significant {
                let reflection = self
                    .dmn
                    .reflect(ReflectionTrigger::Emotional, Some(&content));
                reflections.push(reflection.content);

                // Signal deep processing for acetylcholine
                self.neuromodulators
                    .acetylcholine
                    .signal_deep_processing(0.5);
            }
        }

        // 7. Process prefrontal cycle
        self.prefrontal.process_cycle();

        // 8. DMN cycle (may generate scheduled reflection)
        if let Some(reflection) = self.dmn.process_cycle() {
            reflections.push(reflection.content);
        }

        // 9. Narrate significant experiences
        if emotion.is_significant {
            self.dmn.narrate(&content, emotion.arousal.value());
        }

        // 10. Neuromodulatory system homeostatic update
        self.neuromodulators.update();

        self.cycle_count += 1;

        Ok(ProcessingResult {
            tagged_signal,
            emotion,
            reached_consciousness,
            memory,
            predictions: Vec::new(),
            errors: Vec::new(),
            reflections,
        })
    }

    /// Process multiple inputs in sequence.
    pub fn process_batch(&mut self, inputs: Vec<String>) -> Result<Vec<ProcessingResult>> {
        inputs
            .into_iter()
            .map(|input| self.process(input))
            .collect()
    }

    /// Run a "sleep" cycle - consolidate memories, update beliefs.
    pub fn sleep(&mut self, hours: f64) -> Result<SleepReport> {
        // 1. Decay memories
        let forgotten = self.hippocampus.decay_all(hours)?;

        // 2. Consolidate important memories
        let unconsolidated = self.hippocampus.get_unconsolidated(100)?;
        let to_consolidate: Vec<_> = unconsolidated
            .iter()
            .filter(|m| m.valence.intensity() > 0.3 || m.salience.value() > 0.5)
            .map(|m| m.id)
            .collect();

        let consolidated_count = to_consolidate.len();
        self.hippocampus.mark_consolidated(&to_consolidate)?;

        // 3. Generate sleep reflection
        let reflection = self.dmn.reflect(
            ReflectionTrigger::Scheduled,
            Some("post-sleep consolidation"),
        );

        // 4. Decay emotional state toward neutral
        for _ in 0..(hours as usize) {
            self.amygdala.decay();
        }

        // 5. Neuromodulatory restoration during sleep
        // Sleep restores neuromodulator levels toward baseline
        let sleep_cycles = (hours * 2.0) as usize; // 2 update cycles per hour
        for _ in 0..sleep_cycles {
            self.neuromodulators.update();

            // Extra serotonin restoration during sleep (mood regulation)
            self.neuromodulators.serotonin.level.adjust(0.02);

            // Stress reduction during sleep
            if self.neuromodulators.norepinephrine.stress() > 0.0 {
                self.neuromodulators.signal_safety();
            }
        }

        // Clear focus during sleep
        self.neuromodulators.norepinephrine.clear_focus();

        // Cortisol restoration during sleep (critical for recovery)
        self.neuromodulators.cortisol.rest();

        Ok(SleepReport {
            hours_slept: hours,
            memories_forgotten: forgotten,
            memories_consolidated: consolidated_count,
            reflection: reflection.content,
        })
    }

    /// Recall memories related to a query.
    pub fn recall(&mut self, _query: &str, limit: usize) -> Result<Vec<MemoryTrace>> {
        // For now, just retrieve by valence boost
        // TODO: Implement semantic search
        self.hippocampus.retrieve(limit, true)
    }

    /// Ask the brain to reflect on something.
    pub fn reflect(&mut self, topic: &str) -> String {
        self.dmn
            .reflect(ReflectionTrigger::Query, Some(topic))
            .content
    }

    /// Get the brain's self-description.
    pub fn who_am_i(&self) -> String {
        self.dmn.who_am_i()
    }

    /// Add a belief.
    pub fn believe(&mut self, content: &str, category: BeliefCategory, confidence: f64) {
        let belief = Belief::new(content, confidence, category);
        self.dmn.add_belief(belief);
    }

    /// Focus attention on something.
    pub fn focus(&mut self, topic: &str) {
        self.thalamus.focus_attention(topic);
        // Also update norepinephrine focus
        self.neuromodulators.focus_on(topic.to_string());
    }

    /// Get current neuromodulator state.
    pub fn neuromodulator_state(&self) -> NeuromodulatorState {
        self.neuromodulators.state()
    }

    /// Process a reward event (for external reward signals).
    pub fn reward(&mut self, magnitude: f64, category: RewardCategory, quality: RewardQuality) {
        self.neuromodulators
            .process_reward(magnitude, category, quality);
    }

    /// Signal that a goal was achieved (high-quality reward).
    pub fn goal_achieved(&mut self, description: &str) {
        let quality = RewardQuality::new()
            .with_depth(0.8)
            .with_goal_alignment(1.0)
            .with_durability(0.9)
            .with_intrinsic(0.7)
            .with_novelty(0.5);

        self.neuromodulators
            .process_reward(0.7, RewardCategory::Achievement, quality);

        // Also reward patience if we waited for this
        self.neuromodulators.serotonin.reward_patience(0.3);

        // Signal success to cortisol system (reduces stress)
        self.neuromodulators.signal_success();

        // Narrate the achievement
        self.dmn.narrate(description, 0.8);
    }

    /// Signal a failure event (build error, test failure, etc.)
    /// Optionally provide an error signature to track repeated same errors.
    pub fn signal_failure(&mut self, error_description: &str, error_signature: Option<&str>) {
        self.neuromodulators.signal_failure(error_signature);

        // Process as negative emotional event
        let signal = BrainSignal::new("failure", SignalType::Error, error_description)
            .with_valence(-0.5)
            .with_arousal(0.6);
        self.dmn.update_emotional_state(signal.valence.value());

        // Narrate the failure
        self.dmn.narrate(error_description, 0.5);
    }

    /// Signal a success event (build passed, test passed)
    pub fn signal_success(&mut self, description: &str) {
        self.neuromodulators.signal_success();

        // Process as positive emotional event
        self.dmn.update_emotional_state(0.4);
        self.dmn.narrate(description, 0.4);
    }

    /// Check if we should try a different approach (from cortisol)
    pub fn should_pivot(&self) -> bool {
        self.neuromodulators.should_pivot()
    }

    /// Check if we should ask for help
    pub fn should_seek_help(&self) -> bool {
        self.neuromodulators.should_seek_help()
    }

    /// Check if we should take a break
    pub fn should_take_break(&self) -> bool {
        self.neuromodulators.should_take_break()
    }

    /// Get current frustration level
    pub fn frustration(&self) -> f64 {
        self.neuromodulators.frustration()
    }

    /// Get exploration drive (moderate stress promotes trying new things)
    pub fn exploration_drive(&self) -> f64 {
        self.neuromodulators.exploration_drive()
    }

    // --- GABA (Inhibitory Control) Methods ---

    /// Check if an action should be inhibited for deliberation
    /// Returns InhibitionResult::Proceed or InhibitionResult::Inhibited
    pub fn check_impulse(
        &mut self,
        action: &str,
        urgency: f64,
        risk: f64,
    ) -> crate::core::InhibitionResult {
        self.neuromodulators.check_impulse(action, urgency, risk)
    }

    /// Should we pause before a risky action?
    pub fn should_pause(&self, risk_level: f64) -> bool {
        self.neuromodulators.should_pause(risk_level)
    }

    /// Get impulse control quality (0-1)
    pub fn impulse_control(&self) -> f64 {
        self.neuromodulators.impulse_control()
    }

    /// Signal that deliberation led to a good outcome
    pub fn reward_deliberation(&mut self) {
        self.neuromodulators.reward_deliberation();
    }

    /// Signal that an impulsive action led to a bad outcome
    pub fn penalize_impulsivity(&mut self) {
        self.neuromodulators.penalize_impulsivity();
    }

    // --- Oxytocin (Trust/Cooperation) Methods ---

    /// Record a positive interaction with an entity (builds trust)
    pub fn record_positive_interaction(&mut self, entity: &str) {
        self.neuromodulators.record_positive_interaction(entity);
        // Also record in DMN's agent model
        self.dmn.get_agent_model(entity).record_interaction(true);
    }

    /// Record a negative interaction (betrayal, reduces trust)
    pub fn record_negative_interaction(&mut self, entity: &str) {
        self.neuromodulators.record_negative_interaction(entity);
        // Also record in DMN's agent model
        self.dmn.get_agent_model(entity).record_interaction(false);
    }

    /// Get trust level for an entity (0-1)
    pub fn get_trust(&self, entity: &str) -> f64 {
        self.neuromodulators.get_trust(entity)
    }

    /// Is an entity trusted?
    pub fn is_trusted(&self, entity: &str) -> bool {
        self.neuromodulators.is_trusted(entity)
    }

    /// Should we prefer a cooperative approach?
    pub fn prefer_cooperation(&self) -> bool {
        self.neuromodulators.prefer_cooperation()
    }

    /// Get information weight for a source (trusted = higher weight)
    pub fn source_weight(&self, source: &str) -> f64 {
        self.neuromodulators.source_weight(source)
    }

    /// Check if the brain advises patience (waiting for better outcome).
    pub fn should_wait(&self, immediate_value: f64, delayed_value: f64, delay_cycles: u32) -> bool {
        self.neuromodulators
            .should_wait_for_better(immediate_value, delayed_value, delay_cycles)
    }

    /// Get current conscious contents.
    pub fn conscious_contents(&self) -> Vec<&Broadcast> {
        self.workspace.conscious_contents()
    }

    /// Get working memory contents.
    pub fn working_memory(&self) -> Vec<&crate::regions::prefrontal::WorkingMemoryItem> {
        self.prefrontal.contents()
    }

    /// Get comprehensive brain statistics.
    pub fn stats(&self) -> BrainStats {
        let memory_stats = self.hippocampus.stats().unwrap_or_default();
        let workspace_stats = self.workspace.stats();
        let prefrontal_stats = self.prefrontal.stats();
        let dmn_stats = self.dmn.stats();
        let (thalamus_recv, thalamus_pass, thalamus_filt, _) = self.thalamus.stats();
        let _prediction_stats = self.prediction.stats(); // Kept for potential future use

        BrainStats {
            cycles: self.cycle_count,
            memories: memory_stats.total_memories as usize,
            conscious_items: workspace_stats.current_capacity_used,
            working_memory_items: prefrontal_stats.current_items,
            beliefs: dmn_stats.active_beliefs,
            emotional_state: dmn_stats.emotional_state,
            signals_processed: thalamus_recv,
            signals_passed: thalamus_pass,
            signals_filtered: thalamus_filt,
            learning_rate: self.neuromodulators.learning_rate(), // Use neuromodulator-derived rate
            neuromodulators: self.neuromodulators.state(),
        }
    }

    /// Get nervous system statistics (signal routing between regions).
    pub fn nervous_system_stats(&self) -> NervousSystemStats {
        self.nervous_system.stats()
    }

    /// Visualize the neural pathways between brain regions.
    pub fn visualize_pathways(&self) -> String {
        self.nervous_system.visualize()
    }

    /// Apply neuromodulator effects to nervous system pathways.
    /// Call this to reflect neuromodulator state changes in pathway strengths.
    pub fn sync_neuromodulators_to_pathways(&mut self) {
        let state = self.neuromodulators.state();

        // Norepinephrine increases Thalamus → Amygdala (heightened vigilance)
        let ne_factor = 1.0 + (state.norepinephrine - 0.5) * 0.4;
        self.nervous_system.apply_modulation(
            BrainRegion::Thalamus,
            BrainRegion::Amygdala,
            ne_factor,
        );

        // Acetylcholine enhances Hippocampus pathways (learning)
        let ach_factor = 1.0 + (state.acetylcholine - 0.5) * 0.3;
        self.nervous_system.apply_modulation(
            BrainRegion::Thalamus,
            BrainRegion::Hippocampus,
            ach_factor,
        );
        self.nervous_system.apply_modulation(
            BrainRegion::Amygdala,
            BrainRegion::Hippocampus,
            ach_factor,
        );

        // Dopamine enhances Prefrontal → Workspace (motivation for conscious access)
        let da_factor = 1.0 + (state.dopamine - 0.5) * 0.3;
        self.nervous_system.apply_modulation(
            BrainRegion::Prefrontal,
            BrainRegion::Workspace,
            da_factor,
        );

        // GABA reduces Amygdala → Workspace (inhibition of impulsive reactions)
        let gaba_factor = 1.0 - (state.gaba - 0.5) * 0.2;
        self.nervous_system.apply_modulation(
            BrainRegion::Amygdala,
            BrainRegion::Workspace,
            gaba_factor,
        );
    }

    // --- Schema (Pattern Learning) Methods ---

    /// Learn a new pattern from an experience.
    /// Returns the schema ID.
    pub fn learn_pattern(
        &mut self,
        pattern: &str,
        category: SchemaCategory,
        episode_id: Option<u64>,
        triggers: Vec<&str>,
    ) -> u64 {
        if let Some(ep_id) = episode_id {
            self.schemas
                .create_from_episode(pattern, category, ep_id, triggers)
        } else {
            let id = self.schemas.create(pattern, category);
            if let Some(schema) = self.schemas.get_mut(id) {
                for trigger in triggers {
                    schema.add_trigger(trigger);
                }
            }
            id
        }
    }

    /// Find relevant schemas for a situation.
    pub fn find_schemas(&mut self, query: &str) -> Vec<&Schema> {
        self.schemas.find(query)
    }

    /// Add evidence supporting a schema.
    pub fn support_schema(&mut self, schema_id: u64, episode_id: u64) {
        self.schemas.add_support(schema_id, episode_id);
    }

    /// Add evidence contradicting a schema (triggers potential revision).
    pub fn contradict_schema(&mut self, schema_id: u64, episode_id: u64) {
        self.schemas.add_contradiction(schema_id, episode_id);
    }

    /// Get schemas that need revision due to too many contradictions.
    pub fn schemas_needing_revision(&self) -> Vec<&Schema> {
        self.schemas.find_needs_revision()
    }

    /// Get schema statistics.
    pub fn schema_stats(&self) -> SchemaStats {
        self.schemas.stats()
    }

    /// Get high-confidence schemas (well-established patterns).
    pub fn confident_schemas(&self, min_confidence: f64) -> Vec<&Schema> {
        self.schemas.find_confident(min_confidence)
    }
}

impl Default for Brain {
    fn default() -> Self {
        Self::new().expect("Failed to create default brain")
    }
}

/// Report from a sleep cycle.
#[derive(Debug, Clone)]
pub struct SleepReport {
    pub hours_slept: f64,
    pub memories_forgotten: usize,
    pub memories_consolidated: usize,
    pub reflection: String,
}

/// Comprehensive brain statistics.
#[derive(Debug, Clone)]
pub struct BrainStats {
    pub cycles: u64,
    pub memories: usize,
    pub conscious_items: usize,
    pub working_memory_items: usize,
    pub beliefs: usize,
    pub emotional_state: f64,
    pub signals_processed: u64,
    pub signals_passed: u64,
    pub signals_filtered: u64,
    pub learning_rate: f64,
    /// Neuromodulator levels and derived states
    pub neuromodulators: NeuromodulatorState,
}

impl Default for BrainStats {
    fn default() -> Self {
        Self {
            cycles: 0,
            memories: 0,
            conscious_items: 0,
            working_memory_items: 0,
            beliefs: 0,
            emotional_state: 0.0,
            signals_processed: 0,
            signals_passed: 0,
            signals_filtered: 0,
            learning_rate: 0.1,
            neuromodulators: NeuromodulatorState {
                dopamine: 0.5,
                serotonin: 0.5,
                norepinephrine: 0.4,
                acetylcholine: 0.5,
                cortisol: 0.2,
                gaba: 0.5,
                oxytocin: 0.5,
                motivation: 0.5,
                patience: 0.5,
                stress: 0.0,
                learning_depth: 0.5,
                frustration: 0.0,
                exploration_drive: 0.4,
                impulse_control: 0.5,
                cooperativeness: 0.6,
                is_satiated: false,
                is_stressed: false,
                is_burned_out: false,
                is_deliberating: false,
                should_pivot: false,
                should_seek_help: false,
                prefer_cooperation: true,
                mood_stability: 0.8,
            },
        }
    }
}

// Implement default for MemoryStats
impl Default for crate::regions::hippocampus::MemoryStats {
    fn default() -> Self {
        Self {
            total_memories: 0,
            avg_valence: 0.0,
            avg_strength: 0.0,
            consolidated: 0,
            positive_memories: 0,
            negative_memories: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_brain_creation() {
        let brain = Brain::new().unwrap();
        assert_eq!(brain.cycle_count, 0);
    }

    #[test]
    fn test_process_input() {
        let mut brain = Brain::new().unwrap();

        let _result = brain.process("Hello, world!").unwrap();

        assert!(brain.cycle_count > 0);
    }

    #[test]
    fn test_emotional_processing() {
        let mut brain = Brain::new().unwrap();

        // Process positive input
        let result = brain.process("Amazing success! Victory!").unwrap();
        assert!(result.emotion.valence.is_positive());

        // Process negative input
        let result = brain
            .process("Terrible failure, everything is bad")
            .unwrap();
        assert!(result.emotion.valence.is_negative());
    }

    #[test]
    fn test_memory_encoding() {
        let mut brain = Brain::new().unwrap();

        // Process high-value content multiple times
        brain.process("Amazing important discovery!").unwrap();
        brain.process("Critical success achieved!").unwrap();
        brain.process("Wonderful breakthrough!").unwrap();

        // Check that processing happened
        let stats = brain.stats();
        assert!(stats.cycles >= 3);
    }

    #[test]
    fn test_sleep_cycle() {
        let mut brain = Brain::new().unwrap();

        // Process some inputs
        brain.process("Event one").unwrap();
        brain.process("Event two").unwrap();
        brain.process("Event three").unwrap();

        // Sleep
        let report = brain.sleep(8.0).unwrap();

        assert_eq!(report.hours_slept, 8.0);
    }

    #[test]
    fn test_identity() {
        let mut brain = Brain::new().unwrap();

        let identity = Identity {
            name: "TestBrain".to_string(),
            core_values: vec!["testing".to_string()],
            self_description: "A test brain".to_string(),
            creation_time: chrono::Utc::now(),
        };

        brain.set_identity(identity);

        let description = brain.who_am_i();
        assert!(description.contains("TestBrain"));
    }

    #[test]
    fn test_beliefs() {
        let mut brain = Brain::new().unwrap();

        brain.believe("I can process signals", BeliefCategory::SelfCapability, 0.8);

        let stats = brain.stats();
        assert_eq!(stats.beliefs, 1);
    }

    #[test]
    fn test_full_cognitive_cycle() {
        let mut brain = Brain::new().unwrap();

        // Set identity
        let identity = Identity {
            name: "Rata".to_string(),
            core_values: vec!["curiosity".to_string(), "memory".to_string()],
            self_description: "A digital squirrel".to_string(),
            creation_time: chrono::Utc::now(),
        };
        brain.set_identity(identity);

        // Add beliefs
        brain.believe(
            "I can learn from experience",
            BeliefCategory::SelfCapability,
            0.9,
        );

        // Process experiences
        brain.process("Learning about consciousness").unwrap();
        brain.process("Great success in memory research!").unwrap();
        brain.process("Encountered a difficult problem").unwrap();

        // Reflect
        let reflection = brain.reflect("my progress");
        assert!(!reflection.is_empty());

        // Sleep and consolidate
        let _report = brain.sleep(6.0).unwrap();

        // Check state - verify processing happened
        let stats = brain.stats();
        assert!(stats.cycles >= 3);
        assert!(stats.beliefs >= 1);

        // Verify identity was set
        let who = brain.who_am_i();
        assert!(who.contains("Rata"));
    }
}
