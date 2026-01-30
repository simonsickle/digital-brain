//! Digital Brain - Complete Integration
//!
//! This module wires all brain regions together into a functioning whole.
//! The Brain struct coordinates processing cycles across all components.

use crate::core::prediction::{PredictionEngine, Prediction, PredictionError};
use crate::core::workspace::{GlobalWorkspace, WorkspaceConfig, Broadcast};
use crate::regions::amygdala::{Amygdala, EmotionalAppraisal};
use crate::regions::dmn::{DefaultModeNetwork, Identity, Belief, BeliefCategory, ReflectionTrigger};
use crate::regions::hippocampus::HippocampusStore;
use crate::regions::prefrontal::{PrefrontalCortex, PrefrontalConfig};
use crate::regions::thalamus::{Thalamus, ThalamusConfig, RoutedSignal, Destination};
use crate::signal::{BrainSignal, SignalType, MemoryTrace};
use crate::error::Result;

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
    /// Processing cycle count
    cycle_count: u64,
    /// Configuration
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
        let signal = BrainSignal::new("external", SignalType::Sensory, &content);

        // 2. Gate through thalamus
        self.thalamus.receive(signal);
        let routed_signals = self.thalamus.process_cycle();

        if routed_signals.is_empty() {
            // Signal was filtered - create minimal result
            return Ok(ProcessingResult {
                tagged_signal: BrainSignal::new("external", SignalType::Sensory, &content),
                emotion: self.amygdala.appraise(&BrainSignal::new("external", SignalType::Sensory, &content)),
                reached_consciousness: false,
                memory: None,
                predictions: Vec::new(),
                errors: Vec::new(),
                reflections: Vec::new(),
            });
        }

        let routed = &routed_signals[0];

        // 3. Emotional tagging by amygdala
        let tagged_signal = self.amygdala.tag_signal(routed.signal.clone());
        let emotion = self.amygdala.appraise(&tagged_signal);

        // 4. Update DMN emotional state
        self.dmn.update_emotional_state(tagged_signal.valence.value());

        // 5. Route to appropriate modules
        let mut reached_consciousness = false;
        let mut memory = None;
        let mut reflections = Vec::new();

        for dest in &routed.destinations {
            match dest {
                Destination::Workspace => {
                    self.workspace.submit(tagged_signal.clone());
                }
                Destination::Hippocampus => {
                    memory = Some(self.hippocampus.encode(&tagged_signal)?);
                }
                Destination::Prefrontal => {
                    self.prefrontal.load(&tagged_signal);
                }
                _ => {}
            }
        }

        // 6. Process workspace cycle
        let broadcasts = self.workspace.process_cycle();
        if !broadcasts.is_empty() {
            reached_consciousness = true;
            
            // Strong conscious experience triggers reflection
            if emotion.is_significant {
                let reflection = self.dmn.reflect(
                    ReflectionTrigger::Emotional,
                    Some(&content),
                );
                reflections.push(reflection.content);
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
        inputs.into_iter()
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

        Ok(SleepReport {
            hours_slept: hours,
            memories_forgotten: forgotten,
            memories_consolidated: consolidated_count,
            reflection: reflection.content,
        })
    }

    /// Recall memories related to a query.
    pub fn recall(&mut self, query: &str, limit: usize) -> Result<Vec<MemoryTrace>> {
        // For now, just retrieve by valence boost
        // TODO: Implement semantic search
        self.hippocampus.retrieve(limit, true)
    }

    /// Ask the brain to reflect on something.
    pub fn reflect(&mut self, topic: &str) -> String {
        self.dmn.reflect(ReflectionTrigger::Query, Some(topic)).content
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
        let prediction_stats = self.prediction.stats();

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
            learning_rate: prediction_stats.current_learning_rate,
        }
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
#[derive(Debug, Clone, Default)]
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
        
        let result = brain.process("Hello, world!").unwrap();
        
        assert!(brain.cycle_count > 0);
    }

    #[test]
    fn test_emotional_processing() {
        let mut brain = Brain::new().unwrap();
        
        // Process positive input
        let result = brain.process("Amazing success! Victory!").unwrap();
        assert!(result.emotion.valence.is_positive());
        
        // Process negative input
        let result = brain.process("Terrible failure, everything is bad").unwrap();
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
        brain.believe("I can learn from experience", BeliefCategory::SelfCapability, 0.9);
        
        // Process experiences
        brain.process("Learning about consciousness").unwrap();
        brain.process("Great success in memory research!").unwrap();
        brain.process("Encountered a difficult problem").unwrap();
        
        // Reflect
        let reflection = brain.reflect("my progress");
        assert!(!reflection.is_empty());
        
        // Sleep and consolidate
        let report = brain.sleep(6.0).unwrap();
        
        // Check state - verify processing happened
        let stats = brain.stats();
        assert!(stats.cycles >= 3);
        assert!(stats.beliefs >= 1);
        
        // Verify identity was set
        let who = brain.who_am_i();
        assert!(who.contains("Rata"));
    }
}
