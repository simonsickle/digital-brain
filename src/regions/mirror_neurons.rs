//! Mirror Neuron System - Learning by Observation
//!
//! Mirror neurons fire both when performing an action and when observing
//! others perform it. This mechanism underlies:
//! - **Imitation learning**: Learning new behaviors by watching
//! - **Action understanding**: Grasping the intent behind actions
//! - **Empathy**: Understanding others' mental states
//! - **Skill acquisition**: Picking up techniques through observation
//!
//! For coding, this translates to:
//! - **Learning from examples**: Understanding code patterns by seeing them
//! - **Intent recognition**: Understanding why code was written a certain way
//! - **Style adaptation**: Matching coding style to the codebase
//! - **Pattern acquisition**: Learning idioms and best practices
//!
//! # Theoretical Foundation
//!
//! Based on:
//! - Rizzolatti's mirror neuron discovery in premotor cortex
//! - Action-perception coupling (common coding theory)
//! - Observational learning (Bandura's social learning)
//! - Motor simulation theory
//!
//! # Architecture
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────────────┐
//! │                    MIRROR NEURON SYSTEM                            │
//! ├──────────────────┬──────────────────┬──────────────────┬──────────┤
//! │   Observation    │   Simulation     │   Imitation      │ Intent   │
//! │   Buffer         │   Engine         │   Generator      │ Decoder  │
//! │   - examples     │   - replay       │   - templates    │ - goals  │
//! │   - patterns     │   - mental exec  │   - adaptation   │ - motives│
//! │   - actions      │   - prediction   │   - generation   │ - context│
//! └──────────────────┴──────────────────┴──────────────────┴──────────┘
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

// ============================================================================
// OBSERVED ACTIONS
// ============================================================================

/// An observed coding action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservedAction {
    /// Unique identifier
    pub id: String,
    /// Type of action
    pub action_type: ActionType,
    /// The code/content involved
    pub content: String,
    /// Context in which it occurred
    pub context: ActionContext,
    /// Inferred intent behind the action
    pub intent: Option<InferredIntent>,
    /// When observed
    pub observed_at: DateTime<Utc>,
    /// Quality assessment (0-1)
    pub quality: f64,
    /// How many times similar pattern seen
    pub observation_count: u32,
    /// Whether successfully imitated
    pub imitated: bool,
}

/// Types of coding actions
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ActionType {
    /// Creating new code
    Create { entity: EntityType },
    /// Modifying existing code
    Modify { change_type: ChangeType },
    /// Deleting code
    Delete { reason: Option<String> },
    /// Refactoring code
    Refactor { pattern: RefactorPattern },
    /// Adding tests
    Test { coverage_type: CoverageType },
    /// Fixing bugs
    BugFix { bug_type: BugType },
    /// Adding documentation
    Document,
    /// Configuring/setup
    Configure,
}

/// Types of entities being created
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EntityType {
    Function,
    Struct,
    Enum,
    Trait,
    Module,
    Test,
    Constant,
    Type,
}

/// Types of changes
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChangeType {
    AddFeature,
    FixBehavior,
    Optimize,
    CleanUp,
    UpdateDependency,
}

/// Refactoring patterns
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RefactorPattern {
    ExtractFunction,
    InlineFunction,
    RenameSymbol,
    MoveCode,
    SimplifyLogic,
    RemoveDuplication,
    ImproveTypes,
}

/// Test coverage types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CoverageType {
    UnitTest,
    IntegrationTest,
    EdgeCase,
    ErrorHandling,
}

/// Bug types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BugType {
    Logic,
    OffByOne,
    NullReference,
    TypeMismatch,
    ConcurrencyRace,
    MemoryLeak,
    Security,
    Performance,
}

// ============================================================================
// CONTEXT
// ============================================================================

/// Context in which an action occurs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionContext {
    /// File/module where action occurred
    pub location: String,
    /// Programming language
    pub language: Option<String>,
    /// Surrounding code patterns
    pub patterns: Vec<String>,
    /// Project/codebase identifier
    pub project: Option<String>,
    /// Author/source of the action
    pub source: ActionSource,
}

/// Source of observed action
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActionSource {
    /// Self-performed
    Self_,
    /// Human developer
    Human { identifier: Option<String> },
    /// Another AI
    AI { model: Option<String> },
    /// Documentation/tutorial
    Documentation,
    /// Code review
    Review,
    /// Unknown
    Unknown,
}

// ============================================================================
// INTENT INFERENCE
// ============================================================================

/// Inferred intent behind an action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferredIntent {
    /// Primary goal
    pub goal: IntentGoal,
    /// Confidence in inference (0-1)
    pub confidence: f64,
    /// Supporting evidence
    pub evidence: Vec<String>,
    /// Alternative interpretations
    pub alternatives: Vec<IntentGoal>,
}

/// Possible goals behind actions
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IntentGoal {
    /// Implement new functionality
    ImplementFeature,
    /// Fix a bug
    FixBug,
    /// Improve performance
    Optimize,
    /// Make code cleaner/readable
    ImproveReadability,
    /// Add safety/robustness
    AddSafety,
    /// Satisfy requirements
    MeetRequirement,
    /// Follow best practices
    FollowConvention,
    /// Enable future changes
    PrepareForFuture,
    /// Satisfy tests
    PassTests,
    /// Educational/learning
    Learning,
}

// ============================================================================
// LEARNED PATTERNS
// ============================================================================

/// A learned coding pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedPattern {
    /// Pattern identifier
    pub id: String,
    /// Pattern name
    pub name: String,
    /// Category
    pub category: PatternCategory,
    /// Template code
    pub template: String,
    /// When to use this pattern
    pub use_when: Vec<String>,
    /// When NOT to use
    pub avoid_when: Vec<String>,
    /// Quality of pattern (0-1)
    pub quality: f64,
    /// Times observed
    pub observation_count: u32,
    /// Times successfully used
    pub success_count: u32,
    /// Times it caused issues
    pub failure_count: u32,
    /// Source of learning
    pub learned_from: ActionSource,
    /// When first learned
    pub learned_at: DateTime<Utc>,
}

/// Pattern categories
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PatternCategory {
    /// Error handling
    ErrorHandling,
    /// Resource management
    ResourceManagement,
    /// Concurrency
    Concurrency,
    /// Data structures
    DataStructure,
    /// Algorithm
    Algorithm,
    /// API design
    ApiDesign,
    /// Testing
    Testing,
    /// Configuration
    Configuration,
    /// Logging/debugging
    Debugging,
    /// Style/formatting
    Style,
}

// ============================================================================
// SIMULATION
// ============================================================================

/// A mental simulation of code execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Simulation {
    /// What's being simulated
    pub target: String,
    /// Input state
    pub input: HashMap<String, String>,
    /// Expected output
    pub expected_output: Option<String>,
    /// Actual simulated output
    pub simulated_output: Option<String>,
    /// Steps in simulation
    pub steps: Vec<SimulationStep>,
    /// Did simulation succeed?
    pub success: bool,
    /// Confidence in simulation (0-1)
    pub confidence: f64,
}

/// A step in mental simulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationStep {
    /// Step description
    pub description: String,
    /// State after this step
    pub state: HashMap<String, String>,
    /// Any issues detected
    pub issues: Vec<String>,
}

// ============================================================================
// CONFIGURATION
// ============================================================================

/// Configuration for Mirror Neuron System
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MirrorNeuronConfig {
    /// Maximum observations to keep
    pub max_observations: usize,
    /// Maximum patterns to learn
    pub max_patterns: usize,
    /// Quality threshold for learning
    pub learning_threshold: f64,
    /// Decay rate for observation relevance
    pub observation_decay: f64,
    /// Enable intent inference
    pub enable_intent_inference: bool,
    /// Minimum observations before learning pattern
    pub min_observations_to_learn: u32,
}

impl Default for MirrorNeuronConfig {
    fn default() -> Self {
        Self {
            max_observations: 1000,
            max_patterns: 200,
            learning_threshold: 0.6,
            observation_decay: 0.01,
            enable_intent_inference: true,
            min_observations_to_learn: 3,
        }
    }
}

// ============================================================================
// MIRROR NEURON SYSTEM
// ============================================================================

/// The Mirror Neuron System - learning by observation
pub struct MirrorNeuronSystem {
    config: MirrorNeuronConfig,
    /// Observed actions
    observations: VecDeque<ObservedAction>,
    /// Learned patterns
    patterns: HashMap<String, LearnedPattern>,
    /// Pattern matching cache (action_type -> pattern_ids)
    pattern_index: HashMap<String, Vec<String>>,
    /// Recent simulations
    simulations: VecDeque<Simulation>,
    /// Statistics
    stats: MirrorNeuronStats,
}

/// Statistics for Mirror Neuron System
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MirrorNeuronStats {
    /// Total observations
    pub total_observations: u64,
    /// Patterns learned
    pub patterns_learned: usize,
    /// Successful imitations
    pub successful_imitations: u64,
    /// Failed imitations
    pub failed_imitations: u64,
    /// Simulations run
    pub simulations_run: u64,
    /// Average pattern quality
    pub avg_pattern_quality: f64,
}

impl MirrorNeuronSystem {
    /// Create a new Mirror Neuron System
    pub fn new() -> Self {
        Self::with_config(MirrorNeuronConfig::default())
    }

    /// Create with specific configuration
    pub fn with_config(config: MirrorNeuronConfig) -> Self {
        Self {
            config,
            observations: VecDeque::new(),
            patterns: HashMap::new(),
            pattern_index: HashMap::new(),
            simulations: VecDeque::new(),
            stats: MirrorNeuronStats::default(),
        }
    }

    // ========================================================================
    // OBSERVATION
    // ========================================================================

    /// Observe a coding action
    pub fn observe(&mut self, action: ObservedAction) {
        self.stats.total_observations += 1;

        // Infer intent if enabled and not already present
        let action = if self.config.enable_intent_inference && action.intent.is_none() {
            let mut action = action;
            action.intent = Some(self.infer_intent(&action));
            action
        } else {
            action
        };

        // Check if similar pattern already observed
        let similar = self.find_similar_pattern(&action);
        if let Some(pattern_id) = similar
            && let Some(pattern) = self.patterns.get_mut(&pattern_id)
        {
            pattern.observation_count += 1;
        }

        // Try to learn a new pattern
        self.try_learn_pattern(&action);

        // Add to observations
        self.observations.push_back(action);

        // Enforce size limit
        while self.observations.len() > self.config.max_observations {
            self.observations.pop_front();
        }
    }

    /// Observe from code example
    pub fn observe_example(&mut self, code: &str, _description: &str, source: ActionSource) {
        let action = ObservedAction {
            id: format!("obs:{}", Utc::now().timestamp_millis()),
            action_type: self.infer_action_type(code),
            content: code.to_string(),
            context: ActionContext {
                location: "example".to_string(),
                language: Some(self.detect_language(code)),
                patterns: self.extract_patterns(code),
                project: None,
                source,
            },
            intent: None,
            observed_at: Utc::now(),
            quality: 0.8, // Examples are usually good quality
            observation_count: 1,
            imitated: false,
        };

        self.observe(action);
    }

    /// Infer action type from code
    fn infer_action_type(&self, code: &str) -> ActionType {
        if code.contains("#[test]") || code.contains("fn test_") {
            ActionType::Test {
                coverage_type: CoverageType::UnitTest,
            }
        } else if code.contains("fn ") || code.contains("pub fn ") {
            ActionType::Create {
                entity: EntityType::Function,
            }
        } else if code.contains("struct ") {
            ActionType::Create {
                entity: EntityType::Struct,
            }
        } else if code.contains("enum ") {
            ActionType::Create {
                entity: EntityType::Enum,
            }
        } else if code.contains("impl ") {
            ActionType::Modify {
                change_type: ChangeType::AddFeature,
            }
        } else {
            ActionType::Modify {
                change_type: ChangeType::CleanUp,
            }
        }
    }

    /// Detect programming language
    fn detect_language(&self, code: &str) -> String {
        if code.contains("fn ") && code.contains("let ") {
            "rust".to_string()
        } else if code.contains("def ") && code.contains(":") {
            "python".to_string()
        } else if code.contains("function") || code.contains("const ") {
            "javascript".to_string()
        } else {
            "unknown".to_string()
        }
    }

    /// Extract patterns from code
    fn extract_patterns(&self, code: &str) -> Vec<String> {
        let mut patterns = Vec::new();

        if code.contains("Result<") || code.contains("Result::") {
            patterns.push("error_handling".to_string());
        }
        if code.contains("?") {
            patterns.push("error_propagation".to_string());
        }
        if code.contains("match ") {
            patterns.push("pattern_matching".to_string());
        }
        if code.contains(".iter()") || code.contains(".map(") {
            patterns.push("iterator".to_string());
        }
        if code.contains("impl ") && code.contains("for ") {
            patterns.push("trait_impl".to_string());
        }
        if code.contains("Arc<") || code.contains("Mutex<") {
            patterns.push("concurrency".to_string());
        }
        if code.contains("#[derive(") {
            patterns.push("derive_macro".to_string());
        }
        if code.contains("pub fn new(") {
            patterns.push("constructor".to_string());
        }

        patterns
    }

    // ========================================================================
    // INTENT INFERENCE
    // ========================================================================

    /// Infer intent behind an action
    pub fn infer_intent(&self, action: &ObservedAction) -> InferredIntent {
        let mut evidence = Vec::new();
        let mut goals = Vec::new();

        // Analyze action type
        match &action.action_type {
            ActionType::Test { .. } => {
                goals.push((IntentGoal::PassTests, 0.9));
                evidence.push("Test code detected".to_string());
            }
            ActionType::BugFix { .. } => {
                goals.push((IntentGoal::FixBug, 0.9));
                evidence.push("Bug fix action".to_string());
            }
            ActionType::Refactor { pattern } => {
                goals.push((IntentGoal::ImproveReadability, 0.7));
                evidence.push(format!("Refactoring: {:?}", pattern));
            }
            ActionType::Create { entity } => {
                goals.push((IntentGoal::ImplementFeature, 0.6));
                evidence.push(format!("Creating new {:?}", entity));
            }
            ActionType::Document => {
                goals.push((IntentGoal::ImproveReadability, 0.8));
                evidence.push("Documentation added".to_string());
            }
            _ => {}
        }

        // Analyze content patterns
        if action.content.contains("// Safety:") || action.content.contains("// SAFETY:") {
            goals.push((IntentGoal::AddSafety, 0.8));
            evidence.push("Safety comment present".to_string());
        }
        if action.content.contains("TODO") || action.content.contains("FIXME") {
            goals.push((IntentGoal::PrepareForFuture, 0.5));
            evidence.push("TODO/FIXME comment".to_string());
        }
        if action.content.contains("optimize") || action.content.contains("perf") {
            goals.push((IntentGoal::Optimize, 0.7));
            evidence.push("Performance-related content".to_string());
        }

        // Sort by confidence
        goals.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let (primary_goal, confidence) = goals
            .first()
            .cloned()
            .unwrap_or((IntentGoal::ImplementFeature, 0.3));
        let alternatives: Vec<_> = goals.iter().skip(1).map(|(g, _)| g.clone()).collect();

        InferredIntent {
            goal: primary_goal,
            confidence,
            evidence,
            alternatives,
        }
    }

    // ========================================================================
    // PATTERN LEARNING
    // ========================================================================

    /// Try to learn a pattern from observation
    fn try_learn_pattern(&mut self, action: &ObservedAction) {
        // Only learn from high-quality observations
        if action.quality < self.config.learning_threshold {
            return;
        }

        // Check if we've seen this enough times
        let similar_count = self
            .observations
            .iter()
            .filter(|o| self.are_similar(o, action))
            .count();

        if similar_count >= self.config.min_observations_to_learn as usize {
            let pattern = self.extract_pattern(action);
            let pattern_id = pattern.id.clone();

            self.patterns.insert(pattern_id.clone(), pattern);
            self.stats.patterns_learned = self.patterns.len();

            // Update index
            let type_key = format!("{:?}", action.action_type);
            self.pattern_index
                .entry(type_key)
                .or_default()
                .push(pattern_id);
        }

        // Enforce pattern limit
        if self.patterns.len() > self.config.max_patterns {
            self.prune_low_quality_patterns();
        }

        // Update average quality
        if !self.patterns.is_empty() {
            let total_quality: f64 = self.patterns.values().map(|p| p.quality).sum();
            self.stats.avg_pattern_quality = total_quality / self.patterns.len() as f64;
        }
    }

    /// Check if two actions are similar
    fn are_similar(&self, a: &ObservedAction, b: &ObservedAction) -> bool {
        if a.action_type != b.action_type {
            return false;
        }

        // Check for similar patterns in context
        let pattern_overlap = a
            .context
            .patterns
            .iter()
            .filter(|p| b.context.patterns.contains(p))
            .count();

        pattern_overlap > 0 || a.context.language == b.context.language
    }

    /// Extract a reusable pattern from an action
    fn extract_pattern(&self, action: &ObservedAction) -> LearnedPattern {
        let category = match &action.action_type {
            ActionType::Test { .. } => PatternCategory::Testing,
            ActionType::BugFix { bug_type } => match bug_type {
                BugType::ConcurrencyRace => PatternCategory::Concurrency,
                BugType::MemoryLeak => PatternCategory::ResourceManagement,
                _ => PatternCategory::ErrorHandling,
            },
            ActionType::Refactor { .. } => PatternCategory::Style,
            ActionType::Create { entity } => match entity {
                EntityType::Struct | EntityType::Enum => PatternCategory::DataStructure,
                EntityType::Test => PatternCategory::Testing,
                _ => PatternCategory::ApiDesign,
            },
            _ => PatternCategory::Style,
        };

        let use_when = action
            .intent
            .as_ref()
            .map(|i| vec![format!("When goal is {:?}", i.goal)])
            .unwrap_or_default();

        LearnedPattern {
            id: format!("pattern:{}", Utc::now().timestamp_millis()),
            name: format!("{:?} pattern", action.action_type),
            category,
            template: self.generalize_code(&action.content),
            use_when,
            avoid_when: Vec::new(),
            quality: action.quality,
            observation_count: 1,
            success_count: 0,
            failure_count: 0,
            learned_from: action.context.source.clone(),
            learned_at: Utc::now(),
        }
    }

    /// Generalize code into a template
    fn generalize_code(&self, code: &str) -> String {
        // Simple generalization: replace specific names with placeholders
        let mut template = code.to_string();

        // Replace function names (simple approach without regex)
        for line in code.lines() {
            let trimmed = line.trim();
            if (trimmed.starts_with("fn ") || trimmed.starts_with("pub fn "))
                && let Some(name) = self.extract_identifier(trimmed, "fn ")
            {
                template = template.replace(&format!("fn {}", name), "fn <FUNCTION_NAME>");
            }
        }

        // Replace variable names (simple heuristic)
        for line in code.lines() {
            let trimmed = line.trim();
            if let Some(after_let) = trimmed
                .strip_prefix("let mut ")
                .or_else(|| trimmed.strip_prefix("let "))
                && let Some(name) = after_let
                    .split(|c: char| !c.is_alphanumeric() && c != '_')
                    .next()
                && !name.is_empty()
                && name != "_"
            {
                template = template.replace(&format!("let {}", name), "let <VAR>");
                template = template.replace(&format!("let mut {}", name), "let mut <VAR>");
            }
        }

        template
    }

    /// Extract identifier after a keyword
    fn extract_identifier(&self, line: &str, keyword: &str) -> Option<String> {
        if let Some(pos) = line.find(keyword) {
            let after = &line[pos + keyword.len()..];
            after
                .split(|c: char| !c.is_alphanumeric() && c != '_')
                .next()
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
        } else {
            None
        }
    }

    /// Find a similar existing pattern
    fn find_similar_pattern(&self, action: &ObservedAction) -> Option<String> {
        let type_key = format!("{:?}", action.action_type);

        if let Some(pattern_ids) = self.pattern_index.get(&type_key) {
            for pid in pattern_ids {
                if let Some(pattern) = self.patterns.get(pid) {
                    // Check if patterns match
                    let has_similar_patterns = action
                        .context
                        .patterns
                        .iter()
                        .any(|p| pattern.template.contains(p));

                    if has_similar_patterns {
                        return Some(pid.clone());
                    }
                }
            }
        }

        None
    }

    /// Remove low-quality patterns
    fn prune_low_quality_patterns(&mut self) {
        let threshold = self.config.learning_threshold * 0.8;
        self.patterns.retain(|_, p| p.quality >= threshold);
    }

    // ========================================================================
    // IMITATION
    // ========================================================================

    /// Generate code by imitating a learned pattern
    pub fn imitate(
        &mut self,
        pattern_id: &str,
        context: &HashMap<String, String>,
    ) -> Option<String> {
        let pattern = self.patterns.get_mut(pattern_id)?;

        let mut code = pattern.template.clone();

        // Apply context substitutions
        for (key, value) in context {
            let placeholder = format!("<{}>", key.to_uppercase());
            code = code.replace(&placeholder, value);
        }

        // Check if all placeholders were filled
        if code.contains('<') && code.contains('>') {
            self.stats.failed_imitations += 1;
            pattern.failure_count += 1;
            return None;
        }

        self.stats.successful_imitations += 1;
        pattern.success_count += 1;

        Some(code)
    }

    /// Get patterns suitable for a given intent
    pub fn patterns_for_intent(&self, intent: &IntentGoal) -> Vec<&LearnedPattern> {
        self.patterns
            .values()
            .filter(|p| {
                p.use_when
                    .iter()
                    .any(|w| w.contains(&format!("{:?}", intent)))
            })
            .collect()
    }

    /// Get best patterns by category
    pub fn best_patterns(&self, category: PatternCategory, limit: usize) -> Vec<&LearnedPattern> {
        let mut patterns: Vec<_> = self
            .patterns
            .values()
            .filter(|p| p.category == category)
            .collect();

        patterns.sort_by(|a, b| {
            let a_score = a.quality * (a.success_count as f64 / (a.failure_count + 1) as f64);
            let b_score = b.quality * (b.success_count as f64 / (b.failure_count + 1) as f64);
            b_score.partial_cmp(&a_score).unwrap()
        });

        patterns.truncate(limit);
        patterns
    }

    // ========================================================================
    // SIMULATION
    // ========================================================================

    /// Mentally simulate code execution
    pub fn simulate(&mut self, code: &str, inputs: HashMap<String, String>) -> Simulation {
        self.stats.simulations_run += 1;

        let mut steps = Vec::new();
        let mut state = inputs.clone();
        let mut success = true;
        let mut confidence = 0.8;

        // Simple simulation by parsing code structure
        for line in code.lines() {
            let trimmed = line.trim();

            if trimmed.is_empty() || trimmed.starts_with("//") {
                continue;
            }

            let mut issues = Vec::new();

            // Simulate assignments
            if let Some(after_let) = trimmed.strip_prefix("let ")
                && let Some(eq_pos) = after_let.find('=')
            {
                let var_part = after_let[..eq_pos].trim();
                let var_name = var_part.trim_start_matches("mut ").trim();
                state.insert(var_name.to_string(), "initialized".to_string());
            }

            // Check for potential issues
            if trimmed.contains(".unwrap()") {
                issues.push("Potential panic on unwrap".to_string());
                confidence *= 0.9;
            }
            if trimmed.contains("panic!") {
                issues.push("Explicit panic".to_string());
                success = false;
            }

            steps.push(SimulationStep {
                description: trimmed.to_string(),
                state: state.clone(),
                issues,
            });
        }

        let simulation = Simulation {
            target: code.to_string(),
            input: inputs,
            expected_output: None,
            simulated_output: if success {
                Some("Execution completed".to_string())
            } else {
                None
            },
            steps,
            success,
            confidence,
        };

        self.simulations.push_back(simulation.clone());
        if self.simulations.len() > 100 {
            self.simulations.pop_front();
        }

        simulation
    }

    // ========================================================================
    // QUERIES
    // ========================================================================

    /// Get recent observations
    pub fn recent_observations(&self, limit: usize) -> Vec<&ObservedAction> {
        self.observations.iter().rev().take(limit).collect()
    }

    /// Get pattern by ID
    pub fn get_pattern(&self, id: &str) -> Option<&LearnedPattern> {
        self.patterns.get(id)
    }

    /// Get all patterns
    pub fn all_patterns(&self) -> Vec<&LearnedPattern> {
        self.patterns.values().collect()
    }

    /// Get statistics
    pub fn stats(&self) -> &MirrorNeuronStats {
        &self.stats
    }

    /// Decay observation relevance
    pub fn decay_observations(&mut self) {
        for obs in &mut self.observations {
            obs.quality *= 1.0 - self.config.observation_decay;
        }
    }
}

impl Default for MirrorNeuronSystem {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_observation() {
        let mut system = MirrorNeuronSystem::new();

        system.observe_example(
            "fn hello() { println!(\"Hello\"); }",
            "Simple function",
            ActionSource::Documentation,
        );

        assert_eq!(system.stats.total_observations, 1);
        assert!(!system.observations.is_empty());
    }

    #[test]
    fn test_intent_inference() {
        let system = MirrorNeuronSystem::new();

        let action = ObservedAction {
            id: "test".to_string(),
            action_type: ActionType::Test {
                coverage_type: CoverageType::UnitTest,
            },
            content: "#[test]\nfn test_something() {}".to_string(),
            context: ActionContext {
                location: "test.rs".to_string(),
                language: Some("rust".to_string()),
                patterns: vec!["testing".to_string()],
                project: None,
                source: ActionSource::Self_,
            },
            intent: None,
            observed_at: Utc::now(),
            quality: 0.8,
            observation_count: 1,
            imitated: false,
        };

        let intent = system.infer_intent(&action);
        assert_eq!(intent.goal, IntentGoal::PassTests);
        assert!(intent.confidence > 0.8);
    }

    #[test]
    fn test_language_detection() {
        let system = MirrorNeuronSystem::new();

        assert_eq!(system.detect_language("fn main() { let x = 5; }"), "rust");
        assert_eq!(system.detect_language("def hello(): pass"), "python");
    }

    #[test]
    fn test_pattern_extraction() {
        let system = MirrorNeuronSystem::new();

        let patterns = system.extract_patterns(
            "fn example() -> Result<(), Error> { let x = try_something()?; Ok(()) }",
        );

        assert!(patterns.contains(&"error_handling".to_string()));
        assert!(patterns.contains(&"error_propagation".to_string()));
    }

    #[test]
    fn test_simulation() {
        let mut system = MirrorNeuronSystem::new();

        let code = r#"
let x = 5;
let y = x + 10;
"#;

        let sim = system.simulate(code, HashMap::new());

        assert!(sim.success);
        assert!(sim.confidence > 0.5);
    }
}
