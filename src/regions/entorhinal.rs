//! Entorhinal Cortex - Spatial/Contextual Gateway
//!
//! The entorhinal cortex serves as the primary interface between the
//! hippocampus and neocortex. It provides:
//!
//! - **Context encoding**: Transforms sensory input into contextual frames
//! - **Grid-cell-like spatial representation**: Encodes relational structure
//! - **Pattern separation**: Distinguishes similar but distinct contexts
//! - **Pattern completion**: Reconstructs full context from partial cues
//!
//! In the real brain, the entorhinal cortex contains grid cells, border cells,
//! and head-direction cells that provide a cognitive map. Here we simulate
//! the functional role: encoding the "where/when/what context" that gives
//! memories their episodic quality.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use crate::signal::{BrainSignal, SignalType};

/// A contextual frame representing the current situational context.
/// Analogous to the entorhinal cortex's role in providing contextual
/// coordinates for episodic memory formation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextFrame {
    pub id: Uuid,
    /// Semantic context tags (what domain are we in?)
    pub domain_tags: Vec<String>,
    /// Temporal context (when in the session/lifecycle)
    pub temporal_position: f64,
    /// Relational grid: how concepts relate in this context
    pub relational_grid: HashMap<String, Vec<String>>,
    /// Novelty of this context vs recent contexts (0-1)
    pub novelty: f64,
    /// Stability: how long this context has persisted (0-1)
    pub stability: f64,
    /// Coherence: how internally consistent the context is (0-1)
    pub coherence: f64,
    /// When this frame was created
    pub created_at: DateTime<Utc>,
    /// Number of signals processed in this context
    pub signal_count: u64,
}

impl ContextFrame {
    pub fn new(domain_tags: Vec<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            domain_tags,
            temporal_position: 0.0,
            relational_grid: HashMap::new(),
            novelty: 1.0,
            stability: 0.0,
            coherence: 0.5,
            created_at: Utc::now(),
            signal_count: 0,
        }
    }

    /// How similar is this frame to another?
    pub fn similarity(&self, other: &ContextFrame) -> f64 {
        if self.domain_tags.is_empty() && other.domain_tags.is_empty() {
            return 0.5;
        }

        let my_tags: std::collections::HashSet<&str> =
            self.domain_tags.iter().map(|s| s.as_str()).collect();
        let other_tags: std::collections::HashSet<&str> =
            other.domain_tags.iter().map(|s| s.as_str()).collect();

        let intersection = my_tags.intersection(&other_tags).count() as f64;
        let union = my_tags.union(&other_tags).count() as f64;

        if union == 0.0 {
            0.0
        } else {
            intersection / union
        }
    }
}

/// Result of pattern separation: determining if input matches existing context.
#[derive(Debug, Clone)]
pub struct SeparationResult {
    /// Does this input belong to the current context?
    pub same_context: bool,
    /// Similarity to current context (0-1)
    pub similarity: f64,
    /// Should we create a new context frame?
    pub context_shift: bool,
    /// Suggested domain tags for the input
    pub suggested_tags: Vec<String>,
}

/// Statistics for the entorhinal cortex.
#[derive(Debug, Clone, Default)]
pub struct EntorhinalStats {
    pub total_contexts_created: u64,
    pub context_shifts: u64,
    pub current_context_signals: u64,
    pub pattern_separations: u64,
    pub pattern_completions: u64,
    pub avg_context_stability: f64,
}

/// Configuration for the entorhinal cortex.
#[derive(Debug, Clone)]
pub struct EntorhinalConfig {
    /// Threshold for context shift (below this similarity = new context)
    pub shift_threshold: f64,
    /// Maximum number of recent contexts to retain
    pub max_context_history: usize,
    /// Rate at which stability increases per signal
    pub stability_growth_rate: f64,
}

impl Default for EntorhinalConfig {
    fn default() -> Self {
        Self {
            shift_threshold: 0.3,
            max_context_history: 16,
            stability_growth_rate: 0.05,
        }
    }
}

/// Entorhinal Cortex - the contextual gateway.
pub struct EntorhinalCortex {
    config: EntorhinalConfig,
    /// Current active context frame
    current_context: ContextFrame,
    /// Recent context history for pattern completion
    context_history: Vec<ContextFrame>,
    /// Running statistics
    stats: EntorhinalStats,
}

impl EntorhinalCortex {
    pub fn new() -> Self {
        Self::with_config(EntorhinalConfig::default())
    }

    pub fn with_config(config: EntorhinalConfig) -> Self {
        Self {
            config,
            current_context: ContextFrame::new(vec!["initial".to_string()]),
            context_history: Vec::new(),
            stats: EntorhinalStats::default(),
        }
    }

    /// Process an incoming signal and update the contextual frame.
    /// Returns the current context frame enriched with spatial/contextual data.
    pub fn process_signal(&mut self, signal: &BrainSignal) -> ContextFrame {
        let input_tags = self.extract_domain_tags(signal);
        let separation = self.pattern_separate(&input_tags);
        self.stats.pattern_separations += 1;

        if separation.context_shift {
            self.shift_context(input_tags);
        } else {
            // Update current context with new information
            self.current_context.signal_count += 1;
            self.current_context.stability =
                (self.current_context.stability + self.config.stability_growth_rate).min(1.0);

            // Add new tags
            for tag in &input_tags {
                if !self.current_context.domain_tags.contains(tag) {
                    self.current_context.domain_tags.push(tag.clone());
                }
            }

            // Update relational grid
            self.update_relational_grid(signal);

            // Novelty decreases as context stabilizes
            self.current_context.novelty = (self.current_context.novelty * 0.9).max(0.05);
        }

        // Update temporal position
        self.current_context.temporal_position = self.stats.total_contexts_created as f64
            + self.current_context.signal_count as f64 * 0.01;

        self.current_context.clone()
    }

    /// Attempt to complete a partial context from memory.
    /// Given partial cues, find the best matching historical context.
    pub fn pattern_complete(&mut self, cue_tags: &[String]) -> Option<ContextFrame> {
        self.stats.pattern_completions += 1;

        let cue_frame = ContextFrame::new(cue_tags.to_vec());
        let mut best_match: Option<&ContextFrame> = None;
        let mut best_similarity = 0.0;

        for ctx in &self.context_history {
            let sim = ctx.similarity(&cue_frame);
            if sim > best_similarity && sim > 0.2 {
                best_similarity = sim;
                best_match = Some(ctx);
            }
        }

        // Also check current context
        let current_sim = self.current_context.similarity(&cue_frame);
        if current_sim > best_similarity {
            return Some(self.current_context.clone());
        }

        best_match.cloned()
    }

    /// Generate a spatial/contextual signal for downstream regions.
    pub fn to_signal(&self) -> BrainSignal {
        let mut signal = BrainSignal::new(
            "entorhinal_cortex",
            SignalType::Spatial,
            self.current_context.clone(),
        )
        .with_salience(self.current_context.novelty * 0.5 + 0.3)
        .with_metadata("context_stability", self.current_context.stability)
        .with_metadata("context_coherence", self.current_context.coherence)
        .with_metadata("context_novelty", self.current_context.novelty);

        if let Ok(tags) = serde_json::to_value(&self.current_context.domain_tags) {
            signal.metadata.insert("domain_tags".to_string(), tags);
        }

        signal
    }

    /// Get the current context frame.
    pub fn current_context(&self) -> &ContextFrame {
        &self.current_context
    }

    /// Get statistics.
    pub fn stats(&self) -> EntorhinalStats {
        let avg_stability = if self.context_history.is_empty() {
            self.current_context.stability
        } else {
            let total: f64 = self
                .context_history
                .iter()
                .map(|c| c.stability)
                .sum::<f64>()
                + self.current_context.stability;
            total / (self.context_history.len() + 1) as f64
        };

        EntorhinalStats {
            avg_context_stability: avg_stability,
            ..self.stats.clone()
        }
    }

    fn extract_domain_tags(&self, signal: &BrainSignal) -> Vec<String> {
        let mut tags = Vec::new();

        // Extract from content
        let content = signal.content.as_str().unwrap_or("");
        let words: Vec<&str> = content.split_whitespace().collect();

        // Domain detection heuristics
        let domain_keywords: &[(&[&str], &str)] = &[
            (
                &[
                    "code", "function", "variable", "bug", "compile", "test", "error", "debug",
                ],
                "programming",
            ),
            (
                &["memory", "remember", "forget", "recall", "learn"],
                "memory",
            ),
            (
                &["feel", "emotion", "happy", "sad", "angry", "fear", "love"],
                "emotional",
            ),
            (
                &["plan", "goal", "strategy", "decide", "choose"],
                "planning",
            ),
            (
                &[
                    "see", "look", "bright", "color", "red", "blue", "green", "visual",
                ],
                "visual",
            ),
            (
                &["hear", "sound", "loud", "quiet", "music", "noise"],
                "auditory",
            ),
            (
                &["think", "reason", "logic", "understand", "concept"],
                "cognitive",
            ),
            (
                &["social", "person", "friend", "talk", "communicate"],
                "social",
            ),
            (&["move", "action", "walk", "run", "reach", "grab"], "motor"),
            (
                &["time", "when", "before", "after", "during", "schedule"],
                "temporal",
            ),
        ];

        for (keywords, domain) in domain_keywords {
            for word in &words {
                let lower = word.to_lowercase();
                if keywords.iter().any(|k| lower.contains(k)) {
                    if !tags.contains(&domain.to_string()) {
                        tags.push(domain.to_string());
                    }
                    break;
                }
            }
        }

        // Add signal type as context
        let type_tag = match signal.signal_type {
            SignalType::Sensory => "sensory_input",
            SignalType::Memory => "memory_retrieval",
            SignalType::Prediction => "prediction",
            SignalType::Error => "error_signal",
            SignalType::Emotion => "emotional_event",
            SignalType::Motor => "motor_planning",
            SignalType::Query => "inquiry",
            _ => "general",
        };
        tags.push(type_tag.to_string());

        if tags.is_empty() {
            tags.push("unclassified".to_string());
        }

        tags
    }

    fn pattern_separate(&self, input_tags: &[String]) -> SeparationResult {
        let input_frame = ContextFrame::new(input_tags.to_vec());
        let similarity = self.current_context.similarity(&input_frame);
        let context_shift =
            similarity < self.config.shift_threshold && self.current_context.signal_count > 2;

        SeparationResult {
            same_context: !context_shift,
            similarity,
            context_shift,
            suggested_tags: input_tags.to_vec(),
        }
    }

    fn shift_context(&mut self, new_tags: Vec<String>) {
        // Archive current context
        let old_context = std::mem::replace(&mut self.current_context, ContextFrame::new(new_tags));

        self.context_history.push(old_context);
        if self.context_history.len() > self.config.max_context_history {
            self.context_history.remove(0);
        }

        self.stats.total_contexts_created += 1;
        self.stats.context_shifts += 1;
    }

    fn update_relational_grid(&mut self, signal: &BrainSignal) {
        let content = signal.content.as_str().unwrap_or("");
        let words: Vec<String> = content
            .split_whitespace()
            .filter_map(|w| {
                let trimmed = w.trim_matches(|c: char| !c.is_alphanumeric());
                if trimmed.len() > 2 {
                    Some(trimmed.to_lowercase())
                } else {
                    None
                }
            })
            .take(10)
            .collect();

        // Build co-occurrence relations
        for (i, word) in words.iter().enumerate() {
            let neighbors: Vec<String> = words
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .take(4)
                .map(|(_, w)| w.clone())
                .collect();

            let entry = self
                .current_context
                .relational_grid
                .entry(word.clone())
                .or_default();

            for neighbor in neighbors {
                if !entry.contains(&neighbor) {
                    entry.push(neighbor);
                }
            }

            // Limit grid size per entry
            if entry.len() > 8 {
                entry.truncate(8);
            }
        }

        // Update coherence based on grid interconnectedness
        let grid = &self.current_context.relational_grid;
        if grid.len() > 1 {
            let total_connections: usize = grid.values().map(|v| v.len()).sum();
            let max_possible = grid.len() * (grid.len() - 1);
            self.current_context.coherence = if max_possible > 0 {
                (total_connections as f64 / max_possible as f64).min(1.0)
            } else {
                0.5
            };
        }
    }
}

impl Default for EntorhinalCortex {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_context_from_signal() {
        let mut ec = EntorhinalCortex::new();
        let signal = BrainSignal::new("test", SignalType::Sensory, "I see a bright red circle");
        let context = ec.process_signal(&signal);

        assert!(!context.domain_tags.is_empty());
        assert!(context.domain_tags.contains(&"visual".to_string()));
    }

    #[test]
    fn detects_context_shift() {
        let mut ec = EntorhinalCortex::new();

        // Establish a visual context
        for _ in 0..4 {
            let signal = BrainSignal::new("test", SignalType::Sensory, "bright red visual color");
            ec.process_signal(&signal);
        }

        let initial_shifts = ec.stats().context_shifts;

        // Shift to a completely different domain
        let signal = BrainSignal::new(
            "test",
            SignalType::Sensory,
            "compile the code function debug",
        );
        ec.process_signal(&signal);

        assert!(
            ec.stats().context_shifts > initial_shifts,
            "Expected context shift when domain changes dramatically"
        );
    }

    #[test]
    fn stability_increases_with_consistent_input() {
        let mut ec = EntorhinalCortex::new();
        let signal = BrainSignal::new("test", SignalType::Sensory, "think about reasoning logic");

        ec.process_signal(&signal);
        let stability_1 = ec.current_context().stability;

        ec.process_signal(&signal);
        let stability_2 = ec.current_context().stability;

        assert!(stability_2 > stability_1);
    }

    #[test]
    fn pattern_completion_finds_matching_context() {
        let mut ec = EntorhinalCortex::new();

        // Create and archive a programming context
        for _ in 0..4 {
            let signal =
                BrainSignal::new("test", SignalType::Sensory, "code function variable debug");
            ec.process_signal(&signal);
        }

        // Shift to a different context
        let signal = BrainSignal::new("test", SignalType::Sensory, "feel happy emotion love");
        ec.process_signal(&signal);

        // Try to complete with programming cues
        let result = ec.pattern_complete(&["programming".to_string()]);
        assert!(result.is_some());
    }

    #[test]
    fn generates_spatial_signal() {
        let mut ec = EntorhinalCortex::new();
        let signal = BrainSignal::new("test", SignalType::Sensory, "test input");
        ec.process_signal(&signal);

        let spatial_signal = ec.to_signal();
        assert_eq!(spatial_signal.signal_type, SignalType::Spatial);
        assert!(spatial_signal.metadata.contains_key("context_stability"));
    }
}
