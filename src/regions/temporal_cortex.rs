//! Temporal Cortex - Semantic Association & Meaning Integration
//!
//! The temporal cortex binds linguistic input into semantic associations and
//! maintains a lightweight concept graph for meaning retrieval.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::regions::language_cortex::LanguageRepresentation;
use crate::signal::{BrainSignal, SignalType};

/// Configuration for semantic association.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalCortexConfig {
    pub max_nodes: usize,
    pub decay_rate: f64,
    pub activation_boost: f64,
    pub association_boost: f64,
    pub association_threshold: f64,
    pub insight_threshold: f64,
}

impl Default for TemporalCortexConfig {
    fn default() -> Self {
        Self {
            max_nodes: 512,
            decay_rate: 0.02,
            activation_boost: 0.2,
            association_boost: 0.15,
            association_threshold: 0.6,
            insight_threshold: 0.7,
        }
    }
}

/// A semantic concept node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptNode {
    pub term: String,
    pub activation: f64,
    pub last_updated: DateTime<Utc>,
    pub sentiment_bias: f64,
    pub associations: HashMap<String, f64>,
}

impl ConceptNode {
    fn new(term: &str) -> Self {
        Self {
            term: term.to_string(),
            activation: 0.3,
            last_updated: Utc::now(),
            sentiment_bias: 0.0,
            associations: HashMap::new(),
        }
    }

    fn boost(&mut self, boost: f64, sentiment: f64) {
        self.activation = (self.activation + boost).clamp(0.0, 1.0);
        self.sentiment_bias = (self.sentiment_bias * 0.9 + sentiment * 0.1).clamp(-1.0, 1.0);
        self.last_updated = Utc::now();
    }

    fn decay(&mut self, rate: f64) {
        self.activation = (self.activation * (1.0 - rate)).clamp(0.0, 1.0);
    }

    fn associate(&mut self, term: &str, boost: f64) -> f64 {
        let entry = self.associations.entry(term.to_string()).or_insert(0.0);
        *entry = (*entry + boost).clamp(0.0, 1.0);
        *entry
    }
}

/// A semantic insight derived from association strength.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticInsight {
    pub summary: String,
    pub confidence: f64,
    pub tags: Vec<String>,
    pub source_terms: Vec<String>,
    pub created_at: DateTime<Utc>,
}

impl SemanticInsight {
    pub fn to_signal(&self) -> BrainSignal {
        BrainSignal::new("temporal_cortex", SignalType::Memory, self.clone())
            .with_salience(self.confidence)
            .with_valence(0.1)
    }
}

/// The temporal cortex - semantic association hub.
pub struct TemporalCortex {
    config: TemporalCortexConfig,
    nodes: HashMap<String, ConceptNode>,
}

impl TemporalCortex {
    pub fn new() -> Self {
        Self::with_config(TemporalCortexConfig::default())
    }

    pub fn with_config(config: TemporalCortexConfig) -> Self {
        Self {
            config,
            nodes: HashMap::new(),
        }
    }

    /// Ingest a language representation and update semantic associations.
    pub fn ingest_language(
        &mut self,
        representation: &LanguageRepresentation,
    ) -> Option<SemanticInsight> {
        self.decay();

        let terms = if representation.salient_terms.is_empty() {
            representation.tokens.iter().take(6).cloned().collect()
        } else {
            representation.salient_terms.clone()
        };

        if terms.is_empty() {
            return None;
        }

        let activation_boost = self.config.activation_boost;
        let association_boost = self.config.association_boost;
        for term in &terms {
            self.get_or_create_node(term)
                .boost(activation_boost, representation.sentiment);
        }

        let mut strongest: Option<(String, String, f64)> = None;
        for window in terms.windows(2) {
            let left = &window[0];
            let right = &window[1];
            let weight = {
                let node = self.get_or_create_node(left);
                node.associate(right, association_boost)
            };
            if strongest
                .as_ref()
                .map(|(_, _, best)| weight > *best)
                .unwrap_or(true)
            {
                strongest = Some((left.clone(), right.clone(), weight));
            }
        }

        let (left, right, weight) = strongest?;
        let left_activation = self
            .nodes
            .get(&left)
            .map(|node| node.activation)
            .unwrap_or(0.0);
        let right_activation = self
            .nodes
            .get(&right)
            .map(|node| node.activation)
            .unwrap_or(0.0);
        let confidence =
            (weight * 0.5 + left_activation * 0.25 + right_activation * 0.25).clamp(0.0, 1.0);

        if weight < self.config.association_threshold || confidence < self.config.insight_threshold
        {
            return None;
        }

        Some(SemanticInsight {
            summary: format!("{} is strongly associated with {}", left, right),
            confidence,
            tags: vec![
                "semantic_association".to_string(),
                left.clone(),
                right.clone(),
            ],
            source_terms: vec![left, right],
            created_at: Utc::now(),
        })
    }

    pub fn stats(&self) -> TemporalCortexStats {
        let active_nodes = self
            .nodes
            .values()
            .filter(|node| node.activation > 0.2)
            .count();
        TemporalCortexStats {
            total_nodes: self.nodes.len(),
            active_nodes,
            avg_activation: if self.nodes.is_empty() {
                0.0
            } else {
                self.nodes.values().map(|node| node.activation).sum::<f64>()
                    / self.nodes.len() as f64
            },
        }
    }

    fn decay(&mut self) {
        for node in self.nodes.values_mut() {
            node.decay(self.config.decay_rate);
        }
    }

    fn get_or_create_node(&mut self, term: &str) -> &mut ConceptNode {
        if self.nodes.len() >= self.config.max_nodes
            && !self.nodes.contains_key(term)
            && let Some((evict, _)) = self
                .nodes
                .iter()
                .min_by(|a, b| a.1.activation.partial_cmp(&b.1.activation).unwrap())
        {
            let evict = evict.clone();
            self.nodes.remove(&evict);
        }
        self.nodes
            .entry(term.to_string())
            .or_insert_with(|| ConceptNode::new(term))
    }
}

impl Default for TemporalCortex {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary statistics for the temporal cortex.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalCortexStats {
    pub total_nodes: usize,
    pub active_nodes: usize,
    pub avg_activation: f64,
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regions::language_cortex::{LanguageCortex, LanguageCortexConfig};

    #[test]
    fn test_temporal_cortex_creates_association() {
        let mut cortex = TemporalCortex::new();
        let mut language = LanguageCortex::with_config(LanguageCortexConfig::default());
        let signal = BrainSignal::new("test", SignalType::Sensory, "memory pattern memory pattern");
        let rep = language.process_signal(&signal).expect("rep");

        let mut insight = None;
        for _ in 0..5 {
            insight = cortex.ingest_language(&rep);
            if insight.is_some() {
                break;
            }
        }

        assert!(insight.is_some());
    }

    #[test]
    fn test_temporal_cortex_stats() {
        let mut cortex = TemporalCortex::new();
        let rep = LanguageRepresentation {
            content: "semantic map".to_string(),
            tokens: vec!["semantic".to_string(), "map".to_string()],
            salient_terms: vec!["semantic".to_string(), "map".to_string()],
            key_phrases: vec!["semantic map".to_string()],
            intent: crate::regions::language_cortex::LanguageIntent::Statement,
            sentiment: 0.1,
            confidence: 0.6,
            salience: 0.5,
            arousal: 0.3,
            origin: crate::regions::language_cortex::LanguageOrigin::External,
            inner_speech_type: None,
            created_at: Utc::now(),
        };

        cortex.ingest_language(&rep);
        let stats = cortex.stats();
        assert!(stats.total_nodes >= 2);
    }
}
