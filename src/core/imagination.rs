//! Imagination Engine - Creative Memory Processing
//!
//! Generates novel combinations, simulations, and counterfactuals from memories.
//! Used for creative problem-solving, dream consolidation, and hypothesis generation.
//!
//! # Imagination Types
//!
//! - **Recombination**: Blend elements from different memories into new scenarios
//! - **Simulation**: "What if" projections based on current context
//! - **Counterfactual**: Explore alternate histories ("what if X happened instead")
//! - **Dream**: Free association chains for sleep consolidation
//! - **Synthesis**: Generate insights from pattern recognition
//! - **Hypothesis**: Create testable predictions from observations
//!
//! # Architecture
//!
//! ```text
//! Memories → [Selection] → [LLM Generation] → Imagining
//!                ↓              ↓                  ↓
//!           Related +      Creative           Store with
//!           Distant       Recombination       provenance
//! ```
//!
//! Imaginations are stored separately from real memories but can:
//! - Surface during creative problem-solving
//! - Influence dream consolidation  
//! - Generate hypotheses to test
//! - Be marked useful for reinforcement

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use crate::core::llm::{ChatMessage, LlmBackend, LlmRequestConfig};

/// Types of imagination
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImaginationType {
    /// Blend of existing memories into novel scenario
    Recombination,
    /// "What if" projection into future
    Simulation,
    /// Alternate history exploration
    Counterfactual,
    /// Free association (for dreams)
    Dream,
    /// Pattern-derived insight
    Synthesis,
    /// Testable prediction
    Hypothesis,
}

impl std::fmt::Display for ImaginationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Recombination => write!(f, "recombination"),
            Self::Simulation => write!(f, "simulation"),
            Self::Counterfactual => write!(f, "counterfactual"),
            Self::Dream => write!(f, "dream"),
            Self::Synthesis => write!(f, "synthesis"),
            Self::Hypothesis => write!(f, "hypothesis"),
        }
    }
}

/// Unique identifier for an imagining
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ImaginingId(pub Uuid);

impl ImaginingId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for ImaginingId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for ImaginingId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A single imagined scenario/idea
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Imagining {
    /// Unique identifier
    pub id: ImaginingId,
    /// Type of imagination
    pub imagination_type: ImaginationType,
    /// The imagined content
    pub content: String,
    /// IDs of source memories used
    pub source_memory_ids: Vec<String>,
    /// How plausible this imagining is (0.0-1.0)
    pub confidence: f64,
    /// How novel/creative (0.0-1.0)
    pub novelty: f64,
    /// Potential usefulness (0.0-1.0)
    pub utility: f64,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
    /// When this was created
    pub created_at: DateTime<Utc>,
    /// Optional embedding for semantic search
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
    /// Times this imagining was used
    pub used_count: u32,
    /// Times marked as useful
    pub useful_count: u32,
    /// Whether this has been archived
    pub archived: bool,
}

impl Imagining {
    /// Create a new imagining
    pub fn new(
        imagination_type: ImaginationType,
        content: String,
        source_memory_ids: Vec<String>,
    ) -> Self {
        Self {
            id: ImaginingId::new(),
            imagination_type,
            content,
            source_memory_ids,
            confidence: 0.5,
            novelty: 0.5,
            utility: 0.5,
            metadata: HashMap::new(),
            created_at: Utc::now(),
            embedding: None,
            used_count: 0,
            useful_count: 0,
            archived: false,
        }
    }

    /// Set confidence score
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Set novelty score
    pub fn with_novelty(mut self, novelty: f64) -> Self {
        self.novelty = novelty.clamp(0.0, 1.0);
        self
    }

    /// Set utility score
    pub fn with_utility(mut self, utility: f64) -> Self {
        self.utility = utility.clamp(0.0, 1.0);
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: &str, value: serde_json::Value) -> Self {
        self.metadata.insert(key.to_string(), value);
        self
    }

    /// Calculate overall quality score
    pub fn quality_score(&self) -> f64 {
        // Weighted combination - utility matters most, then confidence, then novelty
        (self.utility * 0.5) + (self.confidence * 0.3) + (self.novelty * 0.2)
    }

    /// Mark as used
    pub fn mark_used(&mut self) {
        self.used_count += 1;
    }

    /// Mark as useful (reinforcement signal)
    pub fn mark_useful(&mut self) {
        self.used_count += 1;
        self.useful_count += 1;
    }

    /// Archive this imagining
    pub fn archive(&mut self) {
        self.archived = true;
    }
}

/// Memory snippet used as source material
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySource {
    pub id: String,
    pub content: String,
    pub similarity: f64,
    pub valence: f64,
}

/// Configuration for the imagination engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImaginationConfig {
    /// LLM temperature for creative generation (higher = more creative)
    pub temperature: f64,
    /// Maximum tokens for generation
    pub max_tokens: u32,
    /// Number of related memories to fetch
    pub related_memory_count: usize,
    /// Number of distant memories to include (for novelty)
    pub distant_memory_count: usize,
    /// Default number of recombinations to generate
    pub default_recombination_count: usize,
    /// Default number of dream associations
    pub default_dream_associations: usize,
    /// Whether to auto-embed imaginations
    pub auto_embed: bool,
    /// Model to use for imagination
    pub model: String,
}

impl Default for ImaginationConfig {
    fn default() -> Self {
        Self {
            temperature: 0.9, // High for creativity
            max_tokens: 500,
            related_memory_count: 5,
            distant_memory_count: 2,
            default_recombination_count: 3,
            default_dream_associations: 5,
            auto_embed: true,
            model: "gpt-4o-mini".to_string(),
        }
    }
}

/// Statistics about imagination usage
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ImaginationStats {
    pub total_imaginations: usize,
    pub by_type: HashMap<ImaginationType, TypeStats>,
    pub total_useful: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TypeStats {
    pub count: usize,
    pub avg_confidence: f64,
    pub avg_novelty: f64,
    pub avg_utility: f64,
    pub useful_count: usize,
}

/// Result of a recombination operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecombinationResult {
    pub imaginings: Vec<Imagining>,
    pub source_memories: Vec<MemorySource>,
}

/// Result of a simulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationResult {
    pub imagining: Imagining,
    pub key_factors: Vec<String>,
}

/// Result of a dream sequence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamResult {
    pub sequence: Vec<Imagining>,
    pub seed: Option<String>,
}

/// The imagination engine
pub struct ImaginationEngine<L: LlmBackend> {
    config: ImaginationConfig,
    llm: L,
    /// In-memory storage (would typically be backed by database)
    imaginations: Vec<Imagining>,
}

impl<L: LlmBackend> ImaginationEngine<L> {
    /// Create a new imagination engine
    pub fn new(config: ImaginationConfig, llm: L) -> Self {
        Self {
            config,
            llm,
            imaginations: Vec::new(),
        }
    }

    /// Get configuration
    pub fn config(&self) -> &ImaginationConfig {
        &self.config
    }

    /// Store an imagining
    pub fn store(&mut self, imagining: Imagining) {
        self.imaginations.push(imagining);
    }

    /// Get recent imaginations
    pub fn recent(&self, limit: usize, filter_type: Option<ImaginationType>) -> Vec<&Imagining> {
        let mut results: Vec<_> = self
            .imaginations
            .iter()
            .filter(|i| !i.archived)
            .filter(|i| filter_type.map_or(true, |t| i.imagination_type == t))
            .collect();

        results.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        results.truncate(limit);
        results
    }

    /// Get statistics
    pub fn stats(&self) -> ImaginationStats {
        let mut stats = ImaginationStats::default();
        
        for imagining in &self.imaginations {
            if imagining.archived {
                continue;
            }
            
            stats.total_imaginations += 1;
            stats.total_useful += imagining.useful_count as usize;
            
            let type_stats = stats
                .by_type
                .entry(imagining.imagination_type)
                .or_default();
            
            type_stats.count += 1;
            type_stats.useful_count += imagining.useful_count as usize;
            
            // Running average
            let n = type_stats.count as f64;
            type_stats.avg_confidence = 
                type_stats.avg_confidence * (n - 1.0) / n + imagining.confidence / n;
            type_stats.avg_novelty = 
                type_stats.avg_novelty * (n - 1.0) / n + imagining.novelty / n;
            type_stats.avg_utility = 
                type_stats.avg_utility * (n - 1.0) / n + imagining.utility / n;
        }
        
        stats
    }

    /// Generate recombinations from topics
    pub async fn recombine(
        &mut self,
        topics: &[String],
        memories: Vec<MemorySource>,
        count: Option<usize>,
    ) -> Result<RecombinationResult, ImaginationError> {
        let count = count.unwrap_or(self.config.default_recombination_count);
        
        if memories.is_empty() {
            return Err(ImaginationError::InsufficientMemories);
        }

        let memory_context: String = memories
            .iter()
            .map(|m| format!("- {}", &m.content[..m.content.len().min(200)]))
            .collect::<Vec<_>>()
            .join("\n");

        let system = r#"You are an imagination engine. Your job is to creatively recombine 
memories and ideas into novel scenarios. Be creative but grounded - the scenarios should 
be plausible given the source material. Output JSON with format:
{"ideas": [{"content": "...", "confidence": 0.0-1.0, "novelty": 0.0-1.0}]}"#;

        let prompt = format!(
            "Given these memories about {}:\n\n{}\n\nGenerate {} creative recombinations - \
             novel scenarios that blend elements from multiple memories. Each should be a \
             complete thought, not just fragments. Focus on interesting connections.",
            topics.join(", "),
            memory_context,
            count
        );

        let response = self.call_llm(system, &prompt).await?;
        let ideas = self.parse_ideas_response(&response)?;

        let source_ids: Vec<String> = memories.iter().map(|m| m.id.clone()).collect();
        
        let mut imaginings = Vec::new();
        for idea in ideas.into_iter().take(count) {
            let imagining = Imagining::new(
                ImaginationType::Recombination,
                idea.content,
                source_ids.clone(),
            )
            .with_confidence(idea.confidence)
            .with_novelty(idea.novelty)
            .with_metadata("topics", serde_json::json!(topics));

            self.store(imagining.clone());
            imaginings.push(imagining);
        }

        Ok(RecombinationResult {
            imaginings,
            source_memories: memories,
        })
    }

    /// Run a "what if" simulation
    pub async fn simulate(
        &mut self,
        scenario: &str,
        memories: Vec<MemorySource>,
        depth: u32,
    ) -> Result<SimulationResult, ImaginationError> {
        let memory_context: String = memories
            .iter()
            .map(|m| format!("- {}", &m.content[..m.content.len().min(200)]))
            .collect::<Vec<_>>()
            .join("\n");

        let system = r#"You are a simulation engine. Given a scenario, project what might happen.
Use available context but extrapolate plausibly. Consider both positive and negative outcomes.
Be specific and concrete, not vague. Output JSON:
{"simulation": "detailed projection", "confidence": 0.0-1.0, "key_factors": ["factor1", "factor2"]}"#;

        let prompt = format!(
            "Scenario to simulate: {}\n\nRelevant context from memory:\n{}\n\n\
             Project forward {} steps. What might happen? What are the key factors?",
            scenario, memory_context, depth
        );

        let response = self.call_llm(system, &prompt).await?;
        let (content, confidence, key_factors) = self.parse_simulation_response(&response)?;

        let source_ids: Vec<String> = memories.iter().map(|m| m.id.clone()).collect();
        
        let imagining = Imagining::new(
            ImaginationType::Simulation,
            content,
            source_ids,
        )
        .with_confidence(confidence)
        .with_novelty(0.6)
        .with_utility(0.7)
        .with_metadata("scenario", serde_json::json!(scenario))
        .with_metadata("depth", serde_json::json!(depth))
        .with_metadata("key_factors", serde_json::json!(&key_factors));

        self.store(imagining.clone());

        Ok(SimulationResult {
            imagining,
            key_factors,
        })
    }

    /// Explore a counterfactual
    pub async fn counterfactual(
        &mut self,
        event: &str,
        alternate: &str,
        memories: Vec<MemorySource>,
    ) -> Result<Imagining, ImaginationError> {
        let memory_context: String = memories
            .iter()
            .map(|m| format!("- {}", &m.content[..m.content.len().min(200)]))
            .collect::<Vec<_>>()
            .join("\n");

        let system = r#"You are a counterfactual reasoning engine. Given an event and an alternate,
explore what would be different. Be concrete and trace specific consequences.
Output JSON: {"counterfactual": "narrative", "divergences": ["point1", "point2"], "confidence": 0.0-1.0}"#;

        let prompt = format!(
            "Actual event: {}\nAlternate scenario: {}\n\nContext:\n{}\n\n\
             Trace how things would be different if the alternate had happened.",
            event, alternate, memory_context
        );

        let response = self.call_llm(system, &prompt).await?;
        let (content, confidence, divergences) = self.parse_counterfactual_response(&response)?;

        let source_ids: Vec<String> = memories.iter().map(|m| m.id.clone()).collect();
        
        let imagining = Imagining::new(
            ImaginationType::Counterfactual,
            content,
            source_ids,
        )
        .with_confidence(confidence)
        .with_novelty(0.8)
        .with_utility(0.4)
        .with_metadata("event", serde_json::json!(event))
        .with_metadata("alternate", serde_json::json!(alternate))
        .with_metadata("divergences", serde_json::json!(divergences));

        self.store(imagining.clone());

        Ok(imagining)
    }

    /// Free association dream sequence
    pub async fn dream(
        &mut self,
        seed: Option<MemorySource>,
        associations: Option<usize>,
    ) -> Result<DreamResult, ImaginationError> {
        let associations = associations.unwrap_or(self.config.default_dream_associations);
        
        let mut current_content = seed
            .as_ref()
            .map(|s| s.content.clone())
            .unwrap_or_else(|| "consciousness emerging from void".to_string());
        
        let seed_str = seed.as_ref().map(|s| s.content.clone());
        let mut sequence = Vec::new();

        let system = r#"You are a dream engine doing free association. Given a memory,
generate a creative associative leap - something the memory reminds you of, suggests,
or connects to in unexpected ways. Be surreal but meaningful.
Output JSON: {"association": "the connection", "leap_type": "metaphor|memory|emotion|pattern"}"#;

        for i in 0..associations {
            let prompt = format!(
                "Current focus: {}\n\nMake an associative leap. What does this remind you of? \
                 What patterns emerge? Be creative and unexpected.",
                &current_content[..current_content.len().min(300)]
            );

            let response = self.call_llm(system, &prompt).await?;
            let (content, leap_type) = self.parse_dream_response(&response)?;

            let imagining = Imagining::new(
                ImaginationType::Dream,
                content.clone(),
                seed.as_ref().map(|s| vec![s.id.clone()]).unwrap_or_default(),
            )
            .with_confidence(0.3) // Dreams are uncertain
            .with_novelty(0.9)    // But highly novel
            .with_utility(0.3)
            .with_metadata("leap_type", serde_json::json!(leap_type))
            .with_metadata("chain_position", serde_json::json!(i + 1));

            self.store(imagining.clone());
            sequence.push(imagining);
            
            current_content = content;
        }

        Ok(DreamResult {
            sequence,
            seed: seed_str,
        })
    }

    /// Synthesize insights from patterns
    pub async fn synthesize(
        &mut self,
        domain: &str,
        memories: Vec<MemorySource>,
    ) -> Result<Imagining, ImaginationError> {
        if memories.is_empty() {
            return Err(ImaginationError::InsufficientMemories);
        }

        let memory_context: String = memories
            .iter()
            .map(|m| format!("- {}", &m.content[..m.content.len().min(200)]))
            .collect::<Vec<_>>()
            .join("\n");

        let system = r#"You are a synthesis engine. Look for patterns across memories and 
generate a novel insight or idea that wasn't explicitly stated anywhere.
Output JSON: {"synthesis": "the insight", "patterns": ["pattern1", "pattern2"], 
"confidence": 0.0-1.0, "utility": 0.0-1.0}"#;

        let prompt = format!(
            "Domain: {}\n\nRecent memories and observations:\n{}\n\n\
             What patterns do you see? What insight or idea emerges from combining these?",
            domain, memory_context
        );

        let response = self.call_llm(system, &prompt).await?;
        let (content, confidence, utility, patterns) = self.parse_synthesis_response(&response)?;

        let source_ids: Vec<String> = memories.iter().map(|m| m.id.clone()).collect();
        
        let imagining = Imagining::new(
            ImaginationType::Synthesis,
            content,
            source_ids,
        )
        .with_confidence(confidence)
        .with_novelty(0.8)
        .with_utility(utility)
        .with_metadata("domain", serde_json::json!(domain))
        .with_metadata("patterns", serde_json::json!(patterns));

        self.store(imagining.clone());

        Ok(imagining)
    }

    /// Generate a testable hypothesis
    pub async fn hypothesize(
        &mut self,
        observation: &str,
        memories: Vec<MemorySource>,
    ) -> Result<Imagining, ImaginationError> {
        let memory_context: String = memories
            .iter()
            .map(|m| format!("- {}", &m.content[..m.content.len().min(200)]))
            .collect::<Vec<_>>()
            .join("\n");

        let system = r#"You are a hypothesis engine. Given an observation, generate a testable
hypothesis. The hypothesis should be specific, falsifiable, and actionable.
Output JSON: {"hypothesis": "if X then Y", "test": "how to test it", 
"confidence": 0.0-1.0, "implications": ["if true", "if false"]}"#;

        let prompt = format!(
            "Observation: {}\n\nRelated context:\n{}\n\n\
             Generate a testable hypothesis. What might be true? How could we test it?",
            observation, memory_context
        );

        let response = self.call_llm(system, &prompt).await?;
        let (content, confidence, test, implications) = self.parse_hypothesis_response(&response)?;

        let source_ids: Vec<String> = memories.iter().map(|m| m.id.clone()).collect();
        
        let imagining = Imagining::new(
            ImaginationType::Hypothesis,
            content,
            source_ids,
        )
        .with_confidence(confidence)
        .with_novelty(0.6)
        .with_utility(0.8) // Hypotheses are highly actionable
        .with_metadata("observation", serde_json::json!(observation))
        .with_metadata("test", serde_json::json!(test))
        .with_metadata("implications", serde_json::json!(implications));

        self.store(imagining.clone());

        Ok(imagining)
    }

    // Helper: call LLM
    async fn call_llm(&self, system: &str, prompt: &str) -> Result<String, ImaginationError> {
        let config = LlmRequestConfig {
            temperature: self.config.temperature,
            max_tokens: self.config.max_tokens,
            ..Default::default()
        };

        let messages = vec![
            ChatMessage::system(system.to_string()),
            ChatMessage::user(prompt.to_string()),
        ];

        let response = self
            .llm
            .chat(&messages, &config)
            .await
            .map_err(|e| ImaginationError::LlmError(e.to_string()))?;

        Ok(response.content)
    }

    // Parse helpers
    fn parse_ideas_response(&self, response: &str) -> Result<Vec<IdeaResponse>, ImaginationError> {
        let json_str = extract_json(response);
        let parsed: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| ImaginationError::ParseError(e.to_string()))?;
        
        let ideas = parsed.get("ideas")
            .and_then(|v| v.as_array())
            .ok_or_else(|| ImaginationError::ParseError("Missing ideas array".to_string()))?;
        
        let mut results = Vec::new();
        for idea in ideas {
            results.push(IdeaResponse {
                content: idea.get("content")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                confidence: idea.get("confidence")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.5),
                novelty: idea.get("novelty")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.7),
            });
        }
        
        Ok(results)
    }

    fn parse_simulation_response(&self, response: &str) -> Result<(String, f64, Vec<String>), ImaginationError> {
        let json_str = extract_json(response);
        let parsed: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| ImaginationError::ParseError(e.to_string()))?;
        
        let content = parsed.get("simulation")
            .and_then(|v| v.as_str())
            .unwrap_or(response)
            .to_string();
        
        let confidence = parsed.get("confidence")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5);
        
        let key_factors = parsed.get("key_factors")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect())
            .unwrap_or_default();
        
        Ok((content, confidence, key_factors))
    }

    fn parse_counterfactual_response(&self, response: &str) -> Result<(String, f64, Vec<String>), ImaginationError> {
        let json_str = extract_json(response);
        let parsed: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| ImaginationError::ParseError(e.to_string()))?;
        
        let content = parsed.get("counterfactual")
            .and_then(|v| v.as_str())
            .unwrap_or(response)
            .to_string();
        
        let confidence = parsed.get("confidence")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.4);
        
        let divergences = parsed.get("divergences")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect())
            .unwrap_or_default();
        
        Ok((content, confidence, divergences))
    }

    fn parse_dream_response(&self, response: &str) -> Result<(String, String), ImaginationError> {
        let json_str = extract_json(response);
        let parsed: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| ImaginationError::ParseError(e.to_string()))?;
        
        let content = parsed.get("association")
            .and_then(|v| v.as_str())
            .unwrap_or(response)
            .to_string();
        
        let leap_type = parsed.get("leap_type")
            .and_then(|v| v.as_str())
            .unwrap_or("pattern")
            .to_string();
        
        Ok((content, leap_type))
    }

    fn parse_synthesis_response(&self, response: &str) -> Result<(String, f64, f64, Vec<String>), ImaginationError> {
        let json_str = extract_json(response);
        let parsed: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| ImaginationError::ParseError(e.to_string()))?;
        
        let content = parsed.get("synthesis")
            .and_then(|v| v.as_str())
            .unwrap_or(response)
            .to_string();
        
        let confidence = parsed.get("confidence")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.6);
        
        let utility = parsed.get("utility")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.6);
        
        let patterns = parsed.get("patterns")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect())
            .unwrap_or_default();
        
        Ok((content, confidence, utility, patterns))
    }

    fn parse_hypothesis_response(&self, response: &str) -> Result<(String, f64, String, Vec<String>), ImaginationError> {
        let json_str = extract_json(response);
        let parsed: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| ImaginationError::ParseError(e.to_string()))?;
        
        let content = parsed.get("hypothesis")
            .and_then(|v| v.as_str())
            .unwrap_or(response)
            .to_string();
        
        let confidence = parsed.get("confidence")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5);
        
        let test = parsed.get("test")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        
        let implications = parsed.get("implications")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect())
            .unwrap_or_default();
        
        Ok((content, confidence, test, implications))
    }
}

/// Extract JSON from response (handles markdown code blocks)
fn extract_json(response: &str) -> String {
    if response.contains("```json") {
        response
            .split("```json")
            .nth(1)
            .and_then(|s| s.split("```").next())
            .unwrap_or(response)
            .trim()
            .to_string()
    } else if response.contains("```") {
        response
            .split("```")
            .nth(1)
            .unwrap_or(response)
            .trim()
            .to_string()
    } else {
        response.trim().to_string()
    }
}

#[derive(Debug)]
struct IdeaResponse {
    content: String,
    confidence: f64,
    novelty: f64,
}

/// Errors from the imagination engine
#[derive(Debug, thiserror::Error)]
pub enum ImaginationError {
    #[error("LLM error: {0}")]
    LlmError(String),
    
    #[error("Parse error: {0}")]
    ParseError(String),
    
    #[error("Insufficient source memories")]
    InsufficientMemories,
    
    #[error("Storage error: {0}")]
    StorageError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_imagining_creation() {
        let imagining = Imagining::new(
            ImaginationType::Dream,
            "A floating library where books read themselves".to_string(),
            vec!["mem_1".to_string()],
        )
        .with_confidence(0.3)
        .with_novelty(0.95)
        .with_utility(0.2);

        assert_eq!(imagining.imagination_type, ImaginationType::Dream);
        assert_eq!(imagining.confidence, 0.3);
        assert_eq!(imagining.novelty, 0.95);
        assert!(!imagining.archived);
    }

    #[test]
    fn test_quality_score() {
        let imagining = Imagining::new(
            ImaginationType::Hypothesis,
            "If we cache more aggressively, latency drops".to_string(),
            vec![],
        )
        .with_confidence(0.8)
        .with_novelty(0.6)
        .with_utility(0.9);

        // utility * 0.5 + confidence * 0.3 + novelty * 0.2
        // 0.9 * 0.5 + 0.8 * 0.3 + 0.6 * 0.2 = 0.45 + 0.24 + 0.12 = 0.81
        let expected = 0.81;
        assert!((imagining.quality_score() - expected).abs() < 0.01);
    }

    #[test]
    fn test_extract_json() {
        let with_markdown = "Here's the result:\n```json\n{\"ideas\": []}\n```\nDone!";
        assert_eq!(extract_json(with_markdown), "{\"ideas\": []}");

        let plain = "{\"ideas\": []}";
        assert_eq!(extract_json(plain), "{\"ideas\": []}");
    }

    #[test]
    fn test_imagination_type_display() {
        assert_eq!(format!("{}", ImaginationType::Dream), "dream");
        assert_eq!(format!("{}", ImaginationType::Recombination), "recombination");
    }
}
