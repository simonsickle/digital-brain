//! Nervous System - Inter-Module Signal Routing
//!
//! The nervous system connects brain regions and routes signals between them.
//! It provides:
//!
//! - **Pathways**: Defined connections between regions
//! - **Signal routing**: Automatic propagation of signals through pathways
//! - **Modulation**: Neuromodulators can strengthen/weaken pathways
//! - **Tracing**: Observe signal flow for debugging/understanding
//!
//! # Architecture
//!
//! ```text
//!                    ┌─────────────────────────────────────┐
//!                    │         NERVOUS SYSTEM              │
//!                    │   (Signal Routing & Pathways)       │
//!                    └─────────────────────────────────────┘
//!                                    │
//!        ┌───────────────────────────┼───────────────────────────┐
//!        │                           │                           │
//!        ▼                           ▼                           ▼
//!   ┌─────────┐               ┌─────────────┐             ┌──────────┐
//!   │THALAMUS │──────────────▶│  AMYGDALA   │────────────▶│   DMN    │
//!   │(Gateway)│               │ (Emotional) │             │ (Self)   │
//!   └────┬────┘               └──────┬──────┘             └────┬─────┘
//!        │                           │                          │
//!        │    ┌──────────────────────┘                          │
//!        │    │                                                 │
//!        ▼    ▼                                                 ▼
//!   ┌──────────────┐         ┌──────────────┐          ┌────────────┐
//!   │ HIPPOCAMPUS  │◀───────▶│  PREFRONTAL  │◀────────▶│ WORKSPACE  │
//!   │  (Memory)    │         │(Working Mem) │          │(Conscious) │
//!   └──────────────┘         └──────────────┘          └────────────┘
//! ```

use crate::signal::{BrainSignal, SignalType};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

/// Brain regions that can be connected via pathways.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BrainRegion {
    Thalamus,
    Amygdala,
    Hippocampus,
    Prefrontal,
    DMN,
    Workspace,
    PredictionEngine,
    BasalGanglia,
    ACC,
    Cerebellum,
    STN,      // Subthalamic Nucleus - response inhibition
    External, // Input/output to outside world
}

impl BrainRegion {
    pub fn name(&self) -> &'static str {
        match self {
            BrainRegion::Thalamus => "Thalamus",
            BrainRegion::Amygdala => "Amygdala",
            BrainRegion::Hippocampus => "Hippocampus",
            BrainRegion::Prefrontal => "Prefrontal",
            BrainRegion::DMN => "DMN",
            BrainRegion::Workspace => "Workspace",
            BrainRegion::PredictionEngine => "PredictionEngine",
            BrainRegion::BasalGanglia => "BasalGanglia",
            BrainRegion::ACC => "ACC",
            BrainRegion::Cerebellum => "Cerebellum",
            BrainRegion::STN => "STN",
            BrainRegion::External => "External",
        }
    }
}

/// A pathway connecting two brain regions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pathway {
    pub from: BrainRegion,
    pub to: BrainRegion,
    /// Base strength of the connection (0.0 to 1.0)
    pub strength: f64,
    /// Current modulated strength
    pub effective_strength: f64,
    /// Signal types this pathway carries
    pub signal_types: Vec<SignalType>,
    /// Number of signals transmitted
    pub transmission_count: u64,
    /// Is this pathway bidirectional?
    pub bidirectional: bool,
}

impl Pathway {
    pub fn new(from: BrainRegion, to: BrainRegion, strength: f64) -> Self {
        Self {
            from,
            to,
            strength: strength.clamp(0.0, 1.0),
            effective_strength: strength.clamp(0.0, 1.0),
            signal_types: vec![
                SignalType::Sensory,
                SignalType::Memory,
                SignalType::Emotion,
                SignalType::Attention,
                SignalType::Broadcast,
            ],
            transmission_count: 0,
            bidirectional: false,
        }
    }

    pub fn bidirectional(mut self) -> Self {
        self.bidirectional = true;
        self
    }

    pub fn with_signal_types(mut self, types: Vec<SignalType>) -> Self {
        self.signal_types = types;
        self
    }

    /// Apply modulation to pathway strength
    pub fn modulate(&mut self, factor: f64) {
        self.effective_strength = (self.strength * factor).clamp(0.0, 1.0);
    }

    /// Reset modulation
    pub fn reset_modulation(&mut self) {
        self.effective_strength = self.strength;
    }

    /// Check if this pathway can carry a signal type
    pub fn can_carry(&self, signal_type: &SignalType) -> bool {
        self.signal_types.contains(signal_type)
    }
}

/// A record of a signal transmission through the nervous system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalTrace {
    pub signal_id: String,
    pub path: Vec<BrainRegion>,
    pub timestamps: Vec<DateTime<Utc>>,
    pub signal_type: SignalType,
    pub initial_salience: f64,
    pub final_salience: f64,
    pub reached_consciousness: bool,
}

/// Configuration for the nervous system.
#[derive(Debug, Clone)]
pub struct NervousSystemConfig {
    /// Maximum number of traces to keep in history
    pub max_trace_history: usize,
    /// Enable signal tracing (has performance cost)
    pub tracing_enabled: bool,
    /// Minimum pathway strength to transmit
    pub transmission_threshold: f64,
}

impl Default for NervousSystemConfig {
    fn default() -> Self {
        Self {
            max_trace_history: 100,
            tracing_enabled: true,
            transmission_threshold: 0.1,
        }
    }
}

/// The nervous system manages inter-module signal routing.
pub struct NervousSystem {
    /// All defined pathways
    pathways: HashMap<(BrainRegion, BrainRegion), Pathway>,
    /// Signal traces for debugging (future use for signal path visualization)
    #[allow(dead_code)]
    traces: VecDeque<SignalTrace>,
    /// Signals queued for each region
    signal_queues: HashMap<BrainRegion, VecDeque<BrainSignal>>,
    /// Configuration
    config: NervousSystemConfig,
    /// Total signals routed
    total_routed: u64,
}

impl NervousSystem {
    /// Create a new nervous system with default pathways.
    pub fn new() -> Self {
        let mut ns = Self {
            pathways: HashMap::new(),
            traces: VecDeque::new(),
            signal_queues: HashMap::new(),
            config: NervousSystemConfig::default(),
            total_routed: 0,
        };
        ns.initialize_default_pathways();
        ns.initialize_queues();
        ns
    }

    /// Create with custom configuration.
    pub fn with_config(config: NervousSystemConfig) -> Self {
        let mut ns = Self {
            pathways: HashMap::new(),
            traces: VecDeque::new(),
            signal_queues: HashMap::new(),
            config,
            total_routed: 0,
        };
        ns.initialize_default_pathways();
        ns.initialize_queues();
        ns
    }

    /// Set up the default brain pathways based on neuroscience.
    fn initialize_default_pathways(&mut self) {
        // External → Thalamus (all sensory input goes through thalamus)
        self.add_pathway(
            Pathway::new(BrainRegion::External, BrainRegion::Thalamus, 1.0)
                .with_signal_types(vec![SignalType::Sensory, SignalType::Query]),
        );

        // Thalamus → Amygdala (fast emotional processing, "low road")
        self.add_pathway(Pathway::new(
            BrainRegion::Thalamus,
            BrainRegion::Amygdala,
            0.9,
        ));

        // Thalamus → Prefrontal (sensory to working memory)
        self.add_pathway(Pathway::new(
            BrainRegion::Thalamus,
            BrainRegion::Prefrontal,
            0.8,
        ));

        // Thalamus → Hippocampus (sensory to memory encoding)
        self.add_pathway(Pathway::new(
            BrainRegion::Thalamus,
            BrainRegion::Hippocampus,
            0.7,
        ));

        // Amygdala → Hippocampus (emotional tagging of memories)
        self.add_pathway(
            Pathway::new(BrainRegion::Amygdala, BrainRegion::Hippocampus, 0.85)
                .with_signal_types(vec![SignalType::Emotion, SignalType::Memory]),
        );

        // Amygdala → Prefrontal (emotional influence on decisions)
        self.add_pathway(
            Pathway::new(BrainRegion::Amygdala, BrainRegion::Prefrontal, 0.7).bidirectional(),
        );

        // Amygdala → Workspace (emotional signals compete for attention)
        self.add_pathway(
            Pathway::new(BrainRegion::Amygdala, BrainRegion::Workspace, 0.8)
                .with_signal_types(vec![SignalType::Emotion, SignalType::Attention]),
        );

        // Hippocampus ↔ Prefrontal (memory and working memory integration)
        self.add_pathway(
            Pathway::new(BrainRegion::Hippocampus, BrainRegion::Prefrontal, 0.75).bidirectional(),
        );

        // Hippocampus → DMN (memories inform self-model)
        self.add_pathway(
            Pathway::new(BrainRegion::Hippocampus, BrainRegion::DMN, 0.7)
                .with_signal_types(vec![SignalType::Memory]),
        );

        // Prefrontal → Workspace (working memory contents compete for consciousness)
        self.add_pathway(Pathway::new(
            BrainRegion::Prefrontal,
            BrainRegion::Workspace,
            0.85,
        ));

        // Workspace → All regions (conscious broadcast)
        for region in [
            BrainRegion::Amygdala,
            BrainRegion::Hippocampus,
            BrainRegion::Prefrontal,
            BrainRegion::DMN,
            BrainRegion::PredictionEngine,
        ] {
            self.add_pathway(
                Pathway::new(BrainRegion::Workspace, region, 1.0)
                    .with_signal_types(vec![SignalType::Broadcast]),
            );
        }

        // DMN ↔ Workspace (self-reflection enters consciousness)
        self.add_pathway(
            Pathway::new(BrainRegion::DMN, BrainRegion::Workspace, 0.6).bidirectional(),
        );

        // PredictionEngine → Amygdala (surprise affects emotion)
        self.add_pathway(
            Pathway::new(BrainRegion::PredictionEngine, BrainRegion::Amygdala, 0.7)
                .with_signal_types(vec![SignalType::Prediction, SignalType::Error]),
        );

        // PredictionEngine → Hippocampus (prediction errors strengthen encoding)
        self.add_pathway(
            Pathway::new(BrainRegion::PredictionEngine, BrainRegion::Hippocampus, 0.8)
                .with_signal_types(vec![SignalType::Prediction, SignalType::Error]),
        );

        // Prefrontal → External (motor output, responses)
        self.add_pathway(
            Pathway::new(BrainRegion::Prefrontal, BrainRegion::External, 0.9)
                .with_signal_types(vec![SignalType::Motor]),
        );
    }

    fn initialize_queues(&mut self) {
        for region in [
            BrainRegion::Thalamus,
            BrainRegion::Amygdala,
            BrainRegion::Hippocampus,
            BrainRegion::Prefrontal,
            BrainRegion::DMN,
            BrainRegion::Workspace,
            BrainRegion::PredictionEngine,
            BrainRegion::External,
        ] {
            self.signal_queues.insert(region, VecDeque::new());
        }
    }

    /// Add a pathway to the nervous system.
    pub fn add_pathway(&mut self, pathway: Pathway) {
        let key = (pathway.from, pathway.to);
        self.pathways.insert(key, pathway.clone());

        // If bidirectional, add reverse pathway
        if pathway.bidirectional {
            let reverse = Pathway {
                from: pathway.to,
                to: pathway.from,
                strength: pathway.strength,
                effective_strength: pathway.effective_strength,
                signal_types: pathway.signal_types,
                transmission_count: 0,
                bidirectional: false, // Prevent infinite recursion
            };
            self.pathways.insert((reverse.from, reverse.to), reverse);
        }
    }

    /// Send a signal from one region to another.
    pub fn transmit(&mut self, from: BrainRegion, to: BrainRegion, signal: BrainSignal) -> bool {
        let key = (from, to);

        if let Some(pathway) = self.pathways.get_mut(&key) {
            // Check if pathway can carry this signal type
            if !pathway.can_carry(&signal.signal_type) {
                return false;
            }

            // Check transmission threshold
            if pathway.effective_strength < self.config.transmission_threshold {
                return false;
            }

            // Modulate signal salience by pathway strength
            let mut transmitted_signal = signal.clone();
            let new_salience = transmitted_signal.salience.value() * pathway.effective_strength;
            transmitted_signal.salience = crate::signal::Salience::new(new_salience);

            // Queue the signal
            if let Some(queue) = self.signal_queues.get_mut(&to) {
                queue.push_back(transmitted_signal);
            }

            // Update stats
            pathway.transmission_count += 1;
            self.total_routed += 1;

            true
        } else {
            false
        }
    }

    /// Broadcast a signal from the workspace to all connected regions.
    pub fn broadcast(&mut self, signal: BrainSignal) {
        let workspace_pathways: Vec<_> = self
            .pathways
            .iter()
            .filter(|((from, _), _)| *from == BrainRegion::Workspace)
            .map(|((_, to), _)| *to)
            .collect();

        for to in workspace_pathways {
            self.transmit(BrainRegion::Workspace, to, signal.clone());
        }
    }

    /// Get queued signals for a region.
    pub fn get_signals(&mut self, region: BrainRegion) -> Vec<BrainSignal> {
        if let Some(queue) = self.signal_queues.get_mut(&region) {
            queue.drain(..).collect()
        } else {
            Vec::new()
        }
    }

    /// Peek at queued signals without removing them.
    pub fn peek_signals(&self, region: BrainRegion) -> Vec<&BrainSignal> {
        if let Some(queue) = self.signal_queues.get(&region) {
            queue.iter().collect()
        } else {
            Vec::new()
        }
    }

    /// Apply neuromodulator modulation to pathways.
    pub fn apply_modulation(&mut self, from: BrainRegion, to: BrainRegion, factor: f64) {
        if let Some(pathway) = self.pathways.get_mut(&(from, to)) {
            pathway.modulate(factor);
        }
    }

    /// Reset all pathway modulations.
    pub fn reset_all_modulations(&mut self) {
        for pathway in self.pathways.values_mut() {
            pathway.reset_modulation();
        }
    }

    /// Get pathway information.
    pub fn get_pathway(&self, from: BrainRegion, to: BrainRegion) -> Option<&Pathway> {
        self.pathways.get(&(from, to))
    }

    /// Get all outgoing pathways from a region.
    pub fn outgoing_pathways(&self, from: BrainRegion) -> Vec<&Pathway> {
        self.pathways
            .iter()
            .filter(|((f, _), _)| *f == from)
            .map(|(_, p)| p)
            .collect()
    }

    /// Get all incoming pathways to a region.
    pub fn incoming_pathways(&self, to: BrainRegion) -> Vec<&Pathway> {
        self.pathways
            .iter()
            .filter(|((_, t), _)| *t == to)
            .map(|(_, p)| p)
            .collect()
    }

    /// Get statistics about the nervous system.
    pub fn stats(&self) -> NervousSystemStats {
        NervousSystemStats {
            total_pathways: self.pathways.len(),
            total_signals_routed: self.total_routed,
            queued_signals: self.signal_queues.values().map(|q| q.len()).sum(),
            pathway_stats: self
                .pathways
                .values()
                .map(|p| {
                    (
                        format!("{} → {}", p.from.name(), p.to.name()),
                        p.transmission_count,
                    )
                })
                .collect(),
        }
    }

    /// Generate a visual representation of the nervous system.
    pub fn visualize(&self) -> String {
        let mut output = String::new();
        output.push_str("┌─────────────────────────────────────────────────────────┐\n");
        output.push_str("│              NERVOUS SYSTEM PATHWAYS                    │\n");
        output.push_str("├─────────────────────────────────────────────────────────┤\n");

        for pathway in self.pathways.values() {
            let direction = if pathway.bidirectional { "↔" } else { "→" };
            output.push_str(&format!(
                "│ {:15} {} {:15} [str: {:.2}, tx: {}]\n",
                pathway.from.name(),
                direction,
                pathway.to.name(),
                pathway.effective_strength,
                pathway.transmission_count
            ));
        }

        output.push_str("└─────────────────────────────────────────────────────────┘\n");
        output
    }
}

impl Default for NervousSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about the nervous system.
#[derive(Debug, Clone)]
pub struct NervousSystemStats {
    pub total_pathways: usize,
    pub total_signals_routed: u64,
    pub queued_signals: usize,
    pub pathway_stats: Vec<(String, u64)>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nervous_system_creation() {
        let ns = NervousSystem::new();
        assert!(ns.pathways.len() > 10); // Should have many default pathways
    }

    #[test]
    fn test_signal_transmission() {
        let mut ns = NervousSystem::new();
        let signal = BrainSignal::new("test", SignalType::Sensory, "hello");

        // Transmit from External to Thalamus (should work)
        let success = ns.transmit(BrainRegion::External, BrainRegion::Thalamus, signal.clone());
        assert!(success);

        // Check signal was queued
        let signals = ns.get_signals(BrainRegion::Thalamus);
        assert_eq!(signals.len(), 1);
    }

    #[test]
    fn test_pathway_modulation() {
        let mut ns = NervousSystem::new();

        // Get original strength
        let original = ns
            .get_pathway(BrainRegion::Thalamus, BrainRegion::Amygdala)
            .map(|p| p.effective_strength)
            .unwrap_or(0.0);

        // Apply modulation
        ns.apply_modulation(BrainRegion::Thalamus, BrainRegion::Amygdala, 0.5);

        let modulated = ns
            .get_pathway(BrainRegion::Thalamus, BrainRegion::Amygdala)
            .map(|p| p.effective_strength)
            .unwrap_or(0.0);

        assert!(modulated < original);
    }

    #[test]
    fn test_broadcast() {
        let mut ns = NervousSystem::new();
        let signal = BrainSignal::new("workspace", SignalType::Broadcast, "conscious content");

        ns.broadcast(signal);

        // Check that multiple regions received the broadcast
        let amygdala_signals = ns.peek_signals(BrainRegion::Amygdala);
        let hippocampus_signals = ns.peek_signals(BrainRegion::Hippocampus);

        assert!(!amygdala_signals.is_empty());
        assert!(!hippocampus_signals.is_empty());
    }
}
