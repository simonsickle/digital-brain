//! Sensory Cortices - Modality-Specific Feature Extraction
//!
//! These cortical modules sit downstream of the thalamus and transform raw
//! sensory signals into richer feature representations that can be consumed by
//! higher-order systems (prefrontal cortex, hippocampus, DMN, etc.).
//! Each cortex maintains a short history to estimate novelty and confidence.

use crate::signal::{BrainSignal, SignalType};
use serde::{Deserialize, Serialize};
use std::collections::{HashSet, VecDeque};

/// Supported sensory modalities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SensoryModality {
    Visual,
    Auditory,
    Somatosensory,
    Gustatory,
    Olfactory,
}

impl std::fmt::Display for SensoryModality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let text = match self {
            SensoryModality::Visual => "visual",
            SensoryModality::Auditory => "auditory",
            SensoryModality::Somatosensory => "somatosensory",
            SensoryModality::Gustatory => "gustatory",
            SensoryModality::Olfactory => "olfactory",
        };
        write!(f, "{}", text)
    }
}

/// Unified cortical representation handed back to the brain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorticalRepresentation {
    /// Which modality produced this representation.
    pub modality: SensoryModality,
    /// Feature descriptors extracted from the stimulus.
    pub detected_features: Vec<String>,
    /// Main focus of the perception (first/highest-salience feature).
    pub primary_focus: Option<String>,
    /// How much detail was extracted (0.0-1.0).
    pub detail_level: f64,
    /// Confidence in this interpretation (0.0-1.0).
    pub confidence: f64,
    /// Novelty relative to recent history (0.0-1.0).
    pub novelty: f64,
}

/// Internal helper for building cortical representations.
fn build_representation(
    modality: SensoryModality,
    mut features: Vec<String>,
    fallback_feature: Option<String>,
    signal: &BrainSignal,
    history: &mut VecDeque<Vec<String>>,
    history_limit: usize,
) -> CorticalRepresentation {
    if features.is_empty()
        && let Some(fallback) = fallback_feature
    {
        features.push(fallback);
    }

    let novelty = compute_novelty(history, &features);
    record_history(history, features.clone(), history_limit);
    let detail_level = (features.len() as f64 / 6.0).clamp(0.0, 1.0);
    let primary_focus = features.first().cloned();

    // Confidence combines salience, detail, and novelty.
    let salience_component = signal.salience.value();
    let confidence =
        ((salience_component * 0.5) + (detail_level * 0.3) + (novelty * 0.2)).clamp(0.0, 1.0);

    CorticalRepresentation {
        modality,
        detected_features: features,
        primary_focus,
        detail_level,
        confidence,
        novelty,
    }
}

fn compute_novelty(history: &VecDeque<Vec<String>>, features: &[String]) -> f64 {
    if history.is_empty() || features.is_empty() {
        return if features.is_empty() { 0.0 } else { 1.0 };
    }

    let latest = history.back().unwrap();
    let new: HashSet<_> = features.iter().collect();
    let old: HashSet<_> = latest.iter().collect();

    if new.is_empty() {
        return 0.0;
    }

    let diff = new.difference(&old).count() as f64;
    (diff / new.len() as f64).clamp(0.0, 1.0)
}

fn record_history(history: &mut VecDeque<Vec<String>>, features: Vec<String>, limit: usize) {
    if history.len() == limit {
        history.pop_front();
    }
    history.push_back(features);
}

fn modality_matches(signal: &BrainSignal, aliases: &[&str]) -> bool {
    signal
        .metadata
        .get("modality")
        .and_then(|v| v.as_str())
        .map(|v| {
            let lower = v.to_ascii_lowercase();
            aliases
                .iter()
                .any(|alias| alias.eq_ignore_ascii_case(&lower))
        })
        .unwrap_or(false)
}

fn extract_text(signal: &BrainSignal) -> String {
    signal
        .content
        .as_str()
        .map(|s| s.to_string())
        .unwrap_or_else(|| signal.content.to_string())
}

fn collect_keywords(text: &str, keywords: &[&str]) -> Vec<String> {
    let mut features = Vec::new();
    for keyword in keywords {
        if text.contains(keyword) {
            features.push(keyword.to_string());
        }
    }
    features
}

/// Visual cortex (V1/V2/V4 abstraction).
pub struct VisualCortex {
    history: VecDeque<Vec<String>>,
    history_limit: usize,
}

impl Default for VisualCortex {
    fn default() -> Self {
        Self::new()
    }
}

impl VisualCortex {
    pub fn new() -> Self {
        Self {
            history: VecDeque::with_capacity(16),
            history_limit: 16,
        }
    }

    pub fn process(&mut self, signal: &BrainSignal) -> Option<CorticalRepresentation> {
        if signal.signal_type != SignalType::Sensory {
            return None;
        }
        let lower = extract_text(signal).to_ascii_lowercase();
        let metadata_flag = modality_matches(signal, &["visual", "vision", "sight"]);

        let mut features = Vec::new();
        for color in VISUAL_COLORS {
            if lower.contains(color) {
                features.push(format!("color:{color}"));
            }
        }
        for shape in VISUAL_SHAPES {
            if lower.contains(shape) {
                features.push(format!("shape:{shape}"));
            }
        }
        if !collect_keywords(&lower, VISUAL_LIGHT).is_empty() {
            features.push("light:level_change".to_string());
        }
        if !collect_keywords(&lower, VISUAL_MOTION).is_empty() {
            features.push("motion:present".to_string());
        }

        if features.is_empty() && !metadata_flag {
            return None;
        }

        let fallback = metadata_flag.then(|| format!("modality:{}", SensoryModality::Visual));
        Some(build_representation(
            SensoryModality::Visual,
            features,
            fallback,
            signal,
            &mut self.history,
            self.history_limit,
        ))
    }
}

/// Auditory cortex (core + belt areas).
pub struct AuditoryCortex {
    history: VecDeque<Vec<String>>,
    history_limit: usize,
}

impl Default for AuditoryCortex {
    fn default() -> Self {
        Self::new()
    }
}

impl AuditoryCortex {
    pub fn new() -> Self {
        Self {
            history: VecDeque::with_capacity(16),
            history_limit: 16,
        }
    }

    pub fn process(&mut self, signal: &BrainSignal) -> Option<CorticalRepresentation> {
        if signal.signal_type != SignalType::Sensory {
            return None;
        }
        let lower = extract_text(signal).to_ascii_lowercase();
        let metadata_flag = modality_matches(signal, &["auditory", "hearing", "sound"]);

        let mut features = Vec::new();
        for (keyword, feature) in AUDITORY_TONES {
            if lower.contains(keyword) {
                features.push(format!("tone:{feature}"));
            }
        }
        if !collect_keywords(&lower, AUDITORY_VOLUME).is_empty() {
            features.push("volume:salient".to_string());
        }
        if !collect_keywords(&lower, AUDITORY_SOURCES).is_empty() {
            features.push("source:identified".to_string());
        }

        if features.is_empty() && !metadata_flag {
            return None;
        }

        let fallback = metadata_flag.then(|| format!("modality:{}", SensoryModality::Auditory));
        Some(build_representation(
            SensoryModality::Auditory,
            features,
            fallback,
            signal,
            &mut self.history,
            self.history_limit,
        ))
    }
}

/// Primary somatosensory cortex (S1) abstraction.
pub struct SomatosensoryCortex {
    history: VecDeque<Vec<String>>,
    history_limit: usize,
}

impl Default for SomatosensoryCortex {
    fn default() -> Self {
        Self::new()
    }
}

impl SomatosensoryCortex {
    pub fn new() -> Self {
        Self {
            history: VecDeque::with_capacity(16),
            history_limit: 16,
        }
    }

    pub fn process(&mut self, signal: &BrainSignal) -> Option<CorticalRepresentation> {
        if signal.signal_type != SignalType::Sensory {
            return None;
        }
        let lower = extract_text(signal).to_ascii_lowercase();
        let metadata_flag = modality_matches(signal, &["somatosensory", "tactile", "touch"]);

        let mut features = Vec::new();
        for texture in SOMATOSENSORY_TEXTURES {
            if lower.contains(texture) {
                features.push(format!("texture:{texture}"));
            }
        }
        for temp in SOMATOSENSORY_TEMPERATURE {
            if lower.contains(temp) {
                features.push(format!("temperature:{temp}"));
            }
        }
        if !collect_keywords(&lower, SOMATOSENSORY_PAIN).is_empty() {
            features.push("nociception:triggered".to_string());
        }

        if features.is_empty() && !metadata_flag {
            return None;
        }

        let fallback =
            metadata_flag.then(|| format!("modality:{}", SensoryModality::Somatosensory));
        Some(build_representation(
            SensoryModality::Somatosensory,
            features,
            fallback,
            signal,
            &mut self.history,
            self.history_limit,
        ))
    }
}

/// Gustatory cortex (insula/frontal operculum).
pub struct GustatoryCortex {
    history: VecDeque<Vec<String>>,
    history_limit: usize,
}

impl Default for GustatoryCortex {
    fn default() -> Self {
        Self::new()
    }
}

impl GustatoryCortex {
    pub fn new() -> Self {
        Self {
            history: VecDeque::with_capacity(16),
            history_limit: 16,
        }
    }

    pub fn process(&mut self, signal: &BrainSignal) -> Option<CorticalRepresentation> {
        if signal.signal_type != SignalType::Sensory {
            return None;
        }
        let lower = extract_text(signal).to_ascii_lowercase();
        let metadata_flag = modality_matches(signal, &["gustatory", "taste", "flavor"]);

        let mut features = Vec::new();
        for taste in GUSTATORY_TASTES {
            if lower.contains(taste) {
                features.push(format!("taste:{taste}"));
            }
        }
        if !collect_keywords(&lower, GUSTATORY_TEXTURES).is_empty() {
            features.push("texture:mouthfeel".to_string());
        }

        if features.is_empty() && !metadata_flag {
            return None;
        }

        let fallback = metadata_flag.then(|| format!("modality:{}", SensoryModality::Gustatory));
        Some(build_representation(
            SensoryModality::Gustatory,
            features,
            fallback,
            signal,
            &mut self.history,
            self.history_limit,
        ))
    }
}

/// Olfactory cortex (piriform + orbitofrontal integration).
pub struct OlfactoryCortex {
    history: VecDeque<Vec<String>>,
    history_limit: usize,
}

impl Default for OlfactoryCortex {
    fn default() -> Self {
        Self::new()
    }
}

impl OlfactoryCortex {
    pub fn new() -> Self {
        Self {
            history: VecDeque::with_capacity(16),
            history_limit: 16,
        }
    }

    pub fn process(&mut self, signal: &BrainSignal) -> Option<CorticalRepresentation> {
        if signal.signal_type != SignalType::Sensory {
            return None;
        }
        let lower = extract_text(signal).to_ascii_lowercase();
        let metadata_flag = modality_matches(signal, &["olfactory", "smell", "scent"]);

        let mut features = Vec::new();
        for scent in OLFACTORY_SCENTS {
            if lower.contains(scent) {
                features.push(format!("scent:{scent}"));
            }
        }
        if !collect_keywords(&lower, OLFACTORY_INTENSITY).is_empty() {
            features.push("intensity:salient".to_string());
        }

        if features.is_empty() && !metadata_flag {
            return None;
        }

        let fallback = metadata_flag.then(|| format!("modality:{}", SensoryModality::Olfactory));
        Some(build_representation(
            SensoryModality::Olfactory,
            features,
            fallback,
            signal,
            &mut self.history,
            self.history_limit,
        ))
    }
}

// --- Keyword dictionaries ---

const VISUAL_COLORS: &[&str] = &[
    "red", "blue", "green", "yellow", "purple", "orange", "black", "white", "pink", "gray",
];
const VISUAL_SHAPES: &[&str] = &[
    "circle", "square", "triangle", "line", "curve", "edge", "pattern", "shape", "form",
];
const VISUAL_LIGHT: &[&str] = &[
    "bright", "glow", "dim", "dark", "shadow", "sparkle", "flash",
];
const VISUAL_MOTION: &[&str] = &[
    "move", "motion", "shift", "sway", "blink", "flicker", "flow",
];

const AUDITORY_TONES: &[(&str, &str)] = &[
    ("melody", "melody"),
    ("tone", "tone"),
    ("pitch", "pitch"),
    ("rhythm", "rhythm"),
    ("beat", "rhythm"),
    ("song", "melody"),
    ("music", "melody"),
];
const AUDITORY_VOLUME: &[&str] = &["loud", "quiet", "soft", "silent", "noisy", "volume"];
const AUDITORY_SOURCES: &[&str] = &["voice", "speech", "echo", "wind", "rain", "footsteps"];

const SOMATOSENSORY_TEXTURES: &[&str] = &["rough", "smooth", "soft", "hard", "gritty", "silky"];
const SOMATOSENSORY_TEMPERATURE: &[&str] = &["warm", "hot", "cold", "cool", "chill"];
const SOMATOSENSORY_PAIN: &[&str] = &["pain", "ache", "sting", "hurt", "sore", "burn"];

const GUSTATORY_TASTES: &[&str] = &["sweet", "bitter", "sour", "salty", "umami", "savory"];
const GUSTATORY_TEXTURES: &[&str] = &[
    "creamy", "crunchy", "chewy", "smooth", "crispy", "silky", "dry",
];

const OLFACTORY_SCENTS: &[&str] = &[
    "floral", "earthy", "fresh", "musty", "citrus", "smoky", "fragrant", "pungent",
];
const OLFACTORY_INTENSITY: &[&str] = &["strong", "faint", "overpowering", "lingering"];

#[cfg(test)]
mod tests {
    use super::*;

    fn sensory_signal(text: &str) -> BrainSignal {
        BrainSignal::new("test", SignalType::Sensory, text).with_salience(0.6)
    }

    #[test]
    fn visual_detects_colors_and_shapes() {
        let mut cortex = VisualCortex::new();
        let signal = sensory_signal("A bright red circle glows in the dark.");
        let rep = cortex
            .process(&signal)
            .expect("should detect visual features");
        assert_eq!(rep.modality, SensoryModality::Visual);
        assert!(
            rep.detected_features
                .iter()
                .any(|f| f.contains("color:red"))
        );
        assert!(
            rep.detected_features
                .iter()
                .any(|f| f.contains("shape:circle"))
        );
        assert!(rep.novelty > 0.0);
    }

    #[test]
    fn auditory_detects_sound_keywords() {
        let mut cortex = AuditoryCortex::new();
        let signal = sensory_signal("Soft music and a gentle rhythm fill the room.");
        let rep = cortex
            .process(&signal)
            .expect("should detect auditory features");
        assert_eq!(rep.modality, SensoryModality::Auditory);
        assert!(
            rep.detected_features
                .iter()
                .any(|f| f.starts_with("tone:melody"))
        );
    }

    #[test]
    fn somatosensory_detects_texture_and_temperature() {
        let mut cortex = SomatosensoryCortex::new();
        let signal = sensory_signal("A warm, smooth stone rests in the palm.");
        let rep = cortex
            .process(&signal)
            .expect("should detect somatosensory features");
        assert!(
            rep.detected_features
                .iter()
                .any(|f| f.starts_with("texture:smooth"))
        );
        assert!(
            rep.detected_features
                .iter()
                .any(|f| f.starts_with("temperature:warm"))
        );
    }

    #[test]
    fn gustatory_responds_to_taste_vocabulary() {
        let mut cortex = GustatoryCortex::new();
        let signal = sensory_signal("The dessert is sweet and creamy.");
        let rep = cortex
            .process(&signal)
            .expect("should detect gustatory features");
        assert!(
            rep.detected_features
                .iter()
                .any(|f| f.starts_with("taste:sweet"))
        );
    }

    #[test]
    fn olfactory_detects_scent_descriptions() {
        let mut cortex = OlfactoryCortex::new();
        let signal = sensory_signal("A strong floral aroma lingers in the air.");
        let rep = cortex
            .process(&signal)
            .expect("should detect olfactory features");
        assert!(
            rep.detected_features
                .iter()
                .any(|f| f.starts_with("scent:floral"))
        );
    }

    #[test]
    fn metadata_triggers_fallback_features() {
        let mut cortex = VisualCortex::new();
        let signal = BrainSignal::new("test", SignalType::Sensory, "Ambiguous input")
            .with_metadata("modality", "visual")
            .with_salience(0.4);
        let rep = cortex
            .process(&signal)
            .expect("metadata should force processing");
        assert!(
            rep.detected_features
                .iter()
                .any(|f| f.contains("modality:visual"))
        );
    }
}
