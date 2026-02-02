//! Posterior Parietal Cortex - Multimodal Binding and Context Integration
//!
//! The posterior parietal cortex binds sensory features into a unified
//! context representation before signals reach the global workspace.

use crate::regions::sensory_cortex::{CorticalRepresentation, SensoryModality};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

/// Integrated multimodal context derived from sensory cortices.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodalContext {
    /// Modalities included in this context.
    pub modalities: Vec<SensoryModality>,
    /// Dominant modality (highest confidence/detail).
    pub dominant_modality: Option<SensoryModality>,
    /// Combined feature list, prefixed with modality.
    pub integrated_features: Vec<String>,
    /// Features per modality.
    pub modality_features: HashMap<SensoryModality, Vec<String>>,
    /// Strength of multimodal binding (0.0 to 1.0).
    pub binding_strength: f64,
    /// Overall confidence in the integrated percept (0.0 to 1.0).
    pub confidence: f64,
    /// Novelty relative to recent contexts (0.0 to 1.0).
    pub novelty: f64,
    /// Aggregate detail level (0.0 to 1.0).
    pub detail_level: f64,
    /// Anchor features that summarize the context.
    pub anchors: Vec<String>,
}

/// Posterior parietal cortex module.
pub struct PosteriorParietalCortex {
    history: VecDeque<Vec<String>>,
    history_limit: usize,
}

impl Default for PosteriorParietalCortex {
    fn default() -> Self {
        Self::new()
    }
}

impl PosteriorParietalCortex {
    pub fn new() -> Self {
        Self {
            history: VecDeque::with_capacity(16),
            history_limit: 16,
        }
    }

    /// Integrate sensory cortical representations into a multimodal context.
    pub fn integrate(
        &mut self,
        representations: &[CorticalRepresentation],
    ) -> Option<MultimodalContext> {
        if representations.is_empty() {
            return None;
        }

        let mut modalities = Vec::new();
        let mut modality_features: HashMap<SensoryModality, Vec<String>> = HashMap::new();
        let mut integrated_features = Vec::new();
        let mut integrated_seen = HashSet::new();
        let mut anchors = Vec::new();

        let mut confidence_sum = 0.0;
        let mut novelty_sum = 0.0;
        let mut detail_sum = 0.0;
        let mut dominant: Option<(SensoryModality, f64)> = None;

        for rep in representations {
            if !modalities.contains(&rep.modality) {
                modalities.push(rep.modality);
            }

            let mut features = rep.detected_features.clone();
            if features.is_empty() {
                if let Some(primary) = rep.primary_focus.clone() {
                    features.push(primary);
                } else {
                    features.push(format!("modality:{}", rep.modality));
                }
            }
            features = dedupe_features(features);

            for feature in &features {
                let labeled = format!("{}:{}", rep.modality, feature);
                if integrated_seen.insert(labeled.clone()) {
                    integrated_features.push(labeled);
                }
            }

            modality_features.insert(rep.modality, features.clone());

            if let Some(primary) = rep.primary_focus.clone()
                && anchors.len() < 3
                && !anchors.contains(&primary)
            {
                anchors.push(primary);
            }

            let weight = rep.confidence * 0.6 + rep.detail_level * 0.4;
            if dominant.as_ref().is_none_or(|(_, score)| weight > *score) {
                dominant = Some((rep.modality, weight));
            }

            confidence_sum += rep.confidence;
            novelty_sum += rep.novelty;
            detail_sum += rep.detail_level;
        }

        let count = representations.len() as f64;
        let avg_confidence = confidence_sum / count;
        let avg_novelty = novelty_sum / count;
        let avg_detail = detail_sum / count;

        let overlap = feature_overlap(&modality_features);
        let modality_bonus = ((modalities.len().saturating_sub(1)) as f64 * 0.15).min(0.4);
        let binding_strength = (overlap + modality_bonus).clamp(0.0, 1.0);

        let context_novelty = compute_novelty(&self.history, &integrated_features);
        record_history(
            &mut self.history,
            integrated_features.clone(),
            self.history_limit,
        );

        let confidence = (avg_confidence * 0.7 + binding_strength * 0.3).clamp(0.0, 1.0);
        let novelty =
            (avg_novelty * 0.6 + context_novelty * 0.3 + modality_bonus * 0.1).clamp(0.0, 1.0);
        let detail_level = (avg_detail + modality_bonus * 0.2).clamp(0.0, 1.0);

        Some(MultimodalContext {
            modalities,
            dominant_modality: dominant.map(|(modality, _)| modality),
            integrated_features,
            modality_features,
            binding_strength,
            confidence,
            novelty,
            detail_level,
            anchors,
        })
    }
}

fn dedupe_features(features: Vec<String>) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut deduped = Vec::new();
    for feature in features {
        if seen.insert(feature.clone()) {
            deduped.push(feature);
        }
    }
    deduped
}

fn feature_overlap(modality_features: &HashMap<SensoryModality, Vec<String>>) -> f64 {
    let modalities: Vec<_> = modality_features.keys().copied().collect();
    if modalities.len() < 2 {
        return 0.0;
    }

    let mut total: f64 = 0.0;
    let mut comparisons: f64 = 0.0;

    for i in 0..modalities.len() {
        for j in (i + 1)..modalities.len() {
            let a = modality_features
                .get(&modalities[i])
                .map(|features| feature_types(features))
                .unwrap_or_default();
            let b = modality_features
                .get(&modalities[j])
                .map(|features| feature_types(features))
                .unwrap_or_default();
            let union = a.union(&b).count() as f64;
            let intersection = a.intersection(&b).count() as f64;
            if union > 0.0 {
                total += intersection / union;
                comparisons += 1.0;
            }
        }
    }

    if comparisons == 0.0 {
        0.0
    } else {
        (total / comparisons).clamp(0.0, 1.0)
    }
}

fn feature_types(features: &[String]) -> HashSet<String> {
    features
        .iter()
        .map(|feature| feature.split(':').next().unwrap_or(feature).to_string())
        .collect()
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

#[cfg(test)]
mod tests {
    use super::*;

    fn rep(
        modality: SensoryModality,
        features: &[&str],
        primary: Option<&str>,
        confidence: f64,
        novelty: f64,
    ) -> CorticalRepresentation {
        CorticalRepresentation {
            modality,
            detected_features: features.iter().map(|f| f.to_string()).collect(),
            primary_focus: primary.map(|p| p.to_string()),
            detail_level: (features.len() as f64 / 4.0).clamp(0.0, 1.0),
            confidence,
            novelty,
        }
    }

    #[test]
    fn integrates_multimodal_context() {
        let mut cortex = PosteriorParietalCortex::new();
        let visual = rep(
            SensoryModality::Visual,
            &["color:red", "shape:circle"],
            Some("color:red"),
            0.7,
            0.6,
        );
        let gustatory = rep(
            SensoryModality::Gustatory,
            &["taste:sweet", "texture:mouthfeel"],
            Some("taste:sweet"),
            0.6,
            0.4,
        );
        let somatic = rep(
            SensoryModality::Somatosensory,
            &["texture:smooth", "temperature:warm"],
            Some("texture:smooth"),
            0.65,
            0.5,
        );

        let context = cortex
            .integrate(&[visual, gustatory, somatic])
            .expect("context should be produced");

        assert_eq!(context.modalities.len(), 3);
        assert!(context.binding_strength > 0.0);
        assert!(
            context
                .integrated_features
                .iter()
                .any(|f| f.contains("visual:color:red"))
        );
        assert!(context.confidence > 0.0);
        assert!(context.novelty > 0.0);
    }

    #[test]
    fn returns_none_for_empty_input() {
        let mut cortex = PosteriorParietalCortex::new();
        assert!(cortex.integrate(&[]).is_none());
    }

    #[test]
    fn novelty_declines_with_repetition() {
        let mut cortex = PosteriorParietalCortex::new();
        let visual = rep(
            SensoryModality::Visual,
            &["color:blue"],
            Some("color:blue"),
            0.6,
            0.4,
        );

        let first = cortex
            .integrate(std::slice::from_ref(&visual))
            .expect("first context");
        let second = cortex.integrate(&[visual]).expect("second context");

        assert!(second.novelty < first.novelty);
    }
}
