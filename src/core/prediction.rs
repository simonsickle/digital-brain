//! Prediction Engine - The Dopamine System
//!
//! Computes prediction errors (surprise) and modulates learning.
//! This is the "dopamine" of the digital brain.
//!
//! Key concepts:
//! - Every module maintains predictions about its domain
//! - Prediction error = actual - expected
//! - High error → increase learning rate
//! - Low error → maintain current model

use crate::error::Result;
use crate::signal::{BrainSignal, SignalType};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use uuid::Uuid;

/// Hierarchical layer for predictive coding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PredictionLayer {
    Sensory,
    Perceptual,
    Conceptual,
}

impl PredictionLayer {
    pub fn weight(&self) -> f64 {
        match self {
            PredictionLayer::Sensory => 0.6,
            PredictionLayer::Perceptual => 0.8,
            PredictionLayer::Conceptual => 1.0,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            PredictionLayer::Sensory => "sensory",
            PredictionLayer::Perceptual => "perceptual",
            PredictionLayer::Conceptual => "conceptual",
        }
    }
}

/// Context passed into the prediction engine for hierarchical coding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionContext {
    pub content: String,
    pub tokens: Vec<String>,
    pub features: Vec<String>,
    pub anchors: Vec<String>,
    pub schemas: Vec<String>,
    pub modalities: Vec<String>,
    pub salience: f64,
    pub valence: f64,
    pub arousal: f64,
    pub novelty: f64,
    pub detail: f64,
    pub confidence: f64,
}

/// A structured state used for predictions and evaluations.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PredictionState {
    pub tokens: Vec<String>,
    pub features: Vec<String>,
    pub anchors: Vec<String>,
    pub schemas: Vec<String>,
}

impl PredictionState {
    fn from_context(context: &PredictionContext, layer: PredictionLayer) -> Self {
        match layer {
            PredictionLayer::Sensory => Self {
                tokens: take_unique(&context.tokens, 6),
                ..Default::default()
            },
            PredictionLayer::Perceptual => Self {
                features: take_unique(&context.features, 6),
                anchors: take_unique(&context.anchors, 3),
                ..Default::default()
            },
            PredictionLayer::Conceptual => Self {
                schemas: take_unique(&context.schemas, 4),
                anchors: take_unique(&context.anchors, 4),
                ..Default::default()
            },
        }
    }
}

/// A prediction about future state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prediction {
    /// Unique identifier
    pub id: Uuid,
    /// What module made this prediction
    pub source: String,
    /// What domain this prediction is about
    pub domain: String,
    /// The predicted value/state
    pub expected: serde_json::Value,
    /// Confidence in this prediction (0-1)
    pub confidence: f64,
    /// Hierarchical layer that produced this prediction
    pub layer: PredictionLayer,
    /// Precision (confidence weighting for error magnitude)
    pub precision: f64,
    /// Parent prediction ID (higher layer)
    pub parent_id: Option<Uuid>,
    /// When this prediction was made
    pub created_at: DateTime<Utc>,
    /// When this prediction should be evaluated
    pub valid_until: Option<DateTime<Utc>>,
}

impl Prediction {
    /// Create a new prediction.
    pub fn new(
        source: impl Into<String>,
        domain: impl Into<String>,
        expected: impl Serialize,
        confidence: f64,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            source: source.into(),
            domain: domain.into(),
            expected: serde_json::to_value(expected).unwrap_or(serde_json::Value::Null),
            confidence: confidence.clamp(0.0, 1.0),
            layer: PredictionLayer::Perceptual,
            precision: confidence.clamp(0.0, 1.0),
            parent_id: None,
            created_at: Utc::now(),
            valid_until: None,
        }
    }

    /// Set expiration time for this prediction.
    pub fn expires_at(mut self, time: DateTime<Utc>) -> Self {
        self.valid_until = Some(time);
        self
    }

    /// Set hierarchical layer for this prediction.
    pub fn with_layer(mut self, layer: PredictionLayer) -> Self {
        self.layer = layer;
        self
    }

    /// Set precision weighting for this prediction.
    pub fn with_precision(mut self, precision: f64) -> Self {
        self.precision = precision.clamp(0.0, 1.0);
        self
    }

    /// Link this prediction to a parent (higher layer).
    pub fn with_parent(mut self, parent_id: Uuid) -> Self {
        self.parent_id = Some(parent_id);
        self
    }
}

/// Result of comparing prediction to actual outcome.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionError {
    /// The original prediction
    pub prediction_id: Uuid,
    /// Layer where the prediction originated
    pub layer: PredictionLayer,
    /// Domain of the prediction
    pub domain: String,
    /// Source module of the prediction
    pub source: String,
    /// Expected state
    pub expected: serde_json::Value,
    /// What actually happened
    pub actual: serde_json::Value,
    /// Error magnitude (0-1, where 1 is completely wrong)
    pub error_magnitude: f64,
    /// Direction: positive means actual > expected, negative means actual < expected
    pub error_direction: f64,
    /// Computed surprise level
    pub surprise: f64,
    /// When this error was computed
    pub computed_at: DateTime<Utc>,
}

impl PredictionError {
    /// Is this a surprising outcome?
    pub fn is_surprising(&self) -> bool {
        self.surprise > 0.7
    }

    /// Should we increase learning rate based on this error?
    pub fn should_boost_learning(&self) -> bool {
        self.error_magnitude > 0.5
    }

    /// Convert to a brain signal for broadcasting.
    pub fn to_signal(&self, source: &str) -> BrainSignal {
        BrainSignal::new(source, SignalType::Error, self.clone())
            .with_salience(self.surprise)
            .with_arousal(self.surprise)
            .with_metadata("prediction_error", self.error_magnitude)
            .with_metadata("prediction_layer", self.layer.label())
            .with_metadata("prediction_domain", self.domain.clone())
            .with_metadata("prediction_source", self.source.clone())
    }
}

/// Active inference policy for reducing prediction errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActiveInferencePolicy {
    SeekInformation,
    RefocusAttention,
    TestHypothesis,
    UpdateBeliefs,
    ExploreNovelty,
}

impl ActiveInferencePolicy {
    pub fn label(&self) -> &'static str {
        match self {
            ActiveInferencePolicy::SeekInformation => "seek_information",
            ActiveInferencePolicy::RefocusAttention => "refocus_attention",
            ActiveInferencePolicy::TestHypothesis => "test_hypothesis",
            ActiveInferencePolicy::UpdateBeliefs => "update_beliefs",
            ActiveInferencePolicy::ExploreNovelty => "explore_novelty",
        }
    }

    pub fn signal_type(&self) -> SignalType {
        match self {
            ActiveInferencePolicy::SeekInformation => SignalType::Query,
            ActiveInferencePolicy::RefocusAttention => SignalType::Attention,
            ActiveInferencePolicy::TestHypothesis => SignalType::Motor,
            ActiveInferencePolicy::UpdateBeliefs => SignalType::Prediction,
            ActiveInferencePolicy::ExploreNovelty => SignalType::Motor,
        }
    }
}

/// A proposal for action/attention to reduce prediction error.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveInferenceProposal {
    pub id: Uuid,
    pub policy: ActiveInferencePolicy,
    pub layer_focus: PredictionLayer,
    pub target: Option<String>,
    pub rationale: String,
    pub expected_error_reduction: f64,
    pub confidence: f64,
    pub created_at: DateTime<Utc>,
}

impl ActiveInferenceProposal {
    pub fn to_signal(&self, source: &str) -> BrainSignal {
        BrainSignal::new(source, self.policy.signal_type(), self.clone())
            .with_salience(self.expected_error_reduction)
            .with_arousal(self.expected_error_reduction)
            .with_metadata("active_inference_policy", self.policy.label())
    }
}

/// The prediction engine - computes surprise and modulates learning.
pub struct PredictionEngine {
    /// Active predictions awaiting evaluation
    predictions: HashMap<Uuid, Prediction>,
    /// History of prediction errors
    error_history: Vec<PredictionError>,
    /// Current baseline learning rate
    base_learning_rate: f64,
    /// Rolling average of recent errors
    rolling_error_avg: f64,
    /// Number of errors in rolling average
    error_count: usize,
    /// Recent contexts for hierarchical predictive coding
    recent_contexts: VecDeque<PredictionContext>,
    /// Maximum number of contexts to keep
    max_context_history: usize,
    /// Last hierarchical prediction IDs
    last_hierarchical_ids: Vec<Uuid>,
    /// Precision defaults for each layer
    layer_precision: HashMap<PredictionLayer, f64>,
}

impl PredictionEngine {
    /// Create a new prediction engine.
    pub fn new() -> Self {
        Self {
            predictions: HashMap::new(),
            error_history: Vec::new(),
            base_learning_rate: 0.1,
            rolling_error_avg: 0.0,
            error_count: 0,
            recent_contexts: VecDeque::with_capacity(12),
            max_context_history: 12,
            last_hierarchical_ids: Vec::new(),
            layer_precision: default_layer_precision(),
        }
    }

    /// Register a new prediction.
    pub fn predict(&mut self, prediction: Prediction) -> Uuid {
        let id = prediction.id;
        self.predictions.insert(id, prediction);
        id
    }

    /// Evaluate a prediction against actual outcome.
    pub fn evaluate(
        &mut self,
        prediction_id: Uuid,
        actual: impl Serialize,
        error_magnitude: f64,
    ) -> Result<PredictionError> {
        self.evaluate_with_direction(prediction_id, actual, error_magnitude, 0.0)
    }

    /// Evaluate a prediction with a signed error direction.
    pub fn evaluate_with_direction(
        &mut self,
        prediction_id: Uuid,
        actual: impl Serialize,
        error_magnitude: f64,
        error_direction: f64,
    ) -> Result<PredictionError> {
        let prediction = self.predictions.remove(&prediction_id);

        let error_magnitude = error_magnitude.clamp(0.0, 1.0);

        // Surprise is weighted by prediction confidence
        // High confidence + high error = very surprising
        let confidence = prediction.as_ref().map(|p| p.confidence).unwrap_or(0.5);
        let surprise = error_magnitude * confidence;

        let (layer, domain, source, expected) = if let Some(prediction) = prediction.as_ref() {
            (
                prediction.layer,
                prediction.domain.clone(),
                prediction.source.clone(),
                prediction.expected.clone(),
            )
        } else {
            (
                PredictionLayer::Perceptual,
                "unknown".to_string(),
                "unknown".to_string(),
                serde_json::Value::Null,
            )
        };

        let error = PredictionError {
            prediction_id,
            layer,
            domain,
            source,
            expected,
            actual: serde_json::to_value(actual).unwrap_or(serde_json::Value::Null),
            error_magnitude,
            error_direction: error_direction.clamp(-1.0, 1.0),
            surprise,
            computed_at: Utc::now(),
        };

        // Update rolling average
        self.update_rolling_error(error_magnitude);

        // Store in history (keep last 1000)
        self.error_history.push(error.clone());
        if self.error_history.len() > 1000 {
            self.error_history.remove(0);
        }

        Ok(error)
    }

    /// Generate hierarchical predictions for the next cycle.
    pub fn predict_hierarchical(&mut self, context: &PredictionContext) -> Vec<Prediction> {
        let blended = self.blend_context(context);

        let conceptual_state = PredictionState::from_context(&blended, PredictionLayer::Conceptual);
        let conceptual = Prediction::new(
            "prediction_engine",
            "conceptual",
            &conceptual_state,
            self.estimate_confidence(PredictionLayer::Conceptual, &blended),
        )
        .with_layer(PredictionLayer::Conceptual)
        .with_precision(self.estimate_precision(PredictionLayer::Conceptual, &blended));
        let conceptual_id = conceptual.id;
        self.predict(conceptual.clone());

        let perceptual_state = PredictionState::from_context(&blended, PredictionLayer::Perceptual);
        let perceptual = Prediction::new(
            "prediction_engine",
            "perceptual",
            &perceptual_state,
            self.estimate_confidence(PredictionLayer::Perceptual, &blended),
        )
        .with_layer(PredictionLayer::Perceptual)
        .with_precision(self.estimate_precision(PredictionLayer::Perceptual, &blended))
        .with_parent(conceptual_id);
        let perceptual_id = perceptual.id;
        self.predict(perceptual.clone());

        let sensory_state = PredictionState::from_context(&blended, PredictionLayer::Sensory);
        let sensory = Prediction::new(
            "prediction_engine",
            "sensory",
            &sensory_state,
            self.estimate_confidence(PredictionLayer::Sensory, &blended),
        )
        .with_layer(PredictionLayer::Sensory)
        .with_precision(self.estimate_precision(PredictionLayer::Sensory, &blended))
        .with_parent(perceptual_id);
        self.predict(sensory.clone());

        self.last_hierarchical_ids = vec![conceptual_id, perceptual_id, sensory.id];
        self.record_context(context.clone());

        vec![conceptual, perceptual, sensory]
    }

    /// Evaluate hierarchical predictions against the current context.
    pub fn evaluate_hierarchical(&mut self, context: &PredictionContext) -> Vec<PredictionError> {
        let mut errors = Vec::new();
        let ids: Vec<_> = self.last_hierarchical_ids.drain(..).collect();

        for id in ids {
            if let Some(prediction) = self.predictions.get(&id).cloned() {
                let expected_state =
                    extract_prediction_state(&prediction.expected).unwrap_or_default();
                let actual_state = PredictionState::from_context(context, prediction.layer);
                let (error_magnitude, error_direction) =
                    self.compare_states(&prediction, &expected_state, &actual_state);

                if let Ok(error) =
                    self.evaluate_with_direction(id, actual_state, error_magnitude, error_direction)
                {
                    errors.push(error);
                }
            }
        }

        errors
    }

    /// Propose active inference actions to reduce prediction error.
    pub fn active_inference(
        &self,
        context: &PredictionContext,
        errors: &[PredictionError],
    ) -> Option<ActiveInferenceProposal> {
        if errors.is_empty() {
            return None;
        }

        let avg_surprise = errors.iter().map(|e| e.surprise).sum::<f64>() / errors.len() as f64;
        if avg_surprise < 0.25 {
            return None;
        }

        let mut layer_scores: HashMap<PredictionLayer, f64> = HashMap::new();
        for error in errors {
            *layer_scores.entry(error.layer).or_insert(0.0) +=
                error.surprise * error.layer.weight();
        }

        let (dominant_layer, _) = layer_scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))?;
        let dominant_layer = *dominant_layer;

        let target = context
            .anchors
            .first()
            .cloned()
            .or_else(|| context.schemas.first().cloned())
            .or_else(|| context.tokens.first().cloned());

        let (policy, rationale) = if context.novelty > 0.6 && avg_surprise > 0.45 {
            (
                ActiveInferencePolicy::ExploreNovelty,
                "Novel context detected with elevated surprise".to_string(),
            )
        } else {
            match dominant_layer {
                PredictionLayer::Sensory => (
                    ActiveInferencePolicy::RefocusAttention,
                    "Low-level mismatch suggests attention realignment".to_string(),
                ),
                PredictionLayer::Perceptual => (
                    ActiveInferencePolicy::SeekInformation,
                    "Perceptual mismatch suggests gathering more evidence".to_string(),
                ),
                PredictionLayer::Conceptual => (
                    ActiveInferencePolicy::UpdateBeliefs,
                    "High-level mismatch suggests revising internal model".to_string(),
                ),
            }
        };

        Some(ActiveInferenceProposal {
            id: Uuid::new_v4(),
            policy,
            layer_focus: dominant_layer,
            target,
            rationale,
            expected_error_reduction: (avg_surprise * 0.6 + context.confidence * 0.2)
                .clamp(0.1, 1.0),
            confidence: (context.confidence * 0.6 + (1.0 - context.novelty) * 0.2 + 0.2)
                .clamp(0.0, 1.0),
            created_at: Utc::now(),
        })
    }

    /// Get current effective learning rate.
    ///
    /// This is modulated by recent prediction errors:
    /// - High errors → increased learning rate
    /// - Low errors → decreased learning rate
    pub fn current_learning_rate(&self) -> f64 {
        // Modulate base rate by rolling error average
        // More errors = higher learning rate (we need to adapt)
        let modulation = 1.0 + self.rolling_error_avg;
        (self.base_learning_rate * modulation).clamp(0.01, 1.0)
    }

    /// Update the rolling error average.
    fn update_rolling_error(&mut self, new_error: f64) {
        // Exponential moving average
        let alpha = 0.1;
        if self.error_count == 0 {
            self.rolling_error_avg = new_error;
        } else {
            self.rolling_error_avg = alpha * new_error + (1.0 - alpha) * self.rolling_error_avg;
        }
        self.error_count += 1;
    }

    /// Check if we're in a high-error state (need to adapt).
    pub fn is_adapting(&self) -> bool {
        self.rolling_error_avg > 0.5
    }

    /// Clear expired predictions.
    pub fn clear_expired(&mut self) -> usize {
        let now = Utc::now();
        let before = self.predictions.len();

        self.predictions
            .retain(|_, p| p.valid_until.map(|t| t > now).unwrap_or(true));

        before - self.predictions.len()
    }

    /// Get statistics about prediction performance.
    pub fn stats(&self) -> PredictionStats {
        let recent_errors: Vec<_> = self.error_history.iter().rev().take(100).collect();

        let avg_error = if recent_errors.is_empty() {
            0.0
        } else {
            recent_errors.iter().map(|e| e.error_magnitude).sum::<f64>()
                / recent_errors.len() as f64
        };

        let surprise_count = recent_errors.iter().filter(|e| e.is_surprising()).count();

        PredictionStats {
            active_predictions: self.predictions.len(),
            total_evaluations: self.error_history.len(),
            recent_avg_error: avg_error,
            recent_surprise_rate: surprise_count as f64 / recent_errors.len().max(1) as f64,
            current_learning_rate: self.current_learning_rate(),
            is_adapting: self.is_adapting(),
        }
    }

    fn record_context(&mut self, context: PredictionContext) {
        if self.recent_contexts.len() >= self.max_context_history {
            self.recent_contexts.pop_front();
        }
        self.recent_contexts.push_back(context);
    }

    fn blend_context(&self, context: &PredictionContext) -> PredictionContext {
        if let Some(last) = self.recent_contexts.back() {
            PredictionContext {
                content: context.content.clone(),
                tokens: merge_unique(&context.tokens, &last.tokens, 8),
                features: merge_unique(&context.features, &last.features, 8),
                anchors: merge_unique(&context.anchors, &last.anchors, 5),
                schemas: merge_unique(&context.schemas, &last.schemas, 5),
                modalities: merge_unique(&context.modalities, &last.modalities, 5),
                salience: (context.salience * 0.7 + last.salience * 0.3).clamp(0.0, 1.0),
                valence: (context.valence * 0.7 + last.valence * 0.3).clamp(-1.0, 1.0),
                arousal: (context.arousal * 0.7 + last.arousal * 0.3).clamp(0.0, 1.0),
                novelty: (context.novelty * 0.6 + last.novelty * 0.4).clamp(0.0, 1.0),
                detail: (context.detail * 0.7 + last.detail * 0.3).clamp(0.0, 1.0),
                confidence: (context.confidence * 0.7 + last.confidence * 0.3).clamp(0.0, 1.0),
            }
        } else {
            context.clone()
        }
    }

    fn estimate_confidence(&self, layer: PredictionLayer, context: &PredictionContext) -> f64 {
        let base = match layer {
            PredictionLayer::Sensory => {
                context.confidence * 0.6 + context.detail * 0.3 + context.salience * 0.1
            }
            PredictionLayer::Perceptual => {
                context.confidence * 0.5 + context.detail * 0.2 + context.novelty * 0.2 + 0.1
            }
            PredictionLayer::Conceptual => {
                let schema_strength = (context.schemas.len() as f64 / 4.0).min(1.0);
                context.confidence * 0.4 + schema_strength * 0.4 + context.novelty * 0.2
            }
        };
        base.clamp(0.1, 1.0)
    }

    fn estimate_precision(&self, layer: PredictionLayer, context: &PredictionContext) -> f64 {
        let base = self.layer_precision.get(&layer).copied().unwrap_or(0.7);
        (base * (0.5 + context.confidence * 0.5)).clamp(0.1, 1.0)
    }

    fn compare_states(
        &self,
        prediction: &Prediction,
        expected: &PredictionState,
        actual: &PredictionState,
    ) -> (f64, f64) {
        let overlap = match prediction.layer {
            PredictionLayer::Sensory => overlap_ratio(&expected.tokens, &actual.tokens),
            PredictionLayer::Perceptual => {
                let feature_overlap = overlap_ratio(&expected.features, &actual.features);
                let anchor_overlap = overlap_ratio(&expected.anchors, &actual.anchors);
                (feature_overlap * 0.7 + anchor_overlap * 0.3).clamp(0.0, 1.0)
            }
            PredictionLayer::Conceptual => {
                let schema_overlap = overlap_ratio(&expected.schemas, &actual.schemas);
                let anchor_overlap = overlap_ratio(&expected.anchors, &actual.anchors);
                (schema_overlap * 0.6 + anchor_overlap * 0.4).clamp(0.0, 1.0)
            }
        };

        let base_error = (1.0 - overlap).clamp(0.0, 1.0);
        let error = (base_error * prediction.precision).clamp(0.0, 1.0);
        let direction = error_direction_from_counts(prediction.layer, expected, actual);
        (error, direction)
    }
}

impl Default for PredictionEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about prediction performance.
#[derive(Debug, Clone)]
pub struct PredictionStats {
    pub active_predictions: usize,
    pub total_evaluations: usize,
    pub recent_avg_error: f64,
    pub recent_surprise_rate: f64,
    pub current_learning_rate: f64,
    pub is_adapting: bool,
}

fn default_layer_precision() -> HashMap<PredictionLayer, f64> {
    let mut map = HashMap::new();
    map.insert(PredictionLayer::Sensory, 0.6);
    map.insert(PredictionLayer::Perceptual, 0.75);
    map.insert(PredictionLayer::Conceptual, 0.85);
    map
}

fn take_unique(values: &[String], max: usize) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for value in values {
        if seen.insert(value) {
            out.push(value.clone());
        }
        if out.len() >= max {
            break;
        }
    }
    out
}

fn merge_unique(primary: &[String], secondary: &[String], max: usize) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for value in primary.iter().chain(secondary.iter()) {
        if seen.insert(value) {
            out.push(value.clone());
        }
        if out.len() >= max {
            break;
        }
    }
    out
}

fn overlap_ratio(expected: &[String], actual: &[String]) -> f64 {
    if expected.is_empty() {
        return 0.5;
    }
    let expected_set: HashSet<_> = expected.iter().collect();
    let actual_set: HashSet<_> = actual.iter().collect();
    let intersection = expected_set.intersection(&actual_set).count() as f64;
    (intersection / expected_set.len() as f64).clamp(0.0, 1.0)
}

fn error_direction_from_counts(
    layer: PredictionLayer,
    expected: &PredictionState,
    actual: &PredictionState,
) -> f64 {
    let (expected_len, actual_len) = match layer {
        PredictionLayer::Sensory => (expected.tokens.len(), actual.tokens.len()),
        PredictionLayer::Perceptual => (expected.features.len(), actual.features.len()),
        PredictionLayer::Conceptual => (expected.schemas.len(), actual.schemas.len()),
    };
    if expected_len == 0 {
        return 0.0;
    }
    let diff = actual_len as f64 - expected_len as f64;
    (diff / expected_len as f64).clamp(-1.0, 1.0)
}

fn extract_prediction_state(value: &serde_json::Value) -> Option<PredictionState> {
    serde_json::from_value(value.clone()).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prediction_creation() {
        let pred = Prediction::new("test_module", "temperature", 72.0, 0.9);
        assert_eq!(pred.source, "test_module");
        assert_eq!(pred.domain, "temperature");
        assert_eq!(pred.confidence, 0.9);
    }

    #[test]
    fn test_prediction_evaluation() {
        let mut engine = PredictionEngine::new();

        // Make a confident prediction
        let pred = Prediction::new("test", "value", 100, 0.95);
        let pred_id = engine.predict(pred);

        // Evaluate with high error
        let error = engine.evaluate(pred_id, 50, 0.8).unwrap();

        // High confidence + high error = high surprise
        assert!(error.surprise > 0.7);
        assert!(error.is_surprising());
    }

    #[test]
    fn test_hierarchical_predictions() {
        let mut engine = PredictionEngine::new();
        let context = PredictionContext {
            content: "red triangle".to_string(),
            tokens: vec!["red".to_string(), "triangle".to_string()],
            features: vec!["visual:red".to_string(), "shape:triangle".to_string()],
            anchors: vec!["triangle".to_string()],
            schemas: vec!["geometry pattern".to_string()],
            modalities: vec!["visual".to_string()],
            salience: 0.6,
            valence: 0.1,
            arousal: 0.4,
            novelty: 0.2,
            detail: 0.7,
            confidence: 0.8,
        };

        let predictions = engine.predict_hierarchical(&context);
        assert_eq!(predictions.len(), 3);
        assert!(
            predictions
                .iter()
                .any(|p| p.layer == PredictionLayer::Sensory)
        );
        assert!(
            predictions
                .iter()
                .any(|p| p.layer == PredictionLayer::Perceptual)
        );
        assert!(
            predictions
                .iter()
                .any(|p| p.layer == PredictionLayer::Conceptual)
        );
    }

    #[test]
    fn test_hierarchical_evaluation_and_active_inference() {
        let mut engine = PredictionEngine::new();
        let context = PredictionContext {
            content: "calm ocean".to_string(),
            tokens: vec!["calm".to_string(), "ocean".to_string()],
            features: vec!["audio:waves".to_string()],
            anchors: vec!["ocean".to_string()],
            schemas: vec!["nature scene".to_string()],
            modalities: vec!["auditory".to_string()],
            salience: 0.4,
            valence: 0.2,
            arousal: 0.3,
            novelty: 0.1,
            detail: 0.5,
            confidence: 0.7,
        };
        engine.predict_hierarchical(&context);

        let new_context = PredictionContext {
            content: "city sirens".to_string(),
            tokens: vec!["city".to_string(), "sirens".to_string()],
            features: vec!["audio:alarm".to_string()],
            anchors: vec!["sirens".to_string()],
            schemas: vec!["urban alert".to_string()],
            modalities: vec!["auditory".to_string()],
            salience: 0.8,
            valence: -0.2,
            arousal: 0.7,
            novelty: 0.7,
            detail: 0.6,
            confidence: 0.7,
        };

        let errors = engine.evaluate_hierarchical(&new_context);
        assert_eq!(errors.len(), 3);
        assert!(errors.iter().any(|e| e.error_magnitude > 0.3));

        let proposal = engine.active_inference(&new_context, &errors);
        assert!(proposal.is_some());
    }

    #[test]
    fn test_learning_rate_modulation() {
        let mut engine = PredictionEngine::new();
        let initial_rate = engine.current_learning_rate();

        // Generate many high errors
        for _ in 0..10 {
            let pred = Prediction::new("test", "value", 100, 0.9);
            let id = engine.predict(pred);
            engine.evaluate(id, 0, 0.9).unwrap();
        }

        // Learning rate should have increased
        assert!(engine.current_learning_rate() > initial_rate);
    }

    #[test]
    fn test_adaptation_state() {
        let mut engine = PredictionEngine::new();
        assert!(!engine.is_adapting());

        // Generate errors
        for _ in 0..20 {
            let pred = Prediction::new("test", "value", 100, 0.9);
            let id = engine.predict(pred);
            engine.evaluate(id, 0, 0.8).unwrap();
        }

        assert!(engine.is_adapting());
    }
}
