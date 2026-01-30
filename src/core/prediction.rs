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
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use uuid::Uuid;

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
            created_at: Utc::now(),
            valid_until: None,
        }
    }

    /// Set expiration time for this prediction.
    pub fn expires_at(mut self, time: DateTime<Utc>) -> Self {
        self.valid_until = Some(time);
        self
    }
}

/// Result of comparing prediction to actual outcome.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionError {
    /// The original prediction
    pub prediction_id: Uuid,
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
        let prediction = self.predictions.remove(&prediction_id);
        
        let error_magnitude = error_magnitude.clamp(0.0, 1.0);
        
        // Surprise is weighted by prediction confidence
        // High confidence + high error = very surprising
        let confidence = prediction.as_ref().map(|p| p.confidence).unwrap_or(0.5);
        let surprise = error_magnitude * confidence;
        
        let error = PredictionError {
            prediction_id,
            actual: serde_json::to_value(actual).unwrap_or(serde_json::Value::Null),
            error_magnitude,
            error_direction: 0.0, // Would need domain-specific comparison
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
        
        self.predictions.retain(|_, p| {
            p.valid_until.map(|t| t > now).unwrap_or(true)
        });
        
        before - self.predictions.len()
    }

    /// Get statistics about prediction performance.
    pub fn stats(&self) -> PredictionStats {
        let recent_errors: Vec<_> = self.error_history.iter()
            .rev()
            .take(100)
            .collect();
        
        let avg_error = if recent_errors.is_empty() {
            0.0
        } else {
            recent_errors.iter().map(|e| e.error_magnitude).sum::<f64>() / recent_errors.len() as f64
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
