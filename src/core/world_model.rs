//! World Model
//!
//! Representation of the external world state, including:
//! - Entities and their properties
//! - Relationships between entities
//! - Predictions about future states
//! - Prediction evaluation and learning
//!
//! This enables the agent to:
//! - Maintain situational awareness
//! - Plan actions based on expected outcomes
//! - Learn from prediction errors

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use crate::signal::Valence;

/// Unique identifier for an entity
pub type EntityId = Uuid;

/// Unique identifier for a prediction
pub type PredictionId = Uuid;

/// An entity in the world model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    /// Unique identifier
    pub id: EntityId,
    /// Entity type/category
    pub entity_type: String,
    /// Name/label
    pub name: String,
    /// Properties (key-value)
    pub properties: HashMap<String, PropertyValue>,
    /// When first observed
    pub first_seen: DateTime<Utc>,
    /// When last updated
    pub last_updated: DateTime<Utc>,
    /// Confidence in existence (0-1)
    pub confidence: f64,
    /// Emotional valence associated with this entity
    pub valence: Valence,
    /// Tags for categorization
    pub tags: Vec<String>,
}

/// Value of a property (typed)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PropertyValue {
    String(String),
    Number(f64),
    Boolean(bool),
    List(Vec<PropertyValue>),
    Null,
}

impl PropertyValue {
    pub fn as_string(&self) -> Option<&str> {
        match self {
            PropertyValue::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_number(&self) -> Option<f64> {
        match self {
            PropertyValue::Number(n) => Some(*n),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            PropertyValue::Boolean(b) => Some(*b),
            _ => None,
        }
    }
}

impl From<&str> for PropertyValue {
    fn from(s: &str) -> Self {
        PropertyValue::String(s.to_string())
    }
}

impl From<String> for PropertyValue {
    fn from(s: String) -> Self {
        PropertyValue::String(s)
    }
}

impl From<f64> for PropertyValue {
    fn from(n: f64) -> Self {
        PropertyValue::Number(n)
    }
}

impl From<i64> for PropertyValue {
    fn from(n: i64) -> Self {
        PropertyValue::Number(n as f64)
    }
}

impl From<bool> for PropertyValue {
    fn from(b: bool) -> Self {
        PropertyValue::Boolean(b)
    }
}

impl Default for Entity {
    fn default() -> Self {
        Self {
            id: Uuid::new_v4(),
            entity_type: String::new(),
            name: String::new(),
            properties: HashMap::new(),
            first_seen: Utc::now(),
            last_updated: Utc::now(),
            confidence: 1.0,
            valence: Valence::new(0.0),
            tags: Vec::new(),
        }
    }
}

impl Entity {
    /// Create a new entity
    pub fn new(entity_type: &str, name: &str) -> Self {
        Self {
            entity_type: entity_type.to_string(),
            name: name.to_string(),
            ..Default::default()
        }
    }

    /// Set a property
    pub fn with_property(mut self, key: &str, value: impl Into<PropertyValue>) -> Self {
        self.properties.insert(key.to_string(), value.into());
        self
    }

    /// Set valence
    pub fn with_valence(mut self, valence: f64) -> Self {
        self.valence = Valence::new(valence);
        self
    }

    /// Set confidence
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Add a tag
    pub fn with_tag(mut self, tag: &str) -> Self {
        self.tags.push(tag.to_string());
        self
    }

    /// Update a property
    pub fn set_property(&mut self, key: &str, value: impl Into<PropertyValue>) {
        self.properties.insert(key.to_string(), value.into());
        self.last_updated = Utc::now();
    }

    /// Get a property
    pub fn get_property(&self, key: &str) -> Option<&PropertyValue> {
        self.properties.get(key)
    }

    /// Decay confidence over time
    pub fn decay_confidence(&mut self, factor: f64) {
        self.confidence = (self.confidence * factor).max(0.0);
    }
}

/// Type of relationship between entities
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RelationType {
    /// Entity A contains entity B
    Contains,
    /// Entity A is part of entity B
    PartOf,
    /// Entity A is near entity B
    Near,
    /// Entity A owns entity B
    Owns,
    /// Entity A knows entity B (social)
    Knows,
    /// Entity A causes entity B
    Causes,
    /// Entity A is similar to entity B
    SimilarTo,
    /// Entity A depends on entity B
    DependsOn,
    /// Custom relationship type
    Custom(String),
}

/// A relationship between two entities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relationship {
    /// Source entity
    pub source: EntityId,
    /// Target entity
    pub target: EntityId,
    /// Type of relationship
    pub relation_type: RelationType,
    /// Strength of relationship (0-1)
    pub strength: f64,
    /// When established
    pub established: DateTime<Utc>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl Relationship {
    pub fn new(source: EntityId, target: EntityId, relation_type: RelationType) -> Self {
        Self {
            source,
            target,
            relation_type,
            strength: 1.0,
            established: Utc::now(),
            metadata: HashMap::new(),
        }
    }

    pub fn with_strength(mut self, strength: f64) -> Self {
        self.strength = strength.clamp(0.0, 1.0);
        self
    }

    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
}

/// A prediction about future world state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldPrediction {
    /// Unique identifier
    pub id: PredictionId,
    /// What is being predicted
    pub description: String,
    /// Entity this prediction is about (if applicable)
    pub entity_id: Option<EntityId>,
    /// Property being predicted (if applicable)
    pub property: Option<String>,
    /// Predicted value
    pub predicted_value: PropertyValue,
    /// Confidence in prediction (0-1)
    pub confidence: f64,
    /// When the prediction was made
    pub made_at: DateTime<Utc>,
    /// When the prediction should be evaluated
    pub evaluate_at: Option<DateTime<Utc>>,
    /// Has this prediction been evaluated?
    pub evaluated: bool,
    /// Was the prediction correct? (after evaluation)
    pub correct: Option<bool>,
    /// Actual value (after evaluation)
    pub actual_value: Option<PropertyValue>,
    /// Prediction error magnitude (after evaluation)
    pub error_magnitude: Option<f64>,
}

impl WorldPrediction {
    pub fn new(description: &str, predicted_value: PropertyValue) -> Self {
        Self {
            id: Uuid::new_v4(),
            description: description.to_string(),
            entity_id: None,
            property: None,
            predicted_value,
            confidence: 0.5,
            made_at: Utc::now(),
            evaluate_at: None,
            evaluated: false,
            correct: None,
            actual_value: None,
            error_magnitude: None,
        }
    }

    pub fn for_entity(mut self, entity_id: EntityId, property: &str) -> Self {
        self.entity_id = Some(entity_id);
        self.property = Some(property.to_string());
        self
    }

    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    pub fn evaluate_by(mut self, time: DateTime<Utc>) -> Self {
        self.evaluate_at = Some(time);
        self
    }

    /// Evaluate the prediction against actual value
    pub fn evaluate(&mut self, actual: PropertyValue) {
        self.actual_value = Some(actual.clone());
        self.evaluated = true;

        // Determine correctness and error
        match (&self.predicted_value, &actual) {
            (PropertyValue::Number(pred), PropertyValue::Number(act)) => {
                let error = (pred - act).abs();
                self.error_magnitude = Some(error);
                // Consider correct if within 10% or 0.1 absolute
                self.correct = Some(error < pred.abs() * 0.1 || error < 0.1);
            }
            (PropertyValue::Boolean(pred), PropertyValue::Boolean(act)) => {
                self.correct = Some(pred == act);
                self.error_magnitude = Some(if pred == act { 0.0 } else { 1.0 });
            }
            (PropertyValue::String(pred), PropertyValue::String(act)) => {
                self.correct = Some(pred == act);
                self.error_magnitude = Some(if pred == act { 0.0 } else { 1.0 });
            }
            _ => {
                self.correct = Some(self.predicted_value == actual);
                self.error_magnitude = Some(if self.predicted_value == actual {
                    0.0
                } else {
                    1.0
                });
            }
        }
    }
}

/// Statistics about the world model
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WorldModelStats {
    pub total_entities: usize,
    pub total_relationships: usize,
    pub total_predictions: usize,
    pub evaluated_predictions: usize,
    pub prediction_accuracy: f64,
    pub average_confidence: f64,
}

/// The world model
#[derive(Debug)]
pub struct WorldModel {
    /// All entities
    entities: HashMap<EntityId, Entity>,
    /// All relationships
    relationships: Vec<Relationship>,
    /// Active predictions
    predictions: Vec<WorldPrediction>,
    /// Prediction history (for learning)
    prediction_history: Vec<WorldPrediction>,
    /// Maximum prediction history
    max_history: usize,
    /// When the model was last updated
    last_updated: DateTime<Utc>,
    /// Statistics
    stats: WorldModelStats,
}

impl Default for WorldModel {
    fn default() -> Self {
        Self::new()
    }
}

impl WorldModel {
    /// Create a new world model
    pub fn new() -> Self {
        Self {
            entities: HashMap::new(),
            relationships: Vec::new(),
            predictions: Vec::new(),
            prediction_history: Vec::new(),
            max_history: 1000,
            last_updated: Utc::now(),
            stats: WorldModelStats::default(),
        }
    }

    /// Add an entity
    pub fn add_entity(&mut self, entity: Entity) -> EntityId {
        let id = entity.id;
        self.entities.insert(id, entity);
        self.last_updated = Utc::now();
        self.update_stats();
        id
    }

    /// Get an entity by ID
    pub fn get_entity(&self, id: EntityId) -> Option<&Entity> {
        self.entities.get(&id)
    }

    /// Get a mutable reference to an entity
    pub fn get_entity_mut(&mut self, id: EntityId) -> Option<&mut Entity> {
        self.entities.get_mut(&id)
    }

    /// Find entities by type
    pub fn find_by_type(&self, entity_type: &str) -> Vec<&Entity> {
        self.entities
            .values()
            .filter(|e| e.entity_type == entity_type)
            .collect()
    }

    /// Find entities by name (partial match)
    pub fn find_by_name(&self, name: &str) -> Vec<&Entity> {
        let name_lower = name.to_lowercase();
        self.entities
            .values()
            .filter(|e| e.name.to_lowercase().contains(&name_lower))
            .collect()
    }

    /// Find entities with a tag
    pub fn find_by_tag(&self, tag: &str) -> Vec<&Entity> {
        self.entities
            .values()
            .filter(|e| e.tags.iter().any(|t| t == tag))
            .collect()
    }

    /// Update an entity's property
    pub fn update_entity_property(
        &mut self,
        id: EntityId,
        key: &str,
        value: impl Into<PropertyValue>,
    ) -> bool {
        if let Some(entity) = self.entities.get_mut(&id) {
            entity.set_property(key, value);
            self.last_updated = Utc::now();
            true
        } else {
            false
        }
    }

    /// Remove an entity
    pub fn remove_entity(&mut self, id: EntityId) -> Option<Entity> {
        // Also remove related relationships
        self.relationships
            .retain(|r| r.source != id && r.target != id);
        let entity = self.entities.remove(&id);
        if entity.is_some() {
            self.update_stats();
        }
        entity
    }

    /// Add a relationship
    pub fn add_relationship(&mut self, relationship: Relationship) {
        // Verify both entities exist
        if self.entities.contains_key(&relationship.source)
            && self.entities.contains_key(&relationship.target)
        {
            self.relationships.push(relationship);
            self.last_updated = Utc::now();
            self.update_stats();
        }
    }

    /// Get relationships for an entity
    pub fn get_relationships(&self, entity_id: EntityId) -> Vec<&Relationship> {
        self.relationships
            .iter()
            .filter(|r| r.source == entity_id || r.target == entity_id)
            .collect()
    }

    /// Get outgoing relationships
    pub fn get_outgoing(&self, entity_id: EntityId) -> Vec<&Relationship> {
        self.relationships
            .iter()
            .filter(|r| r.source == entity_id)
            .collect()
    }

    /// Get incoming relationships
    pub fn get_incoming(&self, entity_id: EntityId) -> Vec<&Relationship> {
        self.relationships
            .iter()
            .filter(|r| r.target == entity_id)
            .collect()
    }

    /// Find related entities
    pub fn related_entities(&self, entity_id: EntityId) -> Vec<&Entity> {
        let related_ids: Vec<EntityId> = self
            .relationships
            .iter()
            .filter_map(|r| {
                if r.source == entity_id {
                    Some(r.target)
                } else if r.target == entity_id {
                    Some(r.source)
                } else {
                    None
                }
            })
            .collect();

        related_ids
            .iter()
            .filter_map(|id| self.entities.get(id))
            .collect()
    }

    /// Make a prediction
    pub fn predict(&mut self, prediction: WorldPrediction) -> PredictionId {
        let id = prediction.id;
        self.predictions.push(prediction);
        self.update_stats();
        id
    }

    /// Evaluate pending predictions
    pub fn evaluate_predictions(&mut self) {
        let now = Utc::now();

        for prediction in &mut self.predictions {
            if prediction.evaluated {
                continue;
            }

            // Check if it's time to evaluate
            if let Some(evaluate_at) = prediction.evaluate_at
                && now < evaluate_at
            {
                continue;
            }

            // If prediction is about an entity property, get actual value
            if let (Some(entity_id), Some(property)) = (prediction.entity_id, &prediction.property)
                && let Some(entity) = self.entities.get(&entity_id)
                && let Some(actual) = entity.get_property(property)
            {
                prediction.evaluate(actual.clone());
            }
        }

        // Move evaluated predictions to history
        let (evaluated, pending): (Vec<_>, Vec<_>) =
            self.predictions.drain(..).partition(|p| p.evaluated);

        self.predictions = pending;
        for p in evaluated {
            self.prediction_history.push(p);
        }

        // Trim history
        while self.prediction_history.len() > self.max_history {
            self.prediction_history.remove(0);
        }

        self.update_stats();
    }

    /// Record a prediction error summary for learning and planning.
    pub fn record_prediction_error(&mut self, description: &str, magnitude: f64, confidence: f64) {
        let mut prediction = WorldPrediction::new(description, PropertyValue::Null);
        prediction.evaluated = true;
        prediction.correct = Some(magnitude < 0.2);
        prediction.error_magnitude = Some(magnitude.clamp(0.0, 1.0));
        prediction.confidence = confidence.clamp(0.0, 1.0);

        self.prediction_history.push(prediction);
        while self.prediction_history.len() > self.max_history {
            self.prediction_history.remove(0);
        }

        self.update_stats();
    }

    /// Get prediction accuracy
    pub fn prediction_accuracy(&self) -> f64 {
        let evaluated: Vec<_> = self
            .prediction_history
            .iter()
            .filter(|p| p.evaluated)
            .collect();

        if evaluated.is_empty() {
            return 0.5; // No data, assume 50%
        }

        let correct = evaluated.iter().filter(|p| p.correct == Some(true)).count();
        correct as f64 / evaluated.len() as f64
    }

    /// Decay confidence in all entities
    pub fn decay_confidence(&mut self, factor: f64) {
        for entity in self.entities.values_mut() {
            entity.decay_confidence(factor);
        }

        // Remove entities with very low confidence
        self.entities.retain(|_, e| e.confidence > 0.01);
        self.update_stats();
    }

    /// Get entities sorted by confidence
    pub fn entities_by_confidence(&self) -> Vec<&Entity> {
        let mut entities: Vec<_> = self.entities.values().collect();
        entities.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        entities
    }

    /// Get entities with positive valence
    pub fn positive_entities(&self) -> Vec<&Entity> {
        self.entities
            .values()
            .filter(|e| e.valence.value() > 0.0)
            .collect()
    }

    /// Get entities with negative valence
    pub fn negative_entities(&self) -> Vec<&Entity> {
        self.entities
            .values()
            .filter(|e| e.valence.value() < 0.0)
            .collect()
    }

    /// Get statistics
    pub fn stats(&self) -> &WorldModelStats {
        &self.stats
    }

    /// Get all entities
    pub fn entities(&self) -> impl Iterator<Item = &Entity> {
        self.entities.values()
    }

    /// Get all relationships
    pub fn relationships(&self) -> &[Relationship] {
        &self.relationships
    }

    /// Get pending predictions
    pub fn pending_predictions(&self) -> &[WorldPrediction] {
        &self.predictions
    }

    /// Get prediction history
    pub fn prediction_history(&self) -> &[WorldPrediction] {
        &self.prediction_history
    }

    /// Update statistics
    fn update_stats(&mut self) {
        self.stats.total_entities = self.entities.len();
        self.stats.total_relationships = self.relationships.len();
        self.stats.total_predictions = self.predictions.len() + self.prediction_history.len();
        self.stats.evaluated_predictions = self.prediction_history.len();
        self.stats.prediction_accuracy = self.prediction_accuracy();

        if !self.entities.is_empty() {
            self.stats.average_confidence =
                self.entities.values().map(|e| e.confidence).sum::<f64>()
                    / self.entities.len() as f64;
        }
    }

    /// Clear all data
    pub fn clear(&mut self) {
        self.entities.clear();
        self.relationships.clear();
        self.predictions.clear();
        self.prediction_history.clear();
        self.update_stats();
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_world_model_creation() {
        let model = WorldModel::new();
        assert_eq!(model.stats().total_entities, 0);
    }

    #[test]
    fn test_add_entity() {
        let mut model = WorldModel::new();

        let entity = Entity::new("person", "Alice")
            .with_property("age", 30.0)
            .with_property("active", true);

        let id = model.add_entity(entity);

        assert!(model.get_entity(id).is_some());
        assert_eq!(model.stats().total_entities, 1);
    }

    #[test]
    fn test_entity_properties() {
        let mut model = WorldModel::new();

        let entity = Entity::new("item", "laptop")
            .with_property("price", 1200.0)
            .with_property("brand", "Apple");

        let id = model.add_entity(entity);

        let e = model.get_entity(id).unwrap();
        assert_eq!(
            e.get_property("price"),
            Some(&PropertyValue::Number(1200.0))
        );
        assert_eq!(
            e.get_property("brand"),
            Some(&PropertyValue::String("Apple".to_string()))
        );
    }

    #[test]
    fn test_update_property() {
        let mut model = WorldModel::new();
        let entity = Entity::new("sensor", "thermometer").with_property("temperature", 20.0);
        let id = model.add_entity(entity);

        model.update_entity_property(id, "temperature", 25.0);

        let e = model.get_entity(id).unwrap();
        assert_eq!(
            e.get_property("temperature"),
            Some(&PropertyValue::Number(25.0))
        );
    }

    #[test]
    fn test_find_by_type() {
        let mut model = WorldModel::new();

        model.add_entity(Entity::new("person", "Alice"));
        model.add_entity(Entity::new("person", "Bob"));
        model.add_entity(Entity::new("item", "Chair"));

        let people = model.find_by_type("person");
        assert_eq!(people.len(), 2);
    }

    #[test]
    fn test_relationships() {
        let mut model = WorldModel::new();

        let alice = model.add_entity(Entity::new("person", "Alice"));
        let bob = model.add_entity(Entity::new("person", "Bob"));

        model.add_relationship(Relationship::new(alice, bob, RelationType::Knows));

        let alice_rels = model.get_outgoing(alice);
        assert_eq!(alice_rels.len(), 1);
        assert_eq!(alice_rels[0].relation_type, RelationType::Knows);
    }

    #[test]
    fn test_related_entities() {
        let mut model = WorldModel::new();

        let room = model.add_entity(Entity::new("room", "Living Room"));
        let chair = model.add_entity(Entity::new("furniture", "Chair"));
        let table = model.add_entity(Entity::new("furniture", "Table"));

        model.add_relationship(Relationship::new(room, chair, RelationType::Contains));
        model.add_relationship(Relationship::new(room, table, RelationType::Contains));

        let related = model.related_entities(room);
        assert_eq!(related.len(), 2);
    }

    #[test]
    fn test_prediction() {
        let mut model = WorldModel::new();
        let entity = Entity::new("stock", "AAPL").with_property("price", 150.0);
        let id = model.add_entity(entity);

        let prediction = WorldPrediction::new("AAPL will be $160", PropertyValue::Number(160.0))
            .for_entity(id, "price")
            .with_confidence(0.7);

        model.predict(prediction);
        assert_eq!(model.pending_predictions().len(), 1);
    }

    #[test]
    fn test_prediction_evaluation() {
        let mut prediction = WorldPrediction::new("Test prediction", PropertyValue::Number(100.0));

        // Correct prediction (within 10%)
        prediction.evaluate(PropertyValue::Number(105.0));
        assert_eq!(prediction.correct, Some(true));

        // Wrong prediction
        let mut wrong_prediction =
            WorldPrediction::new("Test prediction", PropertyValue::Number(100.0));
        wrong_prediction.evaluate(PropertyValue::Number(200.0));
        assert_eq!(wrong_prediction.correct, Some(false));
    }

    #[test]
    fn test_prediction_error_recording() {
        let mut model = WorldModel::new();
        model.record_prediction_error("test_error", 0.4, 0.6);

        assert_eq!(model.prediction_history().len(), 1);
        assert_eq!(model.stats().evaluated_predictions, 1);
    }

    #[test]
    fn test_confidence_decay() {
        let mut model = WorldModel::new();
        let entity = Entity::new("temp", "temp").with_confidence(1.0);
        let id = model.add_entity(entity);

        model.decay_confidence(0.9);

        let e = model.get_entity(id).unwrap();
        assert!((e.confidence - 0.9).abs() < 0.01);
    }

    #[test]
    fn test_valence_filtering() {
        let mut model = WorldModel::new();

        model.add_entity(Entity::new("good", "Ice Cream").with_valence(0.8));
        model.add_entity(Entity::new("bad", "Taxes").with_valence(-0.5));
        model.add_entity(Entity::new("neutral", "Rock").with_valence(0.0));

        assert_eq!(model.positive_entities().len(), 1);
        assert_eq!(model.negative_entities().len(), 1);
    }

    #[test]
    fn test_remove_entity_removes_relationships() {
        let mut model = WorldModel::new();

        let a = model.add_entity(Entity::new("node", "A"));
        let b = model.add_entity(Entity::new("node", "B"));
        let c = model.add_entity(Entity::new("node", "C"));

        model.add_relationship(Relationship::new(a, b, RelationType::Near));
        model.add_relationship(Relationship::new(b, c, RelationType::Near));

        assert_eq!(model.relationships().len(), 2);

        model.remove_entity(b);

        // Both relationships involving B should be removed
        assert_eq!(model.relationships().len(), 0);
    }

    #[test]
    fn test_find_by_name() {
        let mut model = WorldModel::new();

        model.add_entity(Entity::new("person", "Alice Smith"));
        model.add_entity(Entity::new("person", "Bob Smith"));
        model.add_entity(Entity::new("person", "Charlie"));

        let smiths = model.find_by_name("smith");
        assert_eq!(smiths.len(), 2);
    }
}
