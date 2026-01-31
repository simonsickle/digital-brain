//! Communication Interface
//!
//! Structured way to generate outputs/responses based on brain state.
//! This module provides:
//! - Intent classification (inform, request, confirm, etc.)
//! - Emotional tone modulation
//! - Urgency-aware output generation
//! - Multi-turn conversation tracking
//!
//! Integration with neuromodulators:
//! - High stress → more urgent/direct communication
//! - High patience → more elaborate explanations
//! - High oxytocin → warmer, more cooperative tone
//! - Low dopamine → less enthusiastic responses

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use uuid::Uuid;

use crate::core::neuromodulators::NeuromodulatorState;
use crate::signal::Valence;

/// Type of communication intent
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IntentType {
    /// Providing information
    Inform,
    /// Asking for something
    Request,
    /// Confirming understanding
    Confirm,
    /// Asking for clarification
    Clarify,
    /// Expressing emotion/state
    Express,
    /// Greeting/social
    Greet,
    /// Acknowledging receipt
    Acknowledge,
    /// Apologizing
    Apologize,
    /// Thanking
    Thank,
    /// Warning/alerting
    Warn,
    /// Suggesting/recommending
    Suggest,
    /// Refusing/declining
    Refuse,
    /// Agreeing
    Agree,
    /// Disagreeing
    Disagree,
}

impl IntentType {
    /// Get base urgency for this intent type
    pub fn base_urgency(&self) -> f64 {
        match self {
            IntentType::Warn => 0.9,
            IntentType::Request => 0.6,
            IntentType::Clarify => 0.5,
            IntentType::Inform => 0.4,
            IntentType::Confirm => 0.4,
            IntentType::Suggest => 0.4,
            IntentType::Acknowledge => 0.3,
            IntentType::Express => 0.3,
            IntentType::Agree | IntentType::Disagree => 0.3,
            IntentType::Refuse => 0.4,
            IntentType::Apologize => 0.5,
            IntentType::Thank => 0.3,
            IntentType::Greet => 0.2,
        }
    }

    /// Get typical valence for this intent type
    pub fn typical_valence(&self) -> f64 {
        match self {
            IntentType::Thank => 0.6,
            IntentType::Greet => 0.4,
            IntentType::Agree => 0.3,
            IntentType::Acknowledge => 0.1,
            IntentType::Inform | IntentType::Confirm | IntentType::Clarify => 0.0,
            IntentType::Request | IntentType::Suggest => 0.0,
            IntentType::Express => 0.0, // Depends on content
            IntentType::Refuse | IntentType::Disagree => -0.2,
            IntentType::Apologize => -0.1,
            IntentType::Warn => -0.3,
        }
    }
}

/// Communication style preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationStyle {
    /// Formality level (0 = casual, 1 = formal)
    pub formality: f64,
    /// Verbosity (0 = terse, 1 = elaborate)
    pub verbosity: f64,
    /// Directness (0 = indirect/polite, 1 = direct/blunt)
    pub directness: f64,
    /// Warmth (0 = cold/professional, 1 = warm/friendly)
    pub warmth: f64,
    /// Confidence in assertions (0 = hedging, 1 = certain)
    pub confidence: f64,
}

impl Default for CommunicationStyle {
    fn default() -> Self {
        Self {
            formality: 0.5,
            verbosity: 0.5,
            directness: 0.5,
            warmth: 0.5,
            confidence: 0.5,
        }
    }
}

impl CommunicationStyle {
    /// Adjust style based on neuromodulator state
    pub fn modulate(&self, state: &NeuromodulatorState) -> Self {
        Self {
            // High stress → more direct
            directness: (self.directness + state.stress * 0.3).min(1.0),
            // High patience → more verbose/elaborate
            verbosity: (self.verbosity + state.patience * 0.2).min(1.0),
            // High oxytocin → warmer
            warmth: (self.warmth + state.cooperativeness * 0.3).min(1.0),
            // Mood stability affects confidence
            confidence: (self.confidence + state.mood_stability * 0.2).min(1.0),
            // Formality stays relatively stable
            formality: self.formality,
        }
    }

    /// Create a casual style
    pub fn casual() -> Self {
        Self {
            formality: 0.2,
            verbosity: 0.4,
            directness: 0.6,
            warmth: 0.7,
            confidence: 0.5,
        }
    }

    /// Create a formal style
    pub fn formal() -> Self {
        Self {
            formality: 0.9,
            verbosity: 0.6,
            directness: 0.4,
            warmth: 0.3,
            confidence: 0.7,
        }
    }

    /// Create a supportive style
    pub fn supportive() -> Self {
        Self {
            formality: 0.3,
            verbosity: 0.6,
            directness: 0.3,
            warmth: 0.9,
            confidence: 0.5,
        }
    }

    /// Create a concise style
    pub fn concise() -> Self {
        Self {
            formality: 0.5,
            verbosity: 0.2,
            directness: 0.8,
            warmth: 0.4,
            confidence: 0.6,
        }
    }
}

/// A communication intent (what the agent wants to say)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationIntent {
    /// Unique identifier
    pub id: Uuid,
    /// Main content
    pub content: String,
    /// Type of intent
    pub intent_type: IntentType,
    /// Urgency level (0-1)
    pub urgency: f64,
    /// Emotional tone
    pub valence: Valence,
    /// Target audience (if applicable)
    pub target: Option<String>,
    /// Reply to (for conversational context)
    pub reply_to: Option<Uuid>,
    /// When created
    pub created_at: DateTime<Utc>,
    /// Additional context
    pub context: Vec<String>,
    /// Priority (higher = more important to send)
    pub priority: f64,
    /// Has this been sent?
    pub sent: bool,
}

impl CommunicationIntent {
    /// Create a new communication intent
    pub fn new(content: &str, intent_type: IntentType) -> Self {
        Self {
            id: Uuid::new_v4(),
            content: content.to_string(),
            intent_type,
            urgency: intent_type.base_urgency(),
            valence: Valence::new(intent_type.typical_valence()),
            target: None,
            reply_to: None,
            created_at: Utc::now(),
            context: Vec::new(),
            priority: 0.5,
            sent: false,
        }
    }

    /// Set urgency
    pub fn with_urgency(mut self, urgency: f64) -> Self {
        self.urgency = urgency.clamp(0.0, 1.0);
        self
    }

    /// Set valence
    pub fn with_valence(mut self, valence: f64) -> Self {
        self.valence = Valence::new(valence);
        self
    }

    /// Set target
    pub fn to(mut self, target: &str) -> Self {
        self.target = Some(target.to_string());
        self
    }

    /// Set as reply
    pub fn replying_to(mut self, message_id: Uuid) -> Self {
        self.reply_to = Some(message_id);
        self
    }

    /// Add context
    pub fn with_context(mut self, context: &str) -> Self {
        self.context.push(context.to_string());
        self
    }

    /// Set priority
    pub fn with_priority(mut self, priority: f64) -> Self {
        self.priority = priority.clamp(0.0, 1.0);
        self
    }

    /// Mark as sent
    pub fn mark_sent(&mut self) {
        self.sent = true;
    }

    /// Calculate effective priority based on urgency and age
    pub fn effective_priority(&self) -> f64 {
        let age_secs = (Utc::now() - self.created_at).num_seconds() as f64;
        let age_boost = (age_secs / 60.0).min(0.3); // Max 0.3 boost after 1 minute

        (self.priority + self.urgency * 0.3 + age_boost).min(1.0)
    }
}

/// A message in conversation history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationMessage {
    /// Unique identifier
    pub id: Uuid,
    /// Who sent it (self, user, system)
    pub sender: String,
    /// Content
    pub content: String,
    /// When sent
    pub timestamp: DateTime<Utc>,
    /// Intent type (for our messages)
    pub intent_type: Option<IntentType>,
    /// Valence
    pub valence: Valence,
}

impl ConversationMessage {
    pub fn from_self(content: &str, intent_type: IntentType) -> Self {
        Self {
            id: Uuid::new_v4(),
            sender: "self".to_string(),
            content: content.to_string(),
            timestamp: Utc::now(),
            intent_type: Some(intent_type),
            valence: Valence::new(intent_type.typical_valence()),
        }
    }

    pub fn from_user(content: &str) -> Self {
        Self {
            id: Uuid::new_v4(),
            sender: "user".to_string(),
            content: content.to_string(),
            timestamp: Utc::now(),
            intent_type: None,
            valence: Valence::new(0.0),
        }
    }

    pub fn from_system(content: &str) -> Self {
        Self {
            id: Uuid::new_v4(),
            sender: "system".to_string(),
            content: content.to_string(),
            timestamp: Utc::now(),
            intent_type: None,
            valence: Valence::new(0.0),
        }
    }
}

/// Statistics about communication
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CommunicationStats {
    pub total_messages_sent: u64,
    pub total_messages_received: u64,
    pub messages_by_intent: std::collections::HashMap<String, u64>,
    pub average_response_time_ms: f64,
    pub average_message_length: f64,
}

/// The communication system
#[derive(Debug)]
pub struct CommunicationSystem {
    /// Output buffer (pending communications)
    output_buffer: Vec<CommunicationIntent>,
    /// Maximum buffer size
    max_buffer: usize,
    /// Default style
    style: CommunicationStyle,
    /// Conversation history
    conversation: VecDeque<ConversationMessage>,
    /// Maximum conversation history
    max_history: usize,
    /// Current conversation context
    current_context: Vec<String>,
    /// Statistics
    stats: CommunicationStats,
    /// Last response time tracking
    last_input_time: Option<DateTime<Utc>>,
}

impl Default for CommunicationSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl CommunicationSystem {
    /// Create a new communication system
    pub fn new() -> Self {
        Self {
            output_buffer: Vec::new(),
            max_buffer: 50,
            style: CommunicationStyle::default(),
            conversation: VecDeque::new(),
            max_history: 100,
            current_context: Vec::new(),
            stats: CommunicationStats::default(),
            last_input_time: None,
        }
    }

    /// Create with specific style
    pub fn with_style(style: CommunicationStyle) -> Self {
        Self {
            style,
            ..Self::new()
        }
    }

    /// Set communication style
    pub fn set_style(&mut self, style: CommunicationStyle) {
        self.style = style;
    }

    /// Get current style
    pub fn style(&self) -> &CommunicationStyle {
        &self.style
    }

    /// Get modulated style based on neuromodulator state
    pub fn modulated_style(&self, state: &NeuromodulatorState) -> CommunicationStyle {
        self.style.modulate(state)
    }

    /// Record an incoming message
    pub fn receive_message(&mut self, content: &str, sender: &str) {
        let message = ConversationMessage {
            id: Uuid::new_v4(),
            sender: sender.to_string(),
            content: content.to_string(),
            timestamp: Utc::now(),
            intent_type: None,
            valence: Valence::new(0.0),
        };

        self.conversation.push_back(message);
        self.stats.total_messages_received += 1;
        self.last_input_time = Some(Utc::now());

        // Trim history
        while self.conversation.len() > self.max_history {
            self.conversation.pop_front();
        }
    }

    /// Queue a communication intent
    pub fn queue(&mut self, intent: CommunicationIntent) {
        self.output_buffer.push(intent);

        // Trim buffer if too large (remove lowest priority)
        while self.output_buffer.len() > self.max_buffer {
            let min_idx = self
                .output_buffer
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    a.effective_priority()
                        .partial_cmp(&b.effective_priority())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i);

            if let Some(idx) = min_idx {
                self.output_buffer.remove(idx);
            }
        }
    }

    /// Generate a response based on brain state
    pub fn generate_response(
        &mut self,
        context: &str,
        intent_type: IntentType,
        state: &NeuromodulatorState,
    ) -> CommunicationIntent {
        let style = self.modulated_style(state);

        // Calculate urgency based on state
        let urgency =
            intent_type.base_urgency() + state.stress * 0.2 + (1.0 - state.patience) * 0.1;

        // Calculate valence based on state
        let valence = intent_type.typical_valence()
            + state.dopamine * 0.2  // Higher dopamine = more positive
            - state.frustration * 0.3; // Frustration dampens positivity

        // Priority based on motivation
        let priority = 0.5 + state.motivation * 0.3;

        let mut intent = CommunicationIntent::new(context, intent_type)
            .with_urgency(urgency.clamp(0.0, 1.0))
            .with_valence(valence)
            .with_priority(priority);

        // Add style context
        if style.formality > 0.7 {
            intent = intent.with_context("formal");
        } else if style.formality < 0.3 {
            intent = intent.with_context("casual");
        }

        if style.warmth > 0.7 {
            intent = intent.with_context("warm");
        }

        if style.directness > 0.7 {
            intent = intent.with_context("direct");
        }

        intent
    }

    /// Get the next message to send (highest priority)
    pub fn next_to_send(&mut self) -> Option<CommunicationIntent> {
        if self.output_buffer.is_empty() {
            return None;
        }

        // Find highest priority
        let max_idx = self
            .output_buffer
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.effective_priority()
                    .partial_cmp(&b.effective_priority())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)?;

        let mut intent = self.output_buffer.remove(max_idx);
        intent.mark_sent();

        // Record in conversation
        self.conversation.push_back(ConversationMessage::from_self(
            &intent.content,
            intent.intent_type,
        ));

        // Update stats
        self.stats.total_messages_sent += 1;
        let intent_name = format!("{:?}", intent.intent_type);
        *self
            .stats
            .messages_by_intent
            .entry(intent_name)
            .or_insert(0) += 1;

        // Update average message length
        let total = self.stats.total_messages_sent as f64;
        self.stats.average_message_length = self.stats.average_message_length
            * ((total - 1.0) / total)
            + intent.content.len() as f64 / total;

        // Update response time
        if let Some(input_time) = self.last_input_time {
            let response_time = (Utc::now() - input_time).num_milliseconds() as f64;
            self.stats.average_response_time_ms =
                self.stats.average_response_time_ms * 0.9 + response_time * 0.1;
        }

        Some(intent)
    }

    /// Peek at pending messages without removing
    pub fn pending(&self) -> &[CommunicationIntent] {
        &self.output_buffer
    }

    /// Get conversation history
    pub fn history(&self) -> impl Iterator<Item = &ConversationMessage> {
        self.conversation.iter()
    }

    /// Get recent conversation (last n messages)
    pub fn recent_history(&self, n: usize) -> Vec<&ConversationMessage> {
        self.conversation.iter().rev().take(n).collect()
    }

    /// Clear output buffer
    pub fn clear_buffer(&mut self) {
        self.output_buffer.clear();
    }

    /// Clear conversation history
    pub fn clear_history(&mut self) {
        self.conversation.clear();
    }

    /// Add context to current conversation
    pub fn add_context(&mut self, context: &str) {
        self.current_context.push(context.to_string());
        if self.current_context.len() > 10 {
            self.current_context.remove(0);
        }
    }

    /// Get current context
    pub fn context(&self) -> &[String] {
        &self.current_context
    }

    /// Get statistics
    pub fn stats(&self) -> &CommunicationStats {
        &self.stats
    }

    // Common intent helpers

    /// Acknowledge receipt
    pub fn acknowledge(&mut self, content: &str) {
        self.queue(CommunicationIntent::new(content, IntentType::Acknowledge));
    }

    /// Inform/share information
    pub fn inform(&mut self, content: &str) {
        self.queue(CommunicationIntent::new(content, IntentType::Inform));
    }

    /// Request something
    pub fn request(&mut self, content: &str) {
        self.queue(CommunicationIntent::new(content, IntentType::Request));
    }

    /// Ask for clarification
    pub fn clarify(&mut self, content: &str) {
        self.queue(CommunicationIntent::new(content, IntentType::Clarify));
    }

    /// Express emotion/state
    pub fn express(&mut self, content: &str, valence: f64) {
        self.queue(CommunicationIntent::new(content, IntentType::Express).with_valence(valence));
    }

    /// Warn about something
    pub fn warn(&mut self, content: &str) {
        self.queue(CommunicationIntent::new(content, IntentType::Warn).with_urgency(0.9));
    }

    /// Suggest something
    pub fn suggest(&mut self, content: &str) {
        self.queue(CommunicationIntent::new(content, IntentType::Suggest));
    }

    /// Thank
    pub fn thank(&mut self, content: &str) {
        self.queue(CommunicationIntent::new(content, IntentType::Thank).with_valence(0.6));
    }

    /// Apologize
    pub fn apologize(&mut self, content: &str) {
        self.queue(CommunicationIntent::new(content, IntentType::Apologize).with_valence(-0.1));
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_state() -> NeuromodulatorState {
        NeuromodulatorState {
            dopamine: 0.5,
            serotonin: 0.5,
            norepinephrine: 0.5,
            acetylcholine: 0.5,
            cortisol: 0.3,
            gaba: 0.5,
            oxytocin: 0.5,
            motivation: 0.5,
            patience: 0.5,
            stress: 0.3,
            learning_depth: 0.5,
            frustration: 0.2,
            exploration_drive: 0.3,
            impulse_control: 0.5,
            cooperativeness: 0.5,
            is_satiated: false,
            is_stressed: false,
            is_burned_out: false,
            is_deliberating: false,
            should_pivot: false,
            should_seek_help: false,
            prefer_cooperation: false,
            mood_stability: 0.7,
        }
    }

    #[test]
    fn test_communication_system_creation() {
        let comm = CommunicationSystem::new();
        assert_eq!(comm.stats().total_messages_sent, 0);
    }

    #[test]
    fn test_queue_intent() {
        let mut comm = CommunicationSystem::new();

        comm.queue(CommunicationIntent::new("Hello", IntentType::Greet));

        assert_eq!(comm.pending().len(), 1);
    }

    #[test]
    fn test_next_to_send() {
        let mut comm = CommunicationSystem::new();

        comm.queue(CommunicationIntent::new("Low priority", IntentType::Greet).with_priority(0.1));
        comm.queue(CommunicationIntent::new("High priority", IntentType::Warn).with_priority(0.9));

        let next = comm.next_to_send().unwrap();
        assert_eq!(next.content, "High priority");
    }

    #[test]
    fn test_receive_message() {
        let mut comm = CommunicationSystem::new();

        comm.receive_message("Hello there", "user");

        assert_eq!(comm.stats().total_messages_received, 1);
        assert_eq!(comm.history().count(), 1);
    }

    #[test]
    fn test_conversation_history() {
        let mut comm = CommunicationSystem::new();

        comm.receive_message("Hello", "user");
        comm.queue(CommunicationIntent::new("Hi!", IntentType::Greet));
        comm.next_to_send();

        let history: Vec<_> = comm.history().collect();
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].sender, "user");
        assert_eq!(history[1].sender, "self");
    }

    #[test]
    fn test_style_modulation() {
        let style = CommunicationStyle::default();

        let mut high_stress = make_test_state();
        high_stress.stress = 0.9;

        let modulated = style.modulate(&high_stress);
        assert!(modulated.directness > style.directness);
    }

    #[test]
    fn test_generate_response() {
        let mut comm = CommunicationSystem::new();
        let state = make_test_state();

        let intent = comm.generate_response("Test response", IntentType::Inform, &state);

        assert_eq!(intent.intent_type, IntentType::Inform);
        assert!(!intent.content.is_empty());
    }

    #[test]
    fn test_intent_types() {
        assert!(IntentType::Warn.base_urgency() > IntentType::Greet.base_urgency());
        assert!(IntentType::Thank.typical_valence() > IntentType::Apologize.typical_valence());
    }

    #[test]
    fn test_effective_priority() {
        let intent = CommunicationIntent::new("Test", IntentType::Inform)
            .with_priority(0.5)
            .with_urgency(0.8);

        let priority = intent.effective_priority();
        assert!(priority > 0.5); // Should be boosted by urgency
    }

    #[test]
    fn test_helper_methods() {
        let mut comm = CommunicationSystem::new();

        comm.inform("Info");
        comm.request("Request");
        comm.warn("Warning");

        assert_eq!(comm.pending().len(), 3);

        // Warn should have highest urgency
        let next = comm.next_to_send().unwrap();
        assert_eq!(next.intent_type, IntentType::Warn);
    }

    #[test]
    fn test_style_presets() {
        let casual = CommunicationStyle::casual();
        let formal = CommunicationStyle::formal();

        assert!(casual.formality < formal.formality);
        assert!(casual.warmth > formal.warmth);
    }

    #[test]
    fn test_context() {
        let mut comm = CommunicationSystem::new();

        comm.add_context("topic: coding");
        comm.add_context("mood: focused");

        assert_eq!(comm.context().len(), 2);
    }

    #[test]
    fn test_buffer_overflow() {
        let mut comm = CommunicationSystem::new();
        comm.max_buffer = 3;

        for i in 0..5 {
            comm.queue(
                CommunicationIntent::new(&format!("Message {}", i), IntentType::Inform)
                    .with_priority(i as f64 * 0.1),
            );
        }

        // Should only keep 3 highest priority
        assert_eq!(comm.pending().len(), 3);
    }
}
