//! Multi-Agent Interaction
//!
//! Support for multiple agents communicating and collaborating:
//! - Agent registry and discovery
//! - Inter-agent messaging
//! - Shared knowledge/beliefs
//! - Theory of mind (modeling other agents)
//!
//! This enables:
//! - Agent-to-agent communication
//! - Collaborative problem solving
//! - Social dynamics simulation

use std::collections::HashMap;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::signal::Valence;

/// Unique identifier for an agent
pub type AgentId = Uuid;

/// Profile of an agent (how they present themselves)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentProfile {
    /// Unique identifier
    pub id: AgentId,
    /// Display name
    pub name: String,
    /// Description/bio
    pub description: String,
    /// Agent's stated values/goals
    pub values: Vec<String>,
    /// Capabilities the agent claims
    pub capabilities: Vec<String>,
    /// When the profile was created
    pub created_at: DateTime<Utc>,
    /// Last seen/active
    pub last_active: DateTime<Utc>,
    /// Trust level (from our perspective)
    pub trust_level: f64,
    /// How much we like them (valence)
    pub affinity: Valence,
    /// Tags for categorization
    pub tags: Vec<String>,
}

impl AgentProfile {
    pub fn new(name: &str) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            name: name.to_string(),
            description: String::new(),
            values: Vec::new(),
            capabilities: Vec::new(),
            created_at: now,
            last_active: now,
            trust_level: 0.5, // Neutral trust by default
            affinity: Valence::new(0.0),
            tags: Vec::new(),
        }
    }

    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    pub fn with_value(mut self, value: &str) -> Self {
        self.values.push(value.to_string());
        self
    }

    pub fn with_capability(mut self, cap: &str) -> Self {
        self.capabilities.push(cap.to_string());
        self
    }

    pub fn with_trust(mut self, trust: f64) -> Self {
        self.trust_level = trust.clamp(0.0, 1.0);
        self
    }

    pub fn with_affinity(mut self, affinity: f64) -> Self {
        self.affinity = Valence::new(affinity);
        self
    }

    pub fn touch(&mut self) {
        self.last_active = Utc::now();
    }
}

/// Type of message between agents
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MessageType {
    /// General information sharing
    Inform,
    /// Request for action/information
    Request,
    /// Response to a request
    Response,
    /// Proposal for collaboration
    Propose,
    /// Accepting a proposal
    Accept,
    /// Rejecting a proposal
    Reject,
    /// Query/question
    Query,
    /// Greeting/social
    Greet,
    /// Acknowledgment
    Acknowledge,
    /// Broadcast to all
    Broadcast,
}

/// A message between agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMessage {
    /// Unique identifier
    pub id: Uuid,
    /// Sender agent
    pub from: AgentId,
    /// Recipient agent (None = broadcast)
    pub to: Option<AgentId>,
    /// Message type
    pub message_type: MessageType,
    /// Content
    pub content: String,
    /// In reply to (for conversations)
    pub reply_to: Option<Uuid>,
    /// When sent
    pub sent_at: DateTime<Utc>,
    /// Priority (0-1)
    pub priority: f64,
    /// Emotional tone
    pub valence: Valence,
    /// Metadata
    pub metadata: HashMap<String, String>,
    /// Has this been read?
    pub read: bool,
}

impl AgentMessage {
    pub fn new(from: AgentId, to: Option<AgentId>, message_type: MessageType, content: &str) -> Self {
        Self {
            id: Uuid::new_v4(),
            from,
            to,
            message_type,
            content: content.to_string(),
            reply_to: None,
            sent_at: Utc::now(),
            priority: 0.5,
            valence: Valence::new(0.0),
            metadata: HashMap::new(),
            read: false,
        }
    }

    pub fn broadcast(from: AgentId, content: &str) -> Self {
        Self::new(from, None, MessageType::Broadcast, content)
    }

    pub fn inform(from: AgentId, to: AgentId, content: &str) -> Self {
        Self::new(from, Some(to), MessageType::Inform, content)
    }

    pub fn request(from: AgentId, to: AgentId, content: &str) -> Self {
        Self::new(from, Some(to), MessageType::Request, content)
    }

    pub fn query(from: AgentId, to: AgentId, content: &str) -> Self {
        Self::new(from, Some(to), MessageType::Query, content)
    }

    pub fn respond(from: AgentId, to: AgentId, content: &str, reply_to: Uuid) -> Self {
        let mut msg = Self::new(from, Some(to), MessageType::Response, content);
        msg.reply_to = Some(reply_to);
        msg
    }

    pub fn with_priority(mut self, priority: f64) -> Self {
        self.priority = priority.clamp(0.0, 1.0);
        self
    }

    pub fn with_valence(mut self, valence: f64) -> Self {
        self.valence = Valence::new(valence);
        self
    }

    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    pub fn mark_read(&mut self) {
        self.read = true;
    }
}

/// Model of another agent (theory of mind)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentModel {
    /// The agent we're modeling
    pub agent_id: AgentId,
    /// What we believe about their goals
    pub believed_goals: Vec<String>,
    /// What we believe about their beliefs
    pub believed_beliefs: Vec<String>,
    /// What we believe about their capabilities
    pub believed_capabilities: Vec<String>,
    /// Predicted behavior patterns
    pub behavior_patterns: Vec<String>,
    /// Trust history
    pub trust_history: Vec<(DateTime<Utc>, f64)>,
    /// Interaction history summary
    pub interaction_count: u64,
    /// Positive interactions
    pub positive_interactions: u64,
    /// Negative interactions
    pub negative_interactions: u64,
    /// Last updated
    pub updated_at: DateTime<Utc>,
}

impl AgentModel {
    pub fn new(agent_id: AgentId) -> Self {
        Self {
            agent_id,
            believed_goals: Vec::new(),
            believed_beliefs: Vec::new(),
            believed_capabilities: Vec::new(),
            behavior_patterns: Vec::new(),
            trust_history: Vec::new(),
            interaction_count: 0,
            positive_interactions: 0,
            negative_interactions: 0,
            updated_at: Utc::now(),
        }
    }

    /// Record an interaction
    pub fn record_interaction(&mut self, positive: bool) {
        self.interaction_count += 1;
        if positive {
            self.positive_interactions += 1;
        } else {
            self.negative_interactions += 1;
        }
        self.updated_at = Utc::now();
    }

    /// Calculate cooperation score
    pub fn cooperation_score(&self) -> f64 {
        if self.interaction_count == 0 {
            return 0.5;
        }
        self.positive_interactions as f64 / self.interaction_count as f64
    }

    /// Update trust
    pub fn update_trust(&mut self, new_trust: f64) {
        self.trust_history.push((Utc::now(), new_trust.clamp(0.0, 1.0)));
        // Keep last 100 entries
        if self.trust_history.len() > 100 {
            self.trust_history.remove(0);
        }
    }

    /// Get current trust (latest value)
    pub fn current_trust(&self) -> f64 {
        self.trust_history
            .last()
            .map(|(_, t)| *t)
            .unwrap_or(0.5)
    }
}

/// Statistics about multi-agent interactions
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MultiAgentStats {
    pub known_agents: usize,
    pub total_messages_sent: u64,
    pub total_messages_received: u64,
    pub active_conversations: usize,
    pub average_trust: f64,
}

/// The multi-agent registry and messaging system
#[derive(Debug)]
pub struct MultiAgentSystem {
    /// Our own profile
    our_profile: AgentProfile,
    /// Known agents
    agents: HashMap<AgentId, AgentProfile>,
    /// Models of other agents (theory of mind)
    agent_models: HashMap<AgentId, AgentModel>,
    /// Message inbox
    inbox: Vec<AgentMessage>,
    /// Message outbox (pending)
    outbox: Vec<AgentMessage>,
    /// Sent messages
    sent: Vec<AgentMessage>,
    /// Maximum inbox size
    max_inbox: usize,
    /// Statistics
    stats: MultiAgentStats,
}

impl MultiAgentSystem {
    /// Create a new multi-agent system with our profile
    pub fn new(profile: AgentProfile) -> Self {
        Self {
            our_profile: profile,
            agents: HashMap::new(),
            agent_models: HashMap::new(),
            inbox: Vec::new(),
            outbox: Vec::new(),
            sent: Vec::new(),
            max_inbox: 1000,
            stats: MultiAgentStats::default(),
        }
    }

    /// Get our agent ID
    pub fn our_id(&self) -> AgentId {
        self.our_profile.id
    }

    /// Get our profile
    pub fn our_profile(&self) -> &AgentProfile {
        &self.our_profile
    }

    /// Update our profile
    pub fn update_profile(&mut self, profile: AgentProfile) {
        self.our_profile = profile;
    }

    /// Register a known agent
    pub fn register_agent(&mut self, profile: AgentProfile) {
        let id = profile.id;
        self.agents.insert(id, profile);
        self.agent_models.entry(id).or_insert_with(|| AgentModel::new(id));
        self.update_stats();
    }

    /// Get an agent's profile
    pub fn get_agent(&self, id: AgentId) -> Option<&AgentProfile> {
        self.agents.get(&id)
    }

    /// Get a mutable agent profile
    pub fn get_agent_mut(&mut self, id: AgentId) -> Option<&mut AgentProfile> {
        self.agents.get_mut(&id)
    }

    /// Get our model of an agent
    pub fn get_model(&self, id: AgentId) -> Option<&AgentModel> {
        self.agent_models.get(&id)
    }

    /// Get mutable model of an agent
    pub fn get_model_mut(&mut self, id: AgentId) -> Option<&mut AgentModel> {
        self.agent_models.get_mut(&id)
    }

    /// Find agents by capability
    pub fn find_by_capability(&self, capability: &str) -> Vec<&AgentProfile> {
        self.agents
            .values()
            .filter(|a| a.capabilities.iter().any(|c| c.contains(capability)))
            .collect()
    }

    /// Find agents by trust level
    pub fn find_trusted(&self, min_trust: f64) -> Vec<&AgentProfile> {
        self.agents
            .values()
            .filter(|a| a.trust_level >= min_trust)
            .collect()
    }

    /// Queue a message to send
    pub fn send(&mut self, message: AgentMessage) {
        self.outbox.push(message);
    }

    /// Send an inform message
    pub fn inform(&mut self, to: AgentId, content: &str) {
        self.send(AgentMessage::inform(self.our_id(), to, content));
    }

    /// Send a request message
    pub fn request(&mut self, to: AgentId, content: &str) {
        self.send(AgentMessage::request(self.our_id(), to, content));
    }

    /// Send a query message
    pub fn query(&mut self, to: AgentId, content: &str) {
        self.send(AgentMessage::query(self.our_id(), to, content));
    }

    /// Broadcast to all known agents
    pub fn broadcast(&mut self, content: &str) {
        self.send(AgentMessage::broadcast(self.our_id(), content));
    }

    /// Process outbox (send pending messages)
    pub fn flush_outbox(&mut self) -> Vec<AgentMessage> {
        let messages: Vec<_> = self.outbox.drain(..).collect();
        self.stats.total_messages_sent += messages.len() as u64;

        for msg in &messages {
            self.sent.push(msg.clone());
        }

        // Trim sent history
        while self.sent.len() > self.max_inbox {
            self.sent.remove(0);
        }

        messages
    }

    /// Receive a message
    pub fn receive(&mut self, message: AgentMessage) {
        // Update sender's last_active
        if let Some(sender) = self.agents.get_mut(&message.from) {
            sender.touch();
        }

        self.inbox.push(message);
        self.stats.total_messages_received += 1;

        // Trim inbox
        while self.inbox.len() > self.max_inbox {
            self.inbox.remove(0);
        }

        self.update_stats();
    }

    /// Get unread messages
    pub fn unread_messages(&self) -> Vec<&AgentMessage> {
        self.inbox.iter().filter(|m| !m.read).collect()
    }

    /// Get messages from a specific agent
    pub fn messages_from(&self, agent_id: AgentId) -> Vec<&AgentMessage> {
        self.inbox.iter().filter(|m| m.from == agent_id).collect()
    }

    /// Mark a message as read
    pub fn mark_read(&mut self, message_id: Uuid) {
        if let Some(msg) = self.inbox.iter_mut().find(|m| m.id == message_id) {
            msg.mark_read();
        }
    }

    /// Record an interaction with an agent
    pub fn record_interaction(&mut self, agent_id: AgentId, positive: bool) {
        if let Some(model) = self.agent_models.get_mut(&agent_id) {
            model.record_interaction(positive);
        }

        // Update trust based on interaction
        if let Some(agent) = self.agents.get_mut(&agent_id) {
            let delta = if positive { 0.05 } else { -0.1 };
            agent.trust_level = (agent.trust_level + delta).clamp(0.0, 1.0);
        }
    }

    /// Update our model of an agent's goals
    pub fn update_believed_goals(&mut self, agent_id: AgentId, goals: Vec<String>) {
        if let Some(model) = self.agent_models.get_mut(&agent_id) {
            model.believed_goals = goals;
            model.updated_at = Utc::now();
        }
    }

    /// Get statistics
    pub fn stats(&self) -> &MultiAgentStats {
        &self.stats
    }

    /// Get all known agents
    pub fn known_agents(&self) -> impl Iterator<Item = &AgentProfile> {
        self.agents.values()
    }

    /// Get inbox
    pub fn inbox(&self) -> &[AgentMessage] {
        &self.inbox
    }

    /// Clear read messages
    pub fn clear_read(&mut self) {
        self.inbox.retain(|m| !m.read);
    }

    /// Update statistics
    fn update_stats(&mut self) {
        self.stats.known_agents = self.agents.len();

        if !self.agents.is_empty() {
            self.stats.average_trust =
                self.agents.values().map(|a| a.trust_level).sum::<f64>() / self.agents.len() as f64;
        }

        self.stats.active_conversations = self
            .inbox
            .iter()
            .filter(|m| !m.read)
            .map(|m| m.from)
            .collect::<std::collections::HashSet<_>>()
            .len();
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_profile() {
        let profile = AgentProfile::new("TestAgent")
            .with_description("A test agent")
            .with_capability("coding")
            .with_trust(0.8);

        assert_eq!(profile.name, "TestAgent");
        assert_eq!(profile.trust_level, 0.8);
        assert!(profile.capabilities.contains(&"coding".to_string()));
    }

    #[test]
    fn test_multi_agent_system() {
        let our_profile = AgentProfile::new("Us");
        let mut system = MultiAgentSystem::new(our_profile);

        let other = AgentProfile::new("Other").with_capability("research");
        let other_id = other.id;
        system.register_agent(other);

        assert_eq!(system.stats().known_agents, 1);
        assert!(system.get_agent(other_id).is_some());
    }

    #[test]
    fn test_messaging() {
        let our_profile = AgentProfile::new("Us");
        let our_id = our_profile.id;
        let mut system = MultiAgentSystem::new(our_profile);

        let other = AgentProfile::new("Other");
        let other_id = other.id;
        system.register_agent(other);

        // Send message
        system.inform(other_id, "Hello!");

        let messages = system.flush_outbox();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].from, our_id);
        assert_eq!(messages[0].to, Some(other_id));
    }

    #[test]
    fn test_receive_message() {
        let our_profile = AgentProfile::new("Us");
        let mut system = MultiAgentSystem::new(our_profile);

        let other = AgentProfile::new("Other");
        let other_id = other.id;
        system.register_agent(other);

        // Receive message
        let msg = AgentMessage::inform(other_id, system.our_id(), "Hi there!");
        system.receive(msg);

        assert_eq!(system.unread_messages().len(), 1);
        assert_eq!(system.stats().total_messages_received, 1);
    }

    #[test]
    fn test_agent_model() {
        let mut model = AgentModel::new(Uuid::new_v4());

        model.record_interaction(true);
        model.record_interaction(true);
        model.record_interaction(false);

        assert_eq!(model.interaction_count, 3);
        assert!((model.cooperation_score() - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_trust_update() {
        let our_profile = AgentProfile::new("Us");
        let mut system = MultiAgentSystem::new(our_profile);

        let other = AgentProfile::new("Other").with_trust(0.5);
        let other_id = other.id;
        system.register_agent(other);

        // Positive interaction
        system.record_interaction(other_id, true);
        let trust = system.get_agent(other_id).unwrap().trust_level;
        assert!(trust > 0.5);

        // Negative interaction
        system.record_interaction(other_id, false);
        let trust_after = system.get_agent(other_id).unwrap().trust_level;
        assert!(trust_after < trust);
    }

    #[test]
    fn test_find_by_capability() {
        let our_profile = AgentProfile::new("Us");
        let mut system = MultiAgentSystem::new(our_profile);

        system.register_agent(AgentProfile::new("Coder").with_capability("coding"));
        system.register_agent(AgentProfile::new("Researcher").with_capability("research"));
        system.register_agent(AgentProfile::new("Writer").with_capability("writing"));

        let coders = system.find_by_capability("coding");
        assert_eq!(coders.len(), 1);
        assert_eq!(coders[0].name, "Coder");
    }

    #[test]
    fn test_broadcast() {
        let our_profile = AgentProfile::new("Us");
        let mut system = MultiAgentSystem::new(our_profile);

        system.broadcast("Hello everyone!");

        let messages = system.flush_outbox();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].message_type, MessageType::Broadcast);
        assert!(messages[0].to.is_none());
    }

    #[test]
    fn test_message_types() {
        let from = Uuid::new_v4();
        let to = Uuid::new_v4();

        let inform = AgentMessage::inform(from, to, "info");
        assert_eq!(inform.message_type, MessageType::Inform);

        let request = AgentMessage::request(from, to, "please");
        assert_eq!(request.message_type, MessageType::Request);

        let query = AgentMessage::query(from, to, "what?");
        assert_eq!(query.message_type, MessageType::Query);
    }

    #[test]
    fn test_mark_read() {
        let our_profile = AgentProfile::new("Us");
        let mut system = MultiAgentSystem::new(our_profile);

        let other_id = Uuid::new_v4();
        let msg = AgentMessage::inform(other_id, system.our_id(), "Test");
        let msg_id = msg.id;
        system.receive(msg);

        assert_eq!(system.unread_messages().len(), 1);

        system.mark_read(msg_id);

        assert_eq!(system.unread_messages().len(), 0);
    }
}
