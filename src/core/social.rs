//! Social Cognition - Theory of Mind, Reputation, and Hierarchy Tracking
//!
//! Maintains social models of other agents, infers intent/emotion,
//! and tracks reputation/hierarchy signals over time.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Summary of a social agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SocialAgentProfile {
    pub agent_id: String,
    pub reputation: f64,
    pub dominance: f64,
    pub warmth: f64,
    pub competence: f64,
    pub interaction_count: u64,
    pub last_interaction: DateTime<Utc>,
}

impl SocialAgentProfile {
    pub fn new(agent_id: &str) -> Self {
        Self {
            agent_id: agent_id.to_string(),
            reputation: 0.5,
            dominance: 0.5,
            warmth: 0.5,
            competence: 0.5,
            interaction_count: 0,
            last_interaction: Utc::now(),
        }
    }

    pub fn update_from_interaction(&mut self, positive: bool, intensity: f64) {
        let delta = intensity.clamp(0.0, 1.0) * if positive { 0.1 } else { -0.15 };
        self.reputation = (self.reputation + delta).clamp(0.0, 1.0);
        self.warmth = (self.warmth + delta * 0.8).clamp(0.0, 1.0);
        let dominance_delta = intensity.clamp(0.0, 1.0) * if positive { 0.04 } else { -0.03 };
        self.dominance = (self.dominance + dominance_delta).clamp(0.0, 1.0);
        let competence_delta = intensity.clamp(0.0, 1.0) * if positive { 0.05 } else { -0.05 };
        self.competence = (self.competence + competence_delta).clamp(0.0, 1.0);
        self.interaction_count += 1;
        self.last_interaction = Utc::now();
    }
}

/// Inferred theory-of-mind state for an agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TheoryOfMindState {
    pub agent_id: String,
    pub inferred_intent: String,
    pub inferred_emotion: String,
    pub confidence: f64,
    pub evidence: Vec<String>,
}

/// Result of a social update.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SocialUpdate {
    pub agent_id: String,
    pub new_reputation: f64,
    pub notes: Vec<String>,
}

/// Social cognition system.
#[derive(Debug, Default)]
pub struct SocialCognition {
    agents: HashMap<String, SocialAgentProfile>,
}

impl SocialCognition {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn profile(&mut self, agent_id: &str) -> &mut SocialAgentProfile {
        self.agents
            .entry(agent_id.to_string())
            .or_insert_with(|| SocialAgentProfile::new(agent_id))
    }

    pub fn observe_interaction(
        &mut self,
        agent_id: &str,
        positive: bool,
        intensity: f64,
    ) -> SocialUpdate {
        let profile = self.profile(agent_id);
        profile.update_from_interaction(positive, intensity);

        let mut notes = Vec::new();
        if positive {
            notes.push("positive_interaction".to_string());
        } else {
            notes.push("negative_interaction".to_string());
        }

        SocialUpdate {
            agent_id: agent_id.to_string(),
            new_reputation: profile.reputation,
            notes,
        }
    }

    pub fn profile_snapshot(&self, agent_id: &str) -> Option<SocialAgentProfile> {
        self.agents.get(agent_id).cloned()
    }

    pub fn infer_theory_of_mind(&mut self, agent_id: &str, cue: &str) -> TheoryOfMindState {
        let cue_lower = cue.to_lowercase();
        let profile = self.profile(agent_id);

        let (mut intent, mut emotion, confidence, mut evidence) = if cue_lower.contains("help")
            || cue_lower.contains("assist")
            || cue_lower.contains("support")
        {
            (
                "cooperative".to_string(),
                "friendly".to_string(),
                0.7,
                vec!["help_cue".to_string()],
            )
        } else if cue_lower.contains("threat")
            || cue_lower.contains("warn")
            || cue_lower.contains("attack")
        {
            (
                "defensive".to_string(),
                "threatened".to_string(),
                0.75,
                vec!["threat_cue".to_string()],
            )
        } else if cue_lower.contains("ask") || cue_lower.contains("question") {
            (
                "curious".to_string(),
                "uncertain".to_string(),
                0.6,
                vec!["question_cue".to_string()],
            )
        } else {
            (
                "neutral".to_string(),
                "calm".to_string(),
                0.4,
                vec!["baseline".to_string()],
            )
        };

        let reputation_bias = (profile.reputation - 0.5) * 0.2;
        let warmth_bias = (profile.warmth - 0.5) * 0.2;
        let dominance_bias = (profile.dominance - 0.5) * 0.15;

        if intent == "neutral" && evidence.iter().any(|e| e == "baseline") {
            if warmth_bias > 0.08 {
                intent = "cooperative".to_string();
                emotion = "friendly".to_string();
                evidence.push("warmth_bias".to_string());
            } else if warmth_bias < -0.08 {
                intent = "guarded".to_string();
                emotion = "wary".to_string();
                evidence.push("low_warmth_bias".to_string());
            }
        }
        if dominance_bias > 0.1 {
            evidence.push("dominance_bias".to_string());
        }

        TheoryOfMindState {
            agent_id: agent_id.to_string(),
            inferred_intent: intent,
            inferred_emotion: emotion,
            confidence: (confidence + reputation_bias + warmth_bias * 0.5 + dominance_bias * 0.3)
                .clamp(0.0, 1.0),
            evidence,
        }
    }

    pub fn hierarchy(&self) -> Vec<(String, f64)> {
        let mut rankings: Vec<(String, f64)> = self
            .agents
            .values()
            .map(|agent| {
                let score = agent.dominance * 0.6 + agent.competence * 0.4;
                (agent.agent_id.clone(), score)
            })
            .collect();
        rankings.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        rankings
    }

    pub fn reputation(&self, agent_id: &str) -> Option<f64> {
        self.agents.get(agent_id).map(|profile| profile.reputation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracks_reputation_changes() {
        let mut social = SocialCognition::new();
        let update = social.observe_interaction("agent_a", true, 0.8);
        assert!(update.new_reputation > 0.5);
    }

    #[test]
    fn infers_theory_of_mind_from_cues() {
        let mut social = SocialCognition::new();
        let inference = social.infer_theory_of_mind("agent_b", "Can you help me?");
        assert_eq!(inference.inferred_intent, "cooperative");
        assert!(inference.confidence > 0.5);
    }
}
