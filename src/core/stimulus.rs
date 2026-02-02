//! Stimulus Types - All possible inputs to the consciousness loop
//!
//! A stimulus is any event that can trigger processing. This includes:
//! - External prompts (human input)
//! - File system events
//! - Time-based events
//! - Internal drives (curiosity, boredom)
//! - Goal-related triggers
//! - System events

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use uuid::Uuid;

/// Priority level for stimulus processing
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default,
)]
pub enum StimulusPriority {
    /// Background processing, can be deferred
    Background = 0,
    /// Normal priority, process when convenient
    #[default]
    Normal = 1,
    /// Elevated priority, should process soon
    Elevated = 2,
    /// High priority, interrupt current activity
    High = 3,
    /// Critical, must process immediately
    Critical = 4,
}

/// The source of a stimulus
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StimulusSource {
    /// External human input
    Human { identity: Option<String> },
    /// File system event
    FileSystem { path: PathBuf },
    /// Time/scheduler
    Clock,
    /// Internal drive (curiosity, boredom, etc.)
    InternalDrive { drive: String },
    /// Goal system
    GoalSystem { goal_id: Option<Uuid> },
    /// Another brain region
    InternalRegion { region: String },
    /// External API/network
    External { source: String },
    /// System/environment
    System,
}

/// File system event types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FileEvent {
    /// File was created
    Created { path: PathBuf },
    /// File was modified
    Modified { path: PathBuf },
    /// File was deleted
    Deleted { path: PathBuf },
    /// File was renamed
    Renamed { from: PathBuf, to: PathBuf },
    /// Directory contents changed
    DirectoryChanged { path: PathBuf },
}

/// Time-based event types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimeEvent {
    /// Regular tick (heartbeat)
    Tick { cycle: u64 },
    /// Specific time reached
    Alarm { name: String },
    /// Duration elapsed since something
    Elapsed { since: DateTime<Utc>, event: String },
    /// Idle timeout (no activity for duration)
    IdleTimeout { idle_seconds: u64 },
}

/// Internal drive events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DriveEvent {
    /// Curiosity urge - want to explore something
    Curiosity {
        domain: String,
        intensity: f64,
        target: Option<String>,
    },
    /// Boredom trigger - stuck in a rut
    Boredom { level: f64, recommendation: String },
    /// Goal pressure - something needs attention
    GoalPressure {
        goal_id: Uuid,
        urgency: f64,
        reason: String,
    },
    /// Consolidation need - time to organize memories
    ConsolidationNeed {
        pending_memories: usize,
        last_consolidation: Option<DateTime<Utc>>,
    },
    /// Rest need - resources depleted
    RestNeed { fatigue_level: f64 },
    /// Social need - want interaction
    SocialNeed {
        isolation_duration: std::time::Duration,
    },
}

/// Goal-related events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GoalEvent {
    /// Deadline approaching
    DeadlineApproaching {
        goal_id: Uuid,
        time_remaining: std::time::Duration,
    },
    /// Goal completed
    Completed { goal_id: Uuid },
    /// Goal blocked
    Blocked { goal_id: Uuid, reason: String },
    /// New subgoal generated
    SubgoalGenerated { parent_id: Uuid, child_id: Uuid },
    /// Progress made
    Progress { goal_id: Uuid, delta: f64 },
}

/// System events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SystemEvent {
    /// Resource warning (disk, memory, etc.)
    ResourceWarning { resource: String, level: f64 },
    /// Error occurred
    Error { message: String, recoverable: bool },
    /// Configuration changed
    ConfigChanged { key: String },
    /// Startup
    Startup,
    /// Shutdown requested
    ShutdownRequested,
}

/// The main stimulus enum - all possible inputs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StimulusKind {
    /// External prompt from human
    ExternalPrompt {
        content: String,
        context: Option<String>,
    },

    /// File system event
    FileSystem(FileEvent),

    /// Time-based event
    Time(TimeEvent),

    /// Internal drive event
    Drive(DriveEvent),

    /// Goal-related event
    Goal(GoalEvent),

    /// System event
    System(SystemEvent),

    /// Internal thought (from DMN, reflection, etc.)
    InternalThought {
        content: String,
        source_region: String,
    },

    /// Query response (async result came back)
    QueryResponse {
        query_id: Uuid,
        result: serde_json::Value,
    },

    /// Sensory observation (noticed something)
    Observation {
        domain: String,
        content: String,
        novelty: f64,
    },
}

/// A complete stimulus with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stimulus {
    /// Unique identifier
    pub id: Uuid,
    /// When the stimulus occurred
    pub timestamp: DateTime<Utc>,
    /// The stimulus content
    pub kind: StimulusKind,
    /// Priority level
    pub priority: StimulusPriority,
    /// Source of the stimulus
    pub source: StimulusSource,
    /// Estimated salience (0-1)
    pub salience: f64,
    /// Tags for filtering/routing
    pub tags: Vec<String>,
    /// Whether this requires a response
    pub requires_response: bool,
    /// Deadline for processing (if any)
    pub deadline: Option<DateTime<Utc>>,
}

impl Stimulus {
    /// Create a new stimulus
    pub fn new(kind: StimulusKind, source: StimulusSource) -> Self {
        Self {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            kind,
            priority: StimulusPriority::default(),
            source,
            salience: 0.5,
            tags: Vec::new(),
            requires_response: false,
            deadline: None,
        }
    }

    /// Create from external prompt
    pub fn from_prompt(content: impl Into<String>, identity: Option<String>) -> Self {
        Self::new(
            StimulusKind::ExternalPrompt {
                content: content.into(),
                context: None,
            },
            StimulusSource::Human { identity },
        )
        .with_priority(StimulusPriority::High)
        .with_requires_response(true)
    }

    /// Create from file event
    pub fn from_file_event(event: FileEvent) -> Self {
        let path = match &event {
            FileEvent::Created { path } => path.clone(),
            FileEvent::Modified { path } => path.clone(),
            FileEvent::Deleted { path } => path.clone(),
            FileEvent::Renamed { from, .. } => from.clone(),
            FileEvent::DirectoryChanged { path } => path.clone(),
        };

        Self::new(
            StimulusKind::FileSystem(event),
            StimulusSource::FileSystem { path },
        )
    }

    /// Create from time event
    pub fn from_time_event(event: TimeEvent) -> Self {
        Self::new(StimulusKind::Time(event), StimulusSource::Clock)
    }

    /// Create from drive event
    pub fn from_drive(event: DriveEvent, drive_name: &str) -> Self {
        let priority = match &event {
            DriveEvent::Boredom { level, .. } if *level > 0.8 => StimulusPriority::Elevated,
            DriveEvent::GoalPressure { urgency, .. } if *urgency > 0.8 => StimulusPriority::High,
            DriveEvent::RestNeed { fatigue_level } if *fatigue_level > 0.9 => {
                StimulusPriority::High
            }
            _ => StimulusPriority::Normal,
        };

        Self::new(
            StimulusKind::Drive(event),
            StimulusSource::InternalDrive {
                drive: drive_name.to_string(),
            },
        )
        .with_priority(priority)
    }

    /// Create internal thought
    pub fn from_thought(content: impl Into<String>, source_region: impl Into<String>) -> Self {
        let region = source_region.into();
        Self::new(
            StimulusKind::InternalThought {
                content: content.into(),
                source_region: region.clone(),
            },
            StimulusSource::InternalRegion { region },
        )
        .with_priority(StimulusPriority::Background)
    }

    /// Builder: set priority
    pub fn with_priority(mut self, priority: StimulusPriority) -> Self {
        self.priority = priority;
        self
    }

    /// Builder: set salience
    pub fn with_salience(mut self, salience: f64) -> Self {
        self.salience = salience.clamp(0.0, 1.0);
        self
    }

    /// Builder: add tag
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Builder: set requires_response
    pub fn with_requires_response(mut self, requires: bool) -> Self {
        self.requires_response = requires;
        self
    }

    /// Builder: set deadline
    pub fn with_deadline(mut self, deadline: DateTime<Utc>) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// Is this stimulus from an external source?
    pub fn is_external(&self) -> bool {
        matches!(
            self.source,
            StimulusSource::Human { .. }
                | StimulusSource::FileSystem { .. }
                | StimulusSource::External { .. }
        )
    }

    /// Is this stimulus from an internal source?
    pub fn is_internal(&self) -> bool {
        !self.is_external()
    }

    /// Get age of stimulus
    pub fn age(&self) -> chrono::Duration {
        Utc::now() - self.timestamp
    }

    /// Is this stimulus expired (past deadline)?
    pub fn is_expired(&self) -> bool {
        self.deadline.map(|d| Utc::now() > d).unwrap_or(false)
    }

    /// Should this interrupt current processing?
    pub fn should_interrupt(&self, current_priority: StimulusPriority) -> bool {
        self.priority > current_priority
            || (self.priority == StimulusPriority::Critical)
            || (self.requires_response && self.priority >= StimulusPriority::High)
    }
}

/// Response to a stimulus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StimulusResponse {
    /// ID of the stimulus this responds to
    pub stimulus_id: Uuid,
    /// Response content (if any)
    pub content: Option<String>,
    /// Actions taken
    pub actions: Vec<String>,
    /// Was the stimulus fully processed?
    pub complete: bool,
    /// Should be deferred for later?
    pub defer: bool,
    /// Follow-up stimuli generated
    pub follow_ups: Vec<Stimulus>,
    /// Processing duration
    pub processing_time: std::time::Duration,
}

impl StimulusResponse {
    /// Create a simple response
    pub fn simple(stimulus_id: Uuid, content: Option<String>) -> Self {
        Self {
            stimulus_id,
            content,
            actions: Vec::new(),
            complete: true,
            defer: false,
            follow_ups: Vec::new(),
            processing_time: std::time::Duration::ZERO,
        }
    }

    /// Create a deferred response
    pub fn deferred(stimulus_id: Uuid) -> Self {
        Self {
            stimulus_id,
            content: None,
            actions: Vec::new(),
            complete: false,
            defer: true,
            follow_ups: Vec::new(),
            processing_time: std::time::Duration::ZERO,
        }
    }

    /// Add an action
    pub fn with_action(mut self, action: impl Into<String>) -> Self {
        self.actions.push(action.into());
        self
    }

    /// Add a follow-up stimulus
    pub fn with_follow_up(mut self, stimulus: Stimulus) -> Self {
        self.follow_ups.push(stimulus);
        self
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stimulus_creation() {
        let stimulus = Stimulus::from_prompt("Hello", Some("user".to_string()));
        assert!(stimulus.requires_response);
        assert_eq!(stimulus.priority, StimulusPriority::High);
        assert!(stimulus.is_external());
    }

    #[test]
    fn test_file_stimulus() {
        let event = FileEvent::Modified {
            path: PathBuf::from("/tmp/test.txt"),
        };
        let stimulus = Stimulus::from_file_event(event);
        assert!(stimulus.is_external());
        assert!(!stimulus.requires_response);
    }

    #[test]
    fn test_drive_stimulus() {
        let event = DriveEvent::Curiosity {
            domain: "coding".to_string(),
            intensity: 0.8,
            target: Some("rust".to_string()),
        };
        let stimulus = Stimulus::from_drive(event, "curiosity");
        assert!(stimulus.is_internal());
    }

    #[test]
    fn test_priority_ordering() {
        assert!(StimulusPriority::Critical > StimulusPriority::High);
        assert!(StimulusPriority::High > StimulusPriority::Elevated);
        assert!(StimulusPriority::Elevated > StimulusPriority::Normal);
        assert!(StimulusPriority::Normal > StimulusPriority::Background);
    }

    #[test]
    fn test_interrupt_logic() {
        let critical =
            Stimulus::from_prompt("urgent", None).with_priority(StimulusPriority::Critical);

        let normal =
            Stimulus::from_thought("thinking", "dmn").with_priority(StimulusPriority::Normal);

        assert!(critical.should_interrupt(StimulusPriority::Normal));
        assert!(critical.should_interrupt(StimulusPriority::High));
        assert!(!normal.should_interrupt(StimulusPriority::Elevated));
    }

    #[test]
    fn test_internal_thought() {
        let thought = Stimulus::from_thought("I wonder about X", "dmn");
        assert!(thought.is_internal());
        assert_eq!(thought.priority, StimulusPriority::Background);
    }
}
