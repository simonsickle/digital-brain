//! Task Scheduler - Priority Queues and Attention Focus Management
//!
//! This module provides algorithm-based task scheduling for the digital brain's
//! attention system. Key features:
//!
//! - **Priority Queues**: Multiple priority levels with configurable policies
//! - **Scheduling Algorithms**: EDF (Earliest Deadline First), Fixed Priority,
//!   Weighted scoring combining urgency, priority, and salience
//! - **Persistence**: SQLite-backed storage survives daemon restarts
//! - **Pubsub Notifications**: Event-driven task state changes
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                     TASK SCHEDULER                               │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
//! │  │   CRITICAL  │  │    HIGH     │  │         NORMAL          │ │
//! │  │   QUEUE     │  │   QUEUE     │  │    (Medium + Low)       │ │
//! │  └──────┬──────┘  └──────┬──────┘  └───────────┬─────────────┘ │
//! │         │                │                     │               │
//! │         └────────────────┼─────────────────────┘               │
//! │                          ▼                                      │
//! │              ┌───────────────────────┐                          │
//! │              │   SCHEDULING POLICY   │                          │
//! │              │  (EDF/Priority/Weighted)│                        │
//! │              └───────────┬───────────┘                          │
//! │                          ▼                                      │
//! │              ┌───────────────────────┐                          │
//! │              │    NEXT TASK          │──→ BrainSignal           │
//! │              └───────────────────────┘                          │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                    PUBSUB EVENTS                                 │
//! │    TaskReady │ TaskStarted │ TaskCompleted │ TaskBlocked        │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Theoretical Foundation
//!
//! Based on:
//! - Real-time scheduling theory (EDF optimality for periodic tasks)
//! - Attention as a limited resource (capacity constraints)
//! - Urgency × Importance prioritization (Eisenhower matrix)

use chrono::{DateTime, Duration, Utc};
use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use uuid::Uuid;

use crate::error::{BrainError, Result};
use crate::signal::{BrainSignal, SignalType};

/// Unique identifier for a scheduled task
pub type TaskId = Uuid;

/// Unique identifier for a subscription
pub type SubscriptionId = Uuid;

// ============================================================================
// TASK PRIORITY
// ============================================================================

/// Priority level for tasks (matches goals system)
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
pub enum TaskPriority {
    /// Background/nice-to-have tasks
    Low = 1,
    /// Normal priority
    #[default]
    Medium = 2,
    /// Important tasks
    High = 3,
    /// Urgent/critical tasks - preempts all others
    Critical = 4,
}

impl TaskPriority {
    /// Convert to numeric weight for scoring
    pub fn weight(&self) -> f64 {
        match self {
            TaskPriority::Low => 0.25,
            TaskPriority::Medium => 0.5,
            TaskPriority::High => 0.75,
            TaskPriority::Critical => 1.0,
        }
    }

    /// Parse from integer (for database storage)
    pub fn from_i32(value: i32) -> Self {
        match value {
            1 => TaskPriority::Low,
            2 => TaskPriority::Medium,
            3 => TaskPriority::High,
            4 => TaskPriority::Critical,
            _ => TaskPriority::Medium,
        }
    }

    /// Convert to integer (for database storage)
    pub fn to_i32(&self) -> i32 {
        *self as i32
    }
}

// ============================================================================
// TASK STATE
// ============================================================================

/// Current state of a scheduled task
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskState {
    /// Task is waiting to be scheduled
    Pending,
    /// Task is ready to execute (all dependencies met)
    Ready,
    /// Task is currently executing
    Running,
    /// Task is blocked on a condition
    Blocked { reason: String },
    /// Task completed successfully
    Completed { completed_at: DateTime<Utc> },
    /// Task failed
    Failed {
        error: String,
        failed_at: DateTime<Utc>,
    },
    /// Task was cancelled
    Cancelled {
        reason: String,
        cancelled_at: DateTime<Utc>,
    },
    /// Task expired (deadline passed)
    Expired { expired_at: DateTime<Utc> },
}

impl TaskState {
    /// Is this task actionable (can be scheduled)?
    pub fn is_actionable(&self) -> bool {
        matches!(self, TaskState::Pending | TaskState::Ready)
    }

    /// Is this a terminal state?
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            TaskState::Completed { .. }
                | TaskState::Failed { .. }
                | TaskState::Cancelled { .. }
                | TaskState::Expired { .. }
        )
    }

    /// Convert to string for database storage
    pub fn to_db_string(&self) -> String {
        match self {
            TaskState::Pending => "pending".to_string(),
            TaskState::Ready => "ready".to_string(),
            TaskState::Running => "running".to_string(),
            TaskState::Blocked { reason } => format!("blocked:{}", reason),
            TaskState::Completed { .. } => "completed".to_string(),
            TaskState::Failed { error, .. } => format!("failed:{}", error),
            TaskState::Cancelled { reason, .. } => format!("cancelled:{}", reason),
            TaskState::Expired { .. } => "expired".to_string(),
        }
    }

    /// Parse from database string
    pub fn from_db_string(s: &str) -> Self {
        if s == "pending" {
            TaskState::Pending
        } else if s == "ready" {
            TaskState::Ready
        } else if s == "running" {
            TaskState::Running
        } else if s == "completed" {
            TaskState::Completed {
                completed_at: Utc::now(),
            }
        } else if s == "expired" {
            TaskState::Expired {
                expired_at: Utc::now(),
            }
        } else if let Some(reason) = s.strip_prefix("blocked:") {
            TaskState::Blocked {
                reason: reason.to_string(),
            }
        } else if let Some(error) = s.strip_prefix("failed:") {
            TaskState::Failed {
                error: error.to_string(),
                failed_at: Utc::now(),
            }
        } else if let Some(reason) = s.strip_prefix("cancelled:") {
            TaskState::Cancelled {
                reason: reason.to_string(),
                cancelled_at: Utc::now(),
            }
        } else {
            TaskState::Pending
        }
    }
}

// ============================================================================
// SCHEDULED TASK
// ============================================================================

/// A task scheduled for execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduledTask {
    /// Unique identifier
    pub id: TaskId,
    /// Human-readable name
    pub name: String,
    /// Detailed description
    pub description: String,
    /// Priority level
    pub priority: TaskPriority,
    /// Current state
    pub state: TaskState,
    /// Optional deadline
    pub deadline: Option<DateTime<Utc>>,
    /// When to start (earliest start time)
    pub scheduled_at: Option<DateTime<Utc>>,
    /// When the task was created
    pub created_at: DateTime<Utc>,
    /// When the task was last updated
    pub updated_at: DateTime<Utc>,
    /// Estimated duration in milliseconds
    pub estimated_duration_ms: u64,
    /// Salience (attention-grabbing level, 0-1)
    pub salience: f64,
    /// Associated goal ID (if any)
    pub goal_id: Option<String>,
    /// Tags for categorization
    pub tags: Vec<String>,
    /// Dependencies (task IDs that must complete first)
    pub dependencies: Vec<TaskId>,
    /// Number of retry attempts
    pub retry_count: u32,
    /// Maximum retries allowed
    pub max_retries: u32,
    /// Arbitrary metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl ScheduledTask {
    /// Create a new scheduled task
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            description: description.into(),
            priority: TaskPriority::default(),
            state: TaskState::Pending,
            deadline: None,
            scheduled_at: None,
            created_at: now,
            updated_at: now,
            estimated_duration_ms: 0,
            salience: 0.5,
            goal_id: None,
            tags: Vec::new(),
            dependencies: Vec::new(),
            retry_count: 0,
            max_retries: 3,
            metadata: HashMap::new(),
        }
    }

    /// Set priority
    pub fn with_priority(mut self, priority: TaskPriority) -> Self {
        self.priority = priority;
        self
    }

    /// Set deadline
    pub fn with_deadline(mut self, deadline: DateTime<Utc>) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// Set scheduled start time
    pub fn with_scheduled_at(mut self, scheduled_at: DateTime<Utc>) -> Self {
        self.scheduled_at = Some(scheduled_at);
        self
    }

    /// Set estimated duration
    pub fn with_duration(mut self, duration_ms: u64) -> Self {
        self.estimated_duration_ms = duration_ms;
        self
    }

    /// Set salience
    pub fn with_salience(mut self, salience: f64) -> Self {
        self.salience = salience.clamp(0.0, 1.0);
        self
    }

    /// Set goal ID
    pub fn with_goal(mut self, goal_id: impl Into<String>) -> Self {
        self.goal_id = Some(goal_id.into());
        self
    }

    /// Add a tag
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Add a dependency
    pub fn with_dependency(mut self, task_id: TaskId) -> Self {
        self.dependencies.push(task_id);
        self
    }

    /// Set metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Serialize) -> Self {
        if let Ok(v) = serde_json::to_value(value) {
            self.metadata.insert(key.into(), v);
        }
        self
    }

    /// Calculate urgency score based on deadline proximity
    pub fn urgency(&self) -> f64 {
        match self.deadline {
            None => self.priority.weight(),
            Some(deadline) => {
                let now = Utc::now();
                if now >= deadline {
                    1.0 // Overdue = maximum urgency
                } else {
                    let time_remaining = (deadline - now).num_seconds() as f64;
                    let time_since_creation = (now - self.created_at).num_seconds() as f64;
                    let total_time = time_since_creation + time_remaining;

                    if total_time <= 0.0 {
                        return self.priority.weight();
                    }

                    // Urgency increases as deadline approaches
                    let time_pressure = 1.0 - (time_remaining / total_time);
                    (self.priority.weight() + time_pressure) / 2.0
                }
            }
        }
    }

    /// Check if task is overdue
    pub fn is_overdue(&self) -> bool {
        self.deadline.map(|d| Utc::now() > d).unwrap_or(false)
    }

    /// Check if task is ready to execute (scheduled time passed)
    pub fn is_ready_to_start(&self) -> bool {
        match self.scheduled_at {
            None => true,
            Some(scheduled) => Utc::now() >= scheduled,
        }
    }

    /// Mark task as ready
    pub fn mark_ready(&mut self) {
        self.state = TaskState::Ready;
        self.updated_at = Utc::now();
    }

    /// Mark task as running
    pub fn mark_running(&mut self) {
        self.state = TaskState::Running;
        self.updated_at = Utc::now();
    }

    /// Mark task as completed
    pub fn mark_completed(&mut self) {
        self.state = TaskState::Completed {
            completed_at: Utc::now(),
        };
        self.updated_at = Utc::now();
    }

    /// Mark task as failed
    pub fn mark_failed(&mut self, error: impl Into<String>) {
        self.state = TaskState::Failed {
            error: error.into(),
            failed_at: Utc::now(),
        };
        self.updated_at = Utc::now();
    }

    /// Mark task as blocked
    pub fn mark_blocked(&mut self, reason: impl Into<String>) {
        self.state = TaskState::Blocked {
            reason: reason.into(),
        };
        self.updated_at = Utc::now();
    }

    /// Mark task as cancelled
    pub fn mark_cancelled(&mut self, reason: impl Into<String>) {
        self.state = TaskState::Cancelled {
            reason: reason.into(),
            cancelled_at: Utc::now(),
        };
        self.updated_at = Utc::now();
    }

    /// Mark task as expired
    pub fn mark_expired(&mut self) {
        self.state = TaskState::Expired {
            expired_at: Utc::now(),
        };
        self.updated_at = Utc::now();
    }

    /// Convert to BrainSignal for attention system integration
    pub fn to_signal(&self) -> BrainSignal {
        BrainSignal::new("scheduler", SignalType::Attention, &self.name)
            .with_salience(self.salience)
            .with_priority(self.priority.to_i32())
            .with_metadata("task_id", self.id.to_string())
            .with_metadata("urgency", self.urgency())
            .with_metadata("deadline", self.deadline.map(|d| d.to_rfc3339()))
    }
}

// ============================================================================
// SCHEDULING POLICY
// ============================================================================

/// Scheduling policy determines how tasks are ordered
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum SchedulingPolicy {
    /// Earliest Deadline First - optimal for meeting deadlines
    EarliestDeadlineFirst,
    /// Fixed priority - simple priority-based ordering
    #[default]
    FixedPriority,
    /// Weighted scoring combining priority, urgency, and salience
    WeightedScore,
    /// FIFO within priority levels
    FifoByPriority,
    /// Round-robin across priority levels
    RoundRobin,
}

impl SchedulingPolicy {
    /// Calculate task score based on policy
    pub fn score(&self, task: &ScheduledTask) -> f64 {
        match self {
            SchedulingPolicy::EarliestDeadlineFirst => {
                match task.deadline {
                    Some(deadline) => {
                        let now = Utc::now();
                        let seconds_until = (deadline - now).num_seconds() as f64;
                        // Lower score = sooner deadline = higher priority
                        // Add priority as tiebreaker (inverted so higher priority wins)
                        -seconds_until + (1.0 - task.priority.weight()) * 1000.0
                    }
                    None => {
                        // No deadline = lowest priority in EDF
                        f64::MIN + task.priority.weight()
                    }
                }
            }
            SchedulingPolicy::FixedPriority => {
                // Simple priority ordering
                task.priority.weight() * 1000.0
            }
            SchedulingPolicy::WeightedScore => {
                // Combine multiple factors
                let priority_score = task.priority.weight() * 0.4;
                let urgency_score = task.urgency() * 0.3;
                let salience_score = task.salience * 0.2;
                let age_score = {
                    let age_hours = (Utc::now() - task.created_at).num_hours() as f64;
                    (age_hours / 24.0).min(1.0) * 0.1 // Older tasks get slight boost
                };
                (priority_score + urgency_score + salience_score + age_score) * 1000.0
            }
            SchedulingPolicy::FifoByPriority => {
                // Priority first, then creation time
                let priority_component = task.priority.weight() * 1_000_000.0;
                let age_component = -task.created_at.timestamp() as f64;
                priority_component + age_component / 1_000_000.0
            }
            SchedulingPolicy::RoundRobin => {
                // Just use age for round-robin (will be handled specially)
                -task.created_at.timestamp() as f64
            }
        }
    }
}

// ============================================================================
// TASK EVENTS (PUBSUB)
// ============================================================================

/// Events emitted by the scheduler
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskEvent {
    /// Task was added to the scheduler
    TaskAdded { task_id: TaskId, name: String },
    /// Task is ready to execute
    TaskReady { task_id: TaskId, name: String },
    /// Task started executing
    TaskStarted { task_id: TaskId, name: String },
    /// Task completed successfully
    TaskCompleted { task_id: TaskId, name: String },
    /// Task failed
    TaskFailed {
        task_id: TaskId,
        name: String,
        error: String,
    },
    /// Task was blocked
    TaskBlocked {
        task_id: TaskId,
        name: String,
        reason: String,
    },
    /// Task was unblocked
    TaskUnblocked { task_id: TaskId, name: String },
    /// Task was cancelled
    TaskCancelled {
        task_id: TaskId,
        name: String,
        reason: String,
    },
    /// Task expired (deadline passed)
    TaskExpired { task_id: TaskId, name: String },
    /// Task deadline approaching
    DeadlineApproaching {
        task_id: TaskId,
        name: String,
        hours_remaining: f64,
    },
    /// Scheduler state changed
    SchedulerStateChanged {
        running_count: usize,
        pending_count: usize,
    },
}

impl TaskEvent {
    /// Get the task ID associated with this event
    pub fn task_id(&self) -> Option<TaskId> {
        match self {
            TaskEvent::TaskAdded { task_id, .. }
            | TaskEvent::TaskReady { task_id, .. }
            | TaskEvent::TaskStarted { task_id, .. }
            | TaskEvent::TaskCompleted { task_id, .. }
            | TaskEvent::TaskFailed { task_id, .. }
            | TaskEvent::TaskBlocked { task_id, .. }
            | TaskEvent::TaskUnblocked { task_id, .. }
            | TaskEvent::TaskCancelled { task_id, .. }
            | TaskEvent::TaskExpired { task_id, .. }
            | TaskEvent::DeadlineApproaching { task_id, .. } => Some(*task_id),
            TaskEvent::SchedulerStateChanged { .. } => None,
        }
    }

    /// Convert to BrainSignal for broadcast
    pub fn to_signal(&self) -> BrainSignal {
        let (content, salience) = match self {
            TaskEvent::TaskReady { name, .. } => (format!("Task ready: {}", name), 0.7),
            TaskEvent::TaskStarted { name, .. } => (format!("Task started: {}", name), 0.5),
            TaskEvent::TaskCompleted { name, .. } => (format!("Task completed: {}", name), 0.6),
            TaskEvent::TaskFailed { name, error, .. } => {
                (format!("Task failed: {} - {}", name, error), 0.9)
            }
            TaskEvent::TaskBlocked { name, reason, .. } => {
                (format!("Task blocked: {} - {}", name, reason), 0.6)
            }
            TaskEvent::DeadlineApproaching {
                name,
                hours_remaining,
                ..
            } => (
                format!("Deadline approaching: {} ({:.1}h)", name, hours_remaining),
                0.8,
            ),
            TaskEvent::TaskExpired { name, .. } => (format!("Task expired: {}", name), 0.9),
            _ => (format!("{:?}", self), 0.4),
        };

        BrainSignal::new("scheduler", SignalType::Attention, content).with_salience(salience)
    }
}

/// Callback type for event subscribers
pub type EventCallback = Arc<dyn Fn(&TaskEvent) + Send + Sync>;

/// A subscription to scheduler events
pub struct Subscription {
    pub id: SubscriptionId,
    pub filter: Option<Vec<TaskId>>, // None = all tasks
    pub callback: EventCallback,
}

// ============================================================================
// PRIORITY QUEUE ENTRY
// ============================================================================

/// Entry in the priority queue for ordering
#[derive(Debug, Clone)]
struct QueueEntry {
    task_id: TaskId,
    score: f64,
}

impl PartialEq for QueueEntry {
    fn eq(&self, other: &Self) -> bool {
        self.task_id == other.task_id
    }
}

impl Eq for QueueEntry {}

impl PartialOrd for QueueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QueueEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Higher score = higher priority (reverse for BinaryHeap which is max-heap)
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

// ============================================================================
// SCHEDULER STATISTICS
// ============================================================================

/// Statistics about the scheduler
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SchedulerStats {
    pub total_tasks: u64,
    pub pending_tasks: u64,
    pub running_tasks: u64,
    pub completed_tasks: u64,
    pub failed_tasks: u64,
    pub cancelled_tasks: u64,
    pub expired_tasks: u64,
    pub average_wait_time_ms: f64,
    pub average_execution_time_ms: f64,
    pub throughput_per_hour: f64,
}

// ============================================================================
// SCHEDULER CONFIGURATION
// ============================================================================

/// Configuration for the task scheduler
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    /// Maximum concurrent running tasks
    pub max_concurrent: usize,
    /// Default scheduling policy
    pub policy: SchedulingPolicy,
    /// How often to check for deadline warnings (seconds)
    pub deadline_check_interval_secs: u64,
    /// Hours before deadline to warn
    pub deadline_warning_hours: f64,
    /// Maximum events to keep in history
    pub max_event_history: usize,
    /// Auto-expire tasks past deadline
    pub auto_expire: bool,
    /// Auto-retry failed tasks
    pub auto_retry: bool,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_concurrent: 3,
            policy: SchedulingPolicy::default(),
            deadline_check_interval_secs: 60,
            deadline_warning_hours: 24.0,
            max_event_history: 1000,
            auto_expire: true,
            auto_retry: true,
        }
    }
}

// ============================================================================
// TASK SCHEDULER (IN-MEMORY)
// ============================================================================

/// In-memory task scheduler with priority queues
pub struct TaskScheduler {
    /// All tasks by ID
    tasks: HashMap<TaskId, ScheduledTask>,
    /// Priority queue for scheduling
    queue: BinaryHeap<QueueEntry>,
    /// Configuration
    config: SchedulerConfig,
    /// Event subscribers
    subscribers: Vec<Subscription>,
    /// Event history
    event_history: VecDeque<TaskEvent>,
    /// Statistics
    stats: SchedulerStats,
    /// Completed task IDs (for dependency tracking)
    completed_tasks: Vec<TaskId>,
    /// Last deadline check
    last_deadline_check: DateTime<Utc>,
    /// Round-robin index for RoundRobin policy
    round_robin_index: usize,
}

impl TaskScheduler {
    /// Create a new in-memory scheduler
    pub fn new() -> Self {
        Self::with_config(SchedulerConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: SchedulerConfig) -> Self {
        Self {
            tasks: HashMap::new(),
            queue: BinaryHeap::new(),
            config,
            subscribers: Vec::new(),
            event_history: VecDeque::new(),
            stats: SchedulerStats::default(),
            completed_tasks: Vec::new(),
            last_deadline_check: Utc::now(),
            round_robin_index: 0,
        }
    }

    /// Set scheduling policy
    pub fn set_policy(&mut self, policy: SchedulingPolicy) {
        self.config.policy = policy;
        self.rebuild_queue();
    }

    /// Add a task to the scheduler
    pub fn schedule(&mut self, task: ScheduledTask) -> TaskId {
        let task_id = task.id;
        let name = task.name.clone();

        // Check if dependencies are met
        let deps_met = task
            .dependencies
            .iter()
            .all(|dep| self.completed_tasks.contains(dep));

        let mut task = task;
        if deps_met && task.is_ready_to_start() {
            task.mark_ready();
        }

        // Add to queue with score
        let score = self.config.policy.score(&task);
        self.queue.push(QueueEntry { task_id, score });

        self.tasks.insert(task_id, task);
        self.stats.total_tasks += 1;
        self.stats.pending_tasks += 1;

        self.emit_event(TaskEvent::TaskAdded { task_id, name });

        task_id
    }

    /// Get the next task to execute based on scheduling policy
    pub fn next_task(&mut self) -> Option<&ScheduledTask> {
        // Check for expired/deadline tasks first
        self.check_deadlines();

        // Handle round-robin specially
        if self.config.policy == SchedulingPolicy::RoundRobin {
            return self.next_round_robin();
        }

        // Rebuild queue to ensure scores are current
        self.rebuild_queue();

        // Find the highest-scoring ready task
        let ready_tasks: Vec<_> = self
            .tasks
            .values()
            .filter(|t| t.state.is_actionable() && t.is_ready_to_start())
            .filter(|t| {
                t.dependencies
                    .iter()
                    .all(|dep| self.completed_tasks.contains(dep))
            })
            .collect();

        let mut best: Option<&ScheduledTask> = None;
        let mut best_score = f64::MIN;

        for task in ready_tasks {
            let score = self.config.policy.score(task);
            if score > best_score {
                best_score = score;
                best = Some(task);
            }
        }

        best
    }

    /// Round-robin task selection
    fn next_round_robin(&mut self) -> Option<&ScheduledTask> {
        let ready_tasks: Vec<TaskId> = self
            .tasks
            .values()
            .filter(|t| t.state.is_actionable() && t.is_ready_to_start())
            .filter(|t| {
                t.dependencies
                    .iter()
                    .all(|dep| self.completed_tasks.contains(dep))
            })
            .map(|t| t.id)
            .collect();

        if ready_tasks.is_empty() {
            return None;
        }

        self.round_robin_index = (self.round_robin_index + 1) % ready_tasks.len();
        let task_id = ready_tasks[self.round_robin_index];
        self.tasks.get(&task_id)
    }

    /// Start executing a task
    pub fn start_task(&mut self, task_id: TaskId) -> Result<()> {
        // First check if task exists and is actionable
        {
            let task = self
                .tasks
                .get(&task_id)
                .ok_or_else(|| BrainError::InvalidState(format!("Task not found: {}", task_id)))?;

            if !task.state.is_actionable() {
                return Err(BrainError::InvalidState(format!(
                    "Task {} is not actionable: {:?}",
                    task_id, task.state
                )));
            }
        }

        // Check running count
        let running_count = self
            .tasks
            .values()
            .filter(|t| matches!(t.state, TaskState::Running))
            .count();

        if running_count >= self.config.max_concurrent {
            return Err(BrainError::InvalidState(
                "Maximum concurrent tasks reached".to_string(),
            ));
        }

        // Now mutate the task
        let task = self.tasks.get_mut(&task_id).unwrap();
        let name = task.name.clone();
        task.mark_running();
        self.stats.pending_tasks = self.stats.pending_tasks.saturating_sub(1);
        self.stats.running_tasks += 1;

        self.emit_event(TaskEvent::TaskStarted { task_id, name });

        Ok(())
    }

    /// Complete a task
    pub fn complete_task(&mut self, task_id: TaskId) -> Result<()> {
        let task = self
            .tasks
            .get_mut(&task_id)
            .ok_or_else(|| BrainError::InvalidState(format!("Task not found: {}", task_id)))?;

        let name = task.name.clone();
        task.mark_completed();
        self.stats.running_tasks = self.stats.running_tasks.saturating_sub(1);
        self.stats.completed_tasks += 1;
        self.completed_tasks.push(task_id);

        self.emit_event(TaskEvent::TaskCompleted { task_id, name });

        // Check if any blocked tasks can now run
        self.check_dependencies();

        Ok(())
    }

    /// Fail a task
    pub fn fail_task(&mut self, task_id: TaskId, error: impl Into<String>) -> Result<()> {
        let error = error.into();
        let task = self
            .tasks
            .get_mut(&task_id)
            .ok_or_else(|| BrainError::InvalidState(format!("Task not found: {}", task_id)))?;

        let name = task.name.clone();

        // Check for retry
        if self.config.auto_retry && task.retry_count < task.max_retries {
            task.retry_count += 1;
            task.state = TaskState::Pending;
            task.updated_at = Utc::now();
            return Ok(());
        }

        task.mark_failed(&error);
        self.stats.running_tasks = self.stats.running_tasks.saturating_sub(1);
        self.stats.failed_tasks += 1;

        self.emit_event(TaskEvent::TaskFailed {
            task_id,
            name,
            error,
        });

        Ok(())
    }

    /// Block a task
    pub fn block_task(&mut self, task_id: TaskId, reason: impl Into<String>) -> Result<()> {
        let reason = reason.into();
        let task = self
            .tasks
            .get_mut(&task_id)
            .ok_or_else(|| BrainError::InvalidState(format!("Task not found: {}", task_id)))?;

        let name = task.name.clone();
        task.mark_blocked(&reason);

        self.emit_event(TaskEvent::TaskBlocked {
            task_id,
            name,
            reason,
        });

        Ok(())
    }

    /// Unblock a task
    pub fn unblock_task(&mut self, task_id: TaskId) -> Result<()> {
        let task = self
            .tasks
            .get_mut(&task_id)
            .ok_or_else(|| BrainError::InvalidState(format!("Task not found: {}", task_id)))?;

        if !matches!(task.state, TaskState::Blocked { .. }) {
            return Err(BrainError::InvalidState(format!(
                "Task {} is not blocked",
                task_id
            )));
        }

        let name = task.name.clone();
        task.mark_ready();

        self.emit_event(TaskEvent::TaskUnblocked { task_id, name });

        Ok(())
    }

    /// Cancel a task
    pub fn cancel_task(&mut self, task_id: TaskId, reason: impl Into<String>) -> Result<()> {
        let reason = reason.into();
        let task = self
            .tasks
            .get_mut(&task_id)
            .ok_or_else(|| BrainError::InvalidState(format!("Task not found: {}", task_id)))?;

        if task.state.is_terminal() {
            return Err(BrainError::InvalidState(format!(
                "Task {} is already in terminal state",
                task_id
            )));
        }

        let name = task.name.clone();
        let was_running = matches!(task.state, TaskState::Running);
        task.mark_cancelled(&reason);

        if was_running {
            self.stats.running_tasks = self.stats.running_tasks.saturating_sub(1);
        } else {
            self.stats.pending_tasks = self.stats.pending_tasks.saturating_sub(1);
        }
        self.stats.cancelled_tasks += 1;

        self.emit_event(TaskEvent::TaskCancelled {
            task_id,
            name,
            reason,
        });

        Ok(())
    }

    /// Get a task by ID
    pub fn get_task(&self, task_id: TaskId) -> Option<&ScheduledTask> {
        self.tasks.get(&task_id)
    }

    /// Get all tasks
    pub fn all_tasks(&self) -> impl Iterator<Item = &ScheduledTask> {
        self.tasks.values()
    }

    /// Get tasks by state
    pub fn tasks_by_state(&self, state_check: impl Fn(&TaskState) -> bool) -> Vec<&ScheduledTask> {
        self.tasks
            .values()
            .filter(|t| state_check(&t.state))
            .collect()
    }

    /// Get tasks by priority
    pub fn tasks_by_priority(&self, priority: TaskPriority) -> Vec<&ScheduledTask> {
        self.tasks
            .values()
            .filter(|t| t.priority == priority)
            .collect()
    }

    /// Get tasks by tag
    pub fn tasks_by_tag(&self, tag: &str) -> Vec<&ScheduledTask> {
        self.tasks
            .values()
            .filter(|t| t.tags.iter().any(|t| t == tag))
            .collect()
    }

    /// Subscribe to events
    pub fn subscribe(
        &mut self,
        callback: EventCallback,
        filter: Option<Vec<TaskId>>,
    ) -> SubscriptionId {
        let id = Uuid::new_v4();
        self.subscribers.push(Subscription {
            id,
            filter,
            callback,
        });
        id
    }

    /// Unsubscribe from events
    pub fn unsubscribe(&mut self, subscription_id: SubscriptionId) {
        self.subscribers.retain(|s| s.id != subscription_id);
    }

    /// Get event history
    pub fn event_history(&self) -> &VecDeque<TaskEvent> {
        &self.event_history
    }

    /// Get statistics
    pub fn stats(&self) -> &SchedulerStats {
        &self.stats
    }

    /// Check for approaching deadlines
    fn check_deadlines(&mut self) {
        let now = Utc::now();
        let check_interval = Duration::seconds(self.config.deadline_check_interval_secs as i64);

        if now - self.last_deadline_check < check_interval {
            return;
        }
        self.last_deadline_check = now;

        let mut events = Vec::new();

        for task in self.tasks.values_mut() {
            if task.state.is_terminal() {
                continue;
            }

            if let Some(deadline) = task.deadline {
                if now >= deadline {
                    // Task expired
                    if self.config.auto_expire && task.state.is_actionable() {
                        task.mark_expired();
                        self.stats.expired_tasks += 1;
                        if matches!(task.state, TaskState::Running) {
                            self.stats.running_tasks = self.stats.running_tasks.saturating_sub(1);
                        } else {
                            self.stats.pending_tasks = self.stats.pending_tasks.saturating_sub(1);
                        }
                        events.push(TaskEvent::TaskExpired {
                            task_id: task.id,
                            name: task.name.clone(),
                        });
                    }
                } else {
                    // Check for deadline warning
                    let hours_remaining = (deadline - now).num_hours() as f64
                        + (deadline - now).num_minutes() as f64 / 60.0;

                    if hours_remaining <= self.config.deadline_warning_hours {
                        events.push(TaskEvent::DeadlineApproaching {
                            task_id: task.id,
                            name: task.name.clone(),
                            hours_remaining,
                        });
                    }
                }
            }
        }

        for event in events {
            self.emit_event(event);
        }
    }

    /// Check if any blocked tasks have their dependencies met
    fn check_dependencies(&mut self) {
        let mut to_ready = Vec::new();

        for task in self.tasks.values() {
            if !matches!(task.state, TaskState::Pending) {
                continue;
            }

            let deps_met = task
                .dependencies
                .iter()
                .all(|dep| self.completed_tasks.contains(dep));

            if deps_met && task.is_ready_to_start() {
                to_ready.push(task.id);
            }
        }

        for task_id in to_ready {
            if let Some(task) = self.tasks.get_mut(&task_id) {
                let name = task.name.clone();
                task.mark_ready();
                self.emit_event(TaskEvent::TaskReady { task_id, name });
            }
        }
    }

    /// Rebuild the priority queue
    fn rebuild_queue(&mut self) {
        self.queue.clear();
        for task in self.tasks.values() {
            if task.state.is_actionable() {
                let score = self.config.policy.score(task);
                self.queue.push(QueueEntry {
                    task_id: task.id,
                    score,
                });
            }
        }
    }

    /// Emit an event to subscribers
    fn emit_event(&mut self, event: TaskEvent) {
        // Store in history
        self.event_history.push_back(event.clone());
        if self.event_history.len() > self.config.max_event_history {
            self.event_history.pop_front();
        }

        // Notify subscribers
        for subscriber in &self.subscribers {
            // Check filter - skip if event doesn't match filter
            if let Some(filter) = &subscriber.filter
                && let Some(task_id) = event.task_id()
                && !filter.contains(&task_id)
            {
                continue;
            }

            (subscriber.callback)(&event);
        }
    }

    /// Clean up completed tasks older than specified duration
    pub fn cleanup(&mut self, older_than: Duration) {
        let cutoff = Utc::now() - older_than;

        let to_remove: Vec<TaskId> = self
            .tasks
            .iter()
            .filter_map(|(id, task)| match &task.state {
                TaskState::Completed { completed_at } if *completed_at < cutoff => Some(*id),
                TaskState::Failed { failed_at, .. } if *failed_at < cutoff => Some(*id),
                TaskState::Cancelled { cancelled_at, .. } if *cancelled_at < cutoff => Some(*id),
                TaskState::Expired { expired_at } if *expired_at < cutoff => Some(*id),
                _ => None,
            })
            .collect();

        for id in to_remove {
            self.tasks.remove(&id);
            self.completed_tasks.retain(|t| *t != id);
        }
    }
}

impl Default for TaskScheduler {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// PERSISTENT SCHEDULER (SQLITE)
// ============================================================================

/// Persistent task scheduler backed by SQLite
pub struct PersistentScheduler {
    /// In-memory scheduler
    inner: TaskScheduler,
    /// Database connection (thread-safe)
    conn: Arc<Mutex<Connection>>,
}

impl PersistentScheduler {
    /// Create a new persistent scheduler with in-memory database
    pub fn new_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        Self::with_connection(conn)
    }

    /// Create a new persistent scheduler with file-based database
    pub fn new(db_path: &str) -> Result<Self> {
        let conn = Connection::open(db_path)?;
        Self::with_connection(conn)
    }

    /// Create with existing connection
    fn with_connection(conn: Connection) -> Result<Self> {
        let mut scheduler = Self {
            inner: TaskScheduler::new(),
            conn: Arc::new(Mutex::new(conn)),
        };
        scheduler.init_schema()?;
        scheduler.load_tasks()?;
        Ok(scheduler)
    }

    /// Create with custom configuration
    pub fn with_config(db_path: &str, config: SchedulerConfig) -> Result<Self> {
        let conn = Connection::open(db_path)?;
        let mut scheduler = Self {
            inner: TaskScheduler::with_config(config),
            conn: Arc::new(Mutex::new(conn)),
        };
        scheduler.init_schema()?;
        scheduler.load_tasks()?;
        Ok(scheduler)
    }

    /// Initialize database schema
    fn init_schema(&self) -> Result<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| BrainError::InvalidState(format!("Failed to lock connection: {}", e)))?;

        conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS scheduled_tasks (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                priority INTEGER NOT NULL DEFAULT 2,
                state TEXT NOT NULL DEFAULT 'pending',
                deadline TEXT,
                scheduled_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                estimated_duration_ms INTEGER NOT NULL DEFAULT 0,
                salience REAL NOT NULL DEFAULT 0.5,
                goal_id TEXT,
                tags TEXT NOT NULL DEFAULT '[]',
                dependencies TEXT NOT NULL DEFAULT '[]',
                retry_count INTEGER NOT NULL DEFAULT 0,
                max_retries INTEGER NOT NULL DEFAULT 3,
                metadata TEXT NOT NULL DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_tasks_state ON scheduled_tasks(state);
            CREATE INDEX IF NOT EXISTS idx_tasks_priority ON scheduled_tasks(priority);
            CREATE INDEX IF NOT EXISTS idx_tasks_deadline ON scheduled_tasks(deadline);
            CREATE INDEX IF NOT EXISTS idx_tasks_created ON scheduled_tasks(created_at);

            CREATE TABLE IF NOT EXISTS task_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT,
                event_type TEXT NOT NULL,
                event_data TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_events_task ON task_events(task_id);
            CREATE INDEX IF NOT EXISTS idx_events_created ON task_events(created_at);
            ",
        )?;

        Ok(())
    }

    /// Load tasks from database
    fn load_tasks(&mut self) -> Result<()> {
        let tasks: Vec<ScheduledTask> = {
            let conn = self.conn.lock().map_err(|e| {
                BrainError::InvalidState(format!("Failed to lock connection: {}", e))
            })?;

            let mut stmt = conn.prepare(
                "SELECT id, name, description, priority, state, deadline, scheduled_at,
                        created_at, updated_at, estimated_duration_ms, salience, goal_id,
                        tags, dependencies, retry_count, max_retries, metadata
                 FROM scheduled_tasks
                 WHERE state NOT IN ('completed', 'failed', 'cancelled', 'expired')",
            )?;

            stmt.query_map([], |row| {
                let id_str: String = row.get(0)?;
                let deadline_str: Option<String> = row.get(5)?;
                let scheduled_at_str: Option<String> = row.get(6)?;
                let created_at_str: String = row.get(7)?;
                let updated_at_str: String = row.get(8)?;
                let tags_str: String = row.get(12)?;
                let deps_str: String = row.get(13)?;
                let metadata_str: String = row.get(16)?;

                Ok(ScheduledTask {
                    id: Uuid::parse_str(&id_str).unwrap_or_else(|_| Uuid::new_v4()),
                    name: row.get(1)?,
                    description: row.get(2)?,
                    priority: TaskPriority::from_i32(row.get(3)?),
                    state: TaskState::from_db_string(&row.get::<_, String>(4)?),
                    deadline: deadline_str.and_then(|s| {
                        DateTime::parse_from_rfc3339(&s)
                            .ok()
                            .map(|dt| dt.with_timezone(&Utc))
                    }),
                    scheduled_at: scheduled_at_str.and_then(|s| {
                        DateTime::parse_from_rfc3339(&s)
                            .ok()
                            .map(|dt| dt.with_timezone(&Utc))
                    }),
                    created_at: DateTime::parse_from_rfc3339(&created_at_str)
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or_else(|_| Utc::now()),
                    updated_at: DateTime::parse_from_rfc3339(&updated_at_str)
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or_else(|_| Utc::now()),
                    estimated_duration_ms: row.get(9)?,
                    salience: row.get(10)?,
                    goal_id: row.get(11)?,
                    tags: serde_json::from_str(&tags_str).unwrap_or_default(),
                    dependencies: serde_json::from_str(&deps_str).unwrap_or_default(),
                    retry_count: row.get(14)?,
                    max_retries: row.get(15)?,
                    metadata: serde_json::from_str(&metadata_str).unwrap_or_default(),
                })
            })?
            .filter_map(|r| r.ok())
            .collect()
        }; // conn lock released here

        // Schedule loaded tasks
        for task in tasks {
            // Directly insert into inner without going through schedule() to avoid DB write
            let task_id = task.id;
            let score = self.inner.config.policy.score(&task);
            self.inner.tasks.insert(task_id, task);
            self.inner.queue.push(QueueEntry { task_id, score });
        }

        Ok(())
    }

    /// Persist a task to database
    fn persist_task(&self, task: &ScheduledTask) -> Result<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| BrainError::InvalidState(format!("Failed to lock connection: {}", e)))?;

        conn.execute(
            "INSERT OR REPLACE INTO scheduled_tasks (
                id, name, description, priority, state, deadline, scheduled_at,
                created_at, updated_at, estimated_duration_ms, salience, goal_id,
                tags, dependencies, retry_count, max_retries, metadata
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17)",
            params![
                task.id.to_string(),
                task.name,
                task.description,
                task.priority.to_i32(),
                task.state.to_db_string(),
                task.deadline.map(|d| d.to_rfc3339()),
                task.scheduled_at.map(|d| d.to_rfc3339()),
                task.created_at.to_rfc3339(),
                task.updated_at.to_rfc3339(),
                task.estimated_duration_ms as i64,
                task.salience,
                task.goal_id,
                serde_json::to_string(&task.tags)?,
                serde_json::to_string(&task.dependencies)?,
                task.retry_count,
                task.max_retries,
                serde_json::to_string(&task.metadata)?,
            ],
        )?;

        Ok(())
    }

    /// Persist an event to database (reserved for future event logging)
    #[allow(dead_code)]
    fn persist_event(&self, event: &TaskEvent) -> Result<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| BrainError::InvalidState(format!("Failed to lock connection: {}", e)))?;

        let event_type = match event {
            TaskEvent::TaskAdded { .. } => "added",
            TaskEvent::TaskReady { .. } => "ready",
            TaskEvent::TaskStarted { .. } => "started",
            TaskEvent::TaskCompleted { .. } => "completed",
            TaskEvent::TaskFailed { .. } => "failed",
            TaskEvent::TaskBlocked { .. } => "blocked",
            TaskEvent::TaskUnblocked { .. } => "unblocked",
            TaskEvent::TaskCancelled { .. } => "cancelled",
            TaskEvent::TaskExpired { .. } => "expired",
            TaskEvent::DeadlineApproaching { .. } => "deadline_approaching",
            TaskEvent::SchedulerStateChanged { .. } => "state_changed",
        };

        conn.execute(
            "INSERT INTO task_events (task_id, event_type, event_data, created_at)
             VALUES (?1, ?2, ?3, ?4)",
            params![
                event.task_id().map(|id| id.to_string()),
                event_type,
                serde_json::to_string(event)?,
                Utc::now().to_rfc3339(),
            ],
        )?;

        Ok(())
    }

    /// Schedule a task (persistent)
    pub fn schedule(&mut self, task: ScheduledTask) -> Result<TaskId> {
        self.persist_task(&task)?;
        let task_id = self.inner.schedule(task);
        Ok(task_id)
    }

    /// Get the next task to execute
    pub fn next_task(&mut self) -> Option<&ScheduledTask> {
        self.inner.next_task()
    }

    /// Start a task
    pub fn start_task(&mut self, task_id: TaskId) -> Result<()> {
        self.inner.start_task(task_id)?;
        if let Some(task) = self.inner.get_task(task_id) {
            self.persist_task(task)?;
        }
        Ok(())
    }

    /// Complete a task
    pub fn complete_task(&mut self, task_id: TaskId) -> Result<()> {
        self.inner.complete_task(task_id)?;
        if let Some(task) = self.inner.get_task(task_id) {
            self.persist_task(task)?;
        }
        Ok(())
    }

    /// Fail a task
    pub fn fail_task(&mut self, task_id: TaskId, error: impl Into<String>) -> Result<()> {
        self.inner.fail_task(task_id, error)?;
        if let Some(task) = self.inner.get_task(task_id) {
            self.persist_task(task)?;
        }
        Ok(())
    }

    /// Block a task
    pub fn block_task(&mut self, task_id: TaskId, reason: impl Into<String>) -> Result<()> {
        self.inner.block_task(task_id, reason)?;
        if let Some(task) = self.inner.get_task(task_id) {
            self.persist_task(task)?;
        }
        Ok(())
    }

    /// Unblock a task
    pub fn unblock_task(&mut self, task_id: TaskId) -> Result<()> {
        self.inner.unblock_task(task_id)?;
        if let Some(task) = self.inner.get_task(task_id) {
            self.persist_task(task)?;
        }
        Ok(())
    }

    /// Cancel a task
    pub fn cancel_task(&mut self, task_id: TaskId, reason: impl Into<String>) -> Result<()> {
        self.inner.cancel_task(task_id, reason)?;
        if let Some(task) = self.inner.get_task(task_id) {
            self.persist_task(task)?;
        }
        Ok(())
    }

    /// Get a task by ID
    pub fn get_task(&self, task_id: TaskId) -> Option<&ScheduledTask> {
        self.inner.get_task(task_id)
    }

    /// Get all tasks
    pub fn all_tasks(&self) -> impl Iterator<Item = &ScheduledTask> {
        self.inner.all_tasks()
    }

    /// Subscribe to events
    pub fn subscribe(
        &mut self,
        callback: EventCallback,
        filter: Option<Vec<TaskId>>,
    ) -> SubscriptionId {
        self.inner.subscribe(callback, filter)
    }

    /// Unsubscribe from events
    pub fn unsubscribe(&mut self, subscription_id: SubscriptionId) {
        self.inner.unsubscribe(subscription_id)
    }

    /// Get statistics
    pub fn stats(&self) -> &SchedulerStats {
        self.inner.stats()
    }

    /// Set scheduling policy
    pub fn set_policy(&mut self, policy: SchedulingPolicy) {
        self.inner.set_policy(policy);
    }

    /// Get event history
    pub fn event_history(&self) -> &VecDeque<TaskEvent> {
        self.inner.event_history()
    }

    /// Cleanup old completed tasks from memory (keeps in DB for history)
    pub fn cleanup(&mut self, older_than: Duration) {
        self.inner.cleanup(older_than);
    }

    /// Get tasks by state from database (includes historical)
    pub fn query_tasks_by_state(&self, states: &[&str]) -> Result<Vec<ScheduledTask>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| BrainError::InvalidState(format!("Failed to lock connection: {}", e)))?;

        let placeholders: Vec<String> = (1..=states.len()).map(|i| format!("?{}", i)).collect();
        let sql = format!(
            "SELECT id, name, description, priority, state, deadline, scheduled_at,
                    created_at, updated_at, estimated_duration_ms, salience, goal_id,
                    tags, dependencies, retry_count, max_retries, metadata
             FROM scheduled_tasks
             WHERE state IN ({})",
            placeholders.join(", ")
        );

        let mut stmt = conn.prepare(&sql)?;

        let params: Vec<&dyn rusqlite::ToSql> =
            states.iter().map(|s| s as &dyn rusqlite::ToSql).collect();

        let tasks = stmt
            .query_map(params.as_slice(), |row| {
                let id_str: String = row.get(0)?;
                let deadline_str: Option<String> = row.get(5)?;
                let scheduled_at_str: Option<String> = row.get(6)?;
                let created_at_str: String = row.get(7)?;
                let updated_at_str: String = row.get(8)?;
                let tags_str: String = row.get(12)?;
                let deps_str: String = row.get(13)?;
                let metadata_str: String = row.get(16)?;

                Ok(ScheduledTask {
                    id: Uuid::parse_str(&id_str).unwrap_or_else(|_| Uuid::new_v4()),
                    name: row.get(1)?,
                    description: row.get(2)?,
                    priority: TaskPriority::from_i32(row.get(3)?),
                    state: TaskState::from_db_string(&row.get::<_, String>(4)?),
                    deadline: deadline_str.and_then(|s| {
                        DateTime::parse_from_rfc3339(&s)
                            .ok()
                            .map(|dt| dt.with_timezone(&Utc))
                    }),
                    scheduled_at: scheduled_at_str.and_then(|s| {
                        DateTime::parse_from_rfc3339(&s)
                            .ok()
                            .map(|dt| dt.with_timezone(&Utc))
                    }),
                    created_at: DateTime::parse_from_rfc3339(&created_at_str)
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or_else(|_| Utc::now()),
                    updated_at: DateTime::parse_from_rfc3339(&updated_at_str)
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or_else(|_| Utc::now()),
                    estimated_duration_ms: row.get(9)?,
                    salience: row.get(10)?,
                    goal_id: row.get(11)?,
                    tags: serde_json::from_str(&tags_str).unwrap_or_default(),
                    dependencies: serde_json::from_str(&deps_str).unwrap_or_default(),
                    retry_count: row.get(14)?,
                    max_retries: row.get(15)?,
                    metadata: serde_json::from_str(&metadata_str).unwrap_or_default(),
                })
            })?
            .filter_map(|r| r.ok())
            .collect();

        Ok(tasks)
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    #[test]
    fn test_task_creation() {
        let task = ScheduledTask::new("Test Task", "A test task")
            .with_priority(TaskPriority::High)
            .with_salience(0.8)
            .with_tag("test");

        assert_eq!(task.priority, TaskPriority::High);
        assert_eq!(task.salience, 0.8);
        assert!(task.tags.contains(&"test".to_string()));
    }

    #[test]
    fn test_task_urgency_with_deadline() {
        // Create a task that's been around for a while but has an approaching deadline
        let mut task = ScheduledTask::new("Urgent", "Due soon");
        // Simulate task created 23 hours ago, deadline in 1 hour (24h total window)
        task.created_at = Utc::now() - Duration::hours(23);
        task.deadline = Some(Utc::now() + Duration::hours(1));
        task.priority = TaskPriority::High;

        let urgency = task.urgency();
        // With 23/24 time elapsed, time_pressure is high, combined with High priority
        assert!(
            urgency > 0.7,
            "Urgency should be high for approaching deadline: {}",
            urgency
        );
    }

    #[test]
    fn test_scheduler_basic() {
        let mut scheduler = TaskScheduler::new();

        let task1 = ScheduledTask::new("Task 1", "First task").with_priority(TaskPriority::Low);

        let task2 = ScheduledTask::new("Task 2", "Second task").with_priority(TaskPriority::High);

        scheduler.schedule(task1);
        scheduler.schedule(task2);

        // High priority task should come first
        let next = scheduler.next_task().unwrap();
        assert_eq!(next.priority, TaskPriority::High);
    }

    #[test]
    fn test_edf_scheduling() {
        let mut scheduler = TaskScheduler::new();
        scheduler.set_policy(SchedulingPolicy::EarliestDeadlineFirst);

        let task1 = ScheduledTask::new("Later", "Due later")
            .with_deadline(Utc::now() + Duration::hours(24));

        let task2 =
            ScheduledTask::new("Soon", "Due soon").with_deadline(Utc::now() + Duration::hours(1));

        scheduler.schedule(task1);
        scheduler.schedule(task2);

        // Earlier deadline should come first
        let next = scheduler.next_task().unwrap();
        assert_eq!(next.name, "Soon");
    }

    #[test]
    fn test_task_lifecycle() {
        let mut scheduler = TaskScheduler::new();

        let task = ScheduledTask::new("Lifecycle", "Test lifecycle");
        let task_id = scheduler.schedule(task);

        // Start the task
        scheduler.start_task(task_id).unwrap();
        assert!(matches!(
            scheduler.get_task(task_id).unwrap().state,
            TaskState::Running
        ));

        // Complete the task
        scheduler.complete_task(task_id).unwrap();
        assert!(matches!(
            scheduler.get_task(task_id).unwrap().state,
            TaskState::Completed { .. }
        ));
    }

    #[test]
    fn test_task_blocking() {
        let mut scheduler = TaskScheduler::new();

        let task = ScheduledTask::new("Blockable", "Can be blocked");
        let task_id = scheduler.schedule(task);

        scheduler.block_task(task_id, "Waiting for input").unwrap();
        assert!(matches!(
            scheduler.get_task(task_id).unwrap().state,
            TaskState::Blocked { .. }
        ));

        scheduler.unblock_task(task_id).unwrap();
        assert!(matches!(
            scheduler.get_task(task_id).unwrap().state,
            TaskState::Ready
        ));
    }

    #[test]
    fn test_event_subscription() {
        let mut scheduler = TaskScheduler::new();

        let event_count = Arc::new(AtomicU32::new(0));
        let count_clone = Arc::clone(&event_count);

        scheduler.subscribe(
            Arc::new(move |_event| {
                count_clone.fetch_add(1, Ordering::SeqCst);
            }),
            None,
        );

        let task = ScheduledTask::new("Event Test", "Test events");
        scheduler.schedule(task);

        // Should have received TaskAdded event
        assert!(event_count.load(Ordering::SeqCst) >= 1);
    }

    #[test]
    fn test_dependencies() {
        let mut scheduler = TaskScheduler::new();

        let task1 = ScheduledTask::new("First", "Must complete first");
        let task1_id = scheduler.schedule(task1);

        let task2 = ScheduledTask::new("Second", "Depends on first").with_dependency(task1_id);
        let task2_id = scheduler.schedule(task2);

        // Task2 should still be pending (dependency not met)
        assert!(matches!(
            scheduler.get_task(task2_id).unwrap().state,
            TaskState::Pending
        ));

        // Complete task1
        scheduler.start_task(task1_id).unwrap();
        scheduler.complete_task(task1_id).unwrap();

        // Now task2 should be ready
        // Force a check
        let _ = scheduler.next_task();

        // After next_task call, dependency check runs
        assert!(matches!(
            scheduler.get_task(task2_id).unwrap().state,
            TaskState::Ready
        ));
    }

    #[test]
    fn test_persistent_scheduler() -> Result<()> {
        let mut scheduler = PersistentScheduler::new_in_memory()?;

        let task =
            ScheduledTask::new("Persistent", "Survives restart").with_priority(TaskPriority::High);

        let task_id = scheduler.schedule(task)?;

        assert!(scheduler.get_task(task_id).is_some());
        assert_eq!(scheduler.get_task(task_id).unwrap().name, "Persistent");

        Ok(())
    }

    #[test]
    fn test_weighted_score_policy() {
        let mut scheduler = TaskScheduler::new();
        scheduler.set_policy(SchedulingPolicy::WeightedScore);

        // High priority, low salience
        let task1 = ScheduledTask::new("High Pri", "High priority")
            .with_priority(TaskPriority::High)
            .with_salience(0.2);

        // Medium priority, high salience
        let task2 = ScheduledTask::new("High Sal", "High salience")
            .with_priority(TaskPriority::Medium)
            .with_salience(0.9);

        scheduler.schedule(task1);
        scheduler.schedule(task2);

        // The weighted score should consider both factors
        let next = scheduler.next_task().unwrap();
        // High priority should still win with weighted scoring
        assert_eq!(next.priority, TaskPriority::High);
    }

    #[test]
    fn test_task_state_conversion() {
        // Test round-trip conversion
        let states = vec![
            TaskState::Pending,
            TaskState::Ready,
            TaskState::Running,
            TaskState::Blocked {
                reason: "test".to_string(),
            },
            TaskState::Completed {
                completed_at: Utc::now(),
            },
        ];

        for state in states {
            let db_str = state.to_db_string();
            let recovered = TaskState::from_db_string(&db_str);

            // Compare the variant (not the timestamp)
            match (&state, &recovered) {
                (TaskState::Pending, TaskState::Pending) => {}
                (TaskState::Ready, TaskState::Ready) => {}
                (TaskState::Running, TaskState::Running) => {}
                (TaskState::Blocked { reason: r1 }, TaskState::Blocked { reason: r2 }) => {
                    assert_eq!(r1, r2);
                }
                (TaskState::Completed { .. }, TaskState::Completed { .. }) => {}
                _ => panic!("State mismatch: {:?} vs {:?}", state, recovered),
            }
        }
    }

    #[test]
    fn test_priority_ordering() {
        assert!(TaskPriority::Critical > TaskPriority::High);
        assert!(TaskPriority::High > TaskPriority::Medium);
        assert!(TaskPriority::Medium > TaskPriority::Low);
    }
}
