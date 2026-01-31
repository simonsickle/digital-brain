//! Subthalamic Nucleus (STN) - Response Inhibition and Task Watchdog
//!
//! The STN is the brain's emergency brake. It's part of the basal ganglia's
//! "hyperdirect pathway" that can rapidly stop ongoing actions. Key functions:
//!
//! - **Global inhibition**: Stop ALL ongoing actions when needed
//! - **Task monitoring**: Track running tasks and detect hangs
//! - **Progress tracking**: Kill tasks that aren't making progress
//! - **Resource limits**: Stop tasks that consume too much effort
//! - **Timeout enforcement**: Hard time limits on tasks
//!
//! # Biological Basis
//!
//! In the human brain:
//! - STN receives direct input from prefrontal cortex (hyperdirect pathway)
//! - When activated, broadly inhibits basal ganglia output
//! - Results in rapid, global stopping of motor programs
//! - Works with ACC (error detection) and prefrontal (executive control)
//!
//! # Computational Model
//!
//! ```text
//!     Prefrontal Cortex
//!            │
//!            │ "STOP!" signal
//!            ▼
//!     ┌──────────────┐
//!     │     STN      │
//!     │  (Watchdog)  │
//!     └──────┬───────┘
//!            │
//!            │ Global inhibition
//!            ▼
//!     ┌──────────────┐
//!     │    Basal     │
//!     │   Ganglia    │──→ Actions STOPPED
//!     └──────────────┘
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Unique identifier for a monitored task
pub type TaskId = Uuid;

/// Reason why a task was stopped
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StopReason {
    /// Task exceeded its time limit
    Timeout,
    /// Task hasn't made progress in too long
    NoProgress,
    /// Task consumed too much effort/resources
    ResourceExhaustion,
    /// External stop request (manual intervention)
    ExternalRequest,
    /// ACC detected the task is stuck/looping
    StuckDetected,
    /// No reward received despite high effort
    NoRewardHighEffort,
    /// Task completed successfully (not really a "stop")
    Completed,
    /// Habituation - same signal repeated too many times
    Habituation,
    /// Higher priority task needs resources
    Preempted,
}

/// Current state of a monitored task
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskState {
    /// Task is running normally
    Running,
    /// Task is paused (can be resumed)
    Paused,
    /// Task was stopped (see StopReason)
    Stopped(StopReason),
    /// Task completed successfully
    Completed,
}

/// Configuration for task monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskConfig {
    /// Maximum time the task can run (milliseconds)
    pub timeout_ms: u64,
    /// Maximum time without progress before killing (milliseconds)
    pub progress_timeout_ms: u64,
    /// Maximum effort units the task can consume
    pub max_effort: f64,
    /// Minimum reward expected per effort unit
    pub min_reward_per_effort: f64,
    /// How many times the same signal can repeat before habituation
    pub habituation_threshold: u32,
    /// Priority level (higher = harder to preempt)
    pub priority: u8,
}

impl Default for TaskConfig {
    fn default() -> Self {
        Self {
            timeout_ms: 30_000,         // 30 seconds default
            progress_timeout_ms: 5_000, // 5 seconds without progress
            max_effort: 100.0,
            min_reward_per_effort: 0.01,
            habituation_threshold: 10,
            priority: 5,
        }
    }
}

/// A task being monitored by the STN
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoredTask {
    pub id: TaskId,
    /// Human-readable name
    pub name: String,
    /// Current state
    pub state: TaskState,
    /// When the task started
    pub started_at: DateTime<Utc>,
    /// When the task last made progress
    pub last_progress_at: DateTime<Utc>,
    /// Total effort spent so far
    pub effort_spent: f64,
    /// Total reward received so far
    pub reward_received: f64,
    /// Number of times the same signal has been seen
    pub repetition_count: u32,
    /// Last signal/action taken (for detecting loops)
    pub last_signal: Option<String>,
    /// Configuration for this task
    pub config: TaskConfig,
    /// When the task ended (if stopped/completed)
    pub ended_at: Option<DateTime<Utc>>,
    /// Why it was stopped (if applicable)
    pub stop_reason: Option<StopReason>,
}

impl MonitoredTask {
    pub fn new(name: &str, config: TaskConfig) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            name: name.to_string(),
            state: TaskState::Running,
            started_at: now,
            last_progress_at: now,
            effort_spent: 0.0,
            reward_received: 0.0,
            repetition_count: 0,
            last_signal: None,
            config,
            ended_at: None,
            stop_reason: None,
        }
    }

    /// Record progress on the task
    pub fn record_progress(&mut self) {
        self.last_progress_at = Utc::now();
        self.repetition_count = 0; // Reset habituation
    }

    /// Record effort spent
    pub fn record_effort(&mut self, amount: f64) {
        self.effort_spent += amount;
    }

    /// Record reward received
    pub fn record_reward(&mut self, amount: f64) {
        self.reward_received += amount;
        if amount > 0.0 {
            self.record_progress(); // Reward counts as progress
        }
    }

    /// Record a signal/action (for loop detection)
    pub fn record_signal(&mut self, signal: &str) {
        if let Some(ref last) = self.last_signal {
            if last == signal {
                self.repetition_count += 1;
            } else {
                self.repetition_count = 1;
            }
        } else {
            self.repetition_count = 1;
        }
        self.last_signal = Some(signal.to_string());
    }

    /// Check if task has timed out
    pub fn is_timed_out(&self) -> bool {
        let elapsed = (Utc::now() - self.started_at).num_milliseconds() as u64;
        elapsed > self.config.timeout_ms
    }

    /// Check if task has stalled (no progress)
    pub fn is_stalled(&self) -> bool {
        let since_progress = (Utc::now() - self.last_progress_at).num_milliseconds() as u64;
        since_progress > self.config.progress_timeout_ms
    }

    /// Check if task has exhausted its resource budget
    pub fn is_exhausted(&self) -> bool {
        self.effort_spent > self.config.max_effort
    }

    /// Check if task is habituated (stuck in a loop)
    pub fn is_habituated(&self) -> bool {
        self.repetition_count >= self.config.habituation_threshold
    }

    /// Check if effort/reward ratio is too low
    pub fn is_unrewarding(&self) -> bool {
        if self.effort_spent < 10.0 {
            return false; // Give it a chance first
        }
        let ratio = self.reward_received / self.effort_spent;
        ratio < self.config.min_reward_per_effort
    }

    /// Get the elapsed time in milliseconds
    pub fn elapsed_ms(&self) -> u64 {
        (Utc::now() - self.started_at).num_milliseconds() as u64
    }

    /// Get time since last progress in milliseconds
    pub fn since_progress_ms(&self) -> u64 {
        (Utc::now() - self.last_progress_at).num_milliseconds() as u64
    }

    /// Stop the task with a reason
    pub fn stop(&mut self, reason: StopReason) {
        self.state = TaskState::Stopped(reason.clone());
        self.ended_at = Some(Utc::now());
        self.stop_reason = Some(reason);
    }

    /// Mark task as completed
    pub fn complete(&mut self) {
        self.state = TaskState::Completed;
        self.ended_at = Some(Utc::now());
        self.stop_reason = Some(StopReason::Completed);
    }

    /// Check if task is still running
    pub fn is_running(&self) -> bool {
        matches!(self.state, TaskState::Running)
    }
}

/// A stop signal that should be propagated
#[derive(Debug, Clone)]
pub struct StopSignal {
    pub task_id: TaskId,
    pub task_name: String,
    pub reason: StopReason,
    pub issued_at: DateTime<Utc>,
    /// Should this stop ALL tasks (global inhibition)?
    pub global: bool,
}

/// Statistics for the STN
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct STNStats {
    pub tasks_monitored: u64,
    pub tasks_completed: u64,
    pub tasks_stopped: u64,
    pub timeout_stops: u64,
    pub progress_stops: u64,
    pub exhaustion_stops: u64,
    pub habituation_stops: u64,
    pub unrewarding_stops: u64,
    pub external_stops: u64,
    pub global_inhibitions: u64,
    pub average_task_duration_ms: f64,
}

/// The Subthalamic Nucleus - Response Inhibition System
#[derive(Debug)]
pub struct STN {
    /// Currently monitored tasks
    tasks: HashMap<TaskId, MonitoredTask>,
    /// Pending stop signals
    stop_signals: Vec<StopSignal>,
    /// Global inhibition active?
    global_inhibition: bool,
    /// When global inhibition started
    global_inhibition_started: Option<DateTime<Utc>>,
    /// How long global inhibition lasts (ms)
    global_inhibition_duration_ms: u64,
    /// Statistics
    stats: STNStats,
    /// Task duration history (for averaging)
    duration_history: Vec<u64>,
}

impl STN {
    pub fn new() -> Self {
        Self {
            tasks: HashMap::new(),
            stop_signals: Vec::new(),
            global_inhibition: false,
            global_inhibition_started: None,
            global_inhibition_duration_ms: 1000, // 1 second default
            stats: STNStats::default(),
            duration_history: Vec::new(),
        }
    }

    /// Start monitoring a new task
    pub fn monitor(&mut self, name: &str, config: TaskConfig) -> TaskId {
        let task = MonitoredTask::new(name, config);
        let id = task.id;
        self.tasks.insert(id, task);
        self.stats.tasks_monitored += 1;
        id
    }

    /// Start monitoring with default config
    pub fn monitor_default(&mut self, name: &str) -> TaskId {
        self.monitor(name, TaskConfig::default())
    }

    /// Record progress on a task
    pub fn progress(&mut self, task_id: TaskId) {
        if let Some(task) = self.tasks.get_mut(&task_id) {
            task.record_progress();
        }
    }

    /// Record effort spent on a task
    pub fn effort(&mut self, task_id: TaskId, amount: f64) {
        if let Some(task) = self.tasks.get_mut(&task_id) {
            task.record_effort(amount);
        }
    }

    /// Record reward received for a task
    pub fn reward(&mut self, task_id: TaskId, amount: f64) {
        if let Some(task) = self.tasks.get_mut(&task_id) {
            task.record_reward(amount);
        }
    }

    /// Record a signal/action for loop detection
    pub fn signal(&mut self, task_id: TaskId, signal: &str) {
        if let Some(task) = self.tasks.get_mut(&task_id) {
            task.record_signal(signal);
        }
    }

    /// Mark a task as completed
    pub fn complete(&mut self, task_id: TaskId) {
        if let Some(task) = self.tasks.get_mut(&task_id) {
            let duration = task.elapsed_ms();
            task.complete();
            self.stats.tasks_completed += 1;
            self.update_duration_average(duration);
        }
    }

    /// Manually stop a task
    pub fn stop(&mut self, task_id: TaskId, reason: StopReason) {
        // Get task info first
        let task_info = if let Some(task) = self.tasks.get(&task_id) {
            if task.is_running() {
                Some((task.elapsed_ms(), task.name.clone()))
            } else {
                None
            }
        } else {
            None
        };

        // Now mutate
        if let Some((duration, task_name)) = task_info {
            if let Some(task) = self.tasks.get_mut(&task_id) {
                task.stop(reason.clone());
            }

            self.update_stats_for_stop(&reason);
            self.update_duration_average(duration);

            self.stop_signals.push(StopSignal {
                task_id,
                task_name,
                reason,
                issued_at: Utc::now(),
                global: false,
            });
        }
    }

    /// Trigger global inhibition (stop EVERYTHING)
    pub fn global_stop(&mut self, reason: &str) {
        self.global_inhibition = true;
        self.global_inhibition_started = Some(Utc::now());
        self.stats.global_inhibitions += 1;

        // Stop all running tasks
        let running_ids: Vec<TaskId> = self
            .tasks
            .values()
            .filter(|t| t.is_running())
            .map(|t| t.id)
            .collect();

        for id in running_ids {
            self.stop(id, StopReason::ExternalRequest);
        }

        self.stop_signals.push(StopSignal {
            task_id: Uuid::nil(),
            task_name: format!("GLOBAL: {}", reason),
            reason: StopReason::ExternalRequest,
            issued_at: Utc::now(),
            global: true,
        });
    }

    /// Check all tasks and stop any that should be stopped
    pub fn check_all(&mut self) -> Vec<StopSignal> {
        // Check if global inhibition should end
        if self.global_inhibition
            && let Some(started) = self.global_inhibition_started
        {
            let elapsed = (Utc::now() - started).num_milliseconds() as u64;
            if elapsed > self.global_inhibition_duration_ms {
                self.global_inhibition = false;
                self.global_inhibition_started = None;
            }
        }

        let mut signals = Vec::new();

        // Collect tasks that need stopping
        let tasks_to_stop: Vec<(TaskId, StopReason)> = self
            .tasks
            .values()
            .filter(|t| t.is_running())
            .filter_map(|t| {
                if t.is_timed_out() {
                    Some((t.id, StopReason::Timeout))
                } else if t.is_stalled() {
                    Some((t.id, StopReason::NoProgress))
                } else if t.is_exhausted() {
                    Some((t.id, StopReason::ResourceExhaustion))
                } else if t.is_habituated() {
                    Some((t.id, StopReason::Habituation))
                } else if t.is_unrewarding() {
                    Some((t.id, StopReason::NoRewardHighEffort))
                } else {
                    None
                }
            })
            .collect();

        // Stop them
        for (id, reason) in tasks_to_stop {
            self.stop(id, reason.clone());
            if let Some(task) = self.tasks.get(&id) {
                signals.push(StopSignal {
                    task_id: id,
                    task_name: task.name.clone(),
                    reason,
                    issued_at: Utc::now(),
                    global: false,
                });
            }
        }

        signals
    }

    /// Get pending stop signals (and clear them)
    pub fn drain_signals(&mut self) -> Vec<StopSignal> {
        std::mem::take(&mut self.stop_signals)
    }

    /// Peek at pending stop signals
    pub fn pending_signals(&self) -> &[StopSignal] {
        &self.stop_signals
    }

    /// Is global inhibition currently active?
    pub fn is_globally_inhibited(&self) -> bool {
        self.global_inhibition
    }

    /// Get a task by ID
    pub fn get_task(&self, id: TaskId) -> Option<&MonitoredTask> {
        self.tasks.get(&id)
    }

    /// Get all running tasks
    pub fn running_tasks(&self) -> Vec<&MonitoredTask> {
        self.tasks.values().filter(|t| t.is_running()).collect()
    }

    /// Get all stopped tasks
    pub fn stopped_tasks(&self) -> Vec<&MonitoredTask> {
        self.tasks
            .values()
            .filter(|t| matches!(t.state, TaskState::Stopped(_)))
            .collect()
    }

    /// Clean up completed/stopped tasks older than threshold
    pub fn cleanup(&mut self, older_than_ms: u64) {
        let now = Utc::now();
        self.tasks.retain(|_, task| {
            if let Some(ended) = task.ended_at {
                let age = (now - ended).num_milliseconds() as u64;
                age < older_than_ms
            } else {
                true // Keep running tasks
            }
        });
    }

    /// Get statistics
    pub fn stats(&self) -> &STNStats {
        &self.stats
    }

    /// Update stats for a stop
    fn update_stats_for_stop(&mut self, reason: &StopReason) {
        self.stats.tasks_stopped += 1;
        match reason {
            StopReason::Timeout => self.stats.timeout_stops += 1,
            StopReason::NoProgress => self.stats.progress_stops += 1,
            StopReason::ResourceExhaustion => self.stats.exhaustion_stops += 1,
            StopReason::Habituation => self.stats.habituation_stops += 1,
            StopReason::NoRewardHighEffort => self.stats.unrewarding_stops += 1,
            StopReason::ExternalRequest => self.stats.external_stops += 1,
            _ => {}
        }
    }

    /// Update average duration
    fn update_duration_average(&mut self, duration: u64) {
        self.duration_history.push(duration);
        if self.duration_history.len() > 100 {
            self.duration_history.remove(0);
        }
        self.stats.average_task_duration_ms =
            self.duration_history.iter().sum::<u64>() as f64 / self.duration_history.len() as f64;
    }

    /// Set global inhibition duration
    pub fn set_inhibition_duration(&mut self, ms: u64) {
        self.global_inhibition_duration_ms = ms;
    }
}

impl Default for STN {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;
    use std::time::Duration;

    #[test]
    fn test_task_monitoring() {
        let mut stn = STN::new();

        let id = stn.monitor_default("test_task");
        assert!(stn.get_task(id).is_some());
        assert!(stn.get_task(id).unwrap().is_running());
    }

    #[test]
    fn test_task_completion() {
        let mut stn = STN::new();

        let id = stn.monitor_default("test_task");
        stn.complete(id);

        let task = stn.get_task(id).unwrap();
        assert!(!task.is_running());
        assert_eq!(stn.stats().tasks_completed, 1);
    }

    #[test]
    fn test_manual_stop() {
        let mut stn = STN::new();

        let id = stn.monitor_default("test_task");
        stn.stop(id, StopReason::ExternalRequest);

        let task = stn.get_task(id).unwrap();
        assert!(matches!(task.state, TaskState::Stopped(_)));
        assert_eq!(task.stop_reason, Some(StopReason::ExternalRequest));
    }

    #[test]
    fn test_timeout_detection() {
        let mut stn = STN::new();

        let config = TaskConfig {
            timeout_ms: 10, // Very short timeout
            ..Default::default()
        };

        let _id = stn.monitor("quick_task", config);

        // Wait for timeout
        sleep(Duration::from_millis(20));

        let signals = stn.check_all();
        assert!(!signals.is_empty());
        assert_eq!(signals[0].reason, StopReason::Timeout);
    }

    #[test]
    fn test_progress_tracking() {
        let mut stn = STN::new();

        let config = TaskConfig {
            progress_timeout_ms: 10, // Very short progress timeout
            timeout_ms: 1000,
            ..Default::default()
        };

        let _id = stn.monitor("progress_task", config);

        // Wait without making progress
        sleep(Duration::from_millis(20));

        let signals = stn.check_all();
        assert!(!signals.is_empty());
        assert_eq!(signals[0].reason, StopReason::NoProgress);
    }

    #[test]
    fn test_habituation_detection() {
        let mut stn = STN::new();

        let config = TaskConfig {
            habituation_threshold: 3,
            ..Default::default()
        };

        let id = stn.monitor("loop_task", config);

        // Repeat the same signal
        for _ in 0..5 {
            stn.signal(id, "same_action");
        }

        let signals = stn.check_all();
        assert!(!signals.is_empty());
        assert_eq!(signals[0].reason, StopReason::Habituation);
    }

    #[test]
    fn test_global_inhibition() {
        let mut stn = STN::new();

        let _id1 = stn.monitor_default("task1");
        let _id2 = stn.monitor_default("task2");

        assert_eq!(stn.running_tasks().len(), 2);

        stn.global_stop("emergency");

        assert!(stn.is_globally_inhibited());
        assert_eq!(stn.running_tasks().len(), 0);
        assert_eq!(stn.stats().global_inhibitions, 1);
    }

    #[test]
    fn test_effort_exhaustion() {
        let mut stn = STN::new();

        let config = TaskConfig {
            max_effort: 10.0,
            ..Default::default()
        };

        let id = stn.monitor("effort_task", config);

        // Spend too much effort
        stn.effort(id, 15.0);

        let signals = stn.check_all();
        assert!(!signals.is_empty());
        assert_eq!(signals[0].reason, StopReason::ResourceExhaustion);
    }

    #[test]
    fn test_unrewarding_detection() {
        let mut stn = STN::new();

        let config = TaskConfig {
            min_reward_per_effort: 0.1,
            ..Default::default()
        };

        let id = stn.monitor("unrewarding_task", config);

        // High effort, low reward
        stn.effort(id, 100.0);
        stn.reward(id, 0.5);

        let signals = stn.check_all();
        assert!(!signals.is_empty());
        assert_eq!(signals[0].reason, StopReason::NoRewardHighEffort);
    }
}
