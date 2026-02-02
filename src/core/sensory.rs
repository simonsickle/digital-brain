//! Sensory Streams - Input sources for the consciousness loop
//!
//! Sensory streams are continuous or periodic sources of stimuli.
//! They abstract different input modalities:
//! - File system watching
//! - Time/clock events
//! - External prompts
//! - Internal drive signals
//!
//! The consciousness loop polls these streams to gather stimuli.

use async_trait::async_trait;
use chrono::{DateTime, Duration, Utc};
use notify::{Event, RecommendedWatcher, RecursiveMode, Watcher};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc as async_mpsc;

use crate::core::boredom::{BoredomRecommendation, BoredomTracker};
use crate::core::curiosity::CuriositySystem;
use crate::core::goals::GoalManager;
use crate::core::neuromodulators::NeuromodulatorySystem;
use crate::core::stimulus::{DriveEvent, FileEvent, Stimulus, StimulusPriority, TimeEvent};

/// Trait for sensory input streams
#[async_trait]
pub trait SensoryStream: Send + Sync {
    /// Get the name of this stream
    fn name(&self) -> &str;

    /// Poll for new stimuli (non-blocking)
    fn poll(&mut self) -> Vec<Stimulus>;

    /// Is this stream active?
    fn is_active(&self) -> bool;

    /// Shutdown the stream
    fn shutdown(&mut self);
}

// ============================================================================
// FILE SYSTEM STREAM
// ============================================================================

/// Configuration for file system watching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSystemConfig {
    /// Paths to watch
    pub watch_paths: Vec<PathBuf>,
    /// File patterns to include (glob)
    pub include_patterns: Vec<String>,
    /// File patterns to exclude (glob)
    pub exclude_patterns: Vec<String>,
    /// Watch recursively
    pub recursive: bool,
    /// Debounce time (ms)
    pub debounce_ms: u64,
}

impl Default for FileSystemConfig {
    fn default() -> Self {
        Self {
            watch_paths: vec![PathBuf::from(".")],
            include_patterns: vec!["*".to_string()],
            exclude_patterns: vec![
                "*.tmp".to_string(),
                "*.swp".to_string(),
                ".git/*".to_string(),
                "target/*".to_string(),
            ],
            recursive: true,
            debounce_ms: 100,
        }
    }
}

/// File system sensory stream
pub struct FileSystemStream {
    name: String,
    config: FileSystemConfig,
    watcher: Option<RecommendedWatcher>,
    event_rx: Option<Arc<Mutex<Receiver<notify::Result<Event>>>>>,
    active: bool,
    pending_events: VecDeque<FileEvent>,
}

impl FileSystemStream {
    /// Create a new file system stream
    pub fn new(config: FileSystemConfig) -> Self {
        Self {
            name: "filesystem".to_string(),
            config,
            watcher: None,
            event_rx: None,
            active: false,
            pending_events: VecDeque::new(),
        }
    }

    /// Start watching
    pub fn start(&mut self) -> Result<(), notify::Error> {
        let (tx, rx) = mpsc::channel();

        let mut watcher = notify::recommended_watcher(move |res| {
            let _ = tx.send(res);
        })?;

        let mode = if self.config.recursive {
            RecursiveMode::Recursive
        } else {
            RecursiveMode::NonRecursive
        };

        for path in &self.config.watch_paths {
            if path.exists() {
                watcher.watch(path, mode)?;
            }
        }

        self.watcher = Some(watcher);
        self.event_rx = Some(Arc::new(Mutex::new(rx)));
        self.active = true;

        Ok(())
    }

    /// Check if a path matches exclusion patterns
    fn is_excluded(&self, path: &std::path::Path) -> bool {
        let path_str = path.to_string_lossy();

        for pattern in &self.config.exclude_patterns {
            if let Ok(glob) = glob::Pattern::new(pattern)
                && glob.matches(&path_str)
            {
                return true;
            }
        }
        false
    }

    /// Convert notify event to FileEvent
    fn convert_event(&self, event: Event) -> Option<FileEvent> {
        use notify::EventKind;

        let path = event.paths.first()?.clone();

        if self.is_excluded(&path) {
            return None;
        }

        match event.kind {
            EventKind::Create(_) => Some(FileEvent::Created { path }),
            EventKind::Modify(_) => Some(FileEvent::Modified { path }),
            EventKind::Remove(_) => Some(FileEvent::Deleted { path }),
            _ => None,
        }
    }
}

#[async_trait]
impl SensoryStream for FileSystemStream {
    fn name(&self) -> &str {
        &self.name
    }

    fn poll(&mut self) -> Vec<Stimulus> {
        let mut stimuli = Vec::new();

        // Drain any pending events
        while let Some(event) = self.pending_events.pop_front() {
            stimuli.push(Stimulus::from_file_event(event));
        }

        // Check for new events
        if let Some(ref rx) = self.event_rx
            && let Ok(rx_guard) = rx.lock()
        {
            while let Ok(result) = rx_guard.try_recv() {
                if let Ok(event) = result
                    && let Some(file_event) = self.convert_event(event)
                {
                    stimuli.push(Stimulus::from_file_event(file_event));
                }
            }
        }

        stimuli
    }

    fn is_active(&self) -> bool {
        self.active
    }

    fn shutdown(&mut self) {
        self.watcher = None;
        self.event_rx = None;
        self.active = false;
    }
}

// ============================================================================
// CLOCK STREAM
// ============================================================================

/// Configuration for clock/time events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClockConfig {
    /// Tick interval in milliseconds
    pub tick_interval_ms: u64,
    /// Idle timeout in seconds
    pub idle_timeout_secs: u64,
}

impl Default for ClockConfig {
    fn default() -> Self {
        Self {
            tick_interval_ms: 1000, // 1 second ticks
            idle_timeout_secs: 300, // 5 minute idle timeout
        }
    }
}

/// Clock sensory stream - time-based events
pub struct ClockStream {
    name: String,
    config: ClockConfig,
    cycle: u64,
    last_tick: DateTime<Utc>,
    last_activity: DateTime<Utc>,
    alarms: Vec<(String, DateTime<Utc>)>,
    active: bool,
}

impl ClockStream {
    /// Create a new clock stream
    pub fn new(config: ClockConfig) -> Self {
        let now = Utc::now();
        Self {
            name: "clock".to_string(),
            config,
            cycle: 0,
            last_tick: now,
            last_activity: now,
            alarms: Vec::new(),
            active: true,
        }
    }

    /// Record activity (resets idle timer)
    pub fn record_activity(&mut self) {
        self.last_activity = Utc::now();
    }

    /// Set an alarm
    pub fn set_alarm(&mut self, name: impl Into<String>, time: DateTime<Utc>) {
        self.alarms.push((name.into(), time));
    }

    /// Cancel an alarm
    pub fn cancel_alarm(&mut self, name: &str) {
        self.alarms.retain(|(n, _)| n != name);
    }
}

#[async_trait]
impl SensoryStream for ClockStream {
    fn name(&self) -> &str {
        &self.name
    }

    fn poll(&mut self) -> Vec<Stimulus> {
        let mut stimuli = Vec::new();
        let now = Utc::now();

        // Check for tick
        let tick_interval = Duration::milliseconds(self.config.tick_interval_ms as i64);
        if now - self.last_tick >= tick_interval {
            self.cycle += 1;
            self.last_tick = now;

            stimuli.push(
                Stimulus::from_time_event(TimeEvent::Tick { cycle: self.cycle })
                    .with_priority(StimulusPriority::Background)
                    .with_salience(0.1),
            );
        }

        // Check for idle timeout
        let idle_timeout = Duration::seconds(self.config.idle_timeout_secs as i64);
        if now - self.last_activity >= idle_timeout {
            let idle_secs = (now - self.last_activity).num_seconds() as u64;
            stimuli.push(
                Stimulus::from_time_event(TimeEvent::IdleTimeout {
                    idle_seconds: idle_secs,
                })
                .with_priority(StimulusPriority::Normal)
                .with_salience(0.4),
            );
            // Reset to avoid repeated triggers
            self.last_activity = now - idle_timeout + Duration::seconds(60);
        }

        // Check alarms
        let mut triggered = Vec::new();
        for (i, (name, time)) in self.alarms.iter().enumerate() {
            if now >= *time {
                stimuli.push(
                    Stimulus::from_time_event(TimeEvent::Alarm { name: name.clone() })
                        .with_priority(StimulusPriority::High)
                        .with_salience(0.8),
                );
                triggered.push(i);
            }
        }
        // Remove triggered alarms (in reverse order to preserve indices)
        for i in triggered.into_iter().rev() {
            self.alarms.remove(i);
        }

        stimuli
    }

    fn is_active(&self) -> bool {
        self.active
    }

    fn shutdown(&mut self) {
        self.active = false;
    }
}

// ============================================================================
// PROMPT STREAM
// ============================================================================

/// External prompt stream - for human input
pub struct PromptStream {
    name: String,
    rx: async_mpsc::Receiver<(String, Option<String>)>,
    #[allow(dead_code)]
    pending: VecDeque<(String, Option<String>)>,
    active: bool,
}

/// Handle for sending prompts into the stream
#[derive(Clone)]
pub struct PromptSender {
    tx: async_mpsc::Sender<(String, Option<String>)>,
}

impl PromptSender {
    /// Send a prompt
    pub async fn send(&self, prompt: String, identity: Option<String>) -> Result<(), String> {
        self.tx
            .send((prompt, identity))
            .await
            .map_err(|e| e.to_string())
    }

    /// Send a prompt (blocking)
    pub fn send_blocking(&self, prompt: String, identity: Option<String>) -> Result<(), String> {
        self.tx
            .blocking_send((prompt, identity))
            .map_err(|e| e.to_string())
    }
}

impl PromptStream {
    /// Create a new prompt stream and sender
    pub fn new(buffer_size: usize) -> (Self, PromptSender) {
        let (tx, rx) = async_mpsc::channel(buffer_size);

        let stream = Self {
            name: "prompt".to_string(),
            rx,
            pending: VecDeque::new(),
            active: true,
        };

        let sender = PromptSender { tx };

        (stream, sender)
    }
}

#[async_trait]
impl SensoryStream for PromptStream {
    fn name(&self) -> &str {
        &self.name
    }

    fn poll(&mut self) -> Vec<Stimulus> {
        let mut stimuli = Vec::new();

        // Check for new prompts
        while let Ok((prompt, identity)) = self.rx.try_recv() {
            stimuli.push(Stimulus::from_prompt(prompt, identity));
        }

        stimuli
    }

    fn is_active(&self) -> bool {
        self.active
    }

    fn shutdown(&mut self) {
        self.active = false;
    }
}

// ============================================================================
// DRIVE STREAM
// ============================================================================

/// Internal drive stream - generates stimuli from boredom, curiosity, goals
pub struct DriveStream {
    name: String,
    boredom: Arc<Mutex<BoredomTracker>>,
    curiosity: Arc<Mutex<CuriositySystem>>,
    goals: Arc<Mutex<GoalManager>>,
    neuromodulators: Arc<Mutex<NeuromodulatorySystem>>,
    last_check: DateTime<Utc>,
    check_interval: Duration,
    active: bool,
}

impl DriveStream {
    /// Create a new drive stream
    pub fn new(
        boredom: Arc<Mutex<BoredomTracker>>,
        curiosity: Arc<Mutex<CuriositySystem>>,
        goals: Arc<Mutex<GoalManager>>,
        neuromodulators: Arc<Mutex<NeuromodulatorySystem>>,
    ) -> Self {
        Self {
            name: "drives".to_string(),
            boredom,
            curiosity,
            goals,
            neuromodulators,
            last_check: Utc::now(),
            check_interval: Duration::seconds(5),
            active: true,
        }
    }

    /// Check boredom state
    fn check_boredom(&self) -> Option<Stimulus> {
        let mut boredom = self.boredom.lock().ok()?;
        let assessment = boredom.assess();

        if assessment.triggered {
            let rec_str = match assessment.recommendation {
                BoredomRecommendation::Continue => return None,
                BoredomRecommendation::IncreaseExploration => "increase_exploration",
                BoredomRecommendation::SwitchStrategy => "switch_strategy",
                BoredomRecommendation::RequestFreshInput => "request_fresh_input",
                BoredomRecommendation::SeekHelp => "seek_help",
                BoredomRecommendation::ContextReset => "context_reset",
            };

            Some(
                Stimulus::from_drive(
                    DriveEvent::Boredom {
                        level: assessment.level,
                        recommendation: rec_str.to_string(),
                    },
                    "boredom",
                )
                .with_salience(assessment.level),
            )
        } else {
            None
        }
    }

    /// Check curiosity urges
    fn check_curiosity(&self) -> Option<Stimulus> {
        let curiosity = self.curiosity.lock().ok()?;
        let neuro = self.neuromodulators.lock().ok()?;
        let neuro_state = neuro.state();

        // Check explore vs exploit balance
        let explore_prob = curiosity.explore_vs_exploit(&neuro_state);

        if explore_prob > 0.7 {
            // Strong exploration urge
            if let Some(domain) = curiosity.most_uncertain_domain() {
                return Some(
                    Stimulus::from_drive(
                        DriveEvent::Curiosity {
                            domain: domain.0.clone(),
                            intensity: explore_prob,
                            target: None,
                        },
                        "curiosity",
                    )
                    .with_salience(explore_prob * 0.8),
                );
            }
        }

        None
    }

    /// Check goal pressures
    fn check_goals(&self) -> Vec<Stimulus> {
        let mut stimuli = Vec::new();
        let goals = match self.goals.lock() {
            Ok(g) => g,
            Err(_) => return stimuli,
        };

        // Check for urgent goals
        for goal in goals.active_goals() {
            if let Some(deadline) = goal.deadline {
                let remaining = deadline - Utc::now();

                // Urgent if less than 10% time remaining
                if remaining < Duration::hours(1) && remaining > Duration::zero() {
                    stimuli.push(
                        Stimulus::from_drive(
                            DriveEvent::GoalPressure {
                                goal_id: goal.id,
                                urgency: 1.0 - (remaining.num_minutes() as f64 / 60.0),
                                reason: "deadline_approaching".to_string(),
                            },
                            "goals",
                        )
                        .with_priority(StimulusPriority::Elevated),
                    );
                }
            }
        }

        stimuli
    }
}

#[async_trait]
impl SensoryStream for DriveStream {
    fn name(&self) -> &str {
        &self.name
    }

    fn poll(&mut self) -> Vec<Stimulus> {
        let now = Utc::now();

        // Only check at intervals to avoid excessive computation
        if now - self.last_check < self.check_interval {
            return Vec::new();
        }
        self.last_check = now;

        let mut stimuli = Vec::new();

        // Check each drive
        if let Some(s) = self.check_boredom() {
            stimuli.push(s);
        }

        if let Some(s) = self.check_curiosity() {
            stimuli.push(s);
        }

        stimuli.extend(self.check_goals());

        stimuli
    }

    fn is_active(&self) -> bool {
        self.active
    }

    fn shutdown(&mut self) {
        self.active = false;
    }
}

// ============================================================================
// SENSORY CORTEX - Aggregates all streams
// ============================================================================

/// The sensory cortex aggregates all sensory streams
pub struct SensoryCortex {
    streams: Vec<Box<dyn SensoryStream>>,
    buffer: VecDeque<Stimulus>,
    max_buffer_size: usize,
}

impl SensoryCortex {
    /// Create a new sensory cortex
    pub fn new(max_buffer_size: usize) -> Self {
        Self {
            streams: Vec::new(),
            buffer: VecDeque::new(),
            max_buffer_size,
        }
    }

    /// Add a sensory stream
    pub fn add_stream(&mut self, stream: Box<dyn SensoryStream>) {
        self.streams.push(stream);
    }

    /// Poll all streams for stimuli
    pub fn poll_all(&mut self) -> Vec<Stimulus> {
        let mut all_stimuli = Vec::new();

        for stream in &mut self.streams {
            if stream.is_active() {
                all_stimuli.extend(stream.poll());
            }
        }

        // Add to buffer
        for stimulus in &all_stimuli {
            if self.buffer.len() >= self.max_buffer_size {
                self.buffer.pop_front();
            }
            self.buffer.push_back(stimulus.clone());
        }

        all_stimuli
    }

    /// Get stimuli sorted by priority
    pub fn poll_prioritized(&mut self) -> Vec<Stimulus> {
        let mut stimuli = self.poll_all();
        stimuli.sort_by(|a, b| b.priority.cmp(&a.priority));
        stimuli
    }

    /// Shutdown all streams
    pub fn shutdown(&mut self) {
        for stream in &mut self.streams {
            stream.shutdown();
        }
    }

    /// Get number of active streams
    pub fn active_stream_count(&self) -> usize {
        self.streams.iter().filter(|s| s.is_active()).count()
    }

    /// Get stream names
    pub fn stream_names(&self) -> Vec<&str> {
        self.streams.iter().map(|s| s.name()).collect()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::stimulus::StimulusKind;

    #[test]
    fn test_clock_stream_ticks() {
        let config = ClockConfig {
            tick_interval_ms: 10, // Fast ticks for testing
            idle_timeout_secs: 1000,
        };
        let mut stream = ClockStream::new(config);

        // First poll might not have tick
        let _ = stream.poll();

        // Wait and poll again
        std::thread::sleep(std::time::Duration::from_millis(20));
        let stimuli = stream.poll();

        // Should have at least one tick
        assert!(
            stimuli
                .iter()
                .any(|s| matches!(s.kind, StimulusKind::Time(TimeEvent::Tick { .. }))),
            "Expected tick event"
        );
    }

    #[test]
    fn test_clock_alarm() {
        let mut stream = ClockStream::new(ClockConfig::default());

        // Set alarm in the past (immediate trigger)
        stream.set_alarm("test", Utc::now() - Duration::seconds(1));

        let stimuli = stream.poll();

        assert!(
            stimuli.iter().any(|s| matches!(
                &s.kind,
                StimulusKind::Time(TimeEvent::Alarm { name }) if name == "test"
            )),
            "Expected alarm event"
        );
    }

    #[test]
    fn test_sensory_cortex() {
        let mut cortex = SensoryCortex::new(100);

        // Add clock stream
        let clock = ClockStream::new(ClockConfig {
            tick_interval_ms: 10,
            idle_timeout_secs: 1000,
        });
        cortex.add_stream(Box::new(clock));

        assert_eq!(cortex.active_stream_count(), 1);
        assert_eq!(cortex.stream_names(), vec!["clock"]);
    }

    #[tokio::test]
    async fn test_prompt_stream() {
        let (mut stream, sender) = PromptStream::new(10);

        // Send a prompt
        sender
            .send("Hello".to_string(), Some("user".to_string()))
            .await
            .unwrap();

        // Poll should receive it
        let stimuli = stream.poll();
        assert_eq!(stimuli.len(), 1);

        if let StimulusKind::ExternalPrompt { content, .. } = &stimuli[0].kind {
            assert_eq!(content, "Hello");
        } else {
            panic!("Expected ExternalPrompt");
        }
    }
}
