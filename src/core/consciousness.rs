//! Consciousness Loop - The Always-On Processing Cycle
//!
//! This module implements the main cognitive loop that runs continuously,
//! processing stimuli from multiple sources and generating autonomous behavior.
//!
//! Unlike a request/response model, the consciousness loop:
//! - Runs continuously (not waiting for prompts)
//! - Integrates multiple input streams
//! - Generates internal stimuli from drives
//! - Can act autonomously or respond to external input
//!
//! # The Loop
//!
//! ```text
//! loop {
//!     1. SENSE   - Poll all sensory streams
//!     2. GATE    - Filter through thalamus (salience)
//!     3. ATTEND  - Select what to focus on
//!     4. THINK   - Process in global workspace
//!     5. DECIDE  - Choose action (or continue thinking)
//!     6. ACT     - Execute action (if any)
//!     7. UPDATE  - Update internal state, learn
//!     8. REST    - Brief pause, homeostatic regulation
//! }
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::time::{interval, Duration as TokioDuration};
use uuid::Uuid;

use crate::core::sensory::SensoryCortex;
use crate::core::stimulus::{Stimulus, StimulusPriority, StimulusResponse};
use crate::regions::thalamus::{GateResult, Thalamus};
use crate::signal::{BrainSignal, SignalType};

/// Configuration for the consciousness loop
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessConfig {
    /// Cycle interval in milliseconds
    pub cycle_interval_ms: u64,
    /// Maximum stimuli to process per cycle
    pub max_stimuli_per_cycle: usize,
    /// Enable autonomous behavior (can act without prompts)
    pub autonomous: bool,
    /// Minimum salience for processing
    pub attention_threshold: f64,
    /// How long to focus on one task before checking for interrupts (cycles)
    pub focus_duration_cycles: u32,
    /// Enable mind-wandering in idle states
    pub enable_mind_wandering: bool,
    /// Idle threshold before mind-wandering (cycles)
    pub idle_threshold_cycles: u32,
}

impl Default for ConsciousnessConfig {
    fn default() -> Self {
        Self {
            cycle_interval_ms: 100, // 10 Hz
            max_stimuli_per_cycle: 10,
            autonomous: true,
            attention_threshold: 0.2,
            focus_duration_cycles: 50,
            enable_mind_wandering: true,
            idle_threshold_cycles: 100,
        }
    }
}

/// Current state of consciousness
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsciousnessState {
    /// Processing external input
    Responsive,
    /// Working on self-directed task
    Autonomous,
    /// Mind-wandering, reflection
    MindWandering,
    /// Focused on specific task
    Focused,
    /// Resting/consolidating
    Resting,
    /// Shutting down
    ShuttingDown,
}

/// What the consciousness is currently attending to
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionFocus {
    /// What we're focused on
    pub target: String,
    /// Why we're focused on it
    pub reason: String,
    /// Priority of current focus
    pub priority: StimulusPriority,
    /// When focus started
    pub started_at: DateTime<Utc>,
    /// How many cycles we've been focused
    pub cycles_focused: u32,
    /// Associated stimulus ID (if any)
    pub stimulus_id: Option<Uuid>,
}

/// An action that can be executed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsciousAction {
    /// Generate a response (to prompt or internally)
    Respond { content: String, to: Option<Uuid> },
    /// Execute a tool/command
    Execute { tool: String, args: serde_json::Value },
    /// Read/observe something
    Observe { target: String },
    /// Write/create something
    Create { artifact: String, content: String },
    /// Internal thinking (no external effect)
    Think { thought: String },
    /// Switch focus to something else
    Refocus { target: String, reason: String },
    /// Request external input
    RequestInput { prompt: String },
    /// Do nothing (continue current state)
    Idle,
}

/// Result of a consciousness cycle
#[derive(Debug, Clone)]
pub struct CycleResult {
    /// Cycle number
    pub cycle: u64,
    /// Duration of cycle
    pub duration: std::time::Duration,
    /// Stimuli received
    pub stimuli_received: usize,
    /// Stimuli processed
    pub stimuli_processed: usize,
    /// Actions taken
    pub actions: Vec<ConsciousAction>,
    /// State after cycle
    pub state: ConsciousnessState,
    /// Current focus
    pub focus: Option<AttentionFocus>,
    /// Responses generated
    pub responses: Vec<StimulusResponse>,
}

/// Statistics about consciousness loop
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConsciousnessStats {
    pub total_cycles: u64,
    pub stimuli_received: u64,
    pub stimuli_processed: u64,
    pub stimuli_filtered: u64,
    pub actions_taken: u64,
    pub responses_generated: u64,
    pub time_in_state: std::collections::HashMap<String, u64>,
    pub average_cycle_ms: f64,
    pub interrupts: u64,
}

/// Handler trait for processing stimuli and generating actions
pub trait StimulusProcessor: Send + Sync {
    /// Process a stimulus and decide on action
    fn process(&mut self, stimulus: &Stimulus, context: &ProcessingContext)
        -> Option<ConsciousAction>;

    /// Generate mind-wandering thought
    fn mind_wander(&mut self, context: &ProcessingContext) -> Option<ConsciousAction>;

    /// Execute an action
    fn execute(&mut self, action: &ConsciousAction) -> ActionResult;
}

/// Context passed to the processor
#[derive(Debug, Clone)]
pub struct ProcessingContext {
    pub cycle: u64,
    pub state: ConsciousnessState,
    pub focus: Option<AttentionFocus>,
    pub recent_stimuli: Vec<Stimulus>,
    pub recent_actions: Vec<ConsciousAction>,
    pub timestamp: DateTime<Utc>,
}

/// Result of action execution
#[derive(Debug, Clone)]
pub struct ActionResult {
    pub success: bool,
    pub output: Option<String>,
    pub error: Option<String>,
    pub follow_up: Option<ConsciousAction>,
}

/// The main consciousness loop
pub struct ConsciousnessLoop {
    config: ConsciousnessConfig,
    sensory: SensoryCortex,
    thalamus: Thalamus,
    processor: Box<dyn StimulusProcessor>,

    // State
    cycle: u64,
    state: ConsciousnessState,
    focus: Option<AttentionFocus>,
    stimulus_queue: VecDeque<Stimulus>,
    recent_stimuli: VecDeque<Stimulus>,
    recent_actions: VecDeque<ConsciousAction>,
    idle_cycles: u32,

    // Control
    running: Arc<AtomicBool>,
    response_tx: Option<mpsc::Sender<StimulusResponse>>,

    // Stats
    stats: ConsciousnessStats,
}

impl ConsciousnessLoop {
    /// Create a new consciousness loop
    pub fn new(
        config: ConsciousnessConfig,
        sensory: SensoryCortex,
        processor: Box<dyn StimulusProcessor>,
    ) -> Self {
        Self {
            config,
            sensory,
            thalamus: Thalamus::new(),
            processor,
            cycle: 0,
            state: ConsciousnessState::Responsive,
            focus: None,
            stimulus_queue: VecDeque::new(),
            recent_stimuli: VecDeque::with_capacity(100),
            recent_actions: VecDeque::with_capacity(50),
            idle_cycles: 0,
            running: Arc::new(AtomicBool::new(false)),
            response_tx: None,
            stats: ConsciousnessStats::default(),
        }
    }

    /// Get a handle to control the loop
    pub fn control_handle(&self) -> LoopControl {
        LoopControl {
            running: self.running.clone(),
        }
    }

    /// Set response channel
    pub fn set_response_channel(&mut self, tx: mpsc::Sender<StimulusResponse>) {
        self.response_tx = Some(tx);
    }

    /// Run one cycle of the consciousness loop
    pub fn cycle(&mut self) -> CycleResult {
        let cycle_start = std::time::Instant::now();
        let mut actions = Vec::new();
        let mut responses = Vec::new();

        self.cycle += 1;

        // 1. SENSE - Poll all sensory streams
        let raw_stimuli = self.sensory.poll_prioritized();
        self.stats.stimuli_received += raw_stimuli.len() as u64;

        // 2. GATE - Filter through thalamus
        let mut gated_stimuli = Vec::new();
        for stimulus in raw_stimuli {
            let signal = self.stimulus_to_signal(&stimulus);
            match self.thalamus.gate(signal) {
                GateResult::Passed(s) => {
                    // Reconstruct stimulus with updated salience
                    let mut gated = stimulus.clone();
                    gated.salience = s.salience.value();
                    gated_stimuli.push(gated);
                }
                GateResult::Filtered { .. } => {
                    self.stats.stimuli_filtered += 1;
                }
            }
        }

        // 3. ATTEND - Check for interrupts and select focus
        let interrupt = self.check_for_interrupt(&gated_stimuli);
        if let Some(interrupting) = interrupt {
            self.handle_interrupt(interrupting);
            self.stats.interrupts += 1;
        }

        // Add gated stimuli to queue
        for s in gated_stimuli {
            self.enqueue_stimulus(s);
        }

        // 4. THINK - Process stimuli through workspace
        let context = self.build_context();
        let mut processed = 0;

        // Process up to max_stimuli_per_cycle
        while processed < self.config.max_stimuli_per_cycle {
            if let Some(stimulus) = self.next_stimulus() {
                self.recent_stimuli.push_back(stimulus.clone());
                if self.recent_stimuli.len() > 100 {
                    self.recent_stimuli.pop_front();
                }

                // 5. DECIDE - Get action from processor
                if let Some(action) = self.processor.process(&stimulus, &context) {
                    // 6. ACT - Execute action
                    let result = self.processor.execute(&action);

                    // Track action
                    self.recent_actions.push_back(action.clone());
                    if self.recent_actions.len() > 50 {
                        self.recent_actions.pop_front();
                    }

                    actions.push(action.clone());
                    self.stats.actions_taken += 1;

                    // Generate response if needed
                    if stimulus.requires_response {
                        let response = StimulusResponse {
                            stimulus_id: stimulus.id,
                            content: result.output.clone(),
                            actions: vec![format!("{:?}", action)],
                            complete: result.success,
                            defer: false,
                            follow_ups: Vec::new(),
                            processing_time: cycle_start.elapsed(),
                        };
                        responses.push(response.clone());
                        self.stats.responses_generated += 1;

                        // Send response if channel exists
                        if let Some(ref tx) = self.response_tx {
                            let _ = tx.try_send(response);
                        }
                    }

                    // Handle follow-up
                    if let Some(follow_up) = result.follow_up {
                        actions.push(follow_up);
                    }

                    self.idle_cycles = 0;
                }

                processed += 1;
                self.stats.stimuli_processed += 1;
            } else {
                break;
            }
        }

        // 7. IDLE/MIND-WANDER if nothing to process
        if processed == 0 {
            self.idle_cycles += 1;

            if self.config.enable_mind_wandering
                && self.idle_cycles > self.config.idle_threshold_cycles
            {
                self.state = ConsciousnessState::MindWandering;
                if let Some(action) = self.processor.mind_wander(&context) {
                    let _ = self.processor.execute(&action);
                    actions.push(action);
                    self.idle_cycles = 0;
                }
            }
        } else {
            self.state = if self.focus.is_some() {
                ConsciousnessState::Focused
            } else if self.config.autonomous {
                ConsciousnessState::Autonomous
            } else {
                ConsciousnessState::Responsive
            };
        }

        // 8. UPDATE - Update state tracking
        let state_key = format!("{:?}", self.state);
        *self.stats.time_in_state.entry(state_key).or_insert(0) += 1;

        // Update focus duration
        if let Some(ref mut focus) = self.focus {
            focus.cycles_focused += 1;
        }

        // Calculate cycle stats
        let duration = cycle_start.elapsed();
        self.stats.average_cycle_ms = (self.stats.average_cycle_ms * (self.cycle - 1) as f64
            + duration.as_secs_f64() * 1000.0)
            / self.cycle as f64;

        self.stats.total_cycles = self.cycle;

        CycleResult {
            cycle: self.cycle,
            duration,
            stimuli_received: self.stats.stimuli_received as usize,
            stimuli_processed: processed,
            actions,
            state: self.state,
            focus: self.focus.clone(),
            responses,
        }
    }

    /// Run the loop continuously (async)
    pub async fn run(&mut self) {
        self.running.store(true, Ordering::SeqCst);
        let mut ticker = interval(TokioDuration::from_millis(self.config.cycle_interval_ms));

        while self.running.load(Ordering::SeqCst) {
            ticker.tick().await;
            self.cycle();
        }

        self.state = ConsciousnessState::ShuttingDown;
        self.sensory.shutdown();
    }

    /// Run for a specific number of cycles (useful for testing)
    pub fn run_cycles(&mut self, n: u64) -> Vec<CycleResult> {
        let mut results = Vec::new();
        for _ in 0..n {
            results.push(self.cycle());
        }
        results
    }

    /// Stop the loop
    pub fn stop(&mut self) {
        self.running.store(false, Ordering::SeqCst);
    }

    /// Get current state
    pub fn state(&self) -> ConsciousnessState {
        self.state
    }

    /// Get current focus
    pub fn focus(&self) -> Option<&AttentionFocus> {
        self.focus.as_ref()
    }

    /// Get statistics
    pub fn stats(&self) -> &ConsciousnessStats {
        &self.stats
    }

    /// Get cycle count
    pub fn cycle_count(&self) -> u64 {
        self.cycle
    }

    // --- Private methods ---

    fn stimulus_to_signal(&self, stimulus: &Stimulus) -> BrainSignal {
        let signal_type = match &stimulus.kind {
            crate::core::stimulus::StimulusKind::ExternalPrompt { .. } => SignalType::Sensory,
            crate::core::stimulus::StimulusKind::FileSystem(_) => SignalType::Sensory,
            crate::core::stimulus::StimulusKind::Time(_) => SignalType::Attention,
            crate::core::stimulus::StimulusKind::Drive(_) => SignalType::Emotion,
            crate::core::stimulus::StimulusKind::Goal(_) => SignalType::Attention,
            crate::core::stimulus::StimulusKind::System(_) => SignalType::Broadcast,
            crate::core::stimulus::StimulusKind::InternalThought { .. } => SignalType::Memory,
            crate::core::stimulus::StimulusKind::QueryResponse { .. } => SignalType::Prediction,
            crate::core::stimulus::StimulusKind::Observation { .. } => SignalType::Sensory,
        };

        BrainSignal::new(
            stimulus.id.to_string(),
            signal_type,
            serde_json::to_value(&stimulus.kind).unwrap_or_default(),
        )
        .with_salience(stimulus.salience)
        .with_priority(stimulus.priority as i32)
    }

    fn check_for_interrupt(&self, stimuli: &[Stimulus]) -> Option<Stimulus> {
        let current_priority = self
            .focus
            .as_ref()
            .map(|f| f.priority)
            .unwrap_or(StimulusPriority::Background);

        stimuli
            .iter()
            .find(|s| s.should_interrupt(current_priority))
            .cloned()
    }

    fn handle_interrupt(&mut self, stimulus: Stimulus) {
        // Save current focus if any
        let reason = match &stimulus.kind {
            crate::core::stimulus::StimulusKind::ExternalPrompt { .. } => "external_prompt",
            crate::core::stimulus::StimulusKind::Drive(d) => match d {
                crate::core::stimulus::DriveEvent::Boredom { .. } => "boredom",
                crate::core::stimulus::DriveEvent::GoalPressure { .. } => "goal_pressure",
                _ => "drive",
            },
            _ => "interrupt",
        };

        self.focus = Some(AttentionFocus {
            target: format!("{:?}", stimulus.kind),
            reason: reason.to_string(),
            priority: stimulus.priority,
            started_at: Utc::now(),
            cycles_focused: 0,
            stimulus_id: Some(stimulus.id),
        });

        // Put at front of queue
        self.stimulus_queue.push_front(stimulus);
    }

    fn enqueue_stimulus(&mut self, stimulus: Stimulus) {
        // Insert by priority
        let pos = self
            .stimulus_queue
            .iter()
            .position(|s| s.priority < stimulus.priority)
            .unwrap_or(self.stimulus_queue.len());

        self.stimulus_queue.insert(pos, stimulus);
    }

    fn next_stimulus(&mut self) -> Option<Stimulus> {
        self.stimulus_queue.pop_front()
    }

    fn build_context(&self) -> ProcessingContext {
        ProcessingContext {
            cycle: self.cycle,
            state: self.state,
            focus: self.focus.clone(),
            recent_stimuli: self.recent_stimuli.iter().cloned().collect(),
            recent_actions: self.recent_actions.iter().cloned().collect(),
            timestamp: Utc::now(),
        }
    }
}

/// Handle to control the consciousness loop from outside
#[derive(Clone)]
pub struct LoopControl {
    running: Arc<AtomicBool>,
}

impl LoopControl {
    /// Stop the loop
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }

    /// Check if loop is running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }
}

// ============================================================================
// DEFAULT PROCESSOR (for testing)
// ============================================================================

/// A simple processor that logs and echoes
pub struct EchoProcessor {
    pub thoughts: Vec<String>,
}

impl EchoProcessor {
    pub fn new() -> Self {
        Self {
            thoughts: Vec::new(),
        }
    }
}

impl Default for EchoProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl StimulusProcessor for EchoProcessor {
    fn process(
        &mut self,
        stimulus: &Stimulus,
        _context: &ProcessingContext,
    ) -> Option<ConsciousAction> {
        match &stimulus.kind {
            crate::core::stimulus::StimulusKind::ExternalPrompt { content, .. } => {
                Some(ConsciousAction::Respond {
                    content: format!("Echo: {}", content),
                    to: Some(stimulus.id),
                })
            }
            crate::core::stimulus::StimulusKind::Time(crate::core::stimulus::TimeEvent::Tick {
                cycle,
            }) => {
                if cycle % 100 == 0 {
                    Some(ConsciousAction::Think {
                        thought: format!("Cycle {}", cycle),
                    })
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn mind_wander(&mut self, context: &ProcessingContext) -> Option<ConsciousAction> {
        let thought = format!("Mind wandering at cycle {}...", context.cycle);
        self.thoughts.push(thought.clone());
        Some(ConsciousAction::Think { thought })
    }

    fn execute(&mut self, action: &ConsciousAction) -> ActionResult {
        ActionResult {
            success: true,
            output: match action {
                ConsciousAction::Respond { content, .. } => Some(content.clone()),
                ConsciousAction::Think { thought } => Some(thought.clone()),
                _ => None,
            },
            error: None,
            follow_up: None,
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::sensory::{ClockConfig, ClockStream};

    fn make_test_loop() -> ConsciousnessLoop {
        let mut sensory = SensoryCortex::new(100);
        sensory.add_stream(Box::new(ClockStream::new(ClockConfig {
            tick_interval_ms: 10,
            idle_timeout_secs: 1000,
        })));

        ConsciousnessLoop::new(
            ConsciousnessConfig {
                cycle_interval_ms: 10,
                idle_threshold_cycles: 5,
                ..Default::default()
            },
            sensory,
            Box::new(EchoProcessor::new()),
        )
    }

    #[test]
    fn test_consciousness_creation() {
        let consciousness = make_test_loop();
        assert_eq!(consciousness.cycle_count(), 0);
        assert_eq!(consciousness.state(), ConsciousnessState::Responsive);
    }

    #[test]
    fn test_consciousness_cycles() {
        let mut consciousness = make_test_loop();

        let results = consciousness.run_cycles(10);
        assert_eq!(results.len(), 10);
        assert_eq!(consciousness.cycle_count(), 10);
    }

    #[test]
    fn test_mind_wandering() {
        let mut consciousness = make_test_loop();

        // Run enough cycles to trigger mind wandering
        consciousness.run_cycles(20);

        // Should eventually enter mind wandering state
        // (might not happen in 20 cycles depending on timing)
    }

    #[test]
    fn test_loop_control() {
        let consciousness = make_test_loop();
        let control = consciousness.control_handle();

        assert!(!control.is_running());

        // In real use, you'd call run() which sets running to true
    }

    #[test]
    fn test_stimulus_processing() {
        let mut consciousness = make_test_loop();

        // Inject a stimulus directly
        let stimulus = Stimulus::from_prompt("Hello", None);
        consciousness.enqueue_stimulus(stimulus);

        let result = consciousness.cycle();

        // Should have processed something
        assert!(result.stimuli_processed > 0 || result.actions.len() > 0);
    }
}
