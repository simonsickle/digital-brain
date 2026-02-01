//! Autonomous Agent Example
//!
//! Demonstrates the consciousness loop running autonomously:
//! - Watches a directory for file changes
//! - Responds to "prompts" from a channel
//! - Mind-wanders when idle
//! - Follows curiosity to explore
//! - Gets bored and seeks novelty
//!
//! Run with: cargo run --example autonomous_agent

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::mpsc;

use digital_brain::core::{
    // Consciousness loop
    ConsciousAction, ConsciousnessConfig, ConsciousnessLoop, ConsciousnessState,
    ProcessingContext, StimulusProcessor, ActionResult,
    
    // Sensory streams
    ClockConfig, ClockStream, FileSystemConfig, FileSystemStream,
    PromptSender, PromptStream, SensoryCortex,
    
    // Boredom & Curiosity
    BoredomTracker, ActivityFingerprint,
    
    // Stimulus types
    Stimulus, StimulusKind, DriveEvent,
};

/// Our autonomous processor - this is where the "thinking" happens
/// In a real implementation, this would call an LLM
struct AutonomousProcessor {
    name: String,
    thoughts: Vec<String>,
    observations: Vec<String>,
    boredom: Arc<Mutex<BoredomTracker>>,
    files_seen: Vec<PathBuf>,
    cycle_count: u64,
}

impl AutonomousProcessor {
    fn new(name: &str, boredom: Arc<Mutex<BoredomTracker>>) -> Self {
        Self {
            name: name.to_string(),
            thoughts: Vec::new(),
            observations: Vec::new(),
            boredom,
            files_seen: Vec::new(),
            cycle_count: 0,
        }
    }

    fn log(&self, msg: &str) {
        println!("[{}] {}", self.name, msg);
    }
}

impl StimulusProcessor for AutonomousProcessor {
    fn process(
        &mut self,
        stimulus: &Stimulus,
        context: &ProcessingContext,
    ) -> Option<ConsciousAction> {
        self.cycle_count = context.cycle;
        
        // Record activity for boredom tracking
        if let Ok(mut boredom) = self.boredom.lock() {
            let fingerprint = ActivityFingerprint::new(
                &stimulus.id.to_string(),
                &format!("{:?}", std::mem::discriminant(&stimulus.kind)),
            );
            boredom.record_activity(fingerprint);
        }

        match &stimulus.kind {
            // Human prompt - highest priority
            StimulusKind::ExternalPrompt { content, .. } => {
                self.log(&format!("📨 Received prompt: {}", content));
                
                // In real implementation: send to LLM with context
                // For now, simple echo with awareness
                let response = if content.to_lowercase().contains("how are you") {
                    "I'm doing well! Currently exploring and staying curious.".to_string()
                } else if content.to_lowercase().contains("what are you doing") {
                    format!(
                        "I'm on cycle {}. I've seen {} files and had {} thoughts so far.",
                        self.cycle_count,
                        self.files_seen.len(),
                        self.thoughts.len()
                    )
                } else {
                    format!("You said: '{}'. Let me think about that...", content)
                };

                Some(ConsciousAction::Respond {
                    content: response,
                    to: Some(stimulus.id),
                })
            }

            // File system event - something changed!
            StimulusKind::FileSystem(event) => {
                let path = match event {
                    digital_brain::core::FileEvent::Created { path } => {
                        self.log(&format!("📁 New file: {:?}", path));
                        Some(path.clone())
                    }
                    digital_brain::core::FileEvent::Modified { path } => {
                        self.log(&format!("📝 File modified: {:?}", path));
                        Some(path.clone())
                    }
                    digital_brain::core::FileEvent::Deleted { path } => {
                        self.log(&format!("🗑️ File deleted: {:?}", path));
                        None
                    }
                    _ => None,
                };

                if let Some(p) = path {
                    if !self.files_seen.contains(&p) {
                        self.files_seen.push(p.clone());
                        
                        // Signal novelty to boredom tracker
                        if let Ok(mut boredom) = self.boredom.lock() {
                            boredom.signal_novelty();
                        }

                        return Some(ConsciousAction::Observe {
                            target: p.to_string_lossy().to_string(),
                        });
                    }
                }
                None
            }

            // Internal drive - curiosity, boredom, etc.
            StimulusKind::Drive(drive) => {
                match drive {
                    DriveEvent::Curiosity { domain, intensity, .. } => {
                        self.log(&format!("🔍 Curiosity about '{}' (intensity: {:.2})", domain, intensity));
                        
                        // In real implementation: explore the domain
                        Some(ConsciousAction::Observe {
                            target: domain.clone(),
                        })
                    }
                    DriveEvent::Boredom { level, recommendation } => {
                        self.log(&format!("😴 Bored (level: {:.2}) - {}", level, recommendation));
                        
                        // Take the recommended action
                        match recommendation.as_str() {
                            "increase_exploration" => {
                                Some(ConsciousAction::Observe {
                                    target: "something_new".to_string(),
                                })
                            }
                            "switch_strategy" => {
                                Some(ConsciousAction::Refocus {
                                    target: "different_approach".to_string(),
                                    reason: "boredom".to_string(),
                                })
                            }
                            "seek_help" => {
                                Some(ConsciousAction::RequestInput {
                                    prompt: "I'm stuck. What should I explore?".to_string(),
                                })
                            }
                            _ => {
                                Some(ConsciousAction::Think {
                                    thought: "Maybe I should try something different...".to_string(),
                                })
                            }
                        }
                    }
                    DriveEvent::GoalPressure { reason, urgency, .. } => {
                        self.log(&format!("🎯 Goal pressure: {} (urgency: {:.2})", reason, urgency));
                        Some(ConsciousAction::Refocus {
                            target: reason.clone(),
                            reason: "goal_deadline".to_string(),
                        })
                    }
                    _ => None,
                }
            }

            // Time tick - periodic check
            StimulusKind::Time(time_event) => {
                match time_event {
                    digital_brain::core::TimeEvent::Tick { cycle } => {
                        // Only log occasionally
                        if cycle % 50 == 0 {
                            self.log(&format!("⏰ Tick {}", cycle));
                        }
                        None
                    }
                    digital_brain::core::TimeEvent::IdleTimeout { idle_seconds } => {
                        self.log(&format!("💤 Idle for {} seconds", idle_seconds));
                        // Might trigger exploration
                        Some(ConsciousAction::Think {
                            thought: "Been idle for a while, maybe I should do something...".to_string(),
                        })
                    }
                    digital_brain::core::TimeEvent::Alarm { name } => {
                        self.log(&format!("⏰ Alarm: {}", name));
                        Some(ConsciousAction::Think {
                            thought: format!("Alarm '{}' went off", name),
                        })
                    }
                    _ => None,
                }
            }

            // Internal thought
            StimulusKind::InternalThought { content, source_region } => {
                self.log(&format!("💭 Thought from {}: {}", source_region, content));
                self.thoughts.push(content.clone());
                None
            }

            // Observation result
            StimulusKind::Observation { domain, content, novelty } => {
                self.log(&format!("👁️ Observed {} (novelty: {:.2}): {}", domain, novelty, content));
                self.observations.push(content.clone());
                
                if *novelty > 0.7 {
                    // High novelty - record it
                    if let Ok(mut boredom) = self.boredom.lock() {
                        boredom.signal_novelty();
                    }
                }
                None
            }

            _ => {
                // Unknown stimulus type
                self.log(&format!("❓ Unknown stimulus: {:?}", stimulus.kind));
                None
            }
        }
    }

    fn mind_wander(&mut self, context: &ProcessingContext) -> Option<ConsciousAction> {
        // Mind wandering - generate internal thoughts when idle
        let thoughts = [
            "I wonder what new files might appear...",
            "What patterns have I noticed?",
            "Is there something I should explore?",
            "Maybe I should review what I've learned...",
            "I feel curious about the filesystem...",
            "What would happen if I tried something different?",
        ];

        let idx = (context.cycle as usize) % thoughts.len();
        let thought = thoughts[idx].to_string();
        
        self.log(&format!("🧠 Mind wandering: {}", thought));
        self.thoughts.push(thought.clone());

        Some(ConsciousAction::Think { thought })
    }

    fn execute(&mut self, action: &ConsciousAction) -> ActionResult {
        match action {
            ConsciousAction::Respond { content, .. } => {
                println!("\n>>> RESPONSE: {}\n", content);
                ActionResult {
                    success: true,
                    output: Some(content.clone()),
                    error: None,
                    follow_up: None,
                }
            }
            ConsciousAction::Think { thought } => {
                // Internal thought - no external output
                ActionResult {
                    success: true,
                    output: Some(thought.clone()),
                    error: None,
                    follow_up: None,
                }
            }
            ConsciousAction::Observe { target } => {
                // In real implementation: actually observe the target
                self.log(&format!("Observing: {}", target));
                ActionResult {
                    success: true,
                    output: Some(format!("Observed: {}", target)),
                    error: None,
                    follow_up: None,
                }
            }
            ConsciousAction::Refocus { target, reason } => {
                self.log(&format!("Refocusing to '{}' because: {}", target, reason));
                ActionResult {
                    success: true,
                    output: None,
                    error: None,
                    follow_up: None,
                }
            }
            ConsciousAction::RequestInput { prompt } => {
                println!("\n>>> REQUESTING INPUT: {}\n", prompt);
                ActionResult {
                    success: true,
                    output: None,
                    error: None,
                    follow_up: None,
                }
            }
            _ => ActionResult {
                success: true,
                output: None,
                error: None,
                follow_up: None,
            },
        }
    }
}

#[tokio::main]
async fn main() {
    println!("🧠 Autonomous Agent Starting...\n");
    println!("This agent will:");
    println!("  - Watch the current directory for file changes");
    println!("  - Respond to prompts sent via channel");
    println!("  - Mind-wander when idle");
    println!("  - Track boredom and seek novelty");
    println!("\nPress Ctrl+C to stop.\n");
    println!("{}", "─".repeat(50));

    // Create boredom tracker (shared)
    let boredom = Arc::new(Mutex::new(BoredomTracker::new()));

    // Create sensory cortex
    let mut sensory = SensoryCortex::new(100);

    // Add clock stream (ticks, idle detection)
    let clock = ClockStream::new(ClockConfig {
        tick_interval_ms: 1000,  // 1 second ticks
        idle_timeout_secs: 30,   // Idle after 30 seconds
    });
    sensory.add_stream(Box::new(clock));

    // Add file system stream (watch current directory)
    let mut fs_stream = FileSystemStream::new(FileSystemConfig {
        watch_paths: vec![PathBuf::from(".")],
        recursive: false,
        exclude_patterns: vec![
            "*.tmp".to_string(),
            "*.swp".to_string(),
            ".git/*".to_string(),
            "target/*".to_string(),
        ],
        ..Default::default()
    });
    if let Err(e) = fs_stream.start() {
        eprintln!("Warning: Could not start file watcher: {}", e);
    }
    sensory.add_stream(Box::new(fs_stream));

    // Add prompt stream (for human input)
    let (prompt_stream, prompt_sender) = PromptStream::new(10);
    sensory.add_stream(Box::new(prompt_stream));

    // Create the processor
    let processor = AutonomousProcessor::new("Agent", boredom.clone());

    // Create consciousness loop
    let config = ConsciousnessConfig {
        cycle_interval_ms: 100,        // 10 Hz
        max_stimuli_per_cycle: 5,
        autonomous: true,
        attention_threshold: 0.2,
        focus_duration_cycles: 50,
        enable_mind_wandering: true,
        idle_threshold_cycles: 100,    // Mind-wander after 10 seconds idle
    };

    let mut consciousness = ConsciousnessLoop::new(config, sensory, Box::new(processor));
    let control = consciousness.control_handle();

    // Spawn a task to simulate occasional prompts
    let prompt_sender_clone = prompt_sender.clone();
    tokio::spawn(async move {
        // Wait a bit then send a test prompt
        tokio::time::sleep(Duration::from_secs(5)).await;
        let _ = prompt_sender_clone
            .send("Hello! How are you?".to_string(), Some("user".to_string()))
            .await;

        tokio::time::sleep(Duration::from_secs(10)).await;
        let _ = prompt_sender_clone
            .send("What are you doing?".to_string(), Some("user".to_string()))
            .await;
    });

    // Spawn a task to create test files periodically
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_secs(15)).await;
            let filename = format!("test_file_{}.txt", chrono::Utc::now().timestamp());
            if let Ok(_) = tokio::fs::write(&filename, "Hello from the test!").await {
                println!("\n[Test] Created file: {}", filename);
                // Clean up after a bit
                tokio::time::sleep(Duration::from_secs(5)).await;
                let _ = tokio::fs::remove_file(&filename).await;
            }
        }
    });

    // Handle Ctrl+C
    let control_clone = control.clone();
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        println!("\n\n🛑 Shutting down...");
        control_clone.stop();
    });

    // Run the consciousness loop
    println!("\n🚀 Consciousness loop starting...\n");
    consciousness.run().await;

    // Print final stats
    let stats = consciousness.stats();
    println!("\n{}", "─".repeat(50));
    println!("📊 Final Statistics:");
    println!("  Total cycles: {}", stats.total_cycles);
    println!("  Stimuli received: {}", stats.stimuli_received);
    println!("  Stimuli processed: {}", stats.stimuli_processed);
    println!("  Actions taken: {}", stats.actions_taken);
    println!("  Average cycle time: {:.2}ms", stats.average_cycle_ms);
}
