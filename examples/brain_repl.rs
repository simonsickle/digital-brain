//! Brain REPL - Interactive Brain Session
//!
//! A simple command-line interface for interacting with a digital brain.
//! Useful for demos and experimentation.
//!
//! Commands:
//! - process <text>  - Process input through the brain
//! - recall <query>  - Search memories
//! - reflect <topic> - Generate reflection on a topic
//! - sleep <hours>   - Run sleep cycle
//! - stats           - Show brain statistics
//! - who             - Ask "who am I?"
//! - identity <name> - Set brain identity
//! - believe <text>  - Add a belief
//! - save <path>     - Save brain state
//! - load <path>     - Load brain state
//! - help            - Show help
//! - quit            - Exit

use digital_brain::Brain;
use digital_brain::regions::dmn::{Identity, BeliefCategory};
use std::io::{self, Write, BufRead};

fn print_help() {
    println!("
╔══════════════════════════════════════════════════════════════╗
║                    BRAIN REPL COMMANDS                       ║
╚══════════════════════════════════════════════════════════════╝

  process <text>   Process input through the brain
  recall <query>   Search memories for relevant content
  reflect <topic>  Generate reflection on a topic
  sleep <hours>    Run sleep/consolidation cycle (default: 8)
  stats            Show brain statistics
  who              Ask 'who am I?'
  introspect       Full brain introspection report
  identity <name>  Set brain identity
  believe <text>   Add a belief about self-capability
  save <path>      Save brain state to directory
  load <path>      Load brain state from directory
  help             Show this help
  quit             Exit the REPL
");
}

fn main() -> digital_brain::Result<()> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║              🧠 DIGITAL BRAIN REPL 🧠                        ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("Type 'help' for commands, 'quit' to exit.");
    println!();

    let mut brain = Brain::new()?;

    // Set a default identity
    brain.set_identity(Identity {
        name: "REPL Brain".to_string(),
        core_values: vec!["curiosity".to_string(), "learning".to_string()],
        self_description: "An interactive digital brain".to_string(),
        creation_time: chrono::Utc::now(),
    });

    let stdin = io::stdin();
    loop {
        print!("brain> ");
        io::stdout().flush().ok();

        let mut line = String::new();
        if stdin.lock().read_line(&mut line).is_err() {
            break;
        }

        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.splitn(2, ' ').collect();
        let cmd = parts[0].to_lowercase();
        let arg = parts.get(1).map(|s| *s).unwrap_or("");

        match cmd.as_str() {
            "quit" | "exit" | "q" => {
                println!("Goodbye! 👋");
                break;
            }

            "help" | "?" => {
                print_help();
            }

            "process" | "p" => {
                if arg.is_empty() {
                    println!("Usage: process <text>");
                    continue;
                }
                match brain.process(arg) {
                    Ok(result) => {
                        println!("✓ Processed: {:?}", result.emotion);
                        if let Some(mem) = result.memory {
                            println!("  Memory strength: {:.2}", mem.strength);
                        }
                    }
                    Err(e) => println!("Error: {}", e),
                }
            }

            "recall" | "r" => {
                if arg.is_empty() {
                    println!("Usage: recall <query>");
                    continue;
                }
                match brain.recall(arg, 5) {
                    Ok(memories) => {
                        if memories.is_empty() {
                            println!("No memories found for '{}'", arg);
                        } else {
                            println!("Found {} memories:", memories.len());
                            for (i, mem) in memories.iter().enumerate() {
                                let content: String = serde_json::from_value(mem.content.clone())
                                    .unwrap_or_else(|_| "?".to_string());
                                let preview = if content.len() > 60 { &content[..60] } else { &content };
                                println!("  {}. {} (v={:+.2})", i + 1, preview, mem.valence.value());
                            }
                        }
                    }
                    Err(e) => println!("Error: {}", e),
                }
            }

            "reflect" => {
                let topic = if arg.is_empty() { "my current state" } else { arg };
                let reflection = brain.reflect(topic);
                println!("💭 {}", reflection);
            }

            "sleep" => {
                let hours: f64 = arg.parse().unwrap_or(8.0);
                match brain.sleep(hours) {
                    Ok(report) => {
                        println!("😴 Slept for {:.1} hours", report.hours_slept);
                        println!("   Consolidated: {} memories", report.memories_consolidated);
                        println!("   Forgotten: {} memories", report.memories_forgotten);
                    }
                    Err(e) => println!("Error: {}", e),
                }
            }

            "stats" | "s" => {
                let stats = brain.stats();
                println!("📊 Brain Statistics:");
                println!("   Cycles: {}", stats.cycles);
                println!("   Memories: {}", stats.memories);
                println!("   Working memory items: {}", stats.working_memory_items);
                println!("   Beliefs: {}", stats.beliefs);
                println!("   Learning rate: {:.3}", stats.learning_rate);
                println!("   Dopamine: {:.2}", stats.neuromodulators.dopamine);
                println!("   Norepinephrine: {:.2}", stats.neuromodulators.norepinephrine);
            }

            "who" => {
                let identity = brain.who_am_i();
                println!("🪪 {}", identity);
            }

            "introspect" | "i" => {
                let report = brain.introspect();
                println!("{}", report);
            }

            "identity" | "id" => {
                if arg.is_empty() {
                    println!("Usage: identity <name>");
                    continue;
                }
                brain.set_identity(Identity {
                    name: arg.to_string(),
                    core_values: vec!["curiosity".to_string()],
                    self_description: format!("I am {}", arg),
                    creation_time: chrono::Utc::now(),
                });
                println!("✓ Identity set to '{}'", arg);
            }

            "believe" | "b" => {
                if arg.is_empty() {
                    println!("Usage: believe <text>");
                    continue;
                }
                brain.believe(arg, BeliefCategory::SelfCapability, 0.8);
                println!("✓ Added belief: '{}'", arg);
            }

            "save" => {
                if arg.is_empty() {
                    println!("Usage: save <directory_path>");
                    continue;
                }
                match brain.save_to_dir(arg) {
                    Ok(()) => println!("✓ Saved to '{}'", arg),
                    Err(e) => println!("Error: {}", e),
                }
            }

            "load" => {
                if arg.is_empty() {
                    println!("Usage: load <directory_path>");
                    continue;
                }
                match Brain::load_from_dir(arg, None) {
                    Ok(loaded) => {
                        brain = loaded;
                        println!("✓ Loaded from '{}'", arg);
                    }
                    Err(e) => println!("Error: {}", e),
                }
            }

            _ => {
                println!("Unknown command '{}'. Type 'help' for commands.", cmd);
            }
        }
        println!();
    }

    Ok(())
}
