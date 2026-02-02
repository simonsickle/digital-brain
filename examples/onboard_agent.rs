//! Agent Onboarding Example
//!
//! Interactive terminal wizard for creating and configuring a new digital brain agent.
//!
//! Run with:
//! ```bash
//! cargo run --example onboard_agent
//! ```
//!
//! For quick mode (fewer prompts):
//! ```bash
//! cargo run --example onboard_agent -- --quick
//! ```

use digital_brain::Result;
use digital_brain::onboarding::{OnboardingConfig, OnboardingWizard};
use std::env;

fn main() -> Result<()> {
    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    let quick_mode = args.iter().any(|a| a == "--quick" || a == "-q");
    let save_path = args
        .iter()
        .position(|a| a == "--save" || a == "-s")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.to_string());
    let skip_validation = args.iter().any(|a| a == "--no-validate");

    // Build configuration
    let mut config = OnboardingConfig::new();

    if quick_mode {
        config = config.quick();
    }

    if let Some(path) = save_path {
        config = config.save_to(path);
    }

    if skip_validation {
        config = config.skip_validation();
    }

    // Run the wizard
    let wizard = OnboardingWizard::with_config(config);
    let setup = wizard.run()?;

    // Ask if user wants to start a REPL session
    println!();
    println!(
        "Would you like to start an interactive session with {}?",
        setup.name
    );
    print!("  (y/n): ");
    std::io::Write::flush(&mut std::io::stdout()).ok();

    let mut input = String::new();
    std::io::stdin().read_line(&mut input).ok();

    if input.trim().to_lowercase() == "y" || input.trim().to_lowercase() == "yes" {
        run_repl(setup)?;
    } else {
        println!();
        println!("Setup complete! You can create the brain later with:");
        println!();
        println!("  let brain = setup.create_brain()?;");
        println!();
    }

    Ok(())
}

/// Run a simple REPL session with the configured brain.
fn run_repl(setup: digital_brain::AgentSetup) -> Result<()> {
    use std::io::{BufRead, Write};

    println!();
    println!("Creating brain...");

    let mut brain = setup.create_brain()?;

    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!(
        "║              {} is now alive!                         ",
        setup.name
    );
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("Commands: process <text>, recall <query>, who, stats, quit");
    println!();

    let stdin = std::io::stdin();
    loop {
        print!("{}> ", setup.name.to_lowercase());
        std::io::stdout().flush().ok();

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
        let arg = parts.get(1).copied().unwrap_or("");

        match cmd.as_str() {
            "quit" | "exit" | "q" => {
                println!("Goodbye from {}!", setup.name);
                break;
            }

            "who" => {
                println!("{}", brain.who_am_i());
            }

            "stats" | "s" => {
                let stats = brain.stats();
                println!("Cycles: {}", stats.cycles);
                println!("Memories: {}", stats.memories);
                println!("Working memory items: {}", stats.working_memory_items);
                println!("Beliefs: {}", stats.beliefs);
                println!("Learning rate: {:.3}", stats.learning_rate);
            }

            "process" | "p" => {
                if arg.is_empty() {
                    println!("Usage: process <text>");
                    continue;
                }
                match brain.process(arg) {
                    Ok(result) => {
                        println!("Processed: {:?}", result.emotion);
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
                                let preview = if content.len() > 60 {
                                    &content[..60]
                                } else {
                                    &content
                                };
                                println!(
                                    "  {}. {} (v={:+.2})",
                                    i + 1,
                                    preview,
                                    mem.valence.value()
                                );
                            }
                        }
                    }
                    Err(e) => println!("Error: {}", e),
                }
            }

            "reflect" => {
                let topic = if arg.is_empty() {
                    "my current state"
                } else {
                    arg
                };
                let reflection = brain.reflect(topic);
                println!("{}", reflection);
            }

            "help" | "?" => {
                println!("Commands:");
                println!("  process <text>  - Process input through the brain");
                println!("  recall <query>  - Search memories");
                println!("  reflect [topic] - Generate reflection");
                println!("  who             - Who am I?");
                println!("  stats           - Show brain statistics");
                println!("  quit            - Exit the REPL");
            }

            _ => {
                // Treat unknown commands as things to process
                match brain.process(line) {
                    Ok(result) => {
                        println!("Processed: {:?}", result.emotion);
                    }
                    Err(e) => println!("Error: {}", e),
                }
            }
        }
        println!();
    }

    Ok(())
}
