//! Terminal UI Helpers
//!
//! Provides styled terminal output and input prompts for the onboarding wizard.

use std::io::{self, BufRead, Write};

/// Terminal styling configuration.
#[derive(Debug, Clone)]
pub struct TerminalStyle {
    /// Whether to use colors
    pub use_colors: bool,
    /// Whether to use box drawing characters
    pub use_boxes: bool,
    /// Whether to use unicode symbols
    pub use_unicode: bool,
}

impl Default for TerminalStyle {
    fn default() -> Self {
        Self {
            use_colors: true,
            use_boxes: true,
            use_unicode: true,
        }
    }
}

impl TerminalStyle {
    pub fn plain() -> Self {
        Self {
            use_colors: false,
            use_boxes: false,
            use_unicode: false,
        }
    }
}

/// Terminal UI helper for the onboarding wizard.
pub struct TerminalUI {
    style: TerminalStyle,
}

impl TerminalUI {
    pub fn new() -> Self {
        Self {
            style: TerminalStyle::default(),
        }
    }

    pub fn with_style(style: TerminalStyle) -> Self {
        Self { style }
    }

    /// Print a styled header.
    pub fn header(&self, title: &str) {
        println!();
        if self.style.use_boxes {
            let width = title.len() + 4;
            let top = format!(
                "{}{}{}",
                self.top_left(),
                self.horizontal(width),
                self.top_right()
            );
            let bottom = format!(
                "{}{}{}",
                self.bottom_left(),
                self.horizontal(width),
                self.bottom_right()
            );
            println!("{}", top);
            println!("{} {} {}", self.vertical(), title, self.vertical());
            println!("{}", bottom);
        } else {
            println!("=== {} ===", title);
        }
        println!();
    }

    /// Print a section header.
    pub fn section(&self, title: &str) {
        println!();
        if self.style.use_unicode {
            println!("{} {}", self.arrow(), title);
        } else {
            println!(">> {}", title);
        }
        println!();
    }

    /// Print an info message.
    pub fn info(&self, message: &str) {
        if self.style.use_unicode {
            println!("  {} {}", self.info_icon(), message);
        } else {
            println!("  [i] {}", message);
        }
    }

    /// Print a success message.
    pub fn success(&self, message: &str) {
        if self.style.use_unicode {
            println!("  {} {}", self.success_icon(), message);
        } else {
            println!("  [+] {}", message);
        }
    }

    /// Print a warning message.
    pub fn warning(&self, message: &str) {
        if self.style.use_unicode {
            println!("  {} {}", self.warning_icon(), message);
        } else {
            println!("  [!] {}", message);
        }
    }

    /// Print an error message.
    pub fn error(&self, message: &str) {
        if self.style.use_unicode {
            println!("  {} {}", self.error_icon(), message);
        } else {
            println!("  [X] {}", message);
        }
    }

    /// Print a hint/tip.
    pub fn hint(&self, message: &str) {
        if self.style.use_unicode {
            println!("  {} {}", self.hint_icon(), message);
        } else {
            println!("  [?] {}", message);
        }
    }

    /// Print a list item.
    pub fn list_item(&self, index: usize, item: &str) {
        println!("  {}. {}", index + 1, item);
    }

    /// Print a bullet point.
    pub fn bullet(&self, item: &str) {
        if self.style.use_unicode {
            println!("    {} {}", self.bullet_icon(), item);
        } else {
            println!("    - {}", item);
        }
    }

    /// Print blank line.
    pub fn blank(&self) {
        println!();
    }

    /// Print a divider line.
    pub fn divider(&self) {
        if self.style.use_boxes {
            println!("{}", self.horizontal(60));
        } else {
            println!("{}", "-".repeat(60));
        }
    }

    /// Prompt for text input.
    pub fn prompt(&self, message: &str) -> io::Result<String> {
        if self.style.use_unicode {
            print!("  {} {} ", self.prompt_icon(), message);
        } else {
            print!("  > {} ", message);
        }
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().lock().read_line(&mut input)?;
        Ok(input.trim().to_string())
    }

    /// Prompt for text input with a default value.
    pub fn prompt_default(&self, message: &str, default: &str) -> io::Result<String> {
        if self.style.use_unicode {
            print!("  {} {} [{}]: ", self.prompt_icon(), message, default);
        } else {
            print!("  > {} [{}]: ", message, default);
        }
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().lock().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() {
            Ok(default.to_string())
        } else {
            Ok(input.to_string())
        }
    }

    /// Prompt for a yes/no confirmation.
    pub fn confirm(&self, message: &str, default: bool) -> io::Result<bool> {
        let default_str = if default { "Y/n" } else { "y/N" };
        if self.style.use_unicode {
            print!("  {} {} [{}]: ", self.prompt_icon(), message, default_str);
        } else {
            print!("  > {} [{}]: ", message, default_str);
        }
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().lock().read_line(&mut input)?;
        let input = input.trim().to_lowercase();

        Ok(match input.as_str() {
            "y" | "yes" => true,
            "n" | "no" => false,
            "" => default,
            _ => default,
        })
    }

    /// Prompt for selection from a list.
    pub fn select(&self, message: &str, options: &[&str]) -> io::Result<usize> {
        self.info(message);
        self.blank();

        for (i, option) in options.iter().enumerate() {
            self.list_item(i, option);
        }
        self.blank();

        loop {
            let input = self.prompt("Enter number")?;
            if let Ok(num) = input.parse::<usize>()
                && num >= 1
                && num <= options.len()
            {
                return Ok(num - 1);
            }
            let max = options.len();
            self.warning(&format!("Please enter a number between 1 and {max}"));
        }
    }

    /// Prompt for multiple selections from a list.
    pub fn multi_select(&self, message: &str, options: &[&str]) -> io::Result<Vec<usize>> {
        self.info(message);
        self.hint("Enter numbers separated by commas, or 'all' for all options");
        self.blank();

        for (i, option) in options.iter().enumerate() {
            self.list_item(i, option);
        }
        self.blank();

        loop {
            let input = self.prompt("Enter numbers")?;

            if input.to_lowercase() == "all" {
                return Ok((0..options.len()).collect());
            }

            let selections: Result<Vec<usize>, _> = input
                .split(',')
                .map(|s| s.trim().parse::<usize>())
                .collect();

            if let Ok(nums) = selections
                && nums.iter().all(|&n| n >= 1 && n <= options.len())
            {
                return Ok(nums.iter().map(|n| n - 1).collect());
            }

            let max = options.len();
            self.warning(&format!(
                "Please enter valid numbers between 1 and {max}, separated by commas"
            ));
        }
    }

    /// Prompt for a secret (password/API key) - doesn't echo input.
    pub fn prompt_secret(&self, message: &str) -> io::Result<String> {
        if self.style.use_unicode {
            print!("  {} {} ", self.lock_icon(), message);
        } else {
            print!("  > {} ", message);
        }
        io::stdout().flush()?;

        // For now, just read normally - in a real implementation
        // you'd use a library like `rpassword` for hidden input
        let mut input = String::new();
        io::stdin().lock().read_line(&mut input)?;
        Ok(input.trim().to_string())
    }

    /// Prompt for multi-line text input.
    pub fn prompt_multiline(&self, message: &str, end_marker: &str) -> io::Result<Vec<String>> {
        self.info(message);
        self.hint(&format!("Enter '{}' on a new line when done", end_marker));
        self.blank();

        let stdin = io::stdin();
        let mut lines = Vec::new();

        loop {
            print!("  ");
            io::stdout().flush()?;

            let mut line = String::new();
            stdin.lock().read_line(&mut line)?;
            let line = line.trim_end_matches('\n').trim_end_matches('\r');

            if line == end_marker {
                break;
            }

            if !line.is_empty() {
                lines.push(line.to_string());
            }
        }

        Ok(lines)
    }

    /// Display a progress indicator.
    pub fn progress(&self, current: usize, total: usize, message: &str) {
        let percent = (current as f64 / total as f64 * 100.0) as usize;
        let bar_width = 30;
        let filled = bar_width * current / total;
        let empty = bar_width - filled;

        if self.style.use_unicode {
            let bar = format!(
                "{}{}",
                self.progress_filled().repeat(filled),
                self.progress_empty().repeat(empty)
            );
            print!("\r  [{}] {:3}% {}", bar, percent, message);
        } else {
            let bar = format!("{}{}", "#".repeat(filled), "-".repeat(empty));
            print!("\r  [{}] {:3}% {}", bar, percent, message);
        }
        io::stdout().flush().ok();
    }

    /// Complete a progress indicator.
    pub fn progress_complete(&self, message: &str) {
        println!();
        self.success(message);
    }

    /// Display a spinner (for async operations).
    pub fn spinner_frame(&self, frame: usize) -> &'static str {
        if self.style.use_unicode {
            const FRAMES: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
            FRAMES[frame % FRAMES.len()]
        } else {
            const FRAMES: &[&str] = &["|", "/", "-", "\\"];
            FRAMES[frame % FRAMES.len()]
        }
    }

    // Unicode characters for styling

    fn top_left(&self) -> &'static str {
        if self.style.use_unicode { "╔" } else { "+" }
    }

    fn top_right(&self) -> &'static str {
        if self.style.use_unicode { "╗" } else { "+" }
    }

    fn bottom_left(&self) -> &'static str {
        if self.style.use_unicode { "╚" } else { "+" }
    }

    fn bottom_right(&self) -> &'static str {
        if self.style.use_unicode { "╝" } else { "+" }
    }

    fn horizontal(&self, width: usize) -> String {
        let char = if self.style.use_unicode { "═" } else { "=" };
        char.repeat(width)
    }

    fn vertical(&self) -> &'static str {
        if self.style.use_unicode { "║" } else { "|" }
    }

    fn arrow(&self) -> &'static str {
        if self.style.use_unicode { "▸" } else { ">" }
    }

    fn prompt_icon(&self) -> &'static str {
        if self.style.use_unicode { "❯" } else { ">" }
    }

    fn info_icon(&self) -> &'static str {
        if self.style.use_unicode { "ℹ" } else { "i" }
    }

    fn success_icon(&self) -> &'static str {
        if self.style.use_unicode { "✓" } else { "+" }
    }

    fn warning_icon(&self) -> &'static str {
        if self.style.use_unicode { "⚠" } else { "!" }
    }

    fn error_icon(&self) -> &'static str {
        if self.style.use_unicode { "✗" } else { "X" }
    }

    fn hint_icon(&self) -> &'static str {
        if self.style.use_unicode { "💡" } else { "?" }
    }

    fn bullet_icon(&self) -> &'static str {
        if self.style.use_unicode { "•" } else { "-" }
    }

    fn lock_icon(&self) -> &'static str {
        if self.style.use_unicode { "🔑" } else { "*" }
    }

    fn progress_filled(&self) -> &'static str {
        if self.style.use_unicode { "█" } else { "#" }
    }

    fn progress_empty(&self) -> &'static str {
        if self.style.use_unicode { "░" } else { "-" }
    }
}

impl Default for TerminalUI {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_terminal_style_default() {
        let style = TerminalStyle::default();
        assert!(style.use_colors);
        assert!(style.use_boxes);
        assert!(style.use_unicode);
    }

    #[test]
    fn test_terminal_style_plain() {
        let style = TerminalStyle::plain();
        assert!(!style.use_colors);
        assert!(!style.use_boxes);
        assert!(!style.use_unicode);
    }

    #[test]
    fn test_terminal_ui_creation() {
        let ui = TerminalUI::new();
        assert!(ui.style.use_unicode);

        let plain_ui = TerminalUI::with_style(TerminalStyle::plain());
        assert!(!plain_ui.style.use_unicode);
    }

    #[test]
    fn test_spinner_frames() {
        let ui = TerminalUI::new();

        // Should cycle through frames
        let f0 = ui.spinner_frame(0);
        let f1 = ui.spinner_frame(1);
        assert_ne!(f0, f1);

        // Should wrap around
        let f10 = ui.spinner_frame(10);
        assert_eq!(f0, f10);
    }
}
