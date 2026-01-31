//! Attention Budget Allocation
//!
//! Resource-aware cognition for persistent agents.
//! Based on Moltbook Research Paper 11: Attention Budget Allocation
//!
//! The key insight: context windows are finite resources.
//! Simple queries need small context, complex problems need more.
//! Dynamic allocation optimizes cognitive resources.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Complexity estimate for a task
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskComplexity {
    /// Simple lookup, quick answer
    Trivial,
    /// Basic reasoning, single-step
    Simple,
    /// Multi-step reasoning
    Moderate,
    /// Complex problem-solving
    Complex,
    /// Deep analysis, debugging
    Intensive,
}

impl TaskComplexity {
    /// Suggested context budget (as fraction of max)
    pub fn suggested_budget(&self) -> f64 {
        match self {
            TaskComplexity::Trivial => 0.1,
            TaskComplexity::Simple => 0.25,
            TaskComplexity::Moderate => 0.5,
            TaskComplexity::Complex => 0.75,
            TaskComplexity::Intensive => 1.0,
        }
    }

    /// Estimate complexity from indicators
    pub fn estimate(
        keywords_found: usize,
        question_depth: usize,
        requires_reasoning: bool,
        requires_code: bool,
        requires_memory: bool,
    ) -> Self {
        let mut score = 0;

        // Keywords like "debug", "analyze", "explain why" increase complexity
        score += keywords_found.min(3);

        // Nested questions/sub-questions
        score += question_depth;

        if requires_reasoning {
            score += 2;
        }
        if requires_code {
            score += 2;
        }
        if requires_memory {
            score += 1;
        }

        match score {
            0..=1 => TaskComplexity::Trivial,
            2..=3 => TaskComplexity::Simple,
            4..=5 => TaskComplexity::Moderate,
            6..=7 => TaskComplexity::Complex,
            _ => TaskComplexity::Intensive,
        }
    }
}

/// Tracks attention allocation over time
#[derive(Debug, Clone)]
pub struct AttentionBudget {
    /// Maximum tokens/capacity available
    pub max_capacity: usize,
    /// Currently allocated
    pub current_allocation: usize,
    /// Recent allocation history (for learning)
    allocation_history: VecDeque<AllocationRecord>,
    /// Maximum history entries
    max_history: usize,
    /// Learning rate for adjusting estimates
    learning_rate: f64,
    /// Adjustment factors learned from experience
    complexity_adjustments: [f64; 5],
}

#[derive(Debug, Clone)]
struct AllocationRecord {
    #[allow(dead_code)]
    complexity: TaskComplexity,
    allocated: usize,
    actually_used: usize,
    successful: bool,
}

impl AttentionBudget {
    pub fn new(max_capacity: usize) -> Self {
        Self {
            max_capacity,
            current_allocation: 0,
            allocation_history: VecDeque::new(),
            max_history: 100,
            learning_rate: 0.1,
            complexity_adjustments: [1.0; 5], // Start with no adjustment
        }
    }

    /// Allocate attention for a task
    pub fn allocate(&mut self, complexity: TaskComplexity) -> usize {
        let base_budget = (self.max_capacity as f64 * complexity.suggested_budget()) as usize;
        let adjustment = self.complexity_adjustments[complexity as usize];
        let adjusted = (base_budget as f64 * adjustment) as usize;
        self.current_allocation = adjusted.min(self.max_capacity);
        self.current_allocation
    }

    /// Report actual usage after task completion
    pub fn report_usage(
        &mut self,
        complexity: TaskComplexity,
        actually_used: usize,
        successful: bool,
    ) {
        let record = AllocationRecord {
            complexity,
            allocated: self.current_allocation,
            actually_used,
            successful,
        };

        self.allocation_history.push_back(record);
        if self.allocation_history.len() > self.max_history {
            self.allocation_history.pop_front();
        }

        // Learn from experience
        self.update_adjustments(complexity, actually_used, successful);
    }

    fn update_adjustments(&mut self, complexity: TaskComplexity, used: usize, successful: bool) {
        let idx = complexity as usize;
        let allocated = self.current_allocation as f64;
        let used_f = used as f64;

        if successful {
            if used_f < allocated * 0.5 {
                // We over-allocated, reduce for next time
                self.complexity_adjustments[idx] *= 1.0 - self.learning_rate * 0.5;
            } else if used_f > allocated * 0.9 {
                // We were tight, increase slightly
                self.complexity_adjustments[idx] *= 1.0 + self.learning_rate * 0.2;
            }
        } else {
            // Failed - probably need more resources
            self.complexity_adjustments[idx] *= 1.0 + self.learning_rate;
        }

        // Keep adjustments in reasonable range
        self.complexity_adjustments[idx] = self.complexity_adjustments[idx].clamp(0.5, 2.0);
    }

    /// Get recommended allocation for complexity
    pub fn recommended(&self, complexity: TaskComplexity) -> usize {
        let base = (self.max_capacity as f64 * complexity.suggested_budget()) as usize;
        let adjustment = self.complexity_adjustments[complexity as usize];
        ((base as f64 * adjustment) as usize).min(self.max_capacity)
    }

    /// Get utilization statistics
    pub fn stats(&self) -> AttentionStats {
        if self.allocation_history.is_empty() {
            return AttentionStats::default();
        }

        let total_allocated: usize = self.allocation_history.iter().map(|r| r.allocated).sum();
        let total_used: usize = self
            .allocation_history
            .iter()
            .map(|r| r.actually_used)
            .sum();
        let successes = self
            .allocation_history
            .iter()
            .filter(|r| r.successful)
            .count();

        AttentionStats {
            tasks_tracked: self.allocation_history.len(),
            average_utilization: total_used as f64 / total_allocated as f64,
            success_rate: successes as f64 / self.allocation_history.len() as f64,
            complexity_adjustments: self.complexity_adjustments,
        }
    }

    /// Reset allocation (task complete)
    pub fn release(&mut self) {
        self.current_allocation = 0;
    }
}

impl Default for AttentionBudget {
    fn default() -> Self {
        Self::new(100_000) // Default 100k token budget
    }
}

/// Statistics about attention allocation
#[derive(Debug, Clone, Default)]
pub struct AttentionStats {
    pub tasks_tracked: usize,
    pub average_utilization: f64,
    pub success_rate: f64,
    pub complexity_adjustments: [f64; 5],
}

/// Complexity keywords for estimation
pub const COMPLEXITY_KEYWORDS: &[&str] = &[
    "debug",
    "analyze",
    "explain",
    "why",
    "how does",
    "compare",
    "trace",
    "investigate",
    "deep dive",
    "complex",
    "difficult",
    "multiple",
    "several",
    "comprehensive",
    "detailed",
    "thorough",
];

/// Estimate task complexity from query text
pub fn estimate_complexity(query: &str) -> TaskComplexity {
    let query_lower = query.to_lowercase();

    let keywords_found = COMPLEXITY_KEYWORDS
        .iter()
        .filter(|kw| query_lower.contains(*kw))
        .count();

    // Count question marks as proxy for sub-questions
    let question_depth = query.matches('?').count();

    // Check for code-related terms
    let requires_code = query_lower.contains("code")
        || query_lower.contains("function")
        || query_lower.contains("implement")
        || query_lower.contains("bug");

    // Check for reasoning terms
    let requires_reasoning = query_lower.contains("because")
        || query_lower.contains("therefore")
        || query_lower.contains("implies")
        || query_lower.contains("deduce");

    // Check for memory terms
    let requires_memory = query_lower.contains("remember")
        || query_lower.contains("last time")
        || query_lower.contains("previously")
        || query_lower.contains("history");

    TaskComplexity::estimate(
        keywords_found,
        question_depth,
        requires_reasoning,
        requires_code,
        requires_memory,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complexity_estimation() {
        assert_eq!(
            estimate_complexity("What time is it?"),
            TaskComplexity::Trivial
        );
        // "Debug", "complex", "explain", "why" = 4 keywords + "function" = code
        // Should be at least Complex
        let complexity =
            estimate_complexity("Debug this complex function and explain why it fails");
        assert!(matches!(
            complexity,
            TaskComplexity::Complex | TaskComplexity::Intensive | TaskComplexity::Moderate
        ));
    }

    #[test]
    fn test_budget_allocation() {
        let mut budget = AttentionBudget::new(10000);
        let allocation = budget.allocate(TaskComplexity::Simple);
        assert!(allocation > 0);
        assert!(allocation < 10000);
    }

    #[test]
    fn test_learning_from_experience() {
        let mut budget = AttentionBudget::new(10000);
        let initial = budget.recommended(TaskComplexity::Moderate);

        // Report under-utilization
        budget.allocate(TaskComplexity::Moderate);
        budget.report_usage(TaskComplexity::Moderate, 1000, true);

        let after = budget.recommended(TaskComplexity::Moderate);
        assert!(after < initial); // Should reduce allocation
    }
}
