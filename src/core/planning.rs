//! Planning Bridge - Imagination to Goals and Actions
//!
//! Converts imagination outputs into concrete goals and action templates.
//! This bridges creative ideation with execution planning.

use uuid::Uuid;

use crate::core::action::{ActionCategory, ActionTemplate, ExpectedOutcome, Outcome};
use crate::core::goals::{Criterion, Goal, Priority, TimeHorizon};
use crate::core::imagination::{ImaginationType, Imagining, ImaginingId};

/// Configuration for imagination-driven planning.
#[derive(Debug, Clone)]
pub struct ImaginationPlannerConfig {
    /// Minimum utility required to create a goal.
    pub min_goal_utility: f64,
    /// Minimum utility required to create an action.
    pub min_action_utility: f64,
    /// Maximum actions per imagining.
    pub max_actions: usize,
    /// Maximum criteria to derive from metadata.
    pub max_criteria: usize,
    /// Maximum words for summaries.
    pub summary_words: usize,
}

impl Default for ImaginationPlannerConfig {
    fn default() -> Self {
        Self {
            min_goal_utility: 0.4,
            min_action_utility: 0.3,
            max_actions: 2,
            max_criteria: 3,
            summary_words: 12,
        }
    }
}

/// A planning suggestion derived from imagination.
#[derive(Debug, Clone)]
pub struct PlanningSuggestion {
    pub imagining_id: ImaginingId,
    pub imagination_type: ImaginationType,
    pub goals: Vec<Goal>,
    pub actions: Vec<ActionTemplate>,
    pub rationale: String,
    pub score: f64,
}

impl PlanningSuggestion {
    pub fn is_empty(&self) -> bool {
        self.goals.is_empty() && self.actions.is_empty()
    }
}

/// Planner that maps imaginations into actionable plans.
#[derive(Debug, Clone)]
pub struct ImaginationPlanner {
    config: ImaginationPlannerConfig,
}

impl ImaginationPlanner {
    pub fn new(config: ImaginationPlannerConfig) -> Self {
        Self { config }
    }

    pub fn plan_from_imagining(&self, imagining: &Imagining) -> PlanningSuggestion {
        if imagining.archived {
            return PlanningSuggestion {
                imagining_id: imagining.id,
                imagination_type: imagining.imagination_type,
                goals: Vec::new(),
                actions: Vec::new(),
                rationale: "Imagining archived; no plan created".to_string(),
                score: 0.0,
            };
        }

        let score = compute_imagining_score(imagining);
        let summary = summarize_content(&imagining.content, self.config.summary_words);
        let rationale = format!(
            "{} (utility {:.2}, confidence {:.2}, novelty {:.2})",
            imagination_label(imagining.imagination_type),
            imagining.utility,
            imagining.confidence,
            imagining.novelty
        );

        let priority = priority_from_score(score);
        let horizon = horizon_from_type(imagining.imagination_type, imagining.novelty);
        let effort = estimated_effort(imagining);

        let mut goals = Vec::new();
        if should_create_goal(imagining, self.config.min_goal_utility) {
            let goal_desc = format!("{}: {}", goal_prefix(imagining.imagination_type), summary);
            let mut goal = Goal::new(&goal_desc)
                .with_priority(priority)
                .with_horizon(horizon)
                .with_effort(effort)
                .with_tag("imagination");

            let type_tag = imagining.imagination_type.to_string();
            let id_tag = format!("imagining:{}", imagining.id);
            goal = goal.with_tag(&type_tag).with_tag(&id_tag);

            let criteria = metadata_strings(imagining, "key_factors", self.config.max_criteria);
            for criterion in criteria {
                goal.success_criteria.push(Criterion::new(&criterion));
            }

            goal.notes = format!(
                "Derived from {} imagining (confidence {:.2})",
                imagination_label(imagining.imagination_type),
                imagining.confidence
            );

            goals.push(goal);
        }

        let mut actions = Vec::new();
        if should_create_action(imagining, self.config.min_action_utility) {
            let action_category = action_category_for_type(imagining.imagination_type);
            let action_name = format!("Investigate {}", summary);
            let action_description = format!("Follow up on imagining: {}", imagining.content);

            let mut outcome = Outcome::success("Investigated imagining", 0.4);
            if let Some(goal) = goals.first() {
                outcome = outcome.with_goal(&goal.description, 0.25);
            }

            let action = ActionTemplate {
                id: Uuid::new_v4(),
                name: action_name,
                description: action_description,
                preconditions: Vec::new(),
                expected_outcomes: vec![ExpectedOutcome {
                    outcome,
                    probability: 0.6,
                }],
                effort_cost: (effort + 0.1).clamp(0.0, 1.0),
                time_cost: time_cost_for_type(imagining.imagination_type),
                category: action_category,
                tags: vec![
                    "imagination".to_string(),
                    imagining.imagination_type.to_string(),
                ],
            };

            actions.push(action);
        }

        actions.truncate(self.config.max_actions);

        PlanningSuggestion {
            imagining_id: imagining.id,
            imagination_type: imagining.imagination_type,
            goals,
            actions,
            rationale,
            score,
        }
    }

    pub fn plan_from_imaginings(
        &self,
        imaginings: impl IntoIterator<Item = Imagining>,
    ) -> Vec<PlanningSuggestion> {
        imaginings
            .into_iter()
            .map(|imagining| self.plan_from_imagining(&imagining))
            .collect()
    }
}

impl Default for ImaginationPlanner {
    fn default() -> Self {
        Self::new(ImaginationPlannerConfig::default())
    }
}

fn compute_imagining_score(imagining: &Imagining) -> f64 {
    (imagining.utility * 0.5 + imagining.confidence * 0.3 + imagining.novelty * 0.2).clamp(0.0, 1.0)
}

fn should_create_goal(imagining: &Imagining, min_utility: f64) -> bool {
    if imagining.utility >= min_utility {
        return true;
    }
    matches!(
        imagining.imagination_type,
        ImaginationType::Hypothesis | ImaginationType::Simulation | ImaginationType::Synthesis
    )
}

fn should_create_action(imagining: &Imagining, min_utility: f64) -> bool {
    imagining.utility >= min_utility || imagining.confidence > 0.6
}

fn goal_prefix(imagination_type: ImaginationType) -> &'static str {
    match imagination_type {
        ImaginationType::Hypothesis => "Test hypothesis",
        ImaginationType::Simulation => "Prepare for scenario",
        ImaginationType::Counterfactual => "Analyze alternate outcome",
        ImaginationType::Dream => "Interpret dream",
        ImaginationType::Synthesis => "Apply insight",
        ImaginationType::Recombination => "Explore idea",
    }
}

fn imagination_label(imagination_type: ImaginationType) -> &'static str {
    match imagination_type {
        ImaginationType::Hypothesis => "Hypothesis",
        ImaginationType::Simulation => "Simulation",
        ImaginationType::Counterfactual => "Counterfactual",
        ImaginationType::Dream => "Dream",
        ImaginationType::Synthesis => "Synthesis",
        ImaginationType::Recombination => "Recombination",
    }
}

fn summarize_content(content: &str, max_words: usize) -> String {
    let mut words = content.split_whitespace();
    let mut summary = Vec::new();
    for _ in 0..max_words {
        if let Some(word) = words.next() {
            summary.push(word);
        } else {
            break;
        }
    }
    summary.join(" ")
}

fn priority_from_score(score: f64) -> Priority {
    if score >= 0.85 {
        Priority::Critical
    } else if score >= 0.65 {
        Priority::High
    } else if score >= 0.45 {
        Priority::Medium
    } else {
        Priority::Low
    }
}

fn horizon_from_type(imagination_type: ImaginationType, novelty: f64) -> TimeHorizon {
    match imagination_type {
        ImaginationType::Hypothesis => TimeHorizon::ShortTerm,
        ImaginationType::Simulation => TimeHorizon::ShortTerm,
        ImaginationType::Counterfactual => TimeHorizon::MediumTerm,
        ImaginationType::Synthesis => TimeHorizon::MediumTerm,
        ImaginationType::Recombination => {
            if novelty > 0.6 {
                TimeHorizon::LongTerm
            } else {
                TimeHorizon::MediumTerm
            }
        }
        ImaginationType::Dream => TimeHorizon::LongTerm,
    }
}

fn estimated_effort(imagining: &Imagining) -> f64 {
    (0.2 + (1.0 - imagining.confidence) * 0.4 + imagining.novelty * 0.3).clamp(0.0, 1.0)
}

fn action_category_for_type(imagination_type: ImaginationType) -> ActionCategory {
    match imagination_type {
        ImaginationType::Hypothesis | ImaginationType::Simulation => ActionCategory::Exploration,
        ImaginationType::Synthesis | ImaginationType::Recombination => ActionCategory::Creative,
        ImaginationType::Counterfactual => ActionCategory::Learning,
        ImaginationType::Dream => ActionCategory::Rest,
    }
}

fn time_cost_for_type(imagination_type: ImaginationType) -> u32 {
    match imagination_type {
        ImaginationType::Hypothesis => 3,
        ImaginationType::Simulation => 4,
        ImaginationType::Counterfactual => 5,
        ImaginationType::Synthesis => 3,
        ImaginationType::Recombination => 2,
        ImaginationType::Dream => 1,
    }
}

fn metadata_strings(imagining: &Imagining, key: &str, max: usize) -> Vec<String> {
    let mut values = Vec::new();
    let Some(value) = imagining.metadata.get(key) else {
        return values;
    };

    if let Some(array) = value.as_array() {
        for item in array.iter().take(max) {
            if let Some(text) = item.as_str() {
                values.push(text.to_string());
            }
        }
    } else if let Some(text) = value.as_str() {
        values.push(text.to_string());
    }

    values
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_goal_and_action_from_hypothesis() {
        let planner = ImaginationPlanner::default();
        let imagining = Imagining::new(
            ImaginationType::Hypothesis,
            "If we tweak the reward schedule, learning improves".to_string(),
            vec!["mem-1".to_string()],
        )
        .with_confidence(0.7)
        .with_utility(0.8)
        .with_novelty(0.4);

        let suggestion = planner.plan_from_imagining(&imagining);
        assert!(!suggestion.goals.is_empty());
        assert!(!suggestion.actions.is_empty());
        assert!(
            suggestion.goals[0]
                .description
                .to_lowercase()
                .contains("hypothesis")
        );
    }

    #[test]
    fn archived_imagining_creates_no_plan() {
        let planner = ImaginationPlanner::default();
        let mut imagining = Imagining::new(
            ImaginationType::Dream,
            "Ocean made of clocks".to_string(),
            Vec::new(),
        );
        imagining.archive();

        let suggestion = planner.plan_from_imagining(&imagining);
        assert!(suggestion.is_empty());
    }
}
