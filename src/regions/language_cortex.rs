//! Language Cortex - Code Syntax and Semantics Processing
//!
//! Inspired by Broca's and Wernicke's areas in the human brain:
//! - **Broca's Area** (left frontal): Language production, syntax, grammar rules
//! - **Wernicke's Area** (left temporal): Language comprehension, semantic processing
//!
//! For code understanding, this translates to:
//! - **Syntax Analysis**: Parsing code structure (brackets, indentation, keywords)
//! - **Semantic Understanding**: Meaning of identifiers, types, function contracts
//! - **Code Generation**: Producing well-formed, idiomatic code
//! - **Error Detection**: Finding syntax and semantic anomalies
//!
//! # Theoretical Foundation
//!
//! Based on:
//! - Chomsky's Universal Grammar: Hierarchical syntactic structures
//! - Construction Grammar: Form-meaning pairings (patterns)
//! - Embodied Language: Language grounded in sensorimotor experience
//!
//! # Architecture
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────────────┐
//! │                      LANGUAGE CORTEX                               │
//! ├──────────────────┬──────────────────┬──────────────────┬──────────┤
//! │   Broca's Analog │  Wernicke's Analog │  Arcuate Bundle │ Angular  │
//! │   (syntax/prod)  │  (comprehension)  │  (connection)   │ (symbols)│
//! │   - parsing      │  - semantics      │  - syntax-sem   │ - naming │
//! │   - structure    │  - meaning        │  - integration  │ - refs   │
//! │   - generation   │  - intent         │  - feedback     │ - types  │
//! └──────────────────┴──────────────────┴──────────────────┴──────────┘
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

// ============================================================================
// SYNTAX STRUCTURES
// ============================================================================

/// A syntax element representing code structure
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SyntaxElement {
    /// A block of code (function, class, module)
    Block {
        kind: BlockKind,
        name: Option<String>,
        depth: usize,
    },
    /// A statement (assignment, return, expression)
    Statement { kind: StatementKind },
    /// An expression (operation, call, literal)
    Expression { kind: ExpressionKind },
    /// A declaration (variable, function, type)
    Declaration { kind: DeclarationKind, name: String },
    /// Control flow (if, loop, match)
    ControlFlow { kind: ControlFlowKind },
    /// An error in parsing
    Error { message: String, position: usize },
}

/// Kinds of code blocks
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BlockKind {
    Function,
    Class,
    Module,
    Struct,
    Enum,
    Trait,
    Impl,
    Closure,
    Test,
    Other(String),
}

/// Kinds of statements
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StatementKind {
    Assignment,
    Return,
    Expression,
    Import,
    Export,
    Empty,
}

/// Kinds of expressions
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExpressionKind {
    Literal,
    Identifier,
    BinaryOp,
    UnaryOp,
    Call,
    MethodCall,
    Index,
    Field,
    Closure,
}

/// Kinds of declarations
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeclarationKind {
    Variable,
    Constant,
    Function,
    Type,
    Struct,
    Enum,
    Trait,
    Parameter,
}

/// Kinds of control flow
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ControlFlowKind {
    If,
    Else,
    Match,
    Loop,
    While,
    For,
    Break,
    Continue,
    Return,
}

// ============================================================================
// SEMANTIC UNDERSTANDING
// ============================================================================

/// Semantic information about a code entity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticEntity {
    /// The name of the entity
    pub name: String,
    /// What kind of entity (variable, function, type, etc.)
    pub kind: EntityKind,
    /// The inferred type (if known)
    pub inferred_type: Option<String>,
    /// What it represents conceptually
    pub concept: Option<String>,
    /// Confidence in understanding (0-1)
    pub confidence: f64,
    /// Related entities (dependencies, usages)
    pub relations: Vec<EntityRelation>,
    /// When first encountered
    pub first_seen: DateTime<Utc>,
    /// Access count (for familiarity)
    pub access_count: u32,
}

/// Kinds of code entities
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EntityKind {
    Variable,
    Function,
    Type,
    Module,
    Constant,
    Field,
    Parameter,
    Generic,
}

/// Relationships between entities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityRelation {
    /// Target entity name
    pub target: String,
    /// Kind of relationship
    pub kind: RelationKind,
    /// Strength of relationship (0-1)
    pub strength: f64,
}

/// Kinds of entity relationships
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RelationKind {
    /// Uses/depends on
    Uses,
    /// Is used by
    UsedBy,
    /// Contains/has
    Contains,
    /// Is contained in
    ContainedIn,
    /// Extends/inherits from
    Extends,
    /// Is extended by
    ExtendedBy,
    /// Implements
    Implements,
    /// Calls
    Calls,
    /// Is called by
    CalledBy,
}

// ============================================================================
// CODE PATTERNS (Construction Grammar analog)
// ============================================================================

/// A recognized code pattern (idiom, design pattern, anti-pattern)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodePattern {
    /// Pattern identifier
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Pattern category
    pub category: PatternCategory,
    /// Description of what this pattern does
    pub description: String,
    /// Is this considered good practice?
    pub is_idiomatic: bool,
    /// Example snippet
    pub example: Option<String>,
    /// How often recognized
    pub recognition_count: u32,
    /// Confidence when recognizing (0-1)
    pub confidence: f64,
}

/// Categories of code patterns
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PatternCategory {
    /// Design patterns (singleton, factory, etc.)
    DesignPattern,
    /// Idioms (language-specific best practices)
    Idiom,
    /// Anti-patterns (code smells, bad practices)
    AntiPattern,
    /// Error handling patterns
    ErrorHandling,
    /// Resource management
    ResourceManagement,
    /// Concurrency patterns
    Concurrency,
    /// Testing patterns
    Testing,
    /// Other
    Other(String),
}

// ============================================================================
// SYNTAX ERRORS AND ISSUES
// ============================================================================

/// A detected syntax or semantic issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeIssue {
    /// Issue identifier
    pub id: String,
    /// Severity level
    pub severity: IssueSeverity,
    /// Category of issue
    pub category: IssueCategory,
    /// Description of the problem
    pub message: String,
    /// Suggested fix (if available)
    pub suggestion: Option<String>,
    /// Position in code (line, column)
    pub position: Option<(usize, usize)>,
    /// Confidence that this is actually an issue (0-1)
    pub confidence: f64,
    /// When detected
    pub detected_at: DateTime<Utc>,
}

/// Severity of code issues
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum IssueSeverity {
    /// Just a hint or style suggestion
    Hint,
    /// Information, might want to look at
    Info,
    /// Warning, potential problem
    Warning,
    /// Error, definitely broken
    Error,
}

/// Categories of code issues
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IssueCategory {
    /// Syntax error (parsing failed)
    SyntaxError,
    /// Type mismatch
    TypeError,
    /// Undefined reference
    UndefinedReference,
    /// Unused variable/import
    Unused,
    /// Style/formatting issue
    Style,
    /// Potential bug
    PotentialBug,
    /// Security concern
    Security,
    /// Performance issue
    Performance,
}

// ============================================================================
// LANGUAGE CORTEX CONFIGURATION
// ============================================================================

/// Configuration for the Language Cortex
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageCortexConfig {
    /// Maximum entities to track
    pub max_entities: usize,
    /// Maximum patterns to remember
    pub max_patterns: usize,
    /// Maximum issues to track
    pub max_issues: usize,
    /// Confidence threshold for pattern recognition
    pub pattern_confidence_threshold: f64,
    /// Enable syntax error detection
    pub enable_syntax_checking: bool,
    /// Enable semantic analysis
    pub enable_semantic_analysis: bool,
    /// Decay rate for entity familiarity
    pub familiarity_decay_rate: f64,
}

impl Default for LanguageCortexConfig {
    fn default() -> Self {
        Self {
            max_entities: 1000,
            max_patterns: 100,
            max_issues: 500,
            pattern_confidence_threshold: 0.6,
            enable_syntax_checking: true,
            enable_semantic_analysis: true,
            familiarity_decay_rate: 0.01,
        }
    }
}

// ============================================================================
// LANGUAGE CORTEX
// ============================================================================

/// The Language Cortex - processes code syntax and semantics
///
/// Inspired by Broca's and Wernicke's areas, this module handles:
/// - Syntax parsing and structure recognition
/// - Semantic understanding of code meaning
/// - Pattern recognition (idioms, design patterns)
/// - Error detection and suggestions
pub struct LanguageCortex {
    config: LanguageCortexConfig,
    /// Semantic entities (Wernicke's analog)
    entities: HashMap<String, SemanticEntity>,
    /// Recently parsed syntax elements (Broca's analog)
    recent_syntax: VecDeque<SyntaxElement>,
    /// Recognized patterns
    patterns: Vec<CodePattern>,
    /// Detected issues
    issues: VecDeque<CodeIssue>,
    /// Current parsing context
    current_context: ParsingContext,
    /// Statistics
    stats: LanguageCortexStats,
}

/// Current parsing context
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ParsingContext {
    /// Current file/module being analyzed
    pub current_file: Option<String>,
    /// Current function/method context
    pub current_function: Option<String>,
    /// Nesting depth (for blocks)
    pub nesting_depth: usize,
    /// Current language (rust, python, etc.)
    pub language: Option<String>,
    /// Variables in scope
    pub scope_stack: Vec<HashMap<String, String>>,
}

/// Statistics for Language Cortex
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LanguageCortexStats {
    /// Total entities tracked
    pub entities_tracked: usize,
    /// Total patterns recognized
    pub patterns_recognized: u64,
    /// Total issues detected
    pub issues_detected: u64,
    /// Syntax elements parsed
    pub syntax_elements_parsed: u64,
    /// Semantic lookups performed
    pub semantic_lookups: u64,
    /// Average entity familiarity
    pub avg_familiarity: f64,
}

impl LanguageCortex {
    /// Create a new Language Cortex with default configuration
    pub fn new() -> Self {
        Self::with_config(LanguageCortexConfig::default())
    }

    /// Create with specific configuration
    pub fn with_config(config: LanguageCortexConfig) -> Self {
        let mut cortex = Self {
            config,
            entities: HashMap::new(),
            recent_syntax: VecDeque::new(),
            patterns: Vec::new(),
            issues: VecDeque::new(),
            current_context: ParsingContext::default(),
            stats: LanguageCortexStats::default(),
        };

        // Initialize with common patterns
        cortex.initialize_common_patterns();
        cortex
    }

    /// Initialize common code patterns
    fn initialize_common_patterns(&mut self) {
        // Error handling patterns
        self.patterns.push(CodePattern {
            id: "result_unwrap".to_string(),
            name: "Result Unwrap".to_string(),
            category: PatternCategory::ErrorHandling,
            description: "Unwrapping Result without error handling".to_string(),
            is_idiomatic: false,
            example: Some("result.unwrap() // risky!".to_string()),
            recognition_count: 0,
            confidence: 0.8,
        });

        self.patterns.push(CodePattern {
            id: "result_propagation".to_string(),
            name: "Error Propagation".to_string(),
            category: PatternCategory::ErrorHandling,
            description: "Propagating errors with ? operator".to_string(),
            is_idiomatic: true,
            example: Some("let value = operation()?;".to_string()),
            recognition_count: 0,
            confidence: 0.9,
        });

        // Resource management
        self.patterns.push(CodePattern {
            id: "raii".to_string(),
            name: "RAII Pattern".to_string(),
            category: PatternCategory::ResourceManagement,
            description: "Resource Acquisition Is Initialization".to_string(),
            is_idiomatic: true,
            example: None,
            recognition_count: 0,
            confidence: 0.7,
        });

        // Design patterns
        self.patterns.push(CodePattern {
            id: "builder".to_string(),
            name: "Builder Pattern".to_string(),
            category: PatternCategory::DesignPattern,
            description: "Fluent interface for constructing complex objects".to_string(),
            is_idiomatic: true,
            example: Some("Object::new().with_x(1).with_y(2).build()".to_string()),
            recognition_count: 0,
            confidence: 0.75,
        });

        // Anti-patterns
        self.patterns.push(CodePattern {
            id: "god_function".to_string(),
            name: "God Function".to_string(),
            category: PatternCategory::AntiPattern,
            description: "Function that does too many things".to_string(),
            is_idiomatic: false,
            example: None,
            recognition_count: 0,
            confidence: 0.6,
        });
    }

    // ========================================================================
    // SYNTAX PROCESSING (Broca's analog)
    // ========================================================================

    /// Parse and analyze code syntax
    pub fn analyze_syntax(&mut self, code: &str, language: Option<&str>) -> Vec<SyntaxElement> {
        self.current_context.language = language.map(|s| s.to_string());
        let elements = self.parse_syntax(code);

        for element in &elements {
            self.recent_syntax.push_back(element.clone());
            self.stats.syntax_elements_parsed += 1;
        }

        // Keep recent syntax bounded
        while self.recent_syntax.len() > 1000 {
            self.recent_syntax.pop_front();
        }

        elements
    }

    /// Simple syntax parser (structure recognition)
    fn parse_syntax(&mut self, code: &str) -> Vec<SyntaxElement> {
        let mut elements = Vec::new();
        let mut depth = 0;

        for (line_no, line) in code.lines().enumerate() {
            let trimmed = line.trim();

            // Skip empty lines and comments
            if trimmed.is_empty() || trimmed.starts_with("//") || trimmed.starts_with('#') {
                continue;
            }

            // Track nesting depth
            let open_braces = trimmed.matches('{').count();
            let close_braces = trimmed.matches('}').count();

            // Detect block types
            if trimmed.starts_with("fn ") || trimmed.starts_with("pub fn ") {
                let name = self.extract_function_name(trimmed);
                elements.push(SyntaxElement::Block {
                    kind: BlockKind::Function,
                    name,
                    depth,
                });
            } else if trimmed.starts_with("struct ") || trimmed.starts_with("pub struct ") {
                let name = self.extract_name_after(trimmed, "struct ");
                elements.push(SyntaxElement::Block {
                    kind: BlockKind::Struct,
                    name,
                    depth,
                });
            } else if trimmed.starts_with("enum ") || trimmed.starts_with("pub enum ") {
                let name = self.extract_name_after(trimmed, "enum ");
                elements.push(SyntaxElement::Block {
                    kind: BlockKind::Enum,
                    name,
                    depth,
                });
            } else if trimmed.starts_with("impl ") {
                let name = self.extract_name_after(trimmed, "impl ");
                elements.push(SyntaxElement::Block {
                    kind: BlockKind::Impl,
                    name,
                    depth,
                });
            } else if trimmed.starts_with("trait ") || trimmed.starts_with("pub trait ") {
                let name = self.extract_name_after(trimmed, "trait ");
                elements.push(SyntaxElement::Block {
                    kind: BlockKind::Trait,
                    name,
                    depth,
                });
            } else if trimmed.starts_with("mod ") || trimmed.starts_with("pub mod ") {
                let name = self.extract_name_after(trimmed, "mod ");
                elements.push(SyntaxElement::Block {
                    kind: BlockKind::Module,
                    name,
                    depth,
                });
            } else if trimmed.starts_with("#[test]") {
                elements.push(SyntaxElement::Block {
                    kind: BlockKind::Test,
                    name: None,
                    depth,
                });
            }

            // Detect declarations
            if trimmed.starts_with("let ") || trimmed.starts_with("let mut ") {
                let name = self.extract_variable_name(trimmed);
                if let Some(name) = name {
                    elements.push(SyntaxElement::Declaration {
                        kind: DeclarationKind::Variable,
                        name,
                    });
                }
            } else if trimmed.starts_with("const ") {
                let name = self.extract_name_after(trimmed, "const ");
                if let Some(name) = name {
                    elements.push(SyntaxElement::Declaration {
                        kind: DeclarationKind::Constant,
                        name,
                    });
                }
            }

            // Detect control flow
            if trimmed.starts_with("if ") || trimmed == "if" {
                elements.push(SyntaxElement::ControlFlow {
                    kind: ControlFlowKind::If,
                });
            } else if trimmed.starts_with("else ") || trimmed == "else" {
                elements.push(SyntaxElement::ControlFlow {
                    kind: ControlFlowKind::Else,
                });
            } else if trimmed.starts_with("match ") {
                elements.push(SyntaxElement::ControlFlow {
                    kind: ControlFlowKind::Match,
                });
            } else if trimmed.starts_with("loop ") || trimmed == "loop" {
                elements.push(SyntaxElement::ControlFlow {
                    kind: ControlFlowKind::Loop,
                });
            } else if trimmed.starts_with("while ") {
                elements.push(SyntaxElement::ControlFlow {
                    kind: ControlFlowKind::While,
                });
            } else if trimmed.starts_with("for ") {
                elements.push(SyntaxElement::ControlFlow {
                    kind: ControlFlowKind::For,
                });
            } else if trimmed.starts_with("return ") || trimmed == "return" {
                elements.push(SyntaxElement::ControlFlow {
                    kind: ControlFlowKind::Return,
                });
            }

            // Check for unbalanced braces
            depth = depth.saturating_add(open_braces);
            depth = depth.saturating_sub(close_braces);

            // Detect potential syntax errors
            if trimmed.contains(";;") {
                elements.push(SyntaxElement::Error {
                    message: "Double semicolon detected".to_string(),
                    position: line_no,
                });
            }
        }

        self.current_context.nesting_depth = depth;
        elements
    }

    /// Extract function name from function declaration
    fn extract_function_name(&self, line: &str) -> Option<String> {
        let after_fn = line
            .strip_prefix("pub fn ")
            .or_else(|| line.strip_prefix("fn "))?;

        after_fn
            .split(['(', '<', ' '])
            .next()
            .map(|s| s.to_string())
    }

    /// Extract name after a keyword
    fn extract_name_after(&self, line: &str, keyword: &str) -> Option<String> {
        if let Some(pos) = line.find(keyword) {
            let after = &line[pos + keyword.len()..];
            // Find where pub might start
            let after = after.trim_start_matches("pub ").trim();
            after
                .split(|c: char| !c.is_alphanumeric() && c != '_')
                .next()
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
        } else {
            None
        }
    }

    /// Extract variable name from let declaration
    fn extract_variable_name(&self, line: &str) -> Option<String> {
        let after_let = line
            .strip_prefix("let mut ")
            .or_else(|| line.strip_prefix("let "))?;

        after_let
            .split([':', '=', ' '])
            .next()
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
    }

    // ========================================================================
    // SEMANTIC PROCESSING (Wernicke's analog)
    // ========================================================================

    /// Register a semantic entity
    pub fn register_entity(
        &mut self,
        name: String,
        kind: EntityKind,
        inferred_type: Option<String>,
    ) {
        self.stats.entities_tracked = self.entities.len();

        let entity = self
            .entities
            .entry(name.clone())
            .or_insert_with(|| SemanticEntity {
                name: name.clone(),
                kind: kind.clone(),
                inferred_type: None,
                concept: None,
                confidence: 0.5,
                relations: Vec::new(),
                first_seen: Utc::now(),
                access_count: 0,
            });

        entity.access_count += 1;
        if inferred_type.is_some() {
            entity.inferred_type = inferred_type;
            entity.confidence = (entity.confidence + 0.1).min(1.0);
        }

        // Enforce size limit
        if self.entities.len() > self.config.max_entities {
            // Remove least accessed entity
            if let Some(least_accessed) = self
                .entities
                .iter()
                .min_by_key(|(_, e)| e.access_count)
                .map(|(k, _)| k.clone())
            {
                self.entities.remove(&least_accessed);
            }
        }
    }

    /// Look up a semantic entity
    pub fn lookup_entity(&mut self, name: &str) -> Option<&SemanticEntity> {
        self.stats.semantic_lookups += 1;
        if let Some(entity) = self.entities.get_mut(name) {
            entity.access_count += 1;
        }
        self.entities.get(name)
    }

    /// Add a relation between entities
    pub fn add_relation(&mut self, from: &str, to: &str, kind: RelationKind) {
        if let Some(entity) = self.entities.get_mut(from) {
            // Check if relation already exists
            let exists = entity
                .relations
                .iter()
                .any(|r| r.target == to && r.kind == kind);
            if !exists {
                entity.relations.push(EntityRelation {
                    target: to.to_string(),
                    kind,
                    strength: 1.0,
                });
            }
        }
    }

    /// Get all entities of a specific kind
    pub fn entities_of_kind(&self, kind: EntityKind) -> Vec<&SemanticEntity> {
        self.entities.values().filter(|e| e.kind == kind).collect()
    }

    // ========================================================================
    // PATTERN RECOGNITION
    // ========================================================================

    /// Recognize patterns in code
    pub fn recognize_patterns(&mut self, code: &str) -> Vec<CodePattern> {
        let mut recognized = Vec::new();

        for pattern in &mut self.patterns {
            let matched = match pattern.id.as_str() {
                "result_unwrap" => code.contains(".unwrap()"),
                "result_propagation" => code.contains("?;") || code.contains("?)"),
                "builder" => {
                    code.contains(".with_")
                        || (code.contains("::new()") && code.contains(".build()"))
                }
                "god_function" => {
                    // Simple heuristic: function with many lines
                    code.lines().count() > 50
                }
                _ => false,
            };

            if matched && pattern.confidence >= self.config.pattern_confidence_threshold {
                pattern.recognition_count += 1;
                self.stats.patterns_recognized += 1;
                recognized.push(pattern.clone());
            }
        }

        recognized
    }

    /// Get anti-patterns found
    pub fn get_antipatterns(&self) -> Vec<&CodePattern> {
        self.patterns
            .iter()
            .filter(|p| p.category == PatternCategory::AntiPattern && p.recognition_count > 0)
            .collect()
    }

    // ========================================================================
    // ISSUE DETECTION
    // ========================================================================

    /// Detect potential issues in code
    pub fn detect_issues(&mut self, code: &str) -> Vec<CodeIssue> {
        let mut issues = Vec::new();

        // Check for common issues
        for (line_no, line) in code.lines().enumerate() {
            let trimmed = line.trim();

            // Unused variable pattern (starts with _ is ok)
            if trimmed.starts_with("let ")
                && !trimmed.starts_with("let _")
                && !trimmed.contains('=')
            {
                issues.push(CodeIssue {
                    id: format!("unused_decl_{}", line_no),
                    severity: IssueSeverity::Warning,
                    category: IssueCategory::Unused,
                    message: "Variable declared but not assigned".to_string(),
                    suggestion: Some("Consider initializing the variable".to_string()),
                    position: Some((line_no + 1, 1)),
                    confidence: 0.7,
                    detected_at: Utc::now(),
                });
            }

            // Unwrap without context
            if trimmed.contains(".unwrap()") && !trimmed.contains("// ") {
                issues.push(CodeIssue {
                    id: format!("unwrap_{}", line_no),
                    severity: IssueSeverity::Warning,
                    category: IssueCategory::PotentialBug,
                    message: "Unwrap may panic - consider using ? or match".to_string(),
                    suggestion: Some("Use .expect(\"reason\") or handle the error".to_string()),
                    position: Some((line_no + 1, 1)),
                    confidence: 0.8,
                    detected_at: Utc::now(),
                });
            }

            // Todo comments
            if trimmed.to_lowercase().contains("todo") || trimmed.to_lowercase().contains("fixme") {
                issues.push(CodeIssue {
                    id: format!("todo_{}", line_no),
                    severity: IssueSeverity::Info,
                    category: IssueCategory::Style,
                    message: "TODO/FIXME comment found".to_string(),
                    suggestion: None,
                    position: Some((line_no + 1, 1)),
                    confidence: 0.95,
                    detected_at: Utc::now(),
                });
            }

            // Very long lines
            if line.len() > 120 {
                issues.push(CodeIssue {
                    id: format!("long_line_{}", line_no),
                    severity: IssueSeverity::Hint,
                    category: IssueCategory::Style,
                    message: format!("Line exceeds 120 characters ({} chars)", line.len()),
                    suggestion: Some("Consider breaking into multiple lines".to_string()),
                    position: Some((line_no + 1, 121)),
                    confidence: 1.0,
                    detected_at: Utc::now(),
                });
            }
        }

        // Check for unbalanced brackets
        let open_parens = code.matches('(').count();
        let close_parens = code.matches(')').count();
        if open_parens != close_parens {
            issues.push(CodeIssue {
                id: "unbalanced_parens".to_string(),
                severity: IssueSeverity::Error,
                category: IssueCategory::SyntaxError,
                message: format!(
                    "Unbalanced parentheses: {} '(' vs {} ')'",
                    open_parens, close_parens
                ),
                suggestion: None,
                position: None,
                confidence: 0.95,
                detected_at: Utc::now(),
            });
        }

        let open_braces = code.matches('{').count();
        let close_braces = code.matches('}').count();
        if open_braces != close_braces {
            issues.push(CodeIssue {
                id: "unbalanced_braces".to_string(),
                severity: IssueSeverity::Error,
                category: IssueCategory::SyntaxError,
                message: format!(
                    "Unbalanced braces: {} '{{' vs {} '}}'",
                    open_braces, close_braces
                ),
                suggestion: None,
                position: None,
                confidence: 0.95,
                detected_at: Utc::now(),
            });
        }

        // Store issues
        for issue in &issues {
            self.issues.push_back(issue.clone());
            self.stats.issues_detected += 1;
        }

        // Enforce size limit
        while self.issues.len() > self.config.max_issues {
            self.issues.pop_front();
        }

        issues
    }

    /// Get recent issues
    pub fn recent_issues(&self, limit: usize) -> Vec<&CodeIssue> {
        self.issues.iter().rev().take(limit).collect()
    }

    /// Get issues by severity
    pub fn issues_by_severity(&self, severity: IssueSeverity) -> Vec<&CodeIssue> {
        self.issues
            .iter()
            .filter(|i| i.severity == severity)
            .collect()
    }

    // ========================================================================
    // CONTEXT MANAGEMENT
    // ========================================================================

    /// Set current file context
    pub fn set_current_file(&mut self, file: &str) {
        self.current_context.current_file = Some(file.to_string());
    }

    /// Enter a new scope
    pub fn enter_scope(&mut self) {
        self.current_context.scope_stack.push(HashMap::new());
    }

    /// Exit current scope
    pub fn exit_scope(&mut self) {
        self.current_context.scope_stack.pop();
    }

    /// Add variable to current scope
    pub fn add_to_scope(&mut self, name: String, type_info: String) {
        if let Some(scope) = self.current_context.scope_stack.last_mut() {
            scope.insert(name, type_info);
        }
    }

    /// Look up variable in scope chain
    pub fn lookup_in_scope(&self, name: &str) -> Option<&String> {
        for scope in self.current_context.scope_stack.iter().rev() {
            if let Some(type_info) = scope.get(name) {
                return Some(type_info);
            }
        }
        None
    }

    // ========================================================================
    // STATISTICS
    // ========================================================================

    /// Get cortex statistics
    pub fn stats(&self) -> &LanguageCortexStats {
        &self.stats
    }

    /// Get most familiar entities (highest access count)
    pub fn most_familiar(&self, limit: usize) -> Vec<&SemanticEntity> {
        let mut entities: Vec<_> = self.entities.values().collect();
        entities.sort_by(|a, b| b.access_count.cmp(&a.access_count));
        entities.truncate(limit);
        entities
    }

    /// Decay familiarity (call periodically)
    pub fn decay_familiarity(&mut self) {
        for entity in self.entities.values_mut() {
            entity.confidence *= 1.0 - self.config.familiarity_decay_rate;
        }
    }
}

impl Default for LanguageCortex {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_syntax_parsing() {
        let mut cortex = LanguageCortex::new();

        let code = r#"
fn hello_world() {
    let x = 42;
    println!("Hello");
}

struct Point {
    x: f64,
    y: f64,
}
"#;

        let elements = cortex.analyze_syntax(code, Some("rust"));

        // Should find function and struct
        assert!(elements.iter().any(|e| matches!(
            e,
            SyntaxElement::Block {
                kind: BlockKind::Function,
                ..
            }
        )));
        assert!(elements.iter().any(|e| matches!(
            e,
            SyntaxElement::Block {
                kind: BlockKind::Struct,
                ..
            }
        )));
        assert!(elements.iter().any(|e| matches!(
            e,
            SyntaxElement::Declaration {
                kind: DeclarationKind::Variable,
                ..
            }
        )));
    }

    #[test]
    fn test_entity_registration() {
        let mut cortex = LanguageCortex::new();

        cortex.register_entity(
            "my_var".to_string(),
            EntityKind::Variable,
            Some("i32".to_string()),
        );
        cortex.register_entity("my_func".to_string(), EntityKind::Function, None);

        let entity = cortex.lookup_entity("my_var");
        assert!(entity.is_some());
        assert_eq!(entity.unwrap().inferred_type, Some("i32".to_string()));
    }

    #[test]
    fn test_pattern_recognition() {
        let mut cortex = LanguageCortex::new();

        let code_with_unwrap = "let value = result.unwrap();";
        let patterns = cortex.recognize_patterns(code_with_unwrap);

        assert!(patterns.iter().any(|p| p.id == "result_unwrap"));
    }

    #[test]
    fn test_issue_detection() {
        let mut cortex = LanguageCortex::new();

        let problematic_code = r#"
let x = value.unwrap();
// TODO: fix this later
let y
"#;

        let issues = cortex.detect_issues(problematic_code);

        assert!(
            issues
                .iter()
                .any(|i| i.category == IssueCategory::PotentialBug)
        );
        assert!(issues.iter().any(|i| i.message.contains("TODO")));
    }

    #[test]
    fn test_scope_tracking() {
        let mut cortex = LanguageCortex::new();

        cortex.enter_scope();
        cortex.add_to_scope("x".to_string(), "i32".to_string());

        assert_eq!(cortex.lookup_in_scope("x"), Some(&"i32".to_string()));

        cortex.enter_scope();
        cortex.add_to_scope("y".to_string(), "String".to_string());

        // Can see both x and y
        assert_eq!(cortex.lookup_in_scope("x"), Some(&"i32".to_string()));
        assert_eq!(cortex.lookup_in_scope("y"), Some(&"String".to_string()));

        cortex.exit_scope();

        // y is out of scope
        assert!(cortex.lookup_in_scope("y").is_none());
    }

    #[test]
    fn test_unbalanced_brackets() {
        let mut cortex = LanguageCortex::new();

        let code = "fn test() { if true { }";
        let issues = cortex.detect_issues(code);

        assert!(
            issues
                .iter()
                .any(|i| i.category == IssueCategory::SyntaxError)
        );
    }
}
