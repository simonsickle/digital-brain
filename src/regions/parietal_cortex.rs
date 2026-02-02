//! Parietal Cortex - Spatial and Structural Reasoning
//!
//! The parietal cortex handles spatial processing and structural understanding:
//! - **Spatial awareness**: Understanding where things are in relation to each other
//! - **Navigation**: Finding paths through complex structures
//! - **Integration**: Combining information from multiple sources
//! - **Attention**: Directing focus to specific regions
//!
//! For code understanding, this translates to:
//! - **Module structure**: Understanding package/module hierarchy
//! - **Dependency graphs**: Mapping relationships between components
//! - **Code navigation**: Finding paths through the codebase
//! - **Architecture**: Understanding high-level system structure
//!
//! # Theoretical Foundation
//!
//! Based on:
//! - Posterior Parietal Cortex: Spatial attention and coordinate transforms
//! - Superior Parietal Lobule: Visuo-spatial processing, mental rotation
//! - Inferior Parietal Lobule: Semantic/conceptual knowledge integration
//!
//! # Architecture
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────────────┐
//! │                      PARIETAL CORTEX                               │
//! ├──────────────────┬──────────────────┬──────────────────┬──────────┤
//! │   Spatial Map    │   Dependency     │   Navigation     │ Attention │
//! │   (structure)    │   Graph          │   Engine         │ Focus     │
//! │   - hierarchy    │   - imports      │   - pathfinding  │ - regions │
//! │   - modules      │   - exports      │   - traversal    │ - hotspots│
//! │   - boundaries   │   - coupling     │   - landmarks    │ - scope   │
//! └──────────────────┴──────────────────┴──────────────────┴──────────┘
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

// ============================================================================
// SPATIAL STRUCTURES
// ============================================================================

/// A node in the code structure map
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructureNode {
    /// Unique identifier
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// What kind of structure this is
    pub kind: StructureKind,
    /// Path from root (e.g., "src/core/brain.rs")
    pub path: String,
    /// Parent node ID (if any)
    pub parent: Option<String>,
    /// Child node IDs
    pub children: Vec<String>,
    /// Depth in the hierarchy (0 = root)
    pub depth: usize,
    /// Size metric (lines, bytes, complexity)
    pub size: usize,
    /// When first discovered
    pub discovered_at: DateTime<Utc>,
    /// How often accessed (for attention weighting)
    pub access_count: u32,
    /// Importance score (0-1)
    pub importance: f64,
}

/// Kinds of structural elements
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StructureKind {
    /// Root of the codebase
    Root,
    /// A workspace/monorepo
    Workspace,
    /// A package/crate
    Package,
    /// A module/file
    Module,
    /// A class/struct
    Class,
    /// A function/method
    Function,
    /// A test module
    TestModule,
    /// Configuration file
    Config,
    /// Documentation
    Documentation,
    /// External dependency
    ExternalDep,
}

// ============================================================================
// DEPENDENCY GRAPH
// ============================================================================

/// An edge in the dependency graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyEdge {
    /// Source node ID
    pub from: String,
    /// Target node ID
    pub to: String,
    /// Kind of dependency
    pub kind: DependencyKind,
    /// Strength of coupling (0-1)
    pub coupling: f64,
    /// Is this a critical path?
    pub critical: bool,
    /// When discovered
    pub discovered_at: DateTime<Utc>,
}

/// Kinds of dependencies
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DependencyKind {
    /// Module imports another
    Import,
    /// Type uses another type
    TypeUsage,
    /// Function calls another
    FunctionCall,
    /// Inheritance/extension
    Extends,
    /// Implementation
    Implements,
    /// Composition (contains)
    Contains,
    /// Test dependency
    TestDependency,
    /// Build/compile dependency
    BuildDependency,
}

// ============================================================================
// NAVIGATION
// ============================================================================

/// A navigation path through the codebase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NavigationPath {
    /// Sequence of node IDs to traverse
    pub nodes: Vec<String>,
    /// Total distance/cost
    pub cost: f64,
    /// Why this path was chosen
    pub rationale: String,
    /// Landmarks along the way
    pub landmarks: Vec<Landmark>,
}

/// A landmark for navigation (memorable structure)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Landmark {
    /// Node ID
    pub node_id: String,
    /// Why it's memorable
    pub significance: String,
    /// How prominent (0-1)
    pub prominence: f64,
}

/// A region of focused attention
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionRegion {
    /// Region identifier
    pub id: String,
    /// Center node
    pub center: String,
    /// Nodes within this region
    pub nodes: HashSet<String>,
    /// Why attention is here
    pub reason: String,
    /// Attention strength (0-1)
    pub strength: f64,
    /// When focus started
    pub started_at: DateTime<Utc>,
    /// Duration of focus
    pub duration_secs: u64,
}

// ============================================================================
// HOTSPOTS AND COMPLEXITY
// ============================================================================

/// A complexity hotspot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hotspot {
    /// Node ID
    pub node_id: String,
    /// Complexity score
    pub complexity: f64,
    /// What contributes to complexity
    pub factors: Vec<ComplexityFactor>,
    /// Suggested actions
    pub suggestions: Vec<String>,
}

/// Factors contributing to complexity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityFactor {
    /// Factor name
    pub name: String,
    /// Contribution to complexity
    pub contribution: f64,
    /// Description
    pub description: String,
}

// ============================================================================
// CONFIGURATION
// ============================================================================

/// Configuration for Parietal Cortex
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParietalCortexConfig {
    /// Maximum nodes in structure map
    pub max_nodes: usize,
    /// Maximum edges in dependency graph
    pub max_edges: usize,
    /// Maximum landmarks to remember
    pub max_landmarks: usize,
    /// Complexity threshold for hotspot detection
    pub hotspot_threshold: f64,
    /// Decay rate for attention
    pub attention_decay_rate: f64,
    /// Enable cyclic dependency detection
    pub detect_cycles: bool,
}

impl Default for ParietalCortexConfig {
    fn default() -> Self {
        Self {
            max_nodes: 5000,
            max_edges: 20000,
            max_landmarks: 100,
            hotspot_threshold: 0.7,
            attention_decay_rate: 0.05,
            detect_cycles: true,
        }
    }
}

// ============================================================================
// PARIETAL CORTEX
// ============================================================================

/// Parietal Cortex - Spatial and structural reasoning
///
/// Handles understanding of code structure, dependencies, and navigation.
pub struct ParietalCortex {
    config: ParietalCortexConfig,
    /// Structure map (nodes)
    nodes: HashMap<String, StructureNode>,
    /// Dependency graph (edges)
    edges: Vec<DependencyEdge>,
    /// Current attention regions
    attention_regions: Vec<AttentionRegion>,
    /// Discovered landmarks (reserved for future navigation features)
    #[allow(dead_code)]
    landmarks: Vec<Landmark>,
    /// Detected hotspots
    hotspots: Vec<Hotspot>,
    /// Navigation history
    nav_history: VecDeque<String>,
    /// Statistics
    stats: ParietalCortexStats,
}

/// Statistics for Parietal Cortex
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ParietalCortexStats {
    /// Total nodes tracked
    pub total_nodes: usize,
    /// Total edges tracked
    pub total_edges: usize,
    /// Navigation queries performed
    pub navigation_queries: u64,
    /// Cycles detected
    pub cycles_detected: u64,
    /// Hotspots found
    pub hotspots_found: usize,
    /// Average coupling
    pub avg_coupling: f64,
    /// Max depth in hierarchy
    pub max_depth: usize,
}

impl ParietalCortex {
    /// Create a new Parietal Cortex
    pub fn new() -> Self {
        Self::with_config(ParietalCortexConfig::default())
    }

    /// Create with specific configuration
    pub fn with_config(config: ParietalCortexConfig) -> Self {
        Self {
            config,
            nodes: HashMap::new(),
            edges: Vec::new(),
            attention_regions: Vec::new(),
            landmarks: Vec::new(),
            hotspots: Vec::new(),
            nav_history: VecDeque::new(),
            stats: ParietalCortexStats::default(),
        }
    }

    // ========================================================================
    // STRUCTURE MAPPING
    // ========================================================================

    /// Add a structure node
    pub fn add_node(&mut self, node: StructureNode) {
        let id = node.id.clone();
        let depth = node.depth;

        // Update max depth
        if depth > self.stats.max_depth {
            self.stats.max_depth = depth;
        }

        // Add to parent's children if parent exists
        if let Some(ref parent_id) = node.parent
            && let Some(parent) = self.nodes.get_mut(parent_id)
            && !parent.children.contains(&id)
        {
            parent.children.push(id.clone());
        }

        self.nodes.insert(id, node);
        self.stats.total_nodes = self.nodes.len();

        // Enforce size limit by removing least accessed
        if self.nodes.len() > self.config.max_nodes {
            self.prune_least_accessed();
        }
    }

    /// Create and add a module node
    pub fn add_module(&mut self, path: &str, parent: Option<&str>, size: usize) -> String {
        let id = format!("mod:{}", path);
        let name = path.split('/').next_back().unwrap_or(path).to_string();
        let depth = parent
            .map(|p| self.nodes.get(p).map(|n| n.depth + 1).unwrap_or(0))
            .unwrap_or(0);

        let node = StructureNode {
            id: id.clone(),
            name,
            kind: StructureKind::Module,
            path: path.to_string(),
            parent: parent.map(|s| s.to_string()),
            children: Vec::new(),
            depth,
            size,
            discovered_at: Utc::now(),
            access_count: 0,
            importance: 0.5,
        };

        self.add_node(node);
        id
    }

    /// Create and add a function node
    pub fn add_function(&mut self, name: &str, module_id: &str, size: usize) -> String {
        let id = format!("fn:{}::{}", module_id, name);
        let depth = self.nodes.get(module_id).map(|n| n.depth + 1).unwrap_or(0);

        let node = StructureNode {
            id: id.clone(),
            name: name.to_string(),
            kind: StructureKind::Function,
            path: format!("{}::{}", module_id, name),
            parent: Some(module_id.to_string()),
            children: Vec::new(),
            depth,
            size,
            discovered_at: Utc::now(),
            access_count: 0,
            importance: 0.3,
        };

        self.add_node(node);
        id
    }

    /// Get a node by ID
    pub fn get_node(&mut self, id: &str) -> Option<&StructureNode> {
        if let Some(node) = self.nodes.get_mut(id) {
            node.access_count += 1;
        }
        self.nodes.get(id)
    }

    /// Get all children of a node
    pub fn get_children(&self, id: &str) -> Vec<&StructureNode> {
        self.nodes
            .get(id)
            .map(|n| {
                n.children
                    .iter()
                    .filter_map(|cid| self.nodes.get(cid))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get ancestors of a node (path to root)
    pub fn get_ancestors(&self, id: &str) -> Vec<&StructureNode> {
        let mut ancestors = Vec::new();
        let mut current_id = id;

        while let Some(node) = self.nodes.get(current_id) {
            if let Some(ref parent_id) = node.parent {
                if let Some(parent) = self.nodes.get(parent_id) {
                    ancestors.push(parent);
                    current_id = parent_id;
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        ancestors
    }

    /// Prune least accessed nodes
    fn prune_least_accessed(&mut self) {
        // Collect (id, access_count) pairs first
        let mut node_access: Vec<_> = self
            .nodes
            .iter()
            .map(|(id, n)| (id.clone(), n.access_count))
            .collect();
        node_access.sort_by_key(|(_, count)| *count);

        // Remove bottom 10%
        let to_remove = self.config.max_nodes / 10;
        for (id, _) in node_access.into_iter().take(to_remove) {
            self.nodes.remove(&id);
        }
    }

    // ========================================================================
    // DEPENDENCY GRAPH
    // ========================================================================

    /// Add a dependency edge
    pub fn add_dependency(&mut self, from: &str, to: &str, kind: DependencyKind, coupling: f64) {
        // Check for existing edge
        let exists = self
            .edges
            .iter()
            .any(|e| e.from == from && e.to == to && e.kind == kind);
        if exists {
            return;
        }

        let edge = DependencyEdge {
            from: from.to_string(),
            to: to.to_string(),
            kind,
            coupling: coupling.clamp(0.0, 1.0),
            critical: coupling > 0.8,
            discovered_at: Utc::now(),
        };

        self.edges.push(edge);
        self.stats.total_edges = self.edges.len();

        // Update average coupling
        let total_coupling: f64 = self.edges.iter().map(|e| e.coupling).sum();
        self.stats.avg_coupling = total_coupling / self.edges.len() as f64;

        // Enforce size limit
        if self.edges.len() > self.config.max_edges {
            // Remove oldest non-critical edges
            self.edges.retain(|e| e.critical);
            if self.edges.len() > self.config.max_edges {
                self.edges.truncate(self.config.max_edges);
            }
        }

        // Check for cycles if enabled
        if self.config.detect_cycles && self.detect_cycle(from).is_some() {
            self.stats.cycles_detected += 1;
        }
    }

    /// Get dependencies of a node
    pub fn get_dependencies(&self, node_id: &str) -> Vec<&DependencyEdge> {
        self.edges.iter().filter(|e| e.from == node_id).collect()
    }

    /// Get dependents of a node (what depends on this)
    pub fn get_dependents(&self, node_id: &str) -> Vec<&DependencyEdge> {
        self.edges.iter().filter(|e| e.to == node_id).collect()
    }

    /// Detect a cycle starting from a node
    pub fn detect_cycle(&self, start: &str) -> Option<Vec<String>> {
        let mut visited = HashSet::new();
        let mut path = Vec::new();

        fn dfs(
            node: &str,
            edges: &[DependencyEdge],
            visited: &mut HashSet<String>,
            path: &mut Vec<String>,
        ) -> Option<Vec<String>> {
            if path.contains(&node.to_string()) {
                // Found cycle
                let cycle_start = path.iter().position(|n| n == node).unwrap();
                return Some(path[cycle_start..].to_vec());
            }

            if visited.contains(node) {
                return None;
            }

            visited.insert(node.to_string());
            path.push(node.to_string());

            for edge in edges.iter().filter(|e| e.from == node) {
                if let Some(cycle) = dfs(&edge.to, edges, visited, path) {
                    return Some(cycle);
                }
            }

            path.pop();
            None
        }

        dfs(start, &self.edges, &mut visited, &mut path)
    }

    // ========================================================================
    // NAVIGATION
    // ========================================================================

    /// Find shortest path between two nodes
    pub fn find_path(&mut self, from: &str, to: &str) -> Option<NavigationPath> {
        self.stats.navigation_queries += 1;

        // BFS for shortest path
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut predecessors: HashMap<String, String> = HashMap::new();

        queue.push_back(from.to_string());
        visited.insert(from.to_string());

        while let Some(current) = queue.pop_front() {
            if current == to {
                // Reconstruct path
                let mut path = vec![to.to_string()];
                let mut node = to.to_string();
                while let Some(pred) = predecessors.get(&node) {
                    path.push(pred.clone());
                    node = pred.clone();
                }
                path.reverse();

                return Some(NavigationPath {
                    nodes: path.clone(),
                    cost: path.len() as f64 - 1.0,
                    rationale: "Shortest path through dependency graph".to_string(),
                    landmarks: self.find_landmarks_on_path(&path),
                });
            }

            // Explore neighbors (both dependencies and parent/child)
            let mut neighbors = Vec::new();

            // Dependencies
            for edge in self.edges.iter().filter(|e| e.from == current) {
                neighbors.push(edge.to.clone());
            }
            for edge in self.edges.iter().filter(|e| e.to == current) {
                neighbors.push(edge.from.clone());
            }

            // Hierarchy
            if let Some(node) = self.nodes.get(&current) {
                if let Some(ref parent) = node.parent {
                    neighbors.push(parent.clone());
                }
                neighbors.extend(node.children.clone());
            }

            for neighbor in neighbors {
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor.clone());
                    predecessors.insert(neighbor.clone(), current.clone());
                    queue.push_back(neighbor);
                }
            }
        }

        None // No path found
    }

    /// Find landmarks on a path
    fn find_landmarks_on_path(&self, path: &[String]) -> Vec<Landmark> {
        path.iter()
            .filter_map(|id| {
                self.nodes.get(id).and_then(|node| {
                    if node.importance > 0.7 || node.access_count > 10 {
                        Some(Landmark {
                            node_id: id.clone(),
                            significance: format!("{:?} with high importance", node.kind),
                            prominence: node.importance,
                        })
                    } else {
                        None
                    }
                })
            })
            .collect()
    }

    /// Record navigation to a node
    pub fn navigate_to(&mut self, node_id: &str) {
        self.nav_history.push_back(node_id.to_string());
        if self.nav_history.len() > 100 {
            self.nav_history.pop_front();
        }

        // Update access count
        if let Some(node) = self.nodes.get_mut(node_id) {
            node.access_count += 1;
        }
    }

    /// Get navigation history
    pub fn navigation_history(&self) -> Vec<&str> {
        self.nav_history.iter().map(|s| s.as_str()).collect()
    }

    // ========================================================================
    // ATTENTION
    // ========================================================================

    /// Focus attention on a region
    pub fn focus_attention(&mut self, center: &str, reason: &str) -> String {
        let id = format!("attention:{}", Utc::now().timestamp_millis());

        // Find related nodes (within 2 hops)
        let mut nodes = HashSet::new();
        nodes.insert(center.to_string());

        // Add direct dependencies
        for edge in &self.edges {
            if edge.from == center || edge.to == center {
                nodes.insert(edge.from.clone());
                nodes.insert(edge.to.clone());
            }
        }

        // Add children and parent
        if let Some(node) = self.nodes.get(center) {
            if let Some(ref parent) = node.parent {
                nodes.insert(parent.clone());
            }
            for child in &node.children {
                nodes.insert(child.clone());
            }
        }

        let region = AttentionRegion {
            id: id.clone(),
            center: center.to_string(),
            nodes,
            reason: reason.to_string(),
            strength: 1.0,
            started_at: Utc::now(),
            duration_secs: 0,
        };

        self.attention_regions.push(region);
        id
    }

    /// Update attention (call periodically)
    pub fn decay_attention(&mut self) {
        let now = Utc::now();

        for region in &mut self.attention_regions {
            region.duration_secs = (now - region.started_at).num_seconds().max(0) as u64;
            region.strength *= 1.0 - self.config.attention_decay_rate;
        }

        // Remove weak attention regions
        self.attention_regions.retain(|r| r.strength > 0.1);
    }

    /// Get current attention regions
    pub fn current_attention(&self) -> &[AttentionRegion] {
        &self.attention_regions
    }

    // ========================================================================
    // COMPLEXITY ANALYSIS
    // ========================================================================

    /// Analyze and find hotspots
    pub fn find_hotspots(&mut self) -> Vec<&Hotspot> {
        self.hotspots.clear();

        for (id, node) in &self.nodes {
            let mut factors = Vec::new();
            let mut complexity = 0.0;

            // Size contribution
            let size_factor = (node.size as f64 / 500.0).min(1.0);
            if size_factor > 0.3 {
                factors.push(ComplexityFactor {
                    name: "Size".to_string(),
                    contribution: size_factor * 0.3,
                    description: format!("{} lines", node.size),
                });
                complexity += size_factor * 0.3;
            }

            // Dependency count
            let deps = self.edges.iter().filter(|e| e.from == *id).count();
            let dep_factor = (deps as f64 / 10.0).min(1.0);
            if dep_factor > 0.3 {
                factors.push(ComplexityFactor {
                    name: "Dependencies".to_string(),
                    contribution: dep_factor * 0.25,
                    description: format!("{} outgoing dependencies", deps),
                });
                complexity += dep_factor * 0.25;
            }

            // Dependents count (fan-in)
            let dependents = self.edges.iter().filter(|e| e.to == *id).count();
            let fan_in_factor = (dependents as f64 / 10.0).min(1.0);
            if fan_in_factor > 0.3 {
                factors.push(ComplexityFactor {
                    name: "Fan-in".to_string(),
                    contribution: fan_in_factor * 0.25,
                    description: format!("{} dependents (high coupling)", dependents),
                });
                complexity += fan_in_factor * 0.25;
            }

            // Depth in hierarchy
            let depth_factor = (node.depth as f64 / 5.0).min(1.0);
            if depth_factor > 0.5 {
                factors.push(ComplexityFactor {
                    name: "Nesting".to_string(),
                    contribution: depth_factor * 0.2,
                    description: format!("Depth {} in hierarchy", node.depth),
                });
                complexity += depth_factor * 0.2;
            }

            if complexity >= self.config.hotspot_threshold {
                let mut suggestions = Vec::new();
                if size_factor > 0.5 {
                    suggestions.push("Consider breaking into smaller modules".to_string());
                }
                if dep_factor > 0.5 {
                    suggestions.push(
                        "High number of dependencies - consider dependency injection".to_string(),
                    );
                }
                if fan_in_factor > 0.5 {
                    suggestions
                        .push("Many components depend on this - ensure stability".to_string());
                }

                self.hotspots.push(Hotspot {
                    node_id: id.clone(),
                    complexity,
                    factors,
                    suggestions,
                });
            }
        }

        self.stats.hotspots_found = self.hotspots.len();
        self.hotspots.iter().collect()
    }

    // ========================================================================
    // STATISTICS
    // ========================================================================

    /// Get statistics
    pub fn stats(&self) -> &ParietalCortexStats {
        &self.stats
    }

    /// Get summary of structure
    pub fn structure_summary(&self) -> String {
        let modules = self
            .nodes
            .values()
            .filter(|n| n.kind == StructureKind::Module)
            .count();
        let functions = self
            .nodes
            .values()
            .filter(|n| n.kind == StructureKind::Function)
            .count();
        let deps = self.edges.len();

        format!(
            "Structure: {} modules, {} functions, {} dependencies (max depth: {})",
            modules, functions, deps, self.stats.max_depth
        )
    }

    /// Get most important nodes
    pub fn most_important(&self, limit: usize) -> Vec<&StructureNode> {
        let mut nodes: Vec<_> = self.nodes.values().collect();
        nodes.sort_by(|a, b| b.importance.partial_cmp(&a.importance).unwrap());
        nodes.truncate(limit);
        nodes
    }
}

impl Default for ParietalCortex {
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
    fn test_structure_mapping() {
        let mut cortex = ParietalCortex::new();

        let mod1 = cortex.add_module("src/lib.rs", None, 100);
        let mod2 = cortex.add_module("src/brain.rs", Some(&mod1), 500);
        let _fn1 = cortex.add_function("process", &mod2, 50);

        assert_eq!(cortex.nodes.len(), 3);
        assert_eq!(cortex.stats.max_depth, 2);

        let children = cortex.get_children(&mod1);
        assert_eq!(children.len(), 1);
    }

    #[test]
    fn test_dependency_graph() {
        let mut cortex = ParietalCortex::new();

        cortex.add_module("mod_a", None, 100);
        cortex.add_module("mod_b", None, 100);
        cortex.add_module("mod_c", None, 100);

        cortex.add_dependency("mod:mod_a", "mod:mod_b", DependencyKind::Import, 0.5);
        cortex.add_dependency("mod:mod_b", "mod:mod_c", DependencyKind::Import, 0.5);

        let deps = cortex.get_dependencies("mod:mod_a");
        assert_eq!(deps.len(), 1);
    }

    #[test]
    fn test_cycle_detection() {
        let mut cortex = ParietalCortex::new();

        cortex.add_module("mod_a", None, 100);
        cortex.add_module("mod_b", None, 100);
        cortex.add_module("mod_c", None, 100);

        cortex.add_dependency("mod:mod_a", "mod:mod_b", DependencyKind::Import, 0.5);
        cortex.add_dependency("mod:mod_b", "mod:mod_c", DependencyKind::Import, 0.5);
        cortex.add_dependency("mod:mod_c", "mod:mod_a", DependencyKind::Import, 0.5);

        let cycle = cortex.detect_cycle("mod:mod_a");
        assert!(cycle.is_some());
    }

    #[test]
    fn test_navigation() {
        let mut cortex = ParietalCortex::new();

        cortex.add_module("mod_a", None, 100);
        cortex.add_module("mod_b", None, 100);
        cortex.add_module("mod_c", None, 100);

        cortex.add_dependency("mod:mod_a", "mod:mod_b", DependencyKind::Import, 0.5);
        cortex.add_dependency("mod:mod_b", "mod:mod_c", DependencyKind::Import, 0.5);

        let path = cortex.find_path("mod:mod_a", "mod:mod_c");
        assert!(path.is_some());
        assert_eq!(path.unwrap().nodes.len(), 3);
    }

    #[test]
    fn test_attention() {
        let mut cortex = ParietalCortex::new();

        cortex.add_module("mod_a", None, 100);
        cortex.add_module("mod_b", None, 100);
        cortex.add_dependency("mod:mod_a", "mod:mod_b", DependencyKind::Import, 0.5);

        cortex.focus_attention("mod:mod_a", "Investigating issue");

        assert_eq!(cortex.current_attention().len(), 1);
        assert!(cortex.current_attention()[0].nodes.contains("mod:mod_a"));
    }
}
