#!/usr/bin/env python3
"""
KIMERA SWM System - Module Architecture Analyzer
==============================================

Phase 3.1: Module Boundary Clarification Implementation
Analyzes current module structure, dependencies, and provides optimization recommendations.

Features:
- Module dependency analysis and visualization
- Circular dependency detection
- API contract validation
- Code duplication identification
- Module cohesion and coupling metrics
- Architecture optimization recommendations
- Interface design validation

Author: KIMERA Development Team
Date: 2025-01-31
Phase: 3.1 - Module Boundary Clarification
"""

import ast
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque
import importlib.util
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import json
import re
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModuleInfo:
    """Information about a module."""
    name: str
    path: str
    imports: Set[str] = field(default_factory=set)
    exports: Set[str] = field(default_factory=set)
    classes: Set[str] = field(default_factory=set)
    functions: Set[str] = field(default_factory=set)
    lines_of_code: int = 0
    complexity_score: float = 0.0
    last_modified: Optional[datetime] = None
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)

@dataclass
class DependencyRelation:
    """Represents a dependency relationship between modules."""
    source: str
    target: str
    import_type: str  # 'direct', 'from', 'relative'
    usage_count: int = 1
    critical: bool = False

@dataclass
class ArchitecturalIssue:
    """Represents an architectural issue."""
    issue_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    affected_modules: List[str]
    recommendation: str
    estimated_effort: str

@dataclass
class ModuleCluster:
    """Represents a logical grouping of related modules."""
    name: str
    modules: Set[str]
    purpose: str
    internal_cohesion: float = 0.0
    external_coupling: float = 0.0

class ModuleAnalyzer:
    """Comprehensive module architecture analyzer."""
    
    def __init__(self, root_path: str = "src"):
        self.root_path = Path(root_path)
        self.modules: Dict[str, ModuleInfo] = {}
        self.dependencies: List[DependencyRelation] = []
        self.issues: List[ArchitecturalIssue] = []
        self.clusters: List[ModuleCluster] = []
        self.dependency_graph = nx.DiGraph()
        
    def analyze_codebase(self) -> Dict[str, Any]:
        """Perform complete codebase analysis."""
        logger.info("🔍 Starting comprehensive module analysis...")
        
        # Step 1: Discover and analyze modules
        self._discover_modules()
        
        # Step 2: Analyze dependencies
        self._analyze_dependencies()
        
        # Step 3: Build dependency graph
        self._build_dependency_graph()
        
        # Step 4: Detect architectural issues
        self._detect_issues()
        
        # Step 5: Identify module clusters
        self._identify_clusters()
        
        # Step 6: Calculate metrics
        metrics = self._calculate_metrics()
        
        # Step 7: Generate recommendations
        recommendations = self._generate_recommendations()
        
        logger.info("✅ Module analysis completed!")
        
        return {
            "modules": {name: self._module_to_dict(module) for name, module in self.modules.items()},
            "dependencies": [self._dependency_to_dict(dep) for dep in self.dependencies],
            "issues": [self._issue_to_dict(issue) for issue in self.issues],
            "clusters": [self._cluster_to_dict(cluster) for cluster in self.clusters],
            "metrics": metrics,
            "recommendations": recommendations,
            "analysis_date": datetime.now().isoformat()
        }
    
    def _discover_modules(self):
        """Discover all Python modules in the codebase."""
        logger.info("📂 Discovering modules...")
        
        for py_file in self.root_path.rglob("*.py"):
            if self._should_analyze_file(py_file):
                module_name = self._get_module_name(py_file)
                self.modules[module_name] = self._analyze_module(py_file, module_name)
        
        logger.info(f"Found {len(self.modules)} modules")
    
    def _should_analyze_file(self, file_path: Path) -> bool:
        """Check if file should be analyzed."""
        exclude_patterns = [
            "__pycache__",
            ".git",
            "migrations",
            "venv",
            ".venv",
            "build",
            "dist",
            ".pytest_cache",
            ".mypy_cache"
        ]
        
        return not any(pattern in str(file_path) for pattern in exclude_patterns)
    
    def _get_module_name(self, file_path: Path) -> str:
        """Get module name from file path."""
        relative_path = file_path.relative_to(self.root_path)
        module_parts = list(relative_path.parts[:-1])  # Exclude filename
        
        if relative_path.name != "__init__.py":
            module_parts.append(relative_path.stem)
        
        return ".".join(module_parts) if module_parts else relative_path.stem
    
    def _analyze_module(self, file_path: Path, module_name: str) -> ModuleInfo:
        """Analyze a single module."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            module_info = ModuleInfo(
                name=module_name,
                path=str(file_path),
                last_modified=datetime.fromtimestamp(file_path.stat().st_mtime)
            )
            
            # Analyze AST
            visitor = ModuleVisitor()
            visitor.visit(tree)
            
            module_info.imports = visitor.imports
            module_info.exports = visitor.exports
            module_info.classes = visitor.classes
            module_info.functions = visitor.functions
            module_info.lines_of_code = len([line for line in content.split('\n') if line.strip()])
            module_info.complexity_score = self._calculate_complexity(tree)
            
            return module_info
            
        except Exception as e:
            logger.warning(f"Error analyzing {file_path}: {e}")
            return ModuleInfo(name=module_name, path=str(file_path))
    
    def _calculate_complexity(self, tree: ast.AST) -> float:
        """Calculate module complexity score."""
        complexity = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.With)):
                complexity += 1
            elif isinstance(node, ast.FunctionDef):
                complexity += 1
            elif isinstance(node, ast.ClassDef):
                complexity += 2
        
        return complexity
    
    def _analyze_dependencies(self):
        """Analyze dependencies between modules."""
        logger.info("🔗 Analyzing dependencies...")
        
        for module_name, module_info in self.modules.items():
            for import_name in module_info.imports:
                # Resolve import to actual module
                resolved_module = self._resolve_import(import_name, module_name)
                
                if resolved_module and resolved_module in self.modules:
                    dependency = DependencyRelation(
                        source=module_name,
                        target=resolved_module,
                        import_type=self._get_import_type(import_name)
                    )
                    
                    self.dependencies.append(dependency)
                    module_info.dependencies.add(resolved_module)
                    self.modules[resolved_module].dependents.add(module_name)
    
    def _resolve_import(self, import_name: str, source_module: str) -> Optional[str]:
        """Resolve import name to actual module name."""
        # Handle relative imports
        if import_name.startswith('.'):
            source_parts = source_module.split('.')
            if import_name.startswith('..'):
                # Go up levels based on dots
                level = len(import_name) - len(import_name.lstrip('.'))
                if level < len(source_parts):
                    base_parts = source_parts[:-level]
                    relative_name = import_name.lstrip('.')
                    if relative_name:
                        return '.'.join(base_parts + [relative_name])
                    else:
                        return '.'.join(base_parts)
            else:
                # Single dot, same package
                base_parts = source_parts[:-1] if source_parts else []
                relative_name = import_name[1:]
                if relative_name:
                    return '.'.join(base_parts + [relative_name])
        
        # Check if it's a direct module match
        if import_name in self.modules:
            return import_name
        
        # Check for partial matches (import from submodule)
        for module_name in self.modules:
            if module_name.startswith(import_name + '.') or import_name.startswith(module_name + '.'):
                return module_name
        
        return None
    
    def _get_import_type(self, import_name: str) -> str:
        """Determine the type of import."""
        if import_name.startswith('.'):
            return 'relative'
        elif '.' in import_name:
            return 'from'
        else:
            return 'direct'
    
    def _build_dependency_graph(self):
        """Build networkx dependency graph."""
        logger.info("📊 Building dependency graph...")
        
        # Add nodes
        for module_name in self.modules:
            self.dependency_graph.add_node(module_name)
        
        # Add edges
        for dependency in self.dependencies:
            self.dependency_graph.add_edge(
                dependency.source,
                dependency.target,
                weight=dependency.usage_count,
                type=dependency.import_type
            )
    
    def _detect_issues(self):
        """Detect architectural issues."""
        logger.info("🔍 Detecting architectural issues...")
        
        # Detect circular dependencies
        self._detect_circular_dependencies()
        
        # Detect high coupling
        self._detect_high_coupling()
        
        # Detect low cohesion
        self._detect_low_cohesion()
        
        # Detect code duplication
        self._detect_code_duplication()
        
        # Detect interface violations
        self._detect_interface_violations()
    
    def _detect_circular_dependencies(self):
        """Detect circular dependencies."""
        try:
            cycles = list(nx.simple_cycles(self.dependency_graph))
            
            for cycle in cycles:
                if len(cycle) > 1:
                    self.issues.append(ArchitecturalIssue(
                        issue_type="circular_dependency",
                        severity="high" if len(cycle) > 2 else "medium",
                        description=f"Circular dependency detected: {' -> '.join(cycle + [cycle[0]])}",
                        affected_modules=cycle,
                        recommendation="Refactor to break circular dependency using dependency injection or interface segregation",
                        estimated_effort="medium"
                    ))
        except Exception as e:
            logger.warning(f"Error detecting circular dependencies: {e}")
    
    def _detect_high_coupling(self):
        """Detect modules with high coupling."""
        for module_name, module_info in self.modules.items():
            coupling_score = len(module_info.dependencies) + len(module_info.dependents)
            
            if coupling_score > 10:  # Threshold for high coupling
                self.issues.append(ArchitecturalIssue(
                    issue_type="high_coupling",
                    severity="medium" if coupling_score < 20 else "high",
                    description=f"Module {module_name} has high coupling (score: {coupling_score})",
                    affected_modules=[module_name],
                    recommendation="Consider breaking down the module or using interfaces to reduce coupling",
                    estimated_effort="high"
                ))
    
    def _detect_low_cohesion(self):
        """Detect modules with low cohesion."""
        for module_name, module_info in self.modules.items():
            # Simple heuristic: modules with many unrelated functions/classes
            if len(module_info.classes) + len(module_info.functions) > 20:
                self.issues.append(ArchitecturalIssue(
                    issue_type="low_cohesion",
                    severity="medium",
                    description=f"Module {module_name} may have low cohesion (many unrelated components)",
                    affected_modules=[module_name],
                    recommendation="Consider splitting into smaller, more focused modules",
                    estimated_effort="medium"
                ))
    
    def _detect_code_duplication(self):
        """Detect potential code duplication."""
        # Simple heuristic: modules with very similar names or class/function names
        module_names = list(self.modules.keys())
        
        for i, module1 in enumerate(module_names):
            for module2 in module_names[i+1:]:
                similarity = self._calculate_name_similarity(module1, module2)
                
                if similarity > 0.8:  # High similarity threshold
                    self.issues.append(ArchitecturalIssue(
                        issue_type="potential_duplication",
                        severity="low",
                        description=f"Modules {module1} and {module2} have similar names (similarity: {similarity:.2f})",
                        affected_modules=[module1, module2],
                        recommendation="Review for potential code duplication or consolidation opportunities",
                        estimated_effort="low"
                    ))
    
    def _detect_interface_violations(self):
        """Detect interface/API violations."""
        for module_name, module_info in self.modules.items():
            # Check for modules importing from implementation details
            for dep in module_info.dependencies:
                if dep.endswith('_impl') or 'internal' in dep:
                    self.issues.append(ArchitecturalIssue(
                        issue_type="interface_violation",
                        severity="high",
                        description=f"Module {module_name} imports from implementation detail {dep}",
                        affected_modules=[module_name, dep],
                        recommendation="Use public interfaces instead of implementation details",
                        estimated_effort="medium"
                    ))
    
    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between two names."""
        # Simple Jaccard similarity on character n-grams
        def get_ngrams(s, n=2):
            return set(s[i:i+n] for i in range(len(s)-n+1))
        
        ngrams1 = get_ngrams(name1)
        ngrams2 = get_ngrams(name2)
        
        if not ngrams1 and not ngrams2:
            return 1.0
        if not ngrams1 or not ngrams2:
            return 0.0
        
        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)
        
        return intersection / union if union > 0 else 0.0
    
    def _identify_clusters(self):
        """Identify logical module clusters."""
        logger.info("🔍 Identifying module clusters...")
        
        # Use community detection on dependency graph
        try:
            if len(self.dependency_graph.nodes()) > 0:
                # Convert to undirected for community detection
                undirected_graph = self.dependency_graph.to_undirected()
                
                # Simple clustering based on path similarity
                clusters = defaultdict(set)
                
                for module_name in self.modules:
                    # Group by top-level package
                    parts = module_name.split('.')
                    if len(parts) > 1:
                        cluster_key = parts[0]
                    else:
                        cluster_key = "root"
                    
                    clusters[cluster_key].add(module_name)
                
                # Create cluster objects
                for cluster_name, modules in clusters.items():
                    if len(modules) > 1:  # Only meaningful clusters
                        cluster = ModuleCluster(
                            name=cluster_name,
                            modules=modules,
                            purpose=self._infer_cluster_purpose(cluster_name, modules)
                        )
                        
                        cluster.internal_cohesion = self._calculate_cluster_cohesion(modules)
                        cluster.external_coupling = self._calculate_cluster_coupling(modules)
                        
                        self.clusters.append(cluster)
                        
        except Exception as e:
            logger.warning(f"Error identifying clusters: {e}")
    
    def _infer_cluster_purpose(self, cluster_name: str, modules: Set[str]) -> str:
        """Infer the purpose of a module cluster."""
        purpose_keywords = {
            'core': 'Core system functionality',
            'api': 'API and interface definitions',
            'utils': 'Utility functions and helpers',
            'tests': 'Test modules and fixtures',
            'config': 'Configuration management',
            'data': 'Data processing and storage',
            'ui': 'User interface components',
            'models': 'Data models and entities',
            'services': 'Business logic services'
        }
        
        for keyword, purpose in purpose_keywords.items():
            if keyword in cluster_name.lower():
                return purpose
        
        return f"Module group: {cluster_name}"
    
    def _calculate_cluster_cohesion(self, modules: Set[str]) -> float:
        """Calculate internal cohesion of a cluster."""
        if len(modules) <= 1:
            return 1.0
        
        internal_connections = 0
        total_possible = len(modules) * (len(modules) - 1)
        
        for module1 in modules:
            for module2 in modules:
                if module1 != module2:
                    if self.dependency_graph.has_edge(module1, module2):
                        internal_connections += 1
        
        return internal_connections / total_possible if total_possible > 0 else 0.0
    
    def _calculate_cluster_coupling(self, modules: Set[str]) -> float:
        """Calculate external coupling of a cluster."""
        external_connections = 0
        
        for module in modules:
            for dependency in self.modules[module].dependencies:
                if dependency not in modules:
                    external_connections += 1
            
            for dependent in self.modules[module].dependents:
                if dependent not in modules:
                    external_connections += 1
        
        return external_connections / len(modules) if modules else 0.0
    
    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate architecture metrics."""
        logger.info("📊 Calculating metrics...")
        
        total_modules = len(self.modules)
        total_dependencies = len(self.dependencies)
        
        # Calculate various metrics
        metrics = {
            "total_modules": total_modules,
            "total_dependencies": total_dependencies,
            "average_dependencies_per_module": total_dependencies / total_modules if total_modules > 0 else 0,
            "cyclomatic_complexity": {
                "average": sum(m.complexity_score for m in self.modules.values()) / total_modules if total_modules > 0 else 0,
                "max": max((m.complexity_score for m in self.modules.values()), default=0),
                "distribution": self._get_complexity_distribution()
            },
            "coupling_metrics": self._calculate_coupling_metrics(),
            "cohesion_metrics": self._calculate_cohesion_metrics(),
            "size_metrics": self._calculate_size_metrics(),
            "issue_summary": self._get_issue_summary()
        }
        
        return metrics
    
    def _get_complexity_distribution(self) -> Dict[str, int]:
        """Get distribution of complexity scores."""
        distribution = {"low": 0, "medium": 0, "high": 0, "very_high": 0}
        
        for module in self.modules.values():
            if module.complexity_score <= 5:
                distribution["low"] += 1
            elif module.complexity_score <= 15:
                distribution["medium"] += 1
            elif module.complexity_score <= 30:
                distribution["high"] += 1
            else:
                distribution["very_high"] += 1
        
        return distribution
    
    def _calculate_coupling_metrics(self) -> Dict[str, float]:
        """Calculate coupling metrics."""
        if not self.modules:
            return {"average": 0.0, "max": 0.0}
        
        coupling_scores = []
        for module in self.modules.values():
            coupling = len(module.dependencies) + len(module.dependents)
            coupling_scores.append(coupling)
        
        return {
            "average": sum(coupling_scores) / len(coupling_scores),
            "max": max(coupling_scores),
            "standard_deviation": self._calculate_std_dev(coupling_scores)
        }
    
    def _calculate_cohesion_metrics(self) -> Dict[str, float]:
        """Calculate cohesion metrics for clusters."""
        if not self.clusters:
            return {"average": 0.0}
        
        cohesion_scores = [cluster.internal_cohesion for cluster in self.clusters]
        return {
            "average": sum(cohesion_scores) / len(cohesion_scores),
            "min": min(cohesion_scores),
            "max": max(cohesion_scores)
        }
    
    def _calculate_size_metrics(self) -> Dict[str, Any]:
        """Calculate size-related metrics."""
        lines_per_module = [module.lines_of_code for module in self.modules.values()]
        
        return {
            "total_lines_of_code": sum(lines_per_module),
            "average_lines_per_module": sum(lines_per_module) / len(lines_per_module) if lines_per_module else 0,
            "largest_module": max(lines_per_module) if lines_per_module else 0,
            "size_distribution": self._get_size_distribution(lines_per_module)
        }
    
    def _get_size_distribution(self, lines_per_module: List[int]) -> Dict[str, int]:
        """Get distribution of module sizes."""
        distribution = {"small": 0, "medium": 0, "large": 0, "very_large": 0}
        
        for lines in lines_per_module:
            if lines <= 100:
                distribution["small"] += 1
            elif lines <= 300:
                distribution["medium"] += 1
            elif lines <= 500:
                distribution["large"] += 1
            else:
                distribution["very_large"] += 1
        
        return distribution
    
    def _calculate_std_dev(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if not values:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def _get_issue_summary(self) -> Dict[str, int]:
        """Get summary of issues by type and severity."""
        summary = defaultdict(lambda: defaultdict(int))
        
        for issue in self.issues:
            summary[issue.issue_type][issue.severity] += 1
        
        return dict(summary)
    
    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate architecture improvement recommendations."""
        logger.info("💡 Generating recommendations...")
        
        recommendations = []
        
        # High-level architectural recommendations
        if len(self.issues) > 0:
            critical_issues = [i for i in self.issues if i.severity == "critical"]
            high_issues = [i for i in self.issues if i.severity == "high"]
            
            if critical_issues:
                recommendations.append({
                    "priority": "critical",
                    "title": "Address Critical Architectural Issues",
                    "description": f"Found {len(critical_issues)} critical issues that need immediate attention",
                    "action_items": [issue.recommendation for issue in critical_issues],
                    "estimated_effort": "high"
                })
            
            if high_issues:
                recommendations.append({
                    "priority": "high",
                    "title": "Resolve High-Priority Issues",
                    "description": f"Found {len(high_issues)} high-priority issues",
                    "action_items": [issue.recommendation for issue in high_issues],
                    "estimated_effort": "medium"
                })
        
        # Module structure recommendations
        if len(self.clusters) < 3 and len(self.modules) > 10:
            recommendations.append({
                "priority": "medium",
                "title": "Improve Module Organization",
                "description": "Consider organizing modules into logical packages",
                "action_items": [
                    "Group related modules into packages",
                    "Define clear interfaces between packages",
                    "Establish naming conventions"
                ],
                "estimated_effort": "medium"
            })
        
        # Performance recommendations
        large_modules = [m for m in self.modules.values() if m.lines_of_code > 500]
        if large_modules:
            recommendations.append({
                "priority": "medium",
                "title": "Split Large Modules",
                "description": f"Found {len(large_modules)} modules with >500 lines",
                "action_items": [
                    "Break down large modules into smaller, focused modules",
                    "Extract common functionality into utilities",
                    "Consider using composition over inheritance"
                ],
                "estimated_effort": "high"
            })
        
        return recommendations
    
    def visualize_dependencies(self, output_path: str = "dependency_graph.png"):
        """Generate dependency graph visualization."""
        logger.info("📊 Generating dependency visualization...")
        
        try:
            plt.figure(figsize=(16, 12))
            
            # Use spring layout for better visualization
            pos = nx.spring_layout(self.dependency_graph, k=3, iterations=50)
            
            # Draw nodes with different colors for different clusters
            cluster_colors = plt.cm.Set3(range(len(self.clusters)))
            node_colors = []
            
            for node in self.dependency_graph.nodes():
                color_assigned = False
                for i, cluster in enumerate(self.clusters):
                    if node in cluster.modules:
                        node_colors.append(cluster_colors[i])
                        color_assigned = True
                        break
                
                if not color_assigned:
                    node_colors.append('lightgray')
            
            # Draw the graph
            nx.draw_networkx_nodes(
                self.dependency_graph, pos,
                node_color=node_colors,
                node_size=300,
                alpha=0.8
            )
            
            nx.draw_networkx_edges(
                self.dependency_graph, pos,
                edge_color='gray',
                arrows=True,
                arrowsize=20,
                alpha=0.6
            )
            
            # Add labels for important nodes
            important_nodes = {
                node: node.split('.')[-1] if '.' in node else node
                for node in self.dependency_graph.nodes()
                if self.dependency_graph.degree(node) > 3
            }
            
            nx.draw_networkx_labels(
                self.dependency_graph, pos,
                labels=important_nodes,
                font_size=8
            )
            
            plt.title("KIMERA Module Dependency Graph", fontsize=16, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Dependency graph saved to {output_path}")
            
        except Exception as e:
            logger.warning(f"Error generating visualization: {e}")
    
    def generate_report(self, output_path: str = "architecture_analysis_report.json"):
        """Generate comprehensive analysis report."""
        logger.info("📄 Generating analysis report...")
        
        analysis_results = self.analyze_codebase()
        
        with open(output_path, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        logger.info(f"Analysis report saved to {output_path}")
        return analysis_results
    
    # Helper methods for serialization
    def _module_to_dict(self, module: ModuleInfo) -> Dict[str, Any]:
        """Convert ModuleInfo to dictionary."""
        return {
            "name": module.name,
            "path": module.path,
            "imports": list(module.imports),
            "exports": list(module.exports),
            "classes": list(module.classes),
            "functions": list(module.functions),
            "lines_of_code": module.lines_of_code,
            "complexity_score": module.complexity_score,
            "last_modified": module.last_modified.isoformat() if module.last_modified else None,
            "dependencies": list(module.dependencies),
            "dependents": list(module.dependents)
        }
    
    def _dependency_to_dict(self, dependency: DependencyRelation) -> Dict[str, Any]:
        """Convert DependencyRelation to dictionary."""
        return {
            "source": dependency.source,
            "target": dependency.target,
            "import_type": dependency.import_type,
            "usage_count": dependency.usage_count,
            "critical": dependency.critical
        }
    
    def _issue_to_dict(self, issue: ArchitecturalIssue) -> Dict[str, Any]:
        """Convert ArchitecturalIssue to dictionary."""
        return {
            "issue_type": issue.issue_type,
            "severity": issue.severity,
            "description": issue.description,
            "affected_modules": issue.affected_modules,
            "recommendation": issue.recommendation,
            "estimated_effort": issue.estimated_effort
        }
    
    def _cluster_to_dict(self, cluster: ModuleCluster) -> Dict[str, Any]:
        """Convert ModuleCluster to dictionary."""
        return {
            "name": cluster.name,
            "modules": list(cluster.modules),
            "purpose": cluster.purpose,
            "internal_cohesion": cluster.internal_cohesion,
            "external_coupling": cluster.external_coupling
        }

class ModuleVisitor(ast.NodeVisitor):
    """AST visitor to extract module information."""
    
    def __init__(self):
        self.imports = set()
        self.exports = set()
        self.classes = set()
        self.functions = set()
    
    def visit_Import(self, node):
        """Visit import statements."""
        for alias in node.names:
            self.imports.add(alias.name)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        """Visit from-import statements."""
        if node.module:
            self.imports.add(node.module)
        self.generic_visit(node)
    
    def visit_ClassDef(self, node):
        """Visit class definitions."""
        self.classes.add(node.name)
        if not node.name.startswith('_'):
            self.exports.add(node.name)
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node):
        """Visit function definitions."""
        self.functions.add(node.name)
        if not node.name.startswith('_'):
            self.exports.add(node.name)
        self.generic_visit(node)
    
    def visit_AsyncFunctionDef(self, node):
        """Visit async function definitions."""
        self.functions.add(node.name)
        if not node.name.startswith('_'):
            self.exports.add(node.name)
        self.generic_visit(node)

def main():
    """Main function to run module analysis."""
    print("🏗️ KIMERA Module Architecture Analyzer")
    print("=" * 60)
    print("Phase 3.1: Module Boundary Clarification")
    print()
    
    analyzer = ModuleAnalyzer()
    
    try:
        # Run analysis
        results = analyzer.generate_report()
        
        # Generate visualization
        analyzer.visualize_dependencies()
        
        # Print summary
        print(f"📊 Analysis Results:")
        print(f"   Modules analyzed: {results['metrics']['total_modules']}")
        print(f"   Dependencies found: {results['metrics']['total_dependencies']}")
        print(f"   Issues detected: {len(results['issues'])}")
        print(f"   Clusters identified: {len(results['clusters'])}")
        print(f"   Recommendations: {len(results['recommendations'])}")
        print()
        print("📄 Reports generated:")
        print("   - architecture_analysis_report.json")
        print("   - dependency_graph.png")
        print()
        print("🎯 Next steps:")
        print("   1. Review architecture analysis report")
        print("   2. Address critical and high-priority issues")
        print("   3. Implement recommended architectural improvements")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 