#!/usr/bin/env python3
"""
KIMERA SWM Comprehensive Technical Debt Analyzer
===============================================

Advanced technical debt analysis following Martin Fowler's quadrant framework
and industry best practices. Analyzes debt patterns, quantifies impact,
and provides actionable remediation strategies.

Framework Quadrants:
1. Deliberate and Prudent: Strategic choices to ship now, fix later
2. Deliberate and Reckless: Ignoring design due to time pressures
3. Inadvertent and Prudent: Learning what should have been done
4. Inadvertent and Reckless: Poor practices due to lack of knowledge

Author: Kimera SWM Autonomous Architect
Version: 1.0.0
Classification: AEROSPACE-GRADE ANALYSIS
"""

import ast
import os
import re
import sys
import json
import hashlib
import subprocess
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TechnicalDebtAnalyzer:
    """Comprehensive technical debt analysis engine"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.analysis_results = {
            "deliberate_prudent": [],      # Strategic debt
            "deliberate_reckless": [],     # Time pressure debt
            "inadvertent_prudent": [],     # Learning debt
            "inadvertent_reckless": [],    # Knowledge debt
            "metrics": {},
            "hotspots": [],
            "recommendations": []
        }

    def analyze_codebase(self) -> Dict[str, Any]:
        """Perform comprehensive technical debt analysis"""
        logger.info("🔍 Starting comprehensive technical debt analysis...")

        # Core analysis methods
        self._analyze_file_organization()
        self._analyze_code_complexity()
        self._analyze_documentation_debt()
        self._analyze_test_coverage_debt()
        self._analyze_architectural_debt()
        self._analyze_maintenance_debt()
        self._analyze_performance_debt()
        self._analyze_security_debt()

        # Calculate metrics and generate recommendations
        self._calculate_debt_metrics()
        self._identify_hotspots()
        self._generate_recommendations()

        logger.info("✅ Technical debt analysis complete")
        return self.analysis_results

    def _analyze_file_organization(self):
        """Analyze file and directory organization patterns"""
        logger.info("📁 Analyzing file organization...")

        # Root directory pollution
        root_files = [f for f in self.project_root.iterdir()
                      if f.is_file() and f.suffix == '.py']

        if root_files:
            self.analysis_results["deliberate_reckless"].append({
                "type": "file_organization",
                "category": "Root Directory Pollution",
                "severity": "high",
                "description": f"Found {len(root_files)} Python files in root directory",
                "files": [str(f.name) for f in root_files],
                "impact": "Makes project structure unclear, violates organization standards",
                "effort": "low",
                "rationale": "Time pressure led to ignoring proper file placement"
            })

        # Duplicate directory structures
        src_variants = []
        for path in self.project_root.iterdir():
            if path.is_dir() and 'src' in path.name.lower():
                src_variants.append(path.name)

        if len(src_variants) > 1:
            self.analysis_results["inadvertent_reckless"].append({
                "type": "directory_structure",
                "category": "Duplicate Source Directories",
                "severity": "medium",
                "description": f"Multiple source directories found: {src_variants}",
                "impact": "Confusing structure, potential code duplication",
                "effort": "medium",
                "rationale": "Lack of understanding of proper structure"
            })

    def _analyze_code_complexity(self):
        """Analyze code complexity and identify god objects"""
        logger.info("🧮 Analyzing code complexity...")

        large_files = []
        complex_files = []

        for py_file in self.project_root.rglob("*.py"):
            if ".venv" in str(py_file) or "__pycache__" in str(py_file):
                continue

            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = len(content.splitlines())

                    # Large file analysis
                    if lines > 500:
                        large_files.append({
                            "file": str(py_file.relative_to(self.project_root)),
                            "lines": lines,
                            "type": "large_file"
                        })

                    # AST analysis for complexity
                    try:
                        tree = ast.parse(content)
                        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
                        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

                        # Complex class detection
                        for cls in classes:
                            methods = [node for node in cls.body if isinstance(node, ast.FunctionDef)]
                            if len(methods) > 20:
                                complex_files.append({
                                    "file": str(py_file.relative_to(self.project_root)),
                                    "class": cls.name,
                                    "methods": len(methods),
                                    "type": "god_class"
                                })

                    except SyntaxError:
                        pass  # Skip files with syntax errors

            except (UnicodeDecodeError, PermissionError):
                continue

        # Categorize complexity debt
        for file_info in large_files:
            if file_info["lines"] > 1000:
                # Likely deliberate for complex systems
                self.analysis_results["deliberate_prudent"].append({
                    "type": "code_complexity",
                    "category": "Large File",
                    "severity": "medium",
                    "description": f"File {file_info['file']} has {file_info['lines']} lines",
                    "impact": "Harder to maintain, understand, and test",
                    "effort": "high",
                    "rationale": "Strategic choice for complex domain logic"
                })
            else:
                # Likely poor organization
                self.analysis_results["inadvertent_reckless"].append({
                    "type": "code_complexity",
                    "category": "Large File",
                    "severity": "medium",
                    "description": f"File {file_info['file']} has {file_info['lines']} lines",
                    "impact": "Harder to maintain, understand, and test",
                    "effort": "medium",
                    "rationale": "Poor understanding of separation of concerns"
                })

        for file_info in complex_files:
            self.analysis_results["inadvertent_reckless"].append({
                "type": "code_complexity",
                "category": "God Class",
                "severity": "high",
                "description": f"Class {file_info['class']} has {file_info['methods']} methods",
                "impact": "Violates single responsibility principle, hard to test",
                "effort": "high",
                "rationale": "Lack of understanding of SOLID principles"
            })

    def _analyze_documentation_debt(self):
        """Analyze documentation quality and completeness"""
        logger.info("📚 Analyzing documentation debt...")

        # Find TODO/FIXME comments
        todo_pattern = re.compile(r'#\s*(TODO|FIXME|HACK|XXX|BUG)', re.IGNORECASE)
        todos_found = []

        for py_file in self.project_root.rglob("*.py"):
            if ".venv" in str(py_file) or "__pycache__" in str(py_file):
                continue

            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        match = todo_pattern.search(line)
                        if match:
                            todos_found.append({
                                "file": str(py_file.relative_to(self.project_root)),
                                "line": line_num,
                                "type": match.group(1),
                                "content": line.strip()
                            })
            except (UnicodeDecodeError, PermissionError):
                continue

        # Categorize TODO items
        strategic_todos = []
        reckless_todos = []

        for todo in todos_found:
            if "roadmap" in todo["content"].lower() or "week" in todo["content"].lower():
                strategic_todos.append(todo)
            else:
                reckless_todos.append(todo)

        if strategic_todos:
            self.analysis_results["deliberate_prudent"].append({
                "type": "documentation",
                "category": "Strategic TODOs",
                "severity": "low",
                "description": f"Found {len(strategic_todos)} roadmap-related TODO items",
                "items": strategic_todos[:5],  # Show first 5
                "impact": "Planned technical debt with clear timeline",
                "effort": "varies",
                "rationale": "Strategic decision to implement features incrementally"
            })

        if reckless_todos:
            self.analysis_results["deliberate_reckless"].append({
                "type": "documentation",
                "category": "Unplanned TODOs",
                "severity": "medium",
                "description": f"Found {len(reckless_todos)} unplanned TODO/FIXME items",
                "items": reckless_todos[:5],  # Show first 5
                "impact": "Incomplete features, potential bugs",
                "effort": "varies",
                "rationale": "Time pressure led to deferred implementations"
            })

    def _analyze_test_coverage_debt(self):
        """Analyze test coverage and quality"""
        logger.info("🧪 Analyzing test coverage debt...")

        # Count test files vs source files
        src_files = list(self.project_root.rglob("src/**/*.py"))
        test_files = list(self.project_root.rglob("test*.py")) + \
                    list(self.project_root.rglob("tests/**/*.py"))

        src_files = [f for f in src_files if "__pycache__" not in str(f)]
        test_files = [f for f in test_files if "__pycache__" not in str(f)]

        if src_files and len(test_files) < len(src_files) * 0.3:
            self.analysis_results["deliberate_reckless"].append({
                "type": "testing",
                "category": "Low Test Coverage",
                "severity": "high",
                "description": f"Only {len(test_files)} test files for {len(src_files)} source files",
                "impact": "Higher risk of bugs, harder refactoring",
                "effort": "high",
                "rationale": "Time pressure led to skipping test development"
            })

    def _analyze_architectural_debt(self):
        """Analyze architectural patterns and design debt"""
        logger.info("🏗️ Analyzing architectural debt...")

        # Find circular dependencies (basic check)
        import_patterns = {}

        for py_file in self.project_root.rglob("*.py"):
            if ".venv" in str(py_file) or "__pycache__" in str(py_file):
                continue

            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    imports = re.findall(r'from\s+(\S+)\s+import|import\s+(\S+)', content)
                    import_patterns[str(py_file.relative_to(self.project_root))] = imports
            except (UnicodeDecodeError, PermissionError):
                continue

        # Check for wildcard imports (namespace pollution)
        wildcard_imports = []
        for py_file in self.project_root.rglob("*.py"):
            if ".venv" in str(py_file) or "__pycache__" in str(py_file):
                continue

            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if re.search(r'import\s+\*|from\s+\S+\s+import\s+\*', content):
                        wildcard_imports.append(str(py_file.relative_to(self.project_root)))
            except (UnicodeDecodeError, PermissionError):
                continue

        if wildcard_imports:
            self.analysis_results["inadvertent_reckless"].append({
                "type": "architecture",
                "category": "Wildcard Imports",
                "severity": "medium",
                "description": f"Found wildcard imports in {len(wildcard_imports)} files",
                "files": wildcard_imports[:5],
                "impact": "Namespace pollution, unclear dependencies",
                "effort": "low",
                "rationale": "Lack of understanding of import best practices"
            })

    def _analyze_maintenance_debt(self):
        """Analyze maintenance and operational debt"""
        logger.info("🔧 Analyzing maintenance debt...")

        # Check for placeholder implementations
        placeholder_patterns = [
            (r'\bpass\b', "Empty implementations"),
            (r'NotImplementedError', "Unimplemented features"),
            (r'raise NotImplementedError', "Explicitly unimplemented"),
        ]

        placeholders = defaultdict(list)

        for py_file in self.project_root.rglob("*.py"):
            if ".venv" in str(py_file) or "__pycache__" in str(py_file):
                continue

            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    for pattern, description in placeholder_patterns:
                        matches = re.finditer(pattern, content)
                        for match in matches:
                            placeholders[description].append({
                                "file": str(py_file.relative_to(self.project_root)),
                                "pattern": pattern
                            })
            except (UnicodeDecodeError, PermissionError):
                continue

        for description, items in placeholders.items():
            if items:
                self.analysis_results["deliberate_prudent"].append({
                    "type": "maintenance",
                    "category": description,
                    "severity": "medium",
                    "description": f"Found {len(items)} instances of {description.lower()}",
                    "items": items[:5],
                    "impact": "Incomplete functionality, potential runtime errors",
                    "effort": "varies",
                    "rationale": "Strategic decision to implement incrementally"
                })

    def _analyze_performance_debt(self):
        """Analyze performance-related technical debt"""
        logger.info("⚡ Analyzing performance debt...")

        # This is a simplified analysis - real performance debt needs profiling
        potential_performance_issues = []

        # Check for nested loops (simplified)
        for py_file in self.project_root.rglob("*.py"):
            if ".venv" in str(py_file) or "__pycache__" in str(py_file):
                continue

            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Simple check for nested loops
                    if content.count('for ') > 5 and 'for ' in content:
                        potential_performance_issues.append({
                            "file": str(py_file.relative_to(self.project_root)),
                            "issue": "Multiple loops detected"
                        })
            except (UnicodeDecodeError, PermissionError):
                continue

        if potential_performance_issues:
            self.analysis_results["inadvertent_prudent"].append({
                "type": "performance",
                "category": "Potential Performance Issues",
                "severity": "low",
                "description": f"Found {len(potential_performance_issues)} files with potential performance concerns",
                "items": potential_performance_issues[:3],
                "impact": "May affect system performance under load",
                "effort": "medium",
                "rationale": "Learning opportunity to optimize algorithms"
            })

    def _analyze_security_debt(self):
        """Analyze security-related technical debt"""
        logger.info("🔒 Analyzing security debt...")

        # Look for potential security issues
        security_patterns = [
            (r'eval\(', "Use of eval()"),
            (r'exec\(', "Use of exec()"),
            (r'input\(', "Direct input() usage"),
            (r'shell=True', "Shell command execution"),
        ]

        security_issues = defaultdict(list)

        for py_file in self.project_root.rglob("*.py"):
            if ".venv" in str(py_file) or "__pycache__" in str(py_file):
                continue

            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    for pattern, description in security_patterns:
                        if re.search(pattern, content):
                            security_issues[description].append(
                                str(py_file.relative_to(self.project_root))
                            )
            except (UnicodeDecodeError, PermissionError):
                continue

        for issue_type, files in security_issues.items():
            if files:
                self.analysis_results["inadvertent_reckless"].append({
                    "type": "security",
                    "category": issue_type,
                    "severity": "high",
                    "description": f"Found {issue_type.lower()} in {len(files)} files",
                    "files": files[:3],
                    "impact": "Potential security vulnerabilities",
                    "effort": "medium",
                    "rationale": "Lack of security awareness"
                })

    def _calculate_debt_metrics(self):
        """Calculate quantitative debt metrics"""
        logger.info("📊 Calculating debt metrics...")

        total_issues = sum(len(self.analysis_results[quadrant])
                          for quadrant in ["deliberate_prudent", "deliberate_reckless",
                                         "inadvertent_prudent", "inadvertent_reckless"])

        severity_counts = defaultdict(int)
        effort_estimates = defaultdict(int)

        for quadrant in ["deliberate_prudent", "deliberate_reckless",
                        "inadvertent_prudent", "inadvertent_reckless"]:
            for item in self.analysis_results[quadrant]:
                severity_counts[item["severity"]] += 1
                effort_estimates[item["effort"]] += 1

        self.analysis_results["metrics"] = {
            "total_debt_items": total_issues,
            "quadrant_distribution": {
                "deliberate_prudent": len(self.analysis_results["deliberate_prudent"]),
                "deliberate_reckless": len(self.analysis_results["deliberate_reckless"]),
                "inadvertent_prudent": len(self.analysis_results["inadvertent_prudent"]),
                "inadvertent_reckless": len(self.analysis_results["inadvertent_reckless"])
            },
            "severity_distribution": dict(severity_counts),
            "effort_distribution": dict(effort_estimates),
            "debt_score": self._calculate_debt_score()
        }

    def _calculate_debt_score(self) -> float:
        """Calculate overall technical debt score (0-100)"""
        weights = {
            "high": 3,
            "medium": 2,
            "low": 1
        }

        total_weighted_issues = 0
        for quadrant in ["deliberate_prudent", "deliberate_reckless",
                        "inadvertent_prudent", "inadvertent_reckless"]:
            for item in self.analysis_results[quadrant]:
                total_weighted_issues += weights.get(item["severity"], 1)

        # Normalize to 0-100 scale (arbitrarily capped at 50 issues = 100 score)
        return min(100, (total_weighted_issues / 50) * 100)

    def _identify_hotspots(self):
        """Identify critical areas requiring immediate attention"""
        logger.info("🔥 Identifying technical debt hotspots...")

        hotspots = []

        # High severity items
        for quadrant in ["deliberate_reckless", "inadvertent_reckless"]:
            for item in self.analysis_results[quadrant]:
                if item["severity"] == "high":
                    hotspots.append({
                        "area": item["category"],
                        "quadrant": quadrant,
                        "impact": item["impact"],
                        "priority": "urgent" if quadrant == "inadvertent_reckless" else "high"
                    })

        self.analysis_results["hotspots"] = hotspots

    def _generate_recommendations(self):
        """Generate actionable recommendations for debt reduction"""
        logger.info("💡 Generating recommendations...")

        recommendations = []

        # Prioritized recommendations based on quadrant and severity
        if self.analysis_results["deliberate_reckless"]:
            recommendations.append({
                "priority": "immediate",
                "title": "Address Time Pressure Debt",
                "description": "Focus on deliberate reckless debt items that pose immediate risks",
                "actions": [
                    "Review all TODO/FIXME items without clear timeline",
                    "Implement missing test coverage for critical paths",
                    "Clean up root directory file organization"
                ],
                "estimated_effort": "2-4 weeks"
            })

        if self.analysis_results["inadvertent_reckless"]:
            recommendations.append({
                "priority": "high",
                "title": "Knowledge Gap Training",
                "description": "Address inadvertent reckless debt through education and standards",
                "actions": [
                    "Conduct code review training on SOLID principles",
                    "Implement automated linting and formatting",
                    "Create architectural decision records (ADRs)"
                ],
                "estimated_effort": "1-2 weeks"
            })

        if self.analysis_results["deliberate_prudent"]:
            recommendations.append({
                "priority": "medium",
                "title": "Strategic Debt Management",
                "description": "Manage deliberate prudent debt with clear timelines",
                "actions": [
                    "Create detailed roadmap for planned implementations",
                    "Set up debt tracking and review cycles",
                    "Implement feature flags for incremental releases"
                ],
                "estimated_effort": "1 week"
            })

        recommendations.append({
            "priority": "ongoing",
            "title": "Continuous Debt Prevention",
            "description": "Establish practices to prevent new technical debt",
            "actions": [
                "Implement pre-commit hooks for code quality",
                "Set up automated technical debt monitoring",
                "Establish definition of done including debt assessment"
            ],
            "estimated_effort": "2-3 weeks initial setup"
        })

        self.analysis_results["recommendations"] = recommendations

def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(description="Comprehensive Technical Debt Analysis")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--output", help="Output file path")

    args = parser.parse_args()

    # Run analysis
    analyzer = TechnicalDebtAnalyzer(args.project_root)
    results = analyzer.analyze_codebase()

    # Generate output
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')

    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = Path("docs/reports/debt")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{timestamp}_technical_debt_analysis.json"

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"✅ Technical debt analysis complete: {output_path}")
    logger.info(f"📊 Debt Score: {results['metrics']['debt_score']:.1f}/100")
    logger.info(f"🔥 Critical Hotspots: {len(results['hotspots'])}")

    return results

if __name__ == "__main__":
    main()
