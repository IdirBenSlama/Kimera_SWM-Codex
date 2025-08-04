#!/usr/bin/env python3
"""
KIMERA SWM Technical Debt Monitoring Script
===========================================

Automated technical debt monitoring following Martin Fowler's framework.
Generates quantitative metrics and tracks debt evolution over time.

Usage:
    python scripts/analysis/debt_monitor.py
    python scripts/analysis/debt_monitor.py --detailed
    python scripts/analysis/debt_monitor.py --export-json
"""

import os
import re
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import logging
logger = logging.getLogger(__name__)

class KimeraDebtMonitor:
    """
    Kimera SWM Technical Debt Monitor

    Implements automated debt detection and classification following
    aerospace-grade monitoring standards.
    """

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.metrics = {
            "timestamp": datetime.now().isoformat(),
            "structural_debt": {},
            "code_debt": {},
            "dependency_debt": {},
            "test_debt": {},
            "overall_health": {}
        }

    def analyze_structural_debt(self) -> Dict[str, Any]:
        """Analyze directory structure and organization debt."""

        # Count root directories (should be <10)
        root_items = [item for item in self.project_root.iterdir()
                     if item.is_dir() and not item.name.startswith('.')]

        # Find misplaced src directories
        misplaced_src = [item for item in root_items
                        if item.name.startswith('src') and item.name != 'src']

        # Count files in root (should be minimal)
        root_files = [item for item in self.project_root.iterdir()
                     if item.is_file() and not item.name.startswith('.')]

        # Find configuration files scattered around
        config_dirs = [item for item in root_items
                      if 'config' in item.name.lower()]

        structural_debt = {
            "root_directory_count": len(root_items),
            "root_directory_target": 10,
            "misplaced_src_directories": [str(d.name) for d in misplaced_src],
            "root_file_count": len(root_files),
            "root_file_target": 5,
            "config_directory_count": len(config_dirs),
            "config_directories": [str(d.name) for d in config_dirs],
            "debt_severity": self._calculate_structural_severity(
                len(root_items), len(root_files), len(misplaced_src)
            )
        }

        return structural_debt

    def analyze_code_debt(self) -> Dict[str, Any]:
        """Analyze code-level technical debt markers."""

        debt_patterns = {
            'TODO': r'#\s*TODO',
            'FIXME': r'#\s*FIXME',
            'HACK': r'#\s*HACK',
            'XXX': r'#\s*XXX',
            'BUG': r'#\s*BUG',
            'NotImplementedError': r'raise\s+NotImplementedError',
            'pass_statements': r'^\s*pass\s*$'
        }

        debt_counts = defaultdict(list)
        total_python_files = 0

        for py_file in self.project_root.rglob("*.py"):
            # Skip virtual environment and cache files
            if any(part in str(py_file) for part in ['.venv', '__pycache__', '.pytest_cache']):
                continue

            total_python_files += 1

            try:
                # Try different encodings for files
                for encoding in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        with open(py_file, 'r', encoding=encoding) as f:
                            content = f.read()
                            lines = content.split('\n')

                            for line_num, line in enumerate(lines, 1):
                                for debt_type, pattern in debt_patterns.items():
                                    if re.search(pattern, line, re.IGNORECASE):
                                        debt_counts[debt_type].append({
                                            'file': str(py_file.relative_to(self.project_root)),
                                            'line': line_num,
                                            'content': line.strip()
                                        })
                        break  # Success, stop trying other encodings
                    except UnicodeDecodeError:
                        continue  # Try next encoding
            except Exception as e:
                logger.info(f"Warning: Could not analyze {py_file}: {e}")

        code_debt = {
            "total_python_files": total_python_files,
            "debt_markers": {k: len(v) for k, v in debt_counts.items()},
            "debt_details": dict(debt_counts),
            "debt_density": sum(len(v) for v in debt_counts.values()) / max(total_python_files, 1),
            "debt_severity": self._calculate_code_debt_severity(debt_counts)
        }

        return code_debt

    def analyze_dependency_debt(self) -> Dict[str, Any]:
        """Analyze dependency management complexity."""

        requirements_files = list(self.project_root.rglob("requirements*.txt"))
        requirements_files.extend(self.project_root.rglob("*.toml"))

        # Filter out virtual environment files
        requirements_files = [f for f in requirements_files
                            if '.venv' not in str(f) and '__pycache__' not in str(f)]

        dependency_debt = {
            "requirements_file_count": len(requirements_files),
            "requirements_files": [str(f.relative_to(self.project_root))
                                 for f in requirements_files],
            "target_file_count": 1,  # Should ideally have just pyproject.toml
            "dependency_complexity": "HIGH" if len(requirements_files) > 5 else
                                   "MEDIUM" if len(requirements_files) > 2 else "LOW"
        }

        return dependency_debt

    def analyze_test_debt(self) -> Dict[str, Any]:
        """Analyze test organization and structure debt."""

        # Find standalone test scripts (those with if __name__ == "__main__")
        standalone_tests = []
        total_test_files = 0

        for py_file in self.project_root.rglob("test*.py"):
            if '.venv' in str(py_file) or '__pycache__' in str(py_file):
                continue

            total_test_files += 1

            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'if __name__ == "__main__"' in content:
                        standalone_tests.append(str(py_file.relative_to(self.project_root)))
            except Exception as e:
                logger.info(f"Warning: Could not analyze {py_file}: {e}")

        test_debt = {
            "total_test_files": total_test_files,
            "standalone_test_count": len(standalone_tests),
            "standalone_tests": standalone_tests[:10],  # First 10 for brevity
            "test_organization_debt": "HIGH" if len(standalone_tests) > 50 else
                                     "MEDIUM" if len(standalone_tests) > 20 else "LOW",
            "centralized_testing": len(standalone_tests) == 0
        }

        return test_debt

    def _calculate_structural_severity(self, root_dirs: int, root_files: int,
                                     misplaced_src: int) -> str:
        """Calculate structural debt severity."""
        score = 0
        score += max(0, root_dirs - 10) * 2  # Penalty for too many root dirs
        score += max(0, root_files - 5) * 1   # Penalty for too many root files
        score += misplaced_src * 5            # High penalty for misplaced src

        if score >= 20:
            return "CRITICAL"
        elif score >= 10:
            return "HIGH"
        elif score >= 5:
            return "MEDIUM"
        else:
            return "LOW"

    def _calculate_code_debt_severity(self, debt_counts: Dict) -> str:
        """Calculate code debt severity based on markers."""
        critical_debt = len(debt_counts.get('FIXME', [])) + len(debt_counts.get('HACK', []))
        moderate_debt = len(debt_counts.get('TODO', [])) + len(debt_counts.get('BUG', []))

        if critical_debt > 10:
            return "HIGH"
        elif critical_debt > 5 or moderate_debt > 50:
            return "MEDIUM"
        else:
            return "LOW"

    def calculate_overall_health(self) -> Dict[str, Any]:
        """Calculate overall technical debt health score."""

        structural = self.metrics["structural_debt"]
        code = self.metrics["code_debt"]
        dependency = self.metrics["dependency_debt"]
        test = self.metrics["test_debt"]

        # Calculate weighted health score (0-100)
        score = 100

        # Structural penalties
        if structural["debt_severity"] == "CRITICAL":
            score -= 30
        elif structural["debt_severity"] == "HIGH":
            score -= 20
        elif structural["debt_severity"] == "MEDIUM":
            score -= 10

        # Code debt penalties
        if code["debt_severity"] == "HIGH":
            score -= 20
        elif code["debt_severity"] == "MEDIUM":
            score -= 10

        # Dependency penalties
        if dependency["dependency_complexity"] == "HIGH":
            score -= 15
        elif dependency["dependency_complexity"] == "MEDIUM":
            score -= 5

        # Test organization penalties
        if test["test_organization_debt"] == "HIGH":
            score -= 15
        elif test["test_organization_debt"] == "MEDIUM":
            score -= 10

        health_grade = "A" if score >= 90 else "B" if score >= 80 else \
                      "C" if score >= 70 else "D" if score >= 60 else "F"

        return {
            "health_score": max(0, score),
            "health_grade": health_grade,
            "debt_service_estimate": f"{max(10, 110 - score)}%",
            "recommendations": self._generate_recommendations(structural, code, dependency, test)
        }

        def _generate_recommendations(self, structural: Dict, code: Dict,
                                dependency: Dict, test: Dict) -> List[str]:
        """Generate actionable recommendations based on debt analysis."""
        recommendations = []

        if structural["debt_severity"] in ["CRITICAL", "HIGH"]:
            recommendations.append("URGENT: Consolidate directory structure - move misplaced src directories")

        if structural["root_directory_count"] > 15:
            recommendations.append("HIGH: Reduce root directory count to <10 for better organization")

        if code["debt_markers"].get("FIXME", 0) > 5:
            recommendations.append("MEDIUM: Address FIXME comments before next release")

        if test["standalone_test_count"] > 50:
            recommendations.append("HIGH: Migrate to centralized test framework (pytest)")

        if dependency["requirements_file_count"] > 5:
            recommendations.append("MEDIUM: Consolidate dependency management to pyproject.toml")

        if not recommendations:
            recommendations.append("GOOD: Technical debt is well-managed, continue current practices")

        return recommendations

    def run_analysis(self) -> Dict[str, Any]:
        """Run complete technical debt analysis."""
        logger.info("🔍 Running Kimera SWM Technical Debt Analysis...")

        logger.info("  📁 Analyzing structural debt...")
        self.metrics["structural_debt"] = self.analyze_structural_debt()

        logger.info("  💻 Analyzing code debt...")
        self.metrics["code_debt"] = self.analyze_code_debt()

        logger.info("  📦 Analyzing dependency debt...")
        self.metrics["dependency_debt"] = self.analyze_dependency_debt()

        logger.info("  🧪 Analyzing test debt...")
        self.metrics["test_debt"] = self.analyze_test_debt()

        logger.info("  📊 Calculating overall health...")
        self.metrics["overall_health"] = self.calculate_overall_health()

        return self.metrics

    def print_summary(self):
        """Print a formatted summary of the debt analysis."""
        health = self.metrics["overall_health"]
        structural = self.metrics["structural_debt"]
        code = self.metrics["code_debt"]

        logger.info("\n" + "="*60)
        logger.info("🎯 KIMERA SWM TECHNICAL DEBT SUMMARY")
        logger.info("="*60)

        logger.info(f"📊 Overall Health Score: {health['health_score']}/100 (Grade: {health['health_grade']})")
        logger.info(f"⏱️  Estimated Debt Service Time: {health['debt_service_estimate']}")

        logger.info(f"\n📁 Structural Debt: {structural['debt_severity']}")
        logger.info(f"   Root Directories: {structural['root_directory_count']} (target: ≤{structural['root_directory_target']})")
        logger.info(f"   Misplaced Src Dirs: {len(structural['misplaced_src_directories'])}")

        logger.info(f"\n💻 Code Debt: {code['debt_severity']}")
        logger.info(f"   TODO Comments: {code['debt_markers']['TODO']}")
        logger.info(f"   FIXME Comments: {code['debt_markers']['FIXME']}")
        logger.info(f"   Python Files Analyzed: {code['total_python_files']}")

        logger.info(f"\n📦 Dependency Complexity: {self.metrics['dependency_debt']['dependency_complexity']}")
        logger.info(f"   Requirements Files: {self.metrics['dependency_debt']['requirements_file_count']}")

        logger.info(f"\n🧪 Test Organization: {self.metrics['test_debt']['test_organization_debt']}")
        logger.info(f"   Standalone Tests: {self.metrics['test_debt']['standalone_test_count']}")

        logger.info("\n🎯 TOP RECOMMENDATIONS:")
        for i, rec in enumerate(health["recommendations"][:3], 1):
            logger.info(f"   {i}. {rec}")

        logger.info("\n" + "="*60)

def main():
    parser = argparse.ArgumentParser(description="Kimera SWM Technical Debt Monitor")
    parser.add_argument("--detailed", action="store_true", help="Show detailed debt information")
    parser.add_argument("--export-json", action="store_true", help="Export results to JSON")
    parser.add_argument("--project-root", default=".", help="Project root directory")

    args = parser.parse_args()

    # Ensure we're running from project root or handle path properly
    monitor = KimeraDebtMonitor(args.project_root)
    metrics = monitor.run_analysis()

    # Print summary
    monitor.print_summary()

    # Detailed output if requested
    if args.detailed:
        logger.info("\n📋 DETAILED DEBT ANALYSIS:")
        logger.info("-" * 40)

        if metrics["structural_debt"]["misplaced_src_directories"]:
            logger.info("\n📁 Misplaced Src Directories:")
            for dir_name in metrics["structural_debt"]["misplaced_src_directories"]:
                logger.info(f"   - {dir_name}")

        if metrics["code_debt"]["debt_markers"]["TODO"] > 0:
            logger.info(f"\n💻 TODO Comments ({metrics['code_debt']['debt_markers']['TODO']}):")
            for todo in metrics["code_debt"]["debt_details"]["TODO"][:5]:
                logger.info(f"   - {todo['file']}:{todo['line']} - {todo['content'][:60]}...")

        if metrics["test_debt"]["standalone_test_count"] > 0:
            logger.info(f"\n🧪 Standalone Tests (showing first 5 of {metrics['test_debt']['standalone_test_count']}):")
            for test in metrics["test_debt"]["standalone_tests"][:5]:
                logger.info(f"   - {test}")

    # Export to JSON if requested
    if args.export_json:
        date_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        filename = f"docs/reports/debt/{date_str}_debt_metrics.json"

        os.makedirs("docs/reports/debt", exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)

        logger.info(f"\n💾 Metrics exported to: {filename}")

if __name__ == "__main__":
    main()
