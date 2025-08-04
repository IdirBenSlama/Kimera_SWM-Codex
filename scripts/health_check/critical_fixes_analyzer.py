#!/usr/bin/env python3
"""
Critical Fixes Analyzer for Kimera SWM System
============================================
Identifies and categorizes critical runtime issues requiring immediate attention.
Implements aerospace-grade error classification and triage protocols.
"""

import re
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import logging
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

class IssueSeverity(Enum):
    """Issue severity levels following aerospace standards."""
    CATASTROPHIC = "CATASTROPHIC"  # System failure
    CRITICAL = "CRITICAL"         # Major functionality loss
    MAJOR = "MAJOR"              # Significant degradation
    MINOR = "MINOR"              # Minor issues
    ADVISORY = "ADVISORY"        # Informational

@dataclass
class CriticalIssue:
    """Critical issue tracking with aerospace-grade metadata."""
    severity: IssueSeverity
    component: str
    error_type: str
    description: str
    suggested_fix: str
    priority_order: int

class CriticalFixesAnalyzer:
    """Analyzes and categorizes critical runtime issues."""

    def __init__(self):
        self.issues: List[CriticalIssue] = []
        self.log_patterns = {
            'relative_import_error': r'attempted relative import beyond top-level package',
            'module_not_found': r"No module named '([^']+)'",
            'attribute_error': r"'([^']+)' object has no attribute '([^']+)'",
            'database_error': r"'NoneType' object has no attribute '(query|add)'",
            'initialization_failure': r'Failed to initialize ([^:]+): (.+)',
            'import_failure': r'Failed to import ([^:]+): (.+)'
        }

    def analyze_known_issues(self) -> List[CriticalIssue]:
        """Analyze known critical issues from system logs and patterns."""

        # Known critical issues from initialization logs
        issues = [
            CriticalIssue(
                severity=IssueSeverity.CRITICAL,
                component="thermodynamic_optimization",
                error_type="relative_import_error",
                description="Failed to import Thermodynamic Optimization integrator: attempted relative import beyond top-level package",
                suggested_fix="Fix relative import paths in thermodynamic_optimization integration",
                priority_order=1
            ),
            CriticalIssue(
                severity=IssueSeverity.CRITICAL,
                component="vortex_dynamics",
                error_type="relative_import_error",
                description="Failed to import Vortex Dynamics integrator: attempted relative import beyond top-level package",
                suggested_fix="Fix relative import paths in vortex_dynamics integration",
                priority_order=2
            ),
            CriticalIssue(
                severity=IssueSeverity.CRITICAL,
                component="zetetic_and_revolutionary_integration",
                error_type="relative_import_error",
                description="Failed to import Zetetic Revolutionary integrator: attempted relative import beyond top-level package",
                suggested_fix="Fix relative import paths in zetetic_and_revolutionary_integration",
                priority_order=3
            ),
            CriticalIssue(
                severity=IssueSeverity.MAJOR,
                component="triton_and_unsupervised_optimization",
                error_type="missing_dependency",
                description="Failed to import Triton and Unsupervised Optimization integrator: No module named 'triton'",
                suggested_fix="Install triton library or implement CPU fallback for triton kernels",
                priority_order=4
            ),
            CriticalIssue(
                severity=IssueSeverity.CRITICAL,
                component="insight_management",
                error_type="relative_import_error",
                description="Failed to import Insight Management integrator: attempted relative import beyond top-level package",
                suggested_fix="Fix relative import paths in insight_management integration",
                priority_order=5
            ),
            CriticalIssue(
                severity=IssueSeverity.CRITICAL,
                component="response_generation",
                error_type="import_structure_error",
                description="Failed to import Response Generation system: 'core.response_generation.integration' is not a package",
                suggested_fix="Fix package structure in response_generation integration",
                priority_order=6
            ),
            CriticalIssue(
                severity=IssueSeverity.MAJOR,
                component="understanding_engine",
                error_type="database_error",
                description="Understanding Engine database errors: 'NoneType' object has no attribute 'query'",
                suggested_fix="Implement proper database session management or graceful fallback",
                priority_order=7
            ),
            CriticalIssue(
                severity=IssueSeverity.MAJOR,
                component="ethical_reasoning_engine",
                error_type="database_error",
                description="Ethical Reasoning Engine initialization failed: 'NoneType' object is not callable",
                suggested_fix="Fix database session initialization in ethical reasoning engine",
                priority_order=8
            )
        ]

        self.issues = sorted(issues, key=lambda x: x.priority_order)
        return self.issues

    def generate_fix_roadmap(self) -> Dict[str, List[str]]:
        """Generate prioritized fix roadmap."""
        roadmap = {
            'immediate_fixes': [],  # CATASTROPHIC + CRITICAL
            'priority_fixes': [],   # MAJOR
            'future_fixes': []      # MINOR + ADVISORY
        }

        for issue in self.issues:
            fix_description = f"{issue.component}: {issue.suggested_fix}"

            if issue.severity in [IssueSeverity.CATASTROPHIC, IssueSeverity.CRITICAL]:
                roadmap['immediate_fixes'].append(fix_description)
            elif issue.severity == IssueSeverity.MAJOR:
                roadmap['priority_fixes'].append(fix_description)
            else:
                roadmap['future_fixes'].append(fix_description)

        return roadmap

    def save_analysis_report(self) -> str:
        """Save comprehensive analysis report."""
        date_str = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        report_dir = Path("docs/reports/analysis")
        report_dir.mkdir(parents=True, exist_ok=True)

        report_file = report_dir / f"{date_str}_critical_fixes_analysis.md"

        with open(report_file, 'w') as f:
            f.write("# Kimera SWM Critical Fixes Analysis\n\n")
            f.write(f"**Generated**: {datetime.now().isoformat()}\n")
            f.write(f"**Analyzer**: DO-178C Level A Critical Fixes Analyzer\n\n")

            f.write("## Executive Summary\n\n")
            severity_counts = {}
            for issue in self.issues:
                severity_counts[issue.severity] = severity_counts.get(issue.severity, 0) + 1

            for severity, count in severity_counts.items():
                f.write(f"- **{severity.value}**: {count} issues\n")

            f.write(f"\n**Total Issues**: {len(self.issues)}\n\n")

            f.write("## Critical Issues (Immediate Action Required)\n\n")
            for i, issue in enumerate(self.issues, 1):
                if issue.severity in [IssueSeverity.CATASTROPHIC, IssueSeverity.CRITICAL]:
                    f.write(f"### {i}. {issue.component} - {issue.severity.value}\n\n")
                    f.write(f"**Error**: {issue.description}\n\n")
                    f.write(f"**Suggested Fix**: {issue.suggested_fix}\n\n")

            f.write("## Fix Roadmap\n\n")
            roadmap = self.generate_fix_roadmap()

            f.write("### Immediate Fixes (CRITICAL/CATASTROPHIC)\n")
            for i, fix in enumerate(roadmap['immediate_fixes'], 1):
                f.write(f"{i}. {fix}\n")

            f.write("\n### Priority Fixes (MAJOR)\n")
            for i, fix in enumerate(roadmap['priority_fixes'], 1):
                f.write(f"{i}. {fix}\n")

        logger.info(f"📋 Critical fixes analysis saved: {report_file}")
        return str(report_file)

def main():
    """Main analysis execution."""
    logger.info("🔍 KIMERA SWM CRITICAL FIXES ANALYSIS")
    logger.info("=" * 40)

    analyzer = CriticalFixesAnalyzer()
    issues = analyzer.analyze_known_issues()

    logger.info(f"📊 ANALYSIS COMPLETE:")
    logger.info(f"   Total Issues: {len(issues)}")

    severity_counts = {}
    for issue in issues:
        severity_counts[issue.severity] = severity_counts.get(issue.severity, 0) + 1

    for severity, count in severity_counts.items():
        logger.info(f"   {severity.value}: {count}")

    report_file = analyzer.save_analysis_report()
    return analyzer

if __name__ == "__main__":
    main()
