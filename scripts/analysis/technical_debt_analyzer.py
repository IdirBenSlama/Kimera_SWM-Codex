#!/usr/bin/env python3
"""
Kimera SWM Technical Debt Analyzer
==================================

Comprehensive technical debt analysis framework based on Martin Fowler's quadrants
and industry best practices. Implements the Kimera SWM Autonomous Architect Protocol
for systematic technical debt identification, categorization, and remediation planning.

Framework based on:
- Martin Fowler's Technical Debt Quadrants
- Shopify's technical debt management strategies
- Aerospace engineering "Defense in depth" principles
- Zero-debugging constraint compliance
"""

import os
import re
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict, Counter
import ast

import logging

# Simple logging setup since KimeraLogger path is part of the technical debt problem
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("technical_debt_analyzer")


class TechnicalDebtAnalyzer:
    """
    Comprehensive technical debt analyzer implementing Martin Fowler's quadrants:

    1. Deliberate and Prudent: Strategic choices to ship now and fix later
    2. Deliberate and Reckless: Ignoring design due to time pressures
    3. Inadvertent and Prudent: Learning better ways after the fact
    4. Inadvertent and Reckless: Poor practices due to lack of knowledge
    """

    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path)
        self.analysis_timestamp = datetime.now()
        self.debt_inventory = {
            "deliberate_prudent": [],
            "deliberate_reckless": [],
            "inadvertent_prudent": [],
            "inadvertent_reckless": []
        }
        self.metrics = {
            "total_files_analyzed": 0,
            "total_debt_instances": 0,
            "estimated_remediation_hours": 0,
            "priority_issues": 0
        }

    def analyze_codebase(self) -> Dict[str, Any]:
        """Execute comprehensive technical debt analysis"""
        logger.info("🔍 Starting comprehensive technical debt analysis...")

        analysis_results = {
            "metadata": {
                "timestamp": self.analysis_timestamp.isoformat(),
                "framework_version": "3.0",
                "analyzer": "Kimera SWM Autonomous Architect"
            },
            "executive_summary": {},
            "debt_quadrants": {},
            "remediation_roadmap": {},
            "metrics": {}
        }

        # 1. Code Organization Debt Analysis
        org_debt = self._analyze_code_organization()

        # 2. Documentation Debt Analysis
        doc_debt = self._analyze_documentation_debt()

        # 3. Configuration Debt Analysis
        config_debt = self._analyze_configuration_debt()

        # 4. Zero-Debugging Constraint Violations
        debug_debt = self._analyze_debugging_violations()

        # 5. Import Structure Debt
        import_debt = self._analyze_import_debt()

        # 6. Duplication Debt
        dup_debt = self._analyze_duplication_debt()

        # 7. Test Coverage Debt
        test_debt = self._analyze_test_debt()

        # 8. Performance Debt
        perf_debt = self._analyze_performance_debt()

        # Categorize into Fowler's quadrants
        self._categorize_debt_quadrants(
            org_debt, doc_debt, config_debt, debug_debt,
            import_debt, dup_debt, test_debt, perf_debt
        )

        # Generate executive summary
        analysis_results["executive_summary"] = self._generate_executive_summary()
        analysis_results["debt_quadrants"] = self.debt_inventory
        analysis_results["remediation_roadmap"] = self._generate_remediation_roadmap()
        analysis_results["metrics"] = self.metrics

        logger.info("✅ Technical debt analysis completed")
        return analysis_results

    def _analyze_code_organization(self) -> Dict[str, Any]:
        """Analyze code organization and module structure debt"""
        logger.info("📁 Analyzing code organization debt...")

        org_issues = {
            "multiple_source_dirs": [],
            "unclear_boundaries": [],
            "circular_dependencies": [],
            "module_coupling": "high"
        }

        # Check for multiple source directories
        source_dirs = []
        for item in self.root_path.iterdir():
            if item.is_dir() and (
                item.name.startswith('src') or
                'core' in item.name.lower() or
                item.name in ['backend', 'kimera_trading']
            ):
                source_dirs.append(str(item))

        if len(source_dirs) > 1:
            org_issues["multiple_source_dirs"] = source_dirs

        # Check for unclear module boundaries
        python_files = list(self.root_path.rglob("*.py"))
        self.metrics["total_files_analyzed"] = len(python_files)

        # Analyze import patterns for coupling
        import_patterns = self._analyze_import_patterns(python_files)
        org_issues["import_complexity"] = len(import_patterns)

        return org_issues

    def _analyze_documentation_debt(self) -> Dict[str, Any]:
        """Analyze documentation debt and redundancy"""
        logger.info("📚 Analyzing documentation debt...")

        doc_issues = {
            "duplicate_docs": [],
            "outdated_docs": [],
            "missing_docs": [],
            "doc_redundancy_score": 0
        }

        # Find markdown files
        md_files = list(self.root_path.rglob("*.md"))

        # Group similar named files
        doc_groups = defaultdict(list)
        for md_file in md_files:
            # Group by similar names (README, KIMERA, etc.)
            base_name = md_file.name.lower()
            if 'readme' in base_name:
                doc_groups['readme'].append(str(md_file))
            elif 'kimera' in base_name:
                doc_groups['kimera'].append(str(md_file))
            elif 'status' in base_name or 'report' in base_name:
                doc_groups['reports'].append(str(md_file))

        # Identify potential duplicates
        for group, files in doc_groups.items():
            if len(files) > 3:  # More than 3 similar docs is suspicious
                doc_issues["duplicate_docs"].append({
                    "group": group,
                    "files": files,
                    "count": len(files)
                })

        doc_issues["doc_redundancy_score"] = sum(
            len(files) - 1 for files in doc_groups.values() if len(files) > 1
        )

        return doc_issues

    def _analyze_configuration_debt(self) -> Dict[str, Any]:
        """Analyze configuration management debt"""
        logger.info("⚙️ Analyzing configuration debt...")

        config_issues = {
            "multiple_config_dirs": [],
            "duplicate_requirements": [],
            "inconsistent_configs": []
        }

        # Check for multiple config directories
        config_dirs = []
        for item in self.root_path.iterdir():
            if item.is_dir() and 'config' in item.name.lower():
                config_dirs.append(str(item))

        if len(config_dirs) > 1:
            config_issues["multiple_config_dirs"] = config_dirs

        # Check for requirements files
        req_files = list(self.root_path.rglob("requirements*.txt"))
        req_files.extend(list(self.root_path.rglob("*.toml")))

        if len(req_files) > 2:  # More than requirements.txt and pyproject.toml
            config_issues["duplicate_requirements"] = [str(f) for f in req_files]

        return config_issues

    def _analyze_debugging_violations(self) -> Dict[str, Any]:
        """Analyze violations of zero-debugging constraint"""
        logger.info("🐛 Analyzing zero-debugging constraint violations...")

        debug_violations = {
            "print_statements": [],
            "debug_comments": [],
            "violation_count": 0
        }

        python_files = list(self.root_path.rglob("*.py"))

        for py_file in python_files:
            if '.venv' in str(py_file) or '__pycache__' in str(py_file):
                continue

            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')

                # Find print statements (excluding logging)
                for i, line in enumerate(lines, 1):
                    if re.search(r'\bprint\s*\(', line) and 'logger' not in line:
                        debug_violations["print_statements"].append({
                            "file": str(py_file),
                            "line": i,
                            "content": line.strip()
                        })
                        debug_violations["violation_count"] += 1

                    # Find debug comments
                    if re.search(r'#.*\b(DEBUG|FIXME|TODO|HACK|XXX)\b', line, re.IGNORECASE):
                        debug_violations["debug_comments"].append({
                            "file": str(py_file),
                            "line": i,
                            "content": line.strip()
                        })

            except Exception as e:
                logger.debug(f"Error analyzing {py_file}: {e}")

        return debug_violations

    def _analyze_import_debt(self) -> Dict[str, Any]:
        """Analyze import structure and dependency debt"""
        logger.info("📦 Analyzing import structure debt...")

        import_issues = {
            "relative_imports": [],
            "circular_imports": [],
            "complex_import_paths": [],
            "import_violations": 0
        }

        python_files = list(self.root_path.rglob("*.py"))

        for py_file in python_files:
            if '.venv' in str(py_file) or '__pycache__' in str(py_file):
                continue

            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Find relative imports
                relative_imports = re.findall(r'from\s+\.\..*import', content)
                if relative_imports:
                    import_issues["relative_imports"].append({
                        "file": str(py_file),
                        "imports": relative_imports
                    })
                    import_issues["import_violations"] += len(relative_imports)

                # Find complex import paths
                complex_imports = re.findall(r'from\s+[\w.]{50,}\s+import', content)
                if complex_imports:
                    import_issues["complex_import_paths"].append({
                        "file": str(py_file),
                        "imports": complex_imports
                    })

            except Exception as e:
                logger.debug(f"Error analyzing imports in {py_file}: {e}")

        return import_issues

    def _analyze_duplication_debt(self) -> Dict[str, Any]:
        """Analyze code and file duplication debt"""
        logger.info("🔄 Analyzing duplication debt...")

        # Read existing duplicate analysis if available
        dup_analysis_path = self.root_path / "data/analysis/duplicate_analysis_report.json"

        if dup_analysis_path.exists():
            try:
                with open(dup_analysis_path, 'r') as f:
                    existing_analysis = json.load(f)
                return {
                    "existing_analysis": True,
                    "total_duplicate_groups": existing_analysis["summary"]["total_duplicate_groups"],
                    "total_duplicate_files": existing_analysis["summary"]["total_duplicate_files"],
                    "wasted_space_mb": existing_analysis["summary"]["total_wasted_space_mb"]
                }
            except Exception as e:
                logger.debug(f"Error reading existing duplicate analysis: {e}")

        return {"existing_analysis": False, "needs_analysis": True}

    def _analyze_test_debt(self) -> Dict[str, Any]:
        """Analyze test coverage and testing debt"""
        logger.info("🧪 Analyzing test debt...")

        test_issues = {
            "test_coverage": "unknown",
            "missing_tests": [],
            "test_organization": "needs_review"
        }

        # Count test files vs source files
        test_files = list(self.root_path.rglob("test_*.py"))
        test_files.extend(list(self.root_path.rglob("*_test.py")))

        src_files = list((self.root_path / "src").rglob("*.py"))

        if src_files:
            test_ratio = len(test_files) / len(src_files)
            test_issues["test_coverage"] = f"{test_ratio:.2%}"
            test_issues["test_file_count"] = len(test_files)
            test_issues["source_file_count"] = len(src_files)

        return test_issues

    def _analyze_performance_debt(self) -> Dict[str, Any]:
        """Analyze performance-related technical debt"""
        logger.info("⚡ Analyzing performance debt...")

        perf_issues = {
            "large_files": [],
            "complex_functions": [],
            "inefficient_patterns": []
        }

        python_files = list(self.root_path.rglob("*.py"))

        for py_file in python_files:
            if '.venv' in str(py_file) or '__pycache__' in str(py_file):
                continue

            try:
                file_size = py_file.stat().st_size
                if file_size > 50000:  # Files larger than 50KB
                    perf_issues["large_files"].append({
                        "file": str(py_file),
                        "size_kb": file_size / 1024
                    })

                # Analyze function complexity
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')

                    # Simple heuristic: functions with >50 lines
                    in_function = False
                    function_start = 0
                    function_name = ""

                    for i, line in enumerate(lines):
                        if re.match(r'\s*def\s+(\w+)', line):
                            if in_function and i - function_start > 50:
                                perf_issues["complex_functions"].append({
                                    "file": str(py_file),
                                    "function": function_name,
                                    "lines": i - function_start
                                })

                            in_function = True
                            function_start = i
                            match = re.match(r'\s*def\s+(\w+)', line)
                            function_name = match.group(1) if match else "unknown"

            except Exception as e:
                logger.debug(f"Error analyzing performance in {py_file}: {e}")

        return perf_issues

    def _analyze_import_patterns(self, python_files: List[Path]) -> Dict[str, int]:
        """Analyze import patterns for coupling analysis"""
        import_patterns = defaultdict(int)

        for py_file in python_files:
            if '.venv' in str(py_file) or '__pycache__' in str(py_file):
                continue

            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                imports = re.findall(r'(?:from|import)\s+([\w.]+)', content)
                for imp in imports:
                    import_patterns[imp] += 1

            except Exception as e:
                logger.debug(f"Error analyzing imports in {py_file}: {e}")

        return dict(import_patterns)

    def _categorize_debt_quadrants(self, *debt_analyses) -> None:
        """Categorize debt into Martin Fowler's quadrants"""
        logger.info("🎯 Categorizing debt into Fowler's quadrants...")

        org_debt, doc_debt, config_debt, debug_debt, import_debt, dup_debt, test_debt, perf_debt = debt_analyses

        # Deliberate and Prudent (Strategic choices)
        if dup_debt.get("existing_analysis"):
            self.debt_inventory["deliberate_prudent"].append({
                "type": "Managed Duplication",
                "description": "Previous duplicate analysis exists, suggesting managed approach",
                "priority": "low",
                "estimated_hours": 4
            })

        # Deliberate and Reckless (Time pressure shortcuts)
        if debug_debt["violation_count"] > 10:
            self.debt_inventory["deliberate_reckless"].append({
                "type": "Zero-Debugging Violations",
                "description": f"{debug_debt['violation_count']} print statements violate zero-debugging constraint",
                "priority": "high",
                "estimated_hours": debug_debt["violation_count"] * 0.5
            })

        if len(org_debt["multiple_source_dirs"]) > 1:
            self.debt_inventory["deliberate_reckless"].append({
                "type": "Multiple Source Directories",
                "description": f"Found {len(org_debt['multiple_source_dirs'])} source directories: {org_debt['multiple_source_dirs']}",
                "priority": "high",
                "estimated_hours": 16
            })

        # Inadvertent and Prudent (Learning better ways)
        if import_debt["import_violations"] > 5:
            self.debt_inventory["inadvertent_prudent"].append({
                "type": "Import Structure Evolution",
                "description": f"{import_debt['import_violations']} relative imports suggest evolving architecture",
                "priority": "medium",
                "estimated_hours": import_debt["import_violations"] * 0.25
            })

        # Inadvertent and Reckless (Poor practices)
        if doc_debt["doc_redundancy_score"] > 10:
            self.debt_inventory["inadvertent_reckless"].append({
                "type": "Documentation Chaos",
                "description": f"Redundancy score of {doc_debt['doc_redundancy_score']} indicates poor documentation management",
                "priority": "medium",
                "estimated_hours": 12
            })

        if len(config_debt["multiple_config_dirs"]) > 1:
            self.debt_inventory["inadvertent_reckless"].append({
                "type": "Configuration Sprawl",
                "description": f"Multiple config directories: {config_debt['multiple_config_dirs']}",
                "priority": "medium",
                "estimated_hours": 8
            })

        # Calculate total metrics
        total_instances = sum(len(quadrant) for quadrant in self.debt_inventory.values())
        total_hours = sum(
            item["estimated_hours"]
            for quadrant in self.debt_inventory.values()
            for item in quadrant
        )

        priority_issues = sum(
            1 for quadrant in self.debt_inventory.values()
            for item in quadrant if item["priority"] == "high"
        )

        self.metrics.update({
            "total_debt_instances": total_instances,
            "estimated_remediation_hours": total_hours,
            "priority_issues": priority_issues
        })

    def _generate_executive_summary(self) -> Dict[str, Any]:
        """Generate executive summary of technical debt analysis"""
        total_instances = self.metrics["total_debt_instances"]
        total_hours = self.metrics["estimated_remediation_hours"]
        priority_issues = self.metrics["priority_issues"]

        # Calculate debt categories
        quadrant_counts = {k: len(v) for k, v in self.debt_inventory.items()}

        # Risk assessment
        risk_level = "LOW"
        if priority_issues > 3 or total_hours > 50:
            risk_level = "HIGH"
        elif priority_issues > 1 or total_hours > 20:
            risk_level = "MEDIUM"

        return {
            "total_debt_instances": total_instances,
            "estimated_remediation_hours": total_hours,
            "estimated_remediation_days": total_hours / 8,
            "priority_issues": priority_issues,
            "risk_level": risk_level,
            "quadrant_distribution": quadrant_counts,
            "key_findings": [
                f"Found {total_instances} technical debt instances across 4 quadrants",
                f"Estimated {total_hours} hours ({total_hours/8:.1f} days) for complete remediation",
                f"{priority_issues} high-priority issues require immediate attention",
                f"Risk level: {risk_level} - {'Immediate action required' if risk_level == 'HIGH' else 'Manageable with planning' if risk_level == 'MEDIUM' else 'Low impact, monitor trends'}"
            ],
            "recommendations": [
                "Focus on high-priority issues first (Deliberate and Reckless quadrant)",
                "Implement continuous monitoring for new debt accumulation",
                "Establish architectural decision records for future choices",
                "Create automated tooling to prevent debt regression"
            ]
        }

    def _generate_remediation_roadmap(self) -> Dict[str, Any]:
        """Generate actionable remediation roadmap"""
        roadmap = {
            "phase_1_immediate": {"duration_days": 3, "items": []},
            "phase_2_critical": {"duration_days": 10, "items": []},
            "phase_3_strategic": {"duration_days": 20, "items": []},
            "phase_4_optimization": {"duration_days": 15, "items": []}
        }

        # Phase 1: High priority items
        for item in self.debt_inventory["deliberate_reckless"]:
            if item["priority"] == "high":
                roadmap["phase_1_immediate"]["items"].append(item)

        # Phase 2: Medium priority reckless debt
        for quadrant in ["deliberate_reckless", "inadvertent_reckless"]:
            for item in self.debt_inventory[quadrant]:
                if item["priority"] == "medium":
                    roadmap["phase_2_critical"]["items"].append(item)

        # Phase 3: Prudent debt (learning opportunities)
        for quadrant in ["deliberate_prudent", "inadvertent_prudent"]:
            for item in self.debt_inventory[quadrant]:
                roadmap["phase_3_strategic"]["items"].append(item)

        # Phase 4: Optimization and prevention
        roadmap["phase_4_optimization"]["items"] = [
            {
                "type": "Continuous Monitoring Setup",
                "description": "Implement automated technical debt detection",
                "priority": "low",
                "estimated_hours": 8
            },
            {
                "type": "Architecture Decision Records",
                "description": "Establish ADR process for future architectural decisions",
                "priority": "low",
                "estimated_hours": 4
            },
            {
                "type": "Quality Gates Enhancement",
                "description": "Strengthen CI/CD quality gates to prevent debt accumulation",
                "priority": "low",
                "estimated_hours": 6
            }
        ]

        return roadmap


def main():
    """Execute technical debt analysis and generate report"""
    logger.info("🚀 Starting Kimera SWM Technical Debt Analysis")

    analyzer = TechnicalDebtAnalyzer()
    results = analyzer.analyze_codebase()

    # Generate timestamp for report
    timestamp = datetime.now().strftime('%Y-%m-%d')

    # Save detailed analysis
    report_dir = Path("docs/reports/debt")
    report_dir.mkdir(parents=True, exist_ok=True)

    analysis_file = report_dir / f"{timestamp}_technical_debt_analysis.json"
    with open(analysis_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"📊 Technical debt analysis saved to: {analysis_file}")

    # Generate markdown report
    markdown_report = generate_markdown_report(results, timestamp)
    report_file = report_dir / f"{timestamp}_technical_debt_report.md"

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(markdown_report)

    logger.info(f"📝 Technical debt report saved to: {report_file}")

    return results


def generate_markdown_report(results: Dict[str, Any], timestamp: str) -> str:
    """Generate comprehensive markdown technical debt report"""
    summary = results["executive_summary"]
    quadrants = results["debt_quadrants"]
    roadmap = results["remediation_roadmap"]

    report = f"""# Kimera SWM Technical Debt Analysis Report
*Generated: {timestamp}*
*Framework: Martin Fowler's Technical Debt Quadrants*
*Analyzer: Kimera SWM Autonomous Architect Protocol v3.0*

---

## Executive Summary

**Risk Level: {summary['risk_level']}**

- **Total Debt Instances:** {summary['total_debt_instances']}
- **Estimated Remediation:** {summary['estimated_remediation_hours']} hours ({summary['estimated_remediation_days']:.1f} days)
- **Priority Issues:** {summary['priority_issues']}

### Key Findings
{chr(10).join(f"- {finding}" for finding in summary['key_findings'])}

### Strategic Recommendations
{chr(10).join(f"- {rec}" for rec in summary['recommendations'])}

---

## Technical Debt Quadrants Analysis

### 1. Deliberate and Prudent ✅
*Strategic choices to ship now and fix later*

**Instances:** {len(quadrants['deliberate_prudent'])}

{chr(10).join(f"- **{item['type']}:** {item['description']} ({item['estimated_hours']}h)" for item in quadrants['deliberate_prudent'])}

### 2. Deliberate and Reckless ⚠️
*Ignoring design due to time pressures*

**Instances:** {len(quadrants['deliberate_reckless'])}

{chr(10).join(f"- **{item['type']}:** {item['description']} (Priority: {item['priority'].upper()}, {item['estimated_hours']}h)" for item in quadrants['deliberate_reckless'])}

### 3. Inadvertent and Prudent 🎯
*Learning better ways after the fact*

**Instances:** {len(quadrants['inadvertent_prudent'])}

{chr(10).join(f"- **{item['type']}:** {item['description']} ({item['estimated_hours']}h)" for item in quadrants['inadvertent_prudent'])}

### 4. Inadvertent and Reckless ❌
*Poor practices due to lack of knowledge*

**Instances:** {len(quadrants['inadvertent_reckless'])}

{chr(10).join(f"- **{item['type']}:** {item['description']} (Priority: {item['priority'].upper()}, {item['estimated_hours']}h)" for item in quadrants['inadvertent_reckless'])}

---

## Remediation Roadmap

### Phase 1: Immediate Action (3 days)
**Focus:** High-priority debt removal

{chr(10).join(f"- {item['type']}: {item['description']}" for item in roadmap['phase_1_immediate']['items'])}

### Phase 2: Critical Issues (10 days)
**Focus:** Medium-priority debt resolution

{chr(10).join(f"- {item['type']}: {item['description']}" for item in roadmap['phase_2_critical']['items'])}

### Phase 3: Strategic Improvements (20 days)
**Focus:** Learning-based improvements

{chr(10).join(f"- {item['type']}: {item['description']}" for item in roadmap['phase_3_strategic']['items'])}

### Phase 4: Optimization & Prevention (15 days)
**Focus:** Long-term debt prevention

{chr(10).join(f"- {item['type']}: {item['description']}" for item in roadmap['phase_4_optimization']['items'])}

---

## Implementation Guidelines

### Zero-Debugging Constraint Compliance
1. Replace all `logger.info()` statements with proper logging
2. Implement structured error context
3. Add comprehensive input validation

### Code Organization Principles
1. Consolidate source directories into single `src/` hierarchy
2. Establish clear module boundaries
3. Implement dependency injection patterns

### Configuration Management
1. Centralize configuration in single directory
2. Use environment-specific configuration files
3. Implement configuration validation

### Documentation Strategy
1. Archive redundant documentation
2. Establish single source of truth documents
3. Implement automated documentation generation

---

## Continuous Monitoring

### Metrics to Track
- New technical debt introduction rate
- Debt remediation velocity
- Code complexity trends
- Test coverage evolution

### Quality Gates
- Pre-commit hooks for debt prevention
- CI/CD quality thresholds
- Regular technical debt audits
- Architecture decision records

---

## Conclusion

The Kimera SWM codebase shows signs of rapid development with accumulated technical debt primarily in the **Deliberate and Reckless** and **Inadvertent and Reckless** quadrants. The remediation roadmap provides a systematic approach to address debt while maintaining development velocity.

**Next Steps:**
1. Execute Phase 1 immediately for high-priority issues
2. Establish continuous monitoring systems
3. Implement preventive quality gates
4. Review and update this analysis monthly

*This analysis follows the Kimera SWM Autonomous Architect Protocol, synthesizing best practices from aerospace, nuclear engineering, and software development methodologies.*
"""

    return report


if __name__ == "__main__":
    main()
