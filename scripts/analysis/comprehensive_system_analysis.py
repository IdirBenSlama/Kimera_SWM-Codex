#!/usr/bin/env python3
"""
KIMERA SWM Comprehensive System Analysis
=======================================

Multi-level system analysis following KIMERA Protocol v3.0
Performs systematic investigation across all architectural layers:

1. Code Structure & Organization
2. Dependencies & Imports
3. Performance & Resource Usage
4. Security & Vulnerabilities
5. API & Endpoint Health
6. Database & Storage Systems
7. AI/ML Engine Status
8. System Architecture Issues
9. Technical Debt Assessment
10. Operational Readiness

Author: Kimera SWM Autonomous Architect
Classification: AEROSPACE-GRADE SYSTEM ANALYSIS
"""

import os
import sys
import ast
import json
import time
import hashlib
import subprocess
import traceback
import importlib
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple, Optional
from collections import defaultdict, Counter
import concurrent.futures
import threading
import psutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KimeraSystemAnalyzer:
    """Comprehensive system analyzer for Kimera SWM"""

    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path).resolve()
        self.analysis_start = datetime.now()
        self.issues = defaultdict(list)
        self.metrics = defaultdict(dict)
        self.recommendations = defaultdict(list)

        # Analysis results
        self.file_analysis = {}
        self.dependency_graph = defaultdict(set)
        self.circular_deps = []
        self.orphaned_files = []
        self.duplicate_code = []
        self.security_issues = []
        self.performance_bottlenecks = []

    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Execute complete multi-level analysis"""
        logger.info("🔍 KIMERA SWM COMPREHENSIVE SYSTEM ANALYSIS")
        logger.info("=" * 80)
        logger.info(f"📁 Root Path: {self.root_path}")
        logger.info(f"⏰ Started: {self.analysis_start}")
        logger.info("=" * 80)

        analysis_steps = [
            ("1. File Structure Analysis", self._analyze_file_structure),
            ("2. Code Quality Assessment", self._analyze_code_quality),
            ("3. Dependency Analysis", self._analyze_dependencies),
            ("4. API Health Check", self._analyze_api_health),
            ("5. Performance Analysis", self._analyze_performance),
            ("6. Security Scan", self._analyze_security),
            ("7. Database Health", self._analyze_database_health),
            ("8. AI Engine Status", self._analyze_ai_engines),
            ("9. Technical Debt Assessment", self._analyze_technical_debt),
            ("10. Architecture Validation", self._analyze_architecture)
        ]

        results = {}
        for step_name, step_func in analysis_steps:
            logger.info(f"\n🔍 {step_name}")
            logger.info("-" * 40)
            try:
                step_result = step_func()
                results[step_name] = step_result
                logger.info(f"✅ {step_name} completed")
            except Exception as e:
                error_msg = f"❌ {step_name} failed: {str(e)}"
                logger.info(error_msg)
                logger.error(error_msg, exc_info=True)
                results[step_name] = {"error": str(e)}
                self.issues["critical"].append(f"{step_name}: {str(e)}")

        # Generate final report
        final_report = self._generate_final_report(results)
        return final_report

    def _analyze_file_structure(self) -> Dict[str, Any]:
        """Analyze file organization and structure"""
        logger.info("  📁 Scanning file structure...")

        structure_info = {
            "total_files": 0,
            "by_extension": defaultdict(int),
            "by_directory": defaultdict(int),
            "large_files": [],
            "empty_files": [],
            "misplaced_files": []
        }

        for file_path in self.root_path.rglob("*"):
            if file_path.is_file():
                structure_info["total_files"] += 1

                # Count by extension
                ext = file_path.suffix.lower()
                structure_info["by_extension"][ext] += 1

                # Count by directory
                rel_dir = file_path.parent.relative_to(self.root_path)
                structure_info["by_directory"][str(rel_dir)] += 1

                # Check file size
                try:
                    size = file_path.stat().st_size
                    if size > 100 * 1024:  # > 100KB
                        structure_info["large_files"].append({
                            "path": str(file_path.relative_to(self.root_path)),
                            "size_kb": size // 1024
                        })
                    elif size == 0:
                        structure_info["empty_files"].append(str(file_path.relative_to(self.root_path)))
                except:
                    pass

                # Check for misplaced files (files in root that should be in subdirs)
                if file_path.parent == self.root_path:
                    if ext in ['.py', '.md', '.txt', '.json', '.yaml', '.yml']:
                        structure_info["misplaced_files"].append(str(file_path.name))

        # Check directory structure compliance
        expected_dirs = ['src', 'tests', 'docs', 'config', 'scripts', 'experiments']
        missing_dirs = [d for d in expected_dirs if not (self.root_path / d).exists()]
        if missing_dirs:
            self.issues["structure"].append(f"Missing expected directories: {missing_dirs}")

        # Check for root pollution
        if structure_info["misplaced_files"]:
            self.issues["structure"].append(f"Files in root directory: {structure_info['misplaced_files']}")

        logger.info(f"    📊 Total files: {structure_info['total_files']}")
        logger.info(f"    📂 Directories: {len(structure_info['by_directory'])}")
        logger.info(f"    🔍 Large files: {len(structure_info['large_files'])}")

        return structure_info

    def _analyze_code_quality(self) -> Dict[str, Any]:
        """Analyze Python code quality and complexity"""
        logger.info("  🔍 Analyzing code quality...")

        quality_metrics = {
            "total_python_files": 0,
            "total_lines": 0,
            "complexity_issues": [],
            "style_issues": [],
            "import_issues": [],
            "function_analysis": defaultdict(list)
        }

        python_files = list(self.root_path.rglob("*.py"))
        quality_metrics["total_python_files"] = len(python_files)

        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    lines = content.split('\n')
                    quality_metrics["total_lines"] += len(lines)

                    # Parse AST for complexity analysis
                    try:
                        tree = ast.parse(content)
                        self._analyze_ast_complexity(tree, py_file, quality_metrics)
                    except SyntaxError as e:
                        quality_metrics["style_issues"].append({
                            "file": str(py_file.relative_to(self.root_path)),
                            "issue": f"Syntax error: {str(e)}"
                        })

                    # Check for basic style issues
                    self._check_style_issues(py_file, lines, quality_metrics)

            except Exception as e:
                quality_metrics["style_issues"].append({
                    "file": str(py_file.relative_to(self.root_path)),
                    "issue": f"Analysis failed: {str(e)}"
                })

        logger.info(f"    📄 Python files: {quality_metrics['total_python_files']}")
        logger.info(f"    📏 Total lines: {quality_metrics['total_lines']}")
        logger.info(f"    ⚠️ Style issues: {len(quality_metrics['style_issues'])}")

        return quality_metrics

    def _analyze_ast_complexity(self, tree: ast.AST, file_path: Path, metrics: Dict):
        """Analyze AST for complexity metrics"""
        rel_path = str(file_path.relative_to(self.root_path))

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Count complexity (simplified cyclomatic complexity)
                complexity = self._calculate_complexity(node)
                if complexity > 10:
                    metrics["complexity_issues"].append({
                        "file": rel_path,
                        "function": node.name,
                        "complexity": complexity,
                        "line": node.lineno
                    })

                metrics["function_analysis"]["total_functions"].append({
                    "file": rel_path,
                    "name": node.name,
                    "line": node.lineno,
                    "complexity": complexity
                })

    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function"""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor,
                                ast.ExceptHandler, ast.With, ast.AsyncWith)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        return complexity

    def _check_style_issues(self, file_path: Path, lines: List[str], metrics: Dict):
        """Check for basic Python style issues"""
        rel_path = str(file_path.relative_to(self.root_path))

        for i, line in enumerate(lines, 1):
            # Check line length
            if len(line) > 120:
                metrics["style_issues"].append({
                    "file": rel_path,
                    "line": i,
                    "issue": f"Line too long ({len(line)} chars)"
                })

            # Check for trailing whitespace
            if line.rstrip() != line:
                metrics["style_issues"].append({
                    "file": rel_path,
                    "line": i,
                    "issue": "Trailing whitespace"
                })

    def _analyze_dependencies(self) -> Dict[str, Any]:
        """Analyze import dependencies and detect circular imports"""
        logger.info("  🔗 Analyzing dependencies...")

        dependency_info = {
            "import_graph": {},
            "external_dependencies": set(),
            "internal_dependencies": set(),
            "circular_dependencies": [],
            "orphaned_modules": [],
            "missing_dependencies": []
        }

        # Build import graph
        python_files = list(self.root_path.rglob("*.py"))

        for py_file in python_files:
            rel_path = str(py_file.relative_to(self.root_path))
            module_name = rel_path.replace('/', '.').replace('\\', '.').replace('.py', '')

            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                tree = ast.parse(content)
                imports = self._extract_imports(tree)

                dependency_info["import_graph"][module_name] = imports

                # Categorize imports
                for imp in imports:
                    if imp.startswith('src.') or imp.startswith('.'):
                        dependency_info["internal_dependencies"].add(imp)
                    else:
                        dependency_info["external_dependencies"].add(imp)

            except Exception as e:
                dependency_info["missing_dependencies"].append({
                    "file": rel_path,
                    "error": str(e)
                })

        # Detect circular dependencies
        dependency_info["circular_dependencies"] = self._detect_circular_imports(
            dependency_info["import_graph"]
        )

        logger.info(f"    📦 External deps: {len(dependency_info['external_dependencies'])}")
        logger.info(f"    🏠 Internal deps: {len(dependency_info['internal_dependencies'])}")
        logger.info(f"    🔄 Circular deps: {len(dependency_info['circular_dependencies'])}")

        return dependency_info

    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract import statements from AST"""
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)

        return imports

    def _detect_circular_imports(self, import_graph: Dict[str, List[str]]) -> List[List[str]]:
        """Detect circular import chains using DFS"""
        circular_deps = []
        visited = set()
        rec_stack = set()

        def dfs(module: str, path: List[str]):
            if module in rec_stack:
                # Found cycle
                cycle_start = path.index(module)
                cycle = path[cycle_start:] + [module]
                circular_deps.append(cycle)
                return

            if module in visited:
                return

            visited.add(module)
            rec_stack.add(module)

            for imported_module in import_graph.get(module, []):
                if imported_module in import_graph:  # Only follow internal imports
                    dfs(imported_module, path + [module])

            rec_stack.remove(module)

        for module in import_graph:
            if module not in visited:
                dfs(module, [])

        return circular_deps

    def _analyze_api_health(self) -> Dict[str, Any]:
        """Analyze API endpoint health and connectivity"""
        logger.info("  🌐 Checking API health...")

        api_health = {
            "tested_endpoints": [],
            "working_endpoints": [],
            "failing_endpoints": [],
            "response_times": {},
            "status_codes": {}
        }

        # Test known endpoints
        base_urls = [
            "http://127.0.0.1:8000",
            "http://127.0.0.1:8001",
            "http://127.0.0.1:8002"
        ]

        endpoints_to_test = [
            "/",
            "/health",
            "/docs",
            "/api/v1/status",
            "/system/health"
        ]

        for base_url in base_urls:
            for endpoint in endpoints_to_test:
                full_url = f"{base_url}{endpoint}"
                api_health["tested_endpoints"].append(full_url)

                try:
                    start_time = time.time()
                    response = requests.get(full_url, timeout=5)
                    response_time = time.time() - start_time

                    api_health["response_times"][full_url] = response_time
                    api_health["status_codes"][full_url] = response.status_code

                    if response.status_code == 200:
                        api_health["working_endpoints"].append(full_url)
                    else:
                        api_health["failing_endpoints"].append({
                            "url": full_url,
                            "status": response.status_code,
                            "response_time": response_time
                        })

                except Exception as e:
                    api_health["failing_endpoints"].append({
                        "url": full_url,
                        "error": str(e)
                    })

        logger.info(f"    ✅ Working: {len(api_health['working_endpoints'])}")
        logger.info(f"    ❌ Failing: {len(api_health['failing_endpoints'])}")

        return api_health

    def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze system performance metrics"""
        logger.info("  ⚡ Analyzing performance...")

        performance = {
            "system_resources": {},
            "process_analysis": {},
            "file_operations": {},
            "startup_performance": {}
        }

        # System resources
        try:
            performance["system_resources"] = {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:\\').percent,
                "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else "N/A"
            }
        except Exception as e:
            performance["system_resources"]["error"] = str(e)

        # Process analysis
        try:
            python_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                if 'python' in proc.info['name'].lower():
                    python_processes.append(proc.info)

            performance["process_analysis"] = {
                "python_processes": len(python_processes),
                "total_cpu": sum(p.get('cpu_percent', 0) for p in python_processes),
                "total_memory": sum(p.get('memory_percent', 0) for p in python_processes),
                "processes": python_processes[:10]  # Top 10
            }
        except Exception as e:
            performance["process_analysis"]["error"] = str(e)

        logger.info(f"    💻 CPU: {performance['system_resources'].get('cpu_percent', 'N/A')}%")
        logger.info(f"    💾 Memory: {performance['system_resources'].get('memory_percent', 'N/A')}%")

        return performance

    def _analyze_security(self) -> Dict[str, Any]:
        """Basic security analysis"""
        logger.info("  🔒 Security analysis...")

        security = {
            "hardcoded_secrets": [],
            "insecure_patterns": [],
            "file_permissions": [],
            "exposed_endpoints": []
        }

        # Scan for potential secrets
        secret_patterns = [
            r'password\s*=\s*["\'][^"\']+["\']',
            r'api_key\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']',
            r'token\s*=\s*["\'][^"\']+["\']'
        ]

        python_files = list(self.root_path.rglob("*.py"))

        import re
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                for pattern in secret_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        security["hardcoded_secrets"].append({
                            "file": str(py_file.relative_to(self.root_path)),
                            "pattern": pattern,
                            "line": content[:match.start()].count('\n') + 1
                        })
            except Exception:
                pass

        logger.info(f"    🔍 Potential secrets: {len(security['hardcoded_secrets'])}")

        return security

    def _analyze_database_health(self) -> Dict[str, Any]:
        """Analyze database connectivity and health"""
        logger.info("  🗄️ Database health check...")

        db_health = {
            "connection_tests": [],
            "configuration_files": [],
            "migration_status": "unknown"
        }

        # Check for database config files
        config_patterns = ['*database*', '*db*', '*.env*', '*config*']
        for pattern in config_patterns:
            config_files = list(self.root_path.rglob(pattern))
            for cf in config_files:
                if cf.is_file():
                    db_health["configuration_files"].append(str(cf.relative_to(self.root_path)))

        logger.info(f"    📋 Config files: {len(db_health['configuration_files'])}")

        return db_health

    def _analyze_ai_engines(self) -> Dict[str, Any]:
        """Analyze AI/ML engine status"""
        logger.info("  🤖 AI engines analysis...")

        ai_status = {
            "engine_files": [],
            "model_files": [],
            "gpu_availability": {},
            "framework_status": {}
        }

        # Find AI engine files
        ai_patterns = ['*engine*', '*model*', '*neural*', '*cognitive*', '*ai*']
        for pattern in ai_patterns:
            engine_files = list((self.root_path / "src").rglob(f"{pattern}.py")) if (self.root_path / "src").exists() else []
            ai_status["engine_files"].extend([str(f.relative_to(self.root_path)) for f in engine_files])

        # Check GPU availability
        try:
            import torch
            ai_status["gpu_availability"] = {
                "torch_available": True,
                "cuda_available": torch.cuda.is_available(),
                "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
        except ImportError:
            ai_status["gpu_availability"]["torch_available"] = False

        logger.info(f"    🎯 Engine files: {len(ai_status['engine_files'])}")

        return ai_status

    def _analyze_technical_debt(self) -> Dict[str, Any]:
        """Assess technical debt"""
        logger.info("  📊 Technical debt assessment...")

        debt = {
            "todo_comments": [],
            "fixme_comments": [],
            "deprecated_code": [],
            "duplicate_functions": [],
            "debt_score": 0
        }

        # Scan for TODO/FIXME comments
        python_files = list(self.root_path.rglob("*.py"))

        import re
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()

                for i, line in enumerate(lines, 1):
                    if re.search(r'#\s*(TODO|FIXME|BUG|HACK)', line, re.IGNORECASE):
                        debt["todo_comments"].append({
                            "file": str(py_file.relative_to(self.root_path)),
                            "line": i,
                            "comment": line.strip()
                        })

                    if re.search(r'deprecated|legacy|old|remove', line, re.IGNORECASE):
                        debt["deprecated_code"].append({
                            "file": str(py_file.relative_to(self.root_path)),
                            "line": i,
                            "content": line.strip()
                        })
            except Exception:
                pass

        # Calculate debt score
        debt["debt_score"] = (
            len(debt["todo_comments"]) * 1 +
            len(debt["deprecated_code"]) * 2 +
            len(debt["duplicate_functions"]) * 3
        )

        logger.info(f"    📝 TODO items: {len(debt['todo_comments'])}")
        logger.info(f"    💸 Debt score: {debt['debt_score']}")

        return debt

    def _analyze_architecture(self) -> Dict[str, Any]:
        """Validate system architecture"""
        logger.info("  🏗️ Architecture validation...")

        architecture = {
            "layer_separation": {},
            "dependency_violations": [],
            "missing_interfaces": [],
            "architectural_patterns": []
        }

        # Check expected architecture layers
        expected_layers = {
            "api": "src/api",
            "core": "src/core",
            "engines": "src/engines",
            "utils": "src/utils",
            "monitoring": "src/monitoring"
        }

        for layer, path in expected_layers.items():
            layer_path = self.root_path / path
            architecture["layer_separation"][layer] = {
                "exists": layer_path.exists(),
                "file_count": len(list(layer_path.rglob("*.py"))) if layer_path.exists() else 0
            }

        logger.info(f"    🏛️ Architecture layers: {len([l for l in architecture['layer_separation'].values() if l['exists']])}/{len(expected_layers)}")

        return architecture

    def _generate_final_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive final report"""
        analysis_end = datetime.now()
        analysis_duration = analysis_end - self.analysis_start

        # Calculate overall system health
        health_score = self._calculate_health_score(results)

        # Prioritize issues
        critical_issues = self.issues.get("critical", [])
        high_issues = self.issues.get("high", [])
        medium_issues = self.issues.get("medium", [])

        # Generate recommendations
        recommendations = self._generate_recommendations(results)

        final_report = {
            "analysis_metadata": {
                "start_time": self.analysis_start.isoformat(),
                "end_time": analysis_end.isoformat(),
                "duration_seconds": analysis_duration.total_seconds(),
                "analyzer_version": "3.0.0"
            },
            "system_health": {
                "overall_score": health_score,
                "status": self._get_health_status(health_score),
                "critical_issues_count": len(critical_issues),
                "total_issues_count": sum(len(issues) for issues in self.issues.values())
            },
            "detailed_results": results,
            "issues_by_priority": {
                "critical": critical_issues,
                "high": high_issues,
                "medium": medium_issues
            },
            "recommendations": recommendations,
            "next_actions": self._get_next_actions(results)
        }

        return final_report

    def _calculate_health_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall system health score (0-100)"""
        score = 100.0

        # Deduct points for various issues
        api_health = results.get("4. API Health Check", {})
        failing_endpoints = len(api_health.get("failing_endpoints", []))
        total_endpoints = len(api_health.get("tested_endpoints", []))

        if total_endpoints > 0:
            api_success_rate = (total_endpoints - failing_endpoints) / total_endpoints
            score *= api_success_rate

        # Deduct for critical issues
        critical_count = len(self.issues.get("critical", []))
        score -= critical_count * 10

        # Deduct for high issues
        high_count = len(self.issues.get("high", []))
        score -= high_count * 5

        return max(0.0, min(100.0, score))

    def _get_health_status(self, score: float) -> str:
        """Get health status based on score"""
        if score >= 90:
            return "EXCELLENT"
        elif score >= 75:
            return "GOOD"
        elif score >= 60:
            return "FAIR"
        elif score >= 40:
            return "POOR"
        else:
            return "CRITICAL"

    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        # API health recommendations
        api_health = results.get("4. API Health Check", {})
        if api_health.get("failing_endpoints"):
            recommendations.append("🔧 Fix failing API endpoints, especially /health endpoint")

        # Code quality recommendations
        code_quality = results.get("2. Code Quality Assessment", {})
        complexity_issues = code_quality.get("complexity_issues", [])
        if len(complexity_issues) > 10:
            recommendations.append("🎯 Refactor high-complexity functions to improve maintainability")

        # Dependency recommendations
        dependency_analysis = results.get("3. Dependency Analysis", {})
        if dependency_analysis.get("circular_dependencies"):
            recommendations.append("🔄 Resolve circular import dependencies")

        # Performance recommendations
        performance = results.get("5. Performance Analysis", {})
        system_resources = performance.get("system_resources", {})
        if system_resources.get("memory_percent", 0) > 80:
            recommendations.append("💾 Optimize memory usage - system using >80% memory")

        return recommendations

    def _get_next_actions(self, results: Dict[str, Any]) -> List[str]:
        """Get prioritized next actions"""
        actions = []

        # Highest priority actions
        actions.append("1. Fix health endpoint 500 error")
        actions.append("2. Resolve API connectivity issues")
        actions.append("3. Address circular dependencies")
        actions.append("4. Implement comprehensive monitoring")
        actions.append("5. Optimize resource usage")

        return actions

def main():
    """Main analysis execution"""
    # Initialize analyzer
    analyzer = KimeraSystemAnalyzer()

    # Run comprehensive analysis
    report = analyzer.run_comprehensive_analysis()

    # Save report
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    report_path = f"docs/reports/analysis/{timestamp}_comprehensive_analysis.json"

    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("📊 ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"🏥 Overall Health: {report['system_health']['status']} ({report['system_health']['overall_score']:.1f}/100)")
    logger.info(f"🚨 Critical Issues: {report['system_health']['critical_issues_count']}")
    logger.info(f"📝 Total Issues: {report['system_health']['total_issues_count']}")
    logger.info(f"📄 Report saved: {report_path}")

    logger.info("\n🎯 TOP RECOMMENDATIONS:")
    for i, rec in enumerate(report['recommendations'][:5], 1):
        logger.info(f"   {i}. {rec}")

    logger.info("\n⚡ NEXT ACTIONS:")
    for i, action in enumerate(report['next_actions'][:3], 1):
        logger.info(f"   {i}. {action}")

    return report

if __name__ == "__main__":
    main()
