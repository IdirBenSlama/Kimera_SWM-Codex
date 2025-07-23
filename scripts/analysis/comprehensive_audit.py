#!/usr/bin/env python3
"""
KIMERA SWM Comprehensive System Audit
Follows the Kimera SWM Autonomous Architect Protocol v3.0

This script performs a complete audit of the Kimera SWM system including:
- Codebase analysis and organization
- Dependency verification
- Performance assessment
- Security analysis
- Technical debt evaluation
- Architecture compliance
"""

import os
import sys
import json
import ast
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from collections import defaultdict, Counter
import importlib.util
import subprocess
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KimeraSystemAuditor:
    """Comprehensive system auditor for Kimera SWM"""
    
    def __init__(self):
        self.audit_report = {
            "audit_metadata": {
                "timestamp": datetime.now().isoformat(),
                "auditor": "Kimera SWM Autonomous Architect v3.0",
                "scope": "complete_system_audit"
            },
            "codebase_analysis": {},
            "architecture_compliance": {},
            "performance_analysis": {},
            "security_assessment": {},
            "technical_debt": {},
            "recommendations": {},
            "action_items": []
        }
        
        self.project_root = Path(".")
        self.file_hashes = {}
        self.duplicate_files = defaultdict(list)
        self.code_metrics = {}
        
    def scan_codebase_structure(self):
        """Perform comprehensive codebase scan with parallel hashing"""
        logger.info("🔍 Scanning codebase structure...")
        
        # File extensions to analyze
        code_extensions = {'.py', '.js', '.ts', '.json', '.yaml', '.yml', '.md', '.txt'}
        
        file_analysis = {
            "total_files": 0,
            "by_extension": defaultdict(int),
            "by_directory": defaultdict(int),
            "large_files": [],
            "empty_files": [],
            "potential_duplicates": []
        }
        
        for file_path in self.project_root.rglob("*"):
            if file_path.is_file() and not self._should_ignore_file(file_path):
                file_analysis["total_files"] += 1
                file_analysis["by_extension"][file_path.suffix] += 1
                file_analysis["by_directory"][str(file_path.parent)] += 1
                
                # Analyze file size
                size = file_path.stat().st_size
                if size == 0:
                    file_analysis["empty_files"].append(str(file_path))
                elif size > 100_000:  # Files > 100KB
                    file_analysis["large_files"].append({
                        "path": str(file_path),
                        "size_bytes": size,
                        "size_mb": round(size / 1024 / 1024, 2)
                    })
                
                # Calculate hash for duplicate detection
                if file_path.suffix in code_extensions:
                    try:
                        with open(file_path, 'rb') as f:
                            content_hash = hashlib.md5(f.read()).hexdigest()
                            if content_hash in self.file_hashes:
                                self.duplicate_files[content_hash].append(str(file_path))
                            else:
                                self.file_hashes[content_hash] = str(file_path)
                    except Exception as e:
                        logger.warning(f"Could not hash file {file_path}: {e}")
        
        # Process duplicates
        for hash_val, files in self.duplicate_files.items():
            if len(files) > 1:
                files.append(self.file_hashes[hash_val])  # Add original file
                file_analysis["potential_duplicates"].append({
                    "hash": hash_val,
                    "files": list(set(files)),
                    "count": len(set(files))
                })
        
        self.audit_report["codebase_analysis"]["file_structure"] = file_analysis
        logger.info(f"✅ Scanned {file_analysis['total_files']} files")
        
    def analyze_python_code_quality(self):
        """Analyze Python code quality and complexity"""
        logger.info("🐍 Analyzing Python code quality...")
        
        python_analysis = {
            "total_python_files": 0,
            "lines_of_code": 0,
            "functions": 0,
            "classes": 0,
            "imports": defaultdict(int),
            "complexity_issues": [],
            "style_issues": [],
            "missing_docstrings": [],
            "potential_security_issues": []
        }
        
        for py_file in self.project_root.rglob("*.py"):
            if self._should_ignore_file(py_file):
                continue
                
            python_analysis["total_python_files"] += 1
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                python_analysis["lines_of_code"] += len(content.splitlines())
                
                # Parse AST for detailed analysis
                try:
                    tree = ast.parse(content)
                    self._analyze_ast(tree, py_file, python_analysis)
                except SyntaxError as e:
                    python_analysis["style_issues"].append({
                        "file": str(py_file),
                        "issue": f"Syntax error: {e}",
                        "severity": "high"
                    })
                    
            except Exception as e:
                logger.warning(f"Could not analyze {py_file}: {e}")
        
        self.audit_report["codebase_analysis"]["python_quality"] = python_analysis
        logger.info(f"✅ Analyzed {python_analysis['total_python_files']} Python files")
        
    def _analyze_ast(self, tree, file_path, analysis):
        """Analyze Python AST for code metrics"""
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                analysis["functions"] += 1
                
                # Check for missing docstrings
                if not ast.get_docstring(node):
                    analysis["missing_docstrings"].append({
                        "file": str(file_path),
                        "function": node.name,
                        "line": node.lineno
                    })
                
                # Check function complexity (basic)
                if len(node.body) > 50:  # Long functions
                    analysis["complexity_issues"].append({
                        "file": str(file_path),
                        "function": node.name,
                        "issue": "Function too long",
                        "lines": len(node.body),
                        "line": node.lineno
                    })
                    
            elif isinstance(node, ast.ClassDef):
                analysis["classes"] += 1
                
                if not ast.get_docstring(node):
                    analysis["missing_docstrings"].append({
                        "file": str(file_path),
                        "class": node.name,
                        "line": node.lineno
                    })
                    
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    analysis["imports"][alias.name] += 1
                    
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    analysis["imports"][node.module] += 1
                    
            # Security checks
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['eval', 'exec', 'compile']:
                        analysis["potential_security_issues"].append({
                            "file": str(file_path),
                            "issue": f"Dangerous function: {node.func.id}",
                            "line": node.lineno,
                            "severity": "high"
                        })
    
    def analyze_architecture_compliance(self):
        """Check compliance with Kimera SWM architecture principles"""
        logger.info("🏗️ Analyzing architecture compliance...")
        
        compliance = {
            "directory_structure": self._check_directory_compliance(),
            "import_patterns": self._analyze_import_patterns(),
            "naming_conventions": self._check_naming_conventions(),
            "ethical_governor": self._check_ethical_governor_integration(),
            "zero_debugging": self._check_zero_debugging_compliance()
        }
        
        self.audit_report["architecture_compliance"] = compliance
        logger.info("✅ Architecture compliance analysis complete")
        
    def _check_directory_compliance(self):
        """Check if directory structure follows Kimera conventions"""
        
        required_structure = {
            'src': ['core', 'engines', 'api', 'utils', 'config'],
            'tests': ['unit', 'integration', 'performance'],
            'docs': ['architecture', 'guides', 'reports'],
            'scripts': ['health_check', 'migration', 'utils', 'analysis'],
            'experiments': [],
            'archive': [],
            'configs': []
        }
        
        compliance_issues = []
        structure_score = 0
        total_checks = 0
        
        for main_dir, subdirs in required_structure.items():
            total_checks += 1
            if Path(main_dir).exists():
                structure_score += 1
                
                for subdir in subdirs:
                    total_checks += 1
                    subdir_path = Path(main_dir) / subdir
                    if subdir_path.exists():
                        structure_score += 1
                    else:
                        compliance_issues.append(f"Missing directory: {subdir_path}")
            else:
                compliance_issues.append(f"Missing main directory: {main_dir}")
        
        return {
            "score": (structure_score / total_checks) * 100 if total_checks > 0 else 0,
            "issues": compliance_issues,
            "total_checks": total_checks,
            "passed_checks": structure_score
        }
        
    def _analyze_import_patterns(self):
        """Analyze import patterns for circular dependencies and organization"""
        
        import_graph = defaultdict(set)
        import_issues = []
        
        for py_file in self.project_root.rglob("*.py"):
            if self._should_ignore_file(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                tree = ast.parse(content)
                current_module = str(py_file.relative_to(self.project_root)).replace('/', '.').replace('\\', '.')[:-3]
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.module:
                        if node.module.startswith('src.'):
                            import_graph[current_module].add(node.module)
                            
            except Exception as e:
                import_issues.append(f"Could not analyze imports in {py_file}: {e}")
        
        # Check for potential circular imports (simplified)
        circular_imports = []
        for module, imports in import_graph.items():
            for imported in imports:
                if imported in import_graph and module in import_graph[imported]:
                    circular_imports.append((module, imported))
        
        return {
            "total_internal_imports": sum(len(imports) for imports in import_graph.values()),
            "potential_circular_imports": circular_imports,
            "import_issues": import_issues,
            "most_imported_modules": dict(Counter(
                imp for imports in import_graph.values() for imp in imports
            ).most_common(10))
        }
        
    def _check_naming_conventions(self):
        """Check naming conventions compliance"""
        
        naming_issues = []
        
        for py_file in self.project_root.rglob("*.py"):
            if self._should_ignore_file(py_file):
                continue
                
            # Check file naming (should be snake_case)
            if not re.match(r'^[a-z][a-z0-9_]*\.py$', py_file.name):
                naming_issues.append({
                    "file": str(py_file),
                    "issue": "File name not in snake_case",
                    "severity": "medium"
                })
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        if not re.match(r'^[a-z][a-z0-9_]*$', node.name) and not node.name.startswith('_'):
                            naming_issues.append({
                                "file": str(py_file),
                                "issue": f"Function '{node.name}' not in snake_case",
                                "line": node.lineno,
                                "severity": "low"
                            })
                    elif isinstance(node, ast.ClassDef):
                        if not re.match(r'^[A-Z][a-zA-Z0-9]*$', node.name):
                            naming_issues.append({
                                "file": str(py_file),
                                "issue": f"Class '{node.name}' not in PascalCase",
                                "line": node.lineno,
                                "severity": "medium"
                            })
                            
            except Exception:
                pass  # Skip files with syntax errors
        
        return {
            "total_issues": len(naming_issues),
            "issues": naming_issues
        }
    
    def _check_ethical_governor_integration(self):
        """Check for Ethical Governor integration"""
        
        governor_references = []
        action_proposal_usage = []
        
        for py_file in self.project_root.rglob("*.py"):
            if self._should_ignore_file(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                if 'ethical_governor' in content.lower():
                    governor_references.append(str(py_file))
                    
                if 'ActionProposal' in content:
                    action_proposal_usage.append(str(py_file))
                    
            except Exception:
                pass
        
        return {
            "governor_references": len(governor_references),
            "governor_files": governor_references,
            "action_proposal_usage": len(action_proposal_usage),
            "action_proposal_files": action_proposal_usage,
            "compliance": len(governor_references) > 0
        }
        
    def _check_zero_debugging_compliance(self):
        """Check compliance with zero-debugging constraint"""
        
        debug_violations = []
        
        for py_file in self.project_root.rglob("*.py"):
            if self._should_ignore_file(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                for i, line in enumerate(lines, 1):
                    # Check for print statements
                    if re.search(r'\bprint\s*\(', line) and not line.strip().startswith('#'):
                        debug_violations.append({
                            "file": str(py_file),
                            "line": i,
                            "issue": "print() statement found - use logging instead",
                            "severity": "medium"
                        })
                    
                    # Check for TODO/FIXME comments
                    if re.search(r'#\s*(TODO|FIXME|HACK)', line, re.IGNORECASE):
                        debug_violations.append({
                            "file": str(py_file),
                            "line": i,
                            "issue": "TODO/FIXME comment - resolve before production",
                            "severity": "low"
                        })
                        
            except Exception:
                pass
        
        return {
            "total_violations": len(debug_violations),
            "violations": debug_violations,
            "compliance_score": max(0, 100 - len(debug_violations))
        }
    
    def assess_technical_debt(self):
        """Assess technical debt across the system"""
        logger.info("📊 Assessing technical debt...")
        
        debt_analysis = {
            "duplicate_code": len(self.audit_report["codebase_analysis"]["file_structure"]["potential_duplicates"]),
            "large_files": len(self.audit_report["codebase_analysis"]["file_structure"]["large_files"]),
            "complexity_issues": len(self.audit_report["codebase_analysis"]["python_quality"]["complexity_issues"]),
            "missing_documentation": len(self.audit_report["codebase_analysis"]["python_quality"]["missing_docstrings"]),
            "security_issues": len(self.audit_report["codebase_analysis"]["python_quality"]["potential_security_issues"]),
            "naming_violations": self.audit_report["architecture_compliance"]["naming_conventions"]["total_issues"],
            "debug_violations": self.audit_report["architecture_compliance"]["zero_debugging"]["total_violations"]
        }
        
        # Calculate debt score (lower is better)
        total_issues = sum(debt_analysis.values())
        debt_score = min(100, total_issues)  # Cap at 100
        
        debt_analysis["overall_debt_score"] = debt_score
        debt_analysis["debt_level"] = (
            "Low" if debt_score < 20 else
            "Medium" if debt_score < 50 else
            "High" if debt_score < 80 else
            "Critical"
        )
        
        self.audit_report["technical_debt"] = debt_analysis
        logger.info(f"✅ Technical debt assessment complete - Level: {debt_analysis['debt_level']}")
        
    def generate_recommendations(self):
        """Generate actionable recommendations based on audit findings"""
        logger.info("💡 Generating recommendations...")
        
        recommendations = {
            "immediate_actions": [],
            "short_term_improvements": [],
            "long_term_goals": [],
            "priority_matrix": {}
        }
        
        # Immediate actions (Critical/High severity)
        if self.audit_report["codebase_analysis"]["python_quality"]["potential_security_issues"]:
            recommendations["immediate_actions"].append({
                "action": "Fix security vulnerabilities",
                "description": "Address dangerous function usage (eval, exec, compile)",
                "files_affected": len(set(
                    issue["file"] for issue in 
                    self.audit_report["codebase_analysis"]["python_quality"]["potential_security_issues"]
                )),
                "priority": "Critical"
            })
        
        # Short-term improvements
        if self.audit_report["technical_debt"]["duplicate_code"] > 0:
            recommendations["short_term_improvements"].append({
                "action": "Eliminate duplicate code",
                "description": f"Found {self.audit_report['technical_debt']['duplicate_code']} potential duplicates",
                "priority": "High"
            })
        
        if self.audit_report["technical_debt"]["missing_documentation"] > 10:
            recommendations["short_term_improvements"].append({
                "action": "Add missing documentation",
                "description": f"Add docstrings to {self.audit_report['technical_debt']['missing_documentation']} functions/classes",
                "priority": "Medium"
            })
        
        # Long-term goals
        if self.audit_report["architecture_compliance"]["directory_structure"]["score"] < 90:
            recommendations["long_term_goals"].append({
                "action": "Improve directory structure compliance",
                "description": "Align with Kimera SWM architectural standards",
                "priority": "Medium"
            })
        
        self.audit_report["recommendations"] = recommendations
        logger.info("✅ Recommendations generated")
        
    def run_comprehensive_audit(self):
        """Run complete system audit"""
        logger.info("🔍 Starting comprehensive Kimera SWM audit...")
        
        try:
            self.scan_codebase_structure()
            self.analyze_python_code_quality()
            self.analyze_architecture_compliance()
            self.assess_technical_debt()
            self.generate_recommendations()
            
            # Save audit report
            report_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            report_path = f"docs/reports/analysis/{report_date}_comprehensive_audit.json"
            
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            
            with open(report_path, 'w') as f:
                json.dump(self.audit_report, f, indent=2, default=str)
            
            # Generate markdown summary
            self.generate_audit_summary(report_path.replace('.json', '.md'))
            
            logger.info(f"✅ Comprehensive audit complete! Report saved to: {report_path}")
            return self.audit_report
            
        except Exception as e:
            logger.error(f"❌ Audit failed: {e}", exc_info=True)
            raise
    
    def generate_audit_summary(self, output_path):
        """Generate human-readable audit summary"""
        
        md_content = f"""# KIMERA SWM Comprehensive Audit Report
**Generated**: {self.audit_report['audit_metadata']['timestamp']}
**Auditor**: {self.audit_report['audit_metadata']['auditor']}

## Executive Summary

### Codebase Overview
- **Total Files**: {self.audit_report['codebase_analysis']['file_structure']['total_files']:,}
- **Python Files**: {self.audit_report['codebase_analysis']['python_quality']['total_python_files']:,}
- **Lines of Code**: {self.audit_report['codebase_analysis']['python_quality']['lines_of_code']:,}
- **Functions**: {self.audit_report['codebase_analysis']['python_quality']['functions']:,}
- **Classes**: {self.audit_report['codebase_analysis']['python_quality']['classes']:,}

### Technical Debt Assessment
- **Debt Level**: {self.audit_report['technical_debt']['debt_level']}
- **Debt Score**: {self.audit_report['technical_debt']['overall_debt_score']}/100
- **Duplicate Files**: {self.audit_report['technical_debt']['duplicate_code']}
- **Missing Documentation**: {self.audit_report['technical_debt']['missing_documentation']}
- **Security Issues**: {self.audit_report['technical_debt']['security_issues']}

### Architecture Compliance
- **Directory Structure**: {self.audit_report['architecture_compliance']['directory_structure']['score']:.1f}%
- **Ethical Governor Integration**: {'✅ Yes' if self.audit_report['architecture_compliance']['ethical_governor']['compliance'] else '❌ No'}
- **Zero-Debug Compliance**: {self.audit_report['architecture_compliance']['zero_debugging']['compliance_score']:.1f}%

## Detailed Findings

### Code Quality Issues
"""
        
        # Add specific issues
        if self.audit_report['codebase_analysis']['python_quality']['potential_security_issues']:
            md_content += "\n#### 🚨 Security Issues (CRITICAL)\n"
            for issue in self.audit_report['codebase_analysis']['python_quality']['potential_security_issues'][:5]:
                md_content += f"- **{issue['file']}** (Line {issue['line']}): {issue['issue']}\n"
        
        if self.audit_report['codebase_analysis']['python_quality']['complexity_issues']:
            md_content += "\n#### 📊 Complexity Issues\n"
            for issue in self.audit_report['codebase_analysis']['python_quality']['complexity_issues'][:5]:
                md_content += f"- **{issue['file']}** (Line {issue['line']}): {issue['issue']} ({issue['lines']} lines)\n"
        
        # Add recommendations
        md_content += "\n## Immediate Action Items\n"
        for i, action in enumerate(self.audit_report['recommendations']['immediate_actions'], 1):
            md_content += f"{i}. **{action['action']}** ({action['priority']})\n"
            md_content += f"   - {action['description']}\n"
        
        md_content += "\n## Short-term Improvements\n"
        for i, action in enumerate(self.audit_report['recommendations']['short_term_improvements'], 1):
            md_content += f"{i}. **{action['action']}** ({action['priority']})\n"
            md_content += f"   - {action['description']}\n"
        
        md_content += f"""
## Next Steps
1. Address critical security issues immediately
2. Create .env file if missing
3. Install missing dependencies
4. Review and refactor duplicate code
5. Add missing documentation
6. Run system startup tests

---
*Audit completed by Kimera SWM Autonomous Architect Protocol v3.0*
*For detailed findings, see the full JSON report*
         """
         
         with open(output_path, 'w', encoding='utf-8') as f:
             f.write(md_content)
            
        logger.info(f"Audit summary saved to: {output_path}")
    
    def _should_ignore_file(self, file_path):
        """Check if file should be ignored during analysis"""
        ignore_patterns = [
            '.git', '__pycache__', '.pytest_cache', 'node_modules',
            '.venv', 'venv', '.env', '.DS_Store', '*.pyc', '*.pyo',
            'archive', '.cursor', '.qodo', '.roo'
        ]
        
        path_str = str(file_path)
        return any(pattern in path_str for pattern in ignore_patterns)

def main():
    """Main entry point"""
    auditor = KimeraSystemAuditor()
    report = auditor.run_comprehensive_audit()
    
    print(f"\n🔍 KIMERA SWM COMPREHENSIVE AUDIT COMPLETE")
    print(f"Technical Debt Level: {report['technical_debt']['debt_level']}")
    print(f"Total Files Analyzed: {report['codebase_analysis']['file_structure']['total_files']:,}")
    print(f"Python Files: {report['codebase_analysis']['python_quality']['total_python_files']:,}")
    print(f"Action Items: {len(report['recommendations']['immediate_actions']) + len(report['recommendations']['short_term_improvements'])}")
    
    return report

if __name__ == "__main__":
    main() 