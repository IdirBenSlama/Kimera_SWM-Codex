#!/usr/bin/env python3
"""
Kimera SWM Health Check Script
Protocol Version: 3.0
Purpose: Perform comprehensive codebase cartography and health assessment
"""

import os
import sys
import json
import hashlib
import ast
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
import subprocess
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('kimera_health_check')


class CodebaseCartographer:
    """Analyze and map the entire codebase structure"""
    
    def __init__(self, root_path='.'):
        self.root_path = Path(root_path)
        self.files_by_type = defaultdict(list)
        self.file_hashes = {}
        self.duplicates = []
        self.issues = []
        self.metrics = {}
        
    def scan_codebase(self):
        """Perform full recursive scan of the codebase"""
        logger.info("Starting codebase scan...")
        
        total_files = 0
        total_size = 0
        
        for file_path in self.root_path.rglob('*'):
            if file_path.is_file() and not self._should_ignore(file_path):
                total_files += 1
                file_size = file_path.stat().st_size
                total_size += file_size
                
                # Categorize by extension
                ext = file_path.suffix.lower()
                self.files_by_type[ext].append(file_path)
                
                # Calculate file hash
                try:
                    content_hash = self._calculate_file_hash(file_path)
                    if content_hash in self.file_hashes:
                        self.duplicates.append((file_path, self.file_hashes[content_hash]))
                    else:
                        self.file_hashes[content_hash] = file_path
                except Exception as e:
                    logger.warning(f"Error hashing {file_path}: {e}")
        
        self.metrics['total_files'] = total_files
        self.metrics['total_size_mb'] = total_size / (1024 * 1024)
        self.metrics['duplicate_count'] = len(self.duplicates)
        
        logger.info(f"Scan complete: {total_files} files, {len(self.duplicates)} duplicates")
        
    def analyze_python_code(self):
        """Perform AST analysis on Python files"""
        logger.info("Analyzing Python code structure...")
        
        python_files = self.files_by_type.get('.py', [])
        
        undocumented_functions = []
        circular_imports = []
        complex_functions = []
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content, filename=str(file_path))
                
                # Check for documentation
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                        if not ast.get_docstring(node):
                            undocumented_functions.append(f"{file_path}:{node.name}")
                        
                        # Check complexity
                        if isinstance(node, ast.FunctionDef):
                            complexity = self._calculate_complexity(node)
                            if complexity > 10:
                                complex_functions.append({
                                    'file': str(file_path),
                                    'function': node.name,
                                    'complexity': complexity
                                })
                
                # Check imports
                imports = self._extract_imports(tree)
                # TODO: Implement circular import detection
                
            except Exception as e:
                logger.warning(f"Error analyzing {file_path}: {e}")
                self.issues.append(f"Parse error in {file_path}: {e}")
        
        self.metrics['undocumented_functions'] = len(undocumented_functions)
        self.metrics['complex_functions'] = len(complex_functions)
        self.metrics['python_files'] = len(python_files)
        
    def check_dependencies(self):
        """Analyze project dependencies"""
        logger.info("Checking dependencies...")
        
        requirements_files = [
            'requirements.txt',
            'requirements_optimized.txt',
            'pyproject.toml'
        ]
        
        found_deps = set()
        
        for req_file in requirements_files:
            req_path = self.root_path / req_file
            if req_path.exists():
                try:
                    with open(req_path, 'r') as f:
                        content = f.read()
                        # Simple extraction - could be improved
                        for line in content.split('\n'):
                            line = line.strip()
                            if line and not line.startswith('#'):
                                found_deps.add(line.split('==')[0].split('>=')[0])
                except Exception as e:
                    logger.warning(f"Error reading {req_file}: {e}")
        
        self.metrics['total_dependencies'] = len(found_deps)
        
    def check_test_coverage(self):
        """Check test file coverage"""
        logger.info("Checking test coverage...")
        
        source_files = set()
        test_files = set()
        
        # Find all source files
        for file_path in self.files_by_type.get('.py', []):
            if '/test' not in str(file_path) and 'test_' not in file_path.name:
                source_files.add(file_path)
            else:
                test_files.add(file_path)
        
        # Calculate coverage estimate
        self.metrics['source_files'] = len(source_files)
        self.metrics['test_files'] = len(test_files)
        self.metrics['test_coverage_estimate'] = (
            len(test_files) / len(source_files) * 100 if source_files else 0
        )
        
    def generate_report(self):
        """Generate comprehensive health report"""
        logger.info("Generating health report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'protocol_version': '3.0',
            'metrics': self.metrics,
            'file_distribution': {
                ext: len(files) for ext, files in self.files_by_type.items()
            },
            'duplicates': [
                {'file1': str(f1), 'file2': str(f2)} 
                for f1, f2 in self.duplicates[:10]  # Limit to first 10
            ],
            'issues': self.issues[:20],  # Limit to first 20 issues
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _should_ignore(self, path):
        """Check if path should be ignored"""
        ignore_patterns = [
            '.git', '__pycache__', '.pytest_cache', 
            'node_modules', '.venv', 'venv', '.env'
        ]
        path_str = str(path)
        return any(pattern in path_str for pattern in ignore_patterns)
    
    def _calculate_file_hash(self, file_path):
        """Calculate SHA256 hash of file content"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _calculate_complexity(self, node):
        """Calculate cyclomatic complexity of a function"""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
        return complexity
    
    def _extract_imports(self, tree):
        """Extract all imports from an AST"""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module)
        return imports
    
    def _generate_recommendations(self):
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        if self.metrics.get('duplicate_count', 0) > 0:
            recommendations.append(
                f"Found {self.metrics['duplicate_count']} duplicate files. "
                "Consider consolidating or removing duplicates."
            )
        
        if self.metrics.get('undocumented_functions', 0) > 10:
            recommendations.append(
                f"Found {self.metrics['undocumented_functions']} undocumented functions. "
                "Add docstrings for better code maintainability."
            )
        
        if self.metrics.get('test_coverage_estimate', 0) < 50:
            recommendations.append(
                f"Test coverage estimate is {self.metrics['test_coverage_estimate']:.1f}%. "
                "Consider adding more tests."
            )
        
        if self.metrics.get('complex_functions', 0) > 5:
            recommendations.append(
                f"Found {self.metrics['complex_functions']} complex functions. "
                "Consider refactoring for better maintainability."
            )
        
        return recommendations


class EnvironmentChecker:
    """Check the development environment"""
    
    def check_python_version(self):
        """Check Python version"""
        version = sys.version_info
        return {
            'version': f"{version.major}.{version.minor}.{version.micro}",
            'meets_requirement': version >= (3, 9)
        }
    
    def check_gpu_availability(self):
        """Check if GPU is available"""
        try:
            import torch
            return {
                'cuda_available': torch.cuda.is_available(),
                'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            }
        except ImportError:
            return {
                'cuda_available': False,
                'device_count': 0,
                'device_name': None,
                'note': 'PyTorch not installed'
            }
    
    def check_git_status(self):
        """Check git repository status"""
        try:
            # Check if in git repo
            result = subprocess.run(
                ['git', 'status', '--porcelain'],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Get current branch
            branch_result = subprocess.run(
                ['git', 'branch', '--show-current'],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Get last commit
            commit_result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                check=True
            )
            
            return {
                'is_git_repo': True,
                'has_uncommitted_changes': bool(result.stdout.strip()),
                'current_branch': branch_result.stdout.strip(),
                'last_commit': commit_result.stdout.strip()[:8]
            }
        except subprocess.CalledProcessError:
            return {
                'is_git_repo': False
            }


def main():
    """Main health check execution"""
    print("=" * 60)
    print("KIMERA SWM HEALTH CHECK - Protocol v3.0")
    print("=" * 60)
    print()
    
    # Environment checks
    print("Checking environment...")
    env_checker = EnvironmentChecker()
    
    python_info = env_checker.check_python_version()
    print(f"✓ Python version: {python_info['version']}")
    
    gpu_info = env_checker.check_gpu_availability()
    if gpu_info['cuda_available']:
        print(f"✓ GPU available: {gpu_info['device_name']}")
    else:
        print("⚠ GPU not available")
    
    git_info = env_checker.check_git_status()
    if git_info['is_git_repo']:
        print(f"✓ Git repository - Branch: {git_info['current_branch']}")
        if git_info['has_uncommitted_changes']:
            print("⚠ Uncommitted changes detected")
    
    print()
    
    # Codebase analysis
    print("Performing codebase cartography...")
    cartographer = CodebaseCartographer()
    
    cartographer.scan_codebase()
    cartographer.analyze_python_code()
    cartographer.check_dependencies()
    cartographer.check_test_coverage()
    
    # Generate report
    report = cartographer.generate_report()
    
    # Save report
    report_path = Path('health_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print()
    print("SUMMARY")
    print("-" * 40)
    print(f"Total files: {report['metrics']['total_files']}")
    print(f"Total size: {report['metrics']['total_size_mb']:.2f} MB")
    print(f"Python files: {report['metrics']['python_files']}")
    print(f"Duplicate files: {report['metrics']['duplicate_count']}")
    print(f"Test coverage estimate: {report['metrics']['test_coverage_estimate']:.1f}%")
    
    print()
    print("RECOMMENDATIONS")
    print("-" * 40)
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"{i}. {rec}")
    
    print()
    print(f"Full report saved to: {report_path}")
    print()
    print("Health check complete!")
    
    # Return exit code based on critical issues
    critical_issues = len(report['issues'])
    return 0 if critical_issues == 0 else 1


if __name__ == '__main__':
    sys.exit(main()) 