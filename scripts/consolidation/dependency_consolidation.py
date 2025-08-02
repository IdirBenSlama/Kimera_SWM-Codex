#!/usr/bin/env python3
"""
KIMERA SWM Dependency Consolidation Script
==========================================

Consolidates all requirements files into a unified, hierarchical dependency management system.
Follows the Kimera SWM Autonomous Architect principles for infrastructure consolidation.

Author: Kimera SWM Autonomous Architect
Date: January 31, 2025
Version: 1.0.0
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import re
from collections import defaultdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DependencyConsolidator:
    """
    Consolidates multiple requirements files into a unified dependency management system.
    """
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.requirements_dir = self.base_path / "requirements"
        self.consolidated_dir = self.base_path / "requirements_consolidated"
        self.dependencies = defaultdict(dict)  # {package_name: {version: source_files}}
        self.conflicts = []
        self.categories = {
            'base': 'Core Python packages and fundamental dependencies',
            'api': 'FastAPI and web server dependencies',
            'data': 'Data processing and analysis libraries',
            'gpu': 'GPU acceleration and CUDA libraries',
            'thermodynamic': 'Scientific computing and physics libraries',
            'quantum': 'Quantum computing frameworks',
            'trading': 'Trading APIs and financial libraries',
            'ml': 'Machine learning and AI libraries',
            'testing': 'Testing frameworks and tools',
            'dev': 'Development tools and utilities',
            'omnidimensional': 'Advanced cognitive features'
        }
    
    def create_directories(self):
        """Create necessary directories for consolidation."""
        os.makedirs(self.consolidated_dir, exist_ok=True)
        logger.info(f"Created consolidated requirements directory: {self.consolidated_dir}")
    
    def parse_requirements_file(self, file_path: Path) -> List[Tuple[str, str]]:
        """
        Parse a requirements file and return list of (package, version) tuples.
        
        Args:
            file_path: Path to requirements file
            
        Returns:
            List of (package_name, version_spec) tuples
        """
        dependencies = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue
                    
                    # Handle pip options (skip them)
                    if line.startswith('-'):
                        continue
                    
                    # Parse package specification
                    # Handle various formats: package==1.0.0, package>=1.0.0, package, etc.
                    match = re.match(r'^([a-zA-Z0-9._-]+)([><=!~].*)?$', line)
                    if match:
                        package_name = match.group(1).lower().replace('_', '-')
                        version_spec = match.group(2) if match.group(2) else ''
                        dependencies.append((package_name, version_spec))
                    else:
                        logger.warning(f"Could not parse line {line_num} in {file_path}: {line}")
        
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
        
        return dependencies
    
    def analyze_dependencies(self):
        """Analyze all requirements files and identify conflicts."""
        logger.info("Analyzing dependencies across all requirements files...")
        
        if not self.requirements_dir.exists():
            logger.error(f"Requirements directory not found: {self.requirements_dir}")
            return
        
        # Process each requirements file
        for req_file in self.requirements_dir.glob("*.txt"):
            logger.info(f"Processing {req_file.name}...")
            dependencies = self.parse_requirements_file(req_file)
            
            for package, version in dependencies:
                if package not in self.dependencies:
                    self.dependencies[package] = {}
                
                if version not in self.dependencies[package]:
                    self.dependencies[package][version] = []
                
                self.dependencies[package][version].append(req_file.name)
        
        # Identify conflicts
        for package, versions in self.dependencies.items():
            if len(versions) > 1:
                self.conflicts.append({
                    'package': package,
                    'versions': versions
                })
        
        logger.info(f"Found {len(self.dependencies)} unique packages")
        logger.info(f"Found {len(self.conflicts)} conflicts")
    
    def resolve_conflicts(self) -> Dict[str, str]:
        """
        Resolve version conflicts by choosing the most restrictive compatible version.
        
        Returns:
            Dictionary mapping package names to resolved versions
        """
        resolved = {}
        
        for package, versions in self.dependencies.items():
            if len(versions) == 1:
                # No conflict, use the single version
                resolved[package] = list(versions.keys())[0]
            else:
                # Conflict resolution logic
                resolved_version = self._resolve_version_conflict(package, versions)
                resolved[package] = resolved_version
                logger.info(f"Resolved conflict for {package}: {resolved_version}")
        
        return resolved
    
    def _resolve_version_conflict(self, package: str, versions: Dict[str, List[str]]) -> str:
        """
        Resolve version conflict for a single package.
        
        Args:
            package: Package name
            versions: Dictionary of versions and their source files
            
        Returns:
            Resolved version specification
        """
        # Simple resolution strategy: prefer exact versions, then most restrictive
        exact_versions = [v for v in versions.keys() if v.startswith('==')]
        if exact_versions:
            # Choose the highest exact version
            return max(exact_versions, key=lambda x: x.split('==')[1])
        
        # If no exact versions, choose the most restrictive
        version_specs = list(versions.keys())
        if any(v.startswith('>=') for v in version_specs):
            # Prefer minimum version requirements
            min_versions = [v for v in version_specs if v.startswith('>=')]
            return max(min_versions, key=lambda x: x.split('>=')[1])
        
        # Default to first version found
        return list(versions.keys())[0]
    
    def generate_consolidated_requirements(self, resolved: Dict[str, str]):
        """Generate consolidated requirements files."""
        logger.info("Generating consolidated requirements files...")
        
        # Create main consolidated requirements file
        main_req_path = self.consolidated_dir / "requirements.txt"
        with open(main_req_path, 'w', encoding='utf-8') as f:
            f.write(f"# KIMERA SWM Consolidated Requirements\n")
            f.write(f"# Generated on: {datetime.now().isoformat()}\n")
            f.write(f"# Total packages: {len(resolved)}\n\n")
            
            for package in sorted(resolved.keys()):
                version = resolved[package]
                f.write(f"{package}{version}\n")
        
        logger.info(f"Main consolidated requirements written to: {main_req_path}")
        
        # Create category-specific files for modular installation
        for category, description in self.categories.items():
            self._generate_category_requirements(category, description, resolved)
        
        # Generate environment-specific files
        self._generate_environment_requirements(resolved)
    
    def _generate_category_requirements(self, category: str, description: str, resolved: Dict[str, str]):
        """Generate requirements file for a specific category."""
        category_file = self.consolidated_dir / f"{category}.txt"
        
        # Map packages to categories based on original files
        category_packages = set()
        original_file = self.requirements_dir / f"{category}.txt"
        
        if original_file.exists():
            deps = self.parse_requirements_file(original_file)
            category_packages.update(pkg for pkg, _ in deps)
        
        if category_packages:
            with open(category_file, 'w', encoding='utf-8') as f:
                f.write(f"# KIMERA SWM {category.title()} Requirements\n")
                f.write(f"# {description}\n")
                f.write(f"# Generated on: {datetime.now().isoformat()}\n\n")
                
                for package in sorted(category_packages):
                    if package in resolved:
                        f.write(f"{package}{resolved[package]}\n")
            
            logger.info(f"Category requirements written to: {category_file}")
    
    def _generate_environment_requirements(self, resolved: Dict[str, str]):
        """Generate environment-specific requirements files."""
        environments = {
            'development': ['dev', 'testing', 'base', 'api'],
            'production': ['base', 'api', 'gpu'],
            'research': ['base', 'thermodynamic', 'quantum', 'ml'],
            'trading': ['base', 'api', 'trading', 'gpu']
        }
        
        for env_name, categories in environments.items():
            env_file = self.consolidated_dir / f"{env_name}.txt"
            env_packages = set()
            
            for category in categories:
                cat_file = self.requirements_dir / f"{category}.txt"
                if cat_file.exists():
                    deps = self.parse_requirements_file(cat_file)
                    env_packages.update(pkg for pkg, _ in deps)
            
            with open(env_file, 'w', encoding='utf-8') as f:
                f.write(f"# KIMERA SWM {env_name.title()} Environment Requirements\n")
                f.write(f"# Generated on: {datetime.now().isoformat()}\n\n")
                
                for package in sorted(env_packages):
                    if package in resolved:
                        f.write(f"{package}{resolved[package]}\n")
            
            logger.info(f"Environment requirements written to: {env_file}")
    
    def generate_conflict_report(self) -> str:
        """Generate detailed conflict resolution report."""
        report_lines = [
            "# KIMERA SWM Dependency Consolidation Report",
            f"**Generated**: {datetime.now().isoformat()}",
            f"**Total Packages**: {len(self.dependencies)}",
            f"**Conflicts Found**: {len(self.conflicts)}",
            "",
            "## Conflict Resolution Summary",
            ""
        ]
        
        if self.conflicts:
            for conflict in self.conflicts:
                package = conflict['package']
                versions = conflict['versions']
                
                report_lines.extend([
                    f"### {package}",
                    "**Conflicting versions:**",
                    ""
                ])
                
                for version, sources in versions.items():
                    report_lines.append(f"- `{version}` from: {', '.join(sources)}")
                
                report_lines.extend(["", "---", ""])
        else:
            report_lines.append("✅ No conflicts found!")
        
        return "\n".join(report_lines)
    
    def save_report(self, report_content: str):
        """Save consolidation report."""
        date_str = datetime.now().strftime('%Y-%m-%d')
        report_path = self.base_path / "docs" / "reports" / "analysis" / f"{date_str}_dependency_consolidation_report.md"
        
        # Ensure directory exists
        os.makedirs(report_path.parent, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Consolidation report saved to: {report_path}")
    
    def run_consolidation(self):
        """Execute the complete dependency consolidation process."""
        logger.info("Starting KIMERA SWM dependency consolidation...")
        
        # Create directories
        self.create_directories()
        
        # Analyze dependencies
        self.analyze_dependencies()
        
        # Resolve conflicts
        resolved = self.resolve_conflicts()
        
        # Generate consolidated files
        self.generate_consolidated_requirements(resolved)
        
        # Generate and save report
        report = self.generate_conflict_report()
        self.save_report(report)
        
        logger.info("✅ Dependency consolidation completed successfully!")
        logger.info(f"📁 Consolidated files location: {self.consolidated_dir}")
        
        return {
            'total_packages': len(self.dependencies),
            'conflicts_found': len(self.conflicts),
            'resolved_packages': len(resolved),
            'consolidation_successful': True
        }


def main():
    """Main execution function."""
    if len(sys.argv) > 1:
        base_path = sys.argv[1]
    else:
        # Default to current directory's Kimera-SWM
        base_path = os.path.join(os.getcwd(), "Kimera-SWM")
    
    if not os.path.exists(base_path):
        logger.error(f"Base path does not exist: {base_path}")
        sys.exit(1)
    
    consolidator = DependencyConsolidator(base_path)
    result = consolidator.run_consolidation()
    
    if result['consolidation_successful']:
        print(f"✅ Successfully consolidated {result['total_packages']} packages")
        print(f"📊 Resolved {result['conflicts_found']} conflicts")
        print(f"📁 Files generated in: {consolidator.consolidated_dir}")
    else:
        print("❌ Consolidation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()