#!/usr/bin/env python3
"""
KIMERA SWM Core Component Merger
================================

Consolidates duplicate core components from scattered locations into the main core directory.
Handles import updates, dependency resolution, and conflict management.

Author: Kimera SWM Autonomous Architect  
Date: January 31, 2025
Version: 1.0.0
"""

import os
import sys
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import re
import ast

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CoreComponentMerger:
    """
    Merges duplicate core components into a unified core architecture.
    """
    
    def __init__(self, workspace_root: str):
        self.workspace_root = Path(workspace_root)
        self.kimera_root = self.workspace_root / "Kimera-SWM"
        self.main_core_dir = self.kimera_root / "src" / "core"
        self.secondary_core_dir = self.workspace_root / "src" / "core"
        self.backup_dir = self.kimera_root / "archive" / f"2025-07-31_core_merger_backup"
        
        self.conflicts = []
        self.merged_files = []
        self.import_updates = []
        
    def create_backup(self):
        """Create backup of existing core components."""
        logger.info("Creating backup of existing core components...")
        
        os.makedirs(self.backup_dir, exist_ok=True)
        
        # Backup main core directory
        if self.main_core_dir.exists():
            main_backup = self.backup_dir / "main_core"
            shutil.copytree(self.main_core_dir, main_backup, dirs_exist_ok=True)
            logger.info(f"Main core backup created at: {main_backup}")
        
        # Backup secondary core directory  
        if self.secondary_core_dir.exists():
            secondary_backup = self.backup_dir / "secondary_core"
            shutil.copytree(self.secondary_core_dir, secondary_backup, dirs_exist_ok=True)
            logger.info(f"Secondary core backup created at: {secondary_backup}")
        
        # Create backup manifest
        manifest_path = self.backup_dir / "BACKUP_MANIFEST.md"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            f.write(f"# Core Component Merger Backup\n")
            f.write(f"**Created**: {datetime.now().isoformat()}\n")
            f.write(f"**Purpose**: Backup before core component consolidation\n\n")
            f.write(f"## Backup Contents\n")
            f.write(f"- `main_core/` - Original `/Kimera-SWM/src/core/` contents\n")
            f.write(f"- `secondary_core/` - Original `/src/core/` contents\n\n")
            f.write(f"## Restoration\n")
            f.write(f"To restore, copy contents back to original locations.\n")
    
    def analyze_components(self) -> Dict[str, Dict]:
        """Analyze components in both core directories."""
        logger.info("Analyzing core components...")
        
        analysis = {
            'main_core': {},
            'secondary_core': {},
            'conflicts': [],
            'unique_to_main': [],
            'unique_to_secondary': []
        }
        
        # Analyze main core directory
        if self.main_core_dir.exists():
            analysis['main_core'] = self._analyze_directory(self.main_core_dir)
        
        # Analyze secondary core directory
        if self.secondary_core_dir.exists():
            analysis['secondary_core'] = self._analyze_directory(self.secondary_core_dir)
        
        # Find conflicts and unique files
        main_files = set(analysis['main_core'].keys())
        secondary_files = set(analysis['secondary_core'].keys())
        
        conflicts = main_files.intersection(secondary_files)
        analysis['conflicts'] = list(conflicts)
        analysis['unique_to_main'] = list(main_files - secondary_files)
        analysis['unique_to_secondary'] = list(secondary_files - main_files)
        
        logger.info(f"Found {len(main_files)} files in main core")
        logger.info(f"Found {len(secondary_files)} files in secondary core") 
        logger.info(f"Found {len(conflicts)} conflicting files")
        logger.info(f"Found {len(analysis['unique_to_secondary'])} unique files to merge")
        
        return analysis
    
    def _analyze_directory(self, directory: Path) -> Dict[str, Dict]:
        """Analyze Python files in a directory."""
        files = {}
        
        for file_path in directory.rglob("*.py"):
            if file_path.name == "__init__.py":
                continue
                
            relative_path = file_path.relative_to(directory)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                files[str(relative_path)] = {
                    'path': file_path,
                    'size': file_path.stat().st_size,
                    'lines': len(content.splitlines()),
                    'imports': self._extract_imports(content),
                    'classes': self._extract_classes(content),
                    'functions': self._extract_functions(content)
                }
            except Exception as e:
                logger.warning(f"Could not analyze {file_path}: {e}")
        
        return files
    
    def _extract_imports(self, content: str) -> List[str]:
        """Extract import statements from Python code."""
        imports = []
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(f"import {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        imports.append(f"from {module} import {alias.name}")
        except Exception as e:
            logger.error(f"Error in core_component_merger.py: {e}", exc_info=True)
            raise  # Re-raise for proper error handling
            # Fallback to regex if AST parsing fails
            import_pattern = r'^(?:from\s+[\w.]+\s+)?import\s+[\w., ]+$'
            for line in content.splitlines():
                if re.match(import_pattern, line.strip()):
                    imports.append(line.strip())
        
        return imports
    
    def _extract_classes(self, content: str) -> List[str]:
        """Extract class names from Python code."""
        classes = []
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    classes.append(node.name)
        except Exception as e:
            logger.error(f"Error in core_component_merger.py: {e}", exc_info=True)
            raise  # Re-raise for proper error handling
            # Fallback to regex
            class_pattern = r'^class\s+(\w+)'
            for line in content.splitlines():
                match = re.match(class_pattern, line.strip())
                if match:
                    classes.append(match.group(1))
        
        return classes
    
    def _extract_functions(self, content: str) -> List[str]:
        """Extract function names from Python code."""
        functions = []
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
        except Exception as e:
            logger.error(f"Error in core_component_merger.py: {e}", exc_info=True)
            raise  # Re-raise for proper error handling
            # Fallback to regex
            func_pattern = r'^def\s+(\w+)'
            for line in content.splitlines():
                match = re.match(func_pattern, line.strip())
                if match:
                    functions.append(match.group(1))
        
        return functions
    
    def resolve_conflicts(self, analysis: Dict) -> Dict[str, str]:
        """Resolve conflicts between duplicate files."""
        logger.info("Resolving conflicts...")
        
        resolutions = {}
        
        for conflict_file in analysis['conflicts']:
            main_info = analysis['main_core'][conflict_file]
            secondary_info = analysis['secondary_core'][conflict_file]
            
            # Resolution strategy: prefer newer, larger, more feature-rich files
            if secondary_info['size'] > main_info['size'] * 1.1:  # 10% larger
                resolutions[conflict_file] = 'use_secondary'
                logger.info(f"Conflict resolution for {conflict_file}: Using secondary (larger)")
            elif len(secondary_info['classes']) > len(main_info['classes']):
                resolutions[conflict_file] = 'use_secondary'
                logger.info(f"Conflict resolution for {conflict_file}: Using secondary (more classes)")
            elif len(secondary_info['functions']) > len(main_info['functions']):
                resolutions[conflict_file] = 'use_secondary'
                logger.info(f"Conflict resolution for {conflict_file}: Using secondary (more functions)")
            else:
                resolutions[conflict_file] = 'use_main'
                logger.info(f"Conflict resolution for {conflict_file}: Using main (default)")
                
            self.conflicts.append({
                'file': conflict_file,
                'resolution': resolutions[conflict_file],
                'main_size': main_info['size'],
                'secondary_size': secondary_info['size'],
                'main_classes': len(main_info['classes']),
                'secondary_classes': len(secondary_info['classes'])
            })
        
        return resolutions
    
    def merge_components(self, analysis: Dict, resolutions: Dict):
        """Merge components into the main core directory."""
        logger.info("Merging core components...")
        
        # Ensure main core directory exists
        os.makedirs(self.main_core_dir, exist_ok=True)
        
        # Process unique files from secondary core
        for file_rel_path in analysis['unique_to_secondary']:
            source_file = analysis['secondary_core'][file_rel_path]['path']
            target_file = self.main_core_dir / file_rel_path
            
            # Ensure target directory exists
            os.makedirs(target_file.parent, exist_ok=True)
            
            # Copy file
            shutil.copy2(source_file, target_file)
            self.merged_files.append({
                'file': file_rel_path,
                'action': 'added',
                'source': str(source_file),
                'target': str(target_file)
            })
            logger.info(f"Added: {file_rel_path}")
        
        # Process conflicts based on resolutions
        for conflict_file, resolution in resolutions.items():
            if resolution == 'use_secondary':
                source_file = analysis['secondary_core'][conflict_file]['path']
                target_file = self.main_core_dir / conflict_file
                
                # Backup existing main file
                backup_file = target_file.with_suffix(f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py")
                if target_file.exists():
                    shutil.copy2(target_file, backup_file)
                
                # Copy secondary file over main
                shutil.copy2(source_file, target_file)
                self.merged_files.append({
                    'file': conflict_file,
                    'action': 'replaced',
                    'source': str(source_file),
                    'target': str(target_file),
                    'backup': str(backup_file)
                })
                logger.info(f"Replaced: {conflict_file}")
            else:
                logger.info(f"Kept main version: {conflict_file}")
    
    def update_imports(self):
        """Update import statements to point to consolidated core."""
        logger.info("Updating import statements...")
        
        # Pattern to find imports from the old secondary core
        old_import_patterns = [
            r'from src\.core\.',
            r'import src\.core\.',
            r'from \.\.\.src\.core\.',
            r'from \.\.\.\.src\.core\.'
        ]
        
        # Find all Python files that might have imports to update
        for py_file in self.kimera_root.rglob("*.py"):
            if py_file.is_relative_to(self.backup_dir):
                continue  # Skip backup files
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                # Update import patterns
                for pattern in old_import_patterns:
                    # Replace with proper relative imports or absolute imports
                    content = re.sub(
                        pattern,
                        'from src.core.',
                        content
                    )
                
                # If content changed, write it back
                if content != original_content:
                    with open(py_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    self.import_updates.append({
                        'file': str(py_file.relative_to(self.kimera_root)),
                        'changes': 'Updated core imports'
                    })
                    logger.info(f"Updated imports in: {py_file.relative_to(self.kimera_root)}")
            
            except Exception as e:
                logger.warning(f"Could not update imports in {py_file}: {e}")
    
    def generate_merge_report(self) -> str:
        """Generate detailed merge report."""
        report_lines = [
            "# KIMERA SWM Core Component Merger Report",
            f"**Generated**: {datetime.now().isoformat()}",
            f"**Merged Files**: {len(self.merged_files)}",
            f"**Conflicts Resolved**: {len(self.conflicts)}",
            f"**Import Updates**: {len(self.import_updates)}",
            "",
            "## Merge Summary",
            ""
        ]
        
        if self.merged_files:
            report_lines.extend([
                "### Merged Files",
                ""
            ])
            
            for merge_info in self.merged_files:
                action = merge_info['action']
                file_path = merge_info['file']
                report_lines.append(f"- **{action.title()}**: `{file_path}`")
                if 'backup' in merge_info:
                    report_lines.append(f"  - Backup: `{merge_info['backup']}`")
            
            report_lines.extend(["", "---", ""])
        
        if self.conflicts:
            report_lines.extend([
                "### Conflict Resolutions",
                ""
            ])
            
            for conflict in self.conflicts:
                file_path = conflict['file']
                resolution = conflict['resolution']
                report_lines.extend([
                    f"### {file_path}",
                    f"**Resolution**: {resolution}",
                    f"**Main size**: {conflict['main_size']} bytes ({conflict['main_classes']} classes)",
                    f"**Secondary size**: {conflict['secondary_size']} bytes ({conflict['secondary_classes']} classes)",
                    ""
                ])
            
            report_lines.extend(["---", ""])
        
        if self.import_updates:
            report_lines.extend([
                "### Import Updates",
                ""
            ])
            
            for update in self.import_updates:
                report_lines.append(f"- `{update['file']}`: {update['changes']}")
            
            report_lines.extend(["", "---", ""])
        
        report_lines.extend([
            "## Post-Merge Actions Required",
            "",
            "1. **Test all imports**: Verify no broken imports remain",
            "2. **Run tests**: Execute test suite to verify functionality",
            "3. **Update documentation**: Reflect new core structure",
            "4. **Clean up**: Remove old secondary core directory after verification",
            "",
            "## Backup Location",
            f"Complete backup available at: `{self.backup_dir.relative_to(self.kimera_root)}`"
        ])
        
        return "\n".join(report_lines)
    
    def save_report(self, report_content: str):
        """Save merge report."""
        date_str = datetime.now().strftime('%Y-%m-%d')
        report_path = self.kimera_root / "docs" / "reports" / "analysis" / f"{date_str}_core_component_merger_report.md"
        
        # Ensure directory exists
        os.makedirs(report_path.parent, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Merge report saved to: {report_path}")
    
    def run_merger(self):
        """Execute the complete core component merger process."""
        logger.info("Starting KIMERA SWM core component merger...")
        
        # Create backup
        self.create_backup()
        
        # Analyze components
        analysis = self.analyze_components()
        
        # Resolve conflicts
        resolutions = self.resolve_conflicts(analysis)
        
        # Merge components
        self.merge_components(analysis, resolutions)
        
        # Update imports
        self.update_imports()
        
        # Generate and save report
        report = self.generate_merge_report()
        self.save_report(report)
        
        logger.info("✅ Core component merger completed successfully!")
        logger.info(f"📁 Backup location: {self.backup_dir}")
        
        return {
            'merged_files': len(self.merged_files),
            'conflicts_resolved': len(self.conflicts),
            'imports_updated': len(self.import_updates),
            'merger_successful': True
        }


def main():
    """Main execution function."""
    if len(sys.argv) > 1:
        workspace_root = sys.argv[1]
    else:
        # Default to current workspace root
        workspace_root = os.getcwd()
    
    if not os.path.exists(workspace_root):
        logger.error(f"Workspace root does not exist: {workspace_root}")
        sys.exit(1)
    
    merger = CoreComponentMerger(workspace_root)
    result = merger.run_merger()
    
    if result['merger_successful']:
        print(f"✅ Successfully merged {result['merged_files']} files")
        print(f"📊 Resolved {result['conflicts_resolved']} conflicts")
        print(f"🔄 Updated {result['imports_updated']} import statements")
        print(f"📁 Backup available at: {merger.backup_dir}")
    else:
        print("❌ Merger failed")
        sys.exit(1)


if __name__ == "__main__":
    main()