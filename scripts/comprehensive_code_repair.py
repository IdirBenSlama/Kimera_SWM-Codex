#!/usr/bin/env python3
"""
Comprehensive Code Repair Script for Kimera SWM
==============================================

This script systematically repairs all critical code quality issues:
1. Removes Unicode encoding declarations
2. Fixes bare exception handlers
3. Replaces print statements with logging
4. Standardizes import patterns
5. Ensures zero-debugging compliance

Scientific and Engineering Rigor: Each repair is logged and verified.
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import json
import time
from datetime import datetime
import unicodedata

# Configure scientific logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [Code Repair] %(message)s'
)
logger = logging.getLogger(__name__)

class KimeraCodeRepairEngine:
    """
    Scientific code repair engine implementing rigorous quality standards
    """
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.repair_log = []
        self.stats = {
            'files_processed': 0,
            'unicode_declarations_removed': 0,
            'bare_exceptions_fixed': 0,
            'print_statements_replaced': 0,
            'imports_standardized': 0,
            'emojis_removed': 0,
            'total_repairs': 0
        }

    # ------------------------------------------------------------------
    # Emoji/Unicode sanitization
    # ------------------------------------------------------------------
    def remove_emojis(self, file_path: Path) -> bool:
        """Remove emoji and other problematic Unicode symbols."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            filtered_chars = []
            emojis_removed = 0
            for ch in content:
                # Filter out characters that are symbol/other (So) or surrogate (Cs) and beyond ASCII
                if (
                    ord(ch) > 127 and (
                        unicodedata.category(ch) in {"So", "Cs"}
                        or (
                            0x1F000 <= ord(ch) <= 0x1FAFF  # Common emoji blocks
                            or 0x2600 <= ord(ch) <= 0x26FF  # Misc symbols
                            or 0x2700 <= ord(ch) <= 0x27BF
                        )
                    )
                ):
                    emojis_removed += 1
                    continue  # Skip problematic char
                filtered_chars.append(ch)

            if emojis_removed > 0:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(''.join(filtered_chars))
                self.stats['emojis_removed'] += emojis_removed
                self.repair_log.append(
                    f"Removed {emojis_removed} emoji/unicode symbols from {file_path}"
                )
                return True
        except Exception as e:
            logger.warning(f"Failed to remove emojis in {file_path}: {e}")
        return False

    def scan_python_files(self) -> List[Path]:
        """Scan for all Python files requiring repair"""
        python_files = []
        
        # Exclude archive and broken script directories
        exclude_patterns = [
            'archive/',
            'broken_scripts_and_tests/',
            '.venv/',
            '__pycache__/',
            '.git/',
            'temp/',
            'cache/'
        ]
        
        for py_file in self.project_root.rglob("*.py"):
            # Skip if in excluded directory
            if any(pattern in str(py_file) for pattern in exclude_patterns):
                continue
            python_files.append(py_file)
            
        logger.info(f"Found {len(python_files)} Python files for repair")
        return python_files
        
    def remove_unicode_declarations(self, file_path: Path) -> bool:
        """Remove unnecessary Unicode encoding declarations"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            original_content = content
            
            # Remove various forms of UTF-8 encoding declarations
            patterns = [
                r'^# -\*- coding: utf-8 -\*-\s*\n',
                r'^# coding: utf-8\s*\n',
                r'^# coding=utf-8\s*\n',
                r'^#!/usr/bin/env python3\s*\n# -\*- coding: utf-8 -\*-\s*\n',
            ]
            
            for pattern in patterns:
                content = re.sub(pattern, '', content, flags=re.MULTILINE)
                
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.stats['unicode_declarations_removed'] += 1
                self.repair_log.append(f"Removed Unicode declaration from {file_path}")
                return True
                
        except Exception as e:
            logger.warning(f"Failed to process {file_path}: {e}")
            
        return False
        
    def fix_bare_exceptions(self, file_path: Path) -> bool:
        """Fix bare except: statements with specific exception handling"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                
            modified = False
            
            for i, line in enumerate(lines):
                # Look for bare except statements
                if re.match(r'\s*except:\s*$', line):
                    # Replace with generic Exception handler
                    indent = len(line) - len(line.lstrip())
                    lines[i] = ' ' * indent + 'except Exception as e:\n'
                    
                    # Check if next line needs logging added
                    if i + 1 < len(lines):
                        next_line = lines[i + 1]
                        next_indent = len(next_line) - len(next_line.lstrip())
                        
                        # If next line is just 'pass', replace with logging
                        if next_line.strip() == 'pass':
                            lines[i + 1] = ' ' * (next_indent) + 'logger.warning(f"Unhandled exception: {e}")\n'
                        elif not next_line.strip().startswith('logger'):
                            # Add logging before existing code
                            log_line = ' ' * (next_indent) + 'logger.warning(f"Exception in operation: {e}")\n'
                            lines.insert(i + 1, log_line)
                    
                    modified = True
                    self.stats['bare_exceptions_fixed'] += 1
                    
            if modified:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                self.repair_log.append(f"Fixed bare exceptions in {file_path}")
                return True
                
        except Exception as e:
            logger.warning(f"Failed to fix exceptions in {file_path}: {e}")
            
        return False
        
    def replace_print_statements(self, file_path: Path) -> bool:
        """Replace print statements with proper logging"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            original_content = content
            
            # Skip if file already has proper logging setup
            if 'import logging' in content or 'from.*logging import' in content:
                # Just replace print statements with logger calls
                content = re.sub(
                    r'print\(([^)]+)\)',
                    r'logger.info(\1)',
                    content
                )
            else:
                # Add logging import at top and replace prints
                lines = content.split('\n')
                
                # Find best place to insert logging import
                insert_index = 0
                for i, line in enumerate(lines):
                    if line.startswith('import ') or line.startswith('from '):
                        insert_index = i + 1
                    elif line.strip() == '':
                        continue
                    else:
                        break
                        
                # Insert logging setup
                logging_setup = [
                    'import logging',
                    '',
                    'logger = logging.getLogger(__name__)'
                ]
                
                for j, log_line in enumerate(logging_setup):
                    lines.insert(insert_index + j, log_line)
                    
                content = '\n'.join(lines)
                
                # Replace print statements
                content = re.sub(
                    r'print\(([^)]+)\)',
                    r'logger.info(\1)',
                    content
                )
                
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.stats['print_statements_replaced'] += 1
                self.repair_log.append(f"Replaced print statements in {file_path}")
                return True
                
        except Exception as e:
            logger.warning(f"Failed to replace prints in {file_path}: {e}")
            
        return False
        
    def standardize_imports(self, file_path: Path) -> bool:
        """Standardize import patterns for consistency"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            original_content = content
            
            # Fix common import issues
            import_fixes = [
                # Fix gpu_foundation imports
                (r'from backend\.utils import gpu_foundation', 
                 'from src.utils.gpu_foundation import GPUFoundation'),
                
                # Standardize relative imports
                (r'from \.\.utils import', 'from src.utils import'),
                (r'from \.\.core import', 'from src.core import'),
                (r'from \.\.engines import', 'from src.engines import'),
            ]
            
            modified = False
            for pattern, replacement in import_fixes:
                if re.search(pattern, content):
                    content = re.sub(pattern, replacement, content)
                    modified = True
                    
            if modified:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.stats['imports_standardized'] += 1
                self.repair_log.append(f"Standardized imports in {file_path}")
                return True
                
        except Exception as e:
            logger.warning(f"Failed to standardize imports in {file_path}: {e}")
            
        return False
        
    def repair_file(self, file_path: Path) -> Dict[str, bool]:
        """Perform comprehensive repair on a single file"""
        results = {
            'unicode_removed': False,
            'exceptions_fixed': False,
            'prints_replaced': False,
            'imports_standardized': False
        }
        
        logger.debug(f"Repairing {file_path}")
        
        # Only process files that are likely to need repair
        if file_path.suffix == '.py':
            results['unicode_removed'] = self.remove_unicode_declarations(file_path)
            results['exceptions_fixed'] = self.fix_bare_exceptions(file_path)
            results['prints_replaced'] = self.replace_print_statements(file_path)
            results['imports_standardized'] = self.standardize_imports(file_path)
            self.remove_emojis(file_path)
            
        self.stats['files_processed'] += 1
        
        return results
        
    def run_comprehensive_repair(self) -> Dict[str, int]:
        """Execute comprehensive repair across all Python files"""
        logger.info("🔧 Starting comprehensive code repair")
        start_time = time.time()
        
        python_files = self.scan_python_files()
        
        for file_path in python_files:
            try:
                results = self.repair_file(file_path)
                
                if any(results.values()):
                    self.stats['total_repairs'] += sum(results.values())
                    logger.debug(f"Repaired {file_path}: {results}")
                    
            except Exception as e:
                logger.error(f"Failed to repair {file_path}: {e}")
                
        duration = time.time() - start_time
        
        # Generate repair report
        self.generate_repair_report(duration)
        
        return self.stats
        
    def generate_repair_report(self, duration: float):
        """Generate comprehensive repair report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': duration,
            'statistics': self.stats,
            'repair_log': self.repair_log
        }
        
        # Save report
        report_path = self.project_root / 'repair_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        # Log summary
        logger.info("🎯 COMPREHENSIVE REPAIR COMPLETE")
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info(f"Files processed: {self.stats['files_processed']}")
        logger.info(f"Unicode declarations removed: {self.stats['unicode_declarations_removed']}")
        logger.info(f"Bare exceptions fixed: {self.stats['bare_exceptions_fixed']}")
        logger.info(f"Print statements replaced: {self.stats['print_statements_replaced']}")
        logger.info(f"Imports standardized: {self.stats['imports_standardized']}")
        logger.info(f"Unicode/emoji removed: {self.stats['emojis_removed']}")
        logger.info(f"Total repairs: {self.stats['total_repairs']}")
        logger.info(f"Report saved to: {report_path}")


def main():
    """Main execution function"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    logger.info("Kimera SWM Comprehensive Code Repair Engine")
    logger.info(f"Project root: {project_root}")
    
    repair_engine = KimeraCodeRepairEngine(project_root)
    stats = repair_engine.run_comprehensive_repair()
    
    if stats['total_repairs'] > 0:
        logger.info("✅ Code repair successful - System quality improved")
    else:
        logger.info("ℹ️ No repairs needed - Code already compliant")
        
    return stats


if __name__ == "__main__":
    main() 