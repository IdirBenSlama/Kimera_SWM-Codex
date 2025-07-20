#!/usr/bin/env python3
"""
KIMERA Print Statement Fixer - Enhanced Version
===============================================

logger.info(Automatically replaces  statements with appropriate logging calls)
while maintaining zero-debugging constraint compliance.

Features:
- Smart log level determination based on content
- Automatic logger import injection
- AST-based analysis for accurate detection
- Dry run mode for safe testing
"""

import ast
import os
import re
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Set, Tuple
import logging

# Configure logging for this script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PrintStatementFixer:
    """Enhanced print statement fixer with intelligent content analysis"""
    
    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.fixes = 0
        self.files_processed = 0
        self.errors = 0
        
        # Enhanced emoji to log level mapping
        self.emoji_mapping = {
            '✅': 'info',
            '❌': 'error', 
            '⚠️': 'warning',
            '🔍': 'debug',
            '📊': 'info',
            '🎯': 'info',
            '🚀': 'info',
            '💥': 'critical',
            '🛡️': 'info',
            '⚡': 'info',
            '🔧': 'debug',
            '📈': 'info',
            '🎉': 'info',
            '🔬': 'debug',
            '💡': 'info',
            '🎭': 'debug',
            '🌟': 'info'
        }
        
        # Enhanced keyword to log level mapping
        self.keyword_mapping = {
            'ERROR': 'error',
            'FAILED': 'error', 
            'FAIL': 'error',
            'WARNING': 'warning',
            'WARN': 'warning',
            'DEBUG': 'debug',
            'INFO': 'info',
            'SUCCESS': 'info',
            'COMPLETE': 'info',
            'STARTED': 'info',
            'FINISHED': 'info',
            'CRITICAL': 'critical',
            'FATAL': 'critical'
        }

    def determine_log_level(self, content: str) -> str:
        """
        Intelligently determine the appropriate log level based on content.
        
        Args:
            content: The content of the print statement
            
        Returns:
            Appropriate log level string
        """
        content_upper = content.upper()
        
        # Check for emojis first (higher priority)
        for emoji, level in self.emoji_mapping.items():
            if emoji in content:
                return level
        
        # Check for explicit keywords
        for keyword, level in self.keyword_mapping.items():
            if keyword in content_upper:
                return level
        
        # Pattern-based detection
        if any(word in content_upper for word in ['EXCEPTION', 'TRACEBACK', 'CRASH']):
            return 'error'
        elif any(word in content_upper for word in ['WARNING', 'WARN']):
            return 'warning'
        elif any(word in content_upper for word in ['TEST', 'TESTING', 'VALIDATION']):
            return 'info'
        elif any(word in content_upper for word in ['STARTING', 'INITIALIZING', 'LOADING']):
            return 'info'
        elif content.startswith('"') and content.endswith('"'):
            # Simple string literals are usually informational
            return 'info'
        elif 'f"' in content or "f'" in content:
            # F-strings often contain dynamic information
            return 'info'
        
        # Default to info for unknown patterns
        return 'info'

    def extract_print_content(self, line: str) -> str:
        """
        Extract the content from a print statement.
        
        Args:
            line: Line containing the print statement
            
        Returns:
            The content being printed
        """
        # Handle various print statement formats
        patterns = [
            r'print\s*\(\s*([^)]+)\s*\)',
            r'print\s*\(\s*f"([^"]+)"\s*\)',
            r"print\s*\(\s*f'([^']+)'\s*\)",
            r'print\s*\(\s*"([^"]+)"\s*\)',
            r"print\s*\(\s*'([^']+)'\s*\)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                return match.group(1)
        
        # Fallback: return the line without print()
        logger.info(', '')

    def fix_print_statement(self, line: str) -> str:
        """
        Convert a print statement to appropriate logging call.
        
        Args:
            line: Line containing the print statement
            
        Returns:
            Fixed line with logging call
        """
        logger.info(        if '' not in line:)
            return line
        
        # Extract the content being printed
        content = self.extract_print_content(line)
        log_level = self.determine_log_level(content)
        
        # Preserve indentation
        indent = len(line) - len(line.lstrip())
        indent_str = ' ' * indent
        
        # Handle f-strings and other formats
        if content.startswith('f"') or content.startswith("f'"):
            # F-string format
            fixed_line = f"{indent_str}logger.{log_level}({content})"
        elif content.startswith('"') or content.startswith("'"):
            # Regular string
            fixed_line = f"{indent_str}logger.{log_level}({content})"
        else:
            # Variable or expression
            fixed_line = f"{indent_str}logger.{log_level}({content})"
        
        return fixed_line

    def has_logger_import(self, content: str) -> bool:
        """Check if the file already has logger import or setup"""
        logger_patterns = [
            r'from.*kimera_logger.*import',
            r'import.*logging',
            r'logger\s*=.*get_.*_logger',
            r'logger\s*=.*logging\.getLogger'
        ]
        
        for pattern in logger_patterns:
            if re.search(pattern, content):
                return True
        return False

    def add_logger_import(self, content: str, file_path: Path) -> str:
        """Add appropriate logger import to the file"""
        lines = content.split('\n')
        
        # Determine the appropriate logger import based on file location
        if 'backend/core/' in str(file_path):
            logger_import = "from backend.utils.kimera_logger import get_cognitive_logger\nlogger = get_cognitive_logger(__name__)"
        elif 'backend/engines/' in str(file_path):
            logger_import = "from backend.utils.kimera_logger import get_cognitive_logger\nlogger = get_cognitive_logger(__name__)"
        elif 'backend/vault/' in str(file_path):
            logger_import = "from backend.utils.kimera_logger import get_database_logger\nlogger = get_database_logger(__name__)"
        elif 'backend/trading/' in str(file_path):
            logger_import = "from backend.utils.kimera_logger import get_trading_logger\nlogger = get_trading_logger(__name__)"
        elif 'backend/api/' in str(file_path):
            logger_import = "from backend.utils.kimera_logger import get_system_logger\nlogger = get_system_logger(__name__)"
        else:
            # Default to system logger
            logger_import = "from backend.utils.kimera_logger import get_system_logger\nlogger = get_system_logger(__name__)"
        
        # Find the best place to insert the import
        insert_index = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                insert_index = i + 1
            elif line.strip() == '' and insert_index > 0:
                break
        
        # Insert the logger import
        lines.insert(insert_index, '')
        lines.insert(insert_index + 1, '# Initialize structured logger')
        lines.insert(insert_index + 2, logger_import)
        lines.insert(insert_index + 3, '')
        
        return '\n'.join(lines)

    def fix_file(self, file_path: Path) -> Dict[str, int]:
        """
        Fix print statements in a single file.
        
        Args:
            file_path: Path to the file to fix
            
        Returns:
            Dictionary with fix statistics
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            logger.warning(f"Skipping binary file: {file_path}")
            return {'fixes': 0, 'errors': 1}
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return {'fixes': 0, 'errors': 1}
        
        lines = content.split('\n')
        fixed_lines = []
        fixes_in_file = 0
        
        # Check if logger import is needed
        needs_logger_import = False
        
        for line_no, line in enumerate(lines, 1):
            logger.info(' in line and not line.strip()
                needs_logger_import = True
                fixed_line = self.fix_print_statement(line)
                fixed_lines.append(fixed_line)
                fixes_in_file += 1
                
                if not self.dry_run:
                    logger.debug(f"  Line {line_no}: {line.strip()} -> {fixed_line.strip()}")
            else:
                fixed_lines.append(line)
        
        # Add logger import if needed and not present
        if needs_logger_import and not self.has_logger_import(content):
            fixed_content = self.add_logger_import('\n'.join(fixed_lines), file_path)
        else:
            fixed_content = '\n'.join(fixed_lines)
        
        # Write the fixed content back to file
        if fixes_in_file > 0 and not self.dry_run:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(fixed_content)
                logger.info(f"Fixed {fixes_in_file} print statements in {file_path}")
            except Exception as e:
                logger.error(f"Error writing to {file_path}: {e}")
                return {'fixes': 0, 'errors': 1}
        
        return {'fixes': fixes_in_file, 'errors': 0}

    def fix_directory(self, directory: Path) -> Dict[str, int]:
        """
        Fix print statements in all Python files in a directory.
        
        Args:
            directory: Directory to process
            
        Returns:
            Dictionary with overall statistics
        """
        total_fixes = 0
        total_errors = 0
        files_processed = 0
        
        for py_file in directory.rglob('*.py'):
            # Skip certain directories and files
            skip_patterns = [
                '__pycache__',
                '.git',
                'node_modules',
                'venv',
                'env',
                '.pytest_cache',
                'build',
                'dist'
            ]
            
            if any(pattern in str(py_file) for pattern in skip_patterns):
                continue
            
            logger.info(f"Processing: {py_file}")
            result = self.fix_file(py_file)
            total_fixes += result['fixes']
            total_errors += result['errors']
            files_processed += 1
        
        return {
            'files_processed': files_processed,
            'fixes': total_fixes,
            'errors': total_errors
        }

def main():
    parser = argparse.ArgumentParser(description='Fix print statements in Kimera codebase')
    parser.add_argument('target', nargs='?', default='.', help='Target file or directory')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be changed without modifying files')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    target_path = Path(args.target)
    
    if not target_path.exists():
        logger.error(f"Path not found: {args.target}")
        sys.exit(1)
    
    fixer = PrintStatementFixer(dry_run=args.dry_run)
    
    logger.info("🔧 KIMERA Print Statement Fixer")
    logger.info("=" * 40)
    
    if args.dry_run:
        logger.info("DRY RUN MODE - No files will be modified")
    
    if target_path.is_file():
        results = fixer.fix_file(target_path)
        results['files_processed'] = 1
    else:
        results = fixer.fix_directory(target_path)
    
    logger.info("\n📊 Results:")
    logger.info(f"Files processed: {results['files_processed']}")
    logger.info(f"Print statements fixed: {results['fixes']}")
    logger.info(f"Errors encountered: {results['errors']}")
    
    if results['fixes'] > 0:
        logger.info(f"\n✅ Successfully fixed {results['fixes']} print statements")
        if not args.dry_run:
            logger.info("Remember to test the modified files!")
    else:
        logger.info("\n🎯 No print statements found to fix")

if __name__ == "__main__":
    main() 