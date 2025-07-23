#!/usr/bin/env python3
"""
KIMERA SWM Import Path Migration Script
Automatically updates all backend.* imports to src.* imports following reorganization.
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_python_files(directory: str) -> List[Path]:
    """Find all Python files in the given directory and subdirectories."""
    python_files = []
    for root, dirs, files in os.walk(directory):
        # Skip archive directories
        if 'archive' in root:
            continue
        for file in files:
            if file.endswith('.py'):
                python_files.append(Path(root) / file)
    return python_files

def update_imports_in_file(file_path: Path) -> Tuple[bool, int]:
    """
    Update backend imports to src imports in a single file.
    Returns (was_modified, num_changes)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        changes = 0
        
        # Pattern 1: from src.module import ...
        pattern1 = r'from backend\.'
        replacement1 = 'from src.'
        content, count1 = re.subn(pattern1, replacement1, content)
        changes += count1
        
        # Pattern 2: import src.module
        pattern2 = r'import backend\.'
        replacement2 = 'import src.'
        content, count2 = re.subn(pattern2, replacement2, content)
        changes += count2
        
        # Pattern 3: from src import (for relative imports)
        pattern3 = r'from src import'
        replacement3 = 'from src import'
        content, count3 = re.subn(pattern3, replacement3, content)
        changes += count3
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Updated {changes} imports in {file_path}")
            return True, changes
        
        return False, 0
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return False, 0

def main():
    """Main migration function."""
    logger.info("Starting KIMERA SWM import path migration...")
    
    # Directories to process
    directories = ['src', 'tests', 'experiments', 'scripts', 'scientific', 'innovations', 'dashboard']
    
    total_files_processed = 0
    total_files_modified = 0
    total_changes = 0
    
    for directory in directories:
        if not os.path.exists(directory):
            logger.warning(f"Directory {directory} does not exist, skipping...")
            continue
            
        logger.info(f"Processing directory: {directory}")
        python_files = find_python_files(directory)
        
        for file_path in python_files:
            total_files_processed += 1
            was_modified, changes = update_imports_in_file(file_path)
            
            if was_modified:
                total_files_modified += 1
                total_changes += changes
    
    # Summary report
    logger.info("=" * 60)
    logger.info("KIMERA SWM IMPORT MIGRATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Files processed: {total_files_processed}")
    logger.info(f"Files modified: {total_files_modified}")
    logger.info(f"Total import changes: {total_changes}")
    
    if total_changes > 0:
        logger.info("✅ Migration successful! All backend imports updated to src imports.")
    else:
        logger.info("ℹ️  No imports needed updating.")

if __name__ == "__main__":
    main() 