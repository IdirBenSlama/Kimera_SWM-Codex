#!/usr/bin/env python3
"""
KIMERA SWM Documentation Import Path Migration Script
Updates backend.* imports to src.* imports in markdown documentation files.
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_markdown_files(directory: str) -> List[Path]:
    """Find all Markdown files in the given directory and subdirectories."""
    markdown_files = []
    for root, dirs, files in os.walk(directory):
        # Skip archive directories
        if 'archive' in root:
            continue
        for file in files:
            if file.endswith('.md'):
                markdown_files.append(Path(root) / file)
    return markdown_files

def update_imports_in_markdown(file_path: Path) -> Tuple[bool, int]:
    """
    Update backend imports to src imports in a markdown file.
    Returns (was_modified, num_changes)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        changes = 0
        
        # Pattern 1: from backend.module import ...
        pattern1 = r'from backend\.'
        replacement1 = 'from src.'
        content, count1 = re.subn(pattern1, replacement1, content)
        changes += count1
        
        # Pattern 2: import backend.module
        pattern2 = r'import backend\.'
        replacement2 = 'import src.'
        content, count2 = re.subn(pattern2, replacement2, content)
        changes += count2
        
        # Pattern 3: from backend import (for relative imports)
        pattern3 = r'from backend import'
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
    logger.info("Starting KIMERA SWM documentation import path migration...")
    
    # Directory to process
    directory = 'docs'
    
    if not os.path.exists(directory):
        logger.warning(f"Directory {directory} does not exist, skipping...")
        return
        
    logger.info(f"Processing directory: {directory}")
    markdown_files = find_markdown_files(directory)
    
    total_files_processed = 0
    total_files_modified = 0
    total_changes = 0
    
    for file_path in markdown_files:
        total_files_processed += 1
        was_modified, changes = update_imports_in_markdown(file_path)
        
        if was_modified:
            total_files_modified += 1
            total_changes += changes
    
    # Summary report
    logger.info("=" * 60)
    logger.info("KIMERA SWM DOCUMENTATION MIGRATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Markdown files processed: {total_files_processed}")
    logger.info(f"Files modified: {total_files_modified}")
    logger.info(f"Total import changes: {total_changes}")
    
    if total_changes > 0:
        logger.info("✅ Documentation migration successful!")
    else:
        logger.info("ℹ️  No documentation imports needed updating.")

if __name__ == "__main__":
    main() 