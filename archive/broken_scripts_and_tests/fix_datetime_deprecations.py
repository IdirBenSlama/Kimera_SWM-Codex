#!/usr/bin/env python3
"""
Fix deprecated datetime.now(timezone.utc) usage across KIMERA codebase
Replaces with timezone-aware datetime.now(timezone.utc)
"""

import os
import re
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_datetime_in_file(file_path: Path) -> bool:
    """
    Fix datetime.now(timezone.utc) usage in a single file
    
    Returns:
        bool: True if file was modified, False otherwise
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Check if file already imports timezone
        has_timezone_import = 'from datetime import' in content and 'timezone' in content
        
        # Replace datetime.now(timezone.utc) with datetime.now(timezone.utc)
        content = re.sub(r'datetime\.utcnow\(\)', 'datetime.now(timezone.utc)', content)
        
        # Add timezone import if needed and modifications were made
        if content != original_content and not has_timezone_import:
            # Find existing datetime import and add timezone
            datetime_import_pattern = r'from datetime import ([^;]+)'
            match = re.search(datetime_import_pattern, content)
            
            if match:
                # Add timezone to existing import
                existing_imports = match.group(1).strip()
                if 'timezone' not in existing_imports:
                    new_imports = existing_imports + ', timezone'
                    content = content.replace(match.group(0), f'from datetime import {new_imports}')
            else:
                # Look for other imports to place datetime import
                import_pattern = r'(^from [^;]+$\n|^import [^;]+$\n)'
                matches = list(re.finditer(import_pattern, content, re.MULTILINE))
                
                if matches:
                    # Insert after last import
                    last_import = matches[-1]
                    insert_pos = last_import.end()
                    content = content[:insert_pos] + 'from datetime import datetime, timezone\n' + content[insert_pos:]
                else:
                    # Add at beginning after any existing imports
                    content = 'from datetime import datetime, timezone\n' + content
        
        # Write back if changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Fixed datetime usage in: {file_path}")
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return False

def main():
    """Main function to fix datetime usage across the codebase"""
    
    # Target directories
    target_dirs = [
        'backend',
        'scripts',
        'tests',
        'examples'
    ]
    
    fixed_files = []
    total_files = 0
    
    for target_dir in target_dirs:
        target_path = Path(target_dir)
        
        if not target_path.exists():
            logger.warning(f"Directory {target_dir} does not exist, skipping")
            continue
        
        # Find all Python files
        for py_file in target_path.rglob('*.py'):
            total_files += 1
            
            # Skip __pycache__ and other non-source directories
            if any(part.startswith('.') or part == '__pycache__' for part in py_file.parts):
                continue
            
            if fix_datetime_in_file(py_file):
                fixed_files.append(py_file)
    
    logger.info(f"\nDatetime deprecation fix completed:")
    logger.info(f"- Total Python files processed: {total_files}")
    logger.info(f"- Files modified: {len(fixed_files)}")
    
    if fixed_files:
        logger.info(f"\nModified files:")
        for file_path in fixed_files:
            logger.info(f"  - {file_path}")

if __name__ == "__main__":
    main() 