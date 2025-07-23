#!/usr/bin/env python3
"""
KIMERA SWM Final Root Directory Cleanup Script
Systematically organizes all remaining files in the root directory.
"""

import os
import shutil
import logging
from pathlib import Path
from typing import Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_directories():
    """Create necessary directories for organization."""
    dirs_to_create = [
        'data/reports',
        'data/sessions',
        'data/logs',
        'data/analysis',
        'data/exports',
        'config/environments',
        'config/docker',
        'archive/2025-01-08-misc-cleanup',
        'docs/reports',
        'scripts/deployment'
    ]
    
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory exists: {dir_path}")

def get_file_mapping() -> Dict[str, str]:
    """Define where different types of files should go."""
    return {
        # JSON files - reports and data
        'json': {
            'reports': 'data/reports',
            'session': 'data/sessions', 
            'analysis': 'data/analysis',
            'test': 'data/analysis',
            'audit': 'data/reports',
            'comprehensive': 'data/reports',
            'performance': 'data/analysis',
            'verification': 'data/reports',
            'default': 'data/exports'
        },
        
        # Text and log files
        'txt': 'data/logs',
        'log': 'data/logs',
        'csv': 'data/exports',
        
        # Configuration files
        'yml': 'config/docker',
        'yaml': 'config',
        'toml': 'config',
        'ini': 'config',
        'env': 'config/environments',
        
        # Documentation and web files
        'md': 'docs',
        'html': 'docs/reports',
        
        # Docker and deployment
        'Dockerfile': 'config/docker',
        'docker-compose.yml': 'config/docker',
        
        # Temporary and misc files
        'tmp': 'archive/2025-01-08-misc-cleanup',
        'bak': 'archive/2025-01-08-misc-cleanup',
        'old': 'archive/2025-01-08-misc-cleanup',
        
        # Lock files and similar
        'lock': 'data/exports',
        'mmap': 'data/exports',
        'secure': 'data',
    }

def categorize_json_file(filename: str) -> str:
    """Determine the best destination for a JSON file based on its name."""
    mapping = get_file_mapping()['json']
    filename_lower = filename.lower()
    
    for keyword, destination in mapping.items():
        if keyword != 'default' and keyword in filename_lower:
            return destination
    
    return mapping['default']

def move_file(src: str, dest_dir: str) -> bool:
    """Move a file to the destination directory."""
    try:
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir, exist_ok=True)
        
        dest_path = os.path.join(dest_dir, os.path.basename(src))
        
        # If file already exists, add a number suffix
        counter = 1
        original_dest = dest_path
        while os.path.exists(dest_path):
            name, ext = os.path.splitext(original_dest)
            dest_path = f"{name}_{counter}{ext}"
            counter += 1
        
        shutil.move(src, dest_path)
        logger.info(f"Moved: {src} → {dest_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to move {src}: {e}")
        return False

def should_keep_in_root(filename: str) -> bool:
    """Determine if a file should stay in the root directory."""
    keep_in_root = {
        # Essential project files
        'README.md', 'LICENSE', 'CHANGELOG.md', 'CONTRIBUTING.md',
        '.gitignore', '.gitattributes', '.cursorrules',
        'pyproject.toml', 'setup.py', 'requirements.txt',
        'poetry.lock', 'pytest.ini',
        
        # Our new organization files
        'KIMERA_REORGANIZATION_COMPLETE.md',
        'KIMERA_REORGANIZATION_REPORT.md',
    }
    
    # Hidden files and directories should generally stay
    if filename.startswith('.') and os.path.isfile(filename):
        return True
    
    return filename in keep_in_root

def clean_root_directory():
    """Clean up all files in the root directory."""
    root_files = [f for f in os.listdir('.') if os.path.isfile(f)]
    
    mapping = get_file_mapping()
    moved_count = 0
    kept_count = 0
    
    logger.info(f"Found {len(root_files)} files in root directory")
    
    for filename in root_files:
        # Skip files that should stay in root
        if should_keep_in_root(filename):
            kept_count += 1
            logger.info(f"Keeping in root: {filename}")
            continue
        
        # Determine destination based on file extension and name
        file_ext = filename.split('.')[-1].lower() if '.' in filename else ''
        
        dest_dir = None
        
        # Special handling for specific files
        if filename.startswith('Dockerfile'):
            dest_dir = 'config/docker'
        elif filename == 'docker-compose.yml':
            dest_dir = 'config/docker'
        elif file_ext == 'json':
            dest_dir = categorize_json_file(filename)
        elif file_ext in mapping:
            dest_dir = mapping[file_ext]
        else:
            # Default destination for unrecognized files
            dest_dir = 'archive/2025-01-08-misc-cleanup'
        
        if dest_dir and move_file(filename, dest_dir):
            moved_count += 1
    
    return moved_count, kept_count

def main():
    """Main cleanup function."""
    logger.info("🧹 KIMERA SWM Final Root Directory Cleanup")
    logger.info("=" * 60)
    
    # Create necessary directories
    create_directories()
    
    # Clean up root directory
    moved_count, kept_count = clean_root_directory()
    
    # Final summary
    remaining_files = len([f for f in os.listdir('.') if os.path.isfile(f)])
    
    logger.info("=" * 60)
    logger.info("📊 CLEANUP SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Files moved: {moved_count}")
    logger.info(f"Files kept in root: {kept_count}")
    logger.info(f"Files remaining in root: {remaining_files}")
    
    if remaining_files <= 15:  # Reasonable number of essential files
        logger.info("✅ Root directory cleanup SUCCESSFUL!")
        logger.info("🎯 Root directory is now clean and organized!")
    else:
        logger.warning(f"⚠️  Root directory still has {remaining_files} files")
        logger.info("Some files may need manual review")
    
    logger.info("=" * 60)

if __name__ == "__main__":
    main() 