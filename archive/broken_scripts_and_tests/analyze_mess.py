#!/usr/bin/env python3
"""
Directory Mess Analysis
=======================

Analyzes the current chaotic directory structure and shows the problems.
"""

import os
from pathlib import Path
from collections import defaultdict

# Initialize structured logger
from backend.utils.kimera_logger import get_system_logger
logger = get_system_logger(__name__)


def analyze_directory_mess():
    """Analyze the current directory structure."""
    root = Path.cwd()
    
    logger.info("KIMERA SWM Directory Mess Analysis")
    logger.info("=" * 50)
    logger.info(f"Root directory: {root}")
    logger.info()
    
    # Count items in root
    files_in_root = []
    dirs_in_root = []
    
    for item in root.iterdir():
        if item.name.startswith('.'):
            continue
        if item.is_file():
            files_in_root.append(item)
        elif item.is_dir():
            dirs_in_root.append(item)
    
    logger.info(f"📁 Directories in root: {len(dirs_in_root)
    logger.info(f"📄 Files in root: {len(files_in_root)
    logger.info(f"📊 Total items in root: {len(files_in_root)
    logger.info()
    
    # Analyze file types
    file_types = defaultdict(list)
    for file in files_in_root:
        ext = file.suffix.lower() or 'no_extension'
        file_types[ext].append(file.name)
    
    logger.info("📋 Files by type in root directory:")
    for ext, files in sorted(file_types.items()):
        logger.info(f"  {ext}: {len(files)
        if len(files) <= 5:
            for file in files:
                logger.info(f"    - {file}")
        else:
            for file in files[:3]:
                logger.info(f"    - {file}")
            logger.info(f"    ... and {len(files)
    logger.info()
    
    # Identify problematic patterns
    problems = {
        "Test files in root": [f for f in files_in_root if 'test' in f.name.lower()],
        "Database files in root": [f for f in files_in_root if f.suffix in ['.db', '.db-shm', '.db-wal']],
        "JSON result files": [f for f in files_in_root if f.suffix == '.json' and any(x in f.name.lower() for x in ['result', 'analysis', 'report'])],
        "HTML files in root": [f for f in files_in_root if f.suffix == '.html'],
        "Documentation scattered": [f for f in files_in_root if f.suffix == '.md'],
        "Python scripts everywhere": [f for f in files_in_root if f.suffix == '.py'],
    }
    
    logger.info("🚨 PROBLEMS IDENTIFIED:")
    for problem, files in problems.items():
        if files:
            logger.error(f"  ❌ {problem}: {len(files)
            for file in files[:3]:
                logger.info(f"     - {file.name}")
            if len(files) > 3:
                logger.info(f"     ... and {len(files)
    logger.info()
    
    # Show directory structure issues
    logger.info("📁 DIRECTORY STRUCTURE ISSUES:")
    
    # Check for duplicate/similar directories
    dir_names = [d.name.lower() for d in dirs_in_root]
    if 'docs' in dir_names and any('doc' in name for name in dir_names if name != 'docs'):
        logger.error("  ❌ Multiple documentation directories")
    
    if 'tests' in dir_names and any('test' in name for name in dir_names if name != 'tests'):
        logger.error("  ❌ Multiple test directories")
    
    # Check for scattered configs
    config_files = [f for f in files_in_root if 'config' in f.name.lower()]
    if config_files and 'config' in dir_names:
        logger.error(f"  ❌ Config files both in root ({len(config_files)
    
    # Check for build artifacts
    build_artifacts = [f for f in files_in_root if any(x in f.name.lower() for x in ['build', 'dist', 'egg-info'])]
    if build_artifacts:
        logger.error(f"  ❌ Build artifacts in root: {len(build_artifacts)
    
    logger.info()
    
    # Recommendations
    logger.info("💡 RECOMMENDATIONS:")
    logger.info("  1. Move all test files to tests/ directory")
    logger.info("  2. Move all documentation to docs/ directory")
    logger.info("  3. Move database files to data/ directory")
    logger.info("  4. Move JSON results to results/ or data/ directory")
    logger.info("  5. Move HTML files to web/ or dashboard/ directory")
    logger.info("  6. Organize Python scripts by purpose")
    logger.info("  7. Move build artifacts to build/ directory")
    logger.info("  8. Create clear separation between source code and artifacts")
    logger.info()
    
    logger.info("🛠️  AVAILABLE CLEANUP SCRIPTS:")
    logger.info("  - quick_cleanup.py: Conservative cleanup, preserves functionality")
    logger.info("  - reorganize_project.py: Complete restructure to industry standards")
    logger.info()
    
    # Show largest files
    large_files = [(f, f.stat().st_size) for f in files_in_root]
    large_files.sort(key=lambda x: x[1], reverse=True)
    
    logger.info("📊 LARGEST FILES IN ROOT:")
    for file, size in large_files[:10]:
        size_mb = size / (1024 * 1024)
        if size_mb > 0.1:  # Show files > 100KB
            logger.info(f"  {size_mb:6.2f} MB - {file.name}")
    logger.info()
    
    # Summary
    logger.info("📈 MESS SEVERITY: HIGH")
    logger.info(f"  - {len(files_in_root)
    logger.info(f"  - {len([f for f in files_in_root if f.suffix == '.py'])
    logger.info(f"  - {len([f for f in files_in_root if f.suffix == '.md'])
    logger.info(f"  - {len([f for f in files_in_root if f.suffix in ['.json', '.db', '.html']])
    logger.info()
    logger.info("🎯 GOAL: Clean, professional directory structure")
    logger.info("📁 SOLUTION: Run one of the cleanup scripts")

if __name__ == "__main__":
    analyze_directory_mess()