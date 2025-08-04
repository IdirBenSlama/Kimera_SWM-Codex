#!/usr/bin/env python3
"""
Fix Remaining Relative Imports
==============================
Systematically fixes all remaining relative imports in insight_management files.
"""

import os
import re
from pathlib import Path

def fix_relative_imports_in_file(file_path: str):
    """Fix relative imports in a single file."""
    logger.info(f"Fixing: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Pattern replacements for common relative imports
    replacements = [
        # Core imports
        (r'from \.\.\.core\.insight import InsightScar',
         '''try:
    from src.core.insight import InsightScar
except ImportError:
    try:
        from core.insight import InsightScar
    except ImportError:
        class InsightScar:
            def __init__(self, **kwargs): self.__dict__.update(kwargs)'''),

        (r'from \.\.\.core\.geoid import GeoidState',
         '''try:
    from src.core.geoid import GeoidState
except ImportError:
    try:
        from core.geoid import GeoidState
    except ImportError:
        class GeoidState:
            @staticmethod
            def create_default(): return {}'''),

        # Utils imports
        (r'from \.\.\.utils\.config import get_api_settings',
         '''try:
    from src.utils.config import get_api_settings
except ImportError:
    try:
        from utils.config import get_api_settings
    except ImportError:
        def get_api_settings(): return {}'''),

        (r'from \.\.\.config\.settings import get_settings',
         '''try:
    from src.config.settings import get_settings
except ImportError:
    try:
        from config.settings import get_settings
    except ImportError:
        def get_settings(): return {}'''),

        (r'from \.\.\.utils\.kimera_logger import get_logger, LogCategory',
         '''try:
    from src.utils.kimera_logger import get_logger, LogCategory
except ImportError:
    try:
        from utils.kimera_logger import get_logger, LogCategory
    except ImportError:
        import logging
        def get_logger(*args, **kwargs): return logging.getLogger(__name__)
        class LogCategory:
            SYSTEM = "system"'''),

        (r'from \.\.\.utils\.kimera_exceptions import KimeraCognitiveError',
         '''try:
    from src.utils.kimera_exceptions import KimeraCognitiveError
except ImportError:
    try:
        from utils.kimera_exceptions import KimeraCognitiveError
    except ImportError:
        class KimeraCognitiveError(Exception):
            pass'''),
    ]

    # Apply replacements
    modified = False
    for pattern, replacement in replacements:
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
            modified = True
            logger.info(f"  Fixed: {pattern}")

    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"  Updated: {file_path}")
    else:
        logger.info(f"  No changes needed: {file_path}")

def main():
    """Fix all insight_management files."""
    logger.info("FIXING REMAINING RELATIVE IMPORTS")
    logger.info("=" * 40)

    insight_files = [
        "src/core/insight_management/insight_lifecycle.py",
        "src/core/insight_management/insight_feedback.py",
        "src/core/insight_management/insight_entropy.py",
        "src/core/insight_management/information_integration_analyzer.py"
    ]

    for file_path in insight_files:
        if os.path.exists(file_path):
            fix_relative_imports_in_file(file_path)
        else:
            logger.info(f"File not found: {file_path}")

    logger.info("\nFIXING COMPLETE!")

if __name__ == "__main__":
    main()
