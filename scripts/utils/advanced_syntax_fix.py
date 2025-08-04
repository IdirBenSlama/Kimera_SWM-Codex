#!/usr/bin/env python3
"""
Advanced Kimera System Syntax Repair
====================================
Aerospace-grade syntax repair with fault tolerance
"""

import os
import re
import sys
import ast
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

def repair_kimera_system():
    """Apply comprehensive syntax repairs"""
    kimera_path = Path("src/core/kimera_system.py")

    # Read with encoding detection
    content = None
    for encoding in ['utf-8', 'latin-1', 'cp1252']:
        try:
            with open(kimera_path, 'r', encoding=encoding) as f:
                content = f.read()
            logger.info(f"✅ Read with {encoding}")
            break
        except UnicodeDecodeError:
            continue

    if content is None:
        logger.info("❌ Failed to read file")
        return False

    lines = content.split('\n')
    new_lines = []
    i = 0

    logger.info("🔧 ADVANCED SYNTAX REPAIR")
    logger.info("=" * 50)

    while i < len(lines):
        line = lines[i]

        # Check for orphaned except statements
        if 'except ImportError:' in line and i > 0:
            prev_line = lines[i-1].strip()

            # If previous line is not try/except related, we need to add try
            if not any(x in prev_line for x in ['try:', 'except', 'finally:']):
                # Look backwards for the import statement
                j = i - 1
                while j >= 0 and lines[j].strip():
                    if ('from .' in lines[j] or 'import ' in lines[j]) and 'def ' not in lines[j]:
                        # Found the import, wrap it in try
                        indent = len(lines[j]) - len(lines[j].lstrip())
                        import_line = lines[j]

                        # Replace the import line with try block
                        new_lines = new_lines[:-1]  # Remove the import line we already added
                        new_lines.append(' ' * indent + 'try:')
                        new_lines.append(' ' * (indent + 4) + import_line.strip())
                        new_lines.append(line)  # The except line

                        logger.info(f"✅ Fixed try-except block at line {i+1}")
                        i += 1
                        break
                    j -= 1
                else:
                    # Couldn't find import, add pass to except
                    new_lines.append(line)
                    new_lines.append(line.replace('except ImportError:', '    pass'))
                    logger.info(f"✅ Added pass to orphaned except at line {i+1}")
                    i += 1
            else:
                new_lines.append(line)
                i += 1
        else:
            new_lines.append(line)
            i += 1

    # Specific fixes for known patterns
    content = '\n'.join(new_lines)

    # Fix the GPU system section
    content = re.sub(
        r'GPU_SYSTEM_AVAILABLE = True\n\n# Legacy GPU Foundation fallback - fix import path\n(\s+)GPU_FOUNDATION_AVAILABLE = True',
        r'GPU_SYSTEM_AVAILABLE = True\n\n# Legacy GPU Foundation fallback - fix import path\ntry:\n    GPU_FOUNDATION_AVAILABLE = True',
        content
    )

    # Fix method definition spacing
    content = re.sub(r'\n    def _initialize_([^(]+)\(self\) -> None:\n        """([^"]+)"""\n        try:',
                     r'\n\n    def _initialize_\1(self) -> None:\n        """\2"""\n        try:', content)

    # Write the fixed file
    try:
        with open(kimera_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info("✅ Fixed content written")
    except Exception as e:
        logger.info(f"❌ Write error: {e}")
        return False

    # Test syntax
    try:
        ast.parse(content)
        logger.info("✅ Syntax validation passed!")
        return True
    except SyntaxError as e:
        logger.info(f"❌ Still has syntax error: {e}")
        logger.info(f"Line {e.lineno}: {e.text}")

        # Show context around error
        lines = content.split('\n')
        start = max(0, e.lineno - 3)
        end = min(len(lines), e.lineno + 3)
        logger.info("\nContext:")
        for i in range(start, end):
            marker = ">>> " if i + 1 == e.lineno else "    "
            logger.info(f"{marker}{i+1:3}: {lines[i]}")

        return False

if __name__ == "__main__":
    success = repair_kimera_system()
    sys.exit(0 if success else 1)
