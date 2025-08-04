#!/usr/bin/env python3
"""
Complete Kimera System Syntax Repair
====================================
Systematic repair of all syntax issues
"""

import re
import ast
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

def fix_all_syntax_issues():
    """Comprehensively fix all syntax issues in kimera_system.py"""
    kimera_path = Path("src/core/kimera_system.py")

    with open(kimera_path, 'r', encoding='utf-8') as f:
        content = f.read()

    logger.info("🔧 COMPLETE SYNTAX REPAIR")
    logger.info("=" * 50)

    lines = content.split('\n')
    new_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Handle method definitions with orphaned try statements
        if 'def _initialize_' in line and i + 1 < len(lines):
            # Add the method definition
            new_lines.append(line)
            i += 1

            # Check for docstring
            if i < len(lines) and '"""' in lines[i]:
                new_lines.append(lines[i])
                i += 1
                # Find closing docstring
                while i < len(lines) and not ('"""' in lines[i] and lines[i].count('"""') == 1):
                    new_lines.append(lines[i])
                    i += 1
                if i < len(lines):
                    new_lines.append(lines[i])  # closing """
                    i += 1

            # Now look for the import statement that needs try-except
            if i < len(lines):
                next_line = lines[i].strip()

                # Check if it's an import line that needs try-except wrapper
                if (('from .' in next_line or 'import ' in next_line) and
                    'def ' not in next_line and
                    i + 1 < len(lines) and
                    'except ImportError:' in lines[i + 1]):

                    # Get indentation
                    indent = len(lines[i]) - len(lines[i].lstrip())

                    # Add try block
                    new_lines.append(' ' * indent + 'try:')
                    new_lines.append(' ' * (indent + 4) + next_line)
                    i += 1

                    # Add the except block
                    new_lines.append(lines[i])  # except ImportError line
                    i += 1

                    logger.info(f"✅ Fixed import wrapper around line {i}")
                else:
                    new_lines.append(lines[i])
                    i += 1
            continue

        # Handle orphaned except ImportError statements
        elif 'except ImportError:' in stripped:
            # Look backwards for a try statement
            found_try = False
            for j in range(len(new_lines) - 1, max(0, len(new_lines) - 10), -1):
                if 'try:' in new_lines[j]:
                    found_try = True
                    break

            if not found_try:
                # Look forward for what this except is supposed to catch
                j = i - 1
                while j >= 0 and lines[j].strip():
                    if ('from ' in lines[j] or 'import ' in lines[j]) and 'def ' not in lines[j]:
                        # Found the import, add try before it
                        # Remove the import from new_lines and re-add with try
                        indent = len(lines[j]) - len(lines[j].lstrip())
                        if len(new_lines) > 0:
                            new_lines.pop()  # Remove the orphaned import
                        new_lines.append(' ' * indent + 'try:')
                        new_lines.append(' ' * (indent + 4) + lines[j].strip())
                        break
                    j -= 1

            new_lines.append(line)
            i += 1
        else:
            new_lines.append(line)
            i += 1

    # Join and apply final fixes
    content = '\n'.join(new_lines)

    # Remove any duplicate try statements
    content = re.sub(r'(\s+)try:\s*\n\s+try:', r'\1try:', content)

    # Fix any remaining malformed try-except blocks
    content = re.sub(r'(\s+)try:\s*\n(\s+)except ImportError:',
                     r'\1try:\n\2    pass\n\2except ImportError:', content)

    # Write the repaired file
    with open(kimera_path, 'w', encoding='utf-8') as f:
        f.write(content)

    logger.info("✅ Complete syntax repair applied")

    # Validate syntax
    try:
        ast.parse(content)
        logger.info("✅ File syntax is now completely valid!")
        return True
    except SyntaxError as e:
        logger.info(f"❌ Remaining syntax error: {e}")
        logger.info(f"Line {e.lineno}: {e.text}")

        # Show more context
        lines = content.split('\n')
        start = max(0, e.lineno - 5)
        end = min(len(lines), e.lineno + 5)
        logger.info("\nContext:")
        for i in range(start, end):
            marker = ">>> " if i + 1 == e.lineno else "    "
            logger.info(f"{marker}{i+1:3}: {lines[i]}")
        return False

if __name__ == "__main__":
    fix_all_syntax_issues()
