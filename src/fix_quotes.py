#!/usr/bin/env python3
"""
KIMERA SWM Quote Fixer Tool
===========================
Automatically fixes unmatched triple quotes in Python files.
"""

import os
import re
import sys
from typing import List, Tuple

def analyze_quotes(content: str) -> Tuple[int, List[int]]:
    """Analyze triple quote positions and count."""
    quote_positions = []
    lines = content.split('\n')

    for i, line in enumerate(lines, 1):
        if '"""' in line:
            quote_positions.append(i)

    return len(quote_positions), quote_positions

def fix_quotes_in_content(content: str) -> str:
    """Fix unmatched quotes in content."""
    lines = content.split('\n')
    quote_count = 0

    # Find lines with triple quotes
    for i, line in enumerate(lines):
        quote_count += line.count('"""')

    # If odd number, we need to add a closing quote
    if quote_count % 2 != 0:
        # Find the last line with content and add closing docstring
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip():
                # Check if it's likely an incomplete docstring
                if i < len(lines) - 5:  # Not at the very end
                    lines.append('    """')
                    print(f"  Added closing quote at end of file")
                break

    return '\n'.join(lines)

def fix_file_quotes(filepath: str) -> bool:
    """Fix quotes in a specific file."""
    try:
        print(f"Analyzing: {filepath}")

        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            original_content = f.read()

        quote_count, positions = analyze_quotes(original_content)

        if quote_count % 2 == 0:
            print(f"  ✅ Quotes already balanced ({quote_count})")
            return True

        print(f"  ⚠️  Unmatched quotes: {quote_count} (positions: {positions})")

        # Attempt to fix
        fixed_content = fix_quotes_in_content(original_content)

        # Verify fix
        new_count, _ = analyze_quotes(fixed_content)
        if new_count % 2 == 0:
            # Create backup
            backup_path = f"{filepath}.backup_quotes"
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(original_content)

            # Write fixed content
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(fixed_content)

            print(f"  ✅ Fixed! Backup created: {backup_path}")
            return True
        else:
            print(f"  ❌ Fix failed, still {new_count} quotes")
            return False

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def main():
    if len(sys.argv) > 1:
        # Fix specific file
        filepath = sys.argv[1]
        fix_file_quotes(filepath)
    else:
        # Fix critical files
        critical_files = [
            'core/system/kimera_system.py',
            'main.py'
        ]

        print("=== KIMERA SWM QUOTE FIXER ===\n")

        for filepath in critical_files:
            if os.path.exists(filepath):
                fix_file_quotes(filepath)
            else:
                print(f"File not found: {filepath}")
            print()

if __name__ == "__main__":
    main()
