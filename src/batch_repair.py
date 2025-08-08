#!/usr/bin/env python3
"""
KIMERA SWM Batch Repair Tool
============================

Systematic repair tool for processing multiple files with common issues.
Implements the Batch 1 strategy from the repair roadmap.
"""

import os
import sys
import re
from typing import List, Dict, Tuple, Set
from pathlib import Path

def fix_common_quote_issues(content: str) -> Tuple[str, List[str]]:
    """Fix common triple quote issues."""
    fixes = []
    lines = content.split('\n')

    # Fix malformed docstring headers (like utils/__init__.py pattern)
    for i, line in enumerate(lines):
        if line.strip().startswith('"""') and line.strip().endswith('"""') and len(line.strip()) > 6:
            # Single line docstring that should be multi-line
            if i + 1 < len(lines) and lines[i + 1].strip().startswith('='):
                # Header pattern detected
                lines[i] = '"""'
                lines.insert(i + 1, line.strip()[3:-3])
                fixes.append(f"Fixed malformed docstring header at line {i+1}")
                break

    return '\n'.join(lines), fixes

def fix_import_comma_issues(content: str) -> Tuple[str, List[str]]:
    """Fix missing commas in import statements."""
    fixes = []
    lines = content.split('\n')

    for i, line in enumerate(lines):
        # Look for import statements with missing commas
        if 'from ' in line and 'import (' in line:
            # Multi-line import - check next line
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line and not next_line.startswith('#') and not next_line.endswith(',') and not next_line.endswith(')'):
                    # Check if there are more imports after this line
                    if i + 2 < len(lines) and lines[i + 2].strip() and not lines[i + 2].strip().startswith(')'):
                        lines[i + 1] = lines[i + 1].rstrip() + ','
                        fixes.append(f"Added missing comma in import at line {i+2}")

    return '\n'.join(lines), fixes

def fix_indentation_issues(content: str) -> Tuple[str, List[str]]:
    """Fix basic indentation issues."""
    fixes = []
    lines = content.split('\n')

    for i, line in enumerate(lines):
        if line.strip():
            # Check for common indentation errors (non-4-space indentation)
            leading_spaces = len(line) - len(line.lstrip())
            if leading_spaces > 0 and leading_spaces % 4 != 0 and not line.lstrip().startswith('#'):
                # Round to nearest 4-space indentation
                new_indent = (leading_spaces // 4 + (1 if leading_spaces % 4 > 2 else 0)) * 4
                lines[i] = ' ' * new_indent + line.lstrip()
                fixes.append(f"Fixed indentation at line {i+1}: {leading_spaces} -> {new_indent} spaces")

    return '\n'.join(lines), fixes

def repair_file(filepath: str) -> Tuple[bool, List[str]]:
    """Attempt to repair a single file."""
    try:
        print(f"🔧 Repairing: {filepath}")

        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            original_content = f.read()

        content = original_content
        all_fixes = []

        # Apply fixes in order of safety
        content, fixes = fix_common_quote_issues(content)
        all_fixes.extend(fixes)

        content, fixes = fix_import_comma_issues(content)
        all_fixes.extend(fixes)

        content, fixes = fix_indentation_issues(content)
        all_fixes.extend(fixes)

        # Test if the fixes work
        try:
            compile(content, filepath, 'exec')
            syntax_ok = True
        except SyntaxError as e:
            syntax_ok = False
            all_fixes.append(f"❌ Still has syntax error: {e.msg} at line {e.lineno}")

        # If we made fixes and syntax is OK, save the file
        if all_fixes and syntax_ok and content != original_content:
            # Create backup
            backup_path = f"{filepath}.backup_batch"
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(original_content)

            # Save fixed content
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

            print(f"  ✅ Fixed! Changes: {len([f for f in all_fixes if not f.startswith('❌')])}")
            for fix in all_fixes:
                print(f"     - {fix}")
            return True, all_fixes

        elif syntax_ok and not all_fixes:
            print(f"  ✅ Already clean")
            return True, []

        else:
            print(f"  ⚠️ Could not fix automatically")
            for fix in all_fixes:
                print(f"     - {fix}")
            return False, all_fixes

    except Exception as e:
        print(f"  ❌ Error processing file: {e}")
        return False, [f"Error: {e}"]

def get_batch_1_files() -> List[str]:
    """Get the first batch of 50 most critical files to repair."""

    # High priority directories in dependency order
    priority_dirs = [
        'core/architecture',
        'core/cognitive',
        'core/system',
        'api/core',
        'utils',
        'utilities',
        'api',
        'config',
        'core/data',
        'core/processing'
    ]

    files_to_process = []

    # Find Python files in priority directories
    for priority_dir in priority_dirs:
        if os.path.exists(priority_dir):
            for root, dirs, files in os.walk(priority_dir):
                for file in files:
                    if file.endswith('.py'):
                        filepath = os.path.join(root, file)
                        # Skip files we've already fixed
                        if 'kimera_system_clean.py' not in filepath:
                            files_to_process.append(filepath)

                        if len(files_to_process) >= 50:
                            return files_to_process

    # If we need more files, add from other directories
    if len(files_to_process) < 50:
        for root, dirs, files in os.walk('.'):
            if len(files_to_process) >= 50:
                break
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    if filepath not in files_to_process and 'kimera_system_clean.py' not in filepath:
                        files_to_process.append(filepath)
                        if len(files_to_process) >= 50:
                            break

    return files_to_process[:50]

def main():
    print("=== KIMERA SWM BATCH REPAIR TOOL ===")
    print("🎯 Processing Batch 1: 50 Critical Files")
    print("=" * 50)

    batch_files = get_batch_1_files()

    print(f"📋 Found {len(batch_files)} files to process")
    print()

    successful_repairs = 0
    failed_repairs = 0
    already_clean = 0

    for i, filepath in enumerate(batch_files, 1):
        print(f"[{i:2d}/{len(batch_files)}] ", end="")

        success, fixes = repair_file(filepath)

        if success and fixes:
            successful_repairs += 1
        elif success and not fixes:
            already_clean += 1
        else:
            failed_repairs += 1

        print()

    print("=" * 50)
    print("📊 BATCH 1 SUMMARY:")
    print(f"✅ Successfully repaired: {successful_repairs}")
    print(f"🔧 Already clean: {already_clean}")
    print(f"❌ Failed to repair: {failed_repairs}")
    print(f"📈 Success rate: {(successful_repairs + already_clean)/len(batch_files)*100:.1f}%")

    if successful_repairs > 0:
        print(f"\n🎉 Made progress! Run analysis to see improvement.")

if __name__ == "__main__":
    main()
