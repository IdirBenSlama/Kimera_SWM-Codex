#!/usr/bin/env python3
"""
Quick Syntax Check for KIMERA System
====================================

Fast syntax validation tool to identify corrupted Python files
and provide immediate assessment of the codebase health.
"""

import ast
import sys
from pathlib import Path
from typing import List, Tuple
import re

def check_file_syntax(file_path: Path) -> Tuple[bool, str]:
    """
    Check if a Python file has valid syntax.
    
    Returns:
        (is_valid, error_message)
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        # Quick corruption pattern checks
        corruption_patterns = [
            (r'"""[^"]*$', "Unterminated triple quotes"),
            (r"'''[^']*$", "Unterminated triple quotes"),
            (r'(?<!\\)"[^"]*\n[^"]*(?<!\\)"', "Multi-line string without triple quotes"),
        ]
        
        for pattern, description in corruption_patterns:
            if re.search(pattern, content, re.MULTILINE):
                return False, f"Corruption: {description}"
        
        # Try to parse AST
        ast.parse(content)
        return True, "OK"
        
    except SyntaxError as e:
        return False, f"Syntax Error: {e}"
    except Exception as e:
        return False, f"Read Error: {e}"

def main():
    """Main function to check all Python files."""
    if len(sys.argv) > 1:
        root_path = Path(sys.argv[1])
    else:
        root_path = Path('.')
        
    print(f"🔍 Checking Python files in: {root_path.absolute()}")
    print("=" * 60)
    
    python_files = list(root_path.rglob("*.py"))
    total_files = len(python_files)
    corrupted_files = []
    
    print(f"Found {total_files} Python files")
    print()
    
    for i, py_file in enumerate(python_files, 1):
        is_valid, error_msg = check_file_syntax(py_file)
        
        if not is_valid:
            corrupted_files.append((py_file, error_msg))
            status = "❌"
        else:
            status = "✅"
            
        # Show progress for every 50 files or if there's an error
        if i % 50 == 0 or not is_valid:
            relative_path = py_file.relative_to(root_path)
            print(f"{status} [{i:4d}/{total_files}] {relative_path}")
            if not is_valid:
                print(f"    Error: {error_msg}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SYNTAX CHECK SUMMARY")
    print(f"Total files: {total_files}")
    print(f"Valid files: {total_files - len(corrupted_files)}")
    print(f"Corrupted files: {len(corrupted_files)}")
    print(f"Corruption rate: {len(corrupted_files) / total_files * 100:.1f}%")
    
    if corrupted_files:
        print("\n🔴 CORRUPTED FILES:")
        for file_path, error in corrupted_files[:20]:  # Show first 20
            relative_path = file_path.relative_to(root_path)
            print(f"  ❌ {relative_path}")
            print(f"     {error}")
            
        if len(corrupted_files) > 20:
            print(f"  ... and {len(corrupted_files) - 20} more files")
    
    print("\n💡 RECOMMENDATIONS:")
    if len(corrupted_files) > total_files * 0.1:  # More than 10% corrupted
        print("  🚨 High corruption rate detected!")
        print("  📋 Run the full recovery tool: python scripts/system_recovery.py")
        print("  🔧 Use clean versions of core files where available")
        print("  ⚡ Implement automated syntax validation")
    else:
        print("  ✅ Corruption rate is manageable")
        print("  🔧 Fix individual files or run targeted recovery")
    
    return len(corrupted_files)

if __name__ == "__main__":
    exit_code = main()
    sys.exit(min(exit_code, 125))  # Cap exit code for shell compatibility 