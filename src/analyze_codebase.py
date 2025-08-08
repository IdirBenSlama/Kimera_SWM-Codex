#!/usr/bin/env python3
"""KIMERA SWM Codebase Static Analysis Tool"""

import os
import sys
from typing import Tuple, List, Dict, Any

def analyze_python_file(filepath: str) -> Tuple[bool, List[str], List[str]]:
    """Analyze a single Python file for various issues."""
    issues = []
    warnings = []

    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        return False, [f'FILE_READ_ERROR: {str(e)}'], []

    # Check for syntax by compilation
    syntax_ok = True
    try:
        compile(content, filepath, 'exec')
    except SyntaxError as e:
        syntax_ok = False
        issues.append(f'SYNTAX_ERROR: {e.msg} at line {e.lineno}')
    except Exception as e:
        syntax_ok = False
        issues.append(f'COMPILE_ERROR: {str(e)}')

    # Check for triple quote balance
    quote_count = content.count('"""')
    if quote_count % 2 != 0:
        issues.append(f'UNMATCHED_QUOTES: {quote_count} triple quotes (should be even)')

    # Check for common patterns
    if 'TODO: Fix syntax errors' in content:
        warnings.append('TODO_SYNTAX_FIX_NEEDED')

    if '# Temporarily disabled' in content:
        warnings.append('DISABLED_CODE_PRESENT')

    return syntax_ok, issues, warnings

def main() -> Dict[str, Any]:
    print("=== KIMERA SWM STATIC ANALYSIS ===\n")

    total_files = 0
    syntax_errors = 0
    files_with_issues = 0

    critical_issues = {}
    major_issues = {}

    critical_files = ['main.py', 'main_simple.py', 'core/system/kimera_system.py', 'core/__init__.py']

    print("🔍 ANALYZING CRITICAL FILES:")
    print("=" * 50)

    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                relative_path = os.path.relpath(filepath, '.').replace('\\', '/')
                total_files += 1

                syntax_ok, issues, warnings = analyze_python_file(filepath)

                if not syntax_ok:
                    syntax_errors += 1

                if issues:
                    files_with_issues += 1
                    is_critical = any(critical in relative_path for critical in critical_files)

                    if is_critical:
                        critical_issues[relative_path] = {'issues': issues, 'warnings': warnings}
                        print(f"🔴 CRITICAL: {relative_path}")
                        for issue in issues:
                            print(f"   ❌ {issue}")
                        print()
                    elif 'SYNTAX_ERROR' in str(issues):
                        major_issues[relative_path] = {'issues': issues, 'warnings': warnings}

    print(f"\n📊 ANALYSIS SUMMARY:")
    print(f"Total files: {total_files}")
    print(f"Syntax errors: {syntax_errors}")
    print(f"Critical issues: {len(critical_issues)}")
    print(f"Major issues: {len(major_issues)}")

    return {'critical': critical_issues, 'major': major_issues}

if __name__ == "__main__":
    main()
