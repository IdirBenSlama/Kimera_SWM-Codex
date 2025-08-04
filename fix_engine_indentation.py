"""
Fix Engine Indentation Issues
=============================

This script automatically fixes the indentation issues in engine files
where code after logger.debug statements is incorrectly indented.
"""

import os
import re
from typing import List, Tuple
import logging
logger = logging.getLogger(__name__)

def fix_indentation_in_file(filepath: str) -> Tuple[bool, int]:
    """
    Fix indentation issues in a single file.
    
    Returns:
        Tuple of (was_modified, num_fixes)
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    modified = False
    fixes = 0
    fixed_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check if this line is a logger.debug with environment
        if 'logger.debug(f"   Environment: {self.settings.environment}")' in line:
            # Get the indentation of this line
            indent_match = re.match(r'^(\s*)', line)
            if indent_match:
                current_indent = indent_match.group(1)
                
                # Add this line
                fixed_lines.append(line)
                
                # Check next line
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    next_indent_match = re.match(r'^(\s*)', next_line)
                    
                    if next_indent_match:
                        next_indent = next_indent_match.group(1)
                        
                        # If next line has no indentation but contains code, fix it
                        if len(next_indent) == 0 and next_line.strip() and not next_line.strip().startswith('#'):
                            # Add proper indentation (same as logger line)
                            fixed_line = current_indent + next_line
                            fixed_lines.append(fixed_line)
                            modified = True
                            fixes += 1
                            i += 1
                            continue
        
        fixed_lines.append(line)
        i += 1
    
    if modified:
        # Write back the fixed content
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(fixed_lines)
    
    return modified, fixes


def fix_all_engines():
    """Fix indentation in all engine files"""
    engines_dir = "src/engines"
    
    if not os.path.exists(engines_dir):
        logger.info(f"❌ Directory {engines_dir} not found!")
        return
    
    logger.info("🔧 Fixing indentation issues in engine files...")
    
    total_files = 0
    modified_files = 0
    total_fixes = 0
    
    # Get all Python files in engines directory
    for root, dirs, files in os.walk(engines_dir):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                total_files += 1
                
                try:
                    was_modified, num_fixes = fix_indentation_in_file(filepath)
                    
                    if was_modified:
                        modified_files += 1
                        total_fixes += num_fixes
                        logger.info(f"  ✓ Fixed {num_fixes} issues in {filepath}")
                
                except Exception as e:
                    logger.info(f"  ✗ Error processing {filepath}: {e}")
    
    logger.info(f"\n📊 Summary:")
    logger.info(f"  Total files scanned: {total_files}")
    logger.info(f"  Files modified: {modified_files}")
    logger.info(f"  Total fixes applied: {total_fixes}")
    
    if total_fixes > 0:
        logger.info("\n✅ Indentation issues fixed! Please run the audit again to verify.")
    else:
        logger.info("\n✅ No indentation issues found.")


def verify_fix():
    """Verify that the fixes were applied correctly"""
    logger.info("\n🔍 Verifying fixes...")
    
    # Try to import a few previously broken engines
    test_engines = [
        "axiom_mathematical_proof",
        "axiom_of_understanding",
        "axiom_verification"
    ]
    
    success_count = 0
    for engine in test_engines:
        try:
            module = __import__(f"src.engines.{engine}", fromlist=[engine])
            logger.info(f"  ✓ {engine} - import successful")
            success_count += 1
        except Exception as e:
            logger.info(f"  ✗ {engine} - import failed: {e}")
    
    logger.info(f"\n  Success rate: {success_count}/{len(test_engines)}")


if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("ENGINE INDENTATION FIX UTILITY")
    logger.info("=" * 80)
    
    fix_all_engines()
    
    # Optionally verify the fixes
    # verify_fix()