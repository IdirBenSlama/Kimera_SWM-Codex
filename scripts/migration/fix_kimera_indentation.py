#!/usr/bin/env python3
"""
Fix Kimera System Indentation Issues
====================================

Fixes indentation problems in kimera_system.py after import replacements.

Usage:
    python scripts/migration/fix_kimera_indentation.py
"""

import re

def fix_kimera_indentation():
    """Fix indentation issues in kimera_system.py."""
    filepath = "src/core/kimera_system.py"

    logger.info("🔧 FIXING KIMERA SYSTEM INDENTATION")
    logger.info("=" * 50)

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Fix try-except blocks that have incorrect indentation
    # Pattern: Find try: followed by incorrect indentation
    fixes_applied = 0

    # Fix pattern 1: try:\nfrom pattern
    pattern1 = r'(\s+)try:\n    from\s'
    replacement1 = lambda m: f'{m.group(1)}try:\n{m.group(1)}    from '
    content = re.sub(pattern1, replacement1, content)

    # Fix pattern 2: except ImportError:\nSomething = None
    pattern2 = r'(\s+)except ImportError:\n    ([A-Za-z_][A-Za-z0-9_]*) = None'
    replacement2 = lambda m: f'{m.group(1)}except ImportError:\n{m.group(1)}    {m.group(2)} = None'
    content = re.sub(pattern2, replacement2, content)

    # Fix pattern 3: except ImportError:\ndef or class
    pattern3 = r'(\s+)except ImportError:\n    (def |class )'
    replacement3 = lambda m: f'{m.group(1)}except ImportError:\n{m.group(1)}    {m.group(2)}'
    content = re.sub(pattern3, replacement3, content)

    # Manual fixes for specific known issues
    fixes = [
        # Fix vault manager import
        (
            """        try:
    from ..vault.vault_manager import VaultManager
except ImportError:
    VaultManager = None""",
            """        try:
            from ..vault.vault_manager import VaultManager
        except ImportError:
            VaultManager = None"""
        ),

        # Fix embedding utils import
        (
            """        try:
    from . import embedding_utils
except ImportError:
    embedding_utils = None""",
            """        try:
            from . import embedding_utils
        except ImportError:
            embedding_utils = None"""
        ),

        # Fix system monitor import
        (
            """        try:
    from ..monitoring.system_monitor import SystemMonitor
except ImportError:
    SystemMonitor = None""",
            """        try:
            from ..monitoring.system_monitor import SystemMonitor
        except ImportError:
            SystemMonitor = None"""
        ),

        # Fix ethical governor import
        (
            """        try:
    from ..governance.ethical_governor import EthicalGovernor
except ImportError:
    EthicalGovernor = None""",
            """        try:
            from ..governance.ethical_governor import EthicalGovernor
        except ImportError:
            EthicalGovernor = None"""
        ),

        # Fix exception handling import
        (
            """        try:
    from . import exception_handling
except ImportError:
    exception_handling = None""",
            """        try:
            from . import exception_handling
        except ImportError:
            exception_handling = None"""
        )
    ]

    for old, new in fixes:
        if old in content:
            content = content.replace(old, new)
            fixes_applied += 1
            logger.info(f"✅ Fixed indentation block {fixes_applied}")

    # Write back the corrected content
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

    logger.info(f"✅ Applied {fixes_applied} indentation fixes")
    return fixes_applied > 0

def main():
    """Main execution function."""
    try:
        success = fix_kimera_indentation()

        if success:
            logger.info("\n✅ INDENTATION FIXES COMPLETE")
            return 0
        else:
            logger.info("\n⚠️ NO FIXES NEEDED")
            return 0

    except Exception as e:
        logger.info(f"\n❌ INDENTATION FIXING FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    import sys
import logging
logger = logging.getLogger(__name__)
    sys.exit(main())
