#!/usr/bin/env python3
"""
Fix Import Path Issues in Health Check Scripts
=============================================
Resolves "No module named 'src'" errors by ensuring proper import paths
"""

import os
import sys

def fix_audit_script_imports():
    """Fix import issues in comprehensive_system_audit.py"""

    script_path = "scripts/health_check/comprehensive_system_audit.py"

    logger.info(f"🔧 Fixing import paths in {script_path}")

    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Fix the imports from core.kimera_system to src.core.kimera_system
    fixes = [
        ("from core.kimera_system import KimeraSystem", "from src.core.kimera_system import KimeraSystem"),
        ("import core.", "import src.core."),
        ("from core.", "from src.core."),
        ("import engines.", "import src.engines."),
        ("from engines.", "from src.engines."),
        ("import monitoring.", "import src.monitoring."),
        ("from monitoring.", "from src.monitoring."),
        ("import vault.", "import src.vault."),
        ("from vault.", "from src.vault."),
    ]

    for old, new in fixes:
        if old in content:
            content = content.replace(old, new)
            logger.info(f"✅ Fixed: {old} → {new}")

    # Write back
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(content)

    logger.info("✅ Import paths fixed")

def add_src_to_path():
    """Add src directory to Python path for all health check scripts"""

    health_check_dir = "scripts/health_check"

    # Template for path setup
    path_setup = '''# Fix import paths
import sys
import os
import logging
logger = logging.getLogger(__name__)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

'''

    # Process all Python files in health_check directory
    for filename in os.listdir(health_check_dir):
        if filename.endswith('.py') and not filename.startswith('__'):
            filepath = os.path.join(health_check_dir, filename)

            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check if path setup already exists
            if "sys.path.insert(0," not in content:
                # Add after imports
                lines = content.split('\n')
                import_section_end = 0

                for i, line in enumerate(lines):
                    if line.strip() and not line.startswith(('import ', 'from ', '#')) and import_section_end == 0:
                        import_section_end = i
                        break

                # Insert path setup
                lines.insert(import_section_end, path_setup)
                content = '\n'.join(lines)

                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)

                logger.info(f"✅ Added path setup to {filename}")

if __name__ == "__main__":
    fix_audit_script_imports()
    add_src_to_path()
    logger.info("🎉 All import issues fixed")
