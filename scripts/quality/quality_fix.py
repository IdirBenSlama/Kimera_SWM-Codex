#!/usr/bin/env python3
"""
KIMERA SWM Quality Fix Script
Automatically fixes common quality issues
"""

import logging
import subprocess
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_fix_command(cmd, description):
    """Run a fix command"""
    logger.info(f"🔧 {description}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"✅ {description} - FIXED")
            return True
        else:
            logger.warning(f"⚠️ {description} - PARTIAL/FAILED")
            logger.warning(result.stdout)
            return False
    except Exception as e:
        logger.error(f"❌ {description} - ERROR: {e}")
        return False


def main():
    """Run quality auto-fixes"""
    logger.info("🚀 KIMERA SWM Quality Auto-Fix")
    logger.info("=" * 40)

    fixes = [
        (
            "black --config config/quality/black.toml src/ tests/",
            "Code formatting (Black)",
        ),
        ("isort --profile black src/ tests/", "Import sorting (isort)"),
        (
            "ruff check --fix --config config/quality/ruff.toml src/ tests/",
            "Auto-fixable linting issues (Ruff)",
        ),
    ]

    fixed = 0

    for cmd, description in fixes:
        if run_fix_command(cmd, description):
            fixed += 1

    logger.info("=" * 40)
    logger.info(f"🔧 Auto-fixed {fixed}/{len(fixes)} quality categories")
    logger.info("💡 Run quality_check.py to verify all fixes")


if __name__ == "__main__":
    main()
