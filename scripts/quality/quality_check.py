#!/usr/bin/env python3
"""
KIMERA SWM Quality Check Script
Runs comprehensive quality analysis
"""

import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_command(cmd, description):
    """Run a command and return success status"""
    logger.info(f"🔍 {description}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"✅ {description} - PASSED")
            return True
        else:
            logger.error(f"❌ {description} - FAILED")
            logger.error(result.stdout)
            logger.error(result.stderr)
            return False
    except Exception as e:
        logger.error(f"❌ {description} - ERROR: {e}")
        return False


def main():
    """Run comprehensive quality checks"""
    logger.info("🚀 KIMERA SWM Comprehensive Quality Check")
    logger.info("=" * 50)

    checks = [
        (
            "black --check --config config/quality/black.toml src/ tests/",
            "Code formatting (Black)",
        ),
        ("isort --check-only --profile black src/ tests/", "Import sorting (isort)"),
        (
            "ruff check --config config/quality/ruff.toml src/ tests/",
            "Code linting (Ruff)",
        ),
        ("mypy --config-file config/quality/mypy.ini src/", "Type checking (MyPy)"),
        ("bandit -r src/ -c config/quality/bandit.yaml", "Security analysis (Bandit)"),
        ("safety check", "Dependency vulnerabilities (Safety)"),
        (
            "pytest --cov=src --cov-report=term-missing -c config/quality/pytest.toml",
            "Test suite with coverage",
        ),
    ]

    passed = 0
    total = len(checks)

    for cmd, description in checks:
        if run_command(cmd, description):
            passed += 1

    logger.info("=" * 50)
    logger.info(f"📊 Quality Check Results: {passed}/{total} checks passed")

    if passed == total:
        logger.info("🎉 ALL QUALITY CHECKS PASSED - OUTSTANDING!")
        sys.exit(0)
    else:
        logger.error(f"❌ {total - passed} quality checks failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
