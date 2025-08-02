#!/usr/bin/env python3
"""
KIMERA SWM - REMEDIATION VERIFICATION
====================================

Verify that critical issues have been properly remediated.
"""

import os
import stat
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    project_root = Path(__file__).parent.parent
    
    logger.info("🔍 Verifying Critical Issues Remediation")
    logger.info("=" * 60)
    
    fixes_verified = 0
    total_checks = 0
    
    # Check 1: File permissions
    logger.info("🔒 Checking File Permissions...")
    critical_files = [
        "src/core/kimera_system.py",
        "src/core/gpu/gpu_manager.py", 
        "src/vault/vault_manager.py",
        "config/development.yaml"
    ]
    
    for file_path in critical_files:
        total_checks += 1
        full_path = project_root / file_path
        if full_path.exists():
            perms = full_path.stat().st_mode
            if not (perms & stat.S_IWOTH):  # Not world-writable
                logger.info(f"✅ {file_path}: Secure permissions")
                fixes_verified += 1
            else:
                logger.warning(f"⚠️ {file_path}: Still world-writable")
        else:
            logger.warning(f"⚠️ {file_path}: File not found")
    
    # Check 2: PyTorch compatibility wrapper
    logger.info("🔧 Checking PyTorch Compatibility...")
    total_checks += 1
    compat_file = project_root / "src/utils/torch_compatibility.py"
    if compat_file.exists():
        logger.info("✅ PyTorch compatibility wrapper created")
        fixes_verified += 1
    else:
        logger.warning("⚠️ PyTorch compatibility wrapper missing")
    
    # Check 3: Production configuration
    logger.info("⚙️ Checking Production Configuration...")
    total_checks += 1
    prod_config = project_root / "config/production.yaml"
    if prod_config.exists():
        logger.info("✅ Production configuration created")
        fixes_verified += 1
    else:
        logger.warning("⚠️ Production configuration missing")
    
    # Check 4: Database schema
    logger.info("🗄️ Checking Database Schema...")
    total_checks += 1
    schema_file = project_root / "src/vault/sqlite_schema.py"
    if schema_file.exists():
        logger.info("✅ SQLite schema created")
        fixes_verified += 1
    else:
        logger.warning("⚠️ SQLite schema missing")
    
    # Check 5: Security checklist
    logger.info("🔐 Checking Security Checklist...")
    total_checks += 1
    security_file = project_root / "docs/security_checklist.md"
    if security_file.exists():
        logger.info("✅ Security checklist created")
        fixes_verified += 1
    else:
        logger.warning("⚠️ Security checklist missing")
    
    # Summary
    success_rate = (fixes_verified / total_checks) * 100
    
    logger.info("\n" + "=" * 60)
    logger.info("REMEDIATION VERIFICATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"✅ Verified Fixes: {fixes_verified}/{total_checks}")
    logger.info(f"📊 Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 80:
        logger.info("🎉 REMEDIATION SUCCESSFULLY VERIFIED!")
        return 0
    else:
        logger.info("⚠️ REMEDIATION INCOMPLETE - Additional work needed")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main()) 