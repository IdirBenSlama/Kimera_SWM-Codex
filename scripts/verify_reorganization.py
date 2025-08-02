#!/usr/bin/env python3
"""
KIMERA SWM Reorganization Verification Script
Comprehensive validation of the project reorganization.
"""

import os
import sys
import subprocess
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def count_files(directory: str, extension: str = "*.py") -> int:
    """Count files with given extension in directory."""
    if not os.path.exists(directory):
        return 0
    try:
        result = subprocess.run(
            f"find {directory} -name '{extension}' | wc -l", 
            shell=True, 
            capture_output=True, 
            text=True
        )
        return int(result.stdout.strip())
    except Exception as e:
        logger.error(f"Error in verify_reorganization.py: {e}", exc_info=True)
        raise  # Re-raise for proper error handling
        return 0

def check_imports():
    """Verify all imports work correctly."""
    sys.path.insert(0, '.')
    
    test_imports = [
        ("src.utils.config", "Basic utilities"),
        ("src.core.kimera_system", "Core system"),
        ("src.engines.activation_manager", "Engine systems"),
        ("src.api.auth_routes", "API components"),
        ("src.security.authentication", "Security modules"),
        ("src.monitoring.metrics_and_alerting", "Monitoring systems"),
        ("src.trading.core.trading_engine", "Trading systems"),
        ("src.vault.vault_manager", "Vault systems"),
    ]
    
    successful_imports = 0
    failed_imports = []
    
    for module_name, description in test_imports:
        try:
            __import__(module_name)
            logger.info(f"✅ {description}: {module_name}")
            successful_imports += 1
        except Exception as e:
            logger.error(f"❌ {description}: {module_name} - {e}")
            failed_imports.append((module_name, str(e)))
    
    return successful_imports, failed_imports

def verify_structure():
    """Verify the directory structure follows KIMERA protocol."""
    expected_structure = {
        'src': ['core', 'engines', 'api', 'security', 'monitoring', 'trading', 'utils', 'vault'],
        'tests': ['unit', 'integration', 'performance', 'adversarial'],
        'experiments': [],
        'archive': [],
        'docs': [],
        'config': [],
        'scripts': [],
    }
    
    structure_issues = []
    
    for main_dir, subdirs in expected_structure.items():
        if not os.path.exists(main_dir):
            structure_issues.append(f"Missing directory: {main_dir}")
            continue
            
        for subdir in subdirs:
            subdir_path = os.path.join(main_dir, subdir)
            if not os.path.exists(subdir_path):
                structure_issues.append(f"Missing subdirectory: {subdir_path}")
    
    return structure_issues

def check_backend_imports():
    """Check for any remaining backend imports."""
    try:
        result = subprocess.run(
            'grep -r "from backend\\." . --exclude-dir=archive --exclude-dir=backend 2>/dev/null | wc -l',
            shell=True,
            capture_output=True,
            text=True
        )
        return int(result.stdout.strip())
    except Exception as e:
        logger.error(f"Error in verify_reorganization.py: {e}", exc_info=True)
        raise  # Re-raise for proper error handling
        return -1

def main():
    """Main verification function."""
    logger.info("🔍 KIMERA SWM REORGANIZATION VERIFICATION")
    logger.info("=" * 60)
    
    # 1. Structure verification
    logger.info("1. Verifying directory structure...")
    structure_issues = verify_structure()
    if structure_issues:
        logger.warning("Structure issues found:")
        for issue in structure_issues:
            logger.warning(f"  ⚠️  {issue}")
    else:
        logger.info("  ✅ Directory structure is correct")
    
    # 2. File count verification
    logger.info("\n2. Verifying file distribution...")
    src_files = count_files("src")
    test_files = count_files("tests")
    exp_files = count_files("experiments")
    
    logger.info(f"  📁 src/: {src_files} Python files")
    logger.info(f"  🧪 tests/: {test_files} Python files")
    logger.info(f"  🔬 experiments/: {exp_files} Python files")
    
    # 3. Import verification
    logger.info("\n3. Verifying imports...")
    successful, failed = check_imports()
    logger.info(f"  ✅ Successful imports: {successful}")
    if failed:
        logger.warning(f"  ❌ Failed imports: {len(failed)}")
        for module, error in failed:
            logger.warning(f"    - {module}: {error}")
    
    # 4. Backend import check
    logger.info("\n4. Checking for remaining backend imports...")
    backend_imports = check_backend_imports()
    if backend_imports == 0:
        logger.info("  ✅ No remaining backend imports found")
    elif backend_imports > 0:
        logger.warning(f"  ⚠️  {backend_imports} backend imports still remain")
    else:
        logger.error("  ❌ Unable to check backend imports")
    
    # 5. Overall assessment
    logger.info("\n" + "=" * 60)
    logger.info("📊 VERIFICATION SUMMARY")
    logger.info("=" * 60)
    
    overall_score = 0
    total_checks = 4
    
    if not structure_issues:
        overall_score += 1
        logger.info("✅ Directory structure: PASS")
    else:
        logger.warning("⚠️  Directory structure: ISSUES FOUND")
    
    if src_files > 400:
        overall_score += 1
        logger.info("✅ File migration: PASS")
    else:
        logger.warning("⚠️  File migration: INCOMPLETE")
    
    if successful >= 6:
        overall_score += 1
        logger.info("✅ Import functionality: PASS")
    else:
        logger.warning("⚠️  Import functionality: ISSUES FOUND")
    
    if backend_imports == 0:
        overall_score += 1
        logger.info("✅ Import migration: COMPLETE")
    else:
        logger.warning("⚠️  Import migration: INCOMPLETE")
    
    # Final verdict
    logger.info("\n" + "=" * 60)
    if overall_score == total_checks:
        logger.info("🎉 REORGANIZATION STATUS: ✅ COMPLETE AND VERIFIED")
        logger.info("The KIMERA SWM project reorganization is successful!")
    elif overall_score >= 3:
        logger.info("⚠️  REORGANIZATION STATUS: 🟡 MOSTLY COMPLETE")
        logger.info("Minor issues detected, but system should be functional.")
    else:
        logger.error("❌ REORGANIZATION STATUS: 🔴 INCOMPLETE")
        logger.error("Significant issues detected, manual review needed.")
    
    logger.info(f"Overall Score: {overall_score}/{total_checks}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main() 