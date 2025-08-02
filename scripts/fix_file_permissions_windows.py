#!/usr/bin/env python3
"""
KIMERA SWM - WINDOWS FILE PERMISSIONS FIX
=========================================

Fix file permissions on Windows systems using appropriate Windows methods.
"""

import os
import stat
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_windows_file_permissions():
    """Fix file permissions on Windows"""
    project_root = Path(__file__).parent.parent
    
    logger.info("🔒 Fixing File Permissions on Windows...")
    
    # Critical files that need secure permissions
    critical_files = [
        "src/core/kimera_system.py",
        "src/core/gpu/gpu_manager.py", 
        "src/vault/vault_manager.py",
        "config/development.yaml",
        "config/production.yaml"
    ]
    
    fixes_applied = 0
    
    for file_path in critical_files:
        full_path = project_root / file_path
        if full_path.exists():
            try:
                # Method 1: Use Python stat module
                current_mode = full_path.stat().st_mode
                # Remove write permissions for group and others
                secure_mode = current_mode & ~(stat.S_IWGRP | stat.S_IWOTH)
                
                try:
                    full_path.chmod(secure_mode)
                    logger.info(f"✅ Fixed permissions for {file_path} (Python method)")
                    fixes_applied += 1
                    continue
                except Exception as e:
                    logger.debug(f"Python chmod failed for {file_path}: {e}")
                
                # Method 2: Use Windows icacls command
                try:
                    # Remove write permissions for Everyone group
                    cmd = f'icacls "{full_path}" /remove:g "Everyone:(W)" /inheritance:r'
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        logger.info(f"✅ Fixed permissions for {file_path} (icacls method)")
                        fixes_applied += 1
                    else:
                        logger.warning(f"⚠️ icacls failed for {file_path}: {result.stderr}")
                        
                except Exception as e:
                    logger.warning(f"⚠️ icacls method failed for {file_path}: {e}")
                
                # Method 3: Use attrib command to set read-only
                try:
                    cmd = f'attrib +R "{full_path}"'
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        logger.info(f"✅ Set read-only for {file_path} (attrib method)")
                        fixes_applied += 1
                    else:
                        logger.warning(f"⚠️ attrib failed for {file_path}: {result.stderr}")
                        
                except Exception as e:
                    logger.warning(f"⚠️ attrib method failed for {file_path}: {e}")
                
            except Exception as e:
                logger.error(f"❌ All methods failed for {file_path}: {e}")
        else:
            logger.warning(f"⚠️ File not found: {file_path}")
    
    # Fix database permissions
    db_path = project_root / "data/database/kimera_system.db"
    if db_path.exists():
        try:
            # Make database file read-only for others
            current_mode = db_path.stat().st_mode
            secure_mode = current_mode & ~(stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH)
            db_path.chmod(secure_mode)
            logger.info("✅ Fixed database file permissions")
            fixes_applied += 1
        except Exception as e:
            logger.warning(f"⚠️ Database permissions fix failed: {e}")
    
    logger.info(f"\n📊 File Permissions Summary: {fixes_applied} fixes applied")
    
    return fixes_applied > 0

def verify_permissions():
    """Verify that permissions have been fixed"""
    project_root = Path(__file__).parent.parent
    
    logger.info("🔍 Verifying File Permissions...")
    
    critical_files = [
        "src/core/kimera_system.py",
        "src/core/gpu/gpu_manager.py", 
        "src/vault/vault_manager.py",
        "config/development.yaml"
    ]
    
    secure_files = 0
    
    for file_path in critical_files:
        full_path = project_root / file_path
        if full_path.exists():
            perms = full_path.stat().st_mode
            
            # Check if world-writable
            is_world_writable = bool(perms & stat.S_IWOTH)
            
            if not is_world_writable:
                logger.info(f"✅ {file_path}: Secure permissions")
                secure_files += 1
            else:
                logger.warning(f"⚠️ {file_path}: Still world-writable")
        else:
            logger.warning(f"⚠️ {file_path}: File not found")
    
    success_rate = (secure_files / len(critical_files)) * 100
    logger.info(f"\n📊 Security Rate: {success_rate:.1f}% ({secure_files}/{len(critical_files)} files secure)")
    
    return success_rate >= 75

def main():
    """Main function"""
    logger.info("🔒 Windows File Permissions Security Fix")
    logger.info("=" * 50)
    
    # Apply fixes
    fixes_successful = fix_windows_file_permissions()
    
    # Verify fixes
    verification_passed = verify_permissions()
    
    if fixes_successful and verification_passed:
        logger.info("\n🎉 FILE PERMISSIONS SUCCESSFULLY SECURED!")
        return 0
    elif fixes_successful:
        logger.info("\n⚠️ FIXES APPLIED BUT VERIFICATION INCOMPLETE")
        return 1
    else:
        logger.info("\n❌ PERMISSION FIXES FAILED")
        return 2

if __name__ == "__main__":
    import sys
    sys.exit(main()) 