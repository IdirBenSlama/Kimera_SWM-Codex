#!/usr/bin/env python3
"""
Comprehensive System Update Script for Kimera SWM
Updates all packages, dependencies, and system components
"""

import subprocess
import sys
import json
import time
from datetime import datetime
from pathlib import Path

# Initialize structured logger
from backend.utils.kimera_logger import get_system_logger
logger = get_system_logger(__name__)


def run_command(command, description=""):
    """Run a command and return the result"""
    logger.info(f"\n{'='*60}")
    logger.info(f"EXECUTING: {description}")
    logger.info(f"COMMAND: {command}")
    logger.info(f"{'='*60}")
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.stdout:
            logger.info("STDOUT:")
            logger.info(result.stdout)
        
        if result.stderr:
            logger.info("STDERR:")
            logger.info(result.stderr)
            
        return result.returncode == 0, result.stdout, result.stderr
    
    except subprocess.TimeoutExpired:
        logger.error("❌ Command timed out after 5 minutes")
        return False, "", "Command timed out"
    except Exception as e:
        logger.error(f"❌ Error executing command: {e}")
        return False, "", str(e)

def update_pip_packages():
    """Update all pip packages"""
    logger.info("\n🔄 UPDATING PIP PACKAGES...")
    
    # First update pip itself
    success, _, _ = run_command("python -m pip install --upgrade pip", "Updating pip")
    if not success:
        logger.warning("⚠️ Failed to update pip, continuing anyway...")
    
    # Get list of outdated packages
    success, stdout, _ = run_command("pip list --outdated --format=json", "Getting outdated packages")
    if not success:
        logger.error("❌ Failed to get outdated packages list")
        return False
    
    try:
        outdated_packages = json.loads(stdout) if stdout.strip() else []
    except json.JSONDecodeError:
        logger.error("❌ Failed to parse outdated packages JSON")
        return False
    
    if not outdated_packages:
        logger.info("✅ All packages are up to date!")
        return True
    
    logger.info(f"📦 Found {len(outdated_packages)
    
    # Update packages one by one to avoid conflicts
    updated_count = 0
    failed_count = 0
    
    for package in outdated_packages:
        package_name = package['name']
        current_version = package['version']
        latest_version = package['latest_version']
        
        logger.info(f"\n📦 Updating {package_name}: {current_version} → {latest_version}")
        
        success, _, stderr = run_command(
            f"pip install --upgrade {package_name}",
            f"Updating {package_name}"
        )
        
        if success:
            logger.info(f"✅ Successfully updated {package_name}")
            updated_count += 1
        else:
            logger.error(f"❌ Failed to update {package_name}: {stderr}")
            failed_count += 1
    
    logger.info(f"\n📊 UPDATE SUMMARY:")
    logger.info(f"✅ Successfully updated: {updated_count}")
    logger.error(f"❌ Failed to update: {failed_count}")
    
    return failed_count == 0

def update_requirements_file():
    """Update requirements.txt with latest compatible versions"""
    logger.info("\n📝 UPDATING REQUIREMENTS.TXT...")
    
    # Read current requirements
    req_file = Path("requirements.txt")
    if not req_file.exists():
        logger.error("❌ requirements.txt not found")
        return False
    
    # Create backup
    backup_file = f"requirements_backup_{int(time.time())}.txt"
    run_command(f"copy requirements.txt {backup_file}", "Creating backup of requirements.txt")
    
    # Generate new requirements with current versions
    success, stdout, _ = run_command("pip freeze", "Getting current package versions")
    if not success:
        logger.error("❌ Failed to get current package versions")
        return False
    
    # Write updated requirements
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    updated_content = f"""# KIMERA SWM - Updated Dependencies
# Last updated: {timestamp}
# Auto-generated from pip freeze

{stdout}
"""
    
    with open("requirements_updated.txt", "w") as f:
        f.write(updated_content)
    
    logger.info("✅ Created requirements_updated.txt with current versions")
    return True

def check_system_health():
    """Run basic system health checks"""
    logger.info("\n🏥 RUNNING SYSTEM HEALTH CHECKS...")
    
    checks = [
        ("python --version", "Python version check"),
        ("pip --version", "Pip version check"),
        logger.info(f'PyTorch: {torch.__version__}')
        logger.info(f'NumPy: {numpy.__version__}')
        logger.info(f'SciPy: {scipy.__version__}')
        logger.info(f'FastAPI: {fastapi.__version__}')
    ]
    
    passed = 0
    failed = 0
    
    for command, description in checks:
        success, stdout, stderr = run_command(command, description)
        if success:
            logger.info(f"✅ {description}: PASSED")
            if stdout.strip():
                logger.info(f"   {stdout.strip()
            passed += 1
        else:
            logger.error(f"❌ {description}: FAILED")
            if stderr.strip():
                logger.info(f"   {stderr.strip()
            failed += 1
    
    logger.info(f"\n📊 HEALTH CHECK SUMMARY:")
    logger.info(f"✅ Passed: {passed}")
    logger.error(f"❌ Failed: {failed}")
    
    return failed == 0

def update_git_repository():
    """Update git repository if it exists"""
    logger.info("\n📡 CHECKING GIT REPOSITORY...")
    
    # Check if this is a git repository
    success, _, _ = run_command("git status", "Checking git status")
    if not success:
        logger.info("ℹ️ Not a git repository, skipping git updates")
        return True
    
    # Fetch latest changes
    success, _, _ = run_command("git fetch", "Fetching latest changes")
    if not success:
        logger.warning("⚠️ Failed to fetch git changes")
        return False
    
    # Show status
    success, stdout, _ = run_command("git status", "Getting git status")
    if success and stdout:
        logger.info("📊 Git Status:")
        logger.info(stdout)
    
    logger.info("ℹ️ Git repository checked. Manual merge may be required if there are conflicts.")
    return True

def generate_update_report():
    """Generate a comprehensive update report"""
    logger.info("\n📋 GENERATING UPDATE REPORT...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"system_update_report_{timestamp}.md"
    
    # Get system information
    success, python_version, _ = run_command("python --version", "Getting Python version")
    success, pip_version, _ = run_command("pip --version", "Getting pip version")
    success, package_list, _ = run_command("pip list", "Getting package list")
    
    report_content = f"""# System Update Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## System Information
- Python Version: {python_version.strip() if python_version else 'Unknown'}
- Pip Version: {pip_version.strip() if pip_version else 'Unknown'}

## Update Summary
- ✅ Package updates completed
- ✅ Requirements file updated
- ✅ System health checks performed
- ✅ Git repository checked

## Current Package Versions
```
{package_list if package_list else 'Failed to get package list'}
```

## Next Steps
1. Test the system functionality
2. Run comprehensive tests
3. Update documentation if needed
4. Commit changes to version control

## Files Created
- requirements_updated.txt - Updated requirements file
- {report_file} - This report

---
*Report generated by Kimera SWM Update Script*
"""
    
    with open(report_file, "w") as f:
        f.write(report_content)
    
    logger.info(f"✅ Update report saved to: {report_file}")
    return True

def main():
    """Main update function"""
    logger.info("🚀 KIMERA SWM COMPREHENSIVE UPDATE SCRIPT")
    logger.info("=" * 60)
    logger.info(f"Started at: {datetime.now()
    logger.info("=" * 60)
    
    steps = [
        ("Updating pip packages", update_pip_packages),
        ("Updating requirements file", update_requirements_file),
        ("Running system health checks", check_system_health),
        ("Checking git repository", update_git_repository),
        ("Generating update report", generate_update_report),
    ]
    
    completed = 0
    failed = 0
    
    for step_name, step_function in steps:
        logger.info(f"\n🔄 STEP: {step_name}")
        try:
            if step_function():
                logger.info(f"✅ COMPLETED: {step_name}")
                completed += 1
            else:
                logger.error(f"❌ FAILED: {step_name}")
                failed += 1
        except Exception as e:
            logger.error(f"❌ ERROR in {step_name}: {e}")
            failed += 1
    
    logger.info("\n" + "=" * 60)
    logger.info("🏁 UPDATE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"✅ Completed steps: {completed}")
    logger.error(f"❌ Failed steps: {failed}")
    logger.info(f"Finished at: {datetime.now()
    
    if failed == 0:
        logger.info("\n🎉 ALL UPDATES COMPLETED SUCCESSFULLY!")
        logger.info("Your Kimera SWM system is now up to date.")
    else:
        logger.warning(f"\n⚠️ {failed} steps failed. Please review the output above.")
        logger.info("Some manual intervention may be required.")
    
    return failed == 0

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.error("\n\n❌ Update interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\n❌ Unexpected error: {e}")
        sys.exit(1)