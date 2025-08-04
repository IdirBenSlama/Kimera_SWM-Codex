"""
Script to set up and configure the virtual environment properly.
This ensures all dependencies are installed in the venv, not globally.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, cwd=None):
    """Run a command and return success status."""
    logger.info(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    try:
        result = subprocess.run(cmd, shell=isinstance(cmd, str), cwd=cwd, 
                              capture_output=True, text=True)
        if result.stdout:
            logger.info(result.stdout)
        if result.stderr:
            logger.info(f"STDERR: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        logger.info(f"ERROR: {e}")
        return False

def main():
    project_root = Path(__file__).parent.absolute()
    
    # Determine paths based on platform
    if sys.platform == "win32":
        venv_python = project_root / ".venv" / "Scripts" / "python.exe"
        venv_pip = project_root / ".venv" / "Scripts" / "pip.exe"
    else:
        venv_python = project_root / ".venv" / "bin" / "python"
        venv_pip = project_root / ".venv" / "bin" / "pip"
    
    logger.info(f"Project root: {project_root}")
    logger.info(f"Platform: {sys.platform}")
    
    # Check if venv exists
    if not venv_python.exists():
        logger.info("\n1. Creating virtual environment...")
        if not run_command([sys.executable, "-m", "venv", ".venv"], cwd=str(project_root)):
            logger.info("Failed to create virtual environment!")
            return 1
    else:
        logger.info("\n1. Virtual environment already exists.")
    
    # Upgrade pip
    logger.info("\n2. Upgrading pip...")
    run_command([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"])
    
    # Install requirements
    requirements_file = project_root / "requirements.txt"
    if requirements_file.exists():
        logger.info("\n3. Installing requirements...")
        if not run_command([str(venv_pip), "install", "-r", "requirements.txt"], 
                          cwd=str(project_root)):
            logger.info("Failed to install some requirements, but continuing...")
    else:
        logger.info("\n3. No requirements.txt found!")
    
    # Install additional testing dependencies that might be missing
    logger.info("\n4. Installing additional testing dependencies...")
    test_deps = [
        "pytest>=7.0.0",
        "pytest-asyncio",
        "pytest-mock",
        "pytest-cov",
        "httpx",  # For FastAPI testing
        "python-multipart",  # For FastAPI file uploads
    ]
    
    for dep in test_deps:
        logger.info(f"   Installing {dep}...")
        run_command([str(venv_pip), "install", dep])
    
    # Verify installation
    logger.info("\n5. Verifying installation...")
    verification_script = f'''
import sys
logger.info(f"Python: {{sys.executable}}")
logger.info(f"Version: {{sys.version}}")
logger.info(f"Virtual env: {{sys.prefix != sys.base_prefix}}")

try:
    import pytest
    logger.info(f"pytest: {{pytest.__version__}}")
except ImportError:
    logger.info("pytest: NOT INSTALLED")

try:
    import numpy
    logger.info(f"numpy: {{numpy.__version__}}")
except ImportError:
    logger.info("numpy: NOT INSTALLED")

try:
    import torch
import logging
logger = logging.getLogger(__name__)
    logger.info(f"torch: {{torch.__version__}}")
except ImportError:
    logger.info("torch: NOT INSTALLED")
'''
    
    result = subprocess.run([str(venv_python), "-c", verification_script], 
                          capture_output=True, text=True)
    logger.info(result.stdout)
    if result.stderr:
        logger.info(f"STDERR: {result.stderr}")
    
    logger.info("\n6. Setup complete!")
    logger.info(f"\nTo run tests, use: python run_tests.py")
    logger.info(f"Or directly: {venv_python} -m pytest")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())