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
    print(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    try:
        result = subprocess.run(cmd, shell=isinstance(cmd, str), cwd=cwd, 
                              capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(f"STDERR: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"ERROR: {e}")
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
    
    print(f"Project root: {project_root}")
    print(f"Platform: {sys.platform}")
    
    # Check if venv exists
    if not venv_python.exists():
        print("\n1. Creating virtual environment...")
        if not run_command([sys.executable, "-m", "venv", ".venv"], cwd=str(project_root)):
            print("Failed to create virtual environment!")
            return 1
    else:
        print("\n1. Virtual environment already exists.")
    
    # Upgrade pip
    print("\n2. Upgrading pip...")
    run_command([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"])
    
    # Install requirements
    requirements_file = project_root / "requirements.txt"
    if requirements_file.exists():
        print("\n3. Installing requirements...")
        if not run_command([str(venv_pip), "install", "-r", "requirements.txt"], 
                          cwd=str(project_root)):
            print("Failed to install some requirements, but continuing...")
    else:
        print("\n3. No requirements.txt found!")
    
    # Install additional testing dependencies that might be missing
    print("\n4. Installing additional testing dependencies...")
    test_deps = [
        "pytest>=7.0.0",
        "pytest-asyncio",
        "pytest-mock",
        "pytest-cov",
        "httpx",  # For FastAPI testing
        "python-multipart",  # For FastAPI file uploads
    ]
    
    for dep in test_deps:
        print(f"   Installing {dep}...")
        run_command([str(venv_pip), "install", dep])
    
    # Verify installation
    print("\n5. Verifying installation...")
    verification_script = f'''
import sys
print(f"Python: {{sys.executable}}")
print(f"Version: {{sys.version}}")
print(f"Virtual env: {{sys.prefix != sys.base_prefix}}")

try:
    import pytest
    print(f"pytest: {{pytest.__version__}}")
except ImportError:
    print("pytest: NOT INSTALLED")

try:
    import numpy
    print(f"numpy: {{numpy.__version__}}")
except ImportError:
    print("numpy: NOT INSTALLED")

try:
    import torch
    print(f"torch: {{torch.__version__}}")
except ImportError:
    print("torch: NOT INSTALLED")
'''
    
    result = subprocess.run([str(venv_python), "-c", verification_script], 
                          capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"STDERR: {result.stderr}")
    
    print("\n6. Setup complete!")
    print(f"\nTo run tests, use: python run_tests.py")
    print(f"Or directly: {venv_python} -m pytest")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())