"""
Script to run pytest with the correct virtual environment.
This bypasses shell activation issues by directly using the venv Python executable.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    # Get the project root directory
    project_root = Path(__file__).parent.absolute()
    
    # Determine the venv Python executable based on the platform
    if sys.platform == "win32":
        venv_python = project_root / ".venv" / "Scripts" / "python.exe"
    else:
        venv_python = project_root / ".venv" / "bin" / "python"
    
    # Check if venv exists
    if not venv_python.exists():
        print(f"ERROR: Virtual environment Python not found at {venv_python}")
        print("Please create a virtual environment first with: python -m venv .venv")
        return 1
    
    # Prepare the command
    cmd = [str(venv_python), "-m", "pytest"]
    
    # Add any additional pytest arguments passed to this script
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])
    
    print(f"Running: {' '.join(cmd)}")
    print(f"Working directory: {project_root}")
    print("-" * 80)
    
    # Run pytest
    try:
        result = subprocess.run(cmd, cwd=str(project_root))
        return result.returncode
    except Exception as e:
        print(f"ERROR: Failed to run pytest: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())