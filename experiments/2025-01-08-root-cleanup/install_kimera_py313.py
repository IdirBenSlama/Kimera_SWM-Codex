#!/usr/bin/env python3
"""
KIMERA SWM - Python 3.13 Direct Installation
============================================

This script installs Kimera with Python 3.13 compatible packages.
"""

import subprocess
import sys
import os
from pathlib import Path
import shutil

def run_command(cmd, description):
    """Run a command and handle output"""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"{'='*60}")
    
    if isinstance(cmd, str):
        cmd = cmd.split()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr and result.returncode != 0:
            print(f"⚠️ Warning: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║           KIMERA SWM - Python 3.13 Installation                   ║
╚══════════════════════════════════════════════════════════════════╝
""")
    
    project_root = Path(__file__).parent.absolute()
    os.chdir(project_root)
    
    # Step 1: Clean old virtual environments
    print("\n🧹 Cleaning old environments...")
    for venv in [".venv", "venv", ".venv-kimera", ".venv-kimera-clean"]:
        venv_path = project_root / venv
        if venv_path.exists():
            print(f"  Removing {venv}...")
            try:
                shutil.rmtree(venv_path)
            except:
                pass
    
    # Step 2: Create new virtual environment
    venv_name = "venv_py313"
    if not run_command(
        [sys.executable, "-m", "venv", venv_name],
        "Creating virtual environment"
    ):
        print("❌ Failed to create virtual environment")
        return False
    
    # Step 3: Get venv Python path
    if os.name == 'nt':  # Windows
        venv_python = str(project_root / venv_name / "Scripts" / "python.exe")
        venv_pip = str(project_root / venv_name / "Scripts" / "pip.exe")
    else:
        venv_python = str(project_root / venv_name / "bin" / "python")
        venv_pip = str(project_root / venv_name / "bin" / "pip")
    
    # Step 4: Upgrade pip
    run_command(
        [venv_python, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        "Upgrading pip, setuptools, and wheel"
    )
    
    # Step 5: Install packages in stages
    print("\n📦 Installing packages in stages...")
    
    # Stage 1: Core dependencies
    core_packages = [
        "numpy==2.1.3",
        "packaging==24.2",
        "pydantic==2.10.3",
        "fastapi==0.115.5",
        "uvicorn[standard]==0.32.1",
        "requests==2.32.3",
        "python-dotenv==1.0.1"
    ]
    
    for package in core_packages:
        if not run_command(
            [venv_pip, "install", package],
            f"Installing {package}"
        ):
            print(f"⚠️ Failed to install {package}, continuing...")
    
    # Stage 2: Database packages
    db_packages = [
        "psycopg2-binary==2.9.10",
        "sqlalchemy==2.0.36",
        "neo4j==5.26.0"
    ]
    
    for package in db_packages:
        run_command(
            [venv_pip, "install", package],
            f"Installing {package}"
        )
    
    # Stage 3: PyTorch CPU (to avoid CUDA issues)
    print("\n🤖 Installing PyTorch (CPU version for compatibility)...")
    run_command(
        [venv_pip, "install", "torch==2.5.1+cpu", "torchvision==0.20.1+cpu", 
         "torchaudio==2.5.1+cpu", "--index-url", "https://download.pytorch.org/whl/cpu"],
        "Installing PyTorch CPU"
    )
    
    # Stage 4: Transformers
    ml_packages = [
        "transformers==4.47.0",
        "tokenizers==0.21.0",
        "safetensors==0.4.5",
        "accelerate==1.2.1"
    ]
    
    for package in ml_packages:
        run_command(
            [venv_pip, "install", package],
            f"Installing {package}"
        )
    
    # Stage 5: Remaining packages
    run_command(
        [venv_pip, "install", "-r", "requirements_py313.txt"],
        "Installing remaining packages"
    )
    
    # Step 6: Create launcher scripts
    print("\n📄 Creating launcher scripts...")
    
    # Windows batch file
    bat_content = f"""@echo off
echo Starting Kimera SWM with Python 3.13...
call "{venv_name}\\Scripts\\activate.bat"
python kimera.py
pause
"""
    
    bat_file = project_root / "start_kimera_py313.bat"
    bat_file.write_text(bat_content)
    print(f"  ✓ Created: {bat_file}")
    
    # Python launcher
    py_launcher = f"""#!/usr/bin/env python3
import subprocess
import os

os.chdir(r"{project_root}")
subprocess.run([r"{venv_python}", "kimera.py"])
"""
    
    py_file = project_root / "run_kimera_py313.py"
    py_file.write_text(py_launcher)
    print(f"  ✓ Created: {py_file}")
    
    # Step 7: Validate installation
    print("\n🔍 Validating installation...")
    
    test_imports = [
        "import numpy",
        "import torch",
        "import fastapi",
        "import transformers"
    ]
    
    all_good = True
    for test in test_imports:
        result = subprocess.run(
            [venv_python, "-c", test],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"  ✓ {test}")
        else:
            print(f"  ✗ {test}")
            all_good = False
    
    # Final message
    print(f"""
{'='*60}
{'✅ INSTALLATION COMPLETE!' if all_good else '⚠️ INSTALLATION COMPLETED WITH WARNINGS'}
{'='*60}

Kimera SWM is ready to run with Python 3.13!

To start Kimera:
  1. Double-click: start_kimera_py313.bat
  2. Or run: python run_kimera_py313.py

Virtual environment: {venv_name}
Python: {sys.version}

Note: Using CPU-only PyTorch for maximum compatibility.
GPU support can be added later if needed.
""")
    
    return all_good

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)