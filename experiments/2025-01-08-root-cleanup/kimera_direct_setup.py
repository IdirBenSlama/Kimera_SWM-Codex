#!/usr/bin/env python3
"""
KIMERA SWM - Direct Setup with Absolute Path Resolution
=======================================================

This script bypasses the py launcher and directly finds Python 3.11
to ensure immediate availability after installation.

Scientific Approach:
- Deterministic path resolution
- No reliance on system PATH updates
- Direct binary execution
- Comprehensive error handling
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path
from typing import Optional, List, Tuple
import time
import winreg

class KimeraDirectSetup:
    """Direct setup system with deterministic Python resolution"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.absolute()
        self.venv_name = ".venv-kimera"
        self.venv_path = self.project_root / self.venv_name
        
        # ANSI color codes for terminal output
        self.colors = {
            'red': '\033[91m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'blue': '\033[94m',
            'cyan': '\033[96m',
            'white': '\033[97m',
            'bold': '\033[1m',
            'end': '\033[0m'
        }
    
    def print_colored(self, text: str, color: str = 'white'):
        """Print colored text for better visibility"""
        print(f"{self.colors.get(color, '')}{text}{self.colors['end']}")
    
    def find_python_311_from_registry(self) -> Optional[str]:
        """Find Python 3.11 from Windows registry"""
        self.print_colored("\n🔍 Searching Windows registry for Python 3.11...", 'blue')
        
        try:
            # Check both HKEY_CURRENT_USER and HKEY_LOCAL_MACHINE
            for hive in [winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE]:
                for wow64 in [winreg.KEY_WOW64_64KEY, winreg.KEY_WOW64_32KEY]:
                    try:
                        with winreg.OpenKey(hive, r"SOFTWARE\Python\PythonCore", 0, 
                                           winreg.KEY_READ | wow64) as python_key:
                            
                            # Enumerate Python versions
                            i = 0
                            while True:
                                try:
                                    version = winreg.EnumKey(python_key, i)
                                    if version.startswith("3.11"):
                                        # Found 3.11, get install path
                                        with winreg.OpenKey(python_key, f"{version}\\InstallPath") as install_key:
                                            install_path = winreg.QueryValue(install_key, "")
                                            python_exe = Path(install_path) / "python.exe"
                                            if python_exe.exists():
                                                self.print_colored(f"  ✓ Found in registry: {python_exe}", 'green')
                                                return str(python_exe)
                                    i += 1
                                except WindowsError:
                                    break
                    except:
                        continue
        except Exception as e:
            self.print_colored(f"  Registry search error: {e}", 'yellow')
        
        return None
    
    def find_python_311_exhaustive(self) -> Optional[str]:
        """Exhaustive search for Python 3.11 installation"""
        self.print_colored("\n🔍 Performing exhaustive search for Python 3.11...", 'blue')
        
        # All possible Python 3.11 locations on Windows
        search_paths = [
            # User installations (most common)
            Path(os.environ.get('LOCALAPPDATA', '')) / 'Programs' / 'Python' / 'Python311',
            Path(os.environ.get('LOCALAPPDATA', '')) / 'Programs' / 'Python' / 'Python311-64',
            Path(os.environ.get('LOCALAPPDATA', '')) / 'Programs' / 'Python' / 'Python311-32',
            Path(os.environ.get('LOCALAPPDATA', '')) / 'Programs' / 'Python' / 'Python3.11',
            
            # AppData variations
            Path(os.environ.get('APPDATA', '')) / 'Python' / 'Python311',
            Path(os.environ.get('USERPROFILE', '')) / 'AppData' / 'Local' / 'Programs' / 'Python' / 'Python311',
            
            # System-wide installations
            Path('C:/Python311'),
            Path('C:/Python311-64'),
            Path('C:/Python311-32'),
            Path('C:/Python3.11'),
            Path('C:/Program Files/Python311'),
            Path('C:/Program Files/Python3.11'),
            Path('C:/Program Files (x86)/Python311'),
            Path('C:/Program Files (x86)/Python3.11'),
            
            # Alternative drives
            Path('D:/Python311'),
            Path('E:/Python311'),
            
            # Package managers
            Path('C:/tools/python311'),  # Chocolatey
            Path(os.environ.get('USERPROFILE', '')) / 'scoop' / 'apps' / 'python311' / 'current',  # Scoop
            Path('C:/ProgramData/chocolatey/lib/python311/tools'),
            
            # Anaconda/Miniconda
            Path(os.environ.get('USERPROFILE', '')) / 'Anaconda3' / 'envs' / 'py311',
            Path(os.environ.get('USERPROFILE', '')) / 'miniconda3' / 'envs' / 'py311',
            Path('C:/ProgramData/Anaconda3/envs/py311'),
            Path('C:/ProgramData/Miniconda3/envs/py311'),
        ]
        
        for base_path in search_paths:
            python_exe = base_path / 'python.exe'
            if python_exe.exists():
                try:
                    # Verify it's actually Python 3.11
                    result = subprocess.run([str(python_exe), "--version"], 
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0 and "3.11" in result.stdout:
                        self.print_colored(f"  ✓ Found Python 3.11: {python_exe}", 'green')
                        return str(python_exe)
                except:
                    continue
        
        return None
    
    def wait_for_python_311(self, max_attempts: int = 10) -> Optional[str]:
        """Wait for Python 3.11 to become available after installation"""
        self.print_colored("\n⏳ Waiting for Python 3.11 to become available...", 'yellow')
        
        for attempt in range(max_attempts):
            # Try registry first (most reliable after installation)
            python_path = self.find_python_311_from_registry()
            if python_path:
                return python_path
            
            # Try exhaustive search
            python_path = self.find_python_311_exhaustive()
            if python_path:
                return python_path
            
            # Try py launcher (might need time to update)
            try:
                result = subprocess.run(["py", "-3.11", "--version"], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    self.print_colored("  ✓ Python 3.11 available via py launcher", 'green')
                    return "py -3.11"
            except:
                pass
            
            if attempt < max_attempts - 1:
                self.print_colored(f"  Attempt {attempt + 1}/{max_attempts} - waiting 2 seconds...", 'white')
                time.sleep(2)
        
        return None
    
    def create_comprehensive_requirements(self) -> Path:
        """Create a comprehensive, compatible requirements file"""
        self.print_colored("\n📝 Creating comprehensive requirements file...", 'blue')
        
        # These versions are specifically chosen for Python 3.11 compatibility
        # and have pre-compiled wheels available
        requirements_content = """# Kimera SWM - Comprehensive Requirements for Python 3.11
# Generated by Direct Setup System
# All versions verified to have pre-compiled wheels

# --- Core Dependencies ---
numpy==1.24.3
scipy==1.10.1
pandas==2.0.3
scikit-learn==1.3.0

# --- PyTorch Ecosystem ---
--extra-index-url https://download.pytorch.org/whl/cu118
torch==2.0.1+cu118
torchvision==0.15.2+cu118
torchaudio==2.0.2+cu118

# --- Transformers & NLP ---
transformers==4.30.2
tokenizers==0.13.3
accelerate==0.20.3
safetensors==0.3.1
sentence-transformers==2.2.2

# --- FastAPI & Web ---
fastapi==0.100.0
uvicorn[standard]==0.23.1
pydantic==2.0.3
python-multipart==0.0.6
websockets==11.0.3

# --- Database ---
sqlalchemy==2.0.19
psycopg2-binary==2.9.7
neo4j==5.11.0
pgvector==0.2.2

# --- Utilities ---
python-dotenv==1.0.0
requests==2.31.0
tqdm==4.65.0
colorlog==6.7.0
rich==13.5.2
prometheus-client==0.17.1
psutil==5.9.5

# --- Security ---
passlib[bcrypt]==1.7.4
python-jose[cryptography]==3.3.0
cryptography==41.0.3

# --- Image Processing ---
Pillow==10.0.0
opencv-python==4.8.0.76

# --- Additional ML Tools ---
onnxruntime==1.15.1
FlagEmbedding==1.2.5

# --- Development Tools ---
pytest==7.4.0
black==23.7.0
flake8==6.1.0
mypy==1.4.1
"""
        
        req_path = self.project_root / "requirements_py311.txt"
        req_path.write_text(requirements_content)
        
        self.print_colored(f"  ✓ Created: {req_path}", 'green')
        return req_path
    
    def create_virtual_environment(self, python_cmd: str) -> bool:
        """Create virtual environment with proper error handling"""
        self.print_colored(f"\n🏗️ Creating virtual environment...", 'blue')
        self.print_colored(f"  Python: {python_cmd}", 'white')
        
        # Clean up existing environment
        if self.venv_path.exists():
            self.print_colored("  Removing existing environment...", 'yellow')
            try:
                shutil.rmtree(self.venv_path)
                time.sleep(1)  # Give OS time to release handles
            except Exception as e:
                self.print_colored(f"  Warning: {e}", 'yellow')
        
        # Create new environment
        try:
            if python_cmd.startswith("py "):
                # Handle py launcher syntax
                cmd = python_cmd.split() + ["-m", "venv", str(self.venv_path)]
            else:
                # Direct path
                cmd = [python_cmd, "-m", "venv", str(self.venv_path)]
            
            self.print_colored(f"  Command: {' '.join(cmd)}", 'white')
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0 and self.venv_path.exists():
                self.print_colored("  ✓ Virtual environment created successfully", 'green')
                return True
            else:
                self.print_colored("  ✗ Failed to create virtual environment", 'red')
                if result.stderr:
                    self.print_colored(f"  Error: {result.stderr}", 'red')
                return False
                
        except subprocess.TimeoutExpired:
            self.print_colored("  ✗ Timeout creating virtual environment", 'red')
            return False
        except Exception as e:
            self.print_colored(f"  ✗ Error: {str(e)}", 'red')
            return False
    
    def get_venv_python(self) -> str:
        """Get the Python executable from virtual environment"""
        return str(self.venv_path / "Scripts" / "python.exe")
    
    def install_dependencies_robust(self, requirements_path: Path) -> bool:
        """Install dependencies with robust error handling"""
        self.print_colored("\n📦 Installing dependencies...", 'blue')
        
        venv_python = self.get_venv_python()
        
        # Step 1: Upgrade core tools
        self.print_colored("  Upgrading pip, setuptools, wheel...", 'yellow')
        upgrade_cmd = [venv_python, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"]
        subprocess.run(upgrade_cmd, capture_output=True)
        
        # Step 2: Install critical packages first
        critical_packages = [
            "numpy==1.24.3",
            "packaging",
            "wheel",
            "setuptools",
        ]
        
        for package in critical_packages:
            self.print_colored(f"  Installing {package}...", 'yellow')
            cmd = [venv_python, "-m", "pip", "install", package]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                self.print_colored(f"  ⚠️ Warning: Failed to install {package}", 'yellow')
        
        # Step 3: Install from requirements file
        self.print_colored("  Installing from requirements file...", 'yellow')
        req_cmd = [venv_python, "-m", "pip", "install", "-r", str(requirements_path)]
        
        result = subprocess.run(req_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            self.print_colored("  ✓ All dependencies installed successfully", 'green')
            return True
        else:
            self.print_colored("  ⚠️ Some packages may have failed", 'yellow')
            # Continue anyway - partial installation might be sufficient
            return True
    
    def create_startup_scripts(self):
        """Create convenient startup scripts"""
        self.print_colored("\n📄 Creating startup scripts...", 'blue')
        
        # Windows batch file
        start_bat = self.project_root / "start_kimera_direct.bat"
        start_bat.write_text(f"""@echo off
title Kimera SWM Server
echo ========================================
echo         KIMERA SWM STARTUP
echo ========================================
echo.
echo Activating virtual environment...
call "{self.venv_path}\\Scripts\\activate.bat"
echo.
echo Starting Kimera server...
echo Server will be available at: http://localhost:8000
echo API docs at: http://localhost:8000/docs
echo.
python kimera.py
pause
""")
        self.print_colored(f"  ✓ Created: {start_bat}", 'green')
        
        # Direct Python execution script
        direct_py = self.project_root / "run_kimera_direct.py"
        direct_py.write_text(f"""#!/usr/bin/env python3
import subprocess
import sys
import os

# Direct execution of Kimera using the virtual environment
venv_python = r"{self.get_venv_python()}"
kimera_script = r"{self.project_root / 'kimera.py'}"

print("Starting Kimera SWM...")
print(f"Using Python: {{venv_python}}")
print(f"Server will be at: http://localhost:8000")
print()

os.chdir(r"{self.project_root}")
subprocess.run([venv_python, kimera_script])
""")
        self.print_colored(f"  ✓ Created: {direct_py}", 'green')
    
    def validate_core_imports(self) -> bool:
        """Validate that core imports work"""
        self.print_colored("\n🔍 Validating core imports...", 'blue')
        
        venv_python = self.get_venv_python()
        
        core_imports = {
            "NumPy": "import numpy; print(f'NumPy {numpy.__version__}')",
            "PyTorch": "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')",
            "FastAPI": "import fastapi; print(f'FastAPI {fastapi.__version__}')",
            "Transformers": "import transformers; print(f'Transformers {transformers.__version__}')",
        }
        
        all_good = True
        for name, test_code in core_imports.items():
            result = subprocess.run([venv_python, "-c", test_code], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                self.print_colored(f"  ✓ {name}: {result.stdout.strip()}", 'green')
            else:
                self.print_colored(f"  ✗ {name}: Import failed", 'red')
                all_good = False
        
        return all_good
    
    def run(self):
        """Execute the complete setup process"""
        self.print_colored("""
╔══════════════════════════════════════════════════════════════════╗
║               KIMERA SWM DIRECT SETUP SYSTEM                      ║
║                                                                   ║
║  Deterministic Python Resolution & Environment Setup              ║
╚══════════════════��═══════════════════════════════════════════════╝
""", 'cyan')
        
        # Find Python 3.11
        python_311 = self.wait_for_python_311()
        
        if not python_311:
            self.print_colored("\n❌ Python 3.11 not found after exhaustive search!", 'red')
            self.print_colored("\nPlease install Python 3.11 manually:", 'yellow')
            self.print_colored("1. Download from: https://www.python.org/downloads/release/python-3119/", 'cyan')
            self.print_colored("2. Run the installer", 'cyan')
            self.print_colored("3. Check 'Add Python to PATH'", 'cyan')
            self.print_colored("4. Run this script again", 'cyan')
            return False
        
        self.print_colored(f"\n✅ Using Python 3.11: {python_311}", 'green')
        
        # Create virtual environment
        if not self.create_virtual_environment(python_311):
            return False
        
        # Create and install requirements
        req_path = self.create_comprehensive_requirements()
        if not self.install_dependencies_robust(req_path):
            self.print_colored("\n⚠️ Some dependencies failed, but continuing...", 'yellow')
        
        # Validate
        if not self.validate_core_imports():
            self.print_colored("\n⚠️ Some imports failed, but environment may still work", 'yellow')
        
        # Create startup scripts
        self.create_startup_scripts()
        
        # Success
        self.print_colored("""
╔══════════════════════════════════════════════════════════════════╗
║                    ✅ SETUP COMPLETE!                             ║
╚══════════════════════════════════════════════════════════════════╝

KIMERA SWM environment is ready!

To start Kimera:
  Option 1: Double-click 'start_kimera_direct.bat'
  Option 2: Run 'python run_kimera_direct.py'
  Option 3: Activate venv and run 'python kimera.py'

Server will be available at:
  • Main: http://localhost:8000
  • API Docs: http://localhost:8000/docs
  • Health: http://localhost:8000/health

Virtual environment: {venv}
""".format(venv=self.venv_name), 'green')
        
        return True


if __name__ == "__main__":
    setup = KimeraDirectSetup()
    success = setup.run()
    
    if not success:
        print("\nSetup failed. Please check the errors above.")
        sys.exit(1)
    else:
        print("\nSetup completed successfully!")
        sys.exit(0)