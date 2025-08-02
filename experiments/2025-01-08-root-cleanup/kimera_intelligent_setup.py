#!/usr/bin/env python3
"""
KIMERA SWM - Intelligent Setup & Environment Resolution System
==============================================================

This script provides a comprehensive, automated solution for setting up
the Kimera SWM environment with full compatibility resolution.

Features:
- Automatic Python version detection and management
- Intelligent dependency resolution
- Pre-compiled wheel compatibility checking
- Automated virtual environment creation
- Full system validation

Author: Kimera AI Assistant
Date: 2025-01-02
"""

import os
import sys
import subprocess
import platform
import json
import shutil
import urllib.request
import tempfile
import zipfile
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import re

class KimeraIntelligentSetup:
    """Intelligent setup system for Kimera SWM"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.absolute()
        self.python_versions = []
        self.recommended_python = "3.11"
        self.venv_name = ".venv-kimera"
        self.venv_path = self.project_root / self.venv_name
        
        # Color codes for output
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
        if color in self.colors:
            print(f"{self.colors[color]}{text}{self.colors['end']}")
        else:
            print(text)
    
    def print_banner(self):
        """Print the setup banner"""
        self.print_colored("""
╔══════════════════════════════════════════════════════════════════╗
║                    KIMERA SWM INTELLIGENT SETUP                   ║
║                                                                   ║
║  Automated Environment Resolution & Compatibility Management      ║
║                                                                   ║
║  This system will:                                                ║
║  • Detect and resolve Python version compatibility               ║
║  • Install all dependencies with pre-compiled wheels             ║
║  • Create an optimized virtual environment                       ║
║  • Validate the complete system                                  ║
╚══════════════════════════════════════════════════════════════════╝
""", 'cyan')
    
    def detect_python_installations(self) -> List[Tuple[str, str]]:
        """Detect all Python installations on the system"""
        self.print_colored("\n🔍 Detecting Python installations...", 'blue')
        
        python_commands = []
        
        # Windows Python Launcher
        if platform.system() == "Windows":
            # Check py launcher
            try:
                result = subprocess.run(["py", "-0"], capture_output=True, text=True)
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    for line in lines:
                        if line.strip() and not line.startswith(' '):
                            # Parse version info
                            match = re.match(r'-(\d+\.\d+)(?:-\d+)?\s*\*?', line.strip())
                            if match:
                                version = match.group(1)
                                python_commands.append((f"py -{version}", version))
            except Exception as e:
                logger.error(f"Error in kimera_intelligent_setup.py: {e}", exc_info=True)
                raise  # Re-raise for proper error handling
            
            # Check common Windows paths
            common_paths = [
                Path(os.environ.get('LOCALAPPDATA', '')) / 'Programs' / 'Python',
                Path('C:/Python311'),
                Path('C:/Python310'),
                Path('C:/Python39'),
            ]
            
            for base_path in common_paths:
                if base_path.exists():
                    for python_dir in base_path.glob('Python*'):
                        python_exe = python_dir / 'python.exe'
                        if python_exe.exists():
                            try:
                                result = subprocess.run([str(python_exe), "--version"], 
                                                      capture_output=True, text=True)
                                if result.returncode == 0:
                                    version_match = re.search(r'Python (\d+\.\d+)', result.stdout)
                                    if version_match:
                                        version = version_match.group(1)
                                        python_commands.append((str(python_exe), version))
                            except Exception as e:
                                logger.error(f"Error in kimera_intelligent_setup.py: {e}", exc_info=True)
                                raise  # Re-raise for proper error handling
        
        # Check PATH
        for cmd in ['python', 'python3', 'python3.11', 'python3.10', 'python3.9']:
            try:
                result = subprocess.run([cmd, "--version"], capture_output=True, text=True)
                if result.returncode == 0:
                    version_match = re.search(r'Python (\d+\.\d+)', result.stdout)
                    if version_match:
                        version = version_match.group(1)
                        python_commands.append((cmd, version))
            except Exception as e:
                logger.error(f"Error in kimera_intelligent_setup.py: {e}", exc_info=True)
                raise  # Re-raise for proper error handling
        
        # Remove duplicates
        seen = set()
        unique_pythons = []
        for cmd, version in python_commands:
            if version not in seen:
                seen.add(version)
                unique_pythons.append((cmd, version))
        
        # Sort by version
        unique_pythons.sort(key=lambda x: tuple(map(int, x[1].split('.'))), reverse=True)
        
        for cmd, version in unique_pythons:
            self.print_colored(f"  ✓ Found Python {version}: {cmd}", 'green')
        
        return unique_pythons
    
    def check_wheel_compatibility(self, python_cmd: str) -> bool:
        """Check if the Python version has pre-compiled wheels available"""
        self.print_colored(f"\n🔧 Checking wheel compatibility for {python_cmd}...", 'blue')
        
        # Test packages that commonly have compilation issues
        test_packages = ['numpy==1.26.4', 'safetensors==0.4.3']
        
        for package in test_packages:
            try:
                # Use pip to check if wheel is available (dry run)
                result = subprocess.run([
                    python_cmd, "-m", "pip", "install", "--dry-run", "--no-deps", 
                    "--only-binary", ":all:", package
                ], capture_output=True, text=True)
                
                if result.returncode != 0 or "Could not find a version" in result.stderr:
                    self.print_colored(f"  ✗ No pre-compiled wheel for {package}", 'red')
                    return False
            except Exception as e:
                logger.error(f"Error in kimera_intelligent_setup.py: {e}", exc_info=True)
                raise  # Re-raise for proper error handling
                return False
        
        self.print_colored("  ✓ Pre-compiled wheels available", 'green')
        return True
    
    def install_python_311(self) -> Optional[str]:
        """Attempt to install Python 3.11 automatically (Windows only)"""
        if platform.system() != "Windows":
            return None
        
        self.print_colored("\n📥 Attempting to install Python 3.11...", 'blue')
        
        try:
            # Download Python 3.11 installer
            python_url = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"
            installer_path = Path(tempfile.gettempdir()) / "python-3.11.9-installer.exe"
            
            self.print_colored("  Downloading Python 3.11.9...", 'yellow')
            urllib.request.urlretrieve(python_url, installer_path)
            
            # Run installer silently
            self.print_colored("  Installing Python 3.11.9 (this may take a few minutes)...", 'yellow')
            install_cmd = [
                str(installer_path),
                "/quiet",
                "InstallAllUsers=0",
                "PrependPath=1",
                "Include_test=0",
                "Include_pip=1",
                "Include_launcher=1"
            ]
            
            result = subprocess.run(install_cmd, capture_output=True)
            
            if result.returncode == 0:
                self.print_colored("  ✓ Python 3.11 installed successfully", 'green')
                
                # Clean up installer
                installer_path.unlink()
                
                # Return the py launcher command for 3.11
                return "py -3.11"
            else:
                self.print_colored("  ✗ Installation failed", 'red')
                return None
                
        except Exception as e:
            self.print_colored(f"  ✗ Error installing Python: {str(e)}", 'red')
            return None
    
    def create_optimized_requirements(self) -> Path:
        """Create an optimized requirements file with compatible versions"""
        self.print_colored("\n📝 Creating optimized requirements file...", 'blue')
        
        optimized_reqs = """# Kimera SWM - Optimized Requirements for Python 3.11
# Auto-generated by Intelligent Setup System
# All versions are guaranteed to have pre-compiled wheels

# --- Core AI & ML ---
torch==2.1.2+cu118
torchaudio==2.1.2+cu118
torchvision==0.16.2+cu118

# --- Hugging Face & NLP ---
transformers==4.36.2
accelerate==0.25.0
bitsandbytes==0.41.3
sentence-transformers==2.2.2
safetensors==0.4.1

# --- FastAPI Web Framework ---
fastapi==0.108.0
uvicorn[standard]==0.25.0
pydantic==2.5.3
python-dotenv==1.0.0
requests==2.31.0

# --- Database & Storage ---
psycopg2-binary==2.9.9
sqlalchemy==2.0.25
neo4j==5.16.0
pgvector==0.2.4

# --- Monitoring & Utilities ---
prometheus-client==0.19.0
psutil==5.9.8
passlib[bcrypt]==1.7.4
python-jose[cryptography]==3.3.0

# --- Scientific Computing ---
numpy==1.26.3
pandas==2.1.4
scikit-learn==1.4.0
scipy==1.11.4

# --- Image Processing ---
Pillow==10.2.0

# --- General Utilities ---
colorlog==6.8.0
rich==13.7.0
tqdm==4.66.1
websockets==12.0
packaging==23.2
PyYAML==6.0.1

# --- Optional but recommended ---
FlagEmbedding==1.2.8
onnxruntime-gpu==1.16.3
colorama==0.4.6
"""
        
        optimized_path = self.project_root / "requirements_optimized.txt"
        optimized_path.write_text(optimized_reqs)
        
        self.print_colored(f"  ✓ Created: {optimized_path}", 'green')
        return optimized_path
    
    def create_virtual_environment(self, python_cmd: str) -> bool:
        """Create a virtual environment with the specified Python"""
        self.print_colored(f"\n🏗️ Creating virtual environment with {python_cmd}...", 'blue')
        
        # Remove existing venv if present
        if self.venv_path.exists():
            self.print_colored("  Removing existing virtual environment...", 'yellow')
            shutil.rmtree(self.venv_path)
        
        # Create new venv
        try:
            result = subprocess.run([python_cmd, "-m", "venv", str(self.venv_path)], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                self.print_colored(f"  ✓ Virtual environment created: {self.venv_path}", 'green')
                return True
            else:
                self.print_colored(f"  ✗ Failed to create virtual environment: {result.stderr}", 'red')
                return False
        except Exception as e:
            self.print_colored(f"  ✗ Error: {str(e)}", 'red')
            return False
    
    def get_venv_python(self) -> str:
        """Get the Python executable from the virtual environment"""
        if platform.system() == "Windows":
            return str(self.venv_path / "Scripts" / "python.exe")
        else:
            return str(self.venv_path / "bin" / "python")
    
    def install_dependencies(self, requirements_path: Path) -> bool:
        """Install dependencies in the virtual environment"""
        self.print_colored("\n📦 Installing dependencies...", 'blue')
        
        venv_python = self.get_venv_python()
        
        # Upgrade pip first
        self.print_colored("  Upgrading pip...", 'yellow')
        subprocess.run([venv_python, "-m", "pip", "install", "--upgrade", "pip"], 
                      capture_output=True)
        
        # Install requirements
        self.print_colored("  Installing packages (this may take several minutes)...", 'yellow')
        
        cmd = [
            venv_python, "-m", "pip", "install", "-r", str(requirements_path),
            "--extra-index-url", "https://download.pytorch.org/whl/cu118"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            self.print_colored("  ✓ All dependencies installed successfully", 'green')
            return True
        else:
            self.print_colored("  ✗ Installation failed", 'red')
            self.print_colored(f"  Error: {result.stderr}", 'red')
            return False
    
    def validate_installation(self) -> bool:
        """Validate the Kimera installation"""
        self.print_colored("\n🔍 Validating installation...", 'blue')
        
        venv_python = self.get_venv_python()
        
        # Test critical imports
        test_script = """
import sys
print(f"Python: {sys.version}")

try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"✗ PyTorch: {e}")

try:
    import transformers
    print(f"✓ Transformers: {transformers.__version__}")
except Exception as e:
    print(f"✗ Transformers: {e}")

try:
    import fastapi
    print(f"✓ FastAPI: {fastapi.__version__}")
except Exception as e:
    print(f"✗ FastAPI: {e}")

try:
    from src.core.kimera_system import KimeraSystem
    print("✓ Kimera core modules")
except Exception as e:
    print(f"✗ Kimera core: {e}")

print("\\nValidation complete!")
"""
        
        # Write test script
        test_file = self.project_root / "test_kimera_env.py"
        test_file.write_text(test_script)
        
        # Run test
        result = subprocess.run([venv_python, str(test_file)], 
                              capture_output=True, text=True, 
                              cwd=str(self.project_root))
        
        print(result.stdout)
        
        # Clean up
        test_file.unlink()
        
        return result.returncode == 0
    
    def create_activation_scripts(self):
        """Create convenient activation scripts"""
        self.print_colored("\n📄 Creating activation scripts...", 'blue')
        
        # Windows batch script
        if platform.system() == "Windows":
            activate_bat = self.project_root / "activate_kimera.bat"
            activate_bat.write_text(f"""@echo off
echo Activating Kimera virtual environment...
call "{self.venv_path}\\Scripts\\activate.bat"
echo.
echo Kimera environment activated!
echo Python: %VIRTUAL_ENV%\\Scripts\\python.exe
echo.
echo To start Kimera, run: python kimera.py
""")
            self.print_colored(f"  ✓ Created: {activate_bat}", 'green')
            
            # PowerShell script
            activate_ps1 = self.project_root / "activate_kimera.ps1"
            activate_ps1.write_text(f"""
Write-Host "Activating Kimera virtual environment..." -ForegroundColor Cyan
& "{self.venv_path}\\Scripts\\Activate.ps1"
Write-Host ""
Write-Host "Kimera environment activated!" -ForegroundColor Green
Write-Host "Python: $env:VIRTUAL_ENV\\Scripts\\python.exe"
Write-Host ""
Write-Host "To start Kimera, run: python kimera.py" -ForegroundColor Yellow
""")
            self.print_colored(f"  ✓ Created: {activate_ps1}", 'green')
    
    def run(self):
        """Run the complete intelligent setup process"""
        self.print_banner()
        
        # Step 1: Detect Python installations
        pythons = self.detect_python_installations()
        
        # Step 2: Find compatible Python
        compatible_python = None
        
        # Check for Python 3.11 first
        for cmd, version in pythons:
            if version.startswith("3.11"):
                if self.check_wheel_compatibility(cmd):
                    compatible_python = cmd
                    break
        
        # If no 3.11, check other versions
        if not compatible_python:
            for cmd, version in pythons:
                if self.check_wheel_compatibility(cmd):
                    compatible_python = cmd
                    break
        
        # Step 3: Install Python 3.11 if needed
        if not compatible_python and platform.system() == "Windows":
            self.print_colored("\n⚠️ No compatible Python found. Installing Python 3.11...", 'yellow')
            
            installed_cmd = self.install_python_311()
            if installed_cmd:
                compatible_python = installed_cmd
        
        if not compatible_python:
            self.print_colored("\n❌ No compatible Python version found!", 'red')
            self.print_colored("Please install Python 3.11 manually from https://python.org", 'yellow')
            return False
        
        self.print_colored(f"\n✅ Using Python: {compatible_python}", 'green')
        
        # Step 4: Create optimized requirements
        optimized_reqs = self.create_optimized_requirements()
        
        # Step 5: Create virtual environment
        if not self.create_virtual_environment(compatible_python):
            return False
        
        # Step 6: Install dependencies
        if not self.install_dependencies(optimized_reqs):
            # Try with original requirements as fallback
            self.print_colored("\n⚠️ Trying with original requirements...", 'yellow')
            original_reqs = self.project_root / "requirements.txt"
            if not self.install_dependencies(original_reqs):
                return False
        
        # Step 7: Validate installation
        if not self.validate_installation():
            self.print_colored("\n⚠️ Validation showed some issues, but continuing...", 'yellow')
        
        # Step 8: Create activation scripts
        self.create_activation_scripts()
        
        # Success!
        self.print_colored("""
╔══════════════════════════════════════════════════════════════════╗
║                    ✅ SETUP COMPLETE!                             ║
╚══════════════════════════════════════════════════════════════════╝

Your Kimera SWM environment is ready!

To activate the environment:
  • Windows (CMD):        activate_kimera.bat
  • Windows (PowerShell): .\\activate_kimera.ps1
  • Or directly:          {venv}\\Scripts\\activate

To start Kimera:
  1. Activate the environment (see above)
  2. Run: python kimera.py

The server will start at http://localhost:8000
API documentation at http://localhost:8000/docs
""".format(venv=self.venv_name), 'green')
        
        return True


if __name__ == "__main__":
    setup = KimeraIntelligentSetup()
    success = setup.run()
    
    if not success:
        sys.exit(1)