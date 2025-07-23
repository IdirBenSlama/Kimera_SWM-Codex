#!/usr/bin/env python3
"""
KIMERA SWM - Force Setup with Aggressive Cleanup
================================================

This script handles permission issues and forces a clean setup.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import time
import stat

class KimeraForceSetup:
    """Force setup with aggressive cleanup capabilities"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.absolute()
        self.python_311 = r"C:\Users\Loomine\AppData\Local\Programs\Python\Python311\python.exe"
        self.venv_name = ".venv-kimera-clean"  # New name to avoid conflicts
        self.venv_path = self.project_root / self.venv_name
        
    def print_colored(self, text: str, color: str = 'white'):
        """Print colored text"""
        colors = {
            'red': '\033[91m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'blue': '\033[94m',
            'cyan': '\033[96m',
            'end': '\033[0m'
        }
        print(f"{colors.get(color, '')}{text}{colors['end']}")
    
    def force_remove_directory(self, path: Path):
        """Force remove a directory, handling permission issues"""
        if not path.exists():
            return
            
        self.print_colored(f"  Force removing: {path}", 'yellow')
        
        def handle_remove_readonly(func, path, exc):
            """Error handler for shutil.rmtree"""
            if os.path.exists(path):
                os.chmod(path, stat.S_IWRITE)
                func(path)
        
        try:
            # Try normal removal first
            shutil.rmtree(path, onerror=handle_remove_readonly)
        except Exception as e:
            self.print_colored(f"  Standard removal failed: {e}", 'yellow')
            
            # Try Windows-specific force removal
            try:
                subprocess.run(['cmd', '/c', 'rmdir', '/s', '/q', str(path)], 
                             capture_output=True, shell=True)
            except:
                pass
            
            # If still exists, try to rename it
            if path.exists():
                try:
                    temp_name = f"{path}_old_{int(time.time())}"
                    path.rename(temp_name)
                    self.print_colored(f"  Renamed to: {temp_name}", 'yellow')
                except:
                    pass
    
    def clean_all_venvs(self):
        """Clean all virtual environments"""
        self.print_colored("\n🧹 Cleaning all virtual environments...", 'blue')
        
        venv_patterns = [
            ".venv*",
            "venv*",
            "*env",
        ]
        
        for pattern in venv_patterns:
            for venv_dir in self.project_root.glob(pattern):
                if venv_dir.is_dir() and (venv_dir / "Scripts").exists():
                    self.force_remove_directory(venv_dir)
    
    def create_minimal_requirements(self) -> Path:
        """Create minimal requirements for quick testing"""
        content = """# Minimal Kimera requirements
numpy==1.24.3
torch==2.0.1
fastapi==0.100.0
uvicorn[standard]==0.23.1
transformers==4.30.2
pydantic==2.0.3
requests==2.31.0
python-dotenv==1.0.0
sqlalchemy==2.0.19
psycopg2-binary==2.9.7
prometheus-client==0.17.1
colorlog==6.7.0
tqdm==4.65.0
"""
        
        req_path = self.project_root / "requirements_minimal.txt"
        req_path.write_text(content)
        return req_path
    
    def setup_environment(self):
        """Setup the environment"""
        self.print_colored(f"\n🏗️ Creating fresh virtual environment...", 'blue')
        
        cmd = [self.python_311, "-m", "venv", str(self.venv_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            self.print_colored(f"  ✗ Failed: {result.stderr}", 'red')
            return False
            
        self.print_colored("  ✓ Virtual environment created", 'green')
        return True
    
    def install_packages(self, req_path: Path):
        """Install packages"""
        self.print_colored("\n📦 Installing packages...", 'blue')
        
        venv_python = str(self.venv_path / "Scripts" / "python.exe")
        
        # Upgrade pip
        self.print_colored("  Upgrading pip...", 'yellow')
        subprocess.run([venv_python, "-m", "pip", "install", "--upgrade", "pip"], 
                      capture_output=True)
        
        # Install from requirements
        self.print_colored("  Installing requirements...", 'yellow')
        cmd = [venv_python, "-m", "pip", "install", "-r", str(req_path)]
        
        # Add PyTorch index
        if "torch" in req_path.read_text():
            cmd.extend(["--extra-index-url", "https://download.pytorch.org/whl/cu118"])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            self.print_colored("  ✓ Packages installed", 'green')
            return True
        else:
            self.print_colored("  ⚠️ Some packages may have failed", 'yellow')
            return True  # Continue anyway
    
    def create_launcher(self):
        """Create a simple launcher"""
        self.print_colored("\n📄 Creating launcher...", 'blue')
        
        launcher = self.project_root / "kimera_start.bat"
        launcher.write_text(f"""@echo off
echo Starting Kimera SWM...
call "{self.venv_path}\\Scripts\\activate.bat"
python kimera.py
pause
""")
        
        self.print_colored(f"  ✓ Created: {launcher}", 'green')
    
    def run(self):
        """Run the force setup"""
        self.print_colored("""
╔══════════════════════════════════════════════════════════════════╗
║                 KIMERA FORCE SETUP                                ║
╚══════════════════════════════════════════════════════════════════╝
""", 'cyan')
        
        # Clean old environments
        self.clean_all_venvs()
        
        # Create new environment
        if not self.setup_environment():
            return False
        
        # Install packages
        req_path = self.create_minimal_requirements()
        self.install_packages(req_path)
        
        # Create launcher
        self.create_launcher()
        
        self.print_colored("""
✅ Setup complete!

To start Kimera:
  1. Run: kimera_start.bat
  2. Or manually:
     - Activate: .venv-kimera-clean\\Scripts\\activate
     - Run: python kimera.py
""", 'green')
        
        return True


if __name__ == "__main__":
    setup = KimeraForceSetup()
    success = setup.run()
    sys.exit(0 if success else 1)