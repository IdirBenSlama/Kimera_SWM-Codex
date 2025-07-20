#!/usr/bin/env python3
"""
KIMERA SWM - Universal Deployment Script
=======================================

This script provides a comprehensive solution for deploying Kimera on any PC.
It handles all dependencies, configuration, and setup automatically.

Usage:
    python deploy_kimera.py

Features:
- Automatic dependency detection and installation
- Environment configuration
- System verification
- Cross-platform compatibility
- Detailed error reporting and troubleshooting
"""

import os
import sys
import subprocess
import platform
import venv
import shutil
from pathlib import Path
from typing import Dict, List, Optional

# Version requirements
MIN_PYTHON_VERSION = (3, 10)
KIMERA_VERSION = "1.0.0"

class Colors:
    """Terminal colors for better output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_colored(text: str, color: str = Colors.WHITE, bold: bool = False):
    """Print colored text"""
    prefix = f"{Colors.BOLD if bold else ''}{color}"
    print(f"{prefix}{text}{Colors.END}")

def print_banner():
    """Print the Kimera deployment banner"""
    print(f"""{Colors.CYAN}
╔════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                        ║
║                        🚀 KIMERA SWM UNIVERSAL DEPLOYMENT v{KIMERA_VERSION}                       ║
║                                                                                        ║
║                         Automated Setup for Any PC - Windows, Linux, macOS            ║
║                                                                                        ║
╚════════════════════════════════════════════════════════════════════════════════════════╝{Colors.END}""")

class KimeraDeployer:
    """Main deployment class for Kimera SWM"""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.venv_path = self.project_root / "venv"
        self.python_exe = sys.executable
        self.venv_python = None
        self.system_info = self._get_system_info()
        self.errors = []
        self.warnings = []
        
    def _get_system_info(self) -> Dict:
        """Get system information"""
        return {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "architecture": platform.machine(),
            "python_version": platform.python_version(),
        }
    
    def print_system_info(self):
        """Print system information"""
        print_colored("\n📊 System Information:", Colors.BLUE, bold=True)
        print_colored(f"  Platform: {self.system_info['platform']} {self.system_info['platform_release']}", Colors.WHITE)
        print_colored(f"  Architecture: {self.system_info['architecture']}", Colors.WHITE)
        print_colored(f"  Python: {self.system_info['python_version']}", Colors.WHITE)
    
    def check_python_version(self) -> bool:
        """Check if Python version meets requirements"""
        print_colored("\n🐍 Checking Python version...", Colors.BLUE, bold=True)
        
        current_version = sys.version_info[:2]
        if current_version < MIN_PYTHON_VERSION:
            print_colored(f"  ❌ Python {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}+ required", Colors.RED)
            print_colored(f"  ❌ Current version: {current_version[0]}.{current_version[1]}", Colors.RED)
            self.errors.append(f"Python version {current_version[0]}.{current_version[1]} is too old")
            return False
        
        print_colored(f"  ✅ Python {current_version[0]}.{current_version[1]} meets requirements", Colors.GREEN)
        return True
    
    def check_project_structure(self) -> bool:
        """Check if we're in a valid Kimera project directory"""
        print_colored("\n📁 Checking project structure...", Colors.BLUE, bold=True)
        
        required_files = [
            "kimera.py",
            "requirements.txt",
            "backend/main.py",
            "backend/api/main.py",
        ]
        
        missing_files = []
        for file_path in required_files:
            if not (self.project_root / file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            print_colored(f"  ❌ Missing required files: {', '.join(missing_files)}", Colors.RED)
            self.errors.append(f"Missing required files: {', '.join(missing_files)}")
            return False
        
        print_colored("  ✅ Project structure is valid", Colors.GREEN)
        return True
    
    def create_virtual_environment(self) -> bool:
        """Create a virtual environment"""
        print_colored("\n🏗️  Creating virtual environment...", Colors.BLUE, bold=True)
        
        try:
            if self.venv_path.exists():
                print_colored("  ℹ️  Virtual environment already exists, recreating...", Colors.YELLOW)
                shutil.rmtree(self.venv_path)
            
            venv.create(self.venv_path, with_pip=True)
            
            # Get venv python executable
            if platform.system() == "Windows":
                self.venv_python = self.venv_path / "Scripts" / "python.exe"
            else:
                self.venv_python = self.venv_path / "bin" / "python"
            
            # Upgrade pip
            subprocess.run([str(self.venv_python), "-m", "pip", "install", "--upgrade", "pip"], 
                          check=True, capture_output=True)
            
            print_colored("  ✅ Virtual environment created successfully", Colors.GREEN)
            return True
            
        except Exception as e:
            print_colored(f"  ❌ Failed to create virtual environment: {str(e)}", Colors.RED)
            self.errors.append(f"Virtual environment creation failed: {str(e)}")
            return False
    
    def install_dependencies(self) -> bool:
        """Install dependencies from requirements files"""
        print_colored("\n📦 Installing dependencies...", Colors.BLUE, bold=True)
        
        if not self.venv_python:
            print_colored("  ❌ Virtual environment not available", Colors.RED)
            return False
        
        try:
            # Install base requirements first
            base_req_path = self.project_root / "requirements" / "base.txt"
            if base_req_path.exists():
                print_colored("  📥 Installing base requirements...", Colors.CYAN)
                subprocess.run([
                    str(self.venv_python), "-m", "pip", "install", 
                    "-r", str(base_req_path), "--timeout", "300"
                ], check=True)
                print_colored("  ✅ Base requirements installed", Colors.GREEN)
            
            # Install main requirements
            main_req_path = self.project_root / "requirements.txt"
            if main_req_path.exists():
                print_colored("  📥 Installing main requirements (this may take a while)...", Colors.CYAN)
                subprocess.run([
                    str(self.venv_python), "-m", "pip", "install", 
                    "-r", str(main_req_path), "--timeout", "600"
                ], check=True)
                print_colored("  ✅ Main requirements installed", Colors.GREEN)
            
            return True
            
        except subprocess.CalledProcessError as e:
            print_colored(f"  ❌ Failed to install dependencies: {str(e)}", Colors.RED)
            self.errors.append(f"Dependency installation failed: {str(e)}")
            return False
    
    def create_environment_file(self) -> bool:
        """Create environment configuration file"""
        print_colored("\n⚙️  Creating environment configuration...", Colors.BLUE, bold=True)
        
        env_content = """# KIMERA SWM Environment Configuration
# Copy this file to .env and customize as needed

# Server Configuration
KIMERA_ENV=development
KIMERA_HOST=127.0.0.1
KIMERA_PORT=8000
KIMERA_RELOAD=true
KIMERA_LOG_LEVEL=INFO

# Security Settings
KIMERA_SECRET_KEY=your-secret-key-here-change-in-production
KIMERA_API_KEY=your-api-key-here

# Database Configuration
DATABASE_URL=sqlite:///kimera.db
DATABASE_ECHO=false

# Performance Settings
KIMERA_MAX_THREADS=8
KIMERA_GPU_MEMORY_FRACTION=0.8

# External API Keys (Optional)
OPENAI_API_KEY=your-openai-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here

# Monitoring
KIMERA_MONITORING_ENABLED=true
KIMERA_METRICS_ENABLED=true

# Development Settings
KIMERA_DEBUG=false
KIMERA_TESTING=false
"""
        
        try:
            env_example_path = self.project_root / ".env.example"
            env_path = self.project_root / ".env"
            
            # Create .env.example
            with open(env_example_path, "w") as f:
                f.write(env_content)
            
            # Create .env if it doesn't exist
            if not env_path.exists():
                with open(env_path, "w") as f:
                    f.write(env_content)
                print_colored("  ✅ Environment file created (.env)", Colors.GREEN)
            else:
                print_colored("  ℹ️  Environment file already exists (.env)", Colors.YELLOW)
            
            print_colored("  ✅ Environment example created (.env.example)", Colors.GREEN)
            return True
            
        except Exception as e:
            print_colored(f"  ❌ Failed to create environment file: {str(e)}", Colors.RED)
            self.errors.append(f"Environment file creation failed: {str(e)}")
            return False
    
    def verify_installation(self) -> bool:
        """Verify that the installation is working"""
        print_colored("\n🔍 Verifying installation...", Colors.BLUE, bold=True)
        
        try:
            # Test imports
            test_script = """
import sys
sys.path.insert(0, '.')
try:
    import fastapi
    import uvicorn
    print("✅ Core imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
"""
            
            result = subprocess.run([str(self.venv_python), "-c", test_script], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                print_colored("  ✅ Core dependencies verified", Colors.GREEN)
                return True
            else:
                print_colored(f"  ❌ Verification failed: {result.stderr}", Colors.RED)
                self.errors.append(f"Installation verification failed: {result.stderr}")
                return False
                
        except Exception as e:
            print_colored(f"  ❌ Verification error: {str(e)}", Colors.RED)
            self.errors.append(f"Installation verification error: {str(e)}")
            return False
    
    def create_startup_scripts(self) -> bool:
        """Create startup scripts for different platforms"""
        print_colored("\n📝 Creating startup scripts...", Colors.BLUE, bold=True)
        
        try:
            # Windows batch script
            batch_script = f"""@echo off
title KIMERA SWM System
echo Starting KIMERA SWM...
cd /d "{self.project_root}"
"{self.venv_python}" kimera.py
pause
"""
            
            with open(self.project_root / "start_kimera.bat", "w") as f:
                f.write(batch_script)
            
            # Unix shell script
            shell_script = f"""#!/bin/bash
echo "Starting KIMERA SWM..."
cd "{self.project_root}"
"{self.venv_python}" kimera.py
"""
            
            shell_path = self.project_root / "start_kimera.sh"
            with open(shell_path, "w") as f:
                f.write(shell_script)
            
            # Make shell script executable
            if platform.system() != "Windows":
                os.chmod(shell_path, 0o755)
            
            print_colored("  ✅ Startup scripts created", Colors.GREEN)
            return True
            
        except Exception as e:
            print_colored(f"  ❌ Failed to create startup scripts: {str(e)}", Colors.RED)
            self.errors.append(f"Startup script creation failed: {str(e)}")
            return False
    
    def deploy(self) -> bool:
        """Run the complete deployment process"""
        print_banner()
        self.print_system_info()
        
        steps = [
            ("Checking Python version", self.check_python_version),
            ("Checking project structure", self.check_project_structure),
            ("Creating virtual environment", self.create_virtual_environment),
            ("Installing dependencies", self.install_dependencies),
            ("Creating environment configuration", self.create_environment_file),
            ("Verifying installation", self.verify_installation),
            ("Creating startup scripts", self.create_startup_scripts),
        ]
        
        success = True
        for step_name, step_func in steps:
            if not step_func():
                success = False
        
        if success:
            print_colored("\n🎉 DEPLOYMENT SUCCESSFUL!", Colors.GREEN, bold=True)
            print_colored("\nTo start Kimera:", Colors.BLUE, bold=True)
            print_colored("  Windows: Double-click start_kimera.bat", Colors.WHITE)
            print_colored("  Linux/macOS: ./start_kimera.sh", Colors.WHITE)
            print_colored("  Or run: python kimera.py", Colors.WHITE)
            print_colored("\nServer will be available at: http://localhost:8000", Colors.CYAN)
        else:
            print_colored("\n❌ DEPLOYMENT FAILED", Colors.RED, bold=True)
            for error in self.errors:
                print_colored(f"  - {error}", Colors.RED)
        
        return success

def main():
    """Main entry point"""
    deployer = KimeraDeployer()
    success = deployer.deploy()
    
    if success:
        print_colored("\n🚀 Ready to launch Kimera!", Colors.GREEN, bold=True)
        sys.exit(0)
    else:
        print_colored("\n❌ Deployment failed. Please resolve errors and try again.", Colors.RED, bold=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 