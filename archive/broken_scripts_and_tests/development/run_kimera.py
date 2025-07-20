#!/usr/bin/env python3
"""
🚀 KIMERA SYSTEM LAUNCHER - ULTIMATE SOLUTION
============================================

This script solves the KIMERA running problem once and for all.
Run this from anywhere and it will start KIMERA properly.

Usage:
    python run_kimera.py
    python run_kimera.py --setup    (first time setup)
    python run_kimera.py --dev      (development mode)
    python run_kimera.py --help     (show help)
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_banner():
    """Print startup banner"""
    print("=" * 80)
    print("🚀 KIMERA SYSTEM LAUNCHER - ULTIMATE SOLUTION")
    print("   Kinetic Intelligence for Multidimensional Analysis")
    print("=" * 80)
    print(f"📍 Current directory: {os.getcwd()}")
    print(f"🐍 Python: {sys.version.split()[0]}")
    print(f"💻 Platform: {platform.system()}")
    print("=" * 80)

def find_project_root():
    """Find the KIMERA project root directory"""
    current = Path.cwd()
    
    # Look for KIMERA project markers
    markers = ["backend", "requirements.txt", "README.md"]
    
    # Check current directory first
    if all((current / marker).exists() for marker in markers):
        return current
    
    # Check parent directories
    for parent in current.parents:
        if all((parent / marker).exists() for marker in markers):
            return parent
    
    # Check if we're in a subdirectory of the project
    script_dir = Path(__file__).parent
    if all((script_dir / marker).exists() for marker in markers):
        return script_dir
    
    return None

def get_python_executable(project_root):
    """Get the correct Python executable"""
    venv_path = project_root / ".venv"
    
    if platform.system() == "Windows":
        if venv_path.exists():
            return str(venv_path / "Scripts" / "python.exe")
        return "python"
    else:
        if venv_path.exists():
            return str(venv_path / "bin" / "python")
        return "python3"

def setup_environment(project_root, python_exe):
    """Set up the environment"""
    print("🔧 Setting up environment...")
    
    try:
        # Create virtual environment if it doesn't exist
        venv_path = project_root / ".venv"
        if not venv_path.exists():
            print("Creating virtual environment...")
            subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
            print("✅ Virtual environment created")
        
        # Install requirements
        requirements_file = project_root / "requirements.txt"
        if requirements_file.exists():
            print("Installing requirements...")
            subprocess.run([
                python_exe, "-m", "pip", "install", "-r", str(requirements_file)
            ], check=True)
            print("✅ Requirements installed")
        
        return True
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return False

def check_dependencies(python_exe):
    """Check if key dependencies are available"""
    print("📦 Checking dependencies...")
    
    key_deps = ["fastapi", "uvicorn", "torch", "transformers", "sqlalchemy"]
    missing = []
    
    for dep in key_deps:
        try:
            subprocess.run([
                python_exe, "-c", f"import {dep}"
            ], check=True, capture_output=True)
        except subprocess.CalledProcessError:
            missing.append(dep)
    
    if missing:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        print("💡 Run with --setup to install dependencies")
        return False
    
    print("✅ All dependencies available")
    return True

def start_kimera(project_root, python_exe, dev_mode=False):
    """Start KIMERA server"""
    print(f"🚀 Starting KIMERA {'(dev mode)' if dev_mode else ''}...")
    
    # Change to project directory
    os.chdir(project_root)
    
    # Build command
    cmd = [
        python_exe, "-m", "uvicorn", 
        "backend.api.main:app", 
        "--host", "0.0.0.0", 
        "--port", "8001"
    ]
    
    if dev_mode:
        cmd.append("--reload")
    
    try:
        print(f"Executing: {' '.join(cmd)}")
        print("🌐 KIMERA will be available at: http://localhost:8001")
        print("📚 API docs at: http://localhost:8001/docs")
        print("🔍 Health check: http://localhost:8001/system/health")
        print("Press Ctrl+C to stop the server")
        print("-" * 80)
        
        # Start the server
        subprocess.run(cmd, check=True)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Server startup failed: {e}")
        print("\n💡 TROUBLESHOOTING TIPS:")
        print("1. Try: python run_kimera.py --setup")
        print("2. Check if port 8001 is already in use")
        print("3. Make sure you're in the KIMERA project directory")
        return False
    except KeyboardInterrupt:
        print("\n✋ Server stopped by user")
        return True
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def show_help():
    """Show help information"""
    help_text = """
🚀 KIMERA SYSTEM LAUNCHER - HELP
===============================

USAGE:
    python run_kimera.py [OPTIONS]

OPTIONS:
    --setup     Set up environment (create venv, install dependencies)
    --dev       Start in development mode with auto-reload
    --help      Show this help message

EXAMPLES:
    python run_kimera.py                # Start KIMERA normally
    python run_kimera.py --setup        # First time setup
    python run_kimera.py --dev          # Development mode

TROUBLESHOOTING:
    1. First time setup:
       python run_kimera.py --setup

    2. Dependencies missing:
       pip install -r requirements.txt

    3. Port already in use:
       Kill the process using port 8001 or restart your computer

    4. Permission issues (Windows):
       Run Command Prompt as Administrator

    5. Still not working:
       Make sure you're in the KIMERA project directory

SYSTEM REQUIREMENTS:
    - Python 3.10+
    - 8GB+ RAM (16GB+ recommended)
    - NVIDIA GPU (optional but recommended)

WHAT THIS SCRIPT DOES:
    1. Auto-detects KIMERA project directory
    2. Sets up virtual environment if needed
    3. Installs dependencies if needed
    4. Starts KIMERA server with proper configuration
    5. Provides clear error messages and troubleshooting tips

This script solves the "where should I run KIMERA" problem once and for all!
    """
    print(help_text)

def main():
    """Main entry point"""
    # Parse simple arguments
    setup_mode = "--setup" in sys.argv
    dev_mode = "--dev" in sys.argv
    help_mode = "--help" in sys.argv or "-h" in sys.argv
    
    print_banner()
    
    if help_mode:
        show_help()
        return
    
    # Find project root
    project_root = find_project_root()
    if not project_root:
        print("❌ KIMERA project directory not found!")
        print("💡 Make sure you're running this from the KIMERA project directory or its subdirectories.")
        print("   Look for a directory containing: backend/, requirements.txt, README.md")
        sys.exit(1)
    
    print(f"✅ Found KIMERA project at: {project_root}")
    
    # Get Python executable
    python_exe = get_python_executable(project_root)
    print(f"🐍 Using Python: {python_exe}")
    
    # Setup environment if requested
    if setup_mode:
        if not setup_environment(project_root, python_exe):
            sys.exit(1)
        print("✅ Environment setup complete!")
        return
    
    # Check dependencies
    if not check_dependencies(python_exe):
        print("💡 Run with --setup to install dependencies:")
        print("   python run_kimera.py --setup")
        sys.exit(1)
    
    # Start KIMERA
    success = start_kimera(project_root, python_exe, dev_mode)
    
    if not success:
        print("\n💡 If this is your first time, try:")
        print("   python run_kimera.py --setup")
        sys.exit(1)

if __name__ == "__main__":
    main()