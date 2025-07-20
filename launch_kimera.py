#!/usr/bin/env python3
"""
KIMERA SWM - Simple Universal Launcher
=====================================

This script provides a foolproof way to start Kimera from anywhere.
It automatically finds the project directory and handles common startup issues.

Usage:
    python launch_kimera.py
    
Features:
- Works from any directory
- Automatically finds Kimera project
- Handles virtual environments
- Provides clear error messages
- Cross-platform compatibility
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
import time

def print_colored(text, color='white'):
    """Print colored text for better visibility"""
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'bold': '\033[1m',
        'end': '\033[0m'
    }
    
    if color in colors:
        print(f"{colors[color]}{text}{colors['end']}")
    else:
        print(text)

def print_banner():
    """Print the launcher banner"""
    print_colored("""
🚀 KIMERA SWM LAUNCHER
=====================
Starting Kimera from anywhere...
""", 'cyan')

def find_kimera_project(start_path=None):
    """Find the Kimera project directory"""
    if start_path is None:
        start_path = Path.cwd()
    
    # Check current directory and parent directories
    current = Path(start_path).resolve()
    
    for _ in range(10):  # Check up to 10 parent directories
        # Look for key Kimera files
        kimera_indicators = [
            current / "kimera.py",
            current / "backend" / "main.py",
            current / "backend" / "api" / "main.py",
        ]
        
        if all(indicator.exists() for indicator in kimera_indicators):
            return current
        
        # Move to parent directory
        parent = current.parent
        if parent == current:  # Reached root
            break
        current = parent
    
    return None

def find_python_executable(project_root):
    """Find the best Python executable to use"""
    # Check for virtual environment
    venv_paths = [
        project_root / "venv" / "Scripts" / "python.exe",  # Windows
        project_root / "venv" / "bin" / "python",         # Unix
        project_root / ".venv" / "Scripts" / "python.exe", # Windows alt
        project_root / ".venv" / "bin" / "python",        # Unix alt
    ]
    
    for venv_python in venv_paths:
        if venv_python.exists():
            return str(venv_python)
    
    # Fall back to system Python
    return sys.executable

def check_dependencies(python_exe):
    """Check if key dependencies are available"""
    try:
        result = subprocess.run([
            python_exe, "-c", 
            "import fastapi, uvicorn; print('Dependencies OK')"
        ], capture_output=True, text=True, timeout=10)
        
        return result.returncode == 0
    except:
        return False

def start_kimera(project_root, python_exe):
    """Start the Kimera server"""
    print_colored(f"🏃 Starting Kimera server...", 'blue')
    print_colored(f"   Project: {project_root}", 'white')
    print_colored(f"   Python: {python_exe}", 'white')
    
    # Change to project directory
    os.chdir(project_root)
    
    # Start server
    try:
        # Use kimera.py as the main entry point
        cmd = [python_exe, "kimera.py"]
        
        print_colored(f"   Command: {' '.join(cmd)}", 'white')
        print_colored("   Starting server (this may take a moment)...", 'yellow')
        
        # Start the process
        process = subprocess.Popen(cmd, cwd=project_root)
        
        # Wait a moment and check if it's still running
        time.sleep(2)
        if process.poll() is None:
            print_colored("✅ Kimera server started successfully!", 'green')
            print_colored("   🌐 Access at: http://localhost:8000", 'cyan')
            print_colored("   📚 API docs: http://localhost:8000/docs", 'cyan')
            print_colored("   ❤️  Health: http://localhost:8000/health", 'cyan')
            print_colored("\n   Press Ctrl+C to stop the server", 'yellow')
            
            # Wait for the process to complete
            try:
                process.wait()
            except KeyboardInterrupt:
                print_colored("\n⏹️  Stopping server...", 'yellow')
                process.terminate()
                process.wait()
                print_colored("✅ Server stopped.", 'green')
        else:
            print_colored("❌ Server failed to start", 'red')
            return False
            
    except Exception as e:
        print_colored(f"❌ Error starting server: {str(e)}", 'red')
        return False
    
    return True

def show_help():
    """Show help information"""
    print_colored("""
🆘 KIMERA LAUNCHER HELP
======================

WHAT THIS SCRIPT DOES:
• Automatically finds your Kimera project directory
• Uses the correct Python executable (virtual environment if available)
• Starts the Kimera server with proper configuration
• Provides clear error messages and troubleshooting

USAGE:
  python launch_kimera.py

TROUBLESHOOTING:
1. Make sure you're in or near the Kimera project directory
2. Ensure Python 3.10+ is installed
3. Run the deployment script first: python deploy_kimera.py
4. Check that all files exist: kimera.py, backend/, requirements.txt

REQUIREMENTS:
• Python 3.10 or higher
• Kimera project files
• Dependencies installed (run deploy_kimera.py first)
""", 'white')

def main():
    """Main launcher function"""
    print_banner()
    
    # Find Kimera project
    print_colored("🔍 Looking for Kimera project...", 'blue')
    project_root = find_kimera_project()
    
    if not project_root:
        print_colored("❌ Kimera project not found!", 'red')
        print_colored("💡 Make sure you're in the Kimera project directory or a subdirectory", 'yellow')
        print_colored("   Look for a directory containing: kimera.py, backend/, requirements.txt", 'white')
        print_colored("\n   If you need to set up Kimera, run: python deploy_kimera.py", 'cyan')
        return False
    
    print_colored(f"✅ Found Kimera project: {project_root}", 'green')
    
    # Find Python executable
    python_exe = find_python_executable(project_root)
    print_colored(f"🐍 Using Python: {python_exe}", 'blue')
    
    # Check dependencies
    print_colored("📦 Checking dependencies...", 'blue')
    if not check_dependencies(python_exe):
        print_colored("❌ Dependencies not available!", 'red')
        print_colored("💡 Please run the deployment script first:", 'yellow')
        print_colored("   python deploy_kimera.py", 'cyan')
        return False
    
    print_colored("✅ Dependencies OK", 'green')
    
    # Start Kimera
    return start_kimera(project_root, python_exe)

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            print_colored("\n❌ Failed to start Kimera", 'red')
            print_colored("💡 Try running: python deploy_kimera.py", 'yellow')
            sys.exit(1)
    except KeyboardInterrupt:
        print_colored("\n⏹️  Launcher interrupted", 'yellow')
        sys.exit(0)
    except Exception as e:
        print_colored(f"\n❌ Unexpected error: {str(e)}", 'red')
        sys.exit(1) 