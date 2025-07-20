#!/usr/bin/env python3
"""
🚀 KIMERA SYSTEM LAUNCHER - ULTIMATE STARTUP SOLUTION
====================================================

This script provides a foolproof way to start KIMERA from anywhere.
It handles environment setup, dependency checking, and multiple startup methods.

Usage:
    python start_kimera_ultimate.py [method]
    
Methods:
    - server (default): Start full KIMERA API server
    - dev: Start in development mode with auto-reload
    - test: Run system tests first, then start
    - simple: Start minimal server for testing
    - docker: Start using Docker
    - help: Show detailed help

Author: KIMERA AI System
Version: 1.0.0 - Ultimate Startup Solution
"""

import os
import sys
import subprocess
import time
import logging
import json
from pathlib import Path
from typing import List, Dict, Optional
import argparse
import platform

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('kimera_startup.log')
    ]
)
logger = logging.getLogger(__name__)

class KimeraLauncher:
    """Comprehensive KIMERA system launcher"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.absolute()
        self.venv_path = self.project_root / ".venv"
        self.requirements_file = self.project_root / "requirements.txt"
        self.python_exe = self._get_python_executable()
        self.system_info = self._get_system_info()
        
    def _get_system_info(self) -> Dict:
        """Get system information for diagnostics"""
        return {
            "platform": platform.system(),
            "python_version": sys.version,
            "cwd": os.getcwd(),
            "project_root": str(self.project_root),
            "venv_exists": self.venv_path.exists(),
            "requirements_exists": self.requirements_file.exists()
        }
    
    def _get_python_executable(self) -> str:
        """Get the correct Python executable"""
        if platform.system() == "Windows":
            if self.venv_path.exists():
                return str(self.venv_path / "Scripts" / "python.exe")
            return "python"
        else:
            if self.venv_path.exists():
                return str(self.venv_path / "bin" / "python")
            return "python3"
    
    def print_banner(self):
        """Print KIMERA startup banner"""
        banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🚀 KIMERA SYSTEM LAUNCHER v1.0.0                         ║
║              Kinetic Intelligence for Multidimensional Analysis              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  🎯 Comprehensive Startup Solution - No More Running Problems!              ║
║  🔧 Auto-detects environment, handles dependencies, multiple methods        ║
║  📍 Project Root: {}
║  🐍 Python: {}
║  💻 Platform: {}
╚══════════════════════════════════════════════════════════════════════════════╝
        """.format(
            str(self.project_root),
            f"{sys.version.split()[0]} ({'venv' if self.venv_path.exists() else 'system'})",
            f"{platform.system()} {platform.release()}"
        )
        print(banner)
    
    def check_environment(self) -> bool:
        """Check if environment is properly set up"""
        logger.info("🔍 Checking environment setup...")
        
        issues = []
        
        # Check if we're in the right directory
        expected_files = ["backend", "requirements.txt", "README.md"]
        for file in expected_files:
            if not (self.project_root / file).exists():
                issues.append(f"Missing {file} - are you in the KIMERA project directory?")
        
        # Check Python version
        if sys.version_info < (3, 10):
            issues.append(f"Python 3.10+ required, found {sys.version}")
        
        # Check virtual environment
        if not self.venv_path.exists():
            logger.warning("⚠️  Virtual environment not found - will use system Python")
        
        # Check requirements file
        if not self.requirements_file.exists():
            issues.append("requirements.txt not found")
        
        if issues:
            logger.error("❌ Environment issues found:")
            for issue in issues:
                logger.error(f"   • {issue}")
            return False
        
        logger.info("✅ Environment check passed")
        return True
    
    def setup_environment(self) -> bool:
        """Set up the environment if needed"""
        logger.info("🔧 Setting up environment...")
        
        try:
            # Create virtual environment if it doesn't exist
            if not self.venv_path.exists():
                logger.info("Creating virtual environment...")
                subprocess.run([sys.executable, "-m", "venv", str(self.venv_path)], check=True)
                logger.info("✅ Virtual environment created")
            
            # Install/update requirements
            if self.requirements_file.exists():
                logger.info("Installing requirements...")
                subprocess.run([
                    self.python_exe, "-m", "pip", "install", "-r", str(self.requirements_file)
                ], check=True)
                logger.info("✅ Requirements installed")
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Environment setup failed: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected error during setup: {e}")
            return False
    
    def check_dependencies(self) -> bool:
        """Check if key dependencies are available"""
        logger.info("📦 Checking key dependencies...")
        
        key_deps = [
            "fastapi",
            "uvicorn", 
            "torch",
            "transformers",
            "sqlalchemy",
            "numpy"
        ]
        
        missing = []
        for dep in key_deps:
            try:
                subprocess.run([
                    self.python_exe, "-c", f"import {dep}"
                ], check=True, capture_output=True)
            except subprocess.CalledProcessError:
                missing.append(dep)
        
        if missing:
            logger.error(f"❌ Missing dependencies: {', '.join(missing)}")
            logger.info("💡 Run with --setup to install dependencies")
            return False
        
        logger.info("✅ All key dependencies available")
        return True
    
    def start_server(self, method: str = "server") -> bool:
        """Start KIMERA server using specified method"""
        logger.info(f"🚀 Starting KIMERA server using method: {method}")
        
        # Change to project directory
        os.chdir(self.project_root)
        
        # Define startup commands for different methods
        commands = {
            "server": [
                self.python_exe, "-m", "uvicorn", 
                "backend.api.main:app", 
                "--host", "0.0.0.0", 
                "--port", "8001"
            ],
            "dev": [
                self.python_exe, "-m", "uvicorn", 
                "backend.api.main:app", 
                "--host", "0.0.0.0", 
                "--port", "8001", 
                "--reload"
            ],
            "simple": [
                self.python_exe, "-c",
                "from backend.api.main import app; import uvicorn; uvicorn.run(app, host='0.0.0.0', port=8001)"
            ],
            "direct": [
                self.python_exe, "backend/api/main.py"
            ]
        }
        
        if method not in commands:
            logger.error(f"❌ Unknown startup method: {method}")
            return False
        
        cmd = commands[method]
        
        try:
            logger.info(f"Executing: {' '.join(cmd)}")
            logger.info("🌐 KIMERA will be available at: http://localhost:8001")
            logger.info("📚 API docs at: http://localhost:8001/docs")
            logger.info("🔍 Health check: http://localhost:8001/system/health")
            logger.info("Press Ctrl+C to stop the server")
            
            # Start the server
            subprocess.run(cmd, check=True)
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Server startup failed: {e}")
            return False
        except KeyboardInterrupt:
            logger.info("\n✋ Server stopped by user")
            return True
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
            return False
    
    def run_tests(self) -> bool:
        """Run system tests before starting"""
        logger.info("🧪 Running system tests...")
        
        test_commands = [
            [self.python_exe, "-c", "import backend.api.main; print('✅ Main module imports OK')"],
            [self.python_exe, "-c", "from backend.core.embedding_utils import encode_text; print('✅ Embedding utils OK')"],
            [self.python_exe, "-c", "import torch; print(f'✅ PyTorch OK - CUDA: {torch.cuda.is_available()}')"],
        ]
        
        for cmd in test_commands:
            try:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                logger.info(result.stdout.strip())
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Test failed: {' '.join(cmd)}")
                logger.error(f"Error: {e.stderr}")
                return False
        
        logger.info("✅ All tests passed")
        return True
    
    def start_docker(self) -> bool:
        """Start KIMERA using Docker"""
        logger.info("🐳 Starting KIMERA with Docker...")
        
        if not (self.project_root / "docker-compose.yml").exists():
            logger.error("❌ docker-compose.yml not found")
            return False
        
        try:
            subprocess.run(["docker-compose", "up", "-d"], check=True)
            logger.info("✅ KIMERA started with Docker")
            logger.info("🌐 Available at: http://localhost:8001")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Docker startup failed: {e}")
            return False
    
    def show_help(self):
        """Show detailed help information"""
        help_text = """
🚀 KIMERA SYSTEM LAUNCHER - DETAILED HELP
========================================

USAGE:
    python start_kimera_ultimate.py [METHOD] [OPTIONS]

STARTUP METHODS:
    server (default)  - Start full KIMERA API server (production mode)
    dev              - Start with auto-reload for development
    test             - Run tests first, then start server
    simple           - Start minimal server (fallback method)
    docker           - Start using Docker Compose
    help             - Show this help message

OPTIONS:
    --setup          - Set up environment (create venv, install deps)
    --check          - Only check environment, don't start
    --force          - Skip environment checks
    --port PORT      - Use custom port (default: 8001)
    --host HOST      - Use custom host (default: 0.0.0.0)

EXAMPLES:
    python start_kimera_ultimate.py                    # Start server normally
    python start_kimera_ultimate.py dev                # Development mode
    python start_kimera_ultimate.py --setup            # Set up environment first
    python start_kimera_ultimate.py test               # Test then start
    python start_kimera_ultimate.py docker             # Use Docker

TROUBLESHOOTING:
    1. Environment Issues:
       python start_kimera_ultimate.py --setup

    2. Dependency Problems:
       pip install -r requirements.txt

    3. Permission Issues (Windows):
       Run as Administrator

    4. Port Already in Use:
       python start_kimera_ultimate.py --port 8002

    5. Still Not Working:
       python start_kimera_ultimate.py --check
       Check kimera_startup.log for details

SYSTEM REQUIREMENTS:
    - Python 3.10+
    - 8GB+ RAM (16GB+ recommended)
    - NVIDIA GPU (optional but recommended)
    - Windows 10+/Linux/macOS

For more help, see README.md or visit the documentation.
        """
        print(help_text)
    
    def diagnostic_report(self):
        """Generate diagnostic report"""
        logger.info("🔍 Generating diagnostic report...")
        
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system_info": self.system_info,
            "environment": {
                "python_executable": self.python_exe,
                "virtual_env": str(self.venv_path) if self.venv_path.exists() else None,
                "requirements_file": str(self.requirements_file) if self.requirements_file.exists() else None
            },
            "checks": {}
        }
        
        # Run various checks
        checks = [
            ("Python Version", lambda: sys.version_info >= (3, 10)),
            ("Project Directory", lambda: (self.project_root / "backend").exists()),
            ("Requirements File", lambda: self.requirements_file.exists()),
            ("Virtual Environment", lambda: self.venv_path.exists()),
        ]
        
        for name, check_func in checks:
            try:
                result = check_func()
                report["checks"][name] = {"status": "✅ PASS" if result else "❌ FAIL", "result": result}
            except Exception as e:
                report["checks"][name] = {"status": "❌ ERROR", "error": str(e)}
        
        # Save report
        report_file = self.project_root / "kimera_diagnostic_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📋 Diagnostic report saved to: {report_file}")
        
        # Print summary
        print("\n🔍 DIAGNOSTIC SUMMARY:")
        print("=" * 50)
        for name, result in report["checks"].items():
            print(f"{result['status']} {name}")
        print("=" * 50)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="KIMERA System Launcher")
    parser.add_argument("method", nargs="?", default="server", 
                       choices=["server", "dev", "test", "simple", "docker", "help"],
                       help="Startup method")
    parser.add_argument("--setup", action="store_true", help="Set up environment")
    parser.add_argument("--check", action="store_true", help="Only check environment")
    parser.add_argument("--force", action="store_true", help="Skip environment checks")
    parser.add_argument("--port", type=int, default=8001, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--diagnostic", action="store_true", help="Generate diagnostic report")
    
    args = parser.parse_args()
    
    launcher = KimeraLauncher()
    launcher.print_banner()
    
    # Handle special commands
    if args.method == "help":
        launcher.show_help()
        return
    
    if args.diagnostic:
        launcher.diagnostic_report()
        return
    
    # Set up environment if requested
    if args.setup:
        if not launcher.setup_environment():
            sys.exit(1)
    
    # Check environment unless forced to skip
    if not args.force:
        if not launcher.check_environment():
            logger.error("❌ Environment check failed. Use --setup to fix or --force to skip.")
            sys.exit(1)
        
        if not launcher.check_dependencies():
            logger.error("❌ Dependency check failed. Use --setup to install dependencies.")
            sys.exit(1)
    
    # Only check, don't start
    if args.check:
        logger.info("✅ Environment check completed successfully")
        return
    
    # Run tests if requested
    if args.method == "test":
        if not launcher.run_tests():
            logger.error("❌ Tests failed. Fix issues before starting server.")
            sys.exit(1)
        args.method = "server"  # Switch to server mode after tests
    
    # Start the server
    if args.method == "docker":
        success = launcher.start_docker()
    else:
        success = launcher.start_server(args.method)
    
    if not success:
        logger.error("❌ Failed to start KIMERA. Check logs for details.")
        logger.info("💡 Try: python start_kimera_ultimate.py --setup")
        sys.exit(1)

if __name__ == "__main__":
    main() 