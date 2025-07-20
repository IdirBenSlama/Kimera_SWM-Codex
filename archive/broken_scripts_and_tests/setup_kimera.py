#!/usr/bin/env python3
"""
KIMERA SWM Setup Script
Checks dependencies and sets up the environment
"""
import subprocess
import sys
import os
from pathlib import Path

# Initialize structured logger
from backend.utils.kimera_logger import get_system_logger
logger = get_system_logger(__name__)


def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        logger.error("❌ Python 3.8+ is required")
        logger.info(f"Current version: {sys.version}")
        return False
    logger.info(f"✅ Python {sys.version.split()
    return True

def install_requirements():
    """Install requirements from requirements.txt"""
    logger.info("📦 Installing requirements...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        logger.info("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install requirements: {e}")
        logger.info("🔄 Trying to install core dependencies...")
        
        # Try installing core dependencies only
        core_deps = [
            "fastapi>=0.115.12",
            "uvicorn[standard]>=0.34.3", 
            "pydantic>=2.11.5",
            "sqlalchemy>=2.0.41",
            "numpy>=1.24.0",
            "torch>=2.0.0",
            "transformers>=4.44.2"
        ]
        
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install"
            ] + core_deps)
            logger.info("✅ Core dependencies installed")
            return True
        except subprocess.CalledProcessError:
            logger.error("❌ Failed to install core dependencies")
            return False

def check_dependencies():
    """Check if key dependencies are available"""
    required_modules = [
        "fastapi",
        "uvicorn", 
        "pydantic",
        "sqlalchemy",
        "numpy"
    ]
    
    missing = []
    for module in required_modules:
        try:
            __import__(module)
            logger.info(f"✅ {module}")
        except ImportError:
            logger.error(f"❌ {module}")
            missing.append(module)
    
    return len(missing) == 0

def create_directories():
    """Create necessary directories"""
    dirs = [
        "static/images",
        "test_databases",
        "logs"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Created directory: {dir_path}")

def main():
    logger.info("🚀 KIMERA SWM Setup")
    logger.info("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check if dependencies are already installed
    logger.debug("\n🔍 Checking dependencies...")
    if not check_dependencies():
        logger.info("\n📦 Installing missing dependencies...")
        if not install_requirements():
            logger.error("\n❌ Setup failed. Please install dependencies manually:")
            logger.info("pip install -r requirements.txt")
            sys.exit(1)
    else:
        logger.info("✅ All dependencies are available")
    
    # Create directories
    logger.info("\n📁 Creating directories...")
    create_directories()
    
    logger.info("\n✅ Setup complete!")
    logger.info("\nYou can now run KIMERA with:")
    logger.info("  python run_kimera.py")
    logger.info("\nOr use the batch file:")
    logger.info("  start_kimera.bat")

if __name__ == "__main__":
    main()