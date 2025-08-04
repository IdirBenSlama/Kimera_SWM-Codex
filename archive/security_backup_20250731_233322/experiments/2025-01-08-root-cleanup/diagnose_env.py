"""
Comprehensive environment diagnostic script to identify pytest and module loading issues.
"""

import sys
import os
import subprocess
import importlib.util
from pathlib import Path

def run_command(cmd, shell=True):
    """Run a command and return its output."""
    try:
        result = subprocess.run(cmd, shell=shell, capture_output=True, text=True)
        return result.stdout, result.stderr, result.returncode
    except Exception as e:
        return "", str(e), -1

def check_module_location(module_name):
    """Check where a module is located."""
    try:
        spec = importlib.util.find_spec(module_name)
        if spec and spec.origin:
            return f"Found at: {spec.origin}"
        else:
            return "Module spec found but no origin"
    except ImportError:
        return "Module not found"
    except Exception as e:
        return f"Error: {e}"

def main():
    logger.info("=== COMPREHENSIVE ENVIRONMENT DIAGNOSTIC ===\n")
    
    # Basic Python info
    logger.info("1. PYTHON INFORMATION:")
    logger.info(f"   Executable: {sys.executable}")
    logger.info(f"   Version: {sys.version}")
    logger.info(f"   Platform: {sys.platform}")
    logger.info(f"   Prefix: {sys.prefix}")
    logger.info(f"   Base Prefix: {sys.base_prefix}")
    logger.info(f"   Virtual Env: {sys.prefix != sys.base_prefix}")
    
    # Environment variables
    logger.info("\n2. ENVIRONMENT VARIABLES:")
    logger.info(f"   PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
    logger.info(f"   PATH (first 5 entries):")
    path_entries = os.environ.get('PATH', '').split(os.pathsep)[:5]
    for entry in path_entries:
        logger.info(f"      - {entry}")
    
    # sys.path
    logger.info("\n3. PYTHON MODULE SEARCH PATH (sys.path):")
    for i, path in enumerate(sys.path[:10]):  # First 10 entries
        logger.info(f"   [{i}] {path}")
    
    # Check critical modules
    logger.info("\n4. MODULE LOCATIONS:")
    modules_to_check = [
        "pytest", "aiofiles", "yaml", "numpy", "torch", 
        "fastapi", "pydantic", "sqlalchemy"
    ]
    for module in modules_to_check:
        logger.info(f"   {module}: {check_module_location(module)}")
    
    # Check pytest executable
    logger.info("\n5. PYTEST EXECUTABLE CHECK:")
    
    # Try different ways to run pytest
    commands = [
        ("python -m pytest --version", "python -m pytest"),
        ("pytest --version", "pytest command"),
        (f"{sys.executable} -m pytest --version", "explicit python -m pytest"),
        (r".venv\Scripts\pytest.exe --version", "venv pytest.exe"),
        (r".venv\Scripts\python.exe -m pytest --version", "venv python -m pytest")
    ]
    
    for cmd, desc in commands:
        logger.info(f"\n   Trying: {desc}")
        stdout, stderr, code = run_command(cmd)
        logger.info(f"   Return code: {code}")
        if stdout:
            logger.info(f"   Output: {stdout.strip()}")
        if stderr:
            logger.info(f"   Error: {stderr.strip()}")
    
    # Check if we can import backend modules
    logger.info("\n6. BACKEND MODULE IMPORT TEST:")
    
    # Add the current directory to sys.path temporarily
    original_path = sys.path.copy()
    sys.path.insert(0, os.getcwd())
    
    test_imports = [
        "backend",
        "backend.core",
        "backend.core.ethical_governor",
        "backend.core.heart"
    ]
    
    for module in test_imports:
        try:
            importlib.import_module(module)
            logger.info(f"   âœ“ Successfully imported: {module}")
        except ImportError as e:
            logger.info(f"   âœ— Failed to import {module}: {e}")
        except Exception as e:
            logger.info(f"   âœ— Error importing {module}: {type(e).__name__}: {e}")
    
    sys.path = original_path
    
    # Check site-packages
    logger.info("\n7. SITE-PACKAGES LOCATIONS:")
    import site
import logging
logger = logging.getLogger(__name__)
    for path in site.getsitepackages():
        logger.info(f"   - {path}")
        if os.path.exists(path):
            # Count packages
            try:
                packages = len([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
                logger.info(f"     ({packages} packages found)")
            except OSError as e:
                logger.info(f"     (Could not count packages: {e})")
    
    # Check if __init__.py files exist
    logger.info("\n8. CHECKING __init__.py FILES:")
    init_files = [
        "backend/__init__.py",
        "backend/core/__init__.py",
        "backend/engines/__init__.py",
        "backend/vault/__init__.py",
        "tests/__init__.py"
    ]
    
    for init_file in init_files:
        path = Path(init_file)
        if path.exists():
            logger.info(f"   âœ“ {init_file} exists")
        else:
            logger.info(f"   âœ— {init_file} MISSING")
    
    logger.info("\n=== DIAGNOSTIC COMPLETE ===")

if __name__ == "__main__":
    main()