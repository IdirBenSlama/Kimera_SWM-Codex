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
    print("=== COMPREHENSIVE ENVIRONMENT DIAGNOSTIC ===\n")
    
    # Basic Python info
    print("1. PYTHON INFORMATION:")
    print(f"   Executable: {sys.executable}")
    print(f"   Version: {sys.version}")
    print(f"   Platform: {sys.platform}")
    print(f"   Prefix: {sys.prefix}")
    print(f"   Base Prefix: {sys.base_prefix}")
    print(f"   Virtual Env: {sys.prefix != sys.base_prefix}")
    
    # Environment variables
    print("\n2. ENVIRONMENT VARIABLES:")
    print(f"   PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
    print(f"   PATH (first 5 entries):")
    path_entries = os.environ.get('PATH', '').split(os.pathsep)[:5]
    for entry in path_entries:
        print(f"      - {entry}")
    
    # sys.path
    print("\n3. PYTHON MODULE SEARCH PATH (sys.path):")
    for i, path in enumerate(sys.path[:10]):  # First 10 entries
        print(f"   [{i}] {path}")
    
    # Check critical modules
    print("\n4. MODULE LOCATIONS:")
    modules_to_check = [
        "pytest", "aiofiles", "yaml", "numpy", "torch", 
        "fastapi", "pydantic", "sqlalchemy"
    ]
    for module in modules_to_check:
        print(f"   {module}: {check_module_location(module)}")
    
    # Check pytest executable
    print("\n5. PYTEST EXECUTABLE CHECK:")
    
    # Try different ways to run pytest
    commands = [
        ("python -m pytest --version", "python -m pytest"),
        ("pytest --version", "pytest command"),
        (f"{sys.executable} -m pytest --version", "explicit python -m pytest"),
        (r".venv\Scripts\pytest.exe --version", "venv pytest.exe"),
        (r".venv\Scripts\python.exe -m pytest --version", "venv python -m pytest")
    ]
    
    for cmd, desc in commands:
        print(f"\n   Trying: {desc}")
        stdout, stderr, code = run_command(cmd)
        print(f"   Return code: {code}")
        if stdout:
            print(f"   Output: {stdout.strip()}")
        if stderr:
            print(f"   Error: {stderr.strip()}")
    
    # Check if we can import backend modules
    print("\n6. BACKEND MODULE IMPORT TEST:")
    
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
            print(f"   ✓ Successfully imported: {module}")
        except ImportError as e:
            print(f"   ✗ Failed to import {module}: {e}")
        except Exception as e:
            print(f"   ✗ Error importing {module}: {type(e).__name__}: {e}")
    
    sys.path = original_path
    
    # Check site-packages
    print("\n7. SITE-PACKAGES LOCATIONS:")
    import site
    for path in site.getsitepackages():
        print(f"   - {path}")
        if os.path.exists(path):
            # Count packages
            try:
                packages = len([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
                print(f"     ({packages} packages found)")
            except OSError as e:
                print(f"     (Could not count packages: {e})")
    
    # Check if __init__.py files exist
    print("\n8. CHECKING __init__.py FILES:")
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
            print(f"   ✓ {init_file} exists")
        else:
            print(f"   ✗ {init_file} MISSING")
    
    print("\n=== DIAGNOSTIC COMPLETE ===")

if __name__ == "__main__":
    main()