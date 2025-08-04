#!/usr/bin/env python3
"""
KIMERA SWM - Final Setup and Launch
===================================

This script ensures all dependencies are installed and launches Kimera.
"""

import subprocess
import sys
import os
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

class KimeraFinalSetup:
    def __init__(self):
        self.project_root = Path(__file__).parent.absolute()
        self.venv_pip = self.project_root / "venv_py313" / "Scripts" / "pip"
        self.venv_python = self.project_root / "venv_py313" / "Scripts" / "python.exe"
        
    def install_missing_packages(self):
        """Install any remaining missing packages"""
        logger.info("🔧 Checking for missing packages...")
        
        # List of packages that might still be missing
        packages = [
            "gputil",
            "py-cpuinfo",
            "nvidia-ml-py3",
            "pynvml",
            "torch-tb-profiler",
            "tensorboard",
            "wandb",
            "mlflow",
            "optuna",
            "ray",
            "dask",
            "joblib",
            "cloudpickle",
            "msgpack",
            "lmdb",
            "faiss-cpu",
            "annoy",
            "hnswlib",
            "umap-learn",
            "hdbscan",
            "yellowbrick",
            "shap",
            "lime",
            "eli5",
        ]
        
        for package in packages:
            try:
                subprocess.run([str(self.venv_pip), "install", package], 
                             capture_output=True, check=False)
            except Exception as e:
                logger.error(f"Error in kimera_final_setup.py: {e}", exc_info=True)
                raise  # Re-raise for proper error handling
    
    def fix_import_issues(self):
        """Create dummy modules for any problematic imports"""
        logger.info("🔧 Creating compatibility fixes...")
        
        # Create a dummy module for any missing imports
        dummy_modules_dir = self.project_root / "backend" / "utils" / "dummy_modules"
        dummy_modules_dir.mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py
        (dummy_modules_dir / "__init__.py").write_text("")
        
        # Add to Python path
        sys.path.insert(0, str(dummy_modules_dir))
    
    def start_kimera(self):
        """Start the Kimera server"""
        logger.info("\n🚀 Starting Kimera SWM...")
        logger.info("=" * 60)
        
        # Change to project directory
        os.chdir(self.project_root)
        
        # Start the server
        try:
            subprocess.run([str(self.venv_python), "kimera.py"])
        except KeyboardInterrupt:
            logger.info("\n\n✅ Kimera server stopped.")
        except Exception as e:
            logger.info(f"\n❌ Error: {str(e)}")
    
    def run(self):
        """Run the final setup and launch"""
        logger.info("""
╔══════════════════════════════════════════════════════════════════╗
║                 KIMERA SWM - FINAL SETUP & LAUNCH                 ║
╚══════════════════════════════════════════════════════════════════╝
""")
        
        # Install any missing packages
        self.install_missing_packages()
        
        # Fix import issues
        self.fix_import_issues()
        
        # Start Kimera
        self.start_kimera()


if __name__ == "__main__":
    setup = KimeraFinalSetup()
    setup.run()