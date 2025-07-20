"""
A script to verify the Python environment and installed packages directly,
bypassing the pytest runner to diagnose environmental issues.
"""

import sys
import os
import pkg_resources
import logging

# Configure logging properly
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_package_version(package_name):
    try:
        version = pkg_resources.get_distribution(package_name).version
        logger.info(f"SUCCESS: Found {package_name} version {version}")
    except pkg_resources.DistributionNotFound:
        logger.error(f"ERROR: {package_name} is NOT installed.")
    except Exception as e:
        logger.error(f"ERROR: Could not check version for {package_name}. Reason: {e}")

def main():
    logger.info("--- Environment Verification ---")
    logger.info(f"Python Executable: {sys.executable}")
    logger.info("\n--- sys.path ---")
    for path in sys.path:
        logger.info(path)
    
    logger.info("\n--- Package Verification ---")
    required_packages = [
        "fastapi",
        "uvicorn",
        "pydantic",
        "sqlalchemy",
        "alembic",
        "torch",
        "transformers",
        "numpy",
        "pandas",
        "scikit-learn",
        "aiofiles",
        "python-dotenv",
        "requests",
        "httpx",
        "neo4j",
        "redis",
        "psutil",
        "loguru"
    ]
    
    for package in required_packages:
        check_package_version(package)
            
    logger.info("\n--- Verification Complete ---")
    
    logger.info("\n--- Attempting Direct Imports ---")
    try:
        import aiofiles
        logger.info("SUCCESS: import aiofiles")
    except ImportError:
        logger.error("ERROR: import aiofiles failed")
        
    try:
        import yaml
        logger.info("SUCCESS: import yaml")
    except ImportError:
        logger.error("ERROR: import yaml failed")
        
    try:
        import torch
        logger.info("SUCCESS: import torch")
    except ImportError:
        logger.error("ERROR: import torch failed")
        
    try:
        import numpy
        logger.info("SUCCESS: import numpy")
    except ImportError:
        logger.error("ERROR: import numpy failed")

if __name__ == "__main__":
    main() 