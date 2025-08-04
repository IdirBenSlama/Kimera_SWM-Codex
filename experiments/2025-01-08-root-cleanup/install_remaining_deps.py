#!/usr/bin/env python3
"""
Install all remaining dependencies for Kimera SWM
"""

import subprocess
import sys
import os
import logging
logger = logging.getLogger(__name__)

# Get the venv Python path
venv_pip = os.path.join("venv_py313", "Scripts", "pip")

# List of packages that might be missing
packages = [
    # OpenTelemetry instrumentation
    "opentelemetry-instrumentation-httpx",
    "opentelemetry-exporter-prometheus",
    "opentelemetry-exporter-jaeger",
    
    # Additional ML/AI packages
    "sentence-transformers",
    "bitsandbytes-windows",  # Windows-specific version
    "einops",
    "xformers",
    
    # Database and caching
    "redis",
    "aioredis",
    "motor",
    "pymongo",
    
    # Additional utilities
    "httpx",
    "aiofiles",
    "python-multipart",
    "email-validator",
    "orjson",
    "ujson",
    "itsdangerous",
    "starlette",
    "anyio",
    
    # Monitoring and logging
    "structlog",
    "sentry-sdk",
    
    # Testing
    "pytest",
    "pytest-asyncio",
    "pytest-cov",
    
    # Documentation
    "mkdocs",
    "mkdocs-material",
    
    # Code quality
    "black",
    "isort",
    "flake8",
    "mypy",
    
    # Additional scientific computing
    "statsmodels",
    "sympy",
    "networkx",
    
    # Visualization
    "bokeh",
    "altair",
    "graphviz",
    
    # NLP
    "spacy",
    "nltk",
    "textblob",
]

def install_package(package):
    """Try to install a package, return True if successful"""
    try:
        logger.info(f"Installing {package}...")
        result = subprocess.run([venv_pip, "install", package], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"✓ {package} installed successfully")
            return True
        else:
            logger.info(f"✗ {package} failed: {result.stderr.split('ERROR:')[-1].strip()[:100]}...")
            return False
    except Exception as e:
        logger.info(f"✗ {package} error: {str(e)}")
        return False

def main():
    logger.info("Installing remaining dependencies for Kimera SWM...")
    logger.info("=" * 60)
    
    failed = []
    succeeded = []
    
    for package in packages:
        if install_package(package):
            succeeded.append(package)
        else:
            failed.append(package)
    
    logger.info("\n" + "=" * 60)
    logger.info(f"✓ Successfully installed: {len(succeeded)} packages")
    if failed:
        logger.info(f"✗ Failed to install: {len(failed)} packages")
        logger.info("Failed packages:", ", ".join(failed))
    
    # Try to install the specific httpx instrumentation
    logger.info("\nTrying alternative OpenTelemetry packages...")
    alt_packages = [
        "opentelemetry-instrumentation-httpx",
        "opentelemetry-instrumentation-aiohttp-client",
        "opentelemetry-instrumentation-urllib3",
        "opentelemetry-instrumentation-urllib",
    ]
    
    for package in alt_packages:
        install_package(package)

if __name__ == "__main__":
    main()