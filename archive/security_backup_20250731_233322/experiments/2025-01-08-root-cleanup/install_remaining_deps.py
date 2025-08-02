#!/usr/bin/env python3
"""
Install all remaining dependencies for Kimera SWM
"""

import subprocess
import sys
import os

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
        print(f"Installing {package}...")
        result = subprocess.run([venv_pip, "install", package], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"âœ“ {package} installed successfully")
            return True
        else:
            print(f"âœ— {package} failed: {result.stderr.split('ERROR:')[-1].strip()[:100]}...")
            return False
    except Exception as e:
        print(f"âœ— {package} error: {str(e)}")
        return False

def main():
    print("Installing remaining dependencies for Kimera SWM...")
    print("=" * 60)
    
    failed = []
    succeeded = []
    
    for package in packages:
        if install_package(package):
            succeeded.append(package)
        else:
            failed.append(package)
    
    print("\n" + "=" * 60)
    print(f"âœ“ Successfully installed: {len(succeeded)} packages")
    if failed:
        print(f"âœ— Failed to install: {len(failed)} packages")
        print("Failed packages:", ", ".join(failed))
    
    # Try to install the specific httpx instrumentation
    print("\nTrying alternative OpenTelemetry packages...")
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