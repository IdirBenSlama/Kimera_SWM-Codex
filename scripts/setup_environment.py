#!/usr/bin/env python3
"""
KIMERA SWM - Environment Setup Script
====================================

This script creates the optimal environment configuration for Kimera deployment.
It handles environment variables, directory structure, and basic configuration.

Usage:
    python setup_environment.py
"""

import os
import sys
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

def create_environment_file():
    """Create comprehensive environment configuration"""
    
    env_content = """# KIMERA SWM Environment Configuration
# =====================================
# This file contains all configuration settings for Kimera SWM.
# Copy this file to .env and customize the settings below.

# =============================================================================
# SERVER CONFIGURATION
# =============================================================================

# Environment mode (development, production, testing)
KIMERA_ENV=development

# Server host and port
KIMERA_HOST=127.0.0.1
KIMERA_PORT=8000

# Enable auto-reload in development
KIMERA_RELOAD=true

# Logging level (DEBUG, INFO, WARNING, ERROR)
KIMERA_LOG_LEVEL=INFO

# =============================================================================
# SECURITY SETTINGS
# =============================================================================

# Secret key for session management (change in production!)
KIMERA_SECRET_KEY=# Generate with: python -c "import secrets; logger.info(secrets.token_urlsafe(32))"

# API key for authentication (change this!)
KIMERA_API_KEY=# Set your actual API key here

# Rate limiting
KIMERA_RATE_LIMIT_ENABLED=true
KIMERA_RATE_LIMIT_REQUESTS=100
KIMERA_RATE_LIMIT_WINDOW=60

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

# Database URL (SQLite by default, can use PostgreSQL)
DATABASE_URL=sqlite:///data/kimera.db

# Database logging (set to false in production)
DATABASE_ECHO=false

# Connection pool settings
DATABASE_POOL_SIZE=5
DATABASE_MAX_OVERFLOW=10

# =============================================================================
# PERFORMANCE SETTINGS
# =============================================================================

# Maximum number of worker threads
KIMERA_MAX_THREADS=8

# GPU memory fraction (0.0 to 1.0)
KIMERA_GPU_MEMORY_FRACTION=0.8

# Enable GPU acceleration (set to false if no GPU)
KIMERA_USE_GPU=true

# Memory settings
KIMERA_MAX_MEMORY_MB=8192
KIMERA_CACHE_SIZE_MB=1024

# =============================================================================
# EXTERNAL API KEYS (Optional)
# =============================================================================

# OpenAI API key for AI features
OPENAI_API_KEY=# Set your OpenAI API key here

# Anthropic API key for Claude integration
ANTHROPIC_API_KEY=# Set your Anthropic API key here

# Other API keys
HUGGINGFACE_API_KEY=# Set your HuggingFace API key here

# =============================================================================
# MONITORING AND LOGGING
# =============================================================================

# Enable monitoring
KIMERA_MONITORING_ENABLED=true

# Enable metrics collection
KIMERA_METRICS_ENABLED=true

# Health check interval (seconds)
KIMERA_HEALTH_CHECK_INTERVAL=30

# Log file settings
KIMERA_LOG_FILE=logs/kimera.log
KIMERA_LOG_MAX_SIZE=10MB
KIMERA_LOG_BACKUP_COUNT=5

# =============================================================================
# DEVELOPMENT SETTINGS
# =============================================================================

# Enable debug mode (detailed error messages)
KIMERA_DEBUG=false

# Enable testing mode
KIMERA_TESTING=false

# Enable experimental features
KIMERA_EXPERIMENTAL=false

# Development tools
KIMERA_PROFILING=false
KIMERA_MEMORY_TRACKING=false

# =============================================================================
# ADVANCED SETTINGS
# =============================================================================

# Timeout settings (seconds)
KIMERA_REQUEST_TIMEOUT=30
KIMERA_STARTUP_TIMEOUT=60

# Batch processing settings
KIMERA_BATCH_SIZE=32
KIMERA_BATCH_TIMEOUT=10

# Cognitive engine settings
KIMERA_COGNITIVE_DEPTH=3
KIMERA_REASONING_STEPS=5

# =============================================================================
# CONTAINER SETTINGS (for Docker deployment)
# =============================================================================

# Container-specific settings
KIMERA_CONTAINER_MODE=false
KIMERA_CONTAINER_MEMORY_LIMIT=8G
KIMERA_CONTAINER_CPU_LIMIT=4

# =============================================================================
# BACKUP AND RECOVERY
# =============================================================================

# Backup settings
KIMERA_BACKUP_ENABLED=true
KIMERA_BACKUP_INTERVAL=3600
KIMERA_BACKUP_RETENTION=7

# Recovery settings
KIMERA_AUTO_RECOVERY=true
KIMERA_RECOVERY_TIMEOUT=300

# =============================================================================
# NETWORKING
# =============================================================================

# Network timeouts
KIMERA_CONNECT_TIMEOUT=10
KIMERA_READ_TIMEOUT=30

# Proxy settings (if needed)
# HTTP_PROXY=http://proxy.example.com:8080
# HTTPS_PROXY=https://proxy.example.com:8080

# =============================================================================
# FEATURE FLAGS
# =============================================================================

# Enable/disable specific features
KIMERA_FEATURE_CHAT=true
KIMERA_FEATURE_COGNITIVE=true
KIMERA_FEATURE_ANALYTICS=true
KIMERA_FEATURE_TRADING=false
KIMERA_FEATURE_RESEARCH=true

# =============================================================================
# CUSTOMIZATION
# =============================================================================

# Custom paths
KIMERA_DATA_PATH=data
KIMERA_LOGS_PATH=logs
KIMERA_CACHE_PATH=cache

# Custom settings
KIMERA_CUSTOM_CONFIG={}

# =============================================================================
# END OF CONFIGURATION
# =============================================================================

# Note: After modifying this file, restart Kimera for changes to take effect.
# For production deployment, ensure all sensitive values are properly secured.
"""

    # Create .env.example file
    env_example_path = Path(".env.example")
    with open(env_example_path, "w") as f:
        f.write(env_content)
    
    logger.info("✅ Created .env.example with comprehensive configuration")
    
    # Create .env file if it doesn't exist
    env_path = Path(".env")
    if not env_path.exists():
        with open(env_path, "w") as f:
            f.write(env_content)
        logger.info("✅ Created .env file (customize as needed)")
    else:
        logger.info("ℹ️  .env file already exists (not overwritten)")
    
    return True

def create_directories():
    """Create required directories"""
    directories = [
        "data",
        "logs", 
        "cache",
        "backups",
        "temp"
    ]
    
    for directory in directories:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"✅ Created directory: {directory}")
        else:
            logger.info(f"ℹ️  Directory already exists: {directory}")
    
    return True

def create_gitignore():
    """Create or update .gitignore file"""
    gitignore_content = """# Kimera SWM - Generated files and directories to ignore

# Environment files
.env
.env.local
.env.*.local

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
venv/
.venv/
env/
ENV/
env.bak/
venv.bak/

# IDE and editors
.vscode/
.idea/
*.swp
*.swo
*~

# OS generated files
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Logs and data
logs/
*.log
data/
cache/
temp/
backups/

# Model files and large data
*.bin
*.safetensors
*.pkl
*.joblib
*.h5
*.model

# Jupyter notebooks
.ipynb_checkpoints

# Docker
.dockerignore

# Test coverage
htmlcov/
.coverage
.coverage.*
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Local configuration
local_settings.py
secrets.json

# Kimera specific
kimera.db
kimera.db-journal
*.backup
"""
    
    gitignore_path = Path(".gitignore")
    if not gitignore_path.exists():
        with open(gitignore_path, "w") as f:
            f.write(gitignore_content)
        logger.info("✅ Created .gitignore file")
    else:
        logger.info("ℹ️  .gitignore file already exists")
    
    return True

def main():
    """Main setup function"""
    logger.info("🔧 KIMERA SWM - Environment Setup")
    logger.info("=" * 40)
    
    # Create environment configuration
    logger.info("\n📄 Creating environment configuration...")
    if not create_environment_file():
        logger.info("❌ Failed to create environment configuration")
        return False
    
    # Create required directories
    logger.info("\n📁 Creating directories...")
    if not create_directories():
        logger.info("❌ Failed to create directories")
        return False
    
    # Create .gitignore
    logger.info("\n📝 Creating .gitignore...")
    if not create_gitignore():
        logger.info("❌ Failed to create .gitignore")
        return False
    
    logger.info("\n✅ Environment setup complete!")
    logger.info("\n📋 Next steps:")
    logger.info("1. Edit .env file with your specific settings")
    logger.info("2. Add your API keys to .env")
    logger.info("3. Run: python deploy_kimera.py")
    logger.info("4. Start Kimera: python kimera.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 