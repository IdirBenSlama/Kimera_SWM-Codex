#!/usr/bin/env python3
"""
KIMERA Windows Startup Script
=============================

Aerospace-grade startup sequence optimized for Windows environments.
Handles encoding issues, path problems, and provides graceful degradation.
"""

import os
import sys
import asyncio
import logging
import time
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import warnings

# Fix Windows-specific issues
if sys.platform == 'win32':
    # Set console code page to UTF-8
    os.system('chcp 65001 > nul 2>&1')
    
    # Configure Python for UTF-8
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    # Reconfigure stdout/stderr
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    
    # Fix asyncio for Windows
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Configure logging
log_format = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.FileHandler('kimera_startup.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Suppress noisy loggers
for noisy_logger in ['transformers', 'torch', 'urllib3', 'asyncio']:
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

class KimeraWindowsStartup:
    """Windows-optimized KIMERA startup manager."""
    
    def __init__(self):
        self.start_time = time.time()
        self.config = None
        self.components = {}
        self.startup_errors = []
        self.warnings = []
        
    async def startup(self) -> bool:
        """Execute full startup sequence."""
        logger.info("=" * 70)
        logger.info("KIMERA SYSTEM STARTUP - WINDOWS EDITION")
        logger.info(f"Startup Time: {datetime.now()}")
        logger.info(f"Python Version: {sys.version}")
        logger.info(f"Platform: {sys.platform}")
        logger.info("=" * 70)
        
        try:
            # Phase 1: Environment Setup
            if not await self._setup_environment():
                return False
            
            # Phase 2: Configuration
            if not await self._load_configuration():
                return False
            
            # Phase 3: Database Setup
            if not await self._setup_database():
                return False
            
            # Phase 4: Core Components
            if not await self._initialize_core_components():
                return False
            
            # Phase 5: API Server
            if not await self._start_api_server():
                return False
            
            # Success!
            elapsed = time.time() - self.start_time
            logger.info(f"\n[SUCCESS] KIMERA started in {elapsed:.2f} seconds")
            
            if self.warnings:
                logger.warning(f"\nStartup completed with {len(self.warnings)} warnings:")
                for warning in self.warnings:
                    logger.warning(f"  - {warning}")
            
            return True
            
        except Exception as e:
            logger.critical(f"STARTUP FAILED: {e}", exc_info=True)
            return False
    
    async def _setup_environment(self) -> bool:
        """Setup environment and check prerequisites."""
        logger.info("\n>>> PHASE 1: Environment Setup")
        logger.info("-" * 50)
        
        # Set critical environment variables
        env_vars = {
            'KIMERA_HOME': str(project_root),
            'PYTHONPATH': str(project_root),
            'TOKENIZERS_PARALLELISM': 'false',  # Prevent tokenizer warnings
            'TF_CPP_MIN_LOG_LEVEL': '2',  # Reduce TensorFlow verbosity
            'CUDA_LAUNCH_BLOCKING': '1',  # Better CUDA error messages
        }
        
        for key, value in env_vars.items():
            os.environ[key] = value
            logger.info(f"Set {key}={value}")
        
        # Check Python version
        if sys.version_info < (3, 11):
            logger.error(f"Python 3.11+ required, found {sys.version}")
            return False
        
        # Check critical imports
        critical_modules = ['torch', 'transformers', 'fastapi', 'sqlalchemy']
        missing = []
        
        for module in critical_modules:
            try:
                __import__(module)
                logger.info(f"[OK] {module} available")
            except ImportError:
                missing.append(module)
                logger.error(f"[FAIL] {module} not found")
        
        if missing:
            logger.error(f"Missing critical dependencies: {', '.join(missing)}")
            logger.error("Install with: pip install " + " ".join(missing))
            return False
        
        # Check GPU availability
        try:
            import torch
            if torch.cuda.is_available():
                logger.info(f"[OK] CUDA available: {torch.cuda.get_device_name(0)}")
            else:
                logger.warning("[WARN] No CUDA GPU detected - using CPU mode")
                self.warnings.append("Running in CPU mode - performance will be limited")
        except Exception as e:
            logger.warning(f"[WARN] GPU check failed: {e}")
        
        return True
    
    async def _load_configuration(self) -> bool:
        """Load and validate configuration."""
        logger.info("\n>>> PHASE 2: Configuration")
        logger.info("-" * 50)
        
        try:
            # First, set default database to SQLite for Windows
            if 'DATABASE_URL' not in os.environ:
                os.environ['DATABASE_URL'] = 'sqlite:///./kimera_swm.db'
                logger.info("Using SQLite database (default for Windows)")
            
            # Import configuration
            from src.config.kimera_config import get_config, ConfigProfile
            
            self.config = get_config()
            
            # Log configuration
            logger.info(f"Profile: {self.config.profile.value}")
            logger.info(f"Database: {self.config.database.url.split('@')[0]}...")
            logger.info(f"API Port: {self.config.server.port}")
            
            # Validate
            issues = self.config.validate()
            if issues:
                for issue in issues:
                    logger.warning(f"Config issue: {issue}")
                    self.warnings.append(f"Configuration: {issue}")
            
            # Security check
            if self.config.profile == ConfigProfile.PRODUCTION:
                if self.config.security.secret_key.startswith("CHANGE_THIS"):
                    logger.error("SECURITY: Default secret key in production!")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Configuration failed: {e}")
            
            # Create minimal config
            logger.info("Creating minimal configuration...")
            self._create_minimal_config()
            return True
    
    def _create_minimal_config(self):
        """Create minimal working configuration."""
        os.environ['DATABASE_URL'] = 'sqlite:///./kimera_swm.db'
        os.environ['KIMERA_PROFILE'] = 'development'
        os.environ['SECRET_KEY'] = 'dev-only-' + os.urandom(16).hex()
        logger.info("Minimal configuration created")
    
    async def _setup_database(self) -> bool:
        """Setup database connection."""
        logger.info("\n>>> PHASE 3: Database Setup")
        logger.info("-" * 50)
        
        try:
            from sqlalchemy import create_engine, text
            
            # Get database URL
            db_url = os.environ.get('DATABASE_URL', 'sqlite:///./kimera_swm.db')
            
            # Create engine
            engine = create_engine(db_url)
            
            # Test connection
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                logger.info(f"[OK] Database connected: {db_url.split('@')[0]}...")
                
                # Create tables if needed
                if 'sqlite' in db_url:
                    from src.vault.database import Base, get_engine
                    Base.metadata.create_all(bind=engine)
                    logger.info("[OK] Database tables created/verified")
            
            return True
            
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            
            # Try SQLite fallback
            if 'postgresql' in str(e).lower():
                logger.info("Falling back to SQLite...")
                os.environ['DATABASE_URL'] = 'sqlite:///./kimera_swm.db'
                return await self._setup_database()  # Retry with SQLite
            
            return False
    
    async def _initialize_core_components(self) -> bool:
        """Initialize core KIMERA components."""
        logger.info("\n>>> PHASE 4: Core Components")
        logger.info("-" * 50)
        
        components = [
            ("KimeraSystem", self._init_kimera_system),
            ("GovernanceEngine", self._init_governance),
            ("VaultManager", self._init_vault),
            ("TranslatorHub", self._init_translator),
        ]
        
        failed = []
        
        for name, init_func in components:
            try:
                logger.info(f"Initializing {name}...")
                component = await init_func()
                if component:
                    self.components[name] = component
                    logger.info(f"[OK] {name} initialized")
                else:
                    raise Exception("Initialization returned None")
            except Exception as e:
                logger.error(f"[FAIL] {name}: {e}")
                failed.append(name)
                self.startup_errors.append(f"{name}: {str(e)}")
        
        # Determine if we can continue
        critical_components = ["KimeraSystem", "GovernanceEngine"]
        critical_failed = [c for c in failed if c in critical_components]
        
        if critical_failed:
            logger.error(f"Critical components failed: {', '.join(critical_failed)}")
            return False
        elif failed:
            logger.warning(f"Non-critical components failed: {', '.join(failed)}")
            self.warnings.append(f"Some components unavailable: {', '.join(failed)}")
        
        return True
    
    async def _init_kimera_system(self):
        """Initialize KimeraSystem."""
        from src.core.kimera_system import get_kimera_system
        system = get_kimera_system()
        system.initialize()
        return system
    
    async def _init_governance(self):
        """Initialize Governance Engine."""
        from src.governance import GovernanceEngine, create_default_policies
        
        engine = GovernanceEngine()
        
        # Load default policies
        policies = create_default_policies()
        for policy in policies:
            engine.register_policy(policy)
            engine.activate_policy(policy.id)
        
        return engine
    
    async def _init_vault(self):
        """Initialize Vault Manager."""
        from src.vault.vault_manager import VaultManager
        return VaultManager()
    
    async def _init_translator(self):
        """Initialize Universal Translator Hub."""
        from src.engines.universal_translator_hub import create_universal_translator_hub
        return create_universal_translator_hub()
    
    async def _start_api_server(self) -> bool:
        """Start the API server."""
        logger.info("\n>>> PHASE 5: API Server")
        logger.info("-" * 50)
        
        try:
            from src.api.main import create_app
            import uvicorn
            
            # Create app
            app = create_app()
            
            # Get configuration
            host = self.config.server.host if self.config else "0.0.0.0"
            port = self.config.server.port if self.config else 8000
            
            logger.info(f"Starting API server on {host}:{port}")
            
            # Configure uvicorn
            config = uvicorn.Config(
                app,
                host=host,
                port=port,
                log_level="info",
                access_log=False,  # Reduce noise
                loop="asyncio",
                reload=False
            )
            
            # Create server
            server = uvicorn.Server(config)
            
            # Print startup message
            logger.info("\n" + "=" * 70)
            logger.info("KIMERA SYSTEM READY")
            logger.info(f"API: http://localhost:{port}")
            logger.info(f"Docs: http://localhost:{port}/docs")
            logger.info(f"Health: http://localhost:{port}/health")
            logger.info("=" * 70)
            logger.info("\nPress Ctrl+C to shutdown")
            
            # Run server
            await server.serve()
            
            return True
            
        except Exception as e:
            logger.error(f"API server failed: {e}")
            return False

async def main():
    """Main entry point."""
    startup = KimeraWindowsStartup()
    success = await startup.startup()
    
    if not success:
        logger.error("\nStartup failed. Check kimera_startup.log for details.")
        return 1
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nShutdown requested by user")
        exit_code = 0
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}", exc_info=True)
        exit_code = 1
    
    sys.exit(exit_code)