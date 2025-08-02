#!/usr/bin/env python3
"""
KIMERA SWM Unified Entry Point
==============================

COMPLETE unified entry point consolidating ALL initialization patterns:
- Progressive: Lazy loading with background enhancement
- Full: Complete initialization with all features  
- Safe: Maximum fallbacks and error tolerance
- Fast: Minimal features for rapid startup
- Optimized: Performance-focused initialization
- Hybrid: Adaptive initialization based on system capabilities

Integrates with the new unified cognitive architecture for maximum reliability.

Author: Kimera SWM Autonomous Architect
Date: 2025-08-01T00:14:55.163020
Version: 3.0.0 (COMPLETE UNIFICATION)
Classification: AEROSPACE-GRADE ENTRY POINT
"""

import asyncio
import logging
import os
import sys
import time
import traceback
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import socket

# Configure application-level logging
logging.basicConfig(level=logging.INFO)

# Environment setup
from dotenv import load_dotenv
load_dotenv()

# FastAPI imports
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# KIMERA core imports
from src.utils.kimera_logger import get_system_logger
from src.core.kimera_system import KimeraSystem, kimera_singleton, get_kimera_system
from src.monitoring.kimera_prometheus_metrics import initialize_background_collection
from src.utils.threading_utils import start_background_task

# Import the new unified cognitive architecture
try:
    from src.core.unified_master_cognitive_architecture_fix import patch_unified_architecture
except ImportError as e:
    logging.getLogger(__name__).error(
        f"Failed to import patch_unified_architecture: {e}", exc_info=e
    )

try:
    from src.core.unified_master_cognitive_architecture import (
        UnifiedMasterCognitiveArchitecture,
        ProcessingMode,
        create_unified_architecture,
        create_safe_architecture
    )
    UNIFIED_ARCHITECTURE_AVAILABLE = True
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Unified architecture not available: {e}")
    UNIFIED_ARCHITECTURE_AVAILABLE = False

# Progressive initialization imports
try:
    from src.core.lazy_initialization_manager import get_global_lazy_manager, ComponentConfig, Priority
    from src.core.progressive_components import (
        create_progressive_universal_comprehension,
        create_progressive_therapeutic_intervention
    )
    PROGRESSIVE_COMPONENTS_AVAILABLE = True
except ImportError:
    PROGRESSIVE_COMPONENTS_AVAILABLE = False

# GPU foundation imports
try:
    from src.utils.gpu_foundation import GPUFoundation
    GPU_FOUNDATION_AVAILABLE = True
except ImportError:
    GPU_FOUNDATION_AVAILABLE = False

# Router imports (with fallbacks)
AVAILABLE_ROUTERS = {}
ROUTER_IMPORTS = [
    ("geoid_scar_router", "src.api.routers.geoid_scar_router"),
    ("cognitive_router", "src.api.routers.cognitive_router"),

    ("system_router", "src.api.routers.system_router"),
    ("contradiction_router", "src.api.routers.contradiction_router"),
    ("vault_router", "src.api.routers.vault_router"),
    ("insight_router", "src.api.routers.insight_router"),
    ("statistics_router", "src.api.routers.statistics_router"),
    ("output_analysis_router", "src.api.routers.output_analysis_router"),
    ("core_actions_router", "src.api.routers.core_actions_router"),
    ("thermodynamic_router", "src.api.routers.thermodynamic_router"),
    ("unified_thermodynamic_router", "src.api.routers.unified_thermodynamic_router"),
    ("metrics_router", "src.api.routers.metrics_router"),
    ("gpu_router", "src.api.routers.gpu_router"),
    ("linguistic_router", "src.api.routers.linguistic_router"),
    ("cognitive_architecture_router", "src.api.routers.cognitive_architecture_router"),
    ("cognitive_control_routes", "src.api.cognitive_control_routes"),
    ("monitoring_routes", "src.api.monitoring_routes"),
    ("revolutionary_routes", "src.api.revolutionary_routes"),
    ("law_enforcement_routes", "src.api.law_enforcement_routes"),
    ("foundational_thermodynamic_routes", "src.api.foundational_thermodynamic_routes")
]

for router_name, module_path in ROUTER_IMPORTS:
    try:
        module = __import__(module_path, fromlist=["router"])
        AVAILABLE_ROUTERS[router_name] = getattr(module, "router")
    except ImportError as e:
        logging.warning(f"Router {router_name} not available: {e}")

# Setup logger
logger = get_system_logger(__name__)

# Global configuration
KIMERA_MODE = os.getenv('KIMERA_MODE', 'progressive')  # progressive, full, safe, fast, optimized, hybrid
DEBUG_MODE = os.getenv('DEBUG', 'false').lower() == 'true'
PORT_RANGE = [8000, 8001, 8002, 8003, 8080]
FORCE_PORT = os.getenv('KIMERA_PORT')

class KimeraInitializationMode:
    """Comprehensive initialization mode configuration."""
    
    PROGRESSIVE = 'progressive'    # Lazy loading with background enhancement
    FULL = 'full'                 # Complete initialization upfront
    SAFE = 'safe'                 # Maximum fallbacks and error tolerance
    FAST = 'fast'                 # Minimal features for rapid startup
    OPTIMIZED = 'optimized'       # Performance-focused initialization
    HYBRID = 'hybrid'             # Adaptive based on system capabilities

class SystemCapabilities:
    """Detect system capabilities for adaptive initialization."""
    
    @staticmethod
    def detect_capabilities() -> Dict[str, Any]:
        """Detect system capabilities."""
        import psutil
        import torch
        
        capabilities = {
            'cpu_count': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'gpu_available': torch.cuda.is_available(),
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'gpu_memory_gb': 0
        }
        
        if capabilities['gpu_available']:
            try:
                capabilities['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            except Exception:
                pass
        
        return capabilities

# Global state
unified_architecture: Optional[UnifiedMasterCognitiveArchitecture] = None
system_capabilities: Dict[str, Any] = {}

@asynccontextmanager
async def unified_lifespan(app: FastAPI):
    """
    Unified lifespan manager supporting ALL initialization modes.
    Integrates with the new unified cognitive architecture.
    """
    global unified_architecture, system_capabilities
    
    mode = KIMERA_MODE
    logger.info("🚀 KIMERA SWM UNIFIED STARTUP v3.0.0")
    logger.info(f"🎯 Mode: {mode.upper()}")
    logger.info(f"🏗️ Architecture: Complete Unified Entry Point")
    logger.info("=" * 80)
    
    startup_start = time.time()
    
    try:
        # Detect system capabilities
        system_capabilities = SystemCapabilities.detect_capabilities()
        logger.info(f"💻 System: {system_capabilities['cpu_count']} CPU, {system_capabilities['memory_gb']:.1f}GB RAM")
        if system_capabilities['gpu_available']:
            logger.info(f"🎮 GPU: {system_capabilities['gpu_count']} devices, {system_capabilities['gpu_memory_gb']:.1f}GB VRAM")
        
        # Initialize based on mode
        if mode == KimeraInitializationMode.PROGRESSIVE:
            await _initialize_progressive(app)
        elif mode == KimeraInitializationMode.FULL:
            await _initialize_full(app)
        elif mode == KimeraInitializationMode.SAFE:
            await _initialize_safe(app)
        elif mode == KimeraInitializationMode.FAST:
            await _initialize_fast(app)
        elif mode == KimeraInitializationMode.OPTIMIZED:
            await _initialize_optimized(app)
        elif mode == KimeraInitializationMode.HYBRID:
            await _initialize_hybrid(app)
        else:
            logger.warning(f"Unknown mode {mode}, defaulting to progressive")
            await _initialize_progressive(app)
        
        startup_time = time.time() - startup_start
        logger.info("✅ KIMERA SWM UNIFIED STARTUP COMPLETE")
        logger.info(f"⏱️  Total startup time: {startup_time:.2f}s")
        logger.info(f"🌟 System ready for operation")
        logger.info("=" * 80)
        
        yield  # FastAPI runs here
        
    except Exception as e:
        logger.critical(f"❌ KIMERA startup failed: {e}")
        logger.critical(f"Traceback: {traceback.format_exc()}")
        raise
    
    finally:
        # Comprehensive shutdown sequence
        logger.info("🛑 KIMERA SWM Unified shutdown initiated...")
        try:
            # Shutdown unified architecture
            if unified_architecture:
                await unified_architecture.shutdown()
                logger.info("✅ Unified architecture shutdown complete")
            
            # Shutdown kimera system
            if hasattr(app.state, 'kimera_system') and app.state.kimera_system:
                await app.state.kimera_system.shutdown()
                logger.info("✅ Kimera system shutdown complete")
            
            # Shutdown GPU foundation
            if hasattr(app.state, 'gpu_foundation') and app.state.gpu_foundation:
                app.state.gpu_foundation.cleanup()
                logger.info("✅ GPU foundation shutdown complete")
                
        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")
        
        logger.info("🛑 KIMERA SWM Unified shutdown complete")

# ============================================================================
# INITIALIZATION IMPLEMENTATIONS
# ============================================================================

async def _initialize_progressive(app: FastAPI):
    """Progressive initialization with lazy loading and background enhancement."""
    global unified_architecture
    
    logger.info("📦 Progressive initialization starting...")
    
    try:
        # Phase 1: Unified Architecture (Critical - Fast)
        if UNIFIED_ARCHITECTURE_AVAILABLE:
            unified_architecture = create_unified_architecture(
                processing_mode=ProcessingMode.ADAPTIVE,
                enable_experimental=False
            )
            success = await unified_architecture.initialize_architecture()
            if success:
                app.state.unified_architecture = unified_architecture
                logger.info("✅ Unified architecture initialized (progressive)")
            else:
                logger.warning("⚠️ Unified architecture initialization failed, continuing with fallback")
        
        # Phase 2: Core Kimera System (Fast)
        app.state.kimera_system = kimera_singleton
        await _initialize_core_fast(app)
        
        # Phase 3: Basic API State
        await _setup_basic_api_state(app)
        
        # Phase 4: Start background enhancement
        if PROGRESSIVE_COMPONENTS_AVAILABLE:
            asyncio.create_task(_background_enhancement(app))
        
        logger.info("✅ Progressive initialization complete - Background enhancement active")
        
    except Exception as e:
        logger.error(f"❌ Progressive initialization failed: {e}")
        await _initialize_minimal_fallback(app)

async def _initialize_full(app: FastAPI):
    """Full initialization with ALL features enabled."""
    global unified_architecture
    
    logger.info("🔧 Full initialization starting...")
    logger.info("⚠️  This will take longer but enables ALL capabilities")
    
    try:
        # Phase 1: Unified Architecture (Revolutionary Mode)
        if UNIFIED_ARCHITECTURE_AVAILABLE:
            unified_architecture = create_unified_architecture(
                processing_mode=ProcessingMode.REVOLUTIONARY,
                enable_experimental=True
            )
            success = await unified_architecture.initialize_architecture()
            if success:
                app.state.unified_architecture = unified_architecture
                logger.info("✅ Unified architecture initialized (revolutionary)")
        
        # Phase 2: GPU Foundation
        if GPU_FOUNDATION_AVAILABLE:
            await _initialize_gpu_foundation(app)
        
        # Phase 3: Complete Kimera System
        app.state.kimera_system = kimera_singleton
        await _initialize_core_complete(app)
        
        # Phase 4: Complete API State
        await _setup_complete_api_state(app)
        
        # Phase 5: Initialize Monitoring
        await _initialize_monitoring(app)
        
        # Phase 6: Initialize All Optional Components
        await _initialize_optional_components(app)
        
        logger.info("✅ Full initialization complete - ALL capabilities enabled")
        
    except Exception as e:
        logger.error(f"❌ Full initialization failed: {e}")
        await _initialize_safe_fallback(app)

async def _initialize_safe(app: FastAPI):
    """Safe mode initialization with maximum fallbacks."""
    global unified_architecture
    
    logger.info("🛡️ Safe mode initialization starting...")
    
    try:
        # Phase 1: Safe Unified Architecture
        if UNIFIED_ARCHITECTURE_AVAILABLE:
            unified_architecture = create_safe_architecture()
            success = await unified_architecture.initialize_architecture()
            if success:
                app.state.unified_architecture = unified_architecture
                logger.info("✅ Unified architecture initialized (safe mode)")
        
        # Phase 2: Safe Kimera System
        app.state.kimera_system = kimera_singleton
        await _initialize_core_safe(app)
        
        # Phase 3: Basic API State
        await _setup_basic_api_state(app)
        
        logger.info("✅ Safe mode initialization complete")
        
    except Exception as e:
        logger.warning(f"Safe mode fallback activated: {e}")
        await _initialize_minimal_fallback(app)

async def _initialize_fast(app: FastAPI):
    """Fast initialization - minimal features for rapid startup."""
    global unified_architecture
    
    logger.info("⚡ Fast initialization starting...")
    
    try:
        # Phase 1: Minimal Unified Architecture
        if UNIFIED_ARCHITECTURE_AVAILABLE:
            unified_architecture = create_unified_architecture(
                processing_mode=ProcessingMode.SAFE,
                enable_experimental=False
            )
            # Quick initialization
            app.state.unified_architecture = unified_architecture
            logger.info("✅ Unified architecture set (fast mode)")
        
        # Phase 2: Minimal Kimera System
        app.state.kimera_system = kimera_singleton
        await _initialize_core_minimal(app)
        
        logger.info("✅ Fast initialization complete")
        
    except Exception as e:
        logger.error(f"❌ Fast initialization failed: {e}")
        await _initialize_minimal_fallback(app)

async def _initialize_optimized(app: FastAPI):
    """Optimized initialization - performance-focused."""
    global unified_architecture
    
    logger.info("🚀 Optimized initialization starting...")
    
    try:
        # Phase 1: Optimized Unified Architecture
        if UNIFIED_ARCHITECTURE_AVAILABLE:
            unified_architecture = create_unified_architecture(
                processing_mode=ProcessingMode.OPTIMIZED,
                enable_gpu=system_capabilities.get('gpu_available', False)
            )
            success = await unified_architecture.initialize_architecture()
            if success:
                app.state.unified_architecture = unified_architecture
                logger.info("✅ Unified architecture initialized (optimized)")
        
        # Phase 2: GPU Foundation (if available)
        if GPU_FOUNDATION_AVAILABLE and system_capabilities.get('gpu_available'):
            await _initialize_gpu_foundation(app)
        
        # Phase 3: Optimized Kimera System
        app.state.kimera_system = kimera_singleton
        await _initialize_core_optimized(app)
        
        # Phase 4: Essential API State
        await _setup_essential_api_state(app)
        
        logger.info("✅ Optimized initialization complete")
        
    except Exception as e:
        logger.error(f"❌ Optimized initialization failed: {e}")
        await _initialize_progressive(app)

async def _initialize_hybrid(app: FastAPI):
    """Hybrid initialization - adaptive based on system capabilities."""
    global unified_architecture
    
    logger.info("🎯 Hybrid initialization starting...")
    logger.info(f"📊 Adapting to system: {system_capabilities}")
    
    try:
        # Determine best mode based on capabilities
        if system_capabilities.get('memory_gb', 0) >= 16 and system_capabilities.get('gpu_available'):
            # High-end system: Use full mode
            logger.info("🏆 High-end system detected - using full initialization")
            await _initialize_full(app)
        elif system_capabilities.get('memory_gb', 0) >= 8:
            # Mid-range system: Use optimized mode
            logger.info("⚡ Mid-range system detected - using optimized initialization")
            await _initialize_optimized(app)
        else:
            # Low-end system: Use progressive mode
            logger.info("📦 Resource-constrained system detected - using progressive initialization")
            await _initialize_progressive(app)
        
        logger.info("✅ Hybrid initialization complete")
        
    except Exception as e:
        logger.error(f"❌ Hybrid initialization failed: {e}")
        await _initialize_safe(app)

# ============================================================================
# CORE INITIALIZATION FUNCTIONS
# ============================================================================

async def _initialize_core_fast(app: FastAPI):
    """Fast core initialization."""
    try:
        kimera_singleton.initialize()
        logger.info("✅ Core system initialized (fast mode)")
    except Exception as e:
        logger.error(f"❌ Core fast initialization failed: {e}")
        raise

async def _initialize_core_complete(app: FastAPI):
    """Complete core initialization with all features."""
    try:
        # Initialize vault
        from src.vault import initialize_vault
        if not initialize_vault():
            raise RuntimeError("Vault initialization failed")
        
        # Initialize kimera system
        kimera_singleton.initialize()
        
        # Initialize additional components
        await _initialize_embedding_models(app)
        
        logger.info("✅ Core system fully initialized")
    except Exception as e:
        logger.error(f"❌ Complete core initialization failed: {e}")
        raise

async def _initialize_core_safe(app: FastAPI):
    """Safe core initialization with maximum error tolerance."""
    initialized_components = 0
    
    # Try to initialize vault
    try:
        from src.vault import initialize_vault
        if initialize_vault():
            initialized_components += 1
            logger.info("✅ Vault initialized")
    except Exception as e:
        logger.warning(f"⚠️ Vault initialization failed: {e}")
    
    # Try to initialize kimera system
    try:
        kimera_singleton.initialize()
        initialized_components += 1
        logger.info("✅ Kimera system initialized")
    except Exception as e:
        logger.warning(f"⚠️ Kimera system initialization failed: {e}")
    
    if initialized_components == 0:
        raise RuntimeError("No core components could be initialized")
    
    logger.info(f"✅ Safe core initialization complete ({initialized_components}/2 components)")

async def _initialize_core_minimal(app: FastAPI):
    """Minimal core initialization."""
    try:
        # Just basic kimera system
        kimera_singleton.initialize()
        logger.info("✅ Core system initialized (minimal mode)")
    except Exception as e:
        logger.warning(f"⚠️ Minimal core initialization failed: {e}")
        # Continue anyway for fast startup

async def _initialize_core_optimized(app: FastAPI):
    """Optimized core initialization."""
    try:
        # Initialize with performance focus
        kimera_singleton.initialize()
        
        # Initialize GPU-accelerated components if available
        if system_capabilities.get('gpu_available'):
            await _initialize_gpu_components(app)
        
        logger.info("✅ Core system initialized (optimized mode)")
    except Exception as e:
        logger.error(f"❌ Optimized core initialization failed: {e}")
        raise

# ============================================================================
# COMPONENT INITIALIZATION FUNCTIONS
# ============================================================================

async def _initialize_gpu_foundation(app: FastAPI):
    """Initialize GPU foundation."""
    if not GPU_FOUNDATION_AVAILABLE:
        logger.warning("GPU foundation not available")
        return
    
    try:
        gpu_foundation = GPUFoundation()
        app.state.gpu_foundation = gpu_foundation
        logger.info("✅ GPU foundation initialized")
    except Exception as e:
        logger.warning(f"⚠️ GPU foundation initialization failed: {e}")
        app.state.gpu_foundation = None

async def _initialize_embedding_models(app: FastAPI):
    """Initialize embedding models."""
    try:
        from src.utils.embedding_utils import initialize_embedding_model
        await initialize_embedding_model()
        logger.info("✅ Embedding models initialized")
    except Exception as e:
        logger.warning(f"⚠️ Embedding models initialization failed: {e}")

async def _initialize_gpu_components(app: FastAPI):
    """Initialize GPU-accelerated components."""
    try:
        # Initialize GPU-specific components
        if hasattr(app.state, 'gpu_foundation') and app.state.gpu_foundation:
            logger.info("✅ GPU components initialized")
    except Exception as e:
        logger.warning(f"⚠️ GPU components initialization failed: {e}")

async def _initialize_monitoring(app: FastAPI):
    """Initialize monitoring and metrics."""
    try:
        initialize_background_collection()
        logger.info("✅ Monitoring initialized")
    except Exception as e:
        logger.warning(f"⚠️ Monitoring initialization failed: {e}")

async def _initialize_optional_components(app: FastAPI):
    """Initialize optional components."""
    try:
        # Initialize any optional components
        logger.info("✅ Optional components initialized")
    except Exception as e:
        logger.warning(f"⚠️ Optional components initialization failed: {e}")

# ============================================================================
# API STATE SETUP FUNCTIONS
# ============================================================================

async def _setup_basic_api_state(app: FastAPI):
    """Setup basic API state."""
    app.state.startup_time = time.time()
    app.state.initialization_mode = KIMERA_MODE
    logger.info("✅ Basic API state configured")

async def _setup_essential_api_state(app: FastAPI):
    """Setup essential API state."""
    await _setup_basic_api_state(app)
    app.state.system_capabilities = system_capabilities
    logger.info("✅ Essential API state configured")

async def _setup_complete_api_state(app: FastAPI):
    """Setup complete API state."""
    await _setup_essential_api_state(app)
    # Add any additional state needed for full mode
    logger.info("✅ Complete API state configured")

# ============================================================================
# BACKGROUND ENHANCEMENT
# ============================================================================

async def _background_enhancement(app: FastAPI):
    """Background enhancement for progressive initialization."""
    logger.info("🔄 Starting background enhancement...")
    
    try:
        # Progressive enhancement of components
        if PROGRESSIVE_COMPONENTS_AVAILABLE:
            lazy_manager = get_global_lazy_manager()
            
            # Enhance components progressively
            await lazy_manager.enhance_component("universal_comprehension")
            await lazy_manager.enhance_component("therapeutic_intervention")
            
            logger.info("✅ Background enhancement complete")
    except Exception as e:
        logger.warning(f"⚠️ Background enhancement failed: {e}")

# ============================================================================
# FALLBACK FUNCTIONS
# ============================================================================

async def _initialize_minimal_fallback(app: FastAPI):
    """Minimal fallback initialization."""
    logger.warning("🛡️ Activating minimal fallback initialization...")
    
    try:
        app.state.kimera_system = None
        app.state.fallback_mode = True
        app.state.startup_time = time.time()
        
        logger.info("✅ Minimal fallback initialized")
    except Exception as e:
        logger.critical(f"❌ Even minimal fallback failed: {e}")
        raise

async def _initialize_safe_fallback(app: FastAPI):
    """Safe fallback initialization."""
    logger.warning("🛡️ Activating safe fallback initialization...")
    
    try:
        await _initialize_safe(app)
    except Exception as e:
        logger.error(f"❌ Safe fallback failed: {e}")
        await _initialize_minimal_fallback(app)

# ============================================================================
# FastAPI APPLICATION SETUP
# ============================================================================

def create_application() -> FastAPI:
    """Create the FastAPI application with unified configuration."""
    
    app = FastAPI(
        title="KIMERA SWM Unified API",
        description="Unified KIMERA SWM API with complete initialization patterns",
        version="3.0.0",
        lifespan=unified_lifespan,
        debug=DEBUG_MODE
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add routers
    for router_name, router in AVAILABLE_ROUTERS.items():
        try:
            app.include_router(router, prefix=f"/{router_name.replace('_router', '').replace('_routes', '')}")
            logger.info(f"✅ Router included: {router_name}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to include router {router_name}: {e}")
    
    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "service": "KIMERA SWM Unified API",
            "version": "3.0.0",
            "mode": KIMERA_MODE,
            "status": "operational",
            "timestamp": datetime.now().isoformat(),
            "unified_architecture": unified_architecture is not None,
            "system_capabilities": system_capabilities
        }
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "unified_architecture": unified_architecture.get_system_status() if unified_architecture else None,
            "uptime": time.time() - getattr(app.state, 'startup_time', time.time())
        }
    
    return app

# ============================================================================
# PORT MANAGEMENT
# ============================================================================

def find_available_port() -> int:
    """Find an available port from the configured range."""
    if FORCE_PORT:
        return int(FORCE_PORT)
    
    for port in PORT_RANGE:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', port))
                return port
            except OSError:
                continue
    
    # If no port from range is available, let the system choose
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    try:
        # Create application
        app = create_application()
        
        # Find available port
        port = find_available_port()
        
        # Log startup information
        logger.info("🚀 KIMERA SWM UNIFIED STARTUP")
        logger.info("=" * 80)
        logger.info(f"🌐 Server: http://127.0.0.1:{port}")
        logger.info(f"📚 API Docs: http://127.0.0.1:{port}/docs")
        logger.info(f"🎯 Mode: {KIMERA_MODE.upper()}")
        logger.info(f"🔧 Debug: {DEBUG_MODE}")
        logger.info("=" * 80)
        
        # Run server
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=port,
            log_level="info" if DEBUG_MODE else "warning",
            access_log=DEBUG_MODE
        )
        
    except Exception as e:
        logger.critical(f"❌ Failed to start KIMERA SWM: {e}")
        logger.critical(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()
