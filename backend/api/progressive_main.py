#!/usr/bin/env python3
"""
KIMERA Progressive Main Application
==================================

Implements the progressive enhancement architecture to solve startup bottlenecks
while preserving KIMERA's complete functionality and uniqueness.

Key Features:
- Lazy initialization with progressive enhancement
- Critical components available in ~30 seconds
- Full AI capabilities achieved progressively
- Zero-debugging constraint maintained
- Complete cognitive fidelity preserved

Architecture:
1. Basic Level: Fast startup with mock implementations
2. Enhanced Level: Real AI with optimized components
3. Full Level: Complete KIMERA with all validations
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import asyncio
import logging
import time
import os

# KIMERA imports
from backend.utils.kimera_logger import get_system_logger
from backend.core.lazy_initialization_manager import (
    get_global_lazy_manager, ComponentConfig, Priority
)
from backend.core.progressive_components import (
    create_progressive_universal_comprehension,
    create_progressive_therapeutic_intervention
)

# Setup logger
logger = get_system_logger(__name__)

# Global system state
kimera_system = {
    'status': 'initializing',
    'components': {},
    'errors': [],
    'initialization_level': 'basic'
}

@asynccontextmanager
async def progressive_lifespan(app: FastAPI):
    """
    Progressive lifespan manager with lazy initialization
    """
    logger.info("🚀 KIMERA Progressive Startup initiated...")
    logger.info("🎯 Architecture: Lazy Initialization + Progressive Enhancement")
    
    # Initialize app state
    app.state.kimera_system = kimera_system
    app.state.loading_progress = 0
    app.state.lazy_manager = get_global_lazy_manager()
    
    try:
        # Phase 1: Register all components for lazy loading
        logger.info("📦 Phase 1: Registering components for lazy initialization...")
        await register_all_components(app.state.lazy_manager)
        app.state.loading_progress = 20
        
        # Phase 2: Initialize critical components immediately
        logger.info("⚡ Phase 2: Initializing critical components...")
        app.state.lazy_manager.initialize_critical_components()
        kimera_system['status'] = 'basic_ready'
        kimera_system['initialization_level'] = 'basic'
        app.state.loading_progress = 60
        
        # Phase 3: Start background enhancement
        logger.info("🔄 Phase 3: Starting background enhancement...")
        app.state.lazy_manager.start_background_enhancement()
        app.state.loading_progress = 80
        
        # Phase 4: Setup API state
        logger.info("🌐 Phase 4: Setting up API state...")
        await setup_api_state(app)
        app.state.loading_progress = 100
        
        logger.info("✅ KIMERA Progressive Startup Complete!")
        logger.info("🎉 Basic functionality available immediately")
        logger.info("🔄 Enhanced features loading in background")
        logger.info("🌟 Full AI capabilities will be available progressively")
        
        # Schedule progressive enhancement
        asyncio.create_task(progressive_enhancement_scheduler(app))
        
    except Exception as e:
        logger.critical(f"💥 Critical error during progressive startup: {e}", exc_info=True)
        kimera_system['status'] = 'error'
        kimera_system['errors'].append(f"Critical: {e}")

    yield
    
    # Cleanup
    logger.info("🛑 KIMERA shutting down...")
    if hasattr(app.state, 'lazy_manager'):
        app.state.lazy_manager.shutdown()
    kimera_system['status'] = 'shutdown'

async def register_all_components(lazy_manager) -> None:
    """Register all KIMERA components for lazy initialization"""
    
    # GPU Foundation - Critical priority
    def create_gpu_foundation():
        from backend.utils.gpu_foundation import GPUFoundation
        return GPUFoundation()
    
    def create_mock_gpu():
        class MockGPU:
            def __init__(self):
                self.device = "cpu"
                self.available = False
            def get_device(self):
                return self.device
        return MockGPU()
    
    lazy_manager.register_component(ComponentConfig(
        name="gpu_foundation",
        priority=Priority.CRITICAL,
        basic_initializer=create_mock_gpu,
        enhanced_initializer=lambda x: create_gpu_foundation(),
        cache_key="gpu_foundation",
        fallback_factory=create_mock_gpu
    ))
    
    # Embedding Model - High priority
    def create_embedding_model():
        from backend.core.embedding_utils import initialize_embedding_model
        return initialize_embedding_model()
    
    def create_mock_embedding():
        class MockEmbedding:
            def encode(self, text):
                return [0.0] * 768
        return MockEmbedding()
    
    lazy_manager.register_component(ComponentConfig(
        name="embedding_model",
        priority=Priority.HIGH,
        basic_initializer=create_mock_embedding,
        enhanced_initializer=lambda x: create_embedding_model(),
        dependencies=["gpu_foundation"],
        cache_key="embedding_model",
        fallback_factory=create_mock_embedding
    ))
    
    # Universal Output Comprehension - Medium priority (progressive)
    lazy_manager.register_component(ComponentConfig(
        name="universal_comprehension",
        priority=Priority.MEDIUM,
        basic_initializer=lambda: create_progressive_universal_comprehension("basic"),
        enhanced_initializer=lambda x: create_progressive_universal_comprehension("enhanced"),
        full_initializer=lambda x: create_progressive_universal_comprehension("full"),
        cache_key="universal_comprehension_basic"
    ))
    
    # Therapeutic Intervention System - Medium priority (progressive)
    lazy_manager.register_component(ComponentConfig(
        name="therapeutic_intervention",
        priority=Priority.MEDIUM,
        basic_initializer=lambda: create_progressive_therapeutic_intervention("basic"),
        enhanced_initializer=lambda x: create_progressive_therapeutic_intervention("enhanced"),
        full_initializer=lambda x: create_progressive_therapeutic_intervention("full"),
        cache_key="therapeutic_intervention_basic"
    ))
    
    # Vault Manager - High priority
    def create_vault_manager():
        from backend.vault import get_vault_manager
        return get_vault_manager()
    
    def create_mock_vault():
        class MockVault:
            def store(self, key, value):
                return True
            def retrieve(self, key):
                return None
        return MockVault()
    
    lazy_manager.register_component(ComponentConfig(
        name="vault_manager",
        priority=Priority.HIGH,
        basic_initializer=create_mock_vault,
        enhanced_initializer=lambda x: create_vault_manager(),
        cache_key="vault_manager",
        fallback_factory=create_mock_vault
    ))
    
    # Contradiction Engine - Medium priority
    def create_contradiction_engine():
        from backend.engines.contradiction_engine import ContradictionEngine
        return ContradictionEngine(tension_threshold=0.3)
    
    def create_mock_contradiction():
        class MockContradiction:
            def detect_contradiction(self, text):
                return {"contradiction_detected": False, "confidence": 0.5}
        return MockContradiction()
    
    lazy_manager.register_component(ComponentConfig(
        name="contradiction_engine",
        priority=Priority.MEDIUM,
        basic_initializer=create_mock_contradiction,
        enhanced_initializer=lambda x: create_contradiction_engine(),
        cache_key="contradiction_engine",
        fallback_factory=create_mock_contradiction
    ))
    
    # Thermodynamics Engine - Low priority
    def create_thermodynamics_engine():
        from backend.engines.thermodynamics import SemanticThermodynamicsEngine
        return SemanticThermodynamicsEngine()
    
    def create_mock_thermodynamics():
        class MockThermodynamics:
            def calculate_entropy(self, text):
                return 1.5
        return MockThermodynamics()
    
    lazy_manager.register_component(ComponentConfig(
        name="thermodynamics_engine",
        priority=Priority.LOW,
        basic_initializer=create_mock_thermodynamics,
        enhanced_initializer=lambda x: create_thermodynamics_engine(),
        cache_key="thermodynamics_engine",
        fallback_factory=create_mock_thermodynamics
    ))
    
    logger.info(f"📦 Registered {len(lazy_manager.components)} components for lazy initialization")

async def setup_api_state(app: FastAPI) -> None:
    """Setup API state with lazy-loaded components"""
    
    # Get basic components immediately
    app.state.gpu_foundation = app.state.lazy_manager.get_component("gpu_foundation", "basic")
    app.state.embedding_model = app.state.lazy_manager.get_component("embedding_model", "basic")
    app.state.vault_manager = app.state.lazy_manager.get_component("vault_manager", "basic")
    
    # Progressive components will be loaded on-demand
    app.state.universal_comprehension = None  # Loaded on first use
    app.state.therapeutic_intervention = None  # Loaded on first use
    
    # Store component references in kimera_system
    kimera_system['components'] = {
        'gpu_foundation': app.state.gpu_foundation,
        'embedding_model': app.state.embedding_model,
        'vault_manager': app.state.vault_manager,
        'lazy_manager': app.state.lazy_manager
    }
    
    # Initialize metrics
    try:
        from backend.monitoring.kimera_prometheus_metrics import get_kimera_metrics
        metrics = get_kimera_metrics()
        app.state.metrics = metrics
        kimera_system['metrics'] = metrics
        logger.info("✅ Metrics initialized")
    except Exception as e:
        logger.warning(f"Metrics initialization failed: {e}")
    
    # Start background jobs with fallback
    try:
        from backend.engines.background_jobs import start_background_jobs
        
        embedding_model = app.state.embedding_model
        if hasattr(embedding_model, 'encode'):
            start_background_jobs(embedding_model.encode)
        else:
            # Fallback encoder
            def dummy_encode(text: str) -> list:
                return [0.0] * 768
            start_background_jobs(dummy_encode)
        
        logger.info("✅ Background jobs started")
    except Exception as e:
        logger.warning(f"Background jobs failed: {e}")

async def progressive_enhancement_scheduler(app: FastAPI) -> None:
    """Schedule progressive enhancement of components"""
    
    # Wait a bit for basic system to stabilize
    await asyncio.sleep(10)
    
    logger.info("🔄 Starting progressive enhancement scheduler...")
    
    # Enhancement schedule
    enhancement_schedule = [
        (30, "embedding_model", "enhanced"),     # 30 seconds: Real embedding model
        (60, "vault_manager", "enhanced"),       # 1 minute: Real vault manager
        (120, "contradiction_engine", "enhanced"), # 2 minutes: Real contradiction engine
        (180, "universal_comprehension", "enhanced"), # 3 minutes: Enhanced comprehension
        (300, "therapeutic_intervention", "enhanced"), # 5 minutes: Enhanced TIS
        (600, "universal_comprehension", "full"),     # 10 minutes: Full comprehension
        (900, "therapeutic_intervention", "full"),    # 15 minutes: Full TIS
    ]
    
    for delay, component_name, level in enhancement_schedule:
        await asyncio.sleep(delay - time.time() + app.state.lazy_manager.stats.get('start_time', time.time()))
        
        try:
            logger.info(f"🔄 Enhancing {component_name} to {level} level...")
            enhanced_component = app.state.lazy_manager.get_component(component_name, level)
            
            if enhanced_component:
                # Update app state
                setattr(app.state, component_name, enhanced_component)
                kimera_system['components'][component_name] = enhanced_component
                
                # Update initialization level
                if level == "enhanced" and kimera_system['initialization_level'] == "basic":
                    kimera_system['initialization_level'] = "enhanced"
                    kimera_system['status'] = 'enhanced_ready'
                elif level == "full":
                    kimera_system['initialization_level'] = "full"
                    kimera_system['status'] = 'fully_ready'
                
                logger.info(f"✅ {component_name} enhanced to {level} level")
            
        except Exception as e:
            logger.error(f"❌ Failed to enhance {component_name}: {e}")

# Create FastAPI app
app = FastAPI(
    title="KIMERA Spherical Word Methodology AI (Progressive)",
    description="Advanced AI system with progressive enhancement architecture",
    version="0.1.0-alpha-progressive",
    lifespan=progressive_lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
try:
    app.mount("/images", StaticFiles(directory="static/images"), name="images")
except Exception as e:
    logger.warning(f"Failed to mount static files: {e}")

# Health check endpoint
@app.get("/system/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "initialization_level": kimera_system.get('initialization_level', 'basic'),
        "system_status": kimera_system.get('status', 'unknown'),
        "timestamp": time.time()
    }

# Status endpoint
@app.get("/system/status")
async def system_status():
    """Comprehensive system status"""
    if hasattr(app.state, 'lazy_manager'):
        lazy_status = app.state.lazy_manager.get_status_report()
    else:
        lazy_status = {"error": "Lazy manager not available"}
    
    return {
        "kimera_system": kimera_system,
        "lazy_initialization": lazy_status,
        "api_state": {
            "has_gpu_foundation": hasattr(app.state, 'gpu_foundation'),
            "has_embedding_model": hasattr(app.state, 'embedding_model'),
            "has_vault_manager": hasattr(app.state, 'vault_manager'),
        }
    }

# Progressive component access endpoints
@app.get("/system/components/{component_name}")
async def get_component_status(component_name: str, level: str = "basic"):
    """Get status of a specific component"""
    if not hasattr(app.state, 'lazy_manager'):
        raise HTTPException(status_code=503, detail="Lazy manager not available")
    
    component = app.state.lazy_manager.get_component(component_name, level)
    
    if component is None:
        raise HTTPException(status_code=404, detail=f"Component {component_name} not found")
    
    return {
        "component_name": component_name,
        "level": level,
        "available": component is not None,
        "type": type(component).__name__,
        "mock": getattr(component, 'mock', False) if hasattr(component, 'mock') else False
    }

# Enhanced comprehension endpoint
@app.post("/cognitive/comprehend")
async def comprehend_output(request: dict):
    """Comprehend output using progressive universal comprehension"""
    if not hasattr(app.state, 'lazy_manager'):
        raise HTTPException(status_code=503, detail="System not ready")
    
    # Get comprehension engine (will load if not already loaded)
    comprehension_engine = app.state.lazy_manager.get_component("universal_comprehension", "basic")
    
    if not comprehension_engine:
        raise HTTPException(status_code=503, detail="Comprehension engine not available")
    
    output_content = request.get('content', '')
    context = request.get('context')
    
    try:
        result = await comprehension_engine.comprehend_output(output_content, context)
        return result
    except Exception as e:
        logger.error(f"Comprehension failed: {e}")
        raise HTTPException(status_code=500, detail=f"Comprehension failed: {str(e)}")

# Include other routers
try:
    from backend.api.monitoring_routes import router as monitoring_router
    from backend.api.cognitive_field_routes import router as cognitive_field_router
    
    app.include_router(monitoring_router, prefix="/monitoring", tags=["monitoring"])
    app.include_router(cognitive_field_router, prefix="/cognitive", tags=["cognitive"])
    
    logger.info("✅ API routers included")
except Exception as e:
    logger.warning(f"Failed to include some routers: {e}")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "message": "KIMERA Progressive AI System",
        "status": kimera_system.get('status', 'unknown'),
        "initialization_level": kimera_system.get('initialization_level', 'basic'),
        "version": "0.1.0-alpha-progressive",
        "architecture": "lazy_initialization_progressive_enhancement",
        "endpoints": {
            "health": "/system/health",
            "status": "/system/status",
            "docs": "/docs",
            "comprehension": "/cognitive/comprehend"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003, log_level="info") 