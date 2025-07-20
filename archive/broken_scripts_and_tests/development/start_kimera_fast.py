#!/usr/bin/env python3
"""
🚀 KIMERA FAST STARTUP SCRIPT
=============================

This script starts KIMERA with minimal components for faster initialization.
It bypasses the Universal Output Comprehension System which may be causing
the long initialization time.

Author: KIMERA AI System
Version: 1.0.0 - Fast Startup Solution
"""

import uvicorn
import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_environment():
    """Check if we're in the right environment"""
    try:
        # Check if we can import the main app
        from backend.api.main import app
        return True, app
    except ImportError as e:
        logger.error(f"Cannot import KIMERA app: {e}")
        return False, None

def patch_universal_comprehension():
    """Temporarily disable Universal Output Comprehension System"""
    try:
        # Monkey patch the initialization to skip the problematic component
        import backend.api.main as main_module
        
        # Store original lifespan
        original_lifespan = main_module.lifespan
        
        # Create a patched version that skips Universal Output Comprehension
        from contextlib import asynccontextmanager
        
        @asynccontextmanager
        async def patched_lifespan(app):
            """Patched lifespan that skips Universal Output Comprehension System"""
            logger.info("🚀 KIMERA FAST STARTUP - Skipping Universal Output Comprehension")
            logger.info("🚀 KIMERA starting up... Initializing core systems only.")
            
            app.state.gpu_foundation = None
            app.state.embedding_model = None
            app.state.translator_hub = None
            app.state.anthropomorphic_profiler = None
            app.state.gyroscopic_security = None
            
            try:
                # Import necessary components
                from backend.utils.gpu_foundation import GPUFoundation
                from backend.core.embedding_utils import initialize_embedding_model
                from backend.engines.universal_translator_hub import create_universal_translator_hub
                from backend.api.cognitive_control_routes import initialize_cognitive_control_services
                from backend.vault import get_vault_manager
                from backend.engines.contradiction_engine import ContradictionEngine
                from backend.engines.thermodynamics import SemanticThermodynamicsEngine
                from backend.engines.background_jobs import start_background_jobs
                from backend.monitoring.kimera_prometheus_metrics import get_kimera_metrics
                from backend.core.therapeutic_intervention_system import TherapeuticInterventionSystem
                
                # Step 1: Initialize GPU Foundation
                gpu_foundation = GPUFoundation()
                app.state.gpu_foundation = gpu_foundation
                logger.info("✅ GPU Foundation initialized successfully.")

                # Step 2: Initialize Embedding Model
                embedding_model = initialize_embedding_model()
                app.state.embedding_model = embedding_model
                logger.info("✅ Embedding model initialized successfully.")

                # Step 3: Initialize Universal Translator Hub
                translator_hub_config = {}
                translator_hub = create_universal_translator_hub(
                    config=translator_hub_config, 
                    gpu_foundation=gpu_foundation
                )
                if translator_hub:
                    app.state.translator_hub = translator_hub
                    logger.info("✅ Universal Translator Hub initialized successfully.")

                # Step 4: Initialize Enhanced Services
                logger.info("Initializing Enhanced Services (Profiler, Security)...")
                initialize_cognitive_control_services(app)
                logger.info("✅ Enhanced Services initialized successfully.")
                
                # Step 5: Initialize Core KIMERA Engines (minimal set)
                logger.info("Initializing Core KIMERA Engines...")
                
                # Vault Manager
                vault_manager = get_vault_manager()
                main_module.kimera_system['vault_manager'] = vault_manager
                logger.info("✅ Vault Manager initialized successfully.")
                
                # Contradiction Engine
                main_module.kimera_system['contradiction_engine'] = ContradictionEngine(tension_threshold=0.3)
                logger.info("✅ Contradiction Engine initialized successfully.")
                
                # Legacy Thermodynamics Engine only
                main_module.kimera_system['thermodynamics_engine'] = SemanticThermodynamicsEngine()
                logger.info("✅ Legacy Thermodynamics Engine initialized successfully.")
                
                # SKIP Universal Output Comprehension System
                logger.info("⏭️  Skipping Universal Output Comprehension System for fast startup")
                
                # Initialize metrics
                main_module.kimera_system['metrics'] = get_kimera_metrics()
                
                # Start background jobs - fix for embedding model being a dict
                try:
                    if hasattr(embedding_model, 'encode'):
                        start_background_jobs(embedding_model.encode)
                    else:
                        # If embedding_model is a dict, get the actual model
                        actual_model = embedding_model.get('model') if isinstance(embedding_model, dict) else embedding_model
                        if hasattr(actual_model, 'encode'):
                            start_background_jobs(actual_model.encode)
                        else:
                            logger.warning("⚠️ Embedding model doesn't have encode method, starting background jobs without it")
                            start_background_jobs(lambda x: [])
                except Exception as e:
                    logger.warning(f"⚠️ Failed to start background jobs with embedding model: {e}")
                    start_background_jobs(lambda x: [])
                logger.info("✅ Background jobs started successfully.")
                
                # Step 6: Initialize Therapeutic Intervention System
                logger.info("Initializing Therapeutic Intervention System...")
                tis = TherapeuticInterventionSystem()
                app.state.tis = tis
                logger.info("✅ Therapeutic Intervention System initialized successfully.")

                # Step 7: Initialize Prometheus Metrics
                metrics = get_kimera_metrics()
                app.state.metrics = metrics
                metrics.start_background_collection()
                logger.info("✅ Prometheus metrics initialized and background collection started.")
                
                logger.info("🎉 KIMERA FAST STARTUP COMPLETE!")
                logger.info("⚠️  Note: Universal Output Comprehension System is disabled for faster startup")
                
            except Exception as e:
                logger.error(f"Failed to initialize KIMERA engines: {e}", exc_info=True)

            yield
            
            logger.info("🛑 KIMERA shutting down. Releasing resources.")
            app.state.translator_hub = None
            app.state.embedding_model = None
            app.state.gpu_foundation = None
            app.state.anthropomorphic_profiler = None
            app.state.gyroscopic_security = None
        
        # Replace the lifespan function
        main_module.lifespan = patched_lifespan
        
        # Recreate the FastAPI app with the patched lifespan
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.staticfiles import StaticFiles
        
        # Create new app with patched lifespan
        app = FastAPI(
            title="KIMERA Spherical Word Methodology AI (Fast Mode)",
            description="An advanced AI system based on Spherical Word Methodology - Fast Startup Mode.",
            version="0.1.0-alpha-fast",
            lifespan=patched_lifespan
        )
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Copy middleware and routes from original app
        original_app = main_module.app
        
        # Copy static files mount
        app.mount("/images", StaticFiles(directory="static/images"), name="images")
        
        # Import and include routers
        from backend.api.monitoring_routes import router as monitoring_router
        from backend.api.cognitive_field_routes import router as cognitive_field_router
        from backend.api.cognitive_control_routes import router as cognitive_control_router
        from backend.api import chat_routes
        from backend.engines.universal_translator_hub import router as translator_router
        
        app.include_router(monitoring_router)
        app.include_router(cognitive_field_router)
        app.include_router(cognitive_control_router)
        app.include_router(chat_routes.router)
        app.include_router(translator_router)
        
        # Include conditional routers
        try:
            from backend.api.law_enforcement_routes import router as law_enforcement_router
            app.include_router(law_enforcement_router)
            logger.info("⚖️ Law enforcement routes registered")
        except ImportError:
            pass
            
        try:
            from backend.api.revolutionary_routes import router as revolutionary_router
            app.include_router(revolutionary_router)
            logger.info("🧠 Revolutionary intelligence routes registered")
        except ImportError:
            pass
            
        try:
            from backend.api.cognitive_pharmaceutical_routes import router as cognitive_pharmaceutical_router
            app.include_router(cognitive_pharmaceutical_router)
            logger.info("🧠💊 Cognitive pharmaceutical optimization routes registered")
        except ImportError:
            pass
        
        # Copy essential endpoints from original app
        for route in original_app.routes:
            if hasattr(route, 'path') and route.path in ['/system/health', '/system/status', '/metrics']:
                app.routes.append(route)
        
        return app
        
    except Exception as e:
        logger.error(f"Failed to patch Universal Output Comprehension: {e}")
        return None

if __name__ == "__main__":
    print("🚀 KIMERA FAST STARTUP")
    print("=" * 50)
    print("🧠 Starting KIMERA with minimal components for faster initialization")
    print("⚠️  Universal Output Comprehension System will be disabled")
    print("⏰ Expected startup time: 30-60 seconds")
    print("🎯 Server will start on: http://localhost:8001")
    print("")
    
    # Check environment
    can_import, original_app = check_environment()
    if not can_import:
        print("❌ Cannot import KIMERA. Please check your environment.")
        sys.exit(1)
    
    # Try to patch and use fast startup
    fast_app = patch_universal_comprehension()
    if fast_app:
        print("✅ Using patched fast startup mode")
        app_to_use = fast_app
    else:
        print("⚠️  Using original app (may take longer)")
        app_to_use = original_app
    
    print("Starting server...")
    
    try:
        uvicorn.run(
            app_to_use,
            host="0.0.0.0",
            port=8001,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n\n✋ KIMERA shutdown requested by user")
        print("🛑 Stopping all systems...")
    except Exception as e:
        print(f"\n❌ Error starting KIMERA: {e}")
        print("💡 Try using the patient startup script instead:")
        print("   python start_kimera_patient.py") 