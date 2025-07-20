#!/usr/bin/env python3
"""
🚀 KIMERA MINIMAL STARTUP SCRIPT
================================

Ultra-minimal KIMERA startup that loads only the absolute essentials
for immediate API access. No AI models, no complex systems.

Author: KIMERA AI System
Version: 1.0.0 - Minimal Startup Solution
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

def create_minimal_app():
    """Create a minimal FastAPI app with only core endpoints"""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from datetime import datetime
    import json
    
    app = FastAPI(
        title="KIMERA Minimal Mode",
        description="KIMERA in minimal mode - core API only",
        version="0.1.0-minimal"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/")
    async def root():
        return {"message": "KIMERA Minimal Mode - Ready", "timestamp": datetime.now().isoformat()}
    
    @app.get("/system/health")
    async def health():
        return {"status": "healthy", "mode": "minimal", "timestamp": datetime.now().isoformat()}
    
    @app.get("/system/status")
    async def status():
        return {
            "timestamp": datetime.now().isoformat(),
            "mode": "minimal",
            "system_info": {
                "active_geoids": 0,
                "vault_a_scars": 0,
                "vault_b_scars": 0,
                "total_contradictions": 0,
                "total_insights": 0,
                "system_health": "healthy"
            },
            "services": {
                "vault_manager": "minimal",
                "contradiction_engine": "minimal", 
                "thermodynamics_engine": "minimal",
                "embedding_model": "disabled",
                "gpu_foundation": "disabled",
                "universal_translator": "disabled"
            },
            "metrics": {
                "total_geoids": 0,
                "total_scars": 0,
                "total_contradictions": 0,
                "avg_processing_time": 0.0,
                "system_uptime": "minimal_mode"
            }
        }
    
    @app.get("/docs")
    async def docs_redirect():
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/docs")
    
    @app.get("/geoids")
    async def list_geoids():
        return {
            "geoids": [],
            "total": 0,
            "message": "Minimal mode - no geoids available",
            "upgrade_to": "Use python start_kimera_patient.py for full functionality"
        }
    
    @app.post("/geoids")
    async def create_geoid(request: dict):
        return {
            "error": "Minimal mode - geoid creation disabled",
            "message": "Use python start_kimera_patient.py for full functionality",
            "request_received": True
        }
    
    @app.get("/system/engines")
    async def engines_status():
        return {
            "engines": {
                "vault_manager": {"status": "minimal", "description": "Minimal mode"},
                "contradiction_engine": {"status": "minimal", "description": "Minimal mode"},
                "thermodynamics_engine": {"status": "minimal", "description": "Minimal mode"},
                "embedding_model": {"status": "disabled", "description": "Disabled in minimal mode"},
                "gpu_foundation": {"status": "disabled", "description": "Disabled in minimal mode"}
            },
            "mode": "minimal",
            "upgrade_message": "Use python start_kimera_patient.py for full AI capabilities"
        }
    
    @app.get("/metrics")
    async def metrics():
        """Simple metrics endpoint that doesn't use Prometheus"""
        return {
            "kimera_mode": "minimal",
            "kimera_status": "healthy",
            "kimera_uptime_seconds": 0,
            "kimera_requests_total": 0,
            "kimera_geoids_total": 0,
            "kimera_scars_total": 0,
            "timestamp": datetime.now().isoformat()
        }
    
    return app

if __name__ == "__main__":
    print("🚀 KIMERA MINIMAL STARTUP")
    print("=" * 50)
    print("⚡ Ultra-fast startup with core API only")
    print("🚫 AI models and complex systems disabled")
    print("⏰ Expected startup time: 5-10 seconds")
    print("🎯 Server will start on: http://localhost:8001")
    print("")
    print("💡 For full AI capabilities, use:")
    print("   python start_kimera_patient.py")
    print("")
    
    try:
        # Create minimal app
        app = create_minimal_app()
        
        print("✅ Minimal app created successfully")
        print("🚀 Starting server...")
        
        # Start the server
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8001,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n\n✋ KIMERA shutdown requested by user")
        print("🛑 Stopping minimal server...")
    except Exception as e:
        print(f"\n❌ Error starting KIMERA minimal: {e}")
        print("💡 This shouldn't happen with minimal mode")
        print("   Please check your Python environment") 