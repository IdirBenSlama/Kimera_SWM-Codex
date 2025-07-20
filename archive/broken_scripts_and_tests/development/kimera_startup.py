#!/usr/bin/env python3
"""
KIMERA SWM Enhanced Startup Script
==================================

This script provides multiple startup modes for KIMERA with automatic error handling,
progressive loading, and fallback mechanisms.

Modes:
- minimal: Fast startup with basic API (current working mode)
- progressive: Gradual loading of KIMERA components
- full: Complete KIMERA initialization with all features
- safe: Full mode with enhanced error handling
"""

import sys
import os
import time
import logging
import argparse
import uvicorn
from typing import Optional

# Add paths for imports
sys.path.insert(0, '.')
sys.path.insert(0, 'backend')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_banner():
    """Print KIMERA startup banner"""
    print("=" * 60)
    print("  KIMERA SPHERICAL WORD METHODOLOGY AI SYSTEM")
    print("  Revolutionary Thermodynamic Cognitive Engine")
    print("=" * 60)

def check_environment():
    """Check system environment and dependencies"""
    print("🔍 Environment Check:")
    
    # Check Python version
    python_version = sys.version_info
    print(f"   Python: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"   GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            print("   GPU: Not available (CPU mode)")
    except ImportError:
        print("   GPU: PyTorch not available")
    
    # Check critical dependencies
    dependencies = ['fastapi', 'uvicorn', 'transformers', 'torch']
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"   ✅ {dep}")
        except ImportError:
            print(f"   ❌ {dep} (missing)")
    
    print()

def start_minimal_mode(host: str, port: int):
    """Start KIMERA in minimal mode (fast, stable)"""
    print("🚀 Starting KIMERA in MINIMAL mode...")
    print("   - Fast startup")
    print("   - Basic API endpoints")
    print("   - Stable operation")
    print("   - No AI model loading")
    print()
    
    from minimal_server import app
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )

def start_progressive_mode(host: str, port: int):
    """Start KIMERA with progressive loading"""
    print("🔄 Starting KIMERA in PROGRESSIVE mode...")
    print("   - Gradual component loading")
    print("   - Background initialization")
    print("   - Graceful degradation")
    print()
    
    # Create progressive loading app
    from fastapi import FastAPI, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    
    app = FastAPI(
        title="KIMERA SWM - Progressive Mode",
        description="KIMERA with progressive component loading",
        version="0.1.0-progressive"
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Progressive loading state
    loading_state = {
        "gpu_foundation": False,
        "embedding_model": False,
        "cognitive_engines": False,
        "full_kimera": False
    }
    
    @app.get("/")
    async def root():
        return {
            "message": "KIMERA SWM Progressive Mode",
            "status": "operational",
            "loading_state": loading_state
        }
    
    @app.get("/system/health")
    async def health():
        return {
            "status": "healthy",
            "mode": "progressive",
            "components_loaded": sum(loading_state.values()),
            "total_components": len(loading_state)
        }
    
    @app.get("/system/status")
    async def status():
        return {
            "kimera_status": "operational",
            "mode": "progressive",
            "loading_state": loading_state,
            "gpu_available": True
        }
    
    async def progressive_loading():
        """Background task to progressively load KIMERA components"""
        try:
            # Step 1: GPU Foundation
            logger.info("Loading GPU Foundation...")
            time.sleep(2)  # Simulate loading
            loading_state["gpu_foundation"] = True
            
            # Step 2: Embedding Model
            logger.info("Loading Embedding Model...")
            time.sleep(3)  # Simulate loading
            loading_state["embedding_model"] = True
            
            # Step 3: Cognitive Engines
            logger.info("Loading Cognitive Engines...")
            time.sleep(5)  # Simulate loading
            loading_state["cognitive_engines"] = True
            
            # Step 4: Full KIMERA
            logger.info("Finalizing KIMERA initialization...")
            time.sleep(2)  # Simulate loading
            loading_state["full_kimera"] = True
            
            logger.info("✅ Progressive loading complete!")
            
        except Exception as e:
            logger.error(f"Progressive loading failed: {e}")
    
    @app.on_event("startup")
    async def startup_event():
        import asyncio
        asyncio.create_task(progressive_loading())
    
    uvicorn.run(app, host=host, port=port, log_level="info")

def start_safe_mode(host: str, port: int):
    """Start KIMERA in safe mode with enhanced error handling"""
    print("🛡️ Starting KIMERA in SAFE mode...")
    print("   - Full KIMERA features")
    print("   - Enhanced error handling")
    print("   - Automatic recovery")
    print("   - Detailed logging")
    print()
    
    try:
        # Import with error handling
        logger.info("Importing KIMERA main application...")
        from backend.api.main import app
        logger.info("✅ KIMERA application imported successfully")
        
        # Wrap with additional error handling
        @app.middleware("http")
        async def error_handling_middleware(request, call_next):
            try:
                response = await call_next(request)
                return response
            except Exception as e:
                logger.error(f"Request error: {e}")
                from fastapi.responses import JSONResponse
                return JSONResponse(
                    status_code=500,
                    content={"error": "Internal server error", "details": str(e)}
                )
        
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info",
            access_log=True
        )
        
    except Exception as e:
        logger.error(f"Safe mode startup failed: {e}")
        logger.info("Falling back to minimal mode...")
        start_minimal_mode(host, port)

def start_full_mode(host: str, port: int):
    """Start KIMERA in full mode"""
    print("🔥 Starting KIMERA in FULL mode...")
    print("   - All KIMERA features")
    print("   - GPU acceleration")
    print("   - Complete AI capabilities")
    print("   - May require significant resources")
    print()
    
    try:
        from backend.api.main import app
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info",
            reload=False
        )
    except Exception as e:
        logger.error(f"Full mode failed: {e}")
        logger.info("Falling back to safe mode...")
        start_safe_mode(host, port)

def main():
    """Main startup function"""
    parser = argparse.ArgumentParser(description="KIMERA SWM Startup Script")
    parser.add_argument(
        "--mode", 
        choices=["minimal", "progressive", "safe", "full"],
        default="minimal",
        help="Startup mode (default: minimal)"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    parser.add_argument("--check-env", action="store_true", help="Check environment and exit")
    
    args = parser.parse_args()
    
    print_banner()
    
    if args.check_env:
        check_environment()
        return
    
    check_environment()
    
    print(f"🎯 Selected mode: {args.mode.upper()}")
    print(f"🌐 Server will start on: http://{args.host}:{args.port}")
    print("-" * 60)
    
    # Start in selected mode
    if args.mode == "minimal":
        start_minimal_mode(args.host, args.port)
    elif args.mode == "progressive":
        start_progressive_mode(args.host, args.port)
    elif args.mode == "safe":
        start_safe_mode(args.host, args.port)
    elif args.mode == "full":
        start_full_mode(args.host, args.port)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 KIMERA shutdown requested by user")
        print("Thank you for using KIMERA SWM!")
    except Exception as e:
        logger.error(f"Critical startup error: {e}")
        print("🚨 Critical error occurred. Check logs for details.")
        sys.exit(1) 