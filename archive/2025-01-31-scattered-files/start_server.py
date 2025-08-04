#!/usr/bin/env python3
"""
Simple Server Startup for Kimera Revolutionary Thermodynamic System
==================================================================
"""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def start_server():
    logger.info("🌟 KIMERA REVOLUTIONARY THERMODYNAMIC SYSTEM")
    logger.info("=" * 50)
    logger.info("🚀 Starting the world's first physics-compliant AI server...")
    logger.info("")
    
    try:
        # Import the FastAPI app
        from src.main import app
        logger.info("✅ Revolutionary thermodynamic app loaded successfully!")
        
        # Start with uvicorn
        import uvicorn
        logger.info("🌐 Starting API server...")
        logger.info("📡 Server will be available at:")
        logger.info("   • API Documentation: http://localhost:8001/docs")
        logger.info("   • Revolutionary Endpoints: http://localhost:8001/kimera/unified-thermodynamic/")
        logger.info("   • System Status: http://localhost:8001/kimera/unified-thermodynamic/status")
        logger.info("")
        logger.info("🔥 Starting revolutionary thermodynamic server...")
        
        # Configure and run server
        config = uvicorn.Config(
            app=app,
            host="0.0.0.0",
            port=8001,
            log_level="info",
            reload=False
        )
        
        server = uvicorn.Server(config)
        server.run()
        
    except Exception as e:
        logger.info(f"❌ Server startup failed: {e}")
        import traceback
import logging
logger = logging.getLogger(__name__)
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = start_server()
    sys.exit(exit_code) 