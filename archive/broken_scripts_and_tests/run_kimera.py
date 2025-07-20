#!/usr/bin/env python3
"""
KIMERA SWM Startup Script
"""
import uvicorn
import sys
import socket
from backend.api.main import app

# Initialize structured logger
from backend.utils.kimera_logger import get_system_logger
logger = get_system_logger(__name__)


def find_free_port(start_port=8001, max_port=8010):
    """Find a free port starting from start_port"""
    for port in range(start_port, max_port):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('0.0.0.0', port))
                return port
        except OSError:
            continue
    return None

if __name__ == "__main__":
    logger.info("🚀 Starting KIMERA SWM API Server...")
    
    # Find a free port
    port = find_free_port()
    if port is None:
        logger.error("❌ No free ports available between 8000-8010")
        sys.exit(1)
    
    logger.info(f"📡 Server will start on http://localhost:{port}")
    logger.info("📚 API Documentation available at:")
    logger.info(f"   - Swagger UI: http://localhost:{port}/docs")
    logger.info(f"   - ReDoc: http://localhost:{port}/redoc")
    logger.debug("\n🔧 Available endpoints:")
    logger.info("   - POST /geoids - Create semantic geoids")
    logger.info("   - POST /process/contradictions - Process contradictions")
    logger.info("   - GET /system/status - System status")
    logger.info("   - GET /system/stability - Stability metrics")
    logger.info("\n⚡ Press Ctrl+C to stop the server\n")
    
    try:
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=port,
            log_level="info",
            reload=False
        )
    except KeyboardInterrupt:
        logger.info("\n🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        sys.exit(1)
