#!/usr/bin/env python3
"""
Quick & Reliable Kimera Revolutionary Thermodynamic Server
=========================================================
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

logger.info("🌟 KIMERA REVOLUTIONARY THERMODYNAMIC SYSTEM")
logger.info("=" * 60)
logger.info("🚀 Quick server startup for world's first physics-compliant AI")

try:
    logger.info("📦 Loading revolutionary thermodynamic system...")
    from src.main import app
    logger.info("✅ Revolutionary system loaded successfully!")
    
    logger.info("🌐 Starting server on port 8003...")
    import uvicorn
    
    logger.info("📡 Server URLs:")
    logger.info("   • Documentation: http://localhost:8003/docs")
    logger.info("   • Thermodynamic API: http://localhost:8003/kimera/unified-thermodynamic/status")
    logger.info("   • Health Check: http://localhost:8003/health")
    logger.info("")
    logger.info("🔥 Starting now...")
    
    uvicorn.run(
        app, 
        host="127.0.0.1", 
        port=8003, 
        log_level="info",
        access_log=True
    )
    
except Exception as e:
    logger.info(f"❌ Error: {e}")
    import traceback
import logging
logger = logging.getLogger(__name__)
    traceback.print_exc() 