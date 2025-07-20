#!/usr/bin/env python3
"""
Start KIMERA server with Mirror Portal integration
"""

import sys
import os
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'kimera_server_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def start_server():
    """Start the KIMERA server"""
    logger = logging.getLogger(__name__)
    logger.info("="*80)
    logger.info("🚀 STARTING KIMERA SERVER WITH MIRROR PORTAL INTEGRATION")
    logger.info("="*80)
    logger.info(f"Start time: {datetime.now()}")
    logger.info("")
    
    try:
        # Import and run the main app
        from backend.api.main import app
        import uvicorn
        
        logger.info("📡 Starting KIMERA server on http://localhost:8001")
        logger.info("   Press Ctrl+C to stop")
        logger.info()
        
        # Run the server
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8001,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        logger.info("\n⏹️ Server stopped by user")
    except Exception as e:
        logger.error(f"\n❌ Error starting server: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    start_server()