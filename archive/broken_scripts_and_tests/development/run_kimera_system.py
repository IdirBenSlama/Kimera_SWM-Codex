#!/usr/bin/env python3
"""
Run the entire Kimera system
"""

import sys
import os
import subprocess
import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run the entire Kimera system"""
    logger.info("="*80)
    logger.info("🚀 STARTING KIMERA SYSTEM")
    logger.info("="*80)
    
    # Add current directory to Python path
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
    try:
        # Option 1: Try to run the API server directly
        logger.info("Starting Kimera API Server...")
        
        # Import FastAPI app directly
        from backend.api.main import app
        import uvicorn
        
        logger.info("✅ Modules loaded successfully")
        logger.info("📡 Starting server on http://localhost:8001")
        logger.info("📚 API Documentation will be available at:")
        logger.info("   - Swagger UI: http://localhost:8001/docs")
        logger.info("   - ReDoc: http://localhost:8001/redoc")
        logger.info("\n⚡ Press Ctrl+C to stop the server\n")
        
        # Run the server
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8001,
            log_level="info"
        )
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.info("\nTrying alternative approach...")
        
        # Option 2: Run a demo instead
        try:
            logger.info("Running Kimera Ultimate Demo...")
            subprocess.run([sys.executable, "kimera_ultimate_demo.py"], check=True)
        except subprocess.CalledProcessError:
            logger.error("Demo failed to run")
            
            # Option 3: Show available options
            logger.info("\n" + "="*80)
            logger.info("AVAILABLE KIMERA COMPONENTS:")
            logger.info("="*80)
            
            components = [
                ("kimera_ultimate_demo.py", "Comprehensive trading system demo"),
                ("kimera_chat_interface.py", "Interactive chat interface"),
                ("enhanced_kimera_live_demo.py", "Enhanced live demonstration"),
                ("final_kimera_demo.py", "Final integrated demo"),
                ("simple_kimera_test.py", "Simple functionality test"),
                ("test_kimera_live.py", "Live system test"),
                ("kimera_web_chat.py", "Web-based chat interface"),
            ]
            
            logger.info("\nYou can run any of these components:")
            for script, description in components:
                if os.path.exists(script):
                    logger.info(f"  ✅ python {script} - {description}")
                else:
                    logger.info(f"  ❌ {script} - Not found")
            
            logger.info("\nFor the full API server, you may need to:")
            logger.info("1. Install dependencies: pip install -r requirements.txt")
            logger.info("2. Fix import issues in the codebase")
            logger.info("3. Run: python run_kimera.py")
            
    except KeyboardInterrupt:
        logger.info("\n🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()