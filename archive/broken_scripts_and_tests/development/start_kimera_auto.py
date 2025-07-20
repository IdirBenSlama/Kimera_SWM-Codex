#!/usr/bin/env python3
"""
Auto-start KIMERA Full Server without user interaction
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the server class
from start_comprehensive_server import ComprehensiveServerLauncher
import signal
import logging

logger = logging.getLogger(__name__)

def main():
    """Main entry point"""
    print("🌟 KIMERA SWM FULL SERVER - ULTIMATE COMPLETE SYSTEM")
    print("=" * 60)
    print("🚀 Auto-starting with EVERYTHING enabled")
    print("📊 Expected startup time: 5-15 minutes")
    print("🧠 Cognitive fidelity: 100% - NO COMPROMISES")
    print("=" * 60)
    
    # Create full server manager
    full_server = ComprehensiveServerLauncher()
    
    # Setup signal handlers
    def signal_handler(signum, frame):
        logger.info("🛑 Received shutdown signal")
        full_server.cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run full server
    success = full_server.run()
    
    if success:
        logger.info("🌟 KIMERA Full Server completed successfully")
        sys.exit(0)
    else:
        logger.error("❌ KIMERA Full Server failed")
        sys.exit(1)

if __name__ == "__main__":
    main()