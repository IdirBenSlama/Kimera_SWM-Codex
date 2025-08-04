#!/usr/bin/env python3
"""
KIMERA AUTONOMOUS LAUNCHER
=========================

Verifies CDP setup and launches real money autonomous trading
"""

import os
import sys
import subprocess
from dotenv import load_dotenv

def verify_and_launch():
    """Verify setup and launch Kimera"""
    logger.info("🚀 KIMERA AUTONOMOUS LAUNCHER")
    logger.info("=" * 60)
    
    # Load environment
    load_dotenv('kimera_cdp_live.env')
    
    # Check credentials
    api_key = os.getenv('CDP_API_KEY_NAME')
    private_key = os.getenv('CDP_API_KEY_PRIVATE_KEY')
    
    if not api_key or not private_key:
        logger.info("❌ CDP credentials not found!")
        return False
    
    logger.info(f"✅ API Key: {api_key}")
    logger.info(f"✅ Private Key: {'*' * 40}{private_key[-10:]}")
    
    # Check CDP SDK
    try:
        import cdp
import logging
logger = logging.getLogger(__name__)
        logger.info("✅ CDP SDK available")
    except ImportError:
        logger.info("❌ CDP SDK not installed")
        logger.info("Installing CDP SDK...")
        subprocess.run([sys.executable, "-m", "pip", "install", "cdp-sdk"])
    
    # Final confirmation
    logger.info("\n" + "⚠️ " * 10)
    logger.info("REAL MONEY TRADING - KIMERA WILL HAVE FULL CONTROL")
    logger.info("The system will autonomously trade with your real wallet")
    logger.info("⚠️ " * 10)
    
    response = input("\nProceed with REAL MONEY autonomous trading? (yes/no): ")
    
    if response.lower() == 'yes':
        logger.info("\n🔥 LAUNCHING KIMERA AUTONOMOUS MISSION...")
        subprocess.run([sys.executable, "kimera_autonomous_real_money.py"])
    else:
        logger.info("❌ Launch cancelled")

if __name__ == "__main__":
    verify_and_launch() 