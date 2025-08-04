#!/usr/bin/env python3
"""
VERIFY CREDENTIALS AND RUN AUTONOMOUS TRADING
=============================================

This script verifies your CDP credentials and launches autonomous trading.
"""

import os
import sys
from pathlib import Path

def verify_credentials():
    """Verify CDP credentials are configured"""
    logger.info("🔍 VERIFYING CDP CREDENTIALS...")
    
    config_file = "kimera_cdp_live.env"
    
    if not Path(config_file).exists():
        logger.info("❌ Configuration file not found!")
        logger.info("Please ensure kimera_cdp_live.env exists")
        return False
    
    try:
        from dotenv import load_dotenv
        load_dotenv(config_file)
        
        api_key_name = os.getenv('CDP_API_KEY_NAME')
        private_key = os.getenv('CDP_API_KEY_PRIVATE_KEY')
        network_id = os.getenv('CDP_NETWORK_ID')
        
        logger.info(f"✅ API Key Name: {api_key_name}")
        
        if not private_key or private_key == "YOUR_CDP_PRIVATE_KEY_HERE":
            logger.info("❌ Private key not configured!")
            logger.info("Please edit kimera_cdp_live.env and add your CDP private key")
            return False
        
        logger.info(f"✅ Private Key: ***{private_key[-10:] if len(private_key) > 10 else '***'}")
        logger.info(f"✅ Network: {network_id}")
        
        # Test CDP SDK
        try:
            from cdp import CdpClient
            logger.info("✅ CDP SDK available")
        except ImportError:
            logger.info("❌ CDP SDK not available. Run: pip install cdp-sdk")
            return False
        
        logger.info("✅ Credentials verified successfully!")
        return True
        
    except Exception as e:
        logger.info(f"❌ Verification error: {e}")
        return False

def launch_autonomous_trading():
    """Launch the autonomous trading system"""
    logger.info("\n🚀 LAUNCHING KIMERA AUTONOMOUS TRADING...")
    logger.info("⚠️  Kimera will now have autonomous control of your wallet")
    logger.info("⚠️  Starting with testnet for safety")
    logger.info()
    
    try:
        # Import and run the live integration
        import asyncio
        from kimera_cdp_live_integration import main as live_trading_main
import logging
logger = logging.getLogger(__name__)
        
        logger.info("🎯 Starting autonomous trading session...")
        asyncio.run(live_trading_main())
        
    except Exception as e:
        logger.info(f"❌ Launch error: {e}")
        logger.info("Please check the logs for details")

def main():
    """Main function"""
    logger.info("🔐 KIMERA CDP AUTONOMOUS TRADING LAUNCHER")
    logger.info("=" * 50)
    
    # Step 1: Verify credentials
    if not verify_credentials():
        logger.info("\n❌ CREDENTIAL VERIFICATION FAILED")
        logger.info("Please configure your credentials before proceeding.")
        logger.info()
        logger.info("📝 INSTRUCTIONS:")
        logger.info("1. Edit kimera_cdp_live.env")
        logger.info("2. Replace YOUR_CDP_PRIVATE_KEY_HERE with your actual CDP private key")
        logger.info("3. Run this script again")
        return
    
    # Step 2: Launch autonomous trading
    logger.info("\n✅ CREDENTIALS VERIFIED")
    
    proceed = input("\n🚀 Launch autonomous trading? (yes/no): ").strip().lower()
    
    if proceed == 'yes':
        launch_autonomous_trading()
    else:
        logger.info("Launch cancelled.")

if __name__ == "__main__":
    main() 