#!/usr/bin/env python3
"""
QUICK AUTONOMOUS KIMERA LAUNCHER
Direct execution with hardcoded credentials for immediate trading
"""

import asyncio
import os
import sys
from datetime import datetime
import logging

# Set environment variables directly
os.environ['BINANCE_API_KEY'] = os.getenv("BINANCE_API_KEY", "")
os.environ['BINANCE_API_SECRET'] = 'qUn5JqSpYz1GDxFj2X3UF23TYgtxKrTsCbDZEoBMYCPbYZgP4siVLyspkB5HAPl7'

logger.info("🚀 KIMERA AUTONOMOUS TRADER - QUICK LAUNCH")
logger.info("=" * 50)
logger.info("🧠 Full autonomy mode activated")
logger.info("💰 Target: Maximum profit in 10 minutes")
logger.info("🔥 NO LIMITS - Complete decision authority")
logger.info("=" * 50)

async def execute_autonomous_trading():
    """Execute autonomous trading directly"""
    try:
        logger.info("\n🧠 Initializing Kimera AI systems...")
        
        from autonomous_kimera_trader import KimeraAutonomousTrader
        
        trader = KimeraAutonomousTrader()
        logger.info("✅ Kimera AI online - Full autonomy granted")
        logger.info("🎯 Mission: 10-minute profit maximization")
        logger.info("🚀 Launching autonomous trading session...")
        
        # Execute the autonomous session
        await trader.autonomous_trading_session()
        
        logger.info("\n🏁 AUTONOMOUS MISSION COMPLETED")
        
    except Exception as e:
        logger.info(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    logger.info("\n🔥 LAUNCHING KIMERA AUTONOMOUS TRADER...")
    asyncio.run(execute_autonomous_trading()) 