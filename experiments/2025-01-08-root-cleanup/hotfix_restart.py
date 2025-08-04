#!/usr/bin/env python3
"""
HOTFIX RESTART
==============
Quick restart with sell amount validation fixes
"""

import asyncio
import os
import sys

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kimera_working_ultra_aggressive_trader import KimeraWorkingUltraAggressiveTrader
import logging
logger = logging.getLogger(__name__)

async def hotfix_restart():
    logger.info("🔧 HOTFIX RESTART - SELL AMOUNT VALIDATION FIXES")
    logger.info("=" * 60)
    
    logger.info("✅ Applied fixes:")
    logger.info("   - Added null checks for change_24h values")
    logger.info("   - Added validation for sell_amount > 0")
    logger.info("   - Added asset amount validation")
    logger.info("   - Enhanced error messages")
    
    trader = KimeraWorkingUltraAggressiveTrader()
    
    logger.info("\n🚀 Starting 5-minute ultra-aggressive session with fixes...")
    await trader.run_ultra_aggressive_session(5)

if __name__ == "__main__":
    asyncio.run(hotfix_restart()) 