#!/usr/bin/env python3
"""
RESTART FIXED TRADER
===================
Quick restart with the None value fixes
"""

import asyncio
import os
import sys

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kimera_working_ultra_aggressive_trader import KimeraWorkingUltraAggressiveTrader
import logging
logger = logging.getLogger(__name__)

async def restart_trader():
    logger.info("ðŸ”§ RESTARTING WITH FIXES APPLIED")
    logger.info("=" * 50)
    
    trader = KimeraWorkingUltraAggressiveTrader()
    
    logger.info("\nðŸš€ Starting 5-minute ultra-aggressive session...")
    await trader.run_ultra_aggressive_session(5)

if __name__ == "__main__":
    asyncio.run(restart_trader()) 