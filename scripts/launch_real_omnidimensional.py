#!/usr/bin/env python3
"""
Launch REAL Omnidimensional Trading
===================================
WARNING: This executes REAL trades with REAL money!
"""

import asyncio
from kimera_omnidimensional_real_wallet import OmnidimensionalRealTrader
import logging
logger = logging.getLogger(__name__)

async def main():
    logger.info("\n🚀 KIMERA OMNIDIMENSIONAL REAL WALLET TRADING 🚀")
    logger.info("="*50)
    logger.info("⚠️  WARNING: REAL TRADES WITH REAL MONEY!")
    logger.info("📊 Strategies:")
    logger.info("   - Horizontal: Multi-asset momentum & arbitrage")
    logger.info("   - Vertical: Order book microstructure")
    logger.info("="*50)
    
    response = input("\nType 'REAL' to confirm real trading: ")
    
    if response.upper() == 'REAL':
        trader = OmnidimensionalRealTrader()
        await trader.run_real_trading(duration_minutes=5)
    else:
        logger.info("❌ Trading cancelled")

if __name__ == "__main__":
    asyncio.run(main()) 