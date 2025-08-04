#!/usr/bin/env python3
"""
Launch Kimera Omnidimensional Profit Engine
==========================================
Quick launcher for maximum profit across all dimensions
"""

import asyncio
import logging
from kimera_omnidimensional_profit_engine import KimeraOmnidimensionalProfitEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

async def main():
    logger.info("\n🚀 KIMERA OMNIDIMENSIONAL PROFIT ENGINE 🚀")
    logger.info("==========================================")
    logger.info("Maximizing profits across:")
    logger.info("- HORIZONTAL: Multiple assets & strategies")
    logger.info("- VERTICAL: Market depth & microstructure")
    logger.info("==========================================\n")
    
    # Create and run engine
    engine = KimeraOmnidimensionalProfitEngine()
    await engine.initialize()
    
    # Run for 5 minutes
    await engine.run_profit_maximization(duration_minutes=5)
    
    logger.info("\n✅ Profit maximization complete!")
    logger.info("Check omnidimensional_profit_report_*.json for details")

if __name__ == "__main__":
    asyncio.run(main()) 