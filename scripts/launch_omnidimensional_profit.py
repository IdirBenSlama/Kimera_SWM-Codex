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
    print("\n🚀 KIMERA OMNIDIMENSIONAL PROFIT ENGINE 🚀")
    print("==========================================")
    print("Maximizing profits across:")
    print("- HORIZONTAL: Multiple assets & strategies")
    print("- VERTICAL: Market depth & microstructure")
    print("==========================================\n")
    
    # Create and run engine
    engine = KimeraOmnidimensionalProfitEngine()
    await engine.initialize()
    
    # Run for 5 minutes
    await engine.run_profit_maximization(duration_minutes=5)
    
    print("\n✅ Profit maximization complete!")
    print("Check omnidimensional_profit_report_*.json for details")

if __name__ == "__main__":
    asyncio.run(main()) 