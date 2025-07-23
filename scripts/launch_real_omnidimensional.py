#!/usr/bin/env python3
"""
Launch REAL Omnidimensional Trading
===================================
WARNING: This executes REAL trades with REAL money!
"""

import asyncio
from kimera_omnidimensional_real_wallet import OmnidimensionalRealTrader

async def main():
    print("\n🚀 KIMERA OMNIDIMENSIONAL REAL WALLET TRADING 🚀")
    print("="*50)
    print("⚠️  WARNING: REAL TRADES WITH REAL MONEY!")
    print("📊 Strategies:")
    print("   - Horizontal: Multi-asset momentum & arbitrage")
    print("   - Vertical: Order book microstructure")
    print("="*50)
    
    response = input("\nType 'REAL' to confirm real trading: ")
    
    if response.upper() == 'REAL':
        trader = OmnidimensionalRealTrader()
        await trader.run_real_trading(duration_minutes=5)
    else:
        print("❌ Trading cancelled")

if __name__ == "__main__":
    asyncio.run(main()) 