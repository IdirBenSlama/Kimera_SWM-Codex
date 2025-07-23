#!/usr/bin/env python3
"""
AUTO TRADE NOW - IMMEDIATE TRADING EXECUTION
============================================
🚀 INSTANT BULLETPROOF TRADING 🚀
🛡️ NO PROMPTS - IMMEDIATE EXECUTION 🛡️
"""

import asyncio
from kimera_ultimate_bulletproof_trader import KimeraUltimateBulletproofTrader

async def auto_trade_now():
    """Start trading immediately with no prompts"""
    print("🚀" * 80)
    print("🚨 AUTO TRADE NOW - IMMEDIATE EXECUTION")
    print("🛡️ BULLETPROOF TRADING STARTING IN 3 SECONDS...")
    print("🚀" * 80)
    
    # Wait 3 seconds for dramatic effect
    await asyncio.sleep(3)
    
    # Create trader instance
    trader = KimeraUltimateBulletproofTrader()
    
    # Start 5-minute trading session immediately
    await trader.run_ultimate_bulletproof_session(5)

if __name__ == "__main__":
    asyncio.run(auto_trade_now()) 