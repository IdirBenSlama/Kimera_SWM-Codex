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
os.environ['BINANCE_API_KEY'] = 'Y9WyflPyK1tVXnET3CTMvSdCbPia3Nhtd89VYWjS9RaAbQ0KEhHezkcGSCySQ8cL'
os.environ['BINANCE_API_SECRET'] = 'qUn5JqSpYz1GDxFj2X3UF23TYgtxKrTsCbDZEoBMYCPbYZgP4siVLyspkB5HAPl7'

print("🚀 KIMERA AUTONOMOUS TRADER - QUICK LAUNCH")
print("=" * 50)
print("🧠 Full autonomy mode activated")
print("💰 Target: Maximum profit in 10 minutes")
print("🔥 NO LIMITS - Complete decision authority")
print("=" * 50)

async def execute_autonomous_trading():
    """Execute autonomous trading directly"""
    try:
        print("\n🧠 Initializing Kimera AI systems...")
        
        from autonomous_kimera_trader import KimeraAutonomousTrader
        
        trader = KimeraAutonomousTrader()
        print("✅ Kimera AI online - Full autonomy granted")
        print("🎯 Mission: 10-minute profit maximization")
        print("🚀 Launching autonomous trading session...")
        
        # Execute the autonomous session
        await trader.autonomous_trading_session()
        
        print("\n🏁 AUTONOMOUS MISSION COMPLETED")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("\n🔥 LAUNCHING KIMERA AUTONOMOUS TRADER...")
    asyncio.run(execute_autonomous_trading()) 