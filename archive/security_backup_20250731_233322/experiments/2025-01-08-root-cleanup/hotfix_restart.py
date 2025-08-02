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

async def hotfix_restart():
    print("ðŸ”§ HOTFIX RESTART - SELL AMOUNT VALIDATION FIXES")
    print("=" * 60)
    
    print("âœ… Applied fixes:")
    print("   - Added null checks for change_24h values")
    print("   - Added validation for sell_amount > 0")
    print("   - Added asset amount validation")
    print("   - Enhanced error messages")
    
    trader = KimeraWorkingUltraAggressiveTrader()
    
    print("\nðŸš€ Starting 5-minute ultra-aggressive session with fixes...")
    await trader.run_ultra_aggressive_session(5)

if __name__ == "__main__":
    asyncio.run(hotfix_restart()) 