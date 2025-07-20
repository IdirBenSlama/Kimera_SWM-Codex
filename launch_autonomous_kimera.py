#!/usr/bin/env python3
"""
KIMERA AUTONOMOUS TRADER LAUNCHER
Full autonomy mode - No limits, complete decision-making freedom
"""

import asyncio
import os
import sys
from datetime import datetime

def display_mission_briefing():
    """Display the autonomous trading mission briefing"""
    print("\n" + "="*60)
    print("🧠 KIMERA AUTONOMOUS TRADER - FULL AUTONOMY MODE")
    print("="*60)
    print("🎯 MISSION: Maximum profit and growth")
    print("💰 CAPITAL: $10 USD equivalent")
    print("⏱️  DURATION: 10 minutes")
    print("🚀 AUTONOMY LEVEL: MAXIMUM")
    print("📊 STRATEGY: Self-determined by Kimera AI")
    print("🔄 POSITION SIZING: Dynamic, self-calculated")
    print("⚡ RISK MANAGEMENT: Adaptive, AI-controlled")
    print("🎲 TRADING PAIRS: Auto-selected by opportunity")
    print("="*60)
    print("🔥 NO PRESET LIMITS - FULL DECISION AUTHORITY")
    print("🧠 Kimera will autonomously:")
    print("   • Select optimal trading pairs")
    print("   • Determine position sizes")
    print("   • Create strategies in real-time")
    print("   • Manage risk dynamically")
    print("   • Optimize performance continuously")
    print("="*60)

async def launch_autonomous_trader():
    """Launch the autonomous trading system"""
    try:
        from autonomous_kimera_trader import KimeraAutonomousTrader
        
        print("\n🚀 INITIALIZING KIMERA AUTONOMOUS TRADER...")
        trader = KimeraAutonomousTrader()
        
        print("🧠 Kimera AI systems online")
        print("📡 Market data connections established")
        print("🔐 Trading permissions verified")
        print("⚡ Autonomous decision engine activated")
        
        print("\n🔥 LAUNCHING 10-MINUTE AUTONOMOUS TRADING SESSION...")
        print("🎯 Kimera now has FULL CONTROL")
        
        # Execute autonomous trading session
        await trader.autonomous_trading_session()
        
        print("\n✅ AUTONOMOUS TRADING MISSION COMPLETED")
        
    except Exception as e:
        print(f"\n❌ MISSION FAILED: {e}")
        print("🛡️  Emergency protocols activated")

def main():
    """Main launcher function"""
    display_mission_briefing()
    
    print("\n🚨 WARNING: This will execute REAL TRADES with REAL MONEY")
    print("🤖 Kimera will have COMPLETE AUTONOMY over trading decisions")
    print("💸 Maximum potential loss: Entire trading balance")
    print("⏰ Session cannot be paused once started")
    
    confirmation = input("\n🔥 CONFIRM LAUNCH AUTONOMOUS TRADER? (type 'LAUNCH' to proceed): ")
    
    if confirmation == 'LAUNCH':
        print(f"\n🚀 MISSION AUTHORIZED at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("🧠 Transferring control to Kimera AI...")
        
        # Run the autonomous trader
        asyncio.run(launch_autonomous_trader())
        
    else:
        print("\n❌ MISSION ABORTED")
        print("🛡️  Autonomous trader not launched")

if __name__ == "__main__":
    main() 