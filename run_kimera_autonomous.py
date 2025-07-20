#!/usr/bin/env python3
"""
KIMERA AUTONOMOUS TRADER - SIMPLE RUNNER
========================================

Simplified autonomous trader execution
"""

import os
import sys
import asyncio

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.trading.autonomous_kimera_trader import create_autonomous_kimera

async def run_autonomous():
    """Run autonomous trader"""
    print("🧠 STARTING KIMERA AUTONOMOUS TRADER")
    print("   Target: EUR 5 → EUR 100")
    print("   Mode: FULLY AUTONOMOUS")
    print("   Safety Limits: NONE")
    print()
    
    # Create trader
    API_KEY = "9268de76-b5f4-4683-b593-327fb2c19503"
    trader = create_autonomous_kimera(API_KEY, target_eur=100.0)
    
    # Show status
    status = await trader.get_portfolio_status()
    print(f"Portfolio Value: EUR {status['portfolio_value_eur']:.2f}")
    print(f"Target: EUR {status['target_eur']}")
    print(f"Progress: {status['progress_pct']:.1f}%")
    print()
    
    print("🚀 LAUNCHING AUTONOMOUS TRADING...")
    
    # Start autonomous trading
    await trader.run_autonomous_trader(cycle_interval_minutes=15)

if __name__ == "__main__":
    try:
        asyncio.run(run_autonomous())
    except KeyboardInterrupt:
        print("\n🛑 Autonomous trading stopped")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc() 