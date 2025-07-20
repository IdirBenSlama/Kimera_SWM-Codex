#!/usr/bin/env python3
"""
Test single autonomous trading cycle
"""

import os
import sys
import asyncio

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.trading.autonomous_kimera_trader import create_autonomous_kimera

async def test_cycle():
    """Test a single autonomous trading cycle"""
    print("🧠 TESTING KIMERA AUTONOMOUS TRADING CYCLE")
    print("=" * 50)
    
    try:
        # Create trader
        API_KEY = "9268de76-b5f4-4683-b593-327fb2c19503"
        trader = create_autonomous_kimera(API_KEY, target_eur=100.0)
        
        # Show initial status
        status = await trader.get_portfolio_status()
        print(f"📊 INITIAL STATUS:")
        print(f"   Portfolio: EUR {status['portfolio_value_eur']:.2f}")
        print(f"   Target: EUR {status['target_eur']}")
        print(f"   Progress: {status['progress_pct']:.1f}%")
        print(f"   Active Positions: {status['active_positions']}")
        print(f"   Strategy: {status['current_strategy']}")
        print(f"   Market Regime: {status['market_regime']}")
        print()
        
        # Run one autonomous cycle
        print("🚀 RUNNING AUTONOMOUS TRADING CYCLE...")
        result = await trader.autonomous_trading_cycle()
        
        print(f"✅ Cycle completed, result: {result}")
        
        # Show final status
        final_status = await trader.get_portfolio_status()
        print(f"\n📈 FINAL STATUS:")
        print(f"   Portfolio: EUR {final_status['portfolio_value_eur']:.2f}")
        print(f"   Progress: {final_status['progress_pct']:.1f}%")
        print(f"   Active Positions: {final_status['active_positions']}")
        print(f"   Total Trades: {final_status['total_trades']}")
        print(f"   Win Rate: {final_status['win_rate_pct']:.1f}%")
        
        if result:
            print("\n🎉 TARGET REACHED!")
        else:
            print("\n📊 Cycle complete, continuing toward target...")
            
    except Exception as e:
        print(f"\n❌ Error during autonomous cycle: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_cycle()) 