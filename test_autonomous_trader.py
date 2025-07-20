#!/usr/bin/env python3
"""
Test Autonomous Trader Creation
"""

import os
import sys
import asyncio
import logging

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.trading.autonomous_kimera_trader import create_autonomous_kimera

async def test_trader_creation():
    """Test creating the autonomous trader"""
    try:
        print("Testing autonomous trader creation...")
        
        # Create trader
        API_KEY = "9268de76-b5f4-4683-b593-327fb2c19503"
        trader = create_autonomous_kimera(API_KEY, target_eur=100.0)
        print(f"✓ Trader created successfully")
        
        # Test portfolio status
        status = await trader.get_portfolio_status()
        print(f"✓ Portfolio status: {status}")
        
        # Test market data fetch
        print("Testing market data fetch...")
        df = await trader.fetch_market_data('bitcoin')
        print(f"✓ Market data fetched, shape: {df.shape}")
        
        # Test signal generation
        print("Testing signal generation...")
        signal = trader.generate_cognitive_signal('bitcoin')
        print(f"✓ Signal generated: {signal}")
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(test_trader_creation()) 