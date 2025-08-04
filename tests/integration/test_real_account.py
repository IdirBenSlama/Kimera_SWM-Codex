#!/usr/bin/env python3
"""
Test Real Coinbase Account Connection
"""

import asyncio
import os
import sys

import requests

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))

from dataclasses import dataclass
from datetime import datetime

# Import live trader components
import requests

# Import from the direct path
import src.trading.autonomous_kimera_trader as base_trader
from src.trading.autonomous_kimera_trader import (
    CognitiveSignal,
    KimeraAutonomousTrader,
    MarketRegime,
    TradingStrategy,
)


async def test_real_account():
    """Test connection to real Coinbase account"""
    print("🔥 TESTING REAL COINBASE ACCOUNT CONNECTION")
    print("=" * 50)

    API_KEY = os.getenv("CDP_API_KEY_NAME", "")

    try:
        # Create live trader
        trader = create_live_autonomous_kimera(API_KEY, target_eur=100.0)

        # Test account balance retrieval
        print("📊 Checking real account balance...")
        real_balance = await trader.get_real_account_balance()

        if real_balance > 0:
            print(f"✅ SUCCESS: Real EUR balance: €{real_balance:.2f}")
            print(f"📈 Current portfolio value: €{trader.portfolio_value:.2f}")

            if real_balance >= 1.0:
                print("✅ Sufficient balance for autonomous trading")
                print(
                    f"📊 Recommended allocation per trade: €{real_balance * 0.2:.2f} (20%)"
                )
            else:
                print("⚠️ Low balance - consider depositing more EUR for trading")

        else:
            print("❌ No EUR balance found or connection failed")
            print("🔍 Please check:")
            print("   - API key is valid")
            print("   - Account has EUR balance")
            print("   - API permissions include trading")

        # Test symbol mapping
        print("\n🔗 Testing symbol mapping:")
        test_symbols = ["bitcoin", "ethereum", "solana"]
        for symbol in test_symbols:
            cdp_symbol = trader.get_cdp_symbol(symbol)
            print(f"   {symbol} → {cdp_symbol}")

        return real_balance > 0

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_real_account())

    if success:
        print("\n🚀 READY FOR LIVE AUTONOMOUS TRADING")
        print("   Run: python launch_live_kimera.py")
    else:
        print("\n❌ Fix connection issues before live trading")
