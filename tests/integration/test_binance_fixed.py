#!/usr/bin/env python3
"""
Test script for the fixed Binance Ed25519 implementation.

This script tests the corrected Ed25519 signature format based on
official Binance documentation.
"""

import asyncio
import logging
import os
import sys

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))

from dotenv import load_dotenv

from src.trading.api.binance_connector_fixed import BinanceConnectorFixed

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_fixed_binance_connector():
    """Test the fixed Binance Ed25519 implementation."""
    print("🧪 Testing Fixed Binance Ed25519 Implementation")
    print("=" * 60)

    # Get configuration
    api_key = os.getenv("BINANCE_API_KEY")
    private_key_path = os.getenv("BINANCE_PRIVATE_KEY_PATH", "binance_private_key.pem")
    use_testnet = os.getenv("BINANCE_USE_TESTNET", "true").lower() == "true"

    if not api_key:
        print("❌ BINANCE_API_KEY not found in environment")
        return False

    if not os.path.exists(private_key_path):
        print(f"❌ Private key file not found: {private_key_path}")
        return False

    print(f"✅ API Key: {api_key[:10]}...")
    print(f"✅ Private Key: {private_key_path}")
    print(f"✅ Testnet: {use_testnet}")
    print()

    try:
        # Initialize the fixed connector
        print("🔧 Initializing Fixed Binance Connector...")
        connector = BinanceConnectorFixed(
            api_key=api_key, private_key_path=private_key_path, testnet=use_testnet
        )

        async with connector:
            print("✅ Connector initialized successfully")
            print()

            # Test 1: Get account information
            print("🧪 Test 1: Get Account Information")
            print("-" * 40)

            try:
                account_info = await connector.get_account()
                print("✅ Account information retrieved successfully!")
                print(f"   Account type: {account_info.get('accountType', 'Unknown')}")
                print(f"   Can trade: {account_info.get('canTrade', False)}")
                print(f"   Can withdraw: {account_info.get('canWithdraw', False)}")
                print(f"   Can deposit: {account_info.get('canDeposit', False)}")
                print(f"   Number of balances: {len(account_info.get('balances', []))}")

                # Show non-zero balances
                balances = account_info.get("balances", [])
                non_zero_balances = [
                    b
                    for b in balances
                    if float(b.get("free", 0)) > 0 or float(b.get("locked", 0)) > 0
                ]

                if non_zero_balances:
                    print("   Non-zero balances:")
                    for balance in non_zero_balances[:5]:  # Show first 5
                        asset = balance.get("asset", "Unknown")
                        free = float(balance.get("free", 0))
                        locked = float(balance.get("locked", 0))
                        print(f"     {asset}: {free} free, {locked} locked")
                else:
                    print("   No non-zero balances found (testnet)")

                print()

                # Test 2: Get specific balance
                print("🧪 Test 2: Get USDT Balance")
                print("-" * 40)

                usdt_balance = await connector.get_balance("USDT")
                print(f"✅ USDT Balance:")
                print(f"   Free: {usdt_balance['free']}")
                print(f"   Locked: {usdt_balance['locked']}")
                print(f"   Total: {usdt_balance['total']}")
                print()

                # Test 3: Test market data (unsigned request)
                print("🧪 Test 3: Get Market Data (Unsigned)")
                print("-" * 40)

                ticker = await connector.get_ticker("BTCUSDT")
                print(f"✅ BTCUSDT Ticker:")
                print(f"   Price: ${float(ticker['lastPrice']):,.2f}")
                print(f"   24h Change: {float(ticker['priceChangePercent']):.2f}%")
                print(f"   24h Volume: {float(ticker['volume']):,.2f} BTC")
                print()

                # Test 4: Get open orders
                print("🧪 Test 4: Get Open Orders")
                print("-" * 40)

                open_orders = await connector.get_open_orders()
                print(f"✅ Open orders: {len(open_orders)}")
                if open_orders:
                    for order in open_orders[:3]:  # Show first 3
                        symbol = order.get("symbol", "Unknown")
                        side = order.get("side", "Unknown")
                        order_type = order.get("type", "Unknown")
                        status = order.get("status", "Unknown")
                        print(f"   {symbol} {side} {order_type} - {status}")
                else:
                    print("   No open orders")
                print()

                print("🎉 ALL TESTS PASSED!")
                print("✅ Ed25519 signature validation is working correctly!")
                return True

            except Exception as e:
                print(f"❌ Account request failed: {e}")
                logger.error(f"Account request error: {e}", exc_info=True)
                return False

    except Exception as e:
        print(f"❌ Connector initialization failed: {e}")
        logger.error(f"Initialization error: {e}", exc_info=True)
        return False


async def main():
    """Main test function."""
    print("🚀 Kimera Binance Ed25519 Fixed Implementation Test")
    print("=" * 60)

    success = await test_fixed_binance_connector()

    if success:
        print("\n🎯 RESULT: Ed25519 implementation is WORKING!")
        print("✅ Ready to integrate with Kimera trading system")
    else:
        print("\n❌ RESULT: Ed25519 implementation needs further debugging")
        print("💡 Check the logs above for specific error details")

    return success


if __name__ == "__main__":
    asyncio.run(main())
