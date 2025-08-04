#!/usr/bin/env python3
"""
Check Binance Account Balance
============================

Displays all non-zero balances in the account.
"""

import asyncio
import os
import sys

# Add backend to path
sys.path.append("backend")

from trading.api.binance_connector_hmac import BinanceConnector


async def check_balance():
    """Check all account balances."""
    try:
        # Load credentials
        if not os.path.exists("kimera_binance_hmac.env"):
            print("❌ kimera_binance_hmac.env not found!")
            return

        with open("kimera_binance_hmac.env", "r") as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    key, value = line.strip().split("=", 1)
                    os.environ[key] = value

        # Get credentials
        api_key = os.environ.get("BINANCE_API_KEY")
        secret_key = os.environ.get("BINANCE_SECRET_KEY")
        testnet = os.environ.get("BINANCE_USE_TESTNET", "false").lower() == "true"

        if not api_key or not secret_key:
            print("❌ API key and secret key are required")
            return

        # Initialize connector
        connector = BinanceConnector(
            api_key=api_key, secret_key=secret_key, testnet=testnet
        )

        print("🔍 Checking Binance account balances...")
        print(f"🌐 Using {'testnet' if testnet else 'live'} environment")
        print("=" * 50)

        # Get account info
        account_info = await connector.get_account_info()

        if not account_info:
            print("❌ Failed to get account info")
            return

        # Show all non-zero balances
        non_zero_balances = []
        for balance in account_info.get("balances", []):
            free = float(balance["free"])
            locked = float(balance["locked"])
            total = free + locked

            if total > 0:
                non_zero_balances.append(
                    {
                        "asset": balance["asset"],
                        "free": free,
                        "locked": locked,
                        "total": total,
                    }
                )

        if non_zero_balances:
            print("💰 Non-zero balances:")
            for bal in non_zero_balances:
                print(
                    f"  {bal['asset']}: {bal['total']:.8f} (Free: {bal['free']:.8f}, Locked: {bal['locked']:.8f})"
                )
        else:
            print("💸 No balances found - account appears empty")

        # Get current BTC price for reference
        try:
            btc_price = await connector.get_ticker_price("BTCUSDT")
            print(f"\n📊 Current BTC price: ${float(btc_price['price']):,.2f}")
        except Exception as e:
            logger.error(f"Error in test_balance_check.py: {e}", exc_info=True)
            raise  # Re-raise for proper error handling
            print("\n⚠️ Could not get BTC price")

        await connector.close()

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(check_balance())
