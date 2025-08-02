#!/usr/bin/env python3
"""
REAL CDP TRADER - LIVE MONEY
===========================

This trader places ACTUAL ORDERS on your Coinbase account.
You WILL see pending orders in your Coinbase app/website.

WARNING: THIS USES REAL MONEY
"""

import os
import sys
import time
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Optional

# Try to import the CDP Wallet library
try:
    from coinbase.wallet.client import Client
    CDP_AVAILABLE = True
except ImportError:
    print("❌ CDP Wallet library not available")
    print("   Install with: pip install coinbase")
    CDP_AVAILABLE = False

class RealCDPTrader:
    """
    Real trader that places actual orders on Coinbase
    """
    
    def __init__(self, api_key: str, api_secret: str = None):
        """
        Initialize real CDP trader
        
        Args:
            api_key: Your CDP API key
            api_secret: Your CDP API secret (if required)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        
        print("🔥 INITIALIZING REAL CDP TRADER")
        print(f"   API Key: {api_key[:8]}...")
        print("   WARNING: THIS WILL PLACE REAL ORDERS")
        
        if not CDP_AVAILABLE:
            print("❌ Cannot place real orders - CDP library not available")
            self.client = None
            return
        
        try:
            # Initialize CDP client
            if api_secret:
                self.client = Client(api_key, api_secret)
            else:
                # Try without secret (some APIs work this way)
                self.client = Client(api_key, "")
            
            print("✅ CDP Client initialized")
            
        except Exception as e:
            print(f"❌ Failed to initialize CDP client: {e}")
            self.client = None
    
    async def get_real_balances(self):
        """Get real account balances"""
        if not self.client:
            return {}
        
        try:
            accounts = self.client.get_accounts()
            balances = {}
            
            print("\n💰 REAL ACCOUNT BALANCES:")
            for account in accounts.data:
                currency = account.balance.currency
                amount = float(account.balance.amount)
                balances[currency] = amount
                print(f"   {currency}: {amount}")
            
            return balances
            
        except Exception as e:
            print(f"❌ Failed to get balances: {e}")
            return {}
    
    async def get_current_price(self, symbol: str):
        """Get current price for trading pair"""
        if not self.client:
            return None
        
        try:
            # Get EUR price for cryptocurrency
            rates = self.client.get_exchange_rates(currency=symbol)
            eur_rate = rates.rates.get('EUR')
            
            if eur_rate:
                price = float(eur_rate)
                print(f"💰 {symbol} current price: €{price:.2f}")
                return price
            
        except Exception as e:
            print(f"❌ Failed to get price for {symbol}: {e}")
        
        return None
    
    async def place_real_buy_order(self, symbol: str, eur_amount: float):
        """
        Place REAL buy order on Coinbase
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC', 'ETH')
            eur_amount: Amount in EUR to spend
        """
        if not self.client:
            print("❌ Cannot place order - no CDP client")
            return None
        
        try:
            print(f"\n🚀 PLACING REAL BUY ORDER:")
            print(f"   Symbol: {symbol}")
            print(f"   Amount: €{eur_amount:.2f}")
            print("   THIS IS REAL MONEY!")
            
            # Get the EUR account
            accounts = self.client.get_accounts()
            eur_account = None
            
            for account in accounts.data:
                if account.balance.currency == 'EUR':
                    eur_account = account
                    break
            
            if not eur_account:
                print("❌ No EUR account found")
                return None
            
            # Check sufficient balance
            available_eur = float(eur_account.balance.amount)
            if available_eur < eur_amount:
                print(f"❌ Insufficient EUR balance: €{available_eur:.2f} < €{eur_amount:.2f}")
                return None
            
            # Place buy order
            buy_order = eur_account.buy(
                amount=str(eur_amount),
                currency=symbol,
                payment_method='EUR'
            )
            
            print(f"✅ REAL BUY ORDER PLACED!")
            print(f"   Order ID: {buy_order.id}")
            print(f"   Status: {buy_order.status}")
            print(f"   Amount: €{eur_amount:.2f}")
            print(f"   Symbol: {symbol}")
            
            return {
                'order_id': buy_order.id,
                'status': buy_order.status,
                'amount_eur': eur_amount,
                'symbol': symbol,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Failed to place buy order: {e}")
            return None
    
    async def place_real_sell_order(self, symbol: str, crypto_amount: float):
        """
        Place REAL sell order on Coinbase
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC', 'ETH')
            crypto_amount: Amount of crypto to sell
        """
        if not self.client:
            print("❌ Cannot place order - no CDP client")
            return None
        
        try:
            print(f"\n🚀 PLACING REAL SELL ORDER:")
            print(f"   Symbol: {symbol}")
            print(f"   Amount: {crypto_amount} {symbol}")
            print("   THIS IS REAL MONEY!")
            
            # Get the crypto account
            accounts = self.client.get_accounts()
            crypto_account = None
            
            for account in accounts.data:
                if account.balance.currency == symbol:
                    crypto_account = account
                    break
            
            if not crypto_account:
                print(f"❌ No {symbol} account found")
                return None
            
            # Check sufficient balance
            available_crypto = float(crypto_account.balance.amount)
            if available_crypto < crypto_amount:
                print(f"❌ Insufficient {symbol} balance: {available_crypto} < {crypto_amount}")
                return None
            
            # Place sell order
            sell_order = crypto_account.sell(
                amount=str(crypto_amount),
                currency='EUR'
            )
            
            print(f"✅ REAL SELL ORDER PLACED!")
            print(f"   Order ID: {sell_order.id}")
            print(f"   Status: {sell_order.status}")
            print(f"   Amount: {crypto_amount} {symbol}")
            
            return {
                'order_id': sell_order.id,
                'status': sell_order.status,
                'amount_crypto': crypto_amount,
                'symbol': symbol,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Failed to place sell order: {e}")
            return None
    
    async def test_small_order(self):
        """Test with a very small order to verify everything works"""
        print("\n🧪 TESTING SMALL REAL ORDER")
        print("=" * 30)
        
        # Get balances first
        balances = await self.get_real_balances()
        
        if 'EUR' not in balances or balances['EUR'] < 1.0:
            print("❌ Need at least €1 EUR balance for testing")
            return False
        
        # Test small buy order (€0.50)
        test_amount = 0.5
        result = await self.place_real_buy_order('BTC', test_amount)
        
        if result:
            print("✅ Test order successful!")
            print("   Check your Coinbase account - you should see this order")
            return True
        else:
            print("❌ Test order failed")
            return False

async def main():
    """Test real CDP trading"""
    print("🔥 REAL CDP TRADER TEST")
    print("=" * 30)
    print("⚠️  WARNING: THIS USES REAL MONEY")
    print("⚠️  ORDERS WILL APPEAR IN YOUR COINBASE ACCOUNT")
    print("=" * 30)
    
    API_KEY = os.getenv("CDP_API_KEY_NAME", "")
    
    # Create real trader
    trader = RealCDPTrader(API_KEY)
    
    if trader.client:
        # Test account access
        balances = await trader.get_real_balances()
        
        if balances:
            print("\n✅ Account access successful!")
            
            # Ask user if they want to test a small order
            response = input("\n🤔 Test with small €0.50 order? (yes/no): ")
            
            if response.lower() in ['yes', 'y']:
                success = await trader.test_small_order()
                
                if success:
                    print("\n✅ SUCCESS! Check your Coinbase account for the order.")
                    print("   This proves real trading is working.")
                    print("   You can now scale up to larger amounts.")
                else:
                    print("\n❌ Test failed. Check API credentials and account setup.")
            else:
                print("🔍 Skipping test order. Real trading setup is ready.")
        else:
            print("\n❌ Cannot access account balances")
            print("🔍 Check API credentials and permissions")
    else:
        print("\n❌ Cannot initialize CDP client")
        print("🔧 Install: pip install coinbase")

if __name__ == "__main__":
    asyncio.run(main()) 