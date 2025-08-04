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
import logging
logger = logging.getLogger(__name__)
    CDP_AVAILABLE = True
except ImportError:
    logger.info("❌ CDP Wallet library not available")
    logger.info("   Install with: pip install coinbase")
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
        
        logger.info("🔥 INITIALIZING REAL CDP TRADER")
        logger.info(f"   API Key: {api_key[:8]}...")
        logger.info("   WARNING: THIS WILL PLACE REAL ORDERS")
        
        if not CDP_AVAILABLE:
            logger.info("❌ Cannot place real orders - CDP library not available")
            self.client = None
            return
        
        try:
            # Initialize CDP client
            if api_secret:
                self.client = Client(api_key, api_secret)
            else:
                # Try without secret (some APIs work this way)
                self.client = Client(api_key, "")
            
            logger.info("✅ CDP Client initialized")
            
        except Exception as e:
            logger.info(f"❌ Failed to initialize CDP client: {e}")
            self.client = None
    
    async def get_real_balances(self):
        """Get real account balances"""
        if not self.client:
            return {}
        
        try:
            accounts = self.client.get_accounts()
            balances = {}
            
            logger.info("\n💰 REAL ACCOUNT BALANCES:")
            for account in accounts.data:
                currency = account.balance.currency
                amount = float(account.balance.amount)
                balances[currency] = amount
                logger.info(f"   {currency}: {amount}")
            
            return balances
            
        except Exception as e:
            logger.info(f"❌ Failed to get balances: {e}")
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
                logger.info(f"💰 {symbol} current price: €{price:.2f}")
                return price
            
        except Exception as e:
            logger.info(f"❌ Failed to get price for {symbol}: {e}")
        
        return None
    
    async def place_real_buy_order(self, symbol: str, eur_amount: float):
        """
        Place REAL buy order on Coinbase
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC', 'ETH')
            eur_amount: Amount in EUR to spend
        """
        if not self.client:
            logger.info("❌ Cannot place order - no CDP client")
            return None
        
        try:
            logger.info(f"\n🚀 PLACING REAL BUY ORDER:")
            logger.info(f"   Symbol: {symbol}")
            logger.info(f"   Amount: €{eur_amount:.2f}")
            logger.info("   THIS IS REAL MONEY!")
            
            # Get the EUR account
            accounts = self.client.get_accounts()
            eur_account = None
            
            for account in accounts.data:
                if account.balance.currency == 'EUR':
                    eur_account = account
                    break
            
            if not eur_account:
                logger.info("❌ No EUR account found")
                return None
            
            # Check sufficient balance
            available_eur = float(eur_account.balance.amount)
            if available_eur < eur_amount:
                logger.info(f"❌ Insufficient EUR balance: €{available_eur:.2f} < €{eur_amount:.2f}")
                return None
            
            # Place buy order
            buy_order = eur_account.buy(
                amount=str(eur_amount),
                currency=symbol,
                payment_method='EUR'
            )
            
            logger.info(f"✅ REAL BUY ORDER PLACED!")
            logger.info(f"   Order ID: {buy_order.id}")
            logger.info(f"   Status: {buy_order.status}")
            logger.info(f"   Amount: €{eur_amount:.2f}")
            logger.info(f"   Symbol: {symbol}")
            
            return {
                'order_id': buy_order.id,
                'status': buy_order.status,
                'amount_eur': eur_amount,
                'symbol': symbol,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.info(f"❌ Failed to place buy order: {e}")
            return None
    
    async def place_real_sell_order(self, symbol: str, crypto_amount: float):
        """
        Place REAL sell order on Coinbase
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC', 'ETH')
            crypto_amount: Amount of crypto to sell
        """
        if not self.client:
            logger.info("❌ Cannot place order - no CDP client")
            return None
        
        try:
            logger.info(f"\n🚀 PLACING REAL SELL ORDER:")
            logger.info(f"   Symbol: {symbol}")
            logger.info(f"   Amount: {crypto_amount} {symbol}")
            logger.info("   THIS IS REAL MONEY!")
            
            # Get the crypto account
            accounts = self.client.get_accounts()
            crypto_account = None
            
            for account in accounts.data:
                if account.balance.currency == symbol:
                    crypto_account = account
                    break
            
            if not crypto_account:
                logger.info(f"❌ No {symbol} account found")
                return None
            
            # Check sufficient balance
            available_crypto = float(crypto_account.balance.amount)
            if available_crypto < crypto_amount:
                logger.info(f"❌ Insufficient {symbol} balance: {available_crypto} < {crypto_amount}")
                return None
            
            # Place sell order
            sell_order = crypto_account.sell(
                amount=str(crypto_amount),
                currency='EUR'
            )
            
            logger.info(f"✅ REAL SELL ORDER PLACED!")
            logger.info(f"   Order ID: {sell_order.id}")
            logger.info(f"   Status: {sell_order.status}")
            logger.info(f"   Amount: {crypto_amount} {symbol}")
            
            return {
                'order_id': sell_order.id,
                'status': sell_order.status,
                'amount_crypto': crypto_amount,
                'symbol': symbol,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.info(f"❌ Failed to place sell order: {e}")
            return None
    
    async def test_small_order(self):
        """Test with a very small order to verify everything works"""
        logger.info("\n🧪 TESTING SMALL REAL ORDER")
        logger.info("=" * 30)
        
        # Get balances first
        balances = await self.get_real_balances()
        
        if 'EUR' not in balances or balances['EUR'] < 1.0:
            logger.info("❌ Need at least €1 EUR balance for testing")
            return False
        
        # Test small buy order (€0.50)
        test_amount = 0.5
        result = await self.place_real_buy_order('BTC', test_amount)
        
        if result:
            logger.info("✅ Test order successful!")
            logger.info("   Check your Coinbase account - you should see this order")
            return True
        else:
            logger.info("❌ Test order failed")
            return False

async def main():
    """Test real CDP trading"""
    logger.info("🔥 REAL CDP TRADER TEST")
    logger.info("=" * 30)
    logger.info("⚠️  WARNING: THIS USES REAL MONEY")
    logger.info("⚠️  ORDERS WILL APPEAR IN YOUR COINBASE ACCOUNT")
    logger.info("=" * 30)
    
    API_KEY = os.getenv("CDP_API_KEY_NAME", "")
    
    # Create real trader
    trader = RealCDPTrader(API_KEY)
    
    if trader.client:
        # Test account access
        balances = await trader.get_real_balances()
        
        if balances:
            logger.info("\n✅ Account access successful!")
            
            # Ask user if they want to test a small order
            response = input("\n🤔 Test with small €0.50 order? (yes/no): ")
            
            if response.lower() in ['yes', 'y']:
                success = await trader.test_small_order()
                
                if success:
                    logger.info("\n✅ SUCCESS! Check your Coinbase account for the order.")
                    logger.info("   This proves real trading is working.")
                    logger.info("   You can now scale up to larger amounts.")
                else:
                    logger.info("\n❌ Test failed. Check API credentials and account setup.")
            else:
                logger.info("🔍 Skipping test order. Real trading setup is ready.")
        else:
            logger.info("\n❌ Cannot access account balances")
            logger.info("🔍 Check API credentials and permissions")
    else:
        logger.info("\n❌ Cannot initialize CDP client")
        logger.info("🔧 Install: pip install coinbase")

if __name__ == "__main__":
    asyncio.run(main()) 