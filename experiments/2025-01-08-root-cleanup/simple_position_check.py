import asyncio
import aiohttp
import hmac
import hashlib
import time
import os
from urllib.parse import urlencode
import logging
logger = logging.getLogger(__name__)

class SimpleBinanceChecker:
    def __init__(self, api_key, secret_key):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = "https://api.binance.com"
    
    def _generate_signature(self, params):
        """Generate HMAC signature"""
        query_string = urlencode(params)
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    async def get_ticker(self, symbol):
        """Get ticker price"""
        url = f"{self.base_url}/api/v3/ticker/price"
        params = {"symbol": symbol}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                return await response.json()
    
    async def get_account(self):
        """Get account information"""
        url = f"{self.base_url}/api/v3/account"
        
        # Add timestamp
        timestamp = int(time.time() * 1000)
        params = {"timestamp": timestamp}
        
        # Generate signature
        signature = self._generate_signature(params)
        params["signature"] = signature
        
        headers = {"X-MBX-APIKEY": self.api_key}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=headers) as response:
                return await response.json()

async def check_position():
    api_key = os.getenv('BINANCE_API_KEY')
    secret_key = os.getenv('BINANCE_SECRET_KEY')
    
    if not api_key or not secret_key:
        logger.info("❌ Missing API credentials")
        return
    
    checker = SimpleBinanceChecker(api_key, secret_key)
    
    try:
        # Get current BTC price
        logger.info("🔍 Checking current Bitcoin price...")
        ticker = await checker.get_ticker('BTCUSDT')
        current_price = float(ticker['price'])
        logger.info(f"Current BTC Price: ${current_price:,.2f}")
        
        # Get account balance
        logger.info("\n🔍 Checking account balance...")
        account = await checker.get_account()
        
        btc_balance = 0.0
        usdt_balance = 0.0
        
        for balance in account['balances']:
            if balance['asset'] == 'BTC':
                btc_balance = float(balance['free'])
            elif balance['asset'] == 'USDT':
                usdt_balance = float(balance['free'])
        
        logger.info(f"BTC Balance: {btc_balance:.8f} BTC")
        logger.info(f"USDT Balance: {usdt_balance:.2f} USDT")
        
        if btc_balance > 0:
            current_value = btc_balance * current_price
            logger.info(f"\n💰 Current BTC Position Value: ${current_value:.2f}")
            
            # Calculate unrealized P&L (entry price from previous trade)
            entry_price = 117416.81  # From the previous trade
            unrealized_pnl = (current_price - entry_price) * btc_balance
            unrealized_pnl_pct = (current_price - entry_price) / entry_price * 100
            
            logger.info(f"Entry Price: ${entry_price:,.2f}")
            logger.info(f"Unrealized P&L: ${unrealized_pnl:.2f} ({unrealized_pnl_pct:+.2f}%)")
            
            if unrealized_pnl > 0:
                logger.info('🟢 Position is currently in PROFIT')
            else:
                logger.info('🔴 Position is currently at LOSS')
        else:
            logger.info("📍 No BTC position found")
            
    except Exception as e:
        logger.info(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(check_position()) 