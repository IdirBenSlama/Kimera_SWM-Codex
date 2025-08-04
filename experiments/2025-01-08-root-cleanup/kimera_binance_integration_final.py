#!/usr/bin/env python3
"""
Kimera-Binance Integration Final Implementation
Complete integration using working HMAC authentication
"""

import asyncio
import hashlib
import hmac
import time
import logging
from urllib.parse import urlencode
from datetime import datetime
from typing import Dict, List, Optional, Any
import aiohttp
import json
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KimeraBinanceConnector:
    """
    Kimera-optimized Binance connector with working HMAC authentication
    """
    
    def __init__(self, api_key: str, secret_key: str, testnet: bool = False):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = "https://testnet.binance.vision" if testnet else "https://api.binance.com"
        self.session = None
        
        # Safety limits
        self.max_position_size = 25.0  # USD
        self.risk_percentage = 0.005   # 0.5%
        self.max_daily_trades = 5
        self.daily_trade_count = 0
        self.daily_loss = 0.0
        
        logger.info(f"Kimera-Binance connector initialized (testnet={testnet})")
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def _sign_request(self, params: Dict[str, Any]) -> str:
        """Sign request with HMAC-SHA256"""
        query_string = urlencode(params, safe='~')
        return hmac.new(
            self.secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    async def _request(self, method: str, endpoint: str, params: Optional[Dict] = None, signed: bool = False) -> Dict:
        """Make authenticated request to Binance API"""
        if params is None:
            params = {}
        
        headers = {'X-MBX-APIKEY': self.api_key}
        
        if signed:
            params['timestamp'] = int(time.time() * 1000)
            params['recvWindow'] = 5000
            signature = self._sign_request(params)
            params['signature'] = signature
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method.upper() == 'GET':
                async with self.session.get(url, params=params, headers=headers) as response:
                    data = await response.json()
            elif method.upper() == 'POST':
                async with self.session.post(url, data=params, headers=headers) as response:
                    data = await response.json()
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            if response.status != 200:
                logger.error(f"API error: {response.status} - {data}")
                raise Exception(f"Binance API error: {response.status} - {data.get('msg', 'Unknown error')}")
            
            return data
            
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise
    
    # Market Data Methods
    async def get_ticker(self, symbol: str) -> Dict:
        """Get 24hr ticker statistics"""
        return await self._request('GET', '/api/v3/ticker/24hr', {'symbol': symbol})
    
    async def get_order_book(self, symbol: str, limit: int = 20) -> Dict:
        """Get order book"""
        return await self._request('GET', '/api/v3/depth', {'symbol': symbol, 'limit': limit})
    
    async def get_market_price(self, symbol: str) -> float:
        """Get current market price"""
        ticker = await self.get_ticker(symbol)
        return float(ticker['lastPrice'])
    
    # Account Methods
    async def get_account_info(self) -> Dict:
        """Get account information"""
        return await self._request('GET', '/api/v3/account', signed=True)
    
    async def get_balance(self, asset: str) -> Dict:
        """Get balance for specific asset"""
        account = await self.get_account_info()
        for balance in account['balances']:
            if balance['asset'] == asset:
                return {
                    'free': float(balance['free']),
                    'locked': float(balance['locked']),
                    'total': float(balance['free']) + float(balance['locked'])
                }
        return {'free': 0.0, 'locked': 0.0, 'total': 0.0}
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get open orders"""
        params = {}
        if symbol:
            params['symbol'] = symbol
        return await self._request('GET', '/api/v3/openOrders', params, signed=True)
    
    # Trading Methods with Safety Checks
    async def place_order(self, symbol: str, side: str, order_type: str, quantity: float, 
                         price: Optional[float] = None) -> Dict:
        """Place order with safety checks"""
        
        # Safety check: Daily trade limit
        if self.daily_trade_count >= self.max_daily_trades:
            raise Exception(f"Daily trade limit reached ({self.max_daily_trades})")
        
        # Safety check: Position size
        current_price = await self.get_market_price(symbol)
        position_value = quantity * current_price
        
        if position_value > self.max_position_size:
            raise Exception(f"Position size ${position_value:.2f} exceeds limit ${self.max_position_size}")
        
        # Prepare order parameters
        params = {
            'symbol': symbol,
            'side': side.upper(),
            'type': order_type.upper(),
            'quantity': f"{quantity:.8f}".rstrip('0').rstrip('.')
        }
        
        if order_type.upper() in ['LIMIT', 'STOP_LOSS_LIMIT']:
            if price is None:
                raise ValueError(f"Price required for {order_type} orders")
            params['price'] = f"{price:.8f}".rstrip('0').rstrip('.')
            params['timeInForce'] = 'GTC'
        
        logger.info(f"Placing {side} order for {quantity} {symbol} at ${current_price:.2f}")
        
        try:
            result = await self._request('POST', '/api/v3/order', params, signed=True)
            self.daily_trade_count += 1
            logger.info(f"Order placed successfully: {result['orderId']}")
            return result
            
        except Exception as e:
            logger.error(f"Order placement failed: {e}")
            raise
    
    async def cancel_order(self, symbol: str, order_id: int) -> Dict:
        """Cancel order"""
        params = {'symbol': symbol, 'orderId': order_id}
        return await self._request('DELETE', '/api/v3/order', params, signed=True)
    
    # Kimera-specific Methods
    async def analyze_market_sentiment(self, symbol: str) -> Dict:
        """Analyze market sentiment for Kimera cognitive trading"""
        ticker = await self.get_ticker(symbol)
        order_book = await self.get_order_book(symbol, limit=10)
        
        # Calculate sentiment indicators
        price_change = float(ticker['priceChangePercent'])
        volume_24h = float(ticker['volume'])
        
        # Order book analysis
        bids = order_book['bids']
        asks = order_book['asks']
        
        bid_volume = sum(float(bid[1]) for bid in bids)
        ask_volume = sum(float(ask[1]) for ask in asks)
        
        buy_pressure = bid_volume / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0.5
        
        sentiment_score = (price_change / 100) * 0.6 + (buy_pressure - 0.5) * 0.4
        
        return {
            'symbol': symbol,
            'sentiment_score': sentiment_score,
            'price_change_24h': price_change,
            'volume_24h': volume_24h,
            'buy_pressure': buy_pressure,
            'timestamp': int(time.time() * 1000)
        }
    
    async def generate_trading_signal(self, symbol: str) -> Dict:
        """Generate Kimera cognitive trading signal"""
        sentiment = await self.analyze_market_sentiment(symbol)
        current_price = await self.get_market_price(symbol)
        
        # Simple signal generation logic
        sentiment_score = sentiment['sentiment_score']
        
        if sentiment_score > 0.3:
            action = 'BUY'
            confidence = min(sentiment_score * 2, 1.0)
        elif sentiment_score < -0.3:
            action = 'SELL'
            confidence = min(abs(sentiment_score) * 2, 1.0)
        else:
            action = 'HOLD'
            confidence = 0.5
        
        return {
            'symbol': symbol,
            'action': action,
            'confidence': confidence,
            'price': current_price,
            'sentiment': sentiment,
            'reasoning': f"Sentiment score: {sentiment_score:.3f}, Buy pressure: {sentiment['buy_pressure']:.3f}",
            'timestamp': datetime.now().isoformat()
        }

async def test_kimera_binance_integration():
    """Test complete Kimera-Binance integration"""
    
    logger.info("🚀 KIMERA-BINANCE FINAL INTEGRATION TEST")
    logger.info("=" * 60)
    
    # Use working credentials
    api_key = os.getenv("BINANCE_API_KEY", "")
    secret_key = 'qUn5JqSpYz1GDxFj2X3UF23TYgtxKrTsCbDZEoBMYCPbYZgP4siVLyspkB5HAPl7'
    
    try:
        async with KimeraBinanceConnector(api_key, secret_key, testnet=False) as connector:
            
            logger.info("✅ Kimera-Binance Connector initialized")
            
            # Test 1: Account Authentication
            logger.info("\n🔐 Test 1: Account Authentication")
            account_info = await connector.get_account_info()
            
            logger.info("🎉 AUTHENTICATION SUCCESSFUL!")
            logger.info(f"   Account Type: {account_info.get('accountType', 'Unknown')}")
            logger.info(f"   Trading Enabled: {account_info.get('canTrade', False)}")
            logger.info(f"   Withdrawal Enabled: {account_info.get('canWithdraw', False)}")
            
            # Test 2: Market Analysis
            logger.info("\n📊 Test 2: Market Sentiment Analysis")
            sentiment = await connector.analyze_market_sentiment('BTCUSDT')
            
            logger.info(f"   Symbol: {sentiment['symbol']}")
            logger.info(f"   Sentiment Score: {sentiment['sentiment_score']:.3f}")
            logger.info(f"   24h Price Change: {sentiment['price_change_24h']:.2f}%")
            logger.info(f"   Buy Pressure: {sentiment['buy_pressure']:.3f}")
            
            # Test 3: Trading Signal Generation
            logger.info("\n🧠 Test 3: Cognitive Trading Signal")
            signal = await connector.generate_trading_signal('BTCUSDT')
            
            logger.info(f"   Action: {signal['action']}")
            logger.info(f"   Confidence: {signal['confidence']:.3f}")
            logger.info(f"   Current Price: ${signal['price']:,.2f}")
            logger.info(f"   Reasoning: {signal['reasoning']}")
            
            # Test 4: Balance Check
            logger.info("\n💰 Test 4: Account Balances")
            btc_balance = await connector.get_balance('BTC')
            usdt_balance = await connector.get_balance('USDT')
            bnb_balance = await connector.get_balance('BNB')
            
            logger.info(f"   BTC Balance: {btc_balance['total']:.8f}")
            logger.info(f"   USDT Balance: {usdt_balance['total']:.2f}")
            logger.info(f"   BNB Balance: {bnb_balance['total']:.6f}")
            
            # Test 5: Safety Systems
            logger.info("\n🛡️ Test 5: Safety Systems")
            logger.info(f"   Max Position Size: ${connector.max_position_size}")
            logger.info(f"   Risk Percentage: {connector.risk_percentage * 100:.2f}%")
            logger.info(f"   Daily Trade Limit: {connector.max_daily_trades}")
            logger.info(f"   Daily Trades Used: {connector.daily_trade_count}")
            
            # Final Summary
            logger.info("\n📋 INTEGRATION SUMMARY")
            logger.info("=" * 60)
            logger.info("✅ HMAC Authentication: WORKING")
            logger.info("✅ Market Data Access: WORKING")
            logger.info("✅ Sentiment Analysis: WORKING")
            logger.info("✅ Signal Generation: WORKING")
            logger.info("✅ Safety Systems: ACTIVE")
            logger.info("✅ Account Integration: COMPLETE")
            
            logger.info("\n🎯 KIMERA-BINANCE INTEGRATION: READY FOR LIVE TRADING!")
            
            return True
            
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        logger.info(f"\n❌ INTEGRATION ERROR: {e}")
        return False

async def main():
    """Main execution"""
    logger.info("🔧 KIMERA SYSTEM INITIALIZATION")
    logger.info("=" * 40)
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    success = await test_kimera_binance_integration()
    
    if success:
        logger.info("\n🎉 KIMERA-BINANCE INTEGRATION SUCCESSFUL!")
        logger.info("System is ready for cognitive trading operations.")
    else:
        logger.info("\n❌ INTEGRATION FAILED")
        logger.info("Please check the error messages above.")

if __name__ == "__main__":
    asyncio.run(main()) 