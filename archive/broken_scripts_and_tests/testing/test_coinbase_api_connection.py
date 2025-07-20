#!/usr/bin/env python3
"""
COINBASE ADVANCED TRADING API CONNECTION TEST
===========================================

Quick test to verify API connection and check account balance
"""

import json
import time
import hmac
import hashlib
import base64
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CoinbaseAdvancedAPITest:
    """Simple API test client"""
    
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.coinbase.com/api/v3/brokerage"
        
        logger.info("🔑 Testing Coinbase Advanced Trading API")
        logger.info(f"🔑 API Key: {api_key[:8]}...{api_key[-8:]}")
    
    def _generate_signature(self, timestamp: str, method: str, path: str, body: str = '') -> str:
        """Generate CB-ACCESS-SIGN header"""
        message = timestamp + method + path + body
        # The API secret needs to be base64-decoded before being used as the HMAC key
        secret_decoded = base64.b64decode(self.api_secret)
        signature = hmac.new(
            secret_decoded,
            message.encode('utf-8'),
            hashlib.sha256
        ).digest()
        return base64.b64encode(signature).decode('utf-8')
    
    def _make_request(self, method: str, path: str, params: dict = None) -> dict:
        """Make authenticated API request"""
        timestamp = str(int(time.time()))
        body = ''
        
        headers = {
            'CB-ACCESS-KEY': self.api_key,
            'CB-ACCESS-SIGN': self._generate_signature(timestamp, method, path, body),
            'CB-ACCESS-TIMESTAMP': timestamp,
            'Content-Type': 'application/json'
        }
        
        url = self.base_url + path
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            logger.info(f"📡 API Call: {method} {path} -> Status: {response.status_code}")
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"❌ API Error {response.status_code}: {response.text}")
                return {"error": f"HTTP {response.status_code}", "message": response.text}
                
        except Exception as e:
            logger.error(f"❌ Request error: {e}")
            return {"error": "request_failed", "message": str(e)}
    
    def test_connection(self):
        """Test API connection and display results"""
        logger.info("\n🧪 COINBASE ADVANCED TRADING API TEST")
        logger.info("=" * 50)
        
        # Test 1: Get accounts
        logger.info("🔍 Testing account access...")
        accounts = self._make_request('GET', '/accounts')
        
        if 'error' in accounts:
            logger.error(f"❌ Account access failed: {accounts}")
            return False
        
        logger.info("✅ Account access successful!")
        
        # Display account information
        if 'accounts' in accounts:
            logger.info(f"📊 Found {len(accounts['accounts'])} accounts:")
            
            total_usd_balance = 0.0
            for account in accounts['accounts']:
                currency = account.get('currency', 'UNKNOWN')
                balance = float(account.get('available_balance', {}).get('value', 0))
                
                if balance > 0:
                    logger.info(f"   💰 {currency}: {balance:.6f}")
                    
                    if currency == 'USD':
                        total_usd_balance = balance
            
            logger.info(f"💵 Total USD Available: ${total_usd_balance:.6f}")
        
        # Test 2: Get products
        logger.info("\n🔍 Testing product access...")
        products = self._make_request('GET', '/products')
        
        if 'error' in products:
            logger.error(f"❌ Product access failed: {products}")
            return False
        
        logger.info("✅ Product access successful!")
        
        if 'products' in products:
            major_pairs = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'DOGE-USD']
            available_major_pairs = []
            
            for product in products['products']:
                if product['product_id'] in major_pairs and product.get('status') == 'online':
                    available_major_pairs.append(product['product_id'])
            
            logger.info(f"📈 Available major trading pairs: {available_major_pairs}")
        
        # Test 3: Get ticker for BTC-USD
        logger.info("\n🔍 Testing market data access...")
        ticker = self._make_request('GET', '/products/BTC-USD/ticker')
        
        if 'error' in ticker:
            logger.error(f"❌ Market data access failed: {ticker}")
            return False
        
        logger.info("✅ Market data access successful!")
        
        if 'price' in ticker:
            btc_price = float(ticker['price'])
            logger.info(f"₿ BTC-USD Price: ${btc_price:,.2f}")
        
        logger.info("\n🎉 ALL TESTS PASSED!")
        logger.info("✅ API is fully functional and ready for trading")
        logger.info("=" * 50)
        
        return True

def main():
    """Run API connection test"""
    
    # User's actual Coinbase Advanced Trading credentials
    API_KEY = "9268de76-b5f4-4683-b593-327fb2c19503"
    API_SECRET = "BiCUFOxZ4J4Fi8F6mcyzuzreXaGZeBLHxr7q8Puo6VHcSEgyqJ6mIx29RbbAJGAjq6SHBt5K4PieiymRhEWVHw=="
    
    # Create test client
    test_client = CoinbaseAdvancedAPITest(API_KEY, API_SECRET)
    
    # Run tests
    success = test_client.test_connection()
    
    if success:
        print("\n🚀 Ready to launch Kimera live trading!")
        print("💡 Run 'python kimera_coinbase_advanced_integration.py' to start")
    else:
        print("\n❌ API connection issues detected")
        print("💡 Please check your credentials and network connection")

if __name__ == "__main__":
    main() 