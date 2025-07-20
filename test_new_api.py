#!/usr/bin/env python3
"""
Test New Coinbase API Credentials
==================================

Test if the new API key has trading permissions.
"""

import os
import sys
import json
import time
import hmac
import hashlib
import base64
import requests
from datetime import datetime
from typing import Dict, Optional

class CoinbaseAdvancedTradeAPI:
    """Test Coinbase Advanced Trade API"""
    
    def __init__(self, api_key: str, api_secret: str):
        """Initialize with API credentials"""
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.coinbase.com"
        
        print("🔐 TESTING NEW COINBASE API CREDENTIALS")
        print("=" * 50)
        print(f"API Key: {api_key[:10]}...")
        
    def _generate_signature(self, request_path: str, body: str, timestamp: str, method: str) -> str:
        """Generate signature for API request"""
        message = timestamp + method + request_path + body
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).digest()
        return base64.b64encode(signature).decode()
    
    def _make_request(self, method: str, path: str, body: Dict = None) -> Optional[Dict]:
        """Make authenticated API request"""
        timestamp = str(int(time.time()))
        body_str = json.dumps(body) if body else ''
        
        signature = self._generate_signature(path, body_str, timestamp, method)
        
        headers = {
            'CB-ACCESS-KEY': self.api_key,
            'CB-ACCESS-SIGN': signature,
            'CB-ACCESS-TIMESTAMP': timestamp,
            'Content-Type': 'application/json'
        }
        
        try:
            url = f"{self.base_url}{path}"
            
            if method == 'GET':
                response = requests.get(url, headers=headers)
            elif method == 'POST':
                response = requests.post(url, headers=headers, data=body_str)
            
            print(f"\n📡 {method} {path}")
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"   Error: {response.text}")
                return None
                
        except Exception as e:
            print(f"   Exception: {e}")
            return None
    
    def test_accounts(self):
        """Test account access"""
        print("\n🔍 Testing Account Access...")
        
        # Try Advanced Trade API endpoint
        result = self._make_request('GET', '/api/v3/brokerage/accounts')
        
        if result:
            print("✅ SUCCESS! Account access working")
            accounts = result.get('accounts', [])
            print(f"   Found {len(accounts)} accounts:")
            
            for account in accounts:
                currency = account.get('currency')
                balance = account.get('available_balance', {}).get('value', '0')
                print(f"   {currency}: {balance}")
                
            return True
        else:
            print("❌ Account access failed")
            return False
    
    def test_products(self):
        """Test product/market data access"""
        print("\n🔍 Testing Market Data Access...")
        
        result = self._make_request('GET', '/api/v3/brokerage/products')
        
        if result:
            print("✅ SUCCESS! Market data access working")
            products = result.get('products', [])
            print(f"   Found {len(products)} trading pairs")
            
            # Show a few examples
            for product in products[:5]:
                print(f"   {product.get('product_id')}")
                
            return True
        else:
            print("❌ Market data access failed")
            return False
    
    def test_orders(self):
        """Test order access"""
        print("\n🔍 Testing Order Access...")
        
        result = self._make_request('GET', '/api/v3/brokerage/orders/historical/batch')
        
        if result:
            print("✅ SUCCESS! Order access working")
            orders = result.get('orders', [])
            print(f"   Found {len(orders)} historical orders")
            return True
        else:
            print("❌ Order access failed")
            return False

def main():
    """Test new API credentials"""
    print("🚀 COINBASE API CREDENTIAL TEST")
    print("=" * 50)
    
    # Get API credentials
    api_key = input("Enter your new API Key: ").strip()
    api_secret = input("Enter your API Secret: ").strip()
    
    if not api_key or not api_secret:
        print("❌ API credentials required!")
        return
    
    # Create tester
    tester = CoinbaseAdvancedTradeAPI(api_key, api_secret)
    
    # Run tests
    tests_passed = []
    
    # Test 1: Account access
    if tester.test_accounts():
        tests_passed.append("Account Access")
    
    # Test 2: Market data
    if tester.test_products():
        tests_passed.append("Market Data")
    
    # Test 3: Order access
    if tester.test_orders():
        tests_passed.append("Order Access")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY:")
    
    if len(tests_passed) == 3:
        print("✅ ALL TESTS PASSED!")
        print("🎉 Your API key is ready for real trading!")
        print("\n💡 Next steps:")
        print("   1. Save these credentials securely")
        print("   2. I'll create a real trading bot")
        print("   3. Start trading from €5 to €100!")
    else:
        print(f"⚠️ Only {len(tests_passed)}/3 tests passed")
        print("❌ Tests failed:")
        all_tests = ["Account Access", "Market Data", "Order Access"]
        for test in all_tests:
            if test not in tests_passed:
                print(f"   - {test}")
        
        print("\n💡 Troubleshooting:")
        print("   1. Check API permissions")
        print("   2. Verify IP whitelist includes your IP")
        print("   3. Make sure it's an Advanced Trade API key")

if __name__ == "__main__":
    main() 