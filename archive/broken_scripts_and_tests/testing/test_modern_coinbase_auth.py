#!/usr/bin/env python3
"""
TEST MODERN COINBASE AUTHENTICATION
===================================

Simple test to verify modern Coinbase Advanced Trade API
- No passphrase required (JWT authentication)
- Tests connection and basic functionality
"""

import time
import jwt
import requests
import json
from typing import Dict

class ModernCoinbaseAPITest:
    """Test class for modern Coinbase API authentication"""
    
    def __init__(self, api_key: str, api_secret: str, sandbox: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        
        # Use sandbox by default for testing
        if sandbox:
            self.base_url = "https://api.sandbox.coinbase.com"
            print("🧪 Testing with Coinbase SANDBOX (safe)")
        else:
            self.base_url = "https://api.coinbase.com"
            print("💰 Testing with Coinbase PRODUCTION")
        
        self.session = requests.Session()
    
    def generate_jwt_token(self, method: str, path: str) -> str:
        """Generate JWT token for modern authentication"""
        try:
            # Modern JWT payload (no passphrase needed)
            payload = {
                'sub': self.api_key,
                'iss': "coinbase-cloud",
                'nbf': int(time.time()),
                'exp': int(time.time()) + 120,  # 2 minutes expiry
                'aud': ["public_websocket_api"],
                'uri': f"{method} {self.base_url}{path}"
            }
            
            # Generate JWT token using API secret
            token = jwt.encode(payload, self.api_secret, algorithm='ES256')
            return token
            
        except Exception as e:
            print(f"❌ JWT generation error: {e}")
            return ""
    
    def make_authenticated_request(self, method: str, path: str) -> Dict:
        """Make authenticated request using modern JWT method"""
        try:
            # Generate JWT token
            jwt_token = self.generate_jwt_token(method, path)
            
            if not jwt_token:
                return {'error': 'Failed to generate JWT token'}
            
            # Modern authentication headers (no passphrase)
            headers = {
                'Authorization': f'Bearer {jwt_token}',
                'Content-Type': 'application/json',
                'User-Agent': 'Kimera-Modern-Auth-Test/1.0'
            }
            
            url = self.base_url + path
            
            print(f"🔗 Making {method} request to: {path}")
            print(f"🔑 Using JWT authentication (no passphrase)")
            
            response = self.session.get(url, headers=headers, timeout=30)
            
            print(f"📡 Response status: {response.status_code}")
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ Error response: {response.text}")
                return {'error': f"HTTP {response.status_code}: {response.text}"}
                
        except Exception as e:
            print(f"❌ Request error: {e}")
            return {'error': str(e)}
    
    def test_connection(self) -> bool:
        """Test basic API connection"""
        print("\n🔧 Testing Modern API Connection...")
        print("-" * 40)
        
        result = self.make_authenticated_request('GET', '/api/v3/brokerage/accounts')
        
        if 'error' in result:
            print(f"❌ Connection test failed: {result['error']}")
            return False
        
        print("✅ Modern Coinbase API connection successful!")
        print(f"📊 Found {len(result.get('accounts', []))} accounts")
        
        return True
    
    def test_account_info(self) -> bool:
        """Test account information retrieval"""
        print("\n💰 Testing Account Information...")
        print("-" * 40)
        
        result = self.make_authenticated_request('GET', '/api/v3/brokerage/accounts')
        
        if 'error' in result:
            print(f"❌ Account info test failed: {result['error']}")
            return False
        
        accounts = result.get('accounts', [])
        
        print(f"✅ Successfully retrieved {len(accounts)} accounts")
        
        for account in accounts[:3]:  # Show first 3 accounts
            currency = account.get('currency', 'Unknown')
            balance = account.get('available_balance', {}).get('value', '0')
            print(f"   💼 {currency}: {balance}")
        
        return True
    
    def test_market_data(self) -> bool:
        """Test market data retrieval"""
        print("\n📈 Testing Market Data...")
        print("-" * 40)
        
        # Test BTC-USD ticker
        result = self.make_authenticated_request('GET', '/api/v3/brokerage/products/BTC-USD/ticker')
        
        if 'error' in result:
            print(f"❌ Market data test failed: {result['error']}")
            return False
        
        if 'price' in result:
            price = result['price']
            volume = result.get('volume_24h', 'N/A')
            print(f"✅ BTC-USD Price: ${float(price):,.2f}")
            print(f"   📊 24h Volume: {volume}")
        else:
            print("⚠️ Price data not available in response")
        
        return True

def run_authentication_test():
    """Run comprehensive modern authentication test"""
    
    print("="*60)
    print("🧪 MODERN COINBASE API AUTHENTICATION TEST")
    print("="*60)
    print("🔐 JWT Authentication (No Passphrase Required)")
    print("⚡ Testing Modern Coinbase Advanced Trade API")
    print("="*60)
    
    # Your API credentials
    API_KEY = "9268de76-b5f4-4683-b593-327fb2c19503"
    API_SECRET = "BiCUFOxZ4J4Fi8F6mcyzuzreXaGZeBLHxr7q8Puo6VHcSEgyqJ6mIx29RbbAJGAjq6SHBt5K4PieiymRhEWVHw=="
    
    print(f"\n🔑 API Credentials:")
    print(f"   Key: {API_KEY[:8]}...{API_KEY[-8:]}")
    print(f"   Secret: {'*' * 20}")
    print(f"   Passphrase: NOT REQUIRED (Modern API)")
    
    # Initialize test client
    test_client = ModernCoinbaseAPITest(API_KEY, API_SECRET, sandbox=True)
    
    # Run tests
    tests_passed = 0
    total_tests = 3
    
    print(f"\n🚀 Running {total_tests} Authentication Tests...")
    
    # Test 1: Basic connection
    if test_client.test_connection():
        tests_passed += 1
    
    # Test 2: Account information
    if test_client.test_account_info():
        tests_passed += 1
    
    # Test 3: Market data
    if test_client.test_market_data():
        tests_passed += 1
    
    # Results summary
    print(f"\n📊 TEST RESULTS SUMMARY")
    print("="*40)
    print(f"✅ Tests Passed: {tests_passed}/{total_tests}")
    print(f"❌ Tests Failed: {total_tests - tests_passed}/{total_tests}")
    print(f"📈 Success Rate: {(tests_passed/total_tests)*100:.1f}%")
    
    if tests_passed == total_tests:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"✅ Modern Coinbase API authentication working perfectly")
        print(f"🔐 JWT authentication confirmed (no passphrase needed)")
        print(f"🚀 Ready for live trading with Kimera system")
    else:
        print(f"\n⚠️  Some tests failed")
        print(f"🔧 Check API credentials and network connection")
        
        if tests_passed > 0:
            print(f"💡 Partial success indicates authentication method is correct")
    
    print(f"\n🏁 Authentication test completed")

if __name__ == "__main__":
    try:
        run_authentication_test()
    except KeyboardInterrupt:
        print(f"\n\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test error: {e}") 