#!/usr/bin/env python3
"""
CDP JWT AUTHENTICATION TEST
===========================
Proper Coinbase Developer Platform API test with JWT authentication
"""

import json
import time
import jwt
import requests
from datetime import datetime

class CDPJWTTest:
    """CDP API test with proper JWT authentication"""
    
    def __init__(self, api_key_id: str, private_key: str):
        self.api_key_id = api_key_id
        self.private_key = private_key
        self.base_url = "https://api.coinbase.com"
        
    def generate_jwt_token(self, request_method: str, request_path: str) -> str:
        """Generate JWT token for CDP API"""
        
        # JWT payload for CDP
        payload = {
            'sub': self.api_key_id,
            'iss': "cdp",
            'nbf': int(time.time()),
            'exp': int(time.time()) + 120,  # 2 minutes expiry
            'aud': ["retail_rest_api_proxy"],
            'uri': f"{request_method} {request_path}"
        }
        
        # Create JWT token with ES256 algorithm
        try:
            token = jwt.encode(
                payload,
                self.private_key,
                algorithm='ES256',
                headers={'kid': self.api_key_id}
            )
            return token
        except Exception as e:
            print(f"JWT generation error: {e}")
            raise
    
    def make_authenticated_request(self, method: str, endpoint: str) -> dict:
        """Make authenticated request to CDP API"""
        
        request_path = endpoint
        
        try:
            # Generate JWT token
            jwt_token = self.generate_jwt_token(method, request_path)
            
            # Headers
            headers = {
                'Authorization': f'Bearer {jwt_token}',
                'Content-Type': 'application/json'
            }
            
            # Make request
            url = self.base_url + endpoint
            response = requests.request(method, url, headers=headers, timeout=10)
            
            print(f"Request: {method} {url}")
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error: {response.text}")
                return {'error': response.status_code, 'message': response.text}
                
        except Exception as e:
            print(f"Request failed: {e}")
            return {'error': 'request_failed', 'message': str(e)}
    
    def test_connection(self):
        """Test CDP API connection"""
        print("🔗 Testing CDP API connection with JWT...")
        
        # Try different endpoints
        endpoints = [
            "/api/v3/brokerage/accounts",
            "/v2/accounts",
            "/api/v3/brokerage/portfolios"
        ]
        
        for endpoint in endpoints:
            print(f"\nTrying endpoint: {endpoint}")
            result = self.make_authenticated_request('GET', endpoint)
            
            if 'error' not in result:
                print("✅ Connection successful!")
                return True, result
        
        print("❌ All endpoints failed")
        return False, None
    
    def check_accounts(self):
        """Check account balances"""
        print("\n💰 Checking account balances...")
        
        # Try to get accounts
        result = self.make_authenticated_request('GET', '/api/v3/brokerage/accounts')
        
        if 'error' in result:
            print(f"❌ Failed to get accounts: {result['message']}")
            return 0
        
        # Parse accounts
        accounts = result.get('accounts', [])
        total_usd = 0
        
        print(f"\n📊 Found {len(accounts)} accounts:")
        print("-" * 40)
        
        for account in accounts:
            currency = account.get('currency', 'UNKNOWN')
            available = float(account.get('available_balance', {}).get('value', 0))
            
            if available > 0:
                print(f"{currency}: {available:.8f}")
                
                if currency == 'USD':
                    total_usd += available
        
        print("-" * 40)
        print(f"💰 Total USD: ${total_usd:.2f}")
        
        # Trading readiness
        if total_usd >= 100:
            print("✅ READY FOR AUTONOMOUS TRADING")
        elif total_usd >= 25:
            print("⚠️ LIMITED TRADING POSSIBLE")
        else:
            print("❌ INSUFFICIENT FUNDS FOR TRADING")
        
        return total_usd

def main():
    """Main CDP JWT test"""
    
    print("CDP JWT AUTHENTICATION TEST")
    print("=" * 50)
    
    # Load credentials
    try:
        with open('Todelete alater/cdp_api_key.json', 'r') as f:
            credentials = json.load(f)
        
        api_key_id = credentials['id']
        private_key = credentials['privateKey']
        
        print(f"✅ Loaded CDP credentials")
        print(f"   API Key ID: {api_key_id[:8]}...{api_key_id[-8:]}")
        
    except Exception as e:
        print(f"❌ Could not load credentials: {e}")
        return
    
    # Initialize CDP test
    try:
        cdp_test = CDPJWTTest(api_key_id, private_key)
        
        # Test connection
        success, data = cdp_test.test_connection()
        
        if success:
            # Check accounts and funds
            total_funds = cdp_test.check_accounts()
            
            print(f"\n🏆 CDP JWT TEST RESULTS")
            print("=" * 50)
            print(f"Authentication: ✅ SUCCESS")
            print(f"Total USD Funds: ${total_funds:.2f}")
            
            if total_funds >= 100:
                print(f"Trading Status: ✅ READY")
                print(f"Kimera Integration: ✅ POSSIBLE")
            else:
                print(f"Trading Status: ⚠️ LIMITED")
                print(f"Recommendation: Add more funds")
            
            # Save results
            results = {
                'timestamp': datetime.now().isoformat(),
                'test_type': 'cdp_jwt',
                'authentication': 'success',
                'total_usd_funds': total_funds,
                'trading_ready': total_funds >= 100,
                'api_key_preview': f"{api_key_id[:8]}...{api_key_id[-8:]}"
            }
            
            filename = f"cdp_jwt_test_{int(time.time())}.json"
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"📊 Results saved: {filename}")
            
        else:
            print(f"\n❌ CDP JWT TEST FAILED")
            print("This could be due to:")
            print("   - Incorrect private key format")
            print("   - API permissions not enabled")
            print("   - Different authentication method required")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 