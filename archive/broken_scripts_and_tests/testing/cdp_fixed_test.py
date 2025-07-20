#!/usr/bin/env python3
"""
CDP FIXED TEST - Handles base64 private key properly
====================================================
"""

import json
import time
import jwt
import requests
import base64
from datetime import datetime
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.backends import default_backend

class CDPFixedTest:
    """CDP API test with proper private key handling"""
    
    def __init__(self, api_key_id: str, private_key_b64: str):
        self.api_key_id = api_key_id
        self.private_key_b64 = private_key_b64
        self.base_url = "https://api.coinbase.com"
        
        # Convert base64 private key to proper format
        self.private_key = self._prepare_private_key(private_key_b64)
        
    def _prepare_private_key(self, private_key_b64: str) -> str:
        """Convert base64 private key to PEM format"""
        
        try:
            # Method 1: Try direct base64 decode
            key_bytes = base64.b64decode(private_key_b64)
            
            # Try to create EC private key from raw bytes
            if len(key_bytes) == 32:  # secp256r1 private key
                # Create EC private key
                private_value = int.from_bytes(key_bytes, 'big')
                private_key = ec.derive_private_key(private_value, ec.SECP256R1(), default_backend())
                
                # Convert to PEM format
                pem = private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                )
                
                return pem.decode('utf-8')
            
            else:
                # Try as DER format
                private_key = serialization.load_der_private_key(key_bytes, password=None, backend=default_backend())
                pem = private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                )
                return pem.decode('utf-8')
                
        except Exception as e:
            print(f"Private key conversion failed: {e}")
            # Return original for fallback
            return private_key_b64
    
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
            # Try alternative approach
            try:
                # Use raw base64 key
                token = jwt.encode(
                    payload,
                    base64.b64decode(self.private_key_b64),
                    algorithm='ES256',
                    headers={'kid': self.api_key_id}
                )
                return token
            except Exception as e2:
                print(f"Alternative JWT generation also failed: {e2}")
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
                print(f"Error: {response.text[:200]}")
                return {'error': response.status_code, 'message': response.text}
                
        except Exception as e:
            print(f"Request failed: {e}")
            return {'error': 'request_failed', 'message': str(e)}
    
    def test_connection(self):
        """Test CDP API connection"""
        print("🔗 Testing CDP API connection with fixed JWT...")
        
        # Try different endpoints
        endpoints = [
            "/api/v3/brokerage/accounts",
            "/v2/accounts",
            "/api/v3/brokerage/portfolios",
            "/v2/user"
        ]
        
        for endpoint in endpoints:
            print(f"\nTrying endpoint: {endpoint}")
            result = self.make_authenticated_request('GET', endpoint)
            
            if 'error' not in result:
                print("✅ Connection successful!")
                return True, result
        
        print("❌ All endpoints failed")
        return False, None

def main():
    """Main CDP fixed test"""
    
    print("CDP FIXED TEST - Base64 Private Key Handler")
    print("=" * 60)
    
    # Load credentials
    try:
        with open('Todelete alater/cdp_api_key.json', 'r') as f:
            credentials = json.load(f)
        
        api_key_id = credentials['id']
        private_key_b64 = credentials['privateKey']
        
        print(f"✅ Loaded CDP credentials")
        print(f"   API Key ID: {api_key_id[:8]}...{api_key_id[-8:]}")
        print(f"   Private Key: {len(private_key_b64)} characters (base64)")
        
    except Exception as e:
        print(f"❌ Could not load credentials: {e}")
        return
    
    # Initialize CDP test
    try:
        cdp_test = CDPFixedTest(api_key_id, private_key_b64)
        
        # Test connection
        success, data = cdp_test.test_connection()
        
        if success:
            print(f"\n🏆 CDP FIXED TEST SUCCESS!")
            print("=" * 60)
            print(f"✅ Authentication working")
            print(f"✅ API connection established")
            print(f"✅ Ready for fund checking")
            
            # Try to get account info
            if 'accounts' in data:
                accounts = data['accounts']
                print(f"📊 Found {len(accounts)} accounts")
                
                total_usd = 0
                for account in accounts:
                    currency = account.get('currency', 'UNKNOWN')
                    balance = account.get('balance', {}).get('amount', '0')
                    
                    try:
                        balance_float = float(balance)
                        if balance_float > 0:
                            print(f"   {currency}: {balance_float}")
                            
                            if currency == 'USD':
                                total_usd += balance_float
                    except:
                        pass
                
                print(f"💰 Total USD: ${total_usd:.2f}")
                
                if total_usd >= 100:
                    print("🚀 READY FOR KIMERA AUTONOMOUS TRADING!")
                else:
                    print("⚠️ Consider adding more funds for optimal trading")
            
        else:
            print(f"\n❌ CDP FIXED TEST FAILED")
            print("Possible issues:")
            print("   - Private key format still incorrect")
            print("   - API permissions not enabled")
            print("   - Different CDP authentication method")
            print("   - Account setup incomplete")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 