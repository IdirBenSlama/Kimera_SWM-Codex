#!/usr/bin/env python3
"""
Test Coinbase CDP SDK / On-chain Operations
"""

import json
import time
import requests
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend

def test_cdp_sdk():
    """Test if credentials are for CDP SDK/on-chain operations"""
    print("🔍 TESTING CDP SDK / ON-CHAIN OPERATIONS")
    print("=" * 50)
    
    # Your credentials
    ORGANIZATION_ID = "d5c46584-dd70-4be9-a71a-1e5e1b7a7ea3"
    API_KEY_ID = "dfe10f85-ed6c-4e75-a880-5db488c44f73"
    PRIVATE_KEY_PEM = """-----BEGIN EC PRIVATE KEY-----
MHcCAQEEIMyN35R0MUQewQ27M8ljhrGsQgRtIl1I3VMCZMucX4UIoAoGCCqGSM49
AwEHoUQDQgAERK6BZscG6p5nLQzIhPkUjXqIT9m/mw/S81U9di/u2BKRvujr4fUL
k+1M3dZ5l6SjNp2naYaa7oXuQQUm8UsFFA==
-----END EC PRIVATE KEY-----"""
    
    # Load private key
    private_key = serialization.load_pem_private_key(
        PRIVATE_KEY_PEM.encode(),
        password=None,
        backend=default_backend()
    )
    
    def create_jwt_token(base_url: str, request_path: str, method: str = 'GET') -> str:
        """Create JWT token for API call"""
        import jwt
        
        uri = f"{method} {base_url}{request_path}"
        
        payload = {
            'sub': API_KEY_ID,
            'iss': "cdp",
            'nbf': int(time.time()),
            'exp': int(time.time()) + 120,
            'uri': uri
        }
        
        return jwt.encode(
            payload,
            private_key,
            algorithm='ES256',
            headers={'kid': API_KEY_ID}
        )
    
    # Test CDP SDK/on-chain endpoints
    base_urls = [
        "https://api.developer.coinbase.com",
        "https://api.coinbase.com", 
        "https://api.cdp.coinbase.com"
    ]
    
    # SDK/on-chain specific endpoints
    sdk_endpoints = [
        # Wallet operations
        (f"/v1/wallets", "Wallets v1"),
        (f"/v1/users/{API_KEY_ID}/wallets", "User Wallets"),
        (f"/v1/networks", "Networks"),
        
        # On-chain operations
        (f"/onchain/v1/wallets", "On-chain Wallets"),
        (f"/onchain/v1/addresses", "On-chain Addresses"),
        (f"/onchain/v1/transactions", "On-chain Transactions"),
        
        # CDP specific endpoints
        (f"/cdp/v1/wallets", "CDP Wallets"),
        (f"/cdp/v1/organizations/{ORGANIZATION_ID}", "CDP Organization"),
        (f"/cdp/v1/projects", "CDP Projects"),
        
        # Organization endpoints
        (f"/organizations/{ORGANIZATION_ID}/wallets", "Org Wallets"),
        (f"/organizations/{ORGANIZATION_ID}/projects", "Org Projects"),
        
        # API key management
        (f"/v1/api_keys", "API Keys"),
        (f"/v1/users/self", "Current User"),
        
        # Other possible paths
        (f"/wallet/v1", "Wallet API v1"),
        (f"/platform/v1/wallets", "Platform Wallets v1"),
    ]
    
    successful_endpoints = []
    
    for base_url in base_urls:
        print(f"\n🔍 TESTING BASE URL: {base_url}")
        print("=" * 40)
        
        for path, description in sdk_endpoints:
            print(f"\n   🔍 {description} ({path})")
            
            try:
                token = create_jwt_token(base_url, path)
                headers = {
                    'Authorization': f'Bearer {token}',
                    'Content-Type': 'application/json'
                }
                
                response = requests.get(
                    f"{base_url}{path}",
                    headers=headers,
                    timeout=10
                )
                
                print(f"      Status: {response.status_code}")
                
                if response.status_code == 200:
                    print("      ✅ SUCCESS!")
                    try:
                        data = response.json()
                        print(f"      Data: {json.dumps(data, indent=2)[:200]}...")
                        successful_endpoints.append((base_url, path, description, data))
                    except Exception as e:
                        logger.error(f"Error in test_cdp_sdk.py: {e}", exc_info=True)
                        raise  # Re-raise for proper error handling
                        print(f"      Data: {response.text[:100]}...")
                        successful_endpoints.append((base_url, path, description, response.text))
                        
                elif response.status_code == 401:
                    print("      ❌ Unauthorized")
                elif response.status_code == 403:
                    print("      ❌ Forbidden")
                elif response.status_code == 404:
                    print("      ❌ Not Found")
                else:
                    error_text = response.text[:100] if response.text else "Unknown error"
                    print(f"      ❌ Error ({response.status_code}): {error_text}")
                    
            except Exception as e:
                print(f"      ❌ Exception: {e}")
    
    # Summary
    print(f"\n🎯 CDP SDK DISCOVERY RESULTS:")
    print("=" * 50)
    
    if successful_endpoints:
        print("✅ WORKING CDP SDK ENDPOINTS FOUND:")
        for base_url, path, desc, data in successful_endpoints:
            print(f"\n   🚀 {desc}:")
            print(f"      URL: {base_url}{path}")
            if isinstance(data, dict):
                print(f"      Type: JSON Response")
            else:
                print(f"      Type: Text Response")
                
        print(f"\n🎉 CONCLUSION: Your credentials are for CDP SDK!")
        print(f"   This enables:")
        print(f"   - On-chain wallet operations")
        print(f"   - Cryptocurrency transfers")
        print(f"   - Smart contract interactions")
        print(f"   - NOT traditional spot trading")
        
        print(f"\n💡 TO PLACE REAL CRYPTO ORDERS:")
        print(f"   You need Advanced Trading API credentials")
        print(f"   Current credentials: CDP SDK (on-chain operations)")
        
    else:
        print("❌ No working CDP SDK endpoints found")
        
        print(f"\n💡 POSSIBLE SOLUTIONS:")
        print(f"   1. Check if API key has proper permissions")
        print(f"   2. Verify this is the correct organization")
        print(f"   3. Try installing CDP SDK: pip install coinbase-sdk")
        print(f"   4. Create Advanced Trading API key for spot trading")

    return successful_endpoints

if __name__ == "__main__":
    results = test_cdp_sdk()
    
    if results:
        print(f"\n🚀 NEXT STEPS:")
        print(f"   1. Use CDP SDK for on-chain operations")
        print(f"   2. Create Advanced Trading API key for spot trading")
        print(f"   3. Test small transactions with SDK first") 