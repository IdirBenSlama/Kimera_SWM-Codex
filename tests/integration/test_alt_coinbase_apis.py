#!/usr/bin/env python3
"""
Test Alternative Coinbase APIs with User Credentials
"""

import json
import time
import requests
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend

def test_alt_apis():
    """Test alternative Coinbase APIs with user credentials"""
    print("🔍 TESTING ALTERNATIVE COINBASE APIS")
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
    
    def test_api_base_url(base_url: str, api_name: str):
        """Test a specific API base URL"""
        print(f"\n🔍 TESTING: {api_name}")
        print(f"   Base URL: {base_url}")
        print("=" * 30)
        
        # Test endpoints that commonly exist
        endpoints_to_test = [
            ("/", "Root"),
            ("/v2", "API v2"),
            ("/v2/user", "User Info"),
            ("/v2/accounts", "Accounts"),
            ("/v2/account", "Account"),
            ("/accounts", "Accounts (simple)"),
            ("/user", "User (simple)"),
            ("/wallets", "Wallets"),
            ("/portfolio", "Portfolio"),
            ("/products", "Products"),
            ("/currencies", "Currencies"),
            ("/exchange-rates", "Exchange Rates"),
            ("/prices", "Prices"),
        ]
        
        successful_endpoints = []
        
        for path, description in endpoints_to_test:
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
                        print(f"      Data preview: {json.dumps(data, indent=2)[:150]}...")
                        successful_endpoints.append((path, description, data))
                    except Exception as e:
                        logger.error(f"Error in test_alt_coinbase_apis.py: {e}", exc_info=True)
                        raise  # Re-raise for proper error handling
                        print(f"      Data: {response.text[:100]}...")
                        successful_endpoints.append((path, description, response.text))
                elif response.status_code == 401:
                    print("      ❌ Unauthorized")
                elif response.status_code == 403:
                    print("      ❌ Forbidden")
                elif response.status_code == 404:
                    print("      ❌ Not Found")
                else:
                    print(f"      ❌ Error: {response.text[:50]}")
                    
            except Exception as e:
                print(f"      ❌ Exception: {e}")
        
        return successful_endpoints
    
    # Test the promising base URLs
    base_urls_to_test = [
        ("https://api.coinbase.com", "Standard Coinbase API"),
        ("https://api.wallet.coinbase.com", "Coinbase Wallet API"),
    ]
    
    all_successful = {}
    
    for base_url, api_name in base_urls_to_test:
        successful = test_api_base_url(base_url, api_name)
        if successful:
            all_successful[api_name] = successful
    
    # Summary
    print(f"\n🎯 FINAL RESULTS:")
    print("=" * 50)
    
    if all_successful:
        print("✅ WORKING APIS FOUND:")
        for api_name, endpoints in all_successful.items():
            print(f"\n   🚀 {api_name}:")
            for path, desc, data in endpoints:
                print(f"      {path} - {desc}")
                
        # If we found working endpoints, this means we can place real orders!
        print(f"\n🎉 SUCCESS! Your credentials work with:")
        for api_name in all_successful.keys():
            print(f"   - {api_name}")
        print(f"\n💡 Next step: Create a real trader using the working API")
        
    else:
        print("❌ No working APIs found")
        print("💡 Possible issues:")
        print("   - Credentials might be for a different service")
        print("   - API might require different authentication")
        print("   - Might be sandbox/testnet credentials")

if __name__ == "__main__":
    test_alt_apis() 