#!/usr/bin/env python3
"""
Discover CDP API Type and Available Endpoints
"""

import json
import time
import requests
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend

def discover_cdp_api():
    """Discover what type of CDP API these credentials access"""
    print("🔍 DISCOVERING CDP API TYPE")
    print("=" * 40)
    
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
    
    def create_jwt_token(request_path: str, method: str = 'GET') -> str:
        """Create JWT token for API call"""
        import jwt
        
        uri = f"{method} https://api.developer.coinbase.com{request_path}"
        
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
    
    def test_endpoint(path: str, description: str):
        """Test a specific endpoint"""
        print(f"\n🔍 Testing: {description}")
        print(f"   Path: {path}")
        
        try:
            token = create_jwt_token(path)
            headers = {
                'Authorization': f'Bearer {token}',
                'Content-Type': 'application/json'
            }
            
            response = requests.get(
                f"https://api.developer.coinbase.com{path}",
                headers=headers,
                timeout=10
            )
            
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                print("   ✅ SUCCESS!")
                data = response.json()
                print(f"   Data: {json.dumps(data, indent=2)[:200]}...")
                return True
            elif response.status_code == 404:
                print("   ❌ Not Found")
            elif response.status_code == 403:
                print("   ❌ Forbidden (wrong permissions)")
            elif response.status_code == 401:
                print("   ❌ Unauthorized (auth failed)")
            else:
                print(f"   ❌ Error: {response.text[:100]}")
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
        
        return False
    
    # Test various CDP endpoints
    endpoints_to_test = [
        # Wallet API v2 endpoints
        ("/v2/user", "User Info (Wallet API v2)"),
        ("/v2/accounts", "Accounts (Wallet API v2)"),
        
        # Platform API endpoints  
        (f"/platform/organizations/{ORGANIZATION_ID}", "Organization Info"),
        ("/platform/wallets", "Platform Wallets"),
        
        # Base paths
        ("/", "Root"),
        ("/v1", "API v1"),
        ("/v2", "API v2"),
        
        # Trading endpoints
        ("/api/v3/brokerage/accounts", "Advanced Trade Accounts"),
        ("/api/v3/brokerage/products", "Trading Products"),
        
        # Other possible endpoints
        (f"/organizations/{ORGANIZATION_ID}", "Organizations (alt path)"),
        (f"/orgs/{ORGANIZATION_ID}", "Orgs (short path)"),
        ("/wallets", "Wallets (simple path)"),
        ("/projects", "Projects"),
        ("/apps", "Apps"),
    ]
    
    successful_endpoints = []
    
    for path, description in endpoints_to_test:
        if test_endpoint(path, description):
            successful_endpoints.append((path, description))
    
    print(f"\n🎯 DISCOVERY RESULTS:")
    print(f"   Successful endpoints: {len(successful_endpoints)}")
    
    if successful_endpoints:
        print("   ✅ Working endpoints:")
        for path, desc in successful_endpoints:
            print(f"      {path} - {desc}")
    else:
        print("   ❌ No working endpoints found")
        print("   💡 This might be:")
        print("      - A different type of CDP API")
        print("      - Credentials for a specific app/wallet")
        print("      - Wrong base URL")
        
    # Try alternative base URLs
    print(f"\n🔍 Trying alternative base URLs...")
    
    alt_urls = [
        "https://api.coinbase.com",
        "https://api.cdp.coinbase.com", 
        "https://wallet-api.coinbase.com",
        "https://api.wallet.coinbase.com"
    ]
    
    for base_url in alt_urls:
        print(f"\n🔍 Testing base URL: {base_url}")
        try:
            # Simple test without auth first
            response = requests.get(f"{base_url}/", timeout=5)
            print(f"   Status: {response.status_code}")
            if response.status_code != 404:
                print(f"   ✅ Base URL exists!")
        except Exception as e:
            print(f"   ❌ {e}")

if __name__ == "__main__":
    discover_cdp_api() 