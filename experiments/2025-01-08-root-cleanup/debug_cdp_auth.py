#!/usr/bin/env python3
"""
Debug CDP v2 Authentication
"""

import json
import time
import requests
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend

def test_cdp_auth():
    """Debug CDP v2 authentication step by step"""
    print("🔍 DEBUGGING CDP V2 AUTHENTICATION")
    print("=" * 40)
    
    # Your credentials
    ORGANIZATION_ID = "d5c46584-dd70-4be9-a71a-1e5e1b7a7ea3"
    API_KEY_ID = "dfe10f85-ed6c-4e75-a880-5db488c44f73"
    PRIVATE_KEY_PEM = """-----BEGIN EC PRIVATE KEY-----
MHcCAQEEIMyN35R0MUQewQ27M8ljhrGsQgRtIl1I3VMCZMucX4UIoAoGCCqGSM49
AwEHoUQDQgAERK6BZscG6p5nLQzIhPkUjXqIT9m/mw/S81U9di/u2BKRvujr4fUL
k+1M3dZ5l6SjNp2naYaa7oXuQQUm8UsFFA==
-----END EC PRIVATE KEY-----"""
    
    print(f"Organization ID: {ORGANIZATION_ID}")
    print(f"API Key ID: {API_KEY_ID}")
    
    # Step 1: Load private key
    print("\n📝 Step 1: Loading private key...")
    try:
        private_key = serialization.load_pem_private_key(
            PRIVATE_KEY_PEM.encode(),
            password=None,
            backend=default_backend()
        )
        print("✅ Private key loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load private key: {e}")
        return
    
    # Step 2: Create JWT token
    print("\n📝 Step 2: Creating JWT token...")
    try:
        import jwt
        
        request_path = f"/platform/organizations/{ORGANIZATION_ID}"
        uri = f"GET https://api.developer.coinbase.com{request_path}"
        
        payload = {
            'sub': API_KEY_ID,
            'iss': "cdp",
            'nbf': int(time.time()),
            'exp': int(time.time()) + 120,  # 2 minutes
            'uri': uri
        }
        
        print(f"JWT Payload: {payload}")
        
        token = jwt.encode(
            payload,
            private_key,
            algorithm='ES256',
            headers={'kid': API_KEY_ID}
        )
        
        print("✅ JWT token created successfully")
        print(f"Token length: {len(token)}")
        
    except Exception as e:
        print(f"❌ Failed to create JWT token: {e}")
        return
    
    # Step 3: Test API call
    print("\n📝 Step 3: Testing API call...")
    try:
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        
        url = f"https://api.developer.coinbase.com{request_path}"
        print(f"URL: {url}")
        print(f"Headers: {headers}")
        
        # Make the request with timeout
        response = requests.get(url, headers=headers, timeout=30)
        
        print(f"Response Status: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Authentication successful!")
            print(f"Organization data: {data}")
        else:
            print(f"❌ Authentication failed")
            print(f"Response text: {response.text}")
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out")
    except Exception as e:
        print(f"❌ API call failed: {e}")

if __name__ == "__main__":
    test_cdp_auth() 