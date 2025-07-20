#!/usr/bin/env python3
"""
Test Coinbase Connection
========================
Verify API credentials and check account balance
"""

import os
import time
import hmac
import hashlib
import base64
import requests
import json
from dotenv import load_dotenv

# Load credentials
load_dotenv('kimera_cdp_live.env')

api_key = os.getenv('CDP_API_KEY_NAME', '').strip()
api_secret = os.getenv('CDP_API_KEY_PRIVATE_KEY', '').strip()

print(f"API Key: {api_key[:10]}...")
print(f"Secret length: {len(api_secret)}")

# Try different authentication methods
def test_auth_v3():
    """Test v3 API authentication"""
    print("\n🔍 Testing Coinbase Advanced Trade API (v3)...")
    
    timestamp = str(int(time.time()))
    method = "GET"
    path = "/api/v3/brokerage/accounts"
    body = ""
    
    # Create signature
    message = f"{timestamp}{method}{path}{body}"
    
    # Try base64 decoding the secret
    try:
        secret = base64.b64decode(api_secret)
        print("✅ Secret is base64 encoded")
    except:
        secret = api_secret.encode('utf-8')
        print("ℹ️ Secret is plain text")
    
    signature = hmac.new(
        secret,
        message.encode('utf-8'),
        hashlib.sha256
    ).digest()
    
    signature_b64 = base64.b64encode(signature).decode()
    
    headers = {
        "CB-ACCESS-KEY": api_key,
        "CB-ACCESS-SIGN": signature_b64,
        "CB-ACCESS-TIMESTAMP": timestamp,
        "Content-Type": "application/json"
    }
    
    url = f"https://api.coinbase.com{path}"
    
    response = requests.get(url, headers=headers)
    
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text[:200]}...")
    
    if response.status_code == 200:
        data = response.json()
        print("\n💰 ACCOUNTS:")
        for account in data.get('accounts', []):
            currency = account.get('currency')
            balance = account.get('available_balance', {}).get('value', '0')
            print(f"   {currency}: {balance}")

def test_auth_v2():
    """Test v2 API authentication (legacy)"""
    print("\n🔍 Testing Coinbase Pro API (v2)...")
    
    timestamp = str(time.time())
    method = "GET"
    path = "/accounts"
    body = ""
    
    message = f"{timestamp}{method}{path}{body}"
    
    # For v2, secret should be base64
    try:
        secret = base64.b64decode(api_secret)
    except:
        print("❌ V2 API requires base64 encoded secret")
        return
    
    signature = hmac.new(
        secret,
        message.encode('utf-8'),
        hashlib.sha256
    ).digest()
    
    signature_b64 = base64.b64encode(signature).decode()
    
    headers = {
        "CB-ACCESS-KEY": api_key,
        "CB-ACCESS-SIGN": signature_b64,
        "CB-ACCESS-TIMESTAMP": timestamp,
        "CB-ACCESS-PASSPHRASE": os.getenv('CDP_API_PASSPHRASE', ''),  # V2 might need passphrase
        "Content-Type": "application/json"
    }
    
    url = f"https://api.pro.coinbase.com{path}"
    
    response = requests.get(url, headers=headers)
    
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text[:200]}...")

# Run tests
test_auth_v3()
test_auth_v2()

print("\n📝 NOTES:")
print("- If getting 401, check if API key has 'view' permission")
print("- Make sure the API key is for Advanced Trade, not CDP")
print("- The secret might need to be in a specific format") 