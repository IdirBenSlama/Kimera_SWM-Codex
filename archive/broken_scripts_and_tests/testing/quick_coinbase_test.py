#!/usr/bin/env python3
"""
QUICK COINBASE API TEST
"""

import time
import hmac
import hashlib
import base64
import requests
import json

# User's credentials
API_KEY = "9268de76-b5f4-4683-b593-327fb2c19503"
API_SECRET = "BiCUFOxZ4J4Fi8F6mcyzuzreXaGZeBLHxr7q8Puo6VHcSEgyqJ6mIx29RbbAJGAjq6SHBt5K4PieiymRhEWVHw=="

def test_api():
    print("🔑 Testing Coinbase Advanced Trading API...")
    
    # Generate signature
    timestamp = str(int(time.time()))
    method = "GET"
    path = "/accounts"
    body = ""
    message = timestamp + method + path + body
    
    # Decode secret and create signature
    secret_decoded = base64.b64decode(API_SECRET)
    signature = hmac.new(secret_decoded, message.encode('utf-8'), hashlib.sha256).digest()
    signature_b64 = base64.b64encode(signature).decode('utf-8')
    
    # Headers
    headers = {
        'CB-ACCESS-KEY': API_KEY,
        'CB-ACCESS-SIGN': signature_b64,
        'CB-ACCESS-TIMESTAMP': timestamp,
        'Content-Type': 'application/json'
    }
    
    # Make request
    url = "https://api.coinbase.com/api/v3/brokerage/accounts"
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ API connection successful!")
            print(f"Accounts found: {len(data.get('accounts', []))}")
            
            # Check for USD balance
            for account in data.get('accounts', []):
                if account.get('currency') == 'USD':
                    balance = account.get('available_balance', {}).get('value', '0')
                    print(f"💰 USD Balance: ${balance}")
                    break
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
    
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_api() 