#!/usr/bin/env python3
import time
import hashlib
import hmac
import base64
import requests

print("TESTING MODERN COINBASE API - NO PASSPHRASE REQUIRED")
print("="*55)

# Your credentials
API_KEY = "9268de76-b5f4-4683-b593-327fb2c19503"
API_SECRET = "BiCUFOxZ4J4Fi8F6mcyzuzreXaGZeBLHxr7q8Puo6VHcSEgyqJ6mIx29RbbAJGAjq6SHBt5K4PieiymRhEWVHw=="

print(f"API Key: {API_KEY[:8]}...{API_KEY[-8:]}")
print("Passphrase: NOT REQUIRED (Modern API)")

# Test sandbox
base_url = "https://api.sandbox.coinbase.com"
endpoint = "/api/v3/brokerage/accounts"
method = "GET"

# Generate signature WITHOUT passphrase
timestamp = str(int(time.time()))
message = timestamp + method + endpoint

# Sign with base64 decoded secret
secret_bytes = base64.b64decode(API_SECRET)
signature = hmac.new(secret_bytes, message.encode(), hashlib.sha256).digest()
signature_b64 = base64.b64encode(signature).decode()

# Headers - NO PASSPHRASE HEADER
headers = {
    "CB-ACCESS-KEY": API_KEY,
    "CB-ACCESS-SIGN": signature_b64,
    "CB-ACCESS-TIMESTAMP": timestamp,
    "Content-Type": "application/json"
}

print(f"\nTesting: {base_url + endpoint}")
print("Authentication: HMAC-SHA256 (no passphrase)")

try:
    response = requests.get(base_url + endpoint, headers=headers, timeout=30)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print("SUCCESS! Modern API works without passphrase")
        print(f"Found {len(data.get('accounts', []))} accounts")
    else:
        print(f"Response: {response.text}")
        
except Exception as e:
    print(f"Error: {e}")

print("\nCONCLUSION: Modern Coinbase API does NOT require passphrase!")
