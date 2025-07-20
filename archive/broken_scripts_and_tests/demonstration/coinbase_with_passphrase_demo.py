#!/usr/bin/env python3
"""
COINBASE ADVANCED TRADING API - COMPLETE AUTHENTICATION DEMO
==========================================================

This shows the correct authentication format including the required passphrase.
You need to provide your passphrase to complete the authentication.
"""

import time
import hmac
import hashlib
import base64
import requests
import json

# Your actual credentials
API_KEY = "9268de76-b5f4-4683-b593-327fb2c19503"
API_SECRET = "BiCUFOxZ4J4Fi8F6mcyzuzreXaGZeBLHxr7q8Puo6VHcSEgyqJ6mIx29RbbAJGAjq6SHBt5K4PieiymRhEWVHw=="

# ⚠️ YOU NEED TO PROVIDE YOUR PASSPHRASE HERE ⚠️
# This is the passphrase you created when generating your API key
API_PASSPHRASE = "YOUR_PASSPHRASE_HERE"  # Replace with your actual passphrase

def test_api_with_passphrase():
    print("🔑 Testing Coinbase Advanced Trading API with COMPLETE authentication...")
    print(f"🔑 API Key: {API_KEY[:8]}...{API_KEY[-8:]}")
    print(f"🔑 Passphrase: {'*' * len(API_PASSPHRASE) if API_PASSPHRASE != 'YOUR_PASSPHRASE_HERE' else 'NOT SET'}")
    
    if API_PASSPHRASE == "YOUR_PASSPHRASE_HERE":
        print("\n❌ PASSPHRASE NOT SET!")
        print("💡 You need to replace 'YOUR_PASSPHRASE_HERE' with your actual passphrase")
        print("💡 This is the passphrase you created when generating your API key in Coinbase")
        return False
    
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
    
    # Complete headers with passphrase
    headers = {
        'CB-ACCESS-KEY': API_KEY,
        'CB-ACCESS-SIGN': signature_b64,
        'CB-ACCESS-TIMESTAMP': timestamp,
        'CB-ACCESS-PASSPHRASE': API_PASSPHRASE,  # This was missing!
        'Content-Type': 'application/json'
    }
    
    # Make request
    url = "https://api.coinbase.com/api/v3/brokerage/accounts"
    
    try:
        print(f"\n📡 Making authenticated request to: {url}")
        response = requests.get(url, headers=headers, timeout=10)
        print(f"📡 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ API CONNECTION SUCCESSFUL!")
            print(f"📊 Accounts found: {len(data.get('accounts', []))}")
            
            # Display account balances
            total_usd = 0.0
            for account in data.get('accounts', []):
                currency = account.get('currency', 'UNKNOWN')
                balance = float(account.get('available_balance', {}).get('value', 0))
                
                if balance > 0:
                    print(f"   💰 {currency}: {balance:.6f}")
                    
                    if currency == 'USD':
                        total_usd = balance
            
            print(f"\n💵 Total USD Available: ${total_usd:.6f}")
            
            if total_usd > 0:
                print("\n🚀 READY FOR KIMERA LIVE TRADING!")
                print("✅ All authentication working perfectly")
                print("✅ Account has funds available")
                print("💡 You can now run the full Kimera trading system")
            else:
                print("\n⚠️  Account has no USD balance")
                print("💡 You may need to deposit funds or check other currency balances")
            
            return True
            
        elif response.status_code == 401:
            print("❌ Still getting 401 Unauthorized")
            print("💡 Please double-check your passphrase")
            print("💡 The passphrase is case-sensitive and must match exactly")
            return False
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def show_next_steps():
    print("\n" + "="*60)
    print("🎯 NEXT STEPS TO ACTIVATE KIMERA LIVE TRADING")
    print("="*60)
    print("1. 🔑 Find your API passphrase from when you created the API key")
    print("2. 📝 Edit this file and replace 'YOUR_PASSPHRASE_HERE' with your actual passphrase")
    print("3. 🧪 Run this test again to verify authentication")
    print("4. 🚀 Launch the full Kimera trading system")
    print("="*60)

if __name__ == "__main__":
    print("\n🔐 COINBASE ADVANCED TRADING API - AUTHENTICATION TEST")
    print("="*60)
    
    success = test_api_with_passphrase()
    
    if not success:
        show_next_steps()
    
    print("\n📖 AUTHENTICATION REQUIREMENTS:")
    print("   ✅ API Key (provided)")
    print("   ✅ API Secret (provided)")
    print("   ❓ Passphrase (needs to be provided by you)")
    print("\n💡 The passphrase is something you created when generating your API key")
    print("💡 It's different from your Coinbase account password") 