#!/usr/bin/env python3
"""
Simple Coinbase API Connection Test
"""

import requests
import json

def test_coinbase_connection():
    """Test basic connection to Coinbase API"""
    print("🔥 TESTING COINBASE API CONNECTION")
    print("=" * 40)
    
    API_KEY = "9268de76-b5f4-4683-b593-327fb2c19503"
    
    try:
        # Test basic API connection
        headers = {
            'Authorization': f'Bearer {API_KEY}',
            'Content-Type': 'application/json'
        }
        
        print("📡 Testing API connection...")
        response = requests.get("https://api.coinbase.com/v2/user", headers=headers)
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            user_data = response.json()
            print("✅ API connection successful!")
            print(f"User ID: {user_data.get('data', {}).get('id', 'Unknown')}")
            
            # Test account access
            print("\n📊 Testing account access...")
            accounts_response = requests.get("https://api.coinbase.com/v2/accounts", headers=headers)
            
            print(f"Accounts response: {accounts_response.status_code}")
            
            if accounts_response.status_code == 200:
                accounts_data = accounts_response.json()
                print("✅ Account access successful!")
                
                # Look for EUR account
                eur_balance = 0.0
                for account in accounts_data.get('data', []):
                    currency = account.get('currency')
                    balance = float(account.get('balance', {}).get('amount', 0))
                    
                    print(f"   {currency}: {balance}")
                    
                    if currency == 'EUR':
                        eur_balance = balance
                
                if eur_balance > 0:
                    print(f"\n💰 EUR Balance: €{eur_balance:.2f}")
                    if eur_balance >= 1.0:
                        print("✅ Sufficient balance for trading!")
                    else:
                        print("⚠️ Low balance for trading")
                else:
                    print("\n❌ No EUR balance found")
                    
            else:
                print(f"❌ Account access failed: {accounts_response.text}")
                
        else:
            print(f"❌ API connection failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_coinbase_connection() 