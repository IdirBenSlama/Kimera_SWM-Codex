#!/usr/bin/env python3
"""
SIMPLE CDP API TEST - For Coinbase Developer Platform
"""

import json
import requests
import time
from datetime import datetime

def test_cdp_api():
    """Simple test for CDP API with existing credentials"""
    
    print("COINBASE DEVELOPER PLATFORM (CDP) API TEST")
    print("=" * 50)
    
    # Load credentials
    try:
        with open('Todelete alater/cdp_api_key.json', 'r') as f:
            credentials = json.load(f)
        
        api_key_id = credentials['id']
        private_key = credentials['privateKey']
        
        print(f"✅ Loaded CDP credentials")
        print(f"   API Key ID: {api_key_id[:8]}...{api_key_id[-8:]}")
        
    except Exception as e:
        print(f"❌ Could not load credentials: {e}")
        return
    
    # Test different CDP endpoints
    endpoints_to_test = [
        "https://api.coinbase.com/v2/user",
        "https://api.coinbase.com/v2/accounts",
        "https://api.coinbase.com/api/v3/brokerage/accounts",
        "https://coinbase.com/api/v2/accounts"
    ]
    
    print(f"\n🔗 Testing CDP API endpoints...")
    
    for endpoint in endpoints_to_test:
        print(f"\nTesting: {endpoint}")
        
        try:
            # Simple request with API key
            headers = {
                'CB-ACCESS-KEY': api_key_id,
                'Content-Type': 'application/json'
            }
            
            response = requests.get(endpoint, headers=headers, timeout=10)
            
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ SUCCESS - Got response")
                
                # Look for account data
                if 'data' in data:
                    accounts = data['data']
                    if isinstance(accounts, list):
                        print(f"   Found {len(accounts)} accounts")
                        
                        total_usd = 0
                        for account in accounts:
                            if isinstance(account, dict):
                                currency = account.get('currency', {}).get('code', 'UNKNOWN')
                                balance = account.get('balance', {}).get('amount', '0')
                                
                                try:
                                    balance_float = float(balance)
                                    if balance_float > 0:
                                        print(f"   {currency}: {balance_float}")
                                        
                                        if currency == 'USD':
                                            total_usd += balance_float
                                except:
                                    pass
                        
                        if total_usd > 0:
                            print(f"💰 Total USD: ${total_usd:.2f}")
                            
                            if total_usd >= 100:
                                print("✅ READY FOR TRADING")
                            else:
                                print("⚠️ Consider adding more funds")
                
                break  # Found working endpoint
                
            else:
                print(f"❌ HTTP {response.status_code}: {response.text[:100]}")
                
        except Exception as e:
            print(f"❌ Request failed: {str(e)}")
    
    print(f"\n🏆 CDP TEST COMPLETE")
    print("=" * 50)
    
    # Save simple test result
    result = {
        'timestamp': datetime.now().isoformat(),
        'test_type': 'simple_cdp',
        'credentials_loaded': True,
        'api_key_preview': f"{api_key_id[:8]}...{api_key_id[-8:]}"
    }
    
    filename = f"simple_cdp_test_{int(time.time())}.json"
    with open(filename, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"📊 Test results saved to: {filename}")
    
    print(f"\n🚀 NEXT STEPS:")
    print(f"   1. If successful, we can integrate with Kimera trading")
    print(f"   2. If failed, we may need different authentication method")
    print(f"   3. CDP API might require different approach than legacy Coinbase Pro")

if __name__ == "__main__":
    test_cdp_api() 