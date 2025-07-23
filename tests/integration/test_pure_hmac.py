#!/usr/bin/env python3
"""
Pure HMAC Connector Test
Tests the HMAC connector without any Kimera imports
"""

import asyncio
import hashlib
import hmac
import time
from urllib.parse import urlencode
import aiohttp
import json

class SimpleBinanceConnector:
    """Simple HMAC-based Binance connector"""
    
    def __init__(self, api_key: str, secret_key: str):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = "https://api.binance.com"
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def _sign_request(self, params):
        """Sign request with HMAC-SHA256"""
        query_string = urlencode(params, safe='~')
        return hmac.new(
            self.secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    async def get_account_info(self):
        """Get account information"""
        params = {
            'timestamp': int(time.time() * 1000),
            'recvWindow': 5000
        }
        
        signature = self._sign_request(params)
        params['signature'] = signature
        
        headers = {'X-MBX-APIKEY': self.api_key}
        
        url = f"{self.base_url}/api/v3/account"
        
        async with self.session.get(url, params=params, headers=headers) as response:
            data = await response.json()
            
            if response.status != 200:
                raise Exception(f"API Error: {response.status} - {data}")
            
            return data

async def test_pure_hmac():
    """Test pure HMAC authentication"""
    
    print("🔍 PURE HMAC CONNECTOR TEST")
    print("=" * 40)
    
    # Use the working credentials
    api_key = 'Y9WyflPyK1tVXnET3CTMvSdCbPia3Nhtd89VYWjS9RaAbQ0KEhHezkcGSCySQ8cL'
    secret_key = 'qUn5JqSpYz1GDxFj2X3UF23TYgtxKrTsCbDZEoBMYCPbYZgP4siVLyspkB5HAPl7'
    
    print(f"📋 API Key: {api_key[:8]}...{api_key[-8:]}")
    print(f"📋 Secret Key: {'*' * len(secret_key)}")
    
    try:
        async with SimpleBinanceConnector(api_key, secret_key) as connector:
            print("✅ Connector initialized")
            
            # Test account info
            print("\n🔐 Testing Account Information...")
            
            account_info = await connector.get_account_info()
            
            print("🎉 HMAC AUTHENTICATION SUCCESSFUL!")
            print(f"   Account Type: {account_info.get('accountType', 'Unknown')}")
            print(f"   Can Trade: {account_info.get('canTrade', False)}")
            print(f"   Can Withdraw: {account_info.get('canWithdraw', False)}")
            
            balances = account_info.get('balances', [])
            non_zero_balances = [b for b in balances if float(b['free']) > 0 or float(b['locked']) > 0]
            
            print(f"   Total Balances: {len(balances)}")
            print(f"   Non-zero Balances: {len(non_zero_balances)}")
            
            if non_zero_balances:
                print("\n   💰 Asset Holdings:")
                for balance in non_zero_balances[:5]:
                    asset = balance['asset']
                    free = float(balance['free'])
                    locked = float(balance['locked'])
                    total = free + locked
                    if total > 0:
                        print(f"      {asset}: {total:.8f}")
            
            return True
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_pure_hmac())
    
    if result:
        print("\n✅ PURE HMAC TEST: SUCCESS")
        print("The HMAC implementation works correctly!")
    else:
        print("\n❌ PURE HMAC TEST: FAILED")
        print("There's an issue with the HMAC implementation.") 