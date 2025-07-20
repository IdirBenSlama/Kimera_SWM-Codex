#!/usr/bin/env python3
"""
CDP SDK FIXED - Corrected Coinbase SDK usage
============================================
"""

import json
import time
from datetime import datetime
from coinbase.rest import RESTClient

class CDPSDKFixed:
    """Fixed CDP SDK test"""
    
    def __init__(self, api_key: str, api_secret: str):
        """Initialize CDP SDK client"""
        
        self.api_key = api_key
        self.api_secret = api_secret
        
        # Initialize Coinbase REST client (without sandbox parameter)
        try:
            self.client = RESTClient(
                api_key=api_key,
                api_secret=api_secret
            )
            print("✅ Coinbase SDK client initialized")
        except Exception as e:
            print(f"❌ SDK initialization failed: {e}")
            self.client = None
    
    def test_connection(self):
        """Test SDK connection"""
        
        if not self.client:
            return False, "Client not initialized"
        
        try:
            print("🔗 Testing SDK connection...")
            
            # Try different methods to test connection
            try:
                # Method 1: Get accounts
                accounts_response = self.client.get_accounts()
                print("✅ get_accounts() successful")
                return True, accounts_response
            except Exception as e1:
                print(f"get_accounts() failed: {e1}")
                
                try:
                    # Method 2: Get products
                    products = self.client.get_products()
                    print("✅ get_products() successful")
                    return True, products
                except Exception as e2:
                    print(f"get_products() failed: {e2}")
                    
                    try:
                        # Method 3: Get server time
                        server_time = self.client.get_unix_time()
                        print("✅ get_unix_time() successful")
                        return True, server_time
                    except Exception as e3:
                        print(f"get_unix_time() failed: {e3}")
                        return False, f"All methods failed: {e1}, {e2}, {e3}"
                
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False, str(e)
    
    def check_funds_simple(self):
        """Simple funds check"""
        
        print("\n💰 Checking funds (simple method)...")
        
        try:
            # Try to get any account information
            accounts_response = self.client.get_accounts()
            
            print(f"✅ Got accounts response")
            print(f"Response type: {type(accounts_response)}")
            
            # Try to extract account info
            if hasattr(accounts_response, 'accounts'):
                accounts = accounts_response.accounts
                print(f"📊 Found {len(accounts)} accounts")
                
                for i, account in enumerate(accounts):
                    print(f"Account {i+1}:")
                    print(f"  Type: {type(account)}")
                    
                    # Try to get currency
                    if hasattr(account, 'currency'):
                        print(f"  Currency: {account.currency}")
                    
                    # Try to get balance
                    if hasattr(account, 'available_balance'):
                        balance = account.available_balance
                        if hasattr(balance, 'value'):
                            print(f"  Available: {balance.value}")
                    
                    if hasattr(account, 'balance'):
                        balance = account.balance
                        if hasattr(balance, 'value'):
                            print(f"  Balance: {balance.value}")
            
            return True
            
        except Exception as e:
            print(f"❌ Funds check failed: {e}")
            return False
    
    def debug_sdk_methods(self):
        """Debug available SDK methods"""
        
        print("\n🔍 Debugging SDK methods...")
        
        if not self.client:
            print("❌ No client available")
            return
        
        # List available methods
        methods = [method for method in dir(self.client) if not method.startswith('_')]
        print(f"📊 Available methods: {len(methods)}")
        
        # Try safe methods
        safe_methods = [
            'get_unix_time',
            'get_products', 
            'get_accounts',
            'get_portfolios'
        ]
        
        for method_name in safe_methods:
            if hasattr(self.client, method_name):
                try:
                    method = getattr(self.client, method_name)
                    print(f"🔧 Trying {method_name}...")
                    result = method()
                    print(f"✅ {method_name} successful: {type(result)}")
                except Exception as e:
                    print(f"❌ {method_name} failed: {e}")

def main():
    """Main CDP SDK fixed test"""
    
    print("CDP SDK FIXED TEST")
    print("=" * 40)
    
    # Load credentials
    try:
        with open('Todelete alater/cdp_api_key.json', 'r') as f:
            credentials = json.load(f)
        
        api_key = credentials['id']
        api_secret = credentials['privateKey']
        
        print(f"✅ Loaded CDP credentials")
        print(f"   API Key: {api_key[:8]}...{api_key[-8:]}")
        
    except Exception as e:
        print(f"❌ Could not load credentials: {e}")
        return
    
    # Initialize and test SDK
    try:
        sdk_test = CDPSDKFixed(api_key, api_secret)
        
        if sdk_test.client:
            # Debug methods
            sdk_test.debug_sdk_methods()
            
            # Test connection
            success, data = sdk_test.test_connection()
            
            if success:
                print(f"\n🏆 CDP SDK CONNECTION SUCCESS!")
                
                # Try to check funds
                funds_success = sdk_test.check_funds_simple()
                
                if funds_success:
                    print(f"\n✅ FUNDS CHECK SUCCESSFUL!")
                    print(f"🚀 Ready for Kimera integration!")
                else:
                    print(f"\n⚠️ Funds check had issues, but connection works")
            
            else:
                print(f"\n❌ CDP SDK CONNECTION ISSUES")
                print(f"Details: {data}")
        
        # Save test results
        results = {
            'timestamp': datetime.now().isoformat(),
            'test_type': 'cdp_sdk_fixed',
            'sdk_initialized': sdk_test.client is not None,
            'connection_test': success if 'success' in locals() else False,
            'api_key_preview': f"{api_key[:8]}...{api_key[-8:]}"
        }
        
        filename = f"cdp_sdk_fixed_{int(time.time())}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📊 Results saved: {filename}")
        
    except Exception as e:
        print(f"❌ SDK test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 