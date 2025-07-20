#!/usr/bin/env python3
"""
CDP SDK TEST - Official Coinbase Advanced Trading SDK
=====================================================
Using the official coinbase-advanced-py SDK with your CDP credentials
"""

import json
import time
from datetime import datetime
from coinbase.rest import RESTClient

class CDPSDKTest:
    """Test CDP using official Coinbase SDK"""
    
    def __init__(self, api_key: str, api_secret: str):
        """Initialize CDP SDK client"""
        
        self.api_key = api_key
        self.api_secret = api_secret
        
        # Initialize Coinbase REST client
        try:
            self.client = RESTClient(
                api_key=api_key,
                api_secret=api_secret,
                sandbox=False  # Use live environment
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
            # Try to get accounts
            print("🔗 Testing SDK connection...")
            accounts_response = self.client.get_accounts()
            
            if accounts_response and hasattr(accounts_response, 'accounts'):
                print("✅ SDK connection successful!")
                return True, accounts_response
            else:
                print("❌ No accounts data received")
                return False, "No accounts data"
                
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False, str(e)
    
    def check_funds(self):
        """Check account funds using SDK"""
        
        print("\n💰 Checking funds with SDK...")
        
        try:
            # Get accounts
            accounts_response = self.client.get_accounts()
            
            if not accounts_response or not hasattr(accounts_response, 'accounts'):
                print("❌ No accounts found")
                return 0
            
            accounts = accounts_response.accounts
            print(f"📊 Found {len(accounts)} accounts")
            
            total_usd_value = 0
            accounts_with_balance = 0
            
            print("\n💳 Account Balances:")
            print("-" * 50)
            
            for account in accounts:
                currency = account.currency if hasattr(account, 'currency') else 'UNKNOWN'
                
                # Get available balance
                available = 0
                if hasattr(account, 'available_balance') and account.available_balance:
                    available = float(account.available_balance.value)
                elif hasattr(account, 'balance') and account.balance:
                    available = float(account.balance.value)
                
                if available > 0:
                    accounts_with_balance += 1
                    
                    if currency == 'USD':
                        total_usd_value += available
                        print(f"💵 {currency}: ${available:.2f} available")
                    else:
                        print(f"🪙 {currency}: {available:.8f} available")
                        
                        # Try to get current price for major cryptos
                        if currency in ['BTC', 'ETH', 'LTC', 'BCH']:
                            try:
                                product_id = f"{currency}-USD"
                                ticker = self.client.get_product(product_id)
                                
                                if ticker and hasattr(ticker, 'price'):
                                    price = float(ticker.price)
                                    usd_value = available * price
                                    total_usd_value += usd_value
                                    print(f"     = ${usd_value:.2f} USD (@ ${price:.2f})")
                            except:
                                print(f"     = Price unavailable")
            
            print("-" * 50)
            print(f"💰 TOTAL USD VALUE: ${total_usd_value:.2f}")
            print(f"📊 Accounts with balance: {accounts_with_balance}")
            
            return total_usd_value
            
        except Exception as e:
            print(f"❌ Failed to check funds: {e}")
            return 0
    
    def check_trading_products(self):
        """Check available trading products"""
        
        print("\n📈 Checking trading products...")
        
        try:
            # Get available products
            products = self.client.get_products()
            
            if not products:
                print("❌ No products found")
                return []
            
            major_pairs = ['BTC-USD', 'ETH-USD', 'LTC-USD', 'BCH-USD']
            available_pairs = []
            
            # Check if products is iterable
            product_list = products if hasattr(products, '__iter__') else []
            if hasattr(products, 'products'):
                product_list = products.products
            
            for product in product_list:
                product_id = product.product_id if hasattr(product, 'product_id') else str(product)
                
                if product_id in major_pairs:
                    available_pairs.append(product_id)
            
            if available_pairs:
                print(f"✅ Available major pairs: {', '.join(available_pairs)}")
            else:
                print("⚠️ No major trading pairs found")
            
            return available_pairs
            
        except Exception as e:
            print(f"❌ Failed to check trading products: {e}")
            return []
    
    def assess_trading_readiness(self, total_funds: float):
        """Assess trading readiness"""
        
        print(f"\n🎯 TRADING READINESS ASSESSMENT:")
        print("=" * 50)
        
        if total_funds >= 1000:
            status = "EXCELLENT"
            recommendation = "$100-500 per trading session"
            ready = True
        elif total_funds >= 500:
            status = "GOOD"
            recommendation = "$50-200 per trading session"
            ready = True
        elif total_funds >= 100:
            status = "ADEQUATE"
            recommendation = "$25-100 per trading session"
            ready = True
        elif total_funds >= 25:
            status = "MINIMAL"
            recommendation = "$5-25 per trading session"
            ready = False
        else:
            status = "INSUFFICIENT"
            recommendation = "Add more funds before trading"
            ready = False
        
        print(f"Status: {'✅' if ready else '❌'} {status}")
        print(f"Recommendation: {recommendation}")
        
        if ready:
            print(f"🚀 KIMERA AUTONOMOUS TRADING: READY")
        else:
            print(f"⚠️ KIMERA AUTONOMOUS TRADING: NOT RECOMMENDED")
        
        return ready, status

def main():
    """Main CDP SDK test"""
    
    print("CDP SDK TEST - Official Coinbase Advanced Trading")
    print("=" * 60)
    
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
        sdk_test = CDPSDKTest(api_key, api_secret)
        
        # Test connection
        success, data = sdk_test.test_connection()
        
        if success:
            print(f"\n🏆 CDP SDK CONNECTION SUCCESS!")
            
            # Check funds
            total_funds = sdk_test.check_funds()
            
            # Check trading products
            trading_pairs = sdk_test.check_trading_products()
            
            # Assess trading readiness
            ready, status = sdk_test.assess_trading_readiness(total_funds)
            
            # Final summary
            print(f"\n🎯 FINAL RESULTS")
            print("=" * 60)
            print(f"SDK Connection: ✅ SUCCESS")
            print(f"Total USD Value: ${total_funds:.2f}")
            print(f"Trading Pairs: {len(trading_pairs)} available")
            print(f"Trading Status: {status}")
            print(f"Kimera Ready: {'✅ YES' if ready else '❌ NO'}")
            
            # Save results
            results = {
                'timestamp': datetime.now().isoformat(),
                'test_type': 'cdp_sdk',
                'connection': 'success',
                'total_usd_funds': total_funds,
                'trading_pairs_available': trading_pairs,
                'trading_ready': ready,
                'trading_status': status,
                'api_key_preview': f"{api_key[:8]}...{api_key[-8:]}"
            }
            
            filename = f"cdp_sdk_results_{int(time.time())}.json"
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"📊 Results saved: {filename}")
            
            if ready:
                print(f"\n🚀 NEXT STEP: INTEGRATE WITH KIMERA!")
                print(f"   Your CDP credentials are working")
                print(f"   Funds are sufficient for autonomous trading")
                print(f"   Ready to connect to Kimera trading engine")
            else:
                print(f"\n⚠️ RECOMMENDATION: ADD MORE FUNDS")
                print(f"   Current: ${total_funds:.2f}")
                print(f"   Recommended minimum: $100")
                print(f"   Optimal: $500+")
        
        else:
            print(f"\n❌ CDP SDK CONNECTION FAILED")
            print(f"Error: {data}")
            print(f"\nPossible issues:")
            print(f"   - API permissions not enabled")
            print(f"   - Credentials format incorrect")
            print(f"   - Account setup incomplete")
            
    except Exception as e:
        print(f"❌ SDK test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 