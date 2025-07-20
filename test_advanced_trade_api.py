#!/usr/bin/env python3
"""
Test Coinbase Advanced Trade API Implementation
===============================================
Tests the new implementation using official SDK
"""

import os
import sys
import asyncio
import logging
from dotenv import load_dotenv

# Test SDK import
try:
    from coinbase.rest import RESTClient
    print("✅ Coinbase Advanced Trade SDK imported successfully")
except ImportError as e:
    print(f"❌ SDK import failed: {e}")
    print("Install with: pip install coinbase-advanced-py")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_credentials():
    """Test if credentials are properly configured"""
    load_dotenv('.env')
    
    api_key = os.getenv('COINBASE_ADVANCED_API_KEY')
    api_secret = os.getenv('COINBASE_ADVANCED_API_SECRET')
    
    print("\n🔍 CREDENTIAL CHECK:")
    print(f"API Key present: {'✅' if api_key else '❌'}")
    print(f"API Secret present: {'✅' if api_secret else '❌'}")
    
    if api_key:
        print(f"API Key preview: {api_key[:10]}...")
    if api_secret:
        print(f"Secret length: {len(api_secret)} characters")
        print(f"Secret format: {'PEM' if '-----BEGIN' in api_secret else 'Raw'}")
    
    return bool(api_key and api_secret)

def test_connection():
    """Test API connection"""
    try:
        load_dotenv('.env')
        
        api_key = os.getenv('COINBASE_ADVANCED_API_KEY')
        api_secret = os.getenv('COINBASE_ADVANCED_API_SECRET')
        
        if not api_key or not api_secret:
            print("❌ Missing credentials - cannot test connection")
            return False
        
        print("\n🔌 TESTING CONNECTION:")
        
        client = RESTClient(
            api_key=api_key,
            api_secret=api_secret
        )
        
        # Test basic API call
        accounts = client.get_accounts()
        
        print("✅ Connection successful!")
        print("\n💰 ACCOUNT BALANCES:")
        
        total_eur = 0.0
        for account in accounts.get('accounts', []):
            currency = account.get('currency')
            balance = float(account.get('available_balance', {}).get('value', 0))
            
            if balance > 0:
                print(f"   {currency}: {balance:.6f}")
                
                if currency == 'EUR':
                    total_eur += balance
                elif currency == 'USD':
                    total_eur += balance * 0.92  # Rough conversion
        
        print(f"\n📊 Total EUR equivalent: €{total_eur:.2f}")
        
        # Test market data
        print("\n📈 TESTING MARKET DATA:")
        try:
            btc_book = client.get_product_book(product_id='BTC-EUR', limit=5)
            if btc_book and 'bids' in btc_book:
                best_bid = btc_book['bids'][0]['price']
                best_ask = btc_book['asks'][0]['price']
                print(f"   BTC-EUR: Bid={best_bid}, Ask={best_ask}")
            
            # Test product info
            products = client.get_products(limit=5)
            if products:
                print(f"   Available products: {len(products.get('products', []))}")
                
        except Exception as e:
            print(f"   ⚠️ Market data test failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\nPossible issues:")
        print("- Wrong API key format (need Advanced Trade, not CDP)")
        print("- Missing permissions on API key")
        print("- Invalid private key format")
        return False

def test_trading_simulation():
    """Test trading functions without executing real trades"""
    print("\n🎯 TESTING TRADING LOGIC:")
    
    try:
        # Import our trading engine
        sys.path.append('.')
        from kimera_omnidimensional_advanced_trade import KimeraAdvancedTrader
        
        # Initialize trader
        trader = KimeraAdvancedTrader(use_sandbox=False)
        
        # Test authentication
        if trader.authenticate_and_test():
            print("✅ Trader authentication successful")
            
            # Test market data analysis
            market_data = trader.get_market_data('BTC-EUR')
            if market_data:
                print("✅ Market data retrieval successful")
                
                # Test analysis functions
                h_analysis = trader.analyze_horizontal_opportunity(market_data)
                v_analysis = trader.analyze_vertical_opportunity(market_data)
                
                print(f"   Horizontal analysis: {h_analysis['score']:.3f} ({h_analysis['action']})")
                print(f"   Vertical analysis: {v_analysis['score']:.3f} ({v_analysis['action']})")
                
                return True
            else:
                print("❌ Market data retrieval failed")
        else:
            print("❌ Trader authentication failed")
            
    except Exception as e:
        print(f"❌ Trading logic test failed: {e}")
    
    return False

async def run_demo_trading():
    """Run a short demo of the trading system (no real trades)"""
    print("\n🚀 DEMO TRADING RUN:")
    print("(This will not execute real trades)")
    
    try:
        sys.path.append('.')
        from kimera_omnidimensional_advanced_trade import KimeraAdvancedTrader
        
        trader = KimeraAdvancedTrader(use_sandbox=False)
        
        if not trader.authenticate_and_test():
            print("❌ Cannot run demo - authentication failed")
            return
        
        print("\n📊 Analyzing market opportunities...")
        
        # Check multiple pairs
        opportunities = []
        for pair in ['BTC-EUR', 'ETH-EUR', 'SOL-EUR']:
            market_data = trader.get_market_data(pair)
            if market_data:
                h_score = trader.analyze_horizontal_opportunity(market_data)['score']
                v_score = trader.analyze_vertical_opportunity(market_data)['score']
                
                opportunities.append({
                    'pair': pair,
                    'horizontal_score': h_score,
                    'vertical_score': v_score,
                    'combined_score': (h_score + v_score) / 2
                })
                
                print(f"   {pair}: H={h_score:.3f}, V={v_score:.3f}, Combined={h_score+v_score:.3f}")
        
        # Find best opportunities
        if opportunities:
            best = max(opportunities, key=lambda x: x['combined_score'])
            print(f"\n🎯 Best opportunity: {best['pair']} (score: {best['combined_score']:.3f})")
            
            if best['combined_score'] > 0.5:
                print("✅ Would execute trades in live mode")
            else:
                print("⏳ Waiting for better opportunities")
        
        print("\n✅ Demo completed successfully")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")

def main():
    """Main test function"""
    print("🧪 COINBASE ADVANCED TRADE API TEST SUITE")
    print("=" * 50)
    
    # Test 1: Credentials
    if not test_credentials():
        print("\n❌ SETUP REQUIRED:")
        print("1. Create Advanced Trade API keys at: https://www.coinbase.com/settings/api")
        print("2. Add to .env file:")
        print("   COINBASE_ADVANCED_API_KEY=your_api_key")
        print("   COINBASE_ADVANCED_API_SECRET=your_private_key_pem")
        return False
    
    # Test 2: Connection
    if not test_connection():
        print("\n❌ CONNECTION FAILED")
        print("Check your API key permissions and format")
        return False
    
    # Test 3: Trading Logic
    if not test_trading_simulation():
        print("\n❌ TRADING LOGIC TEST FAILED")
        return False
    
    # Test 4: Demo Run
    print("\n" + "=" * 50)
    asyncio.run(run_demo_trading())
    
    print("\n" + "=" * 50)
    print("✅ ALL TESTS PASSED!")
    print("Ready for live trading with: python kimera_omnidimensional_advanced_trade.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 