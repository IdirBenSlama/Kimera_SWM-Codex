#!/usr/bin/env python3
"""
TEST NOTIONAL REQUIREMENTS
=========================
Find exact minimum notional value by testing different sizes
"""

import os
import ccxt
from dotenv import load_dotenv

load_dotenv()

def test_notional_requirements():
    """Test different trade sizes to find minimum notional"""
    print("🔬 TESTING NOTIONAL REQUIREMENTS")
    print("=" * 50)
    
    try:
        exchange = ccxt.binance({
            'apiKey': os.getenv('BINANCE_API_KEY'),
            'secret': os.getenv('BINANCE_SECRET_KEY'),
            'sandbox': False,
            'enableRateLimit': True,
        })
        
        exchange.load_markets()
        
        # Get current BTC price
        ticker = exchange.fetch_ticker('BTC/USDT')
        btc_price = ticker['last']
        print(f"\n📈 Current BTC Price: ${btc_price:,.2f}")
        
        # Test different trade amounts
        test_amounts = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 15.0, 20.0]
        
        print(f"\n🧪 Testing different trade amounts:")
        
        for amount_usd in test_amounts:
            btc_quantity = amount_usd / btc_price
            
            print(f"\n   Testing ${amount_usd:.1f} ({btc_quantity:.8f} BTC)...")
            
            try:
                # Try to create the order (but we'll catch the error)
                order = exchange.create_market_buy_order('BTC/USDT', btc_quantity)
                
                # If we get here, the order was successful
                print(f"   ✅ SUCCESS! Order created: {order['id']}")
                print(f"   ⚠️ WARNING: Real order executed!")
                
                # Immediately sell it back
                balance = exchange.fetch_balance()
                btc_available = balance['BTC']['free']
                if btc_available > 0:
                    sell_order = exchange.create_market_sell_order('BTC/USDT', btc_available)
                    print(f"   🔄 Sold back: {sell_order['id']}")
                
                print(f"   🎯 MINIMUM NOTIONAL FOUND: ${amount_usd:.1f}")
                break
                
            except ccxt.BadRequest as e:
                if "NOTIONAL" in str(e):
                    print(f"   ❌ NOTIONAL failure: ${amount_usd:.1f}")
                else:
                    print(f"   ❌ Other error: {e}")
                    
            except ccxt.InsufficientFunds as e:
                print(f"   ❌ Insufficient funds: {e}")
                break
                
            except Exception as e:
                print(f"   ❌ Unexpected error: {e}")
        
        # Also check exchange info for hidden filters
        print(f"\n🔍 Detailed Exchange Info:")
        try:
            response = exchange.public_get_exchangeinfo()
            
            for symbol_info in response['symbols']:
                if symbol_info['symbol'] == 'BTCUSDT':
                    print(f"   All BTC/USDT Filters:")
                    for filter_info in symbol_info['filters']:
                        print(f"   - {filter_info['filterType']}: {filter_info}")
                    break
                    
        except Exception as e:
            print(f"   Exchange info error: {e}")
            
        # Check current balance
        print(f"\n💰 Current Balance Check:")
        balance = exchange.fetch_balance()
        usdt_balance = balance['USDT']['free']
        print(f"   USDT Available: ${usdt_balance:.2f}")
        
        if usdt_balance < 10:
            print("   💡 Suggestion: You may need more USDT for minimum trade sizes")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    print("⚠️ This will test real trades with real money!")
    print("⚠️ Successful trades will be immediately sold back")
    response = input("Continue? (yes/no): ")
    
    if response.lower() == 'yes':
        test_notional_requirements()
    else:
        print("�� Test cancelled") 