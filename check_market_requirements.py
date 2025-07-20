#!/usr/bin/env python3
"""
CHECK MARKET REQUIREMENTS
========================
Check exact Binance market requirements
"""

import os
import ccxt
from dotenv import load_dotenv

load_dotenv()

def check_market_requirements():
    """Check exact market requirements"""
    print("🔍 CHECKING MARKET REQUIREMENTS")
    print("=" * 50)
    
    try:
        exchange = ccxt.binance({
            'apiKey': os.getenv('BINANCE_API_KEY'),
            'secret': os.getenv('BINANCE_SECRET_KEY'),
            'sandbox': False,
            'enableRateLimit': True,
        })
        
        # Load markets first
        print("\n⏳ Loading markets...")
        exchange.load_markets()
        
        # Check BTC/USDT requirements
        print("\n📊 BTC/USDT Market Requirements:")
        market = exchange.market('BTC/USDT')
        
        print(f"   Symbol: {market['symbol']}")
        print(f"   Base: {market['base']}")
        print(f"   Quote: {market['quote']}")
        print(f"   Active: {market['active']}")
        
        limits = market.get('limits', {})
        
        # Amount limits
        amount_limits = limits.get('amount', {})
        print(f"\n💰 Amount Limits:")
        print(f"   Min: {amount_limits.get('min', 'N/A')}")
        print(f"   Max: {amount_limits.get('max', 'N/A')}")
        
        # Cost limits
        cost_limits = limits.get('cost', {})
        print(f"\n💵 Cost Limits:")
        print(f"   Min: ${cost_limits.get('min', 'N/A')}")
        print(f"   Max: ${cost_limits.get('max', 'N/A')}")
        
        # Price limits
        price_limits = limits.get('price', {})
        print(f"\n💲 Price Limits:")
        print(f"   Min: ${price_limits.get('min', 'N/A')}")
        print(f"   Max: ${price_limits.get('max', 'N/A')}")
        
        # Market precision
        precision = market.get('precision', {})
        print(f"\n🎯 Precision:")
        print(f"   Amount: {precision.get('amount', 'N/A')}")
        print(f"   Price: {precision.get('price', 'N/A')}")
        
        # Get current price
        ticker = exchange.fetch_ticker('BTC/USDT')
        current_price = ticker['last']
        print(f"\n📈 Current BTC Price: ${current_price:,.2f}")
        
        # Calculate minimum trade values
        min_amount = amount_limits.get('min', 0)
        min_cost = cost_limits.get('min', 0)
        
        min_value_by_amount = min_amount * current_price
        
        print(f"\n🧮 Minimum Trade Calculations:")
        print(f"   Min by amount: {min_amount} BTC = ${min_value_by_amount:.2f}")
        print(f"   Min by cost: ${min_cost:.2f}")
        print(f"   Effective minimum: ${max(min_value_by_amount, min_cost):.2f}")
        
        # Check other popular pairs
        popular_pairs = ['ETH/USDT', 'BNB/USDT', 'TRX/USDT', 'ADA/USDT']
        
        print(f"\n📋 Other Popular Pairs:")
        for symbol in popular_pairs:
            try:
                market = exchange.market(symbol)
                ticker = exchange.fetch_ticker(symbol)
                
                amount_min = market.get('limits', {}).get('amount', {}).get('min', 0)
                cost_min = market.get('limits', {}).get('cost', {}).get('min', 0)
                price = ticker['last']
                
                min_value = max(amount_min * price, cost_min)
                
                print(f"   {symbol}: Min ${min_value:.2f} (Price: ${price:.4f})")
                
            except Exception as e:
                print(f"   {symbol}: Error - {e}")
        
        # Check exchange info
        print(f"\n⚙️ Exchange Info:")
        try:
            # Get exchange info for detailed filters
            response = exchange.public_get_exchangeinfo()
            
            for symbol_info in response['symbols']:
                if symbol_info['symbol'] == 'BTCUSDT':
                    print(f"   BTC/USDT Filters:")
                    for filter_info in symbol_info['filters']:
                        if filter_info['filterType'] == 'NOTIONAL':
                            print(f"   - NOTIONAL: Min ${filter_info.get('minNotional', 'N/A')}")
                        elif filter_info['filterType'] == 'MIN_NOTIONAL':
                            print(f"   - MIN_NOTIONAL: Min ${filter_info.get('minNotional', 'N/A')}")
                        elif filter_info['filterType'] == 'LOT_SIZE':
                            print(f"   - LOT_SIZE: Min {filter_info.get('minQty', 'N/A')}")
                    break
                    
        except Exception as e:
            print(f"   Exchange info error: {e}")
            
    except Exception as e:
        print(f"❌ Check failed: {e}")

if __name__ == "__main__":
    check_market_requirements() 