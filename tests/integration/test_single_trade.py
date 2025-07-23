#!/usr/bin/env python3
"""
SINGLE TRADE TEST
================
Test a single trade to identify specific failure points
"""

import os
import ccxt
import time
from dotenv import load_dotenv
import traceback

load_dotenv()

def test_single_trade():
    """Test a single small trade to identify issues"""
    print("🔬 SINGLE TRADE TEST")
    print("=" * 40)
    
    try:
        # Initialize exchange
        exchange = ccxt.binance({
            'apiKey': os.getenv('BINANCE_API_KEY'),
            'secret': os.getenv('BINANCE_SECRET_KEY'),
            'sandbox': False,
            'enableRateLimit': True,
        })
        
        # Get current portfolio
        print("\n1. Getting portfolio...")
        balance = exchange.fetch_balance()
        usdt_balance = balance['USDT']['free']
        print(f"   USDT Available: ${usdt_balance:.2f}")
        
        if usdt_balance < 6:
            print("   ❌ Insufficient USDT for test trade")
            return
        
        # Get BTC price
        print("\n2. Getting BTC price...")
        ticker = exchange.fetch_ticker('BTC/USDT')
        btc_price = ticker['last']
        print(f"   BTC Price: ${btc_price:,.2f}")
        
        # Calculate trade parameters
        trade_amount_usdt = 5.5  # Slightly above minimum
        btc_quantity = trade_amount_usdt / btc_price
        
        print(f"\n3. Trade Parameters:")
        print(f"   Trade Amount: ${trade_amount_usdt:.2f}")
        print(f"   BTC Quantity: {btc_quantity:.8f}")
        
        # Check market limits
        market = exchange.market('BTC/USDT')
        min_amount = market.get('limits', {}).get('amount', {}).get('min', 0)
        min_cost = market.get('limits', {}).get('cost', {}).get('min', 0)
        
        print(f"   Market Min Amount: {min_amount}")
        print(f"   Market Min Cost: ${min_cost}")
        
        # Validate trade
        if btc_quantity < min_amount:
            print(f"   ❌ Quantity {btc_quantity:.8f} below minimum {min_amount}")
            return
        
        if trade_amount_usdt < min_cost:
            print(f"   ❌ Cost ${trade_amount_usdt:.2f} below minimum ${min_cost}")
            return
        
        print("   ✅ Trade parameters valid")
        
        # Execute test trade
        print("\n4. Executing BUY order...")
        try:
            order = exchange.create_market_buy_order('BTC/USDT', btc_quantity)
            
            print(f"   ✅ BUY ORDER SUCCESSFUL!")
            print(f"   Order ID: {order['id']}")
            print(f"   Amount: {order.get('amount', 'N/A')}")
            print(f"   Cost: ${order.get('cost', 0):.2f}")
            print(f"   Fee: ${order.get('fee', {}).get('cost', 0):.4f}")
            
            # Wait a moment then sell back
            print("\n5. Waiting 10 seconds before selling back...")
            time.sleep(10)
            
            # Get updated balance
            balance = exchange.fetch_balance()
            btc_available = balance['BTC']['free']
            
            print(f"   BTC Available: {btc_available:.8f}")
            
            if btc_available > min_amount:
                print("\n6. Executing SELL order...")
                sell_order = exchange.create_market_sell_order('BTC/USDT', btc_available)
                
                print(f"   ✅ SELL ORDER SUCCESSFUL!")
                print(f"   Order ID: {sell_order['id']}")
                print(f"   Amount: {sell_order.get('amount', 'N/A')}")
                print(f"   Received: ${sell_order.get('cost', 0):.2f}")
                print(f"   Fee: ${sell_order.get('fee', {}).get('cost', 0):.4f}")
                
                # Calculate P&L
                buy_cost = order.get('cost', 0)
                sell_received = sell_order.get('cost', 0)
                pnl = sell_received - buy_cost
                pnl_pct = (pnl / buy_cost) * 100 if buy_cost > 0 else 0
                
                print(f"\n📊 TRADE RESULTS:")
                print(f"   Buy Cost: ${buy_cost:.2f}")
                print(f"   Sell Received: ${sell_received:.2f}")
                print(f"   P&L: ${pnl:+.2f}")
                print(f"   P&L %: {pnl_pct:+.3f}%")
                
                if pnl > 0:
                    print("   🎯 PROFITABLE TRADE!")
                else:
                    print("   📊 Small loss (likely due to spread/fees)")
            else:
                print(f"   ⚠️ Insufficient BTC to sell: {btc_available:.8f} < {min_amount}")
                
        except ccxt.InsufficientFunds as e:
            print(f"   ❌ Insufficient Funds: {e}")
        except ccxt.InvalidOrder as e:
            print(f"   ❌ Invalid Order: {e}")
        except ccxt.NetworkError as e:
            print(f"   ❌ Network Error: {e}")
        except Exception as e:
            print(f"   ❌ Trade Error: {e}")
            print(f"   📊 Full traceback:")
            traceback.print_exc()
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    print("⚠️ This will execute a real trade with real money!")
    response = input("Continue? (yes/no): ")
    
    if response.lower() == 'yes':
        test_single_trade()
    else:
        print("�� Test cancelled") 