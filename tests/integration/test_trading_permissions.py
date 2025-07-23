#!/usr/bin/env python3
"""
Detailed trading permissions test to identify the exact issue
"""

import os
from binance.client import Client
from binance.exceptions import BinanceAPIException

# Set credentials
os.environ['BINANCE_API_KEY'] = 'Y9WyflPyK1tVXnET3CTMvSdCbPia3Nhtd89VYWjS9RaAbQ0KEhHezkcGSCySQ8cL'
os.environ['BINANCE_API_SECRET'] = 'qUn5JqSpYz1GDxFj2X3UF23TYgtxKrTsCbDZEoBMYCPbYZgP4siVLyspkB5HAPl7'

def detailed_permissions_test():
    """Detailed test of trading permissions"""
    try:
        client = Client(
            os.getenv('BINANCE_API_KEY'),
            os.getenv('BINANCE_API_SECRET')
        )
        
        print("DETAILED BINANCE TRADING PERMISSIONS TEST")
        print("=" * 60)
        
        # Test 1: Basic account access
        print("1. Testing Account Access...")
        try:
            account = client.get_account()
            print(f"   ✅ Account access successful")
            print(f"   📊 Account Type: {account.get('accountType', 'UNKNOWN')}")
            print(f"   🔄 Can Trade: {account.get('canTrade', False)}")
            print(f"   📥 Can Withdraw: {account.get('canWithdraw', False)}")
            print(f"   📤 Can Deposit: {account.get('canDeposit', False)}")
        except Exception as e:
            print(f"   ❌ Account access failed: {e}")
            return
        
        # Test 2: Market data access
        print("\n2. Testing Market Data Access...")
        try:
            ticker = client.get_symbol_ticker(symbol='TRXUSDT')
            print(f"   ✅ Market data access successful")
            print(f"   💰 TRX Price: ${float(ticker['price']):.6f}")
        except Exception as e:
            print(f"   ❌ Market data failed: {e}")
        
        # Test 3: Order history access
        print("\n3. Testing Order History Access...")
        try:
            orders = client.get_all_orders(symbol='TRXUSDT', limit=5)
            print(f"   ✅ Order history access successful")
            print(f"   📝 Found {len(orders)} historical orders")
        except Exception as e:
            print(f"   ❌ Order history failed: {e}")
        
        # Test 4: Test order (dry run)
        print("\n4. Testing Order Creation (TEST MODE)...")
        try:
            test_order = client.create_test_order(
                symbol='TRXUSDT',
                side='BUY',
                type='MARKET',
                quoteOrderQty=10
            )
            print(f"   ✅ Test order successful")
        except Exception as e:
            print(f"   ❌ Test order failed: {e}")
        
        # Test 5: Real order attempt (very small amount)
        print("\n5. Testing REAL Order Creation (Small Amount)...")
        try:
            # Get current TRX balance
            trx_balance = float([b['free'] for b in account['balances'] if b['asset'] == 'TRX'][0])
            print(f"   📊 Current TRX Balance: {trx_balance:.2f}")
            
            if trx_balance >= 50:  # Only test if we have at least 50 TRX
                print(f"   🔄 Attempting to sell 10 TRX...")
                
                real_order = client.order_market_sell(
                    symbol='TRXUSDT',
                    quantity=10
                )
                print(f"   ✅ REAL ORDER EXECUTED!")
                print(f"   📋 Order ID: {real_order['orderId']}")
                print(f"   💰 Status: {real_order['status']}")
                print(f"   🎯 THIS PROVES TRADING IS WORKING!")
                
                return True
            else:
                print(f"   ⚠️  Insufficient TRX balance for real order test")
        except BinanceAPIException as e:
            print(f"   ❌ Real order failed: {e}")
            print(f"   🔍 Error Code: {e.code}")
            print(f"   📝 Error Message: {e.message}")
            
            # Analyze specific error codes
            if e.code == -2015:
                print(f"   🚨 DIAGNOSIS: API permissions issue")
                print(f"   💡 SOLUTION: Enable 'Spot & Margin Trading' in API settings")
            elif e.code == -1013:
                print(f"   🚨 DIAGNOSIS: Order size/precision issue")
            elif e.code == -2010:
                print(f"   🚨 DIAGNOSIS: Insufficient balance")
            else:
                print(f"   🚨 DIAGNOSIS: Unknown trading error")
        
        # Test 6: Check API restrictions
        print("\n6. Checking API Restrictions...")
        try:
            api_status = client.get_system_status()
            print(f"   ✅ System Status: {api_status}")
        except Exception as e:
            print(f"   ❌ System status check failed: {e}")
        
        # Test 7: Check IP restrictions
        print("\n7. Checking API Key Restrictions...")
        try:
            # This will show if there are IP restrictions
            exchange_info = client.get_exchange_info()
            print(f"   ✅ Exchange info accessible (no IP restrictions)")
        except Exception as e:
            print(f"   ❌ Exchange info failed (possible IP restriction): {e}")
        
        print("\n" + "=" * 60)
        print("DIAGNOSIS COMPLETE")
        print("If test order works but real order fails with -2015:")
        print("1. Go to Binance API Management")
        print("2. Enable 'Spot & Margin Trading' permission")
        print("3. Confirm IP whitelist settings")
        print("=" * 60)
        
        return False
        
    except Exception as e:
        print(f"Critical error in permissions test: {e}")
        return False

if __name__ == "__main__":
    detailed_permissions_test() 