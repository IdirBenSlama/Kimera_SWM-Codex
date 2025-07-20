#!/usr/bin/env python3
"""
Quick balance checker to see what assets are available
"""

import os
from binance.client import Client

# Set credentials
os.environ['BINANCE_API_KEY'] = 'Y9WyflPyK1tVXnET3CTMvSdCbPia3Nhtd89VYWjS9RaAbQ0KEhHezkcGSCySQ8cL'
os.environ['BINANCE_API_SECRET'] = 'qUn5JqSpYz1GDxFj2X3UF23TYgtxKrTsCbDZEoBMYCPbYZgP4siVLyspkB5HAPl7'

def check_balances():
    """Check all account balances"""
    try:
        client = Client(
            os.getenv('BINANCE_API_KEY'),
            os.getenv('BINANCE_API_SECRET')
        )
        
        print("BINANCE ACCOUNT BALANCES")
        print("=" * 40)
        
        account = client.get_account()
        
        # Show all non-zero balances
        non_zero_balances = []
        for balance in account['balances']:
            free = float(balance['free'])
            locked = float(balance['locked'])
            total = free + locked
            
            if total > 0:
                non_zero_balances.append({
                    'asset': balance['asset'],
                    'free': free,
                    'locked': locked,
                    'total': total
                })
        
        print(f"Found {len(non_zero_balances)} assets with non-zero balances:")
        print()
        
        total_usdt_value = 0
        
        for balance in non_zero_balances:
            asset = balance['asset']
            free = balance['free']
            locked = balance['locked']
            total = balance['total']
            
            print(f"{asset}:")
            print(f"  Free: {free:.6f}")
            print(f"  Locked: {locked:.6f}")
            print(f"  Total: {total:.6f}")
            
            # Get USD value for major assets
            if asset == 'USDT':
                usd_value = total
                print(f"  USD Value: ${usd_value:.2f}")
                total_usdt_value += usd_value
            elif asset in ['BTC', 'ETH', 'BNB', 'TRX', 'ADA', 'XRP']:
                try:
                    symbol = f"{asset}USDT"
                    ticker = client.get_symbol_ticker(symbol=symbol)
                    price = float(ticker['price'])
                    usd_value = total * price
                    print(f"  Price: ${price:.6f}")
                    print(f"  USD Value: ${usd_value:.2f}")
                    total_usdt_value += usd_value
                except Exception as e:
                    print(f"  USD Value: Unable to calculate ({e})")
            
            print()
        
        print("=" * 40)
        print(f"ESTIMATED TOTAL USD VALUE: ${total_usdt_value:.2f}")
        print("=" * 40)
        
        # Check if we can trade
        usdt_balance = float([b['free'] for b in account['balances'] if b['asset'] == 'USDT'][0])
        
        if usdt_balance >= 10:
            print(f"✅ READY TO TRADE: ${usdt_balance:.2f} USDT available")
        elif total_usdt_value >= 10:
            print(f"💡 CONVERT TO USDT: You have ${total_usdt_value:.2f} in other assets")
            print("   Consider converting some assets to USDT for trading")
        else:
            print(f"❌ INSUFFICIENT FUNDS: Only ${total_usdt_value:.2f} total value")
        
        return non_zero_balances, total_usdt_value
        
    except Exception as e:
        print(f"Error checking balances: {e}")
        return [], 0

if __name__ == "__main__":
    check_balances() 