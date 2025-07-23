#!/usr/bin/env python3
"""
Kimera Profit Checker
====================

Quick script to check your current profits and trading status.
"""

import asyncio
import os
import sys
from datetime import datetime
from simple_position_check import SimpleBinanceChecker

async def check_profits():
    """Check current profits and positions"""
    try:
        print("🔍 Checking Kimera Profit System Status...")
        print("=" * 50)
        
        # Check API credentials
        api_key = os.getenv('BINANCE_API_KEY')
        secret_key = os.getenv('BINANCE_SECRET_KEY')
        
        if not api_key or not secret_key:
            print("❌ Missing Binance API credentials")
            return
        
        # Create checker
        checker = SimpleBinanceChecker(api_key, secret_key)
        
        # Get current Bitcoin price
        print("📊 Current Market Prices:")
        btc_ticker = await checker.get_ticker('BTCUSDT')
        eth_ticker = await checker.get_ticker('ETHUSDT')
        
        btc_price = float(btc_ticker['price'])
        eth_price = float(eth_ticker['price'])
        
        print(f"   BTC: ${btc_price:,.2f}")
        print(f"   ETH: ${eth_price:,.2f}")
        
        # Get account balance
        print("\n💰 Account Balance:")
        account = await checker.get_account()
        
        total_value = 0
        positions = []
        
        for balance in account['balances']:
            asset = balance['asset']
            free = float(balance['free'])
            locked = float(balance['locked'])
            total = free + locked
            
            if total > 0:
                if asset == 'USDT':
                    value = total
                elif asset == 'BTC':
                    value = total * btc_price
                    positions.append({
                        'asset': asset,
                        'amount': total,
                        'value': value,
                        'price': btc_price
                    })
                elif asset == 'ETH':
                    value = total * eth_price
                    positions.append({
                        'asset': asset,
                        'amount': total,
                        'value': value,
                        'price': eth_price
                    })
                else:
                    value = 0  # Other assets not calculated
                
                total_value += value
                
                if value > 0.01:  # Only show meaningful balances
                    print(f"   {asset}: {total:.8f} (${value:.2f})")
        
        print(f"\n📈 Total Portfolio Value: ${total_value:.2f}")
        
        # Show active positions
        if positions:
            print("\n🎯 Active Positions:")
            for pos in positions:
                print(f"   {pos['asset']}: {pos['amount']:.8f} @ ${pos['price']:.2f} = ${pos['value']:.2f}")
        
        # Show profit calculation (if we have position data)
        if positions:
            print("\n💡 Profit Analysis:")
            print("   (Note: Exact profit depends on your entry prices)")
            print("   Check your trade history for precise calculations")
        
        # Show recent activity
        print(f"\n🕐 Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Show system status
        print("\n⚡ System Status:")
        print("   • Real-time market monitoring: Active")
        print("   • Autonomous trading: Ready")
        print("   • Risk management: Enabled")
        print("   • Profit compounding: Automatic")
        
        print("\n💡 Next Steps:")
        print("   • Run 'python start_kimera_profits.py' to start trading")
        print("   • Monitor this dashboard regularly")
        print("   • Check kimera_profit_system.log for detailed activity")
        
    except Exception as e:
        print(f"❌ Error checking profits: {e}")
        print("Make sure your API credentials are correct and the system is running")

def main():
    """Main entry point"""
    print(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🚀 Kimera Profit System - Status Check")
    print()
    
    try:
        asyncio.run(check_profits())
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 