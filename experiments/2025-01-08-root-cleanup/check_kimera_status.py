#!/usr/bin/env python3
"""
Quick status check for Kimera Ultra-Aggressive session
"""

import os
import json
import glob
from datetime import datetime
from binance import Client

def check_session_files():
    """Check for any session result files"""
    pattern = "kimera_ultra_aggressive_session_*.json"
    session_files = glob.glob(pattern)
    
    if session_files:
        latest_file = max(session_files, key=os.path.getctime)
        print(f"📊 Latest session file: {latest_file}")
        
        with open(latest_file, 'r') as f:
            data = json.load(f)
        
        print("📈 Session Results:")
        perf = data.get('performance', {})
        print(f"   Initial: ${perf.get('initial_allocation', 0):.2f}")
        print(f"   Final: ${perf.get('final_value', 0):.2f}")
        print(f"   Profit: ${perf.get('profit', 0):.2f}")
        print(f"   Trades: {perf.get('total_trades', 0)}")
        return True
    
    return False

def check_current_balances():
    """Check current account balances"""
    try:
        api_key = os.getenv("BINANCE_API_KEY", "")
        api_secret = "qUn5JqSpYz1GDxFj2X3UF23TYgtxKrTsCbDZEoBMYCPbYZgP4siVLyspkB5HAPl7"
        
        client = Client(api_key, api_secret)
        account = client.get_account()
        
        print("\n💰 CURRENT BALANCES:")
        total_value = 0
        
        for balance in account['balances']:
            asset = balance['asset']
            free = float(balance['free'])
            
            if free > 0:
                if asset == 'USDT':
                    value = free
                    print(f"   {asset}: {free:.6f} (${value:.2f})")
                    total_value += value
                elif asset in ['TRX', 'BTC', 'ETH', 'BNB']:
                    try:
                        ticker = client.get_symbol_ticker(symbol=f"{asset}USDT")
                        price = float(ticker['price'])
                        value = free * price
                        print(f"   {asset}: {free:.2f} @ ${price:.6f} = ${value:.2f}")
                        total_value += value
                    except Exception as e:
                        logger.error(f"Error in check_kimera_status.py: {e}", exc_info=True)
                        raise  # Re-raise for proper error handling
                        print(f"   {asset}: {free:.6f}")
        
        print(f"\n💰 TOTAL PORTFOLIO VALUE: ${total_value:.2f}")
        return total_value
        
    except Exception as e:
        print(f"❌ Error checking balances: {e}")
        return 0

def main():
    print("🔍 KIMERA ULTRA-AGGRESSIVE SESSION STATUS CHECK")
    print("="*60)
    
    # Check for session files
    session_found = check_session_files()
    
    if not session_found:
        print("⚠️ No completed session files found yet")
        print("🔄 Session may still be running...")
    
    # Check current balances
    total_value = check_current_balances()
    
    # Check if session should be completed
    now = datetime.now()
    print(f"\n⏰ Current time: {now.strftime('%H:%M:%S')}")
    
    # Check log file size/activity
    if os.path.exists('kimera_ultra_aggressive.log'):
        size = os.path.getsize('kimera_ultra_aggressive.log')
        print(f"📝 Log file size: {size} bytes")
        
        # Get last few lines
        with open('kimera_ultra_aggressive.log', 'r') as f:
            lines = f.readlines()
            if lines:
                print(f"📊 Last log entry: {lines[-1].strip()}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main() 