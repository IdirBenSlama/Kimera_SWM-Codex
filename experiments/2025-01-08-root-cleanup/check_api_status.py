#!/usr/bin/env python3
"""
Check Binance API ban status
"""

import requests
import time
from datetime import datetime

def check_api_status():
    """Check if Binance API is accessible"""
    try:
        # Test with a simple server time request (minimal weight)
        response = requests.get("https://api.binance.com/api/v3/time", timeout=5)
        
        if response.status_code == 200:
            server_time = response.json()
            print(f"✅ API ACCESSIBLE - Server time: {datetime.fromtimestamp(server_time['serverTime']/1000)}")
            return True
        elif response.status_code == 418:
            print(f"❌ API BANNED (418 - I'm a teapot)")
            return False
        else:
            print(f"⚠️ API Status: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ API Error: {e}")
        return False

def monitor_ban_status():
    """Monitor API ban status until lifted"""
    print("🔍 Monitoring Binance API ban status...")
    print("⏰ Checking every 30 seconds...")
    
    while True:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"\n[{timestamp}] Checking API status...")
        
        if check_api_status():
            print("🎉 API BAN LIFTED! Ready to trade!")
            break
        else:
            print("⏳ Still banned, waiting 30 seconds...")
            time.sleep(30)

if __name__ == "__main__":
    monitor_ban_status() 