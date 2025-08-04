#!/usr/bin/env python3
"""
Check Binance API ban status
"""

import requests
import time
from datetime import datetime
import logging
logger = logging.getLogger(__name__)

def check_api_status():
    """Check if Binance API is accessible"""
    try:
        # Test with a simple server time request (minimal weight)
        response = requests.get("https://api.binance.com/api/v3/time", timeout=5)
        
        if response.status_code == 200:
            server_time = response.json()
            logger.info(f"✅ API ACCESSIBLE - Server time: {datetime.fromtimestamp(server_time['serverTime']/1000)}")
            return True
        elif response.status_code == 418:
            logger.info(f"❌ API BANNED (418 - I'm a teapot)")
            return False
        else:
            logger.info(f"⚠️ API Status: {response.status_code}")
            return False
            
    except Exception as e:
        logger.info(f"❌ API Error: {e}")
        return False

def monitor_ban_status():
    """Monitor API ban status until lifted"""
    logger.info("🔍 Monitoring Binance API ban status...")
    logger.info("⏰ Checking every 30 seconds...")
    
    while True:
        timestamp = datetime.now().strftime("%H:%M:%S")
        logger.info(f"\n[{timestamp}] Checking API status...")
        
        if check_api_status():
            logger.info("🎉 API BAN LIFTED! Ready to trade!")
            break
        else:
            logger.info("⏳ Still banned, waiting 30 seconds...")
            time.sleep(30)

if __name__ == "__main__":
    monitor_ban_status() 