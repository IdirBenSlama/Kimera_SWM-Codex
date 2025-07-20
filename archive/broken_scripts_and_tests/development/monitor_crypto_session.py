#!/usr/bin/env python3
"""
KIMERA Crypto Session Monitor
Real-time monitoring of the 6-hour trading session
"""

import time
import json
import os
from datetime import datetime, timedelta
import requests

# Initialize structured logger
from backend.utils.kimera_logger import get_system_logger
logger = get_system_logger(__name__)


def get_crypto_prices():
    """Get current crypto prices"""
    try:
        url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum,solana&vs_currencies=usd"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return {}

def monitor_session():
    """Monitor the trading session"""
    session_start = datetime.now()
    session_end = session_start + timedelta(hours=6)
    
    logger.info("🚀 KIMERA CRYPTO SESSION MONITOR")
    logger.info("=" * 50)
    logger.info(f"Session Start: {session_start.strftime('%H:%M:%S')
    logger.info(f"Session End:   {session_end.strftime('%H:%M:%S')
    logger.info(f"Duration: 6 hours")
    logger.info("=" * 50)
    logger.info()
    
    iteration = 0
    
    while datetime.now() < session_end:
        iteration += 1
        current_time = datetime.now()
        elapsed = (current_time - session_start).total_seconds() / 3600
        remaining = (session_end - current_time).total_seconds() / 3600
        
        # Get market data
        prices = get_crypto_prices()
        btc_price = prices.get('bitcoin', {}).get('usd', 0)
        eth_price = prices.get('ethereum', {}).get('usd', 0)
        sol_price = prices.get('solana', {}).get('usd', 0)
        
        # Check for session reports
        report_files = [f for f in os.listdir('.') if f.startswith('discrete_session_') and f.endswith('.json')]
        
        # Display status
        logger.info(f"⏰ Hour {elapsed:.1f}/6.0 | Remaining: {remaining:.1f}h")
        logger.info(f"📊 BTC: ${btc_price:,.0f} | ETH: ${eth_price:,.0f} | SOL: ${sol_price:.0f}")
        
        if report_files:
            # Read latest report
            latest_report = max(report_files, key=os.path.getmtime)
            try:
                with open(latest_report, 'r') as f:
                    data = json.load(f)
                    balance = data.get('session', {}).get('final_balance', 1.0)
                    return_pct = data.get('session', {}).get('total_return_pct', 0.0)
                    trades = data.get('trading', {}).get('total_trades', 0)
                    risk = data.get('session', {}).get('risk_level', 'LOW')
                    
                    logger.info(f"💰 Balance: ${balance:.4f} | Return: {return_pct:+.2f}% | Trades: {trades} | Risk: {risk}")
            except:
                logger.info("📈 Session active - waiting for trade data...")
        else:
            logger.info("🔄 Session initializing...")
        
        logger.info("-" * 50)
        
        # Wait 10 minutes between updates
        time.sleep(600)
    
    logger.info("🏁 SESSION COMPLETED!")
    
    # Final report
    if report_files:
        latest_report = max(report_files, key=os.path.getmtime)
        try:
            with open(latest_report, 'r') as f:
                data = json.load(f)
                logger.info("\n📊 FINAL RESULTS:")
                logger.info(f"Final Balance: ${data.get('session', {})
                logger.info(f"Total Return: {data.get('session', {})
                logger.info(f"Total Trades: {data.get('trading', {})
                logger.info(f"Risk Level: {data.get('session', {})
        except:
            logger.info("Report file not accessible")

if __name__ == "__main__":
    try:
        monitor_session()
    except KeyboardInterrupt:
        logger.info("\n👋 Monitor stopped by user")
    except Exception as e:
        logger.error(f"\n❌ Monitor error: {e}")