#!/usr/bin/env python3
"""Quick portfolio check"""
import os
from binance.client import Client
import logging
logger = logging.getLogger(__name__)

# Set credentials
os.environ['BINANCE_API_KEY'] = os.getenv("BINANCE_API_KEY", "")
os.environ['BINANCE_API_SECRET'] = 'qUn5JqSpYz1GDxFj2X3UF23TYgtxKrTsCbDZEoBMYCPbYZgP4siVLyspkB5HAPl7'

logger.info('🚀 KIMERA IMMEDIATE PROFIT SYSTEM')
logger.info('=' * 50)

# Quick portfolio check
client = Client(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET'))
account = client.get_account()

logger.info('💰 CURRENT PORTFOLIO:')
total_value = 0
for balance in account['balances']:
    asset = balance['asset']
    free = float(balance['free'])
    if free > 0:
        if asset == 'USDT':
            value = free
        else:
            try:
                ticker = client.get_avg_price(symbol=f'{asset}USDT')
                price = float(ticker['price'])
                value = free * price
            except Exception as e:
                logger.error(f"Error in quick_portfolio_check.py: {e}", exc_info=True)
                raise  # Re-raise for proper error handling
                value = 0
        
        if value > 0.1:
            total_value += value
            logger.info(f'   {asset}: {free:.8f} = ${value:.2f}')

logger.info(f'💵 TOTAL VALUE: ${total_value:.2f}')
logger.info('✅ Portfolio analysis complete')
logger.info()

# Check for profitable opportunities
logger.info('🎯 PROFIT OPPORTUNITIES:')
logger.info('-' * 30)

# Check TRX/ADA volatility for quick profits
symbols = ['TRXUSDT', 'ADAUSDT', 'BNBUSDT']
for symbol in symbols:
    try:
        ticker = client.get_24hr_ticker(symbol=symbol)
        price_change = float(ticker['priceChangePercent'])
        volume = float(ticker['volume'])
        
        if abs(price_change) > 1.0:  # Significant movement
            direction = "UP" if price_change > 0 else "DOWN"
            logger.info(f'   🔥 {symbol}: {price_change:+.2f}% ({direction}) - Vol: {volume/1000000:.1f}M')
    except Exception as e:
        logger.info(f'   ⚠️ {symbol}: Error - {e}')

logger.info()
logger.info('🚀 SYSTEMS READY FOR AUTONOMOUS TRADING') 