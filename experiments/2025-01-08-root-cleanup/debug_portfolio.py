#!/usr/bin/env python3
"""
DEBUG PORTFOLIO
===============
Show current portfolio and analyze sell amount issues
"""

import os
import ccxt
from dotenv import load_dotenv
import logging
logger = logging.getLogger(__name__)

load_dotenv()

def debug_portfolio():
    """Debug portfolio to understand sell amount issues"""
    logger.info("🔍 DEBUGGING PORTFOLIO AND SELL AMOUNTS")
    logger.info("=" * 60)
    
    try:
        exchange = ccxt.binance({
            'apiKey': os.getenv('BINANCE_API_KEY'),
            'secret': os.getenv('BINANCE_SECRET_KEY'),
            'sandbox': False,
            'enableRateLimit': True,
        })
        
        exchange.load_markets()
        
        # Get portfolio
        balance = exchange.fetch_balance()
        tickers = exchange.fetch_tickers()
        
        logger.info("\n📊 CURRENT PORTFOLIO:")
        logger.info("-" * 60)
        
        total_value = 0
        for asset, info in balance.items():
            if asset not in ['free', 'used', 'total', 'info'] and isinstance(info, dict):
                free = float(info.get('free', 0))
                if free > 0:
                    if asset == 'USDT':
                        price = 1.0
                        value = free
                    else:
                        symbol = f"{asset}/USDT"
                        if symbol in tickers:
                            price = tickers[symbol]['last']
                            value = free * price
                        else:
                            continue
                    
                    total_value += value
                    logger.info(f"{asset:>6}: {free:>15.8f} @ ${price:>10.4f} = ${value:>8.2f}")
                    
                    # Analyze sell amounts if not USDT
                    if asset != 'USDT':
                        symbol = f"{asset}/USDT"
                        try:
                            market = exchange.market(symbol)
                            min_amount = market.get('limits', {}).get('amount', {}).get('min', 0)
                            
                            # Calculate potential sell amounts
                            sell_60 = free * 0.6
                            sell_40 = free * 0.4
                            
                            notional_60 = sell_60 * price
                            notional_40 = sell_40 * price
                            
                            logger.info(f"       Market Min: {min_amount:>15.8f} {asset}")
                            logger.info(f"       Sell 60%:   {sell_60:>15.8f} {asset} = ${notional_60:>8.2f}")
                            logger.info(f"       Sell 40%:   {sell_40:>15.8f} {asset} = ${notional_40:>8.2f}")
                            
                            # Check if sells would be valid
                            valid_60 = sell_60 >= min_amount and notional_60 >= 6.5
                            valid_40 = sell_40 >= min_amount and notional_40 >= 6.5
                            
                            logger.info(f"       60% Valid:  {'✅' if valid_60 else '❌'}")
                            logger.info(f"       40% Valid:  {'✅' if valid_40 else '❌'}")
                            
                            if not valid_60 and not valid_40:
                                logger.info(f"       🚨 PROBLEM: Both sell amounts invalid!")
                                if sell_60 < min_amount:
                                    logger.info(f"          - 60% below min quantity: {sell_60:.8f} < {min_amount}")
                                if notional_60 < 6.5:
                                    logger.info(f"          - 60% below min notional: ${notional_60:.2f} < $6.50")
                            
                            logger.info()
                            
                        except Exception as e:
                            logger.info(f"       Error getting market info: {e}")
                            logger.info()
        
        logger.info("-" * 60)
        logger.info(f"TOTAL VALUE: ${total_value:.2f}")
        
        # Check for dust balances
        logger.info(f"\n🧹 DUST ANALYSIS:")
        dust_count = 0
        for asset, info in balance.items():
            if asset not in ['free', 'used', 'total', 'info'] and isinstance(info, dict):
                free = float(info.get('free', 0))
                if free > 0 and asset != 'USDT':
                    symbol = f"{asset}/USDT"
                    if symbol in tickers:
                        price = tickers[symbol]['last']
                        value = free * price
                        if value < 6.5:  # Below minimum trade size
                            dust_count += 1
                            logger.info(f"   {asset}: {free:.8f} = ${value:.2f} (DUST)")
        
        if dust_count > 0:
            logger.info(f"\n💡 SOLUTION: You have {dust_count} dust balances below $6.50")
            logger.info("   These cannot be traded individually due to minimum requirements")
            logger.info("   Consider using Binance's 'Convert Small Assets to BNB' feature")
        else:
            logger.info("   ✅ No dust balances found")
            
    except Exception as e:
        logger.info(f"❌ Debug failed: {e}")

if __name__ == "__main__":
    debug_portfolio() 