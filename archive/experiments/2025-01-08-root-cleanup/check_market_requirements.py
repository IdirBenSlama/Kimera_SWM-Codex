#!/usr/bin/env python3
"""
CHECK MARKET REQUIREMENTS
========================
Check exact Binance market requirements
"""

import os
import ccxt
from dotenv import load_dotenv
import logging
logger = logging.getLogger(__name__)

load_dotenv()

def check_market_requirements():
    """Check exact market requirements"""
    logger.info("🔍 CHECKING MARKET REQUIREMENTS")
    logger.info("=" * 50)
    
    try:
        exchange = ccxt.binance({
            'apiKey': os.getenv('BINANCE_API_KEY'),
            'secret': os.getenv('BINANCE_SECRET_KEY'),
            'sandbox': False,
            'enableRateLimit': True,
        })
        
        # Load markets first
        logger.info("\n⏳ Loading markets...")
        exchange.load_markets()
        
        # Check BTC/USDT requirements
        logger.info("\n📊 BTC/USDT Market Requirements:")
        market = exchange.market('BTC/USDT')
        
        logger.info(f"   Symbol: {market['symbol']}")
        logger.info(f"   Base: {market['base']}")
        logger.info(f"   Quote: {market['quote']}")
        logger.info(f"   Active: {market['active']}")
        
        limits = market.get('limits', {})
        
        # Amount limits
        amount_limits = limits.get('amount', {})
        logger.info(f"\n💰 Amount Limits:")
        logger.info(f"   Min: {amount_limits.get('min', 'N/A')}")
        logger.info(f"   Max: {amount_limits.get('max', 'N/A')}")
        
        # Cost limits
        cost_limits = limits.get('cost', {})
        logger.info(f"\n💵 Cost Limits:")
        logger.info(f"   Min: ${cost_limits.get('min', 'N/A')}")
        logger.info(f"   Max: ${cost_limits.get('max', 'N/A')}")
        
        # Price limits
        price_limits = limits.get('price', {})
        logger.info(f"\n💲 Price Limits:")
        logger.info(f"   Min: ${price_limits.get('min', 'N/A')}")
        logger.info(f"   Max: ${price_limits.get('max', 'N/A')}")
        
        # Market precision
        precision = market.get('precision', {})
        logger.info(f"\n🎯 Precision:")
        logger.info(f"   Amount: {precision.get('amount', 'N/A')}")
        logger.info(f"   Price: {precision.get('price', 'N/A')}")
        
        # Get current price
        ticker = exchange.fetch_ticker('BTC/USDT')
        current_price = ticker['last']
        logger.info(f"\n📈 Current BTC Price: ${current_price:,.2f}")
        
        # Calculate minimum trade values
        min_amount = amount_limits.get('min', 0)
        min_cost = cost_limits.get('min', 0)
        
        min_value_by_amount = min_amount * current_price
        
        logger.info(f"\n🧮 Minimum Trade Calculations:")
        logger.info(f"   Min by amount: {min_amount} BTC = ${min_value_by_amount:.2f}")
        logger.info(f"   Min by cost: ${min_cost:.2f}")
        logger.info(f"   Effective minimum: ${max(min_value_by_amount, min_cost):.2f}")
        
        # Check other popular pairs
        popular_pairs = ['ETH/USDT', 'BNB/USDT', 'TRX/USDT', 'ADA/USDT']
        
        logger.info(f"\n📋 Other Popular Pairs:")
        for symbol in popular_pairs:
            try:
                market = exchange.market(symbol)
                ticker = exchange.fetch_ticker(symbol)
                
                amount_min = market.get('limits', {}).get('amount', {}).get('min', 0)
                cost_min = market.get('limits', {}).get('cost', {}).get('min', 0)
                price = ticker['last']
                
                min_value = max(amount_min * price, cost_min)
                
                logger.info(f"   {symbol}: Min ${min_value:.2f} (Price: ${price:.4f})")
                
            except Exception as e:
                logger.info(f"   {symbol}: Error - {e}")
        
        # Check exchange info
        logger.info(f"\n⚙️ Exchange Info:")
        try:
            # Get exchange info for detailed filters
            response = exchange.public_get_exchangeinfo()
            
            for symbol_info in response['symbols']:
                if symbol_info['symbol'] == 'BTCUSDT':
                    logger.info(f"   BTC/USDT Filters:")
                    for filter_info in symbol_info['filters']:
                        if filter_info['filterType'] == 'NOTIONAL':
                            logger.info(f"   - NOTIONAL: Min ${filter_info.get('minNotional', 'N/A')}")
                        elif filter_info['filterType'] == 'MIN_NOTIONAL':
                            logger.info(f"   - MIN_NOTIONAL: Min ${filter_info.get('minNotional', 'N/A')}")
                        elif filter_info['filterType'] == 'LOT_SIZE':
                            logger.info(f"   - LOT_SIZE: Min {filter_info.get('minQty', 'N/A')}")
                    break
                    
        except Exception as e:
            logger.info(f"   Exchange info error: {e}")
            
    except Exception as e:
        logger.info(f"❌ Check failed: {e}")

if __name__ == "__main__":
    check_market_requirements() 