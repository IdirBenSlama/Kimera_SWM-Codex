#!/usr/bin/env python3
"""
KIMERA TRADING DIAGNOSTICS
=========================
Diagnose why trades are failing
"""

import os
import ccxt
import time
from dotenv import load_dotenv
import traceback
import logging
logger = logging.getLogger(__name__)

load_dotenv()

def diagnose_trading_issues():
    """Comprehensive trading diagnostics"""
    logger.info("🔍 KIMERA TRADING DIAGNOSTICS")
    logger.info("=" * 50)
    
    try:
        # Test API connection
        logger.info("\n1. Testing API Connection...")
        api_key = os.getenv('BINANCE_API_KEY')
        secret_key = os.getenv('BINANCE_SECRET_KEY')
        
        if not api_key or not secret_key:
            logger.info("❌ Missing API credentials!")
            return
        
        exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': secret_key,
            'sandbox': False,
            'enableRateLimit': True,
        })
        
        # Test account access
        logger.info("   Testing account access...")
        balance = exchange.fetch_balance()
        logger.info("   ✅ API connection successful")
        
        # Check portfolio
        logger.info("\n2. Current Portfolio Analysis...")
        total_value = 0
        tradeable_assets = []
        
        for asset, info in balance.items():
            if asset not in ['free', 'used', 'total', 'info'] and isinstance(info, dict):
                free = float(info.get('free', 0))
                if free > 0:
                    if asset == 'USDT':
                        value = free
                        price = 1.0
                    else:
                        try:
                            ticker = exchange.fetch_ticker(f"{asset}/USDT")
                            price = ticker['last']
                            value = free * price
                        except Exception as e:
                            logger.error(f"Error in diagnose_trading_issues.py: {e}", exc_info=True)
                            raise  # Re-raise for proper error handling
                            continue
                    
                    total_value += value
                    logger.info(f"   {asset}: {free:.6f} @ ${price:.4f} = ${value:.2f}")
                    
                    if value >= 5.0:  # Binance minimum
                        tradeable_assets.append({
                            'asset': asset,
                            'amount': free,
                            'value': value,
                            'symbol': f"{asset}/USDT" if asset != 'USDT' else None
                        })
        
        logger.info(f"\n   Total Portfolio Value: ${total_value:.2f}")
        logger.info(f"   Tradeable Assets: {len(tradeable_assets)}")
        
        # Check Binance trading requirements
        logger.info("\n3. Trading Requirements Check...")
        logger.info("   Binance minimum order: $5.00")
        logger.info("   Available for trading:")
        
        usdt_available = 0
        for asset_info in tradeable_assets:
            if asset_info['asset'] == 'USDT':
                usdt_available = asset_info['value']
                logger.info(f"   - USDT: ${usdt_available:.2f} (ready to trade)")
            elif asset_info['value'] >= 5.0:
                logger.info(f"   - {asset_info['asset']}: ${asset_info['value']:.2f} (can sell)")
        
        # Test a small trade
        logger.info("\n4. Testing Trade Execution...")
        
        if usdt_available >= 6:
            logger.info("   Attempting small BTC buy test...")
            try:
                # Get BTC price
                btc_ticker = exchange.fetch_ticker('BTC/USDT')
                btc_price = btc_ticker['last']
                
                # Calculate minimum quantity for $5 order
                min_trade_usdt = 5.1  # Slightly above minimum
                btc_quantity = min_trade_usdt / btc_price
                
                logger.info(f"   BTC Price: ${btc_price:,.2f}")
                logger.info(f"   Test quantity: {btc_quantity:.8f} BTC")
                logger.info(f"   Test value: ${min_trade_usdt:.2f}")
                
                # Check if we have enough
                if usdt_available >= min_trade_usdt:
                    logger.info("   ✅ Sufficient funds for test trade")
                    
                    # Get market info to check minimum quantity
                    market = exchange.market('BTC/USDT')
                    min_amount = market.get('limits', {}).get('amount', {}).get('min', 0)
                    
                    logger.info(f"   Market minimum: {min_amount} BTC")
                    
                    if btc_quantity >= min_amount:
                        logger.info("   ✅ Quantity meets minimum requirements")
                        
                        # Don't actually execute, just validate
                        logger.info("   🔍 Trade would be valid - not executing in diagnostic mode")
                    else:
                        logger.info(f"   ❌ Quantity {btc_quantity:.8f} below minimum {min_amount}")
                else:
                    logger.info(f"   ❌ Insufficient USDT: ${usdt_available:.2f} < ${min_trade_usdt:.2f}")
                    
            except Exception as e:
                logger.info(f"   ❌ Trade test failed: {e}")
                traceback.print_exc()
        
        else:
            logger.info("   ❌ Insufficient USDT for trading")
            logger.info("   💡 Need to convert other assets to USDT first")
            
            # Check largest asset for conversion
            largest_asset = None
            largest_value = 0
            
            for asset_info in tradeable_assets:
                if asset_info['asset'] != 'USDT' and asset_info['value'] > largest_value:
                    largest_value = asset_info['value']
                    largest_asset = asset_info
            
            if largest_asset:
                logger.info(f"   💡 Largest asset: {largest_asset['asset']} (${largest_value:.2f})")
                logger.info(f"   💡 Could convert some {largest_asset['asset']} to USDT for trading")
        
        # Check rate limits
        logger.info("\n5. Rate Limit Analysis...")
        rate_limit = exchange.rateLimit
        logger.info(f"   Rate limit: {rate_limit}ms between requests")
        logger.info(f"   Ultra-aggressive frequency: 2000ms (2s)")
        
        if rate_limit > 1000:
            logger.info("   ⚠️ Rate limit may be too restrictive for ultra-aggressive trading")
        else:
            logger.info("   ✅ Rate limit acceptable")
        
        # Check market conditions
        logger.info("\n6. Market Conditions...")
        try:
            tickers = exchange.fetch_tickers(['BTC/USDT', 'ETH/USDT', 'BNB/USDT'])
            
            for symbol, ticker in tickers.items():
                change_24h = ticker.get('percentage', 0)
                volume_24h = ticker.get('quoteVolume', 0)
                
                logger.info(f"   {symbol}: {change_24h:+.2f}% | Vol: ${volume_24h/1000000:.1f}M")
        
        except Exception as e:
            logger.info(f"   ⚠️ Market data error: {e}")
        
        logger.info("\n" + "=" * 50)
        logger.info("🔍 DIAGNOSIS COMPLETE")
        
        # Recommendations
        logger.info("\n💡 RECOMMENDATIONS:")
        
        if usdt_available < 10:
            logger.info("   1. Convert some assets to USDT for trading flexibility")
        
        if total_value < 50:
            logger.info("   2. Portfolio size may limit trading opportunities")
        
        if len(tradeable_assets) < 3:
            logger.info("   3. Consider diversifying into more tradeable assets")
        
        logger.info("   4. Start with conservative parameters and increase gradually")
        logger.info("   5. Monitor for specific error messages during trading")
        
    except Exception as e:
        logger.info(f"❌ Diagnostic failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    diagnose_trading_issues() 