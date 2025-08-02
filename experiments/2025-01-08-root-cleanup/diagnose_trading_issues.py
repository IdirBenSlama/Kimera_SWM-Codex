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

load_dotenv()

def diagnose_trading_issues():
    """Comprehensive trading diagnostics"""
    print("🔍 KIMERA TRADING DIAGNOSTICS")
    print("=" * 50)
    
    try:
        # Test API connection
        print("\n1. Testing API Connection...")
        api_key = os.getenv('BINANCE_API_KEY')
        secret_key = os.getenv('BINANCE_SECRET_KEY')
        
        if not api_key or not secret_key:
            print("❌ Missing API credentials!")
            return
        
        exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': secret_key,
            'sandbox': False,
            'enableRateLimit': True,
        })
        
        # Test account access
        print("   Testing account access...")
        balance = exchange.fetch_balance()
        print("   ✅ API connection successful")
        
        # Check portfolio
        print("\n2. Current Portfolio Analysis...")
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
                    print(f"   {asset}: {free:.6f} @ ${price:.4f} = ${value:.2f}")
                    
                    if value >= 5.0:  # Binance minimum
                        tradeable_assets.append({
                            'asset': asset,
                            'amount': free,
                            'value': value,
                            'symbol': f"{asset}/USDT" if asset != 'USDT' else None
                        })
        
        print(f"\n   Total Portfolio Value: ${total_value:.2f}")
        print(f"   Tradeable Assets: {len(tradeable_assets)}")
        
        # Check Binance trading requirements
        print("\n3. Trading Requirements Check...")
        print("   Binance minimum order: $5.00")
        print("   Available for trading:")
        
        usdt_available = 0
        for asset_info in tradeable_assets:
            if asset_info['asset'] == 'USDT':
                usdt_available = asset_info['value']
                print(f"   - USDT: ${usdt_available:.2f} (ready to trade)")
            elif asset_info['value'] >= 5.0:
                print(f"   - {asset_info['asset']}: ${asset_info['value']:.2f} (can sell)")
        
        # Test a small trade
        print("\n4. Testing Trade Execution...")
        
        if usdt_available >= 6:
            print("   Attempting small BTC buy test...")
            try:
                # Get BTC price
                btc_ticker = exchange.fetch_ticker('BTC/USDT')
                btc_price = btc_ticker['last']
                
                # Calculate minimum quantity for $5 order
                min_trade_usdt = 5.1  # Slightly above minimum
                btc_quantity = min_trade_usdt / btc_price
                
                print(f"   BTC Price: ${btc_price:,.2f}")
                print(f"   Test quantity: {btc_quantity:.8f} BTC")
                print(f"   Test value: ${min_trade_usdt:.2f}")
                
                # Check if we have enough
                if usdt_available >= min_trade_usdt:
                    print("   ✅ Sufficient funds for test trade")
                    
                    # Get market info to check minimum quantity
                    market = exchange.market('BTC/USDT')
                    min_amount = market.get('limits', {}).get('amount', {}).get('min', 0)
                    
                    print(f"   Market minimum: {min_amount} BTC")
                    
                    if btc_quantity >= min_amount:
                        print("   ✅ Quantity meets minimum requirements")
                        
                        # Don't actually execute, just validate
                        print("   🔍 Trade would be valid - not executing in diagnostic mode")
                    else:
                        print(f"   ❌ Quantity {btc_quantity:.8f} below minimum {min_amount}")
                else:
                    print(f"   ❌ Insufficient USDT: ${usdt_available:.2f} < ${min_trade_usdt:.2f}")
                    
            except Exception as e:
                print(f"   ❌ Trade test failed: {e}")
                traceback.print_exc()
        
        else:
            print("   ❌ Insufficient USDT for trading")
            print("   💡 Need to convert other assets to USDT first")
            
            # Check largest asset for conversion
            largest_asset = None
            largest_value = 0
            
            for asset_info in tradeable_assets:
                if asset_info['asset'] != 'USDT' and asset_info['value'] > largest_value:
                    largest_value = asset_info['value']
                    largest_asset = asset_info
            
            if largest_asset:
                print(f"   💡 Largest asset: {largest_asset['asset']} (${largest_value:.2f})")
                print(f"   💡 Could convert some {largest_asset['asset']} to USDT for trading")
        
        # Check rate limits
        print("\n5. Rate Limit Analysis...")
        rate_limit = exchange.rateLimit
        print(f"   Rate limit: {rate_limit}ms between requests")
        print(f"   Ultra-aggressive frequency: 2000ms (2s)")
        
        if rate_limit > 1000:
            print("   ⚠️ Rate limit may be too restrictive for ultra-aggressive trading")
        else:
            print("   ✅ Rate limit acceptable")
        
        # Check market conditions
        print("\n6. Market Conditions...")
        try:
            tickers = exchange.fetch_tickers(['BTC/USDT', 'ETH/USDT', 'BNB/USDT'])
            
            for symbol, ticker in tickers.items():
                change_24h = ticker.get('percentage', 0)
                volume_24h = ticker.get('quoteVolume', 0)
                
                print(f"   {symbol}: {change_24h:+.2f}% | Vol: ${volume_24h/1000000:.1f}M")
        
        except Exception as e:
            print(f"   ⚠️ Market data error: {e}")
        
        print("\n" + "=" * 50)
        print("🔍 DIAGNOSIS COMPLETE")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        
        if usdt_available < 10:
            print("   1. Convert some assets to USDT for trading flexibility")
        
        if total_value < 50:
            print("   2. Portfolio size may limit trading opportunities")
        
        if len(tradeable_assets) < 3:
            print("   3. Consider diversifying into more tradeable assets")
        
        print("   4. Start with conservative parameters and increase gradually")
        print("   5. Monitor for specific error messages during trading")
        
    except Exception as e:
        print(f"❌ Diagnostic failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    diagnose_trading_issues() 