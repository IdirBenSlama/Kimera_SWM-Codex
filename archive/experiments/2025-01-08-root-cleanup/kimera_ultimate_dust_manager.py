#!/usr/bin/env python3
"""
KIMERA ULTIMATE DUST MANAGER
============================
🧹 ABSOLUTE DUST ELIMINATION SYSTEM 🧹
🛡️ BULLETPROOF PORTFOLIO OPTIMIZATION 🛡️

FEATURES:
- Aggressive dust detection and elimination
- Automatic portfolio consolidation
- Pre-trading dust cleanup
- Post-trading optimization
- Zero dust tolerance
"""

import os
import ccxt
import time
import math
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional
import json
from decimal import Decimal, ROUND_DOWN
import logging
logger = logging.getLogger(__name__)

load_dotenv()

class KimeraUltimateDustManager:
    """Ultimate dust management system with zero tolerance for dust"""
    
    def __init__(self):
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.secret_key = os.getenv('BINANCE_SECRET_KEY')
        
        self.exchange = ccxt.binance({
            'apiKey': self.api_key,
            'secret': self.secret_key,
            'sandbox': False,
            'enableRateLimit': True,
        })
        
        self.exchange.load_markets()
        
        # Ultimate dust parameters
        self.dust_threshold = 8.0       # $8.00 - very aggressive dust threshold
        self.min_trade_size = 7.0       # $7.00 minimum trade size
        self.consolidation_target = 20.0 # Try to create $20+ positions
        self.safety_buffer = 1.2        # 20% safety buffer
        
        logger.info("🧹" * 80)
        logger.info("🤖 KIMERA ULTIMATE DUST MANAGER")
        logger.info("🛡️ ZERO DUST TOLERANCE")
        logger.info("🔥 AGGRESSIVE PORTFOLIO OPTIMIZATION")
        logger.info(f"🧹 DUST THRESHOLD: ${self.dust_threshold}")
        logger.info(f"💰 MIN TRADE SIZE: ${self.min_trade_size}")
        logger.info("🧹" * 80)
    
    def analyze_portfolio_dust(self) -> Dict[str, Any]:
        """Comprehensive dust analysis with detailed breakdown"""
        try:
            balance = self.exchange.fetch_balance()
            tickers = self.exchange.fetch_tickers()
            
            dust_assets = []
            tradeable_assets = []
            problematic_assets = []
            total_dust_value = 0
            total_portfolio_value = 0
            
            logger.info("\n🔍 ULTIMATE DUST ANALYSIS:")
            logger.info("-" * 80)
            
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
                                logger.info(f"   ❌ {asset}: No market data - PROBLEMATIC")
                                problematic_assets.append({
                                    'asset': asset,
                                    'amount': free,
                                    'issue': 'No market data'
                                })
                                continue
                        
                        total_portfolio_value += value
                        
                        asset_data = {
                            'asset': asset,
                            'amount': free,
                            'price': price,
                            'value': value,
                            'symbol': symbol if asset != 'USDT' else None
                        }
                        
                        # Categorize assets
                        if value < self.dust_threshold and asset != 'USDT':
                            dust_assets.append(asset_data)
                            total_dust_value += value
                            logger.info(f"   🧹 {asset}: {free:.8f} = ${value:.2f} (DUST)")
                        elif asset != 'USDT':
                            # Check if it's actually tradeable
                            try:
                                market = self.exchange.market(symbol)
                                min_amount = market.get('limits', {}).get('amount', {}).get('min', 0)
                                
                                # Check if we can trade meaningful amounts
                                test_sell_amount = free * 0.5
                                test_sell_value = test_sell_amount * price
                                
                                if (test_sell_amount >= min_amount * 2 and 
                                    test_sell_value >= self.min_trade_size * self.safety_buffer):
                                    tradeable_assets.append(asset_data)
                                    logger.info(f"   ✅ {asset}: {free:.8f} = ${value:.2f} (TRADEABLE)")
                                else:
                                    # It's above dust threshold but not properly tradeable
                                    problematic_assets.append({
                                        'asset': asset,
                                        'amount': free,
                                        'value': value,
                                        'issue': 'Above dust threshold but not tradeable'
                                    })
                                    logger.info(f"   ⚠️ {asset}: {free:.8f} = ${value:.2f} (PROBLEMATIC)")
                                    
                            except Exception as e:
                                problematic_assets.append({
                                    'asset': asset,
                                    'amount': free,
                                    'value': value,
                                    'issue': f'Market validation failed: {e}'
                                })
                                logger.info(f"   ❌ {asset}: Market validation failed")
                        else:
                            tradeable_assets.append(asset_data)
                            logger.info(f"   ✅ {asset}: {free:.8f} = ${value:.2f} (USDT)")
            
            dust_percentage = (total_dust_value / total_portfolio_value) * 100 if total_portfolio_value > 0 else 0
            
            logger.info("-" * 80)
            logger.info(f"💰 Total Portfolio: ${total_portfolio_value:.2f}")
            logger.info(f"🧹 Total Dust: ${total_dust_value:.2f} ({dust_percentage:.1f}%)")
            logger.info(f"✅ Tradeable Assets: {len(tradeable_assets)}")
            logger.info(f"⚠️ Problematic Assets: {len(problematic_assets)}")
            logger.info(f"🧹 Dust Assets: {len(dust_assets)}")
            
            return {
                'dust_assets': dust_assets,
                'tradeable_assets': tradeable_assets,
                'problematic_assets': problematic_assets,
                'total_dust_value': total_dust_value,
                'total_portfolio_value': total_portfolio_value,
                'dust_percentage': dust_percentage,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.info(f"❌ Dust analysis failed: {e}")
            return {}
    
    def eliminate_dust_by_conversion(self, dust_assets: List[Dict]) -> bool:
        """Eliminate dust by converting to USDT where possible"""
        try:
            logger.info(f"\n🔄 DUST ELIMINATION BY CONVERSION:")
            logger.info("-" * 80)
            
            total_converted = 0
            successful_conversions = 0
            
            for dust in dust_assets:
                if dust['asset'] in ['USDT', 'BNB']:
                    continue
                
                symbol = dust['symbol']
                amount = dust['amount']
                value = dust['value']
                
                logger.info(f"   🔄 Converting {dust['asset']}: {amount:.8f} = ${value:.2f}")
                
                try:
                    # Check if conversion is possible
                    market = self.exchange.market(symbol)
                    min_amount = market.get('limits', {}).get('amount', {}).get('min', 0)
                    
                    if amount >= min_amount and value >= 1.0:  # At least $1 to make it worthwhile
                        # Execute conversion
                        order = self.exchange.create_market_sell_order(symbol, amount)
                        
                        received_usdt = order.get('cost', 0)
                        total_converted += received_usdt
                        successful_conversions += 1
                        
                        logger.info(f"      ✅ Converted to ${received_usdt:.2f} USDT")
                        time.sleep(1)  # Brief pause between conversions
                        
                    else:
                        logger.info(f"      ⚠️ Too small to convert (min: {min_amount:.8f})")
                        
                except Exception as e:
                    logger.info(f"      ❌ Conversion failed: {e}")
            
            logger.info("-" * 80)
            logger.info(f"✅ Converted {successful_conversions} dust assets")
            logger.info(f"💰 Total USDT gained: ${total_converted:.2f}")
            
            return successful_conversions > 0
            
        except Exception as e:
            logger.info(f"❌ Dust conversion failed: {e}")
            return False
    
    def consolidate_small_positions(self, analysis: Dict) -> bool:
        """Consolidate small positions into larger, more tradeable ones"""
        try:
            logger.info(f"\n🔗 POSITION CONSOLIDATION:")
            logger.info("-" * 80)
            
            problematic_assets = analysis.get('problematic_assets', [])
            
            if not problematic_assets:
                logger.info("   ✅ No problematic assets to consolidate")
                return True
            
            total_consolidated_value = 0
            
            for asset_info in problematic_assets:
                if asset_info.get('value', 0) >= 2.0:  # Only consolidate assets worth $2+
                    asset = asset_info['asset']
                    amount = asset_info['amount']
                    value = asset_info['value']
                    
                    logger.info(f"   🔄 Consolidating {asset}: {amount:.8f} = ${value:.2f}")
                    
                    try:
                        symbol = f"{asset}/USDT"
                        order = self.exchange.create_market_sell_order(symbol, amount)
                        
                        received_usdt = order.get('cost', 0)
                        total_consolidated_value += received_usdt
                        
                        logger.info(f"      ✅ Consolidated to ${received_usdt:.2f} USDT")
                        time.sleep(1)
                        
                    except Exception as e:
                        logger.info(f"      ❌ Consolidation failed: {e}")
            
            logger.info("-" * 80)
            logger.info(f"💰 Total consolidated value: ${total_consolidated_value:.2f}")
            
            return total_consolidated_value > 0
            
        except Exception as e:
            logger.info(f"❌ Position consolidation failed: {e}")
            return False
    
    def optimize_portfolio_for_trading(self) -> Dict[str, Any]:
        """Optimize entire portfolio for maximum trading efficiency"""
        try:
            logger.info(f"\n⚡ PORTFOLIO OPTIMIZATION FOR TRADING:")
            logger.info("-" * 80)
            
            # Step 1: Analyze current state
            analysis = self.analyze_portfolio_dust()
            
            # Step 2: Eliminate dust
            if analysis.get('dust_assets'):
                logger.info(f"\n🧹 Eliminating {len(analysis['dust_assets'])} dust assets...")
                self.eliminate_dust_by_conversion(analysis['dust_assets'])
            
            # Step 3: Consolidate problematic positions
            if analysis.get('problematic_assets'):
                logger.info(f"\n🔗 Consolidating {len(analysis['problematic_assets'])} problematic assets...")
                self.consolidate_small_positions(analysis)
            
            # Step 4: Re-analyze after optimization
            time.sleep(3)  # Allow balance to update
            final_analysis = self.analyze_portfolio_dust()
            
            # Step 5: Create optimization report
            optimization_report = {
                'optimization_timestamp': datetime.now().isoformat(),
                'before': {
                    'total_value': analysis.get('total_portfolio_value', 0),
                    'dust_value': analysis.get('total_dust_value', 0),
                    'dust_percentage': analysis.get('dust_percentage', 0),
                    'tradeable_assets': len(analysis.get('tradeable_assets', [])),
                    'dust_assets': len(analysis.get('dust_assets', [])),
                    'problematic_assets': len(analysis.get('problematic_assets', []))
                },
                'after': {
                    'total_value': final_analysis.get('total_portfolio_value', 0),
                    'dust_value': final_analysis.get('total_dust_value', 0),
                    'dust_percentage': final_analysis.get('dust_percentage', 0),
                    'tradeable_assets': len(final_analysis.get('tradeable_assets', [])),
                    'dust_assets': len(final_analysis.get('dust_assets', [])),
                    'problematic_assets': len(final_analysis.get('problematic_assets', []))
                }
            }
            
            # Calculate improvements
            dust_reduction = optimization_report['before']['dust_percentage'] - optimization_report['after']['dust_percentage']
            tradeable_increase = optimization_report['after']['tradeable_assets'] - optimization_report['before']['tradeable_assets']
            
            logger.info(f"\n📊 OPTIMIZATION RESULTS:")
            logger.info("-" * 80)
            logger.info(f"🧹 Dust Reduction: {dust_reduction:.1f}% points")
            logger.info(f"✅ Tradeable Assets: {optimization_report['before']['tradeable_assets']} → {optimization_report['after']['tradeable_assets']}")
            logger.info(f"🧹 Dust Assets: {optimization_report['before']['dust_assets']} → {optimization_report['after']['dust_assets']}")
            logger.info(f"⚠️ Problematic Assets: {optimization_report['before']['problematic_assets']} → {optimization_report['after']['problematic_assets']}")
            
            return optimization_report
            
        except Exception as e:
            logger.info(f"❌ Portfolio optimization failed: {e}")
            return {}
    
    def pre_trading_cleanup(self) -> bool:
        """Complete pre-trading cleanup to ensure clean portfolio"""
        try:
            logger.info(f"\n🚀 PRE-TRADING CLEANUP:")
            logger.info("=" * 80)
            
            # Step 1: Full portfolio optimization
            optimization_report = self.optimize_portfolio_for_trading()
            
            # Step 2: Verify portfolio is clean
            final_analysis = self.analyze_portfolio_dust()
            dust_percentage = final_analysis.get('dust_percentage', 0)
            
            if dust_percentage < 2.0:  # Less than 2% dust is acceptable
                logger.info(f"\n✅ PORTFOLIO CLEAN FOR TRADING!")
                logger.info(f"🧹 Dust: {dust_percentage:.1f}% (ACCEPTABLE)")
                return True
            else:
                logger.info(f"\n⚠️ PORTFOLIO STILL HAS DUST!")
                logger.info(f"🧹 Dust: {dust_percentage:.1f}% (TOO HIGH)")
                return False
                
        except Exception as e:
            logger.info(f"❌ Pre-trading cleanup failed: {e}")
            return False
    
    def post_trading_cleanup(self) -> bool:
        """Post-trading cleanup to maintain clean portfolio"""
        try:
            logger.info(f"\n🧹 POST-TRADING CLEANUP:")
            logger.info("=" * 80)
            
            # Quick dust check and cleanup
            analysis = self.analyze_portfolio_dust()
            
            if analysis.get('dust_assets'):
                logger.info(f"🧹 Cleaning up {len(analysis['dust_assets'])} new dust assets...")
                self.eliminate_dust_by_conversion(analysis['dust_assets'])
                return True
            else:
                logger.info("✅ No post-trading dust detected")
                return True
                
        except Exception as e:
            logger.info(f"❌ Post-trading cleanup failed: {e}")
            return False

def main():
    """Main function for dust manager"""
    logger.info("🧹" * 80)
    logger.info("🚨 KIMERA ULTIMATE DUST MANAGER")
    logger.info("🛡️ ZERO DUST TOLERANCE")
    logger.info("🧹" * 80)
    
    manager = KimeraUltimateDustManager()
    
    logger.info("\nSelect operation:")
    logger.info("1. Analyze dust")
    logger.info("2. Pre-trading cleanup")
    logger.info("3. Post-trading cleanup")
    logger.info("4. Full portfolio optimization")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        manager.analyze_portfolio_dust()
    elif choice == "2":
        manager.pre_trading_cleanup()
    elif choice == "3":
        manager.post_trading_cleanup()
    elif choice == "4":
        manager.optimize_portfolio_for_trading()
    else:
        logger.info("❌ Invalid choice")

if __name__ == "__main__":
    main() 