#!/usr/bin/env python3
"""
KIMERA INTELLIGENT DUST MANAGER
==============================
🧹 AUTOMATIC DUST MANAGEMENT SYSTEM 🧹
- Detects dust balances automatically
- Consolidates dust into tradeable amounts
- Prevents trading issues before they occur
- Optimizes portfolio for maximum trading efficiency
"""

import os
import ccxt
import time
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, List, Any
import json
import logging
logger = logging.getLogger(__name__)

load_dotenv()

class KimeraDustManager:
    """Intelligent dust management system for Kimera"""
    
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
        
        # Dust management parameters
        self.min_trade_size = 6.5  # Minimum notional value
        self.dust_threshold = 5.0  # Consider anything below $5 as dust
        self.consolidation_threshold = 20.0  # Try to consolidate dust into $20+ positions
        
        logger.info("🧹" * 60)
        logger.info("🤖 KIMERA INTELLIGENT DUST MANAGER")
        logger.info("🎯 AUTOMATIC DUST DETECTION & CONSOLIDATION")
        logger.info("⚡ PORTFOLIO OPTIMIZATION FOR TRADING")
        logger.info("🧹" * 60)
    
    def analyze_dust(self) -> Dict[str, Any]:
        """Analyze portfolio for dust and optimization opportunities"""
        try:
            balance = self.exchange.fetch_balance()
            tickers = self.exchange.fetch_tickers()
            
            dust_assets = []
            tradeable_assets = []
            total_dust_value = 0
            total_portfolio_value = 0
            
            logger.info("\n🔍 DUST ANALYSIS:")
            logger.info("-" * 50)
            
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
                        
                        total_portfolio_value += value
                        
                        asset_data = {
                            'asset': asset,
                            'amount': free,
                            'price': price,
                            'value': value,
                            'symbol': symbol if asset != 'USDT' else None
                        }
                        
                        if value < self.dust_threshold and asset != 'USDT':
                            dust_assets.append(asset_data)
                            total_dust_value += value
                            logger.info(f"   🧹 {asset}: {free:.8f} = ${value:.2f} (DUST)")
                        else:
                            tradeable_assets.append(asset_data)
                            logger.info(f"   ✅ {asset}: {free:.8f} = ${value:.2f} (TRADEABLE)")
            
            logger.info("-" * 50)
            logger.info(f"💰 Total Portfolio: ${total_portfolio_value:.2f}")
            logger.info(f"🧹 Total Dust: ${total_dust_value:.2f} ({len(dust_assets)} assets)")
            logger.info(f"✅ Tradeable: ${total_portfolio_value - total_dust_value:.2f} ({len(tradeable_assets)} assets)")
            
            return {
                'dust_assets': dust_assets,
                'tradeable_assets': tradeable_assets,
                'total_dust_value': total_dust_value,
                'total_portfolio_value': total_portfolio_value,
                'dust_percentage': (total_dust_value / total_portfolio_value) * 100 if total_portfolio_value > 0 else 0
            }
            
        except Exception as e:
            logger.info(f"❌ Dust analysis failed: {e}")
            return {}
    
    def consolidate_dust_to_bnb(self, dust_assets: List[Dict]) -> bool:
        """Consolidate dust assets to BNB using Binance's dust conversion"""
        try:
            logger.info(f"\n🔄 CONSOLIDATING DUST TO BNB:")
            logger.info("-" * 50)
            
            # Get list of dust assets that can be converted
            convertible_assets = []
            for dust in dust_assets:
                if dust['asset'] not in ['BNB', 'USDT']:  # Can't convert BNB to BNB or USDT
                    convertible_assets.append(dust['asset'])
            
            if not convertible_assets:
                logger.info("   ⚠️ No convertible dust assets found")
                return False
            
            logger.info(f"   📋 Converting {len(convertible_assets)} dust assets to BNB:")
            for asset in convertible_assets:
                logger.info(f"      - {asset}")
            
            # Use Binance's dust conversion API
            # Note: This requires special API permissions
            try:
                # Attempt dust conversion
                result = self.exchange.sapi_post_asset_dust({
                    'asset': ','.join(convertible_assets)
                })
                
                if result.get('success'):
                    logger.info(f"   ✅ Dust conversion successful!")
                    logger.info(f"   💰 Converted to BNB: {result.get('transferResult', {}).get('totalServiceCharge', 'N/A')}")
                    return True
                else:
                    logger.info(f"   ❌ Dust conversion failed: {result.get('msg', 'Unknown error')}")
                    return False
                    
            except Exception as e:
                logger.info(f"   ⚠️ Dust conversion API not available: {e}")
                logger.info(f"   💡 Manual conversion recommended via Binance web interface")
                return False
                
        except Exception as e:
            logger.info(f"❌ Dust consolidation failed: {e}")
            return False
    
    def consolidate_dust_by_trading(self, dust_assets: List[Dict]) -> bool:
        """Consolidate dust by trading small amounts to USDT"""
        try:
            logger.info(f"\n🔄 CONSOLIDATING DUST VIA TRADING:")
            logger.info("-" * 50)
            
            total_consolidated = 0
            successful_conversions = 0
            
            for dust in dust_assets:
                if dust['asset'] in ['USDT', 'BNB']:
                    continue
                
                symbol = dust['symbol']
                amount = dust['amount']
                value = dust['value']
                
                logger.info(f"   🔄 Converting {dust['asset']}: {amount:.8f} = ${value:.2f}")
                
                try:
                    # Check if we can sell this amount
                    market = self.exchange.market(symbol)
                    min_amount = market.get('limits', {}).get('amount', {}).get('min', 0)
                    
                    if amount >= min_amount and value >= 1.0:  # At least $1 to make it worthwhile
                        # Execute sell order
                        order = self.exchange.create_market_sell_order(symbol, amount)
                        
                        received_usdt = order.get('cost', 0)
                        total_consolidated += received_usdt
                        successful_conversions += 1
                        
                        logger.info(f"      ✅ Sold for ${received_usdt:.2f} USDT")
                        
                    else:
                        logger.info(f"      ⚠️ Too small to trade (min: {min_amount:.8f})")
                        
                except Exception as e:
                    logger.info(f"      ❌ Trade failed: {e}")
            
            logger.info("-" * 50)
            logger.info(f"✅ Consolidated {successful_conversions} assets")
            logger.info(f"💰 Total USDT gained: ${total_consolidated:.2f}")
            
            return successful_conversions > 0
            
        except Exception as e:
            logger.info(f"❌ Trading consolidation failed: {e}")
            return False
    
    def optimize_portfolio_for_trading(self, analysis: Dict) -> Dict[str, Any]:
        """Optimize portfolio structure for maximum trading efficiency"""
        try:
            logger.info(f"\n🎯 PORTFOLIO OPTIMIZATION:")
            logger.info("-" * 50)
            
            recommendations = []
            
            # Analyze dust percentage
            dust_pct = analysis.get('dust_percentage', 0)
            if dust_pct > 5:
                recommendations.append({
                    'type': 'DUST_CONSOLIDATION',
                    'priority': 'HIGH',
                    'action': 'Consolidate dust assets',
                    'reason': f'Dust represents {dust_pct:.1f}% of portfolio'
                })
            
            # Check USDT balance for trading
            usdt_balance = 0
            for asset in analysis.get('tradeable_assets', []):
                if asset['asset'] == 'USDT':
                    usdt_balance = asset['value']
                    break
            
            total_value = analysis.get('total_portfolio_value', 0)
            usdt_percentage = (usdt_balance / total_value) * 100 if total_value > 0 else 0
            
            if usdt_percentage < 10:
                recommendations.append({
                    'type': 'USDT_ALLOCATION',
                    'priority': 'MEDIUM',
                    'action': 'Increase USDT allocation',
                    'reason': f'Only {usdt_percentage:.1f}% in USDT, need more for trading flexibility'
                })
            
            # Check for over-concentration
            for asset in analysis.get('tradeable_assets', []):
                if asset['asset'] != 'USDT':
                    asset_percentage = (asset['value'] / total_value) * 100
                    if asset_percentage > 70:
                        recommendations.append({
                            'type': 'DIVERSIFICATION',
                            'priority': 'MEDIUM',
                            'action': f'Reduce {asset["asset"]} concentration',
                            'reason': f'{asset["asset"]} represents {asset_percentage:.1f}% of portfolio'
                        })
            
            # Display recommendations
            if recommendations:
                logger.info("📋 OPTIMIZATION RECOMMENDATIONS:")
                for i, rec in enumerate(recommendations, 1):
                    priority_emoji = "🔴" if rec['priority'] == 'HIGH' else "🟡" if rec['priority'] == 'MEDIUM' else "🟢"
                    logger.info(f"   {i}. {priority_emoji} {rec['action']}")
                    logger.info(f"      Reason: {rec['reason']}")
                    logger.info()
            else:
                logger.info("✅ Portfolio is well-optimized for trading!")
            
            return {
                'recommendations': recommendations,
                'usdt_percentage': usdt_percentage,
                'dust_percentage': dust_pct
            }
            
        except Exception as e:
            logger.info(f"❌ Portfolio optimization failed: {e}")
            return {}
    
    def auto_dust_management(self) -> bool:
        """Automatically manage dust and optimize portfolio"""
        try:
            logger.info(f"\n🤖 AUTOMATIC DUST MANAGEMENT:")
            logger.info("=" * 60)
            
            # Step 1: Analyze dust
            analysis = self.analyze_dust()
            if not analysis:
                return False
            
            # Step 2: Consolidate dust if significant
            dust_assets = analysis.get('dust_assets', [])
            if len(dust_assets) > 2 or analysis.get('total_dust_value', 0) > 10:
                logger.info(f"\n🔄 Significant dust detected, attempting consolidation...")
                
                # Try BNB conversion first
                if not self.consolidate_dust_to_bnb(dust_assets):
                    # Fall back to trading consolidation
                    self.consolidate_dust_by_trading(dust_assets)
                
                # Re-analyze after consolidation
                time.sleep(2)
                analysis = self.analyze_dust()
            
            # Step 3: Generate optimization recommendations
            optimization = self.optimize_portfolio_for_trading(analysis)
            
            # Step 4: Save results
            results = {
                'timestamp': datetime.now().isoformat(),
                'analysis': analysis,
                'optimization': optimization,
                'dust_managed': len(dust_assets) > 0
            }
            
            filename = f"kimera_dust_management_{int(time.time())}.json"
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"\n💾 Results saved to: {filename}")
            
            return True
            
        except Exception as e:
            logger.info(f"❌ Auto dust management failed: {e}")
            return False
    
    def create_dust_free_portfolio_snapshot(self) -> Dict[str, Any]:
        """Create a clean portfolio snapshot with dust removed"""
        try:
            analysis = self.analyze_dust()
            
            clean_portfolio = {
                'assets': {},
                'total_value': 0,
                'tradeable_count': 0
            }
            
            for asset in analysis.get('tradeable_assets', []):
                clean_portfolio['assets'][asset['asset']] = {
                    'amount': asset['amount'],
                    'price': asset['price'],
                    'value_usd': asset['value'],
                    'tradeable': True,
                    'symbol': asset['symbol']
                }
                clean_portfolio['total_value'] += asset['value']
                clean_portfolio['tradeable_count'] += 1
            
            return clean_portfolio
            
        except Exception as e:
            logger.info(f"❌ Clean portfolio creation failed: {e}")
            return {}

def main():
    logger.info("🧹 KIMERA INTELLIGENT DUST MANAGER")
    logger.info("=" * 60)
    
    dust_manager = KimeraDustManager()
    
    logger.info("\n🔍 Running automatic dust management...")
    success = dust_manager.auto_dust_management()
    
    if success:
        logger.info("\n✅ Dust management completed successfully!")
        
        # Create clean portfolio for trading
        clean_portfolio = dust_manager.create_dust_free_portfolio_snapshot()
        if clean_portfolio:
            logger.info(f"\n🎯 CLEAN PORTFOLIO FOR TRADING:")
            logger.info(f"   💰 Total Value: ${clean_portfolio['total_value']:.2f}")
            logger.info(f"   📊 Tradeable Assets: {clean_portfolio['tradeable_count']}")
            logger.info(f"   🧹 Dust Removed: Portfolio optimized for trading")
    else:
        logger.info("\n⚠️ Dust management encountered issues")
    
    logger.info("\n" + "🧹" * 60)

if __name__ == "__main__":
    main() 