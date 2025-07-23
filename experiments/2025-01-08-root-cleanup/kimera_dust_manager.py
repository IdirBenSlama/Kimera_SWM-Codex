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
        
        print("🧹" * 60)
        print("🤖 KIMERA INTELLIGENT DUST MANAGER")
        print("🎯 AUTOMATIC DUST DETECTION & CONSOLIDATION")
        print("⚡ PORTFOLIO OPTIMIZATION FOR TRADING")
        print("🧹" * 60)
    
    def analyze_dust(self) -> Dict[str, Any]:
        """Analyze portfolio for dust and optimization opportunities"""
        try:
            balance = self.exchange.fetch_balance()
            tickers = self.exchange.fetch_tickers()
            
            dust_assets = []
            tradeable_assets = []
            total_dust_value = 0
            total_portfolio_value = 0
            
            print("\n🔍 DUST ANALYSIS:")
            print("-" * 50)
            
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
                            print(f"   🧹 {asset}: {free:.8f} = ${value:.2f} (DUST)")
                        else:
                            tradeable_assets.append(asset_data)
                            print(f"   ✅ {asset}: {free:.8f} = ${value:.2f} (TRADEABLE)")
            
            print("-" * 50)
            print(f"💰 Total Portfolio: ${total_portfolio_value:.2f}")
            print(f"🧹 Total Dust: ${total_dust_value:.2f} ({len(dust_assets)} assets)")
            print(f"✅ Tradeable: ${total_portfolio_value - total_dust_value:.2f} ({len(tradeable_assets)} assets)")
            
            return {
                'dust_assets': dust_assets,
                'tradeable_assets': tradeable_assets,
                'total_dust_value': total_dust_value,
                'total_portfolio_value': total_portfolio_value,
                'dust_percentage': (total_dust_value / total_portfolio_value) * 100 if total_portfolio_value > 0 else 0
            }
            
        except Exception as e:
            print(f"❌ Dust analysis failed: {e}")
            return {}
    
    def consolidate_dust_to_bnb(self, dust_assets: List[Dict]) -> bool:
        """Consolidate dust assets to BNB using Binance's dust conversion"""
        try:
            print(f"\n🔄 CONSOLIDATING DUST TO BNB:")
            print("-" * 50)
            
            # Get list of dust assets that can be converted
            convertible_assets = []
            for dust in dust_assets:
                if dust['asset'] not in ['BNB', 'USDT']:  # Can't convert BNB to BNB or USDT
                    convertible_assets.append(dust['asset'])
            
            if not convertible_assets:
                print("   ⚠️ No convertible dust assets found")
                return False
            
            print(f"   📋 Converting {len(convertible_assets)} dust assets to BNB:")
            for asset in convertible_assets:
                print(f"      - {asset}")
            
            # Use Binance's dust conversion API
            # Note: This requires special API permissions
            try:
                # Attempt dust conversion
                result = self.exchange.sapi_post_asset_dust({
                    'asset': ','.join(convertible_assets)
                })
                
                if result.get('success'):
                    print(f"   ✅ Dust conversion successful!")
                    print(f"   💰 Converted to BNB: {result.get('transferResult', {}).get('totalServiceCharge', 'N/A')}")
                    return True
                else:
                    print(f"   ❌ Dust conversion failed: {result.get('msg', 'Unknown error')}")
                    return False
                    
            except Exception as e:
                print(f"   ⚠️ Dust conversion API not available: {e}")
                print(f"   💡 Manual conversion recommended via Binance web interface")
                return False
                
        except Exception as e:
            print(f"❌ Dust consolidation failed: {e}")
            return False
    
    def consolidate_dust_by_trading(self, dust_assets: List[Dict]) -> bool:
        """Consolidate dust by trading small amounts to USDT"""
        try:
            print(f"\n🔄 CONSOLIDATING DUST VIA TRADING:")
            print("-" * 50)
            
            total_consolidated = 0
            successful_conversions = 0
            
            for dust in dust_assets:
                if dust['asset'] in ['USDT', 'BNB']:
                    continue
                
                symbol = dust['symbol']
                amount = dust['amount']
                value = dust['value']
                
                print(f"   🔄 Converting {dust['asset']}: {amount:.8f} = ${value:.2f}")
                
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
                        
                        print(f"      ✅ Sold for ${received_usdt:.2f} USDT")
                        
                    else:
                        print(f"      ⚠️ Too small to trade (min: {min_amount:.8f})")
                        
                except Exception as e:
                    print(f"      ❌ Trade failed: {e}")
            
            print("-" * 50)
            print(f"✅ Consolidated {successful_conversions} assets")
            print(f"💰 Total USDT gained: ${total_consolidated:.2f}")
            
            return successful_conversions > 0
            
        except Exception as e:
            print(f"❌ Trading consolidation failed: {e}")
            return False
    
    def optimize_portfolio_for_trading(self, analysis: Dict) -> Dict[str, Any]:
        """Optimize portfolio structure for maximum trading efficiency"""
        try:
            print(f"\n🎯 PORTFOLIO OPTIMIZATION:")
            print("-" * 50)
            
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
                print("📋 OPTIMIZATION RECOMMENDATIONS:")
                for i, rec in enumerate(recommendations, 1):
                    priority_emoji = "🔴" if rec['priority'] == 'HIGH' else "🟡" if rec['priority'] == 'MEDIUM' else "🟢"
                    print(f"   {i}. {priority_emoji} {rec['action']}")
                    print(f"      Reason: {rec['reason']}")
                    print()
            else:
                print("✅ Portfolio is well-optimized for trading!")
            
            return {
                'recommendations': recommendations,
                'usdt_percentage': usdt_percentage,
                'dust_percentage': dust_pct
            }
            
        except Exception as e:
            print(f"❌ Portfolio optimization failed: {e}")
            return {}
    
    def auto_dust_management(self) -> bool:
        """Automatically manage dust and optimize portfolio"""
        try:
            print(f"\n🤖 AUTOMATIC DUST MANAGEMENT:")
            print("=" * 60)
            
            # Step 1: Analyze dust
            analysis = self.analyze_dust()
            if not analysis:
                return False
            
            # Step 2: Consolidate dust if significant
            dust_assets = analysis.get('dust_assets', [])
            if len(dust_assets) > 2 or analysis.get('total_dust_value', 0) > 10:
                print(f"\n🔄 Significant dust detected, attempting consolidation...")
                
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
            
            print(f"\n💾 Results saved to: {filename}")
            
            return True
            
        except Exception as e:
            print(f"❌ Auto dust management failed: {e}")
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
            print(f"❌ Clean portfolio creation failed: {e}")
            return {}

def main():
    print("🧹 KIMERA INTELLIGENT DUST MANAGER")
    print("=" * 60)
    
    dust_manager = KimeraDustManager()
    
    print("\n🔍 Running automatic dust management...")
    success = dust_manager.auto_dust_management()
    
    if success:
        print("\n✅ Dust management completed successfully!")
        
        # Create clean portfolio for trading
        clean_portfolio = dust_manager.create_dust_free_portfolio_snapshot()
        if clean_portfolio:
            print(f"\n🎯 CLEAN PORTFOLIO FOR TRADING:")
            print(f"   💰 Total Value: ${clean_portfolio['total_value']:.2f}")
            print(f"   📊 Tradeable Assets: {clean_portfolio['tradeable_count']}")
            print(f"   🧹 Dust Removed: Portfolio optimized for trading")
    else:
        print("\n⚠️ Dust management encountered issues")
    
    print("\n" + "🧹" * 60)

if __name__ == "__main__":
    main() 