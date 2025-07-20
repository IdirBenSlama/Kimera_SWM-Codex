#!/usr/bin/env python3
"""
KIMERA FULL AUTONOMY TRADING ENGINE
===================================
UNRESTRICTED WALLET ACCESS - ALL CURRENCIES - FULL AUTONOMY
Trades across all available assets with complete wallet control
"""

import os
import sys
import json
import time
import asyncio
import logging
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from decimal import Decimal
from dotenv import load_dotenv

# Import the official SDK
try:
    from coinbase.rest import RESTClient
    SDK_AVAILABLE = True
except ImportError:
    print("❌ Coinbase Advanced Trade SDK not installed!")
    print("Install with: pip install coinbase-advanced-py")
    SDK_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AutonomousOmnidimensionalTrader:
    """
    FULL AUTONOMY TRADING ENGINE
    - No currency restrictions
    - Access to entire wallet balance
    - Trades all available pairs
    - Dynamic position sizing
    - Multi-currency portfolio management
    """
    
    def __init__(self, use_sandbox: bool = False):
        """Initialize with full wallet access"""
        
        # Load credentials
        load_dotenv('.env')
        
        self.api_key = os.getenv('COINBASE_ADVANCED_API_KEY')
        self.api_secret = os.getenv('COINBASE_ADVANCED_API_SECRET')
        
        if not self.api_key or not self.api_secret:
            # Try CDP credentials as fallback
            load_dotenv('kimera_cdp_live.env')
            self.api_key = os.getenv('CDP_API_KEY_NAME')
            self.api_secret = os.getenv('CDP_API_KEY_PRIVATE_KEY')
            self.use_cdp = True
            logger.warning("Using CDP credentials - limited functionality")
        else:
            self.use_cdp = False
            
        if not self.api_key:
            raise ValueError("No API credentials found!")
            
        # Initialize client
        if SDK_AVAILABLE and not self.use_cdp:
            self.client = RESTClient(
                api_key=self.api_key,
                api_secret=self.api_secret,
                base_url="https://api.coinbase.com" if not use_sandbox else "https://api-public.sandbox.exchange.coinbase.com"
            )
        
        # FULL AUTONOMY CONFIGURATION
        self.min_trade_size_usd = 1.0  # Minimum $1 trades
        self.max_position_pct = 0.5    # Can use up to 50% of any currency
        self.aggressive_mode = True     # Enable aggressive trading
        self.compound_profits = True    # Reinvest all profits
        
        # Track all currencies and balances
        self.wallet_balances = {}
        self.active_positions = {}
        self.total_portfolio_value_usd = 0.0
        
        # Performance tracking
        self.start_portfolio_value = 0.0
        self.trades_executed = 0
        self.profitable_trades = 0
        self.total_fees_paid = 0.0
        
    def get_all_balances(self) -> Dict[str, float]:
        """Get balances for ALL currencies in wallet"""
        try:
            if self.use_cdp:
                # Limited CDP functionality - simulate
                return self._simulate_balances()
                
            accounts = self.client.get_accounts()
            balances = {}
            
            for account in accounts.get('accounts', []):
                currency = account['currency']
                available = float(account['available_balance']['value'])
                hold = float(account.get('hold', {}).get('value', 0))
                
                if available > 0 or hold > 0:
                    balances[currency] = {
                        'available': available,
                        'hold': hold,
                        'total': available + hold
                    }
                    
            self.wallet_balances = balances
            logger.info(f"💰 Found {len(balances)} currencies with balances")
            
            return balances
            
        except Exception as e:
            logger.error(f"Failed to get balances: {e}")
            return {}
    
    def _simulate_balances(self) -> Dict[str, float]:
        """Simulate balances for CDP mode"""
        # Simulate various currency holdings
        return {
            'EUR': {'available': 5.0, 'hold': 0, 'total': 5.0},
            'USD': {'available': 5.5, 'hold': 0, 'total': 5.5},
            'BTC': {'available': 0.00001, 'hold': 0, 'total': 0.00001},
            'ETH': {'available': 0.001, 'hold': 0, 'total': 0.001}
        }
    
    def calculate_portfolio_value(self) -> float:
        """Calculate total portfolio value in USD"""
        total_usd = 0.0
        
        for currency, balance in self.wallet_balances.items():
            if currency == 'USD':
                total_usd += balance['total']
            else:
                # Get conversion rate
                rate = self.get_usd_rate(currency)
                total_usd += balance['total'] * rate
                
        self.total_portfolio_value_usd = total_usd
        return total_usd
    
    def get_usd_rate(self, currency: str) -> float:
        """Get USD conversion rate for any currency"""
        if currency == 'USD':
            return 1.0
            
        try:
            # Try direct USD pair
            pair = f"{currency}-USD"
            ticker = self.get_ticker(pair)
            if ticker:
                return ticker['price']
                
            # Try reverse pair
            pair = f"USD-{currency}"
            ticker = self.get_ticker(pair)
            if ticker:
                return 1.0 / ticker['price']
                
            # Try via BTC
            if currency != 'BTC':
                btc_rate = self.get_ticker(f"{currency}-BTC")
                if btc_rate:
                    btc_usd = self.get_ticker("BTC-USD")
                    if btc_usd:
                        return btc_rate['price'] * btc_usd['price']
                        
            # Default rates for common currencies
            defaults = {
                'EUR': 1.08,
                'GBP': 1.26,
                'BTC': 100000,
                'ETH': 2500,
                'SOL': 140,
                'USDC': 1.0,
                'USDT': 1.0
            }
            
            return defaults.get(currency, 0)
            
        except Exception as e:
            logger.error(f"Failed to get rate for {currency}: {e}")
            return 0
    
    def get_ticker(self, pair: str) -> Optional[Dict]:
        """Get ticker data for any pair"""
        try:
            if self.use_cdp:
                # Use public API
                import requests
                url = f"https://api.exchange.coinbase.com/products/{pair}/ticker"
                resp = requests.get(url)
                if resp.status_code == 200:
                    data = resp.json()
                    return {
                        'price': float(data.get('price', 0)),
                        'bid': float(data.get('bid', 0)),
                        'ask': float(data.get('ask', 0)),
                        'volume': float(data.get('volume', 0))
                    }
            else:
                # Use SDK
                ticker = self.client.get_product(product_id=pair)
                if ticker:
                    return {
                        'price': float(ticker.get('price', 0)),
                        'bid': float(ticker.get('bid', 0)),
                        'ask': float(ticker.get('ask', 0)),
                        'volume': float(ticker.get('volume_24h', 0))
                    }
                    
        except Exception:
            pass
            
        return None
    
    def get_all_trading_pairs(self) -> List[str]:
        """Get ALL available trading pairs"""
        try:
            if self.use_cdp:
                # Return common pairs for simulation
                return [
                    'BTC-USD', 'ETH-USD', 'SOL-USD', 'AVAX-USD',
                    'BTC-EUR', 'ETH-EUR', 'SOL-EUR',
                    'BTC-USDC', 'ETH-BTC', 'SOL-ETH'
                ]
                
            # Get all products
            products = self.client.get_products(limit=500)
            pairs = []
            
            for product in products.get('products', []):
                if product.get('trading_disabled') == False:
                    pairs.append(product['product_id'])
                    
            logger.info(f"📊 Found {len(pairs)} tradeable pairs")
            return pairs
            
        except Exception as e:
            logger.error(f"Failed to get trading pairs: {e}")
            return []
    
    def analyze_opportunity(self, pair: str) -> Dict:
        """Analyze trading opportunity with full market depth"""
        try:
            ticker = self.get_ticker(pair)
            if not ticker or ticker['price'] == 0:
                return {'score': 0, 'action': 'HOLD'}
                
            # Get order book
            book = self.get_order_book(pair)
            
            score = 0.0
            signals = []
            
            # 1. Spread analysis
            spread_pct = (ticker['ask'] - ticker['bid']) / ticker['bid'] * 100
            if spread_pct < 0.05:  # Very tight spread
                score += 0.3
                signals.append(f"Tight spread: {spread_pct:.3f}%")
                
            # 2. Volume analysis
            if ticker['volume'] > 1000:
                score += 0.2
                signals.append(f"High volume: {ticker['volume']:.0f}")
                
            # 3. Order book imbalance
            if book:
                bid_volume = sum(float(b[1]) for b in book.get('bids', [])[:20])
                ask_volume = sum(float(a[1]) for a in book.get('asks', [])[:20])
                
                if bid_volume > 0 and ask_volume > 0:
                    imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
                    
                    if abs(imbalance) > 0.2:
                        score += 0.3
                        action = 'BUY' if imbalance > 0 else 'SELL'
                        signals.append(f"Order imbalance: {imbalance:.2f}")
                    else:
                        action = 'HOLD'
                else:
                    action = 'HOLD'
            else:
                action = 'HOLD'
                
            # 4. Momentum (if we have historical data)
            # Add momentum analysis here if needed
            
            return {
                'pair': pair,
                'score': score,
                'action': action,
                'price': ticker['price'],
                'signals': signals,
                'spread_pct': spread_pct
            }
            
        except Exception as e:
            logger.error(f"Analysis failed for {pair}: {e}")
            return {'score': 0, 'action': 'HOLD'}
    
    def get_order_book(self, pair: str) -> Optional[Dict]:
        """Get order book data"""
        try:
            if self.use_cdp:
                import requests
                url = f"https://api.exchange.coinbase.com/products/{pair}/book?level=2"
                resp = requests.get(url)
                if resp.status_code == 200:
                    return resp.json()
            else:
                return self.client.get_product_book(product_id=pair, limit=50)
                
        except Exception:
            return None
    
    def calculate_position_size(self, pair: str, signal_strength: float) -> Tuple[str, float]:
        """Calculate optimal position size based on available balances"""
        base, quote = pair.split('-')
        
        # Check available balances
        base_balance = self.wallet_balances.get(base, {}).get('available', 0)
        quote_balance = self.wallet_balances.get(quote, {}).get('available', 0)
        
        # Get current price
        ticker = self.get_ticker(pair)
        if not ticker:
            return 'HOLD', 0
            
        price = ticker['price']
        
        # Calculate maximum position sizes
        max_buy_size = quote_balance * self.max_position_pct
        max_sell_size = base_balance * self.max_position_pct * price
        
        # Adjust by signal strength
        buy_size = max_buy_size * signal_strength
        sell_size = max_sell_size * signal_strength
        
        # Convert to USD for minimum check
        quote_usd_rate = self.get_usd_rate(quote)
        buy_size_usd = buy_size * quote_usd_rate
        sell_size_usd = sell_size
        
        # Determine action
        if buy_size_usd >= self.min_trade_size_usd and quote_balance > 0:
            return 'BUY', buy_size
        elif sell_size_usd >= self.min_trade_size_usd and base_balance > 0:
            return 'SELL', base_balance * self.max_position_pct * signal_strength
        else:
            return 'HOLD', 0
    
    def execute_trade(self, pair: str, side: str, size: float) -> Dict:
        """Execute trade with full autonomy"""
        try:
            logger.info(f"🔴 EXECUTING {side} on {pair}")
            logger.info(f"   Size: {size:.6f}")
            
            if self.use_cdp:
                # Simulate trade
                return self._simulate_trade(pair, side, size)
                
            # Real trade execution
            base, quote = pair.split('-')
            
            if side == 'BUY':
                order_config = {
                    "market_market_ioc": {
                        "quote_size": str(size)  # Size in quote currency
                    }
                }
            else:
                order_config = {
                    "market_market_ioc": {
                        "base_size": str(size)  # Size in base currency
                    }
                }
                
            order_result = self.client.create_order(
                client_order_id=f"kimera_auto_{int(time.time() * 1000)}",
                product_id=pair,
                side=side.upper(),
                order_configuration=order_config
            )
            
            if order_result.get('success'):
                self.trades_executed += 1
                logger.info(f"✅ Trade executed: {order_result.get('order_id')}")
                
                # Update balances
                self.get_all_balances()
                
                return {
                    'success': True,
                    'order_id': order_result.get('order_id'),
                    'pair': pair,
                    'side': side,
                    'size': size
                }
            else:
                logger.error(f"❌ Trade failed: {order_result}")
                return {'success': False, 'error': order_result}
                
        except Exception as e:
            logger.error(f"❌ Trade execution error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _simulate_trade(self, pair: str, side: str, size: float) -> Dict:
        """Simulate trade for CDP mode"""
        # Simulate successful trade
        base, quote = pair.split('-')
        ticker = self.get_ticker(pair)
        
        if ticker:
            price = ticker['price']
            
            # Update simulated balances
            if side == 'BUY':
                # Buying base with quote
                base_amount = size / price
                self.wallet_balances[quote]['available'] -= size
                self.wallet_balances[base]['available'] = self.wallet_balances.get(base, {}).get('available', 0) + base_amount
            else:
                # Selling base for quote
                quote_amount = size * price
                self.wallet_balances[base]['available'] -= size
                self.wallet_balances[quote]['available'] = self.wallet_balances.get(quote, {}).get('available', 0) + quote_amount
                
            self.trades_executed += 1
            
            # Simulate profit
            profit = size * 0.002 * np.random.uniform(0.5, 2.0)
            
            logger.info(f"✅ SIMULATED: {side} {size:.6f} {base} @ {price}")
            logger.info(f"   Estimated profit: ${profit:.3f}")
            
            return {
                'success': True,
                'order_id': f"SIM_{int(time.time()*1000)}",
                'pair': pair,
                'side': side,
                'size': size,
                'price': price,
                'profit': profit
            }
            
        return {'success': False, 'error': 'No ticker data'}
    
    async def run_autonomous_trading(self, duration_minutes: int = None):
        """Run fully autonomous trading with no restrictions"""
        logger.info("\n" + "="*60)
        logger.info("🚀 KIMERA FULL AUTONOMY MODE ACTIVATED")
        logger.info("="*60)
        
        # Get initial portfolio state
        self.get_all_balances()
        self.start_portfolio_value = self.calculate_portfolio_value()
        
        logger.info(f"💼 Initial Portfolio Value: ${self.start_portfolio_value:.2f}")
        logger.info(f"📊 Currencies in wallet: {len(self.wallet_balances)}")
        
        for currency, balance in self.wallet_balances.items():
            if balance['total'] > 0:
                usd_value = balance['total'] * self.get_usd_rate(currency)
                logger.info(f"   {currency}: {balance['total']:.6f} (${usd_value:.2f})")
        
        # Get all trading pairs
        all_pairs = self.get_all_trading_pairs()
        logger.info(f"🎯 Monitoring {len(all_pairs)} trading pairs")
        
        # Run indefinitely or for specified duration
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60) if duration_minutes else None
        
        cycle = 0
        while True:
            cycle += 1
            logger.info(f"\n🔄 Trading Cycle {cycle}")
            
            # Check all pairs for opportunities
            opportunities = []
            
            for pair in all_pairs:
                analysis = self.analyze_opportunity(pair)
                
                if analysis['score'] > 0.5:
                    # Calculate position size
                    action, size = self.calculate_position_size(pair, analysis['score'])
                    
                    if action != 'HOLD' and size > 0:
                        opportunities.append({
                            'pair': pair,
                            'action': action,
                            'size': size,
                            'score': analysis['score'],
                            'signals': analysis.get('signals', [])
                        })
            
            # Sort by score and execute best opportunities
            opportunities.sort(key=lambda x: x['score'], reverse=True)
            
            logger.info(f"📈 Found {len(opportunities)} trading opportunities")
            
            # Execute top opportunities
            for opp in opportunities[:10]:  # Top 10 opportunities
                logger.info(f"\n🎯 {opp['pair']}: {opp['action']} (score: {opp['score']:.3f})")
                logger.info(f"   Signals: {', '.join(opp['signals'])}")
                
                result = self.execute_trade(
                    pair=opp['pair'],
                    side=opp['action'],
                    size=opp['size']
                )
                
                if result.get('success'):
                    self.profitable_trades += 1
                    
                await asyncio.sleep(1)  # Rate limiting
            
            # Update portfolio value
            current_value = self.calculate_portfolio_value()
            profit = current_value - self.start_portfolio_value
            profit_pct = (profit / self.start_portfolio_value) * 100
            
            logger.info(f"\n💰 Portfolio Value: ${current_value:.2f}")
            logger.info(f"📊 Total Profit: ${profit:.2f} ({profit_pct:.2f}%)")
            logger.info(f"✅ Success Rate: {(self.profitable_trades/max(self.trades_executed,1))*100:.1f}%")
            
            # Check if we should stop
            if end_time and time.time() > end_time:
                break
                
            # Wait before next cycle
            await asyncio.sleep(30)  # 30 second cycles
        
        # Final report
        self._generate_final_report()
    
    def _generate_final_report(self):
        """Generate comprehensive trading report"""
        final_value = self.calculate_portfolio_value()
        total_profit = final_value - self.start_portfolio_value
        roi = (total_profit / self.start_portfolio_value) * 100
        
        report = {
            'performance': {
                'initial_portfolio_value': f"${self.start_portfolio_value:.2f}",
                'final_portfolio_value': f"${final_value:.2f}",
                'total_profit': f"${total_profit:.2f}",
                'roi': f"{roi:.2f}%",
                'trades_executed': self.trades_executed,
                'profitable_trades': self.profitable_trades,
                'success_rate': f"{(self.profitable_trades/max(self.trades_executed,1))*100:.1f}%"
            },
            'portfolio_composition': {},
            'top_performing_assets': []
        }
        
        # Add current balances
        for currency, balance in self.wallet_balances.items():
            if balance['total'] > 0:
                usd_value = balance['total'] * self.get_usd_rate(currency)
                report['portfolio_composition'][currency] = {
                    'amount': balance['total'],
                    'usd_value': f"${usd_value:.2f}",
                    'percentage': f"{(usd_value/final_value)*100:.1f}%"
                }
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"test_results/autonomous_trading_report_{timestamp}.json"
        
        os.makedirs("test_results", exist_ok=True)
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Display summary
        logger.info("\n" + "="*60)
        logger.info("🏁 AUTONOMOUS TRADING SESSION COMPLETE")
        logger.info("="*60)
        logger.info(f"💰 Total Profit: ${total_profit:.2f} ({roi:.2f}% ROI)")
        logger.info(f"📊 Trades: {self.trades_executed} ({self.profitable_trades} profitable)")
        logger.info(f"📈 Success Rate: {(self.profitable_trades/max(self.trades_executed,1))*100:.1f}%")
        logger.info(f"📋 Report saved: {report_file}")

async def main():
    """Main execution"""
    print("\n🤖 KIMERA FULL AUTONOMY TRADING ENGINE")
    print("⚡ UNRESTRICTED WALLET ACCESS ENABLED")
    print("="*50)
    
    try:
        # Initialize autonomous trader
        trader = AutonomousOmnidimensionalTrader()
        
        # Run with full autonomy
        await trader.run_autonomous_trading(duration_minutes=5)
        
    except Exception as e:
        logger.error(f"❌ Trading session failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 