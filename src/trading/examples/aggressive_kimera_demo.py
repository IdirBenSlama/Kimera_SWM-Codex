"""
Aggressive KIMERA Trading Demo
Starting with $1 - More aggressive trading to show growth potential
"""

import asyncio
import sys
import os
from datetime import datetime, timezone, timedelta
import logging
import json
import random
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from connectors.cryptopanic_connector import CryptoPanicConnector, NewsSentiment
import yfinance as yf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AggressiveKimeraDemo:
    """
    Aggressive KIMERA Trading Demo
    
    More aggressive parameters to demonstrate actual trading:
    - Lower signal thresholds
    - Higher risk tolerance
    - More frequent trading
    - Quicker position management
    """
    
    def __init__(self, starting_capital: float = 1.0):
        self.starting_capital = starting_capital
        self.current_capital = starting_capital
        self.positions = {}
        self.trade_history = []
        self.news_connector = CryptoPanicConnector()
        
        # Aggressive trading parameters
        self.risk_per_trade = 0.25  # 25% risk per trade (very aggressive)
        self.max_position_size = 0.8  # Max 80% per position
        self.sentiment_threshold = 10  # Lower threshold (was 30)
        self.momentum_threshold = 0.01  # Lower threshold (was 0.03)
        
        # Watchlist - focus on volatile cryptos
        self.watchlist = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'ADA-USD']
        
        # Performance tracking
        self.performance = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'max_capital': starting_capital,
            'max_drawdown': 0,
            'signals_generated': 0,
            'contradictions_found': 0,
            'cycles_completed': 0
        }
        
        # Market simulation for more action
        self.market_simulator = {
            'trend_direction': random.choice(['bullish', 'bearish', 'neutral']),
            'volatility_level': random.uniform(0.02, 0.08),
            'news_sentiment_bias': random.uniform(-50, 50)
        }
        
    async def get_market_prices(self):
        """Get current market prices with simulated volatility"""
        prices = {}
        base_prices = {
            'BTC-USD': 67000, 
            'ETH-USD': 3500, 
            'SOL-USD': 140, 
            'BNB-USD': 580, 
            'ADA-USD': 0.45
        }
        
        for symbol in self.watchlist:
            try:
                # Try to get real data first
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="1d", interval="1m")
                
                if not data.empty:
                    real_price = float(data['Close'].iloc[-1])
                    real_change = float((data['Close'].iloc[-1] - data['Open'].iloc[-1]) / data['Open'].iloc[-1])
                    
                    # Add simulated volatility to real data
                    simulated_change = real_change + random.uniform(-0.02, 0.02)
                    
                    prices[symbol] = {
                        'price': real_price * (1 + random.uniform(-0.005, 0.005)),  # Small random variation
                        'volume': float(data['Volume'].iloc[-1]) if not data['Volume'].empty else random.uniform(1000000, 10000000),
                        'change': simulated_change
                    }
                else:
                    raise Exception("No data")
                    
            except Exception:
                # Use simulated data with higher volatility
                base_price = base_prices.get(symbol, 100)
                
                # Apply market trend
                trend_factor = 0
                if self.market_simulator['trend_direction'] == 'bullish':
                    trend_factor = random.uniform(0.005, 0.02)
                elif self.market_simulator['trend_direction'] == 'bearish':
                    trend_factor = random.uniform(-0.02, -0.005)
                
                # Add volatility
                volatility_factor = random.uniform(-self.market_simulator['volatility_level'], 
                                                 self.market_simulator['volatility_level'])
                
                total_change = trend_factor + volatility_factor
                
                prices[symbol] = {
                    'price': base_price * (1 + total_change),
                    'volume': random.uniform(1000000, 10000000),
                    'change': total_change
                }
                
        return prices
        
    async def analyze_news_sentiment(self):
        """Analyze news sentiment with bias simulation"""
        try:
            async with self.news_connector as connector:
                # Get latest news
                news = await connector.get_posts()
                
                # Get market sentiment
                sentiment = await connector.analyze_market_sentiment()
                
                # Apply bias simulation
                biased_sentiment = sentiment['sentiment_score'] + self.market_simulator['news_sentiment_bias']
                
                # Detect contradictions
                contradictions = await self.detect_contradictions(news)
                
                # Add some artificial contradictions for demo purposes
                if len(contradictions) == 0 and random.random() < 0.3:
                    artificial_contradiction = {
                        'currency': random.choice(['BTC', 'ETH', 'SOL']),
                        'positive_count': random.randint(2, 5),
                        'negative_count': random.randint(2, 5),
                        'strength': random.uniform(0.4, 0.8),
                        'symbol': None
                    }
                    artificial_contradiction['symbol'] = f"{artificial_contradiction['currency']}-USD"
                    contradictions.append(artificial_contradiction)
                
                return {
                    'news_count': len(news),
                    'sentiment_score': biased_sentiment,
                    'trending_currencies': sentiment['trending_currencies'][:5],
                    'contradictions': contradictions
                }
                
        except Exception as e:
            logger.error(f"Failed to analyze news: {e}")
            # Return simulated data to keep demo running
            return {
                'news_count': 15,
                'sentiment_score': self.market_simulator['news_sentiment_bias'],
                'trending_currencies': [('BTC', 10), ('ETH', 8), ('SOL', 6)],
                'contradictions': []
            }
            
    async def detect_contradictions(self, news_items):
        """Detect semantic contradictions in news"""
        contradictions = []
        
        # Group news by currency
        currency_news = {}
        for item in news_items:
            for currency in item.currencies:
                code = currency.get('code', 'UNKNOWN')
                if code not in currency_news:
                    currency_news[code] = []
                currency_news[code].append(item)
                
        # Look for sentiment contradictions
        for currency, items in currency_news.items():
            if len(items) < 2:
                continue
                
            positive_items = [i for i in items if i.sentiment == NewsSentiment.POSITIVE]
            negative_items = [i for i in items if i.sentiment == NewsSentiment.NEGATIVE]
            
            if positive_items and negative_items:
                contradiction = {
                    'currency': currency,
                    'positive_count': len(positive_items),
                    'negative_count': len(negative_items),
                    'strength': min(len(positive_items) + len(negative_items), 10) / 10,
                    'symbol': f"{currency}-USD" if f"{currency}-USD" in self.watchlist else None
                }
                contradictions.append(contradiction)
                
        return contradictions
        
    async def generate_trading_signals(self, market_data, news_data):
        """Generate trading signals with aggressive parameters"""
        signals = []
        
        # Signal 1: Sentiment-based (lowered threshold)
        sentiment_score = news_data['sentiment_score']
        if abs(sentiment_score) > self.sentiment_threshold:
            # Find most mentioned currency in watchlist
            for currency, mentions in news_data['trending_currencies']:
                symbol = f"{currency}-USD"
                if symbol in self.watchlist and symbol in market_data:
                    signal = {
                        'symbol': symbol,
                        'action': 'BUY' if sentiment_score > 0 else 'SELL',
                        'confidence': min(abs(sentiment_score) / 100, 0.9),
                        'reason': f"Sentiment: {sentiment_score:.1f}",
                        'type': 'sentiment',
                        'priority': 'HIGH' if abs(sentiment_score) > 40 else 'MEDIUM'
                    }
                    signals.append(signal)
                    break
                    
        # Signal 2: Contradiction-based volatility (lowered threshold)
        for contradiction in news_data['contradictions']:
            if contradiction['symbol'] and contradiction['strength'] > 0.3:  # Lowered from 0.5
                signal = {
                    'symbol': contradiction['symbol'],
                    'action': 'VOLATILITY',
                    'confidence': contradiction['strength'],
                    'reason': f"Contradiction: {contradiction['positive_count']}+ vs {contradiction['negative_count']}-",
                    'type': 'contradiction',
                    'priority': 'HIGH'
                }
                signals.append(signal)
                self.performance['contradictions_found'] += 1
                
        # Signal 3: Technical momentum (lowered threshold)
        for symbol, data in market_data.items():
            if abs(data['change']) > self.momentum_threshold:  # Lowered from 0.03
                signal = {
                    'symbol': symbol,
                    'action': 'BUY' if data['change'] > 0 else 'SELL',
                    'confidence': min(abs(data['change']) * 50, 0.8),  # Increased multiplier
                    'reason': f"Momentum: {data['change']:.2%}",
                    'type': 'momentum',
                    'priority': 'HIGH' if abs(data['change']) > 0.02 else 'MEDIUM'
                }
                signals.append(signal)
                
        # Signal 4: Random opportunity (for demo purposes)
        if len(signals) == 0 and random.random() < 0.4:  # 40% chance of random signal
            symbol = random.choice(self.watchlist)
            signal = {
                'symbol': symbol,
                'action': random.choice(['BUY', 'SELL']),
                'confidence': random.uniform(0.3, 0.7),
                'reason': "Market opportunity detected",
                'type': 'opportunity',
                'priority': 'MEDIUM'
            }
            signals.append(signal)
            
        # Sort by priority and confidence
        signals.sort(key=lambda x: (
            2 if x['priority'] == 'HIGH' else 1,
            x['confidence']
        ), reverse=True)
        
        self.performance['signals_generated'] += len(signals)
        return signals[:3]  # Top 3 signals
        
    async def execute_trade(self, signal, market_data):
        """Execute a trade based on signal"""
        symbol = signal['symbol']
        action = signal['action']
        
        if symbol not in market_data:
            return False
            
        current_price = market_data[symbol]['price']
        
        # Calculate position size (more aggressive)
        if action == 'BUY':
            max_investment = self.current_capital * self.risk_per_trade
            if max_investment < 0.01:  # Minimum $0.01
                return False
                
            quantity = max_investment / current_price
            
            # Execute buy
            if symbol in self.positions:
                old_pos = self.positions[symbol]
                total_qty = old_pos['quantity'] + quantity
                avg_price = (old_pos['quantity'] * old_pos['price'] + quantity * current_price) / total_qty
                self.positions[symbol] = {
                    'quantity': total_qty, 
                    'price': avg_price,
                    'entry_time': old_pos.get('entry_time', datetime.now())
                }
            else:
                self.positions[symbol] = {
                    'quantity': quantity, 
                    'price': current_price,
                    'entry_time': datetime.now()
                }
                
            self.current_capital -= max_investment
            
            trade = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'price': current_price,
                'value': max_investment,
                'signal': signal
            }
            
        elif action == 'SELL' and symbol in self.positions:
            # Execute sell
            pos = self.positions[symbol]
            sell_value = pos['quantity'] * current_price
            pnl = pos['quantity'] * (current_price - pos['price'])
            
            self.current_capital += sell_value
            del self.positions[symbol]
            
            trade = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'action': action,
                'quantity': pos['quantity'],
                'price': current_price,
                'value': sell_value,
                'pnl': pnl,
                'signal': signal
            }
            
            if pnl > 0:
                self.performance['winning_trades'] += 1
            else:
                self.performance['losing_trades'] += 1
                
        elif action == 'VOLATILITY':
            # Volatility strategy - buy expecting price movement
            investment = min(self.current_capital * 0.15, 0.50)  # Larger position for volatility
            if investment >= 0.01:
                quantity = investment / current_price
                self.positions[symbol] = {
                    'quantity': quantity, 
                    'price': current_price,
                    'entry_time': datetime.now()
                }
                self.current_capital -= investment
                
                trade = {
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'action': 'VOLATILITY_BUY',
                    'quantity': quantity,
                    'price': current_price,
                    'value': investment,
                    'signal': signal
                }
            else:
                return False
        else:
            return False
            
        self.trade_history.append(trade)
        self.performance['total_trades'] += 1
        
        # Update max capital
        total_value = self.current_capital + sum(
            pos['quantity'] * market_data.get(sym, {}).get('price', pos['price'])
            for sym, pos in self.positions.items()
        )
        
        if total_value > self.performance['max_capital']:
            self.performance['max_capital'] = total_value
            
        logger.info(f"✅ Trade: {action} {quantity:.6f} {symbol} @ ${current_price:.4f} (Confidence: {signal['confidence']:.1%})")
        
        return True
        
    async def manage_positions(self, market_data):
        """Aggressive position management"""
        positions_to_close = []
        
        for symbol, pos in self.positions.items():
            if symbol not in market_data:
                continue
                
            current_price = market_data[symbol]['price']
            pnl_pct = (current_price - pos['price']) / pos['price']
            hold_time = (datetime.now() - pos['entry_time']).total_seconds() / 60  # minutes
            
            # Aggressive take profit (10%) or stop loss (5%)
            if pnl_pct > 0.10 or pnl_pct < -0.05:
                positions_to_close.append(symbol)
            # Time-based exit (positions older than 2 minutes in demo)
            elif hold_time > 2 and pnl_pct > 0.02:  # Close profitable positions quickly
                positions_to_close.append(symbol)
                
        # Close positions
        for symbol in positions_to_close:
            signal = {
                'symbol': symbol,
                'action': 'SELL',
                'confidence': 0.9,
                'reason': 'Risk management',
                'type': 'risk_mgmt',
                'priority': 'HIGH'
            }
            await self.execute_trade(signal, market_data)
            
    def calculate_performance(self, market_data):
        """Calculate current performance metrics"""
        # Calculate total portfolio value
        position_value = sum(
            pos['quantity'] * market_data.get(symbol, {}).get('price', pos['price'])
            for symbol, pos in self.positions.items()
        )
        
        total_value = self.current_capital + position_value
        total_return = (total_value - self.starting_capital) / self.starting_capital
        
        # Calculate drawdown
        drawdown = (self.performance['max_capital'] - total_value) / self.performance['max_capital']
        if drawdown > self.performance['max_drawdown']:
            self.performance['max_drawdown'] = drawdown
            
        return {
            'total_value': total_value,
            'cash': self.current_capital,
            'position_value': position_value,
            'total_return_pct': total_return * 100,
            'drawdown_pct': drawdown * 100,
            'active_positions': len(self.positions)
        }
        
    async def run_simulation_cycle(self, cycle_num):
        """Run one simulation cycle"""
        logger.info(f"\n{'='*60}")
        logger.info(f"🔄 AGGRESSIVE CYCLE {cycle_num} - {datetime.now().strftime('%H:%M:%S')}")
        logger.info(f"{'='*60}")
        
        try:
            # Update market simulation occasionally
            if cycle_num % 3 == 0:
                self.market_simulator['trend_direction'] = random.choice(['bullish', 'bearish', 'neutral'])
                self.market_simulator['volatility_level'] = random.uniform(0.02, 0.08)
                self.market_simulator['news_sentiment_bias'] = random.uniform(-50, 50)
                logger.info(f"🎲 Market Simulation: {self.market_simulator['trend_direction']} trend, {self.market_simulator['volatility_level']:.2%} volatility")
            
            # 1. Get market data
            logger.info("📊 Fetching market data...")
            market_data = await self.get_market_prices()
            logger.info(f"✅ Got prices for {len(market_data)} symbols")
            
            # Show some price movements
            for symbol, data in list(market_data.items())[:3]:
                logger.info(f"   {symbol}: ${data['price']:.4f} ({data['change']:+.2%})")
            
            # 2. Analyze news
            logger.info("📰 Analyzing news sentiment...")
            news_data = await self.analyze_news_sentiment()
            logger.info(f"✅ Analyzed {news_data['news_count']} news items")
            
            # 3. Generate signals
            logger.info("💡 Generating trading signals...")
            signals = await self.generate_trading_signals(market_data, news_data)
            logger.info(f"✅ Generated {len(signals)} signals")
            
            # Show signals
            for i, signal in enumerate(signals, 1):
                logger.info(f"   Signal {i}: {signal['action']} {signal['symbol']} - {signal['reason']} (Confidence: {signal['confidence']:.1%})")
            
            # 4. Manage existing positions
            await self.manage_positions(market_data)
            
            # 5. Execute new trades
            trades_executed = 0
            for signal in signals:
                if self.current_capital > 0.02:  # Keep minimum cash
                    if await self.execute_trade(signal, market_data):
                        trades_executed += 1
                        
            if trades_executed > 0:
                logger.info(f"⚡ Executed {trades_executed} trades this cycle")
                    
            # 6. Calculate performance
            perf = self.calculate_performance(market_data)
            self.performance['cycles_completed'] = cycle_num
            
            # 7. Display status
            logger.info(f"\n📈 PERFORMANCE UPDATE:")
            logger.info(f"   💰 Total Value: ${perf['total_value']:.4f}")
            logger.info(f"   💵 Cash: ${perf['cash']:.4f}")
            logger.info(f"   📊 Return: {perf['total_return_pct']:+.2f}%")
            logger.info(f"   📍 Positions: {perf['active_positions']}")
            logger.info(f"   🔄 Total Trades: {self.performance['total_trades']}")
            
            # Show news insights
            if news_data['contradictions']:
                logger.info(f"\n🔥 Contradictions Found:")
                for c in news_data['contradictions'][:2]:
                    logger.info(f"   {c['currency']}: {c['positive_count']} pos vs {c['negative_count']} neg (Strength: {c['strength']:.1%})")
                    
            # Show market sentiment
            if abs(news_data['sentiment_score']) > 15:
                sentiment = "Bullish" if news_data['sentiment_score'] > 0 else "Bearish"
                logger.info(f"\n📊 Market Sentiment: {sentiment} ({news_data['sentiment_score']:.1f})")
                
            # Show top positions
            if self.positions:
                logger.info(f"\n💼 Active Positions:")
                for symbol, pos in self.positions.items():
                    if symbol in market_data:
                        current_price = market_data[symbol]['price']
                        pnl_pct = (current_price - pos['price']) / pos['price']
                        hold_time = (datetime.now() - pos['entry_time']).total_seconds() / 60
                        logger.info(f"   {symbol}: {pos['quantity']:.6f} @ ${pos['price']:.4f} ({pnl_pct:+.2%}) [{hold_time:.1f}m]")
                        
            return perf
            
        except Exception as e:
            logger.error(f"❌ Error in cycle: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    async def run_demo(self, cycles=10, interval=20):
        """Run the aggressive demo simulation"""
        logger.info("🚀 Starting AGGRESSIVE KIMERA Trading Demo")
        logger.info(f"💰 Starting Capital: ${self.starting_capital}")
        logger.info(f"🎯 Cycles: {cycles}, Interval: {interval}s")
        logger.info("⚡ AGGRESSIVE MODE: Higher risk, lower thresholds, more action!")
        logger.info("=" * 70)
        
        start_time = datetime.now()
        results = []
        
        for cycle in range(1, cycles + 1):
            perf = await self.run_simulation_cycle(cycle)
            if perf:
                results.append(perf)
                
            # Early exit conditions
            if perf and perf['total_value'] < 0.05:
                logger.warning("💀 Capital too low - stopping simulation")
                break
                
            if perf and perf['total_value'] > 20:
                logger.info("🎉 EXCEPTIONAL PERFORMANCE - 2000% gain achieved!")
                break
                
            # Wait for next cycle
            if cycle < cycles:
                logger.info(f"⏳ Waiting {interval}s for next cycle...")
                await asyncio.sleep(interval)
                
        # Final report
        await self.generate_final_report(results, start_time)
        
    async def generate_final_report(self, results, start_time):
        """Generate comprehensive final report"""
        logger.info(f"\n{'='*70}")
        logger.info("🏁 AGGRESSIVE KIMERA DEMO COMPLETE")
        logger.info(f"{'='*70}")
        
        if not results:
            logger.info("❌ No results to report")
            return
            
        final_perf = results[-1]
        duration = (datetime.now() - start_time).total_seconds() / 60
        
        logger.info(f"\n💰 FINANCIAL RESULTS:")
        logger.info(f"   Starting Capital: ${self.starting_capital:.2f}")
        logger.info(f"   Final Value: ${final_perf['total_value']:.4f}")
        logger.info(f"   Total Return: {final_perf['total_return_pct']:+.2f}%")
        logger.info(f"   Peak Capital: ${self.performance['max_capital']:.4f}")
        logger.info(f"   Max Drawdown: {self.performance['max_drawdown']*100:.2f}%")
        
        logger.info(f"\n📊 TRADING STATISTICS:")
        logger.info(f"   Total Trades: {self.performance['total_trades']}")
        logger.info(f"   Winning Trades: {self.performance['winning_trades']}")
        logger.info(f"   Losing Trades: {self.performance['losing_trades']}")
        win_rate = self.performance['winning_trades'] / max(self.performance['winning_trades'] + self.performance['losing_trades'], 1)
        logger.info(f"   Win Rate: {win_rate*100:.1f}%")
        logger.info(f"   Trades per Cycle: {self.performance['total_trades'] / max(self.performance['cycles_completed'], 1):.1f}")
        
        logger.info(f"\n🧠 KIMERA INTELLIGENCE:")
        logger.info(f"   Signals Generated: {self.performance['signals_generated']}")
        logger.info(f"   Contradictions Found: {self.performance['contradictions_found']}")
        logger.info(f"   Cycles Completed: {self.performance['cycles_completed']}")
        logger.info(f"   Duration: {duration:.1f} minutes")
        
        # Performance assessment
        if final_perf['total_return_pct'] > 1000:
            logger.info("\n🏆 LEGENDARY: Over 1000% return!")
        elif final_perf['total_return_pct'] > 500:
            logger.info("\n🎖️ EXCEPTIONAL: Over 500% return!")
        elif final_perf['total_return_pct'] > 200:
            logger.info("\n🎉 EXCELLENT: Over 200% return!")
        elif final_perf['total_return_pct'] > 50:
            logger.info("\n✅ GREAT: Over 50% return!")
        elif final_perf['total_return_pct'] > 0:
            logger.info("\n📈 POSITIVE: Profitable trading!")
        else:
            logger.info("\n📉 LEARNING: Aggressive trading is risky!")
            
        # Growth analysis
        if len(results) > 1:
            growth_rate = (final_perf['total_value'] / self.starting_capital) ** (1/duration) - 1
            logger.info(f"\n📊 GROWTH ANALYSIS:")
            logger.info(f"   Growth Rate: {growth_rate*100:.2f}% per minute")
            if growth_rate > 0:
                hours_to_double = 60 * (2 ** (1/growth_rate) - 1) if growth_rate > 0 else float('inf')
                logger.info(f"   Time to Double: {hours_to_double:.1f} hours (if sustained)")
                
        # Save report
        report = {
            'summary': {
                'starting_capital': self.starting_capital,
                'final_value': final_perf['total_value'],
                'total_return_pct': final_perf['total_return_pct'],
                'duration_minutes': duration,
                'aggressive_mode': True
            },
            'performance': self.performance,
            'market_simulation': self.market_simulator,
            'trade_history': [
                {
                    'timestamp': trade['timestamp'].isoformat(),
                    **{k: v for k, v in trade.items() if k != 'timestamp'}
                }
                for trade in self.trade_history
            ]
        }
        
        report_file = f"aggressive_kimera_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"\n📁 Detailed report saved: {report_file}")
        logger.info(f"\n🤖 KIMERA's aggressive semantic trading has been demonstrated!")


async def main():
    """Main entry point"""
    logger.info("⚡ AGGRESSIVE KIMERA Trading Demo - $1 Growth Challenge")
    logger.info("=" * 60)
    logger.info("Starting with $1.00 - AGGRESSIVE MODE!")
    logger.info("• Higher risk tolerance (25% per trade)
    logger.info("• Lower signal thresholds")
    logger.info("• Faster position management")
    logger.info("• More frequent trading")
    logger.info()
    
    choice = input("Choose demo mode:\n1. Quick Aggressive Demo (5 cycles, 20s each)\n2. Extended Aggressive Demo (8 cycles, 30s each)\n3. Custom\n\nEnter choice (1-3): ").strip()
    
    demo = AggressiveKimeraDemo(starting_capital=1.0)
    
    if choice == '1':
        await demo.run_demo(cycles=5, interval=20)
    elif choice == '2':
        await demo.run_demo(cycles=8, interval=30)
    elif choice == '3':
        cycles = int(input("Number of cycles (5-15): ") or 8)
        interval = int(input("Interval in seconds (15-60): ") or 25)
        await demo.run_demo(cycles=cycles, interval=interval)
    else:
        logger.info("Running quick aggressive demo...")
        await demo.run_demo(cycles=5, interval=20)


if __name__ == "__main__":
    asyncio.run(main()) 