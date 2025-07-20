"""
Simplified KIMERA Trading Demo
Starting with $1 - Testing Growth Potential
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

class SimpleKimeraDemo:
    """
    Simplified KIMERA Trading Demo
    
    Demonstrates:
    - Real-time news sentiment analysis
    - Semantic contradiction detection
    - Trading signal generation
    - Performance tracking
    """
    
    def __init__(self, starting_capital: float = 1.0):
        self.starting_capital = starting_capital
        self.current_capital = starting_capital
        self.positions = {}
        self.trade_history = []
        self.news_connector = CryptoPanicConnector()
        
        # Trading parameters
        self.risk_per_trade = 0.10  # 10% risk per trade (aggressive for $1)
        self.max_position_size = 0.5  # Max 50% per position
        
        # Watchlist
        self.watchlist = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'ADA-USD']
        
        # Performance tracking
        self.performance = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'max_capital': starting_capital,
            'max_drawdown': 0,
            'signals_generated': 0,
            'contradictions_found': 0
        }
        
    async def get_market_prices(self):
        """Get current market prices using yfinance"""
        prices = {}
        for symbol in self.watchlist:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="1d", interval="1m")
                if not data.empty:
                    prices[symbol] = {
                        'price': float(data['Close'].iloc[-1]),
                        'volume': float(data['Volume'].iloc[-1]),
                        'change': float((data['Close'].iloc[-1] - data['Open'].iloc[-1]) / data['Open'].iloc[-1])
                    }
                else:
                    # Fallback to mock data if yfinance fails
                    base_prices = {'BTC-USD': 67000, 'ETH-USD': 3500, 'SOL-USD': 140, 'BNB-USD': 580, 'ADA-USD': 0.45}
                    prices[symbol] = {
                        'price': base_prices.get(symbol, 100) * (1 + random.uniform(-0.02, 0.02)),
                        'volume': random.uniform(1000000, 10000000),
                        'change': random.uniform(-0.05, 0.05)
                    }
            except Exception as e:
                logger.warning(f"Failed to get price for {symbol}: {e}")
                # Use mock data
                base_prices = {'BTC-USD': 67000, 'ETH-USD': 3500, 'SOL-USD': 140, 'BNB-USD': 580, 'ADA-USD': 0.45}
                prices[symbol] = {
                    'price': base_prices.get(symbol, 100) * (1 + random.uniform(-0.02, 0.02)),
                    'volume': random.uniform(1000000, 10000000),
                    'change': random.uniform(-0.05, 0.05)
                }
        return prices
        
    async def analyze_news_sentiment(self):
        """Analyze news sentiment and detect contradictions"""
        try:
            async with self.news_connector as connector:
                # Get latest news
                news = await connector.get_posts()
                
                # Get market sentiment
                sentiment = await connector.analyze_market_sentiment()
                
                # Detect contradictions
                contradictions = await self.detect_contradictions(news)
                
                return {
                    'news_count': len(news),
                    'sentiment_score': sentiment['sentiment_score'],
                    'trending_currencies': sentiment['trending_currencies'][:5],
                    'contradictions': contradictions
                }
                
        except Exception as e:
            logger.error(f"Failed to analyze news: {e}")
            return {
                'news_count': 0,
                'sentiment_score': 0,
                'trending_currencies': [],
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
        """Generate trading signals based on KIMERA's semantic approach"""
        signals = []
        
        # Signal 1: Strong sentiment consensus
        sentiment_score = news_data['sentiment_score']
        if abs(sentiment_score) > 30:
            # Find most mentioned currency in watchlist
            for currency, mentions in news_data['trending_currencies']:
                symbol = f"{currency}-USD"
                if symbol in self.watchlist and symbol in market_data:
                    signal = {
                        'symbol': symbol,
                        'action': 'BUY' if sentiment_score > 0 else 'SELL',
                        'confidence': min(abs(sentiment_score) / 100, 0.8),
                        'reason': f"Strong sentiment: {sentiment_score:.1f}",
                        'type': 'sentiment'
                    }
                    signals.append(signal)
                    break
                    
        # Signal 2: Contradiction-based volatility
        for contradiction in news_data['contradictions']:
            if contradiction['symbol'] and contradiction['strength'] > 0.5:
                signal = {
                    'symbol': contradiction['symbol'],
                    'action': 'VOLATILITY',
                    'confidence': contradiction['strength'],
                    'reason': f"Contradiction detected: {contradiction['positive_count']} pos vs {contradiction['negative_count']} neg",
                    'type': 'contradiction'
                }
                signals.append(signal)
                self.performance['contradictions_found'] += 1
                
        # Signal 3: Technical momentum
        for symbol, data in market_data.items():
            if abs(data['change']) > 0.03:  # 3% change
                signal = {
                    'symbol': symbol,
                    'action': 'BUY' if data['change'] > 0 else 'SELL',
                    'confidence': min(abs(data['change']) * 20, 0.7),
                    'reason': f"Momentum: {data['change']:.2%}",
                    'type': 'momentum'
                }
                signals.append(signal)
                
        # Sort by confidence
        signals.sort(key=lambda x: x['confidence'], reverse=True)
        self.performance['signals_generated'] += len(signals)
        
        return signals[:2]  # Top 2 signals
        
    async def execute_trade(self, signal, market_data):
        """Execute a trade based on signal"""
        symbol = signal['symbol']
        action = signal['action']
        
        if symbol not in market_data:
            return False
            
        current_price = market_data[symbol]['price']
        
        # Calculate position size
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
                self.positions[symbol] = {'quantity': total_qty, 'price': avg_price}
            else:
                self.positions[symbol] = {'quantity': quantity, 'price': current_price}
                
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
            # Volatility strategy - buy small amount expecting price movement
            investment = min(self.current_capital * 0.05, 0.20)  # Small position
            if investment >= 0.01:
                quantity = investment / current_price
                self.positions[symbol] = {'quantity': quantity, 'price': current_price}
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
            
        logger.info(f"✅ Trade: {action} {quantity:.6f} {symbol} @ ${current_price:.4f}")
        
        return True
        
    async def manage_positions(self, market_data):
        """Manage existing positions with simple rules"""
        positions_to_close = []
        
        for symbol, pos in self.positions.items():
            if symbol not in market_data:
                continue
                
            current_price = market_data[symbol]['price']
            pnl_pct = (current_price - pos['price']) / pos['price']
            
            # Take profit at 15% or stop loss at 8%
            if pnl_pct > 0.15 or pnl_pct < -0.08:
                positions_to_close.append(symbol)
                
        # Close positions
        for symbol in positions_to_close:
            signal = {
                'symbol': symbol,
                'action': 'SELL',
                'confidence': 0.9,
                'reason': 'Risk management',
                'type': 'risk_mgmt'
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
        logger.info(f"\n{'='*50}")
        logger.info(f"🔄 CYCLE {cycle_num} - {datetime.now().strftime('%H:%M:%S')}")
        logger.info(f"{'='*50}")
        
        try:
            # 1. Get market data
            logger.info("📊 Fetching market data...")
            market_data = await self.get_market_prices()
            logger.info(f"✅ Got prices for {len(market_data)} symbols")
            
            # 2. Analyze news
            logger.info("📰 Analyzing news sentiment...")
            news_data = await self.analyze_news_sentiment()
            logger.info(f"✅ Analyzed {news_data['news_count']} news items")
            
            # 3. Generate signals
            logger.info("💡 Generating trading signals...")
            signals = await self.generate_trading_signals(market_data, news_data)
            logger.info(f"✅ Generated {len(signals)} signals")
            
            # 4. Manage existing positions
            await self.manage_positions(market_data)
            
            # 5. Execute new trades
            for signal in signals:
                if self.current_capital > 0.02:  # Keep minimum cash
                    await self.execute_trade(signal, market_data)
                    
            # 6. Calculate performance
            perf = self.calculate_performance(market_data)
            
            # 7. Display status
            logger.info(f"\n📈 PERFORMANCE:")
            logger.info(f"   💰 Total Value: ${perf['total_value']:.4f}")
            logger.info(f"   💵 Cash: ${perf['cash']:.4f}")
            logger.info(f"   📊 Return: {perf['total_return_pct']:+.2f}%")
            logger.info(f"   📍 Positions: {perf['active_positions']}")
            logger.info(f"   🔄 Total Trades: {self.performance['total_trades']}")
            
            # Show news insights
            if news_data['contradictions']:
                logger.info(f"\n🔥 Contradictions Found:")
                for c in news_data['contradictions'][:3]:
                    logger.info(f"   {c['currency']}: {c['positive_count']} pos vs {c['negative_count']} neg")
                    
            # Show market sentiment
            if abs(news_data['sentiment_score']) > 20:
                sentiment = "Bullish" if news_data['sentiment_score'] > 0 else "Bearish"
                logger.info(f"\n📊 Market Sentiment: {sentiment} ({news_data['sentiment_score']:.1f})")
                
            # Show top positions
            if self.positions:
                logger.info(f"\n💼 Active Positions:")
                for symbol, pos in self.positions.items():
                    if symbol in market_data:
                        current_price = market_data[symbol]['price']
                        pnl_pct = (current_price - pos['price']) / pos['price']
                        logger.info(f"   {symbol}: {pos['quantity']:.6f} @ ${pos['price']:.4f} ({pnl_pct:+.2%})")
                        
            return perf
            
        except Exception as e:
            logger.error(f"❌ Error in cycle: {e}")
            return None
            
    async def run_demo(self, cycles=10, interval=30):
        """Run the demo simulation"""
        logger.info("🚀 Starting KIMERA Trading Demo")
        logger.info(f"💰 Starting Capital: ${self.starting_capital}")
        logger.info(f"🎯 Cycles: {cycles}, Interval: {interval}s")
        logger.info("=" * 60)
        
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
                
            if perf and perf['total_value'] > 10:
                logger.info("🎉 Exceptional performance - 1000% gain achieved!")
                break
                
            # Wait for next cycle
            if cycle < cycles:
                logger.info(f"⏳ Waiting {interval}s for next cycle...")
                await asyncio.sleep(interval)
                
        # Final report
        await self.generate_final_report(results, start_time)
        
    async def generate_final_report(self, results, start_time):
        """Generate final performance report"""
        logger.info(f"\n{'='*60}")
        logger.info("🏁 KIMERA DEMO COMPLETE")
        logger.info(f"{'='*60}")
        
        if not results:
            logger.info("❌ No results to report")
            return
            
        final_perf = results[-1]
        duration = (datetime.now() - start_time).total_seconds() / 60
        
        logger.info(f"\n💰 FINANCIAL RESULTS:")
        logger.info(f"   Starting Capital: ${self.starting_capital:.2f}")
        logger.info(f"   Final Value: ${final_perf['total_value']:.4f}")
        logger.info(f"   Total Return: {final_perf['total_return_pct']:+.2f}%")
        logger.info(f"   Max Drawdown: {self.performance['max_drawdown']*100:.2f}%")
        
        logger.info(f"\n📊 TRADING STATS:")
        logger.info(f"   Total Trades: {self.performance['total_trades']}")
        logger.info(f"   Winning Trades: {self.performance['winning_trades']}")
        logger.info(f"   Losing Trades: {self.performance['losing_trades']}")
        win_rate = self.performance['winning_trades'] / max(self.performance['winning_trades'] + self.performance['losing_trades'], 1)
        logger.info(f"   Win Rate: {win_rate*100:.1f}%")
        
        logger.info(f"\n🧠 KIMERA INTELLIGENCE:")
        logger.info(f"   Signals Generated: {self.performance['signals_generated']}")
        logger.info(f"   Contradictions Found: {self.performance['contradictions_found']}")
        logger.info(f"   Duration: {duration:.1f} minutes")
        
        # Performance assessment
        if final_perf['total_return_pct'] > 500:
            logger.info("\n🏆 EXCEPTIONAL: Over 500% return!")
        elif final_perf['total_return_pct'] > 100:
            logger.info("\n🎉 EXCELLENT: Over 100% return!")
        elif final_perf['total_return_pct'] > 50:
            logger.info("\n✅ GREAT: Over 50% return!")
        elif final_perf['total_return_pct'] > 0:
            logger.info("\n📈 POSITIVE: Profitable trading!")
        else:
            logger.info("\n📉 LEARNING: Room for improvement")
            
        # Save report
        report = {
            'summary': {
                'starting_capital': self.starting_capital,
                'final_value': final_perf['total_value'],
                'total_return_pct': final_perf['total_return_pct'],
                'duration_minutes': duration
            },
            'performance': self.performance,
            'trade_history': [
                {
                    'timestamp': trade['timestamp'].isoformat(),
                    **{k: v for k, v in trade.items() if k != 'timestamp'}
                }
                for trade in self.trade_history
            ]
        }
        
        report_file = f"kimera_demo_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"\n📁 Report saved: {report_file}")


async def main():
    """Main entry point"""
    logger.info("🎮 KIMERA Trading Demo - $1 Growth Challenge")
    logger.info("=" * 50)
    logger.info("Starting with $1.00 - Let's see the growth potential!")
    logger.info()
    
    choice = input("Choose demo mode:\n1. Quick Demo (5 cycles, 30s each)\n2. Extended Demo (10 cycles, 60s each)\n3. Custom\n\nEnter choice (1-3): ").strip()
    
    demo = SimpleKimeraDemo(starting_capital=1.0)
    
    if choice == '1':
        await demo.run_demo(cycles=5, interval=30)
    elif choice == '2':
        await demo.run_demo(cycles=10, interval=60)
    elif choice == '3':
        cycles = int(input("Number of cycles (5-20): ") or 10)
        interval = int(input("Interval in seconds (30-300): ") or 60)
        await demo.run_demo(cycles=cycles, interval=interval)
    else:
        logger.info("Running quick demo...")
        await demo.run_demo(cycles=5, interval=30)


if __name__ == "__main__":
    asyncio.run(main()) 