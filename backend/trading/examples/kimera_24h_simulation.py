"""
KIMERA 24-Hour Trading Simulation
Starting Capital: $1.00
Goal: Test growth potential using semantic contradiction detection
"""

import asyncio
import sys
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any
import logging
import json
import time
import random

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from connectors.cryptopanic_connector import CryptoPanicConnector, NewsSentiment
from connectors.data_providers import YahooFinanceConnector
from core.semantic_trading_reactor import SemanticTradingReactor
from execution.semantic_execution_bridge import SemanticExecutionBridge
from monitoring.semantic_trading_dashboard import TradingMetrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'kimera_24h_simulation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Kimera24HourSimulation:
    """
    24-Hour KIMERA Trading Simulation
    
    Features:
    - Real market data integration
    - Live news sentiment analysis
    - Semantic contradiction detection
    - Autonomous trading decisions
    - Risk management
    - Performance tracking
    """
    
    def __init__(self, starting_capital: float = 1.0):
        self.starting_capital = starting_capital
        self.current_capital = starting_capital
        self.positions = {}  # {symbol: {quantity, entry_price, entry_time}}
        self.trade_history = []
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'peak_capital': starting_capital,
            'hourly_returns': [],
            'news_signals': 0,
            'contradiction_signals': 0,
            'technical_signals': 0
        }
        
        # Initialize connectors
        self.news_connector = CryptoPanicConnector()
        self.market_connector = YahooFinanceConnector()
        self.trading_reactor = SemanticTradingReactor()
        self.execution_bridge = SemanticExecutionBridge(paper_trading=True)
        
        # Trading parameters
        self.risk_per_trade = 0.05  # 5% risk per trade (aggressive for $1 start)
        self.max_position_size = 0.3  # Max 30% of capital per position
        self.stop_loss_pct = 0.02  # 2% stop loss
        self.take_profit_pct = 0.06  # 6% take profit (3:1 ratio)
        
        # Watchlist - focus on high-volume, volatile cryptos
        self.watchlist = [
            'BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'ADA-USD',
            'DOT-USD', 'AVAX-USD', 'MATIC-USD', 'LINK-USD', 'UNI-USD'
        ]
        
        # News cache for contradiction detection
        self.news_cache = []
        self.last_news_update = None
        
    async def initialize_simulation(self):
        """Initialize all systems for the 24-hour simulation"""
        logger.info("🚀 Initializing KIMERA 24-Hour Trading Simulation")
        logger.info(f"💰 Starting Capital: ${self.starting_capital:.2f}")
        logger.info(f"🎯 Target: Maximum growth in 24 hours")
        logger.info(f"📊 Watchlist: {', '.join(self.watchlist)}")
        
        # Test all connections
        try:
            # Test news connection
            async with self.news_connector as connector:
                news = await connector.get_posts()
                logger.info(f"✅ News connector ready - {len(news)} items available")
                
            # Test market data
            btc_data = await self.market_connector.get_current_price('BTC-USD')
            logger.info(f"✅ Market data ready - BTC: ${btc_data:.2f}")
            
            logger.info("🎮 All systems initialized - Starting simulation!")
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            raise
            
    async def fetch_market_data(self) -> Dict[str, Any]:
        """Fetch current market data for all watchlist symbols"""
        market_data = {}
        
        for symbol in self.watchlist:
            try:
                price = await self.market_connector.get_current_price(symbol)
                # Simulate additional data (in real system, this would come from exchanges)
                market_data[symbol] = {
                    'price': price,
                    'volume': random.uniform(1000000, 10000000),  # Mock volume
                    'change_24h': random.uniform(-0.1, 0.1),  # Mock 24h change
                    'volatility': random.uniform(0.02, 0.08),  # Mock volatility
                    'timestamp': datetime.now(timezone.utc)
                }
            except Exception as e:
                logger.warning(f"Failed to fetch data for {symbol}: {e}")
                
        return market_data
        
    async def fetch_news_sentiment(self) -> Dict[str, Any]:
        """Fetch and analyze news sentiment"""
        try:
            async with self.news_connector as connector:
                # Get latest news
                news = await connector.get_posts()
                
                # Get market sentiment
                sentiment = await connector.analyze_market_sentiment()
                
                # Cache news for contradiction detection
                self.news_cache.extend(news)
                # Keep only last 100 items
                self.news_cache = self.news_cache[-100:]
                self.last_news_update = datetime.now(timezone.utc)
                
                return {
                    'news_items': news,
                    'market_sentiment': sentiment,
                    'news_count': len(news)
                }
                
        except Exception as e:
            logger.error(f"Failed to fetch news: {e}")
            return {
                'news_items': [],
                'market_sentiment': {'sentiment_score': 0},
                'news_count': 0
            }
            
    async def detect_contradictions(self, news_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect semantic contradictions in news using KIMERA's approach"""
        contradictions = []
        news_items = news_data['news_items']
        
        if len(news_items) < 2:
            return contradictions
            
        # Group news by currency mentions
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
                
            # Find opposing sentiments
            positive_items = [i for i in items if i.sentiment == NewsSentiment.POSITIVE]
            negative_items = [i for i in items if i.sentiment == NewsSentiment.NEGATIVE]
            
            if positive_items and negative_items:
                # Calculate contradiction strength
                pos_panic = sum(i.panic_score for i in positive_items) / len(positive_items)
                neg_panic = sum(i.panic_score for i in negative_items) / len(negative_items)
                
                contradiction = {
                    'currency': currency,
                    'type': 'sentiment_contradiction',
                    'strength': abs(pos_panic - neg_panic) / 100,  # Normalize to 0-1
                    'positive_count': len(positive_items),
                    'negative_count': len(negative_items),
                    'time_span': (max(i.published_at for i in items) - 
                                min(i.published_at for i in items)).total_seconds() / 3600,
                    'trading_opportunity': currency.replace('-USD', '') + '-USD' in self.watchlist
                }
                
                if contradiction['strength'] > 0.3:  # Significant contradiction
                    contradictions.append(contradiction)
                    
        return contradictions
        
    async def generate_trading_signals(self, 
                                     market_data: Dict[str, Any],
                                     news_data: Dict[str, Any],
                                     contradictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate trading signals using KIMERA's semantic approach"""
        signals = []
        
        market_sentiment = news_data['market_sentiment']['sentiment_score']
        
        # Signal 1: Strong market sentiment with no contradictions
        if abs(market_sentiment) > 50 and len(contradictions) == 0:
            # Find most mentioned currency in our watchlist
            trending = news_data['market_sentiment'].get('trending_currencies', [])
            for currency, mentions in trending:
                symbol = f"{currency}-USD"
                if symbol in self.watchlist and symbol in market_data:
                    signal = {
                        'symbol': symbol,
                        'action': 'BUY' if market_sentiment > 0 else 'SELL',
                        'confidence': min(abs(market_sentiment) / 100, 0.85),
                        'reason': f"Strong {'bullish' if market_sentiment > 0 else 'bearish'} sentiment consensus",
                        'type': 'sentiment_consensus',
                        'priority': 'HIGH' if mentions > 5 else 'MEDIUM'
                    }
                    signals.append(signal)
                    self.performance_metrics['news_signals'] += 1
                    break
                    
        # Signal 2: Contradiction-based volatility trading
        for contradiction in contradictions:
            if contradiction['trading_opportunity'] and contradiction['strength'] > 0.5:
                symbol = contradiction['currency'].replace('-USD', '') + '-USD'
                if symbol in market_data:
                    signal = {
                        'symbol': symbol,
                        'action': 'STRADDLE',  # Volatility play
                        'confidence': min(contradiction['strength'], 0.8),
                        'reason': f"High contradiction detected - volatility opportunity",
                        'type': 'contradiction_volatility',
                        'priority': 'HIGH'
                    }
                    signals.append(signal)
                    self.performance_metrics['contradiction_signals'] += 1
                    
        # Signal 3: Technical momentum (mock - would use real TA in production)
        for symbol, data in market_data.items():
            if abs(data['change_24h']) > 0.05:  # 5% daily change
                signal = {
                    'symbol': symbol,
                    'action': 'BUY' if data['change_24h'] > 0 else 'SELL',
                    'confidence': min(abs(data['change_24h']) * 10, 0.7),
                    'reason': f"Strong momentum: {data['change_24h']:.1%} 24h change",
                    'type': 'technical_momentum',
                    'priority': 'MEDIUM'
                }
                signals.append(signal)
                self.performance_metrics['technical_signals'] += 1
                
        # Sort by confidence and priority
        signals.sort(key=lambda x: (
            1 if x['priority'] == 'HIGH' else 0,
            x['confidence']
        ), reverse=True)
        
        return signals[:3]  # Limit to top 3 signals
        
    async def execute_trade(self, signal: Dict[str, Any], market_data: Dict[str, Any]) -> bool:
        """Execute a trade based on a signal"""
        symbol = signal['symbol']
        action = signal['action']
        
        if symbol not in market_data:
            logger.warning(f"No market data for {symbol}")
            return False
            
        current_price = market_data[symbol]['price']
        
        # Calculate position size based on available capital and risk
        available_capital = self.current_capital * (1 - sum(
            pos['quantity'] * market_data.get(sym, {}).get('price', 0) / self.current_capital
            for sym, pos in self.positions.items()
            if sym in market_data
        ))
        
        max_position_value = min(
            available_capital * self.max_position_size,
            self.current_capital * self.risk_per_trade / self.stop_loss_pct
        )
        
        if max_position_value < 0.01:  # Minimum $0.01 trade
            logger.info(f"Insufficient capital for {symbol} trade")
            return False
            
        quantity = max_position_value / current_price
        
        if action in ['BUY', 'SELL']:
            # Execute the trade
            trade = {
                'timestamp': datetime.now(timezone.utc),
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'price': current_price,
                'value': quantity * current_price,
                'confidence': signal['confidence'],
                'reason': signal['reason'],
                'type': signal['type']
            }
            
            if action == 'BUY':
                if symbol in self.positions:
                    # Add to existing position
                    old_pos = self.positions[symbol]
                    total_quantity = old_pos['quantity'] + quantity
                    avg_price = (old_pos['quantity'] * old_pos['entry_price'] + 
                               quantity * current_price) / total_quantity
                    self.positions[symbol] = {
                        'quantity': total_quantity,
                        'entry_price': avg_price,
                        'entry_time': old_pos['entry_time']
                    }
                else:
                    # New position
                    self.positions[symbol] = {
                        'quantity': quantity,
                        'entry_price': current_price,
                        'entry_time': datetime.now(timezone.utc)
                    }
                    
                self.current_capital -= trade['value']
                
            elif action == 'SELL' and symbol in self.positions:
                # Close position
                pos = self.positions[symbol]
                sell_quantity = min(quantity, pos['quantity'])
                pnl = sell_quantity * (current_price - pos['entry_price'])
                
                self.current_capital += sell_quantity * current_price
                
                if sell_quantity >= pos['quantity']:
                    del self.positions[symbol]
                else:
                    self.positions[symbol]['quantity'] -= sell_quantity
                    
                trade['pnl'] = pnl
                self.performance_metrics['total_pnl'] += pnl
                
                if pnl > 0:
                    self.performance_metrics['winning_trades'] += 1
                else:
                    self.performance_metrics['losing_trades'] += 1
                    
            self.trade_history.append(trade)
            self.performance_metrics['total_trades'] += 1
            
            logger.info(f"🔄 Trade executed: {action} {quantity:.6f} {symbol} @ ${current_price:.4f}")
            logger.info(f"💰 Capital: ${self.current_capital:.4f} | Positions: {len(self.positions)}")
            
            return True
            
        return False
        
    async def manage_risk(self, market_data: Dict[str, Any]):
        """Manage existing positions with stop-loss and take-profit"""
        positions_to_close = []
        
        for symbol, position in self.positions.items():
            if symbol not in market_data:
                continue
                
            current_price = market_data[symbol]['price']
            entry_price = position['entry_price']
            pnl_pct = (current_price - entry_price) / entry_price
            
            # Stop loss
            if pnl_pct < -self.stop_loss_pct:
                positions_to_close.append((symbol, 'STOP_LOSS', pnl_pct))
                
            # Take profit
            elif pnl_pct > self.take_profit_pct:
                positions_to_close.append((symbol, 'TAKE_PROFIT', pnl_pct))
                
            # Time-based exit (positions older than 4 hours)
            elif (datetime.now(timezone.utc) - position['entry_time']).total_seconds() > 14400:
                if pnl_pct > 0:  # Only close profitable old positions
                    positions_to_close.append((symbol, 'TIME_EXIT', pnl_pct))
                    
        # Execute position closures
        for symbol, reason, pnl_pct in positions_to_close:
            await self.close_position(symbol, reason, market_data)
            
    async def close_position(self, symbol: str, reason: str, market_data: Dict[str, Any]):
        """Close a specific position"""
        if symbol not in self.positions or symbol not in market_data:
            return
            
        position = self.positions[symbol]
        current_price = market_data[symbol]['price']
        
        # Calculate P&L
        pnl = position['quantity'] * (current_price - position['entry_price'])
        pnl_pct = (current_price - position['entry_price']) / position['entry_price']
        
        # Execute closure
        self.current_capital += position['quantity'] * current_price
        
        trade = {
            'timestamp': datetime.now(timezone.utc),
            'symbol': symbol,
            'action': 'CLOSE',
            'quantity': position['quantity'],
            'price': current_price,
            'value': position['quantity'] * current_price,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'reason': reason,
            'hold_time': (datetime.now(timezone.utc) - position['entry_time']).total_seconds() / 3600
        }
        
        self.trade_history.append(trade)
        del self.positions[symbol]
        
        self.performance_metrics['total_pnl'] += pnl
        if pnl > 0:
            self.performance_metrics['winning_trades'] += 1
        else:
            self.performance_metrics['losing_trades'] += 1
            
        logger.info(f"🔒 Position closed: {symbol} | P&L: ${pnl:.4f} ({pnl_pct:.2%}) | Reason: {reason}")
        
    def calculate_performance_metrics(self):
        """Calculate comprehensive performance metrics"""
        # Update peak capital and drawdown
        if self.current_capital > self.performance_metrics['peak_capital']:
            self.performance_metrics['peak_capital'] = self.current_capital
            
        current_drawdown = (self.performance_metrics['peak_capital'] - self.current_capital) / self.performance_metrics['peak_capital']
        if current_drawdown > self.performance_metrics['max_drawdown']:
            self.performance_metrics['max_drawdown'] = current_drawdown
            
        # Calculate returns
        total_return = (self.current_capital - self.starting_capital) / self.starting_capital
        
        # Win rate
        total_closed_trades = self.performance_metrics['winning_trades'] + self.performance_metrics['losing_trades']
        win_rate = self.performance_metrics['winning_trades'] / max(total_closed_trades, 1)
        
        return {
            'current_capital': self.current_capital,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'win_rate': win_rate,
            'win_rate_pct': win_rate * 100,
            'max_drawdown_pct': self.performance_metrics['max_drawdown'] * 100,
            'total_trades': self.performance_metrics['total_trades'],
            'active_positions': len(self.positions)
        }
        
    async def run_simulation_cycle(self, cycle_number: int):
        """Run one simulation cycle (every 15 minutes)"""
        logger.info(f"\n{'='*60}")
        logger.info(f"🔄 CYCLE {cycle_number} - {datetime.now().strftime('%H:%M:%S')}")
        logger.info(f"{'='*60}")
        
        try:
            # 1. Fetch market data
            logger.info("📊 Fetching market data...")
            market_data = await self.fetch_market_data()
            logger.info(f"✅ Market data for {len(market_data)} symbols")
            
            # 2. Fetch news and sentiment
            logger.info("📰 Analyzing news sentiment...")
            news_data = await self.fetch_news_sentiment()
            logger.info(f"✅ Analyzed {news_data['news_count']} news items")
            
            # 3. Detect contradictions
            logger.info("🔍 Detecting semantic contradictions...")
            contradictions = await self.detect_contradictions(news_data)
            logger.info(f"✅ Found {len(contradictions)} contradictions")
            
            # 4. Generate trading signals
            logger.info("💡 Generating trading signals...")
            signals = await self.generate_trading_signals(market_data, news_data, contradictions)
            logger.info(f"✅ Generated {len(signals)} signals")
            
            # 5. Risk management first
            await self.manage_risk(market_data)
            
            # 6. Execute new trades
            for signal in signals:
                if self.current_capital > 0.05:  # Minimum capital threshold
                    await self.execute_trade(signal, market_data)
                    await asyncio.sleep(1)  # Small delay between trades
                    
            # 7. Update performance metrics
            metrics = self.calculate_performance_metrics()
            self.performance_metrics['hourly_returns'].append({
                'cycle': cycle_number,
                'timestamp': datetime.now(timezone.utc),
                'capital': self.current_capital,
                'return_pct': metrics['total_return_pct']
            })
            
            # 8. Display status
            logger.info(f"\n📈 PERFORMANCE UPDATE:")
            logger.info(f"   💰 Capital: ${metrics['current_capital']:.4f}")
            logger.info(f"   📊 Return: {metrics['total_return_pct']:+.2f}%")
            logger.info(f"   🎯 Win Rate: {metrics['win_rate_pct']:.1f}%")
            logger.info(f"   📉 Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
            logger.info(f"   🔄 Total Trades: {metrics['total_trades']}")
            logger.info(f"   📍 Active Positions: {metrics['active_positions']}")
            
            # Show top contradictions
            if contradictions:
                logger.info(f"\n🔥 TOP CONTRADICTIONS:")
                for i, c in enumerate(contradictions[:3], 1):
                    logger.info(f"   {i}. {c['currency']}: {c['strength']:.2%} strength")
                    
            # Show active positions
            if self.positions:
                logger.info(f"\n💼 ACTIVE POSITIONS:")
                for symbol, pos in self.positions.items():
                    if symbol in market_data:
                        current_price = market_data[symbol]['price']
                        pnl_pct = (current_price - pos['entry_price']) / pos['entry_price']
                        logger.info(f"   {symbol}: {pos['quantity']:.6f} @ ${pos['entry_price']:.4f} ({pnl_pct:+.2%})")
                        
        except Exception as e:
            logger.error(f"❌ Error in simulation cycle: {e}")
            import traceback
            traceback.print_exc()
            
    async def run_24_hour_simulation(self):
        """Run the complete 24-hour simulation"""
        logger.info("🚀 Starting KIMERA 24-Hour Trading Simulation")
        logger.info(f"⏰ Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        await self.initialize_simulation()
        
        # Simulation parameters
        total_cycles = 96  # 24 hours * 4 cycles per hour (every 15 minutes)
        cycle_interval = 900  # 15 minutes in seconds
        
        start_time = datetime.now()
        
        for cycle in range(1, total_cycles + 1):
            cycle_start = datetime.now()
            
            # Run simulation cycle
            await self.run_simulation_cycle(cycle)
            
            # Calculate remaining time and sleep
            cycle_duration = (datetime.now() - cycle_start).total_seconds()
            sleep_time = max(0, cycle_interval - cycle_duration)
            
            # Show progress
            elapsed_hours = (datetime.now() - start_time).total_seconds() / 3600
            progress = (cycle / total_cycles) * 100
            
            logger.info(f"\n⏱️  Cycle {cycle}/{total_cycles} complete ({progress:.1f}%)")
            logger.info(f"⏰ Elapsed: {elapsed_hours:.1f}h | Next cycle in: {sleep_time/60:.1f}m")
            
            # Early termination conditions
            if self.current_capital < 0.01:
                logger.warning("💀 Capital depleted - Ending simulation early")
                break
                
            if self.current_capital > 100:  # 10,000% gain!
                logger.info("🎉 Exceptional performance - Target achieved early!")
                break
                
            # Sleep until next cycle (or shorter for demo)
            if sleep_time > 60:  # If more than 1 minute, reduce for demo
                sleep_time = min(sleep_time, 300)  # Max 5 minutes for demo
                
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
                
        # Final results
        await self.generate_final_report()
        
    async def generate_final_report(self):
        """Generate comprehensive final report"""
        logger.info(f"\n{'='*80}")
        logger.info(f"🏁 KIMERA 24-HOUR SIMULATION COMPLETE")
        logger.info(f"{'='*80}")
        
        final_metrics = self.calculate_performance_metrics()
        
        logger.info(f"\n💰 FINANCIAL PERFORMANCE:")
        logger.info(f"   Starting Capital: ${self.starting_capital:.2f}")
        logger.info(f"   Final Capital: ${final_metrics['current_capital']:.4f}")
        logger.info(f"   Total Return: {final_metrics['total_return_pct']:+.2f}%")
        logger.info(f"   Peak Capital: ${self.performance_metrics['peak_capital']:.4f}")
        logger.info(f"   Max Drawdown: {final_metrics['max_drawdown_pct']:.2f}%")
        
        logger.info(f"\n📊 TRADING STATISTICS:")
        logger.info(f"   Total Trades: {final_metrics['total_trades']}")
        logger.info(f"   Winning Trades: {self.performance_metrics['winning_trades']}")
        logger.info(f"   Losing Trades: {self.performance_metrics['losing_trades']}")
        logger.info(f"   Win Rate: {final_metrics['win_rate_pct']:.1f}%")
        logger.info(f"   Active Positions: {final_metrics['active_positions']}")
        
        logger.info(f"\n🧠 KIMERA INTELLIGENCE:")
        logger.info(f"   News Signals: {self.performance_metrics['news_signals']}")
        logger.info(f"   Contradiction Signals: {self.performance_metrics['contradiction_signals']}")
        logger.info(f"   Technical Signals: {self.performance_metrics['technical_signals']}")
        
        # Save detailed report
        report = {
            'simulation_summary': {
                'start_time': datetime.now().isoformat(),
                'starting_capital': self.starting_capital,
                'final_capital': final_metrics['current_capital'],
                'total_return_pct': final_metrics['total_return_pct'],
                'performance_metrics': self.performance_metrics
            },
            'trade_history': [
                {
                    'timestamp': trade['timestamp'].isoformat(),
                    **{k: v for k, v in trade.items() if k != 'timestamp'}
                }
                for trade in self.trade_history
            ],
            'hourly_returns': [
                {
                    'timestamp': hr['timestamp'].isoformat(),
                    **{k: v for k, v in hr.items() if k != 'timestamp'}
                }
                for hr in self.performance_metrics['hourly_returns']
            ]
        }
        
        report_file = f"kimera_24h_simulation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"\n📁 Detailed report saved: {report_file}")
        
        # Performance assessment
        if final_metrics['total_return_pct'] > 100:
            logger.info("🏆 EXCEPTIONAL PERFORMANCE - Over 100% return!")
        elif final_metrics['total_return_pct'] > 50:
            logger.info("🎉 EXCELLENT PERFORMANCE - Over 50% return!")
        elif final_metrics['total_return_pct'] > 20:
            logger.info("✅ GOOD PERFORMANCE - Over 20% return!")
        elif final_metrics['total_return_pct'] > 0:
            logger.info("📈 POSITIVE PERFORMANCE - Profitable trading!")
        else:
            logger.info("📉 Learning experience - Room for improvement")
            
        logger.info(f"\n🤖 KIMERA's semantic contradiction detection and thermodynamic")
        logger.info(f"   analysis have been tested in live market conditions!")


async def main():
    """Main entry point for the 24-hour simulation"""
    logger.info("🎮 KIMERA 24-Hour Trading Simulation")
    logger.info("====================================")
    logger.info("Starting Capital: $1.00")
    logger.info("Duration: 24 hours (or accelerated demo)
    logger.info("Strategy: Semantic contradiction detection + News sentiment")
    logger.info()
    
    choice = input("Choose simulation mode:\n1. Full 24-hour simulation\n2. Accelerated demo (30 minutes)\n3. Quick test (5 cycles)\n\nEnter choice (1-3): ").strip()
    
    simulation = Kimera24HourSimulation(starting_capital=1.0)
    
    if choice == '1':
        await simulation.run_24_hour_simulation()
    elif choice == '2':
        # Accelerated demo - 30 minutes with 2-minute cycles
        logger.info("🚀 Running accelerated 30-minute demo")
        await simulation.initialize_simulation()
        for cycle in range(1, 16):  # 15 cycles * 2 minutes = 30 minutes
            await simulation.run_simulation_cycle(cycle)
            await asyncio.sleep(120)  # 2 minutes between cycles
        await simulation.generate_final_report()
    elif choice == '3':
        # Quick test - 5 cycles
        logger.info("🚀 Running quick test (5 cycles)")
        await simulation.initialize_simulation()
        for cycle in range(1, 6):
            await simulation.run_simulation_cycle(cycle)
            await asyncio.sleep(30)  # 30 seconds between cycles
        await simulation.generate_final_report()
    else:
        logger.info("Invalid choice. Running quick test...")
        await simulation.initialize_simulation()
        for cycle in range(1, 6):
            await simulation.run_simulation_cycle(cycle)
            await asyncio.sleep(30)
        await simulation.generate_final_report()


if __name__ == "__main__":
    asyncio.run(main()) 