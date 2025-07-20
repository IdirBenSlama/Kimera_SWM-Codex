"""
Premium KIMERA Trading Demo
Enhanced with Alpha Vantage, Finnhub, and Twelve Data APIs
Starting with $1 - Testing maximum growth potential with premium data
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
from connectors.premium_data_connectors import PremiumDataManager, DataProvider
import yfinance as yf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PremiumKimeraDemo:
    """
    Premium KIMERA Trading Demo with Enhanced Data Sources
    
    Features:
    - Real-time data from Alpha Vantage, Finnhub, Twelve Data
    - Advanced sentiment analysis from multiple news sources
    - Economic indicators integration
    - Technical analysis suite
    - Enhanced contradiction detection
    - Superior signal generation
    """
    
    def __init__(self, starting_capital: float = 1.0):
        self.starting_capital = starting_capital
        self.current_capital = starting_capital
        self.positions = {}
        self.trade_history = []
        
        # Initialize connectors
        self.news_connector = CryptoPanicConnector()
        self.premium_data = PremiumDataManager()
        
        # Enhanced trading parameters
        self.risk_per_trade = 0.20  # 20% risk per trade
        self.max_position_size = 0.6  # Max 60% per position
        self.sentiment_threshold = 5  # Very low threshold with premium data
        self.momentum_threshold = 0.005  # 0.5% threshold
        
        # Enhanced watchlist with both stocks and crypto
        self.watchlist = {
            'stocks': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'],
            'crypto': ['BTC-USD', 'ETH-USD', 'SOL-USD']
        }
        
        # Performance tracking
        self.performance = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'max_capital': starting_capital,
            'max_drawdown': 0,
            'signals_generated': 0,
            'contradictions_found': 0,
            'premium_signals': 0,
            'economic_signals': 0,
            'cycles_completed': 0
        }
        
        # Intelligence cache
        self.intelligence_cache = {}
        self.last_economic_update = None
        
    async def get_premium_market_data(self):
        """Get enhanced market data from premium sources"""
        market_data = {}
        
        async with self.premium_data as data_manager:
            # Get stock data
            for symbol in self.watchlist['stocks']:
                try:
                    intelligence = await data_manager.generate_trading_intelligence(symbol)
                    
                    # Extract price data
                    if 'market_data' in intelligence and 'finnhub_quote' in intelligence['market_data']:
                        quote = intelligence['market_data']['finnhub_quote']
                        if 'c' in quote:  # current price
                            market_data[symbol] = {
                                'price': quote['c'],
                                'change': quote.get('d', 0),  # change
                                'change_pct': quote.get('dp', 0),  # change percent
                                'volume': quote.get('v', 0),  # volume
                                'high': quote.get('h', quote['c']),  # high
                                'low': quote.get('l', quote['c']),  # low
                                'previous_close': quote.get('pc', quote['c']),
                                'intelligence': intelligence,
                                'data_quality': 'premium'
                            }
                except Exception as e:
                    logger.warning(f"Failed to get premium data for {symbol}: {e}")
                    
            # Get crypto data (fallback to yfinance for crypto)
            for symbol in self.watchlist['crypto']:
                try:
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period="1d", interval="1m")
                    if not data.empty:
                        current_price = float(data['Close'].iloc[-1])
                        open_price = float(data['Open'].iloc[-1])
                        change_pct = (current_price - open_price) / open_price
                        
                        market_data[symbol] = {
                            'price': current_price,
                            'change': current_price - open_price,
                            'change_pct': change_pct * 100,
                            'volume': float(data['Volume'].iloc[-1]) if not data['Volume'].empty else 0,
                            'high': float(data['High'].iloc[-1]),
                            'low': float(data['Low'].iloc[-1]),
                            'previous_close': open_price,
                            'data_quality': 'standard'
                        }
                except Exception as e:
                    logger.warning(f"Failed to get crypto data for {symbol}: {e}")
                    
        return market_data
        
    async def analyze_premium_sentiment(self):
        """Enhanced sentiment analysis using multiple premium sources"""
        sentiment_data = {
            'news_sources': [],
            'overall_sentiment': 0,
            'sentiment_strength': 0,
            'contradictions': [],
            'economic_indicators': {},
            'market_stress': 0
        }
        
        try:
            # Get CryptoPanic news
            async with self.news_connector as connector:
                crypto_news = await connector.get_posts()
                crypto_sentiment = await connector.analyze_market_sentiment()
                
                sentiment_data['news_sources'].append({
                    'source': 'CryptoPanic',
                    'sentiment_score': crypto_sentiment['sentiment_score'],
                    'news_count': len(crypto_news),
                    'trending_currencies': crypto_sentiment['trending_currencies'][:3]
                })
                
            # Get premium news and sentiment
            async with self.premium_data as data_manager:
                # Get general market news
                market_news = await data_manager.get_news_finnhub('general')
                
                # Analyze sentiment from news headlines
                if market_news:
                    positive_words = ['surge', 'rally', 'gain', 'rise', 'bull', 'optimistic', 'growth']
                    negative_words = ['fall', 'drop', 'crash', 'bear', 'decline', 'loss', 'pessimistic']
                    
                    sentiment_score = 0
                    for news in market_news[:20]:
                        headline_lower = news.headline.lower()
                        for word in positive_words:
                            if word in headline_lower:
                                sentiment_score += 1
                        for word in negative_words:
                            if word in headline_lower:
                                sentiment_score -= 1
                                
                    sentiment_data['news_sources'].append({
                        'source': 'Finnhub',
                        'sentiment_score': sentiment_score,
                        'news_count': len(market_news),
                        'latest_headline': market_news[0].headline if market_news else None
                    })
                    
                # Get economic context
                if not self.last_economic_update or (datetime.now() - self.last_economic_update).seconds > 300:
                    economic_data = await data_manager.get_economic_context()
                    sentiment_data['economic_indicators'] = economic_data
                    self.last_economic_update = datetime.now()
                    
            # Calculate overall sentiment
            total_sentiment = sum(source['sentiment_score'] for source in sentiment_data['news_sources'])
            sentiment_data['overall_sentiment'] = total_sentiment
            sentiment_data['sentiment_strength'] = abs(total_sentiment)
            
            # Detect contradictions between sources
            if len(sentiment_data['news_sources']) >= 2:
                source1 = sentiment_data['news_sources'][0]
                source2 = sentiment_data['news_sources'][1]
                
                if (source1['sentiment_score'] > 10 and source2['sentiment_score'] < -10) or \
                   (source1['sentiment_score'] < -10 and source2['sentiment_score'] > 10):
                    contradiction = {
                        'type': 'cross_source_sentiment',
                        'source1': source1['source'],
                        'source2': source2['source'],
                        'sentiment1': source1['sentiment_score'],
                        'sentiment2': source2['sentiment_score'],
                        'strength': abs(source1['sentiment_score'] - source2['sentiment_score']) / 50
                    }
                    sentiment_data['contradictions'].append(contradiction)
                    
        except Exception as e:
            logger.error(f"Error in premium sentiment analysis: {e}")
            
        return sentiment_data
        
    async def generate_premium_signals(self, market_data, sentiment_data):
        """Generate enhanced trading signals using premium data"""
        signals = []
        
        # Signal 1: Premium sentiment with economic context
        overall_sentiment = sentiment_data['overall_sentiment']
        if abs(overall_sentiment) > self.sentiment_threshold:
            # Find best symbol based on premium intelligence
            best_symbol = None
            best_score = 0
            
            for symbol, data in market_data.items():
                if 'intelligence' in data:
                    intelligence = data['intelligence']
                    summary = intelligence.get('summary', {})
                    
                    # Calculate composite score
                    price_change = summary.get('price_change_pct', 0)
                    signal_strength = summary.get('signal_strength', 'low')
                    
                    strength_score = {'high': 3, 'medium': 2, 'low': 1}.get(signal_strength, 1)
                    composite_score = abs(price_change) * strength_score
                    
                    if composite_score > best_score:
                        best_score = composite_score
                        best_symbol = symbol
                        
            if best_symbol:
                signal = {
                    'symbol': best_symbol,
                    'action': 'BUY' if overall_sentiment > 0 else 'SELL',
                    'confidence': min(abs(overall_sentiment) / 100, 0.9),
                    'reason': f"Premium sentiment: {overall_sentiment:.1f} with intelligence score: {best_score:.2f}",
                    'type': 'premium_sentiment',
                    'priority': 'HIGH'
                }
                signals.append(signal)
                self.performance['premium_signals'] += 1
                
        # Signal 2: Cross-source contradictions
        for contradiction in sentiment_data['contradictions']:
            if contradiction['strength'] > 0.4:
                # Find volatile symbols for contradiction trading
                volatile_symbols = [
                    symbol for symbol, data in market_data.items()
                    if abs(data.get('change_pct', 0)) > 1.0
                ]
                
                if volatile_symbols:
                    target_symbol = volatile_symbols[0]
                    signal = {
                        'symbol': target_symbol,
                        'action': 'VOLATILITY',
                        'confidence': contradiction['strength'],
                        'reason': f"Cross-source contradiction: {contradiction['source1']} vs {contradiction['source2']}",
                        'type': 'premium_contradiction',
                        'priority': 'HIGH'
                    }
                    signals.append(signal)
                    self.performance['contradictions_found'] += 1
                    
        # Signal 3: Premium technical analysis
        for symbol, data in market_data.items():
            if 'intelligence' in data:
                intelligence = data['intelligence']
                technical = intelligence.get('technical_analysis', {})
                
                # Check RSI
                if 'rsi' in technical and 'values' in technical['rsi']:
                    try:
                        rsi_values = technical['rsi']['values']
                        if rsi_values:
                            latest_rsi = float(rsi_values[0]['rsi'])
                            
                            if latest_rsi < 30:  # Oversold
                                signal = {
                                    'symbol': symbol,
                                    'action': 'BUY',
                                    'confidence': (30 - latest_rsi) / 30,
                                    'reason': f"RSI oversold: {latest_rsi:.1f}",
                                    'type': 'technical_rsi',
                                    'priority': 'MEDIUM'
                                }
                                signals.append(signal)
                            elif latest_rsi > 70:  # Overbought
                                signal = {
                                    'symbol': symbol,
                                    'action': 'SELL',
                                    'confidence': (latest_rsi - 70) / 30,
                                    'reason': f"RSI overbought: {latest_rsi:.1f}",
                                    'type': 'technical_rsi',
                                    'priority': 'MEDIUM'
                                }
                                signals.append(signal)
                    except (ValueError, KeyError, IndexError):
                        pass
                        
        # Signal 4: Economic indicator signals
        economic_indicators = sentiment_data.get('economic_indicators', {})
        if 'upcoming_earnings' in economic_indicators:
            earnings = economic_indicators['upcoming_earnings']
            for earning in earnings[:3]:  # Top 3 upcoming earnings
                symbol = earning.get('symbol', '')
                if symbol in market_data:
                    signal = {
                        'symbol': symbol,
                        'action': 'VOLATILITY',
                        'confidence': 0.6,
                        'reason': f"Upcoming earnings: {earning.get('date', 'Soon')}",
                        'type': 'earnings_event',
                        'priority': 'MEDIUM'
                    }
                    signals.append(signal)
                    self.performance['economic_signals'] += 1
                    
        # Signal 5: Enhanced momentum with volume confirmation
        for symbol, data in market_data.items():
            change_pct = abs(data.get('change_pct', 0))
            volume = data.get('volume', 0)
            
            if change_pct > self.momentum_threshold * 100:  # Convert to percentage
                # Volume confirmation (higher volume = higher confidence)
                volume_factor = min(volume / 1000000, 2)  # Normalize volume
                confidence = min(change_pct / 100 * volume_factor, 0.8)
                
                if confidence > 0.3:
                    signal = {
                        'symbol': symbol,
                        'action': 'BUY' if data['change_pct'] > 0 else 'SELL',
                        'confidence': confidence,
                        'reason': f"Enhanced momentum: {data['change_pct']:.2f}% with volume {volume:,.0f}",
                        'type': 'enhanced_momentum',
                        'priority': 'HIGH' if confidence > 0.6 else 'MEDIUM'
                    }
                    signals.append(signal)
                    
        # Sort signals by priority and confidence
        signals.sort(key=lambda x: (
            3 if x['priority'] == 'HIGH' else 2 if x['priority'] == 'MEDIUM' else 1,
            x['confidence']
        ), reverse=True)
        
        self.performance['signals_generated'] += len(signals)
        return signals[:4]  # Top 4 signals
        
    async def execute_premium_trade(self, signal, market_data):
        """Execute trade with enhanced position sizing based on signal quality"""
        symbol = signal['symbol']
        action = signal['action']
        
        if symbol not in market_data:
            return False
            
        current_price = market_data[symbol]['price']
        
        # Enhanced position sizing based on signal type and confidence
        base_risk = self.risk_per_trade
        
        # Adjust risk based on signal type
        if signal['type'] in ['premium_sentiment', 'premium_contradiction']:
            risk_multiplier = 1.5  # Higher risk for premium signals
        elif signal['type'] in ['technical_rsi', 'enhanced_momentum']:
            risk_multiplier = 1.2  # Medium risk for technical signals
        else:
            risk_multiplier = 1.0
            
        # Adjust risk based on confidence
        confidence_multiplier = signal['confidence']
        
        # Calculate final position size
        adjusted_risk = base_risk * risk_multiplier * confidence_multiplier
        max_investment = self.current_capital * min(adjusted_risk, self.max_position_size)
        
        if max_investment < 0.01:
            return False
            
        quantity = max_investment / current_price
        
        # Execute trade logic (similar to previous implementation but enhanced)
        if action == 'BUY':
            if symbol in self.positions:
                old_pos = self.positions[symbol]
                total_qty = old_pos['quantity'] + quantity
                avg_price = (old_pos['quantity'] * old_pos['price'] + quantity * current_price) / total_qty
                self.positions[symbol] = {
                    'quantity': total_qty,
                    'price': avg_price,
                    'entry_time': old_pos.get('entry_time', datetime.now()),
                    'signal_type': signal['type']
                }
            else:
                self.positions[symbol] = {
                    'quantity': quantity,
                    'price': current_price,
                    'entry_time': datetime.now(),
                    'signal_type': signal['type']
                }
                
            self.current_capital -= max_investment
            
            trade = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'price': current_price,
                'value': max_investment,
                'signal': signal,
                'data_quality': market_data[symbol].get('data_quality', 'standard')
            }
            
        elif action == 'SELL' and symbol in self.positions:
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
                'signal': signal,
                'data_quality': market_data[symbol].get('data_quality', 'standard')
            }
            
            if pnl > 0:
                self.performance['winning_trades'] += 1
            else:
                self.performance['losing_trades'] += 1
                
        elif action == 'VOLATILITY':
            investment = min(self.current_capital * 0.10, 0.30)  # Smaller volatility positions
            if investment >= 0.01:
                quantity = investment / current_price
                self.positions[symbol] = {
                    'quantity': quantity,
                    'price': current_price,
                    'entry_time': datetime.now(),
                    'signal_type': signal['type']
                }
                self.current_capital -= investment
                
                trade = {
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'action': 'VOLATILITY_BUY',
                    'quantity': quantity,
                    'price': current_price,
                    'value': investment,
                    'signal': signal,
                    'data_quality': market_data[symbol].get('data_quality', 'standard')
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
            
        logger.info(f"✅ PREMIUM Trade: {action} {quantity:.6f} {symbol} @ ${current_price:.4f}")
        logger.info(f"   Signal: {signal['type']} | Confidence: {signal['confidence']:.1%} | Data: {market_data[symbol].get('data_quality', 'standard')}")
        
        return True
        
    async def manage_premium_positions(self, market_data):
        """Enhanced position management with signal-type awareness"""
        positions_to_close = []
        
        for symbol, pos in self.positions.items():
            if symbol not in market_data:
                continue
                
            current_price = market_data[symbol]['price']
            pnl_pct = (current_price - pos['price']) / pos['price']
            hold_time = (datetime.now() - pos['entry_time']).total_seconds() / 60
            signal_type = pos.get('signal_type', 'unknown')
            
            # Dynamic exit rules based on signal type
            if signal_type in ['premium_sentiment', 'premium_contradiction']:
                # More aggressive exits for premium signals
                if pnl_pct > 0.08 or pnl_pct < -0.04:  # 8% profit, 4% loss
                    positions_to_close.append(symbol)
                elif hold_time > 3 and pnl_pct > 0.02:  # 3 minutes, 2% profit
                    positions_to_close.append(symbol)
            elif signal_type in ['technical_rsi', 'enhanced_momentum']:
                # Standard exits for technical signals
                if pnl_pct > 0.12 or pnl_pct < -0.06:  # 12% profit, 6% loss
                    positions_to_close.append(symbol)
                elif hold_time > 5 and pnl_pct > 0.03:  # 5 minutes, 3% profit
                    positions_to_close.append(symbol)
            else:
                # Conservative exits for other signals
                if pnl_pct > 0.15 or pnl_pct < -0.08:  # 15% profit, 8% loss
                    positions_to_close.append(symbol)
                    
        # Close positions
        for symbol in positions_to_close:
            signal = {
                'symbol': symbol,
                'action': 'SELL',
                'confidence': 0.95,
                'reason': 'Premium risk management',
                'type': 'risk_mgmt',
                'priority': 'HIGH'
            }
            await self.execute_premium_trade(signal, market_data)
            
    def calculate_premium_performance(self, market_data):
        """Enhanced performance calculation with data quality metrics"""
        position_value = sum(
            pos['quantity'] * market_data.get(symbol, {}).get('price', pos['price'])
            for symbol, pos in self.positions.items()
        )
        
        total_value = self.current_capital + position_value
        total_return = (total_value - self.starting_capital) / self.starting_capital
        
        drawdown = (self.performance['max_capital'] - total_value) / self.performance['max_capital']
        if drawdown > self.performance['max_drawdown']:
            self.performance['max_drawdown'] = drawdown
            
        # Calculate data quality score
        premium_trades = sum(1 for trade in self.trade_history if trade.get('data_quality') == 'premium')
        data_quality_score = premium_trades / max(len(self.trade_history), 1)
        
        return {
            'total_value': total_value,
            'cash': self.current_capital,
            'position_value': position_value,
            'total_return_pct': total_return * 100,
            'drawdown_pct': drawdown * 100,
            'active_positions': len(self.positions),
            'data_quality_score': data_quality_score,
            'premium_signal_ratio': self.performance['premium_signals'] / max(self.performance['signals_generated'], 1)
        }
        
    async def run_premium_cycle(self, cycle_num):
        """Run enhanced simulation cycle with premium data"""
        logger.info(f"\n{'='*70}")
        logger.info(f"🚀 PREMIUM KIMERA CYCLE {cycle_num} - {datetime.now().strftime('%H:%M:%S')}")
        logger.info(f"{'='*70}")
        
        try:
            # 1. Get premium market data
            logger.info("📊 Fetching premium market data...")
            market_data = await self.get_premium_market_data()
            premium_count = sum(1 for data in market_data.values() if data.get('data_quality') == 'premium')
            logger.info(f"✅ Got data for {len(market_data)} symbols ({premium_count} premium)")
            
            # Show sample premium data
            for symbol, data in list(market_data.items())[:3]:
                quality = data.get('data_quality', 'standard')
                logger.info(f"   {symbol}: ${data['price']:.4f} ({data['change_pct']:+.2f}%) [{quality}]")
                
            # 2. Enhanced sentiment analysis
            logger.info("📰 Analyzing premium sentiment...")
            sentiment_data = await self.analyze_premium_sentiment()
            logger.info(f"✅ Analyzed {len(sentiment_data['news_sources'])} news sources")
            
            # 3. Generate premium signals
            logger.info("💡 Generating premium trading signals...")
            signals = await self.generate_premium_signals(market_data, sentiment_data)
            logger.info(f"✅ Generated {len(signals)} premium signals")
            
            # Show signals with enhanced info
            for i, signal in enumerate(signals, 1):
                logger.info(f"   Signal {i}: {signal['action']} {signal['symbol']} - {signal['reason']}")
                logger.info(f"             Type: {signal['type']} | Confidence: {signal['confidence']:.1%} | Priority: {signal['priority']}")
                
            # 4. Enhanced position management
            await self.manage_premium_positions(market_data)
            
            # 5. Execute premium trades
            trades_executed = 0
            for signal in signals:
                if self.current_capital > 0.03:
                    if await self.execute_premium_trade(signal, market_data):
                        trades_executed += 1
                        
            if trades_executed > 0:
                logger.info(f"⚡ Executed {trades_executed} premium trades this cycle")
                
            # 6. Calculate enhanced performance
            perf = self.calculate_premium_performance(market_data)
            self.performance['cycles_completed'] = cycle_num
            
            # 7. Display enhanced status
            logger.info(f"\n📈 PREMIUM PERFORMANCE:")
            logger.info(f"   💰 Total Value: ${perf['total_value']:.4f}")
            logger.info(f"   💵 Cash: ${perf['cash']:.4f}")
            logger.info(f"   📊 Return: {perf['total_return_pct']:+.2f}%")
            logger.info(f"   📍 Positions: {perf['active_positions']}")
            logger.info(f"   🔄 Total Trades: {self.performance['total_trades']}")
            logger.info(f"   🏆 Data Quality Score: {perf['data_quality_score']:.1%}")
            logger.info(f"   ⭐ Premium Signal Ratio: {perf['premium_signal_ratio']:.1%}")
            
            # Show intelligence insights
            if sentiment_data['contradictions']:
                logger.info(f"\n🔥 Premium Contradictions:")
                for c in sentiment_data['contradictions']:
                    logger.info(f"   {c['source1']} vs {c['source2']}: {c['sentiment1']} vs {c['sentiment2']} (Strength: {c['strength']:.1%})")
                    
            # Show overall sentiment
            if abs(sentiment_data['overall_sentiment']) > 10:
                sentiment_dir = "Bullish" if sentiment_data['overall_sentiment'] > 0 else "Bearish"
                logger.info(f"\n📊 Market Sentiment: {sentiment_dir} ({sentiment_data['overall_sentiment']:.1f})")
                
            # Show premium positions
            if self.positions:
                logger.info(f"\n💼 Premium Positions:")
                for symbol, pos in self.positions.items():
                    if symbol in market_data:
                        current_price = market_data[symbol]['price']
                        pnl_pct = (current_price - pos['price']) / pos['price']
                        hold_time = (datetime.now() - pos['entry_time']).total_seconds() / 60
                        signal_type = pos.get('signal_type', 'unknown')
                        logger.info(f"   {symbol}: {pos['quantity']:.6f} @ ${pos['price']:.4f} ({pnl_pct:+.2%}) [{signal_type}] [{hold_time:.1f}m]")
                        
            return perf
            
        except Exception as e:
            logger.error(f"❌ Error in premium cycle: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    async def run_premium_demo(self, cycles=8, interval=25):
        """Run premium KIMERA demo"""
        logger.info("🚀 Starting PREMIUM KIMERA Trading Demo")
        logger.info(f"💰 Starting Capital: ${self.starting_capital}")
        logger.info(f"🎯 Cycles: {cycles}, Interval: {interval}s")
        logger.info("⭐ PREMIUM MODE: Alpha Vantage + Finnhub + Twelve Data + CryptoPanic")
        logger.info("=" * 80)
        
        start_time = datetime.now()
        results = []
        
        for cycle in range(1, cycles + 1):
            perf = await self.run_premium_cycle(cycle)
            if perf:
                results.append(perf)
                
            # Enhanced exit conditions
            if perf and perf['total_value'] < 0.05:
                logger.warning("💀 Capital depleted - stopping simulation")
                break
                
            if perf and perf['total_value'] > 50:
                logger.info("🎉 EXCEPTIONAL PERFORMANCE - 5000% gain achieved!")
                break
                
            # Wait for next cycle
            if cycle < cycles:
                logger.info(f"⏳ Waiting {interval}s for next premium cycle...")
                await asyncio.sleep(interval)
                
        # Generate premium final report
        await self.generate_premium_final_report(results, start_time)
        
    async def generate_premium_final_report(self, results, start_time):
        """Generate enhanced final report"""
        logger.info(f"\n{'='*80}")
        logger.info("🏁 PREMIUM KIMERA DEMO COMPLETE")
        logger.info(f"{'='*80}")
        
        if not results:
            logger.info("❌ No results to report")
            return
            
        final_perf = results[-1]
        duration = (datetime.now() - start_time).total_seconds() / 60
        
        logger.info(f"\n💰 PREMIUM FINANCIAL RESULTS:")
        logger.info(f"   Starting Capital: ${self.starting_capital:.2f}")
        logger.info(f"   Final Value: ${final_perf['total_value']:.4f}")
        logger.info(f"   Total Return: {final_perf['total_return_pct']:+.2f}%")
        logger.info(f"   Peak Capital: ${self.performance['max_capital']:.4f}")
        logger.info(f"   Max Drawdown: {self.performance['max_drawdown']*100:.2f}%")
        logger.info(f"   Data Quality Score: {final_perf['data_quality_score']:.1%}")
        
        logger.info(f"\n📊 PREMIUM TRADING STATISTICS:")
        logger.info(f"   Total Trades: {self.performance['total_trades']}")
        logger.info(f"   Winning Trades: {self.performance['winning_trades']}")
        logger.info(f"   Losing Trades: {self.performance['losing_trades']}")
        win_rate = self.performance['winning_trades'] / max(self.performance['winning_trades'] + self.performance['losing_trades'], 1)
        logger.info(f"   Win Rate: {win_rate*100:.1f}%")
        logger.info(f"   Premium Signal Ratio: {final_perf['premium_signal_ratio']:.1%}")
        
        logger.info(f"\n🧠 PREMIUM KIMERA INTELLIGENCE:")
        logger.info(f"   Total Signals Generated: {self.performance['signals_generated']}")
        logger.info(f"   Premium Signals: {self.performance['premium_signals']}")
        logger.info(f"   Contradictions Found: {self.performance['contradictions_found']}")
        logger.info(f"   Economic Signals: {self.performance['economic_signals']}")
        logger.info(f"   Duration: {duration:.1f} minutes")
        
        # Enhanced performance assessment
        if final_perf['total_return_pct'] > 2000:
            logger.info("\n🏆 LEGENDARY PERFORMANCE: Over 2000% return with premium data!")
        elif final_perf['total_return_pct'] > 1000:
            logger.info("\n🎖️ EXCEPTIONAL PERFORMANCE: Over 1000% return!")
        elif final_perf['total_return_pct'] > 500:
            logger.info("\n🎉 EXCELLENT PERFORMANCE: Over 500% return!")
        elif final_perf['total_return_pct'] > 100:
            logger.info("\n✅ GREAT PERFORMANCE: Over 100% return!")
        elif final_perf['total_return_pct'] > 0:
            logger.info("\n📈 POSITIVE PERFORMANCE: Premium data advantage!")
        else:
            logger.info("\n📉 LEARNING EXPERIENCE: Market conditions challenging")
            
        # Save enhanced report
        report = {
            'summary': {
                'starting_capital': self.starting_capital,
                'final_value': final_perf['total_value'],
                'total_return_pct': final_perf['total_return_pct'],
                'duration_minutes': duration,
                'premium_mode': True,
                'data_sources': ['Alpha Vantage', 'Finnhub', 'Twelve Data', 'CryptoPanic']
            },
            'performance': self.performance,
            'data_quality_metrics': {
                'premium_trades_ratio': final_perf['data_quality_score'],
                'premium_signals_ratio': final_perf['premium_signal_ratio']
            },
            'trade_history': [
                {
                    'timestamp': trade['timestamp'].isoformat(),
                    **{k: v for k, v in trade.items() if k != 'timestamp'}
                }
                for trade in self.trade_history
            ]
        }
        
        report_file = f"premium_kimera_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"\n📁 Premium report saved: {report_file}")
        logger.info(f"\n🤖 KIMERA's premium semantic trading with enterprise data sources complete!")


async def main():
    """Main entry point for premium demo"""
    logger.info("⭐ PREMIUM KIMERA Trading Demo - $1 Growth Challenge")
    logger.info("=" * 70)
    logger.info("Starting with $1.00 - PREMIUM DATA SOURCES!")
    logger.info("• Alpha Vantage: Real-time market data & economic indicators")
    logger.info("• Finnhub: Premium news, sentiment & earnings data")
    logger.info("• Twelve Data: Technical indicators & forex data")
    logger.info("• CryptoPanic: Crypto news & sentiment analysis")
    logger.info()
    
    choice = input("Choose demo mode:\n1. Quick Premium Demo (5 cycles, 25s each)\n2. Extended Premium Demo (8 cycles, 30s each)\n3. Custom\n\nEnter choice (1-3): ").strip()
    
    demo = PremiumKimeraDemo(starting_capital=1.0)
    
    if choice == '1':
        await demo.run_premium_demo(cycles=5, interval=25)
    elif choice == '2':
        await demo.run_premium_demo(cycles=8, interval=30)
    elif choice == '3':
        cycles = int(input("Number of cycles (5-12): ") or 8)
        interval = int(input("Interval in seconds (20-60): ") or 30)
        await demo.run_premium_demo(cycles=cycles, interval=interval)
    else:
        logger.info("Running quick premium demo...")
        await demo.run_premium_demo(cycles=5, interval=25)


if __name__ == "__main__":
    asyncio.run(main()) 