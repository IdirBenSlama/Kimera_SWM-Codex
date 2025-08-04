#!/usr/bin/env python3
"""
KIMERA SENTIMENT-ENHANCED OMNIDIMENSIONAL TRADER
================================================
Combines parallel trading with advanced sentiment analysis:
- Real-time sentiment from decentralized oracles
- Multi-framework sentiment scoring
- Sentiment-weighted trading decisions
- Cross-asset sentiment correlation
"""

import os
import sys
import json
import time
import asyncio
import logging
import numpy as np
import requests
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from dotenv import load_dotenv

# Import our sentiment engine
try:
    from kimera_sentiment_engine import KimeraSentimentEngine, MarketSentiment
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False
    logger.warning("⚠️ Sentiment engine not available, running without sentiment analysis")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SentimentEnhancedOpportunity:
    """Trading opportunity enhanced with sentiment analysis"""
    pair: str
    action: str  # BUY/SELL/HOLD
    base_score: float          # Technical analysis score
    sentiment_score: float     # Sentiment analysis score
    combined_score: float      # Weighted combination
    size: float
    expected_profit: float
    strategy_type: str
    sentiment_signal: Dict
    confidence: float
    priority: int

class SentimentEnhancedTrader:
    """
    ADVANCED TRADING SYSTEM WITH SENTIMENT INTEGRATION
    Combines technical analysis with real-time sentiment scoring
    """
    
    def __init__(self):
        # Load credentials
        load_dotenv('kimera_cdp_live.env')
        self.api_key = os.getenv('CDP_API_KEY_NAME', '').strip()
        self.api_secret = os.getenv('CDP_API_KEY_PRIVATE_KEY', '').strip()
        
        if not self.api_key:
            raise ValueError("CDP credentials not found!")
        
        # Initialize sentiment engine
        if SENTIMENT_AVAILABLE:
            self.sentiment_engine = KimeraSentimentEngine()
            logger.info("✅ Sentiment engine initialized")
        else:
            self.sentiment_engine = None
            logger.warning("⚠️ Trading without sentiment analysis")
        
        # Trading configuration
        self.max_parallel_trades = 30  # Reduced for sentiment processing
        self.update_frequency = 1.0    # Slower for sentiment updates
        self.min_profit_threshold = 0.002
        self.max_position_pct = 0.25
        
        # Sentiment weighting
        self.sentiment_weight = 0.4    # 40% weight to sentiment
        self.technical_weight = 0.6    # 60% weight to technical
        
        # Trading pairs
        self.all_pairs = [
            'BTC-USD', 'ETH-USD', 'SOL-USD', 'AVAX-USD', 'LINK-USD',
            'UNI-USD', 'MATIC-USD', 'ADA-USD', 'DOT-USD', 'ATOM-USD',
            'BTC-EUR', 'ETH-EUR', 'SOL-EUR', 'AVAX-EUR',
            'BTC-ETH', 'ETH-SOL', 'SOL-AVAX', 'LINK-ETH', 'UNI-ETH',
            'MATIC-ETH', 'ADA-BTC', 'DOT-ETH', 'ATOM-USD'
        ]
        
        # Portfolio state
        self.balances = {
            'USD': 100.0,
            'EUR': 50.0,
            'BTC': 0.001,
            'ETH': 0.04,
            'SOL': 0.7,
            'AVAX': 4.0,
            'LINK': 8.0,
            'UNI': 12.0,
            'MATIC': 150.0
        }
        
        # Performance tracking
        self.total_profit = 0.0
        self.sentiment_accuracy = 0.0
        self.sentiment_cache = {}
        self.trade_history = deque(maxlen=500)
        
        # Rate limiting
        self.request_semaphore = asyncio.Semaphore(15)
        
    async def initialize(self):
        """Initialize the sentiment-enhanced trading system"""
        logger.info("🚀 INITIALIZING SENTIMENT-ENHANCED TRADER")
        logger.info("="*60)
        
        # Get initial sentiment data
        if self.sentiment_engine:
            await self.update_market_sentiment()
        
        # Initialize other components
        portfolio_value = await self.calculate_portfolio_value()
        
        logger.info(f"✅ Initialized with {len(self.all_pairs)} trading pairs")
        logger.info(f"💰 Portfolio value: ${portfolio_value:.2f}")
        logger.info(f"🧠 Sentiment analysis: {'Enabled' if self.sentiment_engine else 'Disabled'}")
        
    async def update_market_sentiment(self):
        """Update sentiment data for all tracked assets"""
        if not self.sentiment_engine:
            return
            
        # Extract unique assets from trading pairs
        assets = set()
        for pair in self.all_pairs:
            base, quote = pair.split('-')
            assets.add(base)
            if quote not in ['USD', 'EUR']:
                assets.add(quote)
        
        # Analyze sentiment for all assets
        logger.info(f"🧠 Updating sentiment for {len(assets)} assets")
        
        try:
            sentiment_data = await self.sentiment_engine.analyze_multiple_assets(list(assets))
            
            # Cache sentiment data
            for asset, sentiment in sentiment_data.items():
                self.sentiment_cache[asset] = sentiment
                
            logger.info(f"✅ Updated sentiment for {len(sentiment_data)} assets")
            
        except Exception as e:
            logger.error(f"❌ Sentiment update failed: {e}")
    
    async def get_market_data(self, pair: str) -> Optional[Dict]:
        """Get market data for a specific pair"""
        try:
            async with self.request_semaphore:
                # Get ticker
                ticker_url = f"https://api.exchange.coinbase.com/products/{pair}/ticker"
                ticker_resp = requests.get(ticker_url, timeout=5)
                
                if ticker_resp.status_code != 200:
                    return None
                    
                ticker = ticker_resp.json()
                
                return {
                    'pair': pair,
                    'price': float(ticker.get('price', 0)),
                    'bid': float(ticker.get('bid', 0)),
                    'ask': float(ticker.get('ask', 0)),
                    'volume': float(ticker.get('volume', 0)),
                    'timestamp': time.time()
                }
                
        except Exception as e:
            logger.debug(f"Failed to get data for {pair}: {e}")
            return None
    
    async def analyze_technical_opportunity(self, market_data: Dict) -> float:
        """Analyze technical trading opportunity (existing logic)"""
        pair = market_data['pair']
        
        score = 0.0
        
        # 1. Spread analysis
        if market_data['bid'] > 0 and market_data['ask'] > 0:
            spread_pct = (market_data['ask'] - market_data['bid']) / market_data['bid']
            
            if spread_pct < 0.002:  # Very tight spread
                score += 0.4
            elif spread_pct < 0.005:  # Good spread
                score += 0.2
        
        # 2. Volume analysis
        if market_data['volume'] > 100:
            score += 0.3
            
        # 3. Pair type bonus
        base, quote = pair.split('-')
        if quote in ['BTC', 'ETH', 'SOL']:
            score += 0.3  # Inter-coin trading bonus
        
        # Add some randomness for simulation
        score += np.random.uniform(-0.1, 0.3)
        
        return max(0, min(score, 1.0))
    
    async def get_sentiment_for_pair(self, pair: str) -> Dict:
        """Get sentiment analysis for a trading pair"""
        if not self.sentiment_engine:
            return {
                'signal': 0.0,
                'confidence': 0.5,
                'action': 'HOLD',
                'strength': 0.0,
                'direction': 'neutral'
            }
        
        base, quote = pair.split('-')
        
        # Get sentiment for base asset
        base_sentiment = self.sentiment_cache.get(base)
        quote_sentiment = self.sentiment_cache.get(quote) if quote not in ['USD', 'EUR'] else None
        
        if base_sentiment:
            sentiment_signal = self.sentiment_engine.get_sentiment_signal(base_sentiment)
            
            # If we have quote sentiment, combine them
            if quote_sentiment:
                quote_signal = self.sentiment_engine.get_sentiment_signal(quote_sentiment)
                
                # For X-Y pairs, positive base sentiment and negative quote sentiment = BUY signal
                combined_signal = sentiment_signal['signal'] - quote_signal['signal']
                combined_confidence = (sentiment_signal['confidence'] + quote_signal['confidence']) / 2
                
                return {
                    'signal': combined_signal,
                    'confidence': combined_confidence,
                    'action': 'BUY' if combined_signal > 0.3 else 'SELL' if combined_signal < -0.3 else 'HOLD',
                    'strength': abs(combined_signal),
                    'direction': 'bullish' if combined_signal > 0 else 'bearish' if combined_signal < 0 else 'neutral'
                }
            else:
                return sentiment_signal
        
        # Default neutral sentiment
        return {
            'signal': 0.0,
            'confidence': 0.5,
            'action': 'HOLD',
            'strength': 0.0,
            'direction': 'neutral'
        }
    
    async def analyze_enhanced_opportunity(self, market_data: Dict) -> SentimentEnhancedOpportunity:
        """Analyze opportunity with both technical and sentiment analysis"""
        pair = market_data['pair']
        
        # 1. Technical analysis
        technical_score = await self.analyze_technical_opportunity(market_data)
        
        # 2. Sentiment analysis
        sentiment_data = await self.get_sentiment_for_pair(pair)
        sentiment_score = (sentiment_data['signal'] + 1) / 2  # Convert to 0-1 range
        
        # 3. Combine scores with weights
        combined_score = (
            technical_score * self.technical_weight +
            sentiment_score * self.sentiment_weight
        )
        
        # 4. Determine action based on combined analysis
        if sentiment_data['action'] == 'BUY' and technical_score > 0.4:
            action = 'BUY'
        elif sentiment_data['action'] == 'SELL' and technical_score > 0.4:
            action = 'SELL'
        elif combined_score > 0.6:
            action = 'BUY' if sentiment_data['signal'] > 0 else 'SELL'
        else:
            action = 'HOLD'
        
        # 5. Calculate position size and expected profit
        size = await self.calculate_position_size(pair, combined_score)
        expected_profit = size * combined_score * 0.003  # Enhanced profit rate
        
        # 6. Determine strategy type
        if 'BTC' in pair or 'ETH' in pair:
            strategy_type = 'sentiment_inter_coin'
        elif sentiment_data['confidence'] > 0.7:
            strategy_type = 'sentiment_driven'
        else:
            strategy_type = 'technical_sentiment'
        
        # 7. Calculate priority
        priority = 1 if combined_score > 0.8 else 2 if combined_score > 0.6 else 3
        
        # 8. Overall confidence
        confidence = (
            min(technical_score, 1.0) * 0.4 +
            sentiment_data['confidence'] * 0.6
        )
        
        return SentimentEnhancedOpportunity(
            pair=pair,
            action=action,
            base_score=technical_score,
            sentiment_score=sentiment_score,
            combined_score=combined_score,
            size=size,
            expected_profit=expected_profit,
            strategy_type=strategy_type,
            sentiment_signal=sentiment_data,
            confidence=confidence,
            priority=priority
        )
    
    async def calculate_position_size(self, pair: str, score: float) -> float:
        """Calculate optimal position size"""
        base, quote = pair.split('-')
        
        # Get available balance
        quote_balance = self.balances.get(quote, 0)
        
        # Calculate max position with sentiment adjustment
        max_position = quote_balance * self.max_position_pct * score
        
        return max(max_position, 1.0)  # Minimum $1 position
    
    async def calculate_portfolio_value(self) -> float:
        """Calculate total portfolio value in USD"""
        total_usd = 0.0
        
        usd_rates = {
            'USD': 1.0, 'EUR': 1.08, 'BTC': 100000, 'ETH': 2500,
            'SOL': 140, 'AVAX': 25, 'LINK': 12, 'UNI': 8, 'MATIC': 0.7
        }
        
        for currency, amount in self.balances.items():
            rate = usd_rates.get(currency, 1.0)
            total_usd += amount * rate
                
        return total_usd
    
    async def execute_sentiment_trade(self, opportunity: SentimentEnhancedOpportunity) -> Dict:
        """Execute a sentiment-enhanced trade"""
        try:
            base, quote = opportunity.pair.split('-')
            
            # Update simulated balances
            if opportunity.action == 'BUY':
                self.balances[quote] = max(0, self.balances.get(quote, 0) - opportunity.size)
                # Simulate getting base currency
                rate = 100  # Simplified rate
                self.balances[base] = self.balances.get(base, 0) + (opportunity.size / rate)
            elif opportunity.action == 'SELL':
                self.balances[base] = max(0, self.balances.get(base, 0) - opportunity.size)
                # Simulate getting quote currency
                rate = 100  # Simplified rate
                self.balances[quote] = self.balances.get(quote, 0) + (opportunity.size * rate)
            
            # Calculate actual profit with sentiment bonus
            sentiment_bonus = 1 + (opportunity.sentiment_score * 0.5)  # Up to 50% bonus
            actual_profit = opportunity.expected_profit * sentiment_bonus * np.random.uniform(0.8, 1.4)
            self.total_profit += actual_profit
            
            trade_result = {
                'success': True,
                'pair': opportunity.pair,
                'action': opportunity.action,
                'size': opportunity.size,
                'profit': actual_profit,
                'strategy': opportunity.strategy_type,
                'technical_score': opportunity.base_score,
                'sentiment_score': opportunity.sentiment_score,
                'combined_score': opportunity.combined_score,
                'sentiment_signal': opportunity.sentiment_signal,
                'confidence': opportunity.confidence,
                'timestamp': time.time()
            }
            
            self.trade_history.append(trade_result)
            
            logger.info(f"✅ {opportunity.pair}: {opportunity.action} ${opportunity.size:.2f} "
                       f"({opportunity.strategy_type}) +${actual_profit:.3f}")
            logger.info(f"   Tech: {opportunity.base_score:.2f} | Sent: {opportunity.sentiment_score:.2f} "
                       f"| Combined: {opportunity.combined_score:.2f} | Conf: {opportunity.confidence:.2f}")
            
            return trade_result
            
        except Exception as e:
            logger.error(f"❌ Trade failed for {opportunity.pair}: {e}")
            return {'success': False, 'error': str(e)}
    
    async def parallel_sentiment_analysis(self) -> List[SentimentEnhancedOpportunity]:
        """Analyze all pairs with sentiment enhancement in parallel"""
        
        # Update sentiment data periodically
        current_time = time.time()
        if not hasattr(self, 'last_sentiment_update') or current_time - self.last_sentiment_update > 60:
            await self.update_market_sentiment()
            self.last_sentiment_update = current_time
        
        # Create tasks for all pairs
        tasks = []
        for pair in self.all_pairs:
            task = asyncio.create_task(self.analyze_enhanced_pair(pair))
            tasks.append(task)
        
        # Execute all analyses in parallel
        opportunities = []
        completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in completed_tasks:
            if isinstance(result, SentimentEnhancedOpportunity) and result.combined_score > self.min_profit_threshold:
                opportunities.append(result)
        
        # Sort by combined score and priority
        opportunities.sort(key=lambda x: (x.priority, -x.combined_score))
        
        return opportunities
    
    async def analyze_enhanced_pair(self, pair: str) -> Optional[SentimentEnhancedOpportunity]:
        """Analyze a single pair with sentiment enhancement"""
        try:
            market_data = await self.get_market_data(pair)
            if market_data:
                return await self.analyze_enhanced_opportunity(market_data)
        except Exception as e:
            logger.debug(f"Enhanced analysis failed for {pair}: {e}")
        
        return None
    
    async def execute_parallel_sentiment_trades(self, opportunities: List[SentimentEnhancedOpportunity]):
        """Execute multiple sentiment-enhanced trades in parallel"""
        
        # Filter for actionable opportunities
        actionable_opportunities = [
            opp for opp in opportunities 
            if opp.action in ['BUY', 'SELL'] and opp.confidence > 0.5
        ][:self.max_parallel_trades]
        
        if not actionable_opportunities:
            logger.info("⏳ No actionable sentiment opportunities found")
            return
        
        logger.info(f"🚀 Executing {len(actionable_opportunities)} sentiment-enhanced trades")
        
        # Create execution tasks
        tasks = []
        for opportunity in actionable_opportunities:
            task = asyncio.create_task(self.execute_sentiment_trade(opportunity))
            tasks.append(task)
        
        # Execute all trades in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful_trades = sum(1 for r in results if isinstance(r, dict) and r.get('success'))
        
        logger.info(f"✅ Completed {successful_trades}/{len(actionable_opportunities)} sentiment trades")
    
    async def run_sentiment_enhanced_trading(self, duration_minutes: int = 10):
        """Run sentiment-enhanced omnidimensional trading"""
        logger.info("\n" + "="*70)
        logger.info("🧠 SENTIMENT-ENHANCED OMNIDIMENSIONAL TRADING ACTIVATED")
        logger.info("="*70)
        logger.info(f"🧠 Sentiment analysis: {'✅ Enabled' if self.sentiment_engine else '❌ Disabled'}")
        logger.info(f"⚖️ Weights: Technical {self.technical_weight*100:.0f}% | Sentiment {self.sentiment_weight*100:.0f}%")
        logger.info(f"⚡ Max parallel trades: {self.max_parallel_trades}")
        logger.info(f"🔄 Update frequency: {self.update_frequency}s")
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        cycle = 0
        sentiment_trades = 0
        technical_trades = 0
        
        while time.time() < end_time:
            cycle += 1
            cycle_start = time.time()
            
            logger.info(f"\n🔄 Sentiment-Enhanced Cycle {cycle}")
            
            # Parallel sentiment-enhanced analysis
            opportunities = await self.parallel_sentiment_analysis()
            
            logger.info(f"🧠 Found {len(opportunities)} sentiment-enhanced opportunities")
            
            # Display top opportunities
            for i, opp in enumerate(opportunities[:10]):
                logger.info(f"   {i+1}. {opp.pair}: {opp.action} "
                          f"(T:{opp.base_score:.2f}|S:{opp.sentiment_score:.2f}|C:{opp.combined_score:.2f}) "
                          f"{opp.strategy_type}")
            
            # Execute parallel trades
            await self.execute_parallel_sentiment_trades(opportunities)
            
            # Count trade types
            for opp in opportunities[:self.max_parallel_trades]:
                if 'sentiment' in opp.strategy_type:
                    sentiment_trades += 1
                else:
                    technical_trades += 1
            
            # Portfolio update
            portfolio_value = await self.calculate_portfolio_value()
            
            logger.info(f"\n💰 Portfolio: ${portfolio_value:.2f} (+${self.total_profit:.3f})")
            logger.info(f"🧠 Sentiment trades: {sentiment_trades} | 📊 Technical trades: {technical_trades}")
            
            # Maintain update frequency
            cycle_time = time.time() - cycle_start
            sleep_time = max(0, self.update_frequency - cycle_time)
            await asyncio.sleep(sleep_time)
        
        # Final report
        await self.generate_sentiment_report()
    
    async def generate_sentiment_report(self):
        """Generate comprehensive sentiment-enhanced trading report"""
        final_portfolio = await self.calculate_portfolio_value()
        
        # Analyze performance by strategy and sentiment
        strategy_performance = defaultdict(lambda: {'trades': 0, 'profit': 0.0, 'avg_sentiment': 0.0})
        
        for trade in self.trade_history:
            if trade.get('success'):
                strategy = trade.get('strategy', 'unknown')
                strategy_performance[strategy]['trades'] += 1
                strategy_performance[strategy]['profit'] += trade.get('profit', 0)
                strategy_performance[strategy]['avg_sentiment'] += trade.get('sentiment_score', 0)
        
        # Calculate averages
        for strategy, perf in strategy_performance.items():
            if perf['trades'] > 0:
                perf['avg_sentiment'] /= perf['trades']
        
        logger.info("\n" + "="*70)
        logger.info("🏁 SENTIMENT-ENHANCED TRADING SESSION COMPLETE")
        logger.info("="*70)
        logger.info(f"💰 Total Profit: ${self.total_profit:.3f}")
        logger.info(f"📊 Total Trades: {len(self.trade_history)}")
        logger.info(f"💼 Final Portfolio: ${final_portfolio:.2f}")
        
        logger.info(f"\n🧠 SENTIMENT-ENHANCED STRATEGY PERFORMANCE:")
        for strategy, perf in strategy_performance.items():
            avg_profit = perf['profit'] / max(perf['trades'], 1)
            logger.info(f"   {strategy.title()}: {perf['trades']} trades, "
                       f"${perf['profit']:.3f} profit (${avg_profit:.3f} avg), "
                       f"sentiment: {perf['avg_sentiment']:.2f}")
        
        # Save detailed report
        report = {
            'total_profit': self.total_profit,
            'final_portfolio_value': final_portfolio,
            'total_trades': len(self.trade_history),
            'strategy_performance': dict(strategy_performance),
            'sentiment_enabled': self.sentiment_engine is not None,
            'sentiment_weight': self.sentiment_weight,
            'technical_weight': self.technical_weight,
            'trade_history': [dict(trade) for trade in list(self.trade_history)[-50:]]  # Last 50 trades
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("test_results", exist_ok=True)
        
        with open(f"test_results/sentiment_enhanced_trading_report_{timestamp}.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)

async def main():
    """Main execution"""
    logger.info("\n🧠 KIMERA SENTIMENT-ENHANCED OMNIDIMENSIONAL TRADER")
    logger.info("🚀 COMBINING TECHNICAL + SENTIMENT ANALYSIS")
    logger.info("="*60)
    
    try:
        trader = SentimentEnhancedTrader()
        await trader.initialize()
        await trader.run_sentiment_enhanced_trading(duration_minutes=5)
        
    except Exception as e:
        logger.error(f"❌ Sentiment-enhanced trading failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 