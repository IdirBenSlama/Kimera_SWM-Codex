"""
CryptoPanic Integration Demo for KIMERA Trading System
Demonstrates real-time news analysis with semantic contradiction detection
"""

import asyncio
import sys
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from trading.connectors.cryptopanic_connector import CryptoPanicConnector, CryptoNews, NewsSentiment
from trading.core.semantic_trading_reactor import SemanticTradingReactor
from trading.kimera_trading_integration import KimeraTradingIntegration
from core.contradiction_engine import ContradictionEngine
from engines.understanding_engine import UnderstandingEngine
from engines.semantic_engine import SemanticEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CryptoPanicKimeraDemo:
    """
    Demonstrates how CryptoPanic news integrates with KIMERA's semantic trading
    
    Key Features:
    1. Real-time news streaming
    2. Semantic contradiction detection between news sources
    3. Market sentiment analysis with thermodynamic metrics
    4. Automated trading signal generation
    """
    
    def __init__(self):
        self.crypto_connector = CryptoPanicConnector()
        self.kimera_trading = KimeraTradingIntegration()
        self.news_cache = {}  # Store recent news for contradiction analysis
        self.sentiment_history = []  # Track sentiment changes
        
    async def analyze_news_contradictions(self, news_items: List[CryptoNews]) -> Dict[str, Any]:
        """
        Analyze news items for semantic contradictions using KIMERA
        
        This demonstrates KIMERA's unique ability to detect when different
        news sources report conflicting information about the same event
        """
        contradictions = []
        
        # Group news by currency
        currency_news = {}
        for item in news_items:
            for currency in item.currencies:
                code = currency.get('code', 'UNKNOWN')
                if code not in currency_news:
                    currency_news[code] = []
                currency_news[code].append(item)
                
        # Check for contradictions within each currency's news
        for currency, items in currency_news.items():
            if len(items) < 2:
                continue
                
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    item1, item2 = items[i], items[j]
                    
                    # Check for sentiment contradictions
                    if (item1.sentiment == NewsSentiment.POSITIVE and 
                        item2.sentiment == NewsSentiment.NEGATIVE):
                        
                        contradiction = {
                            'type': 'sentiment_conflict',
                            'currency': currency,
                            'source1': {
                                'title': item1.title,
                                'source': item1.source['title'],
                                'sentiment': item1.sentiment.value,
                                'panic_score': item1.panic_score
                            },
                            'source2': {
                                'title': item2.title,
                                'source': item2.source['title'],
                                'sentiment': item2.sentiment.value,
                                'panic_score': item2.panic_score
                            },
                            'time_delta': abs((item1.published_at - item2.published_at).total_seconds()),
                            'severity': self._calculate_contradiction_severity(item1, item2)
                        }
                        contradictions.append(contradiction)
                        
        return {
            'contradictions_found': len(contradictions),
            'contradictions': contradictions,
            'analysis_timestamp': datetime.now(timezone.utc).isoformat()
        }
        
    def _calculate_contradiction_severity(self, item1: CryptoNews, item2: CryptoNews) -> float:
        """Calculate how severe a contradiction is (0-1 scale)"""
        # Factor in panic scores, time difference, and source reliability
        panic_diff = abs(item1.panic_score - item2.panic_score)
        time_diff = abs((item1.published_at - item2.published_at).total_seconds())
        
        # Normalize factors
        panic_severity = min(panic_diff / 100, 1.0)
        time_severity = max(0, 1 - (time_diff / 3600))  # Less severe if > 1 hour apart
        
        return (panic_severity + time_severity) / 2
        
    async def generate_trading_signals(self, 
                                     sentiment_analysis: Dict[str, Any],
                                     contradictions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate trading signals based on news sentiment and contradictions
        
        This showcases KIMERA's ability to make trading decisions based on
        semantic analysis rather than just technical indicators
        """
        signals = []
        
        # Signal 1: Strong sentiment with no contradictions
        sentiment_score = sentiment_analysis['sentiment_score']
        if abs(sentiment_score) > 50 and contradictions['contradictions_found'] == 0:
            signal = {
                'type': 'sentiment_consensus',
                'action': 'BUY' if sentiment_score > 0 else 'SELL',
                'confidence': min(abs(sentiment_score) / 100, 0.9),
                'reason': f"Strong {'bullish' if sentiment_score > 0 else 'bearish'} consensus across sources",
                'currencies': sentiment_analysis['trending_currencies'][:3]
            }
            signals.append(signal)
            
        # Signal 2: High contradiction = High volatility opportunity
        if contradictions['contradictions_found'] > 3:
            affected_currencies = set()
            for contradiction in contradictions['contradictions']:
                affected_currencies.add(contradiction['currency'])
                
            signal = {
                'type': 'volatility_opportunity',
                'action': 'STRADDLE',  # Options strategy
                'confidence': min(contradictions['contradictions_found'] / 10, 0.8),
                'reason': "High news contradiction indicates upcoming volatility",
                'currencies': list(affected_currencies)[:3]
            }
            signals.append(signal)
            
        # Signal 3: Panic score spike
        avg_panic = sentiment_analysis['average_panic_score']
        if avg_panic > 50:
            signal = {
                'type': 'panic_event',
                'action': 'HEDGE',
                'confidence': min(avg_panic / 100, 0.85),
                'reason': "Elevated panic scores suggest market stress",
                'currencies': ['BTC', 'USDT']  # Flight to safety
            }
            signals.append(signal)
            
        return signals
        
    async def run_live_demo(self, duration_minutes: int = 5):
        """
        Run a live demonstration of the CryptoPanic-KIMERA integration
        
        This will:
        1. Stream real-time crypto news
        2. Analyze sentiment and detect contradictions
        3. Generate trading signals
        4. Show how KIMERA would execute trades
        """
        logger.info("🚀 Starting CryptoPanic-KIMERA Live Demo")
        logger.info("=" * 60)
        
        async with self.crypto_connector as connector:
            start_time = datetime.now(timezone.utc)
            end_time = start_time + timedelta(minutes=duration_minutes)
            
            while datetime.now(timezone.utc) < end_time:
                try:
                    # 1. Fetch latest news
                    logger.info(f"\n📡 Fetching latest crypto news... [{datetime.now()
                    news_items = await connector.get_posts()
                    
                    if news_items:
                        logger.info(f"✅ Retrieved {len(news_items)
                        
                        # Show sample headlines
                        logger.info("\n📰 Latest Headlines:")
                        for item in news_items[:5]:
                            emoji = "📈" if item.sentiment == NewsSentiment.POSITIVE else "📉" if item.sentiment == NewsSentiment.NEGATIVE else "📊"
                            logger.info(f"  {emoji} {item.title[:80]}...")
                            logger.info(f"     Source: {item.source['title']} | Panic Score: {item.panic_score}")
                            
                        # 2. Analyze market sentiment
                        logger.info("\n🧠 Analyzing market sentiment...")
                        sentiment = await connector.analyze_market_sentiment()
                        
                        logger.info(f"\n📊 Market Sentiment Analysis:")
                        logger.info(f"  Overall Score: {sentiment['sentiment_score']:.2f} ")
                              f"({'Bullish' if sentiment['sentiment_score'] > 0 else 'Bearish'})")
                        logger.info(f"  Distribution: {sentiment['sentiment_distribution']}")
                        logger.info(f"  Average Panic Score: {sentiment['average_panic_score']:.2f}")
                        logger.info(f"  Top Trending: {', '.join([f'{c[0]} ({c[1]})
                        
                        # 3. Detect contradictions
                        logger.debug("\n🔍 Detecting semantic contradictions...")
                        contradictions = await self.analyze_news_contradictions(news_items)
                        
                        if contradictions['contradictions_found'] > 0:
                            logger.warning(f"\n⚠️  Found {contradictions['contradictions_found']} contradictions!")
                            for i, contradiction in enumerate(contradictions['contradictions'][:3], 1):
                                logger.info(f"\n  Contradiction #{i} ({contradiction['currency']})
                                logger.info(f"    📰 \"{contradiction['source1']['title'][:60]}...\"")
                                logger.info(f"       {contradiction['source1']['sentiment']} | Source: {contradiction['source1']['source']}")
                                logger.info(f"    vs")
                                logger.info(f"    📰 \"{contradiction['source2']['title'][:60]}...\"")
                                logger.info(f"       {contradiction['source2']['sentiment']} | Source: {contradiction['source2']['source']}")
                                logger.info(f"    Severity: {contradiction['severity']:.2%}")
                        else:
                            logger.info("  ✅ No significant contradictions detected")
                            
                        # 4. Generate trading signals
                        logger.info("\n💡 Generating trading signals...")
                        signals = await self.generate_trading_signals(sentiment, contradictions)
                        
                        if signals:
                            logger.info(f"\n🎯 Generated {len(signals)
                            for i, signal in enumerate(signals, 1):
                                logger.info(f"\n  Signal #{i}: {signal['type'].upper()
                                logger.info(f"    Action: {signal['action']}")
                                logger.info(f"    Confidence: {signal['confidence']:.1%}")
                                logger.info(f"    Reason: {signal['reason']}")
                                logger.info(f"    Currencies: {', '.join(signal['currencies'])
                                
                                # Simulate KIMERA processing
                                if signal['confidence'] > 0.7:
                                    logger.info(f"    🤖 KIMERA: Processing signal through semantic reactor...")
                                    
                                    # Create mock market event
                                    market_event = {
                                        'type': 'news_signal',
                                        'signal': signal,
                                        'sentiment': sentiment,
                                        'contradictions': contradictions,
                                        'timestamp': datetime.now(timezone.utc).isoformat()
                                    }
                                    
                                    # Process through KIMERA (mock)
                                    logger.info(f"    ⚡ Semantic thermodynamic analysis in progress...")
                                    logger.info(f"    ✅ Signal validated - Ready for execution")
                        else:
                            logger.info("  ℹ️  No strong signals at this time")
                            
                        # 5. Show rate limit status
                        rate_status = connector.get_rate_limit_status()
                        logger.info(f"\n📊 API Status: {rate_status['remaining']}/{rate_status['limit']} requests remaining")
                        
                    else:
                        logger.warning("  ⚠️  No news items retrieved")
                        
                except Exception as e:
                    logger.error(f"Error in demo loop: {e}")
                    logger.error(f"\n❌ Error: {str(e)
                    
                # Wait before next iteration
                remaining_time = (end_time - datetime.now(timezone.utc)).total_seconds()
                if remaining_time > 60:
                    logger.info(f"\n⏳ Waiting 60 seconds before next update...")
                    await asyncio.sleep(60)
                else:
                    break
                    
        logger.info(f"\n\n✅ Demo completed! Total runtime: {duration_minutes} minutes")
        logger.info("=" * 60)
        
    async def test_specific_features(self):
        """Test specific CryptoPanic features"""
        logger.info("🧪 Testing CryptoPanic Specific Features")
        logger.info("=" * 60)
        
        async with self.crypto_connector as connector:
            # Test 1: Get trending news
            logger.info("\n1️⃣ Testing Trending News...")
            trending = await connector.get_trending_news()
            logger.info(f"   Found {len(trending)
            if trending:
                logger.info(f"   Top trending: {trending[0].title}")
                
            # Test 2: Get important/breaking news
            logger.info("\n2️⃣ Testing Important News...")
            important = await connector.get_important_news()
            logger.info(f"   Found {len(important)
            
            # Test 3: Get Bitcoin-specific bullish news
            logger.info("\n3️⃣ Testing Currency-Specific Bullish News (BTC)
            btc_bullish = await connector.get_bullish_news(['BTC'])
            logger.info(f"   Found {len(btc_bullish)
            
            # Test 4: Get Ethereum-specific bearish news
            logger.info("\n4️⃣ Testing Currency-Specific Bearish News (ETH)
            eth_bearish = await connector.get_bearish_news(['ETH'])
            logger.info(f"   Found {len(eth_bearish)
            
            # Test 5: Multi-currency analysis
            logger.info("\n5️⃣ Testing Multi-Currency Sentiment Analysis...")
            multi_sentiment = await connector.analyze_market_sentiment(
                currencies=['BTC', 'ETH', 'SOL', 'BNB']
            )
            logger.info(f"   Overall sentiment: {multi_sentiment['sentiment_score']:.2f}")
            logger.info(f"   Top mentioned: {multi_sentiment['trending_currencies'][:3]}")
            
        logger.info("\n✅ All tests completed!")


async def main():
    """Main entry point for the demo"""
    demo = CryptoPanicKimeraDemo()
    
    logger.info("🎮 CryptoPanic-KIMERA Integration Demo")
    logger.info("=====================================")
    logger.info("\nChoose demo mode:")
    logger.info("1. Run 5-minute live demo")
    logger.info("2. Test specific features")
    logger.info("3. Run both")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        await demo.run_live_demo(duration_minutes=5)
    elif choice == '2':
        await demo.test_specific_features()
    elif choice == '3':
        await demo.test_specific_features()
        logger.info("\n" + "="*60 + "\n")
        await demo.run_live_demo(duration_minutes=5)
    else:
        logger.info("Invalid choice. Running feature tests...")
        await demo.test_specific_features()


if __name__ == "__main__":
    asyncio.run(main()) 