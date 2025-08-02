"""
Simple CryptoPanic API Test
Tests the API key and basic functionality
"""

import asyncio
import sys
import os

# Initialize structured logger
from src.utils.kimera_logger import get_system_logger
logger = get_system_logger(__name__)


# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from connectors.cryptopanic_connector import CryptoPanicConnector

async def test_cryptopanic():
    """Test CryptoPanic API with provided key"""
    logger.info("🧪 Testing CryptoPanic API")
    logger.info("=" * 60)
    logger.info(f"API Key: 23675a49e161477a7b2b3c8c4a25743ba6777e8e")
    logger.info(f"Endpoint: https://cryptopanic.com/api/developer/v2/posts/")
    logger.info("=" * 60)
    
    async with CryptoPanicConnector() as connector:
        try:
            # Test 1: Get latest posts
            logger.info("\n1️⃣ Fetching latest crypto news...")
            news = await connector.get_posts()
            logger.info(f"✅ Successfully retrieved {len(news)}")
            
            if news:
                # Show first 3 headlines
                logger.info("\n📰 Latest Headlines:")
                for i, item in enumerate(news[:3], 1):
                    logger.info(f"\n{i}. {item.title}")
                    logger.info(f"   Source: {item.source['title']}")
                    logger.info(f"   Published: {item.published_at}")
                    logger.info(f"   Sentiment: {item.sentiment.value}")
                    logger.info(f"   Panic Score: {item.panic_score}")
                    if item.currencies:
                        currencies = ', '.join([c['code'] for c in item.currencies[:5]])
                        logger.info(f"   Currencies: {currencies}")
                    logger.info(f"   URL: {item.url}")
                    
            # Test 2: Get market sentiment
            logger.info("\n\n2️⃣ Analyzing market sentiment...")
            sentiment = await connector.analyze_market_sentiment()
            logger.info(f"✅ Sentiment analysis complete")
            logger.info(f"   Overall Score: {sentiment['sentiment_score']:.2f}")
            logger.info(f"   Sentiment Distribution:")
            for sent_type, count in sentiment['sentiment_distribution'].items():
                logger.info(f"     - {sent_type}: {count}")
            logger.info(f"   Average Panic Score: {sentiment['average_panic_score']:.2f}")
            logger.info(f"   Top Trending Currencies:")
            for currency, mentions in sentiment['trending_currencies'][:5]:
                logger.info(f"     - {currency}: {mentions} mentions")
                
            # Test 3: Get Bitcoin-specific news
            logger.info("\n\n3️⃣ Fetching Bitcoin-specific news...")
            btc_news = await connector.get_posts(currencies=['BTC'])
            logger.info(f"✅ Found {len(btc_news)}")
            
            # Test 4: Get important/breaking news
            logger.info("\n\n4️⃣ Fetching important/breaking news...")
            important = await connector.get_important_news()
            logger.info(f"✅ Found {len(important)}")
            
            # Check rate limits
            rate_status = connector.get_rate_limit_status()
            logger.info(f"\n\n📊 Rate Limit Status:")
            logger.info(f"   Remaining: {rate_status['remaining']}/{rate_status['limit']}")
            logger.info(f"   Reset: {rate_status['reset']}")
            
            logger.info("\n\n✅ All tests passed! CryptoPanic API is working correctly.")
            
        except Exception as e:
            logger.error(f"\n❌ Error: {str(e)}")
            logger.info(f"   Type: {type(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_cryptopanic()) 