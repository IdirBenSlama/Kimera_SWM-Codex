"""
Premium API Connection Test
Verifies all premium data sources are working correctly
"""

import asyncio
import sys
import os
import logging
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from connectors.premium_data_connectors import PremiumDataManager, DataProvider
from connectors.cryptopanic_connector import CryptoPanicConnector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_alpha_vantage():
    """Test Alpha Vantage API"""
    logger.info("🔍 Testing Alpha Vantage API...")
    
    async with PremiumDataManager() as manager:
        try:
            # Test intraday data
            data = await manager.get_intraday_data_av("AAPL", "1min")
            if data:
                logger.info(f"✅ Alpha Vantage: Got {len(data)} data points for AAPL")
                logger.info(f"   Latest: ${data[0].close:.2f} at {data[0].timestamp}")
                return True
            else:
                logger.warning("⚠️ Alpha Vantage: No data returned")
                return False
                
        except Exception as e:
            logger.error(f"❌ Alpha Vantage Error: {e}")
            return False

async def test_finnhub():
    """Test Finnhub API"""
    logger.info("🔍 Testing Finnhub API...")
    
    async with PremiumDataManager() as manager:
        try:
            # Test real-time quote
            quote = await manager.get_quote_finnhub("AAPL")
            if quote and 'c' in quote:
                logger.info(f"✅ Finnhub Quote: AAPL at ${quote['c']:.2f}")
                logger.info(f"   Change: {quote.get('d', 0):+.2f} ({quote.get('dp', 0):+.2f}%)")
                
                # Test news
                news = await manager.get_news_finnhub("general")
                logger.info(f"✅ Finnhub News: {len(news)} articles")
                if news:
                    logger.info(f"   Latest: {news[0].headline[:80]}...")
                    
                return True
            else:
                logger.warning("⚠️ Finnhub: No quote data returned")
                return False
                
        except Exception as e:
            logger.error(f"❌ Finnhub Error: {e}")
            return False

async def test_twelve_data():
    """Test Twelve Data API"""
    logger.info("🔍 Testing Twelve Data API...")
    
    async with PremiumDataManager() as manager:
        try:
            # Test real-time price
            price = await manager.get_real_time_price_td("AAPL")
            if price and 'price' in price:
                logger.info(f"✅ Twelve Data: AAPL at ${price['price']}")
                
                # Test technical indicators
                rsi = await manager.get_technical_indicators_td("AAPL", "rsi")
                if rsi and 'values' in rsi:
                    logger.info(f"✅ Twelve Data RSI: {len(rsi['values'])} data points")
                    if rsi['values']:
                        latest_rsi = rsi['values'][0].get('rsi', 'N/A')
                        logger.info(f"   Latest RSI: {latest_rsi}")
                        
                return True
            else:
                logger.warning("⚠️ Twelve Data: No price data returned")
                return False
                
        except Exception as e:
            logger.error(f"❌ Twelve Data Error: {e}")
            return False

async def test_cryptopanic():
    """Test CryptoPanic API"""
    logger.info("🔍 Testing CryptoPanic API...")
    
    try:
        async with CryptoPanicConnector() as connector:
            posts = await connector.get_posts()
            if posts:
                logger.info(f"✅ CryptoPanic: {len(posts)} posts retrieved")
                
                # Test sentiment analysis
                sentiment = await connector.analyze_market_sentiment()
                logger.info(f"✅ CryptoPanic Sentiment: {sentiment['sentiment_score']:.1f}")
                logger.info(f"   Trending: {', '.join(sentiment['trending_currencies'][:3])}")
                
                return True
            else:
                logger.warning("⚠️ CryptoPanic: No posts returned")
                return False
                
    except Exception as e:
        logger.error(f"❌ CryptoPanic Error: {e}")
        return False

async def test_comprehensive_intelligence():
    """Test comprehensive intelligence generation"""
    logger.info("🔍 Testing Comprehensive Intelligence...")
    
    async with PremiumDataManager() as manager:
        try:
            intelligence = await manager.generate_trading_intelligence("AAPL")
            
            logger.info(f"✅ Intelligence Report Generated for AAPL")
            logger.info(f"   Data Sources: {', '.join(intelligence['data_sources'])}")
            
            # Check data quality
            market_data = intelligence.get('market_data', {})
            sentiment_data = intelligence.get('sentiment_analysis', {})
            technical_data = intelligence.get('technical_analysis', {})
            
            data_sources = []
            if market_data.get('alpha_vantage_intraday'):
                data_sources.append("Alpha Vantage")
            if market_data.get('finnhub_quote'):
                data_sources.append("Finnhub")
            if market_data.get('twelve_data_price'):
                data_sources.append("Twelve Data")
            if sentiment_data.get('recent_news'):
                data_sources.append("News")
            if technical_data:
                data_sources.append("Technical")
                
            logger.info(f"   Active Sources: {', '.join(data_sources)}")
            
            # Show summary
            summary = intelligence.get('summary', {})
            if summary:
                logger.info(f"   Signal Strength: {summary.get('signal_strength', 'unknown')}")
                logger.info(f"   Price Change: {summary.get('price_change_pct', 'N/A')}%")
                logger.info(f"   News Activity: {summary.get('news_activity', 'unknown')}")
                
            return True
            
        except Exception as e:
            logger.error(f"❌ Intelligence Generation Error: {e}")
            return False

async def run_api_tests():
    """Run all API tests"""
    logger.info("🚀 Starting Premium API Connection Tests")
    logger.info("=" * 60)
    
    test_results = {}
    
    # Test each API
    test_results['alpha_vantage'] = await test_alpha_vantage()
    await asyncio.sleep(1)  # Rate limiting
    
    test_results['finnhub'] = await test_finnhub()
    await asyncio.sleep(1)
    
    test_results['twelve_data'] = await test_twelve_data()
    await asyncio.sleep(1)
    
    test_results['cryptopanic'] = await test_cryptopanic()
    await asyncio.sleep(1)
    
    # Test comprehensive intelligence
    test_results['intelligence'] = await test_comprehensive_intelligence()
    
    # Generate report
    logger.info("\n" + "=" * 60)
    logger.info("📊 API TEST RESULTS")
    logger.info("=" * 60)
    
    working_apis = []
    failed_apis = []
    
    for api, result in test_results.items():
        status = "✅ WORKING" if result else "❌ FAILED"
        logger.info(f"{api.upper().replace('_', ' ')}: {status}")
        
        if result:
            working_apis.append(api)
        else:
            failed_apis.append(api)
            
    logger.info(f"\n📈 Working APIs: {len(working_apis)}/5")
    logger.info(f"📉 Failed APIs: {len(failed_apis)}/5")
    
    if len(working_apis) >= 3:
        logger.info("\n🎉 PREMIUM MODE READY!")
        logger.info("Sufficient APIs working for premium trading")
    elif len(working_apis) >= 2:
        logger.info("\n⚠️ PARTIAL PREMIUM MODE")
        logger.info("Some APIs working, reduced functionality")
    else:
        logger.info("\n❌ PREMIUM MODE UNAVAILABLE")
        logger.info("Too few APIs working for premium features")
        
    # Recommendations
    if failed_apis:
        logger.info(f"\n💡 RECOMMENDATIONS:")
        for api in failed_apis:
            if api == 'alpha_vantage':
                logger.info("• Check Alpha Vantage API key and rate limits")
            elif api == 'finnhub':
                logger.info("• Verify Finnhub API key and subscription")
            elif api == 'twelve_data':
                logger.info("• Confirm Twelve Data API key is active")
            elif api == 'cryptopanic':
                logger.info("• Check CryptoPanic API key and limits")
                
    return test_results

async def main():
    """Main test runner"""
    logger.debug("🔧 KIMERA Premium API Connection Test")
    logger.info("Testing all premium data sources...")
    logger.info()
    
    results = await run_api_tests()
    
    # Ask user if they want to run a quick demo
    if sum(results.values()) >= 2:
        logger.info("\n" + "=" * 60)
        choice = input("APIs are working! Run premium demo? (y/n): ").lower().strip()
        
        if choice == 'y':
            logger.info("Starting premium demo...")
            # Import and run the premium demo
            try:
                from premium_kimera_demo import PremiumKimeraDemo
                demo = PremiumKimeraDemo(starting_capital=1.0)
                await demo.run_premium_demo(cycles=3, interval=20)
            except Exception as e:
                logger.error(f"Demo failed: {e}")
                logger.error("Demo failed - you can run it manually with:")
                logger.info("python premium_kimera_demo.py")
    else:
        logger.error("\n❌ Not enough APIs working for demo")
        logger.info("Please check your API keys and try again")

if __name__ == "__main__":
    asyncio.run(main()) 