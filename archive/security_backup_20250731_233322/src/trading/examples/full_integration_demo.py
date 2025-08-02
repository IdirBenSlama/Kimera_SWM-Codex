"""
Full Trading Integration Demo
=============================

This script demonstrates the complete KIMERA trading module with:
- CryptoPanic for real-time news sentiment
- TAAPI for technical analysis indicators
- Contradiction detection between news, technicals, and price
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import trading components
from src.trading import create_kimera_trading_system
from src.trading.connectors.cryptopanic_connector import create_cryptopanic_connector
from src.trading.connectors.taapi_connector import create_taapi_connector, Indicator, Timeframe


async def test_full_integration():
    """Test the complete integration with real APIs"""
    logger.info("="*80)
    logger.info("KIMERA Trading - Full Integration Demo")
    logger.info("="*80)
    
    # API Keys (use environment variables or the provided testnet keys)
    cryptopanic_key = os.getenv('CRYPTOPANIC_API_KEY', '23675a49e161477a7b2b3c8c4a25743ba6777e8e')
    taapi_key = os.getenv('TAAPI_API_KEY', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJjbHVlIjoiNjg1NjAxODg4MDZmZjE2NTFlYTE1ZDk5IiwiaWF0IjoxNzUwNDY2OTcwLCJleHAiOjMzMjU0OTMwOTcwfQ.vNLwdY6pKkmcT-Hm1pSjaKnJuw3B0daeDoPvvY4TGfQ')
    
    logger.info(f"CryptoPanic API: {cryptopanic_key[:10]}...")
    logger.info(f"TAAPI API: {taapi_key[:20]}...")
    
    # Test 1: Fetch real-time news sentiment
    logger.info("\n" + "="*60)
    logger.info("TEST 1: Real-time News Sentiment Analysis")
    logger.info("="*60)
    
    crypto_connector = create_cryptopanic_connector(cryptopanic_key, testnet=True)
    
    async with crypto_connector:
        # Get BTC sentiment
        btc_sentiment = await crypto_connector.analyze_sentiment_trend('BTC', hours=3)
        
        logger.info(f"\nBTC News Sentiment:")
        logger.info(f"  Overall Score: {btc_sentiment['sentiment_score']:.2f}")
        logger.info(f"  News Count: {btc_sentiment['total_news']}")
        logger.info(f"  Bullish: {btc_sentiment['bullish_count']}")
        logger.info(f"  Bearish: {btc_sentiment['bearish_count']}")
        
        # Show latest news
        if btc_sentiment.get('latest_news'):
            logger.info("\n  Latest News:")
            for news in btc_sentiment['latest_news'][:3]:
                logger.info(f"    â€¢ {news.title[:80]}...")
                logger.info(f"      Sentiment: {news.sentiment_score:.2f}")
    
    # Test 2: Fetch technical indicators
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Technical Analysis Indicators")
    logger.info("="*60)
    
    taapi_connector = create_taapi_connector(taapi_key)
    
    async with taapi_connector:
        # Get technical indicators for BTC
        indicators = await taapi_connector.get_bulk_indicators(
            indicators=[Indicator.RSI, Indicator.MACD, Indicator.ADX, Indicator.BBands],
            symbol='BTC/USDT',
            exchange='binance',
            timeframe=Timeframe.ONE_HOUR
        )
        
        logger.info(f"\nBTC Technical Indicators (1H):")
        for name, analysis in indicators.items():
            if isinstance(analysis.value, (int, float)):
                logger.info(f"  {name.upper()}: {analysis.value:.2f}")
            elif isinstance(analysis.value, dict):
                logger.info(f"  {name.upper()}: {analysis.value}")
        
        # Analyze trend
        trend_analysis = await taapi_connector.analyze_trend('BTC/USDT')
        logger.info(f"\n  Trend Analysis:")
        logger.info(f"    Direction: {trend_analysis['trend_direction']}")
        logger.info(f"    Score: {trend_analysis['trend_score']:.2f}")
        logger.info(f"    Strength: {trend_analysis['trend_strength']:.2f}")
    
    # Test 3: Integrated trading system with contradiction detection
    logger.info("\n" + "="*60)
    logger.info("TEST 3: Integrated Trading with Contradiction Detection")
    logger.info("="*60)
    
    # Configuration with both APIs
    config = {
        'tension_threshold': 0.4,
        'max_position_size': 1000,
        'risk_per_trade': 0.02,
        'enable_paper_trading': True,
        'enable_sentiment_analysis': True,
        'enable_news_processing': True,
        'exchanges': {
            'cryptopanic_api_key': cryptopanic_key
        },
        'taapi_api_key': taapi_key
    }
    
    # Create trading system
    trading_system = create_kimera_trading_system(config)
    await trading_system.start()
    
    # Create market event with potential contradictions
    market_event = {
        'market_data': {
            'symbol': 'BTC-USD',
            'price': 50000,
            'volume': 2500,
            'momentum': 0.03,  # Positive price momentum
            'volatility': 0.02,
            'trend': 'bullish'
        }
    }
    
    logger.info("\nðŸ“Š Processing market event through KIMERA...")
    logger.info(f"   Symbol: {market_event['market_data']['symbol']}")
    logger.info(f"   Price: ${market_event['market_data']['price']:,.2f}")
    logger.info(f"   Price Momentum: {market_event['market_data']['momentum']:.1%} (Bullish)")
    
    # Process through the trading system
    result = await trading_system.process_market_event(market_event)
    
    logger.info("\nðŸ“ˆ Trading Analysis Result:")
    logger.info(f"   Status: {result['status']}")
    
    if 'analysis' in result:
        analysis = result['analysis']
        logger.info(f"   Action: {analysis.action_taken}")
        logger.info(f"   Confidence: {analysis.confidence:.1%}")
        
        # Show semantic analysis
        if analysis.semantic_analysis:
            logger.info("\n   Semantic Analysis:")
            logger.info(f"     - Total Entropy: {analysis.semantic_analysis.get('total_entropy', 0):.3f}")
            logger.info(f"     - Average Tension: {analysis.semantic_analysis.get('average_tension', 0):.3f}")
            logger.info(f"     - Thermodynamic Pressure: {analysis.semantic_analysis.get('thermodynamic_pressure', 0):.3f}")
        
        # Show detected contradictions
        if analysis.contradiction_map:
            logger.info(f"\n   âš¡ Contradictions Detected: {len(analysis.contradiction_map)}")
            for i, contradiction in enumerate(analysis.contradiction_map):
                logger.info(f"\n   Contradiction #{i+1}:")
                logger.info(f"     Sources: {contradiction.get('source_a')} vs {contradiction.get('source_b')}")
                logger.info(f"     Tension Score: {contradiction.get('tension_score', 0):.3f}")
                logger.info(f"     Opportunity Type: {contradiction.get('opportunity_type', 'unknown')}")
    
    # Test 4: Check for specific contradictions
    logger.info("\n" + "="*60)
    logger.info("TEST 4: Specific Contradiction Analysis")
    logger.info("="*60)
    
    # Compare news sentiment with technical indicators
    news_sentiment_score = btc_sentiment['sentiment_score']
    
    # Get RSI value for comparison
    rsi_value = None
    if 'rsi' in indicators and isinstance(indicators['rsi'].value, (int, float)):
        rsi_value = indicators['rsi'].value
    
    logger.info(f"\nCross-Domain Analysis:")
    logger.info(f"  News Sentiment: {news_sentiment_score:.2f} ({'Bullish' if news_sentiment_score > 0 else 'Bearish'})")
    
    if rsi_value:
        logger.info(f"  RSI: {rsi_value:.2f} ({'Oversold' if rsi_value < 30 else 'Overbought' if rsi_value > 70 else 'Neutral'})")
        
        # Check for contradiction
        if (news_sentiment_score > 0.3 and rsi_value > 70) or (news_sentiment_score < -0.3 and rsi_value < 30):
            logger.info("  âš¡ CONTRADICTION: News sentiment contradicts technical indicators!")
            logger.info("     This could represent a trading opportunity")
    
    logger.info(f"  Price Momentum: {market_event['market_data']['momentum']:.1%}")
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    
    status = trading_system.get_status()
    logger.info(f"System Status:")
    logger.info(f"  - Running: {status['is_running']}")
    logger.info(f"  - Active Positions: {status['active_positions']}")
    logger.info(f"  - Total Trades: {status['total_trades']}")
    logger.info(f"  - Connected Exchanges: {status['connected_exchanges']}")
    
    await trading_system.stop()


async def test_indicator_contradictions():
    """Test contradiction detection between different timeframes"""
    logger.info("\n" + "="*80)
    logger.info("Bonus Test: Multi-Timeframe Contradiction Detection")
    logger.info("="*80)
    
    taapi_key = os.getenv('TAAPI_API_KEY', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJjbHVlIjoiNjg1NjAxODg4MDZmZjE2NTFlYTE1ZDk5IiwiaWF0IjoxNzUwNDY2OTcwLCJleHAiOjMzMjU0OTMwOTcwfQ.vNLwdY6pKkmcT-Hm1pSjaKnJuw3B0daeDoPvvY4TGfQ')
    taapi_connector = create_taapi_connector(taapi_key)
    
    async with taapi_connector:
        # Detect contradictions across timeframes
        contradictions = await taapi_connector.detect_indicator_contradictions(
            symbol='BTC/USDT',
            exchange='binance',
            timeframes=[Timeframe.FIFTEEN_MIN, Timeframe.ONE_HOUR, Timeframe.FOUR_HOUR]
        )
        
        if contradictions:
            logger.info(f"\nFound {len(contradictions)} indicator contradictions:")
            for contradiction in contradictions:
                logger.info(f"\n  â€¢ {contradiction['type']}")
                logger.info(f"    {contradiction['description']}")
                logger.info(f"    Severity: {contradiction['severity']:.2f}")
        else:
            logger.info("\nNo significant contradictions found between timeframes")


async def main():
    """Run all tests"""
    try:
        # Main integration test
        await test_full_integration()
        
        # Bonus timeframe contradiction test
        await test_indicator_contradictions()
        
        logger.info("\n" + "="*80)
        logger.info("âœ… Full integration demo completed successfully!")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main()) 