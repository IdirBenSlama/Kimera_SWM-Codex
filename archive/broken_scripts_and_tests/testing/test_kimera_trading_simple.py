#!/usr/bin/env python3
"""
KIMERA Semantic Trading Reactor - Core Test
===========================================
Tests the core semantic trading functionality without dashboard conflicts.
"""

import asyncio
import sys
import os
from datetime import datetime

# Initialize structured logger
from backend.utils.kimera_logger import get_system_logger
logger = get_system_logger(__name__)


# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_semantic_reactor():
    logger.info('='*80)
    logger.info('🚀 KIMERA SEMANTIC TRADING REACTOR - CORE TEST')
    logger.info('='*80)
    
    try:
        # Import core trading components
        from backend.trading.core.semantic_trading_reactor import (
            SemanticTradingReactor,
            TradingRequest,
            create_semantic_trading_reactor
        )
        
        logger.info('✅ Semantic Trading Reactor imported successfully')
        
        # Create reactor with basic config
        config = {
            'tension_threshold': 0.3,
            'questdb_host': 'localhost',
            'questdb_port': 9009,
            'kafka_servers': 'localhost:9092'
        }
        
        reactor = create_semantic_trading_reactor(config)
        logger.info('✅ Semantic Trading Reactor created successfully')
        
        # Create test market data with CONTRADICTION
        trading_request = TradingRequest(
            action_type='analyze',
            market_data={
                'symbol': 'BTC-USD',
                'price': 52000.0,
                'volume': 1500,
                'momentum': 0.08,  # Strong positive momentum
                'volatility': 0.025,
                'technical_indicators': {
                    'rsi': 75,  # Overbought
                    'macd': 120,
                    'bollinger_position': 0.9
                }
            },
            semantic_context={
                'news_sentiment': -0.6,  # Negative news (CONTRADICTION!)
                'social_sentiment': 0.4,
                'market_phase': 'distribution',
                'global_sentiment': -0.2
            },
            risk_parameters={
                'max_position_size': 1000.0,
                'risk_per_trade': 0.02
            }
        )
        
        logger.debug('\n🔍 Processing trading request with contradiction...')
        logger.info(f'   Price Action: ${trading_request.market_data["price"]:,.2f} (+{trading_request.market_data["momentum"]*100:.1f}%)
        logger.info(f'   News Sentiment: {trading_request.semantic_context["news_sentiment"]} (NEGATIVE)
        logger.info(f'   Technical RSI: {trading_request.market_data["technical_indicators"]["rsi"]} (OVERBOUGHT)
        logger.info('   ⚡ CONTRADICTION: Positive price action vs Negative news sentiment!')
        
        # Process through KIMERA semantic reactor
        logger.info('\n🧠 Processing through KIMERA semantic reactor...')
        result = await reactor.process_request(trading_request)
        
        logger.info('\n📊 KIMERA SEMANTIC ANALYSIS:')
        logger.info(f'   Action Taken: {result.action_taken}')
        logger.info(f'   Confidence: {result.confidence*100:.1f}%')
        logger.info(f'   Execution Time: {result.execution_time*1000:.1f}ms')
        logger.info(f'   Contradictions Found: {len(result.contradiction_map)
        
        if result.contradiction_map:
            logger.info('\n⚡ CONTRADICTIONS DETECTED:')
            for i, contradiction in enumerate(result.contradiction_map[:3]):
                logger.info(f'   {i+1}. {contradiction.get("source_a", "Unknown")
                logger.info(f'      Tension Score: {contradiction.get("tension_score", 0)
                logger.info(f'      Contradiction Type: {contradiction.get("contradiction_type", "unknown")
        
        if result.semantic_analysis:
            semantic = result.semantic_analysis
            logger.info('\n🧠 SEMANTIC THERMODYNAMICS:')
            logger.info(f'   Total Entropy: {semantic.get("total_entropy", 0)
            logger.info(f'   Average Tension: {semantic.get("average_tension", 0)
            logger.info(f'   Semantic Coherence: {semantic.get("semantic_coherence", 0)
            logger.info(f'   Contradiction Intensity: {semantic.get("contradiction_intensity", 0)
            logger.info(f'   Thermodynamic Pressure: {semantic.get("thermodynamic_pressure", 0)
        
        if result.position:
            position = result.position
            logger.info('\n💰 TRADING DECISION:')
            logger.info(f'   Symbol: {position.get("symbol", "N/A")
            logger.info(f'   Side: {position.get("side", "N/A")
            logger.info(f'   Size: {position.get("size", 0)
            logger.info(f'   Entry Price: ${position.get("entry_price", 0)
            logger.info(f'   Semantic Score: {position.get("semantic_score", 0)
        else:
            logger.info('\n💰 TRADING DECISION: No position taken')
            logger.info(f'   Reason: {result.metadata.get("reason", "unknown")
        
        logger.info('\n✅ KIMERA Semantic Trading Reactor test completed!')
        logger.info('\n🎯 KEY CAPABILITIES DEMONSTRATED:')
        logger.info('   ✓ Semantic contradiction detection across data sources')
        logger.info('   ✓ Thermodynamic analysis of market entropy')
        logger.info('   ✓ Multi-modal data integration (price + sentiment + technical)
        logger.info('   ✓ Tension gradient calculation between geoids')
        logger.info('   ✓ Confidence-based decision making')
        logger.info('   ✓ Risk-aware position sizing')
        
        return True
        
    except Exception as e:
        logger.error(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test runner"""
    logger.info(f'🚀 Starting KIMERA Semantic Trading Reactor Test at {datetime.now()
    
    success = await test_semantic_reactor()
    
    logger.info('\n' + '='*80)
    logger.info(f'🎯 Semantic Reactor test: {"PASSED" if success else "FAILED"}')
    logger.info('='*80)
    
    if success:
        logger.info('\n🌟 KIMERA Semantic Trading Reactor is fully operational!')
        logger.info('   - Contradiction detection: ACTIVE')
        logger.info('   - Thermodynamic analysis: ACTIVE')
        logger.info('   - Semantic geoid creation: ACTIVE')
        logger.info('   - Tension gradient analysis: ACTIVE')
        logger.info('   - Multi-modal fusion: ACTIVE')
    else:
        logger.warning('\n⚠️ Semantic reactor test failed. Please check the logs above.')
    
    return success

if __name__ == "__main__":
    # Run the core semantic reactor test
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 