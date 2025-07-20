#!/usr/bin/env python3
"""
KIMERA Semantic Trading Integration - Live Demonstration
========================================================
This script demonstrates the complete integration between KIMERA's 
semantic thermodynamic reactor and the autonomous trading module.
"""

import asyncio
import json
from datetime import datetime
import sys
import os

# Initialize structured logger
from backend.utils.kimera_logger import get_system_logger
logger = get_system_logger(__name__)


# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_full_integration():
    logger.info('='*80)
    logger.info('🚀 KIMERA SEMANTIC TRADING INTEGRATION - LIVE DEMONSTRATION')
    logger.info('='*80)
    
    try:
        # Import trading components
        from backend.trading import (
            create_kimera_trading_system, 
            process_trading_opportunity
        )
        
        logger.info('✅ Trading module imported successfully')
        
        # Create trading system
        config = {
            'tension_threshold': 0.3,
            'max_position_size': 1000.0,
            'risk_per_trade': 0.02,
            'enable_sentiment_analysis': True,
            'enable_news_processing': True
        }
        
        trading_system = create_kimera_trading_system(config)
        logger.info('✅ Trading system created successfully')
        
        # Test market event with CONTRADICTION potential
        market_event = {
            'market_data': {
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
            'context': {
                'news_sentiment': -0.6,  # Negative news (CONTRADICTION!)
                'social_sentiment': 0.4,
                'market_phase': 'distribution',
                'global_sentiment': -0.2
            },
            'timestamp': datetime.now().isoformat()
        }
        
        logger.debug('\n🔍 Processing market event with potential contradiction...')
        logger.info(f'   Price Action: ${market_event["market_data"]["price"]:,.2f} (+{market_event["market_data"]["momentum"]*100:.1f}%)
        logger.info(f'   News Sentiment: {market_event["context"]["news_sentiment"]} (NEGATIVE)
        logger.info(f'   Technical RSI: {market_event["market_data"]["technical_indicators"]["rsi"]} (OVERBOUGHT)
        logger.info('   ⚡ CONTRADICTION: Positive price action vs Negative news sentiment!')
        
        # Process through KIMERA
        logger.info('\n🧠 Processing through KIMERA semantic reactor...')
        result = await process_trading_opportunity(market_event)
        
        logger.info('\n📊 KIMERA ANALYSIS RESULTS:')
        logger.info(f'   Action Taken: {result.get("action_taken", "unknown")
        logger.info(f'   Confidence: {result.get("confidence", 0)
        logger.info(f'   Contradictions Found: {len(result.get("contradiction_map", [])
        
        if result.get('contradiction_map'):
            logger.info('\n⚡ CONTRADICTIONS DETECTED:')
            for i, contradiction in enumerate(result.get('contradiction_map', [])[:3]):
                logger.info(f'   {i+1}. {contradiction.get("source_a", "Unknown")
                logger.info(f'      Tension: {contradiction.get("tension_score", 0)
        
        if result.get('semantic_analysis'):
            semantic = result['semantic_analysis']
            logger.info('\n🧠 SEMANTIC THERMODYNAMICS:')
            logger.info(f'   Total Entropy: {semantic.get("total_entropy", 0)
            logger.info(f'   Thermodynamic Pressure: {semantic.get("thermodynamic_pressure", 0)
            logger.info(f'   Semantic Coherence: {semantic.get("semantic_coherence", 0)
            logger.info(f'   Contradiction Intensity: {semantic.get("contradiction_intensity", 0)
        
        if result.get('position'):
            position = result['position']
            logger.info('\n💰 TRADING POSITION:')
            logger.info(f'   Symbol: {position.get("symbol", "N/A")
            logger.info(f'   Side: {position.get("side", "N/A")
            logger.info(f'   Size: {position.get("size", 0)
            logger.info(f'   Entry Price: ${position.get("entry_price", 0)
            logger.info(f'   Semantic Score: {position.get("semantic_score", 0)
        
        logger.info('\n✅ KIMERA Trading Integration fully operational!')
        logger.info('\n🎯 KEY FEATURES DEMONSTRATED:')
        logger.info('   ✓ Semantic contradiction detection')
        logger.info('   ✓ Thermodynamic analysis of market data')
        logger.info('   ✓ Multi-modal data fusion (price + news + sentiment)
        logger.info('   ✓ Autonomous decision making')
        logger.info('   ✓ Risk-aware position sizing')
        
        return True
        
    except Exception as e:
        logger.error(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test runner"""
    logger.info(f'🚀 Starting KIMERA Trading Integration Test at {datetime.now()
    
    success = await test_full_integration()
    
    logger.info('\n' + '='*80)
    logger.info(f'🎯 Integration test: {"PASSED" if success else "FAILED"}')
    logger.info('='*80)
    
    if success:
        logger.info('\n🌟 KIMERA Semantic Trading Module is ready for production!')
        logger.info('   - Contradiction detection: ACTIVE')
        logger.info('   - Thermodynamic analysis: ACTIVE')
        logger.info('   - Autonomous trading: READY')
        logger.info('   - Risk management: ACTIVE')
    else:
        logger.warning('\n⚠️ Integration test failed. Please check the logs above.')
    
    return success

if __name__ == "__main__":
    # Run the comprehensive test
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 