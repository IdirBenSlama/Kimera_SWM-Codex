"""
Simple test to verify KIMERA Trading Module integration
"""

import sys
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_trading_module():
    logger.info("="*60)
    logger.info("KIMERA TRADING MODULE - INTEGRATION TEST")
    logger.info("="*60)
    
    # Test 1: Import core components
    logger.info("\nTest 1: Importing KIMERA core components...")
    try:
        from backend.engines.contradiction_engine import ContradictionEngine
        from backend.engines.thermodynamics import SemanticThermodynamicsEngine
        from backend.core.geoid import GeoidState
        logger.info("✅ Core components imported successfully")
    except Exception as e:
        logger.error(f"❌ Failed to import core components: {e}")
        return False
    
    # Test 2: Import trading module
    logger.info("\nTest 2: Importing trading module...")
    try:
        from backend.trading import (
            create_kimera_trading_system,
            process_trading_opportunity,
            SemanticTradingReactor,
            TradingRequest
        )
        logger.info("✅ Trading module imported successfully")
    except Exception as e:
        logger.error(f"❌ Failed to import trading module: {e}")
        return False
    
    # Test 3: Create trading reactor
    logger.info("\nTest 3: Creating trading reactor...")
    try:
        reactor = SemanticTradingReactor(
            config={'tension_threshold': 0.4},
            reactor_interface=None
        )
        logger.info("✅ Trading reactor created successfully")
    except Exception as e:
        logger.error(f"❌ Failed to create trading reactor: {e}")
        return False
    
    # Test 4: Process test market data
    logger.info("\nTest 4: Processing test market data...")
    try:
        test_request = TradingRequest(
            action_type='analyze',
            market_data={
                'symbol': 'BTC-USD',
                'price': 50000,
                'volume': 1000,
                'momentum': 0.05,  # Positive
                'volatility': 0.02
            },
            semantic_context={
                'news_sentiment': -0.8,  # Negative (contradiction!)
                'social_sentiment': 0.3
            },
            risk_parameters={
                'max_position_size': 1000,
                'risk_per_trade': 0.02
            }
        )
        
        result = await reactor.process_request(test_request)
        
        logger.info(f"✅ Analysis completed:")
        logger.info(f"   - Action: {result.action_taken}")
        logger.info(f"   - Confidence: {result.confidence:.1%}")
        logger.info(f"   - Contradictions: {len(result.contradiction_map)}")
        
        if result.contradiction_map:
            logger.info("   - Detected contradictions:")
            for c in result.contradiction_map:
                logger.info(f"     • {c['source_a']} vs {c['source_b']} (tension: {c['tension_score']:.2f})")
        
    except Exception as e:
        logger.error(f"❌ Failed to process market data: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 5: Test convenience function
    logger.info("\nTest 5: Testing convenience function...")
    try:
        test_event = {
            'market_data': {
                'symbol': 'ETH-USD',
                'price': 3000,
                'volume': 5000,
                'momentum': 0.02,
                'volatility': 0.015
            },
            'context': {
                'news_sentiment': 0.7,
                'social_sentiment': -0.3
            }
        }
        
        result = await process_trading_opportunity(test_event)
        logger.info(f"✅ Convenience function works: {result['status']}")
        
    except Exception as e:
        logger.error(f"❌ Failed to use convenience function: {e}")
        return False
    
    logger.info("\n" + "="*60)
    logger.info("✅ ALL TESTS PASSED - Trading module is ready!")
    logger.info("="*60)
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_trading_module())
    sys.exit(0 if success else 1) 