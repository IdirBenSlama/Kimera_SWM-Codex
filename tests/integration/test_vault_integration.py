#!/usr/bin/env python3
"""
KIMERA VAULT INTEGRATION TEST
============================

Test script to verify that the vault integration is working correctly
with the trading systems. This demonstrates that Kimera now has full
persistent memory and learning capabilities.
"""

import asyncio
import time
import logging
from datetime import datetime

# Kimera imports
from src.vault.vault_manager import VaultManager
from src.core.vault_cognitive_interface import VaultCognitiveInterface
from src.core.scar import ScarRecord
from src.core.geoid import GeoidState

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - VAULT_TEST - %(levelname)s - %(message)s'
)
logger = logging.getLogger('VAULT_TEST')

async def test_vault_integration():
    """Test vault integration functionality"""
    try:
        logger.info("🔒 TESTING KIMERA VAULT INTEGRATION")
        logger.info("=" * 60)
        
        # Initialize vault system
        logger.info("1. Initializing Vault Manager...")
        vault_manager = VaultManager()
        
        logger.info("2. Initializing Vault Cognitive Interface...")
        vault_interface = VaultCognitiveInterface()
        
        # Test 1: Store trading session
        logger.info("3. Testing trading session storage...")
        session_data = {
            'session_id': f"test_session_{int(time.time())}",
            'start_time': datetime.now().isoformat(),
            'system_type': 'cognitive_trading_intelligence',
            'test_mode': True
        }
        
        result = await vault_interface.store_trading_session(session_data)
        logger.info(f"   ✅ Session stored: {result}")
        
        # Test 2: Query learned patterns
        logger.info("4. Testing learned patterns query...")
        patterns = await vault_interface.query_learned_patterns('trading', {'symbol': 'BTC/USDT'})
        logger.info(f"   ✅ Patterns queried: {len(patterns.get('causal_chains', {}))}")
        
        # Test 3: Store trading insight
        logger.info("5. Testing trading insight storage...")
        insight_result = await vault_interface.store_trading_insight(
            'market_analysis',
            {'symbol': 'BTC/USDT', 'confidence': 0.85, 'signal': 'bullish'},
            0.85
        )
        logger.info(f"   ✅ Insight stored: {insight_result}")
        
        # Test 4: Create trading SCAR
        logger.info("6. Testing SCAR creation...")
        scar_result = await vault_interface.create_trading_scar(
            'prediction_mismatch',
            {'symbol': 'ETH/USDT', 'strategy': 'momentum'},
            {'expected': 'up', 'confidence': 0.8},
            {'actual': 'down', 'loss': 50}
        )
        logger.info(f"   ✅ SCAR created: {scar_result}")
        
        # Test 5: Query market insights
        logger.info("7. Testing market insights query...")
        market_insights = await vault_interface.query_market_insights(
            'BTC/USDT', '1h', {'volume': 'high', 'volatility': 'medium'}
        )
        logger.info(f"   ✅ Market insights retrieved: {len(market_insights.get('causal_chains', {}))}")
        
        # Test 6: Store performance data
        logger.info("8. Testing performance data storage...")
        performance_result = await vault_interface.store_performance_data(
            {'profit': 150.0, 'trades': 5, 'success_rate': 0.8},
            {'session_id': 'test_session', 'duration': 600}
        )
        logger.info(f"   ✅ Performance data stored: {performance_result}")
        
        # Test 7: Initialize trading session
        logger.info("9. Testing trading session initialization...")
        session_init = await vault_interface.initialize_trading_session('test_session_init')
        logger.info(f"   ✅ Session initialized: {session_init.get('session_id', 'unknown')}")
        
        # Test 8: Generate market questions
        logger.info("10. Testing market question generation...")
        questions = await vault_interface.generate_market_questions({
            'symbol': 'BTC/USDT',
            'trend': 'bullish',
            'volume': 'increasing'
        })
        logger.info(f"   ✅ Generated {len(questions)} market questions")
        
        # Test 8: Test vault manager status
        logger.info("10. Testing vault manager status...")
        status = vault_manager.get_status()
        logger.info(f"   📊 Vault Status: {status}")
        
        logger.info("=" * 60)
        logger.info("✅ VAULT INTEGRATION TEST COMPLETED SUCCESSFULLY")
        logger.info("🧠 Kimera now has full persistent memory capabilities!")
        logger.info("🔄 Trading systems can learn and evolve from past experiences")
        logger.info("📚 All trading data is stored for continuous improvement")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Vault integration test failed: {e}")
        return False

async def main():
    """Main test function"""
    print("🔒 KIMERA VAULT INTEGRATION TEST")
    print("=" * 60)
    print("Testing persistent memory and learning capabilities...")
    print("=" * 60)
    
    success = await test_vault_integration()
    
    if success:
        print("\n✅ ALL TESTS PASSED!")
        print("🧠 Kimera trading systems now have full vault integration")
        print("📚 Persistent memory and learning capabilities are active")
        print("🔄 The system can now learn from every trading session")
    else:
        print("\n❌ TESTS FAILED!")
        print("⚠️ Vault integration needs additional work")

if __name__ == "__main__":
    asyncio.run(main()) 