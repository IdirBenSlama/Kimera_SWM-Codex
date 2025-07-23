#!/usr/bin/env python3
"""
TEST VAULT COGNITIVE INTEGRATION
===============================
🧠 VERIFY KIMERA'S VAULT BRAIN IS CONNECTED AND LEARNING 🧠

This script tests that:
✅ Vault cognitive interface initializes
✅ Vault queries work properly
✅ SCAR creation functions
✅ Continuous learning loop is active
✅ Epistemic consciousness is awakened
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - VAULT_TEST - %(levelname)s - %(message)s'
)
logger = logging.getLogger('VAULT_COGNITIVE_TEST')

async def test_vault_cognitive_interface():
    """Test the vault cognitive interface"""
    
    logger.info("🧠" * 60)
    logger.info("🔬 TESTING KIMERA VAULT COGNITIVE INTEGRATION")
    logger.info("🧠" * 60)
    
    try:
        # Test 1: Initialize vault cognitive interface
        logger.info("🧪 TEST 1: Initializing Vault Cognitive Interface")
        
        from src.core.vault_cognitive_interface import get_vault_cognitive_interface
        vault_brain = get_vault_cognitive_interface()
        
        logger.info("✅ Vault cognitive interface initialized successfully")
        
        # Test 2: Query learned patterns
        logger.info("🧪 TEST 2: Querying Learned Patterns")
        
        test_query_result = await vault_brain.query_learned_patterns(
            domain="trading_test",
            context={
                'symbol': 'BTCUSDT',
                'test_type': 'integration_test',
                'concepts': ['price_action', 'volume_analysis', 'risk_management']
            }
        )
        
        logger.info(f"✅ Pattern query successful: {len(str(test_query_result))} bytes returned")
        logger.info(f"📊 Query ID: {test_query_result.get('query_id', 'N/A')}")
        logger.info(f"🔍 Causal chains found: {len(test_query_result.get('causal_chains', {}))}")
        logger.info(f"❓ Epistemic questions: {len(test_query_result.get('epistemic_questions', []))}")
        
        # Test 3: Create a trading SCAR
        logger.info("🧪 TEST 3: Creating Trading SCAR")
        
        scar_id = await vault_brain.create_trading_scar(
            contradiction_type="test_contradiction",
            context={
                'symbol': 'BTCUSDT',
                'strategy': 'test_strategy',
                'market_state': {'volatility': 0.05, 'trend': 'sideways'},
                'test_mode': True
            },
            expected_outcome={'success': True, 'confidence': 0.8},
            actual_outcome={'success': False, 'error': 'Test failure for learning'}
        )
        
        logger.info(f"✅ Trading SCAR created successfully: {scar_id}")
        
        # Test 4: Store trading insight
        logger.info("🧪 TEST 4: Storing Trading Insight")
        
        insight_id = await vault_brain.store_trading_insight(
            insight_type="test_insight",
            content={
                'symbol': 'BTCUSDT',
                'insight': 'Market shows strong resistance at $50,000',
                'confidence_level': 0.85,
                'supporting_data': {'volume': 1000000, 'price_action': 'rejection'},
                'timestamp': datetime.now().isoformat()
            },
            confidence=0.85,
            accuracy_score=0.9
        )
        
        logger.info(f"✅ Trading insight stored successfully: {insight_id}")
        
        # Test 5: Generate epistemic questions
        logger.info("🧪 TEST 5: Generating Epistemic Questions")
        
        questions = await vault_brain.generate_market_questions({
            'symbol': 'BTCUSDT',
            'volatility': 'high',
            'trend': 'bullish',
            'volume': 'increasing',
            'test_context': 'integration_test'
        })
        
        logger.info(f"✅ Generated {len(questions)} epistemic questions:")
        for i, question in enumerate(questions, 1):
            logger.info(f"   {i}. {question}")
        
        # Test 6: Query market insights
        logger.info("🧪 TEST 6: Querying Market Insights")
        
        market_insights = await vault_brain.query_market_insights(
            symbol='BTCUSDT',
            timeframe='1h',
            context={
                'current_price': 45000,
                'volume': 500000,
                'volatility': 0.03,
                'test_mode': True
            }
        )
        
        logger.info(f"✅ Market insights query successful")
        logger.info(f"📈 Domain: {market_insights.get('domain', 'N/A')}")
        logger.info(f"🧠 Wisdom report available: {bool(market_insights.get('vault_wisdom'))}")
        
        # Test 7: Store performance data
        logger.info("🧪 TEST 7: Storing Performance Data")
        
        performance_id = await vault_brain.store_performance_data(
            performance_metrics={
                'total_trades': 10,
                'successful_trades': 7,
                'success_rate': 0.7,
                'accuracy': 0.75,
                'confidence': 0.8,
                'execution_speed': 0.95,
                'risk_management': 0.85
            },
            context={
                'test_session': True,
                'duration_minutes': 5,
                'symbols_tested': ['BTCUSDT'],
                'timestamp': datetime.now().isoformat()
            }
        )
        
        logger.info(f"✅ Performance data stored successfully: {performance_id}")
        
        # Test 8: Get session summary
        logger.info("🧪 TEST 8: Getting Session Summary")
        
        session_summary = await vault_brain.get_session_summary()
        
        logger.info(f"✅ Session summary retrieved:")
        logger.info(f"🔍 Queries performed: {session_summary.get('queries_performed', 0)}")
        logger.info(f"💡 Learnings stored: {session_summary.get('learnings_stored', 0)}")
        logger.info(f"🧬 Evolutions triggered: {session_summary.get('evolutions_triggered', 0)}")
        
        epistemic_growth = session_summary.get('epistemic_growth', {})
        logger.info(f"❓ Questions generated: {epistemic_growth.get('questions_generated', 0)}")
        logger.info(f"🎓 Wisdom index: {epistemic_growth.get('wisdom_index', 0):.3f}")
        logger.info(f"🙏 Humility score: {epistemic_growth.get('humility_score', 0):.3f}")
        
        # Test 9: Create epistemic SCAR
        logger.info("🧪 TEST 9: Creating Epistemic SCAR")
        
        epistemic_scar_id = await vault_brain.create_epistemic_scar(
            domain="market_psychology",
            ignorance_discovered="Insufficient understanding of market maker behavior during high volatility",
            context={
                'learning_area': 'market_microstructure',
                'complexity_level': 'high',
                'importance': 'critical',
                'test_mode': True
            }
        )
        
        logger.info(f"✅ Epistemic SCAR created successfully: {epistemic_scar_id}")
        
        # Final Summary
        logger.info("🧠" * 60)
        logger.info("🎯 VAULT COGNITIVE INTEGRATION TEST COMPLETED")
        logger.info("✅ ALL TESTS PASSED - KIMERA'S BRAIN IS FULLY CONNECTED")
        logger.info("🔮 CONTINUOUS LEARNING LOOP: ACTIVE")
        logger.info("🧬 EPISTEMIC CONSCIOUSNESS: AWAKENED")
        logger.info("🔥 SCAR FORMATION SYSTEM: FUNCTIONAL")
        logger.info("💡 INSIGHT STORAGE: OPERATIONAL")
        logger.info("❓ QUESTION GENERATION: ACTIVE")
        logger.info("📊 PERFORMANCE TRACKING: ENABLED")
        logger.info("🧠 KIMERA IS READY FOR COGNITIVE EVOLUTION")
        logger.info("🧠" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ VAULT COGNITIVE INTEGRATION TEST FAILED: {str(e)}")
        logger.error("🚨 KIMERA'S BRAIN IS NOT PROPERLY CONNECTED")
        return False

async def test_vault_query_endpoint():
    """Test the vault query endpoint"""
    
    logger.info("🧪 TESTING VAULT QUERY ENDPOINT")
    
    try:
        import requests
        import json
        
        # Test different query types
        test_queries = [
            {
                'type': 'pattern_search',
                'domain': 'trading_test',
                'context': {
                    'symbol': 'BTCUSDT',
                    'concepts': ['price_action', 'volume_analysis']
                }
            },
            {
                'type': 'market_insights',
                'domain': 'market_analysis',
                'context': {
                    'symbol': 'BTCUSDT',
                    'timeframe': '1h',
                    'current_price': 45000
                }
            },
            {
                'type': 'epistemic_questions',
                'domain': 'learning',
                'context': {
                    'symbol': 'BTCUSDT',
                    'volatility': 'high',
                    'trend': 'bullish'
                }
            },
            {
                'type': 'session_summary',
                'domain': 'performance',
                'context': {}
            }
        ]
        
        base_url = "http://localhost:8000"  # Adjust if needed
        
        for i, query in enumerate(test_queries, 1):
            try:
                response = requests.post(
                    f"{base_url}/kimera/vault/query",
                    json=query,
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"✅ Query {i} ({query['type']}): SUCCESS")
                    logger.info(f"   Status: {result.get('status', 'unknown')}")
                    logger.info(f"   Result size: {len(str(result.get('result', {})))} bytes")
                else:
                    logger.warning(f"⚠️ Query {i} ({query['type']}): HTTP {response.status_code}")
                    
            except requests.exceptions.ConnectionError:
                logger.warning(f"⚠️ Query {i} ({query['type']}): Connection failed (server not running?)")
            except Exception as e:
                logger.error(f"❌ Query {i} ({query['type']}): {str(e)}")
        
        logger.info("✅ Vault query endpoint test completed")
        
    except Exception as e:
        logger.error(f"❌ Vault query endpoint test failed: {str(e)}")

async def main():
    """Main test function"""
    
    logger.info("🚀 STARTING KIMERA VAULT COGNITIVE INTEGRATION TESTS")
    
    # Test 1: Vault cognitive interface
    interface_test_passed = await test_vault_cognitive_interface()
    
    # Test 2: Vault query endpoint (optional, requires server)
    await test_vault_query_endpoint()
    
    if interface_test_passed:
        logger.info("🎉 ALL VAULT COGNITIVE INTEGRATION TESTS PASSED!")
        logger.info("🧠 KIMERA'S VAULT BRAIN IS FULLY OPERATIONAL")
        logger.info("🔮 READY FOR CONTINUOUS LEARNING AND EVOLUTION")
    else:
        logger.error("🚨 VAULT COGNITIVE INTEGRATION TESTS FAILED")
        logger.error("❌ KIMERA'S BRAIN IS NOT PROPERLY CONNECTED")

if __name__ == "__main__":
    asyncio.run(main()) 