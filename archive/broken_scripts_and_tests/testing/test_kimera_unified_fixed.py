"""
KIMERA Unified System Test - Fixed Version
"""

import asyncio
import aiohttp
import json
import time
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_kimera_system():
    """Test KIMERA system with actual endpoints"""
    base_url = "http://localhost:8000"
    
    async with aiohttp.ClientSession() as session:
        results = []
        
        # Test 1: System Health
        logger.info("🔍 Testing system health...")
        try:
            async with session.get(f"{base_url}/system/health") as response:
                if response.status == 200:
                    health_data = await response.json()
                    results.append({
                        'test': 'system_health',
                        'status': 'PASS',
                        'data': health_data
                    })
                    logger.info("✅ system_health: PASS")
                else:
                    results.append({
                        'test': 'system_health',
                        'status': 'FAIL',
                        'error': f"HTTP {response.status}"
                    })
                    logger.info("❌ system_health: FAIL")
        except Exception as e:
            results.append({
                'test': 'system_health',
                'status': 'FAIL',
                'error': str(e)
            })
            logger.info("❌ system_health: FAIL")
        
        # Test 2: Detailed Health
        logger.info("📊 Testing detailed health...")
        try:
            async with session.get(f"{base_url}/system/health/detailed") as response:
                if response.status == 200:
                    health_data = await response.json()
                    results.append({
                        'test': 'detailed_health',
                        'status': 'PASS',
                        'data': health_data
                    })
                    logger.info("✅ detailed_health: PASS")
                else:
                    results.append({
                        'test': 'detailed_health',
                        'status': 'FAIL',
                        'error': f"HTTP {response.status}"
                    })
                    logger.info("❌ detailed_health: FAIL")
        except Exception as e:
            results.append({
                'test': 'detailed_health',
                'status': 'FAIL',
                'error': str(e)
            })
            logger.info("❌ detailed_health: FAIL")
        
        # Test 3: Cognitive Fields
        logger.info("🧠 Testing cognitive fields...")
        test_geoid = {
            "geoid_id": "test_unified_1",
            "semantic_state": {"concept": "test", "intensity": 0.5},
            "symbolic_state": [["test"]],
            "metadata": {"source": "test"}
        }
        
        try:
            async with session.post(
                f"{base_url}/cognitive-fields/geoid/add",
                json=test_geoid
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    results.append({
                        'test': 'cognitive_fields',
                        'status': 'PASS',
                        'data': result
                    })
                    logger.info("✅ cognitive_fields: PASS")
                else:
                    results.append({
                        'test': 'cognitive_fields',
                        'status': 'FAIL',
                        'error': f"HTTP {response.status}"
                    })
                    logger.info("❌ cognitive_fields: FAIL")
        except Exception as e:
            results.append({
                'test': 'cognitive_fields',
                'status': 'FAIL',
                'error': str(e)
            })
            logger.info("❌ cognitive_fields: FAIL")
        
        # Calculate results
        passed = sum(1 for r in results if r['status'] == 'PASS')
        total = len(results)
        success_rate = passed / total
        
        final_result = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': total,
            'passed_tests': passed,
            'success_rate': success_rate,
            'status': 'SUCCESS' if success_rate >= 0.7 else 'NEEDS_IMPROVEMENT',
            'results': results
        }
        
        # Save results
        with open('kimera_test_results.json', 'w') as f:
            json.dump(final_result, f, indent=2)
        
        logger.info(f"\n🎯 TEST COMPLETE")
        logger.info(f"   Success Rate: {success_rate:.1%} ({passed}/{total})
        logger.info(f"   Status: {final_result['status']}")
        
        return final_result

if __name__ == "__main__":
    asyncio.run(test_kimera_system()) 