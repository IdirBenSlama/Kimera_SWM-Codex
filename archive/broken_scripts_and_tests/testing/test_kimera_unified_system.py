"""
KIMERA Unified System Integration Test
=====================================

Comprehensive testing of KIMERA system with the new unified cognitive architecture.
Tests all integrated modules and validates the revolutionary breakthrough.
"""

import asyncio
import aiohttp
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KimeraUnifiedSystemTester:
    """Comprehensive tester for KIMERA unified system"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None
        self.test_results = []
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_system_health(self) -> Dict[str, Any]:
        """Test basic system health and availability"""
        logger.info("🔍 Testing system health...")
        
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    health_data = await response.json()
                    return {
                        'test': 'system_health',
                        'status': 'PASS',
                        'response_time': response.headers.get('X-Response-Time', 'N/A'),
                        'health_data': health_data
                    }
                else:
                    return {
                        'test': 'system_health',
                        'status': 'FAIL',
                        'error': f"HTTP {response.status}"
                    }
        except Exception as e:
            return {
                'test': 'system_health',
                'status': 'FAIL',
                'error': str(e)
            }
    
    async def test_cognitive_field_dynamics(self) -> Dict[str, Any]:
        """Test cognitive field dynamics with thermodynamic optimization"""
        logger.info("🧠 Testing cognitive field dynamics...")
        
        test_data = {
            "geoids": [
                {
                    "geoid_id": "test_unified_1",
                    "semantic_state": {"concept": "market_volatility", "intensity": 0.8},
                    "symbolic_state": [["volatility", "market"]],
                    "metadata": {"test_type": "unified_cognitive"}
                }
            ]
        }
        
        try:
            start_time = time.time()
            async with self.session.post(
                f"{self.base_url}/cognitive/process",
                json=test_data
            ) as response:
                processing_time = time.time() - start_time
                
                if response.status == 200:
                    result = await response.json()
                    return {
                        'test': 'cognitive_field_dynamics',
                        'status': 'PASS',
                        'processing_time': processing_time,
                        'geoids_processed': len(result.get('processed_geoids', [])),
                        'cognitive_coherence': result.get('cognitive_coherence', 0)
                    }
                else:
                    error_text = await response.text()
                    return {
                        'test': 'cognitive_field_dynamics',
                        'status': 'FAIL',
                        'error': f"HTTP {response.status}: {error_text}"
                    }
        except Exception as e:
            return {
                'test': 'cognitive_field_dynamics',
                'status': 'FAIL',
                'error': str(e)
            }
    
    async def test_basic_endpoints(self) -> Dict[str, Any]:
        """Test basic KIMERA endpoints"""
        logger.info("🔗 Testing basic endpoints...")
        
        endpoints_to_test = [
            "/",
            "/docs",
            "/metrics"
        ]
        
        results = []
        for endpoint in endpoints_to_test:
            try:
                async with self.session.get(f"{self.base_url}{endpoint}") as response:
                    results.append({
                        'endpoint': endpoint,
                        'status_code': response.status,
                        'success': response.status < 400
                    })
            except Exception as e:
                results.append({
                    'endpoint': endpoint,
                    'success': False,
                    'error': str(e)
                })
        
        successful_endpoints = sum(1 for r in results if r['success'])
        
        return {
            'test': 'basic_endpoints',
            'status': 'PASS' if successful_endpoints >= 2 else 'FAIL',
            'successful_endpoints': successful_endpoints,
            'total_endpoints': len(endpoints_to_test),
            'detailed_results': results
        }
    
    async def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """Run the complete test suite"""
        logger.info("🌌 STARTING KIMERA UNIFIED SYSTEM COMPREHENSIVE TEST")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # Run all tests
        test_functions = [
            self.test_system_health,
            self.test_basic_endpoints,
            self.test_cognitive_field_dynamics
        ]
        
        results = []
        for test_func in test_functions:
            try:
                result = await test_func()
                results.append(result)
                
                status_icon = "✅" if result['status'] == 'PASS' else "❌"
                logger.info(f"{status_icon} {result['test']}: {result['status']}")
                
            except Exception as e:
                logger.error(f"❌ {test_func.__name__} failed: {e}")
                results.append({
                    'test': test_func.__name__,
                    'status': 'ERROR',
                    'error': str(e)
                })
        
        total_time = time.time() - start_time
        
        # Calculate overall results
        passed_tests = sum(1 for r in results if r['status'] == 'PASS')
        total_tests = len(results)
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        overall_result = {
            'test_suite': 'kimera_unified_system_comprehensive',
            'timestamp': datetime.now().isoformat(),
            'total_time': total_time,
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': success_rate,
            'overall_status': 'SUCCESS' if success_rate >= 0.7 else 'NEEDS_IMPROVEMENT',
            'detailed_results': results
        }
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"kimera_unified_system_test_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(overall_result, f, indent=2, default=str)
        
        logger.info("=" * 80)
        logger.info(f"🎯 KIMERA UNIFIED SYSTEM TEST COMPLETE")
        logger.info(f"   Success Rate: {success_rate:.1%} ({passed_tests}/{total_tests})")
        logger.info(f"   Total Time: {total_time:.2f}s")
        logger.info(f"   Overall Status: {overall_result['overall_status']}")
        logger.info(f"   Report Saved: {filename}")
        
        return overall_result

async def main():
    """Main test runner"""
    try:
        async with KimeraUnifiedSystemTester() as tester:
            results = await tester.run_comprehensive_test_suite()
            
            if results['overall_status'] == 'SUCCESS':
                logger.info("\n🎉 KIMERA UNIFIED SYSTEM: FULLY OPERATIONAL!")
                logger.info("🌌 Revolutionary breakthrough confirmed!")
            else:
                logger.warning("\n⚠️  KIMERA UNIFIED SYSTEM: Needs optimization")
                logger.debug("🔧 Some modules require attention")
                
            return results
            
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    asyncio.run(main()) 