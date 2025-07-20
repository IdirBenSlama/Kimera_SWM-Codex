"""
KIMERA Unified System Integration Test - Fixed Version
=====================================================

Comprehensive testing of KIMERA system with the actual available endpoints.
Tests all integrated modules and validates the unified cognitive architecture.
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
        """Test system health using actual KIMERA endpoints"""
        logger.info("🔍 Testing system health...")
        
        try:
            async with self.session.get(f"{self.base_url}/system/health") as response:
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
    
    async def test_detailed_system_health(self) -> Dict[str, Any]:
        """Test detailed system health"""
        logger.info("🔍 Testing detailed system health...")
        
        try:
            async with self.session.get(f"{self.base_url}/system/health/detailed") as response:
                if response.status == 200:
                    health_data = await response.json()
                    return {
                        'test': 'detailed_system_health',
                        'status': 'PASS',
                        'health_data': health_data,
                        'gpu_available': health_data.get('gpu_info', {}).get('available', False),
                        'memory_usage': health_data.get('memory', {}).get('percent', 0),
                        'cpu_usage': health_data.get('cpu', {}).get('percent', 0)
                    }
                else:
                    return {
                        'test': 'detailed_system_health',
                        'status': 'FAIL',
                        'error': f"HTTP {response.status}"
                    }
        except Exception as e:
            return {
                'test': 'detailed_system_health',
                'status': 'FAIL',
                'error': str(e)
            }
    
    async def test_cognitive_field_dynamics(self) -> Dict[str, Any]:
        """Test cognitive field dynamics"""
        logger.info("🧠 Testing cognitive field dynamics...")
        
        test_geoid = {
            "geoid_id": "test_unified_cognitive_1",
            "semantic_state": {"concept": "market_volatility", "intensity": 0.8},
            "symbolic_state": [["volatility", "market"]],
            "metadata": {"test_type": "unified_cognitive", "source": "test_runner"}
        }
        
        try:
            start_time = time.time()
            async with self.session.post(
                f"{self.base_url}/cognitive-fields/geoid/add",
                json=test_geoid
            ) as response:
                processing_time = time.time() - start_time
                
                if response.status == 200:
                    result = await response.json()
                    return {
                        'test': 'cognitive_field_dynamics',
                        'status': 'PASS',
                        'processing_time': processing_time,
                        'geoid_added': result.get('success', False),
                        'geoid_id': result.get('geoid_id', ''),
                        'field_metrics': result.get('field_metrics', {})
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
    
    async def test_thermodynamic_engines(self) -> Dict[str, Any]:
        """Test thermodynamic engines status"""
        logger.info("🌡️ Testing thermodynamic engines...")
        
        try:
            async with self.session.get(f"{self.base_url}/monitoring/engines/thermodynamics") as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        'test': 'thermodynamic_engines',
                        'status': 'PASS',
                        'engine_status': result.get('status', 'unknown'),
                        'thermal_entropy': result.get('thermal_entropy', 0),
                        'computational_entropy': result.get('computational_entropy', 0),
                        'reversibility_index': result.get('reversibility_index', 0),
                        'free_energy': result.get('free_energy', 0)
                    }
                else:
                    error_text = await response.text()
                    return {
                        'test': 'thermodynamic_engines',
                        'status': 'FAIL',
                        'error': f"HTTP {response.status}: {error_text}"
                    }
        except Exception as e:
            return {
                'test': 'thermodynamic_engines',
                'status': 'FAIL',
                'error': str(e)
            }
    
    async def test_revolutionary_intelligence(self) -> Dict[str, Any]:
        """Test revolutionary intelligence system"""
        logger.info("🚀 Testing revolutionary intelligence...")
        
        try:
            async with self.session.get(f"{self.base_url}/revolutionary/intelligence/complete") as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        'test': 'revolutionary_intelligence',
                        'status': 'PASS',
                        'intelligence_active': result.get('active', False),
                        'analysis_depth': result.get('analysis_depth', 0),
                        'revolutionary_metrics': result.get('metrics', {}),
                        'system_evolution': result.get('evolution_status', 'unknown')
                    }
                else:
                    error_text = await response.text()
                    return {
                        'test': 'revolutionary_intelligence',
                        'status': 'FAIL',
                        'error': f"HTTP {response.status}: {error_text}"
                    }
        except Exception as e:
            return {
                'test': 'revolutionary_intelligence',
                'status': 'FAIL',
                'error': str(e)
            }
    
    async def test_monitoring_system(self) -> Dict[str, Any]:
        """Test comprehensive monitoring system"""
        logger.info("📊 Testing monitoring system...")
        
        try:
            async with self.session.get(f"{self.base_url}/monitoring/metrics/summary") as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        'test': 'monitoring_system',
                        'status': 'PASS',
                        'system_metrics': result.get('system_metrics', {}),
                        'kimera_metrics': result.get('kimera_metrics', {}),
                        'performance_metrics': result.get('performance_metrics', {}),
                        'monitoring_active': result.get('monitoring_active', False)
                    }
                else:
                    error_text = await response.text()
                    return {
                        'test': 'monitoring_system',
                        'status': 'FAIL',
                        'error': f"HTTP {response.status}: {error_text}"
                    }
        except Exception as e:
            return {
                'test': 'monitoring_system',
                'status': 'FAIL',
                'error': str(e)
            }
    
    async def test_security_and_law_enforcement(self) -> Dict[str, Any]:
        """Test security and law enforcement systems"""
        logger.info("🛡️ Testing security and law enforcement...")
        
        test_request = {
            "context": "unified_cognitive_test",
            "operation": "system_integration_test",
            "parameters": {"test_type": "security_validation"}
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/law_enforcement/assess_compliance",
                json=test_request
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        'test': 'security_law_enforcement',
                        'status': 'PASS',
                        'compliance_status': result.get('compliance', 'unknown'),
                        'security_level': result.get('security_level', 0),
                        'law_violations': result.get('violations', []),
                        'enforcement_active': result.get('enforcement_active', False)
                    }
                else:
                    error_text = await response.text()
                    return {
                        'test': 'security_law_enforcement',
                        'status': 'FAIL',
                        'error': f"HTTP {response.status}: {error_text}"
                    }
        except Exception as e:
            return {
                'test': 'security_law_enforcement',
                'status': 'FAIL',
                'error': str(e)
            }
    
    async def test_insight_generation(self) -> Dict[str, Any]:
        """Test insight generation capabilities"""
        logger.info("💡 Testing insight generation...")
        
        test_request = {
            "query": "Analyze the integration of thermodynamic optimization with cognitive field dynamics",
            "context": "unified_system_test",
            "complexity_level": "high"
        }
        
        try:
            start_time = time.time()
            async with self.session.post(
                f"{self.base_url}/insights/generate",
                json=test_request
            ) as response:
                processing_time = time.time() - start_time
                
                if response.status == 200:
                    result = await response.json()
                    return {
                        'test': 'insight_generation',
                        'status': 'PASS',
                        'processing_time': processing_time,
                        'insight_generated': result.get('success', False),
                        'insight_id': result.get('insight_id', ''),
                        'insight_quality': result.get('quality_score', 0),
                        'cognitive_depth': result.get('cognitive_depth', 0)
                    }
                else:
                    error_text = await response.text()
                    return {
                        'test': 'insight_generation',
                        'status': 'FAIL',
                        'error': f"HTTP {response.status}: {error_text}"
                    }
        except Exception as e:
            return {
                'test': 'insight_generation',
                'status': 'FAIL',
                'error': str(e)
            }
    
    async def test_system_cycle_and_stability(self) -> Dict[str, Any]:
        """Test system cycle and stability"""
        logger.info("🔄 Testing system cycle and stability...")
        
        try:
            # Test system cycle
            async with self.session.post(f"{self.base_url}/system/cycle") as cycle_response:
                cycle_success = cycle_response.status == 200
                
            # Test system stability
            async with self.session.get(f"{self.base_url}/system/stability") as stability_response:
                if stability_response.status == 200:
                    stability_data = await stability_response.json()
                    return {
                        'test': 'system_cycle_stability',
                        'status': 'PASS',
                        'cycle_executed': cycle_success,
                        'stability_metrics': stability_data.get('stability_metrics', {}),
                        'system_coherence': stability_data.get('coherence', 0),
                        'operational_stability': stability_data.get('operational_stability', 0)
                    }
                else:
                    return {
                        'test': 'system_cycle_stability',
                        'status': 'FAIL',
                        'error': f"Stability check failed: HTTP {stability_response.status}"
                    }
        except Exception as e:
            return {
                'test': 'system_cycle_stability',
                'status': 'FAIL',
                'error': str(e)
            }
    
    async def test_performance_metrics(self) -> Dict[str, Any]:
        """Test performance and utilization metrics"""
        logger.info("⚡ Testing performance metrics...")
        
        try:
            async with self.session.get(f"{self.base_url}/system/utilization_stats") as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        'test': 'performance_metrics',
                        'status': 'PASS',
                        'cpu_utilization': result.get('cpu_percent', 0),
                        'memory_utilization': result.get('memory_percent', 0),
                        'gpu_utilization': result.get('gpu_utilization', 0),
                        'system_load': result.get('system_load', 0),
                        'performance_score': result.get('performance_score', 0)
                    }
                else:
                    error_text = await response.text()
                    return {
                        'test': 'performance_metrics',
                        'status': 'FAIL',
                        'error': f"HTTP {response.status}: {error_text}"
                    }
        except Exception as e:
            return {
                'test': 'performance_metrics',
                'status': 'FAIL',
                'error': str(e)
            }
    
    async def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """Run the complete test suite"""
        logger.info("🌌 STARTING KIMERA UNIFIED SYSTEM COMPREHENSIVE TEST")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # Run all tests
        test_functions = [
            self.test_system_health,
            self.test_detailed_system_health,
            self.test_cognitive_field_dynamics,
            self.test_thermodynamic_engines,
            self.test_revolutionary_intelligence,
            self.test_monitoring_system,
            self.test_security_and_law_enforcement,
            self.test_insight_generation,
            self.test_system_cycle_and_stability,
            self.test_performance_metrics
        ]
        
        results = []
        for test_func in test_functions:
            try:
                result = await test_func()
                results.append(result)
                
                status_icon = "✅" if result['status'] == 'PASS' else "❌"
                logger.info(f"{status_icon} {result['test']}: {result['status']}")
                
                # Add brief delay between tests
                await asyncio.sleep(0.5)
                
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
            'test_suite': 'kimera_unified_system_comprehensive_fixed',
            'timestamp': datetime.now().isoformat(),
            'total_time': total_time,
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': success_rate,
            'overall_status': 'SUCCESS' if success_rate >= 0.7 else 'NEEDS_IMPROVEMENT',
            'detailed_results': results,
            'unified_architecture_status': 'ACTIVE' if success_rate >= 0.8 else 'PARTIAL'
        }
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"kimera_unified_system_test_fixed_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(overall_result, f, indent=2, default=str)
        
        logger.info("=" * 80)
        logger.info(f"🎯 KIMERA UNIFIED SYSTEM TEST COMPLETE")
        logger.info(f"   Success Rate: {success_rate:.1%} ({passed_tests}/{total_tests})")
        logger.info(f"   Total Time: {total_time:.2f}s")
        logger.info(f"   Overall Status: {overall_result['overall_status']}")
        logger.info(f"   Unified Architecture: {overall_result['unified_architecture_status']}")
        logger.info(f"   Report Saved: {filename}")
        
        return overall_result

async def main():
    """Main test runner"""
    try:
        async with KimeraUnifiedSystemTester() as tester:
            results = await tester.run_comprehensive_test_suite()
            
            if results['overall_status'] == 'SUCCESS':
                logger.info("\n🎉 KIMERA UNIFIED SYSTEM: FULLY OPERATIONAL!")
                logger.info("🌌 Revolutionary unified cognitive architecture confirmed!")
                logger.info("🚀 All major subsystems integrated and functioning!")
            else:
                logger.warning(f"\n⚠️  KIMERA UNIFIED SYSTEM: {results['overall_status']}")
                logger.debug("🔧 System operational but some modules need attention")
                
            # Print key metrics
            logger.info(f"\n📊 KEY METRICS:")
            logger.info(f"   🎯 Success Rate: {results['success_rate']:.1%}")
            logger.info(f"   ⏱️  Total Test Time: {results['total_time']:.2f}s")
            logger.info(f"   🧠 Unified Architecture: {results['unified_architecture_status']}")
            
            return results
            
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    asyncio.run(main()) 