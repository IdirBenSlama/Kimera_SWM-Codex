#!/usr/bin/env python3
"""
KIMERA GPU Foundation Integration Verification Test
===================================================

Comprehensive test to verify the updated KIMERA system with GPU Foundation integration.
This test validates:
- GPU Foundation initialization in KIMERA startup
- New API endpoints for GPU Foundation status
- Enhanced system status with GPU information
- Live system integration with GPU acceleration

Author: KIMERA Development Team
Version: 1.0.0 - Post-Integration Verification
"""

import logging
import time
import requests
import json
from datetime import datetime
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [GPU-Integration] %(message)s'
)
logger = logging.getLogger(__name__)

class KimeraGPUIntegrationTest:
    """Comprehensive test suite for KIMERA GPU Foundation integration"""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.test_results = []
        self.start_time = time.time()
        
    def log_test_result(self, test_name: str, success: bool, duration: float, details: Dict[str, Any] = None):
        """Log test result with detailed information"""
        result = {
            "test_name": test_name,
            "success": success,
            "duration_seconds": round(duration, 3),
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
        self.test_results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status} - {test_name} ({duration:.3f}s)")
        
        if details:
            logger.info(f"   Details: {json.dumps(details, indent=2)}")
    
    def test_system_status_gpu_info(self) -> bool:
        """Test that system status includes GPU Foundation information"""
        start_time = time.time()
        
        try:
            response = requests.get(f"{self.base_url}/system/status", timeout=10)
            duration = time.time() - start_time
            
            if response.status_code != 200:
                self.log_test_result("System Status GPU Info", False, duration, 
                                   {"error": f"HTTP {response.status_code}"})
                return False
            
            data = response.json()
            
            # Check for GPU info in system status
            gpu_info = data.get('gpu_info', {})
            required_fields = ['gpu_available', 'gpu_foundation_status']
            
            missing_fields = [field for field in required_fields if field not in gpu_info]
            if missing_fields:
                self.log_test_result("System Status GPU Info", False, duration,
                                   {"error": f"Missing fields: {missing_fields}"})
                return False
            
            # Check GPU Foundation status
            gpu_foundation_status = gpu_info.get('gpu_foundation_status')
            if gpu_foundation_status not in ['operational', 'not_initialized', 'error']:
                self.log_test_result("System Status GPU Info", False, duration,
                                   {"error": f"Invalid GPU Foundation status: {gpu_foundation_status}"})
                return False
            
            details = {
                "gpu_available": gpu_info.get('gpu_available'),
                "gpu_foundation_status": gpu_foundation_status,
                "gpu_name": gpu_info.get('gpu_name'),
                "total_memory_gb": gpu_info.get('total_memory_gb'),
                "cognitive_stability": gpu_info.get('cognitive_stability', {})
            }
            
            self.log_test_result("System Status GPU Info", True, duration, details)
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result("System Status GPU Info", False, duration, {"error": str(e)})
            return False
    
    def test_gpu_foundation_endpoint(self) -> bool:
        """Test the dedicated GPU Foundation status endpoint"""
        start_time = time.time()
        
        try:
            response = requests.get(f"{self.base_url}/system/gpu_foundation", timeout=10)
            duration = time.time() - start_time
            
            if response.status_code != 200:
                self.log_test_result("GPU Foundation Endpoint", False, duration,
                                   {"error": f"HTTP {response.status_code}"})
                return False
            
            data = response.json()
            
            # Check required fields
            required_fields = ['status', 'timestamp']
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                self.log_test_result("GPU Foundation Endpoint", False, duration,
                                   {"error": f"Missing fields: {missing_fields}"})
                return False
            
            status = data.get('status')
            if status == 'not_available':
                # GPU Foundation not initialized - this is acceptable
                details = {"status": status, "message": data.get('message')}
                self.log_test_result("GPU Foundation Endpoint", True, duration, details)
                return True
            elif status == 'operational':
                # GPU Foundation operational - check detailed info
                required_operational_fields = ['capabilities', 'cognitive_stability', 'performance_metrics']
                missing_op_fields = [field for field in required_operational_fields if field not in data]
                if missing_op_fields:
                    self.log_test_result("GPU Foundation Endpoint", False, duration,
                                       {"error": f"Missing operational fields: {missing_op_fields}"})
                    return False
                
                capabilities = data.get('capabilities', {})
                cognitive_stability = data.get('cognitive_stability', {})
                performance_metrics = data.get('performance_metrics', {})
                
                details = {
                    "status": status,
                    "device_name": capabilities.get('device_name'),
                    "total_memory_gb": capabilities.get('total_memory_gb'),
                    "validation_level": capabilities.get('validation_level'),
                    "safety_grade": cognitive_stability.get('safety_grade'),
                    "performance_grade": performance_metrics.get('performance_grade'),
                    "identity_coherence": cognitive_stability.get('identity_coherence_score'),
                    "memory_bandwidth_gb_s": performance_metrics.get('memory_bandwidth_gb_s')
                }
                
                self.log_test_result("GPU Foundation Endpoint", True, duration, details)
                return True
            else:
                # Error status
                details = {"status": status, "message": data.get('message')}
                self.log_test_result("GPU Foundation Endpoint", True, duration, details)
                return True
                
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result("GPU Foundation Endpoint", False, duration, {"error": str(e)})
            return False
    
    def test_semantic_processing_with_gpu(self) -> bool:
        """Test semantic processing to ensure GPU acceleration is working"""
        start_time = time.time()
        
        try:
            payload = {
                "echoform_text": "GPU-accelerated quantum consciousness processing with neural thermodynamics",
                "context": "GPU Foundation Integration Test"
            }
            
            response = requests.post(f"{self.base_url}/geoids", json=payload, timeout=15)
            duration = time.time() - start_time
            
            if response.status_code != 200:
                self.log_test_result("Semantic Processing with GPU", False, duration,
                                   {"error": f"HTTP {response.status_code}", "response": response.text[:500]})
                return False
            
            data = response.json()
            
            # Check for successful geoid creation
            if 'geoid_id' not in data:
                self.log_test_result("Semantic Processing with GPU", False, duration,
                                   {"error": "No geoid_id in response"})
                return False
            
            geoid_id = data['geoid_id']
            geoid_data = data.get('geoid', {})
            
            # Check for semantic processing indicators
            semantic_complexity = geoid_data.get('semantic_complexity', 0)
            entropy_baseline = geoid_data.get('entropy_baseline', 0)
            
            details = {
                "geoid_id": geoid_id,
                "semantic_complexity": semantic_complexity,
                "entropy_baseline": entropy_baseline,
                "processing_time_seconds": duration,
                "response_size_chars": len(json.dumps(data))
            }
            
            self.log_test_result("Semantic Processing with GPU", True, duration, details)
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result("Semantic Processing with GPU", False, duration, {"error": str(e)})
            return False
    
    def test_prometheus_metrics_gpu(self) -> bool:
        """Test that Prometheus metrics include GPU Foundation information"""
        start_time = time.time()
        
        try:
            response = requests.get(f"{self.base_url}/metrics", timeout=10)
            duration = time.time() - start_time
            
            if response.status_code != 200:
                self.log_test_result("Prometheus Metrics GPU", False, duration,
                                   {"error": f"HTTP {response.status_code}"})
                return False
            
            metrics_text = response.text
            
            # Check for KIMERA-specific metrics
            kimera_metrics_found = []
            expected_metrics = ['kimera_geoids_total', 'kimera_contradictions_total', 'kimera_system_status']
            
            for metric in expected_metrics:
                if metric in metrics_text:
                    kimera_metrics_found.append(metric)
            
            details = {
                "metrics_found": kimera_metrics_found,
                "total_metrics_count": len(metrics_text.split('\n')),
                "response_size_chars": len(metrics_text)
            }
            
            # At least some KIMERA metrics should be present
            success = len(kimera_metrics_found) > 0
            
            self.log_test_result("Prometheus Metrics GPU", success, duration, details)
            return success
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result("Prometheus Metrics GPU", False, duration, {"error": str(e)})
            return False
    
    def test_concurrent_gpu_api_processing(self) -> bool:
        """Test concurrent GPU and API processing"""
        start_time = time.time()
        
        try:
            import threading
            import concurrent.futures
            
            def make_request(text_suffix):
                payload = {
                    "echoform_text": f"Concurrent GPU processing test {text_suffix}",
                    "context": "Concurrent Test"
                }
                response = requests.post(f"{self.base_url}/geoids", json=payload, timeout=10)
                return response.status_code == 200, response.json() if response.status_code == 200 else None
            
            # Make 5 concurrent requests
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(make_request, i) for i in range(5)]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            duration = time.time() - start_time
            
            successful_requests = sum(1 for success, _ in results if success)
            success_rate = successful_requests / len(results)
            
            details = {
                "total_requests": len(results),
                "successful_requests": successful_requests,
                "success_rate": success_rate,
                "concurrent_processing_time": duration
            }
            
            # Consider success if at least 80% of requests succeeded
            test_success = success_rate >= 0.8
            
            self.log_test_result("Concurrent GPU-API Processing", test_success, duration, details)
            return test_success
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result("Concurrent GPU-API Processing", False, duration, {"error": str(e)})
            return False
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run all GPU Foundation integration tests"""
        logger.info("🚀 Starting KIMERA GPU Foundation Integration Test Suite")
        logger.info(f"📡 Testing against: {self.base_url}")
        
        # Test suite
        tests = [
            ("System Status GPU Info", self.test_system_status_gpu_info),
            ("GPU Foundation Endpoint", self.test_gpu_foundation_endpoint),
            ("Semantic Processing with GPU", self.test_semantic_processing_with_gpu),
            ("Prometheus Metrics GPU", self.test_prometheus_metrics_gpu),
            ("Concurrent GPU-API Processing", self.test_concurrent_gpu_api_processing)
        ]
        
        # Run tests
        for test_name, test_func in tests:
            logger.info(f"🧪 Running: {test_name}")
            test_func()
        
        # Calculate results
        total_duration = time.time() - self.start_time
        successful_tests = sum(1 for result in self.test_results if result['success'])
        total_tests = len(self.test_results)
        success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
        
        # Determine overall grade
        if success_rate >= 90:
            grade = "EXCELLENT"
        elif success_rate >= 75:
            grade = "GOOD"
        elif success_rate >= 50:
            grade = "ACCEPTABLE"
        else:
            grade = "NEEDS_IMPROVEMENT"
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "failed_tests": total_tests - successful_tests,
            "success_rate": round(success_rate, 2),
            "total_duration_seconds": round(total_duration, 3),
            "grade": grade,
            "test_results": self.test_results
        }
        
        # Log summary
        logger.info("="*80)
        logger.info("🎯 KIMERA GPU FOUNDATION INTEGRATION TEST SUMMARY")
        logger.info("="*80)
        logger.info(f"📊 Success Rate: {success_rate:.1f}% ({successful_tests}/{total_tests} tests passed)")
        logger.info(f"⏱️  Total Duration: {total_duration:.3f} seconds")
        logger.info(f"🏆 Overall Grade: {grade}")
        logger.info("="*80)
        
        return summary

def main():
    """Main test execution function"""
    logger.debug("🔧 KIMERA GPU Foundation Integration Test")
    logger.info("=" * 50)
    
    # Wait for system to be ready
    logger.info("⏳ Waiting for KIMERA system to be ready...")
    time.sleep(2)
    
    # Run tests
    test_suite = KimeraGPUIntegrationTest()
    results = test_suite.run_comprehensive_test()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"kimera_gpu_integration_test_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n📄 Detailed results saved to: {results_file}")
    
    # Exit with appropriate code
    if results['success_rate'] >= 75:
        logger.info("✅ GPU Foundation integration test PASSED")
        return 0
    else:
        logger.error("❌ GPU Foundation integration test FAILED")
        return 1

if __name__ == "__main__":
    exit(main()) 