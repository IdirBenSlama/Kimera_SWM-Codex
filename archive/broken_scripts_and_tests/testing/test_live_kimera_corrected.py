#!/usr/bin/env python3
"""
CORRECTED Live KIMERA System Test - Real World Performance Validation
====================================================================

This test runs against the live KIMERA system using ACTUAL API endpoints
to validate real performance, cognitive processing, and system capabilities.

Tests based on the actual running system endpoints discovered.
"""

import requests
import json
import time
import sys
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor

# Configure logging without emoji issues
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/live_kimera_corrected_test.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class RealTestResult:
    """Results from real system testing."""
    test_name: str
    success: bool
    response_time: float
    data_size: int
    cognitive_quality: float
    endpoint: str
    status_code: int
    error_message: Optional[str] = None

class RealKimeraSystemTester:
    """
    Real tester for the live KIMERA system using actual endpoints.
    
    Tests actual system performance and cognitive capabilities
    against the running API using discovered endpoints.
    """
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        """Initialize the real system tester."""
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results: List[RealTestResult] = []
        
        logger.info(f"Connecting to live KIMERA system at {base_url}")
        
    def test_system_health(self) -> bool:
        """Test system health using actual endpoints."""
        logger.info("Testing system health...")
        
        health_endpoints = [
            "/system/health",
            "/system/health/detailed",
            "/system/status",
            "/enhanced/health",
            "/revolutionary/health"
        ]
        
        success_count = 0
        
        for endpoint in health_endpoints:
            logger.info(f"  Testing health endpoint: {endpoint}")
            
            try:
                start_time = time.perf_counter()
                response = self.session.get(f"{self.base_url}{endpoint}", timeout=10)
                response_time = time.perf_counter() - start_time
                
                success = response.status_code == 200
                if success:
                    success_count += 1
                    logger.info(f"    SUCCESS: {response_time*1000:.2f}ms")
                else:
                    logger.warning(f"    FAILED: Status {response.status_code}")
                
                self.test_results.append(RealTestResult(
                    test_name=f"Health {endpoint}",
                    success=success,
                    response_time=response_time,
                    data_size=len(response.content) if success else 0,
                    cognitive_quality=1.0 if success else 0.0,
                    endpoint=endpoint,
                    status_code=response.status_code
                ))
                
            except Exception as e:
                logger.error(f"    ERROR: {e}")
                self.test_results.append(RealTestResult(
                    test_name=f"Health {endpoint}",
                    success=False,
                    response_time=0.0,
                    data_size=0,
                    cognitive_quality=0.0,
                    endpoint=endpoint,
                    status_code=0,
                    error_message=str(e)
                ))
        
        success_rate = success_count / len(health_endpoints)
        logger.info(f"Health tests: {success_rate:.2%} success ({success_count}/{len(health_endpoints)})")
        
        return success_rate >= 0.8
    
    def test_cognitive_processing(self) -> bool:
        """Test cognitive processing using actual geoid endpoints."""
        logger.info("Testing cognitive processing...")
        
        # Test actual geoid creation
        test_payload = {
            "text": "Test cognitive processing with quantum mechanical analysis of artificial intelligence systems",
            "metadata": {"test": "cognitive_processing"}
        }
        
        try:
            start_time = time.perf_counter()
            response = self.session.post(
                f"{self.base_url}/geoids",
                json=test_payload,
                timeout=30
            )
            response_time = time.perf_counter() - start_time
            
            success = response.status_code in [200, 201]
            
            if success:
                logger.info(f"  Cognitive processing SUCCESS: {response_time*1000:.2f}ms")
                result_data = response.json()
                cognitive_quality = self._analyze_geoid_quality(result_data)
            else:
                logger.error(f"  Cognitive processing FAILED: Status {response.status_code}")
                logger.error(f"  Response: {response.text}")
                cognitive_quality = 0.0
            
            self.test_results.append(RealTestResult(
                test_name="Cognitive Geoid Creation",
                success=success,
                response_time=response_time,
                data_size=len(response.content) if success else 0,
                cognitive_quality=cognitive_quality,
                endpoint="/geoids",
                status_code=response.status_code
            ))
            
            return success
            
        except Exception as e:
            logger.error(f"  Cognitive processing ERROR: {e}")
            self.test_results.append(RealTestResult(
                test_name="Cognitive Geoid Creation",
                success=False,
                response_time=0.0,
                data_size=0,
                cognitive_quality=0.0,
                endpoint="/geoids",
                status_code=0,
                error_message=str(e)
            ))
            return False
    
    def test_system_metrics(self) -> bool:
        """Test system metrics and monitoring."""
        logger.info("Testing system metrics...")
        
        metrics_endpoints = [
            "/metrics",
            "/monitoring/metrics",
            "/monitoring/metrics/summary",
            "/monitoring/metrics/system",
            "/monitoring/metrics/kimera",
            "/monitoring/metrics/gpu"
        ]
        
        success_count = 0
        
        for endpoint in metrics_endpoints:
            logger.info(f"  Testing metrics endpoint: {endpoint}")
            
            try:
                start_time = time.perf_counter()
                response = self.session.get(f"{self.base_url}{endpoint}", timeout=15)
                response_time = time.perf_counter() - start_time
                
                success = response.status_code == 200
                if success:
                    success_count += 1
                    logger.info(f"    SUCCESS: {response_time*1000:.2f}ms")
                else:
                    logger.warning(f"    FAILED: Status {response.status_code}")
                
                self.test_results.append(RealTestResult(
                    test_name=f"Metrics {endpoint}",
                    success=success,
                    response_time=response_time,
                    data_size=len(response.content) if success else 0,
                    cognitive_quality=1.0 if success else 0.0,
                    endpoint=endpoint,
                    status_code=response.status_code
                ))
                
            except Exception as e:
                logger.error(f"    ERROR: {e}")
                self.test_results.append(RealTestResult(
                    test_name=f"Metrics {endpoint}",
                    success=False,
                    response_time=0.0,
                    data_size=0,
                    cognitive_quality=0.0,
                    endpoint=endpoint,
                    status_code=0,
                    error_message=str(e)
                ))
        
        success_rate = success_count / len(metrics_endpoints)
        logger.info(f"Metrics tests: {success_rate:.2%} success ({success_count}/{len(metrics_endpoints)})")
        
        return success_rate >= 0.7
    
    def test_revolutionary_intelligence(self) -> bool:
        """Test revolutionary intelligence capabilities."""
        logger.info("Testing revolutionary intelligence...")
        
        revolutionary_endpoints = [
            "/revolutionary/status/complete",
            "/revolutionary/intelligence/complete",
            "/revolutionary/health",
            "/revolutionary/principles",
            "/revolutionary/test/simple"
        ]
        
        success_count = 0
        
        for endpoint in revolutionary_endpoints:
            logger.info(f"  Testing revolutionary endpoint: {endpoint}")
            
            try:
                start_time = time.perf_counter()
                response = self.session.get(f"{self.base_url}{endpoint}", timeout=20)
                response_time = time.perf_counter() - start_time
                
                success = response.status_code == 200
                if success:
                    success_count += 1
                    logger.info(f"    SUCCESS: {response_time*1000:.2f}ms")
                    
                    # Analyze revolutionary intelligence quality
                    if endpoint == "/revolutionary/intelligence/complete":
                        result_data = response.json()
                        cognitive_quality = self._analyze_revolutionary_quality(result_data)
                    else:
                        cognitive_quality = 1.0
                else:
                    logger.warning(f"    FAILED: Status {response.status_code}")
                    cognitive_quality = 0.0
                
                self.test_results.append(RealTestResult(
                    test_name=f"Revolutionary {endpoint}",
                    success=success,
                    response_time=response_time,
                    data_size=len(response.content) if success else 0,
                    cognitive_quality=cognitive_quality,
                    endpoint=endpoint,
                    status_code=response.status_code
                ))
                
            except Exception as e:
                logger.error(f"    ERROR: {e}")
                self.test_results.append(RealTestResult(
                    test_name=f"Revolutionary {endpoint}",
                    success=False,
                    response_time=0.0,
                    data_size=0,
                    cognitive_quality=0.0,
                    endpoint=endpoint,
                    status_code=0,
                    error_message=str(e)
                ))
        
        success_rate = success_count / len(revolutionary_endpoints)
        logger.info(f"Revolutionary tests: {success_rate:.2%} success ({success_count}/{len(revolutionary_endpoints)})")
        
        return success_rate >= 0.6
    
    def test_concurrent_processing(self) -> bool:
        """Test concurrent processing capabilities."""
        logger.info("Testing concurrent processing...")
        
        concurrent_requests = 5
        endpoint = "/system/health"
        
        def make_request(request_id: int) -> tuple:
            """Make a single request and return results."""
            try:
                start_time = time.perf_counter()
                response = requests.get(
                    f"{self.base_url}{endpoint}",
                    timeout=10
                )
                response_time = time.perf_counter() - start_time
                
                return request_id, response.status_code == 200, response_time, len(response.content) if response.status_code == 200 else 0
                
            except Exception as e:
                logger.warning(f"Request {request_id} failed: {e}")
                return request_id, False, 0.0, 0
        
        # Execute concurrent requests
        start_time = time.perf_counter()
        
        with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            futures = [executor.submit(make_request, i) for i in range(concurrent_requests)]
            results = [future.result() for future in futures]
        
        total_time = time.perf_counter() - start_time
        
        # Analyze results
        successful_requests = sum(1 for _, success, _, _ in results if success)
        response_times = [rt for _, success, rt, _ in results if success]
        data_sizes = [ds for _, success, _, ds in results if success]
        
        success_rate = successful_requests / concurrent_requests
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        throughput = concurrent_requests / total_time
        
        logger.info(f"Concurrent processing results:")
        logger.info(f"  Success rate: {success_rate:.2%}")
        logger.info(f"  Average response time: {avg_response_time*1000:.2f}ms")
        logger.info(f"  Throughput: {throughput:.2f} requests/second")
        logger.info(f"  Total time: {total_time:.2f}s")
        
        self.test_results.append(RealTestResult(
            test_name="Concurrent Processing",
            success=success_rate >= 0.8,
            response_time=avg_response_time,
            data_size=sum(data_sizes) // len(data_sizes) if data_sizes else 0,
            cognitive_quality=success_rate,
            endpoint=endpoint,
            status_code=200 if success_rate >= 0.8 else 500
        ))
        
        return success_rate >= 0.8 and avg_response_time < 5.0
    
    def _analyze_geoid_quality(self, result_data: Dict[str, Any]) -> float:
        """Analyze geoid creation quality."""
        try:
            quality_score = 0.0
            total_checks = 0
            
            # Check for geoid structure
            if "geoid_id" in result_data or "id" in result_data:
                quality_score += 1.0
                total_checks += 1
            
            # Check for embeddings or vectors
            if "embeddings" in result_data or "vectors" in result_data:
                quality_score += 1.0
                total_checks += 1
            
            # Check for semantic analysis
            if "analysis" in result_data or "semantic" in result_data:
                quality_score += 1.0
                total_checks += 1
            
            # Check for response completeness
            if len(str(result_data)) > 100:  # Reasonable response size
                quality_score += 0.5
                total_checks += 1
            
            return quality_score / total_checks if total_checks > 0 else 0.5
            
        except Exception:
            return 0.5
    
    def _analyze_revolutionary_quality(self, result_data: Dict[str, Any]) -> float:
        """Analyze revolutionary intelligence quality."""
        try:
            quality_score = 0.0
            total_checks = 0
            
            # Check for intelligence indicators
            intelligence_keys = ["analysis", "insights", "assessment", "intelligence", "revolutionary"]
            for key in intelligence_keys:
                if key in str(result_data).lower():
                    quality_score += 0.2
                    total_checks += 1
            
            # Check for response depth
            if len(str(result_data)) > 200:
                quality_score += 0.5
                total_checks += 1
            
            return min(quality_score / total_checks if total_checks > 0 else 0.5, 1.0)
            
        except Exception:
            return 0.5
    
    def generate_real_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive real performance report."""
        if not self.test_results:
            return {"error": "No test results available"}
        
        # Calculate overall metrics
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results if result.success)
        success_rate = successful_tests / total_tests
        
        response_times = [result.response_time for result in self.test_results if result.response_time > 0]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        cognitive_qualities = [result.cognitive_quality for result in self.test_results]
        avg_cognitive_quality = sum(cognitive_qualities) / len(cognitive_qualities) if cognitive_qualities else 0
        
        data_sizes = [result.data_size for result in self.test_results]
        avg_data_size = sum(data_sizes) / len(data_sizes) if data_sizes else 0
        
        # Performance classification
        if success_rate >= 0.95 and avg_response_time < 2.0:
            performance_grade = "EXCELLENT"
        elif success_rate >= 0.85 and avg_response_time < 5.0:
            performance_grade = "GOOD"
        elif success_rate >= 0.70 and avg_response_time < 10.0:
            performance_grade = "ACCEPTABLE"
        else:
            performance_grade = "NEEDS_IMPROVEMENT"
        
        # Group results by test category
        test_categories = {}
        for result in self.test_results:
            category = result.test_name.split()[0]
            if category not in test_categories:
                test_categories[category] = {"total": 0, "successful": 0}
            test_categories[category]["total"] += 1
            if result.success:
                test_categories[category]["successful"] += 1
        
        category_success_rates = {
            category: data["successful"] / data["total"]
            for category, data in test_categories.items()
        }
        
        report = {
            "test_summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "success_rate": success_rate,
                "performance_grade": performance_grade
            },
            "performance_metrics": {
                "average_response_time_ms": avg_response_time * 1000,
                "average_cognitive_quality": avg_cognitive_quality,
                "average_data_size_bytes": avg_data_size,
                "throughput_estimate": 1.0 / avg_response_time if avg_response_time > 0 else 0
            },
            "category_breakdown": category_success_rates,
            "detailed_results": [
                {
                    "test_name": result.test_name,
                    "success": result.success,
                    "response_time_ms": result.response_time * 1000,
                    "cognitive_quality": result.cognitive_quality,
                    "data_size": result.data_size,
                    "endpoint": result.endpoint,
                    "status_code": result.status_code,
                    "error": result.error_message
                }
                for result in self.test_results
            ],
            "system_assessment": {
                "health_status": "GOOD" if category_success_rates.get("Health", 0) >= 0.8 else "NEEDS_ATTENTION",
                "cognitive_capability": "OPERATIONAL" if category_success_rates.get("Cognitive", 0) >= 0.5 else "LIMITED",
                "revolutionary_intelligence": "ACTIVE" if category_success_rates.get("Revolutionary", 0) >= 0.6 else "IMPAIRED",
                "overall_readiness": "PRODUCTION_READY" if success_rate >= 0.8 else "DEVELOPMENT_STAGE"
            },
            "recommendations": self._generate_real_recommendations(success_rate, avg_response_time, avg_cognitive_quality, category_success_rates)
        }
        
        return report
    
    def _generate_real_recommendations(self, success_rate: float, avg_response_time: float, 
                                     avg_cognitive_quality: float, category_rates: Dict[str, float]) -> List[str]:
        """Generate real performance recommendations."""
        recommendations = []
        
        if success_rate < 0.85:
            recommendations.append("CRITICAL: Improve overall system reliability - success rate below 85%")
        
        if avg_response_time > 5.0:
            recommendations.append("OPTIMIZE: Response time averaging above 5 seconds - performance tuning needed")
        
        if avg_cognitive_quality < 0.7:
            recommendations.append("ENHANCE: Cognitive processing quality below 70% - algorithm improvements needed")
        
        if category_rates.get("Health", 0) < 0.8:
            recommendations.append("URGENT: System health endpoints failing - infrastructure issues detected")
        
        if category_rates.get("Revolutionary", 0) < 0.6:
            recommendations.append("INVESTIGATE: Revolutionary intelligence below 60% - core functionality impaired")
        
        if category_rates.get("Cognitive", 0) < 0.5:
            recommendations.append("REPAIR: Cognitive geoid processing failing - semantic engine requires attention")
        
        if not recommendations:
            recommendations.extend([
                "EXCELLENT: System performing optimally across all metrics",
                "READY: Production deployment recommended",
                "SCALE: Consider load balancing for increased capacity"
            ])
        
        return recommendations
    
    def run_comprehensive_real_test(self) -> bool:
        """Run comprehensive real system test suite."""
        logger.info("Starting Comprehensive REAL KIMERA System Test")
        logger.info("=" * 80)
        
        test_start_time = time.perf_counter()
        overall_success = True
        
        # Real test sequence using actual endpoints
        tests = [
            ("System Health", self.test_system_health),
            ("Cognitive Processing", self.test_cognitive_processing),
            ("System Metrics", self.test_system_metrics),
            ("Revolutionary Intelligence", self.test_revolutionary_intelligence),
            ("Concurrent Processing", self.test_concurrent_processing)
        ]
        
        for test_name, test_function in tests:
            logger.info(f"\nRunning: {test_name}")
            try:
                success = test_function()
                if not success:
                    overall_success = False
                    logger.error(f"FAILED: {test_name}")
                else:
                    logger.info(f"PASSED: {test_name}")
            except Exception as e:
                overall_success = False
                logger.error(f"ERROR in {test_name}: {e}")
        
        test_duration = time.perf_counter() - test_start_time
        
        # Generate and save report
        logger.info("\nGenerating comprehensive performance report...")
        performance_report = self.generate_real_performance_report()
        performance_report["test_duration_seconds"] = test_duration
        
        # Save report
        report_filename = f"logs/real_kimera_test_report_{int(time.time())}.json"
        with open(report_filename, 'w') as f:
            json.dump(performance_report, f, indent=2)
        
        # Display summary
        logger.info("\n" + "=" * 80)
        logger.info("REAL KIMERA SYSTEM TEST RESULTS")
        logger.info("=" * 80)
        
        summary = performance_report["test_summary"]
        metrics = performance_report["performance_metrics"]
        assessment = performance_report["system_assessment"]
        
        logger.info(f"OVERALL RESULTS:")
        logger.info(f"  Success Rate: {summary['success_rate']:.2%}")
        logger.info(f"  Performance Grade: {summary['performance_grade']}")
        logger.info(f"  Test Duration: {test_duration:.2f}s")
        
        logger.info(f"\nPERFORMANCE METRICS:")
        logger.info(f"  Average Response Time: {metrics['average_response_time_ms']:.2f}ms")
        logger.info(f"  Cognitive Quality: {metrics['average_cognitive_quality']:.2%}")
        logger.info(f"  Throughput: {metrics['throughput_estimate']:.2f} req/sec")
        
        logger.info(f"\nSYSTEM ASSESSMENT:")
        logger.info(f"  Health Status: {assessment['health_status']}")
        logger.info(f"  Cognitive Capability: {assessment['cognitive_capability']}")
        logger.info(f"  Revolutionary Intelligence: {assessment['revolutionary_intelligence']}")
        logger.info(f"  Overall Readiness: {assessment['overall_readiness']}")
        
        logger.info(f"\nRECOMMENDATIONS:")
        for recommendation in performance_report["recommendations"]:
            logger.info(f"  {recommendation}")
        
        logger.info(f"\nFull report saved: {report_filename}")
        logger.info("=" * 80)
        
        return overall_success

def main():
    """Main test execution."""
    
    # Check if KIMERA is running
    try:
        with open('.port.tmp', 'r') as f:
            port = f.read().strip()
        base_url = f"http://localhost:{port}"
    except:
        logger.error("KIMERA system not running. Please start with: python run_kimera.py")
        return False
    
    # Run real tests
    tester = RealKimeraSystemTester(base_url)
    success = tester.run_comprehensive_real_test()
    
    if success:
        logger.info("\nLive KIMERA System Test: PASSED")
        logger.info("System ready for production deployment")
        return True
    else:
        logger.info("\nLive KIMERA System Test: ISSUES DETECTED")
        logger.info("Review test results and address identified issues")
        return False

if __name__ == "__main__":
    main() 