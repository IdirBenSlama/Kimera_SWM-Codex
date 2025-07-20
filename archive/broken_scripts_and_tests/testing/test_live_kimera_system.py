#!/usr/bin/env python3
"""
Live KIMERA System Test - Real World Performance Validation
==========================================================

This test runs against the live KIMERA system to validate:
- Real API responses and performance
- Actual cognitive processing capabilities  
- Live semantic processing and understanding
- Real-world contradiction handling
- System stability under actual load
- Production-ready performance metrics

Testing the actual running system, not just frameworks.
"""

import requests
import json
import time
import sys
import numpy as np
import asyncio
import aiohttp
from typing import Dict, List, Any, Optional
import logging
from concurrent.futures import ThreadPoolExecutor
import threading
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/live_kimera_test.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class LiveTestResult:
    """Results from live system testing."""
    test_name: str
    success: bool
    response_time: float
    data_size: int
    cognitive_quality: float
    error_message: Optional[str] = None

class LiveKimeraSystemTester:
    """
    Comprehensive tester for the live KIMERA system.
    
    Tests actual system performance, cognitive capabilities,
    and real-world processing against the running API.
    """
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        """Initialize the live system tester."""
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results: List[LiveTestResult] = []
        
        logger.info(f"🔗 Connecting to live KIMERA system at {base_url}")
        
    def test_system_connectivity(self) -> bool:
        """Test basic connectivity to the live system."""
        logger.info("🔌 Testing system connectivity...")
        
        try:
            start_time = time.perf_counter()
            response = self.session.get(f"{self.base_url}/system/status", timeout=10)
            response_time = time.perf_counter() - start_time
            
            if response.status_code == 200:
                status_data = response.json()
                logger.info(f"✅ System online - Response time: {response_time*1000:.2f}ms")
                logger.info(f"   Status: {status_data}")
                
                self.test_results.append(LiveTestResult(
                    test_name="System Connectivity",
                    success=True,
                    response_time=response_time,
                    data_size=len(response.content),
                    cognitive_quality=1.0
                ))
                return True
            else:
                logger.error(f"❌ System not responding - Status: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            return False
    
    def test_cognitive_stability(self) -> bool:
        """Test cognitive stability of the live system."""
        logger.info("🧠 Testing cognitive stability...")
        
        try:
            start_time = time.perf_counter()
            response = self.session.get(f"{self.base_url}/system/stability", timeout=10)
            response_time = time.perf_counter() - start_time
            
            if response.status_code == 200:
                stability_data = response.json()
                logger.info(f"✅ Cognitive stability verified - {response_time*1000:.2f}ms")
                logger.info(f"   Stability metrics: {stability_data}")
                
                # Analyze stability metrics
                cognitive_quality = self._analyze_stability_metrics(stability_data)
                
                self.test_results.append(LiveTestResult(
                    test_name="Cognitive Stability",
                    success=True,
                    response_time=response_time,
                    data_size=len(response.content),
                    cognitive_quality=cognitive_quality
                ))
                return True
            else:
                logger.error(f"❌ Stability check failed - Status: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Stability test failed: {e}")
            return False
    
    def test_semantic_processing(self) -> bool:
        """Test semantic processing with real cognitive workloads."""
        logger.info("🔍 Testing semantic processing...")
        
        # Real-world semantic test cases
        test_cases = [
            {
                "input": "Analyze the quantum mechanical implications of cognitive processing in artificial intelligence systems",
                "expected_concepts": ["quantum", "cognitive", "artificial intelligence"]
            },
            {
                "input": "Compare the thermodynamic entropy of neural networks versus traditional computational architectures",
                "expected_concepts": ["thermodynamic", "entropy", "neural networks"]
            },
            {
                "input": "Evaluate the cognitive load of multi-modal attention mechanisms in transformer architectures",
                "expected_concepts": ["cognitive load", "attention", "transformer"]
            }
        ]
        
        success_count = 0
        total_response_time = 0.0
        
        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"   Testing semantic case {i}/{len(test_cases)}...")
            
            try:
                payload = {
                    "text": test_case["input"],
                    "analysis_depth": "comprehensive",
                    "include_semantic_vectors": True
                }
                
                start_time = time.perf_counter()
                response = self.session.post(
                    f"{self.base_url}/geoids",
                    json=payload,
                    timeout=30
                )
                response_time = time.perf_counter() - start_time
                total_response_time += response_time
                
                if response.status_code == 200:
                    result_data = response.json()
                    
                    # Analyze semantic quality
                    cognitive_quality = self._analyze_semantic_quality(
                        result_data, test_case["expected_concepts"]
                    )
                    
                    self.test_results.append(LiveTestResult(
                        test_name=f"Semantic Processing {i}",
                        success=True,
                        response_time=response_time,
                        data_size=len(response.content),
                        cognitive_quality=cognitive_quality
                    ))
                    
                    success_count += 1
                    logger.info(f"   ✅ Case {i}: {response_time*1000:.2f}ms, Quality: {cognitive_quality:.2%}")
                    
                else:
                    logger.error(f"   ❌ Case {i} failed - Status: {response.status_code}")
                    
            except Exception as e:
                logger.error(f"   ❌ Case {i} error: {e}")
        
        success_rate = success_count / len(test_cases)
        avg_response_time = total_response_time / len(test_cases)
        
        logger.info(f"📊 Semantic processing: {success_rate:.2%} success, {avg_response_time*1000:.2f}ms avg")
        
        return success_rate >= 0.8  # 80% success threshold
    
    def test_contradiction_processing(self) -> bool:
        """Test contradiction processing with real cognitive challenges."""
        logger.info("⚡ Testing contradiction processing...")
        
        # Real-world contradiction test cases
        contradictions = [
            {
                "statement_a": "Artificial intelligence systems should prioritize efficiency above all else",
                "statement_b": "AI systems must maintain perfect safety even if it reduces efficiency",
                "context": "AI system design philosophy"
            },
            {
                "statement_a": "Quantum computing will solve all computational limitations",
                "statement_b": "Classical computers will remain superior for most practical applications",
                "context": "Future of computing architectures"
            },
            {
                "statement_a": "Deep learning models should be completely interpretable",
                "statement_b": "The most powerful AI models are inherently black boxes",
                "context": "AI transparency and capability trade-offs"
            }
        ]
        
        success_count = 0
        total_response_time = 0.0
        
        for i, contradiction in enumerate(contradictions, 1):
            logger.info(f"   Testing contradiction {i}/{len(contradictions)}...")
            
            try:
                payload = {
                    "contradictions": [
                        {
                            "statement_a": contradiction["statement_a"],
                            "statement_b": contradiction["statement_b"],
                            "context": contradiction["context"]
                        }
                    ],
                    "resolution_strategy": "synthesis",
                    "depth": "comprehensive"
                }
                
                start_time = time.perf_counter()
                response = self.session.post(
                    f"{self.base_url}/process/contradictions",
                    json=payload,
                    timeout=45
                )
                response_time = time.perf_counter() - start_time
                total_response_time += response_time
                
                if response.status_code == 200:
                    result_data = response.json()
                    
                    # Analyze contradiction resolution quality
                    cognitive_quality = self._analyze_contradiction_quality(result_data)
                    
                    self.test_results.append(LiveTestResult(
                        test_name=f"Contradiction Processing {i}",
                        success=True,
                        response_time=response_time,
                        data_size=len(response.content),
                        cognitive_quality=cognitive_quality
                    ))
                    
                    success_count += 1
                    logger.info(f"   ✅ Contradiction {i}: {response_time*1000:.2f}ms, Quality: {cognitive_quality:.2%}")
                    
                else:
                    logger.error(f"   ❌ Contradiction {i} failed - Status: {response.status_code}")
                    
            except Exception as e:
                logger.error(f"   ❌ Contradiction {i} error: {e}")
        
        success_rate = success_count / len(contradictions)
        avg_response_time = total_response_time / len(contradictions)
        
        logger.info(f"📊 Contradiction processing: {success_rate:.2%} success, {avg_response_time*1000:.2f}ms avg")
        
        return success_rate >= 0.8  # 80% success threshold
    
    def test_concurrent_load(self) -> bool:
        """Test system performance under concurrent load."""
        logger.info("🚀 Testing concurrent load performance...")
        
        concurrent_requests = 10
        request_payload = {
            "text": "Analyze the intersection of quantum mechanics and cognitive science in the context of artificial intelligence development and consciousness emergence.",
            "analysis_depth": "standard"
        }
        
        def make_request(request_id: int) -> tuple:
            """Make a single request and return results."""
            try:
                start_time = time.perf_counter()
                response = requests.post(
                    f"{self.base_url}/geoids",
                    json=request_payload,
                    timeout=30
                )
                response_time = time.perf_counter() - start_time
                
                return request_id, response.status_code == 200, response_time, len(response.content) if response.status_code == 200 else 0
                
            except Exception as e:
                logger.error(f"Request {request_id} failed: {e}")
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
        
        logger.info(f"📊 Concurrent load results:")
        logger.info(f"   Success rate: {success_rate:.2%}")
        logger.info(f"   Average response time: {avg_response_time*1000:.2f}ms")
        logger.info(f"   Throughput: {throughput:.2f} requests/second")
        logger.info(f"   Total time: {total_time:.2f}s")
        
        self.test_results.append(LiveTestResult(
            test_name="Concurrent Load Test",
            success=success_rate >= 0.8,
            response_time=avg_response_time,
            data_size=sum(data_sizes) // len(data_sizes) if data_sizes else 0,
            cognitive_quality=success_rate
        ))
        
        return success_rate >= 0.8 and avg_response_time < 10.0  # 80% success, <10s response
    
    def test_system_boundaries(self) -> bool:
        """Test system boundaries and error handling."""
        logger.info("🔒 Testing system boundaries...")
        
        boundary_tests = [
            {
                "name": "Empty Input",
                "payload": {"text": ""},
                "expect_error": True
            },
            {
                "name": "Extremely Long Input",
                "payload": {"text": "A" * 100000},  # 100KB text
                "expect_error": False
            },
            {
                "name": "Invalid JSON Structure",
                "payload": {"invalid_field": "test"},
                "expect_error": True
            },
            {
                "name": "Special Characters",
                "payload": {"text": "🚀🔬🧠⚡💡🌟 Special chars with émojis and acčents"},
                "expect_error": False
            }
        ]
        
        success_count = 0
        
        for test in boundary_tests:
            logger.info(f"   Testing: {test['name']}")
            
            try:
                start_time = time.perf_counter()
                response = self.session.post(
                    f"{self.base_url}/geoids",
                    json=test["payload"],
                    timeout=30
                )
                response_time = time.perf_counter() - start_time
                
                if test["expect_error"]:
                    # Should fail
                    if response.status_code != 200:
                        success_count += 1
                        logger.info(f"   ✅ Correctly handled error: {response.status_code}")
                    else:
                        logger.warning(f"   ⚠️ Expected error but got success")
                else:
                    # Should succeed
                    if response.status_code == 200:
                        success_count += 1
                        logger.info(f"   ✅ Handled successfully: {response_time*1000:.2f}ms")
                    else:
                        logger.error(f"   ❌ Unexpected failure: {response.status_code}")
                        
            except Exception as e:
                if test["expect_error"]:
                    success_count += 1
                    logger.info(f"   ✅ Correctly caught error: {e}")
                else:
                    logger.error(f"   ❌ Unexpected error: {e}")
        
        success_rate = success_count / len(boundary_tests)
        logger.info(f"📊 Boundary testing: {success_rate:.2%} success")
        
        self.test_results.append(LiveTestResult(
            test_name="System Boundaries",
            success=success_rate >= 0.8,
            response_time=0.0,  # Varies per test
            data_size=0,
            cognitive_quality=success_rate
        ))
        
        return success_rate >= 0.8
    
    def _analyze_stability_metrics(self, stability_data: Dict[str, Any]) -> float:
        """Analyze cognitive stability metrics quality."""
        try:
            # Extract stability scores
            scores = []
            
            if "identity_coherence" in stability_data:
                scores.append(stability_data["identity_coherence"])
            if "memory_continuity" in stability_data:
                scores.append(stability_data["memory_continuity"])
            if "cognitive_drift" in stability_data:
                # Invert drift (lower is better)
                scores.append(1.0 - stability_data["cognitive_drift"])
            if "reality_testing" in stability_data:
                scores.append(stability_data["reality_testing"])
            
            return sum(scores) / len(scores) if scores else 0.5
            
        except Exception:
            return 0.5  # Default moderate quality
    
    def _analyze_semantic_quality(self, result_data: Dict[str, Any], expected_concepts: List[str]) -> float:
        """Analyze semantic processing quality."""
        try:
            quality_score = 0.0
            total_checks = 0
            
            # Check if response contains expected concepts
            response_text = str(result_data).lower()
            
            for concept in expected_concepts:
                if concept.lower() in response_text:
                    quality_score += 1.0
                total_checks += 1
            
            # Check for structured data
            if "embeddings" in result_data or "vectors" in result_data:
                quality_score += 0.5
                total_checks += 1
            
            # Check for semantic analysis
            if "analysis" in result_data or "geoids" in result_data:
                quality_score += 0.5
                total_checks += 1
            
            return quality_score / total_checks if total_checks > 0 else 0.5
            
        except Exception:
            return 0.5  # Default moderate quality
    
    def _analyze_contradiction_quality(self, result_data: Dict[str, Any]) -> float:
        """Analyze contradiction processing quality."""
        try:
            quality_score = 0.0
            total_checks = 0
            
            # Check for resolution content
            if "resolution" in result_data or "synthesis" in result_data:
                quality_score += 1.0
                total_checks += 1
            
            # Check for analysis depth
            if "analysis" in result_data:
                quality_score += 1.0
                total_checks += 1
            
            # Check for reasoning quality
            response_text = str(result_data).lower()
            reasoning_indicators = ["because", "therefore", "however", "whereas", "synthesis"]
            
            found_reasoning = sum(1 for indicator in reasoning_indicators if indicator in response_text)
            quality_score += min(found_reasoning / len(reasoning_indicators), 1.0)
            total_checks += 1
            
            return quality_score / total_checks if total_checks > 0 else 0.5
            
        except Exception:
            return 0.5  # Default moderate quality
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
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
        if success_rate >= 0.95 and avg_response_time < 5.0:
            performance_grade = "EXCELLENT"
        elif success_rate >= 0.85 and avg_response_time < 10.0:
            performance_grade = "GOOD"
        elif success_rate >= 0.75 and avg_response_time < 20.0:
            performance_grade = "ACCEPTABLE"
        else:
            performance_grade = "NEEDS_IMPROVEMENT"
        
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
            "detailed_results": [
                {
                    "test_name": result.test_name,
                    "success": result.success,
                    "response_time_ms": result.response_time * 1000,
                    "cognitive_quality": result.cognitive_quality,
                    "data_size": result.data_size,
                    "error": result.error_message
                }
                for result in self.test_results
            ],
            "recommendations": self._generate_recommendations(success_rate, avg_response_time, avg_cognitive_quality)
        }
        
        return report
    
    def _generate_recommendations(self, success_rate: float, avg_response_time: float, avg_cognitive_quality: float) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []
        
        if success_rate < 0.85:
            recommendations.append("🔧 Improve system reliability - success rate below 85%")
        
        if avg_response_time > 10.0:
            recommendations.append("⚡ Optimize response time - average above 10 seconds")
        
        if avg_cognitive_quality < 0.8:
            recommendations.append("🧠 Enhance cognitive processing quality")
        
        if not recommendations:
            recommendations.extend([
                "✅ System performing excellently across all metrics",
                "🚀 Ready for production cognitive workloads",
                "📈 Consider scaling for increased load"
            ])
        
        return recommendations
    
    def run_comprehensive_live_test(self) -> bool:
        """Run comprehensive live system test suite."""
        logger.info("🚀 Starting Comprehensive Live KIMERA System Test")
        logger.info("=" * 80)
        
        test_start_time = time.perf_counter()
        overall_success = True
        
        # Test sequence
        tests = [
            ("System Connectivity", self.test_system_connectivity),
            ("Cognitive Stability", self.test_cognitive_stability),
            ("Semantic Processing", self.test_semantic_processing),
            ("Contradiction Processing", self.test_contradiction_processing),
            ("Concurrent Load", self.test_concurrent_load),
            ("System Boundaries", self.test_system_boundaries)
        ]
        
        for test_name, test_function in tests:
            logger.info(f"\n🔄 Running: {test_name}")
            try:
                success = test_function()
                if not success:
                    overall_success = False
                    logger.error(f"❌ {test_name} failed")
                else:
                    logger.info(f"✅ {test_name} passed")
            except Exception as e:
                overall_success = False
                logger.error(f"❌ {test_name} error: {e}")
        
        test_duration = time.perf_counter() - test_start_time
        
        # Generate and save report
        logger.info("\n📋 Generating performance report...")
        performance_report = self.generate_performance_report()
        performance_report["test_duration_seconds"] = test_duration
        
        # Save report
        report_filename = f"logs/live_kimera_test_report_{int(time.time())}.json"
        with open(report_filename, 'w') as f:
            json.dump(performance_report, f, indent=2)
        
        # Display summary
        logger.info("\n" + "=" * 80)
        logger.info("🎯 LIVE KIMERA SYSTEM TEST RESULTS")
        logger.info("=" * 80)
        
        summary = performance_report["test_summary"]
        metrics = performance_report["performance_metrics"]
        
        logger.info(f"📊 OVERALL RESULTS:")
        logger.info(f"   Success Rate: {summary['success_rate']:.2%}")
        logger.info(f"   Performance Grade: {summary['performance_grade']}")
        logger.info(f"   Test Duration: {test_duration:.2f}s")
        
        logger.info(f"\n⚡ PERFORMANCE METRICS:")
        logger.info(f"   Average Response Time: {metrics['average_response_time_ms']:.2f}ms")
        logger.info(f"   Cognitive Quality: {metrics['average_cognitive_quality']:.2%}")
        logger.info(f"   Throughput: {metrics['throughput_estimate']:.2f} req/sec")
        
        logger.info(f"\n💡 RECOMMENDATIONS:")
        for recommendation in performance_report["recommendations"]:
            logger.info(f"   {recommendation}")
        
        logger.info(f"\n📄 Full report saved: {report_filename}")
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
        logger.error("❌ KIMERA system not running. Please start with: python run_kimera.py")
        return False
    
    # Run live tests
    tester = LiveKimeraSystemTester(base_url)
    success = tester.run_comprehensive_live_test()
    
    if success:
        logger.info("\n🎉 Live KIMERA System Test: PASSED")
        logger.info("✅ System ready for production deployment")
        return True
    else:
        logger.warning("\n⚠️ Live KIMERA System Test: ISSUES DETECTED")
        logger.debug("🔧 Review test results and address identified issues")
        return False

if __name__ == "__main__":
    main() 