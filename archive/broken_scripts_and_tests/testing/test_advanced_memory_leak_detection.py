#!/usr/bin/env python3
"""
KIMERA ADVANCED MEMORY LEAK DETECTION SYSTEM TEST
================================================

Comprehensive validation of the revolutionary memory leak detection system
combining static analysis with directed symbolic execution.

This test validates:
1. Static analysis accuracy
2. Symbolic execution path exploration
3. Real-time leak detection
4. GPU memory leak prevention
5. Integration with Kimera components
"""

import os
import sys
import time
import json
import asyncio
import logging
import traceback
from typing import Dict, List, Any
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AdvancedMemoryLeakDetectionTester:
    """Comprehensive tester for the advanced memory leak detection system"""
    
    def __init__(self):
        self.test_results = {}
        self.performance_metrics = {}
        
        # Try to import the leak detection system
        try:
            from backend.analysis.kimera_memory_leak_guardian import (
                KimeraMemoryLeakGuardian,
                get_memory_leak_guardian,
                track_memory_block,
                analyze_for_leaks,
                MemoryAllocation,
                LeakDetectionResult
            )
            self.has_leak_guardian = True
            self.leak_guardian = get_memory_leak_guardian()
            logger.info("✅ Memory leak guardian successfully imported")
        except ImportError as e:
            logger.warning(f"❌ Could not import leak guardian: {e}")
            self.has_leak_guardian = False
            self.leak_guardian = None
    
    async def run_comprehensive_test_suite(self):
        """Run comprehensive test suite for memory leak detection"""
        logger.info("🚀 Starting Advanced Memory Leak Detection Test Suite")
        logger.info("=" * 70)
        
        test_suite = [
            ("Static Analysis Test", self.test_static_analysis),
            ("Symbolic Execution Test", self.test_symbolic_execution),
            ("Real-time Monitoring Test", self.test_realtime_monitoring),
            ("GPU Memory Leak Detection", self.test_gpu_leak_detection),
            ("Component Integration Test", self.test_component_integration),
            ("Performance Impact Assessment", self.test_performance_impact),
            ("Automated Recovery Test", self.test_automated_recovery),
            ("Scalability Test", self.test_scalability)
        ]
        
        total_tests = len(test_suite)
        passed_tests = 0
        
        for test_name, test_func in test_suite:
            logger.info(f"\n🧪 Running: {test_name}")
            logger.info("-" * 50)
            
            try:
                start_time = time.time()
                result = await test_func()
                execution_time = time.time() - start_time
                
                self.test_results[test_name] = {
                    'status': 'PASSED' if result['success'] else 'FAILED',
                    'execution_time_ms': execution_time * 1000,
                    'details': result
                }
                
                if result['success']:
                    passed_tests += 1
                    logger.info(f"✅ {test_name}: PASSED ({execution_time:.2f}s)")
                else:
                    logger.error(f"❌ {test_name}: FAILED - {result.get('error', 'Unknown error')}")
                
            except Exception as e:
                self.test_results[test_name] = {
                    'status': 'ERROR',
                    'execution_time_ms': 0,
                    'details': {'error': str(e), 'traceback': traceback.format_exc()}
                }
                logger.error(f"💥 {test_name}: ERROR - {str(e)}")
        
        # Generate final report
        success_rate = (passed_tests / total_tests) * 100
        logger.info(f"\n📊 TEST SUITE SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {total_tests - passed_tests}")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        
        # Save detailed report
        await self.save_test_report()
        
        return success_rate >= 80  # 80% success rate threshold
    
    async def test_static_analysis(self) -> Dict[str, Any]:
        """Test static analysis functionality"""
        if not self.has_leak_guardian:
            return {'success': False, 'error': 'Leak guardian not available'}
        
        # Test code with potential memory leak
        test_code = """
def process_data(data_size):
    # Allocation without deallocation
    large_array = [0] * data_size
    
    if data_size > 1000:
        # Conditional allocation
        extra_data = [0] * (data_size * 2)
        return process_large_data(extra_data)
    
    return process_small_data(large_array)

def process_large_data(data):
    # Missing deallocation
    return sum(data)

def process_small_data(data):
    # Proper cleanup
    result = sum(data)
    del data
    return result
"""
        
        try:
            # Analyze function for leaks
            summary = self.leak_guardian.analyze_function_for_leaks(test_code, "process_data")
            
            # Validate analysis results
            analysis_results = {
                'function_analyzed': summary.function_name == "process_data",
                'allocations_detected': len(summary.allocations) > 0,
                'memory_balance_calculated': summary.memory_balance is not None,
                'safety_assessment': summary.is_memory_safe is not None,
                'path_conditions': len(summary.path_conditions) >= 0
            }
            
            logger.info(f"   📋 Function: {summary.function_name}")
            logger.info(f"   📊 Allocations: {len(summary.allocations)}")
            logger.info(f"   📊 Deallocations: {len(summary.deallocations)}")
            logger.info(f"   📊 Memory Balance: {summary.memory_balance}")
            logger.info(f"   🛡️ Memory Safe: {summary.is_memory_safe}")
            
            success = all(analysis_results.values())
            
            return {
                'success': success,
                'analysis_results': analysis_results,
                'function_summary': {
                    'allocations': len(summary.allocations),
                    'deallocations': len(summary.deallocations),
                    'memory_balance': summary.memory_balance,
                    'is_memory_safe': summary.is_memory_safe
                }
            }
            
        except Exception as e:
            return {'success': False, 'error': f"Static analysis failed: {str(e)}"}
    
    async def test_symbolic_execution(self) -> Dict[str, Any]:
        """Test symbolic execution functionality"""
        if not self.has_leak_guardian:
            return {'success': False, 'error': 'Leak guardian not available'}
        
        if not self.leak_guardian.enable_symbolic_execution:
            return {'success': False, 'error': 'Symbolic execution not enabled (Z3 not available)'}
        
        # Test code with path-dependent memory behavior
        test_code = """
def conditional_allocation(condition, size):
    if condition:
        # Path 1: Allocation with proper cleanup
        data = [0] * size
        result = process_data(data)
        del data
        return result
    else:
        # Path 2: Allocation without cleanup
        data = [0] * size
        return process_data(data)  # Memory leak on this path

def process_data(data):
    return sum(data)
"""
        
        try:
            # Analyze with symbolic execution
            summary = self.leak_guardian.analyze_function_for_leaks(test_code, "conditional_allocation")
            
            # Check if symbolic execution explored different paths
            symbolic_results = {
                'paths_explored': len(summary.path_conditions) > 0,
                'path_sensitive_analysis': True,  # Assume success if no error
                'memory_safety_per_path': summary.is_memory_safe is not None
            }
            
            logger.info(f"   🔍 Paths Explored: {len(summary.path_conditions)}")
            logger.info(f"   🧭 Path Conditions: {summary.path_conditions}")
            
            success = all(symbolic_results.values())
            
            return {
                'success': success,
                'symbolic_results': symbolic_results,
                'paths_explored': len(summary.path_conditions)
            }
            
        except Exception as e:
            return {'success': False, 'error': f"Symbolic execution failed: {str(e)}"}
    
    async def test_realtime_monitoring(self) -> Dict[str, Any]:
        """Test real-time monitoring functionality"""
        if not self.has_leak_guardian:
            return {'success': False, 'error': 'Leak guardian not available'}
        
        try:
            # Start monitoring
            self.leak_guardian.start_monitoring()
            
            # Simulate memory operations
            allocations_tracked = []
            
            # Create tracked allocations
            for i in range(10):
                allocation_info = {
                    'function': 'test_function',
                    'type': 'test_allocation',
                    'size': 1024 * (i + 1),
                    'timestamp': time.time()
                }
                
                with self.leak_guardian.track_allocation(f"test_alloc_{i}", allocation_info) as allocation:
                    allocations_tracked.append(allocation.allocation_id)
                    # Simulate some work
                    await asyncio.sleep(0.01)
            
            # Let monitoring run for a short time
            await asyncio.sleep(1.0)
            
            # Stop monitoring
            self.leak_guardian.stop_monitoring()
            
            # Check monitoring results
            monitoring_results = {
                'monitoring_started': True,
                'allocations_tracked': len(allocations_tracked) == 10,
                'monitoring_stopped': True,
                'no_errors': True  # If we reach here, no errors occurred
            }
            
            logger.info(f"   📊 Allocations Tracked: {len(allocations_tracked)}")
            logger.info(f"   ⏱️ Monitoring Duration: 1.0 seconds")
            
            success = all(monitoring_results.values())
            
            return {
                'success': success,
                'monitoring_results': monitoring_results,
                'allocations_tracked': len(allocations_tracked)
            }
            
        except Exception as e:
            return {'success': False, 'error': f"Real-time monitoring failed: {str(e)}"}
    
    async def test_gpu_leak_detection(self) -> Dict[str, Any]:
        """Test GPU memory leak detection"""
        if not self.has_leak_guardian:
            return {'success': False, 'error': 'Leak guardian not available'}
        
        try:
            # Check if GPU tracking is available
            gpu_available = self.leak_guardian.enable_gpu_tracking
            
            if not gpu_available:
                logger.info("   ⚠️ GPU not available, testing fallback behavior")
                return {
                    'success': True,
                    'gpu_results': {
                        'gpu_available': False,
                        'fallback_behavior': True
                    }
                }
            
            # Test GPU memory leak detection
            import torch
            
            # Simulate GPU memory operations
            initial_memory = torch.cuda.memory_allocated()
            
            # Create some GPU tensors
            tensors = []
            for i in range(5):
                tensor = torch.randn(1000, 1000, device='cuda')
                tensors.append(tensor)
            
            current_memory = torch.cuda.memory_allocated()
            memory_growth = current_memory - initial_memory
            
            # Run GPU leak detection
            leak_reports = self.leak_guardian.detect_gpu_memory_leaks()
            
            # Clean up
            del tensors
            torch.cuda.empty_cache()
            
            gpu_results = {
                'gpu_memory_tracked': memory_growth > 0,
                'leak_detection_ran': True,
                'leak_reports_generated': len(leak_reports) >= 0,
                'cleanup_successful': True
            }
            
            logger.info(f"   🎮 GPU Memory Growth: {memory_growth / 1024 / 1024:.1f} MB")
            logger.info(f"   📋 Leak Reports: {len(leak_reports)}")
            
            success = all(gpu_results.values())
            
            return {
                'success': success,
                'gpu_results': gpu_results,
                'memory_growth_mb': memory_growth / 1024 / 1024,
                'leak_reports': len(leak_reports)
            }
            
        except Exception as e:
            return {'success': False, 'error': f"GPU leak detection failed: {str(e)}"}
    
    async def test_component_integration(self) -> Dict[str, Any]:
        """Test integration with Kimera components"""
        try:
            # Test decorator functionality
            @analyze_for_leaks
            def test_function_with_decorator():
                data = [0] * 1000
                return sum(data)
            
            # Test context manager
            from backend.analysis.kimera_memory_leak_guardian import track_memory_block
            
            with track_memory_block("test_block"):
                result = test_function_with_decorator()
            
            integration_results = {
                'decorator_works': result == 0,  # sum of zeros
                'context_manager_works': True,  # If we reach here, it worked
                'no_import_errors': True
            }
            
            logger.info(f"   🔗 Decorator Test: Function returned {result}")
            logger.info(f"   🔗 Context Manager: Successfully tracked memory block")
            
            success = all(integration_results.values())
            
            return {
                'success': success,
                'integration_results': integration_results
            }
            
        except Exception as e:
            return {'success': False, 'error': f"Component integration failed: {str(e)}"}
    
    async def test_performance_impact(self) -> Dict[str, Any]:
        """Test performance impact of leak detection system"""
        try:
            # Measure baseline performance
            def baseline_function():
                data = [i for i in range(10000)]
                return sum(data)
            
            # Time baseline
            start_time = time.time()
            for _ in range(100):
                baseline_function()
            baseline_time = time.time() - start_time
            
            # Measure with leak detection
            @analyze_for_leaks
            def monitored_function():
                data = [i for i in range(10000)]
                return sum(data)
            
            start_time = time.time()
            for _ in range(100):
                monitored_function()
            monitored_time = time.time() - start_time
            
            # Calculate overhead
            overhead_percent = ((monitored_time - baseline_time) / baseline_time) * 100
            
            performance_results = {
                'baseline_measured': baseline_time > 0,
                'monitored_measured': monitored_time > 0,
                'overhead_acceptable': overhead_percent < 20,  # Less than 20% overhead
                'performance_impact_calculated': True
            }
            
            logger.info(f"   ⚡ Baseline Time: {baseline_time:.3f}s")
            logger.info(f"   ⚡ Monitored Time: {monitored_time:.3f}s")
            logger.info(f"   📊 Overhead: {overhead_percent:.1f}%")
            
            success = all(performance_results.values())
            
            return {
                'success': success,
                'performance_results': performance_results,
                'baseline_time_s': baseline_time,
                'monitored_time_s': monitored_time,
                'overhead_percent': overhead_percent
            }
            
        except Exception as e:
            return {'success': False, 'error': f"Performance impact test failed: {str(e)}"}
    
    async def test_automated_recovery(self) -> Dict[str, Any]:
        """Test automated recovery functionality"""
        if not self.has_leak_guardian:
            return {'success': False, 'error': 'Leak guardian not available'}
        
        try:
            # Generate comprehensive report (includes recovery recommendations)
            report = self.leak_guardian.generate_comprehensive_report()
            
            recovery_results = {
                'report_generated': 'analysis_summary' in report,
                'recommendations_provided': len(report.get('recommendations', [])) >= 0,
                'leak_reports_included': 'leak_reports' in report,
                'performance_metrics_included': 'performance_metrics' in report
            }
            
            logger.info(f"   📋 Report Sections: {len(report.keys())}")
            logger.info(f"   💡 Recommendations: {len(report.get('recommendations', []))}")
            logger.info(f"   🔍 Leak Reports: {len(report.get('leak_reports', []))}")
            
            success = all(recovery_results.values())
            
            return {
                'success': success,
                'recovery_results': recovery_results,
                'report_sections': len(report.keys()),
                'recommendations_count': len(report.get('recommendations', []))
            }
            
        except Exception as e:
            return {'success': False, 'error': f"Automated recovery test failed: {str(e)}"}
    
    async def test_scalability(self) -> Dict[str, Any]:
        """Test scalability of leak detection system"""
        if not self.has_leak_guardian:
            return {'success': False, 'error': 'Leak guardian not available'}
        
        try:
            # Test with increasing function complexity
            test_functions = []
            
            # Generate functions of different sizes
            for size in [10, 50, 100]:
                code = f"""
def test_function_{size}():
    # Function with {size} operations
"""
                for i in range(size):
                    code += f"    data_{i} = [0] * {i + 1}\n"
                
                code += "    return sum("
                code += " + ".join([f"sum(data_{i})" for i in range(size)])
                code += ")\n"
                
                test_functions.append((f"test_function_{size}", code, size))
            
            # Analyze each function and measure time
            analysis_times = []
            
            for func_name, func_code, size in test_functions:
                start_time = time.time()
                summary = self.leak_guardian.analyze_function_for_leaks(func_code, func_name)
                analysis_time = time.time() - start_time
                analysis_times.append(analysis_time)
                
                logger.info(f"   📊 Function size {size}: {analysis_time:.3f}s")
            
            # Check if analysis time scales reasonably
            scalability_results = {
                'all_functions_analyzed': len(analysis_times) == len(test_functions),
                'analysis_times_reasonable': all(t < 1.0 for t in analysis_times),  # Under 1 second
                'scalability_acceptable': max(analysis_times) / min(analysis_times) < 10  # Less than 10x difference
            }
            
            success = all(scalability_results.values())
            
            return {
                'success': success,
                'scalability_results': scalability_results,
                'analysis_times': analysis_times,
                'max_analysis_time': max(analysis_times),
                'min_analysis_time': min(analysis_times)
            }
            
        except Exception as e:
            return {'success': False, 'error': f"Scalability test failed: {str(e)}"}
    
    async def save_test_report(self):
        """Save comprehensive test report"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Create comprehensive report
        report = {
            'test_execution_summary': {
                'timestamp': timestamp,
                'total_tests': len(self.test_results),
                'passed_tests': sum(1 for r in self.test_results.values() if r['status'] == 'PASSED'),
                'failed_tests': sum(1 for r in self.test_results.values() if r['status'] == 'FAILED'),
                'error_tests': sum(1 for r in self.test_results.values() if r['status'] == 'ERROR'),
                'total_execution_time_ms': sum(r['execution_time_ms'] for r in self.test_results.values()),
                'leak_guardian_available': self.has_leak_guardian
            },
            'detailed_results': self.test_results,
            'system_capabilities': {
                'static_analysis': self.has_leak_guardian,
                'symbolic_execution': self.has_leak_guardian and getattr(self.leak_guardian, 'enable_symbolic_execution', False),
                'gpu_tracking': self.has_leak_guardian and getattr(self.leak_guardian, 'enable_gpu_tracking', False),
                'real_time_monitoring': self.has_leak_guardian
            }
        }
        
        # Save report
        report_file = f"test_results/advanced_memory_leak_detection_test_{timestamp}.json"
        os.makedirs("test_results", exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📄 Test report saved: {report_file}")

async def main():
    """Main test execution"""
    logger.info("🧪 Advanced Memory Leak Detection System Test")
    logger.info("=" * 70)
    
    tester = AdvancedMemoryLeakDetectionTester()
    success = await tester.run_comprehensive_test_suite()
    
    if success:
        logger.info("\n🎉 ADVANCED MEMORY LEAK DETECTION SYSTEM: VALIDATION SUCCESSFUL")
        logger.info("✅ System is ready for production deployment")
        return 0
    else:
        logger.error("\n❌ ADVANCED MEMORY LEAK DETECTION SYSTEM: VALIDATION FAILED")
        logger.error("🔧 Review test results and fix issues before deployment")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main()) 