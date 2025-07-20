#!/usr/bin/env python3
"""
GPU Foundation Performance Regimes Analysis
==========================================

Comprehensive performance testing across different operational regimes,
pushing toward maximum allowed thresholds for the KIMERA GPU Foundation.

This test explores:
1. Memory allocation regimes (10% -> 80% max threshold)
2. Computational intensity regimes (light -> extreme)
3. Concurrent processing regimes (1 -> 16 streams)
4. Thermal stress regimes (burst -> sustained)
5. Cognitive stability under extreme loads

Author: KIMERA Development Team
Version: 1.0.0 - Performance Regime Analysis
"""

import logging
import time
import torch
import numpy as np
import json
from datetime import datetime
from typing import Dict, Any, List, Tuple
from backend.utils.gpu_foundation import GPUFoundation, GPUValidationLevel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [Performance] %(message)s'
)
logger = logging.getLogger(__name__)

class GPUPerformanceRegimeAnalyzer:
    """Comprehensive GPU performance analysis across operational regimes"""
    
    def __init__(self):
        self.gpu_foundation = GPUFoundation(GPUValidationLevel.RIGOROUS)
        self.results = []
        self.start_time = time.time()
        
        # Get GPU capabilities for threshold calculations
        self.caps = self.gpu_foundation.capabilities
        self.max_memory_gb = self.caps.total_memory_gb
        self.max_allocation_threshold = 0.80  # 80% max safe allocation
        
        logger.info(f"🚀 Initializing Performance Regime Analyzer")
        logger.info(f"📊 GPU: {self.caps.device_name}")
        logger.info(f"💾 Max Memory: {self.max_memory_gb:.1f} GB")
        logger.info(f"🔒 Max Allocation Threshold: {self.max_allocation_threshold*100:.0f}%")
    
    def log_regime_result(self, regime_name: str, regime_level: str, metrics: Dict[str, Any], 
                         cognitive_stability: Dict[str, Any], success: bool):
        """Log performance regime test result"""
        result = {
            "regime_name": regime_name,
            "regime_level": regime_level,
            "timestamp": datetime.now().isoformat(),
            "success": success,
            "metrics": metrics,
            "cognitive_stability": cognitive_stability
        }
        self.results.append(result)
        
        status = "✅ SUCCESS" if success else "❌ FAILED"
        logger.info(f"{status} - {regime_name} [{regime_level}]")
        
        # Log key metrics
        if 'memory_usage_gb' in metrics:
            logger.info(f"   💾 Memory: {metrics['memory_usage_gb']:.2f} GB ({metrics.get('memory_usage_percent', 0):.1f}%)")
        if 'throughput_gflops' in metrics:
            logger.info(f"   🚀 Throughput: {metrics['throughput_gflops']:.2f} GFLOPS")
        if 'processing_time_ms' in metrics:
            logger.info(f"   ⏱️  Processing: {metrics['processing_time_ms']:.2f} ms")
        
        # Log cognitive stability
        stability = cognitive_stability
        logger.info(f"   🧠 Stability: Identity={stability.get('identity_coherence', 0):.3f}, "
                   f"Memory={stability.get('memory_continuity', 0):.3f}, "
                   f"Drift={stability.get('cognitive_drift', 0):.3f}")
    
    def test_memory_allocation_regimes(self) -> List[Dict[str, Any]]:
        """Test memory allocation at different regimes approaching maximum threshold"""
        logger.info("🔬 Testing Memory Allocation Regimes")
        
        # Define memory allocation levels (percentage of total GPU memory)
        allocation_levels = [
            ("Conservative", 0.10),  # 10%
            ("Light", 0.25),         # 25%
            ("Moderate", 0.40),      # 40%
            ("Heavy", 0.60),         # 60%
            ("Maximum", 0.80)        # 80% - maximum safe threshold
        ]
        
        regime_results = []
        
        for level_name, allocation_ratio in allocation_levels:
            try:
                # Calculate allocation size
                allocation_gb = self.max_memory_gb * allocation_ratio
                allocation_bytes = int(allocation_gb * 1024**3)
                
                # Assess cognitive stability before allocation
                stability_before = self.gpu_foundation.assess_cognitive_stability()
                
                start_time = time.time()
                
                # Allocate GPU memory
                device = torch.device('cuda')
                tensor_size = allocation_bytes // 4  # 4 bytes per float32
                test_tensor = torch.randn(tensor_size, device=device, dtype=torch.float32)
                
                # Perform memory operations
                test_tensor = test_tensor * 2.0
                test_tensor = torch.sin(test_tensor)
                torch.cuda.synchronize()
                
                processing_time = (time.time() - start_time) * 1000  # ms
                
                # Get memory usage
                memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
                memory_reserved = torch.cuda.memory_reserved() / (1024**3)    # GB
                memory_usage_percent = (memory_allocated / self.max_memory_gb) * 100
                
                # Assess cognitive stability after allocation
                stability_after = self.gpu_foundation.assess_cognitive_stability()
                
                # Calculate memory bandwidth
                data_transferred = allocation_bytes * 3  # read + write + operation
                bandwidth_gb_s = (data_transferred / (1024**3)) / (processing_time / 1000)
                
                metrics = {
                    "allocation_target_gb": allocation_gb,
                    "memory_usage_gb": memory_allocated,
                    "memory_reserved_gb": memory_reserved,
                    "memory_usage_percent": memory_usage_percent,
                    "processing_time_ms": processing_time,
                    "memory_bandwidth_gb_s": bandwidth_gb_s,
                    "tensor_elements": tensor_size
                }
                
                stability_metrics = {
                    "identity_coherence": stability_after.identity_coherence_score,
                    "memory_continuity": stability_after.memory_continuity_score,
                    "cognitive_drift": stability_after.cognitive_drift_magnitude,
                    "reality_testing": stability_after.reality_testing_score,
                    "stability_maintained": (
                        stability_after.identity_coherence_score >= 0.95 and
                        stability_after.memory_continuity_score >= 0.98 and
                        stability_after.cognitive_drift_magnitude <= 0.02
                    )
                }
                
                success = stability_metrics["stability_maintained"] and memory_usage_percent <= 85
                
                self.log_regime_result("Memory Allocation", level_name, metrics, stability_metrics, success)
                regime_results.append({
                    "level": level_name,
                    "success": success,
                    "metrics": metrics,
                    "stability": stability_metrics
                })
                
                # Clean up
                del test_tensor
                torch.cuda.empty_cache()
                time.sleep(1)  # Brief pause between tests
                
            except Exception as e:
                logger.error(f"❌ Memory allocation failed at {level_name}: {str(e)}")
                regime_results.append({
                    "level": level_name,
                    "success": False,
                    "error": str(e)
                })
        
        return regime_results
    
    def test_computational_intensity_regimes(self) -> List[Dict[str, Any]]:
        """Test computational intensity at different regimes"""
        logger.info("🔬 Testing Computational Intensity Regimes")
        
        # Define computational intensity levels
        intensity_levels = [
            ("Light", 1024, 100),      # 1K x 1K, 100 iterations
            ("Moderate", 2048, 200),   # 2K x 2K, 200 iterations
            ("Heavy", 4096, 300),      # 4K x 4K, 300 iterations
            ("Extreme", 6144, 400),    # 6K x 6K, 400 iterations
            ("Maximum", 8192, 500)     # 8K x 8K, 500 iterations - maximum intensity
        ]
        
        regime_results = []
        
        for level_name, matrix_size, iterations in intensity_levels:
            try:
                stability_before = self.gpu_foundation.assess_cognitive_stability()
                
                start_time = time.time()
                
                # Create test matrices
                device = torch.device('cuda')
                A = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float32)
                B = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float32)
                
                # Perform intensive computations
                total_operations = 0
                for i in range(iterations):
                    # Matrix multiplication
                    C = torch.matmul(A, B)
                    # Element-wise operations
                    C = torch.sin(C) * torch.cos(C)
                    # Reduction operations
                    result = torch.sum(C)
                    total_operations += matrix_size * matrix_size * 3  # Approximate ops
                
                torch.cuda.synchronize()
                processing_time = (time.time() - start_time) * 1000  # ms
                
                # Calculate performance metrics
                total_flops = total_operations * iterations
                throughput_gflops = (total_flops / 1e9) / (processing_time / 1000)
                
                # Memory usage
                memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
                memory_usage_percent = (memory_allocated / self.max_memory_gb) * 100
                
                # Assess cognitive stability after computation
                stability_after = self.gpu_foundation.assess_cognitive_stability()
                
                metrics = {
                    "matrix_size": matrix_size,
                    "iterations": iterations,
                    "total_operations": total_operations * iterations,
                    "processing_time_ms": processing_time,
                    "throughput_gflops": throughput_gflops,
                    "memory_usage_gb": memory_allocated,
                    "memory_usage_percent": memory_usage_percent,
                    "operations_per_second": (total_operations * iterations) / (processing_time / 1000)
                }
                
                stability_metrics = {
                    "identity_coherence": stability_after.identity_coherence_score,
                    "memory_continuity": stability_after.memory_continuity_score,
                    "cognitive_drift": stability_after.cognitive_drift_magnitude,
                    "reality_testing": stability_after.reality_testing_score,
                    "stability_maintained": (
                        stability_after.identity_coherence_score >= 0.95 and
                        stability_after.memory_continuity_score >= 0.98 and
                        stability_after.cognitive_drift_magnitude <= 0.02
                    )
                }
                
                success = stability_metrics["stability_maintained"] and throughput_gflops > 100
                
                self.log_regime_result("Computational Intensity", level_name, metrics, stability_metrics, success)
                regime_results.append({
                    "level": level_name,
                    "success": success,
                    "metrics": metrics,
                    "stability": stability_metrics
                })
                
                # Clean up
                del A, B, C
                torch.cuda.empty_cache()
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ Computational intensity failed at {level_name}: {str(e)}")
                regime_results.append({
                    "level": level_name,
                    "success": False,
                    "error": str(e)
                })
        
        return regime_results
    
    def test_concurrent_processing_regimes(self) -> List[Dict[str, Any]]:
        """Test concurrent processing with multiple CUDA streams"""
        logger.info("🔬 Testing Concurrent Processing Regimes")
        
        # Define concurrency levels
        concurrency_levels = [
            ("Single", 1),
            ("Dual", 2),
            ("Quad", 4),
            ("Octal", 8),
            ("Maximum", 16)  # Maximum concurrent streams
        ]
        
        regime_results = []
        
        for level_name, num_streams in concurrency_levels:
            try:
                stability_before = self.gpu_foundation.assess_cognitive_stability()
                
                start_time = time.time()
                
                # Create CUDA streams
                streams = [torch.cuda.Stream() for _ in range(num_streams)]
                
                # Create work for each stream
                device = torch.device('cuda')
                matrix_size = 2048
                results = []
                
                # Launch concurrent operations
                for i, stream in enumerate(streams):
                    with torch.cuda.stream(stream):
                        A = torch.randn(matrix_size, matrix_size, device=device)
                        B = torch.randn(matrix_size, matrix_size, device=device)
                        C = torch.matmul(A, B)
                        C = torch.sin(C) + torch.cos(C)
                        result = torch.sum(C)
                        results.append(result)
                
                # Synchronize all streams
                for stream in streams:
                    stream.synchronize()
                
                processing_time = (time.time() - start_time) * 1000  # ms
                
                # Calculate performance metrics
                total_operations = num_streams * matrix_size * matrix_size * 4  # Approximate ops per stream
                throughput_gflops = (total_operations / 1e9) / (processing_time / 1000)
                
                # Memory usage
                memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
                memory_usage_percent = (memory_allocated / self.max_memory_gb) * 100
                
                # Assess cognitive stability
                stability_after = self.gpu_foundation.assess_cognitive_stability()
                
                metrics = {
                    "num_streams": num_streams,
                    "matrix_size": matrix_size,
                    "total_operations": total_operations,
                    "processing_time_ms": processing_time,
                    "throughput_gflops": throughput_gflops,
                    "memory_usage_gb": memory_allocated,
                    "memory_usage_percent": memory_usage_percent,
                    "concurrent_efficiency": throughput_gflops / num_streams,
                    "operations_per_stream": total_operations / num_streams
                }
                
                stability_metrics = {
                    "identity_coherence": stability_after.identity_coherence_score,
                    "memory_continuity": stability_after.memory_continuity_score,
                    "cognitive_drift": stability_after.cognitive_drift_magnitude,
                    "reality_testing": stability_after.reality_testing_score,
                    "stability_maintained": (
                        stability_after.identity_coherence_score >= 0.95 and
                        stability_after.memory_continuity_score >= 0.98 and
                        stability_after.cognitive_drift_magnitude <= 0.02
                    )
                }
                
                success = stability_metrics["stability_maintained"] and len(results) == num_streams
                
                self.log_regime_result("Concurrent Processing", level_name, metrics, stability_metrics, success)
                regime_results.append({
                    "level": level_name,
                    "success": success,
                    "metrics": metrics,
                    "stability": stability_metrics
                })
                
                # Clean up
                torch.cuda.empty_cache()
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ Concurrent processing failed at {level_name}: {str(e)}")
                regime_results.append({
                    "level": level_name,
                    "success": False,
                    "error": str(e)
                })
        
        return regime_results
    
    def test_thermal_stress_regimes(self) -> List[Dict[str, Any]]:
        """Test thermal stress regimes with sustained load"""
        logger.info("🔬 Testing Thermal Stress Regimes")
        
        # Define thermal stress levels (duration in seconds)
        stress_levels = [
            ("Burst", 5),      # 5 seconds
            ("Short", 15),     # 15 seconds  
            ("Medium", 30),    # 30 seconds
            ("Long", 60),      # 60 seconds
            ("Maximum", 120)   # 120 seconds - maximum sustained load
        ]
        
        regime_results = []
        
        for level_name, duration_seconds in stress_levels:
            try:
                stability_before = self.gpu_foundation.assess_cognitive_stability()
                
                logger.info(f"   🔥 Starting {level_name} thermal stress test ({duration_seconds}s)")
                
                start_time = time.time()
                end_time = start_time + duration_seconds
                
                # Sustained computational load
                device = torch.device('cuda')
                matrix_size = 3072  # Moderate size for sustained load
                
                iteration_count = 0
                total_operations = 0
                
                while time.time() < end_time:
                    # Create matrices
                    A = torch.randn(matrix_size, matrix_size, device=device)
                    B = torch.randn(matrix_size, matrix_size, device=device)
                    
                    # Intensive computation
                    C = torch.matmul(A, B)
                    C = torch.sin(C) * torch.cos(C) + torch.exp(C * 0.01)
                    result = torch.sum(C)
                    
                    iteration_count += 1
                    total_operations += matrix_size * matrix_size * 5  # Approximate ops
                    
                    # Brief pause to prevent system overload
                    if iteration_count % 10 == 0:
                        torch.cuda.synchronize()
                        time.sleep(0.01)  # 10ms pause every 10 iterations
                
                actual_duration = time.time() - start_time
                
                # Final synchronization
                torch.cuda.synchronize()
                
                # Calculate performance metrics
                avg_throughput_gflops = (total_operations / 1e9) / actual_duration
                
                # Memory usage
                memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
                memory_usage_percent = (memory_allocated / self.max_memory_gb) * 100
                
                # Assess cognitive stability after stress
                stability_after = self.gpu_foundation.assess_cognitive_stability()
                
                metrics = {
                    "target_duration_s": duration_seconds,
                    "actual_duration_s": actual_duration,
                    "iterations_completed": iteration_count,
                    "total_operations": total_operations,
                    "avg_throughput_gflops": avg_throughput_gflops,
                    "memory_usage_gb": memory_allocated,
                    "memory_usage_percent": memory_usage_percent,
                    "operations_per_second": total_operations / actual_duration,
                    "thermal_efficiency": avg_throughput_gflops / duration_seconds
                }
                
                stability_metrics = {
                    "identity_coherence": stability_after.identity_coherence_score,
                    "memory_continuity": stability_after.memory_continuity_score,
                    "cognitive_drift": stability_after.cognitive_drift_magnitude,
                    "reality_testing": stability_after.reality_testing_score,
                    "stability_maintained": (
                        stability_after.identity_coherence_score >= 0.95 and
                        stability_after.memory_continuity_score >= 0.98 and
                        stability_after.cognitive_drift_magnitude <= 0.02
                    )
                }
                
                success = (stability_metrics["stability_maintained"] and 
                          iteration_count > duration_seconds * 2)  # At least 2 iterations per second
                
                self.log_regime_result("Thermal Stress", level_name, metrics, stability_metrics, success)
                regime_results.append({
                    "level": level_name,
                    "success": success,
                    "metrics": metrics,
                    "stability": stability_metrics
                })
                
                # Clean up and cool down
                torch.cuda.empty_cache()
                logger.info(f"   ❄️  Cooling down after {level_name} stress test...")
                time.sleep(min(duration_seconds * 0.1, 10))  # Cool down period
                
            except Exception as e:
                logger.error(f"❌ Thermal stress failed at {level_name}: {str(e)}")
                regime_results.append({
                    "level": level_name,
                    "success": False,
                    "error": str(e)
                })
        
        return regime_results
    
    def run_comprehensive_regime_analysis(self) -> Dict[str, Any]:
        """Run comprehensive performance regime analysis"""
        logger.info("🚀 Starting Comprehensive GPU Performance Regime Analysis")
        logger.info("=" * 80)
        
        # Run all regime tests
        memory_results = self.test_memory_allocation_regimes()
        computation_results = self.test_computational_intensity_regimes()
        concurrency_results = self.test_concurrent_processing_regimes()
        thermal_results = self.test_thermal_stress_regimes()
        
        # Calculate overall statistics
        total_duration = time.time() - self.start_time
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r['success'])
        success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
        
        # Determine overall performance grade
        if success_rate >= 90:
            grade = "EXCELLENT"
        elif success_rate >= 75:
            grade = "GOOD"
        elif success_rate >= 60:
            grade = "ACCEPTABLE"
        else:
            grade = "NEEDS_IMPROVEMENT"
        
        # Compile comprehensive report
        report = {
            "timestamp": datetime.now().isoformat(),
            "gpu_device": self.caps.device_name,
            "total_memory_gb": self.max_memory_gb,
            "max_allocation_threshold": self.max_allocation_threshold,
            "analysis_summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "failed_tests": total_tests - successful_tests,
                "success_rate": round(success_rate, 2),
                "total_duration_seconds": round(total_duration, 3),
                "performance_grade": grade
            },
            "regime_results": {
                "memory_allocation": memory_results,
                "computational_intensity": computation_results,
                "concurrent_processing": concurrency_results,
                "thermal_stress": thermal_results
            },
            "detailed_results": self.results
        }
        
        # Log summary
        logger.info("=" * 80)
        logger.info("🎯 GPU PERFORMANCE REGIME ANALYSIS SUMMARY")
        logger.info("=" * 80)
        logger.info(f"📊 Success Rate: {success_rate:.1f}% ({successful_tests}/{total_tests} tests passed)")
        logger.info(f"⏱️  Total Duration: {total_duration:.1f} seconds")
        logger.info(f"🏆 Performance Grade: {grade}")
        logger.info(f"🖥️  GPU Device: {self.caps.device_name}")
        logger.info(f"💾 Memory Threshold: {self.max_allocation_threshold*100:.0f}% ({self.max_memory_gb*self.max_allocation_threshold:.1f} GB)")
        logger.info("=" * 80)
        
        return report

def main():
    """Main execution function"""
    logger.debug("🔬 GPU Foundation Performance Regime Analysis")
    logger.info("=" * 60)
    
    try:
        # Initialize analyzer
        analyzer = GPUPerformanceRegimeAnalyzer()
        
        # Run comprehensive analysis
        results = analyzer.run_comprehensive_regime_analysis()
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"gpu_performance_regimes_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n📄 Detailed results saved to: {results_file}")
        
        # Return success based on overall performance
        success_rate = results['analysis_summary']['success_rate']
        if success_rate >= 75:
            logger.info("✅ Performance regime analysis PASSED")
            return 0
        else:
            logger.error("❌ Performance regime analysis FAILED")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Performance regime analysis failed: {str(e)}")
        logger.error(f"❌ Analysis failed: {str(e)
        return 1

if __name__ == "__main__":
    exit(main()) 