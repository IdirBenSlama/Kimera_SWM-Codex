#!/usr/bin/env python3
"""
GPU Foundation Performance Regimes Test
=======================================
"""

import logging
import time
import torch
import numpy as np
import json
from datetime import datetime
from backend.utils.gpu_foundation import GPUFoundation, GPUValidationLevel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_performance_regimes():
    """Test GPU performance at different regimes approaching maximum thresholds"""
    
    logger.info("🚀 GPU Performance Regimes Analysis")
    logger.info("=" * 50)
    
    # Initialize GPU Foundation
    gpu_foundation = GPUFoundation(GPUValidationLevel.RIGOROUS)
    caps = gpu_foundation.capabilities
    max_memory_gb = caps.total_memory_gb
    
    logger.info(f"🖥️  GPU: {caps.device_name}")
    logger.info(f"💾 Max Memory: {max_memory_gb:.1f} GB")
    logger.info(f"🔒 Max Safe Allocation: 80% ({max_memory_gb * 0.8:.1f} GB)
    logger.info()
    
    results = []
    
    # Test 1: Memory Allocation Regimes
    logger.debug("🔬 Testing Memory Allocation Regimes")
    logger.info("-" * 40)
    
    memory_levels = [
        ("Conservative", 0.10),
        ("Light", 0.25),
        ("Moderate", 0.40),
        ("Heavy", 0.60),
        ("Maximum", 0.80)  # 80% threshold
    ]
    
    for level_name, allocation_ratio in memory_levels:
        try:
            allocation_gb = max_memory_gb * allocation_ratio
            allocation_bytes = int(allocation_gb * 1024**3)
            
            # Assess cognitive stability before
            stability_before = gpu_foundation.assess_cognitive_stability()
            
            start_time = time.time()
            
            # Allocate and test
            device = torch.device('cuda')
            tensor_size = allocation_bytes // 4  # 4 bytes per float32
            test_tensor = torch.randn(tensor_size, device=device)
            test_tensor = test_tensor * 2.0
            torch.cuda.synchronize()
            
            duration = time.time() - start_time
            
            # Get actual memory usage
            memory_used = torch.cuda.memory_allocated() / (1024**3)
            memory_percent = (memory_used / max_memory_gb) * 100
            
            # Assess cognitive stability after
            stability_after = gpu_foundation.assess_cognitive_stability()
            
            # Calculate bandwidth
            bandwidth = (allocation_bytes * 2 / (1024**3)) / duration  # GB/s
            
            stability_ok = (
                stability_after.identity_coherence_score >= 0.95 and
                stability_after.memory_continuity_score >= 0.98 and
                stability_after.cognitive_drift_magnitude <= 0.02
            )
            
            success = stability_ok and memory_percent <= 85
            
            result = {
                "test": "Memory Allocation",
                "level": level_name,
                "target_gb": allocation_gb,
                "actual_gb": memory_used,
                "percent": memory_percent,
                "duration_ms": duration * 1000,
                "bandwidth_gb_s": bandwidth,
                "cognitive_stability": stability_ok,
                "success": success
            }
            
            results.append(result)
            
            status = "✅" if success else "❌"
            logger.info(f"{status} {level_name:12} | {memory_used:5.1f} GB ({memory_percent:4.1f}%)
            
            # Clean up
            del test_tensor
            torch.cuda.empty_cache()
            time.sleep(0.5)
            
        except Exception as e:
            logger.error(f"❌ {level_name:12} | FAILED: {str(e)
            results.append({"test": "Memory Allocation", "level": level_name, "success": False, "error": str(e)})
    
    logger.info()
    
    # Test 2: Computational Intensity Regimes
    logger.debug("🔬 Testing Computational Intensity Regimes")
    logger.info("-" * 40)
    
    intensity_levels = [
        ("Light", 1024, 50),
        ("Moderate", 2048, 100),
        ("Heavy", 4096, 150),
        ("Extreme", 6144, 200),
        ("Maximum", 8192, 250)
    ]
    
    for level_name, matrix_size, iterations in intensity_levels:
        try:
            stability_before = gpu_foundation.assess_cognitive_stability()
            
            start_time = time.time()
            
            device = torch.device('cuda')
            A = torch.randn(matrix_size, matrix_size, device=device)
            B = torch.randn(matrix_size, matrix_size, device=device)
            
            total_ops = 0
            for i in range(iterations):
                C = torch.matmul(A, B)
                C = torch.sin(C)
                result = torch.sum(C)
                total_ops += matrix_size * matrix_size * 2
            
            torch.cuda.synchronize()
            duration = time.time() - start_time
            
            # Performance metrics
            gflops = (total_ops / 1e9) / duration
            memory_used = torch.cuda.memory_allocated() / (1024**3)
            
            stability_after = gpu_foundation.assess_cognitive_stability()
            stability_ok = (
                stability_after.identity_coherence_score >= 0.95 and
                stability_after.memory_continuity_score >= 0.98 and
                stability_after.cognitive_drift_magnitude <= 0.02
            )
            
            success = stability_ok and gflops > 50
            
            result = {
                "test": "Computational Intensity",
                "level": level_name,
                "matrix_size": matrix_size,
                "iterations": iterations,
                "duration_ms": duration * 1000,
                "gflops": gflops,
                "memory_gb": memory_used,
                "cognitive_stability": stability_ok,
                "success": success
            }
            
            results.append(result)
            
            status = "✅" if success else "❌"
            logger.info(f"{status} {level_name:12} | {matrix_size:4}x{matrix_size} x{iterations:3} | {gflops:6.1f} GFLOPS | {memory_used:4.1f} GB | Stability: {stability_ok}")
            
            # Clean up
            del A, B, C
            torch.cuda.empty_cache()
            time.sleep(0.5)
            
        except Exception as e:
            logger.error(f"❌ {level_name:12} | FAILED: {str(e)
            results.append({"test": "Computational Intensity", "level": level_name, "success": False, "error": str(e)})
    
    logger.info()
    
    # Test 3: Concurrent Processing Regimes
    logger.debug("🔬 Testing Concurrent Processing Regimes")
    logger.info("-" * 40)
    
    concurrency_levels = [
        ("Single", 1),
        ("Dual", 2),
        ("Quad", 4),
        ("Octal", 8),
        ("Maximum", 16)
    ]
    
    for level_name, num_streams in concurrency_levels:
        try:
            stability_before = gpu_foundation.assess_cognitive_stability()
            
            start_time = time.time()
            
            # Create streams and work
            streams = [torch.cuda.Stream() for _ in range(num_streams)]
            device = torch.device('cuda')
            results_tensors = []
            
            # Launch concurrent operations
            for i, stream in enumerate(streams):
                with torch.cuda.stream(stream):
                    A = torch.randn(1024, 1024, device=device)
                    B = torch.randn(1024, 1024, device=device)
                    C = torch.matmul(A, B)
                    result = torch.sum(C)
                    results_tensors.append(result)
            
            # Synchronize all streams
            for stream in streams:
                stream.synchronize()
            
            duration = time.time() - start_time
            
            # Performance metrics
            total_ops = num_streams * 1024 * 1024 * 2  # Approximate ops
            gflops = (total_ops / 1e9) / duration
            memory_used = torch.cuda.memory_allocated() / (1024**3)
            efficiency = gflops / num_streams
            
            stability_after = gpu_foundation.assess_cognitive_stability()
            stability_ok = (
                stability_after.identity_coherence_score >= 0.95 and
                stability_after.memory_continuity_score >= 0.98 and
                stability_after.cognitive_drift_magnitude <= 0.02
            )
            
            success = stability_ok and len(results_tensors) == num_streams
            
            result = {
                "test": "Concurrent Processing",
                "level": level_name,
                "num_streams": num_streams,
                "duration_ms": duration * 1000,
                "gflops": gflops,
                "efficiency": efficiency,
                "memory_gb": memory_used,
                "cognitive_stability": stability_ok,
                "success": success
            }
            
            results.append(result)
            
            status = "✅" if success else "❌"
            logger.info(f"{status} {level_name:12} | {num_streams:2} streams | {gflops:6.1f} GFLOPS | Eff: {efficiency:5.1f} | Stability: {stability_ok}")
            
            # Clean up
            torch.cuda.empty_cache()
            time.sleep(0.5)
            
        except Exception as e:
            logger.error(f"❌ {level_name:12} | FAILED: {str(e)
            results.append({"test": "Concurrent Processing", "level": level_name, "success": False, "error": str(e)})
    
    logger.info()
    
    # Test 4: Thermal Stress Regimes
    logger.debug("🔬 Testing Thermal Stress Regimes")
    logger.info("-" * 40)
    
    stress_levels = [
        ("Burst", 5),
        ("Short", 15),
        ("Medium", 30),
        ("Extended", 60),
        ("Maximum", 120)
    ]
    
    for level_name, duration_seconds in stress_levels:
        try:
            stability_before = gpu_foundation.assess_cognitive_stability()
            
            logger.info(f"   🔥 Running {level_name} stress test ({duration_seconds}s)
            
            start_time = time.time()
            end_time = start_time + duration_seconds
            
            device = torch.device('cuda')
            matrix_size = 2048
            iterations = 0
            total_ops = 0
            
            while time.time() < end_time:
                A = torch.randn(matrix_size, matrix_size, device=device)
                B = torch.randn(matrix_size, matrix_size, device=device)
                C = torch.matmul(A, B)
                C = torch.sin(C) + torch.cos(C)
                result = torch.sum(C)
                
                iterations += 1
                total_ops += matrix_size * matrix_size * 3
                
                if iterations % 5 == 0:
                    torch.cuda.synchronize()
                    time.sleep(0.01)  # Brief pause
            
            actual_duration = time.time() - start_time
            torch.cuda.synchronize()
            
            # Performance metrics
            avg_gflops = (total_ops / 1e9) / actual_duration
            memory_used = torch.cuda.memory_allocated() / (1024**3)
            
            stability_after = gpu_foundation.assess_cognitive_stability()
            stability_ok = (
                stability_after.identity_coherence_score >= 0.95 and
                stability_after.memory_continuity_score >= 0.98 and
                stability_after.cognitive_drift_magnitude <= 0.02
            )
            
            success = stability_ok and iterations > duration_seconds
            
            result = {
                "test": "Thermal Stress",
                "level": level_name,
                "target_duration_s": duration_seconds,
                "actual_duration_s": actual_duration,
                "iterations": iterations,
                "avg_gflops": avg_gflops,
                "memory_gb": memory_used,
                "cognitive_stability": stability_ok,
                "success": success
            }
            
            results.append(result)
            
            status = "✅" if success else "❌"
            logger.info(f"{status} {level_name:12} | {actual_duration:5.1f}s | {iterations:3} iter | {avg_gflops:6.1f} GFLOPS | Stability: {stability_ok}")
            
            # Cool down
            torch.cuda.empty_cache()
            cooldown = min(duration_seconds * 0.1, 5)
            logger.info(f"   ❄️  Cooling down {cooldown:.1f}s...")
            time.sleep(cooldown)
            
        except Exception as e:
            logger.error(f"❌ {level_name:12} | FAILED: {str(e)
            results.append({"test": "Thermal Stress", "level": level_name, "success": False, "error": str(e)})
    
    # Calculate overall results
    total_tests = len(results)
    successful_tests = sum(1 for r in results if r.get('success', False))
    success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
    
    logger.info()
    logger.info("=" * 50)
    logger.info(f"📊 PERFORMANCE REGIMES SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Success Rate: {success_rate:.1f}% ({successful_tests}/{total_tests})
    logger.info(f"GPU Device: {caps.device_name}")
    logger.info(f"Max Memory: {max_memory_gb:.1f} GB")
    logger.info(f"Max Threshold: 80% ({max_memory_gb * 0.8:.1f} GB)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"gpu_performance_regimes_{timestamp}.json"
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "gpu_device": caps.device_name,
        "max_memory_gb": max_memory_gb,
        "success_rate": success_rate,
        "total_tests": total_tests,
        "successful_tests": successful_tests,
        "results": results
    }
    
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"📄 Results saved to: {results_file}")
    
    return success_rate >= 75

if __name__ == "__main__":
    success = test_performance_regimes()
    exit(0 if success else 1) 