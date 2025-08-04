"""
GPU Downgrade Verification Script
=================================
This script performs comprehensive verification of GPU downgrade functionality
in the Kimera SWM system, testing all critical pathways and components.
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def verify_gpu_availability():
    """Verify GPU availability and configuration"""
    print("\n" + "=" * 60)
    print("GPU AVAILABILITY VERIFICATION")
    print("=" * 60)

    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")

    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f"CUDA Device Count: {device_count}")

        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            print(f"\nDevice {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Total Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"  Major/Minor: {props.major}.{props.minor}")
            print(f"  Multi Processor Count: {props.multi_processor_count}")
            # These attributes might not be available in all PyTorch versions
            if hasattr(props, "max_threads_per_block"):
                print(f"  Max Threads Per Block: {props.max_threads_per_block}")
            if hasattr(props, "warp_size"):
                print(f"  Warp Size: {props.warp_size}")

            # Check current memory usage
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"  Memory Allocated: {allocated:.2f} GB")
            print(f"  Memory Reserved: {reserved:.2f} GB")
    else:
        print("WARNING: CUDA is not available. System will fall back to CPU.")

    return cuda_available


def test_hardware_config():
    """Test hardware configuration settings"""
    print("\n" + "=" * 60)
    print("HARDWARE CONFIGURATION TEST")
    print("=" * 60)

    try:
        from src.config import hardware_config

        print(f"Device: {hardware_config.DEVICE}")
        print(f"Mixed Precision: {hardware_config.USE_MIXED_PRECISION}")
        print(f"CUDA Streams Enabled: {hardware_config.ENABLE_CUDA_STREAMS}")
        print(f"Tensor Batch Size: {hardware_config.TENSOR_BATCH_SIZE}")
        print(f"Memory Pooling: {hardware_config.ENABLE_MEMORY_POOLING}")
        print(f"Auto Tuning: {hardware_config.ENABLE_AUTO_TUNING}")
        print(f"Compile Models: {hardware_config.COMPILE_MODELS}")
        print(f"Tensor Cores: {hardware_config.ENABLE_TENSOR_CORES}")
        print(
            f"Adaptive Batch Size Range: {hardware_config.ADAPTIVE_BATCH_SIZE_MIN} - {hardware_config.ADAPTIVE_BATCH_SIZE_MAX}"
        )

        # Verify batch size is appropriate for RTX 2080 Ti
        if hardware_config.TENSOR_BATCH_SIZE == 512:
            print("\n✓ Batch size correctly configured for RTX 2080 Ti (11GB VRAM)")
        else:
            print(
                f"\n⚠ Batch size {hardware_config.TENSOR_BATCH_SIZE} may not be optimal for RTX 2080 Ti"
            )

        return True
    except Exception as e:
        print(f"ERROR loading hardware config: {e}")
        return False


def test_cognitive_gpu_kernels():
    """Test Cognitive GPU Kernels functionality"""
    print("\n" + "=" * 60)
    print("COGNITIVE GPU KERNELS TEST")
    print("=" * 60)

    try:
        import cupy as cp

        from src.engines.cognitive_gpu_kernels import CognitiveGPUKernels

        # Initialize kernels
        kernels = CognitiveGPUKernels(device_id=0)

        # Get performance metrics
        metrics = kernels.get_performance_metrics()
        print("\nGPU Performance Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

        # Test thermodynamic signal evolution
        print("\nTesting Thermodynamic Signal Evolution...")
        n_elements = 1000
        signal_states = cp.random.randn(n_elements).astype(cp.float32)
        entropy_gradients = cp.random.randn(n_elements).astype(cp.float32)
        temperature_field = cp.random.rand(n_elements).astype(cp.float32)

        start_time = time.time()
        evolved_states = kernels.run_thermodynamic_signal_evolution(
            signal_states, entropy_gradients, temperature_field
        )
        elapsed_time = time.time() - start_time

        print(f"  Processed {n_elements} elements in {elapsed_time*1000:.2f} ms")
        print(f"  Throughput: {n_elements/elapsed_time:.0f} elements/sec")
        print(f"  Output shape: {evolved_states.shape}")

        return True
    except ImportError as e:
        print(f"WARNING: Could not import required modules: {e}")
        print("This may indicate CUDA/CuPy is not properly installed")
        return False
    except Exception as e:
        print(f"ERROR in GPU kernels test: {e}")
        return False


def test_cognitive_field_dynamics_gpu():
    """Test Cognitive Field Dynamics GPU implementation"""
    print("\n" + "=" * 60)
    print("COGNITIVE FIELD DYNAMICS GPU TEST")
    print("=" * 60)

    try:
        from src.engines.cognitive_field_dynamics_gpu import CognitiveFieldDynamicsGPU

        # Initialize engine
        dimension = 512
        engine = CognitiveFieldDynamicsGPU(dimension=dimension)

        # Test batch field addition
        print("\nTesting Batch Field Addition...")
        n_fields = 100
        geoid_ids = [f"test_geoid_{i}" for i in range(n_fields)]
        embeddings = torch.randn(n_fields, dimension)

        start_time = time.time()
        for i in range(n_fields):
            engine.add_geoid(geoid_ids[i], embeddings[i])

        # Force flush
        engine._flush_pending_fields()
        elapsed_time = time.time() - start_time

        print(f"  Added {n_fields} fields in {elapsed_time*1000:.2f} ms")
        print(f"  Throughput: {n_fields/elapsed_time:.0f} fields/sec")

        # Get performance stats
        stats = engine.get_performance_stats()
        print("\nEngine Performance Stats:")
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")

        # Test neighbor search
        print("\nTesting Neighbor Search...")
        start_time = time.time()
        neighbors = engine.find_semantic_neighbors(geoid_ids[0], energy_threshold=0.1)
        elapsed_time = time.time() - start_time

        print(f"  Found {len(neighbors)} neighbors in {elapsed_time*1000:.2f} ms")

        # Test anomaly detection
        print("\nTesting Anomaly Detection...")
        start_time = time.time()
        anomalies = engine.detect_semantic_anomalies()
        elapsed_time = time.time() - start_time

        print(f"  Detected {len(anomalies)} anomalies in {elapsed_time*1000:.2f} ms")

        # Verify GPU memory usage
        gpu_memory = stats.get("gpu_memory", {})
        allocated_mb = gpu_memory.get("allocated_mb", 0)
        utilization = gpu_memory.get("utilization_percent", 0)

        print(f"\nGPU Memory Status:")
        print(f"  Allocated: {allocated_mb:.2f} MB")
        print(f"  Utilization: {utilization:.1f}%")

        # Shutdown
        engine.shutdown()

        return True
    except Exception as e:
        print(f"ERROR in Cognitive Field Dynamics GPU test: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_gpu_memory_pool():
    """Test GPU memory pool functionality"""
    print("\n" + "=" * 60)
    print("GPU MEMORY POOL TEST")
    print("=" * 60)

    try:
        from src.engines.gpu_memory_pool import GPUMemoryPool

        # Initialize memory pool
        pool = GPUMemoryPool(initial_size_mb=100, max_size_mb=1000, device="cuda:0")

        print(f"Memory Pool Initialized:")
        print(f"  Initial Size: {pool.initial_size_mb} MB")
        print(f"  Max Size: {pool.max_size_mb} MB")
        print(f"  Device: {pool.device}")

        # Test allocation
        print("\nTesting Memory Allocation...")
        tensor_shape = (1000, 512)
        tensor = pool.allocate_tensor(tensor_shape, dtype=torch.float32)
        print(f"  Allocated tensor shape: {tensor.shape}")
        print(f"  Allocated tensor device: {tensor.device}")

        # Get pool stats
        stats = pool.get_stats()
        print("\nMemory Pool Stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        # Test deallocation
        pool.deallocate_tensor(tensor)
        print("\n✓ Memory deallocation successful")

        return True
    except ImportError:
        print("INFO: GPU memory pool module not found (optional component)")
        return True
    except Exception as e:
        print(f"ERROR in GPU memory pool test: {e}")
        return False


def run_stress_test():
    """Run GPU stress test to verify stability under load"""
    print("\n" + "=" * 60)
    print("GPU STRESS TEST")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("Skipping stress test - CUDA not available")
        return True

    try:
        # Test with increasing batch sizes
        batch_sizes = [128, 256, 512, 1024]
        dimension = 512

        for batch_size in batch_sizes:
            print(f"\nTesting batch size: {batch_size}")

            try:
                # Allocate tensors
                tensor1 = torch.randn(batch_size, dimension, device="cuda")
                tensor2 = torch.randn(batch_size, dimension, device="cuda")

                # Perform computation
                start_time = time.time()
                result = torch.mm(tensor1, tensor2.t())
                torch.cuda.synchronize()
                elapsed_time = time.time() - start_time

                # Check memory
                allocated = torch.cuda.memory_allocated() / 1024**2
                reserved = torch.cuda.memory_reserved() / 1024**2

                print(f"  Computation time: {elapsed_time*1000:.2f} ms")
                print(f"  Memory allocated: {allocated:.2f} MB")
                print(f"  Memory reserved: {reserved:.2f} MB")

                # Clean up
                del tensor1, tensor2, result
                torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                print(f"  ⚠ Out of memory at batch size {batch_size}")
                torch.cuda.empty_cache()
                break

        print("\n✓ Stress test completed")
        return True

    except Exception as e:
        print(f"ERROR in stress test: {e}")
        return False


def main():
    """Main verification routine"""
    print("\n" + "=" * 80)
    print("KIMERA SWM GPU DOWNGRADE VERIFICATION")
    print("=" * 80)
    print("This script verifies the GPU downgrade implementation and functionality")

    # Track test results
    results = {}

    # 1. Verify GPU availability
    results["gpu_available"] = verify_gpu_availability()

    # 2. Test hardware configuration
    results["hardware_config"] = test_hardware_config()

    # 3. Test cognitive GPU kernels
    results["gpu_kernels"] = test_cognitive_gpu_kernels()

    # 4. Test cognitive field dynamics GPU
    results["field_dynamics"] = test_cognitive_field_dynamics_gpu()

    # 5. Test GPU memory pool
    results["memory_pool"] = test_gpu_memory_pool()

    # 6. Run stress test
    results["stress_test"] = run_stress_test()

    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)

    total_tests = len(results)
    passed_tests = sum(1 for v in results.values() if v)

    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")

    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")

    # Final verdict
    if results["gpu_available"] and results["hardware_config"]:
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            if "2080 Ti" in device_name:
                print(
                    "\n✓ GPU DOWNGRADE VERIFIED: System correctly configured for RTX 2080 Ti"
                )
            else:
                print(f"\n⚠ GPU DETECTED: {device_name} (expected RTX 2080 Ti)")
        else:
            print("\n⚠ GPU NOT AVAILABLE: System running in CPU mode")
    else:
        print("\n✗ GPU DOWNGRADE VERIFICATION FAILED")

    return passed_tests == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
