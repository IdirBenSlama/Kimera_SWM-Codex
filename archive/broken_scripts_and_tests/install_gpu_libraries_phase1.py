#!/usr/bin/env python3
"""
GPU Libraries Installation Script - Phase 1, Week 1
===================================================

Scientific installation of GPU-accelerated libraries following
the KIMERA Integration Master Plan with zeteic validation.

This script implements:
1. CuPy - GPU NumPy replacement
2. Rapids/CuDF - GPU data processing  
3. Enhanced PyTorch GPU environment
4. GPU monitoring utilities

Author: KIMERA Development Team
Version: 1.0.0 - Phase 1 Foundation
"""

import subprocess
import sys
import logging
import time
import importlib
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import torch

# Configure scientific logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [GPU Install] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class LibraryInstallation:
    """Track library installation with scientific precision"""
    name: str
    version: str
    cuda_version: str
    install_command: List[str]
    verification_import: str
    verification_test: Optional[str] = None
    installed: bool = False
    install_time: Optional[float] = None
    error_message: Optional[str] = None

class GPULibraryInstaller:
    """
    GPU Library Installation with Scientific Validation
    
    Implements Phase 1, Week 1 GPU Foundation requirements:
    - CuPy for GPU-accelerated array operations
    - Rapids/CuDF for GPU data processing
    - Enhanced GPU monitoring
    - Comprehensive validation testing
    """
    
    def __init__(self):
        """Initialize installer with current system analysis"""
        logger.info("🚀 GPU Library Installer initializing...")
        
        self.cuda_version = self._detect_cuda_version()
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        self.installations: Dict[str, LibraryInstallation] = {}
        
        logger.info(f"📋 System Analysis:")
        logger.info(f"   CUDA Version: {self.cuda_version}")
        logger.info(f"   Python Version: {self.python_version}")
        logger.info(f"   PyTorch Version: {torch.__version__}")
        
        self._define_installation_plan()
    
    def _detect_cuda_version(self) -> str:
        """Detect CUDA version with scientific precision"""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available - cannot install GPU libraries")
        
        cuda_version = torch.version.cuda
        logger.info(f"✅ CUDA {cuda_version} detected")
        return cuda_version
    
    def _define_installation_plan(self) -> None:
        """Define installation plan based on CUDA version"""
        logger.info("📋 Defining installation plan...")
        
        # Determine appropriate versions based on CUDA
        if self.cuda_version.startswith("11.8"):
            cupy_package = "cupy-cuda11x"
            torch_index = "https://download.pytorch.org/whl/cu118"
        elif self.cuda_version.startswith("12."):
            cupy_package = "cupy-cuda12x"
            torch_index = "https://download.pytorch.org/whl/cu121"
        else:
            logger.warning(f"⚠️ CUDA {self.cuda_version} may not be fully supported")
            cupy_package = "cupy-cuda11x"  # Fallback
            torch_index = "https://download.pytorch.org/whl/cu118"
        
        self.installations = {
            "cupy": LibraryInstallation(
                name="CuPy",
                version="latest",
                cuda_version=self.cuda_version,
                install_command=["pip", "install", cupy_package],
                verification_import="cupy",
                verification_test="import cupy as cp; x = cp.array([1, 2, 3]); assert x.device.id >= 0"
            ),
            
            "rapids-cudf": LibraryInstallation(
                name="Rapids CuDF",
                version="24.12",
                cuda_version=self.cuda_version,
                install_command=[
                    "pip", "install", 
                    "--extra-index-url", "https://pypi.nvidia.com",
                    "cudf-cu11" if self.cuda_version.startswith("11") else "cudf-cu12"
                ],
                verification_import="cudf",
                verification_test="import cudf; df = cudf.DataFrame({'x': [1, 2, 3]}); assert len(df) == 3"
            ),
            
            "gpu-monitoring": LibraryInstallation(
                name="GPU Monitoring",
                version="latest",
                cuda_version=self.cuda_version,
                install_command=["pip", "install", "pynvml", "gpustat"],
                verification_import="pynvml",
                verification_test="import pynvml; pynvml.nvmlInit(); assert pynvml.nvmlDeviceGetCount() > 0"
            ),
            
            "numba-cuda": LibraryInstallation(
                name="Numba CUDA",
                version="latest", 
                cuda_version=self.cuda_version,
                install_command=["pip", "install", "--upgrade", "numba"],
                verification_import="numba.cuda",
                verification_test="from numba import cuda; assert cuda.is_available()"
            )
        }
        
        logger.info(f"✅ Installation plan defined for {len(self.installations)} libraries")
    
    def install_all_libraries(self) -> Dict[str, bool]:
        """Install all GPU libraries with scientific validation"""
        logger.info("🔧 Beginning GPU library installation...")
        
        results = {}
        total_start_time = time.perf_counter()
        
        for lib_key, lib_info in self.installations.items():
            logger.info(f"📦 Installing {lib_info.name}...")
            
            try:
                install_start = time.perf_counter()
                
                # Execute installation command
                result = subprocess.run(
                    lib_info.install_command,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 minute timeout
                )
                
                install_time = time.perf_counter() - install_start
                lib_info.install_time = install_time
                
                logger.info(f"✅ {lib_info.name} installed in {install_time:.1f}s")
                
                # Verify installation
                if self._verify_installation(lib_info):
                    lib_info.installed = True
                    results[lib_key] = True
                    logger.info(f"✅ {lib_info.name} verification passed")
                else:
                    results[lib_key] = False
                    logger.error(f"❌ {lib_info.name} verification failed")
                
            except subprocess.TimeoutExpired:
                lib_info.error_message = "Installation timeout"
                results[lib_key] = False
                logger.error(f"❌ {lib_info.name} installation timed out")
                
            except subprocess.CalledProcessError as e:
                lib_info.error_message = f"Install error: {e.stderr}"
                results[lib_key] = False
                logger.error(f"❌ {lib_info.name} installation failed: {e.stderr}")
                
            except Exception as e:
                lib_info.error_message = str(e)
                results[lib_key] = False
                logger.error(f"❌ {lib_info.name} unexpected error: {e}")
        
        total_time = time.perf_counter() - total_start_time
        success_count = sum(results.values())
        
        logger.info(f"📊 Installation Summary:")
        logger.info(f"   Total Time: {total_time:.1f}s")
        logger.info(f"   Success Rate: {success_count}/{len(results)} libraries")
        
        return results
    
    def _verify_installation(self, lib_info: LibraryInstallation) -> bool:
        """Verify library installation with scientific rigor"""
        try:
            # Test import
            importlib.import_module(lib_info.verification_import)
            
            # Run verification test if provided
            if lib_info.verification_test:
                exec(lib_info.verification_test)
            
            return True
            
        except ImportError as e:
            logger.error(f"Import failed for {lib_info.name}: {e}")
            return False
        except Exception as e:
            logger.error(f"Verification test failed for {lib_info.name}: {e}")
            return False
    
    def run_comprehensive_tests(self) -> Dict[str, Dict]:
        """Run comprehensive tests on installed libraries"""
        logger.info("🧪 Running comprehensive GPU library tests...")
        
        test_results = {}
        
        # Test CuPy
        if self.installations["cupy"].installed:
            test_results["cupy"] = self._test_cupy()
        
        # Test CuDF
        if self.installations["rapids-cudf"].installed:
            test_results["cudf"] = self._test_cudf()
        
        # Test GPU monitoring
        if self.installations["gpu-monitoring"].installed:
            test_results["gpu_monitoring"] = self._test_gpu_monitoring()
        
        # Test Numba CUDA
        if self.installations["numba-cuda"].installed:
            test_results["numba_cuda"] = self._test_numba_cuda()
        
        return test_results
    
    def _test_cupy(self) -> Dict:
        """Test CuPy GPU operations"""
        logger.info("🔬 Testing CuPy GPU operations...")
        
        try:
            import cupy as cp
            
            # Basic array operations
            x = cp.random.random((1000, 1000))
            y = cp.random.random((1000, 1000))
            
            start_time = time.perf_counter()
            z = cp.matmul(x, y)
            cp.cuda.Stream.null.synchronize()
            compute_time = time.perf_counter() - start_time
            
            # Memory test
            memory_info = cp.cuda.runtime.memGetInfo()
            free_memory = memory_info[0] / 1e9
            total_memory = memory_info[1] / 1e9
            
            result = {
                "status": "success",
                "compute_time_ms": compute_time * 1000,
                "memory_free_gb": free_memory,
                "memory_total_gb": total_memory,
                "performance_rating": "excellent" if compute_time < 0.01 else "good"
            }
            
            logger.info(f"✅ CuPy test passed: {compute_time*1000:.2f}ms matmul")
            return result
            
        except Exception as e:
            logger.error(f"❌ CuPy test failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def _test_cudf(self) -> Dict:
        """Test CuDF GPU data processing"""
        logger.info("🔬 Testing CuDF GPU data operations...")
        
        try:
            import cudf
            import pandas as pd
            
            # Create test data
            data_size = 100000
            df = cudf.DataFrame({
                'x': range(data_size),
                'y': range(data_size, 2 * data_size),
                'category': ['A', 'B', 'C'] * (data_size // 3 + 1)
            })
            
            # Test operations
            start_time = time.perf_counter()
            result_df = df.groupby('category').agg({'x': 'mean', 'y': 'sum'})
            compute_time = time.perf_counter() - start_time
            
            return {
                "status": "success",
                "compute_time_ms": compute_time * 1000,
                "data_size": data_size,
                "performance_rating": "excellent" if compute_time < 0.1 else "good"
            }
            
        except ImportError:
            logger.warning("⚠️ CuDF not available - skipping test")
            return {"status": "skipped", "reason": "not_installed"}
        except Exception as e:
            logger.error(f"❌ CuDF test failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def _test_gpu_monitoring(self) -> Dict:
        """Test GPU monitoring capabilities"""
        logger.info("🔬 Testing GPU monitoring...")
        
        try:
            import pynvml
            
            # Initialize NVML
            pynvml.nvmlInit()
            
            # Get GPU info
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            return {
                "status": "success",
                "gpu_name": name,
                "memory_total_gb": memory_info.total / 1e9,
                "memory_free_gb": memory_info.free / 1e9,
                "memory_used_gb": memory_info.used / 1e9
            }
            
        except Exception as e:
            logger.error(f"❌ GPU monitoring test failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def _test_numba_cuda(self) -> Dict:
        """Test Numba CUDA compilation"""
        logger.info("🔬 Testing Numba CUDA...")
        
        try:
            from numba import cuda
            import numpy as np
            
            if not cuda.is_available():
                return {"status": "failed", "error": "CUDA not available to Numba"}
            
            @cuda.jit
            def add_kernel(x, y, out):
                idx = cuda.grid(1)
                if idx < out.size:
                    out[idx] = x[idx] + y[idx]
            
            # Test kernel
            n = 1000
            x = np.arange(n, dtype=np.float32)
            y = np.ones(n, dtype=np.float32)
            
            x_gpu = cuda.to_device(x)
            y_gpu = cuda.to_device(y)
            out_gpu = cuda.device_array(n, dtype=np.float32)
            
            threads_per_block = 256
            blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
            
            start_time = time.perf_counter()
            add_kernel[blocks_per_grid, threads_per_block](x_gpu, y_gpu, out_gpu)
            cuda.synchronize()
            compute_time = time.perf_counter() - start_time
            
            return {
                "status": "success",
                "compute_time_ms": compute_time * 1000,
                "kernel_compilation": "success"
            }
            
        except Exception as e:
            logger.error(f"❌ Numba CUDA test failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def generate_installation_report(self) -> str:
        """Generate comprehensive installation report"""
        report_lines = [
            "# GPU Libraries Installation Report",
            f"Generated: {datetime.now().isoformat()}",
            f"CUDA Version: {self.cuda_version}",
            f"Python Version: {self.python_version}",
            "",
            "## Installation Results"
        ]
        
        for lib_key, lib_info in self.installations.items():
            status = "✅ INSTALLED" if lib_info.installed else "❌ FAILED"
            install_time = f" ({lib_info.install_time:.1f}s)" if lib_info.install_time else ""
            
            report_lines.append(f"- **{lib_info.name}**: {status}{install_time}")
            
            if lib_info.error_message:
                report_lines.append(f"  - Error: {lib_info.error_message}")
        
        return "\n".join(report_lines)

def main():
    """Main installation function"""
    logger.info("🚀 KIMERA GPU Libraries Installation - Phase 1, Week 1")
    logger.info("=" * 60)
    
    try:
        installer = GPULibraryInstaller()
        
        # Install libraries
        results = installer.install_all_libraries()
        
        # Run tests
        test_results = installer.run_comprehensive_tests()
        
        # Generate report
        report = installer.generate_installation_report()
        logger.info("\n" + report)
        
        # Save report
        with open("gpu_installation_report.md", "w") as f:
            f.write(report)
        
        success_count = sum(results.values())
        if success_count == len(results):
            logger.info(f"\n🎉 All {success_count} GPU libraries installed successfully!")
            return 0
        else:
            logger.warning(f"\n⚠️ {success_count}/{len(results)
            return 1
            
    except Exception as e:
        logger.error(f"\n❌ Installation failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 