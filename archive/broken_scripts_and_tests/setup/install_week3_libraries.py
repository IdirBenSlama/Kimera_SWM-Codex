"""
KIMERA Week 3 Library Installation Script
=========================================
Phase 1, Week 3: Advanced GPU Computing Libraries

This script installs the required libraries for:
- Numba CUDA kernel development
- Triton kernel implementation
- CuGraph integration

Author: KIMERA Team
Date: June 2025
"""

import subprocess
import sys
import os
import platform
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_command(command, description):
    """Run a shell command and handle errors"""
    logger.info(f"Installing {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"✓ {description} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Failed to install {description}")
        logger.error(f"Error: {e.stderr}")
        return False


def check_cuda():
    """Check if CUDA is available"""
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"✓ CUDA is available: {torch.cuda.get_device_name(0)}")
            logger.info(f"  CUDA version: {torch.version.cuda}")
            return True
        else:
            logger.error("✗ CUDA is not available")
            return False
    except ImportError:
        logger.error("✗ PyTorch not installed")
        return False


def install_week3_libraries():
    """Install all Week 3 libraries"""
    logger.info("=== KIMERA Week 3 Library Installation ===")
    logger.info("Installing advanced GPU computing libraries...")
    
    # Check CUDA availability
    if not check_cuda():
        logger.warning("CUDA not available. Some features may not work.")
    
    # Check Python version
    python_version = sys.version_info
    logger.info(f"Python version: {python_version.major}.{python_version.minor}.{python_version.patch}")
    
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        logger.error("Python 3.8+ is required")
        return False
    
    success = True
    
    # 1. Numba with CUDA support
    logger.info("\n1. Installing Numba CUDA...")
    if not run_command(
        "pip install numba cuda-python",
        "Numba with CUDA support"
    ):
        success = False
    
    # 2. Triton (OpenAI's GPU kernel language)
    logger.info("\n2. Installing Triton...")
    if not run_command(
        "pip install triton",
        "Triton GPU kernel compiler"
    ):
        # Triton might have specific requirements
        logger.warning("Triton installation failed. Trying alternative method...")
        if not run_command(
            "pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly",
            "Triton (nightly build)"
        ):
            logger.warning("Triton installation failed. Some features will be unavailable.")
    
    # 3. RAPIDS cuGraph (GPU graph analytics)
    logger.info("\n3. Installing RAPIDS cuGraph...")
    
    # Detect CUDA version for RAPIDS
    cuda_version = "11.8"  # Default
    try:
        import torch
        cuda_version = torch.version.cuda.replace(".", "")[:3]
        cuda_version = f"{cuda_version[0]}.{cuda_version[1:]}"
    except:
        pass
    
    # RAPIDS installation (conda is preferred, but we'll use pip)
    rapids_success = False
    
    # Try pip installation
    if not rapids_success:
        logger.info("Installing RAPIDS components via pip...")
        rapids_packages = [
            "cudf-cu11",
            "cugraph-cu11",
            "cuml-cu11",
            "cuspatial-cu11"
        ]
        
        for package in rapids_packages:
            if not run_command(
                f"pip install {package}",
                f"RAPIDS {package}"
            ):
                logger.warning(f"Failed to install {package}")
    
    # 4. Additional GPU libraries
    logger.info("\n4. Installing additional GPU libraries...")
    
    additional_libs = [
        ("pycuda", "PyCUDA for low-level GPU access"),
        ("cupy-cuda11x", "CuPy (GPU NumPy) - latest"),
        ("torch-scatter", "PyTorch Scatter for graph operations"),
        ("torch-sparse", "PyTorch Sparse for graph operations"),
        ("torch-geometric", "PyTorch Geometric for GNNs"),
        ("networkx", "NetworkX for graph algorithms"),
        ("scipy", "SciPy for scientific computing"),
        ("scikit-learn", "Scikit-learn for ML utilities")
    ]
    
    for lib, description in additional_libs:
        if not run_command(f"pip install {lib}", description):
            logger.warning(f"Failed to install {lib}")
    
    # 5. Verify installations
    logger.info("\n=== Verifying installations ===")
    
    # Check Numba CUDA
    try:
        from numba import cuda
        logger.info("✓ Numba CUDA is available")
        if cuda.is_available():
            logger.info(f"  Detected {len(cuda.gpus)} GPU(s)")
    except Exception as e:
        logger.error(f"✗ Numba CUDA verification failed: {e}")
        success = False
    
    # Check Triton
    try:
        import triton
        logger.info("✓ Triton is available")
    except ImportError:
        logger.warning("✗ Triton not available (optional)")
    
    # Check CuPy
    try:
        import cupy as cp
        logger.info("✓ CuPy is available")
        logger.info(f"  CuPy version: {cp.__version__}")
    except ImportError:
        logger.error("✗ CuPy not available")
        success = False
    
    # Check cuGraph
    try:
        import cugraph
        logger.info("✓ cuGraph is available")
    except ImportError:
        logger.warning("✗ cuGraph not available (RAPIDS installation may have failed)")
        logger.info("  Note: RAPIDS works best with conda. Consider using:")
        logger.info("  conda install -c rapidsai -c nvidia -c conda-forge cugraph")
    
    # Check PyTorch Geometric
    try:
        import torch_geometric
        logger.info("✓ PyTorch Geometric is available")
    except ImportError:
        logger.warning("✗ PyTorch Geometric not available")
    
    # 6. Create requirements file
    logger.info("\n=== Creating requirements file ===")
    
    requirements = """# KIMERA Week 3: Advanced GPU Computing Requirements
# Generated automatically

# Core GPU libraries (from previous weeks)
torch>=2.0.0
cupy-cuda11x>=11.0.0
numpy>=1.21.0

# Numba CUDA
numba>=0.57.0
cuda-python>=11.7.0

# Triton (GPU kernel compiler)
triton>=2.0.0

# RAPIDS (GPU data science)
# Note: RAPIDS works best with conda installation
# conda install -c rapidsai -c nvidia -c conda-forge rapids=23.04 python=3.10 cudatoolkit=11.8

# Graph processing
networkx>=3.0
torch-scatter>=2.1.0
torch-sparse>=0.6.17
torch-geometric>=2.3.0

# Additional GPU utilities
pycuda>=2022.2
scipy>=1.10.0
scikit-learn>=1.2.0

# Monitoring and profiling
nvidia-ml-py>=11.525.0
gpustat>=1.1.0
py3nvml>=0.2.7
"""
    
    with open("requirements_week3_advanced_gpu.txt", "w") as f:
        f.write(requirements)
    
    logger.info("✓ Created requirements_week3_advanced_gpu.txt")
    
    # Final summary
    logger.info("\n=== Installation Summary ===")
    if success:
        logger.info("✓ Week 3 advanced GPU computing libraries installed successfully!")
        logger.info("\nYou can now run the advanced GPU computing tests:")
        logger.info("  python tests/test_advanced_gpu_computing.py")
    else:
        logger.error("✗ Some installations failed. Please check the errors above.")
        logger.info("\nFor RAPIDS/cuGraph issues, consider using conda:")
        logger.info("  conda create -n kimera-rapids -c rapidsai -c nvidia -c conda-forge rapids python=3.10 cudatoolkit=11.8")
    
    return success


def create_test_script():
    """Create a simple test script to verify installations"""
    test_script = '''#!/usr/bin/env python3
"""Quick test script for Week 3 libraries"""

import sys

logger.info("Testing Week 3 Advanced GPU Computing libraries...")

# Test Numba CUDA
try:
    from numba import cuda
    logger.info("✓ Numba CUDA imported successfully")
    if cuda.is_available():
        logger.info(f"  - {len(cuda.gpus)
except Exception as e:
    logger.error(f"✗ Numba CUDA error: {e}")

# Test Triton
try:
    import triton
    logger.info("✓ Triton imported successfully")
except ImportError:
    logger.info("✗ Triton not available (optional)

# Test CuPy
try:
    import cupy as cp
    arr = cp.array([1, 2, 3])
    logger.info(f"✓ CuPy working: {arr}")
except Exception as e:
    logger.error(f"✗ CuPy error: {e}")

# Test cuGraph
try:
    import cugraph
    logger.info("✓ cuGraph imported successfully")
except ImportError:
    logger.info("✗ cuGraph not available (RAPIDS)

# Test PyTorch Geometric
try:
    import torch_geometric
    logger.info("✓ PyTorch Geometric imported successfully")
except ImportError:
    logger.info("✗ PyTorch Geometric not available")

logger.info("\\nTest complete!")
'''
    
    with open("test_week3_imports.py", "w") as f:
        f.write(test_script)
    
    logger.info("✓ Created test_week3_imports.py")


if __name__ == "__main__":
    # Run installation
    success = install_week3_libraries()
    
    # Create test script
    create_test_script()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)