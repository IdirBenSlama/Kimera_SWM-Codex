#!/usr/bin/env python3
"""
Simple GPU verification script
"""

import sys

def test_pytorch_cuda():
    """Test PyTorch CUDA"""
    logger.info("🔍 Testing PyTorch CUDA...")
    try:
        import torch
        logger.info(f"   PyTorch Version: {torch.__version__}")
        logger.info(f"   CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            logger.info(f"   CUDA Version: {torch.version.cuda}")
            logger.info(f"   GPU Count: {torch.cuda.device_count()}")
            logger.info(f"   GPU Name: {torch.cuda.get_device_name(0)}")
            
            # Test basic operations
            x = torch.randn(100, 100, device='cuda')
            y = torch.randn(100, 100, device='cuda')
            z = torch.matmul(x, y)
            logger.info(f"   ✅ Basic GPU operations working")
            return True
        else:
            logger.info(f"   ❌ CUDA not available")
            return False
            
    except Exception as e:
        logger.info(f"   ❌ PyTorch test failed: {e}")
        return False

def test_cupy():
    """Test CuPy"""
    logger.info("\n🔍 Testing CuPy...")
    try:
        import cupy as cp
import logging
logger = logging.getLogger(__name__)
        logger.info(f"   CuPy Version: {cp.__version__}")
        
        # Test basic operations
        x = cp.random.randn(100, 100)
        y = cp.random.randn(100, 100)
        z = cp.matmul(x, y)
        logger.info(f"   ✅ CuPy operations working")
        return True
        
    except Exception as e:
        logger.info(f"   ❌ CuPy test failed: {e}")
        return False

def main():
    """Main verification"""
    logger.info("🚀 GPU Acceleration Verification")
    logger.info("=" * 40)
    
    pytorch_ok = test_pytorch_cuda()
    cupy_ok = test_cupy()
    
    logger.info("\n📊 Results:")
    logger.info(f"   PyTorch CUDA: {'✅' if pytorch_ok else '❌'}")
    logger.info(f"   CuPy: {'✅' if cupy_ok else '❌'}")
    
    if pytorch_ok:
        logger.info("\n🎉 GPU acceleration is ready!")
        return 0
    else:
        logger.info("\n⚠️ GPU acceleration needs attention")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 