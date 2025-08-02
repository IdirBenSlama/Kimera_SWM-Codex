#!/usr/bin/env python3
"""
Simple GPU verification script
"""

import sys

def test_pytorch_cuda():
    """Test PyTorch CUDA"""
    print("🔍 Testing PyTorch CUDA...")
    try:
        import torch
        print(f"   PyTorch Version: {torch.__version__}")
        print(f"   CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   GPU Count: {torch.cuda.device_count()}")
            print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
            
            # Test basic operations
            x = torch.randn(100, 100, device='cuda')
            y = torch.randn(100, 100, device='cuda')
            z = torch.matmul(x, y)
            print(f"   ✅ Basic GPU operations working")
            return True
        else:
            print(f"   ❌ CUDA not available")
            return False
            
    except Exception as e:
        print(f"   ❌ PyTorch test failed: {e}")
        return False

def test_cupy():
    """Test CuPy"""
    print("\n🔍 Testing CuPy...")
    try:
        import cupy as cp
        print(f"   CuPy Version: {cp.__version__}")
        
        # Test basic operations
        x = cp.random.randn(100, 100)
        y = cp.random.randn(100, 100)
        z = cp.matmul(x, y)
        print(f"   ✅ CuPy operations working")
        return True
        
    except Exception as e:
        print(f"   ❌ CuPy test failed: {e}")
        return False

def main():
    """Main verification"""
    print("🚀 GPU Acceleration Verification")
    print("=" * 40)
    
    pytorch_ok = test_pytorch_cuda()
    cupy_ok = test_cupy()
    
    print("\n📊 Results:")
    print(f"   PyTorch CUDA: {'✅' if pytorch_ok else '❌'}")
    print(f"   CuPy: {'✅' if cupy_ok else '❌'}")
    
    if pytorch_ok:
        print("\n🎉 GPU acceleration is ready!")
        return 0
    else:
        print("\n⚠️ GPU acceleration needs attention")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 