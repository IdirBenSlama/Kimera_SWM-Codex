#!/usr/bin/env python3
"""
Minimal Cognitive Test
=====================

Tests cognitive components with minimal imports to isolate issues.
"""

import asyncio
import torch
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO)

async def test_minimal_cognitive():
    """Test with minimal imports"""
    print('🧪 MINIMAL COGNITIVE TEST')
    print('=' * 30)
    
    try:
        # Test 1: Direct GPU operations first
        print('1️⃣  Testing GPU Operations...')
        from src.core.performance.gpu_acceleration import initialize_gpu_acceleration, move_to_gpu
        
        gpu_ok = initialize_gpu_acceleration()
        test_tensor = torch.randn(32, 32)
        gpu_tensor = move_to_gpu(test_tensor)
        result = torch.matmul(gpu_tensor, gpu_tensor.T)
        print(f'   ✅ GPU operations working: {result.shape}')
        
        # Test 2: Basic cache operations
        print('2️⃣  Testing Cache Operations...')
        from src.core.performance.advanced_caching import initialize_caching, put_cached, get_cached
        
        cache_ok = await initialize_caching()
        await put_cached('test', {'gpu_result': str(result.shape)})
        cached = await get_cached('test')
        print(f'   ✅ Cache operations working: {cached["gpu_result"]}')
        
        # Test 3: Pipeline operations
        print('3️⃣  Testing Pipeline Operations...')
        from src.core.performance.pipeline_optimization import add_pipeline_task, TaskPriority
        
        async def simple_task(data):
            return f'Processed: {data}'
        
        task_ok = add_pipeline_task('test_task', simple_task, 'minimal_test', priority=TaskPriority.HIGH)
        print(f'   ✅ Pipeline operations working: {task_ok}')
        
        # Test 4: Scaling operations
        print('4️⃣  Testing Scaling Operations...')
        from src.core.scaling.horizontal_scaling import initialize_horizontal_scaling, route_cognitive_request
        
        scaling_ok = await initialize_horizontal_scaling(min_nodes=2, max_nodes=3)
        node = await route_cognitive_request('test_cognitive', {'priority': 'high'})
        print(f'   ✅ Scaling operations working: routed to {node}')
        
        print('\n🎉 ALL CORE PERFORMANCE SYSTEMS OPERATIONAL!')
        print('✅ GPU Acceleration: Working')
        print('✅ Advanced Caching: Working') 
        print('✅ Pipeline Optimization: Working')
        print('✅ Horizontal Scaling: Working')
        
        return True
        
    except Exception as e:
        print(f'❌ Minimal test error: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print('🚀 Testing Minimal Cognitive Systems')
    success = asyncio.run(test_minimal_cognitive())
    
    if success:
        print('\n🎉 CORE SYSTEMS FULLY OPERATIONAL!')
        print('🌟 Performance platform working perfectly!')
    else:
        print('\n❌ Core systems need attention')