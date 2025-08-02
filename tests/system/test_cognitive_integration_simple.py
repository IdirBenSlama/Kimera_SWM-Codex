#!/usr/bin/env python3
"""
Simple Cognitive Integration Test
================================

Tests individual cognitive components to identify integration issues.
"""

import asyncio
import torch

async def test_individual_cognitive_components():
    """Test cognitive components individually"""
    print('🧪 SIMPLE COGNITIVE INTEGRATION TEST')
    print('=' * 45)
    
    test_results = {}
    
    try:
        # Test 1: Understanding Core
        print('1️⃣  Testing Understanding Core...')
        from src.core.enhanced_capabilities.understanding_core import UnderstandingCore
        
        understanding = UnderstandingCore()
        print('   ✅ Understanding Core initialized')
        
        # Simple understanding test
        test_text = "Simple test for cognitive integration"
        understand_result = await understanding.understand(test_text)
        print(f'   ✅ Understanding completed: {understand_result.comprehension_quality:.3f}')
        test_results['understanding'] = 'PASSED'
        
        # Test 2: Consciousness Core
        print('\n2️⃣  Testing Consciousness Core...')
        from src.core.enhanced_capabilities.consciousness_core import ConsciousnessCore
        
        consciousness = ConsciousnessCore()
        print('   ✅ Consciousness Core initialized')
        
        # Simple consciousness test
        test_state = torch.randn(128)
        consciousness_result = await consciousness.detect_consciousness(test_state)
        print(f'   ✅ Consciousness detection: {consciousness_result["probability"]:.3f}')
        test_results['consciousness'] = 'PASSED'
        
        # Test 3: Learning Core
        print('\n3️⃣  Testing Learning Core...')
        from src.core.enhanced_capabilities.learning_core import LearningCore
        
        learning = LearningCore()
        print('   ✅ Learning Core initialized')
        
        # Simple learning test
        learning_result = await learning.learn_through_energy_minimization(test_state)
        print(f'   ✅ Learning efficiency: {learning_result["learning_efficiency"]:.3f}')
        test_results['learning'] = 'PASSED'
        
        # Test 4: GPU Integration
        print('\n4️⃣  Testing GPU Integration with Cognitive Components...')
        from src.core.performance.gpu_acceleration import move_to_gpu, optimized_context
        
        with optimized_context():
            gpu_tensor = move_to_gpu(torch.randn(64, 64))
            gpu_result = torch.matmul(gpu_tensor, gpu_tensor.T)
        
        print(f'   ✅ GPU tensor processing: {gpu_result.shape}')
        
        # Test cognitive component with GPU tensor
        gpu_consciousness_result = await consciousness.detect_consciousness(gpu_tensor.flatten()[:128])
        print(f'   ✅ GPU + Consciousness: {gpu_consciousness_result["probability"]:.3f}')
        test_results['gpu_integration'] = 'PASSED'
        
    except Exception as e:
        print(f'❌ Cognitive integration error: {e}')
        test_results['error'] = str(e)
        import traceback
        traceback.print_exc()
    
    # Results Summary
    print('\n' + '=' * 45)
    print('🎯 SIMPLE COGNITIVE INTEGRATION RESULTS')
    print('=' * 45)
    
    passed_tests = sum(1 for result in test_results.values() if result == 'PASSED')
    total_tests = len(test_results) - (1 if 'error' in test_results else 0)
    success_rate = passed_tests / total_tests if total_tests > 0 else 0
    
    print(f'Tests Passed: {passed_tests}/{total_tests}')
    print(f'Success Rate: {success_rate:.1%}')
    
    for test_name, result in test_results.items():
        if test_name != 'error':
            status = '✅' if result == 'PASSED' else '❌'
            print(f'{status} {test_name.replace("_", " ").title()}: {result}')
    
    if success_rate >= 1.0:
        print('\n🎉 SIMPLE COGNITIVE INTEGRATION: SUCCESS!')
        print('🚀 Individual cognitive components operational!')
    else:
        print('\n⚠️  Some cognitive components have issues')
        print('🔧 Review failed components and address issues')
    
    return success_rate >= 1.0

if __name__ == "__main__":
    print('🚀 Starting Simple Cognitive Integration Test')
    success = asyncio.run(test_individual_cognitive_components())
    
    if success:
        print('\n🎉 COGNITIVE INTEGRATION: SUCCESS!')
        print('🌟 Ready for full comprehensive integration!')
    else:
        print('\n🔧 Cognitive integration needs attention')
        print('📋 Review test results and address issues')