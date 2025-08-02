#!/usr/bin/env python3
"""
Debug pattern extraction in learning core
"""

import asyncio
import torch

async def debug_pattern_extraction():
    """Debug why patterns aren't being extracted"""
    print("🔍 DEBUGGING PATTERN EXTRACTION")
    print("=" * 40)
    
    try:
        from src.core.enhanced_capabilities.learning_core import (
            LearningCore, LearningMode
        )
        
        learning_core = LearningCore()
        
        # Test pattern extraction directly
        test_data = torch.sin(torch.linspace(0, 6*3.14159, 100)) + torch.randn(100) * 0.2
        print(f"Test data shape: {test_data.shape}")
        print(f"Test data mean: {torch.mean(test_data):.4f}")
        print(f"Test data std: {torch.std(test_data):.4f}")
        print(f"Test data abs mean: {torch.mean(torch.abs(test_data)):.4f}")
        
        # Check threshold calculation
        state_abs = torch.abs(test_data)
        threshold = torch.mean(state_abs).item() + 0.5 * torch.std(state_abs).item()
        print(f"Threshold for pattern detection: {threshold:.4f}")
        
        above_threshold = state_abs > threshold
        print(f"Points above threshold: {torch.sum(above_threshold).item()}")
        
        if torch.any(above_threshold):
            pattern_indices = torch.where(above_threshold)[0]
            print(f"Pattern indices found: {len(pattern_indices)}")
            
            if len(pattern_indices) > 1:
                pattern_vector = test_data[pattern_indices]
                print(f"Pattern vector shape: {pattern_vector.shape}")
                print("✅ Pattern should be created")
            else:
                print("❌ Not enough pattern indices")
        else:
            print("❌ No points above threshold")
        
        # Test basic pattern creation
        if len(test_data) > 10:
            top_indices = torch.topk(torch.abs(test_data), max(1, len(test_data) // 5)).indices
            print(f"Basic pattern indices: {len(top_indices)}")
            print("✅ Basic pattern should be created")
        
        # Now test actual learning
        print("\n🧪 Testing actual learning process...")
        result = await learning_core.learn_unsupervised(
            test_data,
            learning_mode=LearningMode.THERMODYNAMIC_ORG
        )
        
        print(f"Learning success: {result.success}")
        print(f"Discovered patterns: {len(result.discovered_patterns)}")
        print(f"Learning efficiency: {result.learning_efficiency:.6f}")
        print(f"Knowledge integration: {result.knowledge_integration:.6f}")
        
        if result.discovered_patterns:
            for i, pattern in enumerate(result.discovered_patterns):
                print(f"Pattern {i}: {pattern.pattern_type}, quality: {pattern.pattern_quality}")
        else:
            print("❌ No patterns in result!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_pattern_extraction())