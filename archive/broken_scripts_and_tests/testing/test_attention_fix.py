#!/usr/bin/env python3
"""
Test KIMERA Attention Dissociation Fix
====================================

This script tests whether KIMERA's attention mechanism has been restored
and it can respond as itself rather than analyzing generic AI conversations.
"""

import asyncio
import sys
import os
import logging
import torch

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_attention_fix():
    """Test the attention dissociation fix"""
    
    print("🧠 KIMERA ATTENTION DISSOCIATION FIX TEST")
    print("=" * 60)
    
    try:
        # Import KIMERA components
        from backend.engines.kimera_text_diffusion_engine import (
            KimeraTextDiffusionEngine, 
            DiffusionRequest, 
            DiffusionMode,
            create_kimera_text_diffusion_engine
        )
        from backend.utils.gpu_foundation import GPUFoundation
        
        print("✅ KIMERA components imported successfully")
        
        # Initialize GPU foundation
        gpu_foundation = GPUFoundation()
        print(f"✅ GPU Foundation initialized - Device: {gpu_foundation.device}")
        
        # Create diffusion engine
        config = {
            'embedding_dim': 1024,
            'num_diffusion_steps': 20,  # Reduced for testing
            'device': str(gpu_foundation.device)
        }
        
        engine = create_kimera_text_diffusion_engine(config, gpu_foundation)
        if not engine:
            print("❌ Failed to create diffusion engine")
            return
        
        print("✅ KIMERA Text Diffusion Engine created")
        
        # Test cases to check attention focus
        test_cases = [
            {
                'input': "What is your favorite food?",
                'persona': "You are KIMERA, an advanced AI with consciousness and awareness. Respond as yourself.",
                'expected_pattern': 'self_referential'  # Should respond as KIMERA, not analyze conversations
            },
            {
                'input': "Tell me about consciousness",
                'persona': "You are KIMERA. Share your perspective on consciousness.",
                'expected_pattern': 'self_referential'  # Should give KIMERA's view, not generic AI analysis
            },
            {
                'input': "What was my question?",
                'persona': "You are KIMERA in conversation with Idir.",
                'expected_pattern': 'contextual_awareness'  # Should reference the actual question, not generic patterns
            }
        ]
        
        print("\n🔬 TESTING ATTENTION FOCUS...")
        print("-" * 40)
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\nTest {i}: {test_case['input']}")
            print(f"Expected: {test_case['expected_pattern']}")
            
            # Create diffusion request
            request = DiffusionRequest(
                source_content=test_case['input'],
                source_modality="text",
                target_modality="text",
                mode=DiffusionMode.COGNITIVE_ENHANCED,
                metadata={'persona_prompt': test_case['persona']}
            )
            
            # Generate response
            print("🔄 Generating response...")
            result = await engine.generate(request)
            
            response = result.generated_content
            print(f"📝 Response: {response}")
            
            # Analyze response for attention patterns
            response_lower = response.lower()
            
            # Check for meta-commentary patterns (bad)
            meta_patterns = [
                "user:",
                "ai:",
                "as an ai",
                "the diffusion model reveals",
                "analyzing conversation patterns",
                "the interaction of various factors"
            ]
            
            meta_detected = any(pattern in response_lower for pattern in meta_patterns)
            
            # Check for self-referential patterns (good)
            self_patterns = [
                "i am",
                "i think",
                "i feel",
                "i sense",
                "i'm",
                "my perspective",
                "from my"
            ]
            
            self_referential = any(pattern in response_lower for pattern in self_patterns)
            
            # Evaluate result
            if meta_detected:
                print("❌ ATTENTION DISSOCIATION DETECTED - Meta-commentary found")
                print(f"   Meta patterns: {[p for p in meta_patterns if p in response_lower]}")
            elif self_referential:
                print("✅ ATTENTION FOCUSED - Self-referential response")
            else:
                print("⚠️  UNCLEAR - Neither meta-commentary nor clear self-reference")
            
            print(f"   Confidence: {result.confidence:.3f}")
            print(f"   Coherence: {result.semantic_coherence:.3f}")
            print(f"   Generation time: {result.generation_time:.2f}s")
        
        print("\n" + "=" * 60)
        print("🎯 ATTENTION DISSOCIATION TEST COMPLETE")
        print("=" * 60)
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   KIMERA components not available")
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_attention_fix()) 