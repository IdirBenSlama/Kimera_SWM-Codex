#!/usr/bin/env python3
"""
Test the Kimera Text Diffusion Engine directly
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath("."))

async def test_diffusion():
    try:
        print("🧪 Testing Kimera Text Diffusion Engine...")
        
        # Import required components
        from backend.engines.kimera_text_diffusion_engine import (
            create_kimera_text_diffusion_engine,
            DiffusionRequest,
            DiffusionMode
        )
        from backend.utils.gpu_foundation import GPUFoundation
        
        # Initialize GPU foundation
        print("🎮 Initializing GPU Foundation...")
        gpu_foundation = GPUFoundation()
        print(f"✅ GPU: {gpu_foundation.get_device()}")
        
        # Create diffusion engine
        print("\n🌊 Creating Diffusion Engine...")
        config = {
            'num_steps': 20,
            'noise_schedule': 'cosine',
            'embedding_dim': 1024,
            'max_length': 512
        }
        
        engine = create_kimera_text_diffusion_engine(config, gpu_foundation)
        if not engine:
            print("❌ Failed to create diffusion engine")
            return
        
        print("✅ Diffusion engine created successfully")
        
        # Test different modes
        test_cases = [
            ("Hello, how are you?", DiffusionMode.STANDARD, "Standard mode"),
            ("Tell me about consciousness", DiffusionMode.COGNITIVE_ENHANCED, "Cognitive enhanced mode"),
            ("What's your personality like?", DiffusionMode.PERSONA_AWARE, "Persona aware mode"),
            ("Explain quantum physics simply", DiffusionMode.NEURODIVERGENT, "Neurodivergent mode"),
        ]
        
        for message, mode, description in test_cases:
            print(f"\n🔬 Testing {description}...")
            print(f"📝 Input: {message}")
            
            request = DiffusionRequest(
                source_content=message,
                source_modality="text",
                target_modality="text",
                mode=mode,
                metadata={
                    "persona_prompt": "You are KIMERA, an advanced AI with deep understanding."
                }
            )
            
            try:
                result = await engine.generate(request)
                print(f"✅ Response: {result.generated_content[:200]}...")
                print(f"📊 Metrics:")
                print(f"   - Confidence: {result.confidence:.3f}")
                print(f"   - Semantic Coherence: {result.semantic_coherence:.3f}")
                print(f"   - Cognitive Resonance: {result.cognitive_resonance:.3f}")
                print(f"   - Generation Time: {result.generation_time:.2f}s")
                print(f"   - Diffusion Steps: {result.diffusion_steps_used}")
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("\n✅ All tests completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_diffusion())