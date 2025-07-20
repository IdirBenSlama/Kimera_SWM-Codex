#!/usr/bin/env python3
"""
Quick KIMERA Test - Direct interaction
"""

import time
import asyncio
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_kimera_direct():
    """Test KIMERA directly"""
    try:
        from backend.engines.kimera_text_diffusion_engine import (
            KimeraTextDiffusionEngine,
            DiffusionRequest,
            DiffusionMode,
            create_kimera_text_diffusion_engine
        )
        from backend.utils.gpu_foundation import GPUFoundation
        
        print("🧪 Testing KIMERA Text Diffusion Engine Directly...")
        print("=" * 50)
        
        # Initialize GPU foundation
        gpu_foundation = GPUFoundation()
        
        # Configuration for quick test
        config = {
            'num_steps': 5,  # Faster for testing
            'noise_schedule': 'cosine',
            'embedding_dim': 1024,
            'max_length': 256,
            'temperature': 0.8
        }
        
        # Create diffusion engine
        engine = create_kimera_text_diffusion_engine(config, gpu_foundation)
        
        if engine:
            print("✅ KIMERA Engine initialized successfully")
            
            # Create test request
            request = DiffusionRequest(
                source_content="Hello, what are you?",
                source_modality="text",
                target_modality="text",
                mode=DiffusionMode.STANDARD
            )
            
            print("🌊 Generating response...")
            start_time = time.time()
            
            result = await engine.generate(request)
            
            generation_time = time.time() - start_time
            
            print(f"⏱️ Generation time: {generation_time:.2f}s")
            print(f"🎯 Confidence: {result.confidence:.3f}")
            print(f"🧠 Semantic Coherence: {result.semantic_coherence:.3f}")
            print(f"🌟 Cognitive Resonance: {result.cognitive_resonance:.3f}")
            print(f"💬 Response: {result.generated_content}")
            
        else:
            print("❌ Failed to initialize KIMERA engine")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_kimera_direct()) 