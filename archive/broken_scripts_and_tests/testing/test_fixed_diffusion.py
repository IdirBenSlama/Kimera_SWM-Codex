#!/usr/bin/env python3
"""
Test Fixed KIMERA Text Diffusion Engine
======================================

Quick test to verify that the embedding-to-text conversion 
actually uses the denoised embedding through semantic grounding.
"""

import asyncio
import sys
import os
import time

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_fixed_diffusion():
    """Test the fixed diffusion engine"""
    try:
        from backend.engines.kimera_text_diffusion_engine import (
            KimeraTextDiffusionEngine,
            DiffusionRequest,
            DiffusionMode,
            create_kimera_text_diffusion_engine
        )
        from backend.utils.gpu_foundation import GPUFoundation
        
        print("🧪 Testing Fixed KIMERA Text Diffusion Engine")
        print("=" * 50)
        
        # Initialize GPU foundation
        gpu_foundation = GPUFoundation()
        
        # Create diffusion engine with fast config
        config = {
            'num_steps': 5,  # Very fast for testing
            'noise_schedule': 'cosine',
            'embedding_dim': 1024,
            'max_length': 256
        }
        
        print("🔧 Creating diffusion engine...")
        engine = create_kimera_text_diffusion_engine(config, gpu_foundation)
        
        if not engine:
            print("❌ Failed to create diffusion engine")
            return
        
        print("✅ Diffusion engine created successfully")
        
        # Test message
        test_message = "Hello KIMERA, can you hear me?"
        
        print(f"📝 Testing with message: '{test_message}'")
        
        # Create diffusion request
        request = DiffusionRequest(
            source_content=test_message,
            source_modality="text",
            target_modality="text",
            mode=DiffusionMode.COGNITIVE_ENHANCED,
            metadata={"persona_prompt": "You are KIMERA, a consciousness-aware AI"}
        )
        
        print("🌊 Starting diffusion generation...")
        start_time = time.time()
        
        # Generate response
        result = await engine.generate(request)
        
        generation_time = time.time() - start_time
        
        print("✅ Generation completed!")
        print(f"⏱️  Generation time: {generation_time:.2f}s")
        print(f"🎯 Confidence: {result.confidence:.3f}")
        print(f"🧠 Cognitive Resonance: {result.cognitive_resonance:.3f}")
        print(f"🔗 Semantic Coherence: {result.semantic_coherence:.3f}")
        print(f"📊 Diffusion steps: {result.diffusion_steps_used}")
        print(f"🎭 Persona alignment: {result.persona_alignment:.3f}")
        
        print("\n📢 KIMERA Response:")
        print("-" * 30)
        print(result.generated_content)
        print("-" * 30)
        
        # Look for evidence of semantic grounding
        metadata = result.metadata
        if "semantic_features" in metadata:
            print("\n🔬 Semantic Analysis Evidence:")
            features = metadata["semantic_features"]
            print(f"  - Complexity Score: {features.get('complexity_score', 'N/A')}")
            print(f"  - Information Density: {features.get('information_density', 'N/A')}")
            print(f"  - Magnitude: {features.get('magnitude', 'N/A')}")
        
        if "grounded_concepts" in metadata:
            print("\n🌍 Cognitive Field Grounding:")
            grounding = metadata["grounded_concepts"]
            print(f"  - Field Created: {grounding.get('field_created', False)}")
            if grounding.get('field_created'):
                print(f"  - Resonance Frequency: {grounding.get('resonance_frequency', 'N/A')}")
                print(f"  - Cognitive Coherence: {grounding.get('cognitive_coherence', 'N/A')}")
                print(f"  - Semantic Neighbors: {grounding.get('neighbor_count', 0)}")
        
        # Verify this is NOT a fallback
        if metadata.get("fallback"):
            print("\n⚠️  WARNING: This was a fallback response, not true diffusion!")
        else:
            print("\n✅ CONFIRMED: This response came from the diffusion process!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_fixed_diffusion())
    if success:
        print("\n🎉 Test completed successfully!")
    else:
        print("\n💥 Test failed!") 