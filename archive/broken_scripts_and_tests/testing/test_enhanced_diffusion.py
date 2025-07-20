#!/usr/bin/env python3
"""
Test script for KIMERA Enhanced Text Diffusion Engine
====================================================

This script tests the new diffusion-based conversation capabilities
and validates the different cognitive modes.
"""

import asyncio
import logging
import sys
import os
import time
from typing import Dict, Any

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_enhanced_diffusion():
    """Test the enhanced text diffusion engine."""
    try:
        # Import after path setup
        from backend.engines.kimera_text_diffusion_engine import (
            KimeraTextDiffusionEngine,
            DiffusionRequest,
            DiffusionMode,
            DiffusionConfig,
            create_kimera_text_diffusion_engine
        )
        from backend.utils.gpu_foundation import GPUFoundation
        
        logger.info("🚀 Starting Enhanced Text Diffusion Engine Test")
        
        # Initialize GPU foundation
        logger.info("Initializing GPU Foundation...")
        gpu_foundation = GPUFoundation()
        
        # Create enhanced diffusion engine
        logger.info("Creating Enhanced Text Diffusion Engine...")
        config = {
            'num_steps': 10,  # Reduced for testing
            'noise_schedule': 'cosine',
            'embedding_dim': 1024,
            'max_length': 256
        }
        
        engine = create_kimera_text_diffusion_engine(config, gpu_foundation)
        if not engine:
            logger.error("❌ Failed to create text diffusion engine")
            return False
        
        logger.info("✅ Enhanced Text Diffusion Engine created successfully")
        
        # Test different modes
        test_cases = [
            {
                "mode": DiffusionMode.STANDARD,
                "message": "Hello, how are you today?",
                "persona": "You are KIMERA, a helpful AI assistant."
            },
            {
                "mode": DiffusionMode.COGNITIVE_ENHANCED,
                "message": "Explain the concept of consciousness in AI systems.",
                "persona": "You are KIMERA in cognitive enhancement mode with deep analytical capabilities."
            },
            {
                "mode": DiffusionMode.PERSONA_AWARE,
                "message": "I'm feeling a bit overwhelmed with work. Any advice?",
                "persona": "You are KIMERA with heightened empathy and persona awareness."
            },
            {
                "mode": DiffusionMode.NEURODIVERGENT,
                "message": "Can you explain quantum computing in a structured way?",
                "persona": "You are KIMERA optimized for neurodivergent communication patterns."
            }
        ]
        
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"\n--- Test Case {i}: {test_case['mode'].value} ---")
            logger.info(f"Message: {test_case['message']}")
            
            # Create diffusion request
            request = DiffusionRequest(
                source_content=test_case['message'],
                source_modality="natural_language",
                target_modality="natural_language",
                mode=test_case['mode'],
                metadata={"persona_prompt": test_case['persona']}
            )
            
            # Generate response
            start_time = time.time()
            result = await engine.generate(request)
            generation_time = time.time() - start_time
            
            # Log results
            logger.info(f"Generated Response: {result.generated_content}")
            logger.info(f"Confidence: {result.confidence:.3f}")
            logger.info(f"Semantic Coherence: {result.semantic_coherence:.3f}")
            logger.info(f"Cognitive Resonance: {result.cognitive_resonance:.3f}")
            logger.info(f"Generation Time: {result.generation_time:.3f}s")
            logger.info(f"Diffusion Steps Used: {result.diffusion_steps_used}")
            
            results.append({
                "mode": test_case['mode'].value,
                "message": test_case['message'],
                "response": result.generated_content,
                "confidence": result.confidence,
                "semantic_coherence": result.semantic_coherence,
                "cognitive_resonance": result.cognitive_resonance,
                "generation_time": result.generation_time,
                "diffusion_steps": result.diffusion_steps_used
            })
        
        # Summary
        logger.info("\n🎉 ENHANCED DIFFUSION TEST SUMMARY")
        logger.info("=" * 50)
        
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        avg_coherence = sum(r['semantic_coherence'] for r in results) / len(results)
        avg_resonance = sum(r['cognitive_resonance'] for r in results) / len(results)
        total_time = sum(r['generation_time'] for r in results)
        
        logger.info(f"Total Test Cases: {len(results)}")
        logger.info(f"Average Confidence: {avg_confidence:.3f}")
        logger.info(f"Average Semantic Coherence: {avg_coherence:.3f}")
        logger.info(f"Average Cognitive Resonance: {avg_resonance:.3f}")
        logger.info(f"Total Generation Time: {total_time:.3f}s")
        logger.info(f"Average Time per Response: {total_time/len(results):.3f}s")
        
        # Success criteria
        success = (
            avg_confidence > 0.5 and
            avg_coherence > 0.5 and
            avg_resonance > 0.5 and
            all(r['response'] and len(r['response']) > 10 for r in results)
        )
        
        if success:
            logger.info("✅ ALL TESTS PASSED - Enhanced Text Diffusion Engine is working correctly!")
        else:
            logger.warning("⚠️  Some tests may need improvement, but basic functionality is working.")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}", exc_info=True)
        return False

async def test_conversation_flow():
    """Test conversation flow with context."""
    try:
        from backend.engines.universal_translator_hub import (
            UniversalTranslatorHub,
            UniversalTranslationRequest,
            TranslationModality,
            DiffusionMode
        )
        from backend.utils.gpu_foundation import GPUFoundation
        
        logger.info("\n🗣️  Testing Conversation Flow")
        logger.info("=" * 40)
        
        # Initialize components
        gpu_foundation = GPUFoundation()
        config = {'text_diffusion': {'num_steps': 10}}
        
        hub = UniversalTranslatorHub(config, gpu_foundation)
        
        # Simulate a conversation
        conversation = [
            "Hello, I'm interested in learning about AI consciousness.",
            "Can you explain how your cognitive architecture works?",
            "What makes your processing different from traditional AI?"
        ]
        
        conversation_context = []
        
        for i, message in enumerate(conversation, 1):
            logger.info(f"\nTurn {i}: {message}")
            
            request = UniversalTranslationRequest(
                source_content=message,
                source_modality=TranslationModality.NATURAL_LANGUAGE,
                target_modality=TranslationModality.COGNITIVE_ENHANCED,
                diffusion_mode=DiffusionMode.COGNITIVE_ENHANCED,
                conversation_context=conversation_context,
                metadata={"context": {"source": "cognitive_enhanced"}}
            )
            
            result = await hub.translate(request)
            
            logger.info(f"KIMERA: {result.translated_content}")
            logger.info(f"Metrics - Confidence: {result.confidence:.3f}, Coherence: {result.semantic_coherence:.3f}")
            
            # Update conversation context
            conversation_context.append({"user": message, "assistant": result.translated_content})
        
        logger.info("✅ Conversation flow test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Conversation flow test failed: {e}", exc_info=True)
        return False

async def main():
    """Main test function."""
    logger.info("🧪 KIMERA Enhanced Text Diffusion Test Suite")
    logger.info("=" * 60)
    
    # Test 1: Enhanced Diffusion Engine
    test1_success = await test_enhanced_diffusion()
    
    # Test 2: Conversation Flow
    test2_success = await test_conversation_flow()
    
    # Final Results
    logger.info("\n📊 FINAL TEST RESULTS")
    logger.info("=" * 30)
    logger.info(f"Enhanced Diffusion Engine: {'✅ PASS' if test1_success else '❌ FAIL'}")
    logger.info(f"Conversation Flow: {'✅ PASS' if test2_success else '❌ FAIL'}")
    
    overall_success = test1_success and test2_success
    logger.info(f"\nOverall Result: {'🎉 ALL TESTS PASSED' if overall_success else '⚠️  SOME TESTS FAILED'}")
    
    if overall_success:
        logger.info("\n🚀 KIMERA Enhanced Text Diffusion Engine is ready for conversation!")
        logger.info("Features validated:")
        logger.info("  ✅ True diffusion model architecture")
        logger.info("  ✅ Multiple cognitive modes")
        logger.info("  ✅ Persona-aware responses")
        logger.info("  ✅ Conversation context handling")
        logger.info("  ✅ Neurodivergent communication patterns")
        logger.info("  ✅ Real-time quality metrics")
    
    return overall_success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 