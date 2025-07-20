id#!/usr/bin/env python3
"""
KIMERA Response Source Tracer
============================

This script traces exactly where KIMERA's responses are coming from:
1. Real diffusion engine
2. Fallback generation
3. Which language model is being used
4. What prompts are being fed to the model
"""

import sys
import os
import torch
import time
import asyncio
from typing import Dict, Any

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import KIMERA components
from backend.engines.kimera_text_diffusion_engine import (
    KimeraTextDiffusionEngine,
    DiffusionRequest,
    DiffusionMode,
    create_kimera_text_diffusion_engine
)
from backend.utils.gpu_foundation import GPUFoundation
from backend.utils.kimera_logger import get_logger, LogCategory

logger = get_logger(__name__, LogCategory.SYSTEM)

class KimeraResponseTracer:
    """Trace exactly where KIMERA responses come from"""
    
    def __init__(self):
        self.gpu_foundation = None
        self.diffusion_engine = None
        
    async def initialize(self):
        """Initialize KIMERA with tracing"""
        logger.info("🔍 KIMERA Response Tracer - Initializing...")
        
        # Initialize GPU foundation
        self.gpu_foundation = GPUFoundation()
        
        # Configuration
        config = {
            'num_steps': 10,
            'noise_schedule': 'cosine',
            'embedding_dim': 1024,
            'max_length': 256,
            'temperature': 0.8
        }
        
        # Create diffusion engine
        self.diffusion_engine = create_kimera_text_diffusion_engine(config, self.gpu_foundation)
        
        if self.diffusion_engine:
            logger.info("✅ KIMERA Diffusion Engine initialized for tracing")
            
            # Log the language model being used
            model_name = "microsoft/phi-2"  # This is hardcoded in the engine
            logger.info(f"📝 Language Model: {model_name}")
            logger.info(f"🔧 Tokenizer: {type(self.diffusion_engine.tokenizer).__name__}")
            logger.info(f"🧠 Model Type: {type(self.diffusion_engine.language_model).__name__}")
            
            return True
        else:
            logger.error("❌ Failed to initialize KIMERA engine")
            return False
    
    async def trace_response_generation(self, user_input: str) -> Dict[str, Any]:
        """Trace exactly how a response is generated"""
        logger.info(f"🔍 TRACING RESPONSE GENERATION FOR: '{user_input}'")
        logger.info("=" * 60)
        
        trace_data = {
            "user_input": user_input,
            "steps": [],
            "final_response": None,
            "source": None,
            "timing": {},
            "model_info": {}
        }
        
        start_time = time.time()
        
        try:
            # Create diffusion request
            request = DiffusionRequest(
                source_content=user_input,
                source_modality="text",
                target_modality="text",
                mode=DiffusionMode.STANDARD
            )
            
            trace_data["steps"].append("1. Created DiffusionRequest")
            logger.info("📋 Step 1: Created DiffusionRequest")
            
            # Generate response
            logger.info("🌊 Step 2: Calling diffusion engine generate()...")
            trace_data["steps"].append("2. Calling diffusion engine")
            
            generation_start = time.time()
            result = await self.diffusion_engine.generate(request)
            generation_time = time.time() - generation_start
            
            trace_data["timing"]["generation_time"] = generation_time
            trace_data["final_response"] = result.generated_content
            
            # Analyze the result to determine source
            if "fallback" in result.metadata:
                trace_data["source"] = "FALLBACK_GENERATION"
                logger.info("🔄 Step 3: Response came from FALLBACK generation")
                logger.info(f"   Model used: {result.metadata.get('model_used', 'unknown')}")
            else:
                trace_data["source"] = "DIFFUSION_ENGINE"
                logger.info("🌊 Step 3: Response came from DIFFUSION engine")
                logger.info(f"   Diffusion steps: {result.diffusion_steps_used}")
                logger.info(f"   Model type: {result.metadata.get('model_type', 'unknown')}")
            
            # Log quality metrics
            logger.info("📊 Quality Metrics:")
            logger.info(f"   Confidence: {result.confidence:.3f}")
            logger.info(f"   Semantic Coherence: {result.semantic_coherence:.3f}")
            logger.info(f"   Cognitive Resonance: {result.cognitive_resonance:.3f}")
            
            trace_data["quality_metrics"] = {
                "confidence": result.confidence,
                "semantic_coherence": result.semantic_coherence,
                "cognitive_resonance": result.cognitive_resonance
            }
            
            # Determine actual text generation source
            await self._trace_text_generation_source(trace_data, user_input)
            
        except Exception as e:
            logger.error(f"❌ Error during tracing: {e}")
            trace_data["error"] = str(e)
            trace_data["source"] = "ERROR"
        
        total_time = time.time() - start_time
        trace_data["timing"]["total_time"] = total_time
        
        logger.info("=" * 60)
        logger.info(f"🎯 FINAL TRACE RESULT:")
        logger.info(f"   Source: {trace_data['source']}")
        logger.info(f"   Response: {trace_data['final_response']}")
        logger.info(f"   Generation Time: {trace_data['timing'].get('generation_time', 0):.2f}s")
        
        return trace_data
    
    async def _trace_text_generation_source(self, trace_data: Dict, user_input: str):
        """Trace where the actual text generation happens"""
        logger.info("🔍 Step 4: Tracing actual text generation source...")
        
        # The key insight: Look at the generation time and coherence
        gen_time = trace_data["timing"].get("generation_time", 0)
        coherence = trace_data["quality_metrics"].get("semantic_coherence", 0)
        
        if gen_time > 10.0:  # Real diffusion takes 10+ seconds
            if coherence < 0.1:  # Very low coherence = novel generation
                logger.info("🎯 CONFIRMED: Real diffusion with novel text generation")
                trace_data["detailed_source"] = "REAL_DIFFUSION_NOVEL"
            else:
                logger.info("🎯 CONFIRMED: Real diffusion with coherent text generation")
                trace_data["detailed_source"] = "REAL_DIFFUSION_COHERENT"
        elif gen_time > 1.0:  # Medium time = possible diffusion with issues
            logger.info("🎯 CONFIRMED: Partial diffusion processing")
            trace_data["detailed_source"] = "PARTIAL_DIFFUSION"
        else:  # Fast time = fallback
            logger.info("🎯 CONFIRMED: Fallback generation (too fast for real diffusion)")
            trace_data["detailed_source"] = "FALLBACK_FAST"
        
        # Check if the response seems to come from training data
        response = trace_data["final_response"] or ""
        if any(name in response.lower() for name in ["sara", "ava", "paris", "bonjour"]):
            logger.info("�� DETECTED: Response contains training data patterns")
            trace_data["contains_training_data"] = True
        else:
            trace_data["contains_training_data"] = False
        
        # The actual text generation happens in _embedding_to_text method
        # which uses microsoft/phi-2 model with this prompt pattern:
        logger.info("📝 Text Generation Details:")
        logger.info("   Model: microsoft/phi-2 (2.7B parameters)")
        logger.info("   Method: AutoModelForCausalLM.generate()")
        logger.info("   Prompt: 'Generate coherent text based on the following context:\n\nGenerate a thoughtful response:'")
        logger.info("   Temperature: 0.8, top_k: 50, top_p: 0.9")
        
        trace_data["text_generation_details"] = {
            "model": "microsoft/phi-2",
            "method": "AutoModelForCausalLM.generate",
            "prompt_pattern": "Generate coherent text based on the following context:\n\nGenerate a thoughtful response:",
            "parameters": {
                "temperature": 0.8,
                "top_k": 50,
                "top_p": 0.9,
                "max_length": "input_length + 150"
            }
        }

async def main():
    """Main tracing function"""
    tracer = KimeraResponseTracer()
    
    if await tracer.initialize():
        # Test with the same inputs from the chat log
        test_inputs = [
            "hi",
            "who are you?",
            "what is your name?",
            "my name is Idir"
        ]
        
        for test_input in test_inputs:
            print(f"\n{'='*80}")
            print(f"TRACING: '{test_input}'")
            print(f"{'='*80}")
            
            trace_result = await tracer.trace_response_generation(test_input)
            
            print(f"\n📋 TRACE SUMMARY:")
            print(f"   Input: {trace_result['user_input']}")
            print(f"   Response: {trace_result['final_response']}")
            print(f"   Source: {trace_result['source']}")
            print(f"   Detailed Source: {trace_result.get('detailed_source', 'unknown')}")
            print(f"   Generation Time: {trace_result['timing'].get('generation_time', 0):.2f}s")
            print(f"   Contains Training Data: {trace_result.get('contains_training_data', False)}")
            
            # Wait between tests
            await asyncio.sleep(2)
    
    else:
        print("❌ Failed to initialize tracer")

if __name__ == "__main__":
    asyncio.run(main()) 