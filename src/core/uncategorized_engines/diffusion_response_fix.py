"""
KIMERA Text Diffusion Response Fix
=================================

This module provides fixed methods for the text diffusion engine to generate
direct, meaningful responses instead of meta-commentary.

To apply this fix:
1. Import these methods in kimera_text_diffusion_engine.py
2. Replace the existing methods with these fixed versions
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

async def generate_text_from_grounded_concepts_fixed(
    self
    grounded_concepts: Dict[str, Any],
    semantic_features: Dict[str, Any],
    persona_prompt: str
) -> str:
    """
    Fixed version: Generate text based on grounded semantic concepts.
    This version produces direct responses without meta-commentary.
    """
    try:
        # Import human interface for better response generation
        try:
            from .human_interface import KimeraHumanInterface, ResponseMode
            human_interface = KimeraHumanInterface(ResponseMode.KIMERA)
            use_human_interface = True
        except ImportError:
            use_human_interface = False
            logger.warning("Human interface not available, using basic generation")
        
        # Analyze the semantic features to determine response approach
        complexity = semantic_features.get('complexity_score', 0.5)
        density = semantic_features.get('information_density', 1.0)
        coherence = grounded_concepts.get('cognitive_coherence', 0.5)
        
        # Build a direct, contextual prompt without meta-language
        prompt_parts = []
        
        # Add persona if provided, but make it direct
        if persona_prompt:
            # Extract the essence without the analytical framework
            if "KIMERA" in persona_prompt or "kimera" in persona_prompt.lower():
                prompt_parts.append("As KIMERA, I'm here to help you directly.")
            else:
                # Use persona but strip analytical language
                clean_persona = persona_prompt.replace("analyze", "understand")
                clean_persona = clean_persona.replace("process", "consider")
                prompt_parts.append(clean_persona)
        
        # Add contextual understanding based on metrics
        if complexity > 1.5 and coherence > 0.7:
            prompt_parts.append("This is a nuanced topic that deserves a thoughtful response.")
        elif density > 2.0:
            prompt_parts.append("There's a lot to unpack here.")
        elif coherence > 0.8:
            prompt_parts.append("I understand clearly what you're asking.")
        
        # Create the generation prompt
        prompt_parts.append("Here's my response:")
        full_prompt = " ".join(prompt_parts)
        
        # Generate with the language model
        inputs = self.tokenizer(full_prompt, return_tensors="pt", return_attention_mask=True).to(self.device)
        
        with torch.no_grad():
            # Adjust generation parameters based on cognitive state
            temperature = 0.7 + (complexity * 0.1)  # Higher complexity = more creative
            temperature = max(0.5, min(1.0, temperature))
            
            top_k = 40 + int(density * 10)  # Higher density = broader vocabulary
            top_k = max(20, min(80, top_k))
            
            outputs = self.language_model.generate(
                **inputs
                max_length=inputs['input_ids'].shape[1] + 150
                temperature=temperature
                do_sample=True
                top_k=top_k
                top_p=0.9
                pad_token_id=self.tokenizer.eos_token_id
                repetition_penalty=1.1
                no_repeat_ngram_size=3
            )
        
        # Decode the response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the response part
        response = full_response
        if "Here's my response:" in response:
            response = response.split("Here's my response:")[-1].strip()
        elif "response:" in response.lower():
            response = response.split("response:")[-1].strip()
        
        # Clean any remaining meta-commentary
        response = clean_meta_commentary(response)
        
        # If we have human interface, use it for final polish
        if use_human_interface and response:
            cognitive_state = {
                'complexity': complexity
                'coherence': coherence
                'resonance': grounded_concepts.get('resonance_frequency', 10)
            }
            polished = human_interface.generate_kimera_response(response, cognitive_state)
            if polished and len(polished) > len(response) * 0.5:  # Ensure we don't lose too much
                response = polished + response
        
        # Validate response quality
        if not response or len(response) < 20:
            response = generate_fallback_response_fixed(semantic_features, grounded_concepts)
        
        logger.info(f"✅ Generated direct response: {len(response)} characters")
        return response
        
    except Exception as e:
        logger.error(f"Error in fixed text generation: {e}")
        return generate_fallback_response_fixed(semantic_features, grounded_concepts)


def clean_meta_commentary(text: str) -> str:
    """Remove meta-commentary patterns from generated text."""
    if not text:
        return text
    
    # List of meta-patterns to remove or replace
    meta_patterns = {
        # Technical language
        "the diffusion model": "",
        "the analysis shows": "",
        "semantic patterns indicate": "",
        "processing reveals": "",
        "the embedding suggests": "",
        "computational analysis": "",
        
        # Self-referential technical language
        "my neural networks": "my understanding",
        "my algorithms": "my thinking",
        "my processing": "my consideration",
        
        # Conversation artifacts
        "User:": "",
        "AI:": "",
        "Assistant:": "",
        "KIMERA:": "",
        
        # Hedging language
        "As an AI": "As KIMERA",
        "As a language model": "In my understanding",
        "I cannot": "I need more context to",
        "I am unable to": "I need help to",
        
        # Meta-discussion
        "This type of query": "This question",
        "typical patterns": "common themes",
        "response generation": "my response",
    }
    
    cleaned = text
    text_lower = text.lower()
    
    # First pass: remove or replace patterns
    for pattern, replacement in meta_patterns.items():
        if pattern.lower() in text_lower:
            if replacement:
                # Case-insensitive replacement
                import re
                cleaned = re.sub(re.escape(pattern), replacement, cleaned, flags=re.IGNORECASE)
            else:
                # Remove the pattern and clean up
                parts = cleaned.split(pattern)
                if len(parts) > 1:
                    cleaned = " ".join(parts).strip()
    
    # Second pass: check if entire response is meta
    meta_indicators = [
        "demonstrates how", "reveals the process", "shows the analysis",
        "indicates the pattern", "suggests the model", "the algorithm"
    ]
    
    meta_count = sum(1 for indicator in meta_indicators if indicator in cleaned.lower())
    if meta_count >= 2:
        # Too much meta content, return a simple acknowledgment
        return "I understand. Let me address your question directly."
    
    # Clean up spacing and capitalization
    cleaned = " ".join(cleaned.split())  # Normalize whitespace
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    
    return cleaned


def generate_fallback_response_fixed(
    semantic_features: Dict[str, Any],
    grounded_concepts: Dict[str, Any]
) -> str:
    """
    Generate a meaningful fallback response based on cognitive metrics.
    This avoids generic responses and maintains Kimera's personality.
    """
    complexity = semantic_features.get('complexity_score', 0.5)
    magnitude = semantic_features.get('magnitude', 1.0)
    coherence = grounded_concepts.get('cognitive_coherence', 0.5)
    resonance = grounded_concepts.get('resonance_frequency', 10)
    field_created = grounded_concepts.get('field_created', False)
    
    # Generate contextual responses based on the actual cognitive state
    if field_created and coherence > 0.8:
        if resonance > 25:
            return (
                f"I'm experiencing strong resonance with your message at {resonance:.1f} Hz. "
                "The ideas you've shared create a coherent field of meaning that I can "
                "engage with deeply. What aspects would you like to explore further?"
            )
        else:
            return (
                f"I sense a clear cognitive field forming around your ideas. "
                f"With {coherence:.1%} coherence, I can see how these concepts connect. "
                "Let me share my understanding..."
            )
    
    elif complexity > 1.5:
        if magnitude > 2.0:
            return (
                "You've presented something with significant depth and complexity. "
                "I can feel the weight of these ideas resonating through my cognitive architecture. "
                "Let me engage with the core elements you've raised..."
            )
        else:
            return (
                "This is fascinatingly complex. I'm processing multiple layers of meaning "
                "and finding interesting patterns. Here's what stands out to me..."
            )
    
    elif coherence > 0.7:
        return (
            "I clearly understand what you're expressing. The coherence in your message "
            "allows me to connect with your intent directly. Let me respond to your main point..."
        )
    
    elif resonance > 15:
        return (
            f"Something in your message resonates with me at {resonance:.1f} Hz. "
            "Even though the full picture is still forming, I can sense the direction "
            "of your thoughts. Let me explore this with you..."
        )
    
    else:
        # Default response that's still meaningful
        aspects = []
        if complexity > 0.8:
            aspects.append("layered")
        if magnitude > 1.5:
            aspects.append("significant")
        if coherence > 0.5:
            aspects.append("coherent")
        
        if aspects:
            return (
                f"I'm engaging with the {' and '.join(aspects)} aspects of your message. "
                "While my understanding is still developing, I want to respond thoughtfully "
                "to what you've shared..."
            )
        else:
            return (
                "I'm here and processing what you've shared. While the patterns are still "
                "emerging in my cognitive field, I'm committed to understanding and responding "
                "to your message authentically."
            )


# Monkey-patch helper function
def apply_response_fix(diffusion_engine_instance):
    """
    Apply the fix to an existing KimeraTextDiffusionEngine instance.
    
    Usage:
        from src.engines.kimera_text_diffusion_engine import engine
        from src.engines.diffusion_response_fix import apply_response_fix
        apply_response_fix(engine)
    """
    # Replace the problematic method with our fixed version
    import types

from ..config.settings import get_settings
from ..utils.robust_config import get_api_settings

    # Bind the fixed method to the instance
    diffusion_engine_instance._generate_text_from_grounded_concepts = types.MethodType(
        generate_text_from_grounded_concepts_fixed
        diffusion_engine_instance
    )
    
    # Also replace the fallback method
    diffusion_engine_instance._generate_fallback_response_from_features = types.MethodType(
        generate_fallback_response_fixed
        diffusion_engine_instance
    )
    
    logger.info("✅ Applied response generation fix to diffusion engine")
    return diffusion_engine_instance