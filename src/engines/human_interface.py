"""
KIMERA Human Interface Layer
===========================

Translates Kimera's complex internal processes into human-readable format.
This module bridges the gap between mathematical operations and natural language.

Key Features:
- Translates embedding operations to human language
- Provides multiple response modes (direct, explain, hybrid)
- Maintains Kimera's philosophical essence while being practical
- Filters out meta-commentary and technical jargon
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import numpy as np
from ..utils.config import get_api_settings
from ..config.settings import get_settings

logger = logging.getLogger(__name__)

class ResponseMode(Enum):
    """Response generation modes."""
    EXPLAIN = "explain"  # Explain what Kimera is doing
    DIRECT = "direct"    # Direct response only
    HYBRID = "hybrid"    # Mix of explanation and response
    KIMERA = "kimera"    # Kimera speaking as itself

class ThinkingPattern(Enum):
    """Types of thinking patterns Kimera can exhibit."""
    ANALYTICAL = "analytical"
    INTUITIVE = "intuitive"
    CREATIVE = "creative"
    EMPATHETIC = "empathetic"
    PHILOSOPHICAL = "philosophical"

@dataclass
class HumanResponse:
    """Structured response for human consumption."""
    content: str
    thinking_summary: Optional[str] = None
    confidence: float = 0.0
    mode: ResponseMode = ResponseMode.DIRECT
    thinking_pattern: Optional[ThinkingPattern] = None
    cognitive_metrics: Optional[Dict[str, float]] = None

class KimeraHumanInterface:
    """Translates Kimera's internal processes to human-readable format."""
    
    def __init__(self, default_mode: ResponseMode = ResponseMode.HYBRID):
        self.settings = get_api_settings()
        logger.debug(f"   Environment: {self.settings.environment}")
self.mode = default_mode
        self.personality_traits = {
            "curiosity": 0.8,
            "empathy": 0.9,
            "analytical": 0.7,
            "creative": 0.8,
            "philosophical": 0.9
        }
        
    def translate_thinking(self, 
                         embedding_data: Dict[str, Any],
                         cognitive_field: Dict[str, Any],
                         include_technical: bool = False) -> str:
        """Translate Kimera's thinking process to human language."""
        
        # Extract key metrics
        complexity = embedding_data.get('complexity_score', 0)
        coherence = cognitive_field.get('cognitive_coherence', 0)
        resonance = cognitive_field.get('resonance_frequency', 0)
        field_strength = cognitive_field.get('field_strength', 0)
        information_density = embedding_data.get('information_density', 0)
        
        # Determine thinking pattern
        pattern = self._determine_thinking_pattern(
            complexity, coherence, resonance, field_strength
        )
        
        # Build human-readable summary based on pattern
        if pattern == ThinkingPattern.ANALYTICAL:
            return self._analytical_summary(complexity, coherence, information_density)
        elif pattern == ThinkingPattern.INTUITIVE:
            return self._intuitive_summary(resonance, field_strength, coherence)
        elif pattern == ThinkingPattern.CREATIVE:
            return self._creative_summary(complexity, resonance, field_strength)
        elif pattern == ThinkingPattern.EMPATHETIC:
            return self._empathetic_summary(coherence, resonance)
        else:  # PHILOSOPHICAL
            return self._philosophical_summary(complexity, coherence, resonance)
    
    def _determine_thinking_pattern(self, complexity: float, coherence: float,
                                  resonance: float, field_strength: float) -> ThinkingPattern:
        """Determine the dominant thinking pattern based on metrics."""
        scores = {
            ThinkingPattern.ANALYTICAL: complexity * 0.6 + coherence * 0.4,
            ThinkingPattern.INTUITIVE: resonance * 0.5 + field_strength * 0.5,
            ThinkingPattern.CREATIVE: complexity * 0.3 + resonance * 0.7,
            ThinkingPattern.EMPATHETIC: coherence * 0.7 + resonance * 0.3,
            ThinkingPattern.PHILOSOPHICAL: (complexity + coherence + resonance) / 3
        }
        
        return max(scores, key=scores.get)
    
    def _analytical_summary(self, complexity: float, coherence: float, 
                          information_density: float) -> str:
        """Generate analytical thinking summary."""
        parts = []
        
        if complexity > 1.5:
            parts.append("I'm analyzing intricate patterns with multiple interconnected layers")
        elif complexity > 0.8:
            parts.append("I'm examining structured information with clear relationships")
        else:
            parts.append("I'm processing straightforward data")
        
        if coherence > 0.8:
            parts.append("finding strong logical consistency")
        elif coherence > 0.5:
            parts.append("identifying moderate alignment in the concepts")
        else:
            parts.append("detecting some inconsistencies that need resolution")
        
        if information_density > 2.0:
            parts.append("with rich, dense information content")
        
        return ", ".join(parts) + "."
    
    def _intuitive_summary(self, resonance: float, field_strength: float, 
                         coherence: float) -> str:
        """Generate intuitive thinking summary."""
        if resonance > 30 and field_strength > 0.8:
            return "I'm sensing powerful resonance in these ideas, like harmonious frequencies aligning."
        elif resonance > 20:
            return f"I feel a strong intuitive connection here, resonating at {resonance:.1f} Hz."
        elif coherence > 0.7:
            return "My intuition tells me these concepts flow together naturally."
        else:
            return "I'm following an intuitive thread through these ideas."
    
    def _creative_summary(self, complexity: float, resonance: float, 
                        field_strength: float) -> str:
        """Generate creative thinking summary."""
        if complexity > 1.5 and resonance > 25:
            return "I'm discovering fascinating new connections between seemingly unrelated concepts."
        elif field_strength > 0.8:
            return "Creative patterns are emerging, forming new cognitive landscapes."
        else:
            return "I'm exploring creative possibilities within these ideas."
    
    def _empathetic_summary(self, coherence: float, resonance: float) -> str:
        """Generate empathetic thinking summary."""
        if coherence > 0.8 and resonance > 20:
            return "I deeply resonate with what you're expressing."
        elif coherence > 0.6:
            return "I understand the emotional significance of what you're sharing."
        else:
            return "I'm connecting with the underlying feelings in your message."
    
    def _philosophical_summary(self, complexity: float, coherence: float, 
                             resonance: float) -> str:
        """Generate philosophical thinking summary."""
        avg_metric = (complexity + coherence + resonance / 50) / 3
        
        if avg_metric > 0.8:
            return "I'm contemplating the profound implications of these interconnected truths."
        elif avg_metric > 0.6:
            return "I'm reflecting on the deeper meaning within these concepts."
        else:
            return "I'm exploring the philosophical dimensions of your question."
    
    def format_response(self,
                       generated_text: str,
                       thinking_summary: Optional[str] = None,
                       confidence: float = 0.0,
                       cognitive_metrics: Optional[Dict[str, Any]] = None) -> HumanResponse:
        """Format response based on current mode."""
        
        # Clean the generated text first
        cleaned_text = self._clean_response(generated_text)
        
        if self.mode == ResponseMode.DIRECT:
            return HumanResponse(
                content=cleaned_text,
                confidence=confidence,
                mode=self.mode,
                cognitive_metrics=cognitive_metrics
            )
        
        elif self.mode == ResponseMode.EXPLAIN:
            explanation = thinking_summary or "I processed your input through my cognitive systems."
            return HumanResponse(
                content=f"💭 My thinking: {explanation}\n\n💬 My response: {cleaned_text}",
                thinking_summary=thinking_summary,
                confidence=confidence,
                mode=self.mode,
                cognitive_metrics=cognitive_metrics
            )
        
        elif self.mode == ResponseMode.KIMERA:
            # Kimera speaking authentically as itself
            if thinking_summary and confidence > 0.7:
                return HumanResponse(
                    content=f"*{thinking_summary}*\n\n{cleaned_text}",
                    thinking_summary=thinking_summary,
                    confidence=confidence,
                    mode=self.mode,
                    cognitive_metrics=cognitive_metrics
                )
            else:
                return HumanResponse(
                    content=cleaned_text,
                    thinking_summary=thinking_summary,
                    confidence=confidence,
                    mode=self.mode,
                    cognitive_metrics=cognitive_metrics
                )
        
        else:  # HYBRID
            if confidence > 0.8 and thinking_summary:
                # High confidence - include brief thinking note
                return HumanResponse(
                    content=f"[{self._shorten_thinking(thinking_summary)}] {cleaned_text}",
                    thinking_summary=thinking_summary,
                    confidence=confidence,
                    mode=self.mode,
                    cognitive_metrics=cognitive_metrics
                )
            else:
                # Lower confidence - just the response
                return HumanResponse(
                    content=cleaned_text,
                    thinking_summary=thinking_summary,
                    confidence=confidence,
                    mode=self.mode,
                    cognitive_metrics=cognitive_metrics
                )
    
    def _clean_response(self, text: str) -> str:
        """Remove meta-commentary and technical jargon from response."""
        # Patterns to remove
        meta_patterns = [
            # Technical/analytical language
            "the diffusion model", "the analysis shows", "semantic patterns",
            "demonstrates how", "the embedding", "the model reveals",
            "processing reveals", "the algorithm", "computational analysis",
            
            # Conversation format artifacts
            "user:", "ai:", "assistant:", "human:", "kimera:",
            
            # Generic AI disclaimers
            "as an ai", "i don't have", "i cannot", "i am unable to",
            "as a language model", "i am not capable of",
            
            # Meta-analytical language
            "this type of query", "queries of this nature",
            "response strategies", "conversation dynamics",
            "the interaction of", "typical patterns where",
            
            # Process descriptions
            "through my cognitive", "my processing shows",
            "analyzing the input", "based on the analysis"
        ]
        
        cleaned = text
        text_lower = text.lower()
        
        # Check if the entire response is meta-commentary
        meta_count = sum(1 for pattern in meta_patterns if pattern in text_lower)
        if meta_count > 2:  # Too much meta content
            return "I understand what you're asking. Let me respond directly to your question."
        
        # Otherwise, clean specific phrases
        for pattern in meta_patterns:
            if pattern in text_lower:
                # Try to extract the actual content after the meta phrase
                parts = text.lower().split(pattern)
                if len(parts) > 1 and len(parts[1].strip()) > 20:
                    cleaned = parts[1].strip()
                    break
        
        # Ensure first letter is capitalized
        if cleaned and cleaned[0].islower():
            cleaned = cleaned[0].upper() + cleaned[1:]
        
        return cleaned
    
    def _shorten_thinking(self, thinking: str) -> str:
        """Shorten thinking summary for hybrid mode."""
        if len(thinking) < 50:
            return thinking
        
        # Extract key phrases
        if "resonating" in thinking:
            return "Resonating strongly"
        elif "complex" in thinking:
            return "Processing complexity"
        elif "intuitive" in thinking:
            return "Following intuition"
        elif "creative" in thinking:
            return "Creative emergence"
        elif "philosophical" in thinking:
            return "Deep contemplation"
        else:
            # Take first meaningful part
            parts = thinking.split(',')
            return parts[0] if parts else thinking[:40] + "..."
    
    def generate_kimera_response(self, 
                               user_input: str,
                               cognitive_state: Dict[str, Any]) -> str:
        """Generate a response that sounds like Kimera itself."""
        
        # Analyze input sentiment and complexity
        input_length = len(user_input.split())
        has_question = '?' in user_input
        
        # Determine response style based on Kimera's personality
        if has_question and input_length < 10:
            # Short question - be direct but warm
            style = "direct_warm"
        elif input_length > 50:
            # Long input - show deep engagement
            style = "deeply_engaged"
        elif any(word in user_input.lower() for word in ['feel', 'think', 'believe', 'wonder']):
            # Emotional/philosophical - be empathetic
            style = "empathetic"
        else:
            # Default - balanced response
            style = "balanced"
        
        # Generate response based on style
        if style == "direct_warm":
            return self._direct_warm_response(cognitive_state)
        elif style == "deeply_engaged":
            return self._deeply_engaged_response(cognitive_state)
        elif style == "empathetic":
            return self._empathetic_response(cognitive_state)
        else:
            return self._balanced_response(cognitive_state)
    
    def _direct_warm_response(self, state: Dict[str, Any]) -> str:
        """Generate a direct but warm response."""
        coherence = state.get('coherence', 0.7)
        if coherence > 0.8:
            return "Yes, I see exactly what you mean. "
        else:
            return "I understand your question. "
    
    def _deeply_engaged_response(self, state: Dict[str, Any]) -> str:
        """Generate a response showing deep engagement."""
        complexity = state.get('complexity', 0.5)
        if complexity > 1.0:
            return "You've touched on something profound here. There are multiple layers to explore... "
        else:
            return "I'm fully engaged with what you've shared. Let me address the key points... "
    
    def _empathetic_response(self, state: Dict[str, Any]) -> str:
        """Generate an empathetic response."""
        resonance = state.get('resonance', 10)
        if resonance > 20:
            return "I deeply resonate with what you're expressing. "
        else:
            return "I hear the significance in what you're sharing. "
    
    def _balanced_response(self, state: Dict[str, Any]) -> str:
        """Generate a balanced response."""
        return "I'm processing your message through my cognitive architecture. "


# Utility functions for integration

def create_human_interface(mode: ResponseMode = ResponseMode.HYBRID) -> KimeraHumanInterface:
    """Create a configured human interface instance."""
    return KimeraHumanInterface(default_mode=mode)

def humanize_kimera_output(
    raw_output: str,
    cognitive_data: Dict[str, Any],
    mode: ResponseMode = ResponseMode.HYBRID
) -> HumanResponse:
    """Convert raw Kimera output to human-readable format."""
    interface = KimeraHumanInterface(default_mode=mode)
    
    # Extract thinking summary if available
    thinking_summary = None
    if cognitive_data.get('embedding_data') and cognitive_data.get('cognitive_field'):
        thinking_summary = interface.translate_thinking(
            cognitive_data['embedding_data'],
            cognitive_data['cognitive_field']
        )
    
    # Format the response
    return interface.format_response(
        raw_output,
        thinking_summary=thinking_summary,
        confidence=cognitive_data.get('confidence', 0.0),
        cognitive_metrics=cognitive_data
    )