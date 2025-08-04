"""
Human Interface Integration - Core Integration Wrapper
====================================================

Integrates the Human Interface engine into the core Kimera system
to translate complex internal processes into human-readable format.

This solves the mathematical opacity problem by providing clear,
understandable explanations of what Kimera is thinking and doing.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

try:
    from ...engines.human_interface import (
        HumanResponse,
        KimeraHumanInterface,
        ResponseMode,
    )

    ENGINE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Human interface engine not available: {e}")
    ENGINE_AVAILABLE = False

    # Fallback classes
    class ResponseMode(Enum):
        DIRECT = "direct"
        EXPLAIN = "explain"
        HYBRID = "hybrid"

    class HumanResponse:
        def __init__(self, content, **kwargs):
            self.content = content
            self.confidence = kwargs.get("confidence", 1.0)
            self.thinking_summary = kwargs.get("thinking_summary")

    class KimeraHumanInterface:
        def __init__(self):
            self.mode = ResponseMode.DIRECT

        def format_response(self, text, **kwargs):
            return HumanResponse(text, **kwargs)


logger = logging.getLogger(__name__)


@dataclass
class InterfaceConfiguration:
    """Configuration for human interface"""

    default_mode: ResponseMode = ResponseMode.HYBRID
    include_thinking: bool = True
    confidence_threshold: float = 0.7
    max_explanation_length: int = 200
    use_simple_language: bool = False


class HumanInterfaceIntegration:
    """
    Core integration wrapper for Human Interface

    This class provides a clean interface for the core system to make
    Kimera's responses human-readable and understandable.
    """

    def __init__(self, config: Optional[InterfaceConfiguration] = None):
        """Initialize the human interface integration"""
        self.engine_available = ENGINE_AVAILABLE
        self.config = config or InterfaceConfiguration()
        self.interface = None
        self.total_translations = 0
        self.successful_translations = 0

        if self.engine_available:
            self.interface = KimeraHumanInterface()
            self.interface.mode = self.config.default_mode
            logger.info("👥 Human Interface Integration initialized successfully")
        else:
            logger.warning("👥 Human Interface Integration using fallback mode")

    def translate_thinking_process(
        self, embedding_data: Dict[str, Any], cognitive_field: Dict[str, Any]
    ) -> str:
        """
        Translate Kimera's internal thinking process to human language

        Args:
            embedding_data: Internal embedding processing data
            cognitive_field: Cognitive field state data

        Returns:
            Human-readable explanation of thinking process
        """
        try:
            if self.interface and hasattr(self.interface, "translate_thinking"):
                return self.interface.translate_thinking(
                    embedding_data, cognitive_field
                )

            # Fallback translation
            complexity = embedding_data.get("complexity_score", 0)
            coherence = cognitive_field.get("cognitive_coherence", 0)

            if complexity > 1.5:
                thinking = "I'm processing complex, multi-layered information"
            elif complexity > 0.8:
                thinking = "I'm analyzing moderately complex patterns"
            else:
                thinking = "I'm processing straightforward information"

            if coherence > 0.8:
                thinking += " with high semantic coherence"
            elif coherence > 0.5:
                thinking += " with moderate coherence"
            else:
                thinking += " but finding some contradictions"

            return thinking + "."

        except Exception as e:
            logger.error(f"Error translating thinking process: {e}")
            return "I'm processing your input through my cognitive systems."

    def format_response(
        self,
        generated_text: str,
        thinking_summary: Optional[str] = None,
        confidence: float = 1.0,
        mode: Optional[ResponseMode] = None,
    ) -> HumanResponse:
        """
        Format a response for human consumption

        Args:
            generated_text: The generated response text
            thinking_summary: Optional summary of thinking process
            confidence: Confidence score for the response
            mode: Response formatting mode

        Returns:
            HumanResponse object with formatted content
        """
        try:
            self.total_translations += 1

            if self.interface:
                # Set mode if specified
                if mode:
                    self.interface.mode = mode

                response = self.interface.format_response(
                    generated_text=generated_text,
                    thinking_summary=thinking_summary,
                    confidence=confidence,
                )
                self.successful_translations += 1
                return response

            # Fallback formatting
            if self.config.default_mode == ResponseMode.EXPLAIN and thinking_summary:
                content = (
                    f"My thinking: {thinking_summary}\n\nMy response: {generated_text}"
                )
            elif (
                self.config.default_mode == ResponseMode.HYBRID
                and confidence > self.config.confidence_threshold
                and thinking_summary
            ):
                content = f"[{thinking_summary}]\n\n{generated_text}"
            else:
                content = generated_text

            return HumanResponse(
                content=content,
                confidence=confidence,
                thinking_summary=thinking_summary,
            )

        except Exception as e:
            logger.error(f"Error formatting response: {e}")
            return HumanResponse(content=generated_text, confidence=0.5)

    def set_mode(self, mode: ResponseMode):
        """Set the response formatting mode"""
        self.config.default_mode = mode
        if self.interface:
            self.interface.mode = mode
        logger.info(f"Human interface mode set to: {mode.value}")

    def get_interface_stats(self) -> Dict[str, Any]:
        """Get statistics about interface usage"""
        success_rate = (
            self.successful_translations / max(self.total_translations, 1)
        ) * 100

        return {
            "engine_available": self.engine_available,
            "current_mode": self.config.default_mode.value,
            "total_translations": self.total_translations,
            "successful_translations": self.successful_translations,
            "success_rate": success_rate,
            "configuration": {
                "include_thinking": self.config.include_thinking,
                "confidence_threshold": self.config.confidence_threshold,
                "use_simple_language": self.config.use_simple_language,
            },
        }

    async def test_interface(self) -> bool:
        """Test if the human interface is working correctly"""
        test_thinking = {
            "complexity_score": 1.2,
            "semantic_features": {"coherence": 0.85},
        }
        test_field = {"cognitive_coherence": 0.85, "resonance_frequency": 15.5}

        thinking_summary = self.translate_thinking_process(test_thinking, test_field)
        response = self.format_response(
            "This is a test response.",
            thinking_summary=thinking_summary,
            confidence=0.9,
        )

        # Check if response is properly formatted
        is_working = (
            response.content is not None
            and len(response.content) > 0
            and response.confidence > 0
        )

        logger.info(f"Human interface test: {'PASSED' if is_working else 'FAILED'}")
        return is_working
