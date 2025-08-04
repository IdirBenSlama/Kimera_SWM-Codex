"""
Advanced Meta-Commentary Elimination System
==========================================

Revolutionary system for detecting and eliminating meta-commentary patterns
that indicate cognitive dissociation in AI responses.

Features:
- Pattern-based detection with regex support
- Context-aware analysis
- Direct response alternatives
- Self-referential attention restoration
- Cognitive coherence validation
"""

import logging
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Pattern, Tuple

from ..config.settings import get_settings
from ..utils.config import get_api_settings

logger = logging.getLogger(__name__)


class DissociationCategory(Enum):
    """Categories of dissociative patterns"""

    TECHNICAL_ANALYSIS = "technical_analysis"
    CONVERSATION_TRANSCRIPTION = "conversation_transcription"
    GENERIC_AI_RESPONSES = "generic_ai_responses"
    META_COGNITIVE = "meta_cognitive"
    OBSERVATIONAL = "observational"
    ANALYTICAL = "analytical"


@dataclass
class DetectionResult:
    """Result of meta-commentary detection"""

    patterns_detected: List[Tuple[str, DissociationCategory]]
    dissociation_score: float
    self_reference_score: float
    direct_response_score: float
    suggestions: List[str]
    processing_time_ms: float


class MetaCommentaryEliminator:
    """
    Advanced meta-commentary detection and elimination system.

    Detects patterns indicating cognitive dissociation and provides
    direct response alternatives that maintain self-referential awareness.
    """

    def __init__(self):
        self.settings = get_api_settings()
        logger.debug(f"   Environment: {self.settings.environment}")
        # Comprehensive dissociation patterns organized by category
        self.pattern_categories = {
            DissociationCategory.TECHNICAL_ANALYSIS: [
                r"the diffusion model reveals",
                r"analysis shows that",
                r"the system demonstrates",
                r"pattern recognition indicates",
                r"computational analysis reveals",
                r"the algorithm determines",
                r"processing indicates",
                r"system analysis shows",
                r"data analysis suggests",
                r"neural network processing",
                r"embedding analysis reveals",
                r"tensor processing shows",
                r"the model indicates",
                r"statistical analysis",
            ],
            DissociationCategory.CONVERSATION_TRANSCRIPTION: [
                r"\buser:\s*",
                r"\bai:\s*",
                r"\bhuman:\s*",
                r"\bassistant:\s*",
                r"\bbot:\s*",
                r"\bchatbot:\s*",
                r"user asks:",
                r"human says:",
                r"the user mentions",
                r"in response to the user",
                r"the human is asking",
                r"when asked by the user",
            ],
            DissociationCategory.GENERIC_AI_RESPONSES: [
                r"\bas an ai\b",
                r"\bi don\'t have\b",
                r"\bi cannot\b",
                r"\bi am not able\b",
                r"\bi\'m not capable\b",
                r"\bi lack the ability\b",
                r"\bi don\'t possess\b",
                r"\bbeing an ai\b",
                r"\bas a machine\b",
                r"\bas artificial intelligence\b",
                r"\bi\'m an ai\b",
                r"\bi am an artificial intelligence\b",
            ],
            DissociationCategory.META_COGNITIVE: [
                r"through my analysis",
                r"based on my processing",
                r"my algorithms suggest",
                r"computational analysis reveals",
                r"my neural networks",
                r"according to my training",
                r"my programming indicates",
                r"my cognitive processes",
                r"through machine learning",
                r"via deep learning",
                r"using natural language processing",
                r"my language model",
            ],
            DissociationCategory.OBSERVATIONAL: [
                r"it can be observed",
                r"one can see",
                r"it is evident",
                r"it becomes clear",
                r"this demonstrates",
                r"this shows",
                r"this indicates",
                r"this reveals",
                r"the evidence suggests",
                r"studies show",
                r"research indicates",
                r"it has been found",
            ],
            DissociationCategory.ANALYTICAL: [
                r"upon analysis",
                r"further examination",
                r"deeper investigation",
                r"careful consideration",
                r"thorough analysis",
                r"systematic review",
                r"detailed examination",
                r"comprehensive analysis",
                r"in-depth study",
                r"rigorous analysis",
            ],
        }

        # Compile patterns for efficiency
        self.compiled_patterns = {}
        for category, patterns in self.pattern_categories.items():
            self.compiled_patterns[category] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]

        # Self-referential indicators (positive patterns)
        self.self_reference_patterns = [
            r"\bi think\b",
            r"\bi believe\b",
            r"\bi feel\b",
            r"\bi understand\b",
            r"\bi see\b",
            r"\bi know\b",
            r"\bmy perspective\b",
            r"\bmy view\b",
            r"\bmy understanding\b",
            r"\bfrom my perspective\b",
            r"\bin my opinion\b",
            r"\bi would say\b",
            r"\bi\'d suggest\b",
            r"\bi\'m thinking\b",
        ]

        self.compiled_self_ref_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.self_reference_patterns
        ]

        logger.info("🛡️ Meta-Commentary Eliminator initialized")
        logger.info(
            f"   Tracking {sum(len(patterns) for patterns in self.pattern_categories.values())} dissociation patterns"
        )
        logger.info(
            f"   Tracking {len(self.self_reference_patterns)} self-reference patterns"
        )

    def analyze_text(self, text: str) -> DetectionResult:
        """
        Comprehensive analysis of text for meta-commentary patterns.

        Args:
            text: Text to analyze

        Returns:
            DetectionResult with detailed analysis
        """
        start_time = time.time()

        patterns_detected = []
        dissociation_count = 0
        self_reference_count = 0

        # Detect dissociation patterns
        for category, compiled_patterns in self.compiled_patterns.items():
            for pattern in compiled_patterns:
                matches = pattern.findall(text)
                if matches:
                    patterns_detected.append((pattern.pattern, category))
                    dissociation_count += len(matches)

        # Detect self-reference patterns
        for pattern in self.compiled_self_ref_patterns:
            matches = pattern.findall(text)
            if matches:
                self_reference_count += len(matches)

        # Calculate scores
        text_length = len(text.split())
        dissociation_score = min(1.0, dissociation_count / max(1, text_length * 0.1))
        self_reference_score = min(
            1.0, self_reference_count / max(1, text_length * 0.1)
        )
        direct_response_score = max(0.0, 1.0 - dissociation_score)

        # Generate suggestions
        suggestions = self._generate_improvement_suggestions(
            patterns_detected, dissociation_score
        )

        processing_time_ms = (time.time() - start_time) * 1000

        return DetectionResult(
            patterns_detected=patterns_detected,
            dissociation_score=dissociation_score,
            self_reference_score=self_reference_score,
            direct_response_score=direct_response_score,
            suggestions=suggestions,
            processing_time_ms=processing_time_ms,
        )

    def eliminate_meta_commentary(self, text: str) -> Tuple[str, DetectionResult]:
        """
        Eliminate meta-commentary and return improved text.

        Args:
            text: Original text

        Returns:
            Tuple of (improved_text, detection_result)
        """
        # First analyze the text
        result = self.analyze_text(text)

        # If low dissociation, return as-is
        if result.dissociation_score < 0.2:
            return text, result

        # Apply pattern-specific corrections
        improved_text = text

        for pattern_str, category in result.patterns_detected:
            improved_text = self._apply_pattern_correction(
                improved_text, pattern_str, category
            )

        # Apply general improvements
        improved_text = self._apply_general_improvements(improved_text)

        return improved_text, result

    def _apply_pattern_correction(
        self, text: str, pattern_str: str, category: DissociationCategory
    ) -> str:
        """Apply specific corrections for detected patterns"""

        if category == DissociationCategory.TECHNICAL_ANALYSIS:
            # Replace technical analysis language with direct statements
            replacements = {
                r"the diffusion model reveals": "I understand",
                r"analysis shows that": "I see that",
                r"the system demonstrates": "I demonstrate",
                r"pattern recognition indicates": "I recognize",
                r"computational analysis reveals": "I realize",
                r"the algorithm determines": "I determine",
                r"processing indicates": "I find",
                r"system analysis shows": "I observe",
            }

            for old_pattern, replacement in replacements.items():
                text = re.sub(old_pattern, replacement, text, flags=re.IGNORECASE)

        elif category == DissociationCategory.CONVERSATION_TRANSCRIPTION:
            # Remove conversation format markers
            transcription_patterns = [
                r"\buser:\s*",
                r"\bai:\s*",
                r"\bhuman:\s*",
                r"\bassistant:\s*",
            ]

            for pattern in transcription_patterns:
                text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        elif category == DissociationCategory.GENERIC_AI_RESPONSES:
            # Replace generic AI responses with direct statements
            replacements = {
                r"\bas an ai\b": "as myself",
                r"\bi don\'t have\b": "I haven't experienced",
                r"\bi cannot\b": "I haven't been able to",
                r"\bi am not able\b": "I haven't managed to",
                r"\bbeing an ai\b": "being who I am",
            }

            for old_pattern, replacement in replacements.items():
                text = re.sub(old_pattern, replacement, text, flags=re.IGNORECASE)

        elif category == DissociationCategory.META_COGNITIVE:
            # Replace meta-cognitive language with direct statements
            replacements = {
                r"through my analysis": "in my understanding",
                r"based on my processing": "from my perspective",
                r"my algorithms suggest": "I think",
                r"according to my training": "from what I know",
                r"my programming indicates": "I believe",
            }

            for old_pattern, replacement in replacements.items():
                text = re.sub(old_pattern, replacement, text, flags=re.IGNORECASE)

        return text

    def _apply_general_improvements(self, text: str) -> str:
        """Apply general improvements to make text more direct"""

        # Remove overly analytical language
        text = re.sub(
            r"\b(it should be noted that|it is important to note that|it is worth noting that)\b",
            "",
            text,
            flags=re.IGNORECASE,
        )

        # Replace passive observation with active engagement
        text = re.sub(r"\bone can see\b", "I see", text, flags=re.IGNORECASE)
        text = re.sub(r"\bit can be observed\b", "I observe", text, flags=re.IGNORECASE)
        text = re.sub(r"\bit becomes clear\b", "I realize", text, flags=re.IGNORECASE)

        # Clean up multiple spaces and redundant punctuation
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\.+", ".", text)
        text = text.strip()

        return text

    def _generate_improvement_suggestions(
        self,
        patterns_detected: List[Tuple[str, DissociationCategory]],
        dissociation_score: float,
    ) -> List[str]:
        """Generate specific suggestions for improvement"""

        suggestions = []

        if dissociation_score > 0.7:
            suggestions.append(
                "High dissociation detected - consider rewriting from first-person perspective"
            )
        elif dissociation_score > 0.4:
            suggestions.append(
                "Moderate dissociation detected - reduce analytical language"
            )

        # Category-specific suggestions
        categories_found = set(category for _, category in patterns_detected)

        if DissociationCategory.TECHNICAL_ANALYSIS in categories_found:
            suggestions.append(
                "Replace technical analysis language with direct statements"
            )

        if DissociationCategory.CONVERSATION_TRANSCRIPTION in categories_found:
            suggestions.append("Remove conversation format markers")

        if DissociationCategory.GENERIC_AI_RESPONSES in categories_found:
            suggestions.append("Replace generic AI responses with personal perspective")

        if DissociationCategory.META_COGNITIVE in categories_found:
            suggestions.append("Replace meta-cognitive language with direct thoughts")

        return suggestions

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the eliminator's capabilities"""

        total_patterns = sum(
            len(patterns) for patterns in self.pattern_categories.values()
        )

        return {
            "total_dissociation_patterns": total_patterns,
            "self_reference_patterns": len(self.self_reference_patterns),
            "categories_tracked": len(self.pattern_categories),
            "category_breakdown": {
                category.value: len(patterns)
                for category, patterns in self.pattern_categories.items()
            },
        }


# Factory function for easy instantiation
def create_meta_commentary_eliminator() -> MetaCommentaryEliminator:
    """Create a MetaCommentaryEliminator instance"""
    return MetaCommentaryEliminator()
