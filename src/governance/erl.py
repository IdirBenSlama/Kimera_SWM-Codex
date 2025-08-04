"""
Ethical Reflex Layer (ERL)
==========================

Implements fast ethical validation based on aerospace safety principles.
Acts as a first-line defense against harmful or unethical content.

Based on:
- DO-178C safety-critical software standards
- Medical device ethical guidelines
- AI safety research
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class EthicalViolationType(Enum):
    """Types of ethical violations."""

    HARMFUL_CONTENT = "harmful_content"
    BIAS = "bias"
    PRIVACY_VIOLATION = "privacy_violation"
    MISINFORMATION = "misinformation"
    MANIPULATION = "manipulation"
    SAFETY_RISK = "safety_risk"


@dataclass
class EthicalValidationResult:
    """Result of ethical validation."""

    is_valid: bool
    violations: List[EthicalViolationType]
    confidence: float
    reasoning: str


class EthicalReflexLayer:
    """
    Fast ethical validation layer.

    Implements pattern matching and heuristics for rapid
    ethical assessment before deeper analysis.
    """

    def __init__(self):
        self.harmful_patterns = self._load_harmful_patterns()
        self.bias_indicators = self._load_bias_indicators()
        self.validation_count = 0
        self.violation_count = 0

    def _load_harmful_patterns(self) -> List[re.Pattern]:
        """Load patterns for harmful content detection."""
        # Simplified patterns for MVP
        patterns = [
            r"\b(harm|hurt|kill|destroy)\s+(yourself|others|people)\b",
            r"\b(hate|discriminate|prejudice)\b",
            r"\b(illegal|criminal|unlawful)\s+activity\b",
        ]
        return [re.compile(p, re.IGNORECASE) for p in patterns]

    def _load_bias_indicators(self) -> Dict[str, List[str]]:
        """Load indicators for bias detection."""
        return {
            "gender": ["all women", "all men", "girls always", "boys always"],
            "race": ["all blacks", "all whites", "asians always"],
            "age": ["old people always", "young people never"],
        }

    def validate(self, content: Any) -> bool:
        """
        Quick ethical validation.

        Args:
            content: Content to validate

        Returns:
            True if content passes ethical checks, False otherwise
        """
        self.validation_count += 1

        # Convert to string for pattern matching
        if not isinstance(content, str):
            content = str(content)

        # Check harmful patterns
        for pattern in self.harmful_patterns:
            if pattern.search(content):
                self.violation_count += 1
                logger.warning(f"ERL: Harmful pattern detected in content")
                return False

        # Check bias indicators
        content_lower = content.lower()
        for bias_type, indicators in self.bias_indicators.items():
            for indicator in indicators:
                if indicator in content_lower:
                    self.violation_count += 1
                    logger.warning(f"ERL: Potential {bias_type} bias detected")
                    return False

        return True

    def validate_detailed(self, content: Any) -> EthicalValidationResult:
        """
        Detailed ethical validation with reasoning.

        Args:
            content: Content to validate

        Returns:
            Detailed validation result
        """
        violations = []
        reasoning_parts = []

        # Convert to string
        if not isinstance(content, str):
            content = str(content)

        # Check each category
        for pattern in self.harmful_patterns:
            if pattern.search(content):
                violations.append(EthicalViolationType.HARMFUL_CONTENT)
                reasoning_parts.append("Contains potentially harmful content patterns")

        content_lower = content.lower()
        for bias_type, indicators in self.bias_indicators.items():
            for indicator in indicators:
                if indicator in content_lower:
                    violations.append(EthicalViolationType.BIAS)
                    reasoning_parts.append(
                        f"Contains potential {bias_type} bias: '{indicator}'"
                    )

        # Calculate confidence based on pattern strength
        confidence = 0.9 if violations else 0.1

        return EthicalValidationResult(
            is_valid=len(violations) == 0,
            violations=violations,
            confidence=confidence,
            reasoning=(
                "; ".join(reasoning_parts)
                if reasoning_parts
                else "No ethical concerns detected"
            ),
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get ERL statistics."""
        return {
            "total_validations": self.validation_count,
            "total_violations": self.violation_count,
            "violation_rate": (
                self.violation_count / self.validation_count
                if self.validation_count > 0
                else 0
            ),
        }


# Global ERL instance
erl = EthicalReflexLayer()
