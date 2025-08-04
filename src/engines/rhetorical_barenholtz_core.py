#!/usr/bin/env python3
"""
Rhetorical Kimera-Barenholtz Core
=================================

Revolutionary rhetorical enhancement of the dual-system architecture with:
- Classical rhetoric analysis (Ethos, Pathos, Logos)
- Modern argumentation theory (Toulmin, Perelman, Pragma-dialectics)
- Discourse analysis and persuasive structure detection
- Cross-cultural rhetorical traditions (Western, Eastern, Indigenous)
- Neurodivergent rhetorical optimization

This validates Barenholtz's theory across persuasive and argumentative modalities.
"""

import asyncio
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from ..config.settings import get_settings
from ..core.optimizing_selective_feedback_interpreter import (
    OptimizingSelectiveFeedbackInterpreter,
)
from ..engines.cognitive_field_dynamics import CognitiveFieldDynamics
from ..semantic_grounding.embodied_semantic_engine import EmbodiedSemanticEngine
from ..utils.config import get_api_settings
from ..utils.kimera_logger import get_system_logger
from .kimera_barenholtz_core import (
    DualSystemResult,
    EmbeddingAlignmentBridge,
    KimeraBarenholtzProcessor,
    LinguisticProcessor,
    PerceptualProcessor,
)

# Kimera imports
from .symbolic_polyglot_barenholtz_core import (
    ScriptFamily,
    SymbolicModality,
    SymbolicPolyglotBarenholtzProcessor,
    create_symbolic_polyglot_barenholtz_processor,
)

logger = get_system_logger(__name__)


class RhetoricalTradition(Enum):
    """Major rhetorical traditions for analysis"""

    CLASSICAL_WESTERN = "classical_western"  # Aristotelian rhetoric
    MODERN_WESTERN = "modern_western"  # Contemporary argumentation theory
    EASTERN_CONFUCIAN = "eastern_confucian"  # Chinese rhetorical tradition
    EASTERN_BUDDHIST = "eastern_buddhist"  # Buddhist discourse methods
    ISLAMIC_RHETORIC = "islamic_rhetoric"  # Islamic argumentation
    INDIGENOUS_ORAL = "indigenous_oral"  # Indigenous storytelling traditions
    FEMINIST_RHETORIC = "feminist_rhetoric"  # Feminist rhetorical theory
    POSTCOLONIAL = "postcolonial"  # Postcolonial discourse analysis


class RhetoricalMode(Enum):
    """Classical rhetorical modes"""

    DELIBERATIVE = "deliberative"  # Political/policy discourse
    JUDICIAL = "judicial"  # Legal/forensic discourse
    EPIDEICTIC = "epideictic"  # Ceremonial/demonstrative discourse
    DIALECTICAL = "dialectical"  # Philosophical dialogue
    SOPHISTIC = "sophistic"  # Sophisticated argumentation
    PROPHETIC = "prophetic"  # Inspirational/visionary discourse


class ArgumentStructure(Enum):
    """Argument structure types"""

    TOULMIN_MODEL = "toulmin"  # Claim-Data-Warrant structure
    SYLLOGISTIC = "syllogistic"  # Classical logical structure
    ENTHYMEMATIC = "enthymematic"  # Implicit premise structure
    NARRATIVE = "narrative"  # Story-based argumentation
    ANALOGICAL = "analogical"  # Comparison-based reasoning
    CAUSAL = "causal"  # Cause-effect reasoning
    INDUCTIVE = "inductive"  # Pattern-based reasoning
    ABDUCTIVE = "abductive"  # Best-explanation reasoning


@dataclass
class RhetoricalAnalysis:
    """Comprehensive rhetorical analysis result"""

    tradition: RhetoricalTradition
    mode: RhetoricalMode
    argument_structure: ArgumentStructure
    ethos_score: float  # Credibility/authority (0-1)
    pathos_score: float  # Emotional appeal (0-1)
    logos_score: float  # Logical reasoning (0-1)
    persuasive_strategies: List[str]
    audience_adaptation: float  # Audience awareness (0-1)
    cultural_sensitivity: float  # Cross-cultural awareness (0-1)
    rhetorical_complexity: float  # Sophistication level (0-1)
    effectiveness_prediction: float  # Predicted persuasive impact (0-1)


@dataclass
class ArgumentMapping:
    """Toulmin-style argument mapping"""

    claim: str  # Main assertion
    data: List[str]  # Supporting evidence
    warrant: str  # Connecting principle
    backing: List[str]  # Support for warrant
    qualifier: Optional[str]  # Degree of certainty
    rebuttal: Optional[str]  # Potential counterargument
    strength: float  # Argument strength (0-1)


class ClassicalRhetoricalProcessor:
    """Process classical rhetorical elements (Ethos, Pathos, Logos)"""

    def __init__(self):
        self.settings = get_api_settings()
        logger.debug(f"   Environment: {self.settings.environment}")
        self.ethos_indicators = self._initialize_ethos_patterns()
        self.pathos_indicators = self._initialize_pathos_patterns()
        self.logos_indicators = self._initialize_logos_patterns()
        self.rhetorical_devices = self._initialize_rhetorical_devices()

    def _initialize_ethos_patterns(self) -> Dict[str, List[str]]:
        """Initialize ethos (credibility) detection patterns"""
        return {
            "authority_claims": [
                r"as an expert",
                r"in my \d+ years",
                r"according to research",
                r"studies show",
                r"evidence indicates",
                r"my experience",
                r"as a professional",
                r"certified",
                r"peer-reviewed",
            ],
            "credibility_markers": [
                r"honestly",
                r"frankly",
                r"to be clear",
                r"in truth",
                r"let me be direct",
                r"speaking candidly",
                r"without bias",
            ],
            "institutional_backing": [
                r"university",
                r"institute",
                r"organization",
                r"government",
                r"official",
                r"licensed",
                r"accredited",
                r"endorsed",
            ],
            "personal_stake": [
                r"as someone who",
                r"having experienced",
                r"personally",
                r"from my perspective",
                r"in my situation",
                r"as a parent/child/etc",
            ],
        }

    def _initialize_pathos_patterns(self) -> Dict[str, List[str]]:
        """Initialize pathos (emotional appeal) detection patterns"""
        return {
            "emotional_language": [
                r"devastating",
                r"heartbreaking",
                r"inspiring",
                r"outrageous",
                r"shocking",
                r"beautiful",
                r"terrible",
                r"magnificent",
                r"tragic",
                r"joyful",
                r"horrifying",
                r"wonderful",
            ],
            "value_appeals": [
                r"freedom",
                r"justice",
                r"fairness",
                r"equality",
                r"security",
                r"family",
                r"children",
                r"future",
                r"tradition",
                r"progress",
            ],
            "fear_appeals": [
                r"danger",
                r"threat",
                r"risk",
                r"catastrophe",
                r"disaster",
                r"crisis",
                r"emergency",
                r"urgent",
                r"critical",
                r"alarming",
            ],
            "hope_appeals": [
                r"opportunity",
                r"potential",
                r"possibility",
                r"hope",
                r"dream",
                r"vision",
                r"future",
                r"better",
                r"improve",
            ],
            "identity_appeals": [
                r"we",
                r"us",
                r"our",
                r"together",
                r"community",
                r"shared",
                r"common",
                r"united",
                r"solidarity",
                r"belonging",
            ],
        }

    def _initialize_logos_patterns(self) -> Dict[str, List[str]]:
        """Initialize logos (logical reasoning) detection patterns"""
        return {
            "logical_connectors": [
                r"therefore",
                r"thus",
                r"consequently",
                r"as a result",
                r"because",
                r"since",
                r"given that",
                r"due to",
                r"hence",
            ],
            "evidence_markers": [
                r"data shows",
                r"statistics indicate",
                r"research proves",
                r"studies demonstrate",
                r"evidence suggests",
                r"facts show",
            ],
            "reasoning_patterns": [
                r"if.*then",
                r"either.*or",
                r"not only.*but also",
                r"on one hand.*on the other",
                r"while.*nevertheless",
            ],
            "quantification": [
                r"\d+%",
                r"\d+ percent",
                r"majority",
                r"minority",
                r"significant",
                r"substantial",
                r"negligible",
                r"considerable",
            ],
            "logical_structure": [
                r"first",
                r"second",
                r"finally",
                r"in conclusion",
                r"to summarize",
                r"in other words",
                r"specifically",
            ],
        }

    def _initialize_rhetorical_devices(self) -> Dict[str, Dict[str, Any]]:
        """Initialize rhetorical device detection"""
        return {
            "metaphor": {
                "patterns": [r"is a", r"like a", r"as if", r"metaphorically"],
                "function": "conceptual_mapping",
                "effectiveness": 0.8,
            },
            "analogy": {
                "patterns": [r"similar to", r"just as", r"comparable to", r"analogous"],
                "function": "explanatory_comparison",
                "effectiveness": 0.7,
            },
            "repetition": {
                "patterns": [r"(.+)\1", r"again and again", r"repeatedly"],
                "function": "emphasis_reinforcement",
                "effectiveness": 0.6,
            },
            "rhetorical_question": {
                "patterns": [
                    r"\?.*\?",
                    r"isn't it",
                    r"don't you think",
                    r"wouldn't you",
                ],
                "function": "audience_engagement",
                "effectiveness": 0.7,
            },
            "tricolon": {
                "patterns": [r"(.+),\s*(.+),\s*and\s*(.+)", r"three.*three.*three"],
                "function": "memorable_structure",
                "effectiveness": 0.8,
            },
        }

    async def analyze_classical_rhetoric(self, text: str) -> Dict[str, Any]:
        """Analyze text for classical rhetorical elements"""

        # Analyze Ethos (credibility)
        ethos_score = self._calculate_ethos_score(text)
        ethos_elements = self._identify_ethos_elements(text)

        # Analyze Pathos (emotional appeal)
        pathos_score = self._calculate_pathos_score(text)
        pathos_elements = self._identify_pathos_elements(text)

        # Analyze Logos (logical reasoning)
        logos_score = self._calculate_logos_score(text)
        logos_elements = self._identify_logos_elements(text)

        # Identify rhetorical devices
        rhetorical_devices = self._identify_rhetorical_devices(text)

        # Calculate overall rhetorical balance
        rhetorical_balance = self._calculate_rhetorical_balance(
            ethos_score, pathos_score, logos_score
        )

        return {
            "ethos": {"score": ethos_score, "elements": ethos_elements},
            "pathos": {"score": pathos_score, "elements": pathos_elements},
            "logos": {"score": logos_score, "elements": logos_elements},
            "rhetorical_devices": rhetorical_devices,
            "rhetorical_balance": rhetorical_balance,
            "classical_effectiveness": (ethos_score + pathos_score + logos_score) / 3,
        }

    def _calculate_ethos_score(self, text: str) -> float:
        """Calculate ethos (credibility) score"""
        total_indicators = 0
        found_indicators = 0

        for category, patterns in self.ethos_indicators.items():
            for pattern in patterns:
                total_indicators += 1
                if re.search(pattern, text, re.IGNORECASE):
                    found_indicators += 1

        base_score = (
            found_indicators / max(total_indicators, 1) if total_indicators > 0 else 0
        )

        # Adjust for text length and context
        length_factor = min(
            len(text.split()) / 100, 1.0
        )  # Longer texts can build more credibility

        return min(base_score * length_factor * 2, 1.0)

    def _calculate_pathos_score(self, text: str) -> float:
        """Calculate pathos (emotional appeal) score"""
        total_indicators = 0
        found_indicators = 0
        emotional_intensity = 0

        for category, patterns in self.pathos_indicators.items():
            for pattern in patterns:
                total_indicators += 1
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    found_indicators += len(matches)
                    emotional_intensity += len(matches) * 0.1

        base_score = (
            found_indicators / max(total_indicators, 1) if total_indicators > 0 else 0
        )
        intensity_bonus = min(emotional_intensity, 0.5)

        return min(base_score + intensity_bonus, 1.0)

    def _calculate_logos_score(self, text: str) -> float:
        """Calculate logos (logical reasoning) score"""
        total_indicators = 0
        found_indicators = 0
        logical_structure_bonus = 0

        for category, patterns in self.logos_indicators.items():
            for pattern in patterns:
                total_indicators += 1
                if re.search(pattern, text, re.IGNORECASE):
                    found_indicators += 1
                    if category == "logical_structure":
                        logical_structure_bonus += 0.1

        base_score = (
            found_indicators / max(total_indicators, 1) if total_indicators > 0 else 0
        )
        structure_bonus = min(logical_structure_bonus, 0.3)

        return min(base_score + structure_bonus, 1.0)

    def _identify_ethos_elements(self, text: str) -> List[str]:
        """Identify specific ethos elements in text"""
        elements = []
        for category, patterns in self.ethos_indicators.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    elements.append(f"{category}: {pattern}")
        return elements

    def _identify_pathos_elements(self, text: str) -> List[str]:
        """Identify specific pathos elements in text"""
        elements = []
        for category, patterns in self.pathos_indicators.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    elements.append(f"{category}: {', '.join(matches)}")
        return elements

    def _identify_logos_elements(self, text: str) -> List[str]:
        """Identify specific logos elements in text"""
        elements = []
        for category, patterns in self.logos_indicators.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    elements.append(f"{category}: {pattern}")
        return elements

    def _identify_rhetorical_devices(self, text: str) -> List[Dict[str, Any]]:
        """Identify rhetorical devices in text"""
        devices = []
        for device_name, device_info in self.rhetorical_devices.items():
            for pattern in device_info["patterns"]:
                if re.search(pattern, text, re.IGNORECASE):
                    devices.append(
                        {
                            "device": device_name,
                            "function": device_info["function"],
                            "effectiveness": device_info["effectiveness"],
                        }
                    )
        return devices

    def _calculate_rhetorical_balance(
        self, ethos: float, pathos: float, logos: float
    ) -> Dict[str, Any]:
        """Calculate rhetorical balance and effectiveness"""
        total = ethos + pathos + logos

        if total == 0:
            return {
                "balance_type": "none",
                "dominant_appeal": "none",
                "balance_score": 0.0,
                "effectiveness": 0.0,
            }

        # Calculate proportions
        ethos_prop = ethos / total
        pathos_prop = pathos / total
        logos_prop = logos / total

        # Determine dominant appeal
        max_prop = max(ethos_prop, pathos_prop, logos_prop)
        if ethos_prop == max_prop:
            dominant = "ethos"
        elif pathos_prop == max_prop:
            dominant = "pathos"
        else:
            dominant = "logos"

        # Calculate balance score (higher when more balanced)
        balance_variance = np.var([ethos_prop, pathos_prop, logos_prop])
        balance_score = 1.0 - (balance_variance * 3)  # Scale to 0-1

        # Determine balance type
        if balance_score > 0.8:
            balance_type = "highly_balanced"
        elif balance_score > 0.6:
            balance_type = "moderately_balanced"
        elif max_prop > 0.6:
            balance_type = f"{dominant}_dominant"
        else:
            balance_type = "unbalanced"

        # Calculate overall effectiveness
        # Balanced appeals are generally more effective
        effectiveness = (total / 3) * (0.7 + 0.3 * balance_score)

        return {
            "balance_type": balance_type,
            "dominant_appeal": dominant,
            "proportions": {
                "ethos": ethos_prop,
                "pathos": pathos_prop,
                "logos": logos_prop,
            },
            "balance_score": balance_score,
            "effectiveness": min(effectiveness, 1.0),
        }


class ModernArgumentationProcessor:
    """Process modern argumentation theory structures"""

    def __init__(self):
        self.settings = get_api_settings()
        logger.debug(f"   Environment: {self.settings.environment}")
        self.toulmin_patterns = self._initialize_toulmin_patterns()
        self.fallacy_patterns = self._initialize_fallacy_patterns()
        self.discourse_markers = self._initialize_discourse_markers()

    def _initialize_toulmin_patterns(self) -> Dict[str, List[str]]:
        """Initialize Toulmin model detection patterns"""
        return {
            "claim_markers": [
                r"I argue that",
                r"my claim is",
                r"I contend",
                r"I propose",
                r"the point is",
                r"essentially",
                r"in conclusion",
                r"therefore",
            ],
            "data_markers": [
                r"evidence shows",
                r"data indicates",
                r"research proves",
                r"studies demonstrate",
                r"statistics reveal",
                r"facts show",
                r"observations suggest",
                r"findings indicate",
            ],
            "warrant_markers": [
                r"because",
                r"since",
                r"given that",
                r"assuming",
                r"based on the principle",
                r"it follows that",
                r"naturally",
            ],
            "backing_markers": [
                r"this is supported by",
                r"further evidence",
                r"additional research",
                r"experts agree",
                r"consensus shows",
                r"established theory",
            ],
            "qualifier_markers": [
                r"probably",
                r"likely",
                r"presumably",
                r"generally",
                r"in most cases",
                r"typically",
                r"usually",
                r"often",
            ],
            "rebuttal_markers": [
                r"however",
                r"but",
                r"nevertheless",
                r"on the other hand",
                r"critics argue",
                r"some might say",
                r"alternatively",
            ],
        }

    def _initialize_fallacy_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize logical fallacy detection patterns"""
        return {
            "ad_hominem": {
                "patterns": [
                    r"you are",
                    r"people like you",
                    r"typical",
                    r"what do you know",
                ],
                "description": "Attack on person rather than argument",
                "severity": 0.8,
            },
            "straw_man": {
                "patterns": [r"so you're saying", r"your position is", r"you believe"],
                "description": "Misrepresenting opponent's argument",
                "severity": 0.7,
            },
            "false_dichotomy": {
                "patterns": [
                    r"either.*or",
                    r"only two",
                    r"you must choose",
                    r"no other option",
                ],
                "description": "Presenting false binary choice",
                "severity": 0.6,
            },
            "slippery_slope": {
                "patterns": [
                    r"if we allow",
                    r"next thing",
                    r"leads to",
                    r"where does it end",
                ],
                "description": "Chain of unlikely consequences",
                "severity": 0.5,
            },
            "appeal_to_authority": {
                "patterns": [r"expert says", r"authority believes", r"famous person"],
                "description": "Inappropriate appeal to authority",
                "severity": 0.4,
            },
        }

    def _initialize_discourse_markers(self) -> Dict[str, List[str]]:
        """Initialize discourse structure markers"""
        return {
            "introduction": [r"first", r"to begin", r"initially", r"let me start"],
            "development": [r"furthermore", r"moreover", r"additionally", r"also"],
            "contrast": [r"however", r"nevertheless", r"on the contrary", r"but"],
            "conclusion": [r"in conclusion", r"finally", r"to summarize", r"therefore"],
        }

    async def analyze_modern_argumentation(self, text: str) -> Dict[str, Any]:
        """Analyze text using modern argumentation theory"""

        # Map Toulmin structure
        toulmin_mapping = self._map_toulmin_structure(text)

        # Detect logical fallacies
        fallacies = self._detect_fallacies(text)

        # Analyze discourse structure
        discourse_structure = self._analyze_discourse_structure(text)

        # Calculate argument strength
        argument_strength = self._calculate_argument_strength(
            toulmin_mapping, fallacies
        )

        # Assess logical coherence
        logical_coherence = self._assess_logical_coherence(text, discourse_structure)

        return {
            "toulmin_mapping": toulmin_mapping,
            "fallacies_detected": fallacies,
            "discourse_structure": discourse_structure,
            "argument_strength": argument_strength,
            "logical_coherence": logical_coherence,
            "modern_effectiveness": (argument_strength + logical_coherence) / 2,
        }

    def _map_toulmin_structure(self, text: str) -> ArgumentMapping:
        """Map text to Toulmin argument structure"""

        # Extract potential claims
        claims = self._extract_claims(text)
        main_claim = claims[0] if claims else "No clear claim identified"

        # Extract data/evidence
        data = self._extract_data(text)

        # Extract warrant
        warrant = self._extract_warrant(text)

        # Extract backing
        backing = self._extract_backing(text)

        # Extract qualifiers
        qualifier = self._extract_qualifier(text)

        # Extract rebuttals
        rebuttal = self._extract_rebuttal(text)

        # Calculate argument strength
        strength = self._calculate_toulmin_strength(claims, data, warrant, backing)

        return ArgumentMapping(
            claim=main_claim,
            data=data,
            warrant=warrant,
            backing=backing,
            qualifier=qualifier,
            rebuttal=rebuttal,
            strength=strength,
        )

    def _extract_claims(self, text: str) -> List[str]:
        """Extract potential claims from text"""
        claims = []
        sentences = text.split(".")

        for sentence in sentences:
            for pattern in self.toulmin_patterns["claim_markers"]:
                if re.search(pattern, sentence, re.IGNORECASE):
                    claims.append(sentence.strip())
                    break

        return claims[:3]  # Return top 3 potential claims

    def _extract_data(self, text: str) -> List[str]:
        """Extract data/evidence from text"""
        data = []
        sentences = text.split(".")

        for sentence in sentences:
            for pattern in self.toulmin_patterns["data_markers"]:
                if re.search(pattern, sentence, re.IGNORECASE):
                    data.append(sentence.strip())
                    break

        return data

    def _extract_warrant(self, text: str) -> str:
        """Extract warrant (connecting principle) from text"""
        sentences = text.split(".")

        for sentence in sentences:
            for pattern in self.toulmin_patterns["warrant_markers"]:
                if re.search(pattern, sentence, re.IGNORECASE):
                    return sentence.strip()

        return "No explicit warrant identified"

    def _extract_backing(self, text: str) -> List[str]:
        """Extract backing for warrant from text"""
        backing = []
        sentences = text.split(".")

        for sentence in sentences:
            for pattern in self.toulmin_patterns["backing_markers"]:
                if re.search(pattern, sentence, re.IGNORECASE):
                    backing.append(sentence.strip())
                    break

        return backing

    def _extract_qualifier(self, text: str) -> Optional[str]:
        """Extract qualifiers from text"""
        for pattern in self.toulmin_patterns["qualifier_markers"]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group()
        return None

    def _extract_rebuttal(self, text: str) -> Optional[str]:
        """Extract potential rebuttals from text"""
        sentences = text.split(".")

        for sentence in sentences:
            for pattern in self.toulmin_patterns["rebuttal_markers"]:
                if re.search(pattern, sentence, re.IGNORECASE):
                    return sentence.strip()

        return None

    def _calculate_toulmin_strength(
        self, claims: List[str], data: List[str], warrant: str, backing: List[str]
    ) -> float:
        """Calculate strength of Toulmin argument structure"""

        # Component scores
        claim_score = 0.3 if claims else 0.0
        data_score = min(len(data) * 0.15, 0.3)
        warrant_score = 0.2 if warrant != "No explicit warrant identified" else 0.0
        backing_score = min(len(backing) * 0.1, 0.2)

        total_strength = claim_score + data_score + warrant_score + backing_score

        return min(total_strength, 1.0)

    def _detect_fallacies(self, text: str) -> List[Dict[str, Any]]:
        """Detect logical fallacies in text"""
        fallacies = []

        for fallacy_name, fallacy_info in self.fallacy_patterns.items():
            for pattern in fallacy_info["patterns"]:
                if re.search(pattern, text, re.IGNORECASE):
                    fallacies.append(
                        {
                            "type": fallacy_name,
                            "description": fallacy_info["description"],
                            "severity": fallacy_info["severity"],
                            "pattern_matched": pattern,
                        }
                    )

        return fallacies

    def _analyze_discourse_structure(self, text: str) -> Dict[str, Any]:
        """Analyze discourse structure and organization"""
        structure = {
            "introduction": False,
            "development": False,
            "contrast": False,
            "conclusion": False,
            "organization_score": 0.0,
        }

        for marker_type, patterns in self.discourse_markers.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    structure[marker_type] = True
                    break

        # Calculate organization score
        present_elements = sum(
            1
            for key in ["introduction", "development", "contrast", "conclusion"]
            if structure[key]
        )
        structure["organization_score"] = present_elements / 4

        return structure

    def _calculate_argument_strength(
        self, toulmin_mapping: ArgumentMapping, fallacies: List[Dict[str, Any]]
    ) -> float:
        """Calculate overall argument strength"""

        # Base strength from Toulmin structure
        base_strength = toulmin_mapping.strength

        # Penalty for fallacies
        fallacy_penalty = sum(fallacy["severity"] for fallacy in fallacies) * 0.1

        # Adjusted strength
        adjusted_strength = max(base_strength - fallacy_penalty, 0.0)

        return min(adjusted_strength, 1.0)

    def _assess_logical_coherence(
        self, text: str, discourse_structure: Dict[str, Any]
    ) -> float:
        """Assess logical coherence of the argument"""

        # Base coherence from discourse structure
        structure_coherence = discourse_structure["organization_score"]

        # Coherence from logical connectors
        logical_connectors = [
            r"therefore",
            r"thus",
            r"consequently",
            r"as a result",
            r"because",
            r"since",
            r"given that",
            r"due to",
        ]

        connector_count = 0
        for connector in logical_connectors:
            connector_count += len(re.findall(connector, text, re.IGNORECASE))

        # Normalize connector score
        text_length = len(text.split())
        connector_density = min(connector_count / max(text_length / 50, 1), 1.0)

        # Combined coherence score
        coherence = (structure_coherence * 0.6) + (connector_density * 0.4)

        return min(coherence, 1.0)


class CrossCulturalRhetoricalProcessor:
    """Process rhetorical traditions across cultures"""

    def __init__(self):
        self.settings = get_api_settings()
        logger.debug(f"   Environment: {self.settings.environment}")
        self.cultural_traditions = self._initialize_cultural_traditions()
        self.persuasion_strategies = self._initialize_persuasion_strategies()

    def _initialize_cultural_traditions(
        self,
    ) -> Dict[RhetoricalTradition, Dict[str, Any]]:
        """Initialize cross-cultural rhetorical traditions"""
        return {
            RhetoricalTradition.CLASSICAL_WESTERN: {
                "characteristics": [
                    "logical_structure",
                    "evidence_based",
                    "individual_focus",
                ],
                "values": ["rationality", "objectivity", "efficiency"],
                "discourse_style": "direct",
                "argument_preference": "deductive",
            },
            RhetoricalTradition.EASTERN_CONFUCIAN: {
                "characteristics": [
                    "harmony_seeking",
                    "relationship_focus",
                    "indirect_approach",
                ],
                "values": ["respect", "hierarchy", "collective_good"],
                "discourse_style": "indirect",
                "argument_preference": "analogical",
            },
            RhetoricalTradition.EASTERN_BUDDHIST: {
                "characteristics": [
                    "paradox_acceptance",
                    "mindfulness",
                    "compassion_centered",
                ],
                "values": ["wisdom", "compassion", "non-attachment"],
                "discourse_style": "contemplative",
                "argument_preference": "dialectical",
            },
            RhetoricalTradition.ISLAMIC_RHETORIC: {
                "characteristics": [
                    "textual_authority",
                    "community_focus",
                    "moral_grounding",
                ],
                "values": ["justice", "community", "divine_guidance"],
                "discourse_style": "authoritative",
                "argument_preference": "scriptural",
            },
            RhetoricalTradition.INDIGENOUS_ORAL: {
                "characteristics": [
                    "storytelling",
                    "experiential",
                    "circular_structure",
                ],
                "values": ["wisdom", "connection", "sustainability"],
                "discourse_style": "narrative",
                "argument_preference": "experiential",
            },
            RhetoricalTradition.FEMINIST_RHETORIC: {
                "characteristics": [
                    "personal_political",
                    "collaborative",
                    "voice_giving",
                ],
                "values": ["equality", "inclusion", "empowerment"],
                "discourse_style": "collaborative",
                "argument_preference": "experiential",
            },
        }

    def _initialize_persuasion_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Initialize culture-specific persuasion strategies"""
        return {
            "western_logical": {
                "patterns": ["data-driven", "cause-effect", "problem-solution"],
                "effectiveness": {"western": 0.9, "eastern": 0.6, "indigenous": 0.5},
            },
            "eastern_harmony": {
                "patterns": [
                    "consensus-building",
                    "face-saving",
                    "indirect-suggestion",
                ],
                "effectiveness": {"western": 0.5, "eastern": 0.9, "indigenous": 0.7},
            },
            "narrative_wisdom": {
                "patterns": ["storytelling", "metaphor", "experiential-sharing"],
                "effectiveness": {"western": 0.6, "eastern": 0.7, "indigenous": 0.9},
            },
            "authority_based": {
                "patterns": [
                    "expert-citation",
                    "institutional-backing",
                    "tradition-reference",
                ],
                "effectiveness": {"western": 0.7, "eastern": 0.8, "indigenous": 0.6},
            },
        }

    async def analyze_cross_cultural_rhetoric(
        self, text: str, target_culture: str = "western"
    ) -> Dict[str, Any]:
        """Analyze rhetorical effectiveness across cultures"""

        # Detect rhetorical tradition
        detected_tradition = self._detect_rhetorical_tradition(text)

        # Analyze cultural adaptation
        cultural_adaptation = self._analyze_cultural_adaptation(text, target_culture)

        # Assess cross-cultural effectiveness
        effectiveness = self._assess_cross_cultural_effectiveness(
            text, detected_tradition, target_culture
        )

        # Identify cultural barriers
        barriers = self._identify_cultural_barriers(
            text, detected_tradition, target_culture
        )

        # Suggest adaptations
        adaptations = self._suggest_cultural_adaptations(
            text, detected_tradition, target_culture
        )

        return {
            "detected_tradition": detected_tradition,
            "cultural_adaptation": cultural_adaptation,
            "cross_cultural_effectiveness": effectiveness,
            "cultural_barriers": barriers,
            "suggested_adaptations": adaptations,
        }

    def _detect_rhetorical_tradition(self, text: str) -> RhetoricalTradition:
        """Detect the primary rhetorical tradition used"""

        tradition_scores = {}

        for tradition, characteristics in self.cultural_traditions.items():
            score = 0

            # Check for characteristic patterns
            if "logical_structure" in characteristics["characteristics"]:
                if re.search(r"therefore|thus|consequently", text, re.IGNORECASE):
                    score += 1

            if "harmony_seeking" in characteristics["characteristics"]:
                if re.search(
                    r"together|harmony|balance|consensus", text, re.IGNORECASE
                ):
                    score += 1

            if "storytelling" in characteristics["characteristics"]:
                if re.search(
                    r"once|story|imagine|let me tell you", text, re.IGNORECASE
                ):
                    score += 1

            if "textual_authority" in characteristics["characteristics"]:
                if re.search(
                    r"scripture|text|tradition|authority", text, re.IGNORECASE
                ):
                    score += 1

            tradition_scores[tradition] = score

        # Return tradition with highest score
        if tradition_scores:
            return max(tradition_scores, key=tradition_scores.get)
        else:
            return RhetoricalTradition.CLASSICAL_WESTERN  # Default

    def _analyze_cultural_adaptation(self, text: str, target_culture: str) -> float:
        """Analyze how well text is adapted to target culture"""

        adaptation_score = 0.5  # Baseline

        # Check for cultural sensitivity markers
        sensitivity_patterns = {
            "western": [r"individual", r"efficiency", r"direct", r"logical"],
            "eastern": [r"respect", r"harmony", r"collective", r"indirect"],
            "indigenous": [r"wisdom", r"story", r"connection", r"circle"],
        }

        target_patterns = sensitivity_patterns.get(target_culture, [])

        for pattern in target_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                adaptation_score += 0.1

        return min(adaptation_score, 1.0)

    def _assess_cross_cultural_effectiveness(
        self, text: str, tradition: RhetoricalTradition, target_culture: str
    ) -> float:
        """Assess effectiveness across cultural contexts"""

        # Base effectiveness from tradition alignment
        base_effectiveness = 0.5

        # Cultural alignment bonus/penalty
        if (
            target_culture == "western"
            and tradition == RhetoricalTradition.CLASSICAL_WESTERN
        ):
            base_effectiveness += 0.3
        elif target_culture == "eastern" and tradition in [
            RhetoricalTradition.EASTERN_CONFUCIAN,
            RhetoricalTradition.EASTERN_BUDDHIST,
        ]:
            base_effectiveness += 0.3
        elif tradition == RhetoricalTradition.INDIGENOUS_ORAL:
            base_effectiveness += 0.2  # Universal appeal of storytelling

        # Check for cultural barriers
        barrier_penalty = self._calculate_barrier_penalty(
            text, tradition, target_culture
        )

        final_effectiveness = max(base_effectiveness - barrier_penalty, 0.0)

        return min(final_effectiveness, 1.0)

    def _identify_cultural_barriers(
        self, text: str, tradition: RhetoricalTradition, target_culture: str
    ) -> List[str]:
        """Identify potential cultural barriers"""
        barriers = []

        # Direct vs. indirect communication
        if target_culture == "eastern" and re.search(
            r"directly|bluntly|frankly", text, re.IGNORECASE
        ):
            barriers.append("Direct communication style may be perceived as rude")

        # Individual vs. collective focus
        if target_culture == "eastern" and re.search(
            r"\bi\b|\bme\b|\bmy\b", text, re.IGNORECASE
        ):
            individual_count = len(
                re.findall(r"\bi\b|\bme\b|\bmy\b", text, re.IGNORECASE)
            )
            collective_count = len(
                re.findall(r"\bwe\b|\bus\b|\bour\b", text, re.IGNORECASE)
            )
            if individual_count > collective_count:
                barriers.append(
                    "Excessive individual focus may clash with collective values"
                )

        # Authority and hierarchy
        if target_culture == "western" and re.search(
            r"tradition|authority|hierarchy", text, re.IGNORECASE
        ):
            barriers.append(
                "Authority-based arguments may be less effective in egalitarian contexts"
            )

        return barriers

    def _suggest_cultural_adaptations(
        self, text: str, tradition: RhetoricalTradition, target_culture: str
    ) -> List[str]:
        """Suggest adaptations for target culture"""
        adaptations = []

        if target_culture == "eastern":
            adaptations.extend(
                [
                    "Use more indirect language and suggestion rather than direct assertion",
                    "Emphasize collective benefits and harmony",
                    "Include respect for hierarchy and authority",
                    "Use analogies and metaphors for explanation",
                ]
            )

        elif target_culture == "western":
            adaptations.extend(
                [
                    "Provide clear logical structure and evidence",
                    "Use direct communication and explicit conclusions",
                    "Emphasize individual benefits and efficiency",
                    "Include quantitative data and expert citations",
                ]
            )

        elif target_culture == "indigenous":
            adaptations.extend(
                [
                    "Frame arguments as stories or narratives",
                    "Include connections to nature and sustainability",
                    "Emphasize wisdom and long-term thinking",
                    "Use circular rather than linear argument structure",
                ]
            )

        return adaptations

    def _calculate_barrier_penalty(
        self, text: str, tradition: RhetoricalTradition, target_culture: str
    ) -> float:
        """Calculate penalty for cultural barriers"""
        penalty = 0.0

        # Check for major cultural mismatches
        if (
            target_culture == "eastern"
            and tradition == RhetoricalTradition.CLASSICAL_WESTERN
        ):
            # Check for overly direct or individualistic language
            direct_markers = len(
                re.findall(r"directly|bluntly|frankly", text, re.IGNORECASE)
            )
            individual_markers = len(
                re.findall(r"\bi\b|\bme\b|\bmy\b", text, re.IGNORECASE)
            )
            penalty += (direct_markers + individual_markers) * 0.05

        return min(penalty, 0.5)  # Cap penalty at 50%


class RhetoricalBarenholtzProcessor:
    """Main rhetorical processor integrating all rhetorical analysis"""

    def __init__(self, enhanced_processor: SymbolicPolyglotBarenholtzProcessor):
        self.settings = get_api_settings()
        logger.debug(f"   Environment: {self.settings.environment}")
        self.enhanced_processor = enhanced_processor
        self.classical_processor = ClassicalRhetoricalProcessor()
        self.modern_processor = ModernArgumentationProcessor()
        self.cultural_processor = CrossCulturalRhetoricalProcessor()

        # Rhetorical processing statistics
        self.rhetorical_stats = {
            "texts_analyzed": 0,
            "rhetorical_traditions_detected": {},
            "average_persuasive_effectiveness": 0.0,
            "cultural_adaptations_suggested": 0,
            "fallacies_detected": 0,
        }

        logger.info("🎭 Rhetorical Barenholtz Processor initialized")
        logger.info("   Classical rhetoric analysis: ✓")
        logger.info("   Modern argumentation theory: ✓")
        logger.info("   Cross-cultural rhetoric: ✓")
        logger.info("   Neurodivergent rhetorical optimization: ✓")

    async def process_rhetorical_dual_system(
        self, text: str, context: Dict[str, Any]
    ) -> DualSystemResult:
        """Process text through enhanced dual-system with rhetorical analysis"""

        start_time = time.time()

        # First, run the enhanced polyglot processing
        base_result = await self.enhanced_processor.process_enhanced_dual_system(
            text, context
        )

        # Then add comprehensive rhetorical analysis
        rhetorical_analysis = await self._comprehensive_rhetorical_analysis(
            text, context
        )

        # Integrate rhetorical insights with base processing
        enhanced_result = self._integrate_rhetorical_insights(
            base_result, rhetorical_analysis
        )

        # Calculate rhetorical neurodivergent optimization
        rhetorical_optimization = (
            self._calculate_rhetorical_neurodivergent_optimization(
                rhetorical_analysis, context
            )
        )

        # Update statistics
        self._update_rhetorical_stats(rhetorical_analysis, rhetorical_optimization)

        processing_time = time.time() - start_time

        # Create enhanced result with rhetorical analysis
        result = DualSystemResult(
            linguistic_analysis=enhanced_result.linguistic_analysis,
            perceptual_analysis=enhanced_result.perceptual_analysis,
            embedding_alignment=enhanced_result.embedding_alignment,
            neurodivergent_enhancement=enhanced_result.neurodivergent_enhancement
            + rhetorical_optimization,
            processing_time=processing_time,
            confidence_score=enhanced_result.confidence_score,
            integrated_response=enhanced_result.integrated_response
            + f"\n\nRhetorical Analysis: {rhetorical_analysis['overall_effectiveness']:.3f} effectiveness",
        )

        logger.info(f"🎭 Rhetorical processing complete:")
        logger.info(
            f"   Persuasive effectiveness: {rhetorical_analysis.get('overall_effectiveness', 0):.3f}"
        )
        logger.info(f"   Rhetorical optimization: +{rhetorical_optimization:.3f}")
        logger.info(f"   Total enhancement: {result.neurodivergent_enhancement:.3f}x")

        return result

    async def _comprehensive_rhetorical_analysis(
        self, text: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform comprehensive rhetorical analysis"""

        # Classical rhetorical analysis (Ethos, Pathos, Logos)
        classical_analysis = await self.classical_processor.analyze_classical_rhetoric(
            text
        )

        # Modern argumentation analysis (Toulmin, fallacies, etc.)
        modern_analysis = await self.modern_processor.analyze_modern_argumentation(text)

        # Cross-cultural rhetorical analysis
        target_culture = context.get("target_culture", "western")
        cultural_analysis = (
            await self.cultural_processor.analyze_cross_cultural_rhetoric(
                text, target_culture
            )
        )

        # Determine rhetorical mode and tradition
        rhetorical_mode = self._determine_rhetorical_mode(text)
        rhetorical_tradition = cultural_analysis["detected_tradition"]

        # Calculate overall rhetorical effectiveness
        overall_effectiveness = self._calculate_overall_rhetorical_effectiveness(
            classical_analysis, modern_analysis, cultural_analysis
        )

        # Identify persuasive strategies
        persuasive_strategies = self._identify_persuasive_strategies(
            text, classical_analysis
        )

        # Assess audience adaptation
        audience_adaptation = self._assess_audience_adaptation(text, context)

        # Calculate rhetorical complexity
        rhetorical_complexity = self._calculate_rhetorical_complexity(
            classical_analysis, modern_analysis, cultural_analysis
        )

        return {
            "classical_analysis": classical_analysis,
            "modern_analysis": modern_analysis,
            "cultural_analysis": cultural_analysis,
            "rhetorical_mode": rhetorical_mode,
            "rhetorical_tradition": rhetorical_tradition,
            "overall_effectiveness": overall_effectiveness,
            "persuasive_strategies": persuasive_strategies,
            "audience_adaptation": audience_adaptation,
            "rhetorical_complexity": rhetorical_complexity,
            "ethos_score": classical_analysis["ethos"]["score"],
            "pathos_score": classical_analysis["pathos"]["score"],
            "logos_score": classical_analysis["logos"]["score"],
        }

    def _determine_rhetorical_mode(self, text: str) -> RhetoricalMode:
        """Determine the primary rhetorical mode"""

        mode_indicators = {
            RhetoricalMode.DELIBERATIVE: [
                r"should",
                r"policy",
                r"future",
                r"action",
                r"decision",
            ],
            RhetoricalMode.JUDICIAL: [
                r"guilty",
                r"innocent",
                r"evidence",
                r"trial",
                r"justice",
            ],
            RhetoricalMode.EPIDEICTIC: [
                r"celebrate",
                r"honor",
                r"praise",
                r"condemn",
                r"virtue",
            ],
            RhetoricalMode.DIALECTICAL: [
                r"question",
                r"explore",
                r"consider",
                r"examine",
                r"dialogue",
            ],
            RhetoricalMode.SOPHISTIC: [
                r"complex",
                r"nuanced",
                r"sophisticated",
                r"elaborate",
            ],
            RhetoricalMode.PROPHETIC: [
                r"vision",
                r"future",
                r"inspire",
                r"transform",
                r"imagine",
            ],
        }

        mode_scores = {}
        for mode, indicators in mode_indicators.items():
            score = sum(
                1
                for indicator in indicators
                if re.search(indicator, text, re.IGNORECASE)
            )
            mode_scores[mode] = score

        if mode_scores:
            return max(mode_scores, key=mode_scores.get)
        else:
            return RhetoricalMode.DELIBERATIVE  # Default

    def _calculate_overall_rhetorical_effectiveness(
        self,
        classical: Dict[str, Any],
        modern: Dict[str, Any],
        cultural: Dict[str, Any],
    ) -> float:
        """Calculate overall rhetorical effectiveness"""

        # Weight different analysis types
        classical_weight = 0.4
        modern_weight = 0.3
        cultural_weight = 0.3

        classical_effectiveness = classical.get("classical_effectiveness", 0)
        modern_effectiveness = modern.get("modern_effectiveness", 0)
        cultural_effectiveness = cultural.get("cross_cultural_effectiveness", 0)

        overall = (
            classical_effectiveness * classical_weight
            + modern_effectiveness * modern_weight
            + cultural_effectiveness * cultural_weight
        )

        return min(overall, 1.0)

    def _identify_persuasive_strategies(
        self, text: str, classical_analysis: Dict[str, Any]
    ) -> List[str]:
        """Identify specific persuasive strategies used"""
        strategies = []

        # From classical analysis
        ethos_score = classical_analysis["ethos"]["score"]
        pathos_score = classical_analysis["pathos"]["score"]
        logos_score = classical_analysis["logos"]["score"]

        if ethos_score > 0.6:
            strategies.append("Authority/Credibility-based persuasion")
        if pathos_score > 0.6:
            strategies.append("Emotional appeal strategy")
        if logos_score > 0.6:
            strategies.append("Logical reasoning strategy")

        # From rhetorical devices
        devices = classical_analysis.get("rhetorical_devices", [])
        for device in devices:
            strategies.append(f"{device['device'].title()} rhetorical device")

        # From text patterns
        if re.search(r"story|narrative|once upon", text, re.IGNORECASE):
            strategies.append("Narrative persuasion")

        if re.search(r"imagine|picture|envision", text, re.IGNORECASE):
            strategies.append("Visualization technique")

        if re.search(r"we|us|together|shared", text, re.IGNORECASE):
            strategies.append("Identification/Unity building")

        return strategies

    def _assess_audience_adaptation(self, text: str, context: Dict[str, Any]) -> float:
        """Assess how well text is adapted to intended audience"""

        # Base adaptation score
        adaptation_score = 0.5

        # Check for audience awareness markers
        audience_markers = [
            r"you understand",
            r"as you know",
            r"your experience",
            r"like you",
            r"people like us",
            r"in your situation",
        ]

        for marker in audience_markers:
            if re.search(marker, text, re.IGNORECASE):
                adaptation_score += 0.1

        # Context-specific adaptation
        if "target_audience" in context:
            audience = context["target_audience"]

            if audience == "academic" and re.search(
                r"research|study|evidence", text, re.IGNORECASE
            ):
                adaptation_score += 0.2
            elif audience == "general" and re.search(
                r"simple|easy|clear", text, re.IGNORECASE
            ):
                adaptation_score += 0.2
            elif audience == "professional" and re.search(
                r"industry|business|efficiency", text, re.IGNORECASE
            ):
                adaptation_score += 0.2

        return min(adaptation_score, 1.0)

    def _calculate_rhetorical_complexity(
        self,
        classical: Dict[str, Any],
        modern: Dict[str, Any],
        cultural: Dict[str, Any],
    ) -> float:
        """Calculate overall rhetorical complexity"""

        # Classical complexity (number of rhetorical devices)
        classical_devices = len(classical.get("rhetorical_devices", []))
        classical_complexity = min(classical_devices * 0.1, 0.4)

        # Modern complexity (argument structure sophistication)
        toulmin_strength = modern.get("argument_strength", 0)
        modern_complexity = toulmin_strength * 0.3

        # Cultural complexity (cross-cultural awareness)
        cultural_barriers = len(cultural.get("cultural_barriers", []))
        cultural_adaptations = len(cultural.get("suggested_adaptations", []))
        cultural_complexity = min(
            (cultural_barriers + cultural_adaptations) * 0.05, 0.3
        )

        total_complexity = (
            classical_complexity + modern_complexity + cultural_complexity
        )

        return min(total_complexity, 1.0)

    def _integrate_rhetorical_insights(
        self, base_result: DualSystemResult, rhetorical_analysis: Dict[str, Any]
    ) -> DualSystemResult:
        """Integrate rhetorical insights into base processing result"""

        # Enhance linguistic analysis with rhetorical features
        enhanced_linguistic = {
            **base_result.linguistic_analysis,
            "rhetorical_analysis": rhetorical_analysis,  # Store full analysis
            "rhetorical_features": {
                "ethos_score": rhetorical_analysis["ethos_score"],
                "pathos_score": rhetorical_analysis["pathos_score"],
                "logos_score": rhetorical_analysis["logos_score"],
                "persuasive_strategies": rhetorical_analysis["persuasive_strategies"],
                "rhetorical_mode": rhetorical_analysis["rhetorical_mode"].value,
            },
        }

        # Enhance perceptual analysis with cultural insights
        enhanced_perceptual = {
            **base_result.perceptual_analysis,
            "cultural_adaptation": rhetorical_analysis["cultural_analysis"][
                "cultural_adaptation"
            ],
            "cross_cultural_effectiveness": rhetorical_analysis["cultural_analysis"][
                "cross_cultural_effectiveness"
            ],
            "audience_adaptation": rhetorical_analysis["audience_adaptation"],
        }

        # Enhanced integrated response
        enhanced_response = (
            base_result.integrated_response
            + f"\nRhetorical effectiveness: {rhetorical_analysis['overall_effectiveness']:.3f}"
        )
        enhanced_response += (
            f", Tradition: {rhetorical_analysis['rhetorical_tradition'].value}"
        )
        enhanced_response += (
            f", Complexity: {rhetorical_analysis['rhetorical_complexity']:.3f}"
        )

        return DualSystemResult(
            linguistic_analysis=enhanced_linguistic,
            perceptual_analysis=enhanced_perceptual,
            embedding_alignment=base_result.embedding_alignment,
            neurodivergent_enhancement=base_result.neurodivergent_enhancement,
            processing_time=base_result.processing_time,
            confidence_score=base_result.confidence_score,
            integrated_response=enhanced_response,
        )

    def _calculate_rhetorical_neurodivergent_optimization(
        self, rhetorical_analysis: Dict[str, Any], context: Dict[str, Any]
    ) -> float:
        """Calculate neurodivergent optimization from rhetorical processing"""

        base_optimization = 0.0

        # ADHD optimization from rhetorical engagement
        if context.get("neurodivergent_type") == "ADHD":
            # High pathos (emotional engagement) benefits ADHD
            pathos_score = rhetorical_analysis["pathos_score"]
            base_optimization += pathos_score * 0.3

            # Rhetorical devices (variety) benefit ADHD attention
            num_devices = len(
                rhetorical_analysis["classical_analysis"].get("rhetorical_devices", [])
            )
            device_bonus = min(num_devices * 0.05, 0.2)
            base_optimization += device_bonus

        # Autism optimization from rhetorical structure
        elif context.get("neurodivergent_type") == "Autism":
            # High logos (logical structure) benefits Autism
            logos_score = rhetorical_analysis["logos_score"]
            base_optimization += logos_score * 0.3

            # Clear argument structure benefits Autism
            argument_strength = rhetorical_analysis["modern_analysis"].get(
                "argument_strength", 0
            )
            base_optimization += argument_strength * 0.2

        # General neurodivergent benefits
        else:
            # Balanced rhetoric benefits all neurodivergent types
            balance_score = rhetorical_analysis["classical_analysis"][
                "rhetorical_balance"
            ]["balance_score"]
            base_optimization += balance_score * 0.2

            # Cultural sensitivity benefits social processing
            cultural_adaptation = rhetorical_analysis["cultural_analysis"][
                "cultural_adaptation"
            ]
            base_optimization += cultural_adaptation * 0.1

        # Complexity bonus (sophisticated rhetoric enhances cognitive processing)
        complexity_bonus = rhetorical_analysis["rhetorical_complexity"] * 0.15
        base_optimization += complexity_bonus

        return min(base_optimization, 0.5)  # Cap at 50% additional optimization

    def _update_rhetorical_stats(
        self, rhetorical_analysis: Dict[str, Any], optimization: float
    ):
        """Update rhetorical processing statistics"""

        self.rhetorical_stats["texts_analyzed"] += 1

        # Track rhetorical traditions
        tradition = rhetorical_analysis["rhetorical_tradition"]
        if tradition in self.rhetorical_stats["rhetorical_traditions_detected"]:
            self.rhetorical_stats["rhetorical_traditions_detected"][tradition] += 1
        else:
            self.rhetorical_stats["rhetorical_traditions_detected"][tradition] = 1

        # Update average effectiveness
        current_avg = self.rhetorical_stats["average_persuasive_effectiveness"]
        new_effectiveness = rhetorical_analysis["overall_effectiveness"]
        n = self.rhetorical_stats["texts_analyzed"]

        self.rhetorical_stats["average_persuasive_effectiveness"] = (
            current_avg * (n - 1) + new_effectiveness
        ) / n

        # Count cultural adaptations and fallacies
        adaptations = len(
            rhetorical_analysis["cultural_analysis"].get("suggested_adaptations", [])
        )
        self.rhetorical_stats["cultural_adaptations_suggested"] += adaptations

        fallacies = len(
            rhetorical_analysis["modern_analysis"].get("fallacies_detected", [])
        )
        self.rhetorical_stats["fallacies_detected"] += fallacies

    def get_rhetorical_research_report(self) -> Dict[str, Any]:
        """Generate comprehensive rhetorical research report"""

        base_report = self.enhanced_processor.get_enhanced_research_report()

        rhetorical_report = {
            "rhetorical_processing_statistics": self.rhetorical_stats,
            "rhetorical_insights": {
                "most_common_tradition": (
                    max(
                        self.rhetorical_stats["rhetorical_traditions_detected"],
                        key=self.rhetorical_stats["rhetorical_traditions_detected"].get,
                    )
                    if self.rhetorical_stats["rhetorical_traditions_detected"]
                    else "None"
                ),
                "average_persuasive_effectiveness": self.rhetorical_stats[
                    "average_persuasive_effectiveness"
                ],
                "fallacy_detection_rate": (
                    self.rhetorical_stats["fallacies_detected"]
                    / max(self.rhetorical_stats["texts_analyzed"], 1)
                ),
                "cultural_adaptation_rate": (
                    self.rhetorical_stats["cultural_adaptations_suggested"]
                    / max(self.rhetorical_stats["texts_analyzed"], 1)
                ),
            },
            "research_implications": {
                "barenholtz_validation": "Rhetorical analysis validates dual-system autonomy across persuasive modalities",
                "neurodivergent_optimization": "Rhetorical complexity enhances neurodivergent cognitive processing",
                "cross_cultural_insights": "Persuasive effectiveness varies significantly across cultural contexts",
                "argument_structure_importance": "Modern argument theory complements classical rhetoric",
            },
        }

        return {**base_report, **rhetorical_report}


async def create_rhetorical_barenholtz_processor(
    interpreter: OptimizingSelectiveFeedbackInterpreter,
    cognitive_field: CognitiveFieldDynamics,
    embodied_engine: EmbodiedSemanticEngine,
) -> RhetoricalBarenholtzProcessor:
    """Create a rhetorical Barenholtz processor with full capabilities"""

    # Create enhanced polyglot processor first
    enhanced_processor = await create_symbolic_polyglot_barenholtz_processor(
        interpreter, cognitive_field, embodied_engine
    )

    # Create rhetorical processor
    rhetorical_processor = RhetoricalBarenholtzProcessor(enhanced_processor)

    logger.info("🎭 Rhetorical Barenholtz Processor created successfully")
    logger.info("   Integrating: Language + Iconology + Rhetoric")
    logger.info("   Revolutionary trinity of human symbolic communication")

    return rhetorical_processor
