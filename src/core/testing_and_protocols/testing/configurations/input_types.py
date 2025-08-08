#!/usr/bin/env python3
"""
Input Types Configuration for Large-Scale Testing
=================================================

DO-178C Level A compliant input type definitions for comprehensive
cognitive system testing. Each input type represents a distinct
category of cognitive processing challenges.

Key Features:
- Six distinct input types covering cognitive spectrum
- Realistic test data generation
- Input validation and sanitization
- Performance characteristics per type

Author: KIMERA Development Team
Version: 1.0.0 (DO-178C Level A)
"""

import json
import random
import string
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from utils.kimera_exceptions import KimeraValidationError
from utils.kimera_logger import LogCategory, get_logger

logger = get_logger(__name__, LogCategory.SYSTEM)


class InputType(Enum):
    """
    Cognitive input types for systematic testing coverage

    Based on cognitive science research into different
    modalities of human information processing.
    """

    LINGUISTIC = "linguistic"  # Natural language processing
    PERCEPTUAL = "perceptual"  # Pattern recognition and sensory data
    MIXED = "mixed"  # Combined linguistic and perceptual
    CONCEPTUAL = "conceptual"  # Abstract reasoning and symbols
    SCIENTIFIC = "scientific"  # Mathematical and scientific computation
    ARTISTIC = "artistic"  # Creative and aesthetic processing


@dataclass
class InputCharacteristics:
    """Auto-generated class."""
    pass
    """Characteristics of an input type for testing"""

    complexity_factors: Dict[str, float]  # Factors affecting processing complexity
    typical_size_range: tuple  # (min_chars, max_chars)
    processing_requirements: Dict[str, bool]  # System requirements
    validation_patterns: List[str]  # Regex patterns for validation
    performance_expectations: Dict[str, float]  # Expected performance metrics
    error_probability: float  # Expected error rate


@dataclass
class InputSample:
    """Auto-generated class."""
    pass
    """Generated input sample for testing"""

    content: str
    input_type: InputType
    complexity_score: float
    metadata: Dict[str, Any]
    expected_processing_time: float
    validation_checksum: str
class InputGenerator:
    """Auto-generated class."""
    pass
    """
    Generator for realistic test inputs across all cognitive modalities

    Implements scientific rigor in test data generation:
    - Reproducible with seeds
    - Statistically valid distributions
    - Realistic cognitive challenges
    """

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed or 42
        random.seed(self.seed)
        np.random.seed(self.seed)

        self.characteristics = self._initialize_characteristics()
        self.sample_templates = self._load_sample_templates()

        logger.info("📝 Input Generator initialized (DO-178C Level A)")
        logger.info(f"   Seed: {self.seed}")
        logger.info(f"   Input types: {len(self.characteristics)}")

    def _initialize_characteristics(self) -> Dict[InputType, InputCharacteristics]:
        """Initialize characteristics for each input type"""
        return {
            InputType.LINGUISTIC: InputCharacteristics(
                complexity_factors={
                    "vocabulary_diversity": 0.3
                    "sentence_length": 0.2
                    "syntactic_complexity": 0.3
                    "semantic_ambiguity": 0.2
                },
                typical_size_range=(50, 2000),  # 50 to 2000 characters
                processing_requirements={
                    "nlp_engine": True
                    "semantic_analysis": True
                    "grammar_parsing": True
                    "context_memory": True
                },
                validation_patterns=[
                    r"^[a-zA-Z0-9\s\.,;:!?\-'\"()]+$",  # Standard text
                    r".*[.!?]$",  # Must end with punctuation
                ],
                performance_expectations={
                    "processing_time_per_char": 0.001,  # 1ms per character
                    "memory_per_word": 10,  # 10 bytes per word
                    "cpu_utilization": 30.0,  # 30% CPU utilization
                },
                error_probability=0.01,  # 1% error rate
            ),
            InputType.PERCEPTUAL: InputCharacteristics(
                complexity_factors={
                    "pattern_density": 0.4
                    "noise_level": 0.3
                    "spatial_complexity": 0.3
                },
                typical_size_range=(100, 5000),  # Pattern descriptions
                processing_requirements={
                    "pattern_recognition": True
                    "visual_processing": True
                    "feature_extraction": True
                    "gpu_acceleration": True
                },
                validation_patterns=[
                    r"pattern:|shape:|color:|texture:",  # Must contain pattern descriptors
                    r"coordinates:\s*\(\d+,\s*\d+\)",  # Must have coordinates
                ],
                performance_expectations={
                    "processing_time_per_char": 0.002,  # 2ms per character
                    "memory_per_pattern": 100,  # 100 bytes per pattern
                    "cpu_utilization": 60.0,  # 60% CPU utilization
                    "gpu_utilization": 40.0,  # 40% GPU utilization
                },
                error_probability=0.05,  # 5% error rate (more complex)
            ),
            InputType.MIXED: InputCharacteristics(
                complexity_factors={
                    "modality_integration": 0.4
                    "context_switching": 0.3
                    "cross_modal_coherence": 0.3
                },
                typical_size_range=(200, 3000),  # Combined content
                processing_requirements={
                    "nlp_engine": True
                    "pattern_recognition": True
                    "multimodal_fusion": True
                    "context_management": True
                },
                validation_patterns=[
                    r".*text:.*pattern:.*",  # Must have both text and pattern
                    r"coherence_score:\s*\d+\.\d+",  # Must have coherence metric
                ],
                performance_expectations={
                    "processing_time_per_char": 0.003,  # 3ms per character
                    "memory_per_element": 50,  # 50 bytes per element
                    "cpu_utilization": 70.0,  # 70% CPU utilization
                    "gpu_utilization": 30.0,  # 30% GPU utilization
                },
                error_probability=0.03,  # 3% error rate
            ),
            InputType.CONCEPTUAL: InputCharacteristics(
                complexity_factors={
                    "abstraction_level": 0.4
                    "logical_complexity": 0.3
                    "symbol_density": 0.3
                },
                typical_size_range=(100, 1500),  # Abstract concepts
                processing_requirements={
                    "symbolic_reasoning": True
                    "logical_inference": True
                    "concept_hierarchy": True
                    "working_memory": True
                },
                validation_patterns=[
                    r"concept:|relation:|property:",  # Conceptual descriptors
                    r"abstraction_level:\s*\d+",  # Abstraction level
                ],
                performance_expectations={
                    "processing_time_per_char": 0.004,  # 4ms per character
                    "memory_per_concept": 200,  # 200 bytes per concept
                    "cpu_utilization": 80.0,  # 80% CPU utilization
                },
                error_probability=0.04,  # 4% error rate
            ),
            InputType.SCIENTIFIC: InputCharacteristics(
                complexity_factors={
                    "mathematical_complexity": 0.5
                    "precision_requirements": 0.3
                    "computational_depth": 0.2
                },
                typical_size_range=(150, 2500),  # Scientific notation
                processing_requirements={
                    "mathematical_engine": True
                    "numerical_computation": True
                    "precision_arithmetic": True
                    "formula_parsing": True
                },
                validation_patterns=[
                    r"equation:|formula:|calculation:",  # Mathematical content
                    r"\d+\.\d+([eE][+-]?\d+)?",  # Scientific notation
                    r"units:\s*[a-zA-Z/\^0-9]+",  # Physical units
                ],
                performance_expectations={
                    "processing_time_per_char": 0.005,  # 5ms per character
                    "memory_per_operation": 500,  # 500 bytes per operation
                    "cpu_utilization": 90.0,  # 90% CPU utilization
                    "precision_digits": 15,  # 15 decimal precision
                },
                error_probability=0.02,  # 2% error rate (high precision)
            ),
            InputType.ARTISTIC: InputCharacteristics(
                complexity_factors={
                    "creative_novelty": 0.4
                    "aesthetic_complexity": 0.3
                    "emotional_depth": 0.3
                },
                typical_size_range=(80, 1800),  # Creative expressions
                processing_requirements={
                    "creative_analysis": True
                    "aesthetic_evaluation": True
                    "emotional_processing": True
                    "style_recognition": True
                },
                validation_patterns=[
                    r"style:|mood:|emotion:",  # Artistic descriptors
                    r"creativity_score:\s*\d+\.\d+",  # Creativity metric
                    r"aesthetic_value:\s*\d+\.\d+",  # Aesthetic metric
                ],
                performance_expectations={
                    "processing_time_per_char": 0.006,  # 6ms per character
                    "memory_per_element": 300,  # 300 bytes per element
                    "cpu_utilization": 50.0,  # 50% CPU utilization
                    "creativity_threshold": 0.7,  # 70% creativity threshold
                },
                error_probability=0.08,  # 8% error rate (subjective)
            ),
        }

    def _load_sample_templates(self) -> Dict[InputType, List[str]]:
        """Load templates for realistic input generation"""
        return {
            InputType.LINGUISTIC: [
                "Analyze the implications of quantum consciousness in cognitive architectures. How does the observer effect influence decision-making processes in artificial intelligence systems?",
                "The thermodynamic principles underlying cognitive coherence suggest that information entropy decreases as understanding increases. Evaluate this hypothesis through computational analysis.",
                "Consider the philosophical ramifications of dual-system processing in artificial general intelligence. What ethical frameworks should govern System 1 versus System 2 decision pathways?",
                "Examine the role of emergent properties in complex adaptive systems. How do simple rules generate sophisticated cognitive behaviors?",
                "The intersection of consciousness and computation raises fundamental questions about the nature of experience. Discuss the hard problem of consciousness in artificial systems.",
            ],
            InputType.PERCEPTUAL: [
                "pattern: fractal_spiral coordinates: (128, 256) complexity: 0.85 noise: 0.15 texture: smooth_gradient color: blue_to_green spatial_frequency: high",
                "pattern: hexagonal_lattice coordinates: (64, 128) complexity: 0.65 noise: 0.25 texture: crystalline color: silver_metallic spatial_frequency: medium",
                "pattern: wave_interference coordinates: (256, 512) complexity: 0.95 noise: 0.05 texture: ripple_effect color: spectrum_shift spatial_frequency: variable",
                "pattern: mandala_geometric coordinates: (192, 192) complexity: 0.75 noise: 0.20 texture: intricate_detail color: warm_palette spatial_frequency: mixed",
                "pattern: neural_network coordinates: (384, 384) complexity: 0.90 noise: 0.10 texture: connected_nodes color: electric_blue spatial_frequency: high",
            ],
            InputType.MIXED: [
                "text: The recursive nature of consciousness creates strange loops in cognitive processing. pattern: recursive_spiral coordinates: (160, 320) coherence_score: 0.88 modality_strength: {text: 0.6, visual: 0.4}",
                "text: Quantum entanglement in biological systems suggests non-local cognitive processes. pattern: entangled_particles coordinates: (240, 240) coherence_score: 0.92 modality_strength: {text: 0.7, visual: 0.3}",
                "text: The emergence of complexity from simple rules mirrors natural selection in ideas. pattern: cellular_automata coordinates: (128, 256) coherence_score: 0.85 modality_strength: {text: 0.5, visual: 0.5}",
                "text: Thermodynamic equilibrium in cognitive systems requires energy minimization. pattern: energy_landscape coordinates: (320, 160) coherence_score: 0.90 modality_strength: {text: 0.8, visual: 0.2}",
                "text: The holographic principle suggests reality is encoded on boundaries. pattern: holographic_projection coordinates: (256, 256) coherence_score: 0.94 modality_strength: {text: 0.6, visual: 0.4}",
            ],
            InputType.CONCEPTUAL: [
                "concept: consciousness relation: emerges_from property: information_integration abstraction_level: 4 logical_complexity: high symbol_density: 0.7",
                "concept: intelligence relation: depends_on property: adaptive_behavior abstraction_level: 3 logical_complexity: medium symbol_density: 0.6",
                "concept: reality relation: constructed_by property: observation abstraction_level: 5 logical_complexity: very_high symbol_density: 0.8",
                "concept: causality relation: implies property: temporal_ordering abstraction_level: 2 logical_complexity: low symbol_density: 0.4",
                "concept: emergence relation: transcends property: reductionism abstraction_level: 4 logical_complexity: high symbol_density: 0.75",
            ],
            InputType.SCIENTIFIC: [
                "equation: E = mc² calculation: energy_mass_equivalence units: joules precision: 1.602176634e-19 domain: physics complexity: fundamental",
                "formula: ∇²φ = 0 calculation: laplace_equation units: dimensionless precision: 1e-15 domain: mathematics complexity: partial_differential",
                "equation: ΔS ≥ 0 calculation: entropy_increase units: J/K precision: 1.380649e-23 domain: thermodynamics complexity: statistical",
                "formula: Ψ(x,t) = Ae^(i(kx-ωt)) calculation: wave_function units: √(1/length) precision: 6.62607015e-34 domain: quantum_mechanics complexity: complex_exponential",
                "equation: ∂u/∂t = α∇²u calculation: heat_equation units: temperature/time precision: 1e-12 domain: physics complexity: parabolic_pde",
            ],
            InputType.ARTISTIC: [
                "style: abstract_expressionism mood: contemplative emotion: transcendent creativity_score: 0.85 aesthetic_value: 0.78 novelty: high",
                "style: digital_surrealism mood: mysterious emotion: wonder creativity_score: 0.92 aesthetic_value: 0.84 novelty: very_high",
                "style: minimalist_geometry mood: serene emotion: peaceful creativity_score: 0.67 aesthetic_value: 0.89 novelty: medium",
                "style: neo_impressionism mood: vibrant emotion: joyful creativity_score: 0.74 aesthetic_value: 0.82 novelty: medium_high",
                "style: cyberpunk_aesthetic mood: intense emotion: rebellious creativity_score: 0.88 aesthetic_value: 0.76 novelty: high",
            ],
        }

    def generate_sample(
        self
        input_type: InputType
        complexity_level: str = "medium",
        custom_parameters: Optional[Dict[str, Any]] = None
    ) -> InputSample:
        """
        Generate a realistic input sample for testing

        Args:
            input_type: Type of input to generate
            complexity_level: Target complexity ("simple", "medium", "complex", "expert")
            custom_parameters: Optional custom generation parameters

        Returns:
            Generated input sample with metadata
        """
        characteristics = self.characteristics[input_type]
        templates = self.sample_templates[input_type]

        # Select base template
        base_template = random.choice(templates)

        # Modify for complexity level
        content = self._adjust_for_complexity(
            base_template, complexity_level, input_type
        )

        # Calculate complexity score
        complexity_score = self._calculate_complexity_score(content, input_type)

        # Generate metadata
        metadata = self._generate_metadata(
            input_type, complexity_level, custom_parameters
        )

        # Estimate processing time
        expected_time = self._estimate_processing_time(content, characteristics)

        # Generate validation checksum
        checksum = self._generate_checksum(content, input_type)

        sample = InputSample(
            content=content
            input_type=input_type
            complexity_score=complexity_score
            metadata=metadata
            expected_processing_time=expected_time
            validation_checksum=checksum
        )

        logger.debug(
            f"Generated {input_type.value} sample: {len(content)} chars, "
            f"complexity={complexity_score:.3f}, time={expected_time:.3f}s"
        )

        return sample

    def _adjust_for_complexity(
        self, template: str, complexity_level: str, input_type: InputType
    ) -> str:
        """Adjust template content for target complexity level"""

        if complexity_level == "simple":
            # Simplify by reducing length and complexity
            if input_type == InputType.LINGUISTIC:
                # Use first sentence only
                sentences = template.split(". ")
                return sentences[0] + "."
            elif input_type == InputType.SCIENTIFIC:
                # Use basic formulas
                return template.replace("∇²", "d²/dx²").replace("∂", "d")
            else:
                # Reduce parameter complexity
                return template.replace("high", "low").replace("complex", "simple")

        elif complexity_level == "expert":
            # Increase complexity by adding elements
            if input_type == InputType.LINGUISTIC:
                # Add philosophical depth
                return (
                    template
                    + " Furthermore, consider the meta-cognitive implications of this analysis within a broader framework of emergent intelligence."
                )
            elif input_type == InputType.SCIENTIFIC:
                # Add mathematical complexity
                return (
                    template
                    + " with boundary conditions ∂u/∂n = 0 and initial state u(x,0) = φ(x)"
                )
            else:
                # Increase parameter complexity
                return template.replace("medium", "very_high").replace("0.5", "0.95")

        # Medium and complex use template as-is with minor modifications
        return template

    def _calculate_complexity_score(self, content: str, input_type: InputType) -> float:
        """Calculate complexity score for generated content"""
        characteristics = self.characteristics[input_type]

        # Base metrics
        length_factor = len(content) / 1000.0  # Normalize by 1000 chars
        word_count = len(content.split())
        unique_words = len(set(content.lower().split()))

        # Lexical diversity
        lexical_diversity = unique_words / word_count if word_count > 0 else 0

        # Special pattern complexity
        special_chars = sum(1 for c in content if not c.isalnum() and not c.isspace())
        special_factor = special_chars / len(content) if len(content) > 0 else 0

        # Combine factors based on input type characteristics
        complexity_score = 0.0
        for factor, weight in characteristics.complexity_factors.items():
            if factor == "vocabulary_diversity":
                complexity_score += lexical_diversity * weight
            elif factor in [
                "pattern_density",
                "mathematical_complexity",
                "creative_novelty",
            ]:
                complexity_score += special_factor * weight
            else:
                # Generic complexity based on length and structure
                complexity_score += min(length_factor, 1.0) * weight

        return min(complexity_score, 1.0)

    def _generate_metadata(
        self
        input_type: InputType
        complexity_level: str
        custom_parameters: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate comprehensive metadata for input sample"""
        characteristics = self.characteristics[input_type]

        metadata = {
            "input_type": input_type.value
            "complexity_level": complexity_level
            "generation_seed": self.seed
            "processing_requirements": characteristics.processing_requirements.copy(),
            "expected_error_rate": characteristics.error_probability
            "validation_patterns": characteristics.validation_patterns.copy(),
            "generation_timestamp": np.datetime64("now").isoformat(),
            "generator_version": "1.0.0",
        }

        # Add custom parameters if provided
        if custom_parameters:
            metadata["custom_parameters"] = custom_parameters

        return metadata

    def _estimate_processing_time(
        self, content: str, characteristics: InputCharacteristics
    ) -> float:
        """Estimate expected processing time for content"""
        base_time = (
            len(content)
            * characteristics.performance_expectations["processing_time_per_char"]
        )

        # Add complexity overhead
        complexity_multiplier = 1.0 + (
            len(content) / 10000.0
        )  # Longer content = more complexity

        return base_time * complexity_multiplier

    def _generate_checksum(self, content: str, input_type: InputType) -> str:
        """Generate validation checksum for content integrity"""
        import hashlib

        # Create checksum from content + input type + seed
        checksum_input = f"{content}{input_type.value}{self.seed}".encode("utf-8")
        return hashlib.sha256(checksum_input).hexdigest()[:16]

    def validate_sample(self, sample: InputSample) -> bool:
        """Validate generated sample meets requirements"""
        characteristics = self.characteristics[sample.input_type]

        # Check content length
        min_size, max_size = characteristics.typical_size_range
        if not (min_size <= len(sample.content) <= max_size):
            logger.warning(
                f"Sample size {len(sample.content)} outside range {min_size}-{max_size}"
            )
            return False

        # Check validation patterns
        import re

        for pattern in characteristics.validation_patterns:
            if not re.search(pattern, sample.content):
                logger.warning(f"Sample failed validation pattern: {pattern}")
                return False

        # Verify checksum
        expected_checksum = self._generate_checksum(sample.content, sample.input_type)
        if sample.validation_checksum != expected_checksum:
            logger.warning("Sample checksum validation failed")
            return False

        return True

    def generate_test_batch(
        self, batch_size: int, distribution: Optional[Dict[InputType, float]] = None
    ) -> List[InputSample]:
        """
        Generate a batch of test samples with specified distribution

        Args:
            batch_size: Number of samples to generate
            distribution: Optional distribution of input types (must sum to 1.0)

        Returns:
            List of generated input samples
        """
        if distribution is None:
            # Equal distribution across all input types
            distribution = {input_type: 1.0 / 6 for input_type in InputType}

        # Validate distribution
        if abs(sum(distribution.values()) - 1.0) > 0.01:
            raise KimeraValidationError("Distribution must sum to 1.0")

        samples = []
        for input_type, proportion in distribution.items():
            count = int(batch_size * proportion)
            for _ in range(count):
                complexity = random.choice(["simple", "medium", "complex", "expert"])
                sample = self.generate_sample(input_type, complexity)
                samples.append(sample)

        # Shuffle for random order
        random.shuffle(samples)

        logger.info(f"Generated test batch: {len(samples)} samples")
        return samples

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about input generation capabilities"""
        return {
            "supported_types": len(InputType),
            "templates_per_type": {
                input_type.value: len(templates)
                for input_type, templates in self.sample_templates.items()
            },
            "size_ranges": {
                input_type.value: self.characteristics[input_type].typical_size_range
                for input_type in InputType
            },
            "complexity_factors": {
                input_type.value: list(
                    self.characteristics[input_type].complexity_factors.keys()
                )
                for input_type in InputType
            },
            "seed": self.seed
        }


# Global instance for module access
_input_generator: Optional[InputGenerator] = None


def get_input_generator(seed: Optional[int] = None) -> InputGenerator:
    """Get global input generator instance"""
    global _input_generator
    if _input_generator is None:
        _input_generator = InputGenerator(seed)
    return _input_generator
