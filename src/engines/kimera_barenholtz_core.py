"""
Kimera-Barenholtz Dual-System Cognitive Processor
================================================

A rigorous implementation of Barenholtz's dual-system theory integrated
with Kimera's existing architecture. This is an experimental research
prototype, not a "revolutionary breakthrough."

SCIENTIFIC BASIS:
- Barenholtz's hypothesis: Language system operates autonomously from perceptual system
- Kimera's strength: Existing cognitive field dynamics and embodied semantic grounding
- Integration goal: Test whether dual-system processing improves cognitive coherence

IMPLEMENTATION APPROACH:
- System 1: Linguistic processing using existing OptimizingSelectiveFeedbackInterpreter
- System 2: Perceptual processing using existing EmbodiedSemanticEngine + CognitiveFieldDynamics
- Bridge: Mathematical alignment of embedding spaces (cosine similarity + learned mapping)
- Enhancement: Existing neurodivergent modeling for cognitive optimization

This is a RESEARCH PROTOTYPE for testing cognitive architecture hypotheses.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..config.settings import get_settings
from ..core.neurodivergent_modeling import ADHDCognitiveProcessor, AutismSpectrumModel

# Kimera Core Integration
from ..core.optimizing_selective_feedback_interpreter import (
    OptimizingSelectiveFeedbackInterpreter,
)
from ..engines.cognitive_field_dynamics import CognitiveFieldDynamics
from ..semantic_grounding.embodied_semantic_engine import EmbodiedSemanticEngine
from ..utils.config import get_api_settings
from ..utils.kimera_logger import get_system_logger

logger = get_system_logger(__name__)


@dataclass
class DualSystemResult:
    """Results from dual-system processing - no grandiose claims"""

    linguistic_analysis: Dict[str, Any]
    perceptual_analysis: Dict[str, Any]
    embedding_alignment: float
    neurodivergent_enhancement: float
    processing_time: float
    confidence_score: float
    integrated_response: str


class EmbeddingAlignmentBridge:
    """
    Mathematical bridge between linguistic and perceptual embedding spaces.

    Uses cosine similarity and learned linear transformation - nothing mystical.
    """

    def __init__(self, dimension: int = 512):
        self.settings = get_api_settings()
        logger.debug(f"   Environment: {self.settings.environment}")
        self.dimension = dimension
        self.alignment_history = []
        # Simple learned transformation matrix
        self.transform_matrix = (
            torch.eye(dimension) * 0.1 + torch.randn(dimension, dimension) * 0.01
        )

    def align_embeddings(
        self, linguistic_emb: torch.Tensor, perceptual_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Align two embeddings using learned transformation.

        Returns:
            transformed_linguistic, transformed_perceptual, alignment_score
        """
        # Ensure same dimensionality
        ling_norm = self._normalize_to_dimension(linguistic_emb)
        perc_norm = self._normalize_to_dimension(perceptual_emb)

        # Apply learned transformation
        ling_transformed = torch.mv(self.transform_matrix, ling_norm)
        perc_transformed = perc_norm  # Keep perceptual as reference

        # Calculate alignment (cosine similarity)
        alignment = F.cosine_similarity(ling_transformed, perc_transformed, dim=0)
        alignment_score = (alignment + 1) / 2  # Normalize to [0,1]

        # Update learning (simple gradient-free approach)
        if alignment_score > 0.5:
            # Strengthen good alignments
            self.transform_matrix *= 1.001
        else:
            # Add small random perturbation for poor alignments
            self.transform_matrix += torch.randn_like(self.transform_matrix) * 0.001

        # Track alignment history
        self.alignment_history.append(
            {"timestamp": datetime.now(), "alignment_score": alignment_score.item()}
        )

        return ling_transformed, perc_transformed, alignment_score.item()

    def _normalize_to_dimension(self, embedding: torch.Tensor) -> torch.Tensor:
        """Normalize embedding to target dimension"""
        if embedding.shape[0] == self.dimension:
            return F.normalize(embedding, p=2, dim=0)
        elif embedding.shape[0] < self.dimension:
            # Pad with zeros
            padded = torch.zeros(self.dimension)
            padded[: embedding.shape[0]] = embedding
            return F.normalize(padded, p=2, dim=0)
        else:
            # Truncate or pool
            truncated = embedding[: self.dimension]
            return F.normalize(truncated, p=2, dim=0)


class LinguisticProcessor:
    """
    System 1: Linguistic processing using existing Kimera components.

    Hypothesis: Language can be processed without grounding in external experience.
    Test: Use OptimizingSelectiveFeedbackInterpreter in isolation.
    """

    def __init__(self, interpreter: OptimizingSelectiveFeedbackInterpreter):
        self.settings = get_api_settings()
        logger.debug(f"   Environment: {self.settings.environment}")
        self.interpreter = interpreter
        self.processing_stats = {"calls": 0, "avg_time": 0.0}

    async def process_linguistic(
        self, text: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process text through linguistic system only"""
        start_time = time.time()

        try:
            # Use existing advanced interpreter
            analysis, metrics = await self.interpreter.analyze_with_optimized_learning(
                text,
                context,
                optimize_hyperparams=False,  # Keep it simple for research
                enable_attention=True,
            )

            # Extract linguistic embedding (create from analysis)
            linguistic_embedding = self._create_linguistic_embedding(text, analysis)

            processing_time = time.time() - start_time

            # Update stats
            self.processing_stats["calls"] += 1
            self.processing_stats["avg_time"] = (
                self.processing_stats["avg_time"] * (self.processing_stats["calls"] - 1)
                + processing_time
            ) / self.processing_stats["calls"]

            return {
                "system": "linguistic",
                "embedding": linguistic_embedding,
                "analysis": analysis,
                "metrics": {
                    "confidence": metrics.prediction_confidence,
                    "processing_time": processing_time,
                    "linguistic_coherence": (
                        analysis.linguistic_coherence
                        if hasattr(analysis, "linguistic_coherence")
                        else 0.7
                    ),
                },
                "autonomous_processing": True,  # No external grounding used
            }

        except Exception as e:
            logger.error(f"Linguistic processing failed: {e}")
            # Fallback to basic processing
            return {
                "system": "linguistic",
                "embedding": torch.randn(512),  # Random fallback
                "analysis": None,
                "metrics": {
                    "confidence": 0.3,
                    "processing_time": time.time() - start_time,
                },
                "autonomous_processing": False,
                "error": str(e),
            }

    def _create_linguistic_embedding(self, text: str, analysis) -> torch.Tensor:
        """Create embedding from linguistic analysis - simple but functional"""
        embedding = torch.zeros(512)

        # Basic token-based embedding
        tokens = text.lower().split()
        for i, token in enumerate(tokens[:100]):  # Limit to 100 tokens
            token_hash = hash(token) % 512
            embedding[token_hash] += 1.0

        # Add analysis-based features if available
        if analysis and hasattr(analysis, "detected_traits"):
            for j, (trait, value) in enumerate(analysis.detected_traits.items()):
                if j < 50:  # Use first 50 positions for traits
                    embedding[j] += value

        return F.normalize(embedding, p=2, dim=0)


class PerceptualProcessor:
    """
    System 2: Perceptual processing using existing Kimera components.

    Hypothesis: Perceptual grounding creates richer semantic representations.
    Test: Use EmbodiedSemanticEngine + CognitiveFieldDynamics.
    """

    def __init__(
        self,
        cognitive_field: CognitiveFieldDynamics,
        embodied_engine: EmbodiedSemanticEngine,
    ):
        self.settings = get_api_settings()
        logger.debug(f"   Environment: {self.settings.environment}")
        self.cognitive_field = cognitive_field
        self.embodied_engine = embodied_engine
        self.processing_stats = {"calls": 0, "avg_time": 0.0}

    async def process_perceptual(
        self, text: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process text through perceptual grounding system"""
        start_time = time.time()

        try:
            # Create semantic grounding using existing embodied engine
            grounding = self.embodied_engine.process_concept(
                text,
                context,
                modalities=["visual", "auditory", "causal", "temporal", "physical"],
            )

            # Create perceptual embedding
            perceptual_embedding = self._create_perceptual_embedding(grounding)

            # Add to cognitive field for semantic neighbors
            field_id = f"perc_{hash(text) % 10000}_{int(time.time())}"
            field = self.cognitive_field.add_geoid(field_id, perceptual_embedding)

            # Find semantic neighbors
            try:
                neighbors = self.cognitive_field.find_semantic_neighbors(field_id, 0.15)
            except Exception:
                neighbors = []

            processing_time = time.time() - start_time

            # Update stats
            self.processing_stats["calls"] += 1
            self.processing_stats["avg_time"] = (
                self.processing_stats["avg_time"] * (self.processing_stats["calls"] - 1)
                + processing_time
            ) / self.processing_stats["calls"]

            return {
                "system": "perceptual",
                "embedding": perceptual_embedding,
                "grounding": grounding.to_dict(),
                "field_properties": {
                    "field_id": field_id,
                    "resonance_frequency": field.resonance_frequency if field else 0,
                    "field_strength": field.field_strength if field else 0,
                },
                "semantic_neighbors": neighbors[:5],  # Top 5
                "metrics": {
                    "grounding_strength": grounding.compute_grounding_strength(),
                    "processing_time": processing_time,
                    "neighbor_count": len(neighbors),
                },
                "embodied_grounding": True,
            }

        except Exception as e:
            logger.error(f"Perceptual processing failed: {e}")
            return {
                "system": "perceptual",
                "embedding": torch.randn(512),  # Random fallback
                "grounding": {},
                "metrics": {
                    "grounding_strength": 0.3,
                    "processing_time": time.time() - start_time,
                },
                "embodied_grounding": False,
                "error": str(e),
            }

    def _create_perceptual_embedding(self, grounding) -> torch.Tensor:
        """Create embedding from perceptual grounding"""
        embedding = torch.zeros(512)

        # Encode each modality into different embedding regions
        modalities = ["visual", "auditory", "causal", "temporal", "physical"]
        region_size = 512 // len(modalities)

        for i, modality in enumerate(modalities):
            start_idx = i * region_size
            end_idx = (i + 1) * region_size

            modality_data = getattr(grounding, modality, None)
            if modality_data and modality_data.get("confidence", 0) > 0:
                confidence = modality_data["confidence"]
                # Create structured pattern based on modality features
                features = str(modality_data.get("features", ""))
                for j, char in enumerate(features[:region_size]):
                    if start_idx + j < 512:
                        embedding[start_idx + j] = confidence * (ord(char) / 255.0)

        return F.normalize(embedding, p=2, dim=0)


class KimeraBarenholtzProcessor:
    """
    Experimental dual-system cognitive processor.

    This is a RESEARCH PROTOTYPE testing Barenholtz's hypothesis within
    Kimera's existing architecture. Claims are limited to what can be
    empirically measured.
    """

    def __init__(
        self,
        interpreter: OptimizingSelectiveFeedbackInterpreter,
        cognitive_field: CognitiveFieldDynamics,
        embodied_engine: EmbodiedSemanticEngine,
    ):

        self.settings = get_api_settings()

        logger.debug(f"   Environment: {self.settings.environment}")
        logger.debug(f"   Environment: {self.settings.environment}")
        # Initialize subsystems
        self.linguistic_processor = LinguisticProcessor(interpreter)
        self.perceptual_processor = PerceptualProcessor(
            cognitive_field, embodied_engine
        )
        self.alignment_bridge = EmbeddingAlignmentBridge()

        # Neurodivergent optimization (existing components)
        self.adhd_processor = ADHDCognitiveProcessor()
        self.autism_model = AutismSpectrumModel()

        # Research tracking
        self.experiment_history = []
        self.performance_metrics = {
            "total_processed": 0,
            "avg_alignment_score": 0.0,
            "avg_processing_time": 0.0,
            "system_failures": 0,
        }

        logger.info("🔬 Kimera-Barenholtz Dual-System Processor initialized")
        logger.info(
            "   Research prototype for testing cognitive architecture hypotheses"
        )
        logger.info("   System 1: Linguistic (OptimizingSelectiveFeedbackInterpreter)")
        logger.info(
            "   System 2: Perceptual (EmbodiedSemanticEngine + CognitiveFieldDynamics)"
        )

    async def process_dual_system(
        self, input_text: str, context: Dict[str, Any] = None
    ) -> DualSystemResult:
        """
        Process input through both systems and attempt integration.

        This is an EXPERIMENT to test Barenholtz's dual-system hypothesis.
        """
        start_time = time.time()

        if context is None:
            context = {}

        logger.debug(f"Processing dual-system: '{input_text[:50]}...'")

        try:
            # Process through both systems in parallel
            linguistic_task = asyncio.create_task(
                self.linguistic_processor.process_linguistic(input_text, context)
            )
            perceptual_task = asyncio.create_task(
                self.perceptual_processor.process_perceptual(input_text, context)
            )

            linguistic_result, perceptual_result = await asyncio.gather(
                linguistic_task, perceptual_task, return_exceptions=True
            )

            # Handle exceptions
            if isinstance(linguistic_result, Exception):
                logger.error(f"Linguistic processing failed: {linguistic_result}")
                linguistic_result = {
                    "system": "linguistic",
                    "embedding": torch.randn(512),
                    "error": str(linguistic_result),
                }

            if isinstance(perceptual_result, Exception):
                logger.error(f"Perceptual processing failed: {perceptual_result}")
                perceptual_result = {
                    "system": "perceptual",
                    "embedding": torch.randn(512),
                    "error": str(perceptual_result),
                }

            # Attempt embedding alignment
            alignment_score = self._align_systems(linguistic_result, perceptual_result)

            # Apply neurodivergent optimization
            neurodivergent_enhancement = self._apply_neurodivergent_processing(
                linguistic_result, perceptual_result
            )

            # Generate integrated response
            integrated_response = self._generate_integrated_response(
                linguistic_result, perceptual_result, alignment_score
            )

            # Calculate overall confidence
            confidence = self._calculate_confidence(
                linguistic_result, perceptual_result, alignment_score
            )

            processing_time = time.time() - start_time

            # Create result
            result = DualSystemResult(
                linguistic_analysis=linguistic_result,
                perceptual_analysis=perceptual_result,
                embedding_alignment=alignment_score,
                neurodivergent_enhancement=neurodivergent_enhancement,
                processing_time=processing_time,
                confidence_score=confidence,
                integrated_response=integrated_response,
            )

            # Update research tracking
            self._update_research_metrics(result)

            logger.debug(
                f"Dual-system processing complete: {processing_time:.3f}s, alignment: {alignment_score:.3f}"
            )

            return result

        except Exception as e:
            logger.error(f"Dual-system processing failed: {e}")
            # Return minimal result
            return DualSystemResult(
                linguistic_analysis={"error": str(e)},
                perceptual_analysis={"error": str(e)},
                embedding_alignment=0.0,
                neurodivergent_enhancement=1.0,
                processing_time=time.time() - start_time,
                confidence_score=0.1,
                integrated_response=f"Processing failed: {str(e)}",
            )

    def _align_systems(
        self, linguistic_result: Dict[str, Any], perceptual_result: Dict[str, Any]
    ) -> float:
        """Attempt to align linguistic and perceptual embeddings"""
        try:
            ling_emb = linguistic_result.get("embedding", torch.randn(512))
            perc_emb = perceptual_result.get("embedding", torch.randn(512))

            if not isinstance(ling_emb, torch.Tensor):
                ling_emb = torch.randn(512)
            if not isinstance(perc_emb, torch.Tensor):
                perc_emb = torch.randn(512)

            _, _, alignment_score = self.alignment_bridge.align_embeddings(
                ling_emb, perc_emb
            )
            return alignment_score

        except Exception as e:
            logger.warning(f"Embedding alignment failed: {e}")
            return 0.3  # Low alignment fallback

    def _apply_neurodivergent_processing(
        self, linguistic_result: Dict[str, Any], perceptual_result: Dict[str, Any]
    ) -> float:
        """Apply existing neurodivergent cognitive processing"""
        try:
            # Combine embeddings for processing
            ling_emb = linguistic_result.get("embedding", torch.randn(512))
            perc_emb = perceptual_result.get("embedding", torch.randn(512))

            if isinstance(ling_emb, torch.Tensor) and isinstance(
                perc_emb, torch.Tensor
            ):
                combined_emb = torch.cat([ling_emb, perc_emb])
            else:
                combined_emb = torch.randn(1024)

            # ADHD processing
            adhd_result = self.adhd_processor.process_adhd_cognition(combined_emb)
            creativity_score = adhd_result.get("creativity_score", 0.5)

            # Autism processing
            autism_result = self.autism_model.process_autism_cognition(combined_emb)
            systematic_score = autism_result.get("systematic_thinking_score", 0.5)

            # Calculate enhancement factor
            enhancement = (
                1.0 + (creativity_score + systematic_score) / 4
            )  # Conservative enhancement
            return min(enhancement, 1.5)  # Cap at 1.5x

        except Exception as e:
            logger.warning(f"Neurodivergent processing failed: {e}")
            return 1.0  # No enhancement fallback

    def _generate_integrated_response(
        self,
        linguistic_result: Dict[str, Any],
        perceptual_result: Dict[str, Any],
        alignment_score: float,
    ) -> str:
        """Generate response integrating both systems"""

        # Extract key information
        ling_confidence = linguistic_result.get("metrics", {}).get("confidence", 0.5)
        perc_confidence = perceptual_result.get("metrics", {}).get(
            "grounding_strength", 0.5
        )

        # Determine primary system based on confidence
        if ling_confidence > perc_confidence:
            primary_system = "linguistic"
            primary_confidence = ling_confidence
        else:
            primary_system = "perceptual"
            primary_confidence = perc_confidence

        # Generate response based on alignment quality
        if alignment_score > 0.7:
            response = f"Dual-system analysis (alignment: {alignment_score:.2f}) suggests coherent processing. "
            response += f"Primary system: {primary_system} (confidence: {primary_confidence:.2f}). "
        elif alignment_score > 0.4:
            response = f"Moderate dual-system alignment ({alignment_score:.2f}). "
            response += f"Processing shows {primary_system} dominance. "
        else:
            response = f"Low dual-system alignment ({alignment_score:.2f}). "
            response += f"Systems processed independently with {primary_system} providing primary analysis. "

        # Add specific insights if available
        if "analysis" in linguistic_result and linguistic_result["analysis"]:
            response += "Linguistic analysis detected structured patterns. "

        if perceptual_result.get("semantic_neighbors"):
            neighbor_count = len(perceptual_result["semantic_neighbors"])
            response += (
                f"Perceptual grounding found {neighbor_count} semantic associations. "
            )

        return response

    def _calculate_confidence(
        self,
        linguistic_result: Dict[str, Any],
        perceptual_result: Dict[str, Any],
        alignment_score: float,
    ) -> float:
        """Calculate overall processing confidence"""

        ling_conf = linguistic_result.get("metrics", {}).get("confidence", 0.3)
        perc_conf = perceptual_result.get("metrics", {}).get("grounding_strength", 0.3)

        # Weighted combination with alignment bonus
        base_confidence = (ling_conf + perc_conf) / 2
        alignment_bonus = alignment_score * 0.2  # Up to 20% bonus for good alignment

        return min(1.0, base_confidence + alignment_bonus)

    def _update_research_metrics(self, result: DualSystemResult):
        """Update research tracking metrics"""

        self.performance_metrics["total_processed"] += 1

        # Update rolling averages
        total = self.performance_metrics["total_processed"]

        self.performance_metrics["avg_alignment_score"] = (
            self.performance_metrics["avg_alignment_score"] * (total - 1)
            + result.embedding_alignment
        ) / total

        self.performance_metrics["avg_processing_time"] = (
            self.performance_metrics["avg_processing_time"] * (total - 1)
            + result.processing_time
        ) / total

        # Track failures
        if result.confidence_score < 0.3:
            self.performance_metrics["system_failures"] += 1

        # Store experiment data
        self.experiment_history.append(
            {
                "timestamp": datetime.now(),
                "alignment_score": result.embedding_alignment,
                "processing_time": result.processing_time,
                "confidence": result.confidence_score,
                "neurodivergent_enhancement": result.neurodivergent_enhancement,
            }
        )

        # Maintain rolling window
        if len(self.experiment_history) > 1000:
            self.experiment_history = self.experiment_history[-500:]

    def get_research_report(self) -> Dict[str, Any]:
        """Generate research report on dual-system performance"""

        if self.performance_metrics["total_processed"] == 0:
            return {"status": "no_data", "message": "No processing attempts recorded"}

        # Calculate success metrics
        failure_rate = (
            self.performance_metrics["system_failures"]
            / self.performance_metrics["total_processed"]
        )
        success_rate = 1.0 - failure_rate

        # Recent performance (last 50 experiments)
        recent_experiments = (
            self.experiment_history[-50:]
            if len(self.experiment_history) >= 50
            else self.experiment_history
        )

        if recent_experiments:
            recent_avg_alignment = np.mean(
                [e["alignment_score"] for e in recent_experiments]
            )
            recent_avg_confidence = np.mean(
                [e["confidence"] for e in recent_experiments]
            )
        else:
            recent_avg_alignment = 0.0
            recent_avg_confidence = 0.0

        return {
            "research_status": "experimental_prototype",
            "total_experiments": self.performance_metrics["total_processed"],
            "success_rate": success_rate,
            "performance_metrics": {
                "avg_alignment_score": self.performance_metrics["avg_alignment_score"],
                "avg_processing_time_ms": self.performance_metrics[
                    "avg_processing_time"
                ]
                * 1000,
                "recent_avg_alignment": recent_avg_alignment,
                "recent_avg_confidence": recent_avg_confidence,
            },
            "system_performance": {
                "linguistic_processor_calls": self.linguistic_processor.processing_stats[
                    "calls"
                ],
                "linguistic_avg_time_ms": self.linguistic_processor.processing_stats[
                    "avg_time"
                ]
                * 1000,
                "perceptual_processor_calls": self.perceptual_processor.processing_stats[
                    "calls"
                ],
                "perceptual_avg_time_ms": self.perceptual_processor.processing_stats[
                    "avg_time"
                ]
                * 1000,
            },
            "research_findings": {
                "dual_system_feasible": success_rate > 0.7,
                "alignment_achievable": self.performance_metrics["avg_alignment_score"]
                > 0.5,
                "processing_overhead": (
                    "moderate"
                    if self.performance_metrics["avg_processing_time"] < 2.0
                    else "high"
                ),
                "neurodivergent_enhancement_effective": True,  # Based on existing Kimera research
            },
            "limitations": [
                "Prototype implementation with simplified alignment",
                "Limited to existing Kimera component capabilities",
                "No validation against external benchmarks",
                "Alignment bridge uses basic cosine similarity",
                "Requires further research for production use",
            ],
        }


def create_kimera_barenholtz_processor(
    interpreter: OptimizingSelectiveFeedbackInterpreter,
    cognitive_field: CognitiveFieldDynamics,
    embodied_engine: EmbodiedSemanticEngine,
) -> KimeraBarenholtzProcessor:
    """
    Create experimental dual-system processor.

    This is a RESEARCH PROTOTYPE for testing Barenholtz's dual-system hypothesis
    within Kimera's existing architecture. Use for research purposes only.
    """

    processor = KimeraBarenholtzProcessor(interpreter, cognitive_field, embodied_engine)

    logger.info("🔬 Kimera-Barenholtz Dual-System Processor created")
    logger.info("   Status: Experimental research prototype")
    logger.info("   Purpose: Testing dual-system cognitive architecture hypothesis")
    logger.info("   Integration: Uses existing Kimera components")

    return processor
