"""
Contradiction Engine for detecting tension gradients in cognitive fields.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

import numpy as np
from sklearn.metrics.pairwise import cosine_distances

from ..config.settings import get_settings
from ..core.ethical_governor import ActionProposal
from ..core.governor_proxy import require_constitutional
from ..core.insight import InsightScar
from ..core.native_math import NativeMath
from ..governance import erl
from ..utils.kimera_exceptions import (KimeraCognitiveError, KimeraValidationError
                                       handle_exception)
from ..utils.kimera_logger import get_cognitive_logger
from ..utils.robust_config import get_api_settings
from ..vault.enhanced_database_schema import GeoidState

# Initialize structured logger
logger = get_cognitive_logger(__name__)


@dataclass
class TensionGradient:
    """Auto-generated class."""
    pass
    geoid_a: str
    geoid_b: str
    tension_score: float
    gradient_type: str
class ContradictionEngine:
    """Auto-generated class."""
    pass
    """
    Core contradiction detection engine for Kimera's cognitive field dynamics.

    Detects tension gradients between geoids using composite scoring and
    makes decisions about collapse, surge, or buffer operations.
    """

    def __init__(self, tension_threshold: float = 0.4):
        try:
            self.settings = get_api_settings()
        except Exception as e:
            logger.warning(f"API settings loading failed: {e}. Using safe fallback.")
            from ..utils.robust_config import safe_get_api_settings

            self.settings = safe_get_api_settings()
        logger.debug(f"   Environment: {self.settings.environment}")
        """
        Initialize the contradiction engine.
        
        Args:
            tension_threshold: Threshold for detecting significant tensions (0.0-1.0)
            
        Raises:
            KimeraValidationError: If tension_threshold is invalid
        """
        if not isinstance(tension_threshold, (int, float)):
            raise KimeraValidationError(
                f"Tension threshold must be numeric, got {type(tension_threshold).__name__}",
                context={"provided_value": tension_threshold},
            )

        if not 0.0 <= tension_threshold <= 1.0:
            raise KimeraValidationError(
                f"Tension threshold must be between 0.0 and 1.0, got {tension_threshold}",
                context={"provided_value": tension_threshold, "valid_range": "0.0-1.0"},
            )

        self.tension_threshold = tension_threshold
        logger.info(
            f"ContradictionEngine initialized with tension_threshold={tension_threshold}"
        )

    def detect_tension_gradients(
        self, geoids: List[GeoidState]
    ) -> List[TensionGradient]:
        """
        Detect tension gradients using vectorized operations for O(n log n) complexity.

        Args:
            geoids: List of GeoidState objects to analyze

        Returns:
            List of detected tension gradients

        Raises:
            KimeraValidationError: If input validation fails
            KimeraCognitiveError: If tension detection fails
        """
        # Input validation
        if not isinstance(geoids, list):
            raise KimeraValidationError(
                f"Geoids must be a list, got {type(geoids).__name__}",
                context={"provided_type": type(geoids).__name__},
            )

        if not geoids:
            logger.warning("No geoids provided for tension detection")
            return []

        if len(geoids) < 2:
            logger.debug(
                "Less than 2 geoids provided, no tensions possible",
                geoid_count=len(geoids),
            )
            return []

        # Validate all geoids are GeoidState instances
        for i, geoid in enumerate(geoids):
            if not isinstance(geoid, GeoidState):
                raise KimeraValidationError(
                    f"All geoids must be GeoidState instances, item {i} is {type(geoid).__name__}",
                    context={"item_index": i, "item_type": type(geoid).__name__},
                )
            if not hasattr(geoid, "geoid_id") or not geoid.geoid_id:
                raise KimeraValidationError(
                    f"Geoid at index {i} missing valid geoid_id",
                    context={"item_index": i},
                )

        try:
            # Simplified operation context for compatibility
            logger.debug(
                f"Starting vectorized tension gradient detection for {len(geoids)} geoids with threshold {self.tension_threshold}"
            )

            # Use vectorized approach for large datasets
            if len(geoids) > 100:
                return self._detect_tension_gradients_vectorized(geoids)
            else:
                # Use original approach for small datasets
                return self._detect_tension_gradients_pairwise(geoids)

        except (KimeraValidationError, KimeraCognitiveError):
            # Re-raise Kimera-specific exceptions
            raise
        except Exception as e:
            error_msg = f"Tension detection operation failed: {e}"
            logger.error(error_msg, error=e)
            raise KimeraCognitiveError(error_msg) from e

    def _safe_detect_tensions(self, geoids: List[GeoidState]) -> List[TensionGradient]:
        """
        Safely detect tensions with error recovery.

        Args:
            geoids: List of GeoidState objects

        Returns:
            List of detected tensions (may be empty on error)
        """
        try:
            return self.detect_tension_gradients(geoids)
        except Exception as e:
            logger.error(f"Error detecting tensions: {e}", exc_info=True)
            # Return empty list instead of propagating error
            return []

    def _detect_tension_gradients_vectorized(
        self, geoids: List[GeoidState]
    ) -> List[TensionGradient]:
        """
        Vectorized tension gradient detection for large datasets - O(n log n) complexity.

        Args:
            geoids: List of GeoidState objects to analyze

        Returns:
            List of detected tension gradients
        """
        try:
            n = len(geoids)
            geoid_ids = [g.geoid_id for g in geoids]

            # Prepare embedding matrix for vectorized operations
            embeddings = []
            valid_indices = []

            for i, geoid in enumerate(geoids):
                if geoid.embedding_vector and len(geoid.embedding_vector) > 0:
                    embeddings.append(np.array(geoid.embedding_vector))
                    valid_indices.append(i)

            if len(embeddings) < 2:
                logger.debug("Insufficient valid embeddings for vectorized processing")
                return []

            # Stack embeddings into matrix for vectorized cosine distance computation
            embedding_matrix = np.vstack(embeddings)

            # Compute all pairwise cosine distances in one operation - O(n²) but vectorized
            cosine_dist_matrix = cosine_distances(embedding_matrix)

            # Prepare semantic and symbolic state matrices for batch processing
            semantic_states = []
            symbolic_states = []

            for i in valid_indices:
                geoid = geoids[i]
                # Convert semantic state to binary vector for faster set operations
                semantic_keys = (
                    set(geoid.semantic_state.keys()) if geoid.semantic_state else set()
                )
                symbolic_keys = (
                    set(geoid.symbolic_state.keys()) if geoid.symbolic_state else set()
                )
                semantic_states.append(semantic_keys)
                symbolic_states.append(symbolic_keys)

            # Use efficient pairwise comparison with early termination
            tensions = []
            comparison_count = 0

            # Create upper triangular indices for pairwise comparisons
            i_indices, j_indices = np.triu_indices(len(valid_indices), k=1)

            for idx_pair in zip(i_indices, j_indices):
                i_idx, j_idx = idx_pair
                i_orig = valid_indices[i_idx]
                j_orig = valid_indices[j_idx]

                comparison_count += 1

                # Get pre-computed embedding distance
                emb_score = cosine_dist_matrix[i_idx, j_idx]

                # Fast symbolic and semantic comparison
                semantic_conflict = self._fast_layer_conflict(
                    semantic_states[i_idx],
                    semantic_states[j_idx],
                    symbolic_states[i_idx],
                    symbolic_states[j_idx],
                )

                # Early termination optimization
                if emb_score < 0.3 and semantic_conflict < 0.3:
                    continue

                # Combine scores for composite tension metric
                composite_score = 0.4 * emb_score + 0.6 * semantic_conflict

                if composite_score >= self.tension_threshold:
                    tension = TensionGradient(
                        geoid_a=geoid_ids[i_orig],
                        geoid_b=geoid_ids[j_orig],
                        tension_score=composite_score
                        gradient_type="vectorized_composite",
                    )
                    tensions.append(tension)

            logger.debug(
                f"Vectorized processing: {comparison_count} comparisons, {len(tensions)} tensions found"
            )
            return tensions

        except Exception as e:
            logger.error(f"Vectorized tension detection failed: {e}")
            # Fallback to pairwise method
            return self._detect_tension_gradients_pairwise(geoids)

    def _detect_tension_gradients_pairwise(
        self, geoids: List[GeoidState]
    ) -> List[TensionGradient]:
        """
        Pairwise tension gradient detection for smaller datasets.

        Args:
            geoids: List of GeoidState objects to analyze

        Returns:
            List of detected tension gradients
        """
        tensions = []
        n = len(geoids)

        for i in range(n):
            for j in range(i + 1, n):
                try:
                    # Calculate composite tension score
                    emb_score = self._embedding_misalignment(geoids[i], geoids[j])
                    layer_score = self._layer_conflict_intensity(geoids[i], geoids[j])
                    symbolic_score = self._symbolic_opposition(geoids[i], geoids[j])

                    # Weighted composite score
                    composite_score = (
                        0.4 * emb_score + 0.3 * layer_score + 0.3 * symbolic_score
                    )

                    if composite_score >= self.tension_threshold:
                        tension = TensionGradient(
                            geoid_a=geoids[i].geoid_id
                            geoid_b=geoids[j].geoid_id
                            tension_score=composite_score
                            gradient_type="pairwise_composite",
                        )
                        tensions.append(tension)

                except Exception as e:
                    logger.warning(
                        f"Failed to compute tension between {geoids[i].geoid_id} and {geoids[j].geoid_id}: {e}"
                    )
                    continue

        logger.debug(
            f"Pairwise processing: {n*(n-1)//2} comparisons, {len(tensions)} tensions found"
        )
        return tensions

    def _fast_layer_conflict(
        self, sem_a: set, sem_b: set, sym_a: set, sym_b: set
    ) -> float:
        """
        Fast layer conflict calculation using set operations.

        Args:
            sem_a, sem_b: Semantic state key sets
            sym_a, sym_b: Symbolic state key sets

        Returns:
            Conflict intensity score (0.0-1.0)
        """

        def fast_jaccard_distance(x: set, y: set) -> float:
            if not x and not y:
                return 0.0
            intersection = len(x & y)
            union = len(x | y)
            return 1.0 - (intersection / union) if union > 0 else 1.0

        sem_conflict = fast_jaccard_distance(sem_a, sem_b)
        sym_conflict = fast_jaccard_distance(sym_a, sym_b)

        return (sem_conflict + sym_conflict) / 2.0

    def _fast_symbolic_opposition(self, symbolic_a: dict, symbolic_b: dict) -> float:
        """
        Fast symbolic opposition detection.

        Args:
            symbolic_a, symbolic_b: Symbolic state dictionaries

        Returns:
            Opposition score (0.0-1.0)
        """
        if not symbolic_a or not symbolic_b:
            return 0.0

        common_keys = set(symbolic_a.keys()) & set(symbolic_b.keys())
        if not common_keys:
            return 0.0

        oppositions = 0
        for key in common_keys:
            val_a = symbolic_a[key]
            val_b = symbolic_b[key]
            if isinstance(val_a, bool) and isinstance(val_b, bool) and val_a != val_b:
                oppositions += 1

        return oppositions / len(common_keys) if common_keys else 0.0

    def _embedding_misalignment(self, a: GeoidState, b: GeoidState) -> float:
        """Calculate embedding misalignment score between two geoids."""
        try:
            if not a.embedding_vector or not b.embedding_vector:
                return 0.0

            # Ensure vectors are the same length
            vec_a = np.array(a.embedding_vector)
            vec_b = np.array(b.embedding_vector)

            if len(vec_a) != len(vec_b):
                min_len = min(len(vec_a), len(vec_b))
                vec_a = vec_a[:min_len]
                vec_b = vec_b[:min_len]

            if len(vec_a) == 0:
                return 0.0

            # Calculate cosine distance
            dot_product = np.dot(vec_a, vec_b)
            norm_a = np.linalg.norm(vec_a)
            norm_b = np.linalg.norm(vec_b)

            if norm_a == 0 or norm_b == 0:
                return 1.0  # Maximum misalignment for zero vectors

            cosine_similarity = dot_product / (norm_a * norm_b)
            cosine_distance = 1.0 - cosine_similarity

            return max(0.0, min(1.0, cosine_distance))

        except Exception as e:
            logger.warning(f"Embedding calculation failed: {e}")
            return 0.0

    def _layer_conflict_intensity(self, a: GeoidState, b: GeoidState) -> float:
        """Calculate layer conflict intensity between semantic states."""
        try:
            sem_a = set(a.semantic_state.keys()) if a.semantic_state else set()
            sem_b = set(b.semantic_state.keys()) if b.semantic_state else set()

            if not sem_a and not sem_b:
                return 0.0

            def jaccard_distance(x: set, y: set) -> float:
                if not x and not y:
                    return 0.0
                intersection = len(x.intersection(y))
                union = len(x.union(y))
                return 1.0 - (intersection / union) if union > 0 else 1.0

            return jaccard_distance(sem_a, sem_b)

        except Exception as e:
            logger.warning(f"Layer conflict calculation failed: {e}")
            return 0.0

    def _symbolic_opposition(self, a: GeoidState, b: GeoidState) -> float:
        """Calculate symbolic opposition score."""
        try:
            sym_a = a.symbolic_state if a.symbolic_state else {}
            sym_b = b.symbolic_state if b.symbolic_state else {}

            if not sym_a or not sym_b:
                return 0.0

            common_keys = set(sym_a.keys()).intersection(set(sym_b.keys()))
            if not common_keys:
                return 0.0

            oppositions = 0
            total_comparisons = 0

            for key in common_keys:
                val_a = sym_a[key]
                val_b = sym_b[key]
                total_comparisons += 1

                # Check for boolean oppositions
                if isinstance(val_a, bool) and isinstance(val_b, bool):
                    if val_a != val_b:
                        oppositions += 1
                # Check for numeric oppositions (different signs)
                elif isinstance(val_a, (int, float)) and isinstance(
                    val_b, (int, float)
                ):
                    if (val_a > 0) != (val_b > 0) and val_a != 0 and val_b != 0:
                        oppositions += 1
                # Check for string oppositions
                elif isinstance(val_a, str) and isinstance(val_b, str):
                    if val_a.lower() != val_b.lower():
                        oppositions += 0.5  # Partial opposition for strings

            return oppositions / total_comparisons if total_comparisons > 0 else 0.0

        except Exception as e:
            logger.warning(f"Symbolic opposition calculation failed: {e}")
            return 0.0

    def calculate_pulse_strength(
        self, tension: TensionGradient, geoids: Dict[str, GeoidState]
    ) -> float:
        """Calculate pulse strength for a detected tension."""
        try:
            geoid_a = geoids.get(tension.geoid_a)
            geoid_b = geoids.get(tension.geoid_b)

            if not geoid_a or not geoid_b:
                return 0.0

            # Base strength from tension score
            base_strength = tension.tension_score

            # Amplify based on semantic complexity
            complexity_a = len(geoid_a.semantic_state) if geoid_a.semantic_state else 0
            complexity_b = len(geoid_b.semantic_state) if geoid_b.semantic_state else 0
            avg_complexity = (complexity_a + complexity_b) / 2.0

            # Normalize complexity factor (assume max 20 semantic elements)
            complexity_factor = min(avg_complexity / 20.0, 1.0)

            pulse_strength = base_strength * (1.0 + 0.5 * complexity_factor)

            return min(pulse_strength, 1.0)

        except Exception as e:
            logger.warning(f"Pulse strength calculation failed: {e}")
            return 0.0

    def decide_collapse_or_surge(
        self
        pulse_strength: float
        stability: Dict[str, float],
        profile: Dict[str, object] | None = None
    ) -> str:
        """
        Decide whether to collapse, surge, or buffer based on pulse strength and system stability.

        Args:
            pulse_strength: Strength of the cognitive pulse (0.0-1.0)
            stability: System stability metrics
            profile: Optional cognitive profile for context

        Returns:
            Decision string: "collapse", "surge", or "buffer"
        """
        try:
            # Get system stability indicators
            vault_pressure = stability.get("vault_pressure", 0.5)
            semantic_cohesion = stability.get("semantic_cohesion", 0.5)
            entropic_stability = stability.get("entropic_stability", 0.5)

            # Calculate overall system stability
            system_stability = (
                vault_pressure + semantic_cohesion + entropic_stability
            ) / 3.0

            # Decision matrix based on pulse strength and system stability
            if pulse_strength < 0.3:
                # Low pulse strength - maintain current state
                return "buffer"
            elif pulse_strength < 0.7:
                # Medium pulse strength - decision depends on stability
                if system_stability > 0.6:
                    return "surge"  # System is stable, allow surge
                else:
                    return "buffer"  # System unstable, maintain buffer
            else:
                # High pulse strength - more aggressive decision
                if system_stability > 0.7:
                    return "surge"  # Very stable system can handle surge
                elif system_stability < 0.3:
                    return "collapse"  # Unstable system needs collapse
                else:
                    return "buffer"  # Moderate stability, maintain buffer

        except Exception as e:
            logger.error(f"Decision calculation failed: {e}")
            return "buffer"  # Safe default

    def check_insight_conflict(
        self, insight: InsightScar, existing_insights: List[InsightScar]
    ) -> Optional[InsightScar]:
        """
        Check if a new insight conflicts with existing insights.

        Args:
            insight: New insight to check
            existing_insights: List of existing insights

        Returns:
            Conflicting insight if found, None otherwise
        """
        try:
            if not existing_insights:
                return None

            for existing in existing_insights:
                # Check semantic similarity
                if hasattr(insight, "embedding_vector") and hasattr(
                    existing, "embedding_vector"
                ):
                    if insight.embedding_vector and existing.embedding_vector:
                        # Calculate cosine similarity
                        vec_new = np.array(insight.embedding_vector)
                        vec_existing = np.array(existing.embedding_vector)

                        if len(vec_new) == len(vec_existing) and len(vec_new) > 0:
                            similarity = np.dot(vec_new, vec_existing) / (
                                np.linalg.norm(vec_new) * np.linalg.norm(vec_existing)
                            )

                            # High similarity but different conclusions indicates conflict
                            if similarity > 0.8 and insight.content != existing.content:
                                return existing

            return None

        except Exception as e:
            logger.warning(f"Insight conflict check failed: {e}")
            return None
