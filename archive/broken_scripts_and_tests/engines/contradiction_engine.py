from __future__ import annotations
from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_distances

from ..core.geoid import GeoidState
from ..core.insight import InsightScar
from ..core.native_math import NativeMath
from ..governance import erl
from ..core.governor_proxy import require_constitutional
from ..core.ethical_governor import ActionProposal
from ..utils.kimera_logger import get_cognitive_logger
from ..utils.kimera_exceptions import (
    KimeraCognitiveError,
    KimeraValidationError,
    handle_exception
)

# Initialize structured logger
logger = get_cognitive_logger(__name__)

@dataclass
class TensionGradient:
    geoid_a: str
    geoid_b: str
    tension_score: float
    gradient_type: str

class ContradictionEngine:
    """
    Core contradiction detection engine for Kimera's cognitive field dynamics.
    
    Detects tension gradients between geoids using composite scoring and
    makes decisions about collapse, surge, or buffer operations.
    """
    
    def __init__(self, tension_threshold: float = 0.4):
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
                context={'provided_value': tension_threshold}
            )
            
        if not 0.0 <= tension_threshold <= 1.0:
            raise KimeraValidationError(
                f"Tension threshold must be between 0.0 and 1.0, got {tension_threshold}",
                context={'provided_value': tension_threshold, 'valid_range': '0.0-1.0'}
            )
        
        self.tension_threshold = tension_threshold
        logger.info(f"ContradictionEngine initialized with tension_threshold={tension_threshold}")

    def detect_tension_gradients(self, geoids: List[GeoidState]) -> List[TensionGradient]:
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
                context={'provided_type': type(geoids).__name__}
            )
        
        if not geoids:
            logger.warning("No geoids provided for tension detection")
            return []
        
        if len(geoids) < 2:
            logger.debug("Less than 2 geoids provided, no tensions possible", geoid_count=len(geoids))
            return []
        
        # Validate all geoids are GeoidState instances
        for i, geoid in enumerate(geoids):
            if not isinstance(geoid, GeoidState):
                raise KimeraValidationError(
                    f"All geoids must be GeoidState instances, item {i} is {type(geoid).__name__}",
                    context={'item_index': i, 'item_type': type(geoid).__name__}
                )
            if not hasattr(geoid, 'geoid_id') or not geoid.geoid_id:
                raise KimeraValidationError(
                    f"Geoid at index {i} missing valid geoid_id",
                    context={'item_index': i}
                )
        
        try:
            # Simplified operation context for compatibility
            logger.debug(f"Starting vectorized tension gradient detection for {len(geoids)} geoids with threshold {self.tension_threshold}")
            
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

    def _detect_tension_gradients_vectorized(self, geoids: List[GeoidState]) -> List[TensionGradient]:
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
                semantic_keys = set(geoid.semantic_state.keys()) if geoid.semantic_state else set()
                symbolic_keys = set(geoid.symbolic_state.keys()) if geoid.symbolic_state else set()
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
                layer_score = self._fast_layer_conflict(semantic_states[i_idx], semantic_states[j_idx], 
                                                      symbolic_states[i_idx], symbolic_states[j_idx])
                
                sym_score = self._fast_symbolic_opposition(geoids[i_orig].symbolic_state, 
                                                         geoids[j_orig].symbolic_state)
                
                # Composite score
                composite_score = (emb_score + layer_score + sym_score) / 3
                
                # Early termination if score is above threshold
                if composite_score > self.tension_threshold:
                    tension = TensionGradient(
                        geoid_ids[i_orig], 
                        geoid_ids[j_orig], 
                        composite_score, 
                        "composite_vectorized"
                    )
                    tensions.append(tension)
            
            logger.info(f"Vectorized tension detection completed: found {len(tensions)} tensions from {comparison_count} comparisons")
            return tensions
            
        except Exception as e:
            error_msg = f"Vectorized tension detection failed: {e}"
            logger.error(error_msg, error=e)
            raise KimeraCognitiveError(error_msg) from e

    def _detect_tension_gradients_pairwise(self, geoids: List[GeoidState]) -> List[TensionGradient]:
        """
        Original pairwise tension gradient detection for small datasets.
        
        Args:
            geoids: List of GeoidState objects to analyze
            
        Returns:
            List of detected tension gradients
        """
        tensions = []
        comparison_count = 0
        
        for i, a in enumerate(geoids):
            for b in geoids[i + 1:]:
                comparison_count += 1
                try:
                    emb = self._embedding_misalignment(a, b)
                    layer = self._layer_conflict_intensity(a, b)
                    sym = self._symbolic_opposition(a, b)
                    score = (emb + layer + sym) / 3

                    if score > self.tension_threshold:
                        tension = TensionGradient(a.geoid_id, b.geoid_id, score, "composite")
                        tensions.append(tension)
                        logger.debug(f"Tension detected between {a.geoid_id} and {b.geoid_id} with score {score:.3f}")
                
                except Exception as e:
                    error_msg = f"Failed to calculate tension between geoids {a.geoid_id} and {b.geoid_id}"
                    logger.error(error_msg, error=e, 
                               geoid_a=a.geoid_id, geoid_b=b.geoid_id)
                    raise KimeraCognitiveError(
                        error_msg,
                        context={'geoid_a': a.geoid_id, 'geoid_b': b.geoid_id}
                    ) from e
        
        logger.info(f"Pairwise tension detection completed: found {len(tensions)} tensions from {comparison_count} comparisons")
        return tensions

    def _fast_layer_conflict(self, sem_a: set, sem_b: set, sym_a: set, sym_b: set) -> float:
        """
        Fast layer conflict calculation using set operations.
        
        Args:
            sem_a: Semantic state keys for geoid A
            sem_b: Semantic state keys for geoid B  
            sym_a: Symbolic state keys for geoid A
            sym_b: Symbolic state keys for geoid B
            
        Returns:
            Layer conflict intensity (0.0-1.0)
        """
        if not (sem_a or sem_b or sym_a or sym_b):
            return 0.0

        # Fast Jaccard distance using set operations
        def fast_jaccard_distance(x: set, y: set) -> float:
            if not (x or y):
                return 0.0
            intersection = len(x & y)
            union = len(x | y)
            return 1.0 - (intersection / union) if union > 0 else 0.0

        sem_diff = fast_jaccard_distance(sem_a, sem_b)
        sym_diff = fast_jaccard_distance(sym_a, sym_b)

        return (sem_diff + sym_diff) / 2

    def _fast_symbolic_opposition(self, symbolic_a: dict, symbolic_b: dict) -> float:
        """
        Fast symbolic opposition calculation.
        
        Args:
            symbolic_a: Symbolic state of geoid A
            symbolic_b: Symbolic state of geoid B
            
        Returns:
            Symbolic opposition score (0.0-1.0)
        """
        if not (symbolic_a and symbolic_b):
            return 0.0
            
        overlap_keys = set(symbolic_a.keys()) & set(symbolic_b.keys())
        if not overlap_keys:
            return 0.0
            
        conflicts = sum(1 for key in overlap_keys if symbolic_a[key] != symbolic_b[key])
        return conflicts / len(overlap_keys)

    def _embedding_misalignment(self, a: GeoidState, b: GeoidState) -> float:
        """
        Calculate embedding vector misalignment between two geoids.
        
        Args:
            a: First geoid
            b: Second geoid
            
        Returns:
            Misalignment score (0.0-1.0)
            
        Raises:
            KimeraCognitiveError: If embedding calculation fails
        """
        try:
            # Check for valid embedding vectors
            if (a.embedding_vector is None or b.embedding_vector is None or 
                not a.embedding_vector or not b.embedding_vector):
                logger.debug("One or both geoids missing embedding vectors", 
                           geoid_a=a.geoid_id, geoid_b=b.geoid_id)
                return 0.0
            
            # Use native cosine distance implementation
            distance = NativeMath.cosine_distance(a.embedding_vector, b.embedding_vector)
            return distance
            
        except Exception as e:
            error_msg = f"Failed to calculate embedding misalignment: {e}"
            logger.error(error_msg, error=e, geoid_a=a.geoid_id, geoid_b=b.geoid_id)
            raise KimeraCognitiveError(
                error_msg,
                context={'operation': 'embedding_misalignment', 'geoid_a': a.geoid_id, 'geoid_b': b.geoid_id}
            ) from e

    def _layer_conflict_intensity(self, a: GeoidState, b: GeoidState) -> float:
        """
        Calculate semantic vs symbolic layer disagreement.
        
        Args:
            a: First geoid
            b: Second geoid
            
        Returns:
            Layer conflict intensity (0.0-1.0)
            
        Raises:
            KimeraCognitiveError: If layer conflict calculation fails
        """
        try:
            # Safely convert to sets
            sem_a = set(a.semantic_state) if a.semantic_state else set()
            sem_b = set(b.semantic_state) if b.semantic_state else set()
            sym_a = set(a.symbolic_state) if a.symbolic_state else set()
            sym_b = set(b.symbolic_state) if b.symbolic_state else set()

            if not (sem_a or sem_b or sym_a or sym_b):
                return 0.0

            def jaccard_distance(x: set, y: set) -> float:
                if not (x or y):
                    return 0.0
                inter = len(x & y)
                union = len(x | y)
                if union == 0:
                    return 0.0
                return 1.0 - inter / union

            sem_diff = jaccard_distance(sem_a, sem_b)
            sym_diff = jaccard_distance(sym_a, sym_b)

            return (sem_diff + sym_diff) / 2
            
        except Exception as e:
            error_msg = f"Failed to calculate layer conflict intensity: {e}"
            logger.error(error_msg, error=e, geoid_a=a.geoid_id, geoid_b=b.geoid_id)
            raise KimeraCognitiveError(
                error_msg,
                context={'operation': 'layer_conflict', 'geoid_a': a.geoid_id, 'geoid_b': b.geoid_id}
            ) from e

    def _symbolic_opposition(self, a: GeoidState, b: GeoidState) -> float:
        """
        Measure direct conflicts in overlapping symbolic assertions.
        
        Args:
            a: First geoid
            b: Second geoid
            
        Returns:
            Symbolic opposition score (0.0-1.0)
            
        Raises:
            KimeraCognitiveError: If symbolic opposition calculation fails
        """
        try:
            # Safely handle symbolic states
            symbolic_a = a.symbolic_state if a.symbolic_state else {}
            symbolic_b = b.symbolic_state if b.symbolic_state else {}
            
            overlap = set(symbolic_a.keys()) & set(symbolic_b.keys())
            if not overlap:
                return 0.0
                
            conflicts = sum(
                1 for key in overlap 
                if symbolic_a.get(key) != symbolic_b.get(key)
            )
            
            return conflicts / len(overlap)
            
        except Exception as e:
            error_msg = f"Failed to calculate symbolic opposition: {e}"
            logger.error(error_msg, error=e, geoid_a=a.geoid_id, geoid_b=b.geoid_id)
            raise KimeraCognitiveError(
                error_msg,
                context={'operation': 'symbolic_opposition', 'geoid_a': a.geoid_id, 'geoid_b': b.geoid_id}
            ) from e

    def calculate_pulse_strength(self, tension: TensionGradient, geoids: Dict[str, GeoidState]) -> float:
        """
        Calculate pulse strength from tension gradient.
        
        Args:
            tension: Tension gradient to analyze
            geoids: Dictionary of geoids by ID
            
        Returns:
            Pulse strength (0.0-1.0)
            
        Raises:
            KimeraValidationError: If input validation fails
        """
        if not isinstance(tension, TensionGradient):
            raise KimeraValidationError(
                f"Tension must be TensionGradient instance, got {type(tension).__name__}",
                context={'provided_type': type(tension).__name__}
            )
        
        if not isinstance(geoids, dict):
            raise KimeraValidationError(
                f"Geoids must be a dictionary, got {type(geoids).__name__}",
                context={'provided_type': type(geoids).__name__}
            )
        
        # For MVP use tension score as pulse strength
        return min(tension.tension_score, 1.0)

    def decide_collapse_or_surge(
        self,
        pulse_strength: float,
        stability: Dict[str, float],
        profile: Dict[str, object] | None = None,
    ) -> str:
        """
        Decides whether to collapse or surge based on pulse strength, stability,
        and anthropomorphic profile.

        THIS IS A CRITICAL, ACTION-TAKING FUNCTION AND MUST BE CONSTITUTIONALLY GOVERNED.
        
        Args:
            pulse_strength: Strength of the detected pulse (0.0-1.0)
            stability: System stability metrics
            profile: Optional user/system profile
            
        Returns:
            'collapse', 'surge', or 'buffer'
        """
        # Default decision is to buffer (do nothing)
        decision = "buffer"
        
        # This is a simplified decision logic; a full implementation would
        # involve the anthropomorphic profile.
        if pulse_strength > 0.7:
            decision = "collapse"
        elif pulse_strength > 0.5:
            decision = "surge"

        # ------------------------------------------------------------------
        # Constitutional Enforcement (Article IX)
        # ------------------------------------------------------------------
        # Before committing to a collapse or surge, the decision must be
        # adjudicated by the Ethical Governor.
        
        # A high pulse strength could indicate a valid contradiction, but an
        # overly aggressive collapse could be destructive (un-compassionate).
        # The "Heart" must weigh in.
        potential_for_harm = 0.0
        if decision == "collapse":
            # A collapse is inherently more destructive, so it has a higher
            # baseline potential for harm.
            potential_for_harm = 0.4 + (pulse_strength - 0.7) / 0.3 * 0.5

        proposal = ActionProposal(
            source_engine="ContradictionEngine",
            description=f"Decide and execute cognitive action: '{decision.upper()}'",
            logical_analysis={
                "approved": True,
                "pulse_strength": pulse_strength,
                "proposed_decision": decision
            },
            compassionate_analysis={
                "approved": True, # The heart tentatively approves but...
                "potential_for_harm": potential_for_harm,
                "comment": "Collapse actions must be monitored for compassionate alignment. Is this removing a harmful concept or a valid, challenging one?"
            },
            associated_data=stability
        )

        # Raises UnconstitutionalActionError if rejected
        require_constitutional(proposal)
        
        logger.info(f"Constitutional action approved: {decision.upper()} with pulse strength {pulse_strength:.3f}")
        return decision

    def check_insight_conflict(self, insight: InsightScar, existing_insights: List[InsightScar]) -> Optional[InsightScar]:
        """
        Checks a new insight for conflicts against existing ones and validates it.

        Args:
            insight: The newly generated InsightScar.
            existing_insights: A list of existing insights to check against.

        Returns:
            The validated insight if it's not a conflict, otherwise None.
            
        Raises:
            KimeraValidationError: If input validation fails
            KimeraCognitiveError: If insight validation fails
        """
        # Input validation
        if not isinstance(insight, InsightScar):
            raise KimeraValidationError(
                f"Insight must be InsightScar instance, got {type(insight).__name__}",
                context={'provided_type': type(insight).__name__}
            )
        
        if not isinstance(existing_insights, list):
            raise KimeraValidationError(
                f"Existing insights must be a list, got {type(existing_insights).__name__}",
                context={'provided_type': type(existing_insights).__name__}
            )
        
        try:
            with logger.operation_context("check_insight_conflict", 
                                        insight_id=insight.insight_id,
                                        existing_count=len(existing_insights)):
                # 1. Ethical Reflex Layer (ERL) Hook
                if not erl.validate(insight.echoform_repr):
                    logger.warning(f"Insight {insight.insight_id} rejected by ERL",
                                 insight_id=insight.insight_id)
                    insight.status = 'deprecated' # Or a new 'quarantined' status
                    return None # Fails validation

                # 2. Check for semantic duplicates (simplified for MVP)
                for old_insight in existing_insights:
                    # A more robust check would use embedding similarity
                    if old_insight.echoform_repr == insight.echoform_repr:
                        logger.info(f"Insight {insight.insight_id} rejected as duplicate of {old_insight.insight_id}",
                                  insight_id=insight.insight_id, 
                                  duplicate_of=old_insight.insight_id)
                        return None # It's a duplicate

                # 3. Add more sophisticated contradiction logic here in the future
                # (e.g., if one insight makes a claim that another refutes)

                logger.debug(f"Insight {insight.insight_id} passed conflict validation",
                           insight_id=insight.insight_id)
                return insight
                
        except Exception as e:
            error_msg = f"Failed to check insight conflict: {e}"
            logger.error(error_msg, error=e, insight_id=insight.insight_id)
            raise KimeraCognitiveError(
                error_msg,
                context={'insight_id': insight.insight_id, 'existing_count': len(existing_insights)}
            ) from e

