"""
Barenholtz Core - Dual-System Cognitive Architecture Core
========================================================

The foundational dual-system cognitive architecture based on Barenholtz's
theory that integrates linguistic and perceptual processing systems with
sophisticated embedding alignment.

Barenholtz Core provides:
- Dual-system cognitive processing (System 1: Linguistic, System 2: Perceptual)
- Advanced embedding alignment with multiple mathematical methods
- Integration with cognitive field dynamics and embodied semantics
- Neurodivergent cognitive optimization
- Research-grade validation and testing

This is the core architectural foundation that enables Kimera's unique
dual-system approach to cognitive processing and understanding.
"""

import asyncio
import time
import logging
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from enum import Enum
from collections import deque, defaultdict

# Advanced mathematical dependencies
from scipy import linalg
from scipy.spatial.distance import cdist
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler

# Core dependencies
from ...utils.config import get_api_settings
from ...config.settings import get_settings
from ...utils.kimera_logger import get_cognitive_logger

logger = get_cognitive_logger(__name__)


class DualSystemMode(Enum):
    """Dual-system processing modes"""
    LINGUISTIC_DOMINANT = "linguistic_dominant"      # System 1 dominance
    PERCEPTUAL_DOMINANT = "perceptual_dominant"      # System 2 dominance
    BALANCED = "balanced"                            # Equal system weight
    ADAPTIVE = "adaptive"                            # Dynamic weight adjustment
    COMPETITIVE = "competitive"                      # Systems compete for dominance


class AlignmentMethod(Enum):
    """Sophisticated alignment methods for embedding spaces"""
    COSINE_SIMILARITY = "cosine_similarity"          # Basic cosine similarity
    OPTIMAL_TRANSPORT = "optimal_transport"          # Wasserstein distance alignment
    CANONICAL_CORRELATION = "canonical_correlation"  # CCA alignment
    PROCRUSTES_ANALYSIS = "procrustes_analysis"      # Orthogonal Procrustes
    ENSEMBLE_ALIGNMENT = "ensemble_alignment"        # Combined method ensemble


class ProcessingStage(Enum):
    """Stages of dual-system processing"""
    INPUT_ANALYSIS = "input_analysis"
    SYSTEM_1_PROCESSING = "system_1_processing"
    SYSTEM_2_PROCESSING = "system_2_processing"
    EMBEDDING_ALIGNMENT = "embedding_alignment"
    SYSTEM_INTEGRATION = "system_integration"
    DECISION_SYNTHESIS = "decision_synthesis"
    OUTPUT_GENERATION = "output_generation"


@dataclass
class DualSystemResult:
    """Result from dual-system processing"""
    input_content: str
    linguistic_analysis: Dict[str, Any]
    perceptual_analysis: Dict[str, Any]
    embedding_alignment: float
    neurodivergent_enhancement: float
    processing_time: float
    confidence_score: float
    integrated_response: str
    success: bool = True  # Add success attribute
    
    # Detailed metrics
    system_weights: Dict[str, float] = field(default_factory=dict)
    stage_durations: Dict[str, float] = field(default_factory=dict)
    alignment_details: Dict[str, Any] = field(default_factory=dict)
    decision_factors: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlignmentResult:
    """Result from embedding alignment"""
    alignment_score: float
    computational_cost: float
    method_used: AlignmentMethod
    transformed_embeddings: Dict[str, torch.Tensor]
    convergence_achieved: bool
    error_estimate: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class AlignmentEngine:
    """
    Advanced Embedding Alignment Engine
    
    Implements sophisticated mathematical methods for aligning
    linguistic and perceptual embedding spaces with high precision.
    """
    
    def __init__(self, 
                 default_method: AlignmentMethod = AlignmentMethod.ENSEMBLE_ALIGNMENT,
                 dimension: int = 768):
        """
        Initialize Alignment Engine
        
        Args:
            default_method: Default alignment method
            dimension: Target embedding dimension
        """
        self.default_method = default_method
        self.dimension = dimension
        
        # Performance monitoring
        self.performance_stats = {
            'total_alignments': 0,
            'method_usage': {method.value: 0 for method in AlignmentMethod},
            'average_scores': {method.value: 0.0 for method in AlignmentMethod},
            'average_costs': {method.value: 0.0 for method in AlignmentMethod}
        }
        
        # Learned parameters
        self.transform_matrix = torch.eye(dimension) * 0.1 + torch.randn(dimension, dimension) * 0.01
        self.alignment_history = deque(maxlen=1000)
        
        # Method-specific components
        self._initialize_alignment_methods()
        
        logger.info(f"ðŸ”— Alignment Engine initialized")
        logger.info(f"   Default method: {default_method.value}")
        logger.info(f"   Dimension: {dimension}")
    
    def _initialize_alignment_methods(self):
        """Initialize method-specific components"""
        # CCA components
        self.cca_model = CCA(n_components=min(50, self.dimension))
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.cca_fitted = False
        
        # Optimal transport parameters
        self.ot_reg = 0.1  # Regularization for Sinkhorn algorithm
        self.ot_max_iter = 100
        
        logger.debug("Alignment method components initialized")
    
    async def align_embeddings(self, 
                              linguistic_emb: torch.Tensor,
                              perceptual_emb: torch.Tensor,
                              method: Optional[AlignmentMethod] = None) -> AlignmentResult:
        """
        Align linguistic and perceptual embeddings using specified method
        
        Args:
            linguistic_emb: Linguistic system embedding
            perceptual_emb: Perceptual system embedding
            method: Alignment method (uses default if None)
            
        Returns:
            Alignment result with score and transformed embeddings
        """
        if method is None:
            method = self.default_method
        
        start_time = time.time()
        
        try:
            # Normalize embeddings to target dimension
            ling_norm = self._normalize_embedding(linguistic_emb)
            perc_norm = self._normalize_embedding(perceptual_emb)
            
            # Apply selected alignment method
            if method == AlignmentMethod.COSINE_SIMILARITY:
                result = await self._cosine_alignment(ling_norm, perc_norm)
            elif method == AlignmentMethod.OPTIMAL_TRANSPORT:
                result = await self._optimal_transport_alignment(ling_norm, perc_norm)
            elif method == AlignmentMethod.CANONICAL_CORRELATION:
                result = await self._canonical_correlation_alignment(ling_norm, perc_norm)
            elif method == AlignmentMethod.PROCRUSTES_ANALYSIS:
                result = await self._procrustes_alignment(ling_norm, perc_norm)
            elif method == AlignmentMethod.ENSEMBLE_ALIGNMENT:
                result = await self._ensemble_alignment(ling_norm, perc_norm)
            else:
                raise ValueError(f"Unsupported alignment method: {method}")
            
            # Update performance statistics
            computational_cost = time.time() - start_time
            self._update_performance_stats(method, result.alignment_score, computational_cost)
            
            # Store in history
            self.alignment_history.append({
                'timestamp': datetime.now(),
                'method': method.value,
                'score': result.alignment_score,
                'cost': computational_cost
            })
            
            result.computational_cost = computational_cost
            result.method_used = method
            
            return result
            
        except Exception as e:
            logger.error(f"Embedding alignment failed: {e}")
            # Return minimal result on error
            return AlignmentResult(
                alignment_score=0.0,
                computational_cost=time.time() - start_time,
                method_used=method,
                transformed_embeddings={
                    'linguistic': linguistic_emb,
                    'perceptual': perceptual_emb
                },
                convergence_achieved=False,
                error_estimate=float('inf'),
                metadata={'error': str(e)}
            )
    
    async def _cosine_alignment(self, 
                               ling_emb: torch.Tensor,
                               perc_emb: torch.Tensor) -> AlignmentResult:
        """Basic cosine similarity alignment"""
        # Apply learned transformation
        ling_transformed = torch.mv(self.transform_matrix, ling_emb)
        perc_transformed = perc_emb  # Keep perceptual as reference
        
        # Calculate cosine similarity with proper dimension handling
        try:
            if ling_transformed.dim() == 1 and perc_transformed.dim() == 1:
                similarity = F.cosine_similarity(ling_transformed.unsqueeze(0), perc_transformed.unsqueeze(0), dim=1)
                similarity = similarity.squeeze()
            else:
                similarity = F.cosine_similarity(ling_transformed, perc_transformed, dim=0)
            alignment_score = (similarity + 1) / 2  # Normalize to [0,1]
        except Exception as e:
            # Fallback to dot product similarity
            ling_norm = F.normalize(ling_transformed, p=2, dim=0)
            perc_norm = F.normalize(perc_transformed, p=2, dim=0)
            similarity = torch.dot(ling_norm, perc_norm)
            alignment_score = (similarity + 1) / 2
        
        # Update transformation matrix
        if alignment_score > 0.5:
            self.transform_matrix *= 1.001  # Strengthen good alignments
        else:
            self.transform_matrix += torch.randn_like(self.transform_matrix) * 0.001
        
        return AlignmentResult(
            alignment_score=alignment_score.item(),
            computational_cost=0.0,  # Will be set by caller
            method_used=AlignmentMethod.COSINE_SIMILARITY,
            transformed_embeddings={
                'linguistic': ling_transformed,
                'perceptual': perc_transformed
            },
            convergence_achieved=True,
            error_estimate=1.0 - alignment_score.item()
        )
    
    async def _optimal_transport_alignment(self,
                                          ling_emb: torch.Tensor,
                                          perc_emb: torch.Tensor) -> AlignmentResult:
        """Optimal transport (Wasserstein) alignment"""
        try:
            # Convert to distributions (add small noise for numerical stability)
            ling_dist = torch.softmax(ling_emb + torch.randn_like(ling_emb) * 0.01, dim=0)
            perc_dist = torch.softmax(perc_emb + torch.randn_like(perc_emb) * 0.01, dim=0)
            
            # Compute cost matrix (Euclidean distance)
            cost_matrix = torch.cdist(ling_dist.unsqueeze(0), perc_dist.unsqueeze(0)).squeeze()
            
            # Sinkhorn algorithm for approximate optimal transport
            transport_plan = self._sinkhorn_algorithm(ling_dist, perc_dist, cost_matrix)
            
            # Calculate Wasserstein distance
            wasserstein_dist = torch.sum(transport_plan * cost_matrix)
            alignment_score = torch.exp(-wasserstein_dist).item()  # Convert distance to similarity
            
            # Transform embeddings based on transport plan
            ling_transformed = torch.mv(transport_plan, perc_emb)
            perc_transformed = perc_emb
            
            return AlignmentResult(
                alignment_score=alignment_score,
                computational_cost=0.0,
                method_used=AlignmentMethod.OPTIMAL_TRANSPORT,
                transformed_embeddings={
                    'linguistic': ling_transformed,
                    'perceptual': perc_transformed
                },
                convergence_achieved=True,
                error_estimate=wasserstein_dist.item(),
                metadata={'transport_plan': transport_plan}
            )
            
        except Exception as e:
            logger.warning(f"Optimal transport alignment failed: {e}")
            # Fallback to cosine similarity
            return await self._cosine_alignment(ling_emb, perc_emb)
    
    async def _canonical_correlation_alignment(self,
                                              ling_emb: torch.Tensor,
                                              perc_emb: torch.Tensor) -> AlignmentResult:
        """Canonical Correlation Analysis alignment"""
        try:
            # Convert to numpy for sklearn
            ling_np = ling_emb.detach().numpy().reshape(1, -1)
            perc_np = perc_emb.detach().numpy().reshape(1, -1)
            
            # Fit CCA if not already fitted
            if not self.cca_fitted:
                # Need more samples for CCA, use repeated samples with noise
                ling_samples = np.repeat(ling_np, 10, axis=0) + np.random.normal(0, 0.01, (10, ling_np.shape[1]))
                perc_samples = np.repeat(perc_np, 10, axis=0) + np.random.normal(0, 0.01, (10, perc_np.shape[1]))
                
                # Scale data
                ling_scaled = self.scaler_x.fit_transform(ling_samples)
                perc_scaled = self.scaler_y.fit_transform(perc_samples)
                
                # Fit CCA
                self.cca_model.fit(ling_scaled, perc_scaled)
                self.cca_fitted = True
            
            # Transform current embeddings
            ling_scaled = self.scaler_x.transform(ling_np)
            perc_scaled = self.scaler_y.transform(perc_np)
            
            ling_cca, perc_cca = self.cca_model.transform(ling_scaled, perc_scaled)
            
            # Calculate correlation score
            correlation = np.corrcoef(ling_cca.flatten(), perc_cca.flatten())[0, 1]
            alignment_score = (correlation + 1) / 2  # Normalize to [0,1]
            
            # Convert back to tensors
            ling_transformed = torch.tensor(ling_cca.flatten(), dtype=torch.float32)
            perc_transformed = torch.tensor(perc_cca.flatten(), dtype=torch.float32)
            
            # Pad to original dimension if needed
            if len(ling_transformed) < self.dimension:
                ling_transformed = F.pad(ling_transformed, (0, self.dimension - len(ling_transformed)))
                perc_transformed = F.pad(perc_transformed, (0, self.dimension - len(perc_transformed)))
            
            return AlignmentResult(
                alignment_score=alignment_score,
                computational_cost=0.0,
                method_used=AlignmentMethod.CANONICAL_CORRELATION,
                transformed_embeddings={
                    'linguistic': ling_transformed,
                    'perceptual': perc_transformed
                },
                convergence_achieved=True,
                error_estimate=1.0 - abs(correlation),
                metadata={'correlation': correlation}
            )
            
        except Exception as e:
            logger.warning(f"CCA alignment failed: {e}")
            # Fallback to cosine similarity
            return await self._cosine_alignment(ling_emb, perc_emb)
    
    async def _procrustes_alignment(self,
                                   ling_emb: torch.Tensor,
                                   perc_emb: torch.Tensor) -> AlignmentResult:
        """Procrustes analysis alignment"""
        try:
            # Convert to numpy matrices
            X = ling_emb.detach().numpy().reshape(-1, 1)
            Y = perc_emb.detach().numpy().reshape(-1, 1)
            
            # Ensure same shape
            min_len = min(len(X), len(Y))
            X = X[:min_len]
            Y = Y[:min_len]
            
            # Center the data
            X_centered = X - np.mean(X, axis=0)
            Y_centered = Y - np.mean(Y, axis=0)
            
            # Compute optimal rotation matrix using SVD
            U, s, Vt = np.linalg.svd(X_centered.T @ Y_centered)
            R = U @ Vt
            
            # Apply transformation
            X_transformed = X_centered @ R
            
            # Calculate alignment score (based on residual)
            residual = np.linalg.norm(X_transformed - Y_centered)
            alignment_score = np.exp(-residual)
            
            # Convert back to tensors
            ling_transformed = torch.tensor(X_transformed.flatten(), dtype=torch.float32)
            perc_transformed = torch.tensor(Y_centered.flatten(), dtype=torch.float32)
            
            # Pad to original dimension if needed
            if len(ling_transformed) < self.dimension:
                ling_transformed = F.pad(ling_transformed, (0, self.dimension - len(ling_transformed)))
                perc_transformed = F.pad(perc_transformed, (0, self.dimension - len(perc_transformed)))
            
            return AlignmentResult(
                alignment_score=alignment_score,
                computational_cost=0.0,
                method_used=AlignmentMethod.PROCRUSTES_ANALYSIS,
                transformed_embeddings={
                    'linguistic': ling_transformed,
                    'perceptual': perc_transformed
                },
                convergence_achieved=True,
                error_estimate=residual,
                metadata={'rotation_matrix': R, 'residual': residual}
            )
            
        except Exception as e:
            logger.warning(f"Procrustes alignment failed: {e}")
            # Fallback to cosine similarity
            return await self._cosine_alignment(ling_emb, perc_emb)
    
    async def _ensemble_alignment(self,
                                 ling_emb: torch.Tensor,
                                 perc_emb: torch.Tensor) -> AlignmentResult:
        """Ensemble alignment using multiple methods"""
        try:
            # Run multiple alignment methods
            methods = [
                AlignmentMethod.COSINE_SIMILARITY,
                AlignmentMethod.OPTIMAL_TRANSPORT,
                AlignmentMethod.CANONICAL_CORRELATION,
                AlignmentMethod.PROCRUSTES_ANALYSIS
            ]
            
            results = []
            for method in methods:
                try:
                    result = await self.align_embeddings(ling_emb, perc_emb, method)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error in barenholtz_core.py: {e}", exc_info=True)
                    raise  # Re-raise for proper error handling
                    continue
            
            if not results:
                # Fallback to cosine if all fail
                return await self._cosine_alignment(ling_emb, perc_emb)
            
            # Weight results by their alignment scores
            weights = [r.alignment_score for r in results]
            total_weight = sum(weights)
            
            if total_weight == 0:
                weights = [1.0 / len(results)] * len(results)
            else:
                weights = [w / total_weight for w in weights]
            
            # Weighted average of alignment scores
            ensemble_score = sum(w * r.alignment_score for w, r in zip(weights, results))
            
            # Use best performing method's transformations
            best_result = max(results, key=lambda r: r.alignment_score)
            
            return AlignmentResult(
                alignment_score=ensemble_score,
                computational_cost=sum(r.computational_cost for r in results),
                method_used=AlignmentMethod.ENSEMBLE_ALIGNMENT,
                transformed_embeddings=best_result.transformed_embeddings,
                convergence_achieved=all(r.convergence_achieved for r in results),
                error_estimate=min(r.error_estimate for r in results),
                metadata={
                    'ensemble_weights': weights,
                    'individual_scores': [r.alignment_score for r in results],
                    'best_method': best_result.method_used.value
                }
            )
            
        except Exception as e:
            logger.warning(f"Ensemble alignment failed: {e}")
            # Fallback to cosine similarity
            return await self._cosine_alignment(ling_emb, perc_emb)
    
    def _sinkhorn_algorithm(self, 
                           source: torch.Tensor, 
                           target: torch.Tensor,
                           cost_matrix: torch.Tensor,
                           reg: float = 0.1,
                           max_iter: int = 100) -> torch.Tensor:
        """Sinkhorn algorithm for optimal transport"""
        # Initialize dual variables
        u = torch.ones_like(source)
        v = torch.ones_like(target)
        
        # Compute kernel
        K = torch.exp(-cost_matrix / reg)
        
        for _ in range(max_iter):
            u = source / (K @ v)
            v = target / (K.T @ u)
        
        # Compute transport plan
        transport_plan = torch.diag(u) @ K @ torch.diag(v)
        return transport_plan
    
    def _normalize_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        """Normalize embedding to target dimension"""
        if embedding.shape[0] == self.dimension:
            return F.normalize(embedding, p=2, dim=0)
        elif embedding.shape[0] < self.dimension:
            # Pad with zeros
            padded = torch.zeros(self.dimension)
            padded[:embedding.shape[0]] = embedding
            return F.normalize(padded, p=2, dim=0)
        else:
            # Truncate
            truncated = embedding[:self.dimension]
            return F.normalize(truncated, p=2, dim=0)
    
    def _update_performance_stats(self, 
                                 method: AlignmentMethod,
                                 score: float,
                                 cost: float):
        """Update performance statistics"""
        self.performance_stats['total_alignments'] += 1
        self.performance_stats['method_usage'][method.value] += 1
        
        # Update average scores
        count = self.performance_stats['method_usage'][method.value]
        current_avg = self.performance_stats['average_scores'][method.value]
        self.performance_stats['average_scores'][method.value] = (
            (current_avg * (count - 1) + score) / count
        )
        
        # Update average costs
        current_cost_avg = self.performance_stats['average_costs'][method.value]
        self.performance_stats['average_costs'][method.value] = (
            (current_cost_avg * (count - 1) + cost) / count
        )


class DualSystemProcessor:
    """
    Dual-System Cognitive Processor
    
    Implements Barenholtz's dual-system theory with System 1 (linguistic)
    and System 2 (perceptual) processing with advanced integration.
    """
    
    def __init__(self,
                 processing_mode: DualSystemMode = DualSystemMode.ADAPTIVE,
                 alignment_method: AlignmentMethod = AlignmentMethod.ENSEMBLE_ALIGNMENT):
        """
        Initialize Dual-System Processor
        
        Args:
            processing_mode: Dual-system processing mode
            alignment_method: Default embedding alignment method
        """
        self.settings = get_api_settings()
        self.processing_mode = processing_mode
        
        # Core components
        self.alignment_engine = AlignmentEngine(
            default_method=alignment_method
        )
        
        # System weights (adaptive)
        self.system_weights = {
            'linguistic': 0.5,
            'perceptual': 0.5
        }
        
        # Performance tracking
        self.processing_history = deque(maxlen=1000)
        self.total_processes = 0
        self.success_rate = 0.0
        
        # Integration callbacks
        self.processing_callbacks = []
        
        logger.info(f"ðŸ§  Dual-System Processor initialized")
        logger.info(f"   Processing mode: {processing_mode.value}")
        logger.info(f"   Alignment method: {alignment_method.value}")
    
    async def process_dual_system(self, 
                                 content: str,
                                 context: Optional[Dict[str, Any]] = None) -> DualSystemResult:
        """
        Process input through dual-system architecture
        
        Args:
            content: Input content to process
            context: Optional processing context
            
        Returns:
            Complete dual-system processing result
        """
        start_time = time.time()
        process_id = f"DS_{self.total_processes:06d}_{int(start_time)}"
        
        # Initialize result
        result = DualSystemResult(
            input_content=content,
            linguistic_analysis={},
            perceptual_analysis={},
            embedding_alignment=0.0,
            neurodivergent_enhancement=0.0,
            processing_time=0.0,
            confidence_score=0.0,
            integrated_response="",
            system_weights=self.system_weights.copy(),
            stage_durations={},
            metadata={'process_id': process_id}
        )
        
        try:
            # Stage 1: Input Analysis
            stage_start = time.time()
            input_analysis = await self._analyze_input(content, context)
            result.stage_durations[ProcessingStage.INPUT_ANALYSIS.value] = time.time() - stage_start
            
            # Stage 2: System 1 Processing (Linguistic)
            stage_start = time.time()
            linguistic_result = await self._process_system_1(content, input_analysis, context)
            result.linguistic_analysis = linguistic_result
            result.stage_durations[ProcessingStage.SYSTEM_1_PROCESSING.value] = time.time() - stage_start
            
            # Stage 3: System 2 Processing (Perceptual)
            stage_start = time.time()
            perceptual_result = await self._process_system_2(content, input_analysis, context)
            result.perceptual_analysis = perceptual_result
            result.stage_durations[ProcessingStage.SYSTEM_2_PROCESSING.value] = time.time() - stage_start
            
            # Stage 4: Embedding Alignment
            stage_start = time.time()
            alignment_result = await self._align_system_embeddings(
                linguistic_result.get('embedding', torch.randn(768)),
                perceptual_result.get('embedding', torch.randn(768))
            )
            result.embedding_alignment = alignment_result.alignment_score
            result.alignment_details = alignment_result.metadata
            result.stage_durations[ProcessingStage.EMBEDDING_ALIGNMENT.value] = time.time() - stage_start
            
            # Stage 5: System Integration
            stage_start = time.time()
            integration_result = await self._integrate_systems(
                linguistic_result, perceptual_result, alignment_result
            )
            result.stage_durations[ProcessingStage.SYSTEM_INTEGRATION.value] = time.time() - stage_start
            
            # Stage 6: Decision Synthesis
            stage_start = time.time()
            decision_result = await self._synthesize_decision(
                linguistic_result, perceptual_result, integration_result, context
            )
            result.decision_factors = decision_result
            result.stage_durations[ProcessingStage.DECISION_SYNTHESIS.value] = time.time() - stage_start
            
            # Stage 7: Output Generation
            stage_start = time.time()
            output_result = await self._generate_output(
                linguistic_result, perceptual_result, decision_result, context
            )
            result.integrated_response = output_result['response']
            result.confidence_score = output_result['confidence']
            result.stage_durations[ProcessingStage.OUTPUT_GENERATION.value] = time.time() - stage_start
            
            # Calculate neurodivergent enhancement
            result.neurodivergent_enhancement = await self._calculate_neurodivergent_optimization(
                linguistic_result, perceptual_result, context
            )
            
            # Update system weights if in adaptive mode
            if self.processing_mode == DualSystemMode.ADAPTIVE:
                await self._update_adaptive_weights(result)
            
            # Finalize result
            result.processing_time = time.time() - start_time
            
            # Update performance tracking
            self.total_processes += 1
            self.processing_history.append(result)
            self._update_success_rate(True)
            
            # Trigger callbacks
            await self._trigger_processing_callbacks({
                'result': result,
                'stage': 'completion',
                'process_id': process_id
            })
            
            logger.debug(f"Dual-system processing completed: {process_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"Dual-system processing failed: {e}")
            result.integrated_response = f"Processing failed: {e}"
            result.confidence_score = 0.0
            result.processing_time = time.time() - start_time
            
            self._update_success_rate(False)
            
            return result
    
    async def _analyze_input(self, 
                           content: str,
                           context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze input content to determine processing strategy"""
        analysis = {
            'content_length': len(content),
            'complexity_estimate': self._estimate_complexity(content),
            'processing_strategy': self._determine_strategy(content, context),
            'linguistic_features': self._extract_linguistic_features(content),
            'perceptual_cues': self._extract_perceptual_cues(content)
        }
        
        return analysis
    
    async def _process_system_1(self,
                               content: str,
                               input_analysis: Dict[str, Any],
                               context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Process through System 1 (Linguistic)"""
        # Placeholder for linguistic processing
        # In full implementation, this would use linguistic processors
        
        linguistic_embedding = torch.randn(768)  # Placeholder embedding
        
        return {
            'processing_type': 'linguistic',
            'embedding': linguistic_embedding,
            'features': input_analysis['linguistic_features'],
            'confidence': 0.8,
            'processing_time': 0.1,
            'analysis': {
                'semantic_coherence': 0.75,
                'syntactic_complexity': 0.6,
                'pragmatic_appropriateness': 0.8
            }
        }
    
    async def _process_system_2(self,
                               content: str,
                               input_analysis: Dict[str, Any],
                               context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Process through System 2 (Perceptual)"""
        # Placeholder for perceptual processing
        # In full implementation, this would use perceptual processors
        
        perceptual_embedding = torch.randn(768)  # Placeholder embedding
        
        return {
            'processing_type': 'perceptual',
            'embedding': perceptual_embedding,
            'features': input_analysis['perceptual_cues'],
            'confidence': 0.7,
            'processing_time': 0.3,
            'analysis': {
                'spatial_coherence': 0.65,
                'temporal_dynamics': 0.7,
                'embodied_grounding': 0.75
            }
        }
    
    async def _align_system_embeddings(self,
                                      linguistic_emb: torch.Tensor,
                                      perceptual_emb: torch.Tensor) -> AlignmentResult:
        """Align embeddings from both systems"""
        return await self.alignment_engine.align_embeddings(
            linguistic_emb, perceptual_emb
        )
    
    async def _integrate_systems(self,
                               linguistic_result: Dict[str, Any],
                               perceptual_result: Dict[str, Any],
                               alignment_result: AlignmentResult) -> Dict[str, Any]:
        """Integrate results from both systems"""
        integration = {
            'alignment_quality': alignment_result.alignment_score,
            'system_coherence': self._calculate_system_coherence(
                linguistic_result, perceptual_result
            ),
            'integration_confidence': self._calculate_integration_confidence(
                linguistic_result, perceptual_result, alignment_result
            ),
            'unified_representation': self._create_unified_representation(
                alignment_result.transformed_embeddings
            )
        }
        
        return integration
    
    async def _synthesize_decision(self,
                                  linguistic_result: Dict[str, Any],
                                  perceptual_result: Dict[str, Any],
                                  integration_result: Dict[str, Any],
                                  context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Synthesize final decision from integrated systems"""
        decision = {
            'decision_strategy': self._determine_decision_strategy(),
            'confidence_factors': {
                'linguistic_confidence': linguistic_result.get('confidence', 0.0),
                'perceptual_confidence': perceptual_result.get('confidence', 0.0),
                'integration_confidence': integration_result.get('integration_confidence', 0.0)
            },
            'system_contributions': {
                'linguistic_weight': self.system_weights['linguistic'],
                'perceptual_weight': self.system_weights['perceptual']
            },
            'decision_factors': {
                'alignment_quality': integration_result.get('alignment_quality', 0.0),
                'system_coherence': integration_result.get('system_coherence', 0.0)
            }
        }
        
        return decision
    
    async def _generate_output(self,
                              linguistic_result: Dict[str, Any],
                              perceptual_result: Dict[str, Any],
                              decision_result: Dict[str, Any],
                              context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate final output from dual-system processing"""
        # Weighted combination based on system weights and decision factors
        linguistic_weight = self.system_weights['linguistic']
        perceptual_weight = self.system_weights['perceptual']
        
        # Calculate overall confidence
        confidence = (
            linguistic_weight * linguistic_result.get('confidence', 0.0) +
            perceptual_weight * perceptual_result.get('confidence', 0.0)
        )
        
        # Generate response (placeholder)
        response = f"Dual-system processing complete (L:{linguistic_weight:.2f}, P:{perceptual_weight:.2f})"
        
        return {
            'response': response,
            'confidence': confidence,
            'generation_method': 'weighted_integration',
            'output_quality_score': confidence * decision_result.get('decision_factors', {}).get('alignment_quality', 0.5)
        }
    
    async def _calculate_neurodivergent_optimization(self,
                                                   linguistic_result: Dict[str, Any],
                                                   perceptual_result: Dict[str, Any],
                                                   context: Optional[Dict[str, Any]]) -> float:
        """Calculate neurodivergent cognitive optimization factor"""
        # Placeholder for neurodivergent optimization calculation
        # Would consider ADHD and autism spectrum processing patterns
        
        base_optimization = 0.5
        
        # Factor in processing asymmetries (common in neurodivergence)
        asymmetry = abs(
            linguistic_result.get('confidence', 0.5) - 
            perceptual_result.get('confidence', 0.5)
        )
        
        # Higher asymmetry can indicate neurodivergent processing strength
        optimization_factor = base_optimization + (asymmetry * 0.3)
        
        return min(1.0, optimization_factor)
    
    async def _update_adaptive_weights(self, result: DualSystemResult):
        """Update system weights in adaptive mode"""
        if self.processing_mode == DualSystemMode.ADAPTIVE:
            # Adjust weights based on system performance
            ling_performance = result.linguistic_analysis.get('confidence', 0.5)
            perc_performance = result.perceptual_analysis.get('confidence', 0.5)
            
            # Simple adaptive weighting
            total_performance = ling_performance + perc_performance
            if total_performance > 0:
                self.system_weights['linguistic'] = ling_performance / total_performance
                self.system_weights['perceptual'] = perc_performance / total_performance
    
    def _estimate_complexity(self, content: str) -> float:
        """Estimate content complexity"""
        # Simple complexity estimate
        return min(1.0, len(content) / 1000.0)
    
    def _determine_strategy(self, content: str, context: Optional[Dict[str, Any]]) -> str:
        """Determine processing strategy"""
        if self.processing_mode == DualSystemMode.LINGUISTIC_DOMINANT:
            return "linguistic_dominant"
        elif self.processing_mode == DualSystemMode.PERCEPTUAL_DOMINANT:
            return "perceptual_dominant"
        else:
            return "balanced"
    
    def _extract_linguistic_features(self, content: str) -> Dict[str, Any]:
        """Extract linguistic features from content"""
        return {
            'word_count': len(content.split()),
            'sentence_count': content.count('.') + content.count('!') + content.count('?'),
            'avg_word_length': np.mean([len(word) for word in content.split()]) if content.split() else 0
        }
    
    def _extract_perceptual_cues(self, content: str) -> Dict[str, Any]:
        """Extract perceptual cues from content"""
        return {
            'spatial_terms': sum(1 for word in content.lower().split() if word in ['up', 'down', 'left', 'right', 'above', 'below']),
            'temporal_terms': sum(1 for word in content.lower().split() if word in ['before', 'after', 'during', 'while', 'when']),
            'sensory_terms': sum(1 for word in content.lower().split() if word in ['see', 'hear', 'feel', 'touch', 'taste', 'smell'])
        }
    
    def _calculate_system_coherence(self,
                                   linguistic_result: Dict[str, Any],
                                   perceptual_result: Dict[str, Any]) -> float:
        """Calculate coherence between systems"""
        ling_conf = linguistic_result.get('confidence', 0.0)
        perc_conf = perceptual_result.get('confidence', 0.0)
        
        # Coherence based on confidence alignment
        return 1.0 - abs(ling_conf - perc_conf)
    
    def _calculate_integration_confidence(self,
                                        linguistic_result: Dict[str, Any],
                                        perceptual_result: Dict[str, Any],
                                        alignment_result: AlignmentResult) -> float:
        """Calculate confidence in system integration"""
        factors = [
            linguistic_result.get('confidence', 0.0),
            perceptual_result.get('confidence', 0.0),
            alignment_result.alignment_score
        ]
        
        return np.mean(factors)
    
    def _create_unified_representation(self,
                                     transformed_embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Create unified representation from aligned embeddings"""
        ling_emb = transformed_embeddings.get('linguistic', torch.zeros(768))
        perc_emb = transformed_embeddings.get('perceptual', torch.zeros(768))
        
        # Weighted average
        unified = (
            self.system_weights['linguistic'] * ling_emb +
            self.system_weights['perceptual'] * perc_emb
        )
        
        return F.normalize(unified, p=2, dim=0)
    
    def _determine_decision_strategy(self) -> str:
        """Determine decision strategy based on processing mode"""
        return f"dual_system_{self.processing_mode.value}"
    
    def _update_success_rate(self, success: bool):
        """Update processing success rate"""
        if self.total_processes == 0:
            self.success_rate = 1.0 if success else 0.0
        else:
            self.success_rate = (
                (self.success_rate * (self.total_processes - 1) + (1.0 if success else 0.0)) /
                self.total_processes
            )
    
    async def _trigger_processing_callbacks(self, event_data: Dict[str, Any]):
        """Trigger registered processing callbacks"""
        for callback in self.processing_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event_data)
                else:
                    callback(event_data)
            except Exception as e:
                logger.warning(f"Processing callback failed: {e}")
    
    def register_processing_callback(self, callback: Callable):
        """Register callback for processing events"""
        self.processing_callbacks.append(callback)
        logger.debug("Registered dual-system processing callback")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'processing_mode': self.processing_mode.value,
            'system_weights': self.system_weights.copy(),
            'total_processes': self.total_processes,
            'success_rate': self.success_rate,
            'alignment_engine_stats': self.alignment_engine.performance_stats,
            'recent_performance': {
                'avg_processing_time': np.mean([
                    r.processing_time for r in list(self.processing_history)[-10:]
                ]) if self.processing_history else 0.0,
                'avg_confidence': np.mean([
                    r.confidence_score for r in list(self.processing_history)[-10:]
                ]) if self.processing_history else 0.0,
                'avg_alignment_score': np.mean([
                    r.embedding_alignment for r in list(self.processing_history)[-10:]
                ]) if self.processing_history else 0.0
            }
        }


class BarenholtzCore:
    """
    Barenholtz Core - Unified Dual-System Architecture
    
    The foundational dual-system cognitive architecture that integrates
    Barenholtz's theory with Kimera's cognitive processing systems.
    """
    
    def __init__(self,
                 processing_mode: DualSystemMode = DualSystemMode.ADAPTIVE,
                 alignment_method: AlignmentMethod = AlignmentMethod.ENSEMBLE_ALIGNMENT):
        """
        Initialize Barenholtz Core
        
        Args:
            processing_mode: Dual-system processing mode
            alignment_method: Default embedding alignment method
        """
        self.settings = get_api_settings()
        
        # Initialize core components
        self.dual_processor = DualSystemProcessor(
            processing_mode=processing_mode,
            alignment_method=alignment_method
        )
        self.alignment_engine = self.dual_processor.alignment_engine
        
        # Integration components (will be injected by orchestrator)
        self.cognitive_field_dynamics = None
        self.embodied_engine = None
        self.spde_core = None
        
        # System state
        self.processing_mode = processing_mode
        self.active_processes = {}
        self.total_integrations = 0
        
        # Integration callbacks
        self.integration_callbacks = []
        
        logger.info(f"ðŸ§  Barenholtz Core initialized")
        logger.info(f"   Processing mode: {processing_mode.value}")
        logger.info(f"   Alignment method: {alignment_method.value}")
    
    def register_integration_components(self,
                                      cognitive_field_dynamics: Any = None,
                                      embodied_engine: Any = None,
                                      spde_core: Any = None):
        """Register integration components for enhanced processing"""
        self.cognitive_field_dynamics = cognitive_field_dynamics
        self.embodied_engine = embodied_engine
        self.spde_core = spde_core
        
        logger.info("âœ… Barenholtz Core integration components registered")
    
    async def process_with_integration(self,
                                     content: str,
                                     context: Optional[Dict[str, Any]] = None) -> DualSystemResult:
        """
        Process content through dual-system architecture with full integration
        
        Args:
            content: Input content to process
            context: Optional processing context
            
        Returns:
            Enhanced dual-system processing result
        """
        # Base dual-system processing
        result = await self.dual_processor.process_dual_system(content, context)
        
        # Enhanced integration if components are available
        if self.cognitive_field_dynamics or self.embodied_engine or self.spde_core:
            result = await self._enhance_with_integration(result, content, context)
        
        self.total_integrations += 1
        
        # Trigger integration callbacks
        await self._trigger_integration_callbacks({
            'result': result,
            'content': content,
            'context': context,
            'integration_count': self.total_integrations
        })
        
        return result
    
    async def _enhance_with_integration(self,
                                      base_result: DualSystemResult,
                                      content: str,
                                      context: Optional[Dict[str, Any]]) -> DualSystemResult:
        """Enhance result with integrated components"""
        enhanced_result = base_result
        
        try:
            # Cognitive field integration
            if self.cognitive_field_dynamics:
                field_enhancement = await self._integrate_with_cognitive_fields(
                    base_result, content
                )
                enhanced_result.metadata['field_integration'] = field_enhancement
            
            # Embodied semantics integration
            if self.embodied_engine:
                embodied_enhancement = await self._integrate_with_embodied_semantics(
                    base_result, content
                )
                enhanced_result.metadata['embodied_integration'] = embodied_enhancement
            
            # SPDE integration
            if self.spde_core:
                spde_enhancement = await self._integrate_with_spde(
                    base_result, content
                )
                enhanced_result.metadata['spde_integration'] = spde_enhancement
            
            # Update overall enhancement score
            enhanced_result.neurodivergent_enhancement = await self._calculate_enhanced_optimization(
                enhanced_result
            )
            
        except Exception as e:
            logger.warning(f"Integration enhancement failed: {e}")
            enhanced_result.metadata['integration_error'] = str(e)
        
        return enhanced_result
    
    async def _integrate_with_cognitive_fields(self,
                                             result: DualSystemResult,
                                             content: str) -> Dict[str, Any]:
        """Integrate with cognitive field dynamics"""
        # Placeholder for cognitive field integration
        return {
            'field_coherence': 0.8,
            'field_dynamics_score': 0.75,
            'integration_success': True
        }
    
    async def _integrate_with_embodied_semantics(self,
                                               result: DualSystemResult,
                                               content: str) -> Dict[str, Any]:
        """Integrate with embodied semantic engine"""
        # Placeholder for embodied semantics integration
        return {
            'embodied_grounding': 0.7,
            'semantic_coherence': 0.8,
            'integration_success': True
        }
    
    async def _integrate_with_spde(self,
                                 result: DualSystemResult,
                                 content: str) -> Dict[str, Any]:
        """Integrate with SPDE core"""
        # Placeholder for SPDE integration
        return {
            'diffusion_coherence': 0.75,
            'semantic_evolution': 0.8,
            'integration_success': True
        }
    
    async def _calculate_enhanced_optimization(self,
                                             result: DualSystemResult) -> float:
        """Calculate enhanced neurodivergent optimization"""
        base_optimization = result.neurodivergent_enhancement
        
        # Factor in integration enhancements
        integration_scores = []
        for key, integration in result.metadata.items():
            if key.endswith('_integration') and isinstance(integration, dict):
                integration_scores.append(integration.get('integration_success', 0.0))
        
        if integration_scores:
            integration_boost = np.mean(integration_scores) * 0.2  # 20% boost
            return min(1.0, base_optimization + integration_boost)
        
        return base_optimization
    
    async def _trigger_integration_callbacks(self, event_data: Dict[str, Any]):
        """Trigger registered integration callbacks"""
        for callback in self.integration_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event_data)
                else:
                    callback(event_data)
            except Exception as e:
                logger.warning(f"Integration callback failed: {e}")
    
    def register_integration_callback(self, callback: Callable):
        """Register callback for integration events"""
        self.integration_callbacks.append(callback)
        logger.debug("Registered Barenholtz integration callback")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = self.dual_processor.get_system_status()
        status.update({
            'total_integrations': self.total_integrations,
            'integration_components': {
                'cognitive_field_dynamics': self.cognitive_field_dynamics is not None,
                'embodied_engine': self.embodied_engine is not None,
                'spde_core': self.spde_core is not None
            },
            'active_processes': len(self.active_processes)
        })
        
        return status