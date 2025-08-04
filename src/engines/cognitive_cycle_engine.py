"""
Cognitive Cycle Engine - Iterative Cognitive Processing System
============================================================

Advanced cognitive processing engine that implements iterative cognitive cycles
for deep semantic understanding, insight generation, and cognitive state evolution.

Key Features:
- Iterative cognitive processing cycles
- Multi-stage cognitive state evolution
- Attention mechanism integration
- Memory consolidation and retrieval
- Cognitive load management
- Insight emergence tracking
- Adaptive cycle timing

Scientific Foundation:
- Cognitive Architecture Theory
- Attention and Working Memory Models
- Iterative Processing Frameworks
- Cognitive Load Theory
- Insight Generation Mechanisms
"""

import asyncio
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.engines.advanced_barenholtz_alignment_engine import (
    AdvancedBarenholtzAlignmentEngine,
    AlignmentMethod,
)

from ..config.settings import get_settings
from ..utils.config import get_api_settings

logger = logging.getLogger(__name__)


class CyclePhase(Enum):
    """Phases of a cognitive cycle"""

    PERCEPTION = "perception"
    ATTENTION = "attention"
    WORKING_MEMORY = "working_memory"
    PROCESSING = "processing"
    INTEGRATION = "integration"
    CONSOLIDATION = "consolidation"
    OUTPUT = "output"


class CognitiveState(Enum):
    """States of cognitive processing"""

    IDLE = "idle"
    ACTIVE = "active"
    FOCUSED = "focused"
    OVERLOADED = "overloaded"
    INSIGHT = "insight"
    CONSOLIDATING = "consolidating"


@dataclass
class CognitiveContent:
    """Content processed in cognitive cycles"""

    content_id: str
    data: torch.Tensor
    attention_weights: torch.Tensor
    semantic_embedding: torch.Tensor
    priority: float = 0.0
    processing_depth: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CycleMetrics:
    """Metrics for a cognitive cycle"""

    cycle_id: str
    phase_durations: Dict[CyclePhase, float]
    attention_entropy: float
    working_memory_load: float
    processing_efficiency: float
    insight_score: float
    total_duration: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CognitiveMemory:
    """Cognitive memory system"""

    short_term: deque = field(default_factory=lambda: deque(maxlen=50))
    long_term: List[CognitiveContent] = field(default_factory=list)
    episodic: List[CycleMetrics] = field(default_factory=list)
    semantic_network: Dict[str, torch.Tensor] = field(default_factory=dict)
    attention_history: deque = field(default_factory=lambda: deque(maxlen=100))


class AttentionMechanism:
    """
    Attention mechanism for cognitive processing

    Implements multi-head attention with cognitive load awareness
    """

    def __init__(
        self, embedding_dim: int = 768, num_heads: int = 8, device: str = "cpu"
    ):
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.device = torch.device(device)
        self.head_dim = embedding_dim // num_heads

        # Attention matrices
        self.query_proj = torch.nn.Linear(
            embedding_dim, embedding_dim, device=self.device
        )
        self.key_proj = torch.nn.Linear(
            embedding_dim, embedding_dim, device=self.device
        )
        self.value_proj = torch.nn.Linear(
            embedding_dim, embedding_dim, device=self.device
        )
        self.output_proj = torch.nn.Linear(
            embedding_dim, embedding_dim, device=self.device
        )

        # Cognitive load tracking
        self.load_history = deque(maxlen=100)
        self.attention_patterns = deque(maxlen=50)

    def compute_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute multi-head attention with cognitive load awareness

        Args:
            query: Query tensor [batch_size, seq_len, embedding_dim]
            key: Key tensor [batch_size, seq_len, embedding_dim]
            value: Value tensor [batch_size, seq_len, embedding_dim]
            mask: Optional attention mask

        Returns:
            Tuple of (attended_output, attention_weights)
        """

        batch_size, seq_len, _ = query.shape

        # Project to query, key, value
        Q = self.query_proj(query)
        K = self.key_proj(key)
        V = self.value_proj(value)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Compute attention weights
        attention_weights = torch.softmax(scores, dim=-1)

        # Apply attention to values
        attended = torch.matmul(attention_weights, V)

        # Reshape and project output
        attended = (
            attended.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.embedding_dim)
        )
        output = self.output_proj(attended)

        # Compute cognitive load
        load = self._compute_cognitive_load(attention_weights)
        self.load_history.append(load)

        # Store attention pattern
        avg_attention = attention_weights.mean(dim=1).mean(
            dim=0
        )  # Average across heads and batch
        self.attention_patterns.append(avg_attention.detach().cpu())

        return output, attention_weights

    def _compute_cognitive_load(self, attention_weights: torch.Tensor) -> float:
        """Compute cognitive load based on attention patterns"""
        # Entropy-based load measure
        entropy = -torch.sum(
            attention_weights * torch.log(attention_weights + 1e-8), dim=-1
        )
        load = entropy.mean().item()
        return load

    def get_attention_entropy(self) -> float:
        """Get current attention entropy"""
        if not self.load_history:
            return 0.0
        return float(np.mean(self.load_history))

    def reset_attention_state(self):
        """Reset attention mechanism state"""
        self.load_history.clear()
        self.attention_patterns.clear()


class WorkingMemorySystem:
    """
    Working memory system for cognitive processing

    Manages active cognitive content with capacity limits
    """

    def __init__(self, capacity: int = 7, decay_rate: float = 0.1):
        self.capacity = capacity
        self.decay_rate = decay_rate
        self.contents: List[CognitiveContent] = []
        self.access_times: Dict[str, datetime] = {}
        self.load_threshold = 0.8

    def add_content(self, content: CognitiveContent) -> bool:
        """
        Add content to working memory

        Args:
            content: Content to add

        Returns:
            True if successfully added, False if capacity exceeded
        """

        # Check capacity
        if len(self.contents) >= self.capacity:
            # Remove least recently used content
            self._evict_lru()

        # Add new content
        self.contents.append(content)
        self.access_times[content.content_id] = datetime.now()

        logger.debug(f"Added content {content.content_id} to working memory")
        return True

    def retrieve_content(self, content_id: str) -> Optional[CognitiveContent]:
        """Retrieve content from working memory"""
        for content in self.contents:
            if content.content_id == content_id:
                self.access_times[content_id] = datetime.now()
                return content
        return None

    def get_active_contents(self) -> List[CognitiveContent]:
        """Get all active contents in working memory"""
        self._apply_decay()
        return self.contents.copy()

    def get_memory_load(self) -> float:
        """Get current working memory load"""
        return len(self.contents) / self.capacity

    def is_overloaded(self) -> bool:
        """Check if working memory is overloaded"""
        return self.get_memory_load() > self.load_threshold

    def _evict_lru(self):
        """Evict least recently used content"""
        if not self.contents:
            return

        # Find least recently used
        lru_content = min(
            self.contents,
            key=lambda c: self.access_times.get(c.content_id, datetime.min),
        )

        # Remove from working memory
        self.contents.remove(lru_content)
        if lru_content.content_id in self.access_times:
            del self.access_times[lru_content.content_id]

        logger.debug(f"Evicted content {lru_content.content_id} from working memory")

    def _apply_decay(self):
        """Apply decay to working memory contents"""
        current_time = datetime.now()
        to_remove = []

        for content in self.contents:
            access_time = self.access_times.get(content.content_id, current_time)
            age = (current_time - access_time).total_seconds()

            # Apply decay based on age
            decay_factor = np.exp(-self.decay_rate * age)
            content.priority *= decay_factor

            # Remove if priority too low
            if content.priority < 0.1:
                to_remove.append(content)

        # Remove decayed content
        for content in to_remove:
            self.contents.remove(content)
            if content.content_id in self.access_times:
                del self.access_times[content.content_id]

    def clear_memory(self):
        """Clear working memory"""
        self.contents.clear()
        self.access_times.clear()


class CognitiveCycleEngine:
    """
    Main Cognitive Cycle Engine

    Orchestrates iterative cognitive processing cycles with attention,
    working memory, and insight generation capabilities.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        num_attention_heads: int = 8,
        working_memory_capacity: int = 7,
        device: str = "cpu",
    ):

        self.settings = get_api_settings()

        logger.debug(f"   Environment: {self.settings.environment}")
        self.embedding_dim = embedding_dim
        self.device = torch.device(device)

        # Core components
        self.attention_mechanism = AttentionMechanism(
            embedding_dim=embedding_dim, num_heads=num_attention_heads, device=device
        )
        self.working_memory = WorkingMemorySystem(capacity=working_memory_capacity)
        self.cognitive_memory = CognitiveMemory()
        self.alignment_engine = AdvancedBarenholtzAlignmentEngine()

        # State management
        self.current_state = CognitiveState.IDLE
        self.current_cycle_id = None
        self.cycle_count = 0
        self.total_processing_time = 0.0

        # Cycle configuration
        self.cycle_phases = list(CyclePhase)
        self.phase_processors = {
            CyclePhase.PERCEPTION: self._process_perception,
            CyclePhase.ATTENTION: self._process_attention,
            CyclePhase.WORKING_MEMORY: self._process_working_memory,
            CyclePhase.PROCESSING: self._process_cognitive_processing,
            CyclePhase.INTEGRATION: self._process_integration,
            CyclePhase.CONSOLIDATION: self._process_consolidation,
            CyclePhase.OUTPUT: self._process_output,
        }

        # Metrics tracking
        self.cycle_metrics_history = deque(maxlen=1000)
        self.insight_events = []

        # Threading
        self.processing_lock = threading.Lock()

        logger.info(f"Cognitive Cycle Engine initialized on device: {device}")

    async def process_cognitive_cycle(
        self, input_data: torch.Tensor, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a complete cognitive cycle (now async for system-level integration)
        """
        with self.processing_lock:
            cycle_start_time = time.time()
            cycle_id = f"cycle_{self.cycle_count:06d}_{int(cycle_start_time)}"
            self.current_cycle_id = cycle_id
            self.cycle_count += 1

            # Initialize cycle metrics
            phase_durations = {}
            cycle_context = context or {}

            # Create initial cognitive content
            initial_content = CognitiveContent(
                content_id=f"{cycle_id}_input",
                data=input_data,
                attention_weights=torch.ones(input_data.shape[0], device=self.device),
                semantic_embedding=input_data,
                priority=1.0,
                metadata=cycle_context,
            )

            # Execute cycle phases
            current_content = initial_content
            phase_outputs = {}

            for phase in self.cycle_phases:
                phase_start = time.time()

                try:
                    # Process phase
                    phase_processor = self.phase_processors[phase]
                    if asyncio.iscoroutinefunction(phase_processor):
                        phase_result = await phase_processor(
                            current_content, cycle_context
                        )
                    else:
                        phase_result = phase_processor(current_content, cycle_context)

                    # Update content for next phase
                    if isinstance(phase_result, CognitiveContent):
                        current_content = phase_result
                    elif isinstance(phase_result, dict) and "content" in phase_result:
                        current_content = phase_result["content"]
                        phase_outputs[phase.value] = phase_result

                    # Update state based on phase
                    self._update_cognitive_state(phase, current_content)

                except Exception as e:
                    logger.error(f"Error in phase {phase.value}: {e}")
                    phase_outputs[phase.value] = {"error": str(e)}

                # Record phase duration
                phase_duration = time.time() - phase_start
                phase_durations[phase] = phase_duration

                logger.debug(f"Phase {phase.value} completed in {phase_duration:.3f}s")

            # Compute cycle metrics
            total_duration = time.time() - cycle_start_time
            self.total_processing_time += total_duration

            cycle_metrics = CycleMetrics(
                cycle_id=cycle_id,
                phase_durations=phase_durations,
                attention_entropy=self.attention_mechanism.get_attention_entropy(),
                working_memory_load=self.working_memory.get_memory_load(),
                processing_efficiency=self._compute_processing_efficiency(
                    phase_durations
                ),
                insight_score=self._compute_insight_score(current_content),
                total_duration=total_duration,
            )

            # Store metrics
            self.cycle_metrics_history.append(cycle_metrics)
            self.cognitive_memory.episodic.append(cycle_metrics)

            # Check for insights
            if cycle_metrics.insight_score > 0.7:
                self._record_insight_event(cycle_id, current_content, cycle_metrics)

            # Return cycle results
            return {
                "cycle_id": cycle_id,
                "final_content": current_content,
                "phase_outputs": phase_outputs,
                "metrics": cycle_metrics,
                "cognitive_state": self.current_state.value,
                "insights_generated": cycle_metrics.insight_score > 0.7,
            }

    async def _process_perception(
        self, content: CognitiveContent, context: Dict[str, Any]
    ) -> CognitiveContent:
        """Process perception phase"""
        # Simulate perceptual processing
        processed_data = content.data

        # Add perceptual noise and filtering
        if processed_data.dim() > 1:
            noise = torch.randn_like(processed_data) * 0.01
            processed_data = processed_data + noise

        # Update content
        content.data = processed_data
        content.processing_depth += 1
        content.metadata["perception_processed"] = True

        return content

    async def _process_attention(
        self, content: CognitiveContent, context: Dict[str, Any]
    ) -> CognitiveContent:
        """Process attention phase"""
        # Apply attention mechanism
        if content.data.dim() < 3:
            # Reshape for attention if needed
            data = (
                content.data.unsqueeze(0)
                if content.data.dim() == 2
                else content.data.unsqueeze(0).unsqueeze(0)
            )
        else:
            data = content.data

        # Compute attention
        attended_output, attention_weights = self.attention_mechanism.compute_attention(
            query=data, key=data, value=data
        )

        # Update content
        content.data = (
            attended_output.squeeze(0)
            if attended_output.shape[0] == 1
            else attended_output
        )
        content.attention_weights = attention_weights.mean(dim=1).squeeze(
            0
        )  # Average across heads
        content.processing_depth += 1
        content.metadata["attention_processed"] = True

        return content

    async def _process_working_memory(
        self, content: CognitiveContent, context: Dict[str, Any]
    ) -> CognitiveContent:
        """Process working memory phase"""
        # Add to working memory
        self.working_memory.add_content(content)

        # Retrieve related content
        active_contents = self.working_memory.get_active_contents()

        # Integrate with active contents if available
        if len(active_contents) > 1:
            # Simple integration - average embeddings
            all_embeddings = torch.stack(
                [c.semantic_embedding for c in active_contents]
            )
            integrated_embedding = all_embeddings.mean(dim=0)
            content.semantic_embedding = integrated_embedding

        content.processing_depth += 1
        content.metadata["working_memory_processed"] = True
        content.metadata["working_memory_load"] = self.working_memory.get_memory_load()

        return content

    async def _process_cognitive_processing(
        self, content: CognitiveContent, context: Dict[str, Any]
    ) -> CognitiveContent:
        """Process cognitive processing phase"""
        # Deep cognitive processing
        processed_embedding = content.semantic_embedding

        # Apply nonlinear transformations
        if processed_embedding.dim() > 0:
            # Cognitive transformation
            processed_embedding = torch.tanh(processed_embedding)
            processed_embedding = processed_embedding + 0.1 * torch.randn_like(
                processed_embedding
            )

        content.semantic_embedding = processed_embedding
        content.processing_depth += 1
        content.metadata["cognitive_processed"] = True

        return content

    async def _process_integration(
        self, content: CognitiveContent, context: Dict[str, Any]
    ) -> CognitiveContent:
        """Process integration phase using advanced alignment engine."""
        semantic_network = self.cognitive_memory.semantic_network
        if semantic_network:
            similarities = {}
            for concept_id, concept_embedding in semantic_network.items():
                if concept_embedding.shape == content.semantic_embedding.shape:
                    # Use advanced alignment engine
                    result = await self.alignment_engine.align_embeddings(
                        content.semantic_embedding.flatten(),
                        concept_embedding.flatten(),
                        method=AlignmentMethod.ENSEMBLE_ALIGNMENT,
                    )
                    similarity = result.alignment_score
                    similarities[concept_id] = similarity
                    logger.info(
                        f"[Kimera] Alignment method: {AlignmentMethod.ENSEMBLE_ALIGNMENT.value}, Score: {similarity:.4f}, Device: {'GPU' if torch.cuda.is_available() else 'CPU'}"
                    )
            content.metadata["similar_concepts"] = similarities
        content.processing_depth += 1
        content.metadata["integration_processed"] = True
        return content

    async def _process_consolidation(
        self, content: CognitiveContent, context: Dict[str, Any]
    ) -> CognitiveContent:
        """Process consolidation phase"""
        # Consolidate to long-term memory if important
        if content.priority > 0.5:
            self.cognitive_memory.long_term.append(content)

            # Update semantic network
            concept_id = f"concept_{len(self.cognitive_memory.semantic_network)}"
            self.cognitive_memory.semantic_network[concept_id] = (
                content.semantic_embedding.clone()
            )

        content.processing_depth += 1
        content.metadata["consolidation_processed"] = True

        return content

    async def _process_output(
        self, content: CognitiveContent, context: Dict[str, Any]
    ) -> CognitiveContent:
        """Process output phase"""
        # Final output processing
        content.metadata["output_processed"] = True
        content.metadata["final_priority"] = content.priority
        content.metadata["total_processing_depth"] = content.processing_depth

        return content

    def _update_cognitive_state(self, phase: CyclePhase, content: CognitiveContent):
        """Update cognitive state based on current phase and content"""
        memory_load = self.working_memory.get_memory_load()
        attention_entropy = self.attention_mechanism.get_attention_entropy()

        if memory_load > 0.8:
            self.current_state = CognitiveState.OVERLOADED
        elif attention_entropy > 2.0:
            self.current_state = CognitiveState.FOCUSED
        elif content.priority > 0.8:
            self.current_state = CognitiveState.ACTIVE
        elif phase == CyclePhase.CONSOLIDATION:
            self.current_state = CognitiveState.CONSOLIDATING
        else:
            self.current_state = CognitiveState.ACTIVE

    def _compute_processing_efficiency(
        self, phase_durations: Dict[CyclePhase, float]
    ) -> float:
        """Compute processing efficiency metric"""
        total_time = sum(phase_durations.values())
        if total_time == 0:
            return 1.0

        # Efficiency based on balanced phase timing
        expected_time_per_phase = total_time / len(phase_durations)
        variance = np.var(list(phase_durations.values()))
        efficiency = 1.0 / (1.0 + variance / expected_time_per_phase)

        return efficiency

    def _compute_insight_score(self, content: CognitiveContent) -> float:
        """Compute insight score based on content characteristics"""
        # Insight based on processing depth, priority, and novelty
        depth_score = min(content.processing_depth / 10.0, 1.0)
        priority_score = content.priority

        # Novelty score based on similarity to existing concepts
        novelty_score = 1.0
        if "similar_concepts" in content.metadata:
            similarities = content.metadata["similar_concepts"]
            if similarities:
                max_similarity = max(similarities.values())
                novelty_score = 1.0 - max_similarity

        insight_score = (depth_score + priority_score + novelty_score) / 3.0
        return insight_score

    def _record_insight_event(
        self, cycle_id: str, content: CognitiveContent, metrics: CycleMetrics
    ):
        """Record an insight event"""
        insight_event = {
            "cycle_id": cycle_id,
            "timestamp": datetime.now(),
            "insight_score": metrics.insight_score,
            "content_id": content.content_id,
            "processing_depth": content.processing_depth,
            "priority": content.priority,
            "metadata": content.metadata.copy(),
        }

        self.insight_events.append(insight_event)
        self.current_state = CognitiveState.INSIGHT

        logger.info(
            f"Insight generated in cycle {cycle_id} with score {metrics.insight_score:.3f}"
        )

    def get_engine_status(self) -> Dict[str, Any]:
        """Get current engine status and metrics"""
        return {
            "status": "operational",
            "device": str(self.device),
            "current_state": self.current_state.value,
            "cycle_count": self.cycle_count,
            "total_processing_time": self.total_processing_time,
            "average_cycle_time": self.total_processing_time / max(self.cycle_count, 1),
            "working_memory_load": self.working_memory.get_memory_load(),
            "attention_entropy": self.attention_mechanism.get_attention_entropy(),
            "long_term_memory_size": len(self.cognitive_memory.long_term),
            "semantic_network_size": len(self.cognitive_memory.semantic_network),
            "insight_events_count": len(self.insight_events),
            "recent_insights": len(
                [
                    e
                    for e in self.insight_events
                    if (datetime.now() - e["timestamp"]).total_seconds() < 3600
                ]
            ),
            "last_updated": datetime.now().isoformat(),
        }

    def get_cycle_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent cycle history"""
        recent_cycles = list(self.cycle_metrics_history)[-limit:]
        return [
            {
                "cycle_id": cycle.cycle_id,
                "total_duration": cycle.total_duration,
                "attention_entropy": cycle.attention_entropy,
                "working_memory_load": cycle.working_memory_load,
                "processing_efficiency": cycle.processing_efficiency,
                "insight_score": cycle.insight_score,
                "timestamp": cycle.timestamp.isoformat(),
            }
            for cycle in recent_cycles
        ]

    def get_insights_summary(self) -> Dict[str, Any]:
        """Get summary of insights generated"""
        if not self.insight_events:
            return {
                "total_insights": 0,
                "recent_insights": 0,
                "average_insight_score": 0.0,
                "top_insights": [],
            }

        # Recent insights (last hour)
        recent_insights = [
            e
            for e in self.insight_events
            if (datetime.now() - e["timestamp"]).total_seconds() < 3600
        ]

        # Top insights
        top_insights = sorted(
            self.insight_events, key=lambda x: x["insight_score"], reverse=True
        )[:10]

        return {
            "total_insights": len(self.insight_events),
            "recent_insights": len(recent_insights),
            "average_insight_score": np.mean(
                [e["insight_score"] for e in self.insight_events]
            ),
            "top_insights": [
                {
                    "cycle_id": insight["cycle_id"],
                    "insight_score": insight["insight_score"],
                    "timestamp": insight["timestamp"].isoformat(),
                }
                for insight in top_insights
            ],
        }

    def reset_engine(self):
        """Reset engine state"""
        self.working_memory.clear_memory()
        self.attention_mechanism.reset_attention_state()
        self.cognitive_memory = CognitiveMemory()
        self.cycle_metrics_history.clear()
        self.insight_events.clear()
        self.current_state = CognitiveState.IDLE
        self.cycle_count = 0
        self.total_processing_time = 0.0

        logger.info("Cognitive Cycle Engine reset")


# Factory function for easy instantiation
def create_cognitive_cycle_engine(
    embedding_dim: int = 768,
    num_attention_heads: int = 8,
    working_memory_capacity: int = 7,
    device: str = "cpu",
) -> CognitiveCycleEngine:
    """
    Create and initialize Cognitive Cycle Engine

    Args:
        embedding_dim: Dimension of embeddings
        num_attention_heads: Number of attention heads
        working_memory_capacity: Working memory capacity
        device: Computing device ("cpu" or "cuda")

    Returns:
        Initialized Cognitive Cycle Engine
    """
    return CognitiveCycleEngine(
        embedding_dim=embedding_dim,
        num_attention_heads=num_attention_heads,
        working_memory_capacity=working_memory_capacity,
        device=device,
    )
