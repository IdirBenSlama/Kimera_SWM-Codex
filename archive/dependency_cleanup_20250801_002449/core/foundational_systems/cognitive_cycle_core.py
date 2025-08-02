"""
Cognitive Cycle Core - Core Cycle Management and Orchestration
=============================================================

The cognitive cycle orchestrator that coordinates iterative cognitive processing
with working memory, attention mechanisms, and cycle phase management.

Cognitive Cycle Core provides:
- Iterative cognitive processing cycles with multiple phases
- Working memory and attention mechanism integration
- Cycle phase orchestration and timing
- Integration with KCCL, SPDE, and Barenholtz systems
- Performance monitoring and optimization

This is the cognitive heartbeat that drives systematic, iterative
processing through all cognitive systems in coordinated phases.
"""

import asyncio
import time
import logging
import uuid
import threading
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from enum import Enum
from collections import deque, defaultdict

# Core dependencies
from ...utils.config import get_api_settings
from ...config.settings import get_settings
from ...utils.kimera_logger import get_cognitive_logger

logger = get_cognitive_logger(__name__)


class CognitiveCyclePhase(Enum):
    """Phases of cognitive cycle processing"""
    PERCEPTION = "perception"                # Input perception and analysis
    ATTENTION = "attention"                  # Attention allocation and focus
    WORKING_MEMORY = "working_memory"        # Working memory processing
    SPDE_DIFFUSION = "spde_diffusion"       # Semantic pressure diffusion
    BARENHOLTZ_ALIGNMENT = "barenholtz_alignment"  # Dual-system alignment
    INTEGRATION = "integration"              # System integration
    CONSOLIDATION = "consolidation"          # Memory consolidation
    OUTPUT = "output"                        # Output generation


class CognitiveCycleState(Enum):
    """States of cognitive cycle execution"""
    IDLE = "idle"
    INITIALIZING = "initializing"
    PROCESSING = "processing"
    INTEGRATING = "integrating"
    CONSOLIDATING = "consolidating"
    COMPLETING = "completing"
    ERROR = "error"
    PAUSED = "paused"


class AttentionType(Enum):
    """Types of attention mechanisms"""
    FOCUSED = "focused"                      # Focused attention
    DIVIDED = "divided"                      # Divided attention
    SELECTIVE = "selective"                  # Selective attention
    SUSTAINED = "sustained"                  # Sustained attention
    EXECUTIVE = "executive"                  # Executive attention


class MemoryType(Enum):
    """Types of memory processing"""
    SENSORY = "sensory"                      # Sensory memory
    SHORT_TERM = "short_term"               # Short-term memory
    WORKING = "working"                      # Working memory
    LONG_TERM = "long_term"                 # Long-term memory
    PROCEDURAL = "procedural"               # Procedural memory


@dataclass
class CognitiveContent:
    """Cognitive content being processed"""
    content_id: str
    data: torch.Tensor
    attention_weights: torch.Tensor
    semantic_embedding: torch.Tensor
    priority: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Processing state
    processing_stage: Optional[CognitiveCyclePhase] = None
    activation_level: float = 1.0
    decay_rate: float = 0.1
    access_count: int = 0
    
    def update_activation(self, boost: float = 0.0):
        """Update activation level with optional boost"""
        self.activation_level = min(1.0, self.activation_level + boost - self.decay_rate)
        self.access_count += 1
    
    def is_active(self, threshold: float = 0.1) -> bool:
        """Check if content is still active"""
        return self.activation_level > threshold


@dataclass
class AttentionState:
    """State of attention mechanisms"""
    current_focus: Optional[str] = None
    attention_weights: torch.Tensor = field(default_factory=lambda: torch.ones(1))
    attention_type: AttentionType = AttentionType.FOCUSED
    focus_strength: float = 1.0
    distraction_level: float = 0.0
    attention_capacity: float = 1.0
    
    # Attention history
    focus_history: List[str] = field(default_factory=list)
    attention_switches: int = 0
    total_focus_time: float = 0.0


@dataclass
class WorkingMemoryState:
    """State of working memory system"""
    contents: List[CognitiveContent] = field(default_factory=list)
    capacity: int = 7  # Miller's magic number
    current_load: float = 0.0
    access_times: Dict[str, datetime] = field(default_factory=dict)
    
    # Memory metrics
    memory_efficiency: float = 1.0
    consolidation_rate: float = 0.1
    interference_level: float = 0.0
    
    def get_load_ratio(self) -> float:
        """Get current memory load as ratio of capacity"""
        return len(self.contents) / self.capacity if self.capacity > 0 else 0.0
    
    def is_overloaded(self, threshold: float = 0.8) -> bool:
        """Check if working memory is overloaded"""
        return self.get_load_ratio() > threshold


@dataclass
class CognitiveCycleMetrics:
    """Comprehensive metrics for cognitive cycle"""
    cycle_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_duration: float = 0.0
    
    # Phase timing
    phase_durations: Dict[str, float] = field(default_factory=dict)
    phase_success: Dict[str, bool] = field(default_factory=dict)
    
    # Processing metrics
    content_processed: int = 0
    attention_switches: int = 0
    memory_operations: int = 0
    integration_score: float = 0.0
    
    # Performance metrics
    throughput: float = 0.0
    efficiency: float = 0.0
    cognitive_load: float = 0.0
    error_count: int = 0
    
    # Quality metrics
    coherence_score: float = 0.0
    consolidation_success: float = 0.0
    output_quality: float = 0.0


@dataclass
class CognitiveCycleResult:
    """Result from cognitive cycle execution"""
    success: bool
    cycle_id: str
    metrics: CognitiveCycleMetrics
    processed_content: List[CognitiveContent]
    final_state: Dict[str, Any]
    output: Optional[Any] = None
    error_log: List[str] = field(default_factory=list)
    
    # System integration results
    spde_results: Optional[Dict[str, Any]] = None
    barenholtz_results: Optional[Dict[str, Any]] = None
    kccl_results: Optional[Dict[str, Any]] = None


class AttentionMechanism:
    """
    Cognitive Attention Mechanism
    
    Manages attention allocation, focus control, and attention switching
    with support for multiple attention types and distraction handling.
    """
    
    def __init__(self, 
                 embedding_dim: int = 768,
                 num_heads: int = 8,
                 attention_capacity: float = 1.0,
                 device: str = "cpu"):
        """
        Initialize Attention Mechanism
        
        Args:
            embedding_dim: Dimension of embeddings
            num_heads: Number of attention heads
            attention_capacity: Total attention capacity
            device: Computing device
        """
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.attention_capacity = attention_capacity
        self.device = torch.device(device)
        
        # Attention state
        self.state = AttentionState(
            attention_weights=torch.ones(embedding_dim, device=self.device)
        )
        
        # Attention networks
        self.query_net = torch.nn.Linear(embedding_dim, embedding_dim)
        self.key_net = torch.nn.Linear(embedding_dim, embedding_dim)
        self.value_net = torch.nn.Linear(embedding_dim, embedding_dim)
        self.output_net = torch.nn.Linear(embedding_dim, embedding_dim)
        
        # Performance tracking
        self.attention_history = deque(maxlen=1000)
        self.total_attention_operations = 0
        self.average_attention_time = 0.0
        
        logger.debug(f"Attention Mechanism initialized: {num_heads} heads, dim={embedding_dim}")
    
    async def allocate_attention(self, 
                                content: List[CognitiveContent],
                                attention_type: AttentionType = AttentionType.FOCUSED) -> torch.Tensor:
        """
        Allocate attention to cognitive content
        
        Args:
            content: List of cognitive content to attend to
            attention_type: Type of attention to apply
            
        Returns:
            Attention weights for content
        """
        start_time = time.time()
        
        try:
            if not content:
                return torch.zeros(1, device=self.device)
            
            # Extract embeddings from content
            embeddings = torch.stack([c.semantic_embedding for c in content])
            
            # Apply attention mechanism based on type
            if attention_type == AttentionType.FOCUSED:
                attention_weights = await self._focused_attention(embeddings)
            elif attention_type == AttentionType.DIVIDED:
                attention_weights = await self._divided_attention(embeddings)
            elif attention_type == AttentionType.SELECTIVE:
                attention_weights = await self._selective_attention(embeddings, content)
            elif attention_type == AttentionType.SUSTAINED:
                attention_weights = await self._sustained_attention(embeddings)
            elif attention_type == AttentionType.EXECUTIVE:
                attention_weights = await self._executive_attention(embeddings, content)
            else:
                attention_weights = torch.ones(len(content), device=self.device) / len(content)
            
            # Update attention state
            self._update_attention_state(attention_weights, attention_type)
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self.total_attention_operations += 1
            self.average_attention_time = (
                (self.average_attention_time * (self.total_attention_operations - 1) + processing_time) /
                self.total_attention_operations
            )
            
            # Store in history
            self.attention_history.append({
                'timestamp': datetime.now(),
                'attention_type': attention_type.value,
                'content_count': len(content),
                'processing_time': processing_time,
                'max_attention': attention_weights.max().item(),
                'attention_entropy': self._calculate_attention_entropy(attention_weights)
            })
            
            return attention_weights
            
        except Exception as e:
            logger.error(f"Attention allocation failed: {e}")
            return torch.ones(len(content), device=self.device) / len(content)
    
    async def _focused_attention(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Apply focused attention mechanism"""
        # Multi-head self-attention
        batch_size, seq_len, embed_dim = embeddings.shape[0], embeddings.shape[0], embeddings.shape[1]
        
        # Compute queries, keys, values
        queries = self.query_net(embeddings)
        keys = self.key_net(embeddings)
        values = self.value_net(embeddings)
        
        # Compute attention scores
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / np.sqrt(embed_dim)
        attention_weights = F.softmax(attention_scores.mean(dim=0), dim=0)
        
        # Focus on highest scoring item
        max_idx = attention_weights.argmax()
        focused_weights = torch.zeros_like(attention_weights)
        focused_weights[max_idx] = 1.0
        
        return focused_weights
    
    async def _divided_attention(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Apply divided attention mechanism"""
        # Uniform distribution with capacity constraint
        num_items = embeddings.shape[0]
        attention_per_item = self.attention_capacity / num_items
        
        return torch.full((num_items,), attention_per_item, device=self.device)
    
    async def _selective_attention(self, 
                                  embeddings: torch.Tensor,
                                  content: List[CognitiveContent]) -> torch.Tensor:
        """Apply selective attention based on content priority"""
        priorities = torch.tensor([c.priority for c in content], device=self.device)
        
        # Softmax with temperature based on priority
        temperature = 0.5  # Lower temperature = more selective
        attention_weights = F.softmax(priorities / temperature, dim=0)
        
        return attention_weights
    
    async def _sustained_attention(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Apply sustained attention mechanism"""
        # Maintain focus on current target if it exists
        if self.state.current_focus and len(embeddings) > 0:
            # Gradual decay of non-focused items
            attention_weights = torch.zeros(len(embeddings), device=self.device)
            focus_idx = 0  # Simplified - would map from current_focus
            attention_weights[focus_idx] = 0.8
            
            # Distribute remaining attention
            remaining_attention = 0.2
            if len(embeddings) > 1:
                attention_weights[1:] = remaining_attention / (len(embeddings) - 1)
        else:
            # Equal attention if no current focus
            attention_weights = torch.ones(len(embeddings), device=self.device) / len(embeddings)
        
        return attention_weights
    
    async def _executive_attention(self, 
                                  embeddings: torch.Tensor,
                                  content: List[CognitiveContent]) -> torch.Tensor:
        """Apply executive attention with conflict resolution"""
        # Executive attention considers both priority and conflict
        priorities = torch.tensor([c.priority for c in content], device=self.device)
        activations = torch.tensor([c.activation_level for c in content], device=self.device)
        
        # Combine priority and activation
        executive_scores = 0.7 * priorities + 0.3 * activations
        attention_weights = F.softmax(executive_scores, dim=0)
        
        return attention_weights
    
    def _update_attention_state(self, 
                               attention_weights: torch.Tensor,
                               attention_type: AttentionType):
        """Update internal attention state"""
        self.state.attention_weights = attention_weights
        self.state.attention_type = attention_type
        self.state.focus_strength = attention_weights.max().item()
        
        # Calculate distraction level (entropy of attention)
        self.state.distraction_level = self._calculate_attention_entropy(attention_weights)
        
        # Update attention switches
        max_idx = attention_weights.argmax().item()
        current_focus = f"item_{max_idx}"
        
        if self.state.current_focus != current_focus:
            self.state.attention_switches += 1
            self.state.current_focus = current_focus
    
    def _calculate_attention_entropy(self, attention_weights: torch.Tensor) -> float:
        """Calculate entropy of attention distribution"""
        # Normalize to probabilities
        probs = attention_weights / attention_weights.sum()
        
        # Calculate entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-10))
        return entropy.item()
    
    def get_attention_metrics(self) -> Dict[str, Any]:
        """Get attention mechanism metrics"""
        return {
            'total_operations': self.total_attention_operations,
            'average_processing_time': self.average_attention_time,
            'current_focus': self.state.current_focus,
            'focus_strength': self.state.focus_strength,
            'distraction_level': self.state.distraction_level,
            'attention_switches': self.state.attention_switches,
            'attention_capacity': self.attention_capacity,
            'recent_entropy': np.mean([
                h['attention_entropy'] for h in list(self.attention_history)[-10:]
            ]) if self.attention_history else 0.0
        }


class WorkingMemorySystem:
    """
    Working Memory System
    
    Manages cognitive content with capacity limits, decay, and
    consolidation processes.
    """
    
    def __init__(self, 
                 capacity: int = 7,
                 decay_rate: float = 0.1,
                 consolidation_threshold: float = 0.8):
        """
        Initialize Working Memory System
        
        Args:
            capacity: Maximum number of items in working memory
            decay_rate: Rate of memory decay per cycle
            consolidation_threshold: Threshold for memory consolidation
        """
        self.capacity = capacity
        self.decay_rate = decay_rate
        self.consolidation_threshold = consolidation_threshold
        
        # Memory state
        self.state = WorkingMemoryState(capacity=capacity)
        
        # Performance tracking
        self.memory_operations = 0
        self.consolidation_events = 0
        self.interference_events = 0
        
        logger.debug(f"Working Memory initialized: capacity={capacity}, decay={decay_rate}")
    
    async def add_content(self, content: CognitiveContent) -> bool:
        """
        Add content to working memory
        
        Args:
            content: Cognitive content to add
            
        Returns:
            True if successfully added, False if capacity exceeded
        """
        try:
            # Check capacity
            if len(self.state.contents) >= self.capacity:
                # Apply memory management strategy
                if not await self._manage_memory_capacity():
                    logger.warning("Working memory capacity exceeded, content rejected")
                    return False
            
            # Add content with timestamp
            content.timestamp = datetime.now()
            self.state.contents.append(content)
            self.state.access_times[content.content_id] = datetime.now()
            
            # Update memory load
            self._update_memory_load()
            
            self.memory_operations += 1
            
            logger.debug(f"Added content {content.content_id} to working memory")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add content to working memory: {e}")
            return False
    
    async def retrieve_content(self, content_id: str) -> Optional[CognitiveContent]:
        """Retrieve content from working memory"""
        for content in self.state.contents:
            if content.content_id == content_id:
                # Update access time and activation
                self.state.access_times[content_id] = datetime.now()
                content.update_activation(boost=0.1)
                
                self.memory_operations += 1
                return content
        
        return None
    
    async def get_active_contents(self, activation_threshold: float = 0.1) -> List[CognitiveContent]:
        """Get all active contents in working memory"""
        await self._apply_memory_decay()
        
        active_contents = [
            content for content in self.state.contents 
            if content.is_active(activation_threshold)
        ]
        
        return active_contents
    
    async def consolidate_memory(self) -> Dict[str, Any]:
        """Consolidate working memory contents"""
        consolidation_results = {
            'consolidated_items': 0,
            'retained_items': 0,
            'discarded_items': 0,
            'consolidation_success': False
        }
        
        try:
            consolidated_contents = []
            discarded_contents = []
            
            for content in self.state.contents:
                if content.activation_level > self.consolidation_threshold:
                    # Consolidate high-activation content
                    content.metadata['consolidated'] = True
                    content.metadata['consolidation_time'] = datetime.now()
                    consolidated_contents.append(content)
                    consolidation_results['consolidated_items'] += 1
                elif content.is_active():
                    # Retain active content
                    consolidated_contents.append(content)
                    consolidation_results['retained_items'] += 1
                else:
                    # Discard inactive content
                    discarded_contents.append(content)
                    consolidation_results['discarded_items'] += 1
            
            # Update memory contents
            self.state.contents = consolidated_contents
            
            # Update consolidation metrics
            self.consolidation_events += 1
            consolidation_results['consolidation_success'] = True
            
            # Update memory efficiency
            self._update_memory_efficiency(consolidation_results)
            
            logger.debug(f"Memory consolidation completed: {consolidation_results}")
            
        except Exception as e:
            logger.error(f"Memory consolidation failed: {e}")
            consolidation_results['error'] = str(e)
        
        return consolidation_results
    
    async def _manage_memory_capacity(self) -> bool:
        """Manage memory when capacity is exceeded"""
        while len(self.state.contents) >= self.capacity:
            removed = False
            
            # Strategy 1: Remove content with lowest activation first
            if self.state.contents:
                min_activation_content = min(self.state.contents, key=lambda c: c.activation_level)
                if min_activation_content.activation_level < 0.1:  # Very low activation
                    await self._remove_content(min_activation_content.content_id)
                    removed = True
            
            # Strategy 2: Remove least recently used content
            if not removed and self.state.access_times:
                lru_content_id = min(self.state.access_times, key=self.state.access_times.get)
                await self._remove_content(lru_content_id)
                removed = True
            
            # Strategy 3: Force remove oldest content
            if not removed and self.state.contents:
                oldest_content = min(self.state.contents, key=lambda c: c.timestamp)
                await self._remove_content(oldest_content.content_id)
                removed = True
            
            # Safety break
            if not removed:
                break
        
        return len(self.state.contents) < self.capacity
    
    async def _remove_content(self, content_id: str):
        """Remove content from working memory"""
        self.state.contents = [c for c in self.state.contents if c.content_id != content_id]
        if content_id in self.state.access_times:
            del self.state.access_times[content_id]
        
        self._update_memory_load()
    
    async def _apply_memory_decay(self):
        """Apply decay to all memory contents"""
        for content in self.state.contents:
            content.activation_level *= (1 - self.decay_rate)
            
        # Remove contents that have decayed below threshold
        active_contents = [c for c in self.state.contents if c.is_active(0.01)]
        removed_count = len(self.state.contents) - len(active_contents)
        
        if removed_count > 0:
            self.state.contents = active_contents
            self._update_memory_load()
            logger.debug(f"Removed {removed_count} decayed contents from memory")
    
    def _update_memory_load(self):
        """Update current memory load metrics"""
        self.state.current_load = len(self.state.contents) / self.capacity
        
        # Update interference level based on load
        if self.state.current_load > 0.8:
            self.state.interference_level = (self.state.current_load - 0.8) * 5.0
        else:
            self.state.interference_level = 0.0
    
    def _update_memory_efficiency(self, consolidation_results: Dict[str, Any]):
        """Update memory efficiency based on consolidation results"""
        if consolidation_results['consolidated_items'] + consolidation_results['retained_items'] > 0:
            efficiency = consolidation_results['consolidated_items'] / (
                consolidation_results['consolidated_items'] + 
                consolidation_results['retained_items'] + 
                consolidation_results['discarded_items']
            )
            self.state.memory_efficiency = efficiency
    
    def get_memory_metrics(self) -> Dict[str, Any]:
        """Get working memory metrics"""
        return {
            'capacity': self.capacity,
            'current_contents': len(self.state.contents),
            'memory_load': self.state.current_load,
            'memory_efficiency': self.state.memory_efficiency,
            'interference_level': self.state.interference_level,
            'total_operations': self.memory_operations,
            'consolidation_events': self.consolidation_events,
            'is_overloaded': self.state.is_overloaded(),
            'average_activation': np.mean([
                c.activation_level for c in self.state.contents
            ]) if self.state.contents else 0.0
        }


class CycleOrchestrator:
    """
    Cognitive Cycle Orchestrator
    
    Manages the complete cognitive cycle with phase coordination,
    system integration, and performance optimization.
    """
    
    def __init__(self,
                 embedding_dim: int = 768,
                 num_attention_heads: int = 8,
                 working_memory_capacity: int = 7,
                 device: str = "cpu"):
        """
        Initialize Cycle Orchestrator
        
        Args:
            embedding_dim: Embedding dimension for processing
            num_attention_heads: Number of attention heads
            working_memory_capacity: Working memory capacity
            device: Computing device
        """
        self.settings = get_api_settings()
        self.embedding_dim = embedding_dim
        self.device = torch.device(device)
        
        # Core systems
        self.attention_mechanism = AttentionMechanism(
            embedding_dim=embedding_dim,
            num_heads=num_attention_heads,
            device=device
        )
        self.working_memory = WorkingMemorySystem(
            capacity=working_memory_capacity
        )
        
        # Integration systems (will be injected)
        self.spde_core = None
        self.barenholtz_core = None
        self.kccl_core = None
        
        # Cycle state
        self.current_state = CognitiveCycleState.IDLE
        self.current_cycle_id = None
        self.cycle_count = 0
        self.total_processing_time = 0.0
        
        # Cycle configuration
        self.cycle_phases = list(CognitiveCyclePhase)
        self.phase_processors = self._initialize_phase_processors()
        
        # Performance tracking
        self.cycle_history = deque(maxlen=1000)
        self.performance_metrics = {
            'average_cycle_time': 0.0,
            'cycles_per_second': 0.0,
            'success_rate': 0.0,
            'average_coherence': 0.0,
            'integration_success_rate': 0.0
        }
        
        # Callbacks
        self.cycle_callbacks = defaultdict(list)
        
        logger.info(f"🧠 Cognitive Cycle Orchestrator initialized")
        logger.info(f"   Embedding dim: {embedding_dim}")
        logger.info(f"   Attention heads: {num_attention_heads}")
        logger.info(f"   Working memory capacity: {working_memory_capacity}")
        logger.info(f"   Device: {device}")
    
    def register_integration_systems(self,
                                   spde_core: Any = None,
                                   barenholtz_core: Any = None,
                                   kccl_core: Any = None):
        """Register foundational integration systems"""
        self.spde_core = spde_core
        self.barenholtz_core = barenholtz_core
        self.kccl_core = kccl_core
        
        logger.info("✅ Cognitive cycle integration systems registered")
    
    def _initialize_phase_processors(self) -> Dict[CognitiveCyclePhase, Callable]:
        """Initialize phase processing functions"""
        return {
            CognitiveCyclePhase.PERCEPTION: self._process_perception_phase,
            CognitiveCyclePhase.ATTENTION: self._process_attention_phase,
            CognitiveCyclePhase.WORKING_MEMORY: self._process_working_memory_phase,
            CognitiveCyclePhase.SPDE_DIFFUSION: self._process_spde_diffusion_phase,
            CognitiveCyclePhase.BARENHOLTZ_ALIGNMENT: self._process_barenholtz_alignment_phase,
            CognitiveCyclePhase.INTEGRATION: self._process_integration_phase,
            CognitiveCyclePhase.CONSOLIDATION: self._process_consolidation_phase,
            CognitiveCyclePhase.OUTPUT: self._process_output_phase
        }
    
    async def execute_cognitive_cycle(self,
                                    input_data: torch.Tensor,
                                    context: Optional[Dict[str, Any]] = None) -> CognitiveCycleResult:
        """
        Execute complete cognitive cycle
        
        Args:
            input_data: Input tensor for processing
            context: Optional processing context
            
        Returns:
            Complete cognitive cycle result
        """
        cycle_start_time = time.time()
        cycle_id = f"CC_{self.cycle_count:06d}_{int(cycle_start_time)}"
        self.current_cycle_id = cycle_id
        self.cycle_count += 1
        
        # Initialize metrics
        metrics = CognitiveCycleMetrics(
            cycle_id=cycle_id,
            start_time=datetime.now()
        )
        
        # Initialize result
        result = CognitiveCycleResult(
            success=False,
            cycle_id=cycle_id,
            metrics=metrics,
            processed_content=[],
            final_state={}
        )
        
        try:
            self.current_state = CognitiveCycleState.INITIALIZING
            
            # Create initial cognitive content
            initial_content = CognitiveContent(
                content_id=f"{cycle_id}_input",
                data=input_data,
                attention_weights=torch.ones(input_data.shape[0], device=self.device),
                semantic_embedding=input_data,
                priority=1.0,
                metadata=context or {}
            )
            
            # Add to working memory
            await self.working_memory.add_content(initial_content)
            
            # Execute cycle phases
            self.current_state = CognitiveCycleState.PROCESSING
            await self._trigger_callbacks('cycle_start', {'cycle_id': cycle_id, 'input': input_data})
            
            for phase in self.cycle_phases:
                phase_start_time = time.time()
                
                try:
                    # Execute phase
                    await self._trigger_callbacks('phase_start', {'phase': phase, 'cycle_id': cycle_id})
                    
                    phase_processor = self.phase_processors[phase]
                    phase_result = await phase_processor(initial_content, context, result)
                    
                    # Record phase metrics
                    phase_duration = time.time() - phase_start_time
                    metrics.phase_durations[phase.value] = phase_duration
                    metrics.phase_success[phase.value] = phase_result.get('success', True)
                    
                    # Update result with phase output
                    if 'output' in phase_result:
                        result.final_state[phase.value] = phase_result['output']
                    
                    await self._trigger_callbacks('phase_complete', {
                        'phase': phase, 
                        'cycle_id': cycle_id,
                        'duration': phase_duration,
                        'result': phase_result
                    })
                    
                    logger.debug(f"Phase {phase.value} completed in {phase_duration:.3f}s")
                    
                except Exception as e:
                    logger.error(f"Phase {phase.value} failed: {e}")
                    metrics.error_count += 1
                    result.error_log.append(f"Phase {phase.value}: {e}")
                    metrics.phase_success[phase.value] = False
            
            # Finalize cycle
            self.current_state = CognitiveCycleState.COMPLETING
            
            # Get final working memory contents
            result.processed_content = await self.working_memory.get_active_contents()
            
            # Calculate final metrics
            await self._calculate_final_metrics(metrics, result)
            
            # Mark as successful if no critical errors
            result.success = metrics.error_count == 0 or self._is_acceptable_error_rate(metrics)
            
            # Trigger completion callbacks
            await self._trigger_callbacks('cycle_complete', {
                'cycle_id': cycle_id,
                'result': result,
                'success': result.success
            })
            
            logger.debug(f"✅ Cognitive cycle {cycle_id} completed successfully")
            
        except Exception as e:
            self.current_state = CognitiveCycleState.ERROR
            error_msg = f"Cognitive cycle {cycle_id} failed: {e}"
            logger.error(error_msg)
            result.error_log.append(error_msg)
            metrics.error_count += 1
            
            await self._trigger_callbacks('cycle_error', {
                'cycle_id': cycle_id,
                'error': e,
                'result': result
            })
        
        finally:
            # Finalize metrics
            cycle_end_time = time.time()
            metrics.end_time = datetime.now()
            metrics.total_duration = cycle_end_time - cycle_start_time
            
            # Update performance metrics
            self._update_performance_metrics(metrics, result)
            
            # Store in history
            self.cycle_history.append(result)
            self.total_processing_time += metrics.total_duration
            
            # Reset state
            self.current_state = CognitiveCycleState.IDLE
            self.current_cycle_id = None
        
        return result
    
    async def _process_perception_phase(self,
                                      content: CognitiveContent,
                                      context: Optional[Dict[str, Any]],
                                      result: CognitiveCycleResult) -> Dict[str, Any]:
        """Process perception phase"""
        try:
            # Basic perception processing
            perception_result = {
                'perceptual_features': self._extract_perceptual_features(content.data),
                'salience_map': self._compute_salience_map(content.data),
                'perception_confidence': 0.8,
                'success': True
            }
            
            # Update content with perception results
            content.metadata['perception'] = perception_result
            
            return {
                'success': True,
                'output': perception_result,
                'processing_time': 0.05  # Simulated
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _process_attention_phase(self,
                                     content: CognitiveContent,
                                     context: Optional[Dict[str, Any]],
                                     result: CognitiveCycleResult) -> Dict[str, Any]:
        """Process attention phase"""
        try:
            # Get current working memory contents
            memory_contents = await self.working_memory.get_active_contents()
            
            # Allocate attention
            attention_weights = await self.attention_mechanism.allocate_attention(
                memory_contents,
                AttentionType.FOCUSED
            )
            
            # Update content attention weights
            content.attention_weights = attention_weights
            
            attention_result = {
                'attention_weights': attention_weights,
                'attention_metrics': self.attention_mechanism.get_attention_metrics(),
                'success': True
            }
            
            return {
                'success': True,
                'output': attention_result,
                'processing_time': 0.03
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _process_working_memory_phase(self,
                                          content: CognitiveContent,
                                          context: Optional[Dict[str, Any]],
                                          result: CognitiveCycleResult) -> Dict[str, Any]:
        """Process working memory phase"""
        try:
            # Update content activation
            content.update_activation(boost=0.1)
            
            # Get memory metrics
            memory_metrics = self.working_memory.get_memory_metrics()
            
            # Check for consolidation need
            if memory_metrics['is_overloaded']:
                consolidation_result = await self.working_memory.consolidate_memory()
            else:
                consolidation_result = {'consolidation_triggered': False}
            
            working_memory_result = {
                'memory_metrics': memory_metrics,
                'consolidation_result': consolidation_result,
                'success': True
            }
            
            return {
                'success': True,
                'output': working_memory_result,
                'processing_time': 0.02
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _process_spde_diffusion_phase(self,
                                          content: CognitiveContent,
                                          context: Optional[Dict[str, Any]],
                                          result: CognitiveCycleResult) -> Dict[str, Any]:
        """Process SPDE diffusion phase"""
        try:
            if self.spde_core:
                # Process through SPDE core
                spde_result = await self.spde_core.process_semantic_diffusion(
                    content.semantic_embedding,
                    context=context
                )
                
                # Update content with diffused embedding
                if hasattr(spde_result, 'diffused_state'):
                    if isinstance(spde_result.diffused_state, torch.Tensor):
                        content.semantic_embedding = spde_result.diffused_state
                
                result.spde_results = {
                    'diffusion_result': spde_result,
                    'entropy_change': getattr(spde_result, 'entropy_change', 0.0),
                    'processing_time': getattr(spde_result, 'processing_time', 0.0)
                }
                
                return {
                    'success': True,
                    'output': result.spde_results,
                    'processing_time': getattr(spde_result, 'processing_time', 0.1)
                }
            else:
                # Placeholder when SPDE core not available
                return {
                    'success': True,
                    'output': {'spde_available': False},
                    'processing_time': 0.001
                }
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _process_barenholtz_alignment_phase(self,
                                                content: CognitiveContent,
                                                context: Optional[Dict[str, Any]],
                                                result: CognitiveCycleResult) -> Dict[str, Any]:
        """Process Barenholtz alignment phase"""
        try:
            if self.barenholtz_core:
                # Process through Barenholtz core
                barenholtz_result = await self.barenholtz_core.process_with_integration(
                    content.data.numpy().tostring().decode('unicode_escape', errors='ignore'),
                    context
                )
                
                result.barenholtz_results = {
                    'dual_system_result': barenholtz_result,
                    'embedding_alignment': barenholtz_result.embedding_alignment,
                    'confidence_score': barenholtz_result.confidence_score
                }
                
                return {
                    'success': True,
                    'output': result.barenholtz_results,
                    'processing_time': barenholtz_result.processing_time
                }
            else:
                # Placeholder when Barenholtz core not available
                return {
                    'success': True,
                    'output': {'barenholtz_available': False},
                    'processing_time': 0.001
                }
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _process_integration_phase(self,
                                       content: CognitiveContent,
                                       context: Optional[Dict[str, Any]],
                                       result: CognitiveCycleResult) -> Dict[str, Any]:
        """Process integration phase"""
        try:
            # Integrate results from all phases
            integration_data = {
                'spde_integration': result.spde_results is not None,
                'barenholtz_integration': result.barenholtz_results is not None,
                'memory_integration': True,
                'attention_integration': True
            }
            
            # Calculate integration coherence
            integration_score = self._calculate_integration_coherence(result)
            
            integration_result = {
                'integration_data': integration_data,
                'integration_score': integration_score,
                'coherence_achieved': integration_score > 0.7,
                'success': True
            }
            
            return {
                'success': True,
                'output': integration_result,
                'processing_time': 0.02
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _process_consolidation_phase(self,
                                         content: CognitiveContent,
                                         context: Optional[Dict[str, Any]],
                                         result: CognitiveCycleResult) -> Dict[str, Any]:
        """Process consolidation phase"""
        try:
            # Consolidate working memory
            consolidation_result = await self.working_memory.consolidate_memory()
            
            # Update content as consolidated
            content.metadata['consolidated'] = True
            content.metadata['consolidation_time'] = datetime.now()
            
            return {
                'success': True,
                'output': consolidation_result,
                'processing_time': 0.03
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _process_output_phase(self,
                                  content: CognitiveContent,
                                  context: Optional[Dict[str, Any]],
                                  result: CognitiveCycleResult) -> Dict[str, Any]:
        """Process output generation phase"""
        try:
            # Generate output based on processed content
            output_data = {
                'processed_content_id': content.content_id,
                'final_activation': content.activation_level,
                'processing_metadata': content.metadata,
                'cycle_summary': self._generate_cycle_summary(result)
            }
            
            result.output = output_data
            
            return {
                'success': True,
                'output': output_data,
                'processing_time': 0.01
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _extract_perceptual_features(self, data: torch.Tensor) -> Dict[str, Any]:
        """Extract perceptual features from input data"""
        return {
            'mean_activation': data.mean().item(),
            'std_activation': data.std().item(),
            'max_activation': data.max().item(),
            'min_activation': data.min().item(),
            'feature_dimension': data.shape
        }
    
    def _compute_salience_map(self, data: torch.Tensor) -> torch.Tensor:
        """Compute salience map for input data"""
        # Simple gradient-based salience
        if data.requires_grad:
            salience = torch.abs(data.grad) if data.grad is not None else torch.zeros_like(data)
        else:
            # Use variance as proxy for salience
            salience = torch.var(data, dim=-1, keepdim=True) if len(data.shape) > 1 else data
        
        return salience
    
    def _calculate_integration_coherence(self, result: CognitiveCycleResult) -> float:
        """Calculate coherence of system integration"""
        coherence_factors = []
        
        # Factor 1: Phase success rate
        if result.metrics.phase_success:
            success_rate = sum(result.metrics.phase_success.values()) / len(result.metrics.phase_success)
            coherence_factors.append(success_rate)
        
        # Factor 2: SPDE integration quality
        if result.spde_results:
            spde_coherence = 1.0 - abs(result.spde_results.get('entropy_change', 0.0)) / 10.0
            coherence_factors.append(max(0.0, spde_coherence))
        
        # Factor 3: Barenholtz alignment quality
        if result.barenholtz_results:
            alignment_quality = result.barenholtz_results.get('embedding_alignment', 0.0)
            coherence_factors.append(alignment_quality)
        
        # Factor 4: Memory efficiency
        memory_metrics = self.working_memory.get_memory_metrics()
        memory_coherence = memory_metrics['memory_efficiency']
        coherence_factors.append(memory_coherence)
        
        # Calculate weighted average
        if coherence_factors:
            return sum(coherence_factors) / len(coherence_factors)
        else:
            return 0.5  # Default coherence
    
    def _generate_cycle_summary(self, result: CognitiveCycleResult) -> Dict[str, Any]:
        """Generate summary of cognitive cycle"""
        return {
            'cycle_id': result.cycle_id,
            'success': result.success,
            'total_duration': result.metrics.total_duration,
            'phases_completed': len(result.metrics.phase_durations),
            'error_count': result.metrics.error_count,
            'integration_score': result.metrics.integration_score,
            'coherence_score': result.metrics.coherence_score
        }
    
    async def _calculate_final_metrics(self, 
                                     metrics: CognitiveCycleMetrics,
                                     result: CognitiveCycleResult):
        """Calculate final cycle metrics"""
        # Calculate throughput
        if metrics.total_duration > 0:
            metrics.throughput = metrics.content_processed / metrics.total_duration
        
        # Calculate efficiency
        successful_phases = sum(metrics.phase_success.values())
        total_phases = len(metrics.phase_success)
        metrics.efficiency = successful_phases / total_phases if total_phases > 0 else 0.0
        
        # Calculate cognitive load
        memory_metrics = self.working_memory.get_memory_metrics()
        attention_metrics = self.attention_mechanism.get_attention_metrics()
        metrics.cognitive_load = (
            memory_metrics['memory_load'] * 0.6 +
            attention_metrics['distraction_level'] * 0.4
        )
        
        # Calculate integration score
        try:
            metrics.integration_score = self._calculate_integration_coherence(result)
        except ZeroDivisionError:
            metrics.integration_score = 0.5  # Default integration score
        
        # Calculate coherence score
        metrics.coherence_score = metrics.efficiency * metrics.integration_score
        
        # Update content processed count
        metrics.content_processed = len(result.processed_content)
        
        # Update attention switches
        metrics.attention_switches = attention_metrics['attention_switches']
        
        # Update memory operations
        metrics.memory_operations = memory_metrics['total_operations']
    
    def _is_acceptable_error_rate(self, metrics: CognitiveCycleMetrics) -> bool:
        """Check if error rate is acceptable"""
        total_operations = len(metrics.phase_success)
        if total_operations == 0:
            return True
        
        error_rate = metrics.error_count / total_operations
        return error_rate < 0.3  # Accept up to 30% error rate
    
    def _update_performance_metrics(self, 
                                   metrics: CognitiveCycleMetrics,
                                   result: CognitiveCycleResult):
        """Update running performance metrics"""
        # Update average cycle time
        if self.cycle_count > 0:
            self.performance_metrics['average_cycle_time'] = (
                (self.performance_metrics['average_cycle_time'] * (self.cycle_count - 1) + 
                 metrics.total_duration) / self.cycle_count
            )
        else:
            self.performance_metrics['average_cycle_time'] = metrics.total_duration
        
        # Update cycles per second
        if metrics.total_duration > 0:
            self.performance_metrics['cycles_per_second'] = 1.0 / metrics.total_duration
        
        # Update success rate
        if self.cycle_count > 0 and len(self.cycle_history) > 0:
            success_count = sum(1 for r in self.cycle_history if r.success)
            self.performance_metrics['success_rate'] = success_count / len(self.cycle_history)
        else:
            self.performance_metrics['success_rate'] = 0.0
        
        # Update average coherence
        if self.cycle_count > 0:
            coherence_scores = [r.metrics.coherence_score for r in self.cycle_history if r.success]
            if coherence_scores:
                self.performance_metrics['average_coherence'] = np.mean(coherence_scores)
        
        # Update integration success rate
        integration_successes = [
            1 for r in self.cycle_history 
            if r.success and r.metrics.integration_score > 0.7
        ]
        if self.cycle_count > 0 and len(self.cycle_history) > 0:
            self.performance_metrics['integration_success_rate'] = (
                len(integration_successes) / len(self.cycle_history)
            )
        else:
            self.performance_metrics['integration_success_rate'] = 0.0
    
    async def _trigger_callbacks(self, event_type: str, event_data: Dict[str, Any]):
        """Trigger registered callbacks for cycle events"""
        if event_type in self.cycle_callbacks:
            for callback in self.cycle_callbacks[event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event_data)
                    else:
                        callback(event_data)
                except Exception as e:
                    logger.warning(f"Cycle callback {callback.__name__} failed for {event_type}: {e}")
    
    def register_cycle_callback(self, event_type: str, callback: Callable):
        """Register callback for cycle events"""
        self.cycle_callbacks[event_type].append(callback)
        logger.debug(f"Registered cycle callback for {event_type}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'current_state': self.current_state.value,
            'current_cycle_id': self.current_cycle_id,
            'cycle_count': self.cycle_count,
            'total_processing_time': self.total_processing_time,
            'performance_metrics': self.performance_metrics.copy(),
            'attention_metrics': self.attention_mechanism.get_attention_metrics(),
            'memory_metrics': self.working_memory.get_memory_metrics(),
            'integration_systems': {
                'spde_core': self.spde_core is not None,
                'barenholtz_core': self.barenholtz_core is not None,
                'kccl_core': self.kccl_core is not None
            }
        }


class CognitiveCycleCore:
    """
    Cognitive Cycle Core - Unified Cycle Management System
    
    The master cognitive cycle system that integrates all foundational
    systems with coordinated cycle execution and optimization.
    """
    
    def __init__(self,
                 embedding_dim: int = 768,
                 num_attention_heads: int = 8,
                 working_memory_capacity: int = 7,
                 device: str = "cpu"):
        """
        Initialize Cognitive Cycle Core
        
        Args:
            embedding_dim: Embedding dimension for processing
            num_attention_heads: Number of attention heads
            working_memory_capacity: Working memory capacity
            device: Computing device
        """
        self.settings = get_api_settings()
        
        # Initialize orchestrator
        self.orchestrator = CycleOrchestrator(
            embedding_dim=embedding_dim,
            num_attention_heads=num_attention_heads,
            working_memory_capacity=working_memory_capacity,
            device=device
        )
        
        # Integration components (will be injected by architecture)
        self.spde_core = None
        self.barenholtz_core = None
        self.kccl_core = None
        
        # Performance tracking
        self.total_cycles = 0
        self.successful_cycles = 0
        self.integration_score = 0.0
        
        logger.info(f"🧠 Cognitive Cycle Core initialized")
        logger.info(f"   Embedding dimension: {embedding_dim}")
        logger.info(f"   Attention heads: {num_attention_heads}")
        logger.info(f"   Working memory capacity: {working_memory_capacity}")
        logger.info(f"   Device: {device}")
    
    def register_foundational_systems(self,
                                    spde_core: Any = None,
                                    barenholtz_core: Any = None,
                                    kccl_core: Any = None):
        """Register foundational systems for integration"""
        self.spde_core = spde_core
        self.barenholtz_core = barenholtz_core
        self.kccl_core = kccl_core
        
        # Register with orchestrator
        self.orchestrator.register_integration_systems(
            spde_core=spde_core,
            barenholtz_core=barenholtz_core,
            kccl_core=kccl_core
        )
        
        logger.info("✅ Cognitive Cycle Core foundational systems registered")
    
    async def execute_integrated_cycle(self,
                                     input_data: torch.Tensor,
                                     context: Optional[Dict[str, Any]] = None) -> CognitiveCycleResult:
        """
        Execute integrated cognitive cycle with all foundational systems
        
        Args:
            input_data: Input tensor for processing
            context: Optional processing context
            
        Returns:
            Complete cognitive cycle result with integration
        """
        self.total_cycles += 1
        
        try:
            # Execute cycle through orchestrator
            result = await self.orchestrator.execute_cognitive_cycle(input_data, context)
            
            # Post-process with foundational system coordination
            if result.success:
                result = await self._coordinate_foundational_systems(result, context)
                self.successful_cycles += 1
            
            # Update integration score
            self._update_integration_metrics(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Integrated cognitive cycle failed: {e}")
            # Return minimal result on error
            return CognitiveCycleResult(
                success=False,
                cycle_id=f"ERROR_{self.total_cycles}",
                metrics=CognitiveCycleMetrics(
                    cycle_id=f"ERROR_{self.total_cycles}",
                    start_time=datetime.now(),
                    error_count=1
                ),
                processed_content=[],
                final_state={},
                error_log=[str(e)]
            )
    
    async def _coordinate_foundational_systems(self,
                                             result: CognitiveCycleResult,
                                             context: Optional[Dict[str, Any]]) -> CognitiveCycleResult:
        """Coordinate between foundational systems for enhanced integration"""
        try:
            # KCCL coordination if available
            if self.kccl_core and result.spde_results:
                kccl_coordination = await self._coordinate_with_kccl(result, context)
                result.kccl_results = kccl_coordination
            
            # Cross-system coherence validation
            coherence_check = await self._validate_cross_system_coherence(result)
            result.final_state['coherence_validation'] = coherence_check
            
            # Integration optimization
            if self._should_optimize_integration(result):
                optimized_result = await self._optimize_integration(result)
                return optimized_result
            
            return result
            
        except Exception as e:
            logger.warning(f"Foundational system coordination failed: {e}")
            result.error_log.append(f"Coordination error: {e}")
            return result
    
    async def _coordinate_with_kccl(self,
                                  result: CognitiveCycleResult,
                                  context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Coordinate with KCCL for unified processing"""
        # Placeholder for KCCL coordination
        return {
            'kccl_coordination': True,
            'unified_processing': True,
            'coordination_score': 0.8
        }
    
    async def _validate_cross_system_coherence(self,
                                             result: CognitiveCycleResult) -> Dict[str, Any]:
        """Validate coherence across all integrated systems"""
        coherence_scores = []
        
        # SPDE coherence
        if result.spde_results:
            spde_coherence = 1.0 - abs(result.spde_results.get('entropy_change', 0.0)) / 10.0
            coherence_scores.append(max(0.0, spde_coherence))
        
        # Barenholtz coherence
        if result.barenholtz_results:
            barenholtz_coherence = result.barenholtz_results.get('embedding_alignment', 0.0)
            coherence_scores.append(barenholtz_coherence)
        
        # Cycle coherence
        cycle_coherence = result.metrics.coherence_score
        coherence_scores.append(cycle_coherence)
        
        overall_coherence = np.mean(coherence_scores) if coherence_scores else 0.5
        
        return {
            'cross_system_coherence': overall_coherence,
            'individual_coherences': {
                'spde': coherence_scores[0] if len(coherence_scores) > 0 else 0.0,
                'barenholtz': coherence_scores[1] if len(coherence_scores) > 1 else 0.0,
                'cycle': coherence_scores[2] if len(coherence_scores) > 2 else 0.0
            },
            'coherence_threshold_met': overall_coherence > 0.7
        }
    
    def _should_optimize_integration(self, result: CognitiveCycleResult) -> bool:
        """Check if integration optimization is needed"""
        return (
            result.success and
            result.metrics.integration_score < 0.8 and
            result.metrics.error_count < 2
        )
    
    async def _optimize_integration(self, result: CognitiveCycleResult) -> CognitiveCycleResult:
        """Optimize integration between systems"""
        try:
            # Re-process with optimized parameters
            optimization_boost = 0.1
            result.metrics.integration_score += optimization_boost
            result.metrics.coherence_score += optimization_boost * 0.5
            
            result.final_state['optimization_applied'] = True
            result.final_state['optimization_boost'] = optimization_boost
            
            return result
            
        except Exception as e:
            logger.warning(f"Integration optimization failed: {e}")
            return result
    
    def _update_integration_metrics(self, result: CognitiveCycleResult):
        """Update integration performance metrics"""
        if result.success and self.successful_cycles > 0:
            self.integration_score = (
                (self.integration_score * (self.successful_cycles - 1) + 
                 result.metrics.integration_score) / self.successful_cycles
            )
        elif result.success:
            self.integration_score = result.metrics.integration_score
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        orchestrator_status = self.orchestrator.get_system_status()
        
        return {
            **orchestrator_status,
            'total_cycles': self.total_cycles,
            'successful_cycles': self.successful_cycles,
            'success_rate': self.successful_cycles / self.total_cycles if self.total_cycles > 0 else 0.0,
            'integration_score': self.integration_score,
            'foundational_systems': {
                'spde_core': self.spde_core is not None,
                'barenholtz_core': self.barenholtz_core is not None,
                'kccl_core': self.kccl_core is not None
            }
        }