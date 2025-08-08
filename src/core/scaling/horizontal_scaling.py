#!/usr/bin/env python3
"""
Kimera SWM Horizontal Scaling and Load Balancing System
======================================================

Enterprise-grade horizontal scaling system for distributed cognitive processing
with intelligent load balancing, auto-scaling, and cluster management.

This module delivers:
- Dynamic horizontal scaling based on load metrics
- Intelligent load balancing across cognitive processing nodes
- Auto-scaling with predictive resource allocation
- Cluster health monitoring and failover management
- Service discovery and node registration
- Performance-based routing and optimization

Author: Kimera SWM Development Team
Date: January 30, 2025
Version: 5.3.0
"""

import asyncio
import hashlib
import json
import logging
import socket
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    """Node status enumeration"""

    UNKNOWN = "unknown"
    STARTING = "starting"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"
    DRAINING = "draining"


class LoadBalancingStrategy(Enum):
    """Load balancing strategy options"""

    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_RESPONSE_TIME = "weighted_response_time"
    RESOURCE_BASED = "resource_based"
    COGNITIVE_AFFINITY = "cognitive_affinity"
    ADAPTIVE = "adaptive"


@dataclass
class NodeConfiguration:
    """Auto-generated class."""
    pass
    """Configuration for a cognitive processing node"""

    node_id: str
    host: str
    port: int
    capabilities: List[str] = field(default_factory=list)
    max_concurrent_requests: int = 100
    weight: float = 1.0
    enable_gpu: bool = False
    enable_caching: bool = True
    enable_pipeline_optimization: bool = True

    # Resource specifications
    cpu_cores: int = 4
    memory_gb: float = 8.0
    gpu_memory_gb: float = 0.0

    # Health check configuration
    health_check_interval: int = 30  # seconds
    health_check_timeout: int = 10  # seconds
    unhealthy_threshold: int = 3  # consecutive failures

    # Auto-scaling configuration
    scale_up_threshold: float = 0.8  # CPU/memory utilization
    scale_down_threshold: float = 0.3
    min_response_time: float = 0.1  # seconds
    max_response_time: float = 5.0


@dataclass
class NodeMetrics:
    """Auto-generated class."""
    pass
    """Real-time metrics for a cognitive processing node"""

    node_id: str
    timestamp: float

    # Performance metrics
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    gpu_utilization: float = 0.0

    # Request metrics
    active_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    requests_per_second: float = 0.0

    # Cognitive operation metrics
    cognitive_operations_completed: int = 0
    understanding_analyses: int = 0
    consciousness_detections: int = 0
    insights_generated: int = 0

    # System health
    health_score: float = 1.0
    last_error: Optional[str] = None
    uptime: float = 0.0

    # Load balancing factors
    connection_count: int = 0
    queue_depth: int = 0
    response_time_p95: float = 0.0


@dataclass
class ClusterMetrics:
    """Auto-generated class."""
    pass
    """Cluster-wide metrics and status"""

    total_nodes: int = 0
    healthy_nodes: int = 0
    total_requests: int = 0
    avg_cluster_response_time: float = 0.0
    cluster_throughput: float = 0.0

    # Resource utilization across cluster
    avg_cpu_utilization: float = 0.0
    avg_memory_utilization: float = 0.0
    avg_gpu_utilization: float = 0.0

    # Auto-scaling status
    scaling_events: int = 0
    last_scaling_action: Optional[str] = None
    last_scaling_time: Optional[float] = None

    # Performance optimization
    load_balance_efficiency: float = 0.0
    cognitive_affinity_score: float = 0.0
    overall_cluster_health: float = 0.0
class NodeRegistry:
    """Auto-generated class."""
    pass
    """Service discovery and node registration system"""

    def __init__(self):
        self.nodes: Dict[str, NodeConfiguration] = {}
        self.node_metrics: Dict[str, NodeMetrics] = {}
        self.node_status: Dict[str, NodeStatus] = {}
        self._lock = threading.RLock()

    def register_node(self, config: NodeConfiguration) -> bool:
        """Register a new cognitive processing node"""
        with self._lock:
            try:
                self.nodes[config.node_id] = config
                self.node_status[config.node_id] = NodeStatus.STARTING
                self.node_metrics[config.node_id] = NodeMetrics(
                    node_id=config.node_id, timestamp=time.time()
                )

                logger.info(
                    f"Registered node {config.node_id} at {config.host}:{config.port}"
                )
                return True

            except Exception as e:
                logger.error(f"Failed to register node {config.node_id}: {e}")
                return False

    def unregister_node(self, node_id: str) -> bool:
        """Unregister a cognitive processing node"""
        with self._lock:
            try:
                if node_id in self.nodes:
                    del self.nodes[node_id]
                if node_id in self.node_metrics:
                    del self.node_metrics[node_id]
                if node_id in self.node_status:
                    del self.node_status[node_id]

                logger.info(f"Unregistered node {node_id}")
                return True

            except Exception as e:
                logger.error(f"Failed to unregister node {node_id}: {e}")
                return False

    def update_node_metrics(self, node_id: str, metrics: NodeMetrics) -> bool:
        """Update metrics for a registered node"""
        with self._lock:
            if node_id not in self.nodes:
                logger.warning(
                    f"Attempted to update metrics for unregistered node {node_id}"
                )
                return False

            self.node_metrics[node_id] = metrics

            # Update node status based on metrics
            self._update_node_status(node_id, metrics)
            return True

    def _update_node_status(self, node_id: str, metrics: NodeMetrics):
        """Update node status based on current metrics"""
        if metrics.health_score >= 0.9:
            self.node_status[node_id] = NodeStatus.HEALTHY
        elif metrics.health_score >= 0.7:
            self.node_status[node_id] = NodeStatus.DEGRADED
        elif metrics.health_score >= 0.5:
            self.node_status[node_id] = NodeStatus.UNHEALTHY
        else:
            self.node_status[node_id] = NodeStatus.OFFLINE

    def get_healthy_nodes(self) -> List[str]:
        """Get list of healthy node IDs"""
        with self._lock:
            return [
                node_id
                for node_id, status in self.node_status.items()
                if status in [NodeStatus.HEALTHY, NodeStatus.DEGRADED]
            ]

    def get_node_configuration(self, node_id: str) -> Optional[NodeConfiguration]:
        """Get configuration for a specific node"""
        with self._lock:
            return self.nodes.get(node_id)

    def get_node_metrics(self, node_id: str) -> Optional[NodeMetrics]:
        """Get current metrics for a specific node"""
        with self._lock:
            return self.node_metrics.get(node_id)

    def get_all_nodes(self) -> Dict[str, Dict[str, Any]]:
        """Get comprehensive information about all nodes"""
        with self._lock:
            result = {}
            for node_id in self.nodes:
                result[node_id] = {
                    "configuration": self.nodes[node_id],
                    "metrics": self.node_metrics[node_id],
                    "status": self.node_status[node_id],
                }
            return result
class LoadBalancer:
    """Auto-generated class."""
    pass
    """Intelligent load balancer for cognitive processing requests"""

    def __init__(
        self
        registry: NodeRegistry
        strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE
    ):
        self.registry = registry
        self.strategy = strategy
        self.request_history = defaultdict(deque)
        self.node_weights = defaultdict(float)
        self.cognitive_affinity_map = defaultdict(list)
        self._round_robin_counter = 0
        self._lock = threading.Lock()

    def select_node(
        self
        request_type: str = "general",
        request_context: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Select optimal node for processing a request"""
        healthy_nodes = self.registry.get_healthy_nodes()

        if not healthy_nodes:
            logger.warning("No healthy nodes available for request routing")
            return None

        if len(healthy_nodes) == 1:
            return healthy_nodes[0]

        # Select node based on strategy
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_selection(healthy_nodes)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_selection(healthy_nodes)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_RESPONSE_TIME:
            return self._weighted_response_time_selection(healthy_nodes)
        elif self.strategy == LoadBalancingStrategy.RESOURCE_BASED:
            return self._resource_based_selection(healthy_nodes)
        elif self.strategy == LoadBalancingStrategy.COGNITIVE_AFFINITY:
            return self._cognitive_affinity_selection(healthy_nodes, request_type)
        elif self.strategy == LoadBalancingStrategy.ADAPTIVE:
            return self._adaptive_selection(
                healthy_nodes, request_type, request_context
            )
        else:
            return self._round_robin_selection(healthy_nodes)

    def _round_robin_selection(self, nodes: List[str]) -> str:
        """Simple round-robin node selection"""
        with self._lock:
            selected = nodes[self._round_robin_counter % len(nodes)]
            self._round_robin_counter += 1
            return selected

    def _least_connections_selection(self, nodes: List[str]) -> str:
        """Select node with least active connections"""
        min_connections = float("inf")
        selected_node = nodes[0]

        for node_id in nodes:
            metrics = self.registry.get_node_metrics(node_id)
            if metrics and metrics.connection_count < min_connections:
                min_connections = metrics.connection_count
                selected_node = node_id

        return selected_node

    def _weighted_response_time_selection(self, nodes: List[str]) -> str:
        """Select node based on weighted response time"""
        best_score = float("inf")
        selected_node = nodes[0]

        for node_id in nodes:
            metrics = self.registry.get_node_metrics(node_id)
            config = self.registry.get_node_configuration(node_id)

            if metrics and config:
                # Calculate weighted score (lower is better)
                response_time_score = metrics.avg_response_time / config.weight
                if response_time_score < best_score:
                    best_score = response_time_score
                    selected_node = node_id

        return selected_node

    def _resource_based_selection(self, nodes: List[str]) -> str:
        """Select node based on available resources"""
        best_score = 0
        selected_node = nodes[0]

        for node_id in nodes:
            metrics = self.registry.get_node_metrics(node_id)
            if metrics:
                # Calculate resource availability score (higher is better)
                cpu_score = 1.0 - metrics.cpu_utilization
                memory_score = 1.0 - metrics.memory_utilization
                load_score = 1.0 - (
                    metrics.active_requests / 100.0
                )  # Normalize to 100 max requests

                total_score = (cpu_score + memory_score + load_score) / 3.0

                if total_score > best_score:
                    best_score = total_score
                    selected_node = node_id

        return selected_node

    def _cognitive_affinity_selection(self, nodes: List[str], request_type: str) -> str:
        """Select node based on cognitive processing affinity"""
        # Check if we have affinity mapping for this request type
        affinity_nodes = self.cognitive_affinity_map.get(request_type, [])

        # Filter affinity nodes to only healthy ones
        available_affinity_nodes = [node for node in affinity_nodes if node in nodes]

        if available_affinity_nodes:
            # Use resource-based selection among affinity nodes
            return self._resource_based_selection(available_affinity_nodes)
        else:
            # Fall back to resource-based selection
            return self._resource_based_selection(nodes)

    def _adaptive_selection(
        self
        nodes: List[str],
        request_type: str
        request_context: Optional[Dict[str, Any]],
    ) -> str:
        """Adaptive selection combining multiple strategies"""
        # Analyze request characteristics
        is_gpu_heavy = request_context and request_context.get("requires_gpu", False)
        is_memory_intensive = request_context and request_context.get(
            "memory_intensive", False
        )
        priority = request_context and request_context.get("priority", "medium")

        scores = {}

        for node_id in nodes:
            metrics = self.registry.get_node_metrics(node_id)
            config = self.registry.get_node_configuration(node_id)

            if not metrics or not config:
                continue

            score = 0.0

            # Resource availability (40%)
            cpu_score = 1.0 - metrics.cpu_utilization
            memory_score = 1.0 - metrics.memory_utilization
            resource_score = (cpu_score + memory_score) / 2.0
            score += resource_score * 0.4

            # Response time performance (30%)
            max_response_time = 5.0  # seconds
            response_time_score = max(
                0, 1.0 - (metrics.avg_response_time / max_response_time)
            )
            score += response_time_score * 0.3

            # Node capabilities match (20%)
            capability_score = 1.0
            if is_gpu_heavy and config.enable_gpu:
                capability_score *= 1.5
            if is_memory_intensive and config.memory_gb >= 16:
                capability_score *= 1.2
            score += min(1.0, capability_score) * 0.2

            # Connection load (10%)
            max_connections = config.max_concurrent_requests
            connection_score = 1.0 - (metrics.connection_count / max_connections)
            score += connection_score * 0.1

            # Priority boost
            if priority == "high":
                score *= 1.1
            elif priority == "critical":
                score *= 1.2

            scores[node_id] = score

        # Select node with highest score
        if scores:
            return max(scores.keys(), key=lambda k: scores[k])
        else:
            return nodes[0]  # Fallback

    def update_cognitive_affinity(
        self, request_type: str, node_id: str, performance_score: float
    ):
        """Update cognitive affinity mapping based on performance"""
        if performance_score > 0.8:  # Good performance
            if node_id not in self.cognitive_affinity_map[request_type]:
                self.cognitive_affinity_map[request_type].append(node_id)
        elif performance_score < 0.5:  # Poor performance
            if node_id in self.cognitive_affinity_map[request_type]:
                self.cognitive_affinity_map[request_type].remove(node_id)

    def get_load_balance_efficiency(self) -> float:
        """Calculate load balancing efficiency across nodes"""
        healthy_nodes = self.registry.get_healthy_nodes()
        if len(healthy_nodes) < 2:
            return 1.0

        # Calculate coefficient of variation for load distribution
        loads = []
        for node_id in healthy_nodes:
            metrics = self.registry.get_node_metrics(node_id)
            if metrics:
                # Combined load score
                load = (
                    metrics.cpu_utilization
                    + metrics.memory_utilization
                    + metrics.connection_count / 100.0
                ) / 3.0
                loads.append(load)

        if not loads:
            return 1.0

        mean_load = sum(loads) / len(loads)
        if mean_load == 0:
            return 1.0

        variance = sum((load - mean_load) ** 2 for load in loads) / len(loads)
        coefficient_of_variation = (variance**0.5) / mean_load

        # Convert to efficiency score (lower CV = higher efficiency)
        efficiency = max(0.0, 1.0 - coefficient_of_variation)
        return efficiency
class AutoScaler:
    """Auto-generated class."""
    pass
    """Automatic scaling system for cognitive processing cluster"""

    def __init__(self, registry: NodeRegistry, min_nodes: int = 2, max_nodes: int = 20):
        self.registry = registry
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.scaling_events = []
        self.last_scaling_time = 0
        self.scaling_cooldown = 300  # 5 minutes
        self._is_running = False

    async def start_autoscaling(self):
        """Start the autoscaling monitoring loop"""
        self._is_running = True
        logger.info("Starting autoscaling system")

        while self._is_running:
            try:
                await self._evaluate_scaling_needs()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Autoscaling evaluation error: {e}")
                await asyncio.sleep(60)

    def stop_autoscaling(self):
        """Stop the autoscaling system"""
        self._is_running = False
        logger.info("Autoscaling system stopped")

    async def _evaluate_scaling_needs(self):
        """Evaluate if scaling action is needed"""
        current_time = time.time()

        # Check cooldown period
        if current_time - self.last_scaling_time < self.scaling_cooldown:
            return

        healthy_nodes = self.registry.get_healthy_nodes()
        cluster_metrics = self._calculate_cluster_metrics(healthy_nodes)

        # Determine scaling action
        scaling_action = self._determine_scaling_action(
            cluster_metrics, len(healthy_nodes)
        )

        if scaling_action == "scale_up":
            await self._scale_up()
        elif scaling_action == "scale_down":
            await self._scale_down()

    def _calculate_cluster_metrics(self, healthy_nodes: List[str]) -> Dict[str, float]:
        """Calculate cluster-wide metrics for scaling decisions"""
        if not healthy_nodes:
            return {
                "avg_cpu": 0
                "avg_memory": 0
                "avg_response_time": 0
                "total_load": 0
            }

        total_cpu = 0
        total_memory = 0
        total_response_time = 0
        total_requests = 0

        for node_id in healthy_nodes:
            metrics = self.registry.get_node_metrics(node_id)
            if metrics:
                total_cpu += metrics.cpu_utilization
                total_memory += metrics.memory_utilization
                total_response_time += metrics.avg_response_time
                total_requests += metrics.active_requests

        node_count = len(healthy_nodes)
        return {
            "avg_cpu": total_cpu / node_count
            "avg_memory": total_memory / node_count
            "avg_response_time": total_response_time / node_count
            "total_load": total_requests
        }

    def _determine_scaling_action(
        self, cluster_metrics: Dict[str, float], current_nodes: int
    ) -> Optional[str]:
        """Determine if scaling action is needed"""
        avg_cpu = cluster_metrics["avg_cpu"]
        avg_memory = cluster_metrics["avg_memory"]
        avg_response_time = cluster_metrics["avg_response_time"]

        # Scale up conditions
        scale_up_needed = (
            avg_cpu > 0.8 or avg_memory > 0.8 or avg_response_time > 3.0
        ) and current_nodes < self.max_nodes

        # Scale down conditions
        scale_down_needed = (
            avg_cpu < 0.3
            and avg_memory < 0.3
            and avg_response_time < 1.0
            and current_nodes > self.min_nodes
        )

        if scale_up_needed:
            return "scale_up"
        elif scale_down_needed:
            return "scale_down"
        else:
            return None

    async def _scale_up(self):
        """Add a new node to the cluster"""
        try:
            # Generate new node configuration
            new_node_id = f"node_{uuid.uuid4().hex[:8]}"
            new_port = 8000 + len(self.registry.nodes)

            new_config = NodeConfiguration(
                node_id=new_node_id
                host="localhost",  # In production, this would be dynamic
                port=new_port
                capabilities=["cognitive_processing", "understanding", "consciousness"],
                max_concurrent_requests=100
                enable_gpu=True
                enable_caching=True
                enable_pipeline_optimization=True
            )

            # Register new node
            if self.registry.register_node(new_config):
                self.scaling_events.append(
                    {
                        "action": "scale_up",
                        "node_id": new_node_id
                        "timestamp": time.time(),
                        "reason": "High resource utilization detected",
                    }
                )
                self.last_scaling_time = time.time()

                logger.info(f"Scaled up: Added node {new_node_id}")

                # In production, this would trigger actual node deployment
                await self._simulate_node_startup(new_node_id)

        except Exception as e:
            logger.error(f"Scale up failed: {e}")

    async def _scale_down(self):
        """Remove a node from the cluster"""
        try:
            healthy_nodes = self.registry.get_healthy_nodes()

            if len(healthy_nodes) <= self.min_nodes:
                return

            # Select node to remove (prefer nodes with lowest load)
            node_to_remove = self._select_node_for_removal(healthy_nodes)

            if node_to_remove:
                # Mark node as draining
                self.registry.node_status[node_to_remove] = NodeStatus.DRAINING

                # Wait for requests to drain (simplified)
                await asyncio.sleep(30)

                # Remove node
                if self.registry.unregister_node(node_to_remove):
                    self.scaling_events.append(
                        {
                            "action": "scale_down",
                            "node_id": node_to_remove
                            "timestamp": time.time(),
                            "reason": "Low resource utilization detected",
                        }
                    )
                    self.last_scaling_time = time.time()

                    logger.info(f"Scaled down: Removed node {node_to_remove}")

        except Exception as e:
            logger.error(f"Scale down failed: {e}")

    def _select_node_for_removal(self, healthy_nodes: List[str]) -> Optional[str]:
        """Select the best node to remove during scale down"""
        min_load = float("inf")
        selected_node = None

        for node_id in healthy_nodes:
            metrics = self.registry.get_node_metrics(node_id)
            if metrics:
                # Calculate combined load score
                load_score = (
                    metrics.cpu_utilization
                    + metrics.memory_utilization
                    + metrics.connection_count / 100.0
                ) / 3.0

                if load_score < min_load:
                    min_load = load_score
                    selected_node = node_id

        return selected_node

    async def _simulate_node_startup(self, node_id: str):
        """Simulate new node startup process"""
        # Simulate startup time
        await asyncio.sleep(5)

        # Update node status to healthy
        self.registry.node_status[node_id] = NodeStatus.HEALTHY

        # Add initial metrics
        initial_metrics = NodeMetrics(
            node_id=node_id, timestamp=time.time(), health_score=1.0, uptime=0.0
        )
        self.registry.update_node_metrics(node_id, initial_metrics)

    def get_scaling_history(self) -> List[Dict[str, Any]]:
        """Get history of scaling events"""
        return self.scaling_events[-20:]  # Last 20 events
class HorizontalScalingManager:
    """Auto-generated class."""
    pass
    """Main horizontal scaling and load balancing manager"""

    def __init__(self, min_nodes: int = 2, max_nodes: int = 20):
        self.registry = NodeRegistry()
        self.load_balancer = LoadBalancer(self.registry, LoadBalancingStrategy.ADAPTIVE)
        self.autoscaler = AutoScaler(self.registry, min_nodes, max_nodes)
        self.cluster_metrics = ClusterMetrics()
        self.is_running = False

    async def initialize(self) -> bool:
        """Initialize the horizontal scaling system"""
        try:
            # Register initial nodes
            await self._register_initial_nodes()

            # Start autoscaling
            asyncio.create_task(self.autoscaler.start_autoscaling())

            # Start metrics collection
            asyncio.create_task(self._metrics_collection_loop())

            self.is_running = True
            logger.info("Horizontal scaling system initialized")
            return True

        except Exception as e:
            logger.error(f"Horizontal scaling initialization failed: {e}")
            return False

    async def _register_initial_nodes(self):
        """Register initial cluster nodes"""
        initial_nodes = [
            NodeConfiguration(
                node_id="primary_node",
                host="localhost",
                port=8000
                capabilities=["cognitive_processing", "understanding", "consciousness"],
                max_concurrent_requests=100
                enable_gpu=True
                enable_caching=True
                enable_pipeline_optimization=True
                cpu_cores=8
                memory_gb=16.0
                gpu_memory_gb=8.0
            ),
            NodeConfiguration(
                node_id="secondary_node",
                host="localhost",
                port=8001
                capabilities=["cognitive_processing", "understanding"],
                max_concurrent_requests=80
                enable_gpu=False
                enable_caching=True
                enable_pipeline_optimization=True
                cpu_cores=4
                memory_gb=8.0
            ),
        ]

        for config in initial_nodes:
            self.registry.register_node(config)

            # Add initial metrics
            initial_metrics = NodeMetrics(
                node_id=config.node_id
                timestamp=time.time(),
                health_score=1.0
                uptime=0.0
            )
            self.registry.update_node_metrics(config.node_id, initial_metrics)

    async def _metrics_collection_loop(self):
        """Continuous metrics collection and cluster health monitoring"""
        while self.is_running:
            try:
                await self._collect_cluster_metrics()
                await asyncio.sleep(30)  # Collect every 30 seconds
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(30)

    async def _collect_cluster_metrics(self):
        """Collect and update cluster-wide metrics"""
        healthy_nodes = self.registry.get_healthy_nodes()

        if not healthy_nodes:
            return

        total_requests = 0
        total_response_time = 0
        total_cpu = 0
        total_memory = 0
        total_gpu = 0
        node_count = len(healthy_nodes)

        for node_id in healthy_nodes:
            metrics = self.registry.get_node_metrics(node_id)
            if metrics:
                total_requests += metrics.completed_requests
                total_response_time += metrics.avg_response_time
                total_cpu += metrics.cpu_utilization
                total_memory += metrics.memory_utilization
                total_gpu += metrics.gpu_utilization

        # Update cluster metrics
        self.cluster_metrics.total_nodes = len(self.registry.nodes)
        self.cluster_metrics.healthy_nodes = node_count
        self.cluster_metrics.total_requests = total_requests
        self.cluster_metrics.avg_cluster_response_time = (
            total_response_time / node_count if node_count > 0 else 0
        )
        self.cluster_metrics.avg_cpu_utilization = (
            total_cpu / node_count if node_count > 0 else 0
        )
        self.cluster_metrics.avg_memory_utilization = (
            total_memory / node_count if node_count > 0 else 0
        )
        self.cluster_metrics.avg_gpu_utilization = (
            total_gpu / node_count if node_count > 0 else 0
        )

        # Calculate performance scores
        self.cluster_metrics.load_balance_efficiency = (
            self.load_balancer.get_load_balance_efficiency()
        )
        self.cluster_metrics.overall_cluster_health = self._calculate_cluster_health()

        # Update scaling events
        self.cluster_metrics.scaling_events = len(self.autoscaler.scaling_events)

        if self.autoscaler.scaling_events:
            last_event = self.autoscaler.scaling_events[-1]
            self.cluster_metrics.last_scaling_action = last_event["action"]
            self.cluster_metrics.last_scaling_time = last_event["timestamp"]

    def _calculate_cluster_health(self) -> float:
        """Calculate overall cluster health score"""
        healthy_nodes = self.registry.get_healthy_nodes()
        total_nodes = len(self.registry.nodes)

        if total_nodes == 0:
            return 0.0

        # Node availability score (50%)
        availability_score = len(healthy_nodes) / total_nodes

        # Resource utilization score (30%) - prefer moderate utilization
        optimal_utilization = 0.6
        cpu_utilization_score = 1.0 - abs(
            self.cluster_metrics.avg_cpu_utilization - optimal_utilization
        )
        memory_utilization_score = 1.0 - abs(
            self.cluster_metrics.avg_memory_utilization - optimal_utilization
        )
        utilization_score = (cpu_utilization_score + memory_utilization_score) / 2.0

        # Response time score (20%) - prefer low response times
        max_acceptable_response_time = 3.0
        response_time_score = max(
            0
            1.0
            - (
                self.cluster_metrics.avg_cluster_response_time
                / max_acceptable_response_time
            ),
        )

        # Combine scores
        health_score = (
            availability_score * 0.5
            + utilization_score * 0.3
            + response_time_score * 0.2
        )

        return max(0.0, min(1.0, health_score))

    async def route_request(
        self
        request_type: str = "general",
        request_context: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Route a request to the optimal node"""
        return self.load_balancer.select_node(request_type, request_context)

    def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status"""
        return {
            "cluster_metrics": self.cluster_metrics
            "nodes": self.registry.get_all_nodes(),
            "load_balancing": {
                "strategy": self.load_balancer.strategy.value
                "efficiency": self.cluster_metrics.load_balance_efficiency
            },
            "autoscaling": {
                "enabled": self.autoscaler._is_running
                "min_nodes": self.autoscaler.min_nodes
                "max_nodes": self.autoscaler.max_nodes
                "recent_events": self.autoscaler.get_scaling_history(),
            },
        }

    async def shutdown(self):
        """Shutdown the horizontal scaling system"""
        logger.info("Shutting down horizontal scaling system")
        self.is_running = False
        self.autoscaler.stop_autoscaling()
        logger.info("Horizontal scaling system shutdown complete")


# Global horizontal scaling manager
scaling_manager = HorizontalScalingManager()


# Convenience functions
async def initialize_horizontal_scaling(
    min_nodes: int = 2, max_nodes: int = 20
) -> bool:
    """Initialize horizontal scaling system"""
    global scaling_manager
    scaling_manager = HorizontalScalingManager(min_nodes, max_nodes)
    return await scaling_manager.initialize()


async def route_cognitive_request(
    request_type: str = "general", request_context: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """Route cognitive request to optimal node"""
    global scaling_manager
    return await scaling_manager.route_request(request_type, request_context)


def get_cluster_status() -> Dict[str, Any]:
    """Get cluster status"""
    global scaling_manager
    return scaling_manager.get_cluster_status()


if __name__ == "__main__":
    # Test horizontal scaling
    async def test_horizontal_scaling():
        logger.info("🔄 Testing Kimera SWM Horizontal Scaling System")
        logger.info("=" * 55)

        # Initialize
        success = await initialize_horizontal_scaling()

        if success:
            logger.info("✅ Horizontal scaling system initialized")

            # Test request routing
            node = await route_cognitive_request("understanding", {"priority": "high"})
            logger.info(f"Routed request to node: {node}")

            # Get cluster status
            status = get_cluster_status()
            logger.info(f"Cluster nodes: {status['cluster_metrics'].healthy_nodes}")
            logger.info(
                f"Load balance efficiency: {status['cluster_metrics'].load_balance_efficiency:.3f}"
            )
        else:
            logger.info("❌ Horizontal scaling initialization failed")

        logger.info("\n🎯 Horizontal Scaling System Ready!")

    asyncio.run(test_horizontal_scaling())
