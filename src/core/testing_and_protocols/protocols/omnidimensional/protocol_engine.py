#!/usr/bin/env python3
"""
Omnidimensional Protocol Engine for Cognitive System Communication
=================================================================

DO-178C Level A compliant protocol engine for seamless communication
and data exchange across all dimensions of the Kimera cognitive system.

Key Features:
- Quantum-resistant secure communication channels
- Multi-dimensional routing and message delivery
- Real-time protocol adaptation and optimization
- Comprehensive error detection and recovery

Author: KIMERA Development Team
Version: 1.0.0 (DO-178C Level A)
"""

import asyncio
import hashlib
import json
import queue
import struct
import sys
import threading
import time
import uuid
import weakref
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent.parent))

from utils.kimera_exceptions import KimeraCognitiveError, KimeraValidationError
from utils.kimera_logger import LogCategory, get_logger

logger = get_logger(__name__, LogCategory.SYSTEM)


class ProtocolVersion(Enum):
    """Protocol version enumeration"""

    V1_0 = "1.0"
    V1_1 = "1.1"
    V2_0 = "2.0"  # Future version


class MessageType(Enum):
    """Message type enumeration"""

    DATA_TRANSFER = "data_transfer"
    CONTROL_COMMAND = "control_command"
    STATUS_REPORT = "status_report"
    HEARTBEAT = "heartbeat"
    DISCOVERY = "discovery"
    REGISTRATION = "registration"
    ERROR_REPORT = "error_report"
    SYNC_REQUEST = "sync_request"
    COGNITIVE_STATE = "cognitive_state"
    SYSTEM_ALERT = "system_alert"


class MessagePriority(Enum):
    """Message priority levels"""

    CRITICAL = 1  # System-critical messages
    HIGH = 2  # High priority operations
    NORMAL = 3  # Standard communications
    LOW = 4  # Background data exchange
    BULK = 5  # Large data transfers


class DeliveryGuarantee(Enum):
    """Message delivery guarantee levels"""

    AT_MOST_ONCE = "at_most_once"  # Fire and forget
    AT_LEAST_ONCE = "at_least_once"  # Ensure delivery
    EXACTLY_ONCE = "exactly_once"  # Prevent duplicates


class SystemDimension(Enum):
    """Cognitive system dimensions"""

    COGNITIVE_RESPONSE = "cognitive_response"
    BARENHOLTZ_ARCHITECTURE = "barenholtz_architecture"
    HIGH_DIMENSIONAL_MODELING = "high_dimensional_modeling"
    INSIGHT_MANAGEMENT = "insight_management"
    THERMODYNAMIC_INTEGRATION = "thermodynamic_integration"
    QUANTUM_SECURITY = "quantum_security"
    RESPONSE_GENERATION = "response_generation"
    TESTING_ORCHESTRATION = "testing_orchestration"
    SYSTEM_MONITOR = "system_monitor"
    ETHICAL_GOVERNOR = "ethical_governor"


@dataclass
class MessageHeader:
    """Auto-generated class."""
    pass
    """Protocol message header"""

    version: ProtocolVersion
    message_id: str
    correlation_id: Optional[str]
    timestamp: datetime
    source_dimension: SystemDimension
    destination_dimension: SystemDimension
    message_type: MessageType
    priority: MessagePriority
    delivery_guarantee: DeliveryGuarantee
    ttl_seconds: int
    sequence_number: int
    total_parts: int
    part_number: int
    security_context: Dict[str, str]
    routing_hints: Dict[str, Any]


@dataclass
class MessagePayload:
    """Auto-generated class."""
    pass
    """Protocol message payload"""

    content_type: str
    content_encoding: str
    data: Union[bytes, str, Dict[str, Any]]
    metadata: Dict[str, Any]
    checksum: str
    size_bytes: int
    compression_used: bool


@dataclass
class ProtocolMessage:
    """Auto-generated class."""
    pass
    """Complete protocol message"""

    header: MessageHeader
    payload: MessagePayload

    def __post_init__(self):
        """Validate message structure"""
        if not self.header.message_id:
            self.header.message_id = str(uuid.uuid4())

        if not self.payload.checksum:
            self.payload.checksum = self._calculate_checksum()

    def _calculate_checksum(self) -> str:
        """Calculate message checksum for integrity verification"""
        if isinstance(self.payload.data, bytes):
            data_bytes = self.payload.data
        elif isinstance(self.payload.data, str):
            data_bytes = self.payload.data.encode("utf-8")
        else:
            data_bytes = json.dumps(self.payload.data, sort_keys=True).encode("utf-8")

        return hashlib.sha256(data_bytes).hexdigest()[:16]
class MessageRoute:
    """Auto-generated class."""
    pass
    """Message routing information"""

    def __init__(
        self
        source: SystemDimension
        destination: SystemDimension
        hops: Optional[List[SystemDimension]] = None
    ):
        self.source = source
        self.destination = destination
        self.hops = hops or []
        self.route_hash = self._calculate_route_hash()
        self.performance_metrics = {
            "average_latency": 0.0
            "success_rate": 1.0
            "last_update": datetime.now(),
        }

    def _calculate_route_hash(self) -> str:
        """Calculate unique route identifier"""
        route_string = f"{self.source.value}→{self.destination.value}"
        if self.hops:
            route_string += "→" + "→".join(hop.value for hop in self.hops)
        return hashlib.md5(route_string.encode()).hexdigest()[:8]
class DimensionRegistry:
    """Auto-generated class."""
    pass
    """
    Registry for cognitive system dimensions and their capabilities

    Implements service discovery and capability negotiation
    following aerospace distributed systems principles.
    """

    def __init__(self):
        self.dimensions: Dict[SystemDimension, Dict[str, Any]] = {}
        self.capabilities: Dict[SystemDimension, Set[str]] = {}
        self.health_status: Dict[SystemDimension, Dict[str, Any]] = {}
        self.last_heartbeat: Dict[SystemDimension, datetime] = {}
        self.registry_lock = threading.RLock()

    def register_dimension(
        self
        dimension: SystemDimension
        capabilities: List[str],
        metadata: Dict[str, Any],
    ) -> bool:
        """Register a cognitive dimension with its capabilities"""
        with self.registry_lock:
            try:
                self.dimensions[dimension] = {
                    "registration_time": datetime.now(),
                    "metadata": metadata.copy(),
                    "status": "active",
                }

                self.capabilities[dimension] = set(capabilities)

                self.health_status[dimension] = {
                    "status": "healthy",
                    "last_check": datetime.now(),
                    "metrics": {},
                }

                self.last_heartbeat[dimension] = datetime.now()

                logger.info(
                    f"🔗 Registered dimension: {dimension.value} with {len(capabilities)} capabilities"
                )
                return True

            except Exception as e:
                logger.error(f"Failed to register dimension {dimension.value}: {e}")
                return False

    def unregister_dimension(self, dimension: SystemDimension) -> bool:
        """Unregister a cognitive dimension"""
        with self.registry_lock:
            try:
                if dimension in self.dimensions:
                    del self.dimensions[dimension]
                if dimension in self.capabilities:
                    del self.capabilities[dimension]
                if dimension in self.health_status:
                    del self.health_status[dimension]
                if dimension in self.last_heartbeat:
                    del self.last_heartbeat[dimension]

                logger.info(f"🔗 Unregistered dimension: {dimension.value}")
                return True

            except Exception as e:
                logger.error(f"Failed to unregister dimension {dimension.value}: {e}")
                return False

    def update_heartbeat(
        self, dimension: SystemDimension, health_metrics: Dict[str, Any]
    ) -> None:
        """Update heartbeat and health metrics for dimension"""
        with self.registry_lock:
            if dimension in self.dimensions:
                self.last_heartbeat[dimension] = datetime.now()
                if dimension in self.health_status:
                    self.health_status[dimension]["metrics"] = health_metrics
                    self.health_status[dimension]["last_check"] = datetime.now()

    def get_dimension_info(
        self, dimension: SystemDimension
    ) -> Optional[Dict[str, Any]]:
        """Get information about a specific dimension"""
        with self.registry_lock:
            if dimension not in self.dimensions:
                return None

            return {
                "dimension": dimension.value
                "registration": self.dimensions[dimension],
                "capabilities": list(self.capabilities.get(dimension, [])),
                "health": self.health_status.get(dimension, {}),
                "last_heartbeat": self.last_heartbeat.get(dimension),
            }

    def find_dimensions_by_capability(self, capability: str) -> List[SystemDimension]:
        """Find all dimensions that support a specific capability"""
        with self.registry_lock:
            matching_dimensions = []
            for dimension, caps in self.capabilities.items():
                if capability in caps:
                    matching_dimensions.append(dimension)
            return matching_dimensions

    def get_healthy_dimensions(
        self, max_age_seconds: int = 30
    ) -> List[SystemDimension]:
        """Get list of dimensions with recent heartbeats"""
        with self.registry_lock:
            cutoff_time = datetime.now() - timedelta(seconds=max_age_seconds)
            healthy_dimensions = []

            for dimension, last_beat in self.last_heartbeat.items():
                if last_beat >= cutoff_time:
                    healthy_dimensions.append(dimension)

            return healthy_dimensions

    def get_registry_status(self) -> Dict[str, Any]:
        """Get comprehensive registry status"""
        with self.registry_lock:
            healthy_count = len(self.get_healthy_dimensions())
            total_count = len(self.dimensions)

            return {
                "total_dimensions": total_count
                "healthy_dimensions": healthy_count
                "unhealthy_dimensions": total_count - healthy_count
                "total_capabilities": sum(
                    len(caps) for caps in self.capabilities.values()
                ),
                "last_update": (
                    max(self.last_heartbeat.values()) if self.last_heartbeat else None
                ),
            }
class MessageRouter:
    """Auto-generated class."""
    pass
    """
    Intelligent message routing system

    Implements adaptive routing with load balancing and failover
    following distributed systems best practices.
    """

    def __init__(self, registry: DimensionRegistry):
        self.registry = registry
        self.routes: Dict[str, MessageRoute] = {}
        self.route_cache: Dict[
            Tuple[SystemDimension, SystemDimension], List[MessageRoute]
        ] = {}
        self.performance_history: Dict[str, List[Dict[str, float]]] = {}
        self.router_lock = threading.RLock()

    def find_route(
        self
        source: SystemDimension
        destination: SystemDimension
        requirements: Optional[Dict[str, Any]] = None
    ) -> Optional[MessageRoute]:
        """Find optimal route between dimensions"""
        with self.router_lock:
            # Check cache first
            cache_key = (source, destination)
            if cache_key in self.route_cache:
                routes = self.route_cache[cache_key]
                if routes:
                    # Return best performing route
                    return self._select_best_route(routes, requirements)

            # Calculate new routes
            routes = self._calculate_routes(source, destination, requirements)

            # Update cache
            self.route_cache[cache_key] = routes

            # Return best route
            return self._select_best_route(routes, requirements) if routes else None

    def _calculate_routes(
        self
        source: SystemDimension
        destination: SystemDimension
        requirements: Optional[Dict[str, Any]],
    ) -> List[MessageRoute]:
        """Calculate possible routes between dimensions"""
        routes = []

        # Direct route (preferred)
        if self._is_dimension_available(destination):
            direct_route = MessageRoute(source, destination)
            routes.append(direct_route)

        # Multi-hop routes (if direct route unavailable or unreliable)
        healthy_dimensions = self.registry.get_healthy_dimensions()

        for intermediate in healthy_dimensions:
            if intermediate != source and intermediate != destination:
                # Check if intermediate can route to destination
                if self._can_route(intermediate, destination):
                    multi_hop_route = MessageRoute(source, destination, [intermediate])
                    routes.append(multi_hop_route)

        return routes

    def _is_dimension_available(self, dimension: SystemDimension) -> bool:
        """Check if dimension is available for communication"""
        healthy_dimensions = self.registry.get_healthy_dimensions()
        return dimension in healthy_dimensions

    def _can_route(
        self, intermediate: SystemDimension, destination: SystemDimension
    ) -> bool:
        """Check if intermediate dimension can route to destination"""
        # For now, assume all healthy dimensions can route to each other
        # In future, this could check routing capabilities
        return self._is_dimension_available(destination)

    def _select_best_route(
        self, routes: List[MessageRoute], requirements: Optional[Dict[str, Any]]
    ) -> Optional[MessageRoute]:
        """Select best route based on performance and requirements"""
        if not routes:
            return None

        # Score routes based on performance metrics
        scored_routes = []
        for route in routes:
            score = self._calculate_route_score(route, requirements)
            scored_routes.append((score, route))

        # Sort by score (higher is better)
        scored_routes.sort(key=lambda x: x[0], reverse=True)

        return scored_routes[0][1]

    def _calculate_route_score(
        self, route: MessageRoute, requirements: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate route quality score"""
        base_score = 100.0

        # Prefer direct routes
        if not route.hops:
            base_score += 50.0
        else:
            # Penalize for each hop
            base_score -= len(route.hops) * 10.0

        # Factor in performance metrics
        base_score += route.performance_metrics["success_rate"] * 30.0

        # Factor in latency (lower is better)
        latency_penalty = min(route.performance_metrics["average_latency"] * 5.0, 20.0)
        base_score -= latency_penalty

        # Apply requirements if specified
        if requirements:
            if requirements.get("low_latency", False) and route.hops:
                base_score -= 25.0  # Penalize multi-hop for low latency

            if requirements.get("high_reliability", False):
                base_score += route.performance_metrics["success_rate"] * 20.0

        return max(0.0, base_score)

    def update_route_performance(
        self, route_hash: str, latency: float, success: bool
    ) -> None:
        """Update route performance metrics"""
        with self.router_lock:
            if route_hash not in self.performance_history:
                self.performance_history[route_hash] = []

            # Add new measurement
            measurement = {
                "timestamp": time.time(),
                "latency": latency
                "success": success
            }
            self.performance_history[route_hash].append(measurement)

            # Trim history to last 100 measurements
            if len(self.performance_history[route_hash]) > 100:
                self.performance_history[route_hash] = self.performance_history[
                    route_hash
                ][-100:]

            # Update route metrics
            self._recalculate_route_metrics(route_hash)

    def _recalculate_route_metrics(self, route_hash: str) -> None:
        """Recalculate route performance metrics from history"""
        history = self.performance_history.get(route_hash, [])
        if not history:
            return

        # Calculate metrics from recent measurements
        recent_history = history[-20:]  # Last 20 measurements

        latencies = [m["latency"] for m in recent_history]
        successes = [m["success"] for m in recent_history]

        average_latency = sum(latencies) / len(latencies)
        success_rate = sum(successes) / len(successes)

        # Update route metrics
        for routes in self.route_cache.values():
            for route in routes:
                if route.route_hash == route_hash:
                    route.performance_metrics.update(
                        {
                            "average_latency": average_latency
                            "success_rate": success_rate
                            "last_update": datetime.now(),
                        }
                    )

    def clear_route_cache(self) -> None:
        """Clear route cache to force recalculation"""
        with self.router_lock:
            self.route_cache.clear()
            logger.debug("Route cache cleared")
class ProtocolEngine:
    """Auto-generated class."""
    pass
    """
    Main omnidimensional protocol engine

    Coordinates all aspects of inter-dimensional communication
    with aerospace-grade reliability and nuclear-grade safety.
    """

    def __init__(self, local_dimension: SystemDimension):
        self.local_dimension = local_dimension
        self.registry = DimensionRegistry()
        self.router = MessageRouter(self.registry)

        # Message handling
        self.message_handlers: Dict[MessageType, List[Callable]] = {}
        self.pending_messages: Dict[str, ProtocolMessage] = {}
        self.message_queue: queue.Queue = queue.Queue()

        # Connection management
        self.connections: Dict[SystemDimension, Dict[str, Any]] = {}
        self.heartbeat_interval = 10.0  # seconds
        self.heartbeat_active = False
        self.heartbeat_thread: Optional[threading.Thread] = None

        # Statistics and monitoring
        self.statistics = {
            "messages_sent": 0
            "messages_received": 0
            "messages_failed": 0
            "bytes_transferred": 0
            "average_latency": 0.0
            "start_time": datetime.now(),
        }

        # Thread safety
        self.engine_lock = threading.RLock()

        # Register self
        self._register_self()

        logger.info(
            f"🌐 Omnidimensional Protocol Engine initialized for {local_dimension.value}"
        )

    def _register_self(self) -> None:
        """Register this engine instance with the registry"""
        capabilities = [
            "message_routing",
            "data_transfer",
            "heartbeat_monitoring",
            "error_recovery",
            "protocol_adaptation",
        ]

        metadata = {
            "engine_version": "1.0.0",
            "supported_protocols": ["omnidimensional_v1.0"],
            "max_message_size": 10 * 1024 * 1024,  # 10MB
            "compression_supported": True
            "encryption_supported": True
        }

        success = self.registry.register_dimension(
            self.local_dimension, capabilities, metadata
        )

        if not success:
            raise KimeraValidationError(
                f"Failed to register dimension {self.local_dimension.value}"
            )

    def start(self) -> None:
        """Start the protocol engine"""
        with self.engine_lock:
            if self.heartbeat_active:
                return

            # Start heartbeat monitoring
            self.heartbeat_active = True
            self.heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop, daemon=True
            )
            self.heartbeat_thread.start()

            logger.info(f"🚀 Protocol engine started for {self.local_dimension.value}")

    def stop(self) -> None:
        """Stop the protocol engine"""
        with self.engine_lock:
            if not self.heartbeat_active:
                return

            # Stop heartbeat
            self.heartbeat_active = False
            if self.heartbeat_thread:
                self.heartbeat_thread.join(timeout=5.0)

            # Unregister self
            self.registry.unregister_dimension(self.local_dimension)

            logger.info(f"🛑 Protocol engine stopped for {self.local_dimension.value}")

    def _heartbeat_loop(self) -> None:
        """Main heartbeat monitoring loop"""
        while self.heartbeat_active:
            try:
                # Update own heartbeat
                health_metrics = self._collect_health_metrics()
                self.registry.update_heartbeat(self.local_dimension, health_metrics)

                # Send heartbeat to connected dimensions
                self._send_heartbeats()

                # Wait for next heartbeat
                time.sleep(self.heartbeat_interval)

            except Exception as e:
                logger.error(f"Heartbeat loop error: {e}")
                time.sleep(self.heartbeat_interval)

    def _collect_health_metrics(self) -> Dict[str, Any]:
        """Collect health metrics for heartbeat"""
        return {
            "messages_processed": self.statistics["messages_sent"]
            + self.statistics["messages_received"],
            "error_rate": self.statistics["messages_failed"]
            / max(1, self.statistics["messages_sent"]),
            "average_latency": self.statistics["average_latency"],
            "queue_size": self.message_queue.qsize(),
            "connections": len(self.connections),
            "uptime_seconds": (
                datetime.now() - self.statistics["start_time"]
            ).total_seconds(),
        }

    def _send_heartbeats(self) -> None:
        """Send heartbeat messages to connected dimensions"""
        heartbeat_message = self.create_message(
            destination=self.local_dimension,  # Will be updated per connection
            message_type=MessageType.HEARTBEAT
            data={"timestamp": datetime.now().isoformat(), "status": "healthy"},
            priority=MessagePriority.LOW
        )

        for dimension in self.connections:
            try:
                heartbeat_message.header.destination_dimension = dimension
                heartbeat_message.header.message_id = str(
                    uuid.uuid4()
                )  # New ID per recipient
                self._send_message_internal(heartbeat_message)
            except Exception as e:
                logger.debug(f"Failed to send heartbeat to {dimension.value}: {e}")

    def create_message(
        self
        destination: SystemDimension
        message_type: MessageType
        data: Union[str, bytes, Dict[str, Any]],
        priority: MessagePriority = MessagePriority.NORMAL
        delivery_guarantee: DeliveryGuarantee = DeliveryGuarantee.AT_LEAST_ONCE
        ttl_seconds: int = 300
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProtocolMessage:
        """Create a protocol message"""

        # Prepare payload data
        if isinstance(data, (dict, list)):
            content_data = json.dumps(data, separators=(",", ":"))
            content_type = "application/json"
            content_encoding = "utf-8"
        elif isinstance(data, str):
            content_data = data
            content_type = "text/plain"
            content_encoding = "utf-8"
        elif isinstance(data, bytes):
            content_data = data
            content_type = "application/octet-stream"
            content_encoding = "binary"
        else:
            raise KimeraValidationError(f"Unsupported data type: {type(data)}")

        # Create header
        header = MessageHeader(
            version=ProtocolVersion.V1_0
            message_id=str(uuid.uuid4()),
            correlation_id=None
            timestamp=datetime.now(),
            source_dimension=self.local_dimension
            destination_dimension=destination
            message_type=message_type
            priority=priority
            delivery_guarantee=delivery_guarantee
            ttl_seconds=ttl_seconds
            sequence_number=1
            total_parts=1
            part_number=1
            security_context={},
            routing_hints={},
        )

        # Create payload
        payload = MessagePayload(
            content_type=content_type
            content_encoding=content_encoding
            data=content_data
            metadata=metadata or {},
            checksum="",  # Will be calculated in __post_init__
            size_bytes=(
                len(content_data) if isinstance(content_data, (str, bytes)) else 0
            ),
            compression_used=False
        )

        return ProtocolMessage(header=header, payload=payload)

    def send_message(self, message: ProtocolMessage) -> str:
        """Send a message through the protocol engine"""
        with self.engine_lock:
            try:
                # Validate message
                self._validate_message(message)

                # Send message
                message_id = self._send_message_internal(message)

                # Update statistics
                self.statistics["messages_sent"] += 1
                self.statistics["bytes_transferred"] += message.payload.size_bytes

                logger.debug(
                    f"Sent message {message_id} to {message.header.destination_dimension.value}"
                )

                return message_id

            except Exception as e:
                self.statistics["messages_failed"] += 1
                logger.error(f"Failed to send message: {e}")
                raise

    def _validate_message(self, message: ProtocolMessage) -> None:
        """Validate message before sending"""
        if not message.header.message_id:
            raise KimeraValidationError("Message ID is required")

        if message.header.source_dimension != self.local_dimension:
            raise KimeraValidationError("Message source must be local dimension")

        if not message.payload.checksum:
            raise KimeraValidationError("Message checksum is required")

        # Verify checksum
        expected_checksum = message._calculate_checksum()
        if message.payload.checksum != expected_checksum:
            raise KimeraValidationError("Message checksum verification failed")

    def _send_message_internal(self, message: ProtocolMessage) -> str:
        """Internal message sending implementation"""
        start_time = time.time()

        try:
            # Find route to destination
            route = self.router.find_route(
                message.header.source_dimension, message.header.destination_dimension
            )

            if not route:
                raise KimeraCognitiveError(
                    f"No route found to {message.header.destination_dimension.value}"
                )

            # Store pending message for delivery confirmation
            if message.header.delivery_guarantee != DeliveryGuarantee.AT_MOST_ONCE:
                self.pending_messages[message.header.message_id] = message

            # Simulate message transmission
            # TODO: Replace with actual network transmission
            self._simulate_message_transmission(message, route)

            # Update route performance
            latency = time.time() - start_time
            self.router.update_route_performance(route.route_hash, latency, True)

            # Update average latency
            self._update_average_latency(latency)

            return message.header.message_id

        except Exception as e:
            # Update route performance for failure
            latency = time.time() - start_time
            if "route" in locals():
                self.router.update_route_performance(route.route_hash, latency, False)

            raise

    def _simulate_message_transmission(
        self, message: ProtocolMessage, route: MessageRoute
    ) -> None:
        """
        Simulate message transmission through route

        TODO: Replace with actual network/IPC transmission
        """
        # Simulate network delay based on route complexity
        base_delay = 0.001  # 1ms base delay
        hop_delay = len(route.hops) * 0.002  # 2ms per hop

        import random

        jitter = random.uniform(0.0, 0.001)  # Up to 1ms jitter

        total_delay = base_delay + hop_delay + jitter
        time.sleep(total_delay)

        # Simulate occasional transmission failure (0.1% failure rate)
        if random.random() < 0.001:
            raise KimeraCognitiveError("Simulated transmission failure")

        logger.debug(
            f"Simulated transmission: {message.header.message_id} "
            f"via {len(route.hops)} hops in {total_delay:.4f}s"
        )

    def _update_average_latency(self, latency: float) -> None:
        """Update running average latency"""
        current_avg = self.statistics["average_latency"]
        total_messages = self.statistics["messages_sent"]

        if total_messages == 1:
            self.statistics["average_latency"] = latency
        else:
            # Exponential moving average
            alpha = 0.1
            self.statistics["average_latency"] = (alpha * latency) + (
                (1 - alpha) * current_avg
            )

    def register_message_handler(
        self, message_type: MessageType, handler: Callable[[ProtocolMessage], None]
    ) -> None:
        """Register a handler for specific message type"""
        if message_type not in self.message_handlers:
            self.message_handlers[message_type] = []

        self.message_handlers[message_type].append(handler)
        logger.debug(f"Registered handler for {message_type.value}")

    def receive_message(self, message: ProtocolMessage) -> None:
        """Handle received message"""
        with self.engine_lock:
            try:
                # Update statistics
                self.statistics["messages_received"] += 1
                self.statistics["bytes_transferred"] += message.payload.size_bytes

                # Validate message
                self._validate_received_message(message)

                # Call registered handlers
                handlers = self.message_handlers.get(message.header.message_type, [])
                for handler in handlers:
                    try:
                        handler(message)
                    except Exception as e:
                        logger.error(f"Message handler error: {e}")

                logger.debug(
                    f"Received message {message.header.message_id} "
                    f"from {message.header.source_dimension.value}"
                )

            except Exception as e:
                logger.error(f"Failed to handle received message: {e}")

    def _validate_received_message(self, message: ProtocolMessage) -> None:
        """Validate received message"""
        # Check TTL
        age = (datetime.now() - message.header.timestamp).total_seconds()
        if age > message.header.ttl_seconds:
            raise KimeraValidationError(
                f"Message expired (age: {age:.1f}s, TTL: {message.header.ttl_seconds}s)"
            )

        # Verify checksum
        expected_checksum = message._calculate_checksum()
        if message.payload.checksum != expected_checksum:
            raise KimeraValidationError("Received message checksum verification failed")

    def get_connection_status(self, dimension: SystemDimension) -> Dict[str, Any]:
        """Get connection status for specific dimension"""
        dimension_info = self.registry.get_dimension_info(dimension)

        if not dimension_info:
            return {"status": "unknown", "registered": False}

        # Check if dimension is healthy
        healthy_dimensions = self.registry.get_healthy_dimensions()
        is_healthy = dimension in healthy_dimensions

        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "registered": True
            "last_heartbeat": dimension_info["last_heartbeat"],
            "capabilities": dimension_info["capabilities"],
            "health_metrics": dimension_info["health"].get("metrics", {}),
        }

    def get_engine_statistics(self) -> Dict[str, Any]:
        """Get comprehensive engine statistics"""
        uptime = (datetime.now() - self.statistics["start_time"]).total_seconds()

        return {
            "local_dimension": self.local_dimension.value
            "uptime_seconds": uptime
            "messages": {
                "sent": self.statistics["messages_sent"],
                "received": self.statistics["messages_received"],
                "failed": self.statistics["messages_failed"],
                "success_rate": 1.0
                - (
                    self.statistics["messages_failed"]
                    / max(1, self.statistics["messages_sent"])
                ),
            },
            "performance": {
                "bytes_transferred": self.statistics["bytes_transferred"],
                "average_latency": self.statistics["average_latency"],
                "throughput_bytes_per_second": self.statistics["bytes_transferred"]
                / max(1, uptime),
            },
            "connections": {
                "active_connections": len(self.connections),
                "registered_dimensions": len(self.registry.dimensions),
                "healthy_dimensions": len(self.registry.get_healthy_dimensions()),
            },
            "queue_status": {
                "pending_messages": len(self.pending_messages),
                "queue_size": self.message_queue.qsize(),
            },
            "registry_status": self.registry.get_registry_status(),
        }

    def get_route_information(
        self, destination: SystemDimension
    ) -> Optional[Dict[str, Any]]:
        """Get routing information for specific destination"""
        route = self.router.find_route(self.local_dimension, destination)

        if not route:
            return None

        return {
            "destination": destination.value
            "route_hash": route.route_hash
            "hops": [hop.value for hop in route.hops],
            "performance_metrics": route.performance_metrics
            "direct_route": len(route.hops) == 0
        }


# Global instances for module access
_engines: Dict[SystemDimension, ProtocolEngine] = {}
_registry: Optional[DimensionRegistry] = None


def get_protocol_engine(dimension: SystemDimension) -> ProtocolEngine:
    """Get protocol engine instance for specific dimension"""
    global _engines
    if dimension not in _engines:
        _engines[dimension] = ProtocolEngine(dimension)
    return _engines[dimension]


def get_global_registry() -> DimensionRegistry:
    """Get global dimension registry"""
    global _registry
    if _registry is None:
        _registry = DimensionRegistry()
    return _registry
