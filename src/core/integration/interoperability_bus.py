"""
Cognitive Interoperability Bus - High-Performance Component Communication
========================================================================

The interoperability bus provides seamless, high-performance communication
between all cognitive components with intelligent routing, priority handling
and event-driven architecture.

Key Features:
- High-performance message passing with priority queues
- Intelligent message routing and load balancing
- Event-driven publish-subscribe architecture
- Request-response patterns for synchronous operations
- Component registration and discovery
- Message transformation and protocol adaptation
- Performance monitoring and optimization

This is the nervous system that enables all cognitive components to
communicate efficiently and reliably.
"""

import asyncio
import json
import logging
import threading
import time
import uuid
import weakref
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from ...config.settings import get_settings
from ...utils.config import get_api_settings
from ...utils.kimera_logger import get_cognitive_logger

logger = get_cognitive_logger(__name__)


class MessagePriority(Enum):
    """Message priority levels"""

    CRITICAL = 1  # Critical system messages
    HIGH = 2  # High priority operations
    NORMAL = 3  # Normal operations
    LOW = 4  # Background tasks
    BULK = 5  # Bulk data transfers


class MessageType(Enum):
    """Types of messages in the system"""

    EVENT = "event"  # Asynchronous event notifications
    REQUEST = "request"  # Synchronous request messages
    RESPONSE = "response"  # Response to request messages
    BROADCAST = "broadcast"  # Broadcast to all subscribers
    TARGETED = "targeted"  # Targeted to specific component
    SYSTEM = "system"  # System control messages


class ComponentState(Enum):
    """States of registered components"""

    REGISTERING = "registering"
    ACTIVE = "active"
    BUSY = "busy"
    IDLE = "idle"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"


@dataclass
class Message:
    """Auto-generated class."""
    pass
    """Message structure for interoperability bus"""

    message_id: str
    message_type: MessageType
    priority: MessagePriority
    source_component: str
    target_component: Optional[str] = None
    event_type: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Timing information
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    processing_deadline: Optional[datetime] = None

    # Routing information
    routing_path: List[str] = field(default_factory=list)
    broadcast_targets: Set[str] = field(default_factory=set)

    # Response tracking
    requires_response: bool = False
    correlation_id: Optional[str] = None
    response_timeout: float = 30.0

    def is_expired(self) -> bool:
        """Check if message has expired"""
        if self.expires_at:
            return datetime.now() > self.expires_at
        return False

    def add_routing_hop(self, component: str):
        """Add component to routing path"""
        self.routing_path.append(component)

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization"""
        return {
            "message_id": self.message_id
            "message_type": self.message_type.value
            "priority": self.priority.value
            "source_component": self.source_component
            "target_component": self.target_component
            "event_type": self.event_type
            "payload": self.payload
            "metadata": self.metadata
            "created_at": self.created_at.isoformat(),
            "requires_response": self.requires_response
            "correlation_id": self.correlation_id
        }


@dataclass
class ComponentInfo:
    """Auto-generated class."""
    pass
    """Information about registered components"""

    component_id: str
    component_type: str
    capabilities: List[str]
    event_subscriptions: Set[str]
    state: ComponentState

    # Connection information
    registered_at: datetime = field(default_factory=datetime.now)
    last_heartbeat: datetime = field(default_factory=datetime.now)
    heartbeat_interval: float = 30.0

    # Performance metrics
    messages_sent: int = 0
    messages_received: int = 0
    messages_processed: int = 0
    average_processing_time: float = 0.0
    error_count: int = 0

    # Resource information
    load_factor: float = 0.0
    resource_usage: Dict[str, float] = field(default_factory=dict)

    def update_heartbeat(self):
        """Update heartbeat timestamp"""
        self.last_heartbeat = datetime.now()

    def is_alive(self) -> bool:
        """Check if component is alive based on heartbeat"""
        timeout = timedelta(seconds=self.heartbeat_interval * 2)
        return datetime.now() - self.last_heartbeat < timeout

    def update_performance(self, processing_time: float, success: bool):
        """Update performance metrics"""
        self.messages_processed += 1
        if success:
            # Update average processing time
            total_time = self.average_processing_time * (self.messages_processed - 1)
            self.average_processing_time = (
                total_time + processing_time
            ) / self.messages_processed
        else:
            self.error_count += 1


@dataclass
class EventStream:
    """Auto-generated class."""
    pass
    """Event stream for specific event types"""

    event_type: str
    subscribers: Set[str] = field(default_factory=set)
    message_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    total_messages: int = 0
    dropped_messages: int = 0
    max_queue_size: int = 1000

    async def publish(self, message: Message) -> bool:
        """Publish message to event stream"""
        try:
            if self.message_queue.qsize() >= self.max_queue_size:
                # Drop oldest message
                try:
                    self.message_queue.get_nowait()
                    self.dropped_messages += 1
                except asyncio.QueueEmpty:
                    pass

            await self.message_queue.put(message)
            self.total_messages += 1
            return True

        except Exception as e:
            logger.error(
                f"Failed to publish message to event stream {self.event_type}: {e}"
            )
            return False

    async def consume(self, timeout: float = 1.0) -> Optional[Message]:
        """Consume message from event stream"""
        try:
            return await asyncio.wait_for(self.message_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    def add_subscriber(self, component_id: str):
        """Add subscriber to event stream"""
        self.subscribers.add(component_id)

    def remove_subscriber(self, component_id: str):
        """Remove subscriber from event stream"""
        self.subscribers.discard(component_id)
class MessageRouter:
    """Auto-generated class."""
    pass
    """
    Intelligent Message Router

    Routes messages between components with load balancing
    priority handling, and performance optimization.
    """

    def __init__(self):
        self.routing_table: Dict[str, List[str]] = defaultdict(list)
        self.load_balancer = LoadBalancer()
        self.performance_monitor = RouterPerformanceMonitor()

        # Routing strategies
        self.routing_strategies = {
            "round_robin": self._round_robin_route
            "least_loaded": self._least_loaded_route
            "fastest_response": self._fastest_response_route
            "affinity_based": self._affinity_based_route
        }

        self.default_strategy = "least_loaded"

        logger.debug("Message router initialized")

    def register_route(
        self, event_type: str, target_component: str, strategy: str = None
    ):
        """Register routing rule for event type"""
        if event_type not in self.routing_table:
            self.routing_table[event_type] = []

        if target_component not in self.routing_table[event_type]:
            self.routing_table[event_type].append(target_component)

        logger.debug(f"Registered route: {event_type} -> {target_component}")

    def unregister_route(self, event_type: str, target_component: str):
        """Unregister routing rule"""
        if event_type in self.routing_table:
            self.routing_table[event_type] = [
                comp
                for comp in self.routing_table[event_type]
                if comp != target_component
            ]

    async def route_message(
        self
        message: Message
        available_components: Dict[str, ComponentInfo],
        strategy: str = None
    ) -> List[str]:
        """
        Route message to appropriate components

        Args:
            message: Message to route
            available_components: Available target components
            strategy: Routing strategy to use

        Returns:
            List of target component IDs
        """
        routing_strategy = strategy or self.default_strategy

        # Handle targeted messages
        if message.target_component:
            if message.target_component in available_components:
                return [message.target_component]
            else:
                logger.warning(
                    f"Target component {message.target_component} not available"
                )
                return []

        # Handle broadcast messages
        if message.message_type == MessageType.BROADCAST:
            return list(available_components.keys())

        # Handle event-based routing
        if message.event_type and message.event_type in self.routing_table:
            candidates = [
                comp
                for comp in self.routing_table[message.event_type]
                if comp in available_components
            ]

            if candidates:
                strategy_func = self.routing_strategies.get(
                    routing_strategy, self._least_loaded_route
                )
                return await strategy_func(message, candidates, available_components)

        # Default: route to least loaded component
        if available_components:
            return await self._least_loaded_route(
                message, list(available_components.keys()), available_components
            )

        return []

    async def _round_robin_route(
        self
        message: Message
        candidates: List[str],
        components: Dict[str, ComponentInfo],
    ) -> List[str]:
        """Round-robin routing strategy"""
        if not candidates:
            return []

        # Use message hash for consistent round-robin
        index = hash(message.message_id) % len(candidates)
        return [candidates[index]]

    async def _least_loaded_route(
        self
        message: Message
        candidates: List[str],
        components: Dict[str, ComponentInfo],
    ) -> List[str]:
        """Route to least loaded component"""
        if not candidates:
            return []

        # Find component with lowest load factor
        least_loaded = min(candidates, key=lambda comp: components[comp].load_factor)

        return [least_loaded]

    async def _fastest_response_route(
        self
        message: Message
        candidates: List[str],
        components: Dict[str, ComponentInfo],
    ) -> List[str]:
        """Route to component with fastest average response time"""
        if not candidates:
            return []

        fastest = min(
            candidates, key=lambda comp: components[comp].average_processing_time
        )

        return [fastest]

    async def _affinity_based_route(
        self
        message: Message
        candidates: List[str],
        components: Dict[str, ComponentInfo],
    ) -> List[str]:
        """Route based on component affinity (sticky sessions)"""
        # Use source component for affinity
        affinity_key = message.source_component

        if affinity_key and candidates:
            # Consistent hashing for affinity
            index = hash(affinity_key) % len(candidates)
            return [candidates[index]]

        return await self._least_loaded_route(message, candidates, components)
class LoadBalancer:
    """Auto-generated class."""
    pass
    """Load balancer for component resource management"""

    def __init__(self):
        self.load_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.load_thresholds = {"cpu": 0.8, "memory": 0.9, "queue": 0.7}

    def update_component_load(self, component_id: str, load_metrics: Dict[str, float]):
        """Update load metrics for component"""
        self.load_history[component_id].append(
            {"timestamp": datetime.now(), "metrics": load_metrics}
        )

    def calculate_load_factor(self, component_id: str) -> float:
        """Calculate overall load factor for component"""
        if component_id not in self.load_history:
            return 0.0

        recent_loads = list(self.load_history[component_id])[
            -10:
        ]  # Last 10 measurements
        if not recent_loads:
            return 0.0

        # Calculate weighted average of different load metrics
        total_load = 0.0
        weight_sum = 0.0

        for load_entry in recent_loads:
            metrics = load_entry["metrics"]

            # Weight different metrics
            cpu_weight = 0.4
            memory_weight = 0.3
            queue_weight = 0.3

            load_value = (
                metrics.get("cpu", 0.0) * cpu_weight
                + metrics.get("memory", 0.0) * memory_weight
                + metrics.get("queue", 0.0) * queue_weight
            )

            total_load += load_value
            weight_sum += 1.0

        return total_load / weight_sum if weight_sum > 0 else 0.0

    def is_overloaded(self, component_id: str) -> bool:
        """Check if component is overloaded"""
        load_factor = self.calculate_load_factor(component_id)
        return load_factor > 0.8  # 80% threshold for overload
class RouterPerformanceMonitor:
    """Auto-generated class."""
    pass
    """Performance monitoring for message router"""

    def __init__(self):
        self.routing_times: deque = deque(maxlen=1000)
        self.routing_failures: deque = deque(maxlen=1000)
        self.total_routes = 0
        self.failed_routes = 0

    def record_routing_time(self, routing_time: float):
        """Record routing operation time"""
        self.routing_times.append(routing_time)
        self.total_routes += 1

    def record_routing_failure(self, error: Exception):
        """Record routing failure"""
        self.routing_failures.append({"timestamp": datetime.now(), "error": str(error)})
        self.failed_routes += 1

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get routing performance metrics"""
        return {
            "total_routes": self.total_routes
            "failed_routes": self.failed_routes
            "success_rate": (
                (self.total_routes - self.failed_routes) / self.total_routes
                if self.total_routes > 0
                else 0.0
            ),
            "average_routing_time": (
                sum(self.routing_times) / len(self.routing_times)
                if self.routing_times
                else 0.0
            ),
            "recent_failures": len(
                [
                    f
                    for f in self.routing_failures
                    if datetime.now() - f["timestamp"] < timedelta(minutes=5)
                ]
            ),
        }
class ComponentRegistry:
    """Auto-generated class."""
    pass
    """
    Component Registry and Discovery

    Manages registration, discovery, and lifecycle of cognitive components.
    """

    def __init__(self):
        self.components: Dict[str, ComponentInfo] = {}
        self.component_capabilities: Dict[str, Set[str]] = defaultdict(set)
        self.heartbeat_monitor = HeartbeatMonitor(self)

        # Registry events
        self.registry_callbacks: Dict[str, List[Callable]] = defaultdict(list)

        logger.debug("Component registry initialized")

    async def register_component(
        self
        component_id: str
        component_type: str
        capabilities: List[str],
        heartbeat_interval: float = 30.0
    ) -> bool:
        """
        Register a component with the registry

        Args:
            component_id: Unique component identifier
            component_type: Type of component
            capabilities: List of component capabilities
            heartbeat_interval: Heartbeat interval in seconds

        Returns:
            True if registration successful
        """
        try:
            if component_id in self.components:
                logger.warning(f"Component {component_id} already registered, updating")

            component_info = ComponentInfo(
                component_id=component_id
                component_type=component_type
                capabilities=capabilities
                event_subscriptions=set(),
                state=ComponentState.REGISTERING
                heartbeat_interval=heartbeat_interval
            )

            self.components[component_id] = component_info

            # Update capability index
            for capability in capabilities:
                self.component_capabilities[capability].add(component_id)

            # Update state to active
            component_info.state = ComponentState.ACTIVE

            # Trigger registration callbacks
            await self._trigger_registry_callbacks(
                "component_registered",
                {"component_id": component_id, "component_info": component_info},
            )

            logger.info(f"✅ Component {component_id} registered successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to register component {component_id}: {e}")
            return False

    async def unregister_component(self, component_id: str) -> bool:
        """Unregister component from registry"""
        try:
            if component_id not in self.components:
                logger.warning(f"Component {component_id} not found for unregistration")
                return False

            component_info = self.components[component_id]
            component_info.state = ComponentState.DISCONNECTING

            # Remove from capability index
            for capability in component_info.capabilities:
                self.component_capabilities[capability].discard(component_id)

            # Remove component
            del self.components[component_id]

            # Trigger unregistration callbacks
            await self._trigger_registry_callbacks(
                "component_unregistered",
                {"component_id": component_id, "component_info": component_info},
            )

            logger.info(f"Component {component_id} unregistered")
            return True

        except Exception as e:
            logger.error(f"Failed to unregister component {component_id}: {e}")
            return False

    def discover_components(
        self, capability: Optional[str] = None, component_type: Optional[str] = None
    ) -> List[ComponentInfo]:
        """
        Discover components by capability or type

        Args:
            capability: Required capability
            component_type: Required component type

        Returns:
            List of matching components
        """
        candidates = list(self.components.values())

        # Filter by capability
        if capability:
            candidates = [
                comp for comp in candidates if capability in comp.capabilities
            ]

        # Filter by type
        if component_type:
            candidates = [
                comp for comp in candidates if comp.component_type == component_type
            ]

        # Filter by state (only active components)
        candidates = [
            comp
            for comp in candidates
            if comp.state == ComponentState.ACTIVE and comp.is_alive()
        ]

        return candidates

    def get_component(self, component_id: str) -> Optional[ComponentInfo]:
        """Get component information by ID"""
        return self.components.get(component_id)

    def update_component_heartbeat(self, component_id: str) -> bool:
        """Update component heartbeat"""
        if component_id in self.components:
            self.components[component_id].update_heartbeat()
            return True
        return False

    def update_component_state(self, component_id: str, state: ComponentState) -> bool:
        """Update component state"""
        if component_id in self.components:
            old_state = self.components[component_id].state
            self.components[component_id].state = state

            logger.debug(
                f"Component {component_id} state changed: {old_state.value} -> {state.value}"
            )
            return True
        return False

    async def _trigger_registry_callbacks(
        self, event_type: str, event_data: Dict[str, Any]
    ):
        """Trigger registry event callbacks"""
        for callback in self.registry_callbacks[event_type]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event_data)
                else:
                    callback(event_data)
            except Exception as e:
                logger.warning(f"Registry callback failed for {event_type}: {e}")

    def register_callback(self, event_type: str, callback: Callable):
        """Register callback for registry events"""
        self.registry_callbacks[event_type].append(callback)

    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        active_components = sum(
            1
            for comp in self.components.values()
            if comp.state == ComponentState.ACTIVE
        )

        return {
            "total_components": len(self.components),
            "active_components": active_components
            "component_types": list(
                set(comp.component_type for comp in self.components.values())
            ),
            "total_capabilities": len(self.component_capabilities),
            "capability_distribution": {
                cap: len(comps) for cap, comps in self.component_capabilities.items()
            },
        }
class HeartbeatMonitor:
    """Auto-generated class."""
    pass
    """Monitors component heartbeats and handles disconnections"""

    def __init__(self, registry: ComponentRegistry):
        self.registry = registry
        self.monitoring_task = None
        self.monitoring_interval = 10.0  # Check every 10 seconds
        self.running = False

    async def start_monitoring(self):
        """Start heartbeat monitoring"""
        if self.running:
            return

        self.running = True
        self.monitoring_task = asyncio.create_task(self._monitor_heartbeats())
        logger.debug("Heartbeat monitoring started")

    async def stop_monitoring(self):
        """Stop heartbeat monitoring"""
        self.running = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        logger.debug("Heartbeat monitoring stopped")

    async def _monitor_heartbeats(self):
        """Monitor component heartbeats"""
        while self.running:
            try:
                current_time = datetime.now()
                dead_components = []

                for component_id, component_info in self.registry.components.items():
                    if not component_info.is_alive():
                        dead_components.append(component_id)
                        logger.warning(f"Component {component_id} heartbeat timeout")

                # Mark dead components as disconnected
                for component_id in dead_components:
                    await self.registry.update_component_state(
                        component_id, ComponentState.DISCONNECTED
                    )

                await asyncio.sleep(self.monitoring_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat monitoring error: {e}")
                await asyncio.sleep(1.0)
class CognitiveInteroperabilityBus:
    """Auto-generated class."""
    pass
    """
    Main Cognitive Interoperability Bus

    High-performance message bus for cognitive component communication
    with intelligent routing, priority handling, and performance optimization.
    """

    def __init__(
        self
        max_queue_size: int = 10000
        max_workers: int = 4
        enable_persistence: bool = False
    ):
        """
        Initialize Cognitive Interoperability Bus

        Args:
            max_queue_size: Maximum message queue size
            max_workers: Maximum worker threads for processing
            enable_persistence: Enable message persistence
        """
        self.settings = get_api_settings()
        self.max_queue_size = max_queue_size
        self.max_workers = max_workers
        self.enable_persistence = enable_persistence

        # Core components
        self.message_router = MessageRouter()
        self.component_registry = ComponentRegistry()
        self.load_balancer = LoadBalancer()

        # Message queues by priority
        self.message_queues = {
            priority: asyncio.Queue(maxsize=max_queue_size)
            for priority in MessagePriority
        }

        # Event streams
        self.event_streams: Dict[str, EventStream] = {}

        # Request-response tracking
        self.pending_requests: Dict[str, asyncio.Future] = {}
        self.request_timeouts: Dict[str, asyncio.Task] = {}

        # Performance metrics
        self.performance_metrics = {
            "total_messages": 0
            "messages_processed": 0
            "messages_dropped": 0
            "average_latency": 0.0
            "throughput": 0.0
            "error_rate": 0.0
        }

        # Bus state
        self.running = False
        self.worker_tasks: List[asyncio.Task] = []

        # Thread pool for CPU-intensive operations
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)

        logger.info(f"🚌 Cognitive Interoperability Bus initialized")
        logger.info(f"   Max queue size: {max_queue_size}")
        logger.info(f"   Max workers: {max_workers}")
        logger.info(f"   Persistence enabled: {enable_persistence}")

    async def start(self):
        """Start the interoperability bus"""
        if self.running:
            return

        self.running = True

        # Start component registry heartbeat monitoring
        await self.component_registry.heartbeat_monitor.start_monitoring()

        # Start message processing workers
        for i in range(self.max_workers):
            worker_task = asyncio.create_task(self._message_worker(f"worker_{i}"))
            self.worker_tasks.append(worker_task)

        # Start performance monitoring
        performance_task = asyncio.create_task(self._performance_monitor())
        self.worker_tasks.append(performance_task)

        logger.info("✅ Cognitive Interoperability Bus started")

    async def stop(self):
        """Stop the interoperability bus"""
        if not self.running:
            return

        self.running = False

        # Stop heartbeat monitoring
        await self.component_registry.heartbeat_monitor.stop_monitoring()

        # Cancel all worker tasks
        for task in self.worker_tasks:
            task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)

        # Close thread pool
        self.thread_pool.shutdown(wait=True)

        logger.info("Cognitive Interoperability Bus stopped")

    async def register_component(
        self
        component_id: str
        component_type: str
        capabilities: List[str],
        event_subscriptions: List[str] = None
    ) -> bool:
        """Register component with the bus"""
        success = await self.component_registry.register_component(
            component_id=component_id
            component_type=component_type
            capabilities=capabilities
        )

        if success and event_subscriptions:
            await self._setup_event_subscriptions(component_id, event_subscriptions)

        return success

    async def unregister_component(self, component_id: str) -> bool:
        """Unregister component from the bus"""
        # Remove from event subscriptions
        for event_stream in self.event_streams.values():
            event_stream.remove_subscriber(component_id)

        return await self.component_registry.unregister_component(component_id)

    async def publish(
        self
        source_component: str
        event_type: str
        payload: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Publish event message

        Args:
            source_component: Source component ID
            event_type: Type of event
            payload: Event payload
            priority: Message priority
            metadata: Optional metadata

        Returns:
            Message ID
        """
        message_id = str(uuid.uuid4())

        message = Message(
            message_id=message_id
            message_type=MessageType.EVENT
            priority=priority
            source_component=source_component
            event_type=event_type
            payload=payload
            metadata=metadata or {},
        )

        await self._enqueue_message(message)

        # Update performance metrics
        self.performance_metrics["total_messages"] += 1

        return message_id

    async def subscribe(
        self, component_id: str, event_types: List[str], callback: Callable
    ) -> bool:
        """
        Subscribe component to event types

        Args:
            component_id: Component ID
            event_types: List of event types to subscribe to
            callback: Callback function for events

        Returns:
            True if subscription successful
        """
        try:
            for event_type in event_types:
                # Create event stream if it doesn't exist
                if event_type not in self.event_streams:
                    self.event_streams[event_type] = EventStream(event_type=event_type)

                # Add subscriber
                self.event_streams[event_type].add_subscriber(component_id)

                # Register route for this event type
                self.message_router.register_route(event_type, component_id)

            logger.debug(
                f"Component {component_id} subscribed to events: {event_types}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to subscribe component {component_id}: {e}")
            return False

    async def request_response(
        self
        source_component: str
        target_component: str
        request_type: str
        payload: Dict[str, Any],
        timeout: float = 30.0
        priority: MessagePriority = MessagePriority.NORMAL
    ) -> Dict[str, Any]:
        """
        Send request and wait for response

        Args:
            source_component: Source component ID
            target_component: Target component ID
            request_type: Type of request
            payload: Request payload
            timeout: Response timeout in seconds
            priority: Message priority

        Returns:
            Response payload
        """
        message_id = str(uuid.uuid4())
        correlation_id = str(uuid.uuid4())

        # Create response future
        response_future = asyncio.Future()
        self.pending_requests[correlation_id] = response_future

        # Create timeout task
        timeout_task = asyncio.create_task(
            self._handle_request_timeout(correlation_id, timeout)
        )
        self.request_timeouts[correlation_id] = timeout_task

        try:
            # Create request message
            message = Message(
                message_id=message_id
                message_type=MessageType.REQUEST
                priority=priority
                source_component=source_component
                target_component=target_component
                event_type=request_type
                payload=payload
                requires_response=True
                correlation_id=correlation_id
                response_timeout=timeout
            )

            await self._enqueue_message(message)

            # Wait for response
            response = await response_future

            return response

        except asyncio.TimeoutError:
            raise asyncio.TimeoutError(f"Request timeout after {timeout} seconds")

        finally:
            # Cleanup
            self.pending_requests.pop(correlation_id, None)
            timeout_task.cancel()
            self.request_timeouts.pop(correlation_id, None)

    async def send_response(
        self
        response_to_message_id: str
        correlation_id: str
        payload: Dict[str, Any],
        source_component: str
    ) -> bool:
        """Send response to request"""
        try:
            if correlation_id in self.pending_requests:
                future = self.pending_requests[correlation_id]
                if not future.done():
                    future.set_result(payload)
                    return True

            logger.warning(
                f"No pending request found for correlation_id: {correlation_id}"
            )
            return False

        except Exception as e:
            logger.error(f"Failed to send response: {e}")
            return False

    async def broadcast(
        self
        source_component: str
        event_type: str
        payload: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL
    ) -> str:
        """Broadcast message to all subscribers"""
        message_id = str(uuid.uuid4())

        message = Message(
            message_id=message_id
            message_type=MessageType.BROADCAST
            priority=priority
            source_component=source_component
            event_type=event_type
            payload=payload
        )

        await self._enqueue_message(message)
        return message_id

    async def _setup_event_subscriptions(
        self, component_id: str, event_subscriptions: List[str]
    ):
        """Setup event subscriptions for component"""
        component_info = self.component_registry.get_component(component_id)
        if component_info:
            component_info.event_subscriptions.update(event_subscriptions)

            for event_type in event_subscriptions:
                if event_type not in self.event_streams:
                    self.event_streams[event_type] = EventStream(event_type=event_type)

                self.event_streams[event_type].add_subscriber(component_id)
                self.message_router.register_route(event_type, component_id)

    async def _enqueue_message(self, message: Message):
        """Enqueue message for processing"""
        try:
            queue = self.message_queues[message.priority]

            if queue.full():
                # Handle queue overflow
                await self._handle_queue_overflow(message)
            else:
                await queue.put(message)

        except Exception as e:
            logger.error(f"Failed to enqueue message {message.message_id}: {e}")
            self.performance_metrics["messages_dropped"] += 1

    async def _handle_queue_overflow(self, message: Message):
        """Handle message queue overflow"""
        # Drop lower priority message to make room
        if message.priority.value <= MessagePriority.HIGH.value:
            # Try to drop a low priority message
            for priority in reversed(list(MessagePriority)):
                queue = self.message_queues[priority]
                if not queue.empty():
                    try:
                        dropped_message = queue.get_nowait()
                        logger.warning(
                            f"Dropped message {dropped_message.message_id} due to queue overflow"
                        )
                        self.performance_metrics["messages_dropped"] += 1

                        # Enqueue the new message
                        await queue.put(message)
                        return
                    except asyncio.QueueEmpty:
                        continue

        # If we can't drop anything, drop the new message
        logger.warning(f"Dropped message {message.message_id} - all queues full")
        self.performance_metrics["messages_dropped"] += 1

    async def _message_worker(self, worker_id: str):
        """Message processing worker"""
        logger.debug(f"Message worker {worker_id} started")

        while self.running:
            try:
                # Process messages by priority
                message = None

                for priority in MessagePriority:
                    queue = self.message_queues[priority]
                    try:
                        message = queue.get_nowait()
                        break
                    except asyncio.QueueEmpty:
                        continue

                if message is None:
                    await asyncio.sleep(0.01)  # Short sleep if no messages
                    continue

                # Process message
                await self._process_message(message)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Message worker {worker_id} error: {e}")
                await asyncio.sleep(0.1)

        logger.debug(f"Message worker {worker_id} stopped")

    async def _process_message(self, message: Message):
        """Process individual message"""
        start_time = time.time()

        try:
            # Check if message is expired
            if message.is_expired():
                logger.debug(f"Message {message.message_id} expired, dropping")
                self.performance_metrics["messages_dropped"] += 1
                return

            # Route message to target components
            available_components = {
                comp_id: comp_info
                for comp_id, comp_info in self.component_registry.components.items()
                if comp_info.state == ComponentState.ACTIVE and comp_info.is_alive()
            }

            target_components = await self.message_router.route_message(
                message, available_components
            )

            # Deliver message to target components
            for target_component in target_components:
                await self._deliver_message(message, target_component)

            # Update performance metrics
            processing_time = time.time() - start_time
            self.performance_metrics["messages_processed"] += 1

            # Update average latency
            total_latency = (
                self.performance_metrics["average_latency"]
                * (self.performance_metrics["messages_processed"] - 1)
                + processing_time
            )
            self.performance_metrics["average_latency"] = (
                total_latency / self.performance_metrics["messages_processed"]
            )

        except Exception as e:
            logger.error(f"Failed to process message {message.message_id}: {e}")
            self.performance_metrics["error_rate"] += 1

    async def _deliver_message(self, message: Message, target_component: str):
        """Deliver message to specific component"""
        try:
            # Add routing hop
            message.add_routing_hop(target_component)

            # Check if component has event stream subscription
            if message.event_type and message.event_type in self.event_streams:
                event_stream = self.event_streams[message.event_type]
                if target_component in event_stream.subscribers:
                    await event_stream.publish(message)

            # Update component metrics
            component_info = self.component_registry.get_component(target_component)
            if component_info:
                component_info.messages_received += 1

            logger.debug(
                f"Message {message.message_id} delivered to {target_component}"
            )

        except Exception as e:
            logger.error(
                f"Failed to deliver message {message.message_id} to {target_component}: {e}"
            )

    async def _handle_request_timeout(self, correlation_id: str, timeout: float):
        """Handle request timeout"""
        await asyncio.sleep(timeout)

        future = self.pending_requests.get(correlation_id)
        if future and not future.done():
            future.set_exception(asyncio.TimeoutError("Request timeout"))

    async def _performance_monitor(self):
        """Monitor bus performance"""
        last_processed = 0

        while self.running:
            try:
                await asyncio.sleep(5.0)  # Monitor every 5 seconds

                # Calculate throughput
                current_processed = self.performance_metrics["messages_processed"]
                throughput = (current_processed - last_processed) / 5.0
                self.performance_metrics["throughput"] = throughput
                last_processed = current_processed

                # Log performance metrics
                if self.performance_metrics["total_messages"] > 0:
                    logger.debug(
                        f"Bus performance: {throughput:.2f} msg/sec, "
                        f"avg latency: {self.performance_metrics['average_latency']:.3f}s"
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "bus_status": {
                "running": self.running
                "worker_count": len(self.worker_tasks),
                "queue_sizes": {
                    priority.name: queue.qsize()
                    for priority, queue in self.message_queues.items()
                },
            },
            "performance_metrics": self.performance_metrics.copy(),
            "registry_stats": self.component_registry.get_registry_stats(),
            "router_metrics": self.message_router.performance_monitor.get_performance_metrics(),
            "event_streams": {
                event_type: {
                    "subscribers": len(stream.subscribers),
                    "total_messages": stream.total_messages
                    "dropped_messages": stream.dropped_messages
                    "queue_size": stream.message_queue.qsize(),
                }
                for event_type, stream in self.event_streams.items()
            },
        }
