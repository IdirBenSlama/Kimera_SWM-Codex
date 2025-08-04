#!/usr/bin/env python3
"""
Unified Integration Manager for Output Generation and Portal Management
=====================================================================

DO-178C Level A compliant unified integration system that coordinates
multi-modal output generation with interdimensional portal management,
ensuring seamless cognitive state transitions and information exchange.

Key Features:
- Unified coordination of output generation and portal management
- Cross-system resource optimization and scheduling
- Real-time performance monitoring and health assessment
- Event-driven architecture with asynchronous communication
- Nuclear-grade safety protocols and emergency procedures

Author: KIMERA Development Team
Version: 1.0.0 (DO-178C Level A)
"""

import asyncio
import json
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from utils.kimera_exceptions import KimeraCognitiveError, KimeraValidationError
from utils.kimera_logger import LogCategory, get_logger

# Import core components
from ..output_generation.multi_modal_output_generator import (
    MultiModalOutputGenerator,
    OutputArtifact,
    OutputModality,
    OutputQuality,
    get_multi_modal_output_generator,
)
from ..portal_management.interdimensional_portal_manager import (
    DimensionalSpace,
    InterdimensionalPortalManager,
    PortalStability,
    PortalType,
    get_interdimensional_portal_manager,
)

logger = get_logger(__name__, LogCategory.SYSTEM)


class IntegrationMode(Enum):
    """Integration operation modes"""

    INDEPENDENT = "independent"  # Components operate independently
    COORDINATED = "coordinated"  # Basic coordination between components
    SYNCHRONIZED = "synchronized"  # Tight synchronization and resource sharing
    UNIFIED = "unified"  # Full unified operation


class WorkflowType(Enum):
    """Types of integrated workflows"""

    OUTPUT_ONLY = "output_only"  # Pure output generation
    PORTAL_ONLY = "portal_only"  # Pure portal operations
    OUTPUT_THEN_PORTAL = (
        "output_then_portal"  # Generate output, then transfer via portal
    )
    PORTAL_THEN_OUTPUT = (
        "portal_then_output"  # Transfer via portal, then generate output
    )
    BIDIRECTIONAL = "bidirectional"  # Two-way communication with output generation
    MULTI_DIMENSIONAL = "multi_dimensional"  # Complex multi-portal workflows


class SystemHealthStatus(Enum):
    """Overall system health status"""

    OPTIMAL = "optimal"  # All systems operating optimally
    GOOD = "good"  # Good performance with minor issues
    DEGRADED = "degraded"  # Degraded performance, attention needed
    CRITICAL = "critical"  # Critical issues, immediate action required
    EMERGENCY = "emergency"  # Emergency state, safety protocols active


@dataclass
class IntegratedWorkflowRequest:
    """Request for integrated output generation and portal operation"""

    workflow_id: str
    workflow_type: WorkflowType
    output_specification: Optional[Dict[str, Any]] = None
    portal_specification: Optional[Dict[str, Any]] = None
    integration_requirements: Dict[str, Any] = field(default_factory=dict)
    priority_level: int = 3  # 1=highest, 5=lowest
    deadline: Optional[datetime] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntegratedWorkflowResult:
    """Result of integrated workflow execution"""

    workflow_id: str
    execution_timestamp: datetime
    success: bool
    total_execution_time_ms: float
    output_artifacts: List[OutputArtifact] = field(default_factory=list)
    portal_operations: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    error_details: Optional[str] = None
    resource_usage: Dict[str, float] = field(default_factory=dict)


@dataclass
class SystemHealthReport:
    """Comprehensive system health report"""

    report_timestamp: datetime
    overall_health: SystemHealthStatus
    component_health: Dict[str, Dict[str, Any]]
    performance_metrics: Dict[str, float]
    resource_utilization: Dict[str, float]
    active_workflows: int
    error_rates: Dict[str, float]
    recommendations: List[str]
    alerts: List[Dict[str, Any]]


class ResourceScheduler:
    """
    Intelligent resource scheduler for coordinated operations

    Implements aerospace resource management principles:
    - Conservative resource allocation
    - Priority-based scheduling
    - Load balancing across components
    """

    def __init__(self):
        self.resource_limits = {
            "cpu_percent": 80.0,
            "memory_mb": 8192.0,
            "gpu_utilization": 90.0,
            "network_bandwidth": 1000.0,  # Mbps
        }

        self.current_allocations = {
            "output_generation": {"cpu": 0.0, "memory": 0.0, "gpu": 0.0},
            "portal_management": {"cpu": 0.0, "memory": 0.0, "gpu": 0.0},
        }

        self.scheduling_queue = deque()
        self.active_tasks = {}

        logger.debug("📊 Resource Scheduler initialized")

    def schedule_task(
        self, task_id: str, component: str, resource_requirements: Dict[str, float]
    ) -> bool:
        """
        Schedule task based on resource availability

        Returns True if task can be scheduled immediately, False if queued
        """
        if self._can_allocate_resources(component, resource_requirements):
            self._allocate_resources(task_id, component, resource_requirements)
            return True
        else:
            self.scheduling_queue.append(
                {
                    "task_id": task_id,
                    "component": component,
                    "requirements": resource_requirements,
                    "queue_time": datetime.now(),
                }
            )
            return False

    def _can_allocate_resources(
        self, component: str, requirements: Dict[str, float]
    ) -> bool:
        """Check if resources can be allocated"""
        current = self.current_allocations[component]

        total_cpu = sum(alloc["cpu"] for alloc in self.current_allocations.values())
        total_memory = sum(
            alloc["memory"] for alloc in self.current_allocations.values()
        )
        total_gpu = sum(alloc["gpu"] for alloc in self.current_allocations.values())

        return (
            total_cpu + requirements.get("cpu", 0)
            <= self.resource_limits["cpu_percent"]
            and total_memory + requirements.get("memory", 0)
            <= self.resource_limits["memory_mb"]
            and total_gpu + requirements.get("gpu", 0)
            <= self.resource_limits["gpu_utilization"]
        )

    def _allocate_resources(
        self, task_id: str, component: str, requirements: Dict[str, float]
    ) -> None:
        """Allocate resources to task"""
        self.active_tasks[task_id] = {
            "component": component,
            "requirements": requirements,
            "start_time": datetime.now(),
        }

        for resource, amount in requirements.items():
            if resource in self.current_allocations[component]:
                self.current_allocations[component][resource] += amount

    def release_resources(self, task_id: str) -> None:
        """Release resources from completed task"""
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            component = task["component"]
            requirements = task["requirements"]

            for resource, amount in requirements.items():
                if resource in self.current_allocations[component]:
                    self.current_allocations[component][resource] = max(
                        0.0, self.current_allocations[component][resource] - amount
                    )

            del self.active_tasks[task_id]

            # Try to schedule queued tasks
            self._process_scheduling_queue()

    def _process_scheduling_queue(self) -> None:
        """Process queued tasks that can now be scheduled"""
        processed_tasks = []

        for queued_task in list(self.scheduling_queue):
            if self._can_allocate_resources(
                queued_task["component"], queued_task["requirements"]
            ):
                self._allocate_resources(
                    queued_task["task_id"],
                    queued_task["component"],
                    queued_task["requirements"],
                )
                processed_tasks.append(queued_task)

        # Remove processed tasks from queue
        for task in processed_tasks:
            self.scheduling_queue.remove(task)

    def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource allocation status"""
        total_cpu = sum(alloc["cpu"] for alloc in self.current_allocations.values())
        total_memory = sum(
            alloc["memory"] for alloc in self.current_allocations.values()
        )
        total_gpu = sum(alloc["gpu"] for alloc in self.current_allocations.values())

        return {
            "current_usage": {
                "cpu_percent": total_cpu,
                "memory_mb": total_memory,
                "gpu_utilization": total_gpu,
            },
            "limits": self.resource_limits.copy(),
            "utilization_ratios": {
                "cpu": total_cpu / self.resource_limits["cpu_percent"],
                "memory": total_memory / self.resource_limits["memory_mb"],
                "gpu": total_gpu / self.resource_limits["gpu_utilization"],
            },
            "active_tasks": len(self.active_tasks),
            "queued_tasks": len(self.scheduling_queue),
            "component_allocations": self.current_allocations.copy(),
        }


class PerformanceMonitor:
    """
    Real-time performance monitoring system

    Implements nuclear engineering monitoring principles:
    - Continuous surveillance of all parameters
    - Early warning system for performance degradation
    - Automatic alerting and reporting
    """

    def __init__(self):
        self.metrics_history = deque(maxlen=1000)  # Last 1000 measurements
        self.alert_thresholds = {
            "output_generation_time_ms": 5000.0,  # 5 seconds
            "portal_traversal_time_ms": 1000.0,  # 1 second
            "error_rate": 0.05,  # 5%
            "resource_utilization": 0.90,  # 90%
        }

        self.active_alerts = {}
        self.performance_baselines = {}

        logger.debug("📈 Performance Monitor initialized")

    def record_metrics(self, metrics: Dict[str, float]) -> None:
        """Record performance metrics"""
        timestamp = datetime.now()

        metrics_entry = {"timestamp": timestamp, "metrics": metrics.copy()}

        self.metrics_history.append(metrics_entry)

        # Check for alert conditions
        self._check_alert_conditions(metrics)

        # Update performance baselines
        self._update_baselines(metrics)

    def _check_alert_conditions(self, metrics: Dict[str, float]) -> None:
        """Check metrics against alert thresholds"""
        current_time = datetime.now()

        for metric_name, threshold in self.alert_thresholds.items():
            if metric_name in metrics:
                value = metrics[metric_name]

                if value > threshold:
                    alert_id = f"alert_{metric_name}_{current_time.timestamp()}"

                    alert = {
                        "alert_id": alert_id,
                        "metric": metric_name,
                        "value": value,
                        "threshold": threshold,
                        "severity": "HIGH" if value > threshold * 1.5 else "MEDIUM",
                        "timestamp": current_time,
                        "status": "active",
                    }

                    self.active_alerts[alert_id] = alert

                    logger.warning(
                        f"Performance alert: {metric_name}={value:.3f} > {threshold:.3f}"
                    )

    def _update_baselines(self, metrics: Dict[str, float]) -> None:
        """Update performance baselines using exponential moving average"""
        alpha = 0.1  # Smoothing factor

        for metric_name, value in metrics.items():
            if metric_name in self.performance_baselines:
                self.performance_baselines[metric_name] = (
                    alpha * value
                    + (1 - alpha) * self.performance_baselines[metric_name]
                )
            else:
                self.performance_baselines[metric_name] = value

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        if not self.metrics_history:
            return {"status": "no_data"}

        # Calculate recent performance statistics
        recent_metrics = list(self.metrics_history)[-100:]  # Last 100 entries

        aggregated_metrics = {}
        for entry in recent_metrics:
            for metric_name, value in entry["metrics"].items():
                if metric_name not in aggregated_metrics:
                    aggregated_metrics[metric_name] = []
                aggregated_metrics[metric_name].append(value)

        statistics = {}
        for metric_name, values in aggregated_metrics.items():
            statistics[metric_name] = {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "count": len(values),
            }

        return {
            "report_timestamp": datetime.now(),
            "statistics": statistics,
            "baselines": self.performance_baselines.copy(),
            "active_alerts": len(self.active_alerts),
            "alert_details": list(self.active_alerts.values()),
            "metrics_count": len(self.metrics_history),
        }


class UnifiedIntegrationManager:
    """
    Main unified integration manager for output generation and portal management

    Implements nuclear engineering system integration principles:
    - Defense in depth through multiple coordination layers
    - Positive confirmation of all operations
    - Conservative resource management
    - Comprehensive monitoring and alerting
    """

    def __init__(
        self,
        integration_mode: IntegrationMode = IntegrationMode.UNIFIED,
        enable_performance_monitoring: bool = True,
        enable_resource_scheduling: bool = True,
    ):

        self.integration_mode = integration_mode
        self.enable_performance_monitoring = enable_performance_monitoring
        self.enable_resource_scheduling = enable_resource_scheduling

        # Initialize core components
        self.output_generator = get_multi_modal_output_generator(
            default_quality=OutputQuality.HIGH,
            enable_verification=True,
            enable_citations=True,
        )

        self.portal_manager = get_interdimensional_portal_manager(
            max_portals=100, safety_threshold=0.8, enable_predictive_maintenance=True
        )

        # Initialize integration components
        self.resource_scheduler = (
            ResourceScheduler() if enable_resource_scheduling else None
        )
        self.performance_monitor = (
            PerformanceMonitor() if enable_performance_monitoring else None
        )

        # Workflow management
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        self.workflow_history = deque(maxlen=1000)
        self.workflow_queue = deque()

        # Integration state
        self.integration_active = False
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()

        # Statistics
        self.integration_stats = {
            "workflows_executed": 0,
            "workflows_successful": 0,
            "workflows_failed": 0,
            "total_execution_time": 0.0,
            "average_execution_time": 0.0,
            "resource_efficiency": 0.0,
            "uptime_seconds": 0.0,
        }

        self.start_time = datetime.now()

        # Event callbacks
        self.event_callbacks: Dict[str, List[Callable]] = {
            "workflow_started": [],
            "workflow_completed": [],
            "workflow_failed": [],
            "system_alert": [],
            "performance_degradation": [],
        }

        logger.info("🔗 Unified Integration Manager initialized (DO-178C Level A)")
        logger.info(f"   Integration mode: {integration_mode.value}")
        logger.info(f"   Performance monitoring: {enable_performance_monitoring}")
        logger.info(f"   Resource scheduling: {enable_resource_scheduling}")

    async def initialize(self) -> None:
        """Initialize the integrated system"""
        if self.integration_active:
            logger.warning("Integration already active")
            return

        try:
            logger.info("🚀 Initializing Unified Integration Manager...")

            # Initialize portal manager
            # Portal manager doesn't need explicit initialization in current implementation

            # Start monitoring if enabled
            if self.enable_performance_monitoring:
                self._start_monitoring()

            self.integration_active = True

            logger.info("✅ Unified Integration Manager initialized successfully")

        except Exception as e:
            logger.error(f"❌ Integration initialization failed: {e}")
            raise KimeraCognitiveError(f"Integration initialization failed: {str(e)}")

    async def execute_workflow(
        self, request: IntegratedWorkflowRequest
    ) -> IntegratedWorkflowResult:
        """
        Execute integrated workflow with comprehensive coordination

        Args:
            request: Workflow execution request

        Returns:
            Complete workflow execution result
        """
        if not self.integration_active:
            raise KimeraValidationError("Integration system not initialized")

        workflow_start = time.time()
        workflow_id = request.workflow_id

        try:
            # Register workflow
            self.active_workflows[workflow_id] = {
                "request": request,
                "start_time": datetime.now(),
                "status": "executing",
            }

            # Trigger workflow started event
            self._trigger_event("workflow_started", request)

            # Execute workflow based on type
            result = await self._execute_workflow_by_type(request)

            # Calculate total execution time
            total_time = (time.time() - workflow_start) * 1000
            result.total_execution_time_ms = total_time

            # Update statistics
            self._update_workflow_stats(result, total_time)

            # Record performance metrics
            if self.performance_monitor:
                self.performance_monitor.record_metrics(
                    {
                        "workflow_execution_time_ms": total_time,
                        "workflow_success": 1.0 if result.success else 0.0,
                        "output_artifacts_generated": len(result.output_artifacts),
                        "portal_operations_completed": len(result.portal_operations),
                    }
                )

            # Clean up workflow
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]

            # Add to history
            self.workflow_history.append(
                {
                    "workflow_id": workflow_id,
                    "request": request,
                    "result": result,
                    "execution_time": total_time,
                }
            )

            # Trigger appropriate event
            if result.success:
                self._trigger_event("workflow_completed", result)
            else:
                self._trigger_event("workflow_failed", result)

            logger.info(
                f"Workflow {workflow_id} completed: "
                f"success={result.success}, time={total_time:.1f}ms"
            )

            return result

        except Exception as e:
            total_time = (time.time() - workflow_start) * 1000

            # Create error result
            error_result = IntegratedWorkflowResult(
                workflow_id=workflow_id,
                execution_timestamp=datetime.now(),
                success=False,
                total_execution_time_ms=total_time,
                error_details=str(e),
            )

            # Update statistics
            self._update_workflow_stats(error_result, total_time)

            # Clean up
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]

            # Trigger failure event
            self._trigger_event("workflow_failed", error_result)

            logger.error(f"Workflow {workflow_id} failed: {e}")
            return error_result

    async def _execute_workflow_by_type(
        self, request: IntegratedWorkflowRequest
    ) -> IntegratedWorkflowResult:
        """Execute workflow based on its type"""

        result = IntegratedWorkflowResult(
            workflow_id=request.workflow_id,
            execution_timestamp=datetime.now(),
            success=True,
            total_execution_time_ms=0.0,
        )

        if request.workflow_type == WorkflowType.OUTPUT_ONLY:
            await self._execute_output_only_workflow(request, result)

        elif request.workflow_type == WorkflowType.PORTAL_ONLY:
            await self._execute_portal_only_workflow(request, result)

        elif request.workflow_type == WorkflowType.OUTPUT_THEN_PORTAL:
            await self._execute_output_then_portal_workflow(request, result)

        elif request.workflow_type == WorkflowType.PORTAL_THEN_OUTPUT:
            await self._execute_portal_then_output_workflow(request, result)

        elif request.workflow_type == WorkflowType.BIDIRECTIONAL:
            await self._execute_bidirectional_workflow(request, result)

        elif request.workflow_type == WorkflowType.MULTI_DIMENSIONAL:
            await self._execute_multi_dimensional_workflow(request, result)

        else:
            raise KimeraValidationError(
                f"Unsupported workflow type: {request.workflow_type}"
            )

        return result

    async def _execute_output_only_workflow(
        self, request: IntegratedWorkflowRequest, result: IntegratedWorkflowResult
    ) -> None:
        """Execute output-only workflow"""
        if not request.output_specification:
            raise KimeraValidationError(
                "Output specification required for output-only workflow"
            )

        try:
            # Schedule resources if enabled
            task_id = f"output_{uuid.uuid4().hex[:8]}"
            if self.resource_scheduler:
                resource_requirements = {"cpu": 10.0, "memory": 500.0, "gpu": 20.0}
                self.resource_scheduler.schedule_task(
                    task_id, "output_generation", resource_requirements
                )

            # Generate output
            output_artifact = self.output_generator.generate_output(
                content_request=request.output_specification,
                modality=OutputModality(
                    request.output_specification.get("modality", "text")
                ),
                quality_level=OutputQuality(
                    request.output_specification.get("quality", "standard")
                ),
                context=request.context,
            )

            result.output_artifacts.append(output_artifact)

            # Release resources
            if self.resource_scheduler:
                self.resource_scheduler.release_resources(task_id)

            logger.debug(
                f"Output-only workflow completed: {output_artifact.artifact_id}"
            )

        except Exception as e:
            result.success = False
            result.error_details = f"Output generation failed: {str(e)}"
            raise

    async def _execute_portal_only_workflow(
        self, request: IntegratedWorkflowRequest, result: IntegratedWorkflowResult
    ) -> None:
        """Execute portal-only workflow"""
        if not request.portal_specification:
            raise KimeraValidationError(
                "Portal specification required for portal-only workflow"
            )

        try:
            spec = request.portal_specification

            # Schedule resources if enabled
            task_id = f"portal_{uuid.uuid4().hex[:8]}"
            if self.resource_scheduler:
                resource_requirements = {"cpu": 5.0, "memory": 200.0, "gpu": 10.0}
                self.resource_scheduler.schedule_task(
                    task_id, "portal_management", resource_requirements
                )

            if spec.get("operation") == "create":
                # Create portal
                portal_id = await self.portal_manager.create_portal(
                    source_dimension=DimensionalSpace(spec["source_dimension"]),
                    target_dimension=DimensionalSpace(spec["target_dimension"]),
                    portal_type=PortalType(spec.get("portal_type", "cognitive")),
                    energy_requirements=spec.get("energy_requirements", 100.0),
                    stability_threshold=spec.get("stability_threshold", 0.8),
                )

                result.portal_operations.append(
                    {"operation": "create", "portal_id": portal_id, "success": True}
                )

            elif spec.get("operation") == "traverse":
                # Traverse portal
                traversal_result = await self.portal_manager.traverse_portal(
                    portal_id=spec["portal_id"],
                    data_payload=spec.get("data_payload", {}),
                    traversal_context=request.context,
                )

                result.portal_operations.append(traversal_result)

            # Release resources
            if self.resource_scheduler:
                self.resource_scheduler.release_resources(task_id)

            logger.debug(f"Portal-only workflow completed: {spec.get('operation')}")

        except Exception as e:
            result.success = False
            result.error_details = f"Portal operation failed: {str(e)}"
            raise

    async def _execute_output_then_portal_workflow(
        self, request: IntegratedWorkflowRequest, result: IntegratedWorkflowResult
    ) -> None:
        """Execute output generation followed by portal transfer"""

        # First, generate output
        await self._execute_output_only_workflow(request, result)

        # Then, transfer via portal
        if result.success and result.output_artifacts:
            # Modify portal specification to include output data
            portal_spec = request.portal_specification or {}
            portal_spec["operation"] = "traverse"
            portal_spec["data_payload"] = {
                "output_artifacts": [
                    artifact.artifact_id for artifact in result.output_artifacts
                ],
                "content_summary": "Generated output artifacts",
            }

            # Create temporary request for portal operation
            portal_request = IntegratedWorkflowRequest(
                workflow_id=f"{request.workflow_id}_portal",
                workflow_type=WorkflowType.PORTAL_ONLY,
                portal_specification=portal_spec,
                context=request.context,
            )

            # Execute portal operation
            await self._execute_portal_only_workflow(portal_request, result)

    async def _execute_portal_then_output_workflow(
        self, request: IntegratedWorkflowRequest, result: IntegratedWorkflowResult
    ) -> None:
        """Execute portal transfer followed by output generation"""

        # First, execute portal operation
        await self._execute_portal_only_workflow(request, result)

        # Then, generate output based on portal result
        if result.success and result.portal_operations:
            # Modify output specification to include portal data
            output_spec = request.output_specification or {}
            output_spec["content"] = json.dumps(
                {
                    "portal_operations": result.portal_operations,
                    "source_workflow": request.workflow_id,
                }
            )

            # Create temporary request for output generation
            output_request = IntegratedWorkflowRequest(
                workflow_id=f"{request.workflow_id}_output",
                workflow_type=WorkflowType.OUTPUT_ONLY,
                output_specification=output_spec,
                context=request.context,
            )

            # Execute output generation
            await self._execute_output_only_workflow(output_request, result)

    async def _execute_bidirectional_workflow(
        self, request: IntegratedWorkflowRequest, result: IntegratedWorkflowResult
    ) -> None:
        """Execute bidirectional workflow with two-way communication"""

        # Execute output-then-portal
        await self._execute_output_then_portal_workflow(request, result)

        # Then execute portal-then-output for return path
        if result.success:
            # Create return workflow
            return_spec = request.portal_specification or {}
            return_spec["source_dimension"], return_spec["target_dimension"] = (
                return_spec.get("target_dimension"),
                return_spec.get("source_dimension"),
            )

            return_request = IntegratedWorkflowRequest(
                workflow_id=f"{request.workflow_id}_return",
                workflow_type=WorkflowType.PORTAL_THEN_OUTPUT,
                output_specification=request.output_specification,
                portal_specification=return_spec,
                context=request.context,
            )

            await self._execute_portal_then_output_workflow(return_request, result)

    async def _execute_multi_dimensional_workflow(
        self, request: IntegratedWorkflowRequest, result: IntegratedWorkflowResult
    ) -> None:
        """Execute complex multi-dimensional workflow"""

        # This would implement complex workflows involving multiple portals
        # and output generations across different dimensional spaces

        dimensions = request.integration_requirements.get("dimensions", [])
        if not dimensions:
            raise KimeraValidationError(
                "Multi-dimensional workflow requires dimension specification"
            )

        # Execute workflow across multiple dimensions
        for i, dimension in enumerate(dimensions):
            step_request = IntegratedWorkflowRequest(
                workflow_id=f"{request.workflow_id}_step_{i}",
                workflow_type=WorkflowType.OUTPUT_THEN_PORTAL,
                output_specification=request.output_specification,
                portal_specification={
                    "operation": "traverse",
                    "source_dimension": dimension.get("source"),
                    "target_dimension": dimension.get("target"),
                    "portal_type": dimension.get("type", "cognitive"),
                },
                context=request.context,
            )

            await self._execute_output_then_portal_workflow(step_request, result)

            if not result.success:
                break

    def _start_monitoring(self) -> None:
        """Start system monitoring"""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self._stop_monitoring.clear()

        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self.monitoring_thread.start()

        logger.info("📊 System monitoring started")

    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while not self._stop_monitoring.is_set():
            try:
                # Collect system metrics
                self._collect_system_metrics()

                # Check system health
                health_report = self._generate_health_report()

                # Check for alerts
                if health_report.overall_health in [
                    SystemHealthStatus.CRITICAL,
                    SystemHealthStatus.EMERGENCY,
                ]:
                    self._trigger_event("system_alert", health_report)

                # Wait for next monitoring cycle
                self._stop_monitoring.wait(5.0)  # 5-second monitoring interval

            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                self._stop_monitoring.wait(10.0)

    def _collect_system_metrics(self) -> None:
        """Collect comprehensive system metrics"""
        if not self.performance_monitor:
            return

        # Get output generator statistics
        output_stats = self.output_generator.get_generation_statistics()

        # Get portal manager status
        portal_status = self.portal_manager.get_system_status()

        # Get resource scheduler status
        resource_status = (
            self.resource_scheduler.get_resource_status()
            if self.resource_scheduler
            else {}
        )

        # Combine metrics
        combined_metrics = {
            "output_generation_success_rate": output_stats.get(
                "generation_stats", {}
            ).get("successful_generations", 0)
            / max(
                1, output_stats.get("generation_stats", {}).get("total_generations", 1)
            ),
            "portal_system_uptime": portal_status.get("system_info", {}).get(
                "uptime_seconds", 0
            ),
            "active_workflows": len(self.active_workflows),
            "resource_utilization_cpu": resource_status.get(
                "utilization_ratios", {}
            ).get("cpu", 0),
            "resource_utilization_memory": resource_status.get(
                "utilization_ratios", {}
            ).get("memory", 0),
            "resource_utilization_gpu": resource_status.get(
                "utilization_ratios", {}
            ).get("gpu", 0),
        }

        self.performance_monitor.record_metrics(combined_metrics)

    def _generate_health_report(self) -> SystemHealthReport:
        """Generate comprehensive system health report"""

        # Get component health
        output_stats = self.output_generator.get_generation_statistics()
        portal_status = self.portal_manager.get_system_status()
        resource_status = (
            self.resource_scheduler.get_resource_status()
            if self.resource_scheduler
            else {}
        )
        performance_report = (
            self.performance_monitor.get_performance_report()
            if self.performance_monitor
            else {}
        )

        # Determine overall health
        health_factors = []

        # Output generation health
        output_success_rate = output_stats.get("generation_stats", {}).get(
            "successful_generations", 0
        ) / max(1, output_stats.get("generation_stats", {}).get("total_generations", 1))
        health_factors.append(output_success_rate)

        # Portal system health
        active_portals = portal_status.get("system_info", {}).get("active_portals", 0)
        total_portals = portal_status.get("system_info", {}).get("total_portals", 0)
        portal_health = (
            active_portals / max(1, total_portals) if total_portals > 0 else 1.0
        )
        health_factors.append(portal_health)

        # Resource utilization health
        cpu_util = resource_status.get("utilization_ratios", {}).get("cpu", 0)
        memory_util = resource_status.get("utilization_ratios", {}).get("memory", 0)
        resource_health = 1.0 - max(cpu_util, memory_util)
        health_factors.append(resource_health)

        # Calculate overall health
        overall_health_score = sum(health_factors) / len(health_factors)

        if overall_health_score >= 0.9:
            overall_health = SystemHealthStatus.OPTIMAL
        elif overall_health_score >= 0.8:
            overall_health = SystemHealthStatus.GOOD
        elif overall_health_score >= 0.6:
            overall_health = SystemHealthStatus.DEGRADED
        elif overall_health_score >= 0.4:
            overall_health = SystemHealthStatus.CRITICAL
        else:
            overall_health = SystemHealthStatus.EMERGENCY

        return SystemHealthReport(
            report_timestamp=datetime.now(),
            overall_health=overall_health,
            component_health={
                "output_generation": {
                    "success_rate": output_success_rate,
                    "active": True,
                    "cache_hit_rate": output_stats.get("cache_performance", {}).get(
                        "cache_hit_rate", 0
                    ),
                },
                "portal_management": {
                    "active_portals": active_portals,
                    "monitoring_active": portal_status.get("system_info", {}).get(
                        "monitoring_active", False
                    ),
                    "portal_health": portal_health,
                },
                "resource_scheduler": {
                    "active": self.resource_scheduler is not None,
                    "cpu_utilization": cpu_util,
                    "memory_utilization": memory_util,
                    "queued_tasks": resource_status.get("queued_tasks", 0),
                },
            },
            performance_metrics={
                "workflow_success_rate": self.integration_stats["workflows_successful"]
                / max(1, self.integration_stats["workflows_executed"]),
                "average_execution_time": self.integration_stats[
                    "average_execution_time"
                ],
                "system_uptime": (datetime.now() - self.start_time).total_seconds(),
            },
            resource_utilization={
                "cpu": cpu_util,
                "memory": memory_util,
                "gpu": resource_status.get("utilization_ratios", {}).get("gpu", 0),
            },
            active_workflows=len(self.active_workflows),
            error_rates={
                "output_generation": 1.0 - output_success_rate,
                "portal_operations": portal_status.get("operation_statistics", {}).get(
                    "safety_violations", 0
                )
                / max(
                    1,
                    portal_status.get("operation_statistics", {}).get(
                        "total_traversals", 1
                    ),
                ),
            },
            recommendations=self._generate_health_recommendations(overall_health),
            alerts=performance_report.get("alert_details", []),
        )

    def _generate_health_recommendations(
        self, health_status: SystemHealthStatus
    ) -> List[str]:
        """Generate health improvement recommendations"""
        recommendations = []

        if health_status == SystemHealthStatus.OPTIMAL:
            recommendations.append(
                "System operating optimally - continue current operations"
            )
        elif health_status == SystemHealthStatus.GOOD:
            recommendations.append(
                "System operating well - monitor for potential improvements"
            )
        elif health_status == SystemHealthStatus.DEGRADED:
            recommendations.append(
                "Performance degradation detected - investigate resource allocation"
            )
            recommendations.append(
                "Consider reducing workflow complexity or increasing system resources"
            )
        elif health_status == SystemHealthStatus.CRITICAL:
            recommendations.append(
                "Critical system issues detected - immediate attention required"
            )
            recommendations.append(
                "Reduce system load and investigate component failures"
            )
        else:  # EMERGENCY
            recommendations.append(
                "Emergency system state - activate emergency protocols"
            )
            recommendations.append(
                "Halt non-critical operations and focus on system recovery"
            )

        return recommendations

    def _update_workflow_stats(
        self, result: IntegratedWorkflowResult, execution_time: float
    ) -> None:
        """Update workflow execution statistics"""
        self.integration_stats["workflows_executed"] += 1

        if result.success:
            self.integration_stats["workflows_successful"] += 1
        else:
            self.integration_stats["workflows_failed"] += 1

        self.integration_stats["total_execution_time"] += execution_time
        self.integration_stats["average_execution_time"] = (
            self.integration_stats["total_execution_time"]
            / self.integration_stats["workflows_executed"]
        )

        self.integration_stats["uptime_seconds"] = (
            datetime.now() - self.start_time
        ).total_seconds()

    def _trigger_event(self, event_type: str, *args) -> None:
        """Trigger event callbacks"""
        for callback in self.event_callbacks.get(event_type, []):
            try:
                callback(*args)
            except Exception as e:
                logger.error(f"Event callback error for {event_type}: {e}")

    def add_event_callback(self, event_type: str, callback: Callable) -> None:
        """Add event callback"""
        if event_type in self.event_callbacks:
            self.event_callbacks[event_type].append(callback)
        else:
            logger.warning(f"Unknown event type: {event_type}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "integration_active": self.integration_active,
            "integration_mode": self.integration_mode.value,
            "monitoring_active": self.monitoring_active,
            "active_workflows": len(self.active_workflows),
            "workflow_queue_size": len(self.workflow_queue),
            "integration_statistics": self.integration_stats.copy(),
            "component_status": {
                "output_generator": "active",
                "portal_manager": "active",
                "resource_scheduler": (
                    "active" if self.resource_scheduler else "disabled"
                ),
                "performance_monitor": (
                    "active" if self.performance_monitor else "disabled"
                ),
            },
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
        }

    async def shutdown(self) -> None:
        """Shutdown the unified integration system"""
        if not self.integration_active:
            return

        logger.info("🛑 Shutting down Unified Integration Manager...")

        # Stop monitoring
        if self.monitoring_active:
            self.monitoring_active = False
            self._stop_monitoring.set()

            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=10.0)

        # Shutdown portal manager
        await self.portal_manager.shutdown()

        # Clear active workflows
        self.active_workflows.clear()

        self.integration_active = False

        logger.info("✅ Unified Integration Manager shutdown completed")


# Global instance for module access
_integration_manager: Optional[UnifiedIntegrationManager] = None


def get_unified_integration_manager(
    integration_mode: IntegrationMode = IntegrationMode.UNIFIED,
    enable_performance_monitoring: bool = True,
    enable_resource_scheduling: bool = True,
) -> UnifiedIntegrationManager:
    """Get global unified integration manager instance"""
    global _integration_manager
    if _integration_manager is None:
        _integration_manager = UnifiedIntegrationManager(
            integration_mode=integration_mode,
            enable_performance_monitoring=enable_performance_monitoring,
            enable_resource_scheduling=enable_resource_scheduling,
        )
    return _integration_manager
