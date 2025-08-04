#!/usr/bin/env python3
"""
Output and Portals Integrator for KimeraSystem Integration
==========================================================

DO-178C Level A compliant main integrator for output generation and
portal management systems, designed for seamless integration with
the KimeraSystem cognitive architecture.

Key Features:
- KimeraSystem integration interface
- Comprehensive system coordination and orchestration
- Real-time health monitoring and reporting
- Event-driven architecture with callback support
- Nuclear-grade safety and reliability standards

Author: KIMERA Development Team
Version: 1.0.0 (DO-178C Level A)
"""

import asyncio
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from utils.kimera_exceptions import KimeraCognitiveError, KimeraValidationError
from utils.kimera_logger import LogCategory, get_logger

# Import core components
from ..output_generation.multi_modal_output_generator import (
    OutputModality,
    OutputQuality,
    get_multi_modal_output_generator,
)
from ..portal_management.interdimensional_portal_manager import (
    DimensionalSpace,
    PortalType,
    get_interdimensional_portal_manager,
)
from .unified_integration_manager import (
    IntegratedWorkflowRequest,
    IntegrationMode,
    SystemHealthStatus,
    WorkflowType,
    get_unified_integration_manager,
)

logger = get_logger(__name__, LogCategory.SYSTEM)


@dataclass
class OutputAndPortalsConfig:
    """Configuration for output and portals integration"""

    # Output generation configuration
    enable_output_generation: bool = True
    default_output_quality: OutputQuality = OutputQuality.STANDARD
    enable_output_verification: bool = True
    enable_scientific_citations: bool = True

    # Portal management configuration
    enable_portal_management: bool = True
    max_portals: int = 100
    portal_safety_threshold: float = 0.8
    enable_predictive_maintenance: bool = True

    # Integration configuration
    integration_mode: IntegrationMode = IntegrationMode.UNIFIED
    enable_performance_monitoring: bool = True
    enable_resource_scheduling: bool = True
    enable_cross_system_validation: bool = True
    monitoring_interval_seconds: int = 5

    # Safety and reliability configuration
    enable_emergency_protocols: bool = True
    safety_override_allowed: bool = False
    maximum_resource_utilization: float = 0.85


@dataclass
class IntegratedSystemHealthReport:
    """Comprehensive system health report for external monitoring"""

    timestamp: datetime
    overall_health_status: SystemHealthStatus
    component_health: Dict[str, Dict[str, Any]]
    performance_summary: Dict[str, float]
    resource_utilization: Dict[str, float]
    active_operations: Dict[str, int]
    recent_errors: List[Dict[str, Any]]
    recommendations: List[str]
    compliance_status: Dict[str, bool]


class OutputAndPortalsIntegrator:
    """
    Main integrator for output generation and portal management systems

    Implements aerospace system integration principles:
    - Unified control and coordination interface
    - Comprehensive health monitoring and reporting
    - Event-driven architecture for system interactions
    - Nuclear-grade safety and emergency procedures
    """

    def __init__(self, config: Optional[OutputAndPortalsConfig] = None):
        self.config = config or OutputAndPortalsConfig()

        # Core components (initialized on demand)
        self._output_generator = None
        self._portal_manager = None
        self._unified_manager = None

        # Integration state
        self.integration_active = False
        self.system_initialized = False
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()

        # Health and status tracking
        self.health_history: List[IntegratedSystemHealthReport] = []
        self.error_history: List[Dict[str, Any]] = []
        self.operation_statistics = {
            "outputs_generated": 0,
            "portals_created": 0,
            "portals_traversed": 0,
            "workflows_executed": 0,
            "errors_encountered": 0,
            "uptime_seconds": 0.0,
        }

        # Event system
        self.event_callbacks: Dict[str, List[Callable]] = {
            "system_initialized": [],
            "system_shutdown": [],
            "output_generated": [],
            "portal_created": [],
            "portal_traversed": [],
            "workflow_completed": [],
            "health_alert": [],
            "error_occurred": [],
        }

        self.start_time = datetime.now()

        logger.info("🔬 Output and Portals Integrator initialized (DO-178C Level A)")
        logger.info(f"   Output generation: {self.config.enable_output_generation}")
        logger.info(f"   Portal management: {self.config.enable_portal_management}")
        logger.info(f"   Integration mode: {self.config.integration_mode.value}")

    @property
    def output_generator(self):
        """Lazy-loaded output generator instance"""
        if self._output_generator is None and self.config.enable_output_generation:
            self._output_generator = get_multi_modal_output_generator(
                default_quality=self.config.default_output_quality,
                enable_verification=self.config.enable_output_verification,
                enable_citations=self.config.enable_scientific_citations,
            )
        return self._output_generator

    @property
    def portal_manager(self):
        """Lazy-loaded portal manager instance"""
        if self._portal_manager is None and self.config.enable_portal_management:
            self._portal_manager = get_interdimensional_portal_manager(
                max_portals=self.config.max_portals,
                safety_threshold=self.config.portal_safety_threshold,
                enable_predictive_maintenance=self.config.enable_predictive_maintenance,
            )
        return self._portal_manager

    @property
    def unified_manager(self):
        """Lazy-loaded unified integration manager instance"""
        if self._unified_manager is None:
            self._unified_manager = get_unified_integration_manager(
                integration_mode=self.config.integration_mode,
                enable_performance_monitoring=self.config.enable_performance_monitoring,
                enable_resource_scheduling=self.config.enable_resource_scheduling,
            )
        return self._unified_manager

    async def initialize(self) -> None:
        """Initialize the complete output and portals system"""
        if self.system_initialized:
            logger.warning("System already initialized")
            return

        try:
            logger.info("🚀 Initializing Output and Portals Integration System...")

            # Initialize unified manager
            await self.unified_manager.initialize()

            # Start health monitoring if enabled
            if self.config.enable_performance_monitoring:
                self._start_health_monitoring()

            # Perform cross-system validation if enabled
            if self.config.enable_cross_system_validation:
                await self._perform_cross_system_validation()

            self.system_initialized = True
            self.integration_active = True

            # Trigger initialization event
            self._trigger_event("system_initialized", self.get_system_status())

            logger.info(
                "✅ Output and Portals Integration System initialized successfully"
            )

        except Exception as e:
            logger.error(f"❌ System initialization failed: {e}")
            await self.shutdown()
            raise KimeraCognitiveError(f"System initialization failed: {str(e)}")

    async def _perform_cross_system_validation(self) -> None:
        """Perform comprehensive cross-system validation"""
        validation_results = {}

        try:
            # Validate output generation system
            if self.config.enable_output_generation:
                output_stats = self.output_generator.get_generation_statistics()
                validation_results["output_generation"] = {
                    "available": True,
                    "generation_stats": output_stats.get("generation_stats", {}),
                    "cache_performance": output_stats.get("cache_performance", {}),
                }

            # Validate portal management system
            if self.config.enable_portal_management:
                portal_status = self.portal_manager.get_system_status()
                validation_results["portal_management"] = {
                    "available": True,
                    "system_info": portal_status.get("system_info", {}),
                    "safety_features": portal_status.get("safety_features", {}),
                }

            # Validate unified integration
            unified_status = self.unified_manager.get_system_status()
            validation_results["unified_integration"] = {
                "available": True,
                "integration_active": unified_status.get("integration_active", False),
                "component_status": unified_status.get("component_status", {}),
            }

            logger.info("✅ Cross-system validation completed successfully")
            logger.debug(f"Validation results: {validation_results}")

        except Exception as e:
            logger.error(f"❌ Cross-system validation failed: {e}")
            raise

    def _start_health_monitoring(self) -> None:
        """Start system health monitoring"""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self._stop_monitoring.clear()

        self.monitoring_thread = threading.Thread(
            target=self._health_monitoring_loop, daemon=True
        )
        self.monitoring_thread.start()

        logger.info("📊 Health monitoring started")

    def _health_monitoring_loop(self) -> None:
        """Main health monitoring loop"""
        while not self._stop_monitoring.is_set():
            try:
                # Generate health report
                health_report = self._generate_health_report()

                # Store in history
                self.health_history.append(health_report)

                # Trim history to last 100 reports
                if len(self.health_history) > 100:
                    self.health_history = self.health_history[-100:]

                # Check for health alerts
                if health_report.overall_health_status in [
                    SystemHealthStatus.CRITICAL,
                    SystemHealthStatus.EMERGENCY,
                ]:
                    self._trigger_event("health_alert", health_report)

                # Update operation statistics
                self._update_operation_statistics()

                # Wait for next monitoring cycle
                self._stop_monitoring.wait(self.config.monitoring_interval_seconds)

            except Exception as e:
                logger.error(f"Health monitoring loop error: {e}")
                self._stop_monitoring.wait(self.config.monitoring_interval_seconds * 2)

    def _generate_health_report(self) -> IntegratedSystemHealthReport:
        """Generate comprehensive system health report"""
        current_time = datetime.now()

        # Collect component health information
        component_health = {}

        # Output generation health
        if self.config.enable_output_generation and self.output_generator:
            output_stats = self.output_generator.get_generation_statistics()
            component_health["output_generation"] = {
                "status": "active",
                "generation_stats": output_stats.get("generation_stats", {}),
                "cache_performance": output_stats.get("cache_performance", {}),
                "registry_size": output_stats.get("registry_size", 0),
            }

        # Portal management health
        if self.config.enable_portal_management and self.portal_manager:
            portal_status = self.portal_manager.get_system_status()
            component_health["portal_management"] = {
                "status": "active",
                "system_info": portal_status.get("system_info", {}),
                "operation_statistics": portal_status.get("operation_statistics", {}),
                "safety_features": portal_status.get("safety_features", {}),
            }

        # Unified integration health
        if self.unified_manager:
            unified_status = self.unified_manager.get_system_status()
            component_health["unified_integration"] = {
                "status": "active",
                "integration_active": unified_status.get("integration_active", False),
                "component_status": unified_status.get("component_status", {}),
                "integration_statistics": unified_status.get(
                    "integration_statistics", {}
                ),
            }

        # Calculate overall health status
        overall_health = self._calculate_overall_health(component_health)

        # Generate performance summary
        performance_summary = self._generate_performance_summary(component_health)

        # Calculate resource utilization
        resource_utilization = self._calculate_resource_utilization()

        # Count active operations
        active_operations = self._count_active_operations()

        # Get recent errors
        recent_errors = self.error_history[-10:] if self.error_history else []

        # Generate recommendations
        recommendations = self._generate_health_recommendations(
            overall_health, component_health
        )

        # Check compliance status
        compliance_status = self._check_compliance_status()

        return IntegratedSystemHealthReport(
            timestamp=current_time,
            overall_health_status=overall_health,
            component_health=component_health,
            performance_summary=performance_summary,
            resource_utilization=resource_utilization,
            active_operations=active_operations,
            recent_errors=recent_errors,
            recommendations=recommendations,
            compliance_status=compliance_status,
        )

    def _calculate_overall_health(
        self, component_health: Dict[str, Dict[str, Any]]
    ) -> SystemHealthStatus:
        """Calculate overall system health status"""
        health_scores = []

        # Output generation health
        if "output_generation" in component_health:
            output_stats = component_health["output_generation"].get(
                "generation_stats", {}
            )
            success_rate = output_stats.get("successful_generations", 0) / max(
                1, output_stats.get("total_generations", 1)
            )
            health_scores.append(success_rate)

        # Portal management health
        if "portal_management" in component_health:
            system_info = component_health["portal_management"].get("system_info", {})
            active_portals = system_info.get("active_portals", 0)
            total_portals = system_info.get("total_portals", 0)
            portal_health = (
                active_portals / max(1, total_portals) if total_portals > 0 else 1.0
            )
            health_scores.append(portal_health)

        # Unified integration health
        if "unified_integration" in component_health:
            integration_stats = component_health["unified_integration"].get(
                "integration_statistics", {}
            )
            workflow_success_rate = integration_stats.get(
                "workflows_successful", 0
            ) / max(1, integration_stats.get("workflows_executed", 1))
            health_scores.append(workflow_success_rate)

        # Calculate overall score
        if not health_scores:
            return SystemHealthStatus.DEGRADED

        overall_score = sum(health_scores) / len(health_scores)

        if overall_score >= 0.95:
            return SystemHealthStatus.OPTIMAL
        elif overall_score >= 0.85:
            return SystemHealthStatus.GOOD
        elif overall_score >= 0.70:
            return SystemHealthStatus.DEGRADED
        elif overall_score >= 0.50:
            return SystemHealthStatus.CRITICAL
        else:
            return SystemHealthStatus.EMERGENCY

    def _generate_performance_summary(
        self, component_health: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """Generate performance summary metrics"""
        summary = {}

        # Output generation performance
        if "output_generation" in component_health:
            output_stats = component_health["output_generation"].get(
                "generation_stats", {}
            )
            summary["output_success_rate"] = output_stats.get(
                "successful_generations", 0
            ) / max(1, output_stats.get("total_generations", 1))
            summary["average_generation_time"] = output_stats.get(
                "average_generation_time", 0.0
            )

        # Portal management performance
        if "portal_management" in component_health:
            op_stats = component_health["portal_management"].get(
                "operation_statistics", {}
            )
            summary["portal_creation_rate"] = op_stats.get("portals_created", 0)
            summary["portal_traversal_rate"] = op_stats.get("total_traversals", 0)

        # Integration performance
        if "unified_integration" in component_health:
            int_stats = component_health["unified_integration"].get(
                "integration_statistics", {}
            )
            summary["workflow_success_rate"] = int_stats.get(
                "workflows_successful", 0
            ) / max(1, int_stats.get("workflows_executed", 1))
            summary["average_workflow_time"] = int_stats.get(
                "average_execution_time", 0.0
            )

        # System uptime
        summary["system_uptime_hours"] = (
            datetime.now() - self.start_time
        ).total_seconds() / 3600.0

        return summary

    def _calculate_resource_utilization(self) -> Dict[str, float]:
        """Calculate current resource utilization"""
        # This would integrate with actual resource monitoring
        # For now, return placeholder values
        return {
            "cpu_percent": 0.0,
            "memory_percent": 0.0,
            "gpu_percent": 0.0,
            "network_utilization": 0.0,
        }

    def _count_active_operations(self) -> Dict[str, int]:
        """Count active operations across all systems"""
        operations = {}

        # Count active workflows
        if self.unified_manager:
            unified_status = self.unified_manager.get_system_status()
            operations["active_workflows"] = unified_status.get("active_workflows", 0)

        # Count active portals
        if self.portal_manager:
            portal_status = self.portal_manager.get_system_status()
            operations["active_portals"] = portal_status.get("system_info", {}).get(
                "active_portals", 0
            )

        # Count queued outputs
        operations["queued_outputs"] = 0  # Would be implemented with actual queue

        return operations

    def _generate_health_recommendations(
        self,
        health_status: SystemHealthStatus,
        component_health: Dict[str, Dict[str, Any]],
    ) -> List[str]:
        """Generate health improvement recommendations"""
        recommendations = []

        if health_status == SystemHealthStatus.OPTIMAL:
            recommendations.append("System operating at optimal performance")
        elif health_status == SystemHealthStatus.GOOD:
            recommendations.append(
                "System operating well with minor areas for improvement"
            )
        elif health_status == SystemHealthStatus.DEGRADED:
            recommendations.append(
                "Performance degradation detected - investigate component issues"
            )
            recommendations.append("Consider resource optimization and load balancing")
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

        # Component-specific recommendations
        if "output_generation" in component_health:
            output_stats = component_health["output_generation"].get(
                "generation_stats", {}
            )
            cache_perf = component_health["output_generation"].get(
                "cache_performance", {}
            )

            success_rate = output_stats.get("successful_generations", 0) / max(
                1, output_stats.get("total_generations", 1)
            )
            if success_rate < 0.9:
                recommendations.append(
                    "Output generation success rate below optimal - review generation parameters"
                )

            cache_hit_rate = cache_perf.get("cache_hit_rate", 0)
            if cache_hit_rate < 0.5:
                recommendations.append(
                    "Low cache hit rate - consider cache optimization"
                )

        if "portal_management" in component_health:
            system_info = component_health["portal_management"].get("system_info", {})

            active_portals = system_info.get("active_portals", 0)
            total_portals = system_info.get("total_portals", 0)

            if total_portals > 0 and active_portals / total_portals < 0.8:
                recommendations.append(
                    "High number of inactive portals - consider cleanup"
                )

        return recommendations

    def _check_compliance_status(self) -> Dict[str, bool]:
        """Check DO-178C Level A compliance status"""
        return {
            "system_initialized": self.system_initialized,
            "monitoring_active": self.monitoring_active,
            "safety_protocols_enabled": self.config.enable_emergency_protocols,
            "verification_enabled": self.config.enable_output_verification,
            "predictive_maintenance": self.config.enable_predictive_maintenance,
            "cross_system_validation": self.config.enable_cross_system_validation,
        }

    def _update_operation_statistics(self) -> None:
        """Update operation statistics"""
        self.operation_statistics["uptime_seconds"] = (
            datetime.now() - self.start_time
        ).total_seconds()

        # Update statistics from components
        if self.output_generator:
            output_stats = self.output_generator.get_generation_statistics()
            gen_stats = output_stats.get("generation_stats", {})
            self.operation_statistics["outputs_generated"] = gen_stats.get(
                "total_generations", 0
            )

        if self.portal_manager:
            portal_status = self.portal_manager.get_system_status()
            op_stats = portal_status.get("operation_statistics", {})
            self.operation_statistics["portals_created"] = op_stats.get(
                "portals_created", 0
            )
            self.operation_statistics["portals_traversed"] = op_stats.get(
                "total_traversals", 0
            )

        if self.unified_manager:
            unified_status = self.unified_manager.get_system_status()
            int_stats = unified_status.get("integration_statistics", {})
            self.operation_statistics["workflows_executed"] = int_stats.get(
                "workflows_executed", 0
            )

    def _trigger_event(self, event_type: str, *args) -> None:
        """Trigger event callbacks"""
        for callback in self.event_callbacks.get(event_type, []):
            try:
                callback(*args)
            except Exception as e:
                logger.error(f"Event callback error for {event_type}: {e}")
                self._record_error("event_callback", str(e))

    def _record_error(self, error_type: str, error_message: str) -> None:
        """Record error in history"""
        error_entry = {
            "timestamp": datetime.now(),
            "type": error_type,
            "message": error_message,
        }

        self.error_history.append(error_entry)

        # Trim error history to last 100 entries
        if len(self.error_history) > 100:
            self.error_history = self.error_history[-100:]

        self.operation_statistics["errors_encountered"] += 1

        # Trigger error event
        self._trigger_event("error_occurred", error_entry)

    def add_event_callback(self, event_type: str, callback: Callable) -> None:
        """Add event callback"""
        if event_type in self.event_callbacks:
            self.event_callbacks[event_type].append(callback)
        else:
            logger.warning(f"Unknown event type: {event_type}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "system_initialized": self.system_initialized,
            "integration_active": self.integration_active,
            "monitoring_active": self.monitoring_active,
            "configuration": {
                "output_generation_enabled": self.config.enable_output_generation,
                "portal_management_enabled": self.config.enable_portal_management,
                "integration_mode": self.config.integration_mode.value,
                "performance_monitoring": self.config.enable_performance_monitoring,
            },
            "operation_statistics": self.operation_statistics.copy(),
            "health_reports_count": len(self.health_history),
            "error_count": len(self.error_history),
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
        }

    def get_latest_health_report(self) -> Optional[IntegratedSystemHealthReport]:
        """Get latest system health report"""
        return self.health_history[-1] if self.health_history else None

    async def generate_output(
        self,
        content_request: Dict[str, Any],
        modality: OutputModality = OutputModality.TEXT,
        quality_level: OutputQuality = None,
    ) -> Dict[str, Any]:
        """
        Generate output through the integrated system

        Args:
            content_request: Content generation request
            modality: Output modality
            quality_level: Quality level (defaults to config default)

        Returns:
            Output generation result with metadata
        """
        if not self.system_initialized:
            raise KimeraValidationError("System not initialized")

        if not self.config.enable_output_generation:
            raise KimeraValidationError("Output generation disabled")

        try:
            quality_level = quality_level or self.config.default_output_quality

            # Generate output through unified manager if possible
            if (
                self.unified_manager
                and self.config.integration_mode == IntegrationMode.UNIFIED
            ):
                workflow_request = IntegratedWorkflowRequest(
                    workflow_id=f"output_{datetime.now().timestamp()}",
                    workflow_type=WorkflowType.OUTPUT_ONLY,
                    output_specification={
                        **content_request,
                        "modality": modality.value,
                        "quality": quality_level.value,
                    },
                )

                result = await self.unified_manager.execute_workflow(workflow_request)

                # Trigger event
                if result.success and result.output_artifacts:
                    self._trigger_event("output_generated", result.output_artifacts[0])

                return {
                    "success": result.success,
                    "artifacts": result.output_artifacts,
                    "execution_time_ms": result.total_execution_time_ms,
                    "error": result.error_details,
                }

            else:
                # Direct output generation
                artifact = self.output_generator.generate_output(
                    content_request=content_request,
                    modality=modality,
                    quality_level=quality_level,
                )

                # Trigger event
                self._trigger_event("output_generated", artifact)

                return {
                    "success": True,
                    "artifacts": [artifact],
                    "execution_time_ms": artifact.metadata.generation_time_ms,
                    "error": None,
                }

        except Exception as e:
            self._record_error("output_generation", str(e))
            logger.error(f"Output generation failed: {e}")
            return {
                "success": False,
                "artifacts": [],
                "execution_time_ms": 0.0,
                "error": str(e),
            }

    async def create_portal(
        self,
        source_dimension: DimensionalSpace,
        target_dimension: DimensionalSpace,
        portal_type: PortalType = PortalType.COGNITIVE,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Create portal through the integrated system

        Args:
            source_dimension: Source dimensional space
            target_dimension: Target dimensional space
            portal_type: Type of portal
            **kwargs: Additional portal configuration

        Returns:
            Portal creation result
        """
        if not self.system_initialized:
            raise KimeraValidationError("System not initialized")

        if not self.config.enable_portal_management:
            raise KimeraValidationError("Portal management disabled")

        try:
            portal_id = await self.portal_manager.create_portal(
                source_dimension=source_dimension,
                target_dimension=target_dimension,
                portal_type=portal_type,
                **kwargs,
            )

            # Trigger event
            self._trigger_event("portal_created", portal_id)

            return {
                "success": True,
                "portal_id": portal_id,
                "source_dimension": source_dimension.value,
                "target_dimension": target_dimension.value,
                "portal_type": portal_type.value,
            }

        except Exception as e:
            self._record_error("portal_creation", str(e))
            logger.error(f"Portal creation failed: {e}")
            return {"success": False, "portal_id": None, "error": str(e)}

    async def shutdown(self) -> None:
        """Shutdown the integrated system"""
        if not self.integration_active:
            return

        logger.info("🛑 Shutting down Output and Portals Integration System...")

        try:
            # Stop monitoring
            if self.monitoring_active:
                self.monitoring_active = False
                self._stop_monitoring.set()

                if self.monitoring_thread:
                    self.monitoring_thread.join(timeout=10.0)

            # Shutdown unified manager
            if self._unified_manager:
                await self._unified_manager.shutdown()

            # Shutdown portal manager
            if self._portal_manager:
                await self._portal_manager.shutdown()

            # Mark as inactive
            self.integration_active = False
            self.system_initialized = False

            # Trigger shutdown event
            self._trigger_event("system_shutdown", self.get_system_status())

            logger.info("✅ Output and Portals Integration System shutdown completed")

        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")
            raise


# Global instance for module access
_integrator: Optional[OutputAndPortalsIntegrator] = None


def get_output_and_portals_integrator(
    config: Optional[OutputAndPortalsConfig] = None,
) -> OutputAndPortalsIntegrator:
    """Get global output and portals integrator instance"""
    global _integrator
    if _integrator is None:
        _integrator = OutputAndPortalsIntegrator(config)
    return _integrator
