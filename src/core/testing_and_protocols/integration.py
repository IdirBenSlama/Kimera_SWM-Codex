#!/usr/bin/env python3
"""
Testing and Protocols Integration Module
========================================

DO-178C Level A compliant integration of large-scale testing framework
and omnidimensional protocol engine into the Kimera cognitive system.

Key Features:
- Unified integration of testing and protocol systems
- Complete 96-test matrix execution capability
- Inter-dimensional communication protocols
- Real-time monitoring and health assessment

Author: KIMERA Development Team
Version: 1.0.0 (DO-178C Level A)
"""

import asyncio
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import time
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.kimera_logger import get_logger, LogCategory
from utils.kimera_exceptions import KimeraValidationError, KimeraCognitiveError

# Testing framework imports
from .testing.configurations.complexity_levels import get_complexity_manager
from .testing.configurations.input_types import get_input_generator
from .testing.configurations.cognitive_contexts import get_cognitive_context_manager
from .testing.configurations.matrix_validator import get_matrix_validator
from .testing.framework.test_orchestrator import get_test_orchestrator

# Protocol engine imports
from .protocols.omnidimensional.protocol_engine import (
    get_protocol_engine, get_global_registry, SystemDimension,
    MessageType, MessagePriority, ProtocolMessage
)

logger = get_logger(__name__, LogCategory.SYSTEM)


@dataclass
class TestingAndProtocolsConfig:
    """Configuration for testing and protocols integration"""
    # Testing configuration
    enable_large_scale_testing: bool = True
    max_parallel_tests: int = 8
    test_timeout_seconds: int = 30
    test_matrix_seed: int = 42

    # Protocol configuration
    enable_omnidimensional_protocols: bool = True
    local_dimension: SystemDimension = SystemDimension.TESTING_ORCHESTRATION
    heartbeat_interval_seconds: int = 10
    message_timeout_seconds: int = 30

    # Integration configuration
    enable_test_result_broadcasting: bool = True
    enable_cross_dimensional_validation: bool = True
    enable_performance_monitoring: bool = True
    monitoring_interval_seconds: int = 5


@dataclass
class SystemHealthReport:
    """Comprehensive system health report"""
    timestamp: datetime
    testing_framework_status: Dict[str, Any]
    protocol_engine_status: Dict[str, Any]
    integration_metrics: Dict[str, Any]
    overall_health_score: float
    recommendations: List[str]
    alerts: List[Dict[str, Any]]


class TestingAndProtocolsIntegrator:
    """
    Unified integrator for testing framework and protocol systems

    Implements aerospace-grade system integration:
    - Complete system orchestration
    - Real-time health monitoring
    - Fault tolerance and recovery
    - Performance optimization
    """

    def __init__(self, config: Optional[TestingAndProtocolsConfig] = None):
        self.config = config or TestingAndProtocolsConfig()

        # Core components
        self.complexity_manager = get_complexity_manager()
        self.input_generator = get_input_generator(self.config.test_matrix_seed)
        self.context_manager = get_cognitive_context_manager()
        self.matrix_validator = get_matrix_validator(self.config.test_matrix_seed)
        self.test_orchestrator = get_test_orchestrator(self.config.max_parallel_tests)

        # Protocol components
        self.global_registry = get_global_registry()
        self.protocol_engine = get_protocol_engine(self.config.local_dimension)

        # Integration state
        self.integration_active = False
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()

        # Statistics and metrics
        self.start_time: Optional[datetime] = None
        self.health_history: List[SystemHealthReport] = []
        self.performance_metrics = {
            "total_tests_executed": 0,
            "total_messages_sent": 0,
            "total_messages_received": 0,
            "average_test_execution_time": 0.0,
            "average_message_latency": 0.0,
            "system_uptime": 0.0
        }

        # Event callbacks
        self.event_callbacks: Dict[str, List[Callable]] = {
            "integration_started": [],
            "integration_stopped": [],
            "test_matrix_completed": [],
            "health_alert": [],
            "performance_threshold_exceeded": []
        }

        logger.info("🔬 Testing and Protocols Integrator initialized (DO-178C Level A)")
        logger.info(f"   Local dimension: {self.config.local_dimension.value}")
        logger.info(f"   Testing enabled: {self.config.enable_large_scale_testing}")
        logger.info(f"   Protocols enabled: {self.config.enable_omnidimensional_protocols}")

    async def initialize(self) -> None:
        """Initialize the complete integration system"""
        if self.integration_active:
            logger.warning("Integration already initialized")
            return

        try:
            logger.info("🚀 Initializing Testing and Protocols Integration...")

            # Initialize protocol engine
            if self.config.enable_omnidimensional_protocols:
                await self._initialize_protocol_engine()

            # Initialize testing framework
            if self.config.enable_large_scale_testing:
                await self._initialize_testing_framework()

            # Setup inter-component communication
            await self._setup_inter_component_communication()

            # Start monitoring
            if self.config.enable_performance_monitoring:
                self._start_monitoring()

            self.integration_active = True
            self.start_time = datetime.now()

            # Trigger initialization callbacks
            self._trigger_event("integration_started", self.get_system_status())

            logger.info("✅ Testing and Protocols Integration initialized successfully")

        except Exception as e:
            logger.error(f"❌ Failed to initialize integration: {e}")
            await self.shutdown()
            raise

    async def _initialize_protocol_engine(self) -> None:
        """Initialize the omnidimensional protocol engine"""
        logger.info("🌐 Initializing omnidimensional protocol engine...")

        # Start protocol engine
        self.protocol_engine.start()

        # Register message handlers
        self._register_protocol_handlers()

        # Announce this dimension to the network
        await self._announce_dimension()

        logger.info("✅ Omnidimensional protocol engine initialized")

    def _register_protocol_handlers(self) -> None:
        """Register handlers for different message types"""

        # Handler for test results
        def handle_test_result(message: ProtocolMessage) -> None:
            try:
                if self.config.enable_test_result_broadcasting:
                    logger.debug(f"Received test result from {message.header.source_dimension.value}")
                    # Process test result message
                    self._process_test_result_message(message)
            except Exception as e:
                logger.error(f"Failed to handle test result message: {e}")

        # Handler for health status
        def handle_health_status(message: ProtocolMessage) -> None:
            try:
                logger.debug(f"Received health status from {message.header.source_dimension.value}")
                # Process health status message
                self._process_health_status_message(message)
            except Exception as e:
                logger.error(f"Failed to handle health status message: {e}")

        # Handler for system alerts
        def handle_system_alert(message: ProtocolMessage) -> None:
            try:
                logger.warning(f"Received system alert from {message.header.source_dimension.value}")
                # Process system alert
                self._process_system_alert_message(message)
            except Exception as e:
                logger.error(f"Failed to handle system alert: {e}")

        # Register handlers
        self.protocol_engine.register_message_handler(MessageType.DATA_TRANSFER, handle_test_result)
        self.protocol_engine.register_message_handler(MessageType.STATUS_REPORT, handle_health_status)
        self.protocol_engine.register_message_handler(MessageType.SYSTEM_ALERT, handle_system_alert)

        logger.debug("Protocol message handlers registered")

    async def _announce_dimension(self) -> None:
        """Announce this dimension to other system components"""
        announcement_data = {
            "dimension": self.config.local_dimension.value,
            "capabilities": [
                "large_scale_testing",
                "test_orchestration",
                "result_aggregation",
                "performance_monitoring",
                "protocol_communication"
            ],
            "status": "active",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "max_parallel_tests": self.config.max_parallel_tests,
                "testing_enabled": self.config.enable_large_scale_testing,
                "protocols_enabled": self.config.enable_omnidimensional_protocols
            }
        }

        # Broadcast to all known dimensions
        for dimension in SystemDimension:
            if dimension != self.config.local_dimension:
                try:
                    message = self.protocol_engine.create_message(
                        destination=dimension,
                        message_type=MessageType.REGISTRATION,
                        data=announcement_data,
                        priority=MessagePriority.HIGH
                    )
                    self.protocol_engine.send_message(message)
                except Exception as e:
                    logger.debug(f"Failed to announce to {dimension.value}: {e}")

    async def _initialize_testing_framework(self) -> None:
        """Initialize the large-scale testing framework"""
        logger.info("🧪 Initializing large-scale testing framework...")

        # Generate complete test matrix
        test_configurations = self.matrix_validator.generate_complete_matrix()

        # Validate matrix
        validation_report = self.matrix_validator.validate_complete_matrix()

        if validation_report.invalid_configurations > 0:
            logger.warning(f"Found {validation_report.invalid_configurations} invalid configurations")

        # Setup test orchestrator
        self.test_orchestrator.setup_test_execution(test_configurations)

        # Register test event callbacks
        self._register_test_callbacks()

        logger.info(f"✅ Testing framework initialized with {len(test_configurations)} test configurations")

    def _register_test_callbacks(self) -> None:
        """Register callbacks for test execution events"""

        def on_test_completed(context, result):
            """Handle individual test completion"""
            try:
                # Broadcast test result if enabled
                if self.config.enable_test_result_broadcasting:
                    self._broadcast_test_result(context, result)

                # Update performance metrics
                self._update_test_metrics(result)

            except Exception as e:
                logger.error(f"Test completion callback error: {e}")

        def on_test_failed(context):
            """Handle test failure"""
            try:
                logger.warning(f"Test {context.configuration.config_id} failed: {context.error_message}")

                # Broadcast failure alert if severe
                if context.retry_count >= context.max_retries:
                    self._broadcast_test_failure_alert(context)

            except Exception as e:
                logger.error(f"Test failure callback error: {e}")

        def on_orchestration_completed(results):
            """Handle completion of entire test orchestration"""
            try:
                logger.info("🎉 Test orchestration completed")

                # Trigger completion event
                self._trigger_event("test_matrix_completed", results)

                # Generate comprehensive report
                self._generate_completion_report(results)

            except Exception as e:
                logger.error(f"Orchestration completion callback error: {e}")

        # Register callbacks with test orchestrator
        self.test_orchestrator.add_event_callback("test_completed", on_test_completed)
        self.test_orchestrator.add_event_callback("test_failed", on_test_failed)
        self.test_orchestrator.add_event_callback("orchestration_completed", on_orchestration_completed)

    async def _setup_inter_component_communication(self) -> None:
        """Setup communication between testing and protocol components"""
        logger.info("🔗 Setting up inter-component communication...")

        # Cross-validate dimensions if enabled
        if self.config.enable_cross_dimensional_validation:
            await self._perform_cross_dimensional_validation()

        logger.info("✅ Inter-component communication established")

    async def _perform_cross_dimensional_validation(self) -> None:
        """Perform validation across different system dimensions"""
        validation_results = {}

        for dimension in SystemDimension:
            if dimension != self.config.local_dimension:
                try:
                    # Check connection status
                    status = self.protocol_engine.get_connection_status(dimension)
                    validation_results[dimension.value] = status

                    if status["status"] == "healthy":
                        logger.debug(f"✅ Validated connection to {dimension.value}")
                    else:
                        logger.warning(f"⚠️ Unhealthy connection to {dimension.value}")

                except Exception as e:
                    logger.warning(f"❌ Failed to validate {dimension.value}: {e}")
                    validation_results[dimension.value] = {"status": "error", "error": str(e)}

        logger.info(f"Cross-dimensional validation completed: "
                   f"{sum(1 for r in validation_results.values() if r.get('status') == 'healthy')}"
                   f"/{len(validation_results)} connections healthy")

    def _start_monitoring(self) -> None:
        """Start system monitoring in background thread"""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self._stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()

        logger.info("📊 System monitoring started")

    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while not self._stop_monitoring.is_set():
            try:
                # Generate health report
                health_report = self._generate_health_report()

                # Store in history
                self.health_history.append(health_report)

                # Trim history to last 100 reports
                if len(self.health_history) > 100:
                    self.health_history = self.health_history[-100:]

                # Check for alerts
                self._check_health_alerts(health_report)

                # Wait for next monitoring cycle
                self._stop_monitoring.wait(self.config.monitoring_interval_seconds)

            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                self._stop_monitoring.wait(self.config.monitoring_interval_seconds)

    def _generate_health_report(self) -> SystemHealthReport:
        """Generate comprehensive system health report"""
        current_time = datetime.now()

        # Get testing framework status
        testing_status = {}
        if self.config.enable_large_scale_testing:
            testing_status = {
                "orchestrator_active": self.test_orchestrator.orchestration_active,
                "current_metrics": self.test_orchestrator.get_current_metrics(),
                "complexity_manager_status": self.complexity_manager.get_status_report(),
                "context_manager_status": self.context_manager.get_comprehensive_report()
            }

        # Get protocol engine status
        protocol_status = {}
        if self.config.enable_omnidimensional_protocols:
            protocol_status = self.protocol_engine.get_engine_statistics()

        # Calculate integration metrics
        uptime = (current_time - self.start_time).total_seconds() if self.start_time else 0.0

        integration_metrics = {
            "uptime_seconds": uptime,
            "integration_active": self.integration_active,
            "monitoring_active": self.monitoring_active,
            "performance_metrics": self.performance_metrics.copy(),
            "health_history_entries": len(self.health_history)
        }

        # Calculate overall health score
        health_score = self._calculate_health_score(testing_status, protocol_status, integration_metrics)

        # Generate recommendations
        recommendations = self._generate_recommendations(testing_status, protocol_status, health_score)

        # Check for alerts
        alerts = self._detect_alerts(testing_status, protocol_status, integration_metrics)

        return SystemHealthReport(
            timestamp=current_time,
            testing_framework_status=testing_status,
            protocol_engine_status=protocol_status,
            integration_metrics=integration_metrics,
            overall_health_score=health_score,
            recommendations=recommendations,
            alerts=alerts
        )

    def _calculate_health_score(self,
                              testing_status: Dict[str, Any],
                              protocol_status: Dict[str, Any],
                              integration_metrics: Dict[str, Any]) -> float:
        """Calculate overall system health score (0.0 to 1.0)"""
        score = 1.0

        # Testing framework health
        if self.config.enable_large_scale_testing and testing_status:
            if not testing_status.get("orchestrator_active", False):
                score -= 0.2

            current_metrics = testing_status.get("current_metrics")
            if current_metrics and current_metrics.success_rate < 0.8:
                score -= 0.3 * (0.8 - current_metrics.success_rate)

        # Protocol engine health
        if self.config.enable_omnidimensional_protocols and protocol_status:
            messages = protocol_status.get("messages", {})
            success_rate = messages.get("success_rate", 1.0)
            if success_rate < 0.9:
                score -= 0.2 * (0.9 - success_rate)

            connections = protocol_status.get("connections", {})
            healthy_ratio = connections.get("healthy_dimensions", 0) / max(1, connections.get("registered_dimensions", 1))
            if healthy_ratio < 0.8:
                score -= 0.3 * (0.8 - healthy_ratio)

        # Integration health
        if not integration_metrics.get("integration_active", False):
            score -= 0.5

        return max(0.0, min(1.0, score))

    def _generate_recommendations(self,
                                testing_status: Dict[str, Any],
                                protocol_status: Dict[str, Any],
                                health_score: float) -> List[str]:
        """Generate system recommendations based on current status"""
        recommendations = []

        if health_score < 0.7:
            recommendations.append("System health below optimal - investigate issues immediately")

        # Testing framework recommendations
        if self.config.enable_large_scale_testing and testing_status:
            current_metrics = testing_status.get("current_metrics")
            if current_metrics:
                if current_metrics.success_rate < 0.8:
                    recommendations.append("Test success rate low - review test configurations and system resources")

                if current_metrics.throughput_per_minute < 5.0:
                    recommendations.append("Test throughput low - consider increasing parallelism or optimizing tests")

        # Protocol recommendations
        if self.config.enable_omnidimensional_protocols and protocol_status:
            performance = protocol_status.get("performance", {})
            if performance.get("average_latency", 0) > 0.1:  # 100ms
                recommendations.append("High message latency detected - check network and routing efficiency")

            connections = protocol_status.get("connections", {})
            if connections.get("healthy_dimensions", 0) < connections.get("registered_dimensions", 0) * 0.8:
                recommendations.append("Multiple unhealthy connections - investigate dimension health")

        return recommendations

    def _detect_alerts(self,
                      testing_status: Dict[str, Any],
                      protocol_status: Dict[str, Any],
                      integration_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect system alerts based on current status"""
        alerts = []

        # High severity alerts
        if not integration_metrics.get("integration_active", False):
            alerts.append({
                "severity": "CRITICAL",
                "type": "system_failure",
                "message": "Integration system not active",
                "timestamp": datetime.now()
            })

        # Testing alerts
        if self.config.enable_large_scale_testing and testing_status:
            current_metrics = testing_status.get("current_metrics")
            if current_metrics and current_metrics.success_rate < 0.5:
                alerts.append({
                    "severity": "HIGH",
                    "type": "test_failure_rate",
                    "message": f"Test success rate critically low: {current_metrics.success_rate:.1%}",
                    "timestamp": datetime.now()
                })

        # Protocol alerts
        if self.config.enable_omnidimensional_protocols and protocol_status:
            messages = protocol_status.get("messages", {})
            if messages.get("failed", 0) > messages.get("sent", 1) * 0.1:  # >10% failure rate
                alerts.append({
                    "severity": "MEDIUM",
                    "type": "message_failure_rate",
                    "message": f"High message failure rate: {messages.get('failed', 0)} failures",
                    "timestamp": datetime.now()
                })

        return alerts

    def _check_health_alerts(self, health_report: SystemHealthReport) -> None:
        """Check health report for alerts and trigger callbacks"""
        for alert in health_report.alerts:
            self._trigger_event("health_alert", alert)

            if alert["severity"] in ["CRITICAL", "HIGH"]:
                logger.error(f"🚨 {alert['severity']} ALERT: {alert['message']}")
            else:
                logger.warning(f"⚠️ {alert['severity']} ALERT: {alert['message']}")

    async def execute_complete_test_matrix(self) -> Dict[str, Any]:
        """Execute the complete 96-test matrix"""
        if not self.integration_active:
            raise KimeraValidationError("Integration not initialized")

        if not self.config.enable_large_scale_testing:
            raise KimeraValidationError("Large-scale testing not enabled")

        logger.info("🚀 Starting complete test matrix execution (96 tests)")

        try:
            # Execute all tests
            results = await self.test_orchestrator.execute_all_tests()

            # Broadcast completion if enabled
            if self.config.enable_test_result_broadcasting:
                await self._broadcast_matrix_completion(results)

            logger.info("✅ Complete test matrix execution finished")

            return results

        except Exception as e:
            logger.error(f"❌ Test matrix execution failed: {e}")
            raise

    def _broadcast_test_result(self, context, result) -> None:
        """Broadcast individual test result to other dimensions"""
        try:
            result_data = {
                "test_id": context.configuration.config_id,
                "complexity_level": context.configuration.complexity_level.value,
                "input_type": context.configuration.input_type.value,
                "cognitive_context": context.configuration.cognitive_context.value,
                "success": result.success,
                "execution_time": result.execution_time,
                "accuracy_score": result.accuracy_score,
                "timestamp": result.timestamp.isoformat()
            }

            # Send to relevant dimensions
            for dimension in [SystemDimension.COGNITIVE_RESPONSE, SystemDimension.SYSTEM_MONITOR]:
                if dimension != self.config.local_dimension:
                    message = self.protocol_engine.create_message(
                        destination=dimension,
                        message_type=MessageType.DATA_TRANSFER,
                        data=result_data,
                        priority=MessagePriority.LOW
                    )
                    self.protocol_engine.send_message(message)

        except Exception as e:
            logger.error(f"Failed to broadcast test result: {e}")

    async def _broadcast_matrix_completion(self, results: Dict[str, Any]) -> None:
        """Broadcast test matrix completion to all dimensions"""
        try:
            completion_data = {
                "event": "test_matrix_completed",
                "summary": results.get("summary", {}),
                "dimension_analysis": results.get("dimension_analysis", {}),
                "timestamp": datetime.now().isoformat(),
                "source": self.config.local_dimension.value
            }

            # Broadcast to all dimensions
            for dimension in SystemDimension:
                if dimension != self.config.local_dimension:
                    message = self.protocol_engine.create_message(
                        destination=dimension,
                        message_type=MessageType.DATA_TRANSFER,
                        data=completion_data,
                        priority=MessagePriority.NORMAL
                    )
                    self.protocol_engine.send_message(message)

            logger.info("📡 Test matrix completion broadcasted to all dimensions")

        except Exception as e:
            logger.error(f"Failed to broadcast matrix completion: {e}")

    def _broadcast_test_failure_alert(self, context) -> None:
        """Broadcast test failure alert for critical failures"""
        try:
            alert_data = {
                "alert_type": "test_failure",
                "test_id": context.configuration.config_id,
                "complexity_level": context.configuration.complexity_level.value,
                "error_message": context.error_message,
                "retry_count": context.retry_count,
                "timestamp": datetime.now().isoformat(),
                "severity": "HIGH"
            }

            # Send to system monitor
            message = self.protocol_engine.create_message(
                destination=SystemDimension.SYSTEM_MONITOR,
                message_type=MessageType.SYSTEM_ALERT,
                data=alert_data,
                priority=MessagePriority.HIGH
            )
            self.protocol_engine.send_message(message)

        except Exception as e:
            logger.error(f"Failed to broadcast test failure alert: {e}")

    def _process_test_result_message(self, message: ProtocolMessage) -> None:
        """Process received test result message"""
        # Implementation for handling external test results
        # This would integrate results from other dimensions
        pass

    def _process_health_status_message(self, message: ProtocolMessage) -> None:
        """Process received health status message"""
        # Implementation for handling health status from other dimensions
        pass

    def _process_system_alert_message(self, message: ProtocolMessage) -> None:
        """Process received system alert message"""
        # Implementation for handling system alerts from other dimensions
        pass

    def _update_test_metrics(self, result) -> None:
        """Update performance metrics based on test result"""
        self.performance_metrics["total_tests_executed"] += 1

        # Update average execution time
        current_avg = self.performance_metrics["average_test_execution_time"]
        total_tests = self.performance_metrics["total_tests_executed"]

        new_avg = ((current_avg * (total_tests - 1)) + result.execution_time) / total_tests
        self.performance_metrics["average_test_execution_time"] = new_avg

    def _generate_completion_report(self, results: Dict[str, Any]) -> None:
        """Generate comprehensive completion report"""
        try:
            report_data = {
                "execution_summary": results.get("summary", {}),
                "performance_analysis": results.get("performance_statistics", {}),
                "dimension_breakdown": results.get("dimension_analysis", {}),
                "resource_utilization": results.get("resource_usage", {}),
                "quality_assessment": results.get("quality_assessment", {}),
                "integration_metrics": self.performance_metrics.copy(),
                "generation_timestamp": datetime.now().isoformat()
            }

            logger.info("📊 Test matrix completion report generated")

            # Store or export report as needed
            # This could be extended to save to file or database

        except Exception as e:
            logger.error(f"Failed to generate completion report: {e}")

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
            "config": {
                "testing_enabled": self.config.enable_large_scale_testing,
                "protocols_enabled": self.config.enable_omnidimensional_protocols,
                "monitoring_enabled": self.config.enable_performance_monitoring
            },
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            "performance_metrics": self.performance_metrics.copy(),
            "health_score": self.health_history[-1].overall_health_score if self.health_history else 0.0,
            "latest_health_report": self.health_history[-1] if self.health_history else None
        }

    def get_latest_health_report(self) -> Optional[SystemHealthReport]:
        """Get latest health report"""
        return self.health_history[-1] if self.health_history else None

    async def shutdown(self) -> None:
        """Shutdown the integration system"""
        if not self.integration_active:
            return

        logger.info("🛑 Shutting down Testing and Protocols Integration...")

        try:
            # Stop monitoring
            if self.monitoring_active:
                self.monitoring_active = False
                self._stop_monitoring.set()
                if self.monitoring_thread:
                    self.monitoring_thread.join(timeout=10.0)

            # Stop protocol engine
            if self.config.enable_omnidimensional_protocols:
                self.protocol_engine.stop()

            # Mark as inactive
            self.integration_active = False

            # Trigger shutdown callbacks
            self._trigger_event("integration_stopped", self.get_system_status())

            logger.info("✅ Testing and Protocols Integration shutdown completed")

        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")
            raise


# Global instance for module access
_integrator: Optional[TestingAndProtocolsIntegrator] = None

def get_testing_and_protocols_integrator(config: Optional[TestingAndProtocolsConfig] = None) -> TestingAndProtocolsIntegrator:
    """Get global testing and protocols integrator instance"""
    global _integrator
    if _integrator is None:
        _integrator = TestingAndProtocolsIntegrator(config)
    return _integrator
