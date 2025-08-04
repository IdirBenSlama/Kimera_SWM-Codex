"""
Services Integration Module
==========================

This module integrates all background services and jobs into the Kimera
cognitive architecture, providing a unified interface for service management.

Integration Points:
- KimeraSystem: Initializes and manages all services
- Background Jobs: Scheduled tasks for system maintenance
- CLIP Service: Visual grounding capabilities
- Service Monitoring: Health checks and recovery

This module follows aerospace service management standards with:
- Service lifecycle management
- Dependency resolution
- Health monitoring
- Graceful degradation
"""

import asyncio
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional

# Kimera imports
try:
    from ...utils.kimera_logger import get_system_logger
except ImportError:
    try:
        from utils.kimera_logger import get_system_logger
    except ImportError:
        # Create placeholders for utils.kimera_logger
        def get_system_logger(*args, **kwargs):
            return None


from .background_job_manager import (
    BackgroundJobManager,
    JobConfiguration,
    JobPriority,
    get_job_manager,
)
from .clip_service_integration import CLIPServiceIntegration, get_clip_service

logger = get_system_logger(__name__)


class ServiceStatus(Enum):
    """Service health status"""

    HEALTHY = auto()
    DEGRADED = auto()
    UNHEALTHY = auto()
    STOPPED = auto()


@dataclass
class ServiceHealth:
    """Health status of a service"""

    service_name: str
    status: ServiceStatus
    last_check: datetime
    error_count: int
    last_error: Optional[str] = None
    metrics: Dict[str, Any] = None


class ServicesIntegration:
    """
    Integration layer for all Kimera services.

    This class provides:
    - Unified service initialization
    - Health monitoring
    - Service dependency management
    - Graceful degradation
    """

    def __init__(self):
        # Service instances
        self.job_manager: Optional[BackgroundJobManager] = None
        self.clip_service: Optional[CLIPServiceIntegration] = None

        # Health monitoring
        self.service_health: Dict[str, ServiceHealth] = {}
        self._health_check_interval = 60  # seconds
        self._health_check_task = None

        # State
        self._initialized = False
        self._lock = threading.Lock()
        self._shutdown_event = threading.Event()

        # Metrics
        self.initialization_time = None
        self.total_health_checks = 0

    async def initialize(
        self, embedding_fn: Optional[Callable[[str], List[float]]] = None
    ) -> bool:
        """
        Initialize all services.

        Args:
            embedding_fn: Optional embedding function for background jobs

        Returns:
            bool: True if initialization successful
        """
        with self._lock:
            if self._initialized:
                return True

            start_time = datetime.now(timezone.utc)

            try:
                logger.info("Initializing Services Integration...")

                # Initialize Background Job Manager
                logger.info("Initializing Background Job Manager...")
                self.job_manager = get_job_manager()

                # Initialize Kimera-specific jobs if embedding function provided
                if embedding_fn:
                    self.job_manager.initialize_kimera_jobs(embedding_fn)

                # Add system maintenance jobs
                self._add_system_maintenance_jobs()

                # Start job manager
                self.job_manager.start()

                self.service_health["job_manager"] = ServiceHealth(
                    service_name="Background Job Manager",
                    status=ServiceStatus.HEALTHY,
                    last_check=datetime.now(timezone.utc),
                    error_count=0,
                )

                # Initialize CLIP Service
                logger.info("Initializing CLIP Service...")
                self.clip_service = get_clip_service()

                clip_status = self.clip_service.get_status()
                service_status = (
                    ServiceStatus.HEALTHY
                    if clip_status["available"]
                    else ServiceStatus.DEGRADED
                )

                self.service_health["clip_service"] = ServiceHealth(
                    service_name="CLIP Service",
                    status=service_status,
                    last_check=datetime.now(timezone.utc),
                    error_count=0,
                    metrics=clip_status,
                )

                if service_status == ServiceStatus.DEGRADED:
                    logger.warning(
                        "CLIP Service running in degraded mode (lightweight)"
                    )

                # Start health monitoring
                self._health_check_task = asyncio.create_task(
                    self._health_monitoring_loop()
                )

                self._initialized = True
                self.initialization_time = (
                    datetime.now(timezone.utc) - start_time
                ).total_seconds()

                logger.info(
                    f"✅ Services Integration initialized successfully in {self.initialization_time:.2f}s"
                )
                return True

            except Exception as e:
                logger.error(f"Failed to initialize Services Integration: {e}")
                return False

    def _add_system_maintenance_jobs(self):
        """Add system-level maintenance jobs"""
        # Health check job
        self.job_manager.add_job(
            JobConfiguration(
                name="service_health_check",
                func=self._perform_health_check,
                trigger="interval",
                priority=JobPriority.HIGH,
                kwargs={"minutes": 5},
            )
        )

        # Cache cleanup job
        self.job_manager.add_job(
            JobConfiguration(
                name="cache_cleanup",
                func=self._cleanup_caches,
                trigger="interval",
                priority=JobPriority.MAINTENANCE,
                kwargs={"hours": 6},
            )
        )

        # Metrics collection job
        self.job_manager.add_job(
            JobConfiguration(
                name="metrics_collection",
                func=self._collect_metrics,
                trigger="interval",
                priority=JobPriority.LOW,
                kwargs={"minutes": 15},
            )
        )

    def _perform_health_check(self):
        """Perform health check on all services"""
        self.total_health_checks += 1

        # Check Job Manager
        if self.job_manager:
            try:
                job_metrics = self.job_manager.get_metrics_summary()

                # Determine health based on metrics
                if job_metrics["jobs_with_circuit_breaker_open"] > 0:
                    status = ServiceStatus.DEGRADED
                elif job_metrics["overall_success_rate"] < 50:
                    status = ServiceStatus.UNHEALTHY
                else:
                    status = ServiceStatus.HEALTHY

                self.service_health["job_manager"] = ServiceHealth(
                    service_name="Background Job Manager",
                    status=status,
                    last_check=datetime.now(timezone.utc),
                    error_count=0,
                    metrics=job_metrics,
                )

            except Exception as e:
                logger.error(f"Job Manager health check failed: {e}")
                if "job_manager" in self.service_health:
                    self.service_health["job_manager"].status = ServiceStatus.UNHEALTHY
                    self.service_health["job_manager"].error_count += 1
                    self.service_health["job_manager"].last_error = str(e)

        # Check CLIP Service
        if self.clip_service:
            try:
                clip_status = self.clip_service.get_status()

                if clip_status["available"]:
                    status = ServiceStatus.HEALTHY
                elif clip_status["lightweight_mode"]:
                    status = ServiceStatus.DEGRADED
                else:
                    status = ServiceStatus.UNHEALTHY

                self.service_health["clip_service"] = ServiceHealth(
                    service_name="CLIP Service",
                    status=status,
                    last_check=datetime.now(timezone.utc),
                    error_count=0,
                    metrics=clip_status,
                )

            except Exception as e:
                logger.error(f"CLIP Service health check failed: {e}")
                if "clip_service" in self.service_health:
                    self.service_health["clip_service"].status = ServiceStatus.UNHEALTHY
                    self.service_health["clip_service"].error_count += 1
                    self.service_health["clip_service"].last_error = str(e)

    def _cleanup_caches(self):
        """Clean up service caches"""
        logger.info("Performing cache cleanup...")

        # Clear CLIP cache
        if self.clip_service:
            try:
                self.clip_service.clear_cache()
            except Exception as e:
                logger.error(f"Failed to clear CLIP cache: {e}")

    def _collect_metrics(self):
        """Collect metrics from all services"""
        metrics = {"timestamp": datetime.now(timezone.utc).isoformat(), "services": {}}

        # Job Manager metrics
        if self.job_manager:
            try:
                metrics["services"][
                    "job_manager"
                ] = self.job_manager.get_metrics_summary()
            except Exception as e:
                logger.error(f"Failed to collect job manager metrics: {e}")

        # CLIP Service metrics
        if self.clip_service:
            try:
                metrics["services"]["clip_service"] = self.clip_service.get_status()
            except Exception as e:
                logger.error(f"Failed to collect CLIP service metrics: {e}")

        # Log metrics (in production, would send to monitoring system)
        logger.debug(f"Service metrics: {metrics}")

    async def _health_monitoring_loop(self):
        """Background health monitoring loop"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self._health_check_interval)
                self._perform_health_check()

                # Check for critical issues
                unhealthy_services = [
                    name
                    for name, health in self.service_health.items()
                    if health.status == ServiceStatus.UNHEALTHY
                ]

                if unhealthy_services:
                    logger.warning(f"Unhealthy services detected: {unhealthy_services}")
                    # In production, would trigger alerts

            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")

    def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all services"""
        with self._lock:
            return {
                "initialized": self._initialized,
                "initialization_time": self.initialization_time,
                "total_health_checks": self.total_health_checks,
                "services": {
                    name: {
                        "name": health.service_name,
                        "status": health.status.name,
                        "last_check": health.last_check.isoformat(),
                        "error_count": health.error_count,
                        "last_error": health.last_error,
                        "healthy": health.status == ServiceStatus.HEALTHY,
                    }
                    for name, health in self.service_health.items()
                },
            }

    def get_job_manager(self) -> Optional[BackgroundJobManager]:
        """Get the job manager instance"""
        return self.job_manager

    def get_clip_service(self) -> Optional[CLIPServiceIntegration]:
        """Get the CLIP service instance"""
        return self.clip_service

    def add_job(self, job_config: JobConfiguration) -> Optional[str]:
        """Add a new job to the job manager"""
        if self.job_manager:
            return self.job_manager.add_job(job_config)
        else:
            logger.error("Cannot add job: Job Manager not initialized")
            return None

    def remove_job(self, job_id: str):
        """Remove a job from the job manager"""
        if self.job_manager:
            self.job_manager.remove_job(job_id)
        else:
            logger.error("Cannot remove job: Job Manager not initialized")

    async def shutdown(self):
        """Clean shutdown of all services"""
        logger.info("Shutting down Services Integration...")

        # Signal shutdown
        self._shutdown_event.set()

        # Cancel health monitoring
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        # Shutdown services
        if self.job_manager:
            self.job_manager.shutdown(wait=True)
            logger.info("✅ Job Manager shutdown complete")

        if self.clip_service:
            self.clip_service.shutdown()
            logger.info("✅ CLIP Service shutdown complete")

        logger.info("Services Integration shutdown complete")


# Module-level instance
_services_instance = None
_services_lock = threading.Lock()


def get_services_integration() -> ServicesIntegration:
    """Get the singleton instance of the Services Integration"""
    global _services_instance

    if _services_instance is None:
        with _services_lock:
            if _services_instance is None:
                _services_instance = ServicesIntegration()

    return _services_instance


__all__ = [
    "ServicesIntegration",
    "get_services_integration",
    "ServiceStatus",
    "ServiceHealth",
]
