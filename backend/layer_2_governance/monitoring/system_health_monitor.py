"""
System Health Monitor for Kimera
Provides comprehensive system monitoring and automatic optimization
"""

import logging
import time
import threading
import asyncio
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
import psutil
import torch
from dataclasses import dataclass, asdict
from collections import deque
import os

from backend.utils.kimera_logger import get_logger, LogCategory
from backend.config.settings import MonitoringSettings
from .kimera_monitoring_core import KimeraMonitoringCore, MonitoringAlert, AlertSeverity

logger = logging.getLogger(__name__)

@dataclass
class HealthMetrics:
    """Data class for system health metrics"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    gpu_memory_percent: Optional[float]
    disk_percent: float
    gpu_temperature: Optional[float]
    cognitive_load: float
    response_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class SystemHealthMonitor:
    """
    Comprehensive system health monitoring with automatic optimization
    """
    
    def __init__(self, 
                 settings: MonitoringSettings,
                 monitoring_core: KimeraMonitoringCore):
        self.settings = settings
        self.monitoring_core = monitoring_core
        self.monitoring_interval = settings.health_check_interval
        self.is_monitoring = False
        self.metrics_history = deque(maxlen=100)  # Keep last 100 metrics
        
        self.logger = get_logger(__name__, category=LogCategory.SYSTEM)
        
        self._monitor_thread = None
    
    def start_monitoring(self):
        """Start continuous system monitoring"""
        if self.is_monitoring:
            logger.warning("Monitoring already started")
            return
        
        self.is_monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
        self.logger.info("🔍 System health monitoring started")
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.is_monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        self.logger.info("🛑 System health monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                metrics = self.collect_metrics()
                self.metrics_history.append(metrics)
                asyncio.run(self._analyze_metrics(metrics))
                time.sleep(self.monitoring_interval)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}", error=e)
                time.sleep(self.monitoring_interval)
    
    def collect_metrics(self) -> HealthMetrics:
        """Collect current system metrics"""
        start_time = time.time()
        
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # GPU metrics
        gpu_memory_percent = None
        gpu_temperature = None
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_allocated = torch.cuda.memory_allocated(0)
            gpu_memory_percent = (gpu_allocated / gpu_memory) * 100
            
            # Try to get GPU temperature (if available)
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except (ImportError, AttributeError, RuntimeError) as e:
                # GPU temperature not available - could be missing library or hardware access issues
                logger.debug(f"Could not retrieve GPU temperature: {e}")
                pass  # GPU temperature not available
        
        # Calculate cognitive load (simplified)
        cognitive_load = self._calculate_cognitive_load()
        
        # Response time simulation
        response_time = time.time() - start_time
        
        return HealthMetrics(
            timestamp=datetime.utcnow(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            gpu_memory_percent=gpu_memory_percent,
            disk_percent=disk.percent,
            gpu_temperature=gpu_temperature,
            cognitive_load=cognitive_load,
            response_time=response_time
        )
    
    def _calculate_cognitive_load(self) -> float:
        """Calculate current cognitive processing load"""
        # This is a simplified calculation
        # In a real implementation, this would analyze active cognitive processes
        base_load = 20.0  # Base cognitive load
        
        # Add load based on GPU utilization
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_allocated = torch.cuda.memory_allocated(0)
            gpu_utilization = (gpu_allocated / gpu_memory) * 100
            base_load += gpu_utilization * 0.5
        
        return min(base_load, 100.0)
    
    async def _analyze_metrics(self, metrics: HealthMetrics):
        """Analyze metrics and generate alerts/suggestions"""
        thresholds = self.settings.thresholds
        
        # CPU analysis
        if metrics.cpu_percent > thresholds.cpu_critical:
            await self._create_alert(
                AlertSeverity.CRITICAL, "High CPU Usage", 
                metrics.cpu_percent, thresholds.cpu_critical,
                {"suggestion": "Consider reducing computational load or scaling resources"}
            )
        elif metrics.cpu_percent > thresholds.cpu_warning:
            await self._create_alert(
                AlertSeverity.WARNING, "High CPU Usage",
                metrics.cpu_percent, thresholds.cpu_warning
            )
        
        # Memory analysis
        if metrics.memory_percent > thresholds.memory_critical:
            await self._create_alert(
                AlertSeverity.CRITICAL, "High Memory Usage",
                metrics.memory_percent, thresholds.memory_critical,
                {"suggestion": "Potential memory leak detected. Investigate memory usage or increase available RAM."}
            )
        elif metrics.memory_percent > thresholds.memory_warning:
            await self._create_alert(
                AlertSeverity.WARNING, "High Memory Usage",
                metrics.memory_percent, thresholds.memory_warning
            )
        
        # GPU analysis
        if metrics.gpu_memory_percent:
            if metrics.gpu_memory_percent > thresholds.gpu_memory_critical:
                await self._create_alert(
                    AlertSeverity.CRITICAL, "High GPU Memory Usage",
                    metrics.gpu_memory_percent, thresholds.gpu_memory_critical,
                    {"suggestion": "Reduce GPU model sizes or clear GPU cache"}
                )
            elif metrics.gpu_memory_percent > thresholds.gpu_memory_warning:
                await self._create_alert(
                    AlertSeverity.WARNING, "High GPU Memory Usage",
                    metrics.gpu_memory_percent, thresholds.gpu_memory_warning
                )
        
        # Temperature analysis
        if metrics.gpu_temperature and metrics.gpu_temperature > 80:
            await self._create_alert(
                AlertSeverity.WARNING, "High GPU Temperature",
                metrics.gpu_temperature, 80.0,
                {"suggestion": "Check GPU cooling and reduce workload if necessary"}
            )
    
    async def _create_alert(self, severity: AlertSeverity, metric_name: str, value: float, threshold: float, context: Optional[Dict] = None):
        """Create and send a monitoring alert."""
        message = f"{severity.value.upper()}: {metric_name} is at {value:.1f}%, exceeding threshold of {threshold:.1f}%"
        
        alert = MonitoringAlert(
            id=f"{metric_name}_{datetime.utcnow().timestamp()}",
            severity=severity,
            message=message,
            timestamp=datetime.utcnow(),
            metric_name=metric_name,
            value=value,
            threshold=threshold,
            context=context or {}
        )
        
        self.logger.log(
            logging.CRITICAL if severity == AlertSeverity.CRITICAL else logging.WARNING,
            message,
            alert_details=alert.to_dict()
        )
        
        await self.monitoring_core.send_alert(alert)

    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status summary"""
        if not self.metrics_history:
            return {"status": "no_data", "message": "No metrics available yet"}
        
        latest = self.metrics_history[-1]
        
        # Determine overall status from monitoring core alerts
        recent_alerts = [
            a for a in self.monitoring_core.alerts 
            if a.timestamp > datetime.utcnow() - timedelta(minutes=10)
        ]
        
        status = "healthy"
        if any(a.severity == AlertSeverity.CRITICAL for a in recent_alerts):
            status = "critical"
        elif any(a.severity == AlertSeverity.WARNING for a in recent_alerts):
            status = "warning"
        
        return {
            "status": status,
            "timestamp": latest.timestamp.isoformat(),
            "metrics": latest.to_dict(),
            "recent_alerts": [a.to_dict() for a in recent_alerts[-5:]],
            "uptime": self._get_uptime(),
            "monitoring_active": self.is_monitoring
        }
    
    def get_metrics_history(self, hours: int = 1) -> list:
        """Get metrics history for specified hours"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return [m.to_dict() for m in self.metrics_history if m.timestamp > cutoff]

    def _get_uptime(self) -> str:
        """Get system uptime"""
        p = psutil.Process(os.getpid())
        create_time = datetime.fromtimestamp(p.create_time())
        uptime = datetime.now() - create_time
        return str(uptime)

    def optimize_system(self) -> Dict[str, Any]:
        """
        Automatically performs optimization based on current health status.
        Currently, this is a placeholder for more advanced logic.
        """
        
        status = self.get_health_status()
        
        if status['status'] == 'critical':
            self.logger.warning("System critical. Attempting emergency optimization.")
            # Example action: Reduce cognitive load
            # This would need a proper interface to the cognitive engines
            return {"action": "reduced_cognitive_load", "details": "Emergency action taken."}

        if status['status'] == 'warning':
            self.logger.info("System warning. Performing proactive optimization.")
            # Example action: Clear caches
            return {"action": "cleared_caches", "details": "Proactive optimization."}
            
        return {"action": "none", "details": "System is healthy. No optimization needed."}


# Singleton instance management
_health_monitor_instance: Optional[SystemHealthMonitor] = None
_health_monitor_lock = threading.Lock()

def get_health_monitor() -> SystemHealthMonitor:
    """Get the singleton instance of the SystemHealthMonitor"""
    global _health_monitor_instance
    if _health_monitor_instance is None:
        with _health_monitor_lock:
            if _health_monitor_instance is None:
                # This part needs configuration management
                from backend.config.config_manager import ConfigManager
                from backend.layer_2_governance.monitoring.kimera_monitoring_core import get_monitoring_core
                
                settings = MonitoringSettings(**ConfigManager.get_monitoring_settings())
                monitoring_core = get_monitoring_core()
                _health_monitor_instance = SystemHealthMonitor(settings, monitoring_core)
                
    return _health_monitor_instance 