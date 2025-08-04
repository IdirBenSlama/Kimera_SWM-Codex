#!/usr/bin/env python3
"""
KIMERA Real-time Performance Monitoring Dashboard
Comprehensive system monitoring with GPU utilization, engine status, and performance metrics
"""

import asyncio
import time
import psutil
import torch
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import threading
from dataclasses import dataclass, asdict
from collections import deque
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """System performance metrics snapshot"""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    gpu_available: bool
    gpu_name: Optional[str] = None
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None
    gpu_utilization_percent: Optional[float] = None
    gpu_temperature: Optional[float] = None

@dataclass
class EngineStatus:
    """Engine status information"""
    name: str
    status: str  # operational, error, unknown
    last_activity: str
    performance_score: float
    initialization_time: float
    error_count: int

@dataclass
class PerformanceAlert:
    """Performance alert information"""
    timestamp: str
    level: str  # info, warning, critical
    component: str
    message: str
    value: float
    threshold: float

class KimeraPerformanceMonitor:
    """Real-time performance monitoring system for KIMERA"""
    
    def __init__(self, history_size: int = 100):
        self.history_size = history_size
        self.metrics_history = deque(maxlen=history_size)
        self.engine_statuses = {}
        self.alerts = deque(maxlen=50)
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Performance thresholds
        self.thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'gpu_memory_percent': 90.0,
            'gpu_temperature': 83.0,  # Celsius
            'response_time': 5.0  # seconds
        }
        
        # Initialize GPU monitoring
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            try:
                import pynvml
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.nvml_available = True
                logger.info("🖥️ NVIDIA-ML monitoring initialized")
            except ImportError:
                self.nvml_available = False
                logger.warning("⚠️ pynvml not available - limited GPU monitoring")
        else:
            self.nvml_available = False
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system performance metrics"""
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_available_gb = memory.available / (1024**3)
        
        # GPU metrics
        gpu_memory_used_mb = None
        gpu_memory_total_mb = None
        gpu_utilization_percent = None
        gpu_temperature = None
        gpu_name = None
        
        if self.gpu_available:
            gpu_name = torch.cuda.get_device_name()
            
            if torch.cuda.is_available():
                gpu_memory_used_mb = torch.cuda.memory_allocated() / (1024**2)
                gpu_memory_total_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
            
            if self.nvml_available:
                try:
                    import pynvml
                    # GPU utilization
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                    gpu_utilization_percent = utilization.gpu
                    
                    # GPU temperature
                    gpu_temperature = pynvml.nvmlDeviceGetTemperature(self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
                    
                    # More accurate memory info
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                    gpu_memory_used_mb = memory_info.used / (1024**2)
                    gpu_memory_total_mb = memory_info.total / (1024**2)
                    
                except Exception as e:
                    logger.debug(f"NVML error: {e}")
        
        return SystemMetrics(
            timestamp=datetime.now().isoformat(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_available_gb=memory_available_gb,
            gpu_available=self.gpu_available,
            gpu_name=gpu_name,
            gpu_memory_used_mb=gpu_memory_used_mb,
            gpu_memory_total_mb=gpu_memory_total_mb,
            gpu_utilization_percent=gpu_utilization_percent,
            gpu_temperature=gpu_temperature
        )
    
    def check_performance_thresholds(self, metrics: SystemMetrics):
        """Check metrics against thresholds and generate alerts"""
        alerts = []
        
        # CPU threshold
        if metrics.cpu_percent > self.thresholds['cpu_percent']:
            alerts.append(PerformanceAlert(
                timestamp=metrics.timestamp,
                level='warning',
                component='CPU',
                message=f'High CPU usage: {metrics.cpu_percent:.1f}%',
                value=metrics.cpu_percent,
                threshold=self.thresholds['cpu_percent']
            ))
        
        # Memory threshold
        if metrics.memory_percent > self.thresholds['memory_percent']:
            alerts.append(PerformanceAlert(
                timestamp=metrics.timestamp,
                level='warning',
                component='Memory',
                message=f'High memory usage: {metrics.memory_percent:.1f}%',
                value=metrics.memory_percent,
                threshold=self.thresholds['memory_percent']
            ))
        
        # GPU memory threshold
        if metrics.gpu_memory_used_mb and metrics.gpu_memory_total_mb:
            gpu_memory_percent = (metrics.gpu_memory_used_mb / metrics.gpu_memory_total_mb) * 100
            if gpu_memory_percent > self.thresholds['gpu_memory_percent']:
                alerts.append(PerformanceAlert(
                    timestamp=metrics.timestamp,
                    level='warning',
                    component='GPU Memory',
                    message=f'High GPU memory usage: {gpu_memory_percent:.1f}%',
                    value=gpu_memory_percent,
                    threshold=self.thresholds['gpu_memory_percent']
                ))
        
        # GPU temperature threshold
        if metrics.gpu_temperature and metrics.gpu_temperature > self.thresholds['gpu_temperature']:
            alerts.append(PerformanceAlert(
                timestamp=metrics.timestamp,
                level='critical',
                component='GPU Temperature',
                message=f'High GPU temperature: {metrics.gpu_temperature}°C',
                value=metrics.gpu_temperature,
                threshold=self.thresholds['gpu_temperature']
            ))
        
        # Add alerts to queue
        for alert in alerts:
            self.alerts.append(alert)
            level_emoji = "🔥" if alert.level == 'critical' else "⚠️"
            logger.warning(f"{level_emoji} ALERT: {alert.message}")
    
    def discover_engine_status(self) -> Dict[str, EngineStatus]:
        """Discover and check status of KIMERA engines"""
        engines = {}
        
        # Mock engine discovery - in real implementation, this would
        # check actual engine instances, API endpoints, or process status
        engine_list = [
            "ThermodynamicEngine",
            "QuantumCognitiveEngine", 
            "GPUCryptographicEngine",
            "RevolutionaryIntelligenceEngine",
            "UniversalCompassionEngine",
            "LivingNeutralityEngine",
            "ContextSupremacyEngine",
            "GeniusDriftEngine"
        ]
        
        for engine_name in engine_list:
            # Simulate engine status check
            status = "operational"  # Could be: operational, error, unknown
            performance_score = 95.0 + (hash(engine_name) % 10) - 5  # Mock score
            
            engines[engine_name] = EngineStatus(
                name=engine_name,
                status=status,
                last_activity=datetime.now().isoformat(),
                performance_score=performance_score,
                initialization_time=1.5 + (hash(engine_name) % 100) / 100,
                error_count=0
            )
        
        return engines
    
    def generate_dashboard_data(self) -> Dict[str, Any]:
        """Generate complete dashboard data"""
        current_metrics = self.collect_system_metrics()
        engine_statuses = self.discover_engine_status()
        
        # Calculate trends if we have history
        trends = {}
        if len(self.metrics_history) > 1:
            recent_metrics = list(self.metrics_history)[-10:]  # Last 10 readings
            
            if len(recent_metrics) >= 2:
                cpu_trend = recent_metrics[-1].cpu_percent - recent_metrics[0].cpu_percent
                memory_trend = recent_metrics[-1].memory_percent - recent_metrics[0].memory_percent
                
                trends = {
                    'cpu_trend': cpu_trend,
                    'memory_trend': memory_trend,
                    'trend_direction': 'up' if cpu_trend > 5 else 'down' if cpu_trend < -5 else 'stable'
                }
        
        # Recent alerts (last 10)
        recent_alerts = list(self.alerts)[-10:] if self.alerts else []
        
        dashboard_data = {
            'timestamp': datetime.now().isoformat(),
            'system_status': {
                'overall': 'healthy',
                'uptime': self.get_system_uptime(),
                'kimera_version': '0.1.0'
            },
            'current_metrics': asdict(current_metrics),
            'metrics_history': [asdict(m) for m in list(self.metrics_history)[-20:]],  # Last 20 readings
            'engine_statuses': {name: asdict(status) for name, status in engine_statuses.items()},
            'performance_trends': trends,
            'recent_alerts': [asdict(alert) for alert in recent_alerts],
            'thresholds': self.thresholds,
            'monitoring_config': {
                'history_size': self.history_size,
                'update_interval': 5.0,
                'gpu_monitoring': self.nvml_available
            }
        }
        
        return dashboard_data
    
    def get_system_uptime(self) -> str:
        """Get system uptime in human-readable format"""
        try:
            uptime_seconds = time.time() - psutil.boot_time()
            uptime_delta = timedelta(seconds=uptime_seconds)
            return str(uptime_delta).split('.')[0]  # Remove microseconds
        except Exception as e:
            logger.error(f"Error in realtime_performance_monitor.py: {e}", exc_info=True)
            raise  # Re-raise for proper error handling
            return "unknown"
    
    def monitoring_loop(self):
        """Main monitoring loop"""
        logger.info("🔄 Starting performance monitoring loop")
        
        while self.monitoring_active:
            try:
                # Collect metrics
                metrics = self.collect_system_metrics()
                self.metrics_history.append(metrics)
                
                # Check thresholds
                self.check_performance_thresholds(metrics)
                
                # Update engine statuses
                self.engine_statuses = self.discover_engine_status()
                
                # Log status every 30 seconds
                if len(self.metrics_history) % 6 == 0:  # Every 30 seconds at 5s intervals
                    self.log_status_summary(metrics)
                
                # Sleep until next collection
                time.sleep(5.0)  # 5-second intervals
                
            except Exception as e:
                logger.error(f"❌ Monitoring loop error: {e}")
                time.sleep(5.0)
        
        logger.info("🛑 Performance monitoring stopped")
    
    def log_status_summary(self, metrics: SystemMetrics):
        """Log periodic status summary"""
        gpu_info = ""
        if metrics.gpu_available and metrics.gpu_memory_used_mb and metrics.gpu_memory_total_mb:
            gpu_usage = (metrics.gpu_memory_used_mb / metrics.gpu_memory_total_mb) * 100
            gpu_info = f" | GPU: {gpu_usage:.1f}%"
            if metrics.gpu_utilization_percent:
                gpu_info += f" ({metrics.gpu_utilization_percent}% util)"
        
        logger.info(f"📊 System Status: CPU {metrics.cpu_percent:.1f}% | RAM {metrics.memory_percent:.1f}%{gpu_info}")
    
    def start_monitoring(self):
        """Start the monitoring system"""
        if self.monitoring_active:
            logger.warning("⚠️ Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self.monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("🚀 Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop the monitoring system"""
        if not self.monitoring_active:
            logger.warning("⚠️ Monitoring not active")
            return
        
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        logger.info("🛑 Performance monitoring stopped")
    
    def save_dashboard_snapshot(self) -> str:
        """Save current dashboard data to file"""
        dashboard_data = self.generate_dashboard_data()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"kimera_dashboard_snapshot_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(dashboard_data, f, indent=2, default=str)
        
        logger.info(f"💾 Dashboard snapshot saved: {filename}")
        return filename
    
    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report"""
        dashboard_data = self.generate_dashboard_data()
        
        report_content = f"""# KIMERA Real-time Performance Report
**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 🖥️ System Overview

**Status**: {dashboard_data['system_status']['overall'].upper()}  
**Uptime**: {dashboard_data['system_status']['uptime']}  
**Version**: {dashboard_data['system_status']['kimera_version']}

## 📊 Current Metrics

### System Resources
- **CPU Usage**: {dashboard_data['current_metrics']['cpu_percent']:.1f}%
- **Memory Usage**: {dashboard_data['current_metrics']['memory_percent']:.1f}%
- **Available Memory**: {dashboard_data['current_metrics']['memory_available_gb']:.1f} GB

### GPU Status
"""
        
        if dashboard_data['current_metrics']['gpu_available']:
            gpu_name = dashboard_data['current_metrics']['gpu_name']
            gpu_mem_used = dashboard_data['current_metrics']['gpu_memory_used_mb']
            gpu_mem_total = dashboard_data['current_metrics']['gpu_memory_total_mb']
            
            report_content += f"""- **GPU**: {gpu_name} ✅
- **GPU Memory**: {gpu_mem_used:.0f}MB / {gpu_mem_total:.0f}MB ({(gpu_mem_used/gpu_mem_total*100):.1f}%)
"""
            
            if dashboard_data['current_metrics']['gpu_utilization_percent']:
                report_content += f"- **GPU Utilization**: {dashboard_data['current_metrics']['gpu_utilization_percent']}%\n"
            
            if dashboard_data['current_metrics']['gpu_temperature']:
                report_content += f"- **GPU Temperature**: {dashboard_data['current_metrics']['gpu_temperature']}°C\n"
        else:
            report_content += "- **GPU**: Not Available ❌\n"
        
        # Engine status
        report_content += "\n## 🔧 Engine Status\n\n"
        
        operational_engines = 0
        total_engines = len(dashboard_data['engine_statuses'])
        
        for engine_name, status in dashboard_data['engine_statuses'].items():
            status_emoji = "✅" if status['status'] == 'operational' else "❌"
            report_content += f"- **{engine_name}**: {status_emoji} {status['status']} ({status['performance_score']:.1f}%)\n"
            if status['status'] == 'operational':
                operational_engines += 1
        
        # Alerts
        if dashboard_data['recent_alerts']:
            report_content += "\n## ⚠️ Recent Alerts\n\n"
            for alert in dashboard_data['recent_alerts'][-5:]:  # Last 5 alerts
                level_emoji = "🔥" if alert['level'] == 'critical' else "⚠️"
                report_content += f"- {level_emoji} **{alert['component']}**: {alert['message']}\n"
        
        # Summary
        engine_health = (operational_engines / total_engines * 100) if total_engines > 0 else 0
        
        report_content += f"""
## 🎯 Performance Summary

- **Engine Health**: {engine_health:.1f}% ({operational_engines}/{total_engines} operational)
- **System Load**: {'Normal' if dashboard_data['current_metrics']['cpu_percent'] < 70 else 'High'}
- **Memory Status**: {'Healthy' if dashboard_data['current_metrics']['memory_percent'] < 80 else 'Warning'}
- **Alert Level**: {'Normal' if not dashboard_data['recent_alerts'] else 'Active Alerts'}

**Overall Status**: {"🟢 EXCELLENT" if engine_health >= 90 and dashboard_data['current_metrics']['cpu_percent'] < 70 else "🟡 GOOD" if engine_health >= 70 else "🔴 NEEDS ATTENTION"}
"""
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"kimera_performance_report_{timestamp}.md"
        
        with open(report_filename, 'w') as f:
            f.write(report_content)
        
        logger.info(f"📄 Performance report saved: {report_filename}")
        return report_filename

class KimeraDashboardServer:
    """Simple HTTP server for dashboard visualization"""
    
    def __init__(self, monitor: KimeraPerformanceMonitor, port: int = 8080):
        self.monitor = monitor
        self.port = port
    
    def generate_html_dashboard(self) -> str:
        """Generate HTML dashboard"""
        dashboard_data = self.monitor.generate_dashboard_data()
        
        html_template = f"""<!DOCTYPE html>
<html>
<head>
    <title>KIMERA Performance Dashboard</title>
    <meta charset="utf-8">
    <meta http-equiv="refresh" content="10">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f0f0f0; }}
        .header {{ background: #2c3e50; color: white; padding: 20px; margin-bottom: 20px; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
        .metric-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #3498db; }}
        .metric-label {{ color: #7f8c8d; margin-bottom: 10px; }}
        .status-good {{ color: #27ae60; }}
        .status-warning {{ color: #f39c12; }}
        .status-critical {{ color: #e74c3c; }}
        .engines-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; }}
        .engine-card {{ background: white; padding: 15px; border-radius: 6px; border-left: 4px solid #27ae60; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 KIMERA Performance Dashboard</h1>
        <p>Real-time system monitoring | Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>
    
    <div class="metrics">
        <div class="metric-card">
            <div class="metric-label">CPU Usage</div>
            <div class="metric-value">{dashboard_data['current_metrics']['cpu_percent']:.1f}%</div>
        </div>
        
        <div class="metric-card">
            <div class="metric-label">Memory Usage</div>
            <div class="metric-value">{dashboard_data['current_metrics']['memory_percent']:.1f}%</div>
        </div>
        
        <div class="metric-card">
            <div class="metric-label">GPU Status</div>
            <div class="metric-value">{'✅ Available' if dashboard_data['current_metrics']['gpu_available'] else '❌ N/A'}</div>
        </div>
        
        <div class="metric-card">
            <div class="metric-label">System Uptime</div>
            <div class="metric-value">{dashboard_data['system_status']['uptime']}</div>
        </div>
    </div>
    
    <h2>🔧 Engine Status</h2>
    <div class="engines-grid">
"""
        
        for engine_name, status in dashboard_data['engine_statuses'].items():
            status_class = "status-good" if status['status'] == 'operational' else "status-critical"
            html_template += f"""
        <div class="engine-card">
            <strong>{engine_name}</strong><br>
            <span class="{status_class}">{status['status'].upper()}</span><br>
            Performance: {status['performance_score']:.1f}%
        </div>
"""
        
        html_template += """
    </div>
    
    <script>
        // Auto-refresh every 10 seconds
        setTimeout(function(){ location.reload(); }, 10000);
    </script>
</body>
</html>"""
        
        return html_template

def main():
    """Main function to run performance monitoring"""
    logger.info("📊 KIMERA Real-time Performance Monitor")
    logger.info("=" * 50)
    
    # Create monitor
    monitor = KimeraPerformanceMonitor(history_size=200)
    
    # Start monitoring
    monitor.start_monitoring()
    
    try:
        # Run for a demo period
        logger.info("🔄 Monitoring active - collecting metrics...")
        logger.info("Press Ctrl+C to stop and generate report")
        
        # Let it run and collect data
        while True:
            time.sleep(10)
            
            # Generate periodic reports
            dashboard_data = monitor.generate_dashboard_data()
            current = dashboard_data['current_metrics']
            
            logger.info(f"📈 CPU: {current['cpu_percent']:.1f}% | RAM: {current['memory_percent']:.1f}% | Alerts: {len(dashboard_data['recent_alerts'])}")
            
    except KeyboardInterrupt:
        logger.info("\n🛑 Stopping monitoring...")
        
    finally:
        # Stop monitoring
        monitor.stop_monitoring()
        
        # Generate final reports
        snapshot_file = monitor.save_dashboard_snapshot()
        report_file = monitor.generate_performance_report()
        
        logger.info(f"\n📊 Final Reports Generated:")
        logger.info(f"  📄 Performance Report: {report_file}")
        logger.info(f"  💾 Dashboard Snapshot: {snapshot_file}")
        logger.info("\n✅ Monitoring complete!")

if __name__ == "__main__":
    main() 