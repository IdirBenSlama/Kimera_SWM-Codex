"""
Kimera Performance Monitor
=========================
Real-time performance monitoring dashboard.
"""

import time
import psutil
import logging
from datetime import datetime
from typing import Dict, Any

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Real-time performance monitoring"""
    
    def __init__(self):
        self.start_time = time.time()
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except:
                self.gpu_handle = None
        else:
            self.gpu_handle = None
            
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": time.time() - self.start_time,
            "cpu": {
                "percent": psutil.cpu_percent(interval=1),
                "cores": psutil.cpu_count(),
                "frequency": psutil.cpu_freq().current if psutil.cpu_freq() else 0
            },
            "memory": {
                "percent": psutil.virtual_memory().percent,
                "used_gb": psutil.virtual_memory().used / (1024**3),
                "available_gb": psutil.virtual_memory().available / (1024**3)
            },
            "disk": {
                "percent": psutil.disk_usage('/').percent,
                "free_gb": psutil.disk_usage('/').free / (1024**3)
            }
        }
        
        # Add GPU metrics if available
        if self.gpu_handle:
            try:
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                gpu_temp = pynvml.nvmlDeviceGetTemperature(self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
                
                metrics["gpu"] = {
                    "utilization": gpu_util.gpu,
                    "memory_percent": (gpu_mem.used / gpu_mem.total) * 100,
                    "memory_used_gb": gpu_mem.used / (1024**3),
                    "temperature": gpu_temp
                }
            except:
                pass
                
        return metrics
        
    def print_dashboard(self):
        """Print performance dashboard"""
        metrics = self.get_system_metrics()
        
        print("\n" + "="*60)
        print(f"KIMERA PERFORMANCE MONITOR - {metrics['timestamp']}")
        print("="*60)
        
        print(f"\n📊 SYSTEM METRICS (Uptime: {metrics['uptime_seconds']:.0f}s)")
        print(f"  CPU:    {metrics['cpu']['percent']:5.1f}% ({metrics['cpu']['cores']} cores @ {metrics['cpu']['frequency']:.0f}MHz)")
        print(f"  Memory: {metrics['memory']['percent']:5.1f}% ({metrics['memory']['used_gb']:.1f}GB used)")
        print(f"  Disk:   {metrics['disk']['percent']:5.1f}% ({metrics['disk']['free_gb']:.1f}GB free)")
        
        if "gpu" in metrics:
            print(f"\n🎮 GPU METRICS")
            print(f"  Utilization: {metrics['gpu']['utilization']:5.1f}%")
            print(f"  Memory:      {metrics['gpu']['memory_percent']:5.1f}% ({metrics['gpu']['memory_used_gb']:.1f}GB used)")
            print(f"  Temperature: {metrics['gpu']['temperature']:5.1f}°C")
            
        # Performance assessment
        print(f"\n🎯 PERFORMANCE ASSESSMENT")
        cpu_status = "🟢 Optimal" if metrics['cpu']['percent'] < 70 else "🟡 High" if metrics['cpu']['percent'] < 90 else "🔴 Critical"
        mem_status = "🟢 Optimal" if metrics['memory']['percent'] < 70 else "🟡 High" if metrics['memory']['percent'] < 85 else "🔴 Critical"
        
        print(f"  CPU Status:    {cpu_status}")
        print(f"  Memory Status: {mem_status}")
        
        if "gpu" in metrics:
            gpu_status = "🟢 Optimal" if metrics['gpu']['utilization'] > 60 else "🟡 Underutilized" if metrics['gpu']['utilization'] > 30 else "🔴 Idle"
            print(f"  GPU Status:    {gpu_status}")
            
    def monitor_loop(self, interval: int = 5):
        """Run monitoring loop"""
        logger.info("Starting performance monitoring...")
        try:
            while True:
                self.print_dashboard()
                time.sleep(interval)
        except KeyboardInterrupt:
            logger.info("\nMonitoring stopped")
            

if __name__ == "__main__":
    monitor = PerformanceMonitor()
    monitor.monitor_loop()
