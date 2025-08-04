#!/usr/bin/env python3
"""
Improved KIMERA SWM System Monitor
Reduces noise and provides intelligent monitoring

This monitor distinguishes between:
- Normal idle states
- Actual system issues
- Performance optimization opportunities
"""

import time
import psutil
import logging
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ImprovedSystemMonitor:
    """Intelligent system monitor that reduces false alarms"""
    
    def __init__(self):
        self.last_activity_time = time.time()
        self.baseline_metrics = self._collect_baseline()
        
    def _collect_baseline(self) -> Dict[str, float]:
        """Collect baseline system metrics"""
        return {
            'cpu_idle': psutil.cpu_percent(interval=1),
            'memory_available': psutil.virtual_memory().available,
            'timestamp': time.time()
        }
    
    def is_system_idle(self) -> bool:
        """Determine if system is in normal idle state"""
        current_cpu = psutil.cpu_percent(interval=0.1)
        current_memory = psutil.virtual_memory()
        
        # System is considered idle if:
        # - CPU usage is low
        # - Memory usage is stable
        # - No active processing tasks
        return (
            current_cpu < 10.0 and
            current_memory.percent < 80.0 and
            time.time() - self.last_activity_time > 30
        )
    
    def get_intelligent_health_status(self) -> Dict[str, Any]:
        """Get health status with intelligent idle detection"""
        is_idle = self.is_system_idle()
        
        status = {
            "timestamp": datetime.now().isoformat(),
            "system_state": "idle" if is_idle else "active",
            "health_level": "normal" if is_idle else "monitoring",
            "efficiency_context": "idle_state_normal" if is_idle else "active_monitoring",
            "recommendations": []
        }
        
        if is_idle:
            status["recommendations"].append("System is in normal idle state - no action needed")
        else:
            status["recommendations"].append("System is active - monitoring performance")
            
        return status

# Global monitor instance
_monitor = None

def get_system_monitor() -> ImprovedSystemMonitor:
    """Get the global system monitor instance"""
    global _monitor
    if _monitor is None:
        _monitor = ImprovedSystemMonitor()
    return _monitor
