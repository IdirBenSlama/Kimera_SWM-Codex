"""
Kimera SWM - Real-Time Vault Monitor
===================================

Comprehensive monitoring system for the Kimera vault that tracks:
- Database connectivity and health
- Memory formations (Geoids) in real-time
- Scar formations and cognitive transitions
- Understanding system activity
- Behavior pattern analysis
- Performance metrics and anomaly detection

This system provides real-time insights into the cognitive state of Kimera.
"""

import asyncio
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import json
import statistics

from ..vault.database import get_engine, get_db_status, GeoidDB, ScarDB
from ..vault.vault_manager import VaultManager
from ..engines.contradiction_engine import ContradictionEngine
from ..core.geoid import GeoidState
from ..core.scar import ScarRecord
from ..vault.enhanced_database_schema import InsightDB

logger = logging.getLogger(__name__)

@dataclass
class VaultHealthMetrics:
    """Health metrics for the vault system"""
    timestamp: datetime
    database_connected: bool
    database_latency_ms: float
    total_geoids: int
    total_scars: int
    total_insights: int
    recent_activity_rate: float  # per minute
    memory_efficiency: float
    cognitive_state: str
    anomalies_detected: int
    
    def to_dict(self):
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat()
        }

@dataclass
class CognitiveActivity:
    """Represents a cognitive activity event"""
    timestamp: datetime
    activity_type: str  # 'geoid_creation', 'scar_formation', 'insight_generation'
    entity_id: str
    metadata: Dict[str, Any]
    
    def to_dict(self):
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat()
        }

class RealTimeVaultMonitor:
    """
    Real-time monitoring system for Kimera's vault infrastructure.
    
    Provides comprehensive insights into cognitive processes, memory formation,
    and vault health with real-time analytics and anomaly detection.
    """
    
    def __init__(self, vault_manager: VaultManager, contradiction_engine: ContradictionEngine):
        self.vault_manager = vault_manager
        self.contradiction_engine = contradiction_engine
        self.geoid_db = GeoidDB()
        self.scar_db = ScarDB()
        self.insight_db = InsightDB()
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread = None
        self.activity_history = deque(maxlen=1000)
        self.health_history = deque(maxlen=100)
        self.performance_metrics = defaultdict(list)
        
        # Configuration
        self.monitoring_interval = 5.0  # seconds
        self.health_check_interval = 30.0  # seconds
        self.activity_window = timedelta(minutes=10)
        
        # Anomaly detection
        self.baseline_metrics = {}
        self.anomaly_threshold = 2.0  # standard deviations
        
        # Last known state for delta detection
        self.last_geoid_count = 0
        self.last_scar_count = 0
        self.last_insight_count = 0
        
        logger.info("Real-time vault monitor initialized")
    
    def start_monitoring(self):
        """Start the real-time monitoring system"""
        if self.is_monitoring:
            logger.warning("Monitoring already active")
            return
            
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("🔍 Real-time vault monitoring started")
    
    def stop_monitoring(self):
        """Stop the monitoring system"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        logger.info("🛑 Real-time vault monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        last_health_check = 0
        
        while self.is_monitoring:
            try:
                current_time = time.time()
                
                # Activity monitoring (every interval)
                self._monitor_cognitive_activity()
                
                # Health check (every health_check_interval)
                if current_time - last_health_check >= self.health_check_interval:
                    self._perform_health_check()
                    last_health_check = current_time
                
                # Performance analysis
                self._analyze_performance()
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                time.sleep(self.monitoring_interval)
    
    def _monitor_cognitive_activity(self):
        """Monitor cognitive activity and detect new formations"""
        try:
            # Check for new geoids
            current_geoid_count = self._get_geoid_count()
            if current_geoid_count > self.last_geoid_count:
                delta = current_geoid_count - self.last_geoid_count
                activity = CognitiveActivity(
                    timestamp=datetime.now(),
                    activity_type='geoid_creation',
                    entity_id=f'delta_{delta}',
                    metadata={'new_count': delta, 'total': current_geoid_count}
                )
                self.activity_history.append(activity)
                logger.info(f"🧠 New geoid formations detected: +{delta} (total: {current_geoid_count})")
                self.last_geoid_count = current_geoid_count
            
            # Check for new scars
            current_scar_count = self._get_scar_count()
            if current_scar_count > self.last_scar_count:
                delta = current_scar_count - self.last_scar_count
                activity = CognitiveActivity(
                    timestamp=datetime.now(),
                    activity_type='scar_formation',
                    entity_id=f'delta_{delta}',
                    metadata={'new_count': delta, 'total': current_scar_count}
                )
                self.activity_history.append(activity)
                logger.info(f"⚡ New scar formations detected: +{delta} (total: {current_scar_count})")
                self.last_scar_count = current_scar_count
            
            # Check for new insights
            current_insight_count = self._get_insight_count()
            if current_insight_count > self.last_insight_count:
                delta = current_insight_count - self.last_insight_count
                activity = CognitiveActivity(
                    timestamp=datetime.now(),
                    activity_type='insight_generation',
                    entity_id=f'delta_{delta}',
                    metadata={'new_count': delta, 'total': current_insight_count}
                )
                self.activity_history.append(activity)
                logger.info(f"💡 New insights generated: +{delta} (total: {current_insight_count})")
                self.last_insight_count = current_insight_count
                
        except Exception as e:
            logger.error(f"Error monitoring cognitive activity: {e}")
    
    def _perform_health_check(self):
        """Perform comprehensive health check"""
        try:
            start_time = time.time()
            
            # Database connectivity test
            db_connected = self._test_database_connection()
            db_latency = (time.time() - start_time) * 1000  # ms
            
            # Count entities
            geoid_count = self._get_geoid_count()
            scar_count = self._get_scar_count()
            insight_count = self._get_insight_count()
            
            # Calculate activity rate
            recent_activities = [a for a in self.activity_history 
                               if datetime.now() - a.timestamp <= self.activity_window]
            activity_rate = len(recent_activities) / self.activity_window.total_seconds() * 60
            
            # Memory efficiency (placeholder calculation)
            memory_efficiency = min(100.0, (geoid_count + scar_count) / max(1, geoid_count + scar_count + 1) * 100)
            
            # Determine cognitive state
            cognitive_state = self._determine_cognitive_state(geoid_count, scar_count, activity_rate)
            
            # Anomaly detection
            anomalies = self._detect_anomalies(geoid_count, scar_count, activity_rate)
            
            # Create health metrics
            health_metrics = VaultHealthMetrics(
                timestamp=datetime.now(),
                database_connected=db_connected,
                database_latency_ms=db_latency,
                total_geoids=geoid_count,
                total_scars=scar_count,
                total_insights=insight_count,
                recent_activity_rate=activity_rate,
                memory_efficiency=memory_efficiency,
                cognitive_state=cognitive_state,
                anomalies_detected=len(anomalies)
            )
            
            self.health_history.append(health_metrics)
            
            # Log health status
            logger.info(f"🏥 Vault Health: {cognitive_state} | "
                       f"DB: {'✅' if db_connected else '❌'} ({db_latency:.1f}ms) | "
                       f"G:{geoid_count} S:{scar_count} I:{insight_count} | "
                       f"Activity: {activity_rate:.1f}/min | "
                       f"Anomalies: {len(anomalies)}")
            
        except Exception as e:
            logger.error(f"Error in health check: {e}")
    
    def _analyze_performance(self):
        """Analyze performance trends and patterns"""
        try:
            if len(self.health_history) < 2:
                return
            
            # Calculate trends
            recent_metrics = list(self.health_history)[-10:]
            
            # Database latency trend
            latencies = [m.database_latency_ms for m in recent_metrics]
            avg_latency = statistics.mean(latencies)
            self.performance_metrics['db_latency'].append(avg_latency)
            
            # Activity trend
            activities = [m.recent_activity_rate for m in recent_metrics]
            avg_activity = statistics.mean(activities)
            self.performance_metrics['activity_rate'].append(avg_activity)
            
            # Keep only recent performance data
            for metric_list in self.performance_metrics.values():
                if len(metric_list) > 100:
                    metric_list[:] = metric_list[-100:]
                    
        except Exception as e:
            logger.error(f"Error analyzing performance: {e}")
    
    def _test_database_connection(self) -> bool:
        """Test database connectivity"""
        try:
            status = get_db_status()
            return status.get('status') == 'connected'
        except Exception:
            return False
    
    def _get_geoid_count(self) -> int:
        """Get current geoid count"""
        try:
            return self.vault_manager.get_geoid_count()
        except Exception as e:
            logger.debug(f"Error getting geoid count: {e}")
            return 0
    
    def _get_scar_count(self) -> int:
        """Get current scar count"""
        try:
            return self.vault_manager.get_scar_count()
        except Exception as e:
            logger.debug(f"Error getting scar count: {e}")
            return 0
    
    def _get_insight_count(self) -> int:
        """Get current insight count"""
        try:
            # Try to get insights from insight database
            if hasattr(self.insight_db, 'get_insight_count'):
                return self.insight_db.get_insight_count()
            return 0
        except Exception as e:
            logger.debug(f"Error getting insight count: {e}")
            return 0
    
    def _determine_cognitive_state(self, geoid_count: int, scar_count: int, activity_rate: float) -> str:
        """Determine the current cognitive state based on metrics"""
        if geoid_count == 0 and scar_count == 0:
            return "NASCENT"
        elif activity_rate > 1.0:
            return "HIGHLY_ACTIVE"
        elif activity_rate > 0.1:
            return "ACTIVE"
        elif geoid_count > 0 or scar_count > 0:
            return "DORMANT"
        else:
            return "UNKNOWN"
    
    def _detect_anomalies(self, geoid_count: int, scar_count: int, activity_rate: float) -> List[str]:
        """Detect anomalies in the metrics"""
        anomalies = []
        
        # Database latency anomaly
        if len(self.health_history) > 10:
            recent_latencies = [m.database_latency_ms for m in list(self.health_history)[-10:]]
            if recent_latencies:
                avg_latency = statistics.mean(recent_latencies)
                if avg_latency > 1000:  # > 1 second
                    anomalies.append("HIGH_DB_LATENCY")
        
        # Activity burst detection
        if activity_rate > 10.0:
            anomalies.append("ACTIVITY_BURST")
        
        # Stagnation detection
        if len(self.activity_history) == 0 and geoid_count > 0:
            anomalies.append("COGNITIVE_STAGNATION")
        
        return anomalies
    
    # Public API methods
    
    def get_current_health(self) -> Optional[Dict[str, Any]]:
        """Get the most recent health metrics"""
        if not self.health_history:
            return None
        return self.health_history[-1].to_dict()
    
    def get_recent_activities(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent cognitive activities"""
        activities = list(self.activity_history)[-limit:]
        return [a.to_dict() for a in activities]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary and trends"""
        if not self.performance_metrics:
            return {"status": "insufficient_data"}
        
        summary = {}
        for metric_name, values in self.performance_metrics.items():
            if values:
                summary[metric_name] = {
                    "current": values[-1],
                    "average": statistics.mean(values),
                    "trend": "improving" if len(values) > 1 and values[-1] < values[-2] else "stable"
                }
        
        return summary
    
    def get_cognitive_analytics(self) -> Dict[str, Any]:
        """Get advanced cognitive analytics"""
        if not self.activity_history:
            return {"status": "no_activity_data"}
        
        activities = list(self.activity_history)
        
        # Activity type distribution
        type_counts = defaultdict(int)
        for activity in activities:
            type_counts[activity.activity_type] += 1
        
        # Time-based patterns
        recent_hour = datetime.now() - timedelta(hours=1)
        recent_activities = [a for a in activities if a.timestamp >= recent_hour]
        
        return {
            "total_activities": len(activities),
            "recent_hour_activities": len(recent_activities),
            "activity_distribution": dict(type_counts),
            "cognitive_patterns": {
                "formation_rate": len([a for a in activities if "formation" in a.activity_type]) / max(1, len(activities)),
                "creation_rate": len([a for a in activities if "creation" in a.activity_type]) / max(1, len(activities)),
                "insight_rate": len([a for a in activities if "insight" in a.activity_type]) / max(1, len(activities))
            }
        }
    
    def generate_monitoring_report(self) -> Dict[str, Any]:
        """Generate comprehensive monitoring report"""
        return {
            "timestamp": datetime.now().isoformat(),
            "monitoring_status": "active" if self.is_monitoring else "inactive",
            "current_health": self.get_current_health(),
            "recent_activities": self.get_recent_activities(10),
            "performance_summary": self.get_performance_summary(),
            "cognitive_analytics": self.get_cognitive_analytics(),
            "system_info": {
                "monitoring_interval": self.monitoring_interval,
                "health_check_interval": self.health_check_interval,
                "activity_window_minutes": self.activity_window.total_seconds() / 60,
                "history_size": {
                    "activities": len(self.activity_history),
                    "health_checks": len(self.health_history)
                }
            }
        } 