"""
Alert Manager - Intelligent Alert System
========================================

Implements alert management based on:
- Prometheus Alertmanager patterns
- NASA Mission Control alert systems
- Medical device alarm standards (IEC 60601-1-8)
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import threading
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    """Alert severity levels (aligned with syslog)."""
    DEBUG = 7
    INFO = 6
    NOTICE = 5
    WARNING = 4
    ERROR = 3
    CRITICAL = 2
    ALERT = 1
    EMERGENCY = 0

class AlertState(Enum):
    """Alert lifecycle states."""
    PENDING = "pending"      # Waiting for confirmation
    FIRING = "firing"        # Active alert
    RESOLVED = "resolved"    # Alert condition cleared
    SILENCED = "silenced"    # Temporarily muted

@dataclass
class Alert:
    """Alert instance with full context."""
    id: str
    name: str
    level: AlertLevel
    state: AlertState
    message: str
    
    # Context
    component: str
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    
    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    
    # Alert management
    count: int = 1  # Number of occurrences
    fingerprint: str = ""  # For deduplication
    silence_until: Optional[datetime] = None
    
    def __post_init__(self):
        if not self.fingerprint:
            # Generate fingerprint for deduplication
            key_parts = [self.name, self.component]
            key_parts.extend(f"{k}:{v}" for k, v in sorted(self.labels.items()))
            self.fingerprint = str(hash(tuple(key_parts)))

@dataclass
class AlertRule:
    """Alert rule definition."""
    name: str
    condition: Callable[[], bool]
    level: AlertLevel
    message_template: str
    component: str
    
    # Rule configuration
    for_duration: timedelta = timedelta(seconds=0)  # How long before firing
    repeat_interval: timedelta = timedelta(minutes=5)  # Re-alert interval
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    
    # State tracking
    pending_since: Optional[datetime] = None
    last_alert: Optional[datetime] = None

class AlertManager:
    """
    Centralized alert management system.
    
    Features:
    - Alert deduplication
    - Alert routing
    - Silence management
    - Alert aggregation
    - Notification dispatch
    """
    
    def __init__(self):
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=10000)
        
        # Notification channels
        self.notification_handlers: Dict[AlertLevel, List[Callable]] = defaultdict(list)
        
        # Silence rules
        self.silences: List[Dict[str, Any]] = []
        
        # Aggregation groups
        self.aggregation_groups: Dict[str, List[Alert]] = defaultdict(list)
        
        # Threading
        self._lock = threading.RLock()
        self._evaluation_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Metrics
        self.metrics = {
            'alerts_total': 0,
            'alerts_firing': 0,
            'alerts_resolved': 0,
            'notifications_sent': 0
        }
        
        logger.info("AlertManager initialized")
    
    def register_rule(self, rule: AlertRule):
        """Register an alert rule."""
        with self._lock:
            self.rules[rule.name] = rule
            logger.info(f"Registered alert rule: {rule.name}")
    
    def register_handler(
        self,
        handler: Callable,
        min_level: AlertLevel = AlertLevel.WARNING
    ):
        """Register a notification handler."""
        for level in AlertLevel:
            if level.value <= min_level.value:
                self.notification_handlers[level].append(handler)
        
        logger.info(f"Registered notification handler for {min_level.name}+")
    
    async def start(self):
        """Start alert evaluation loop."""
        if self._running:
            return
        
        self._running = True
        self._evaluation_task = asyncio.create_task(self._evaluation_loop())
        logger.info("AlertManager started")
    
    async def stop(self):
        """Stop alert evaluation."""
        self._running = False
        
        if self._evaluation_task:
            self._evaluation_task.cancel()
            try:
                await self._evaluation_task
            except asyncio.CancelledError:
                pass
        
        logger.info("AlertManager stopped")
    
    async def _evaluation_loop(self):
        """Main alert evaluation loop."""
        while self._running:
            try:
                await self._evaluate_rules()
                await self._check_resolutions()
                await self._cleanup_expired()
                
                await asyncio.sleep(10)  # Evaluate every 10 seconds
                
            except Exception as e:
                logger.error(f"Alert evaluation error: {e}")
                await asyncio.sleep(10)
    
    async def _evaluate_rules(self):
        """Evaluate all alert rules."""
        with self._lock:
            for rule in self.rules.values():
                try:
                    # Check condition
                    condition_met = await self._evaluate_condition(rule.condition)
                    
                    if condition_met:
                        await self._handle_condition_met(rule)
                    else:
                        await self._handle_condition_cleared(rule)
                        
                except Exception as e:
                    logger.error(f"Error evaluating rule {rule.name}: {e}")
    
    async def _evaluate_condition(self, condition: Callable) -> bool:
        """Safely evaluate a condition."""
        try:
            if asyncio.iscoroutinefunction(condition):
                return await condition()
            else:
                return condition()
        except Exception as e:
            logger.error(f"Condition evaluation error: {e}")
            return False
    
    async def _handle_condition_met(self, rule: AlertRule):
        """Handle when alert condition is met."""
        now = datetime.now()
        
        # Check if already pending
        if rule.pending_since is None:
            rule.pending_since = now
            logger.debug(f"Alert {rule.name} pending")
        
        # Check if pending long enough
        if now - rule.pending_since < rule.for_duration:
            return
        
        # Check if we should re-alert
        if rule.last_alert and now - rule.last_alert < rule.repeat_interval:
            return
        
        # Fire alert
        await self._fire_alert(rule)
    
    async def _handle_condition_cleared(self, rule: AlertRule):
        """Handle when alert condition is cleared."""
        rule.pending_since = None
        
        # Resolve any active alerts for this rule
        for alert_id, alert in list(self.active_alerts.items()):
            if alert.name == rule.name and alert.state == AlertState.FIRING:
                await self._resolve_alert(alert_id)
    
    async def _fire_alert(self, rule: AlertRule):
        """Fire an alert from a rule."""
        # Create alert
        alert = Alert(
            id=f"{rule.name}-{int(time.time() * 1000)}",
            name=rule.name,
            level=rule.level,
            state=AlertState.FIRING,
            message=rule.message_template,
            component=rule.component,
            labels=rule.labels.copy(),
            annotations=rule.annotations.copy()
        )
        
        # Check for existing alert with same fingerprint
        existing = None
        for existing_id, existing_alert in self.active_alerts.items():
            if existing_alert.fingerprint == alert.fingerprint:
                existing = existing_alert
                break
        
        if existing:
            # Update existing alert
            existing.count += 1
            existing.updated_at = datetime.now()
            alert = existing
        else:
            # New alert
            with self._lock:
                self.active_alerts[alert.id] = alert
                self.alert_history.append(alert)
                self.metrics['alerts_total'] += 1
                self.metrics['alerts_firing'] += 1
        
        # Check silences
        if self._is_silenced(alert):
            alert.state = AlertState.SILENCED
            return
        
        # Send notifications
        await self._send_notifications(alert)
        
        # Update rule
        rule.last_alert = datetime.now()
        
        logger.warning(f"Alert fired: {alert.name} - {alert.message}")
    
    async def _resolve_alert(self, alert_id: str):
        """Resolve an active alert."""
        with self._lock:
            if alert_id not in self.active_alerts:
                return
            
            alert = self.active_alerts[alert_id]
            alert.state = AlertState.RESOLVED
            alert.resolved_at = datetime.now()
            
            # Move to history
            del self.active_alerts[alert_id]
            self.metrics['alerts_firing'] -= 1
            self.metrics['alerts_resolved'] += 1
        
        logger.info(f"Alert resolved: {alert.name}")
    
    async def _send_notifications(self, alert: Alert):
        """Send notifications for an alert."""
        handlers = self.notification_handlers.get(alert.level, [])
        
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
                
                self.metrics['notifications_sent'] += 1
                
            except Exception as e:
                logger.error(f"Notification handler error: {e}")
    
    def _is_silenced(self, alert: Alert) -> bool:
        """Check if an alert is silenced."""
        now = datetime.now()
        
        # Check alert-specific silence
        if alert.silence_until and now < alert.silence_until:
            return True
        
        # Check global silences
        for silence in self.silences:
            if self._matches_silence(alert, silence):
                return True
        
        return False
    
    def _matches_silence(self, alert: Alert, silence: Dict[str, Any]) -> bool:
        """Check if an alert matches a silence rule."""
        # Match by component
        if 'component' in silence and alert.component != silence['component']:
            return False
        
        # Match by name pattern
        if 'name_pattern' in silence:
            import re
            if not re.match(silence['name_pattern'], alert.name):
                return False
        
        # Match by labels
        if 'labels' in silence:
            for key, value in silence['labels'].items():
                if alert.labels.get(key) != value:
                    return False
        
        # Check time window
        now = datetime.now()
        if 'start_time' in silence and now < silence['start_time']:
            return False
        if 'end_time' in silence and now > silence['end_time']:
            return False
        
        return True
    
    async def _check_resolutions(self):
        """Check if any alerts should be auto-resolved."""
        # This could check if alert conditions have cleared
        pass
    
    async def _cleanup_expired(self):
        """Clean up expired silences and old alerts."""
        now = datetime.now()
        
        # Remove expired silences
        self.silences = [
            s for s in self.silences
            if 'end_time' not in s or s['end_time'] > now
        ]
    
    def silence_alert(
        self,
        alert_id: Optional[str] = None,
        duration: timedelta = timedelta(hours=1),
        **match_criteria
    ):
        """Silence an alert or group of alerts."""
        if alert_id:
            # Silence specific alert
            with self._lock:
                if alert_id in self.active_alerts:
                    alert = self.active_alerts[alert_id]
                    alert.silence_until = datetime.now() + duration
                    logger.info(f"Silenced alert {alert_id} for {duration}")
        else:
            # Create silence rule
            silence = {
                'start_time': datetime.now(),
                'end_time': datetime.now() + duration,
                **match_criteria
            }
            self.silences.append(silence)
            logger.info(f"Created silence rule: {match_criteria}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        with self._lock:
            return list(self.active_alerts.values())
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alert statistics."""
        with self._lock:
            level_counts = defaultdict(int)
            component_counts = defaultdict(int)
            
            for alert in self.active_alerts.values():
                level_counts[alert.level.name] += 1
                component_counts[alert.component] += 1
            
            return {
                'metrics': self.metrics.copy(),
                'active_by_level': dict(level_counts),
                'active_by_component': dict(component_counts),
                'total_active': len(self.active_alerts),
                'total_silences': len(self.silences)
            }

# Predefined alert rules
def create_system_alerts() -> List[AlertRule]:
    """Create standard system alert rules."""
    import psutil
    
    rules = []
    
    # High CPU alert
    rules.append(AlertRule(
        name="high_cpu_usage",
        condition=lambda: psutil.cpu_percent(interval=1) > 90,
        level=AlertLevel.WARNING,
        message_template="CPU usage above 90%",
        component="system",
        for_duration=timedelta(minutes=2),
        labels={"resource": "cpu"}
    ))
    
    # High memory alert
    rules.append(AlertRule(
        name="high_memory_usage",
        condition=lambda: psutil.virtual_memory().percent > 85,
        level=AlertLevel.WARNING,
        message_template="Memory usage above 85%",
        component="system",
        for_duration=timedelta(minutes=1),
        labels={"resource": "memory"}
    ))
    
    # Disk space alert
    rules.append(AlertRule(
        name="low_disk_space",
        condition=lambda: psutil.disk_usage('/').percent > 90,
        level=AlertLevel.ERROR,
        message_template="Disk usage above 90%",
        component="system",
        for_duration=timedelta(seconds=30),
        labels={"resource": "disk"}
    ))
    
    return rules