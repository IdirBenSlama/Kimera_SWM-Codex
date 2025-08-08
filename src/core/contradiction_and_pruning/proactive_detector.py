"""
Proactive Detector - Predictive Analysis and Early Warning System
================================================================

Advanced detection engine that provides predictive analysis, early warning
systems, and proactive monitoring for cognitive, system, and environmental
anomalies and opportunities.

Key Features:
- Predictive anomaly detection
- Early warning systems
- Proactive opportunity identification
- Multi-modal threat detection
- Cognitive state prediction
- System performance forecasting
- Environmental monitoring
- Adaptive threshold management

Scientific Foundation:
- Time Series Analysis
- Anomaly Detection Algorithms
- Predictive Modeling
- Statistical Process Control
- Machine Learning for Prediction
- Signal Processing
- Pattern Recognition
"""

import asyncio
import logging
import threading
import time
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from ..config.settings import get_settings
from ..utils.robust_config import get_api_settings

logger = logging.getLogger(__name__)


class DetectionType(Enum):
    """Types of detection capabilities"""

    ANOMALY = "anomaly"
    THREAT = "threat"
    OPPORTUNITY = "opportunity"
    DEGRADATION = "degradation"
    EMERGENCE = "emergence"
    PATTERN_CHANGE = "pattern_change"
    SYSTEM_FAILURE = "system_failure"
    COGNITIVE_OVERLOAD = "cognitive_overload"


class AlertLevel(Enum):
    """Alert severity levels"""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class PredictionHorizon(Enum):
    """Prediction time horizons"""

    IMMEDIATE = "immediate"  # 0-1 minutes
    SHORT_TERM = "short_term"  # 1-10 minutes
    MEDIUM_TERM = "medium_term"  # 10-60 minutes
    LONG_TERM = "long_term"  # 1+ hours


@dataclass
class DetectionEvent:
    """Auto-generated class."""
    pass
    """Represents a detection event"""

    event_id: str
    detection_type: DetectionType
    alert_level: AlertLevel
    confidence: float
    description: str
    data_source: str
    timestamp: datetime = field(default_factory=datetime.now)
    predicted_time: Optional[datetime] = None
    prediction_horizon: Optional[PredictionHorizon] = None
    context: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    false_positive_probability: float = 0.0
    resolution_status: str = "open"


@dataclass
class TimeSeriesData:
    """Auto-generated class."""
    pass
    """Time series data for analysis"""

    timestamps: List[datetime]
    values: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Convert to numpy arrays"""
        timestamps_numeric = [
            (ts - self.timestamps[0]).total_seconds() for ts in self.timestamps
        ]
        return np.array(timestamps_numeric), np.array(self.values)


@dataclass
class PredictionModel:
    """Auto-generated class."""
    pass
    """Prediction model configuration"""

    model_type: str
    parameters: Dict[str, Any]
    training_data: Optional[TimeSeriesData] = None
    accuracy_metrics: Dict[str, float] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)
class AnomalyDetector:
    """Auto-generated class."""
    pass
    """
    Statistical anomaly detection using multiple algorithms
    """

    def __init__(self, window_size: int = 100, sensitivity: float = 0.95):
        try:
            self.settings = get_api_settings()
        except Exception as e:
            logger.warning(f"API settings loading failed: {e}. Using safe fallback.")
            from ..utils.robust_config import safe_get_api_settings

            self.settings = safe_get_api_settings()
        logger.debug(f"   Environment: {self.settings.environment}")
        self.window_size = window_size
        self.sensitivity = sensitivity
        self.data_buffer = deque(maxlen=window_size)
        self.statistics = {
            "mean": 0.0
            "std": 0.0
            "min": float("inf"),
            "max": float("-inf"),
        }
        self.detection_history = deque(maxlen=1000)

    def add_data_point(
        self, value: float, timestamp: datetime = None
    ) -> Optional[DetectionEvent]:
        """
        Add a data point and check for anomalies

        Args:
            value: Data value to analyze
            timestamp: Timestamp of the data point

        Returns:
            DetectionEvent if anomaly detected, None otherwise
        """

        if timestamp is None:
            timestamp = datetime.now()

        # Add to buffer
        self.data_buffer.append(value)

        # Update statistics
        self._update_statistics()

        # Check for anomalies if we have enough data
        if len(self.data_buffer) >= min(30, self.window_size):
            anomaly_score = self._compute_anomaly_score(value)

            if anomaly_score > self.sensitivity:
                # Create detection event
                event = DetectionEvent(
                    event_id=f"anomaly_{int(time.time())}_{len(self.detection_history)}",
                    detection_type=DetectionType.ANOMALY
                    alert_level=self._determine_alert_level(anomaly_score),
                    confidence=anomaly_score
                    description=f"Statistical anomaly detected: value {value:.3f} deviates from normal range",
                    data_source="statistical_analysis",
                    timestamp=timestamp
                    context={
                        "value": value
                        "anomaly_score": anomaly_score
                        "mean": self.statistics["mean"],
                        "std": self.statistics["std"],
                        "z_score": abs(value - self.statistics["mean"])
                        / (self.statistics["std"] + 1e-8),
                    },
                )

                self.detection_history.append(event)
                return event

        return None

    def _update_statistics(self):
        """Update statistical measures"""
        if not self.data_buffer:
            return

        data = list(self.data_buffer)
        self.statistics["mean"] = np.mean(data)
        self.statistics["std"] = np.std(data)
        self.statistics["min"] = np.min(data)
        self.statistics["max"] = np.max(data)

    def _compute_anomaly_score(self, value: float) -> float:
        """Compute anomaly score for a value"""
        if self.statistics["std"] == 0:
            return 0.0

        # Z-score based anomaly detection
        z_score = abs(value - self.statistics["mean"]) / self.statistics["std"]

        # Convert to probability-like score
        anomaly_score = min(1.0, z_score / 3.0)  # 3-sigma rule

        return anomaly_score

    def _determine_alert_level(self, anomaly_score: float) -> AlertLevel:
        """Determine alert level based on anomaly score"""
        if anomaly_score >= 0.95:
            return AlertLevel.CRITICAL
        elif anomaly_score >= 0.8:
            return AlertLevel.HIGH
        elif anomaly_score >= 0.6:
            return AlertLevel.MEDIUM
        else:
            return AlertLevel.LOW
class TrendAnalyzer:
    """Auto-generated class."""
    pass
    """
    Analyzes trends in time series data for predictive insights
    """

    def __init__(self, min_data_points: int = 10):
        try:
            self.settings = get_api_settings()
        except Exception as e:
            logger.warning(f"API settings loading failed: {e}. Using safe fallback.")
            from ..utils.robust_config import safe_get_api_settings

            self.settings = safe_get_api_settings()
        logger.debug(f"   Environment: {self.settings.environment}")
        self.min_data_points = min_data_points
        self.trend_history = deque(maxlen=500)

    def analyze_trend(self, data: TimeSeriesData) -> Dict[str, Any]:
        """
        Analyze trend in time series data

        Args:
            data: Time series data to analyze

        Returns:
            Trend analysis results
        """

        if len(data.values) < self.min_data_points:
            return {
                "trend_direction": "insufficient_data",
                "trend_strength": 0.0
                "prediction": None
                "confidence": 0.0
            }

        # Convert to numpy
        timestamps, values = data.to_numpy()

        # Linear regression for trend
        coefficients = np.polyfit(timestamps, values, 1)
        slope, intercept = coefficients

        # Trend analysis
        trend_direction = (
            "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
        )
        trend_strength = abs(slope)

        # Correlation coefficient for trend strength
        correlation = np.corrcoef(timestamps, values)[0, 1]
        trend_confidence = abs(correlation)

        # Prediction
        last_timestamp = timestamps[-1]
        future_timestamp = last_timestamp + 300  # 5 minutes ahead
        predicted_value = slope * future_timestamp + intercept

        # Trend analysis result
        result = {
            "trend_direction": trend_direction
            "trend_strength": trend_strength
            "trend_confidence": trend_confidence
            "slope": slope
            "intercept": intercept
            "correlation": correlation
            "prediction": {
                "timestamp": future_timestamp
                "value": predicted_value
                "confidence": trend_confidence
            },
        }

        # Store in history
        self.trend_history.append(
            {"timestamp": datetime.now(), "source": data.source, "result": result}
        )

        return result

    def detect_trend_changes(self, data: TimeSeriesData) -> List[DetectionEvent]:
        """
        Detect significant changes in trends

        Args:
            data: Time series data to analyze

        Returns:
            List of trend change detection events
        """

        events = []

        # Current trend
        current_trend = self.analyze_trend(data)

        # Compare with recent trends
        if len(self.trend_history) > 0:
            recent_trend = self.trend_history[-1]["result"]

            # Check for significant slope change
            slope_change = abs(current_trend["slope"] - recent_trend["slope"])
            if slope_change > 0.1:  # Threshold for significant change

                event = DetectionEvent(
                    event_id=f"trend_change_{int(time.time())}",
                    detection_type=DetectionType.PATTERN_CHANGE
                    alert_level=AlertLevel.MEDIUM
                    confidence=min(1.0, slope_change),
                    description=f"Significant trend change detected: slope changed from {recent_trend['slope']:.3f} to {current_trend['slope']:.3f}",
                    data_source="trend_analysis",
                    context={
                        "previous_trend": recent_trend
                        "current_trend": current_trend
                        "slope_change": slope_change
                    },
                )

                events.append(event)

        return events
class PatternRecognizer:
    """Auto-generated class."""
    pass
    """
    Recognizes patterns in data for predictive analysis
    """

    def __init__(self, pattern_window: int = 50):
        try:
            self.settings = get_api_settings()
        except Exception as e:
            logger.warning(f"API settings loading failed: {e}. Using safe fallback.")
            from ..utils.robust_config import safe_get_api_settings

            self.settings = safe_get_api_settings()
        logger.debug(f"   Environment: {self.settings.environment}")
        self.pattern_window = pattern_window
        self.known_patterns = {}
        self.pattern_history = deque(maxlen=1000)

    def recognize_patterns(self, data: TimeSeriesData) -> Dict[str, Any]:
        """
        Recognize patterns in time series data

        Args:
            data: Time series data to analyze

        Returns:
            Pattern recognition results
        """

        if len(data.values) < self.pattern_window:
            return {
                "patterns_found": [],
                "pattern_confidence": 0.0
                "next_expected_pattern": None
            }

        # Extract recent window
        recent_values = data.values[-self.pattern_window :]

        # Detect periodic patterns
        periodic_patterns = self._detect_periodic_patterns(recent_values)

        # Detect shape patterns
        shape_patterns = self._detect_shape_patterns(recent_values)

        # Combine results
        all_patterns = periodic_patterns + shape_patterns

        result = {
            "patterns_found": all_patterns
            "pattern_confidence": (
                np.mean([p["confidence"] for p in all_patterns])
                if all_patterns
                else 0.0
            ),
            "next_expected_pattern": self._predict_next_pattern(all_patterns),
            "pattern_count": len(all_patterns),
        }

        # Store in history
        self.pattern_history.append(
            {"timestamp": datetime.now(), "source": data.source, "result": result}
        )

        return result

    def _detect_periodic_patterns(self, values: List[float]) -> List[Dict[str, Any]]:
        """Detect periodic patterns in data"""
        patterns = []

        # Simple autocorrelation-based period detection
        for period in range(2, min(20, len(values) // 2)):
            correlation = self._compute_autocorrelation(values, period)

            if correlation > 0.7:  # Strong periodic pattern
                patterns.append(
                    {
                        "type": "periodic",
                        "period": period
                        "confidence": correlation
                        "description": f"Periodic pattern with period {period}",
                    }
                )

        return patterns

    def _detect_shape_patterns(self, values: List[float]) -> List[Dict[str, Any]]:
        """Detect shape patterns in data"""
        patterns = []

        # Detect peaks and valleys
        peaks = self._find_peaks(values)
        valleys = self._find_valleys(values)

        if peaks:
            patterns.append(
                {
                    "type": "peaks",
                    "count": len(peaks),
                    "confidence": min(1.0, len(peaks) / 10),
                    "description": f"Found {len(peaks)} peaks in data",
                }
            )

        if valleys:
            patterns.append(
                {
                    "type": "valleys",
                    "count": len(valleys),
                    "confidence": min(1.0, len(valleys) / 10),
                    "description": f"Found {len(valleys)} valleys in data",
                }
            )

        return patterns

    def _compute_autocorrelation(self, values: List[float], lag: int) -> float:
        """Compute autocorrelation at given lag"""
        if lag >= len(values):
            return 0.0

        n = len(values) - lag
        if n <= 0:
            return 0.0

        mean_val = np.mean(values)

        # Compute autocorrelation
        numerator = sum(
            (values[i] - mean_val) * (values[i + lag] - mean_val) for i in range(n)
        )
        denominator = sum((values[i] - mean_val) ** 2 for i in range(len(values)))

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def _find_peaks(self, values: List[float]) -> List[int]:
        """Find peaks in data"""
        peaks = []
        for i in range(1, len(values) - 1):
            if values[i] > values[i - 1] and values[i] > values[i + 1]:
                peaks.append(i)
        return peaks

    def _find_valleys(self, values: List[float]) -> List[int]:
        """Find valleys in data"""
        valleys = []
        for i in range(1, len(values) - 1):
            if values[i] < values[i - 1] and values[i] < values[i + 1]:
                valleys.append(i)
        return valleys

    def _predict_next_pattern(
        self, patterns: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Predict next pattern based on found patterns"""
        if not patterns:
            return None

        # Find most confident pattern
        best_pattern = max(patterns, key=lambda p: p["confidence"])

        if best_pattern["type"] == "periodic":
            return {
                "type": "periodic_continuation",
                "expected_period": best_pattern["period"],
                "confidence": best_pattern["confidence"]
                * 0.8,  # Slightly lower confidence for prediction
                "description": f"Expect periodic pattern to continue with period {best_pattern['period']}",
            }

        return None
class ThresholdManager:
    """Auto-generated class."""
    pass
    """
    Manages adaptive thresholds for detection systems
    """

    def __init__(self, initial_threshold: float = 0.5, adaptation_rate: float = 0.1):
        try:
            self.settings = get_api_settings()
        except Exception as e:
            logger.warning(f"API settings loading failed: {e}. Using safe fallback.")
            from ..utils.robust_config import safe_get_api_settings

            self.settings = safe_get_api_settings()
        logger.debug(f"   Environment: {self.settings.environment}")
        self.initial_threshold = initial_threshold
        self.adaptation_rate = adaptation_rate
        self.thresholds = {}
        self.performance_history = deque(maxlen=100)

    def get_threshold(self, detection_type: DetectionType) -> float:
        """Get current threshold for detection type"""
        return self.thresholds.get(detection_type, self.initial_threshold)

    def update_threshold(
        self, detection_type: DetectionType, performance_feedback: Dict[str, Any]
    ):
        """
        Update threshold based on performance feedback

        Args:
            detection_type: Type of detection
            performance_feedback: Feedback containing false positives, false negatives, etc.
        """

        current_threshold = self.get_threshold(detection_type)

        # Calculate adjustment based on performance
        false_positive_rate = performance_feedback.get("false_positive_rate", 0.0)
        false_negative_rate = performance_feedback.get("false_negative_rate", 0.0)

        # Adjust threshold
        if false_positive_rate > 0.1:  # Too many false positives
            new_threshold = current_threshold + self.adaptation_rate
        elif false_negative_rate > 0.1:  # Too many false negatives
            new_threshold = current_threshold - self.adaptation_rate
        else:
            new_threshold = current_threshold

        # Clamp threshold
        new_threshold = max(0.1, min(0.9, new_threshold))

        # Update threshold
        self.thresholds[detection_type] = new_threshold

        # Record performance
        self.performance_history.append(
            {
                "timestamp": datetime.now(),
                "detection_type": detection_type.value
                "old_threshold": current_threshold
                "new_threshold": new_threshold
                "performance": performance_feedback
            }
        )

    def get_threshold_history(self) -> List[Dict[str, Any]]:
        """Get threshold adaptation history"""
        return list(self.performance_history)
class ProactiveDetector:
    """Auto-generated class."""
    pass
    """
    Main Proactive Detector Engine

    Coordinates multiple detection systems for comprehensive
    predictive analysis and early warning capabilities.
    """

    def __init__(self, device: str = "cpu"):
        try:
            self.settings = get_api_settings()
        except Exception as e:
            logger.warning(f"API settings loading failed: {e}. Using safe fallback.")
            from ..utils.robust_config import safe_get_api_settings

            self.settings = safe_get_api_settings()
        logger.debug(f"   Environment: {self.settings.environment}")
        self.device = torch.device(device)

        # Detection components
        self.anomaly_detector = AnomalyDetector()
        self.trend_analyzer = TrendAnalyzer()
        self.pattern_recognizer = PatternRecognizer()
        self.threshold_manager = ThresholdManager()

        # Data management
        self.data_streams = {}
        self.detection_events = deque(maxlen=5000)
        self.active_alerts = {}

        # Prediction models
        self.prediction_models = {}

        # Metrics
        self.total_detections = 0
        self.false_positive_count = 0
        self.false_negative_count = 0
        self.detection_accuracy = 0.0

        # Threading
        self.detection_lock = threading.Lock()

        logger.info(f"Proactive Detector initialized on device: {device}")

    def add_data_stream(self, stream_id: str, initial_data: TimeSeriesData):
        """
        Add a new data stream for monitoring

        Args:
            stream_id: Unique identifier for the data stream
            initial_data: Initial time series data
        """

        with self.detection_lock:
            self.data_streams[stream_id] = {
                "data": initial_data
                "last_update": datetime.now(),
                "anomaly_detector": AnomalyDetector(),
                "detection_count": 0
                "last_detection": None
            }

            logger.info(f"Added data stream: {stream_id}")

    def update_data_stream(
        self, stream_id: str, new_value: float, timestamp: datetime = None
    ) -> List[DetectionEvent]:
        """
        Update a data stream with new data point

        Args:
            stream_id: Data stream identifier
            new_value: New data value
            timestamp: Timestamp of the data point

        Returns:
            List of detection events triggered
        """

        if timestamp is None:
            timestamp = datetime.now()

        events = []

        with self.detection_lock:
            if stream_id not in self.data_streams:
                logger.warning(f"Data stream {stream_id} not found")
                return events

            stream = self.data_streams[stream_id]

            # Update data
            stream["data"].timestamps.append(timestamp)
            stream["data"].values.append(new_value)
            stream["last_update"] = timestamp

            # Limit data size
            max_size = 1000
            if len(stream["data"].values) > max_size:
                stream["data"].timestamps = stream["data"].timestamps[-max_size:]
                stream["data"].values = stream["data"].values[-max_size:]

            # Run anomaly detection
            anomaly_event = stream["anomaly_detector"].add_data_point(
                new_value, timestamp
            )
            if anomaly_event:
                anomaly_event.context["stream_id"] = stream_id
                events.append(anomaly_event)
                stream["detection_count"] += 1
                stream["last_detection"] = timestamp

            # Run trend analysis
            if len(stream["data"].values) >= 10:
                trend_events = self.trend_analyzer.detect_trend_changes(stream["data"])
                for event in trend_events:
                    event.context["stream_id"] = stream_id
                    events.append(event)

            # Run pattern recognition
            if len(stream["data"].values) >= 20:
                pattern_result = self.pattern_recognizer.recognize_patterns(
                    stream["data"]
                )

                # Generate events for strong patterns
                for pattern in pattern_result["patterns_found"]:
                    if pattern["confidence"] > 0.8:
                        pattern_event = DetectionEvent(
                            event_id=f"pattern_{stream_id}_{int(time.time())}",
                            detection_type=DetectionType.PATTERN_CHANGE
                            alert_level=AlertLevel.INFO
                            confidence=pattern["confidence"],
                            description=f"Strong pattern detected: {pattern['description']}",
                            data_source="pattern_recognition",
                            timestamp=timestamp
                            context={
                                "stream_id": stream_id
                                "pattern": pattern
                                "pattern_result": pattern_result
                            },
                        )
                        events.append(pattern_event)

            # Store events
            for event in events:
                self.detection_events.append(event)
                self.total_detections += 1

                # Update active alerts
                if event.alert_level in [
                    AlertLevel.HIGH
                    AlertLevel.CRITICAL
                    AlertLevel.EMERGENCY
                ]:
                    self.active_alerts[event.event_id] = event

        return events

    def predict_future_values(
        self, stream_id: str, horizon: PredictionHorizon = PredictionHorizon.SHORT_TERM
    ) -> Dict[str, Any]:
        """
        Predict future values for a data stream

        Args:
            stream_id: Data stream identifier
            horizon: Prediction time horizon

        Returns:
            Prediction results
        """

        with self.detection_lock:
            if stream_id not in self.data_streams:
                return {"error": f"Data stream {stream_id} not found"}

            stream = self.data_streams[stream_id]
            data = stream["data"]

            if len(data.values) < 10:
                return {"error": "Insufficient data for prediction"}

            # Trend-based prediction
            trend_result = self.trend_analyzer.analyze_trend(data)

            # Pattern-based prediction
            pattern_result = self.pattern_recognizer.recognize_patterns(data)

            # Time horizon mapping
            horizon_minutes = {
                PredictionHorizon.IMMEDIATE: 1
                PredictionHorizon.SHORT_TERM: 5
                PredictionHorizon.MEDIUM_TERM: 30
                PredictionHorizon.LONG_TERM: 120
            }

            minutes_ahead = horizon_minutes.get(horizon, 5)

            # Generate predictions
            predictions = []
            current_time = data.timestamps[-1]

            for i in range(1, minutes_ahead + 1):
                future_time = current_time + timedelta(minutes=i)

                # Trend-based prediction
                if trend_result["prediction"]:
                    trend_value = trend_result["prediction"]["value"]
                    trend_confidence = trend_result["prediction"]["confidence"]
                else:
                    trend_value = data.values[-1]
                    trend_confidence = 0.5

                # Pattern-based adjustment
                pattern_adjustment = 0.0
                pattern_confidence = 0.0

                if pattern_result["next_expected_pattern"]:
                    pattern_confidence = pattern_result["next_expected_pattern"][
                        "confidence"
                    ]
                    # Simple pattern adjustment (would be more sophisticated in practice)
                    pattern_adjustment = np.sin(i * 0.1) * 0.1  # Placeholder

                # Combined prediction
                combined_value = trend_value + pattern_adjustment
                combined_confidence = (trend_confidence + pattern_confidence) / 2

                predictions.append(
                    {
                        "timestamp": future_time
                        "predicted_value": combined_value
                        "confidence": combined_confidence
                        "trend_component": trend_value
                        "pattern_component": pattern_adjustment
                    }
                )

            return {
                "stream_id": stream_id
                "prediction_horizon": horizon.value
                "predictions": predictions
                "trend_analysis": trend_result
                "pattern_analysis": pattern_result
                "prediction_timestamp": datetime.now(),
            }

    def get_active_alerts(
        self, alert_level: Optional[AlertLevel] = None
    ) -> List[DetectionEvent]:
        """
        Get currently active alerts

        Args:
            alert_level: Filter by alert level

        Returns:
            List of active alerts
        """

        with self.detection_lock:
            alerts = list(self.active_alerts.values())

            if alert_level:
                alerts = [alert for alert in alerts if alert.alert_level == alert_level]

            return alerts

    def resolve_alert(self, event_id: str, resolution_notes: str = ""):
        """
        Resolve an active alert

        Args:
            event_id: Event identifier
            resolution_notes: Optional resolution notes
        """

        with self.detection_lock:
            if event_id in self.active_alerts:
                alert = self.active_alerts[event_id]
                alert.resolution_status = "resolved"
                alert.context["resolution_notes"] = resolution_notes
                alert.context["resolution_timestamp"] = datetime.now()

                del self.active_alerts[event_id]
                logger.info(f"Resolved alert: {event_id}")

    def get_detection_summary(self) -> Dict[str, Any]:
        """Get summary of detection activity"""

        with self.detection_lock:
            recent_events = [
                e
                for e in self.detection_events
                if (datetime.now() - e.timestamp).total_seconds() < 3600
            ]

            # Event type distribution
            type_distribution = {}
            for event in recent_events:
                event_type = event.detection_type.value
                type_distribution[event_type] = type_distribution.get(event_type, 0) + 1

            # Alert level distribution
            alert_distribution = {}
            for event in recent_events:
                alert_level = event.alert_level.value
                alert_distribution[alert_level] = (
                    alert_distribution.get(alert_level, 0) + 1
                )

            return {
                "total_detections": self.total_detections
                "recent_detections": len(recent_events),
                "active_alerts_count": len(self.active_alerts),
                "data_streams_count": len(self.data_streams),
                "detection_accuracy": self.detection_accuracy
                "false_positive_rate": self.false_positive_count
                / max(self.total_detections, 1),
                "type_distribution": type_distribution
                "alert_distribution": alert_distribution
                "threshold_adaptations": len(
                    self.threshold_manager.get_threshold_history()
                ),
            }

    def get_engine_status(self) -> Dict[str, Any]:
        """Get current engine status and metrics"""

        with self.detection_lock:
            return {
                "status": "operational",
                "device": str(self.device),
                "total_detections": self.total_detections
                "active_alerts": len(self.active_alerts),
                "data_streams": len(self.data_streams),
                "detection_accuracy": self.detection_accuracy
                "false_positive_count": self.false_positive_count
                "false_negative_count": self.false_negative_count
                "detection_events_history": len(self.detection_events),
                "prediction_models": len(self.prediction_models),
                "threshold_adaptations": len(
                    self.threshold_manager.get_threshold_history()
                ),
                "last_updated": datetime.now().isoformat(),
            }

    def reset_engine(self):
        """Reset engine state"""

        with self.detection_lock:
            self.data_streams.clear()
            self.detection_events.clear()
            self.active_alerts.clear()
            self.prediction_models.clear()

            self.total_detections = 0
            self.false_positive_count = 0
            self.false_negative_count = 0
            self.detection_accuracy = 0.0

            # Reset components
            self.anomaly_detector = AnomalyDetector()
            self.trend_analyzer = TrendAnalyzer()
            self.pattern_recognizer = PatternRecognizer()
            self.threshold_manager = ThresholdManager()

            logger.info("Proactive Detector reset")


# Factory function for easy instantiation
def create_proactive_detector(device: str = "cpu") -> ProactiveDetector:
    """
    Create and initialize Proactive Detector

    Args:
        device: Computing device ("cpu" or "cuda")

    Returns:
        Initialized Proactive Detector
    """
    return ProactiveDetector(device=device)
