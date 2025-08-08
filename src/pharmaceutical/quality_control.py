import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Add the backend directory to the sys.path to allow for engine imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from engines.thermodynamic_engine import ThermodynamicEngine
class QualityControlSystem:
    """Auto-generated class."""
    pass
    """
    A pharmaceutical quality control system implementing Statistical Process Control (SPC)
    with integration to thermodynamic analysis for process stability assessment.
    """

    def __init__(self):
        self.control_limits = {}
        self.quality_history = {}
        self.thermodynamic_engine = ThermodynamicEngine()

    def establish_control_limits(
        self, attribute: str, historical_data: List[float]
    ) -> Dict[str, Any]:
        """
        Calculates statistical process control limits using 3-sigma methodology.

        Args:
            attribute: Name of the quality attribute
            historical_data: Historical measurement data

        Returns:
            Dictionary containing control limits and statistical parameters
        """
        if not historical_data or len(historical_data) < 2:
            raise ValueError(
                "At least 2 data points required for control limit establishment"
            )

        data_array = np.array(historical_data)

        # Calculate statistical parameters
        mean_value = np.mean(data_array)
        std_dev = np.std(data_array, ddof=1)  # Sample standard deviation

        if std_dev == 0:
            raise ValueError(
                "Cannot establish control limits - all data points are identical"
            )

        # Calculate 3-sigma control limits
        ucl = mean_value + 3 * std_dev  # Upper Control Limit
        lcl = mean_value - 3 * std_dev  # Lower Control Limit

        # Calculate 2-sigma warning limits
        uwl = mean_value + 2 * std_dev  # Upper Warning Limit
        lwl = mean_value - 2 * std_dev  # Lower Warning Limit

        control_limits = {
            "attribute": attribute,
            "center_line": mean_value,
            "ucl": ucl,
            "lcl": lcl,
            "uwl": uwl,
            "lwl": lwl,
            "std_dev": std_dev,
            "n_samples": len(historical_data),
            "established_date": datetime.now().isoformat(),
        }

        self.control_limits[attribute] = control_limits

        # Initialize quality history for this attribute
        self.quality_history[attribute] = list(historical_data)

        return control_limits

    def monitor_quality_point(self, attribute: str, value: float) -> Dict[str, Any]:
        """
        Evaluates a single measurement against established control limits.

        Args:
            attribute: Quality attribute being monitored
            value: Measured value

        Returns:
            Monitoring result with status and recommendations
        """
        if attribute not in self.control_limits:
            raise ValueError(
                f"Control limits not established for attribute '{attribute}'"
            )

        limits = self.control_limits[attribute]

        # Add to quality history
        if attribute not in self.quality_history:
            self.quality_history[attribute] = []
        self.quality_history[attribute].append(value)

        # Determine status
        status = "IN_CONTROL"
        alerts = []

        # Check control limits
        if value > limits["ucl"]:
            status = "OUT_OF_CONTROL"
            alerts.append("ABOVE_UCL")
        elif value < limits["lcl"]:
            status = "OUT_OF_CONTROL"
            alerts.append("BELOW_LCL")
        elif value > limits["uwl"]:
            alerts.append("ABOVE_WARNING")
        elif value < limits["lwl"]:
            alerts.append("BELOW_WARNING")

        # Check for trending patterns (last 7 points)
        recent_data = self.quality_history[attribute][-7:]
        if len(recent_data) >= 7:
            trend = self._detect_trend(recent_data)
            if trend:
                alerts.append(f"TRENDING_{trend}")

        # Calculate process entropy using thermodynamic engine
        process_entropy = 0.0
        if len(self.quality_history[attribute]) >= 10:
            # Convert quality measurements to embedding-like vectors for thermodynamic analysis
            recent_measurements = self.quality_history[attribute][-10:]
            measurement_vectors = [
                np.array([val, val**2, np.sin(val)]) for val in recent_measurements
            ]
            process_entropy = self.thermodynamic_engine.calculate_semantic_temperature(
                measurement_vectors
            )

        result = {
            "attribute": attribute,
            "value": value,
            "status": status,
            "alerts": alerts,
            "center_line": limits["center_line"],
            "deviation_from_center": abs(value - limits["center_line"]),
            "process_entropy": process_entropy,
            "timestamp": datetime.now().isoformat(),
        }

        return result

    def analyze_process_capability(
        self, attribute: str, specification_limits: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Calculates process capability indices (Cp, Cpk).

        Args:
            attribute: Quality attribute to analyze
            specification_limits: Dictionary with 'usl' (upper) and 'lsl' (lower) specification limits

        Returns:
            Process capability analysis results
        """
        if attribute not in self.control_limits:
            raise ValueError(
                f"Control limits not established for attribute '{attribute}'"
            )

        if "usl" not in specification_limits or "lsl" not in specification_limits:
            raise ValueError(
                "Both 'usl' and 'lsl' must be provided in specification_limits"
            )

        limits = self.control_limits[attribute]
        usl = specification_limits["usl"]
        lsl = specification_limits["lsl"]

        mean = limits["center_line"]
        std_dev = limits["std_dev"]

        # Calculate Cp (process capability)
        cp = (usl - lsl) / (6 * std_dev)

        # Calculate Cpk (process capability index accounting for centering)
        cpu = (usl - mean) / (3 * std_dev)  # Upper capability
        cpl = (mean - lsl) / (3 * std_dev)  # Lower capability
        cpk = min(cpu, cpl)

        # Interpret capability
        if cpk >= 1.33:
            capability_assessment = "EXCELLENT"
        elif cpk >= 1.0:
            capability_assessment = "ADEQUATE"
        elif cpk >= 0.67:
            capability_assessment = "MARGINAL"
        else:
            capability_assessment = "INADEQUATE"

        return {
            "attribute": attribute,
            "cp": cp,
            "cpk": cpk,
            "cpu": cpu,
            "cpl": cpl,
            "capability_assessment": capability_assessment,
            "specification_limits": specification_limits,
            "process_mean": mean,
            "process_std_dev": std_dev,
        }

    def _detect_trend(self, data: List[float]) -> str:
        """
        Detects trending patterns in quality data.

        Args:
            data: List of recent quality measurements

        Returns:
            Trend direction ('UP', 'DOWN') or empty string if no trend
        """
        if len(data) < 5:
            return ""

        # Calculate linear regression slope
        x = np.arange(len(data))
        y = np.array(data)

        # Simple linear regression
        n = len(data)
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (
            n * np.sum(x**2) - (np.sum(x)) ** 2
        )

        # Determine trend significance
        if abs(slope) > 0.1:  # Threshold for significant trend
            return "UP" if slope > 0 else "DOWN"

        return ""
