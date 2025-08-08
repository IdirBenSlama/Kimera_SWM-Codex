"""
Pharmaceutical Validation Engine

Comprehensive validation framework for pharmaceutical development integrating
with Kimera's scientific validation principles and cognitive fidelity goals.

Provides end-to-end validation from raw materials to finished products.
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy import stats

from ...utils.gpu_foundation import GPUFoundation
from ...utils.kimera_exceptions import KimeraBaseException as KimeraException
from ...utils.kimera_logger import get_logger
from ..analysis.dissolution_analyzer import DissolutionAnalyzer, DissolutionComparison
from ..core.kcl_testing_engine import FormulationPrototype, KClTestingEngine
from ..protocols.usp_protocols import USPProtocolEngine, USPTestResult

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Auto-generated class."""
    pass
    """Comprehensive validation result."""

    validation_id: str
    validation_type: str
    status: str  # PASSED, FAILED, WARNING, PENDING
    confidence_score: float
    critical_failures: List[str]
    warnings: List[str]
    test_results: Dict[str, Any]
    compliance_assessment: Dict[str, Any]
    timestamp: str


@dataclass
class RegulatoryCompliance:
    """Auto-generated class."""
    pass
    """Regulatory compliance assessment."""

    fda_ready: bool
    ema_ready: bool
    ich_compliant: bool
    usp_compliant: bool
    compliance_score: float
    regulatory_gaps: List[str]
    submission_readiness: str


@dataclass
class QualityProfile:
    """Auto-generated class."""
    pass
    """Product quality profile."""

    batch_id: str
    quality_score: float
    critical_quality_attributes: Dict[str, float]
    specification_compliance: Dict[str, bool]
    risk_assessment: Dict[str, str]
    shelf_life_prediction: Optional[float]
class QualityControlMonitor:
    """Auto-generated class."""
    pass
    """Real-time quality control monitoring system."""

    def __init__(self):
        self.logger = get_logger(__name__)
        self.monitoring_thresholds = self._load_monitoring_thresholds()
        self.process_parameters = {}
        self.quality_trends = {}

    def _load_monitoring_thresholds(self) -> Dict[str, Any]:
        """Load quality control monitoring thresholds."""
        return {
            "dissolution_variability": {"warning": 5.0, "critical": 10.0},
            "content_uniformity": {"warning": 2.5, "critical": 5.0},
            "hardness_variation": {"warning": 15.0, "critical": 25.0},
            "disintegration_time": {"warning": 10.0, "critical": 20.0},
            "moisture_content": {"warning": 1.5, "critical": 2.0},
            "particle_size_drift": {"warning": 10.0, "critical": 20.0},
        }

    def monitor_batch_quality(
        self, batch_data: Dict[str, Any], historical_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Monitor batch quality against historical trends and thresholds.

        Args:
            batch_data: Current batch test results
            historical_data: Historical batch data for trend analysis

        Returns:
            Dict[str, Any]: Quality monitoring results
        """
        try:
            self.logger.info("📊 Monitoring batch quality in real-time...")

            monitoring_result = {
                "batch_id": batch_data.get("batch_id", "Unknown"),
                "timestamp": datetime.now().isoformat(),
                "quality_score": 0.0,
                "alerts": [],
                "trends": {},
                "recommendations": [],
            }

            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(batch_data)

            # Trend analysis against historical data
            trend_analysis = self._analyze_quality_trends(
                quality_metrics, historical_data
            )
            monitoring_result["trends"] = trend_analysis

            # Threshold monitoring
            alerts = self._check_quality_thresholds(quality_metrics)
            monitoring_result["alerts"] = alerts

            # Calculate overall quality score
            quality_score = self._calculate_overall_quality_score(
                quality_metrics, trend_analysis
            )
            monitoring_result["quality_score"] = quality_score

            # Generate recommendations
            recommendations = self._generate_quality_recommendations(
                quality_metrics, trend_analysis, alerts
            )
            monitoring_result["recommendations"] = recommendations

            # Log monitoring results
            alert_level = (
                "CRITICAL"
                if any(a["level"] == "CRITICAL" for a in alerts)
                else "NORMAL"
            )
            self.logger.info(
                f"✅ Quality monitoring completed - Score: {quality_score:.2f}, Level: {alert_level}"
            )

            return monitoring_result

        except Exception as e:
            self.logger.error(f"❌ Quality monitoring failed: {e}")
            raise KimeraException(f"Quality monitoring failed: {e}")

    def _calculate_quality_metrics(
        self, batch_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate key quality metrics from batch data."""
        metrics = {}

        # Dissolution variability
        if "dissolution_profile" in batch_data:
            dissolution_data = batch_data["dissolution_profile"]
            if len(dissolution_data) > 1:
                cv = np.std(dissolution_data) / np.mean(dissolution_data) * 100
                metrics["dissolution_cv"] = cv

        # Content uniformity
        if "content_uniformity" in batch_data:
            content_data = batch_data["content_uniformity"]
            if len(content_data) > 1:
                rsd = np.std(content_data) / np.mean(content_data) * 100
                metrics["content_rsd"] = rsd

        # Other quality parameters
        metrics.update(
            {
                "hardness_avg": batch_data.get("hardness_average", 0.0),
                "hardness_rsd": batch_data.get("hardness_rsd", 0.0),
                "disintegration_time": batch_data.get("disintegration_time", 0.0),
                "moisture_content": batch_data.get("moisture_content", 0.0),
                "particle_size_d50": batch_data.get("particle_size_d50", 0.0),
            }
        )

        return metrics

    def _analyze_quality_trends(
        self, current_metrics: Dict[str, float], historical_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze quality trends against historical data."""
        trends = {}

        if len(historical_data) < 5:
            return {
                "status": "INSUFFICIENT_DATA",
                "message": "Need at least 5 historical batches",
            }

        # Extract historical metrics
        historical_metrics = {}
        for metric in current_metrics.keys():
            historical_values = []
            for batch in historical_data:
                if metric in batch:
                    historical_values.append(batch[metric])

            if len(historical_values) >= 3:
                historical_metrics[metric] = historical_values

        # Trend analysis for each metric
        for metric, current_value in current_metrics.items():
            if metric in historical_metrics:
                historical_values = historical_metrics[metric]

                # Calculate trend statistics
                mean_historical = np.mean(historical_values)
                std_historical = np.std(historical_values)

                # Z-score analysis
                z_score = (current_value - mean_historical) / max(std_historical, 1e-6)

                # Trend direction
                recent_values = historical_values[-5:]  # Last 5 batches
                if len(recent_values) >= 3:
                    slope, _, r_value, _, _ = stats.linregress(
                        range(len(recent_values)), recent_values
                    )
                    trend_direction = (
                        "INCREASING"
                        if slope > 0.1
                        else "DECREASING" if slope < -0.1 else "STABLE"
                    )
                else:
                    trend_direction = "UNKNOWN"

                trends[metric] = {
                    "current_value": current_value,
                    "historical_mean": mean_historical,
                    "historical_std": std_historical,
                    "z_score": z_score,
                    "trend_direction": trend_direction,
                    "deviation_from_norm": abs(z_score) > 2.0,
                }

        return trends

    def _check_quality_thresholds(
        self, quality_metrics: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Check quality metrics against monitoring thresholds."""
        alerts = []

        for metric, value in quality_metrics.items():
            if metric in self.monitoring_thresholds:
                thresholds = self.monitoring_thresholds[metric]

                if value > thresholds["critical"]:
                    alerts.append(
                        {
                            "metric": metric,
                            "level": "CRITICAL",
                            "value": value,
                            "threshold": thresholds["critical"],
                            "message": f'{metric} exceeds critical threshold: {value:.2f} > {thresholds["critical"]}',
                        }
                    )
                elif value > thresholds["warning"]:
                    alerts.append(
                        {
                            "metric": metric,
                            "level": "WARNING",
                            "value": value,
                            "threshold": thresholds["warning"],
                            "message": f'{metric} exceeds warning threshold: {value:.2f} > {thresholds["warning"]}',
                        }
                    )

        return alerts

    def _calculate_overall_quality_score(
        self, quality_metrics: Dict[str, float], trend_analysis: Dict[str, Any]
    ) -> float:
        """Calculate overall quality score (0-100)."""
        score = 100.0

        # Deduct points for threshold violations
        for metric, value in quality_metrics.items():
            if metric in self.monitoring_thresholds:
                thresholds = self.monitoring_thresholds[metric]

                if value > thresholds["critical"]:
                    score -= 25.0  # Major deduction for critical violations
                elif value > thresholds["warning"]:
                    score -= 10.0  # Moderate deduction for warnings

        # Deduct points for negative trends
        for metric_trend in trend_analysis.values():
            if isinstance(metric_trend, dict) and metric_trend.get(
                "deviation_from_norm", False
            ):
                score -= 5.0  # Minor deduction for trend deviations

        return max(0.0, score)

    def _generate_quality_recommendations(
        self,
        quality_metrics: Dict[str, float],
        trend_analysis: Dict[str, Any],
        alerts: List[Dict[str, Any]],
    ) -> List[str]:
        """Generate quality improvement recommendations."""
        recommendations = []

        # Recommendations based on alerts
        for alert in alerts:
            metric = alert["metric"]

            if metric == "dissolution_cv" and alert["level"] in ["WARNING", "CRITICAL"]:
                recommendations.append(
                    "Review mixing parameters and coating uniformity to reduce dissolution variability"
                )

            elif metric == "content_rsd" and alert["level"] in ["WARNING", "CRITICAL"]:
                recommendations.append(
                    "Investigate blending process and check for segregation issues"
                )

            elif metric == "hardness_rsd" and alert["level"] in ["WARNING", "CRITICAL"]:
                recommendations.append(
                    "Review compression parameters and tablet weight control"
                )

            elif metric == "moisture_content" and alert["level"] in [
                "WARNING",
                "CRITICAL",
            ]:
                recommendations.append(
                    "Check drying conditions and storage environment"
                )

        # Recommendations based on trends
        for metric, trend_data in trend_analysis.items():
            if (
                isinstance(trend_data, dict)
                and trend_data.get("trend_direction") == "INCREASING"
            ):
                if metric == "dissolution_cv":
                    recommendations.append(
                        "Dissolution variability is trending upward - implement process control improvements"
                    )
                elif metric == "moisture_content":
                    recommendations.append(
                        "Moisture content is increasing - review environmental controls"
                    )

        if not recommendations:
            recommendations.append(
                "Quality metrics are within acceptable ranges - continue current process controls"
            )

        return recommendations
class PredictiveQualityAnalytics:
    """Auto-generated class."""
    pass
    """Predictive analytics for quality control."""

    def __init__(self, use_gpu: bool = True):
        self.logger = get_logger(__name__)
        self.use_gpu = use_gpu
        self.device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )

        # Initialize predictive models
        self.prediction_models = {}
        self.model_performance = {}

    def predict_quality_issues(
        self,
        current_batch_data: Dict[str, Any],
        process_parameters: Dict[str, Any],
        prediction_horizon: int = 5,
    ) -> Dict[str, Any]:
        """
        Predict potential quality issues based on current trends.

        Args:
            current_batch_data: Current batch quality data
            process_parameters: Current process parameters
            prediction_horizon: Number of future batches to predict

        Returns:
            Dict[str, Any]: Quality predictions and risk assessment
        """
        try:
            self.logger.info(
                f"🔮 Predicting quality issues for next {prediction_horizon} batches..."
            )

            predictions = {
                "prediction_horizon": prediction_horizon,
                "quality_risk_score": 0.0,
                "predicted_issues": [],
                "preventive_actions": [],
                "confidence_intervals": {},
            }

            # Simple predictive analysis based on current trends
            # In a real implementation, this would use sophisticated ML models

            # Risk factors analysis
            risk_factors = self._analyze_risk_factors(
                current_batch_data, process_parameters
            )
            predictions["risk_factors"] = risk_factors

            # Calculate overall risk score
            risk_score = self._calculate_quality_risk_score(risk_factors)
            predictions["quality_risk_score"] = risk_score

            # Predict specific issues
            predicted_issues = self._predict_specific_issues(risk_factors, risk_score)
            predictions["predicted_issues"] = predicted_issues

            # Generate preventive actions
            preventive_actions = self._generate_preventive_actions(predicted_issues)
            predictions["preventive_actions"] = preventive_actions

            self.logger.info(
                f"✅ Quality prediction completed - Risk Score: {risk_score:.2f}"
            )

            return predictions

        except Exception as e:
            self.logger.error(f"❌ Quality prediction failed: {e}")
            raise KimeraException(f"Quality prediction failed: {e}")

    def _analyze_risk_factors(
        self, batch_data: Dict[str, Any], process_params: Dict[str, Any]
    ) -> Dict[str, float]:
        """Analyze risk factors for quality issues."""
        risk_factors = {}

        # Dissolution risk factors
        dissolution_cv = batch_data.get("dissolution_cv", 0.0)
        if dissolution_cv > 3.0:
            risk_factors["dissolution_risk"] = min(dissolution_cv / 10.0, 1.0)

        # Content uniformity risk
        content_rsd = batch_data.get("content_rsd", 0.0)
        if content_rsd > 2.0:
            risk_factors["content_uniformity_risk"] = min(content_rsd / 5.0, 1.0)

        # Process parameter risks
        compression_force = process_params.get("compression_force", 15.0)
        if compression_force < 10.0 or compression_force > 25.0:
            risk_factors["compression_risk"] = 0.3

        environmental_humidity = process_params.get("humidity", 50.0)
        if environmental_humidity > 65.0:
            risk_factors["environmental_risk"] = min(
                (environmental_humidity - 65.0) / 20.0, 1.0
            )

        return risk_factors

    def _calculate_quality_risk_score(self, risk_factors: Dict[str, float]) -> float:
        """Calculate overall quality risk score (0-1)."""
        if not risk_factors:
            return 0.0

        # Weighted average of risk factors
        weights = {
            "dissolution_risk": 0.3,
            "content_uniformity_risk": 0.25,
            "compression_risk": 0.2,
            "environmental_risk": 0.15,
            "default": 0.1,
        }

        weighted_score = 0.0
        total_weight = 0.0

        for factor, risk_value in risk_factors.items():
            weight = weights.get(factor, weights["default"])
            weighted_score += risk_value * weight
            total_weight += weight

        return weighted_score / max(total_weight, 1e-6)

    def _predict_specific_issues(
        self, risk_factors: Dict[str, float], overall_risk: float
    ) -> List[Dict[str, Any]]:
        """Predict specific quality issues."""
        issues = []

        # High-risk threshold
        high_risk_threshold = 0.6

        for factor, risk_value in risk_factors.items():
            if risk_value > high_risk_threshold:
                issue = {
                    "issue_type": factor.replace("_risk", ""),
                    "probability": risk_value,
                    "severity": "HIGH" if risk_value > 0.8 else "MEDIUM",
                    "estimated_impact": self._estimate_issue_impact(factor, risk_value),
                }
                issues.append(issue)

        return issues

    def _estimate_issue_impact(self, risk_factor: str, risk_value: float) -> str:
        """Estimate the impact of a quality issue."""
        impact_mapping = {
            "dissolution_risk": "Dissolution profile deviation, potential regulatory issues",
            "content_uniformity_risk": "Dose variability, efficacy concerns",
            "compression_risk": "Tablet hardness variation, disintegration issues",
            "environmental_risk": "Moisture uptake, stability concerns",
        }

        return impact_mapping.get(risk_factor, "General quality degradation")

    def _generate_preventive_actions(
        self, predicted_issues: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate preventive actions based on predicted issues."""
        actions = []

        for issue in predicted_issues:
            issue_type = issue["issue_type"]

            if issue_type == "dissolution":
                actions.append(
                    "Implement tighter coating process controls and monitor spray rate variability"
                )
            elif issue_type == "content_uniformity":
                actions.append(
                    "Review blending parameters and check for powder segregation"
                )
            elif issue_type == "compression":
                actions.append(
                    "Calibrate tablet press and verify compression force consistency"
                )
            elif issue_type == "environmental":
                actions.append(
                    "Enhance environmental controls and monitor humidity levels"
                )

        if not actions:
            actions.append(
                "Continue current monitoring protocols - no immediate preventive actions required"
            )

        return actions
class QualityAlertSystem:
    """Auto-generated class."""
    pass
    """Quality alert and notification system."""

    def __init__(self):
        self.logger = get_logger(__name__)
        self.alert_thresholds = self._load_alert_thresholds()
        self.notification_history = []

    def _load_alert_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Load alert thresholds for quality parameters."""
        return {
            "dissolution_cv": {"warning": 5.0, "critical": 10.0},
            "content_rsd": {"warning": 2.5, "critical": 5.0},
            "quality_score": {"warning": 80.0, "critical": 70.0},
            "risk_score": {"warning": 0.6, "critical": 0.8},
        }

    def process_quality_alerts(
        self, monitoring_result: Dict[str, Any], prediction_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process and generate quality alerts."""
        try:
            alert_summary = {
                "timestamp": datetime.now().isoformat(),
                "active_alerts": [],
                "alert_level": "NORMAL",
                "notifications_sent": 0,
                "alert_actions": [],
            }

            # Process monitoring alerts
            monitoring_alerts = monitoring_result.get("alerts", [])
            for alert in monitoring_alerts:
                alert_summary["active_alerts"].append(
                    {
                        "source": "MONITORING",
                        "type": alert["metric"],
                        "level": alert["level"],
                        "message": alert["message"],
                    }
                )

            # Process predictive alerts
            risk_score = prediction_result.get("quality_risk_score", 0.0)
            if risk_score > self.alert_thresholds["risk_score"]["critical"]:
                alert_summary["active_alerts"].append(
                    {
                        "source": "PREDICTION",
                        "type": "quality_risk",
                        "level": "CRITICAL",
                        "message": f"High quality risk predicted: {risk_score:.2f}",
                    }
                )
            elif risk_score > self.alert_thresholds["risk_score"]["warning"]:
                alert_summary["active_alerts"].append(
                    {
                        "source": "PREDICTION",
                        "type": "quality_risk",
                        "level": "WARNING",
                        "message": f"Elevated quality risk predicted: {risk_score:.2f}",
                    }
                )

            # Determine overall alert level
            if any(
                alert["level"] == "CRITICAL" for alert in alert_summary["active_alerts"]
            ):
                alert_summary["alert_level"] = "CRITICAL"
            elif any(
                alert["level"] == "WARNING" for alert in alert_summary["active_alerts"]
            ):
                alert_summary["alert_level"] = "WARNING"

            # Generate alert actions
            alert_actions = self._generate_alert_actions(alert_summary["active_alerts"])
            alert_summary["alert_actions"] = alert_actions

            self.logger.info(
                f"🚨 Quality alerts processed: {len(alert_summary['active_alerts'])} alerts, Level: {alert_summary['alert_level']}"
            )

            return alert_summary

        except Exception as e:
            self.logger.error(f"❌ Alert processing failed: {e}")
            raise KimeraException(f"Alert processing failed: {e}")

    def _generate_alert_actions(self, active_alerts: List[Dict[str, Any]]) -> List[str]:
        """Generate recommended actions based on active alerts."""
        actions = []

        critical_alerts = [a for a in active_alerts if a["level"] == "CRITICAL"]
        warning_alerts = [a for a in active_alerts if a["level"] == "WARNING"]

        if critical_alerts:
            actions.append(
                "IMMEDIATE ACTION REQUIRED: Stop production and investigate critical quality issues"
            )
            actions.append("Notify quality assurance manager and production supervisor")
            actions.append("Quarantine current batch pending investigation")

        if warning_alerts:
            actions.append(
                "Increase monitoring frequency and prepare for process adjustments"
            )
            actions.append(
                "Review recent process parameters and identify potential causes"
            )

        if not active_alerts:
            actions.append(
                "Continue normal operations with standard monitoring protocols"
            )

        return actions
class PharmaceuticalValidator:
    """Auto-generated class."""
    pass
    """
    Comprehensive pharmaceutical validation engine.

    Integrates all testing components to provide end-to-end validation
    from raw materials through finished product testing and regulatory compliance.
    """

    def __init__(self, use_gpu: bool = True):
        """
        Initialize the pharmaceutical validator.

        Args:
            use_gpu: Whether to use GPU acceleration

        Raises:
            KimeraException: If initialization fails
        """
        self.logger = logger
        self.use_gpu = use_gpu

        # Initialize component engines
        try:
            self.kcl_engine = KClTestingEngine(use_gpu=use_gpu)
            self.usp_engine = USPProtocolEngine()
            self.dissolution_analyzer = DissolutionAnalyzer(use_gpu=use_gpu)
        except Exception as e:
            raise KimeraException(f"Failed to initialize validation engines: {e}")

        # Validation standards and thresholds
        self.validation_standards = self._load_validation_standards()
        self.validation_history = {}
        self.quality_profiles = {}

        # Real-time monitoring components
        self.quality_control_monitor = QualityControlMonitor()
        self.predictive_analytics = PredictiveQualityAnalytics(use_gpu=use_gpu)
        self.alert_system = QualityAlertSystem()

        # Continuous validation tracking
        self.validation_metrics = []
        self.trend_analysis = {}
        self.quality_score_history = []

        self.logger.info(
            "🔬 Pharmaceutical Validator initialized with real-time QC monitoring"
        )

    def _load_validation_standards(self) -> Dict[str, Any]:
        """Load comprehensive validation standards."""
        return {
            "critical_quality_attributes": {
                "assay": {"min": 90.0, "max": 110.0, "critical": True},
                "content_uniformity": {"acceptance_value": 15.0, "critical": True},
                "dissolution_f2": {"min": 50.0, "critical": True},
                "related_substances": {"max": 0.5, "critical": False},
                "moisture_content": {"max": 5.0, "critical": False},
            },
            "regulatory_requirements": {
                "fda": {
                    "dissolution_points": 4,
                    "stability_conditions": 3,
                    "batch_size_minimum": 100000,
                },
                "ema": {
                    "dissolution_points": 4,
                    "stability_conditions": 3,
                    "bioequivalence_required": True,
                },
                "ich": {
                    "stability_duration_months": 24,
                    "accelerated_testing_required": True,
                    "photostability_required": False,
                },
            },
            "validation_thresholds": {
                "confidence_minimum": 0.85,
                "quality_score_minimum": 0.80,
                "compliance_score_minimum": 0.95,
            },
        }

    def validate_complete_development(
        self,
        raw_materials: Dict[str, Any],
        formulation_data: Dict[str, Any],
        manufacturing_data: Dict[str, Any],
        testing_data: Dict[str, Any],
    ) -> ValidationResult:
        """
        Perform complete pharmaceutical development validation.

        Args:
            raw_materials: Raw material specifications and test data
            formulation_data: Formulation parameters and prototype data
            manufacturing_data: Manufacturing process parameters
            testing_data: All analytical testing data

        Returns:
            ValidationResult: Comprehensive validation assessment

        Raises:
            KimeraException: If validation fails
        """
        try:
            self.logger.info(
                "🧪 Starting complete pharmaceutical development validation..."
            )

            validation_id = f"PHARMA_VAL_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Phase 1: Raw Material Validation
            self.logger.info("📋 Phase 1: Raw Material Validation")
            rm_validation = self._validate_raw_materials(raw_materials)

            # Phase 2: Formulation Validation
            self.logger.info("🧪 Phase 2: Formulation Validation")
            formulation_validation = self._validate_formulation(formulation_data)

            # Phase 3: Manufacturing Process Validation
            self.logger.info("🏭 Phase 3: Manufacturing Process Validation")
            manufacturing_validation = self._validate_manufacturing(manufacturing_data)

            # Phase 4: Analytical Testing Validation
            self.logger.info("📊 Phase 4: Analytical Testing Validation")
            analytical_validation = self._validate_analytical_testing(testing_data)

            # Phase 5: Regulatory Compliance Assessment
            self.logger.info("📜 Phase 5: Regulatory Compliance Assessment")
            regulatory_assessment = self._assess_regulatory_compliance(
                {
                    "raw_materials": rm_validation,
                    "formulation": formulation_validation,
                    "manufacturing": manufacturing_validation,
                    "analytical": analytical_validation,
                }
            )

            # Consolidate results
            all_test_results = {
                "raw_materials": rm_validation,
                "formulation": formulation_validation,
                "manufacturing": manufacturing_validation,
                "analytical": analytical_validation,
                "regulatory": regulatory_assessment,
            }

            # Determine overall status
            critical_failures = self._identify_critical_failures(all_test_results)
            warnings = self._identify_warnings(all_test_results)

            overall_status = (
                "FAILED" if critical_failures else ("WARNING" if warnings else "PASSED")
            )

            # Calculate confidence score
            confidence_score = self._calculate_validation_confidence(all_test_results)

            # Create compliance assessment
            compliance_assessment = self._create_compliance_assessment(
                regulatory_assessment
            )

            validation_result = ValidationResult(
                validation_id=validation_id,
                validation_type="Complete Development Validation",
                status=overall_status,
                confidence_score=confidence_score,
                critical_failures=critical_failures,
                warnings=warnings,
                test_results=all_test_results,
                compliance_assessment=compliance_assessment,
                timestamp=datetime.now().isoformat(),
            )

            # Store validation history
            self.validation_history[validation_id] = validation_result

            self.logger.info(
                f"✅ Complete validation finished: {overall_status} "
                f"(Confidence: {confidence_score:.1%})"
            )

            return validation_result

        except Exception as e:
            self.logger.error(f"❌ Complete pharmaceutical validation failed: {e}")
            raise KimeraException(f"Complete validation failed: {e}")

    def validate_batch_quality(
        self,
        batch_data: Dict[str, Any],
        specification_limits: Dict[str, Dict[str, float]],
    ) -> QualityProfile:
        """
        Validate batch quality against specifications.

        Args:
            batch_data: Batch testing data
            specification_limits: Product specification limits

        Returns:
            QualityProfile: Batch quality assessment

        Raises:
            KimeraException: If batch validation fails
        """
        try:
            batch_id = batch_data.get(
                "batch_id", f"BATCH_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            self.logger.info(f"🔍 Validating batch quality for {batch_id}...")

            # Extract critical quality attributes
            cqa_values = {}
            cqa_compliance = {}

            for attribute, limits in specification_limits.items():
                measured_value = batch_data.get(attribute)
                if measured_value is not None:
                    cqa_values[attribute] = measured_value

                    # Check compliance
                    min_limit = limits.get("min", float("-inf"))
                    max_limit = limits.get("max", float("inf"))
                    compliant = min_limit <= measured_value <= max_limit
                    cqa_compliance[attribute] = compliant

                    if not compliant:
                        self.logger.warning(
                            f"⚠️ {attribute}: {measured_value} outside limits "
                            f"({min_limit}-{max_limit})"
                        )

            # Calculate overall quality score
            quality_score = (
                sum(cqa_compliance.values()) / len(cqa_compliance)
                if cqa_compliance
                else 0
            )

            # Risk assessment
            risk_assessment = self._assess_batch_risks(cqa_values, specification_limits)

            # Predict shelf life (simplified)
            shelf_life_prediction = self._predict_shelf_life(batch_data)

            quality_profile = QualityProfile(
                batch_id=batch_id,
                quality_score=quality_score,
                critical_quality_attributes=cqa_values,
                specification_compliance=cqa_compliance,
                risk_assessment=risk_assessment,
                shelf_life_prediction=shelf_life_prediction,
            )

            # Store quality profile
            self.quality_profiles[batch_id] = quality_profile

            self.logger.info(
                f"✅ Batch quality validation completed: {quality_score:.1%} compliance"
            )

            return quality_profile

        except Exception as e:
            self.logger.error(f"❌ Batch quality validation failed: {e}")
            raise KimeraException(f"Batch quality validation failed: {e}")

    def validate_bioequivalence_study(
        self,
        test_product_data: Dict[str, Any],
        reference_product_data: Dict[str, Any],
        study_design: Dict[str, Any],
    ) -> ValidationResult:
        """
        Validate bioequivalence study design and results.

        Args:
            test_product_data: Test product dissolution and characteristics
            reference_product_data: Reference product data
            study_design: Bioequivalence study design parameters

        Returns:
            ValidationResult: Bioequivalence validation result

        Raises:
            KimeraException: If bioequivalence validation fails
        """
        try:
            self.logger.info("🔬 Validating bioequivalence study...")

            validation_id = f"BE_VAL_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Dissolution profile comparison
            test_profile = test_product_data.get("dissolution_profile", {})
            ref_profile = reference_product_data.get("dissolution_profile", {})

            if test_profile and ref_profile:
                dissolution_comparison = (
                    self.dissolution_analyzer.compare_dissolution_profiles(
                        test_profile.get("time_points", []),
                        test_profile.get("release_percentages", []),
                        ref_profile.get("time_points", []),
                        ref_profile.get("release_percentages", []),
                    )
                )

                # Assess if in vitro dissolution supports bioequivalence
                dissolution_similar = dissolution_comparison.f2_similarity >= 50.0

                self.logger.info(
                    f"📊 Dissolution f2 similarity: {dissolution_comparison.f2_similarity:.2f}"
                )
            else:
                dissolution_similar = False
                dissolution_comparison = None

            # Study design validation
            design_validation = self._validate_be_study_design(study_design)

            # Statistical power analysis
            power_analysis = self._analyze_statistical_power(study_design)

            # Overall bioequivalence assessment
            be_assessment = {
                "dissolution_similar": dissolution_similar,
                "design_adequate": design_validation["adequate"],
                "statistical_power": power_analysis["power"],
                "recommendation": self._generate_be_recommendation(
                    dissolution_similar, design_validation, power_analysis
                ),
            }

            # Determine validation status
            if (
                dissolution_similar
                and design_validation["adequate"]
                and power_analysis["power"] >= 0.8
            ):
                status = "PASSED"
                critical_failures = []
            else:
                status = "FAILED"
                critical_failures = []
                if not dissolution_similar:
                    critical_failures.append(
                        "Dissolution profiles not similar (f2 < 50)"
                    )
                if not design_validation["adequate"]:
                    critical_failures.extend(design_validation["issues"])
                if power_analysis["power"] < 0.8:
                    critical_failures.append(
                        f"Statistical power too low ({power_analysis['power']:.2f})"
                    )

            test_results = {
                "dissolution_comparison": (
                    dissolution_comparison.__dict__ if dissolution_comparison else None
                ),
                "study_design_validation": design_validation,
                "statistical_power_analysis": power_analysis,
                "bioequivalence_assessment": be_assessment,
            }

            validation_result = ValidationResult(
                validation_id=validation_id,
                validation_type="Bioequivalence Study Validation",
                status=status,
                confidence_score=0.95 if status == "PASSED" else 0.60,
                critical_failures=critical_failures,
                warnings=[],
                test_results=test_results,
                compliance_assessment={"bioequivalence_ready": status == "PASSED"},
                timestamp=datetime.now().isoformat(),
            )

            self.logger.info(f"✅ Bioequivalence validation completed: {status}")

            return validation_result

        except Exception as e:
            self.logger.error(f"❌ Bioequivalence validation failed: {e}")
            raise KimeraException(f"Bioequivalence validation failed: {e}")

    # Helper methods for validation phases
    def _validate_raw_materials(self, raw_materials: Dict[str, Any]) -> Dict[str, Any]:
        """Validate raw materials using KCl testing engine."""
        try:
            # Use KCl testing engine for raw material characterization
            rm_spec = self.kcl_engine.characterize_raw_materials(raw_materials)

            # Additional validation checks
            validation_results = {
                "specification": rm_spec.__dict__,
                "usp_compliant": all(rm_spec.identification_tests.values()),
                "purity_acceptable": (
                    self.validation_standards["critical_quality_attributes"]["assay"][
                        "min"
                    ]
                    <= rm_spec.purity_percent
                    <= self.validation_standards["critical_quality_attributes"][
                        "assay"
                    ]["max"]
                ),
                "moisture_acceptable": (
                    rm_spec.moisture_content
                    <= self.validation_standards["critical_quality_attributes"][
                        "moisture_content"
                    ]["max"]
                ),
                "status": "PASSED",
            }

            # Check for failures
            if not validation_results["usp_compliant"]:
                validation_results["status"] = "FAILED"
                validation_results["failure_reason"] = "USP identification tests failed"
            elif not validation_results["purity_acceptable"]:
                validation_results["status"] = "FAILED"
                validation_results["failure_reason"] = "Purity outside acceptable range"
            elif not validation_results["moisture_acceptable"]:
                validation_results["status"] = "FAILED"
                validation_results["failure_reason"] = "Moisture content too high"

            return validation_results

        except Exception as e:
            return {"status": "FAILED", "error": str(e)}

    def _validate_formulation(self, formulation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate formulation parameters and prototype."""
        try:
            # Extract formulation parameters
            coating_thickness = formulation_data.get("coating_thickness_percent", 12.0)
            polymer_ratios = formulation_data.get(
                "polymer_ratios", {"ethylcellulose": 0.8, "hpc": 0.2}
            )
            process_params = formulation_data.get("process_parameters", {})

            # Create and test prototype using KCl engine
            prototype = self.kcl_engine.create_formulation_prototype(
                coating_thickness, polymer_ratios, process_params
            )

            # Validate encapsulation efficiency
            efficiency_acceptable = prototype.encapsulation_efficiency >= 0.95

            # Validate morphology
            morphology_acceptable = "spherical" in prototype.particle_morphology.lower()

            # Test dissolution if profile provided
            dissolution_acceptable = True
            if "dissolution_profile" in formulation_data:
                test_conditions = formulation_data.get(
                    "test_conditions",
                    {
                        "apparatus": 1,
                        "medium": "water",
                        "volume_ml": 900,
                        "temperature_c": 37.0,
                        "rotation_rpm": 100,
                    },
                )

                from ..protocols.usp_protocols import DissolutionTestUSP711

                dissolution_test = DissolutionTestUSP711(
                    apparatus=test_conditions["apparatus"],
                    medium=test_conditions["medium"],
                    volume_ml=test_conditions["volume_ml"],
                    temperature_c=test_conditions["temperature_c"],
                    rotation_rpm=test_conditions["rotation_rpm"],
                    sampling_times=[1, 2, 4, 6],
                    acceptance_table="2",
                )

                dissolution_result = self.usp_engine.perform_dissolution_test_711(
                    formulation_data["dissolution_profile"], dissolution_test
                )

                dissolution_acceptable = dissolution_result.status == "PASSED"

            validation_results = {
                "prototype": prototype.__dict__,
                "encapsulation_efficiency_acceptable": efficiency_acceptable,
                "morphology_acceptable": morphology_acceptable,
                "dissolution_acceptable": dissolution_acceptable,
                "status": (
                    "PASSED"
                    if all(
                        [
                            efficiency_acceptable,
                            morphology_acceptable,
                            dissolution_acceptable,
                        ]
                    )
                    else "FAILED"
                ),
            }

            return validation_results

        except Exception as e:
            return {"status": "FAILED", "error": str(e)}

    def _validate_manufacturing(
        self, manufacturing_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate manufacturing process parameters."""
        try:
            # Extract manufacturing parameters
            batch_size = manufacturing_data.get("batch_size", 0)
            process_controls = manufacturing_data.get("process_controls", {})
            equipment_qualification = manufacturing_data.get(
                "equipment_qualified", False
            )

            # Validate batch size
            min_batch_size = self.validation_standards["regulatory_requirements"][
                "fda"
            ]["batch_size_minimum"]
            batch_size_acceptable = batch_size >= min_batch_size

            # Validate process controls
            critical_controls = [
                "temperature",
                "spray_rate",
                "inlet_air_flow",
                "product_temperature",
            ]
            controls_adequate = all(
                control in process_controls for control in critical_controls
            )

            # Validate equipment qualification
            equipment_acceptable = equipment_qualification

            validation_results = {
                "batch_size_acceptable": batch_size_acceptable,
                "process_controls_adequate": controls_adequate,
                "equipment_qualified": equipment_acceptable,
                "status": (
                    "PASSED"
                    if all(
                        [batch_size_acceptable, controls_adequate, equipment_acceptable]
                    )
                    else "FAILED"
                ),
            }

            return validation_results

        except Exception as e:
            return {"status": "FAILED", "error": str(e)}

    def _validate_analytical_testing(
        self, testing_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate analytical testing methods and results."""
        try:
            validation_results = {
                "tests_performed": [],
                "all_passed": True,
                "status": "PASSED",
            }

            # Assay test validation
            if "assay" in testing_data:
                assay_result = self.usp_engine.perform_assay_test(
                    testing_data["assay"]["sample_concentration"],
                    testing_data["assay"]["standard_concentration"],
                    testing_data["assay"]["labeled_amount"],
                )
                validation_results["tests_performed"].append(assay_result)
                if assay_result.status != "PASSED":
                    validation_results["all_passed"] = False

            # Content uniformity validation
            if "content_uniformity" in testing_data:
                cu_result = self.usp_engine.perform_content_uniformity_905(
                    testing_data["content_uniformity"]["measurements"],
                    testing_data["content_uniformity"]["labeled_amount"],
                )
                validation_results["tests_performed"].append(cu_result)
                if cu_result.status != "PASSED":
                    validation_results["all_passed"] = False

            # Disintegration test validation
            if "disintegration" in testing_data:
                disint_result = self.usp_engine.perform_disintegration_test_701(
                    testing_data["disintegration"]["times"]
                )
                validation_results["tests_performed"].append(disint_result)
                if disint_result.status != "PASSED":
                    validation_results["all_passed"] = False

            # Stability validation
            if "stability" in testing_data:
                for condition, data in testing_data["stability"].items():
                    stability_result = self.usp_engine.validate_stability_ich_q1a(
                        data, condition
                    )
                    validation_results["tests_performed"].append(stability_result)
                    if stability_result.status != "PASSED":
                        validation_results["all_passed"] = False

            validation_results["status"] = (
                "PASSED" if validation_results["all_passed"] else "FAILED"
            )

            return validation_results

        except Exception as e:
            return {"status": "FAILED", "error": str(e)}

    def _assess_regulatory_compliance(
        self, validation_data: Dict[str, Any]
    ) -> RegulatoryCompliance:
        """Assess regulatory compliance readiness."""
        try:
            # Check FDA readiness
            fda_requirements = self.validation_standards["regulatory_requirements"][
                "fda"
            ]
            fda_ready = all(
                [
                    validation_data["analytical"]["status"] == "PASSED",
                    validation_data["manufacturing"]["status"] == "PASSED",
                    len(validation_data["analytical"].get("tests_performed", []))
                    >= fda_requirements["dissolution_points"],
                ]
            )

            # Check EMA readiness (similar to FDA for this example)
            ema_ready = fda_ready

            # Check ICH compliance
            ich_compliant = validation_data["analytical"]["status"] == "PASSED"

            # Check USP compliance
            usp_compliant = validation_data["raw_materials"]["usp_compliant"]

            # Calculate overall compliance score
            compliance_factors = [fda_ready, ema_ready, ich_compliant, usp_compliant]
            compliance_score = sum(compliance_factors) / len(compliance_factors)

            # Identify regulatory gaps
            regulatory_gaps = []
            if not fda_ready:
                regulatory_gaps.append("FDA submission requirements not met")
            if not ema_ready:
                regulatory_gaps.append("EMA submission requirements not met")
            if not ich_compliant:
                regulatory_gaps.append("ICH guidelines not followed")
            if not usp_compliant:
                regulatory_gaps.append("USP standards not met")

            # Determine submission readiness
            if compliance_score >= 1.0:
                submission_readiness = "READY_FOR_SUBMISSION"
            elif compliance_score >= 0.8:
                submission_readiness = "MINOR_GAPS_IDENTIFIED"
            else:
                submission_readiness = "MAJOR_REMEDIATION_REQUIRED"

            return RegulatoryCompliance(
                fda_ready=fda_ready,
                ema_ready=ema_ready,
                ich_compliant=ich_compliant,
                usp_compliant=usp_compliant,
                compliance_score=compliance_score,
                regulatory_gaps=regulatory_gaps,
                submission_readiness=submission_readiness,
            )

        except Exception as e:
            self.logger.error(f"Regulatory compliance assessment failed: {e}")
            return RegulatoryCompliance(
                fda_ready=False,
                ema_ready=False,
                ich_compliant=False,
                usp_compliant=False,
                compliance_score=0.0,
                regulatory_gaps=["Assessment failed"],
                submission_readiness="ASSESSMENT_FAILED",
            )

    def _identify_critical_failures(self, test_results: Dict[str, Any]) -> List[str]:
        """Identify critical failures from all test results."""
        critical_failures = []

        for phase, results in test_results.items():
            if isinstance(results, dict) and results.get("status") == "FAILED":
                if phase in ["raw_materials", "formulation", "analytical"]:
                    critical_failures.append(
                        f"Critical failure in {phase}: {results.get('failure_reason', 'Unknown')}"
                    )

        return critical_failures

    def _identify_warnings(self, test_results: Dict[str, Any]) -> List[str]:
        """Identify warnings from test results."""
        warnings = []

        # Check for borderline results
        for phase, results in test_results.items():
            if isinstance(results, dict):
                if phase == "regulatory" and hasattr(results, "compliance_score"):
                    if 0.8 <= results.compliance_score < 0.95:
                        warnings.append(
                            "Regulatory compliance score below optimal threshold"
                        )

        return warnings

    def _calculate_validation_confidence(self, test_results: Dict[str, Any]) -> float:
        """Calculate overall validation confidence score."""
        phase_scores = []

        for phase, results in test_results.items():
            if isinstance(results, dict):
                if results.get("status") == "PASSED":
                    phase_scores.append(1.0)
                elif results.get("status") == "WARNING":
                    phase_scores.append(0.7)
                else:
                    phase_scores.append(0.0)

        return sum(phase_scores) / len(phase_scores) if phase_scores else 0.0

    def _create_compliance_assessment(
        self, regulatory_assessment: RegulatoryCompliance
    ) -> Dict[str, Any]:
        """Create compliance assessment summary."""
        return {
            "overall_compliance_score": regulatory_assessment.compliance_score,
            "submission_readiness": regulatory_assessment.submission_readiness,
            "regulatory_gaps_count": len(regulatory_assessment.regulatory_gaps),
            "fda_ready": regulatory_assessment.fda_ready,
            "ema_ready": regulatory_assessment.ema_ready,
            "recommendation": self._generate_compliance_recommendation(
                regulatory_assessment
            ),
        }

    def _generate_compliance_recommendation(
        self, assessment: RegulatoryCompliance
    ) -> str:
        """Generate compliance recommendation."""
        if assessment.submission_readiness == "READY_FOR_SUBMISSION":
            return "Product ready for regulatory submission"
        elif assessment.submission_readiness == "MINOR_GAPS_IDENTIFIED":
            return "Address minor gaps before submission"
        else:
            return "Major remediation required before submission"

    def _assess_batch_risks(
        self,
        cqa_values: Dict[str, float],
        specification_limits: Dict[str, Dict[str, float]],
    ) -> Dict[str, str]:
        """Assess batch-specific risks."""
        risks = {}

        for attribute, value in cqa_values.items():
            if attribute in specification_limits:
                limits = specification_limits[attribute]
                min_limit = limits.get("min", float("-inf"))
                max_limit = limits.get("max", float("inf"))

                # Calculate distance from limits
                range_size = (
                    max_limit - min_limit
                    if max_limit != float("inf") and min_limit != float("-inf")
                    else 100
                )

                if value < min_limit or value > max_limit:
                    risks[attribute] = "HIGH - Out of specification"
                elif (
                    abs(value - min_limit) / range_size < 0.1
                    or abs(value - max_limit) / range_size < 0.1
                ):
                    risks[attribute] = "MEDIUM - Near specification limit"
                else:
                    risks[attribute] = "LOW - Well within specification"

        return risks

    def _predict_shelf_life(self, batch_data: Dict[str, Any]) -> Optional[float]:
        """Predict shelf life based on batch characteristics."""
        # Simplified shelf life prediction
        # In practice, this would use Arrhenius modeling and stability data

        base_shelf_life = 24.0  # months

        # Adjust based on moisture content
        moisture = batch_data.get("moisture_content", 0)
        if moisture > 3.0:
            base_shelf_life *= 0.8
        elif moisture < 1.0:
            base_shelf_life *= 1.1

        # Adjust based on assay
        assay = batch_data.get("assay_percent", 100)
        if assay < 95:
            base_shelf_life *= 0.9
        elif assay > 105:
            base_shelf_life *= 0.85

        return max(12.0, min(36.0, base_shelf_life))  # Constrain to 12-36 months

    def _validate_be_study_design(self, study_design: Dict[str, Any]) -> Dict[str, Any]:
        """Validate bioequivalence study design."""
        design_issues = []

        # Check sample size
        sample_size = study_design.get("sample_size", 0)
        if sample_size < 12:
            design_issues.append("Sample size too small (minimum 12)")

        # Check study design
        design_type = study_design.get("design_type", "")
        if design_type not in ["crossover", "parallel"]:
            design_issues.append("Invalid study design type")

        # Check washout period for crossover
        if design_type == "crossover":
            washout_period = study_design.get("washout_period_hours", 0)
            if washout_period < 168:  # 1 week
                design_issues.append("Washout period too short")

        return {
            "adequate": len(design_issues) == 0,
            "issues": design_issues,
            "sample_size": sample_size,
            "design_type": design_type,
        }

    def _analyze_statistical_power(
        self, study_design: Dict[str, Any]
    ) -> Dict[str, float]:
        """Analyze statistical power of bioequivalence study."""
        # Simplified power analysis
        sample_size = study_design.get("sample_size", 12)
        expected_cv = study_design.get("expected_cv_percent", 20) / 100

        # Power calculation for bioequivalence (simplified)
        # Assumes 80-125% acceptance criteria
        power = min(0.99, 0.5 + (sample_size - 12) * 0.03 - expected_cv * 2)
        power = max(0.05, power)

        return {
            "power": power,
            "sample_size": sample_size,
            "expected_cv": expected_cv,
            "adequate_power": power >= 0.8,
        }

    def _generate_be_recommendation(
        self,
        dissolution_similar: bool,
        design_validation: Dict[str, Any],
        power_analysis: Dict[str, float],
    ) -> str:
        """Generate bioequivalence study recommendation."""
        if (
            dissolution_similar
            and design_validation["adequate"]
            and power_analysis["adequate_power"]
        ):
            return "Study design adequate - proceed with bioequivalence study"

        recommendations = []
        if not dissolution_similar:
            recommendations.append("improve dissolution similarity")
        if not design_validation["adequate"]:
            recommendations.append("address study design issues")
        if not power_analysis["adequate_power"]:
            recommendations.append("increase sample size for adequate power")

        return f"Study modifications needed: {', '.join(recommendations)}"

    def generate_master_validation_report(self) -> Dict[str, Any]:
        """
        Generate master validation report combining all validation activities.

        Returns:
            Dict[str, Any]: Master validation report
        """
        try:
            self.logger.info("📋 Generating master validation report...")

            # Aggregate all validation results
            total_validations = len(self.validation_history)
            passed_validations = sum(
                1 for v in self.validation_history.values() if v.status == "PASSED"
            )

            # Aggregate quality profiles
            total_batches = len(self.quality_profiles)
            avg_quality_score = (
                np.mean([qp.quality_score for qp in self.quality_profiles.values()])
                if self.quality_profiles
                else 0
            )

            # Generate comprehensive report
            master_report = {
                "report_id": f"MASTER_VAL_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "generation_time": datetime.now().isoformat(),
                "validation_summary": {
                    "total_validations_performed": total_validations,
                    "validations_passed": passed_validations,
                    "validation_success_rate": (
                        passed_validations / total_validations
                        if total_validations > 0
                        else 0
                    ),
                    "total_batches_assessed": total_batches,
                    "average_batch_quality_score": avg_quality_score,
                },
                "detailed_validation_history": {
                    vid: v.__dict__ for vid, v in self.validation_history.items()
                },
                "quality_profile_summary": {
                    bid: qp.__dict__ for bid, qp in self.quality_profiles.items()
                },
                "regulatory_readiness_assessment": self._assess_overall_regulatory_readiness(),
                "recommendations": self._generate_master_recommendations(),
                "next_steps": self._identify_next_steps(),
            }

            self.logger.info(
                f"✅ Master validation report generated with {total_validations} validations"
            )

            return master_report

        except Exception as e:
            self.logger.error(f"❌ Master validation report generation failed: {e}")
            raise KimeraException(f"Master report generation failed: {e}")

    def _assess_overall_regulatory_readiness(self) -> Dict[str, Any]:
        """Assess overall regulatory readiness across all validations."""
        if not self.validation_history:
            return {"status": "NO_VALIDATIONS", "readiness_score": 0.0}

        # Get latest validation with regulatory assessment
        latest_regulatory = None
        for validation in reversed(list(self.validation_history.values())):
            if "regulatory" in validation.test_results:
                latest_regulatory = validation.test_results["regulatory"]
                break

        if latest_regulatory:
            return {
                "status": latest_regulatory.submission_readiness,
                "readiness_score": latest_regulatory.compliance_score,
                "fda_ready": latest_regulatory.fda_ready,
                "ema_ready": latest_regulatory.ema_ready,
            }
        else:
            return {"status": "ASSESSMENT_PENDING", "readiness_score": 0.0}

    def _generate_master_recommendations(self) -> List[str]:
        """Generate master recommendations across all validations."""
        recommendations = []

        # Analyze validation patterns
        failed_validations = [
            v for v in self.validation_history.values() if v.status == "FAILED"
        ]

        if failed_validations:
            common_failures = {}
            for validation in failed_validations:
                for failure in validation.critical_failures:
                    common_failures[failure] = common_failures.get(failure, 0) + 1

            # Recommend addressing most common failures
            if common_failures:
                most_common = max(common_failures, key=common_failures.get)
                recommendations.append(f"Address recurring issue: {most_common}")

        # Quality score recommendations
        if self.quality_profiles:
            low_quality_batches = [
                qp for qp in self.quality_profiles.values() if qp.quality_score < 0.8
            ]
            if low_quality_batches:
                recommendations.append(
                    f"Investigate {len(low_quality_batches)} batches with low quality scores"
                )

        if not recommendations:
            recommendations.append(
                "All validations successful - maintain current quality standards"
            )

        return recommendations

    def _identify_next_steps(self) -> List[str]:
        """Identify next steps in pharmaceutical development."""
        next_steps = []

        # Check if ready for regulatory submission
        regulatory_status = self._assess_overall_regulatory_readiness()

        if regulatory_status["status"] == "READY_FOR_SUBMISSION":
            next_steps.append("Prepare regulatory submission dossier")
            next_steps.append("Schedule pre-submission meeting with regulatory agency")
        elif regulatory_status["status"] == "MINOR_GAPS_IDENTIFIED":
            next_steps.append("Address identified regulatory gaps")
            next_steps.append("Perform additional validation studies if needed")
        else:
            next_steps.append("Complete major remediation activities")
            next_steps.append("Re-validate critical processes")

        # Manufacturing readiness
        if self.quality_profiles:
            next_steps.append("Scale up manufacturing process")
            next_steps.append("Perform process validation batches")

        return next_steps
