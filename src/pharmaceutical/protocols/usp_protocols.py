"""
USP Protocol Implementation Engine

Official United States Pharmacopeia (USP) testing protocols for pharmaceutical
development and validation, specifically for KCl extended-release capsules.

Implements USP <711> Dissolution, USP <905> Content Uniformity, and other standards.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

from ...utils.kimera_exceptions import KimeraBaseException as KimeraException
from ...utils.kimera_logger import get_logger

logger = get_logger(__name__)


@dataclass
class USPTestResult:
    """Standard USP test result structure."""

    test_id: str
    test_name: str
    method: str
    result_value: float
    acceptance_criteria: Dict[str, float]
    status: str  # PASSED, FAILED, WARNING
    confidence_level: float
    measurement_uncertainty: float
    test_conditions: Dict[str, Any]
    timestamp: str


@dataclass
class DissolutionTestUSP711:
    """USP <711> Dissolution Test implementation."""

    apparatus: int  # 1 = Basket, 2 = Paddle
    medium: str
    volume_ml: int
    temperature_c: float
    rotation_rpm: int
    sampling_times: List[float]
    acceptance_table: str  # "2" for extended release


@dataclass
class ContentUniformityUSP905:
    """USP <905> Content Uniformity test parameters."""

    sample_size: int
    acceptance_value: float
    reference_value: float
    individual_limits: Tuple[float, float]


class USPProtocolEngine:
    """
    Engine implementing official USP testing protocols.

    Provides standardized testing methods following USP guidelines
    for pharmaceutical quality control and validation.
    """

    def __init__(self):
        """Initialize USP Protocol Engine."""
        self.logger = logger
        self.test_results = {}
        self.protocol_standards = self._load_usp_standards()

        self.logger.info("📋 USP Protocol Engine initialized")

    def _load_usp_standards(self) -> Dict[str, Any]:
        """Load official USP standards and acceptance criteria."""
        return {
            "dissolution_711": {
                "apparatus_1_rpm": 100,
                "apparatus_2_rpm": 50,
                "standard_volume": 900,  # mL
                "standard_temperature": 37.0,  # °C
                "kcl_er_test_2": {
                    "times_hours": [1, 2, 4, 6],
                    "tolerances_750mg": {
                        1: {"min": 25, "max": 45},
                        2: {"min": 45, "max": 65},
                        4: {"min": 70, "max": 90},
                        6: {"min": 85, "max": 100},
                    },
                },
                "f2_threshold": 50.0,
                "cv_limits": {"first_point": 20.0, "other_points": 10.0},
            },
            "content_uniformity_905": {
                "sample_size": 10,
                "individual_range": (85.0, 115.0),
                "mean_range": (98.5, 101.5),
                "acceptance_value_l1": 15.0,
                "reference_value": 100.0,
            },
            "assay_standards": {
                "kcl_range": (90.0, 110.0),
                "precision_rsd": 2.0,
                "accuracy_range": (98.0, 102.0),
            },
            "disintegration_701": {
                "time_limit_capsules": 30,  # minutes
                "apparatus_temperature": 37.0,
                "medium": "water",
            },
            "stability_ich_q1a": {
                "long_term": {"temp": 25, "humidity": 60, "duration_months": 24},
                "accelerated": {"temp": 40, "humidity": 75, "duration_months": 6},
                "intermediate": {"temp": 30, "humidity": 65, "duration_months": 12},
            },
        }

    def perform_dissolution_test_711(
        self,
        sample_data: Dict[str, Any],
        test_conditions: DissolutionTestUSP711,
        reference_profile: Optional[List[float]] = None,
    ) -> USPTestResult:
        """
        Perform USP <711> Dissolution Test for extended-release capsules.

        Args:
            sample_data: Sample dissolution data
            test_conditions: Test conditions following USP <711>
            reference_profile: Reference dissolution profile for f2 calculation

        Returns:
            USPTestResult: Standardized test result

        Raises:
            KimeraException: If test execution fails
        """
        try:
            self.logger.info("🧪 Performing USP <711> Dissolution Test...")

            # Validate test conditions
            self._validate_dissolution_conditions(test_conditions)

            # Extract dissolution data
            time_points = sample_data.get("time_points", [])
            release_percentages = sample_data.get("release_percentages", [])

            if len(time_points) != len(release_percentages):
                raise KimeraException(
                    "Time points and release percentages must have same length"
                )

            # Check against USP tolerances
            tolerances = self.protocol_standards["dissolution_711"]["kcl_er_test_2"][
                "tolerances_750mg"
            ]
            compliance_results = []

            for time, release in zip(time_points, release_percentages):
                if time in tolerances:
                    min_release = tolerances[time]["min"]
                    max_release = tolerances[time]["max"]
                    compliant = min_release <= release <= max_release
                    compliance_results.append(compliant)

                    if not compliant:
                        self.logger.warning(
                            f"⚠️ Time {time}h: {release:.1f}% outside range "
                            f"({min_release}-{max_release}%)"
                        )

            # Calculate f2 similarity if reference provided
            f2_similarity = None
            if reference_profile and len(reference_profile) == len(release_percentages):
                f2_similarity = self._calculate_f2_similarity_factor(
                    release_percentages, reference_profile
                )
                self.logger.info(f"📊 f2 similarity factor: {f2_similarity:.2f}")

            # Determine overall test status
            overall_compliance = (
                all(compliance_results) if compliance_results else False
            )
            f2_compliant = (
                f2_similarity is None
                or f2_similarity
                >= self.protocol_standards["dissolution_711"]["f2_threshold"]
            )

            status = "PASSED" if overall_compliance and f2_compliant else "FAILED"

            # Calculate measurement uncertainty
            uncertainty = np.std(release_percentages) / np.sqrt(
                len(release_percentages)
            )

            result = USPTestResult(
                test_id=f"USP711_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                test_name="Dissolution Test USP <711>",
                method="USP <711> Apparatus 1, Test 2 for Extended Release",
                result_value=(
                    f2_similarity if f2_similarity else np.mean(release_percentages)
                ),
                acceptance_criteria={
                    "f2_threshold": self.protocol_standards["dissolution_711"][
                        "f2_threshold"
                    ],
                    "individual_tolerances": tolerances,
                },
                status=status,
                confidence_level=0.95,
                measurement_uncertainty=uncertainty,
                test_conditions={
                    "apparatus": test_conditions.apparatus,
                    "medium": test_conditions.medium,
                    "volume_ml": test_conditions.volume_ml,
                    "temperature_c": test_conditions.temperature_c,
                    "rotation_rpm": test_conditions.rotation_rpm,
                },
                timestamp=datetime.now().isoformat(),
            )

            self.test_results[result.test_id] = result
            self.logger.info(f"✅ USP <711> Dissolution Test completed: {status}")

            return result

        except Exception as e:
            self.logger.error(f"❌ USP <711> Dissolution Test failed: {e}")
            raise KimeraException(f"USP <711> test failed: {e}")

    def perform_content_uniformity_905(
        self, sample_measurements: List[float], labeled_amount: float
    ) -> USPTestResult:
        """
        Perform USP <905> Content Uniformity test.

        Args:
            sample_measurements: Individual capsule content measurements
            labeled_amount: Labeled amount of active ingredient

        Returns:
            USPTestResult: Standardized test result

        Raises:
            KimeraException: If test execution fails
        """
        try:
            self.logger.info("📊 Performing USP <905> Content Uniformity Test...")

            if len(sample_measurements) < 10:
                raise KimeraException(
                    "Minimum 10 samples required for content uniformity"
                )

            # Convert to percentage of labeled amount
            percentages = [
                (measurement / labeled_amount) * 100
                for measurement in sample_measurements
            ]

            # Calculate statistics
            mean_percentage = np.mean(percentages)
            std_dev = np.std(percentages, ddof=1)

            # Check individual units
            individual_range = self.protocol_standards["content_uniformity_905"][
                "individual_range"
            ]
            individual_compliance = all(
                individual_range[0] <= p <= individual_range[1] for p in percentages
            )

            # Calculate Acceptance Value (AV)
            reference_value = self.protocol_standards["content_uniformity_905"][
                "reference_value"
            ]

            # |M - reference_value| where M is mean
            mean_deviation = abs(mean_percentage - reference_value)

            # k*s where k depends on sample size and s is standard deviation
            k_value = 2.4  # For n=10 samples
            ks_value = k_value * std_dev

            acceptance_value = max(mean_deviation, ks_value)
            av_threshold = self.protocol_standards["content_uniformity_905"][
                "acceptance_value_l1"
            ]

            # Determine compliance
            av_compliant = acceptance_value <= av_threshold
            overall_compliant = individual_compliance and av_compliant

            status = "PASSED" if overall_compliant else "FAILED"

            if not individual_compliance:
                self.logger.warning("⚠️ Individual units outside 85.0-115.0% range")
            if not av_compliant:
                self.logger.warning(
                    f"⚠️ Acceptance Value {acceptance_value:.2f} exceeds limit {av_threshold}"
                )

            result = USPTestResult(
                test_id=f"USP905_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                test_name="Content Uniformity USP <905>",
                method="USP <905> Uniformity of Dosage Units",
                result_value=acceptance_value,
                acceptance_criteria={
                    "acceptance_value_limit": av_threshold,
                    "individual_range": individual_range,
                    "mean_range": self.protocol_standards["content_uniformity_905"][
                        "mean_range"
                    ],
                },
                status=status,
                confidence_level=0.95,
                measurement_uncertainty=std_dev / np.sqrt(len(sample_measurements)),
                test_conditions={
                    "sample_size": len(sample_measurements),
                    "labeled_amount": labeled_amount,
                    "test_method": "Individual content determination",
                },
                timestamp=datetime.now().isoformat(),
            )

            self.test_results[result.test_id] = result
            self.logger.info(
                f"✅ USP <905> Content Uniformity Test completed: {status}"
            )

            return result

        except Exception as e:
            self.logger.error(f"❌ USP <905> Content Uniformity Test failed: {e}")
            raise KimeraException(f"USP <905> test failed: {e}")

    def perform_assay_test(
        self,
        sample_concentration: float,
        standard_concentration: float,
        labeled_amount: float,
    ) -> USPTestResult:
        """
        Perform pharmaceutical assay test for KCl content.

        Args:
            sample_concentration: Measured sample concentration
            standard_concentration: Standard reference concentration
            labeled_amount: Labeled amount of KCl

        Returns:
            USPTestResult: Standardized test result

        Raises:
            KimeraException: If assay test fails
        """
        try:
            self.logger.info("🔬 Performing KCl Assay Test...")

            # Calculate percentage of labeled amount
            percentage = (sample_concentration / standard_concentration) * 100

            # Check against USP acceptance criteria
            assay_range = self.protocol_standards["assay_standards"]["kcl_range"]
            compliant = assay_range[0] <= percentage <= assay_range[1]

            status = "PASSED" if compliant else "FAILED"

            if not compliant:
                self.logger.warning(
                    f"⚠️ Assay result {percentage:.2f}% outside USP range "
                    f"({assay_range[0]}-{assay_range[1]}%)"
                )

            # Estimate measurement uncertainty (typical for assay methods)
            uncertainty = 1.0  # ±1% typical for pharmaceutical assays

            result = USPTestResult(
                test_id=f"ASSAY_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                test_name="KCl Assay Test",
                method="USP Potassium Chloride Assay",
                result_value=percentage,
                acceptance_criteria={
                    "range": assay_range,
                    "method": "Atomic Absorption Spectrophotometry",
                },
                status=status,
                confidence_level=0.95,
                measurement_uncertainty=uncertainty,
                test_conditions={
                    "sample_concentration": sample_concentration,
                    "standard_concentration": standard_concentration,
                    "labeled_amount": labeled_amount,
                },
                timestamp=datetime.now().isoformat(),
            )

            self.test_results[result.test_id] = result
            self.logger.info(f"✅ KCl Assay Test completed: {status}")

            return result

        except Exception as e:
            self.logger.error(f"❌ KCl Assay Test failed: {e}")
            raise KimeraException(f"Assay test failed: {e}")

    def perform_disintegration_test_701(
        self, disintegration_times: List[float]
    ) -> USPTestResult:
        """
        Perform USP <701> Disintegration Test for capsules.

        Args:
            disintegration_times: Disintegration times for 6 capsules (minutes)

        Returns:
            USPTestResult: Standardized test result

        Raises:
            KimeraException: If disintegration test fails
        """
        try:
            self.logger.info("⏱️ Performing USP <701> Disintegration Test...")

            if len(disintegration_times) != 6:
                raise KimeraException("Disintegration test requires exactly 6 capsules")

            # USP requirement: all units must disintegrate within time limit
            time_limit = self.protocol_standards["disintegration_701"][
                "time_limit_capsules"
            ]

            all_compliant = all(time <= time_limit for time in disintegration_times)
            max_time = max(disintegration_times)
            mean_time = np.mean(disintegration_times)

            status = "PASSED" if all_compliant else "FAILED"

            if not all_compliant:
                self.logger.warning(
                    f"⚠️ Maximum disintegration time {max_time:.1f} min exceeds limit {time_limit} min"
                )

            result = USPTestResult(
                test_id=f"USP701_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                test_name="Disintegration Test USP <701>",
                method="USP <701> Disintegration of Dosage Forms",
                result_value=max_time,
                acceptance_criteria={
                    "time_limit_minutes": time_limit,
                    "requirement": "All units must disintegrate within time limit",
                },
                status=status,
                confidence_level=0.95,
                measurement_uncertainty=np.std(disintegration_times),
                test_conditions={
                    "apparatus": "USP Disintegration Apparatus",
                    "medium": self.protocol_standards["disintegration_701"]["medium"],
                    "temperature": self.protocol_standards["disintegration_701"][
                        "apparatus_temperature"
                    ],
                    "sample_size": len(disintegration_times),
                },
                timestamp=datetime.now().isoformat(),
            )

            self.test_results[result.test_id] = result
            self.logger.info(f"✅ USP <701> Disintegration Test completed: {status}")

            return result

        except Exception as e:
            self.logger.error(f"❌ USP <701> Disintegration Test failed: {e}")
            raise KimeraException(f"USP <701> test failed: {e}")

    def validate_stability_ich_q1a(
        self, stability_data: Dict[str, Dict[str, float]], storage_condition: str
    ) -> USPTestResult:
        """
        Validate stability according to ICH Q1A guidelines.

        Args:
            stability_data: Stability data by time point
            storage_condition: Storage condition (long_term, accelerated, intermediate)

        Returns:
            USPTestResult: Stability validation result

        Raises:
            KimeraException: If stability validation fails
        """
        try:
            self.logger.info(
                f"📈 Validating ICH Q1A Stability - {storage_condition}..."
            )

            if storage_condition not in self.protocol_standards["stability_ich_q1a"]:
                raise KimeraException(f"Unknown storage condition: {storage_condition}")

            condition_spec = self.protocol_standards["stability_ich_q1a"][
                storage_condition
            ]

            # Analyze trends in critical quality attributes
            time_points = sorted(
                stability_data.keys(), key=lambda x: int(x.split("_")[0])
            )

            # Extract assay values over time
            assay_values = [
                stability_data[tp].get("assay_percent", 100.0) for tp in time_points
            ]
            dissolution_f2_values = [
                stability_data[tp].get("dissolution_f2", 100.0) for tp in time_points
            ]

            # Statistical trend analysis
            time_numeric = [int(tp.split("_")[0]) for tp in time_points]

            # Linear regression for assay degradation
            if len(assay_values) > 2:
                (
                    slope_assay,
                    intercept_assay,
                    r_value_assay,
                    p_value_assay,
                    std_err_assay,
                ) = stats.linregress(time_numeric, assay_values)
            else:
                slope_assay = 0
                p_value_assay = 1.0

            # Check if degradation is significant
            significant_degradation = (
                p_value_assay < 0.05 and slope_assay < -0.1
            )  # >0.1% per month

            # Check final values against acceptance criteria
            final_assay = assay_values[-1] if assay_values else 100.0
            final_f2 = dissolution_f2_values[-1] if dissolution_f2_values else 100.0

            assay_compliant = 90.0 <= final_assay <= 110.0
            dissolution_compliant = final_f2 >= 50.0

            overall_stable = (
                assay_compliant
                and dissolution_compliant
                and not significant_degradation
            )
            status = "PASSED" if overall_stable else "FAILED"

            if significant_degradation:
                self.logger.warning(
                    f"⚠️ Significant degradation detected: {slope_assay:.3f}%/month"
                )
            if not assay_compliant:
                self.logger.warning(
                    f"⚠️ Final assay {final_assay:.1f}% outside 90-110% range"
                )
            if not dissolution_compliant:
                self.logger.warning(
                    f"⚠️ Final dissolution f2 {final_f2:.1f} below 50.0 threshold"
                )

            result = USPTestResult(
                test_id=f"ICH_Q1A_{storage_condition}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                test_name=f"ICH Q1A Stability - {storage_condition.title()}",
                method="ICH Q1A(R2) Stability Testing Guidelines",
                result_value=final_assay,
                acceptance_criteria={
                    "assay_range": (90.0, 110.0),
                    "dissolution_f2_min": 50.0,
                    "significant_degradation_threshold": 0.05,
                },
                status=status,
                confidence_level=0.95,
                measurement_uncertainty=std_err_assay if len(assay_values) > 2 else 1.0,
                test_conditions={
                    "storage_condition": storage_condition,
                    "temperature": condition_spec["temp"],
                    "humidity": condition_spec["humidity"],
                    "duration_months": condition_spec["duration_months"],
                    "time_points_tested": len(time_points),
                },
                timestamp=datetime.now().isoformat(),
            )

            self.test_results[result.test_id] = result
            self.logger.info(f"✅ ICH Q1A Stability Validation completed: {status}")

            return result

        except Exception as e:
            self.logger.error(f"❌ ICH Q1A Stability Validation failed: {e}")
            raise KimeraException(f"ICH Q1A validation failed: {e}")

    # Helper methods
    def _validate_dissolution_conditions(
        self, conditions: DissolutionTestUSP711
    ) -> None:
        """Validate dissolution test conditions against USP standards."""
        standards = self.protocol_standards["dissolution_711"]

        if conditions.volume_ml != standards["standard_volume"]:
            self.logger.warning(f"Non-standard volume: {conditions.volume_ml} mL")

        if abs(conditions.temperature_c - standards["standard_temperature"]) > 0.5:
            raise KimeraException(
                f"Temperature {conditions.temperature_c}°C outside ±0.5°C tolerance"
            )

        expected_rpm = (
            standards["apparatus_1_rpm"]
            if conditions.apparatus == 1
            else standards["apparatus_2_rpm"]
        )
        if conditions.rotation_rpm != expected_rpm:
            self.logger.warning(
                f"Non-standard rotation speed: {conditions.rotation_rpm} rpm"
            )

    def _calculate_f2_similarity_factor(
        self, test_profile: List[float], reference_profile: List[float]
    ) -> float:
        """
        Calculate f2 similarity factor according to FDA/EMA guidelines.

        f2 = 50 * log{[1 + (1/n) * Σ(Rt - Tt)²]^(-0.5) * 100}
        """
        if len(test_profile) != len(reference_profile):
            raise KimeraException("Test and reference profiles must have same length")

        n = len(test_profile)
        sum_squared_diff = sum(
            (r - t) ** 2 for r, t in zip(reference_profile, test_profile)
        )

        # Calculate f2
        f2 = 50 * np.log10(((1 + sum_squared_diff / n) ** -0.5) * 100)

        # Constrain to valid range
        return max(0, min(100, f2))

    def generate_usp_compliance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive USP compliance report.

        Returns:
            Dict[str, Any]: Complete compliance report
        """
        try:
            self.logger.info("📋 Generating USP Compliance Report...")

            # Categorize test results
            dissolution_tests = [
                r
                for r in self.test_results.values()
                if "dissolution" in r.test_name.lower()
            ]
            content_tests = [
                r
                for r in self.test_results.values()
                if "content" in r.test_name.lower()
            ]
            assay_tests = [
                r for r in self.test_results.values() if "assay" in r.test_name.lower()
            ]
            stability_tests = [
                r
                for r in self.test_results.values()
                if "stability" in r.test_name.lower()
            ]

            # Calculate compliance statistics
            total_tests = len(self.test_results)
            passed_tests = sum(
                1 for r in self.test_results.values() if r.status == "PASSED"
            )
            failed_tests = total_tests - passed_tests

            compliance_rate = (
                (passed_tests / total_tests * 100) if total_tests > 0 else 0
            )

            report = {
                "report_id": f"USP_Compliance_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "generation_time": datetime.now().isoformat(),
                "summary": {
                    "total_tests": total_tests,
                    "passed_tests": passed_tests,
                    "failed_tests": failed_tests,
                    "compliance_rate_percent": compliance_rate,
                    "overall_status": (
                        "COMPLIANT" if compliance_rate >= 100 else "NON_COMPLIANT"
                    ),
                },
                "test_categories": {
                    "dissolution_usp_711": {
                        "count": len(dissolution_tests),
                        "passed": sum(
                            1 for t in dissolution_tests if t.status == "PASSED"
                        ),
                        "critical": True,
                    },
                    "content_uniformity_usp_905": {
                        "count": len(content_tests),
                        "passed": sum(1 for t in content_tests if t.status == "PASSED"),
                        "critical": True,
                    },
                    "assay_tests": {
                        "count": len(assay_tests),
                        "passed": sum(1 for t in assay_tests if t.status == "PASSED"),
                        "critical": True,
                    },
                    "stability_ich_q1a": {
                        "count": len(stability_tests),
                        "passed": sum(
                            1 for t in stability_tests if t.status == "PASSED"
                        ),
                        "critical": False,
                    },
                },
                "detailed_results": {
                    test_id: result.__dict__
                    for test_id, result in self.test_results.items()
                },
                "regulatory_assessment": self._assess_regulatory_readiness(),
                "recommendations": self._generate_compliance_recommendations(),
            }

            self.logger.info(
                f"✅ USP Compliance Report generated: {compliance_rate:.1f}% compliance"
            )
            return report

        except Exception as e:
            self.logger.error(f"❌ USP Compliance Report generation failed: {e}")
            raise KimeraException(f"Compliance report generation failed: {e}")

    def _assess_regulatory_readiness(self) -> Dict[str, Any]:
        """Assess regulatory submission readiness."""
        critical_tests = ["dissolution", "content", "assay"]
        critical_passed = 0
        critical_total = 0

        for result in self.test_results.values():
            if any(
                test_type in result.test_name.lower() for test_type in critical_tests
            ):
                critical_total += 1
                if result.status == "PASSED":
                    critical_passed += 1

        readiness_score = (
            (critical_passed / critical_total * 100) if critical_total > 0 else 0
        )

        return {
            "readiness_score": readiness_score,
            "critical_tests_passed": critical_passed,
            "critical_tests_total": critical_total,
            "regulatory_status": "READY" if readiness_score >= 100 else "NOT_READY",
            "submission_recommendation": (
                "APPROVED" if readiness_score >= 100 else "REQUIRES_REMEDIATION"
            ),
        }

    def _generate_compliance_recommendations(self) -> List[str]:
        """Generate compliance recommendations based on test results."""
        recommendations = []

        failed_tests = [r for r in self.test_results.values() if r.status == "FAILED"]

        for test in failed_tests:
            if "dissolution" in test.test_name.lower():
                recommendations.append(
                    "Dissolution failure: Optimize coating parameters or polymer ratios"
                )
            elif "content" in test.test_name.lower():
                recommendations.append(
                    "Content uniformity failure: Improve mixing or filling process"
                )
            elif "assay" in test.test_name.lower():
                recommendations.append(
                    "Assay failure: Verify raw material quality and analytical method"
                )
            elif "stability" in test.test_name.lower():
                recommendations.append(
                    "Stability failure: Investigate packaging or storage conditions"
                )

        if not recommendations:
            recommendations.append(
                "All USP tests passed - formulation ready for regulatory submission"
            )

        return recommendations
