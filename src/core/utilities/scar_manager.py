"""
KIMERA SWM - SCAR MANAGER
=========================

The SCAR Manager handles the collection, analysis, tracking, and resolution
of Semantic Contextual Anomaly Reports throughout the Kimera SWM system.
It provides automated anomaly detection, intelligent analysis, and resolution
coordination across all engines.

This is the system's immune system for cognitive health.
"""

import asyncio
import logging
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from ..data_structures.geoid_state import GeoidState
from ..data_structures.scar_state import (
    ScarSeverity,
    ScarState,
    ScarStatus,
    ScarType,
    create_coherence_breakdown_scar,
    create_emergence_anomaly_scar,
    create_energy_violation_scar,
    create_logical_contradiction_scar,
    create_processing_error_scar,
)

# Configure logging
logger = logging.getLogger(__name__)


class AnalysisMode(Enum):
    """Modes of SCAR analysis"""

    IMMEDIATE = "immediate"  # Immediate analysis and response
    BATCH = "batch"  # Batch analysis for efficiency
    PREDICTIVE = "predictive"  # Predictive anomaly detection
    CONTINUOUS = "continuous"  # Continuous monitoring
    FORENSIC = "forensic"  # Deep forensic analysis


class ResolutionStrategy(Enum):
    """Strategies for SCAR resolution"""

    AUTOMATIC = "automatic"  # Automatic resolution
    SEMI_AUTOMATIC = "semi_auto"  # Semi-automatic with validation
    MANUAL = "manual"  # Manual resolution required
    ESCALATION = "escalation"  # Escalate to higher level
    ISOLATION = "isolation"  # Isolate affected components


@dataclass
class ScarStatistics:
    """Statistics about SCAR management"""

    total_scars: int
    scars_by_type: Dict[ScarType, int]
    scars_by_severity: Dict[ScarSeverity, int]
    scars_by_status: Dict[ScarStatus, int]
    average_resolution_time: float
    resolution_success_rate: float
    most_common_anomaly: Optional[ScarType]
    most_affected_engine: Optional[str]
    system_health_score: float


@dataclass
class AnalysisResult:
    """Result of SCAR analysis"""

    scar_id: str
    analysis_type: str
    root_cause_identified: bool
    root_cause: Optional[str]
    recommended_actions: List[Dict[str, Any]]
    confidence_score: float
    analysis_duration: float
    metadata: Dict[str, Any]


class ScarAnalyzer:
    """
    Analyzes SCARs to identify root causes and recommend resolutions.
    """

    def __init__(self):
        self.analysis_patterns = {}
        self.resolution_patterns = {}
        self.learning_data = []

    def analyze_scar(self, scar: ScarState) -> AnalysisResult:
        """Analyze a SCAR to identify root cause and recommendations"""
        start_time = time.time()

        # Pattern-based analysis
        root_cause, confidence = self._identify_root_cause(scar)

        # Generate recommendations
        recommendations = self._generate_recommendations(scar, root_cause)

        # Create analysis result
        result = AnalysisResult(
            scar_id=scar.scar_id,
            analysis_type="pattern_analysis",
            root_cause_identified=root_cause is not None,
            root_cause=root_cause,
            recommended_actions=recommendations,
            confidence_score=confidence,
            analysis_duration=time.time() - start_time,
            metadata={
                "analysis_patterns_used": len(self.analysis_patterns),
                "evidence_count": len(scar.evidence),
                "affected_geoids_count": len(scar.affected_geoids),
            },
        )

        # Update SCAR with analysis results
        scar.analysis_results[result.analysis_type] = result.metadata
        if root_cause:
            scar.root_cause = root_cause

        return result

    def _identify_root_cause(self, scar: ScarState) -> Tuple[Optional[str], float]:
        """Identify the root cause of an anomaly"""
        if scar.scar_type == ScarType.LOGICAL_CONTRADICTION:
            return self._analyze_logical_contradiction(scar)
        elif scar.scar_type == ScarType.ENERGY_VIOLATION:
            return self._analyze_energy_violation(scar)
        elif scar.scar_type == ScarType.COHERENCE_BREAKDOWN:
            return self._analyze_coherence_breakdown(scar)
        elif scar.scar_type == ScarType.PROCESSING_ERROR:
            return self._analyze_processing_error(scar)
        elif scar.scar_type == ScarType.EMERGENCE_ANOMALY:
            return self._analyze_emergence_anomaly(scar)
        else:
            return self._generic_analysis(scar)

    def _analyze_logical_contradiction(
        self, scar: ScarState
    ) -> Tuple[Optional[str], float]:
        """Analyze logical contradiction"""
        # Look for contradictory predicates in evidence
        contradictory_predicates = []
        for evidence in scar.evidence:
            if evidence.evidence_type == "logical_analysis":
                predicates = evidence.evidence_data.get("predicates", [])
                contradictory_predicates.extend(predicates)

        if contradictory_predicates:
            return (
                f"Contradictory logical predicates detected: {contradictory_predicates[:3]}",
                0.9,
            )
        else:
            return "Unknown logical contradiction source", 0.3

    def _analyze_energy_violation(self, scar: ScarState) -> Tuple[Optional[str], float]:
        """Analyze energy conservation violation"""
        for evidence in scar.evidence:
            if evidence.evidence_type == "energy_measurement":
                violation_magnitude = evidence.evidence_data.get(
                    "violation_magnitude", 0.0
                )
                if violation_magnitude > 10.0:
                    return "Major energy leak or creation detected", 0.95
                elif violation_magnitude > 5.0:
                    return "Significant energy imbalance in processing", 0.85
                else:
                    return "Minor energy accounting error", 0.7

        return "Unknown energy violation cause", 0.4

    def _analyze_coherence_breakdown(
        self, scar: ScarState
    ) -> Tuple[Optional[str], float]:
        """Analyze coherence breakdown"""
        for evidence in scar.evidence:
            if evidence.evidence_type == "coherence_analysis":
                coherence_loss = evidence.evidence_data.get("coherence_loss", 0.0)
                if coherence_loss > 0.5:
                    return "Severe semantic-symbolic misalignment", 0.9
                elif coherence_loss > 0.3:
                    return "Moderate processing inconsistency", 0.8
                else:
                    return "Minor coherence degradation", 0.6

        return "Unknown coherence breakdown cause", 0.3

    def _analyze_processing_error(self, scar: ScarState) -> Tuple[Optional[str], float]:
        """Analyze processing error"""
        for evidence in scar.evidence:
            if evidence.evidence_type == "error_trace":
                error_message = evidence.evidence_data.get("error_message", "")
                if "validation" in error_message.lower():
                    return "Input validation failure", 0.85
                elif "memory" in error_message.lower():
                    return "Memory or resource limitation", 0.8
                elif "timeout" in error_message.lower():
                    return "Processing timeout exceeded", 0.9
                else:
                    return f"Engine-specific error: {error_message[:100]}", 0.7

        return "Unknown processing error", 0.4

    def _analyze_emergence_anomaly(
        self, scar: ScarState
    ) -> Tuple[Optional[str], float]:
        """Analyze emergence anomaly"""
        affected_count = len(scar.affected_geoids)
        if affected_count > 10:
            return "Large-scale collective behavior anomaly", 0.8
        elif affected_count > 5:
            return "Medium-scale emergent pattern", 0.7
        else:
            return "Small-scale interaction anomaly", 0.6

    def _generic_analysis(self, scar: ScarState) -> Tuple[Optional[str], float]:
        """Generic analysis for unknown anomaly types"""
        severity_mapping = {
            ScarSeverity.CRITICAL: (
                "System-critical anomaly requiring immediate attention",
                0.9,
            ),
            ScarSeverity.HIGH: (
                "High-impact anomaly affecting system performance",
                0.8,
            ),
            ScarSeverity.MEDIUM: ("Moderate anomaly with localized impact", 0.6),
            ScarSeverity.LOW: ("Minor anomaly with minimal impact", 0.4),
            ScarSeverity.INFORMATIONAL: ("Informational anomaly for tracking", 0.2),
        }

        return severity_mapping.get(scar.severity, ("Unknown anomaly", 0.3))

    def _generate_recommendations(
        self, scar: ScarState, root_cause: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Generate resolution recommendations"""
        recommendations = []

        if scar.scar_type == ScarType.LOGICAL_CONTRADICTION:
            recommendations.extend(
                [
                    {
                        "action_type": "predicate_reconciliation",
                        "description": "Reconcile contradictory logical predicates",
                        "priority": "high",
                        "estimated_duration": 300,
                        "parameters": {"affected_geoids": scar.affected_geoids},
                    },
                    {
                        "action_type": "symbolic_state_cleanup",
                        "description": "Clean up inconsistent symbolic states",
                        "priority": "medium",
                        "estimated_duration": 180,
                        "parameters": {"validation_level": "strict"},
                    },
                ]
            )

        elif scar.scar_type == ScarType.ENERGY_VIOLATION:
            recommendations.extend(
                [
                    {
                        "action_type": "energy_rebalancing",
                        "description": "Rebalance system energy conservation",
                        "priority": "critical",
                        "estimated_duration": 120,
                        "parameters": {"conservation_mode": "strict"},
                    },
                    {
                        "action_type": "thermodynamic_audit",
                        "description": "Audit thermodynamic calculations",
                        "priority": "high",
                        "estimated_duration": 600,
                        "parameters": {"audit_depth": "comprehensive"},
                    },
                ]
            )

        elif scar.scar_type == ScarType.COHERENCE_BREAKDOWN:
            recommendations.extend(
                [
                    {
                        "action_type": "coherence_restoration",
                        "description": "Restore semantic-symbolic coherence",
                        "priority": "high",
                        "estimated_duration": 240,
                        "parameters": {"coherence_threshold": 0.8},
                    },
                    {
                        "action_type": "state_synchronization",
                        "description": "Synchronize semantic and symbolic states",
                        "priority": "medium",
                        "estimated_duration": 300,
                        "parameters": {"sync_mode": "bidirectional"},
                    },
                ]
            )

        elif scar.scar_type == ScarType.PROCESSING_ERROR:
            recommendations.extend(
                [
                    {
                        "action_type": "error_recovery",
                        "description": "Recover from processing error",
                        "priority": "high",
                        "estimated_duration": 60,
                        "parameters": {"recovery_mode": "safe"},
                    },
                    {
                        "action_type": "input_validation",
                        "description": "Enhance input validation",
                        "priority": "medium",
                        "estimated_duration": 120,
                        "parameters": {"validation_strictness": "enhanced"},
                    },
                ]
            )

        else:
            # Generic recommendations
            recommendations.append(
                {
                    "action_type": "anomaly_isolation",
                    "description": "Isolate anomaly to prevent spread",
                    "priority": "medium",
                    "estimated_duration": 180,
                    "parameters": {"isolation_scope": "local"},
                }
            )

        return recommendations


class ScarManager:
    """
    SCAR Manager - System Cognitive Health Management
    ================================================

    The ScarManager serves as the central hub for managing all Semantic
    Contextual Anomaly Reports in the Kimera SWM system. It provides
    automated detection, intelligent analysis, and coordinated resolution
    of cognitive anomalies.

    Key Responsibilities:
    - Automated anomaly detection across all engines
    - Intelligent root cause analysis
    - Coordinated resolution strategies
    - System health monitoring and reporting
    - Preventive anomaly prediction
    """

    def __init__(self, mode: AnalysisMode = AnalysisMode.CONTINUOUS):
        self.mode = mode
        self.active_scars: Dict[str, ScarState] = {}
        self.resolved_scars: Dict[str, ScarState] = {}
        self.scar_history: List[ScarState] = []

        # Analysis and resolution
        self.analyzer = ScarAnalyzer()
        self.resolution_handlers: Dict[str, Callable] = {}
        self.auto_resolution_enabled = True

        # Monitoring and statistics
        self.total_scars_processed = 0
        self.resolution_success_count = 0
        self.system_health_score = 1.0
        self.last_health_check = datetime.now()

        # Performance tracking
        self.processing_metrics = {
            "average_analysis_time": 0.0,
            "average_resolution_time": 0.0,
            "detection_accuracy": 0.95,
            "false_positive_rate": 0.05,
        }

        # Threading for background processing
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.background_tasks = set()

        # Register default resolution handlers
        self._register_default_handlers()

        logger.info(f"ScarManager initialized with mode: {mode.value}")

    def report_anomaly(self, scar: ScarState) -> str:
        """Report a new anomaly to the SCAR system"""
        scar_id = scar.scar_id

        # Deduplicate similar SCARs
        similar_scar = self._find_similar_active_scar(scar)
        if similar_scar:
            self._merge_scars(similar_scar, scar)
            logger.info(
                f"Merged SCAR {scar_id[:8]} with similar active SCAR {similar_scar.scar_id[:8]}"
            )
            return similar_scar.scar_id

        # Add to active SCARs
        self.active_scars[scar_id] = scar
        self.total_scars_processed += 1

        # Immediate analysis for critical SCARs
        if scar.severity == ScarSeverity.CRITICAL:
            self._analyze_scar_immediately(scar)
        elif self.mode == AnalysisMode.IMMEDIATE:
            self._analyze_scar_immediately(scar)
        else:
            # Queue for batch processing
            self._queue_for_analysis(scar)

        logger.info(f"Reported new SCAR: {scar_id[:8]} - {scar.title}")
        return scar_id

    def get_scar(self, scar_id: str) -> Optional[ScarState]:
        """Get a SCAR by ID"""
        return self.active_scars.get(scar_id) or self.resolved_scars.get(scar_id)

    def get_active_scars(
        self,
        severity: Optional[ScarSeverity] = None,
        scar_type: Optional[ScarType] = None,
    ) -> List[ScarState]:
        """Get active SCARs with optional filtering"""
        scars = list(self.active_scars.values())

        if severity:
            scars = [s for s in scars if s.severity == severity]

        if scar_type:
            scars = [s for s in scars if s.scar_type == scar_type]

        return scars

    def resolve_scar(
        self, scar_id: str, resolution_summary: str, effectiveness: float = 1.0
    ) -> bool:
        """Mark a SCAR as resolved"""
        if scar_id not in self.active_scars:
            logger.warning(f"Cannot resolve non-existent SCAR: {scar_id[:8]}")
            return False

        scar = self.active_scars[scar_id]
        scar.mark_resolved(resolution_summary, effectiveness)

        # Move to resolved SCARs
        self.resolved_scars[scar_id] = scar
        del self.active_scars[scar_id]

        # Update success metrics
        if effectiveness >= 0.8:
            self.resolution_success_count += 1

        # Add to history
        self.scar_history.append(scar)

        logger.info(
            f"Resolved SCAR {scar_id[:8]} with effectiveness {effectiveness:.2f}"
        )
        return True

    def analyze_system_health(self) -> float:
        """Analyze overall system health based on SCAR patterns"""
        if not self.active_scars and not self.resolved_scars:
            return 1.0

        # Base health score
        health_score = 1.0

        # Penalty for active SCARs
        critical_scars = len(
            [
                s
                for s in self.active_scars.values()
                if s.severity == ScarSeverity.CRITICAL
            ]
        )
        high_scars = len(
            [s for s in self.active_scars.values() if s.severity == ScarSeverity.HIGH]
        )
        medium_scars = len(
            [s for s in self.active_scars.values() if s.severity == ScarSeverity.MEDIUM]
        )

        health_score -= critical_scars * 0.2
        health_score -= high_scars * 0.1
        health_score -= medium_scars * 0.05

        # Bonus for resolution success rate
        if self.total_scars_processed > 0:
            success_rate = self.resolution_success_count / self.total_scars_processed
            health_score += success_rate * 0.1

        # Time factor - recent SCARs have more impact
        recent_threshold = datetime.now() - timedelta(hours=1)
        recent_scars = [
            s
            for s in self.active_scars.values()
            if s.metrics.detection_time > recent_threshold
        ]

        if recent_scars:
            health_score -= len(recent_scars) * 0.02

        self.system_health_score = max(0.0, min(1.0, health_score))
        self.last_health_check = datetime.now()

        return self.system_health_score

    def get_statistics(self) -> ScarStatistics:
        """Get comprehensive SCAR statistics"""
        all_scars = list(self.active_scars.values()) + list(
            self.resolved_scars.values()
        )

        if not all_scars:
            return ScarStatistics(
                total_scars=0,
                scars_by_type={},
                scars_by_severity={},
                scars_by_status={},
                average_resolution_time=0.0,
                resolution_success_rate=0.0,
                most_common_anomaly=None,
                most_affected_engine=None,
                system_health_score=1.0,
            )

        # Count by type
        scars_by_type = defaultdict(int)
        for scar in all_scars:
            scars_by_type[scar.scar_type] += 1

        # Count by severity
        scars_by_severity = defaultdict(int)
        for scar in all_scars:
            scars_by_severity[scar.severity] += 1

        # Count by status
        scars_by_status = defaultdict(int)
        for scar in all_scars:
            scars_by_status[scar.status] += 1

        # Calculate average resolution time
        resolved_scars_with_time = [
            s
            for s in self.resolved_scars.values()
            if s.metrics.resolution_duration is not None
        ]
        avg_resolution_time = (
            (
                sum(s.metrics.resolution_duration for s in resolved_scars_with_time)
                / len(resolved_scars_with_time)
            )
            if resolved_scars_with_time
            else 0.0
        )

        # Resolution success rate
        success_rate = (
            (self.resolution_success_count / self.total_scars_processed)
            if self.total_scars_processed > 0
            else 0.0
        )

        # Most common anomaly
        most_common_anomaly = (
            max(scars_by_type.items(), key=lambda x: x[1])[0] if scars_by_type else None
        )

        # Most affected engine
        engine_counts = defaultdict(int)
        for scar in all_scars:
            for engine in scar.affected_engines:
                engine_counts[engine] += 1

        most_affected_engine = (
            max(engine_counts.items(), key=lambda x: x[1])[0] if engine_counts else None
        )

        return ScarStatistics(
            total_scars=len(all_scars),
            scars_by_type=dict(scars_by_type),
            scars_by_severity=dict(scars_by_severity),
            scars_by_status=dict(scars_by_status),
            average_resolution_time=avg_resolution_time,
            resolution_success_rate=success_rate,
            most_common_anomaly=most_common_anomaly,
            most_affected_engine=most_affected_engine,
            system_health_score=self.analyze_system_health(),
        )

    def _find_similar_active_scar(self, new_scar: ScarState) -> Optional[ScarState]:
        """Find similar active SCAR to avoid duplicates"""
        for active_scar in self.active_scars.values():
            # Same type and similar affected geoids
            if active_scar.scar_type == new_scar.scar_type and set(
                active_scar.affected_geoids
            ) & set(new_scar.affected_geoids):
                return active_scar

            # Same type and same affected engines
            if active_scar.scar_type == new_scar.scar_type and set(
                active_scar.affected_engines
            ) & set(new_scar.affected_engines):
                return active_scar

        return None

    def _merge_scars(self, existing_scar: ScarState, new_scar: ScarState) -> None:
        """Merge a new SCAR with an existing similar one"""
        # Merge evidence
        existing_scar.evidence.extend(new_scar.evidence)

        # Merge affected geoids and engines
        for geoid_id in new_scar.affected_geoids:
            if geoid_id not in existing_scar.affected_geoids:
                existing_scar.affected_geoids.append(geoid_id)

        for engine in new_scar.affected_engines:
            if engine not in existing_scar.affected_engines:
                existing_scar.affected_engines.append(engine)

        # Update severity if new SCAR is more severe
        if new_scar.severity.value > existing_scar.severity.value:
            existing_scar.severity = new_scar.severity

        # Increment recurrence count
        existing_scar.metrics.recurrence_count += 1

    def _analyze_scar_immediately(self, scar: ScarState) -> None:
        """Analyze a SCAR immediately"""
        try:
            analysis_result = self.analyzer.analyze_scar(scar)

            # Apply automatic resolution if enabled and confidence is high
            if (
                self.auto_resolution_enabled
                and analysis_result.confidence_score >= 0.8
                and analysis_result.recommended_actions
            ):

                self._apply_automatic_resolution(scar, analysis_result)

        except Exception as e:
            logger.error(f"Error analyzing SCAR {scar.scar_id[:8]}: {str(e)}")

    def _queue_for_analysis(self, scar: ScarState) -> None:
        """Queue a SCAR for batch analysis"""
        # Submit to background processing
        future = self.executor.submit(self._analyze_scar_immediately, scar)
        self.background_tasks.add(future)

        # Clean up completed tasks
        self.background_tasks = {
            task for task in self.background_tasks if not task.done()
        }

    def _apply_automatic_resolution(
        self, scar: ScarState, analysis: AnalysisResult
    ) -> None:
        """Apply automatic resolution actions"""
        for recommendation in analysis.recommended_actions:
            action_type = recommendation["action_type"]

            if action_type in self.resolution_handlers:
                try:
                    handler = self.resolution_handlers[action_type]
                    success = handler(scar, recommendation["parameters"])

                    # Record resolution action
                    action = scar.apply_resolution_action(
                        action_type=action_type,
                        description=recommendation["description"],
                        parameters=recommendation["parameters"],
                        applied_by="AutomaticResolution",
                    )
                    action.success = success

                    if success:
                        logger.info(
                            f"Applied automatic resolution {action_type} to SCAR {scar.scar_id[:8]}"
                        )
                    else:
                        logger.warning(
                            f"Failed automatic resolution {action_type} for SCAR {scar.scar_id[:8]}"
                        )

                except Exception as e:
                    logger.error(
                        f"Error applying automatic resolution {action_type}: {str(e)}"
                    )

    def _register_default_handlers(self) -> None:
        """Register default resolution handlers"""
        self.resolution_handlers.update(
            {
                "predicate_reconciliation": self._handle_predicate_reconciliation,
                "energy_rebalancing": self._handle_energy_rebalancing,
                "coherence_restoration": self._handle_coherence_restoration,
                "error_recovery": self._handle_error_recovery,
                "anomaly_isolation": self._handle_anomaly_isolation,
            }
        )

    def _handle_predicate_reconciliation(
        self, scar: ScarState, parameters: Dict[str, Any]
    ) -> bool:
        """Handle predicate reconciliation"""
        # Placeholder for actual implementation
        logger.info(f"Reconciling predicates for SCAR {scar.scar_id[:8]}")
        return True

    def _handle_energy_rebalancing(
        self, scar: ScarState, parameters: Dict[str, Any]
    ) -> bool:
        """Handle energy rebalancing"""
        # Placeholder for actual implementation
        logger.info(f"Rebalancing energy for SCAR {scar.scar_id[:8]}")
        return True

    def _handle_coherence_restoration(
        self, scar: ScarState, parameters: Dict[str, Any]
    ) -> bool:
        """Handle coherence restoration"""
        # Placeholder for actual implementation
        logger.info(f"Restoring coherence for SCAR {scar.scar_id[:8]}")
        return True

    def _handle_error_recovery(
        self, scar: ScarState, parameters: Dict[str, Any]
    ) -> bool:
        """Handle error recovery"""
        # Placeholder for actual implementation
        logger.info(f"Recovering from error for SCAR {scar.scar_id[:8]}")
        return True

    def _handle_anomaly_isolation(
        self, scar: ScarState, parameters: Dict[str, Any]
    ) -> bool:
        """Handle anomaly isolation"""
        # Placeholder for actual implementation
        logger.info(f"Isolating anomaly for SCAR {scar.scar_id[:8]}")
        return True


# Global SCAR manager instance
_global_scar_manager: Optional[ScarManager] = None


def get_global_scar_manager() -> ScarManager:
    """Get the global SCAR manager instance"""
    global _global_scar_manager
    if _global_scar_manager is None:
        _global_scar_manager = ScarManager()
    return _global_scar_manager


def initialize_scar_manager(
    mode: AnalysisMode = AnalysisMode.CONTINUOUS,
) -> ScarManager:
    """Initialize the global SCAR manager with custom parameters"""
    global _global_scar_manager
    _global_scar_manager = ScarManager(mode)
    return _global_scar_manager


# Convenience functions for common SCAR operations
def report_logical_contradiction(
    conflicting_geoids: List[GeoidState], description: str
) -> str:
    """Convenience function to report logical contradiction"""
    scar = create_logical_contradiction_scar(conflicting_geoids, description)
    manager = get_global_scar_manager()
    return manager.report_anomaly(scar)


def report_energy_violation(geoid: GeoidState, expected: float, actual: float) -> str:
    """Convenience function to report energy violation"""
    scar = create_energy_violation_scar(geoid, expected, actual)
    manager = get_global_scar_manager()
    return manager.report_anomaly(scar)


def report_coherence_breakdown(
    geoid: GeoidState, prev_coherence: float, curr_coherence: float
) -> str:
    """Convenience function to report coherence breakdown"""
    scar = create_coherence_breakdown_scar(geoid, prev_coherence, curr_coherence)
    manager = get_global_scar_manager()
    return manager.report_anomaly(scar)


def get_system_health() -> float:
    """Convenience function to get system health score"""
    manager = get_global_scar_manager()
    return manager.analyze_system_health()
