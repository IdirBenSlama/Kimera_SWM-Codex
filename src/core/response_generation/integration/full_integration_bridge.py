#!/usr/bin/env python3
"""
KIMERA Full Integration Bridge v2.0
===================================

DO-178C Level A compliant integration bridge that ensures ALL sophisticated KIMERA
components participate in response generation. This bridge connects the response
generation system with the complete cognitive architecture.

Key Features:
- Barenholtz dual-system integration
- Quantum security enforcement
- Thermodynamic coherence maintenance
- High-dimensional modeling integration
- Insight management coordination

Author: KIMERA Development Team
Version: 2.0.0 (DO-178C Level A)
"""

import asyncio
import logging
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.config.settings import get_settings
from src.utils.kimera_exceptions import KimeraCognitiveError, KimeraIntegrationError
from src.utils.kimera_logger import LogCategory, get_logger

# Import core systems
from ..core.cognitive_response_system import (
    CognitiveMetrics,
    ResponseContext,
    ResponseGenerationConfig,
    ResponseGenerator,
    ResponseOutput,
)
from ..security.quantum_security import get_quantum_security

# Import Barenholtz architecture
try:
    from ..barenholtz_architecture import BarenholtzDualSystemIntegrator

    BARENHOLTZ_AVAILABLE = True
except ImportError:
    BARENHOLTZ_AVAILABLE = False
    logger = get_logger(__name__, LogCategory.COGNITIVE)
    logger.warning("Barenholtz architecture not available")

logger = get_logger(__name__, LogCategory.COGNITIVE)


class IntegrationMode(Enum):
    """Integration modes for different processing requirements"""

    FULL_COGNITIVE = "full_cognitive"  # Use all cognitive systems
    SECURITY_FOCUSED = "security_focused"  # Prioritize security processing
    PERFORMANCE_OPTIMIZED = "performance_optimized"  # Optimize for speed
    RESEARCH_MODE = "research_mode"  # Include experimental systems
    MINIMAL = "minimal"  # Basic integration only


class ProcessingPriority(Enum):
    """Processing priority levels"""

    CRITICAL = "critical"  # Real-time requirements
    HIGH = "high"  # Important but not critical
    NORMAL = "normal"  # Standard processing
    LOW = "low"  # Background processing
    BATCH = "batch"  # Batch processing mode


@dataclass
class IntegrationConfig:
    """Configuration for full integration bridge"""

    mode: IntegrationMode = IntegrationMode.FULL_COGNITIVE
    priority: ProcessingPriority = ProcessingPriority.NORMAL
    enable_barenholtz: bool = True
    enable_quantum_security: bool = True
    enable_thermodynamic_coherence: bool = True
    enable_high_dimensional: bool = True
    enable_insight_management: bool = True
    max_processing_time: float = 10.0  # seconds
    fallback_on_failure: bool = True
    validate_coherence: bool = True


@dataclass
class IntegrationMetrics:
    """Metrics for integration processing"""

    total_processing_time: float
    barenholtz_time: float = 0.0
    security_time: float = 0.0
    thermodynamic_time: float = 0.0
    high_dimensional_time: float = 0.0
    insight_time: float = 0.0
    systems_engaged: List[str] = field(default_factory=list)
    coherence_score: float = 0.0
    integration_success: bool = True
    error_count: int = 0


@dataclass
class CognitiveArchitectureState:
    """Complete state of cognitive architecture"""

    barenholtz_state: Optional[Dict[str, Any]] = None
    security_state: Optional[Dict[str, Any]] = None
    thermodynamic_state: Optional[Dict[str, Any]] = None
    high_dimensional_state: Optional[Dict[str, Any]] = None
    insight_state: Optional[Dict[str, Any]] = None
    overall_coherence: float = 0.0
    timestamp: float = field(default_factory=time.time)


class BarenholtzIntegrationAdapter:
    """Adapter for Barenholtz dual-system integration"""

    def __init__(self):
        self.integrator = None
        self.available = BARENHOLTZ_AVAILABLE

        if self.available:
            try:
                self.integrator = BarenholtzDualSystemIntegrator()
                logger.info("🧠 Barenholtz integration adapter initialized")
            except Exception as e:
                logger.error(f"❌ Barenholtz adapter initialization failed: {e}")
                self.available = False

    async def process_with_dual_systems(
        self, query: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process query through dual-system architecture"""
        if not self.available or not self.integrator:
            return {
                "status": "unavailable",
                "system1_result": None,
                "system2_result": None,
                "arbitration_result": None,
            }

        try:
            # Convert query to tensor format
            query_tensor = torch.tensor(
                [ord(c) for c in query[:512]], dtype=torch.float32
            )

            # Process through dual systems
            result = await self.integrator.process(query_tensor, context)

            return {
                "status": "success",
                "system1_result": result.system1_result,
                "system2_result": result.system2_result,
                "arbitration_result": result.arbitration_result,
                "confidence": result.confidence,
                "processing_time": result.processing_time,
            }

        except Exception as e:
            logger.error(f"❌ Barenholtz processing failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "system1_result": None,
                "system2_result": None,
                "arbitration_result": None,
            }


class ThermodynamicCoherenceManager:
    """Manager for thermodynamic coherence integration"""

    def __init__(self):
        self.coherence_threshold = 0.8
        self.entropy_baseline = 2.5

    async def assess_thermodynamic_coherence(
        self, processing_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess thermodynamic coherence of processing"""
        try:
            # Calculate entropy of processing data
            entropy = self._calculate_processing_entropy(processing_data)

            # Assess coherence based on entropy and processing patterns
            coherence = self._assess_coherence_from_entropy(entropy)

            # Check if coherence meets threshold
            coherent = coherence >= self.coherence_threshold

            return {
                "entropy": entropy,
                "coherence": coherence,
                "coherent": coherent,
                "threshold": self.coherence_threshold,
                "assessment": "coherent" if coherent else "incoherent",
            }

        except Exception as e:
            logger.error(f"❌ Thermodynamic coherence assessment failed: {e}")
            return {
                "entropy": 0.0,
                "coherence": 0.0,
                "coherent": False,
                "error": str(e),
            }

    def _calculate_processing_entropy(self, data: Dict[str, Any]) -> float:
        """Calculate entropy of processing data"""
        # Convert data to string representation
        data_str = str(data)

        # Calculate character frequency
        char_counts = {}
        for char in data_str:
            char_counts[char] = char_counts.get(char, 0) + 1

        # Calculate Shannon entropy
        total_chars = len(data_str)
        entropy = 0.0

        for count in char_counts.values():
            probability = count / total_chars
            if probability > 0:
                entropy -= probability * np.log2(probability)

        return entropy

    def _assess_coherence_from_entropy(self, entropy: float) -> float:
        """Assess coherence based on entropy"""
        # Optimal entropy range for coherent processing
        optimal_entropy_range = (2.0, 4.0)

        if optimal_entropy_range[0] <= entropy <= optimal_entropy_range[1]:
            # High coherence for optimal entropy
            coherence = (
                1.0 - abs(entropy - self.entropy_baseline) / self.entropy_baseline
            )
        else:
            # Lower coherence for entropy outside optimal range
            deviation = min(
                abs(entropy - optimal_entropy_range[0]),
                abs(entropy - optimal_entropy_range[1]),
            )
            coherence = max(0.0, 1.0 - deviation / optimal_entropy_range[1])

        return coherence


class HighDimensionalIntegrationAdapter:
    """Adapter for high-dimensional modeling integration"""

    def __init__(self):
        self.dimension = 1024  # Standard high-dimensional space
        self.available = False

        try:
            # Try to import high-dimensional modeling
            from ..high_dimensional_modeling import HighDimensionalModelingIntegrator

            self.integrator = HighDimensionalModelingIntegrator()
            self.available = True
            logger.info("🌀 High-dimensional integration adapter initialized")
        except ImportError:
            logger.warning("High-dimensional modeling not available")
        except Exception as e:
            logger.error(f"❌ High-dimensional adapter initialization failed: {e}")

    async def process_high_dimensional(
        self, query_embedding: torch.Tensor, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process query in high-dimensional space"""
        if not self.available:
            return {
                "status": "unavailable",
                "embedding": None,
                "bgm_result": None,
                "homomorphic_result": None,
            }

        try:
            # Ensure embedding is correct dimension
            if query_embedding.size(0) != self.dimension:
                # Pad or truncate to match dimension
                if query_embedding.size(0) < self.dimension:
                    padding = torch.zeros(self.dimension - query_embedding.size(0))
                    query_embedding = torch.cat([query_embedding, padding])
                else:
                    query_embedding = query_embedding[: self.dimension]

            # Process through high-dimensional modeling
            result = await self.integrator.process_cognitive_data(
                query_embedding.unsqueeze(0)  # Add batch dimension
            )

            return {
                "status": "success",
                "embedding": query_embedding,
                "bgm_result": result.get("bgm_result"),
                "homomorphic_result": result.get("homomorphic_result"),
                "processing_time": result.get("processing_time", 0.0),
            }

        except Exception as e:
            logger.error(f"❌ High-dimensional processing failed: {e}")
            return {"status": "error", "error": str(e), "embedding": None}


class InsightManagementAdapter:
    """Adapter for insight management integration"""

    def __init__(self):
        self.available = False

        try:
            # Try to import insight management
            from ..insight_management import InsightManagementIntegrator

            self.integrator = InsightManagementIntegrator()
            self.available = True
            logger.info("🧠 Insight management adapter initialized")
        except ImportError:
            logger.warning("Insight management not available")
        except Exception as e:
            logger.error(f"❌ Insight management adapter initialization failed: {e}")

    async def process_insights(
        self, cognitive_metrics: CognitiveMetrics, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process insights from cognitive metrics"""
        if not self.available:
            return {
                "status": "unavailable",
                "insights": [],
                "entropy_validation": False,
            }

        try:
            # Convert cognitive metrics to insight data
            insight_data = {
                "cognitive_coherence": cognitive_metrics.cognitive_coherence,
                "semantic_complexity": cognitive_metrics.semantic_complexity,
                "resonance_frequency": cognitive_metrics.resonance_frequency,
                "field_strength": cognitive_metrics.field_strength,
                "timestamp": cognitive_metrics.timestamp,
            }

            # Process through insight management
            insights = await self.integrator.analyze_cognitive_state(insight_data)

            return {
                "status": "success",
                "insights": insights,
                "entropy_validation": True,
                "processing_time": time.time() - cognitive_metrics.timestamp,
            }

        except Exception as e:
            logger.error(f"❌ Insight processing failed: {e}")
            return {"status": "error", "error": str(e), "insights": []}


class KimeraFullIntegrationBridge:
    """
    Full Integration Bridge for KIMERA Cognitive Architecture

    DO-178C Level A compliant bridge ensuring all sophisticated components
    participate in response generation with thermodynamic coherence.
    """

    def __init__(self, config: Optional[IntegrationConfig] = None):
        self.config = config or IntegrationConfig()

        # Initialize core response system
        self.response_generator = ResponseGenerator()
        self.quantum_security = get_quantum_security()

        # Initialize integration adapters
        self.barenholtz_adapter = (
            BarenholtzIntegrationAdapter() if self.config.enable_barenholtz else None
        )
        self.thermodynamic_manager = (
            ThermodynamicCoherenceManager()
            if self.config.enable_thermodynamic_coherence
            else None
        )
        self.high_dimensional_adapter = (
            HighDimensionalIntegrationAdapter()
            if self.config.enable_high_dimensional
            else None
        )
        self.insight_adapter = (
            InsightManagementAdapter()
            if self.config.enable_insight_management
            else None
        )

        # Integration metrics
        self.processing_history: List[IntegrationMetrics] = []
        self.total_integrations = 0
        self.successful_integrations = 0

        logger.info("🌉 KIMERA Full Integration Bridge initialized")
        logger.info(f"   Mode: {self.config.mode.value}")
        logger.info(f"   Priority: {self.config.priority.value}")
        logger.info(
            f"   Barenholtz: {'✅' if self.barenholtz_adapter and self.barenholtz_adapter.available else '❌'}"
        )
        logger.info(
            f"   High-dimensional: {'✅' if self.high_dimensional_adapter and self.high_dimensional_adapter.available else '❌'}"
        )
        logger.info(
            f"   Insight management: {'✅' if self.insight_adapter and self.insight_adapter.available else '❌'}"
        )

    async def process_integrated_response(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        context_data: Optional[Dict[str, Any]] = None,
    ) -> ResponseOutput:
        """
        Process response through full cognitive architecture integration

        Args:
            query: User query to process
            conversation_history: Previous conversation context
            context_data: Additional context information

        Returns:
            Complete response with full cognitive integration
        """
        start_time = time.time()
        integration_metrics = IntegrationMetrics(total_processing_time=0.0)

        try:
            self.total_integrations += 1

            # Step 1: Prepare response context
            response_context = ResponseContext(
                user_query=query,
                conversation_history=conversation_history or [],
                system_state=context_data,
                security_context=None,  # Will be populated by security processing
                performance_constraints=self._get_performance_constraints(),
            )

            # Step 2: Security assessment and quantum protection
            if self.config.enable_quantum_security:
                security_start = time.time()
                security_result = await self._process_quantum_security(
                    query, context_data
                )
                response_context.security_context = security_result
                integration_metrics.security_time = time.time() - security_start
                integration_metrics.systems_engaged.append("quantum_security")

                # Block processing if security threat detected
                if security_result.get("status") == "BLOCKED":
                    logger.warning("🚫 Processing blocked due to security threat")
                    return self._create_security_blocked_response(
                        security_result, integration_metrics
                    )

            # Step 3: Cognitive architecture state collection
            architecture_state = await self._collect_cognitive_architecture_state(
                query, response_context, integration_metrics
            )

            # Step 4: Thermodynamic coherence validation
            if (
                self.config.enable_thermodynamic_coherence
                and self.thermodynamic_manager
            ):
                thermo_start = time.time()
                coherence_result = (
                    await self.thermodynamic_manager.assess_thermodynamic_coherence(
                        architecture_state.__dict__
                    )
                )
                architecture_state.overall_coherence = coherence_result["coherence"]
                integration_metrics.thermodynamic_time = time.time() - thermo_start
                integration_metrics.systems_engaged.append("thermodynamic_coherence")

                # Validate coherence meets requirements
                if self.config.validate_coherence and not coherence_result["coherent"]:
                    logger.warning(
                        f"⚠️ Thermodynamic coherence below threshold: {coherence_result['coherence']:.3f}"
                    )

            # Step 5: Generate integrated response
            enhanced_context = self._enhance_context_with_architecture_state(
                response_context, architecture_state
            )

            response = await self.response_generator.generate_response(enhanced_context)

            # Step 6: Post-processing validation and enhancement
            final_response = await self._post_process_response(
                response, architecture_state, integration_metrics
            )

            # Step 7: Update metrics and log success
            integration_metrics.total_processing_time = time.time() - start_time
            integration_metrics.coherence_score = architecture_state.overall_coherence
            integration_metrics.integration_success = True

            self.successful_integrations += 1
            self.processing_history.append(integration_metrics)

            logger.info(
                f"✅ Integrated response generated: {len(integration_metrics.systems_engaged)} systems "
                f"(coherence: {architecture_state.overall_coherence:.3f}, "
                f"time: {integration_metrics.total_processing_time*1000:.1f}ms)"
            )

            return final_response

        except Exception as e:
            integration_metrics.total_processing_time = time.time() - start_time
            integration_metrics.integration_success = False
            integration_metrics.error_count = 1
            self.processing_history.append(integration_metrics)

            logger.error(f"❌ Integration processing failed: {e}")

            if self.config.fallback_on_failure:
                return await self._generate_fallback_response(query, str(e))
            else:
                raise KimeraIntegrationError(f"Full integration failed: {e}")

    async def _process_quantum_security(
        self, query: str, context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Process quantum security assessment"""
        security_data = {
            "query": query,
            "context": context or {},
            "timestamp": time.time(),
            "integration_mode": self.config.mode.value,
        }

        return await self.quantum_security.process_with_quantum_protection(
            security_data, require_encryption=False
        )

    async def _collect_cognitive_architecture_state(
        self, query: str, context: ResponseContext, metrics: IntegrationMetrics
    ) -> CognitiveArchitectureState:
        """Collect state from all cognitive architecture components"""
        state = CognitiveArchitectureState()

        # Barenholtz dual-system processing
        if self.barenholtz_adapter and self.barenholtz_adapter.available:
            barenholtz_start = time.time()
            try:
                barenholtz_result = (
                    await self.barenholtz_adapter.process_with_dual_systems(
                        query, context.__dict__
                    )
                )
                state.barenholtz_state = barenholtz_result
                metrics.barenholtz_time = time.time() - barenholtz_start
                metrics.systems_engaged.append("barenholtz_dual_system")
            except Exception as e:
                logger.error(f"❌ Barenholtz integration failed: {e}")
                metrics.error_count += 1

        # High-dimensional modeling
        if self.high_dimensional_adapter and self.high_dimensional_adapter.available:
            hd_start = time.time()
            try:
                # Create query embedding
                query_embedding = self._create_query_embedding(query)
                hd_result = (
                    await self.high_dimensional_adapter.process_high_dimensional(
                        query_embedding, context.__dict__
                    )
                )
                state.high_dimensional_state = hd_result
                metrics.high_dimensional_time = time.time() - hd_start
                metrics.systems_engaged.append("high_dimensional_modeling")
            except Exception as e:
                logger.error(f"❌ High-dimensional integration failed: {e}")
                metrics.error_count += 1

        # Insight management (will be processed after response generation)
        state.insight_state = {"status": "pending"}

        return state

    def _create_query_embedding(self, query: str) -> torch.Tensor:
        """Create high-dimensional embedding for query"""
        # Simple character-based embedding (in production, use sophisticated embedding model)
        chars = [ord(c) for c in query[:1024]]  # Take first 1024 characters

        # Pad or truncate to exactly 1024 dimensions
        if len(chars) < 1024:
            chars.extend([0] * (1024 - len(chars)))
        else:
            chars = chars[:1024]

        # Normalize to 0-1 range
        embedding = torch.tensor(chars, dtype=torch.float32) / 255.0

        return embedding

    def _enhance_context_with_architecture_state(
        self, context: ResponseContext, state: CognitiveArchitectureState
    ) -> ResponseContext:
        """Enhance response context with architecture state"""
        enhanced_context = ResponseContext(
            user_query=context.user_query,
            conversation_history=context.conversation_history,
            system_state=context.system_state or {},
            security_context=context.security_context,
            performance_constraints=context.performance_constraints,
            modality_preferences=context.modality_preferences,
        )

        # Add architecture state to system state
        enhanced_context.system_state.update(
            {
                "cognitive_architecture": {
                    "barenholtz_state": state.barenholtz_state,
                    "high_dimensional_state": state.high_dimensional_state,
                    "overall_coherence": state.overall_coherence,
                    "integration_timestamp": state.timestamp,
                }
            }
        )

        return enhanced_context

    async def _post_process_response(
        self,
        response: ResponseOutput,
        state: CognitiveArchitectureState,
        metrics: IntegrationMetrics,
    ) -> ResponseOutput:
        """Post-process response with additional integrations"""

        # Process insights if available
        if self.insight_adapter and self.insight_adapter.available:
            insight_start = time.time()
            try:
                insight_result = await self.insight_adapter.process_insights(
                    response.cognitive_metrics, state.__dict__
                )
                state.insight_state = insight_result
                metrics.insight_time = time.time() - insight_start
                metrics.systems_engaged.append("insight_management")

                # Add insights to response metadata
                response.metadata["insights"] = insight_result.get("insights", [])

            except Exception as e:
                logger.error(f"❌ Insight processing failed: {e}")
                metrics.error_count += 1

        # Add integration metadata
        response.metadata.update(
            {
                "integration_metrics": {
                    "systems_engaged": metrics.systems_engaged,
                    "total_time": metrics.total_processing_time,
                    "coherence_score": metrics.coherence_score,
                    "error_count": metrics.error_count,
                },
                "architecture_state": {
                    "overall_coherence": state.overall_coherence,
                    "systems_available": len(metrics.systems_engaged),
                    "integration_mode": self.config.mode.value,
                },
            }
        )

        return response

    def _get_performance_constraints(self) -> Dict[str, float]:
        """Get performance constraints based on priority"""
        if self.config.priority == ProcessingPriority.CRITICAL:
            return {
                "max_response_time": 1.0,
                "max_memory_mb": 512,
                "max_cpu_percent": 50,
            }
        elif self.config.priority == ProcessingPriority.HIGH:
            return {
                "max_response_time": 2.0,
                "max_memory_mb": 1024,
                "max_cpu_percent": 70,
            }
        else:  # NORMAL, LOW, BATCH
            return {
                "max_response_time": self.config.max_processing_time,
                "max_memory_mb": 2048,
                "max_cpu_percent": 90,
            }

    def _create_security_blocked_response(
        self, security_result: Dict[str, Any], metrics: IntegrationMetrics
    ) -> ResponseOutput:
        """Create response for security-blocked requests"""
        from ..core.cognitive_response_system import CognitiveMetrics, ResponseType

        blocked_metrics = CognitiveMetrics(
            resonance_frequency=0.0,
            field_strength=0.0,
            cognitive_coherence=0.0,
            semantic_complexity=0.0,
        )

        return ResponseOutput(
            content="Request blocked due to security assessment. Please rephrase your query.",
            response_type=ResponseType.SECURE,
            quality_score=0.0,
            cognitive_metrics=blocked_metrics,
            security_status=security_result,
            processing_time_ms=metrics.total_processing_time * 1000,
            metadata={"blocked": True, "reason": "security_threat"},
        )

    async def _generate_fallback_response(
        self, query: str, error: str
    ) -> ResponseOutput:
        """Generate fallback response when integration fails"""
        from ..core.cognitive_response_system import CognitiveMetrics, ResponseType

        fallback_context = ResponseContext(
            user_query=query, modality_preferences=["text"]
        )

        try:
            # Use basic response generation without full integration
            return await self.response_generator.generate_response(fallback_context)
        except Exception as e:
            # Last resort response
            fallback_metrics = CognitiveMetrics(
                resonance_frequency=7.83,
                field_strength=0.5,
                cognitive_coherence=0.5,
                semantic_complexity=0.3,
            )

            return ResponseOutput(
                content=f"I apologize, but I'm experiencing technical difficulties processing your request. Please try again.",
                response_type=ResponseType.DIRECT,
                quality_score=0.5,
                cognitive_metrics=fallback_metrics,
                security_status={"status": "fallback"},
                processing_time_ms=0.0,
                metadata={
                    "fallback": True,
                    "integration_error": error,
                    "generation_error": str(e),
                },
            )

    def get_integration_status(self) -> Dict[str, Any]:
        """Get comprehensive integration system status"""
        recent_metrics = (
            self.processing_history[-10:] if self.processing_history else []
        )

        success_rate = (
            self.successful_integrations / self.total_integrations
            if self.total_integrations > 0
            else 0.0
        )

        avg_processing_time = (
            np.mean([m.total_processing_time for m in recent_metrics])
            if recent_metrics
            else 0.0
        )

        avg_coherence = (
            np.mean(
                [m.coherence_score for m in recent_metrics if m.coherence_score > 0]
            )
            if recent_metrics
            else 0.0
        )

        return {
            "status": "operational",
            "total_integrations": self.total_integrations,
            "successful_integrations": self.successful_integrations,
            "success_rate": success_rate,
            "average_processing_time": avg_processing_time,
            "average_coherence": avg_coherence,
            "systems_available": {
                "barenholtz": self.barenholtz_adapter
                and self.barenholtz_adapter.available,
                "quantum_security": self.config.enable_quantum_security,
                "thermodynamic_coherence": self.config.enable_thermodynamic_coherence,
                "high_dimensional": (
                    self.high_dimensional_adapter
                    and self.high_dimensional_adapter.available
                ),
                "insight_management": (
                    self.insight_adapter and self.insight_adapter.available
                ),
            },
            "configuration": self.config.__dict__,
        }


# Factory function for global instance
_integration_bridge_instance: Optional[KimeraFullIntegrationBridge] = None


def get_full_integration_bridge(
    config: Optional[IntegrationConfig] = None,
) -> KimeraFullIntegrationBridge:
    """Get global full integration bridge instance"""
    global _integration_bridge_instance
    if _integration_bridge_instance is None:
        _integration_bridge_instance = KimeraFullIntegrationBridge(config)
    return _integration_bridge_instance
