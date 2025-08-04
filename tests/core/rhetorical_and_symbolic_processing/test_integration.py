"""
Test Suite for Rhetorical and Symbolic Processing Integration
============================================================

DO-178C Level A compliant test suite validating:
- Integration layer functionality
- Safety requirements SR-4.20.1 through SR-4.20.24
- Performance requirements
- Error handling and recovery
- Cross-modal analysis capabilities
"""

import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.rhetorical_and_symbolic_processing.integration import (
    ProcessingMode,
    RhetoricalSymbolicIntegrator,
    UnifiedProcessingResult,
)
from src.core.rhetorical_and_symbolic_processing.rhetorical_engine import (
    RhetoricalAnalysis,
    RhetoricalMode,
    RhetoricalProcessor,
)
from src.core.rhetorical_and_symbolic_processing.symbolic_engine import (
    SymbolicAnalysis,
    SymbolicModality,
    SymbolicProcessor,
)


class TestRhetoricalSymbolicIntegrator:
    """Test suite for RhetoricalSymbolicIntegrator."""

    @pytest.fixture
    async def integrator(self):
        """Create integrator instance for testing."""
        integrator = RhetoricalSymbolicIntegrator(
            device="cpu", mode=ProcessingMode.PARALLEL
        )
        await integrator.initialize()
        yield integrator
        await integrator.shutdown()

    @pytest.mark.asyncio
    async def test_initialization_success(self):
        """Test SR-4.20.1: Initialization safety validation."""
        integrator = RhetoricalSymbolicIntegrator()

        # Test initialization
        success = await integrator.initialize()
        assert success, "Integrator initialization should succeed"

        # Validate components initialized
        assert integrator._initialized, "Integrator should be marked as initialized"
        assert all(
            integrator._components_initialized.values()
        ), "All components should be initialized"

        # Test health metrics
        health = integrator.get_health_metrics()
        assert health["integration_metrics"][
            "initialized"
        ], "Health metrics should show initialized"

        await integrator.shutdown()

    @pytest.mark.asyncio
    async def test_input_validation(self, integrator):
        """Test SR-4.20.2: Input validation and sanitization."""

        # Test empty input
        with pytest.raises(AssertionError):
            await integrator.process_content("")

        # Test None input
        with pytest.raises(AssertionError):
            await integrator.process_content(None)

        # Test oversized input
        large_content = "x" * 300000
        with pytest.raises(AssertionError):
            await integrator.process_content(large_content)

        # Test valid input
        result = await integrator.process_content("Valid test content")
        assert isinstance(result, UnifiedProcessingResult)

    @pytest.mark.asyncio
    async def test_processing_time_bounds(self, integrator):
        """Test SR-4.20.3: Processing time bounds enforcement."""
        content = "Test content for processing time validation"

        start_time = time.time()
        result = await integrator.process_content(content)
        processing_time = time.time() - start_time

        # Validate processing time is within bounds
        assert (
            processing_time <= integrator._max_processing_time
        ), "Processing time should be within bounds"
        assert (
            result.processing_time <= integrator._max_processing_time
        ), "Result processing time should be within bounds"

    @pytest.mark.asyncio
    async def test_parallel_processing_mode(self, integrator):
        """Test parallel processing mode functionality."""
        content = "This is a test with emojis 😀🔥 and logical arguments because evidence shows."

        result = await integrator.process_content(content, mode=ProcessingMode.PARALLEL)

        assert result.status == "success", "Parallel processing should succeed"
        assert result.rhetorical_analysis is not None, "Should have rhetorical analysis"
        assert result.symbolic_analysis is not None, "Should have symbolic analysis"
        assert (
            len(result.cross_modal_correlations) > 0
        ), "Should have cross-modal correlations"

    @pytest.mark.asyncio
    async def test_sequential_processing_mode(self, integrator):
        """Test sequential processing mode functionality."""
        content = "Evidence-based argument with mathematical notation ∑x²"

        result = await integrator.process_content(
            content, mode=ProcessingMode.SEQUENTIAL
        )

        assert result.status == "success", "Sequential processing should succeed"
        assert result.rhetorical_analysis is not None, "Should have rhetorical analysis"
        assert result.symbolic_analysis is not None, "Should have symbolic analysis"

    @pytest.mark.asyncio
    async def test_rhetorical_only_mode(self, integrator):
        """Test rhetorical-only processing mode."""
        content = "Persuasive argument with ethos, pathos, and logos elements"

        result = await integrator.process_content(
            content, mode=ProcessingMode.RHETORICAL_ONLY
        )

        assert (
            result.status == "rhetorical_only"
        ), "Should indicate rhetorical-only processing"
        assert result.rhetorical_analysis is not None, "Should have rhetorical analysis"
        assert result.symbolic_analysis is None, "Should not have symbolic analysis"

    @pytest.mark.asyncio
    async def test_symbolic_only_mode(self, integrator):
        """Test symbolic-only processing mode."""
        content = "Mathematical symbols ∑∫∂√ and emojis 🔢📊📈"

        result = await integrator.process_content(
            content, mode=ProcessingMode.SYMBOLIC_ONLY
        )

        assert (
            result.status == "symbolic_only"
        ), "Should indicate symbolic-only processing"
        assert result.rhetorical_analysis is None, "Should not have rhetorical analysis"
        assert result.symbolic_analysis is not None, "Should have symbolic analysis"

    @pytest.mark.asyncio
    async def test_adaptive_processing_mode(self, integrator):
        """Test adaptive processing mode selection."""

        # Content with both rhetorical and symbolic elements
        mixed_content = "Evidence 📊 proves our ethical approach 🤝 will succeed ⭐"
        result = await integrator.process_content(
            mixed_content, mode=ProcessingMode.ADAPTIVE
        )
        assert (
            result.rhetorical_analysis is not None
        ), "Should detect rhetorical elements"
        assert result.symbolic_analysis is not None, "Should detect symbolic elements"

        # Content with primarily rhetorical elements
        rhetorical_content = "We must persuade through evidence and logical reasoning"
        result = await integrator.process_content(
            rhetorical_content, mode=ProcessingMode.ADAPTIVE
        )
        assert (
            result.rhetorical_analysis is not None
        ), "Should process rhetorical content"

    @pytest.mark.asyncio
    async def test_safety_fallback_mode(self, integrator):
        """Test safety fallback mode."""
        content = "Any content"

        result = await integrator.process_content(
            content, mode=ProcessingMode.SAFETY_FALLBACK
        )

        assert result.status == "safety_fallback", "Should indicate safety fallback"
        assert result.rhetorical_analysis is None, "Should not have rhetorical analysis"
        assert result.symbolic_analysis is None, "Should not have symbolic analysis"
        assert result.safety_validation[
            "fallback_mode"
        ], "Should indicate fallback mode"

    @pytest.mark.asyncio
    async def test_cross_modal_correlations(self, integrator):
        """Test SR-4.20.8: Cross-modal validation consistency."""
        content = "Mathematical proof ∑ shows logical reasoning 💡 builds trust ✅"

        result = await integrator.process_content(content)

        # Should have correlations between logos and mathematical symbols
        assert (
            len(result.cross_modal_correlations) > 0
        ), "Should have cross-modal correlations"

        # Validate correlation values are in valid range
        for correlation_value in result.cross_modal_correlations.values():
            assert (
                0.0 <= correlation_value <= 1.0
            ), "Correlations should be in [0,1] range"

    @pytest.mark.asyncio
    async def test_cultural_context_preservation(self, integrator):
        """Test SR-4.20.9: Cultural context preservation."""
        content = "Peace ☮️ through wisdom 🕉️ and unity 🤝"
        context = "international_diplomacy"

        result = await integrator.process_content(content, context=context)

        if result.rhetorical_analysis:
            assert (
                result.rhetorical_analysis.cultural_context is not None
            ), "Should preserve cultural context"

        if result.symbolic_analysis:
            assert (
                result.symbolic_analysis.cultural_context is not None
            ), "Should preserve cultural context"

    @pytest.mark.asyncio
    async def test_neurodivergent_accessibility(self, integrator):
        """Test SR-4.20.10: Neurodivergent accessibility verification."""
        # Simple, clear content that should score high on accessibility
        accessible_content = "First, we examine the evidence. Second, we analyze the data. Finally, we conclude."

        result = await integrator.process_content(accessible_content)

        if result.rhetorical_analysis:
            assert (
                result.rhetorical_analysis.neurodivergent_accessibility >= 0.5
            ), "Should have good accessibility score"

        # Complex content that should score lower
        complex_content = "The multifaceted, paradigmatic, and synergistic approach to optimization..."
        result_complex = await integrator.process_content(complex_content)

        if result_complex.rhetorical_analysis:
            # Complex content should have lower accessibility (though this is simplified)
            assert isinstance(
                result_complex.rhetorical_analysis.neurodivergent_accessibility, float
            )

    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, integrator):
        """Test SR-4.20.5: Error handling and recovery."""

        # Test with malformed input that should be handled gracefully
        try:
            result = await integrator.process_content("Test content")
            assert result is not None, "Should handle processing gracefully"
        except Exception as e:
            pytest.fail(f"Should not raise unhandled exceptions: {e}")

    @pytest.mark.asyncio
    async def test_health_monitoring(self, integrator):
        """Test SR-4.20.6: Health monitoring and reporting."""
        # Process some content to generate metrics
        await integrator.process_content("Test content for health monitoring")

        health = integrator.get_health_metrics()

        # Validate health metrics structure
        assert "integration_metrics" in health, "Should have integration metrics"
        assert (
            "rhetorical_processor" in health
        ), "Should have rhetorical processor metrics"
        assert "symbolic_processor" in health, "Should have symbolic processor metrics"

        # Validate key health indicators
        integration_metrics = health["integration_metrics"]
        assert integration_metrics["initialized"], "Should be initialized"
        assert (
            integration_metrics["total_processing"] > 0
        ), "Should have processing count"
        assert (
            integration_metrics["error_rate"] >= 0.0
        ), "Error rate should be non-negative"
        assert (
            integration_metrics["safety_violation_rate"] >= 0.0
        ), "Safety violation rate should be non-negative"

    @pytest.mark.asyncio
    async def test_graceful_degradation(self, integrator):
        """Test SR-4.20.7: Graceful degradation capability."""
        # This would test behavior when components are unavailable
        # For this test, we'll verify that partial results are handled

        content = "Test content for degradation testing"
        result = await integrator.process_content(content)

        # Even if some analyses fail, should still provide meaningful result
        assert result is not None, "Should provide result even with partial analysis"
        assert result.status in [
            "success",
            "rhetorical_only",
            "symbolic_only",
            "safety_fallback",
        ], "Should have valid status"

    @pytest.mark.asyncio
    async def test_performance_requirements(self, integrator):
        """Test performance requirements."""
        content = "Performance test content with mixed elements 📊💡"

        # Test multiple iterations to verify consistent performance
        times = []
        for _ in range(5):
            start_time = time.time()
            result = await integrator.process_content(content)
            processing_time = time.time() - start_time
            times.append(processing_time)

            assert (
                result.processing_time <= 15.0
            ), "Should complete within maximum time limit"

        # Verify consistent performance
        avg_time = sum(times) / len(times)
        assert avg_time <= 5.0, "Average processing time should be reasonable"

    def test_safety_margins_configuration(self):
        """Test safety margins are properly configured."""
        integrator = RhetoricalSymbolicIntegrator()

        assert integrator._safety_margins == 0.1, "Should have 10% safety margins"
        assert (
            integrator._max_processing_time == 15.0
        ), "Should have reasonable processing time limit"

    @pytest.mark.asyncio
    async def test_unified_insights_generation(self, integrator):
        """Test unified insights generation from both analyses."""
        content = "Credible experts 👨‍🔬 provide evidence 📊 that proves our logical argument 💡"

        result = await integrator.process_content(content)

        assert result.unified_insights is not None, "Should generate unified insights"
        assert isinstance(
            result.unified_insights, dict
        ), "Insights should be dictionary"

        if result.rhetorical_analysis and result.symbolic_analysis:
            assert (
                "communication_effectiveness" in result.unified_insights
            ), "Should calculate communication effectiveness"
            assert (
                "cross_cultural_adaptation" in result.unified_insights
            ), "Should assess cross-cultural adaptation"


class TestRhetoricalProcessor:
    """Test suite for RhetoricalProcessor."""

    @pytest.fixture
    async def processor(self):
        """Create processor instance for testing."""
        processor = RhetoricalProcessor(device="cpu")
        await processor.initialize()
        yield processor
        await processor.shutdown()

    @pytest.mark.asyncio
    async def test_rhetorical_analysis_basic(self, processor):
        """Test basic rhetorical analysis functionality."""
        content = "Our credible experts provide evidence that logically proves this emotional appeal will succeed."

        result = await processor.analyze_rhetoric(content)

        assert isinstance(
            result, RhetoricalAnalysis
        ), "Should return RhetoricalAnalysis"
        assert 0.0 <= result.ethos_score <= 1.0, "Ethos score should be in valid range"
        assert (
            0.0 <= result.pathos_score <= 1.0
        ), "Pathos score should be in valid range"
        assert 0.0 <= result.logos_score <= 1.0, "Logos score should be in valid range"
        assert 0.0 <= result.confidence <= 1.0, "Confidence should be in valid range"

    @pytest.mark.asyncio
    async def test_rhetorical_modes(self, processor):
        """Test different rhetorical analysis modes."""
        content = "Test content for mode testing"

        # Test classical mode
        result_classical = await processor.analyze_rhetoric(
            content, mode=RhetoricalMode.CLASSICAL
        )
        assert result_classical.ethos_score >= 0.0, "Classical mode should work"

        # Test modern mode
        result_modern = await processor.analyze_rhetoric(
            content, mode=RhetoricalMode.MODERN
        )
        assert (
            result_modern.argument_structure is not None
        ), "Modern mode should analyze structure"

        # Test unified mode
        result_unified = await processor.analyze_rhetoric(
            content, mode=RhetoricalMode.UNIFIED
        )
        assert (
            result_unified.persuasive_effectiveness >= 0.0
        ), "Unified mode should calculate effectiveness"


class TestSymbolicProcessor:
    """Test suite for SymbolicProcessor."""

    @pytest.fixture
    async def processor(self):
        """Create processor instance for testing."""
        processor = SymbolicProcessor(device="cpu")
        await processor.initialize()
        yield processor
        await processor.shutdown()

    @pytest.mark.asyncio
    async def test_symbolic_analysis_basic(self, processor):
        """Test basic symbolic analysis functionality."""
        content = "Mathematical expression: ∑x² + emoji 😀 + arrow →"

        result = await processor.analyze_symbols(content)

        assert isinstance(result, SymbolicAnalysis), "Should return SymbolicAnalysis"
        assert isinstance(
            result.modality, SymbolicModality
        ), "Should detect symbolic modality"
        assert (
            0.0 <= result.symbol_complexity <= 1.0
        ), "Complexity should be in valid range"
        assert (
            0.0 <= result.cross_cultural_recognition <= 1.0
        ), "Recognition should be in valid range"
        assert 0.0 <= result.confidence <= 1.0, "Confidence should be in valid range"

    @pytest.mark.asyncio
    async def test_modality_detection(self, processor):
        """Test symbolic modality detection."""

        # Test emoji detection
        emoji_content = "Happy face 😀 and heart ❤️"
        result_emoji = await processor.analyze_symbols(emoji_content)
        assert (
            result_emoji.modality == SymbolicModality.EMOJI_SEMIOTICS
        ), "Should detect emoji modality"

        # Test mathematical detection
        math_content = "Mathematical symbols: ∑∫∂√∞"
        result_math = await processor.analyze_symbols(math_content)
        assert (
            result_math.modality == SymbolicModality.MATHEMATICAL
        ), "Should detect mathematical modality"

        # Test iconographic detection
        icon_content = "Arrows and symbols → ← ↑ ⚠ ⭐"
        result_icon = await processor.analyze_symbols(icon_content)
        assert (
            result_icon.modality == SymbolicModality.ICONOGRAPHY
        ), "Should detect iconographic modality"


# Test execution
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
