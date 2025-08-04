"""
Unit Tests for Axiom of Understanding
=====================================

Tests the implementation of the fundamental axiom:
"Understanding reduces semantic entropy while preserving information"
"""

import asyncio
import unittest
from unittest.mock import Mock, patch

import numpy as np

from src.core.axiomatic_foundation.axiom_of_understanding import (
    AxiomOfUnderstanding,
    SemanticState,
    UnderstandingMode,
    UnderstandingResult,
)


class TestSemanticState(unittest.TestCase):
    """Test the SemanticState data structure"""

    def test_semantic_state_creation(self):
        """Test creating semantic states"""
        vector = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        state = SemanticState(
            vector=vector, entropy=2.0, information=1.5, meaning_label="test_concept"
        )

        self.assertIsNotNone(state)
        np.testing.assert_array_equal(state.vector, vector)
        self.assertEqual(state.entropy, 2.0)
        self.assertEqual(state.information, 1.5)
        self.assertEqual(state.meaning_label, "test_concept")

    def test_semantic_state_validation(self):
        """Test semantic state validation"""
        # Test negative entropy (invalid)
        with self.assertRaises(ValueError):
            SemanticState(vector=np.array([1.0, 0.0]), entropy=-1.0, information=1.0)

        # Test negative information (invalid)
        with self.assertRaises(ValueError):
            SemanticState(vector=np.array([1.0, 0.0]), entropy=1.0, information=-1.0)

    def test_semantic_state_composition(self):
        """Test composition of semantic states"""
        state1 = SemanticState(
            vector=np.array([1.0, 0.0, 0.0]), entropy=1.0, information=0.8
        )

        state2 = SemanticState(
            vector=np.array([0.0, 1.0, 0.0]), entropy=1.0, information=0.8
        )

        composed = state1.compose(state2)

        self.assertIsInstance(composed, SemanticState)
        self.assertEqual(len(composed.vector), len(state1.vector))
        # Entropy should not decrease in composition
        self.assertGreaterEqual(composed.entropy, max(state1.entropy, state2.entropy))


class TestAxiomOfUnderstanding(unittest.TestCase):
    """Test the Axiom of Understanding implementation"""

    def setUp(self):
        """Set up test fixtures"""
        self.axiom = AxiomOfUnderstanding()

        # Create test states
        self.test_state = SemanticState(
            vector=np.array([1.0, 0.5, 0.0, 0.0, 0.0]),
            entropy=2.0,
            information=1.5,
            meaning_label="test",
        )

    def test_initialization(self):
        """Test proper initialization"""
        self.assertIsNotNone(self.axiom)
        self.assertTrue(hasattr(self.axiom, "apply_understanding"))
        self.assertTrue(hasattr(self.axiom, "verify_axiom"))

    def test_apply_understanding_reduces_entropy(self):
        """Test that understanding reduces entropy"""
        result = self.axiom.apply_understanding(self.test_state)

        self.assertIsInstance(result, UnderstandingResult)
        self.assertLess(result.output_state.entropy, self.test_state.entropy)
        self.assertAlmostEqual(
            result.output_state.information,
            self.test_state.information,
            delta=0.1,  # Small tolerance for information preservation
        )

    def test_apply_understanding_preserves_information(self):
        """Test that understanding preserves information"""
        result = self.axiom.apply_understanding(self.test_state)

        # Information should be preserved within tolerance
        info_ratio = result.output_state.information / self.test_state.information
        self.assertGreater(info_ratio, 0.95)  # At least 95% preserved
        self.assertLess(info_ratio, 1.05)  # At most 105% (small increase ok)

    def test_understanding_modes(self):
        """Test different understanding modes"""
        modes = [
            UnderstandingMode.ANALYTICAL,
            UnderstandingMode.INTUITIVE,
            UnderstandingMode.HOLISTIC,
        ]

        for mode in modes:
            with self.subTest(mode=mode):
                result = self.axiom.apply_understanding(self.test_state, mode=mode)

                self.assertIsInstance(result, UnderstandingResult)
                self.assertEqual(result.mode, mode)
                self.assertLess(result.output_state.entropy, self.test_state.entropy)

    def test_verify_axiom_homomorphism(self):
        """Test axiom verification for homomorphism property"""
        state_a = SemanticState(
            vector=np.array([1.0, 0.0, 0.0, 0.0, 0.0]), entropy=1.5, information=1.0
        )

        state_b = SemanticState(
            vector=np.array([0.0, 1.0, 0.0, 0.0, 0.0]), entropy=1.5, information=1.0
        )

        verification = self.axiom.verify_axiom(state_a, state_b)

        self.assertIn("is_valid", verification)
        self.assertIn("error", verification)
        self.assertIn("left_side", verification)
        self.assertIn("right_side", verification)

        # Error should be small for valid axiom
        self.assertLess(verification["error"], 0.1)

    def test_riemannian_geometry(self):
        """Test Riemannian geometry calculations"""
        # Test geodesic distance
        state1 = SemanticState(
            vector=np.array([1.0, 0.0, 0.0, 0.0, 0.0]), entropy=1.0, information=0.8
        )

        state2 = SemanticState(
            vector=np.array([0.0, 1.0, 0.0, 0.0, 0.0]), entropy=1.0, information=0.8
        )

        distance = self.axiom.geodesic_distance(state1, state2)

        self.assertGreater(distance, 0)
        self.assertIsInstance(distance, float)

        # Distance to self should be zero
        self_distance = self.axiom.geodesic_distance(state1, state1)
        self.assertAlmostEqual(self_distance, 0.0, places=6)

    def test_parallel_transport(self):
        """Test parallel transport on the understanding manifold"""
        # Create a tangent vector
        tangent = np.array([0.1, 0.2, 0.0, 0.0, 0.0])

        # Transport along a path
        transported = self.axiom.parallel_transport(self.test_state, tangent, steps=10)

        self.assertEqual(len(transported), len(tangent))
        # Magnitude should be preserved in parallel transport
        self.assertAlmostEqual(
            np.linalg.norm(transported), np.linalg.norm(tangent), places=5
        )

    def test_curvature_tensor(self):
        """Test Riemann curvature tensor computation"""
        curvature = self.axiom.compute_curvature_at(self.test_state)

        self.assertIsInstance(curvature, np.ndarray)
        # Curvature tensor should have correct shape
        dim = len(self.test_state.vector)
        self.assertEqual(curvature.shape, (dim, dim, dim, dim))

        # Test symmetries of Riemann tensor
        # R_ijkl = -R_jikl
        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    for l in range(dim):
                        self.assertAlmostEqual(
                            curvature[i, j, k, l], -curvature[j, i, k, l], places=10
                        )

    def test_information_metric(self):
        """Test Fisher information metric"""
        metric = self.axiom.compute_information_metric(self.test_state)

        self.assertIsInstance(metric, np.ndarray)
        dim = len(self.test_state.vector)
        self.assertEqual(metric.shape, (dim, dim))

        # Metric should be symmetric
        np.testing.assert_array_almost_equal(metric, metric.T)

        # Metric should be positive definite
        eigenvalues = np.linalg.eigvals(metric)
        self.assertTrue(np.all(eigenvalues > 0))

    def test_entropy_gradient_flow(self):
        """Test entropy reduction via gradient flow"""
        # Apply multiple understanding steps
        current_state = self.test_state
        entropies = [current_state.entropy]

        for _ in range(5):
            result = self.axiom.apply_understanding(current_state)
            current_state = result.output_state
            entropies.append(current_state.entropy)

        # Entropy should decrease monotonically
        for i in range(1, len(entropies)):
            self.assertLessEqual(entropies[i], entropies[i - 1])

    def test_async_understanding(self):
        """Test asynchronous understanding application"""

        async def run_async_test():
            result = await self.axiom.apply_understanding_async(self.test_state)
            self.assertIsInstance(result, UnderstandingResult)
            return result

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(run_async_test())
            self.assertLess(result.output_state.entropy, self.test_state.entropy)
        finally:
            loop.close()

    def test_batch_understanding(self):
        """Test batch processing of multiple states"""
        states = [
            SemanticState(vector=np.random.randn(5), entropy=2.0, information=1.5)
            for _ in range(10)
        ]

        results = self.axiom.apply_understanding_batch(states)

        self.assertEqual(len(results), len(states))

        # All should have reduced entropy
        for i, result in enumerate(results):
            self.assertLess(result.output_state.entropy, states[i].entropy)

    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        # Zero entropy state
        zero_entropy_state = SemanticState(
            vector=np.array([1.0, 0.0, 0.0, 0.0, 0.0]), entropy=0.0, information=1.0
        )

        result = self.axiom.apply_understanding(zero_entropy_state)
        # Can't reduce entropy below zero
        self.assertEqual(result.output_state.entropy, 0.0)

        # Very high entropy state
        high_entropy_state = SemanticState(
            vector=np.ones(5) / np.sqrt(5), entropy=100.0, information=50.0
        )

        result = self.axiom.apply_understanding(high_entropy_state)
        # Should still reduce entropy
        self.assertLess(result.output_state.entropy, high_entropy_state.entropy)

    def test_axiom_consistency(self):
        """Test consistency of the axiom across multiple applications"""
        # Apply understanding multiple times
        state = self.test_state

        # First application
        result1 = self.axiom.apply_understanding(state)

        # Second application on same input
        result2 = self.axiom.apply_understanding(state)

        # Results should be consistent (deterministic)
        np.testing.assert_array_almost_equal(
            result1.output_state.vector, result2.output_state.vector
        )
        self.assertAlmostEqual(
            result1.output_state.entropy, result2.output_state.entropy
        )


class TestAxiomIntegration(unittest.TestCase):
    """Integration tests for the Axiom of Understanding"""

    def test_complete_understanding_pipeline(self):
        """Test complete understanding pipeline"""
        axiom = AxiomOfUnderstanding()

        # 1. Create initial high-entropy state
        initial_state = SemanticState(
            vector=np.random.randn(10),
            entropy=5.0,
            information=3.0,
            meaning_label="complex_concept",
        )

        # 2. Apply understanding
        result = axiom.apply_understanding(
            initial_state, mode=UnderstandingMode.HOLISTIC
        )

        # 3. Verify axiom properties
        self.assertLess(result.output_state.entropy, initial_state.entropy)
        self.assertAlmostEqual(
            result.output_state.information, initial_state.information, delta=0.2
        )

        # 4. Test composition
        another_state = SemanticState(
            vector=np.random.randn(10),
            entropy=4.0,
            information=2.5,
            meaning_label="related_concept",
        )

        verification = axiom.verify_axiom(initial_state, another_state)
        self.assertTrue(verification["is_valid"])

        # 5. Compute geometric properties
        distance = axiom.geodesic_distance(result.output_state, another_state)
        self.assertGreater(distance, 0)

        # 6. Generate understanding report
        report = axiom.generate_understanding_report(result)
        self.assertIn("entropy_reduction", report)
        self.assertIn("information_preservation", report)
        self.assertIn("geometric_properties", report)


if __name__ == "__main__":
    unittest.main()
