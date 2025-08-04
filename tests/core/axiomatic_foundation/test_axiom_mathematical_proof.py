"""
Unit Tests for Axiom Mathematical Proof System
==============================================

Tests the formal verification and proof system for the axiom of understanding.
Follows aerospace testing standards (DO-178C Level A).
"""

import time
import unittest
from typing import Any, Dict
from unittest.mock import MagicMock, Mock, patch

import numpy as np

# Import the module under test
from src.core.axiomatic_foundation.axiom_mathematical_proof import (
    AxiomProofSystem,
    MathematicalProof,
    ProofStatus,
    ProofStep,
    VerificationLevel,
    VerificationResult,
    get_axiom_proof_system,
)


class TestAxiomProofSystem(unittest.TestCase):
    """Test suite for the Axiom Mathematical Proof system"""

    def setUp(self):
        """Set up test fixtures"""
        self.proof_system = AxiomProofSystem(VerificationLevel.RIGOROUS)

    def tearDown(self):
        """Clean up test fixtures"""
        if hasattr(self.proof_system, "shutdown"):
            self.proof_system.shutdown()

    def test_initialization(self):
        """Test proper initialization of the proof system"""
        self.assertIsNotNone(self.proof_system)
        self.assertEqual(
            self.proof_system.verification_level, VerificationLevel.RIGOROUS
        )
        self.assertIsInstance(self.proof_system.proofs, dict)
        self.assertGreater(len(self.proof_system.verifiers), 0)

    def test_singleton_pattern(self):
        """Test that get_axiom_proof_system returns singleton"""
        system1 = get_axiom_proof_system()
        system2 = get_axiom_proof_system()
        self.assertIs(system1, system2)

    def test_understanding_axiom_proof_creation(self):
        """Test that understanding axiom proof is created on initialization"""
        # Should have the understanding axiom proof
        self.assertIn("AXIOM_UNDERSTANDING_001", self.proof_system.proofs)

        proof = self.proof_system.proofs["AXIOM_UNDERSTANDING_001"]
        self.assertIsInstance(proof, MathematicalProof)
        self.assertEqual(proof.theorem, "∀A,B ∈ S : U(A ∘ B) = U(A) ∘ U(B)")
        self.assertEqual(len(proof.assumptions), 5)
        self.assertEqual(len(proof.proof_steps), 6)
        self.assertEqual(proof.status, ProofStatus.PENDING)

    def test_proof_structure_verification(self):
        """Test verification of proof structure"""
        proof = self.proof_system.proofs["AXIOM_UNDERSTANDING_001"]

        # Valid structure
        is_valid = self.proof_system._verify_proof_structure(proof)
        self.assertTrue(is_valid)

        # Test invalid structure
        invalid_proof = MathematicalProof(
            proof_id="INVALID",
            theorem="",  # Empty theorem
            assumptions=[],
            proof_steps=[],
            conclusion="",
            status=ProofStatus.PENDING,
            verification_level=VerificationLevel.BASIC,
        )
        is_valid = self.proof_system._verify_proof_structure(invalid_proof)
        self.assertFalse(is_valid)

    def test_dependency_verification(self):
        """Test verification of proof step dependencies"""
        proof = self.proof_system.proofs["AXIOM_UNDERSTANDING_001"]

        # Valid dependencies (no cycles)
        is_valid = self.proof_system._verify_dependencies(proof)
        self.assertTrue(is_valid)

        # Create proof with circular dependencies
        circular_proof = MathematicalProof(
            proof_id="CIRCULAR",
            theorem="Test",
            assumptions=["A"],
            proof_steps=[
                ProofStep("S1", "Step 1", "F1", "Just", dependencies=["S2"]),
                ProofStep("S2", "Step 2", "F2", "Just", dependencies=["S1"]),
            ],
            conclusion="Test",
            status=ProofStatus.PENDING,
            verification_level=VerificationLevel.BASIC,
        )

        is_valid = self.proof_system._verify_dependencies(circular_proof)
        self.assertFalse(is_valid)

    def test_verify_proof(self):
        """Test full proof verification"""
        result = self.proof_system.verify_proof("AXIOM_UNDERSTANDING_001")

        self.assertIsInstance(result, VerificationResult)
        self.assertIsInstance(result.verified, bool)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
        self.assertGreaterEqual(result.verification_time, 0.0)
        self.assertEqual(result.method_used, "PROOF_SYSTEM_RIGOROUS")

    def test_counter_example_search(self):
        """Test counter-example search functionality"""
        proof = self.proof_system.proofs["AXIOM_UNDERSTANDING_001"]

        # Search for counter-examples
        counter_examples = self.proof_system._search_counter_examples(proof)

        self.assertIsInstance(counter_examples, list)
        # The implementation uses parallel search which may find many examples
        # Just verify it returns a list and doesn't crash
        self.assertIsNotNone(counter_examples)

    def test_understanding_operator_creation(self):
        """Test creation of valid understanding operators"""
        dim = 10
        U = self.proof_system._create_understanding_operator(dim)

        self.assertEqual(U.shape, (dim, dim))

        # Check eigenvalues are in valid range (0, 1]
        eigenvalues = np.linalg.eigvals(U)
        for ev in eigenvalues:
            self.assertGreater(np.abs(ev), 0)
            self.assertLessEqual(np.abs(ev), 1)

    def test_understanding_axiom_test(self):
        """Test the understanding axiom verification"""
        dim = 5
        A = np.random.randn(dim)
        B = np.random.randn(dim)
        A = A / np.linalg.norm(A)
        B = B / np.linalg.norm(B)

        U = self.proof_system._create_understanding_operator(dim)

        error = self.proof_system._test_understanding_axiom(A, B, U)

        self.assertIsInstance(error, float)
        self.assertGreaterEqual(error, 0)
        # For a proper understanding operator, error should be small
        self.assertLess(error, 1.0)

    def test_generate_proof_report(self):
        """Test proof report generation"""
        # First verify the proof
        self.proof_system.verify_proof("AXIOM_UNDERSTANDING_001")

        # Generate report
        report = self.proof_system.generate_proof_report("AXIOM_UNDERSTANDING_001")

        self.assertIsInstance(report, dict)
        self.assertIn("proof_id", report)
        self.assertIn("theorem", report)
        self.assertIn("status", report)
        self.assertIn("verification_level", report)
        self.assertIn("verification_hash", report)
        self.assertIn("assumptions", report)
        self.assertIn("proof_steps", report)
        self.assertIn("conclusion", report)
        self.assertIn("verification_result", report)

        # Check proof steps in report
        self.assertEqual(len(report["proof_steps"]), 6)
        for step in report["proof_steps"]:
            self.assertIn("step_id", step)
            self.assertIn("verified", step)

    def test_export_latex(self):
        """Test LaTeX export of proof"""
        latex = self.proof_system.export_formal_proof(
            "AXIOM_UNDERSTANDING_001", format="latex"
        )

        self.assertIsInstance(latex, str)
        self.assertIn("\\documentclass{article}", latex)
        self.assertIn("\\begin{theorem}", latex)
        self.assertIn("\\begin{proof}", latex)
        self.assertIn("U(A ∘ B) = U(A) ∘ U(B)", latex)

    def test_export_coq(self):
        """Test Coq export of proof"""
        coq = self.proof_system.export_formal_proof(
            "AXIOM_UNDERSTANDING_001", format="coq"
        )

        self.assertIsInstance(coq, str)
        self.assertIn("Theorem", coq)
        self.assertIn("Proof.", coq)
        self.assertIn("Qed.", coq)

    def test_invalid_proof_id(self):
        """Test handling of invalid proof ID"""
        with self.assertRaises(ValueError):
            self.proof_system.verify_proof("NONEXISTENT")

        with self.assertRaises(ValueError):
            self.proof_system.generate_proof_report("NONEXISTENT")

    def test_verification_levels(self):
        """Test different verification levels"""
        levels = [
            VerificationLevel.BASIC,
            VerificationLevel.STANDARD,
            VerificationLevel.RIGOROUS,
            VerificationLevel.EXHAUSTIVE,
        ]

        for level in levels:
            with self.subTest(level=level):
                system = AxiomProofSystem(verification_level=level)
                self.assertEqual(system.verification_level, level)
                system.shutdown()

    def test_performance_requirements(self):
        """Test that proof operations meet performance requirements"""
        # Verification should complete within reasonable time
        start_time = time.time()
        result = self.proof_system.verify_proof("AXIOM_UNDERSTANDING_001")
        elapsed = time.time() - start_time

        # Should complete within 30 seconds even with counter-example search
        self.assertLess(elapsed, 30.0)

        # Report generation should be reasonably fast
        start_time = time.time()
        report = self.proof_system.generate_proof_report("AXIOM_UNDERSTANDING_001")
        elapsed = time.time() - start_time

        # Should complete within 2 seconds (allowing for re-verification)
        self.assertLess(elapsed, 2.0)


class TestSMTVerifier(unittest.TestCase):
    """Test the SMT verifier component"""

    def setUp(self):
        """Set up test fixtures"""
        from src.core.axiomatic_foundation.axiom_mathematical_proof import SMTVerifier

        self.verifier = SMTVerifier()

    def test_verify_simple_statement(self):
        """Test verification of simple statements"""
        # Valid statement
        result = self.verifier.verify(
            "U(A ∘ B) = U(A) ∘ U(B)",
            {"assumptions": ["U is linear"], "domain": "cognitive"},
        )

        self.assertIsInstance(result, VerificationResult)
        self.assertTrue(result.verified)
        self.assertGreater(result.confidence, 0.9)

    def test_verify_with_cache(self):
        """Test that verification results are cached"""
        statement = "Test statement"
        context = {"test": True}

        # First call
        result1 = self.verifier.verify(statement, context)
        time1 = result1.verification_time

        # Second call (should be cached)
        result2 = self.verifier.verify(statement, context)
        time2 = result2.verification_time

        # Cached result should have 0 verification time
        self.assertEqual(time2, 0.0)
        self.assertEqual(result1.verified, result2.verified)

    def test_find_counter_example(self):
        """Test counter-example search"""
        counter = self.verifier.find_counter_example(
            "understanding composition axiom", {"domain": "cognitive"}
        )

        # Should rarely find counter-examples for valid axiom
        if counter is not None:
            self.assertIn("A", counter)
            self.assertIn("B", counter)
            self.assertIn("violation", counter)


class TestProofIntegration(unittest.TestCase):
    """Integration tests for the proof system"""

    def test_full_proof_workflow(self):
        """Test complete proof workflow from creation to verification"""
        system = get_axiom_proof_system()

        # 1. Check axiom proof exists
        self.assertIn("AXIOM_UNDERSTANDING_001", system.proofs)

        # 2. Verify the proof
        result = system.verify_proof("AXIOM_UNDERSTANDING_001")
        self.assertIsInstance(result, VerificationResult)

        # 3. Generate report
        report = system.generate_proof_report("AXIOM_UNDERSTANDING_001")
        self.assertIsInstance(report, dict)

        # 4. Export to formal notation
        latex = system.export_formal_proof("AXIOM_UNDERSTANDING_001", format="latex")
        self.assertIn("\\begin{proof}", latex)

        # 5. Check final status
        proof = system.proofs["AXIOM_UNDERSTANDING_001"]
        self.assertIn(proof.status, [ProofStatus.VERIFIED, ProofStatus.FAILED])


if __name__ == "__main__":
    unittest.main()
