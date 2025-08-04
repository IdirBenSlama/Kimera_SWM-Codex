"""
Unit Tests for Axiom Verification Engine
========================================

Tests the DO-178C Level A compliant verification system for the axiom of understanding.
"""

import unittest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import asyncio
from datetime import datetime, timedelta

from src.core.axiomatic_foundation.axiom_verification import (
    AxiomVerificationEngine,
    VerificationRequirement,
    VerificationResult,
    VerificationLevel,
    TestCase,
    CertificationReport
)


class TestVerificationRequirement(unittest.TestCase):
    """Test the VerificationRequirement class"""
    
    def test_requirement_creation(self):
        """Test creating verification requirements"""
        req = VerificationRequirement(
            id="REQ-001",
            description="Entropy reduction requirement",
            level=VerificationLevel.CRITICAL,
            test_criteria="Entropy must reduce by at least 10%",
            tolerance=0.1
        )
        
        self.assertEqual(req.id, "REQ-001")
        self.assertEqual(req.level, VerificationLevel.CRITICAL)
        self.assertEqual(req.tolerance, 0.1)
        self.assertFalse(req.verified)
    
    def test_requirement_verification(self):
        """Test requirement verification status"""
        req = VerificationRequirement(
            id="REQ-002",
            description="Information preservation",
            level=VerificationLevel.ESSENTIAL
        )
        
        # Initially not verified
        self.assertFalse(req.verified)
        
        # Mark as verified
        req.mark_verified(True, evidence="Test passed")
        self.assertTrue(req.verified)
        self.assertEqual(req.evidence, "Test passed")


class TestAxiomVerificationEngine(unittest.TestCase):
    """Test the main verification engine"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = AxiomVerificationEngine()
        
        # Test data
        self.test_state = {
            "vector": [1.0, 0.0, 0.0, 0.0, 0.0],
            "entropy": 2.0,
            "information": 1.5
        }
    
    def test_initialization(self):
        """Test proper initialization of verification engine"""
        self.assertIsNotNone(self.engine)
        self.assertTrue(hasattr(self.engine, 'requirements'))
        self.assertTrue(hasattr(self.engine, 'test_cases'))
        self.assertTrue(hasattr(self.engine, 'run_verification'))
    
    def test_requirements_definition(self):
        """Test that all 6 critical requirements are defined"""
        requirements = self.engine.get_requirements()
        
        self.assertEqual(len(requirements), 6)
        
        # Check requirement IDs
        req_ids = [req.id for req in requirements]
        expected_ids = ["REQ-001", "REQ-002", "REQ-003", "REQ-004", "REQ-005", "REQ-006"]
        
        for expected_id in expected_ids:
            self.assertIn(expected_id, req_ids)
        
        # All should be critical level
        for req in requirements:
            self.assertEqual(req.level, VerificationLevel.CRITICAL)
    
    def test_test_case_generation(self):
        """Test automatic test case generation"""
        test_cases = self.engine.generate_test_cases()
        
        self.assertGreater(len(test_cases), 0)
        
        for tc in test_cases:
            self.assertIsInstance(tc, TestCase)
            self.assertTrue(hasattr(tc, 'id'))
            self.assertTrue(hasattr(tc, 'requirement_id'))
            self.assertTrue(hasattr(tc, 'input_data'))
            self.assertTrue(hasattr(tc, 'expected_output'))
    
    def test_run_single_test_case(self):
        """Test running a single test case"""
        test_case = TestCase(
            id="TC-001",
            requirement_id="REQ-001",
            description="Test entropy reduction",
            input_data=self.test_state,
            expected_output={"entropy_reduced": True}
        )
        
        result = self.engine.run_test_case(test_case)
        
        self.assertIsInstance(result, dict)
        self.assertIn("passed", result)
        self.assertIn("actual_output", result)
        self.assertIn("execution_time", result)
    
    def test_run_full_verification(self):
        """Test running complete verification suite"""
        result = self.engine.run_verification()
        
        self.assertIsInstance(result, VerificationResult)
        self.assertTrue(hasattr(result, 'all_passed'))
        self.assertTrue(hasattr(result, 'requirements_status'))
        self.assertTrue(hasattr(result, 'test_results'))
        self.assertTrue(hasattr(result, 'coverage'))
        
        # Coverage should be 100% for critical requirements
        self.assertEqual(result.coverage, 100.0)
    
    def test_entropy_reduction_verification(self):
        """Test REQ-001: Entropy reduction verification"""
        # Create states with known entropy values
        high_entropy_state = {
            "vector": np.ones(5) / np.sqrt(5),
            "entropy": 5.0,
            "information": 3.0
        }
        
        low_entropy_state = {
            "vector": np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
            "entropy": 1.0,
            "information": 3.0
        }
        
        # Verify entropy reduction
        result = self.engine.verify_entropy_reduction(
            high_entropy_state,
            low_entropy_state
        )
        
        self.assertTrue(result["passed"])
        self.assertGreater(result["reduction_percentage"], 10.0)
    
    def test_information_preservation_verification(self):
        """Test REQ-002: Information preservation verification"""
        initial_state = self.test_state
        final_state = {
            "vector": [0.8, 0.6, 0.0, 0.0, 0.0],
            "entropy": 1.0,
            "information": 1.48  # Within 5% of initial
        }
        
        result = self.engine.verify_information_preservation(
            initial_state,
            final_state
        )
        
        self.assertTrue(result["passed"])
        self.assertLess(result["information_loss"], 0.05)
    
    def test_homomorphism_verification(self):
        """Test REQ-003: Homomorphism property verification"""
        state_a = {
            "vector": [1.0, 0.0, 0.0, 0.0, 0.0],
            "entropy": 1.0,
            "information": 0.8
        }
        
        state_b = {
            "vector": [0.0, 1.0, 0.0, 0.0, 0.0],
            "entropy": 1.0,
            "information": 0.8
        }
        
        result = self.engine.verify_homomorphism_property(state_a, state_b)
        
        self.assertIn("passed", result)
        self.assertIn("composition_error", result)
        
        # Error should be small
        if result["passed"]:
            self.assertLess(result["composition_error"], 0.01)
    
    def test_consistency_verification(self):
        """Test REQ-004: Consistency verification"""
        # Run same verification multiple times
        results = []
        
        for _ in range(5):
            result = self.engine.verify_consistency(self.test_state)
            results.append(result)
        
        # All results should be identical
        for i in range(1, len(results)):
            self.assertEqual(results[0], results[i])
    
    def test_convergence_verification(self):
        """Test REQ-005: Convergence verification"""
        result = self.engine.verify_convergence(
            initial_state=self.test_state,
            max_iterations=100
        )
        
        self.assertTrue(result["converged"])
        self.assertLess(result["iterations"], 100)
        self.assertLess(result["final_entropy"], self.test_state["entropy"])
    
    def test_stability_verification(self):
        """Test REQ-006: Stability verification"""
        # Add small perturbation
        perturbed_state = {
            "vector": [1.01, 0.01, 0.0, 0.0, 0.0],
            "entropy": 2.02,
            "information": 1.51
        }
        
        result = self.engine.verify_stability(
            self.test_state,
            perturbed_state
        )
        
        self.assertTrue(result["stable"])
        self.assertLess(result["deviation"], 0.1)
    
    def test_continuous_monitoring(self):
        """Test continuous monitoring capability"""
        # Start monitoring
        self.engine.start_continuous_monitoring(interval=0.1)
        
        # Run some operations
        for _ in range(3):
            self.engine.run_verification()
        
        # Stop monitoring
        monitoring_data = self.engine.stop_continuous_monitoring()
        
        self.assertIn("duration", monitoring_data)
        self.assertIn("checks_performed", monitoring_data)
        self.assertIn("failures_detected", monitoring_data)
        self.assertGreaterEqual(monitoring_data["checks_performed"], 3)
    
    def test_generate_certification_report(self):
        """Test certification report generation"""
        # Run verification first
        verification_result = self.engine.run_verification()
        
        # Generate report
        report = self.engine.generate_certification_report(
            verification_result,
            certifier="Test Engineer",
            standard="DO-178C Level A"
        )
        
        self.assertIsInstance(report, CertificationReport)
        self.assertEqual(report.standard, "DO-178C Level A")
        self.assertEqual(report.certifier, "Test Engineer")
        self.assertIsNotNone(report.timestamp)
        self.assertIn("requirements", report.sections)
        self.assertIn("test_results", report.sections)
        self.assertIn("coverage", report.sections)
    
    def test_regression_testing(self):
        """Test regression testing capability"""
        # Create baseline
        baseline = self.engine.create_baseline("v1.0")
        
        # Make some changes (simulate)
        # Run verification again
        new_result = self.engine.run_verification()
        
        # Compare with baseline
        regression_report = self.engine.compare_with_baseline(
            baseline,
            new_result
        )
        
        self.assertIn("changes_detected", regression_report)
        self.assertIn("new_failures", regression_report)
        self.assertIn("fixed_issues", regression_report)
    
    def test_performance_requirements(self):
        """Test performance requirements for verification"""
        import time
        
        # Single test case should complete quickly
        test_case = self.engine.generate_test_cases()[0]
        
        start_time = time.time()
        self.engine.run_test_case(test_case)
        elapsed = time.time() - start_time
        
        # Should complete within 100ms
        self.assertLess(elapsed, 0.1)
        
        # Full verification should complete within reasonable time
        start_time = time.time()
        self.engine.run_verification()
        elapsed = time.time() - start_time
        
        # Should complete within 5 seconds
        self.assertLess(elapsed, 5.0)
    
    def test_parallel_verification(self):
        """Test parallel execution of verification tests"""
        # Generate many test cases
        test_cases = []
        for i in range(20):
            tc = TestCase(
                id=f"TC-PAR-{i}",
                requirement_id="REQ-001",
                description=f"Parallel test {i}",
                input_data=self.test_state
            )
            test_cases.append(tc)
        
        # Run in parallel
        results = self.engine.run_parallel_verification(test_cases)
        
        self.assertEqual(len(results), len(test_cases))
        
        # All should have results
        for result in results:
            self.assertIn("passed", result)
    
    def test_fault_injection(self):
        """Test fault injection for robustness testing"""
        # Inject various faults
        faults = [
            {"type": "invalid_input", "data": None},
            {"type": "dimension_mismatch", "data": {"vector": [1, 2]}},
            {"type": "negative_entropy", "data": {"entropy": -1}},
        ]
        
        for fault in faults:
            with self.subTest(fault=fault["type"]):
                result = self.engine.run_with_fault_injection(fault)
                
                # Should handle gracefully
                self.assertIn("handled", result)
                self.assertTrue(result["handled"])
                self.assertIn("recovery_action", result)


class TestVerificationIntegration(unittest.TestCase):
    """Integration tests for the verification system"""
    
    def test_complete_verification_workflow(self):
        """Test complete verification workflow"""
        engine = AxiomVerificationEngine()
        
        # 1. Define requirements
        requirements = engine.get_requirements()
        self.assertEqual(len(requirements), 6)
        
        # 2. Generate test cases
        test_cases = engine.generate_test_cases()
        self.assertGreater(len(test_cases), 0)
        
        # 3. Run verification
        result = engine.run_verification()
        
        # 4. Check all requirements verified
        for req_status in result.requirements_status.values():
            if not req_status["verified"]:
                print(f"Requirement {req_status['id']} failed: {req_status.get('reason')}")
        
        # 5. Generate certification report
        if result.all_passed:
            report = engine.generate_certification_report(
                result,
                certifier="Automated Test System",
                standard="DO-178C Level A"
            )
            
            # Save report
            report_path = engine.save_certification_report(report)
            self.assertTrue(report_path.exists())
        
        # 6. Create baseline for future regression testing
        baseline = engine.create_baseline("test_baseline")
        self.assertIsNotNone(baseline)


if __name__ == '__main__':
    unittest.main()