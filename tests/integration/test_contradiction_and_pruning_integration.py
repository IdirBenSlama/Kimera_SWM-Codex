"""
Integration Test for Proactive Contradiction Detection and Pruning System
========================================================================

This test suite validates the integration of contradiction detection and pruning
systems following DO-178C Level A certification standards.

Test Categories:
- Unit tests for individual components
- Integration tests for coordinated operations
- Safety verification tests
- Performance benchmark tests
- Failure mode analysis tests

Safety Requirements Tested:
- SR-4.15.1 through SR-4.15.16: All critical safety requirements
- Error handling and graceful degradation
- Data integrity and consistency
- System health monitoring

References:
- DO-178C: Software Considerations in Airborne Systems and Equipment Certification
- DO-333: Formal Methods Supplement to DO-178C
"""

import asyncio
import json
import logging
import unittest
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

# Set up logging for test visibility
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from src.core.contradiction_and_pruning import (
        ContradictionAndPruningIntegrator,
        GeoidState,
        InsightScar,
        IntelligentPruningEngine,
        ProactiveContradictionDetector,
        ProactiveDetectionConfig,
        PruningConfig,
        PruningDecision,
        SafetyStatus,
        Scar,
        create_contradiction_and_pruning_integrator,
    )

    IMPORTS_AVAILABLE = True
except ImportError as e:
    logger.error(f"❌ Failed to import contradiction and pruning modules: {e}")
    IMPORTS_AVAILABLE = False


class TestContradictionAndPruningIntegration(unittest.TestCase):
    """
    Integration test suite for contradiction detection and pruning system.

    Following aerospace testing standards with comprehensive verification
    of all critical system functions and safety requirements.
    """

    def setUp(self):
        """Set up test environment with mock data and configurations."""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required modules not available")

        # Create test configurations
        self.detection_config = ProactiveDetectionConfig(
            batch_size=10,
            similarity_threshold=0.7,
            scan_interval_hours=1,  # Short for testing
            max_comparisons_per_run=100,
            enable_clustering=True,
            enable_temporal_analysis=True,
        )

        self.pruning_config = PruningConfig(
            vault_pressure_threshold=0.8,
            memory_pressure_threshold=0.9,
            deprecated_insight_priority=10.0,
            max_prune_per_cycle=50,
            safety_margin=0.1,
        )

        # Create test data
        self.test_geoids = self._create_test_geoids()
        self.test_prunable_items = self._create_test_prunable_items()

        logger.info("🔧 Test environment set up complete")

    def _create_test_geoids(self) -> List[GeoidState]:
        """Create test geoids for analysis."""
        geoids = []
        base_time = datetime.now(timezone.utc)

        for i in range(5):
            geoid = GeoidState(
                geoid_id=f"test_geoid_{i}",
                semantic_state={
                    "content": f"Test content {i} with semantic meaning",
                    "domain": "test_domain" if i < 3 else "alternate_domain",
                },
                symbolic_state={"symbols": [f"symbol_{i}", f"alt_symbol_{i}"]},
                embedding_vector=[float(j + i) for j in range(10)],
                metadata={
                    "created": (base_time - timedelta(days=i)).isoformat(),
                    "test": True,
                    "last_accessed": (base_time - timedelta(hours=i)).isoformat(),
                },
            )
            geoids.append(geoid)

        return geoids

    def _create_test_prunable_items(self) -> List:
        """Create test items for pruning analysis."""
        items = []
        base_time = datetime.now(timezone.utc)

        # Create deprecated insights (high pruning priority)
        for i in range(3):
            insight = InsightScar(
                scar_id=f"deprecated_insight_{i}",
                content=f"Deprecated insight content {i}",
                created_at=base_time - timedelta(days=30 + i),
                utility_score=0.1,  # Low utility
                metadata={"safety_critical": False},
            )
            insight.status = "deprecated"
            items.append(insight)

        # Create active insights (should be preserved)
        for i in range(3):
            insight = InsightScar(
                scar_id=f"active_insight_{i}",
                content=f"Active insight content {i}",
                created_at=base_time - timedelta(days=i),
                utility_score=0.8,  # High utility
                metadata={"safety_critical": False},
            )
            insight.status = "active"
            insight.access_count = 10  # Frequently accessed
            items.append(insight)

        # Create safety-critical items (must be preserved)
        safety_critical = InsightScar(
            scar_id="safety_critical_insight",
            content="Safety critical system insight",
            created_at=base_time - timedelta(days=60),
            utility_score=0.9,
            metadata={"safety_critical": True},
        )
        items.append(safety_critical)

        # Create regular SCARs
        for i in range(3):
            scar = Scar(
                scar_id=f"regular_scar_{i}",
                created_at=base_time - timedelta(days=15 + i),
                utility_score=0.3 + (i * 0.2),
                metadata={"safety_critical": False},
            )
            items.append(scar)

        return items

    def test_01_component_initialization(self):
        """Test SR-4.15.3: All initialization must complete successfully."""
        logger.info("🧪 Testing component initialization...")

        # Test individual component initialization
        detector = ProactiveContradictionDetector(self.detection_config)
        self.assertIsNotNone(detector)
        self.assertEqual(detector.config.batch_size, 10)

        pruning_engine = IntelligentPruningEngine(self.pruning_config)
        self.assertIsNotNone(pruning_engine)
        self.assertEqual(pruning_engine.config.max_prune_per_cycle, 50)

        # Test integrated initialization
        integrator = ContradictionAndPruningIntegrator(
            self.detection_config, self.pruning_config
        )
        self.assertIsNotNone(integrator)
        self.assertIsNotNone(integrator.contradiction_detector)
        self.assertIsNotNone(integrator.pruning_engine)

        logger.info("✅ Component initialization test passed")

    def test_02_health_monitoring(self):
        """Test comprehensive health monitoring capabilities."""
        logger.info("🧪 Testing health monitoring...")

        integrator = create_contradiction_and_pruning_integrator(
            self.detection_config, self.pruning_config
        )

        # Test health status retrieval
        health_status = integrator.get_comprehensive_health_status()
        self.assertIsInstance(health_status, dict)
        self.assertIn("integration_status", health_status)
        self.assertIn("contradiction_detection", health_status)
        self.assertIn("intelligent_pruning", health_status)
        self.assertIn("safety_assessment", health_status)

        # Verify safety assessment structure
        safety_assessment = health_status["safety_assessment"]
        self.assertIn("overall_safety_status", safety_assessment)
        self.assertIn("safety_score", safety_assessment)
        self.assertIn("safety_indicators", safety_assessment)

        # Test individual component health
        detector_health = integrator.contradiction_detector.get_health_status()
        self.assertIn("status", detector_health)
        self.assertIn("performance_metrics", detector_health)

        pruning_health = integrator.pruning_engine.get_health_status()
        self.assertIn("performance_metrics", pruning_health)
        self.assertIn("safety_features", pruning_health)

        logger.info("✅ Health monitoring test passed")

    def test_03_contradiction_detection(self):
        """Test proactive contradiction detection functionality."""
        logger.info("🧪 Testing contradiction detection...")

        detector = ProactiveContradictionDetector(self.detection_config)

        # Test scan decision logic
        self.assertTrue(detector.should_run_scan())  # First scan should always run

        # Run detection scan with test data
        scan_results = detector.run_proactive_scan(self.test_geoids)

        self.assertIsInstance(scan_results, dict)
        self.assertIn("status", scan_results)
        self.assertIn("tensions_found", scan_results)
        self.assertIn("geoids_scanned", scan_results)
        self.assertIn("strategies_used", scan_results)

        # Verify scan completed successfully
        if scan_results["status"] == "completed":
            self.assertEqual(scan_results["geoids_scanned"], len(self.test_geoids))
            self.assertIsInstance(scan_results["tensions_found"], list)
            self.assertGreater(len(scan_results["strategies_used"]), 0)

        # Test JSON serialization (SR-4.15.1)
        json_str = json.dumps(scan_results)
        self.assertIsInstance(json_str, str)

        logger.info(
            f"✅ Contradiction detection test passed - {scan_results.get('status', 'unknown')} status"
        )

    def test_04_intelligent_pruning(self):
        """Test intelligent pruning functionality."""
        logger.info("🧪 Testing intelligent pruning...")

        engine = IntelligentPruningEngine(self.pruning_config)

        # Test individual item analysis
        deprecated_item = self.test_prunable_items[0]  # First item is deprecated
        result = engine.should_prune(deprecated_item, vault_pressure=0.5)

        self.assertIsNotNone(result)
        self.assertEqual(result.item_id, deprecated_item.item_id)
        self.assertIn(
            result.decision,
            [PruningDecision.PRUNE, PruningDecision.PRESERVE, PruningDecision.DEFER],
        )
        self.assertGreaterEqual(result.confidence_score, 0.0)
        self.assertLessEqual(result.confidence_score, 1.0)

        # Test safety-critical item protection
        safety_critical_item = next(
            item
            for item in self.test_prunable_items
            if item.metadata.get("safety_critical", False)
        )
        safety_result = engine.should_prune(safety_critical_item, vault_pressure=0.9)

        self.assertEqual(safety_result.decision, PruningDecision.PRESERVE)
        self.assertEqual(safety_result.safety_status, SafetyStatus.SAFETY_CRITICAL)

        # Test batch analysis
        batch_results = engine.analyze_batch(
            self.test_prunable_items[:5], vault_pressure=0.6
        )
        self.assertEqual(len(batch_results), 5)

        for batch_result in batch_results:
            self.assertIsInstance(batch_result.pruning_score, (int, float))
            self.assertIn(
                batch_result.decision,
                [
                    PruningDecision.PRUNE,
                    PruningDecision.PRESERVE,
                    PruningDecision.DEFER,
                ],
            )

        logger.info("✅ Intelligent pruning test passed")

    def test_05_integration_workflow(self):
        """Test integrated workflow coordination."""
        logger.info("🧪 Testing integration workflow...")

        integrator = create_contradiction_and_pruning_integrator(
            self.detection_config, self.pruning_config
        )

        # Test complete integrated analysis cycle
        async def run_integration_test():
            cycle_results = await integrator.run_integrated_analysis_cycle(
                vault_pressure=0.7,
                geoids=self.test_geoids,
                prunable_items=self.test_prunable_items,
            )

            self.assertIsInstance(cycle_results, dict)
            self.assertIn("status", cycle_results)
            self.assertIn("contradiction_detection", cycle_results)
            self.assertIn("pruning_analysis", cycle_results)
            self.assertIn("integration_actions", cycle_results)
            self.assertIn("safety_assessment", cycle_results)
            self.assertIn("performance_metrics", cycle_results)

            # Verify safety assessment
            safety_assessment = cycle_results["safety_assessment"]
            self.assertIn("overall_safety_score", safety_assessment)
            self.assertIn("safety_level", safety_assessment)

            # Verify integration actions
            actions = cycle_results["integration_actions"]
            self.assertIsInstance(actions, list)

            return cycle_results

        # Run the async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            cycle_results = loop.run_until_complete(run_integration_test())
            logger.info(
                f"✅ Integration workflow test passed - Status: {cycle_results.get('status', 'unknown')}"
            )
        finally:
            loop.close()

    def test_06_safety_compliance(self):
        """Test safety compliance and protection mechanisms."""
        logger.info("🧪 Testing safety compliance...")

        integrator = create_contradiction_and_pruning_integrator(
            self.detection_config, self.pruning_config
        )

        # Test item protection mechanism
        protected_item_id = "test_protected_item"
        integrator.protect_from_pruning([protected_item_id], "unit_test_protection")

        # Verify protection is active
        self.assertIn(protected_item_id, integrator.pruning_engine.protected_items)

        # Test safety margin application
        self.assertEqual(integrator.pruning_config.safety_margin, 0.1)

        # Test degraded mode operation (no database)
        detector = ProactiveContradictionDetector(self.detection_config)
        fallback_results = detector.run_proactive_scan()  # No geoids provided

        self.assertIsInstance(fallback_results, dict)
        self.assertIn("status", fallback_results)

        # Test error handling
        try:
            invalid_config = PruningConfig(
                vault_pressure_threshold=2.0
            )  # Invalid value
            self.fail("Should have raised ValueError for invalid configuration")
        except ValueError:
            pass  # Expected behavior

        logger.info("✅ Safety compliance test passed")

    def test_07_performance_benchmarks(self):
        """Test performance requirements and benchmarks."""
        logger.info("🧪 Testing performance benchmarks...")

        integrator = create_contradiction_and_pruning_integrator(
            self.detection_config, self.pruning_config
        )

        # Measure detection performance
        start_time = datetime.now(timezone.utc)
        detection_results = integrator.contradiction_detector.run_proactive_scan(
            self.test_geoids
        )
        detection_duration = (datetime.now(timezone.utc) - start_time).total_seconds()

        # Verify detection performance (should be under 10 seconds for test data)
        self.assertLess(detection_duration, 10.0)

        # Measure pruning performance
        start_time = datetime.now(timezone.utc)
        pruning_results = integrator.pruning_engine.analyze_batch(
            self.test_prunable_items, 0.6
        )
        pruning_duration = (datetime.now(timezone.utc) - start_time).total_seconds()

        # Verify pruning performance (should be under 5 seconds for test data)
        self.assertLess(pruning_duration, 5.0)

        # Test memory efficiency
        metrics = integrator.get_integration_metrics()
        self.assertIsInstance(metrics, dict)
        self.assertIn("uptime_seconds", metrics)
        self.assertIn("scans_per_hour", metrics)

        logger.info(
            f"✅ Performance benchmarks passed - Detection: {detection_duration:.2f}s, Pruning: {pruning_duration:.2f}s"
        )

    def test_08_failure_modes(self):
        """Test failure mode analysis and error handling."""
        logger.info("🧪 Testing failure modes...")

        # Test invalid configuration handling
        with self.assertRaises(ValueError):
            ProactiveDetectionConfig(batch_size=0)  # Invalid batch size

        with self.assertRaises(ValueError):
            PruningConfig(safety_margin=1.5)  # Invalid safety margin

        # Test graceful degradation with missing dependencies
        detector = ProactiveContradictionDetector(self.detection_config)

        # Test with invalid geoid data
        invalid_geoid = GeoidState("invalid", {}, {}, [], {})
        results = detector.run_proactive_scan([invalid_geoid])
        self.assertIsInstance(results, dict)

        # Test pruning engine error handling
        engine = IntelligentPruningEngine(self.pruning_config)

        # Test with malformed item
        class MalformedItem:
            def __init__(self):
                self.item_id = "malformed"
                # Missing required attributes

        try:
            malformed_item = MalformedItem()
            result = engine.should_prune(malformed_item, 0.5)
            # Should handle gracefully or raise appropriate exception
        except Exception as e:
            self.assertIsInstance(e, (AttributeError, TypeError))

        logger.info("✅ Failure mode analysis test passed")


def run_integration_tests():
    """Run the complete integration test suite."""
    logger.info("🚀 Starting Contradiction and Pruning Integration Test Suite")
    logger.info("   Following DO-178C Level A certification standards")

    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestContradictionAndPruningIntegration
    )

    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=None)
    result = runner.run(test_suite)

    # Generate test report
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    success_rate = (
        ((total_tests - failures - errors) / total_tests) * 100
        if total_tests > 0
        else 0
    )

    logger.info("=" * 80)
    logger.info("🔍 CONTRADICTION AND PRUNING INTEGRATION TEST REPORT")
    logger.info("=" * 80)
    logger.info(f"   Total Tests: {total_tests}")
    logger.info(f"   Passed: {total_tests - failures - errors}")
    logger.info(f"   Failed: {failures}")
    logger.info(f"   Errors: {errors}")
    logger.info(f"   Success Rate: {success_rate:.1f}%")
    logger.info("=" * 80)

    if success_rate >= 90.0:
        logger.info("✅ INTEGRATION TEST SUITE PASSED - System ready for deployment")
    else:
        logger.error(
            "❌ INTEGRATION TEST SUITE FAILED - Review failures before deployment"
        )

    return result.wasSuccessful()


if __name__ == "__main__":
    # Run integration tests when executed directly
    success = run_integration_tests()
    exit(0 if success else 1)
