"""
Rigorous Test of Unsupervised Test Suite Reduction
=================================================

This script performs a series of rigorous tests on the test suite
reduction feature to validate its behavior in various scenarios,
including edge cases.

Test Scenarios:
1.  Reduction Enabled (Standard): Reduces a 5-test category to 3.
2.  Reduction Skipped (Edge Case): Attempts to reduce a 5-test
    category to 5 clusters. The reduction should be gracefully skipped.
3.  Reduction Disabled: Runs a 5-test category with reduction disabled
    to ensure the original functionality is unaffected.
4.  Different Category: Runs reduction on the 4-test KIMERA_COGNITIVE
    category to ensure the reducer is general-purpose.

To run this test:
- Execute from the root directory: python -m scripts.rigorous_test_reduction
"""

import asyncio
import logging
import os
import sys
from typing import List

# Ensure the project root is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.kimera_ai_test_suite_integration import (
    KimeraAITestSuiteIntegration,
    KimeraAITestConfig,
    TestCategory
)
from backend.utils.kimera_logger import get_system_logger

# Configure logger for clear output
logger = get_system_logger(__name__)
logging.basicConfig(level=logging.INFO)


async def run_test_scenario(
    scenario_name: str,
    test_categories: List[TestCategory],
    reduction_enabled: bool,
    cluster_size: int,
    expected_tests: int
):
    """Runs a single test scenario with a specific configuration."""
    logger.info(f"--- SCENARIO: {scenario_name} ---")
    
    config = KimeraAITestConfig(
        test_categories=test_categories,
        enable_test_suite_reduction=reduction_enabled,
        reduction_cluster_size=cluster_size,
        enable_gpu_optimization=False,
        save_detailed_logs=False,
        generate_visualizations=False
    )
    
    test_suite = KimeraAITestSuiteIntegration(config)
    report = await test_suite.run_comprehensive_test_suite()
    
    total_tests_executed = report['overall_results']['total_tests']
    
    logger.info(f"Expected tests: {expected_tests}, Executed tests: {total_tests_executed}")
    
    if total_tests_executed == expected_tests:
        logger.info(f"✔️ VERIFICATION SUCCESS for scenario: {scenario_name}")
        return True
    else:
        logger.error(f"❌ VERIFICATION FAILED for scenario: {scenario_name}")
        return False


async def main():
    """Main function to run all rigorous test scenarios."""
    logger.info("🚀 Starting Rigorous Test of Suite Reduction")
    logger.info("="*50)

    results = {}

    # Scenario 1: Standard reduction
    results['standard_reduction'] = await run_test_scenario(
        scenario_name="Standard Reduction",
        test_categories=[TestCategory.MLPERF_INFERENCE],
        reduction_enabled=True,
        cluster_size=3,
        expected_tests=3
    )

    # Scenario 2: Reduction skipped (n_clusters == n_tests)
    results['reduction_skipped'] = await run_test_scenario(
        scenario_name="Reduction Skipped (Edge Case)",
        test_categories=[TestCategory.MLPERF_INFERENCE],
        reduction_enabled=True,
        cluster_size=5,
        expected_tests=5
    )

    # Scenario 3: Reduction disabled
    results['reduction_disabled'] = await run_test_scenario(
        scenario_name="Reduction Disabled",
        test_categories=[TestCategory.MLPERF_INFERENCE],
        reduction_enabled=False,
        cluster_size=3,  # This value should be ignored
        expected_tests=5
    )
    
    # Scenario 4: Different category reduction
    results['different_category'] = await run_test_scenario(
        scenario_name="Different Category (KIMERA_COGNITIVE)",
        test_categories=[TestCategory.KIMERA_COGNITIVE],
        reduction_enabled=True,
        cluster_size=2,
        expected_tests=2
    )
    
    logger.info("="*50)
    logger.info("🏁 Rigorous Test Summary:")
    
    all_passed = True
    for scenario, passed in results.items():
        status = "✔️ PASSED" if passed else "❌ FAILED"
        logger.info(f"  - {scenario}: {status}")
        if not passed:
            all_passed = False
            
    if all_passed:
        logger.info("\n✅ All rigorous test scenarios passed successfully.")
    else:
        logger.error("\n❌ One or more rigorous test scenarios failed.")


if __name__ == "__main__":
    asyncio.run(main()) 