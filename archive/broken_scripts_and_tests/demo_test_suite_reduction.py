"""
Demonstration of Unsupervised Test Suite Reduction
=================================================

This script demonstrates the newly implemented unsupervised test suite
reduction feature within the Kimera AI Test Suite.

It configures the test suite to run only the 'mlperf_inference' category
and enables the test suite reduction with a small target cluster size
to clearly show the effect of the reduction process.

To run this demo:
- Ensure all dependencies, including 'scikit-learn', are installed.
- Execute from the root directory: python -m scripts.demo_test_suite_reduction

Expected output:
The logs will show the original number of tests in the category,
the application of the TestSuiteReducer, and the final reduced number
of tests being executed.
"""

import asyncio
import logging
import os
import sys

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

async def main():
    """Main function to run the demonstration."""
    logger.info("🚀 Starting Test Suite Reduction Demonstration")
    logger.info("="*50)

    # 1. Configure the test suite for reduction
    # We enable reduction and set a small cluster size for a clear demo.
    # We select only one category to keep the output concise.
    config = KimeraAITestConfig(
        test_categories=[TestCategory.MLPERF_INFERENCE],
        enable_test_suite_reduction=True,
        reduction_cluster_size=2,  # Reduce to 2 tests
        enable_gpu_optimization=False, # Disable GPU to run on any machine
        save_detailed_logs=False
    )

    logger.info(f"Configuration: Test Suite Reduction ENABLED")
    logger.info(f"Configuration: Target Cluster Size = {config.reduction_cluster_size}")
    logger.info(f"Configuration: Test Categories = {[c.value for c in config.test_categories]}")


    # 2. Initialize and run the test suite
    test_suite = KimeraAITestSuiteIntegration(config)

    try:
        report = await test_suite.run_comprehensive_test_suite()
        logger.info("✅ Demonstration Finished Successfully")
        logger.info("="*50)
        logger.info("📊 Final Report Summary:")
        
        total_tests = report['overall_results']['total_tests']
        passed = report['overall_results']['passed_tests']
        
        logger.info(f"  - Total tests executed: {total_tests}")
        logger.info(f"  - Passed: {passed}")
        
        if total_tests == config.reduction_cluster_size:
            logger.info("✔️ VERIFICATION SUCCESS: The number of executed tests matches the target cluster size.")
        else:
            logger.error("❌ VERIFICATION FAILED: The number of executed tests does not match the target cluster size.")

    except Exception as e:
        logger.error(f"An error occurred during the demonstration: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main()) 