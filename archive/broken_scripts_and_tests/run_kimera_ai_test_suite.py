#!/usr/bin/env python3
"""
Kimera AI Test Suite Execution Script
=====================================

Command-line interface for running the comprehensive AI test suite
with various configuration options and execution modes.

Usage:
    python scripts/run_kimera_ai_test_suite.py [options]

Examples:
    # Run quick test suite
    python scripts/run_kimera_ai_test_suite.py --quick

    # Run full test suite
    python scripts/run_kimera_ai_test_suite.py --full

    # Run only MLPerf tests
    python scripts/run_kimera_ai_test_suite.py --mlperf-only

    # Run with custom config
    python scripts/run_kimera_ai_test_suite.py --config config/custom_ai_test_config.json

    # Run without GPU optimization
    python scripts/run_kimera_ai_test_suite.py --no-gpu

    # Dry run (show what would be executed)
    python scripts/run_kimera_ai_test_suite.py --dry-run

Author: Kimera Development Team
Version: 1.0.0
"""

import asyncio
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.kimera_ai_test_suite_integration import (
    KimeraAITestSuiteIntegration,
    KimeraAITestConfig,
    TestCategory,
    run_quick_test_suite,
    run_full_test_suite,
    run_kimera_cognitive_tests
)
from backend.utils.kimera_logger import get_system_logger
from backend.utils.gpu_foundation import GPUValidationLevel
from backend.monitoring.kimera_monitoring_core import MonitoringLevel

logger = get_system_logger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        logger.info(f"✅ Configuration loaded from {config_path}")
        return config_data
    except Exception as e:
        logger.error(f"❌ Failed to load configuration from {config_path}: {e}")
        raise


def create_test_config_from_args(args: argparse.Namespace, config_data: Optional[Dict[str, Any]] = None) -> KimeraAITestConfig:
    """Create test configuration from command line arguments and config file"""
    
    # Default categories
    test_categories = list(TestCategory)
    
    # Override based on command line arguments
    if args.mlperf_only:
        test_categories = [TestCategory.MLPERF_INFERENCE]
    elif args.safety_only:
        test_categories = [TestCategory.SAFETY_ASSESSMENT]
    elif args.cognitive_only:
        test_categories = [TestCategory.KIMERA_COGNITIVE]
    elif args.domain_only:
        test_categories = [TestCategory.DOMAIN_SPECIFIC]
    elif args.cert_only:
        test_categories = [TestCategory.CERTIFICATION_PREP]
    
    # GPU validation level
    gpu_validation_level = GPUValidationLevel.RIGOROUS
    if args.gpu_validation_level:
        gpu_validation_level = GPUValidationLevel(args.gpu_validation_level)
    
    # Monitoring level
    monitoring_level = MonitoringLevel.DETAILED
    if args.monitoring_level:
        monitoring_level = MonitoringLevel(args.monitoring_level)
    
    # Create configuration
    config = KimeraAITestConfig(
        test_categories=test_categories,
        gpu_validation_level=gpu_validation_level,
        monitoring_level=monitoring_level,
        max_test_duration_minutes=args.max_duration or 60,
        enable_gpu_optimization=not args.no_gpu,
        enable_cognitive_monitoring=not args.no_cognitive_monitoring,
        output_directory=args.output_dir or "test_results",
        save_detailed_logs=not args.no_detailed_logs,
        generate_visualizations=not args.no_visualizations
    )
    
    # Override with config file settings if provided
    if config_data and 'execution_settings' in config_data:
        settings = config_data['execution_settings']
        if not args.max_duration:
            config.max_test_duration_minutes = settings.get('max_test_duration_minutes', 60)
        if not hasattr(args, 'no_gpu') or not args.no_gpu:
            config.enable_gpu_optimization = settings.get('enable_gpu_optimization', True)
        if not hasattr(args, 'no_cognitive_monitoring') or not args.no_cognitive_monitoring:
            config.enable_cognitive_monitoring = settings.get('enable_cognitive_monitoring', True)
        if not args.output_dir:
            config.output_directory = settings.get('output_directory', 'test_results')
    
    return config


def print_test_plan(config: KimeraAITestConfig, config_data: Optional[Dict[str, Any]] = None):
    """Print the test execution plan"""
    logger.info("📋 TEST EXECUTION PLAN")
    logger.info("=" * 50)
    
    logger.info(f"📂 Output Directory: {config.output_directory}")
    logger.info(f"⏱️  Max Duration: {config.max_test_duration_minutes} minutes")
    logger.info(f"🖥️  GPU Optimization: {'Enabled' if config.enable_gpu_optimization else 'Disabled'}")
    logger.info(f"🧠 Cognitive Monitoring: {'Enabled' if config.enable_cognitive_monitoring else 'Disabled'}")
    logger.info(f"📊 Monitoring Level: {config.monitoring_level.value}")
    logger.info(f"🔬 GPU Validation: {config.gpu_validation_level.value}")
    
    logger.info("\n📋 TEST CATEGORIES:")
    for category in config.test_categories:
        logger.info(f"  ✅ {category.value}")
        
        # Show test details from config if available
        if config_data and 'test_categories' in config_data:
            category_config = config_data['test_categories'].get(category.value, {})
            if 'tests' in category_config:
                test_count = len(category_config['tests'])
                logger.info(f"     ({test_count} tests)")
    
    logger.info("=" * 50)


async def run_test_suite(config: KimeraAITestConfig) -> Dict[str, Any]:
    """Run the test suite with the given configuration"""
    logger.info("🚀 Starting Kimera AI Test Suite Execution")
    
    try:
        # Create and run test suite
        suite = KimeraAITestSuiteIntegration(config)
        results = await suite.run_comprehensive_test_suite()
        
        # Log summary
        overall_results = results['overall_results']
        logger.info("🏁 TEST SUITE COMPLETED")
        logger.info("=" * 50)
        logger.info(f"📊 Total Tests: {overall_results['total_tests']}")
        logger.info(f"✅ Passed: {overall_results['passed_tests']}")
        logger.info(f"❌ Failed: {overall_results['failed_tests']}")
        logger.info(f"📈 Pass Rate: {overall_results['pass_rate']:.1f}%")
        logger.info(f"🎯 Average Accuracy: {overall_results['average_accuracy']:.2f}%")
        logger.info(f"⚡ Average Throughput: {overall_results['average_throughput']:.1f} ops/sec")
        logger.info(f"🏆 Overall Status: {overall_results['status']}")
        
        # Show recommendations
        if results.get('recommendations'):
            logger.info("\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(results['recommendations'], 1):
                logger.info(f"  {i}. {rec}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Test suite execution failed: {e}")
        raise


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Kimera AI Test Suite - Comprehensive AI Benchmarking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --quick                    # Run quick test suite (MLPerf + Safety)
  %(prog)s --full                     # Run complete test suite
  %(prog)s --mlperf-only              # Run only MLPerf inference tests
  %(prog)s --safety-only              # Run only safety assessment tests
  %(prog)s --cognitive-only           # Run only Kimera cognitive tests
  %(prog)s --config custom.json       # Use custom configuration file
  %(prog)s --no-gpu                   # Disable GPU optimization
  %(prog)s --dry-run                  # Show execution plan without running tests
        """
    )
    
    # Execution modes
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--quick', action='store_true',
                           help='Run quick test suite (MLPerf + Safety)')
    mode_group.add_argument('--full', action='store_true',
                           help='Run complete test suite with all categories')
    
    # Test category filters
    category_group = parser.add_mutually_exclusive_group()
    category_group.add_argument('--mlperf-only', action='store_true',
                               help='Run only MLPerf inference tests')
    category_group.add_argument('--safety-only', action='store_true',
                               help='Run only safety assessment tests')
    category_group.add_argument('--cognitive-only', action='store_true',
                               help='Run only Kimera cognitive tests')
    category_group.add_argument('--domain-only', action='store_true',
                               help='Run only domain-specific tests')
    category_group.add_argument('--cert-only', action='store_true',
                               help='Run only certification preparation tests')
    
    # Configuration options
    parser.add_argument('--config', type=str,
                       help='Path to configuration JSON file')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory for results')
    parser.add_argument('--max-duration', type=int,
                       help='Maximum test duration in minutes')
    
    # Hardware and optimization options
    parser.add_argument('--no-gpu', action='store_true',
                       help='Disable GPU optimization')
    parser.add_argument('--gpu-validation-level', 
                       choices=['basic', 'standard', 'rigorous', 'zeteic'],
                       help='GPU validation level')
    parser.add_argument('--monitoring-level',
                       choices=['minimal', 'standard', 'detailed', 'extreme'],
                       help='Monitoring detail level')
    
    # Logging and output options
    parser.add_argument('--no-cognitive-monitoring', action='store_true',
                       help='Disable Kimera cognitive monitoring')
    parser.add_argument('--no-detailed-logs', action='store_true',
                       help='Disable detailed logging')
    parser.add_argument('--no-visualizations', action='store_true',
                       help='Disable visualization generation')
    
    # Execution options
    parser.add_argument('--dry-run', action='store_true',
                       help='Show execution plan without running tests')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Load configuration if specified
    config_data = None
    if args.config:
        config_data = load_config(args.config)
    else:
        # Try to load default config
        default_config_path = project_root / "config" / "ai_test_suite_config.json"
        if default_config_path.exists():
            config_data = load_config(str(default_config_path))
    
    # Create test configuration
    config = create_test_config_from_args(args, config_data)
    
    # Print test plan
    print_test_plan(config, config_data)
    
    # Dry run - just show the plan
    if args.dry_run:
        logger.info("🔍 DRY RUN - No tests will be executed")
        return
    
    # Confirm execution for full test suite
    if args.full or (not args.quick and not any([args.mlperf_only, args.safety_only, 
                                                args.cognitive_only, args.domain_only, args.cert_only])):
        response = input("\n❓ Run full test suite? This may take significant time. (y/N): ")
        if response.lower() not in ['y', 'yes']:
            logger.info("❌ Test execution cancelled by user")
            return
    
    # Execute the test suite
    try:
        if args.quick:
            logger.info("🏃 Running QUICK test suite...")
            results = asyncio.run(run_quick_test_suite())
        elif args.cognitive_only:
            logger.info("🧠 Running COGNITIVE tests only...")
            results = asyncio.run(run_kimera_cognitive_tests())
        else:
            logger.info("🔬 Running CUSTOM test suite...")
            results = asyncio.run(run_test_suite(config))
        
        # Show final results location
        if 'test_suite_info' in results:
            logger.info(f"\n📁 Results saved to: {config.output_directory}")
            logger.info(f"📊 Execution completed at: {results['test_suite_info']['execution_time']}")
        
        # Exit with appropriate code
        overall_status = results.get('overall_results', {}).get('status', 'UNKNOWN')
        if overall_status in ['EXCELLENT', 'GOOD']:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("🛑 Test execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"💥 Test execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 