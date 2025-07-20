#!/usr/bin/env python3
"""
Scientific Tests Runner for Kimera SWM

This script runs all scientific tests with proper Python path configuration
to ensure the backend modules can be imported correctly.
"""

import sys
import os
import unittest
from pathlib import Path

# Initialize structured logger
from backend.utils.kimera_logger import get_system_logger
logger = get_system_logger(__name__)


# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import test modules
sys.path.insert(0, str(project_root / "scientific"))

def run_all_scientific_tests():
    """Run all scientific tests and report results."""
    
    logger.info("=" * 60)
    logger.info("KIMERA SWM SCIENTIFIC TESTS")
    logger.info("=" * 60)
    
    # Test results tracking
    results = {
        'total_tests': 0,
        'passed_tests': 0,
        'failed_tests': 0,
        'error_tests': 0,
        'test_files': []
    }
    
    # List of test files to run
    test_files = [
        'test_draconic_thermodynamic_analysis.py',
        'test_thermodynamics_foundations.py', 
        'test_thermodynamics_system.py'
    ]
    
    scientific_dir = project_root / "scientific"
    
    for test_file in test_files:
        test_path = scientific_dir / test_file
        
        if not test_path.exists():
            logger.error(f"\n❌ Test file not found: {test_file}")
            continue
            
        logger.info(f"\n🧪 Running {test_file}...")
        logger.info("-" * 40)
        
        try:
            # Load and run the test module
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromName(test_file[:-3])  # Remove .py extension
            
            # Run tests with detailed output
            runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
            result = runner.run(suite)
            
            # Track results
            file_results = {
                'file': test_file,
                'tests_run': result.testsRun,
                'failures': len(result.failures),
                'errors': len(result.errors),
                'success': result.wasSuccessful()
            }
            
            results['test_files'].append(file_results)
            results['total_tests'] += result.testsRun
            results['passed_tests'] += result.testsRun - len(result.failures) - len(result.errors)
            results['failed_tests'] += len(result.failures)
            results['error_tests'] += len(result.errors)
            
            if result.wasSuccessful():
                logger.info(f"✅ {test_file}: All {result.testsRun} tests passed!")
            else:
                logger.error(f"❌ {test_file}: {len(result.failures)
                
        except ImportError as e:
            logger.error(f"❌ Import error in {test_file}: {e}")
            results['test_files'].append({
                'file': test_file,
                'tests_run': 0,
                'failures': 0,
                'errors': 1,
                'success': False,
                'import_error': str(e)
            })
            results['error_tests'] += 1
            
        except Exception as e:
            logger.error(f"❌ Unexpected error in {test_file}: {e}")
            results['test_files'].append({
                'file': test_file,
                'tests_run': 0,
                'failures': 0,
                'errors': 1,
                'success': False,
                'error': str(e)
            })
            results['error_tests'] += 1
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    for file_result in results['test_files']:
        status = "✅ PASS" if file_result['success'] else "❌ FAIL"
        logger.info(f"{status} {file_result['file']}: {file_result['tests_run']} tests")
        
        if not file_result['success']:
            if 'import_error' in file_result:
                logger.error(f"    Import Error: {file_result['import_error']}")
            elif 'error' in file_result:
                logger.error(f"    Error: {file_result['error']}")
            else:
                logger.error(f"    Failures: {file_result['failures']}, Errors: {file_result['errors']}")
    
    logger.info(f"\nOverall Results:")
    logger.info(f"  Total Tests: {results['total_tests']}")
    logger.info(f"  Passed: {results['passed_tests']}")
    logger.error(f"  Failed: {results['failed_tests']}")
    logger.error(f"  Errors: {results['error_tests']}")
    
    success_rate = (results['passed_tests'] / results['total_tests'] * 100) if results['total_tests'] > 0 else 0
    logger.info(f"  Success Rate: {success_rate:.1f}%")
    
    return results


if __name__ == "__main__":
    # Change to scientific directory for test discovery
    os.chdir(project_root / "scientific")
    
    try:
        results = run_all_scientific_tests()
        
        # Exit with appropriate code
        if results['failed_tests'] > 0 or results['error_tests'] > 0:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except KeyboardInterrupt:
        logger.warning("\n\n⚠️  Tests interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.critical(f"\n\n💥 Fatal error running tests: {e}")
        sys.exit(1)