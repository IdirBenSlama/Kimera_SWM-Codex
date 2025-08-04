"""
KIMERA SWM Test Runner
======================

Runs all unit tests for the completed integrations with detailed reporting.
"""

import unittest
import sys
import os
from datetime import datetime
import json
from io import StringIO

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class ColoredTextTestResult(unittest.TextTestResult):
    """Custom test result class with colored output"""
    
    COLORS = {
        'green': '\033[92m',
        'red': '\033[91m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'reset': '\033[0m'
    }
    
    def addSuccess(self, test):
        super().addSuccess(test)
        if self.showAll:
            self.stream.writeln(f"{self.COLORS['green']}✓ PASS{self.COLORS['reset']}")
    
    def addError(self, test, err):
        super().addError(test, err)
        if self.showAll:
            self.stream.writeln(f"{self.COLORS['red']}✗ ERROR{self.COLORS['reset']}")
    
    def addFailure(self, test, err):
        super().addFailure(test, err)
        if self.showAll:
            self.stream.writeln(f"{self.COLORS['red']}✗ FAIL{self.COLORS['reset']}")
    
    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        if self.showAll:
            self.stream.writeln(f"{self.COLORS['yellow']}⊘ SKIP: {reason}{self.COLORS['reset']}")


class TestRunner:
    """Main test runner for KIMERA SWM"""
    
    def __init__(self):
        self.test_suites = {
            "Axiomatic Foundation": [
                "tests.core.axiomatic_foundation.test_axiom_mathematical_proof",
                "tests.core.axiomatic_foundation.test_axiom_of_understanding",
                "tests.core.axiomatic_foundation.test_axiom_verification"
            ],
            "Background Services": [
                "tests.core.services.test_background_job_manager",
                "tests.core.services.test_clip_service_integration"
            ]
        }
        
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "suites": {},
            "summary": {
                "total_tests": 0,
                "passed": 0,
                "failed": 0,
                "errors": 0,
                "skipped": 0
            }
        }
    
    def run_suite(self, suite_name, test_modules):
        """Run a test suite"""
        print(f"\n{'='*80}")
        print(f"Running {suite_name} Tests")
        print(f"{'='*80}")
        
        suite_results = {
            "modules": {},
            "total": 0,
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "skipped": 0
        }
        
        for module_name in test_modules:
            print(f"\n📦 Testing {module_name}...")
            
            try:
                # Load test module
                module = __import__(module_name, fromlist=[''])
                
                # Create test suite
                loader = unittest.TestLoader()
                suite = loader.loadTestsFromModule(module)
                
                # Run tests
                stream = StringIO()
                runner = unittest.TextTestRunner(
                    stream=stream,
                    verbosity=2,
                    resultclass=ColoredTextTestResult
                )
                
                result = runner.run(suite)
                
                # Collect results
                module_results = {
                    "tests_run": result.testsRun,
                    "passed": result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped),
                    "failures": len(result.failures),
                    "errors": len(result.errors),
                    "skipped": len(result.skipped),
                    "success": result.wasSuccessful()
                }
                
                # Update totals
                suite_results["total"] += result.testsRun
                suite_results["passed"] += module_results["passed"]
                suite_results["failed"] += len(result.failures)
                suite_results["errors"] += len(result.errors)
                suite_results["skipped"] += len(result.skipped)
                
                suite_results["modules"][module_name] = module_results
                
                # Print summary for module
                print(f"\n  Summary: {module_results['tests_run']} tests")
                print(f"    ✓ Passed: {module_results['passed']}")
                if module_results['failures'] > 0:
                    print(f"    ✗ Failed: {module_results['failures']}")
                if module_results['errors'] > 0:
                    print(f"    ✗ Errors: {module_results['errors']}")
                if module_results['skipped'] > 0:
                    print(f"    ⊘ Skipped: {module_results['skipped']}")
                
            except ImportError as e:
                print(f"  ❌ Failed to import {module_name}: {e}")
                suite_results["errors"] += 1
            except Exception as e:
                print(f"  ❌ Unexpected error in {module_name}: {e}")
                suite_results["errors"] += 1
        
        self.results["suites"][suite_name] = suite_results
        
        # Update global summary
        self.results["summary"]["total_tests"] += suite_results["total"]
        self.results["summary"]["passed"] += suite_results["passed"]
        self.results["summary"]["failed"] += suite_results["failed"]
        self.results["summary"]["errors"] += suite_results["errors"]
        self.results["summary"]["skipped"] += suite_results["skipped"]
    
    def run_all_tests(self):
        """Run all test suites"""
        print("\n" + "="*80)
        print("KIMERA SWM UNIT TEST RUNNER")
        print("="*80)
        print(f"Started at: {self.results['timestamp']}")
        
        # Run each suite
        for suite_name, test_modules in self.test_suites.items():
            self.run_suite(suite_name, test_modules)
        
        # Print final summary
        self.print_summary()
        
        # Save detailed report
        self.save_report()
        
        # Return exit code
        return 0 if self.results["summary"]["failed"] == 0 and self.results["summary"]["errors"] == 0 else 1
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*80)
        print("FINAL TEST SUMMARY")
        print("="*80)
        
        summary = self.results["summary"]
        total = summary["total_tests"]
        passed = summary["passed"]
        failed = summary["failed"]
        errors = summary["errors"]
        skipped = summary["skipped"]
        
        print(f"\nTotal Tests: {total}")
        print(f"  ✓ Passed:  {passed} ({passed/total*100:.1f}%)" if total > 0 else "  ✓ Passed:  0")
        
        if failed > 0:
            print(f"  ✗ Failed:  {failed} ({failed/total*100:.1f}%)")
        
        if errors > 0:
            print(f"  ✗ Errors:  {errors} ({errors/total*100:.1f}%)")
        
        if skipped > 0:
            print(f"  ⊘ Skipped: {skipped} ({skipped/total*100:.1f}%)")
        
        # Overall status
        print("\n" + "-"*40)
        if failed == 0 and errors == 0:
            print("✅ ALL TESTS PASSED!")
        else:
            print("❌ SOME TESTS FAILED!")
        
        # Suite breakdown
        print("\nSuite Results:")
        for suite_name, suite_data in self.results["suites"].items():
            status = "✅" if suite_data["failed"] == 0 and suite_data["errors"] == 0 else "❌"
            print(f"  {status} {suite_name}: {suite_data['passed']}/{suite_data['total']} passed")
    
    def save_report(self):
        """Save detailed test report"""
        report_file = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: {report_file}")


def main():
    """Main entry point"""
    runner = TestRunner()
    
    # Check for specific suite argument
    if len(sys.argv) > 1:
        suite_name = sys.argv[1]
        if suite_name in runner.test_suites:
            runner.run_suite(suite_name, runner.test_suites[suite_name])
            runner.print_summary()
            return 0
        else:
            print(f"Unknown test suite: {suite_name}")
            print(f"Available suites: {', '.join(runner.test_suites.keys())}")
            return 1
    
    # Run all tests
    return runner.run_all_tests()


if __name__ == "__main__":
    sys.exit(main())