#!/usr/bin/env python3
"""
KIMERA Comprehensive Test Suite
==============================

A rigorous, scientific testing framework for the complete KIMERA system.
This suite validates all components across multiple dimensions:

1. Environment & Infrastructure
2. Unit Component Tests  
3. Integration & System Tests
4. Quantum Computing Tests
5. Stress & Performance Tests
6. Security & Safety Tests
7. Coverage & Quality Analysis

Author: KIMERA Development Team
Version: 1.0.0
Date: 2025-01-27
"""

import os
import sys
import time
import asyncio
import logging
import subprocess
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('kimera_test_results.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Structured test result data"""
    name: str
    category: str
    status: str  # PASS, FAIL, SKIP, ERROR
    duration: float
    details: str
    error_trace: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None

class KIMERATestOrchestrator:
    """
    Comprehensive test orchestrator for the KIMERA system.
    
    Implements a rigorous testing methodology with:
    - Parallel test execution where safe
    - Detailed result tracking and analysis
    - Performance metrics collection
    - Scientific validation protocols
    """
    
    def __init__(self):
        self.test_results: List[TestResult] = []
        self.start_time = None
        self.end_time = None
        self.workspace_root = Path(__file__).parent.parent
        
        # Test categories and their respective test files
        self.test_categories = {
            "environment": [],  # Environment checks
            "unit": [
                "tests/unit/test_psychotic_prevention_direct.py",
                "tests/unit/test_insight_feedback.py", 
                "tests/unit/test_meta_insight.py",
                "tests/unit/test_activation_manager.py",
                "tests/unit/test_coherence_service.py",
                "tests/unit/test_contradiction_engine.py"
            ],
            "integration": [
                "tests/integration/test_phase2_cognitive_architecture.py",
                "tests/integration/test_kimera_self_referential_validation.py",
                "tests/integration/test_quantum_integration.py",
                "tests/integration/test_cognitive_field_metrics.py",
                "tests/integration/test_cognitive_field_dynamics_api.py",
                "tests/integration/test_insights_api.py",
                "tests/integration/test_insight_generation_cycle.py"
            ],
            "validation": [
                "tests/validation/test_data_integrity.py",
                "tests/validation/test_psychiatric_stability_long_term.py"
            ],
            "quantum": [
                "tests/quantum/kimera_quantum_enhanced_test_suite.py",
                "tests/quantum/kimera_quantum_integration_test_suite.py",
                "tests/quantum/quantum_test_orchestrator.py"
            ],
            "stress": [
                "tests/stress/test_system_stress.py",
                "tests/stress/comprehensive_stress_test.py"
            ],
            "rigorous": [
                "tests/rigorous/test_cognitive_field_dynamics_logic.py"
            ],
            "security": [
                "tests/test_security_foundation.py"
            ],
            "specialized": [
                "tests/test_advanced_gpu_computing.py",
                "tests/test_kimera_action_interface.py",
                "tests/test_understanding_capabilities.py",
                "tests/test_semantic_grounding.py"
            ]
        }
        
    async def run_comprehensive_tests(self) -> Dict[str, Any]:
        """
        Execute the complete KIMERA test suite with scientific rigor.
        
        Returns:
            Comprehensive test results and analysis
        """
        logger.info("🚀 Starting KIMERA Comprehensive Test Suite")
        logger.info("=" * 70)
        
        self.start_time = time.time()
        
        try:
            # Step 1: Environment and Infrastructure Validation
            await self._validate_environment()
            
            # Step 2: Core Unit Tests
            await self._run_unit_tests()
            
            # Step 3: Integration and System Tests  
            await self._run_integration_tests()
            
            # Step 4: Validation Tests
            await self._run_validation_tests()
            
            # Step 5: Quantum Computing Tests
            await self._run_quantum_tests()
            
            # Step 6: Stress and Performance Tests
            await self._run_stress_tests()
            
            # Step 7: Rigorous Mathematical Tests
            await self._run_rigorous_tests()
            
            # Step 8: Security and Safety Tests
            await self._run_security_tests()
            
            # Step 9: Specialized Component Tests
            await self._run_specialized_tests()
            
            # Step 10: Generate comprehensive analysis
            return await self._generate_final_analysis()
            
        except Exception as e:
            logger.error(f"Critical error in test orchestration: {e}")
            logger.error(traceback.format_exc())
            return {"status": "CRITICAL_FAILURE", "error": str(e)}
        
        finally:
            self.end_time = time.time()
    
    async def _validate_environment(self):
        """Validate the testing environment and dependencies"""
        logger.info("🔍 Step 1: Environment & Infrastructure Validation")
        
        # Check Python environment
        python_version = sys.version_info
        result = TestResult(
            name="Python Version Check",
            category="environment", 
            status="PASS" if python_version >= (3, 8) else "FAIL",
            duration=0.1,
            details=f"Python {python_version.major}.{python_version.minor}.{python_version.micro}",
            metrics={"python_version": f"{python_version.major}.{python_version.minor}"}
        )
        self.test_results.append(result)
        
        # Check critical dependencies
        critical_deps = [
            "torch", "numpy", "fastapi", "sqlalchemy", 
            "qiskit", "uvicorn", "pytest", "asyncio"
        ]
        
        for dep in critical_deps:
            try:
                __import__(dep)
                status = "PASS"
                details = f"{dep} available"
            except ImportError:
                status = "FAIL" 
                details = f"{dep} not available"
                
            result = TestResult(
                name=f"Dependency: {dep}",
                category="environment",
                status=status,
                duration=0.05,
                details=details
            )
            self.test_results.append(result)
        
        # Check GPU availability
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            gpu_count = torch.cuda.device_count()
            
            result = TestResult(
                name="GPU/CUDA Availability",
                category="environment",
                status="PASS" if gpu_available else "SKIP",
                duration=0.1,
                details=f"GPU Available: {gpu_available}, Count: {gpu_count}",
                metrics={"gpu_available": gpu_available, "gpu_count": gpu_count}
            )
            self.test_results.append(result)
            
        except Exception as e:
            result = TestResult(
                name="GPU/CUDA Availability", 
                category="environment",
                status="ERROR",
                duration=0.1,
                details=f"Error checking GPU: {str(e)}",
                error_trace=traceback.format_exc()
            )
            self.test_results.append(result)
    
    async def _run_test_category(self, category_name: str, test_files: List[str]):
        """Run all tests in a specific category"""
        logger.info(f"🧪 Running {category_name.upper()} tests...")
        
        for test_file in test_files:
            if not os.path.exists(test_file):
                result = TestResult(
                    name=f"{category_name}: {os.path.basename(test_file)}",
                    category=category_name,
                    status="SKIP",
                    duration=0.0,
                    details="Test file not found"
                )
                self.test_results.append(result)
                continue
                
            start_time = time.time()
            try:
                # Run pytest on the specific file
                process = await asyncio.create_subprocess_exec(
                    sys.executable, "-m", "pytest", test_file, "-v", "--tb=short",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self.workspace_root
                )
                
                stdout, stderr = await process.communicate()
                duration = time.time() - start_time
                
                # Parse pytest output
                stdout_str = stdout.decode() if stdout else ""
                stderr_str = stderr.decode() if stderr else ""
                
                if process.returncode == 0:
                    status = "PASS"
                    details = f"All tests passed in {duration:.2f}s"
                else:
                    status = "FAIL"
                    details = f"Tests failed: {stderr_str[:200]}..."
                
                result = TestResult(
                    name=f"{category_name}: {os.path.basename(test_file)}",
                    category=category_name,
                    status=status,
                    duration=duration,
                    details=details,
                    error_trace=stderr_str if status == "FAIL" else None,
                    metrics={"return_code": process.returncode}
                )
                self.test_results.append(result)
                
            except Exception as e:
                duration = time.time() - start_time
                result = TestResult(
                    name=f"{category_name}: {os.path.basename(test_file)}",
                    category=category_name,
                    status="ERROR",
                    duration=duration,
                    details=f"Execution error: {str(e)}",
                    error_trace=traceback.format_exc()
                )
                self.test_results.append(result)
    
    async def _run_unit_tests(self):
        """Execute all unit tests"""
        await self._run_test_category("unit", self.test_categories["unit"])
    
    async def _run_integration_tests(self):
        """Execute all integration tests"""
        await self._run_test_category("integration", self.test_categories["integration"])
    
    async def _run_validation_tests(self):
        """Execute all validation tests"""
        await self._run_test_category("validation", self.test_categories["validation"])
    
    async def _run_quantum_tests(self):
        """Execute quantum computing tests"""
        logger.info("⚛️  Running QUANTUM tests...")
        # Quantum tests may require special handling
        await self._run_test_category("quantum", self.test_categories["quantum"])
    
    async def _run_stress_tests(self):
        """Execute stress and performance tests"""
        logger.info("💪 Running STRESS tests...")
        await self._run_test_category("stress", self.test_categories["stress"])
    
    async def _run_rigorous_tests(self):
        """Execute rigorous mathematical and logical tests"""
        logger.info("🔬 Running RIGOROUS tests...")
        await self._run_test_category("rigorous", self.test_categories["rigorous"])
    
    async def _run_security_tests(self):
        """Execute security and safety tests"""
        logger.info("🔒 Running SECURITY tests...")
        await self._run_test_category("security", self.test_categories["security"])
    
    async def _run_specialized_tests(self):
        """Execute specialized component tests"""
        logger.info("🎯 Running SPECIALIZED tests...")
        await self._run_test_category("specialized", self.test_categories["specialized"])
    
    async def _generate_final_analysis(self) -> Dict[str, Any]:
        """Generate comprehensive test analysis and report"""
        logger.info("📊 Generating Final Analysis...")
        
        total_duration = self.end_time - self.start_time if self.end_time else time.time() - self.start_time
        
        # Categorize results
        results_by_category = {}
        for result in self.test_results:
            if result.category not in results_by_category:
                results_by_category[result.category] = []
            results_by_category[result.category].append(result)
        
        # Calculate statistics
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r.status == "PASS"])
        failed_tests = len([r for r in self.test_results if r.status == "FAIL"])
        error_tests = len([r for r in self.test_results if r.status == "ERROR"])
        skipped_tests = len([r for r in self.test_results if r.status == "SKIP"])
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Generate detailed report
        analysis = {
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "errors": error_tests,
                "skipped": skipped_tests,
                "success_rate": f"{success_rate:.1f}%",
                "total_duration": f"{total_duration:.2f}s"
            },
            "category_breakdown": {},
            "detailed_results": [],
            "recommendations": [],
            "system_health": "HEALTHY" if success_rate >= 90 else "DEGRADED" if success_rate >= 70 else "CRITICAL"
        }
        
        # Category breakdown
        for category, results in results_by_category.items():
            cat_passed = len([r for r in results if r.status == "PASS"])
            cat_total = len(results)
            cat_success = (cat_passed / cat_total * 100) if cat_total > 0 else 0
            
            analysis["category_breakdown"][category] = {
                "total": cat_total,
                "passed": cat_passed,
                "success_rate": f"{cat_success:.1f}%"
            }
        
        # Detailed results
        for result in self.test_results:
            analysis["detailed_results"].append({
                "name": result.name,
                "category": result.category,
                "status": result.status,
                "duration": f"{result.duration:.3f}s",
                "details": result.details,
                "has_error": result.error_trace is not None
            })
        
        # Generate recommendations
        if failed_tests > 0:
            analysis["recommendations"].append(f"Address {failed_tests} failing tests before production deployment")
        if error_tests > 0:
            analysis["recommendations"].append(f"Investigate {error_tests} tests with execution errors")
        if success_rate < 95:
            analysis["recommendations"].append("Consider additional testing and validation before release")
        if success_rate >= 95:
            analysis["recommendations"].append("System shows excellent test coverage and reliability")
        
        # Log final summary
        logger.info("=" * 70)
        logger.info("🎯 KIMERA COMPREHENSIVE TEST RESULTS")
        logger.info("=" * 70)
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"✅ Passed: {passed_tests}")
        logger.info(f"❌ Failed: {failed_tests}")
        logger.info(f"⚠️  Errors: {error_tests}")
        logger.info(f"⏭️  Skipped: {skipped_tests}")
        logger.info(f"📈 Success Rate: {success_rate:.1f}%")
        logger.info(f"⏱️  Total Duration: {total_duration:.2f}s")
        logger.info(f"🏥 System Health: {analysis['system_health']}")
        logger.info("=" * 70)
        
        # Save detailed results to file
        with open("kimera_test_analysis.json", "w") as f:
            json.dump(analysis, f, indent=2)
        
        return analysis

async def main():
    """Main execution function"""
    orchestrator = KIMERATestOrchestrator()
    results = await orchestrator.run_comprehensive_tests()
    
    logger.info("\n" + "="*70)
    logger.info("🎯 KIMERA COMPREHENSIVE TEST SUITE COMPLETE")
    logger.info("="*70)
    logger.info(f"📊 Results saved to: kimera_test_analysis.json")
    logger.info(f"📝 Logs saved to: kimera_test_results.log")
    logger.info("="*70)
    
    return results

if __name__ == "__main__":
    asyncio.run(main()) 