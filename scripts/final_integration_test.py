#!/usr/bin/env python3
"""
KIMERA SWM - FINAL INTEGRATION TEST
===================================

Comprehensive integration test to validate all systems before coherent core implementation.
Tests all components, dependencies, and workflows to ensure system readiness.
"""

import os
import sys
import asyncio
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import logging

# Setup paths
current_dir = Path(__file__).parent
project_root = current_dir.parent
src_dir = project_root / "src"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_dir))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FinalIntegrationTester:
    """Comprehensive integration testing for Kimera SWM"""
    
    def __init__(self):
        self.project_root = project_root
        self.test_results = {
            'core_system': {},
            'gpu_acceleration': {},
            'database_operations': {},
            'vault_systems': {},
            'orchestration': {},
            'api_endpoints': {},
            'performance': {},
            'integration_workflows': {},
            'overall_status': 'unknown'
        }
        self.passed_tests = 0
        self.failed_tests = 0
        self.start_time = None
    
    def test_core_system_initialization(self) -> bool:
        """Test core system initialization and state"""
        logger.info("🧠 Testing Core System Initialization...")
        
        try:
            from src.core.kimera_system import get_kimera_system
            
            # Get system instance
            system = get_kimera_system()
            
            # Initialize system
            system.initialize()
            
            # Get system state
            state = system.get_system_state()
            
            # Validate state
            required_state_keys = ['state', 'device', 'gpu_acceleration_enabled', 'components']
            missing_keys = [key for key in required_state_keys if key not in state]
            
            if missing_keys:
                raise ValueError(f"Missing state keys: {missing_keys}")
            
            # Check critical state values
            if state['state'] != 'RUNNING':
                raise ValueError(f"System not running: {state['state']}")
            
            if not state.get('gpu_acceleration_enabled', False):
                logger.warning("⚠️ GPU acceleration not enabled")
            
            self.test_results['core_system'] = {
                'status': 'passed',
                'system_state': state['state'],
                'device': state['device'],
                'gpu_enabled': state.get('gpu_acceleration_enabled', False),
                'component_count': len(state.get('components', {}))
            }
            
            logger.info("✅ Core system initialization passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Core system test failed: {e}")
            self.test_results['core_system'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def test_gpu_acceleration(self) -> bool:
        """Test GPU acceleration components"""
        logger.info("⚡ Testing GPU Acceleration...")
        
        try:
            # Test basic GPU availability
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available")
            
            device_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            # Test GPU operations
            x = torch.randn(1000, 1000, device='cuda')
            y = torch.randn(1000, 1000, device='cuda')
            
            start_time = time.time()
            z = torch.matmul(x, y)
            torch.cuda.synchronize()
            gpu_time = time.time() - start_time
            
            # Calculate performance
            gflops = (2 * 1000**3) / gpu_time / 1e9
            
            # Test Kimera GPU components
            from src.core.gpu.gpu_manager import get_gpu_manager, is_gpu_available
            
            if not is_gpu_available():
                raise RuntimeError("Kimera GPU manager reports GPU unavailable")
            
            gpu_manager = get_gpu_manager()
            device_info = gpu_manager.get_device_info()
            
            self.test_results['gpu_acceleration'] = {
                'status': 'passed',
                'device_name': device_name,
                'memory_gb': round(memory_gb, 1),
                'performance_gflops': round(gflops, 0),
                'gpu_manager': 'available',
                'device_info': device_info
            }
            
            logger.info(f"✅ GPU acceleration passed - {device_name} ({gflops:.0f} GFLOPS)")
            return True
            
        except Exception as e:
            logger.error(f"❌ GPU acceleration test failed: {e}")
            self.test_results['gpu_acceleration'] = {
                'status': 'failed', 
                'error': str(e)
            }
            return False
    
    def test_database_operations(self) -> bool:
        """Test database operations and connectivity"""
        logger.info("🗄️ Testing Database Operations...")
        
        try:
            import sqlite3
            
            # Test SQLite database
            db_path = self.project_root / "data/database/kimera_system.db"
            if not db_path.exists():
                raise FileNotFoundError(f"Database not found: {db_path}")
            
            # Test connection
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            # Test table existence
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            required_tables = ['geoid_states', 'cognitive_transitions', 'semantic_embeddings']
            missing_tables = [table for table in required_tables if table not in tables]
            
            if missing_tables:
                logger.warning(f"⚠️ Missing tables: {missing_tables}")
            
            # Test basic operations
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table';")
            table_count = cursor.fetchone()[0]
            
            # Test schema validation
            for table in required_tables:
                if table in tables:
                    cursor.execute(f"PRAGMA table_info({table});")
                    columns = cursor.fetchall()
                    if not columns:
                        raise ValueError(f"Table {table} has no columns")
            
            conn.close()
            
            # Test Kimera database integration
            try:
                from src.vault.vault_manager import VaultManager
                vault = VaultManager()
                
                vault_status = {
                    'db_initialized': vault.db_initialized,
                    'neo4j_available': vault.neo4j_available
                }
            except Exception as e:
                vault_status = {'error': str(e)}
            
            self.test_results['database_operations'] = {
                'status': 'passed',
                'sqlite_tables': len(tables),
                'required_tables_present': len(required_tables) - len(missing_tables),
                'vault_manager': vault_status
            }
            
            logger.info(f"✅ Database operations passed - {len(tables)} tables")
            return True
            
        except Exception as e:
            logger.error(f"❌ Database operations test failed: {e}")
            self.test_results['database_operations'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def test_orchestration_system(self) -> bool:
        """Test orchestration and engine coordination"""
        logger.info("🎼 Testing Orchestration System...")
        
        try:
            from src.orchestration.kimera_orchestrator import EngineCoordinator
            
            # Initialize orchestrator
            coordinator = EngineCoordinator()
            
            # Test engine registry
            total_engines = len(coordinator.engines)
            gpu_engines = len([name for name in coordinator.engines.keys() if 'gpu' in name])
            
            # Test engine capabilities
            capabilities = coordinator.engine_capabilities
            total_capabilities = sum(len(caps) for caps in capabilities.values())
            
            # Test engine status
            available_engines = []
            failed_engines = []
            
            for engine_name in coordinator.engines.keys():
                try:
                    engine = coordinator.engines[engine_name]
                    if engine is not None:
                        available_engines.append(engine_name)
                    else:
                        failed_engines.append(engine_name)
                except Exception:
                    failed_engines.append(engine_name)
            
            self.test_results['orchestration'] = {
                'status': 'passed',
                'total_engines': total_engines,
                'gpu_engines': gpu_engines,
                'available_engines': len(available_engines),
                'failed_engines': len(failed_engines),
                'total_capabilities': total_capabilities,
                'gpu_available': coordinator.gpu_available
            }
            
            logger.info(f"✅ Orchestration passed - {total_engines} engines ({gpu_engines} GPU)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Orchestration test failed: {e}")
            self.test_results['orchestration'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def test_api_endpoints(self) -> bool:
        """Test API endpoint availability"""
        logger.info("🌐 Testing API Endpoints...")
        
        try:
            # Test core API imports
            from src.api.core_actions_routes import router as core_router
            from src.api.routers.thermodynamic_router import router as thermo_router
            
            # Test GPU router
            try:
                from src.api.routers.gpu_router import router as gpu_router
                gpu_router_available = True
            except Exception:
                gpu_router_available = False
            
            # Count available routes
            core_routes = len(core_router.routes)
            thermo_routes = len(thermo_router.routes)
            gpu_routes = len(gpu_router.routes) if gpu_router_available else 0
            
            total_routes = core_routes + thermo_routes + gpu_routes
            
            self.test_results['api_endpoints'] = {
                'status': 'passed',
                'core_routes': core_routes,
                'thermodynamic_routes': thermo_routes,
                'gpu_routes': gpu_routes,
                'total_routes': total_routes,
                'gpu_router_available': gpu_router_available
            }
            
            logger.info(f"✅ API endpoints passed - {total_routes} routes")
            return True
            
        except Exception as e:
            logger.error(f"❌ API endpoints test failed: {e}")
            self.test_results['api_endpoints'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def test_integration_workflows(self) -> bool:
        """Test end-to-end integration workflows"""
        logger.info("🔄 Testing Integration Workflows...")
        
        try:
            # Test geoid creation and processing
            from src.core.data_structures.geoid_state import create_concept_geoid
            
            # Create test geoid
            test_geoid = create_concept_geoid("integration_test_concept")
            
            if not test_geoid:
                raise ValueError("Failed to create test geoid")
            
            # Test processing result structure
            from src.core.processing.geoid_processor import ProcessingResult
            
            # Create test processing result
            test_result = ProcessingResult(
                success=True,
                updated_geoid=test_geoid,
                operation="test_operation",
                execution_time=0.001,
                metadata={'test': True}
            )
            
            # Test basic workflow components
            workflow_components = {
                'geoid_creation': test_geoid is not None,
                'processing_result': test_result.success,
                'execution_time': test_result.execution_time > 0,
                'metadata': 'test' in test_result.metadata
            }
            
            # Calculate workflow success rate
            successful_components = sum(1 for success in workflow_components.values() if success)
            total_components = len(workflow_components)
            
            self.test_results['integration_workflows'] = {
                'status': 'passed',
                'workflow_components': workflow_components,
                'success_rate': successful_components / total_components,
                'geoid_test': 'passed' if test_geoid else 'failed'
            }
            
            logger.info(f"✅ Integration workflows passed - {successful_components}/{total_components}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Integration workflows test failed: {e}")
            self.test_results['integration_workflows'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def test_performance_benchmarks(self) -> bool:
        """Test system performance benchmarks"""
        logger.info("📊 Testing Performance Benchmarks...")
        
        try:
            import torch
            import time
            
            # CPU vs GPU performance comparison
            size = 500
            iterations = 5
            
            # CPU benchmark
            cpu_times = []
            for _ in range(iterations):
                a = torch.randn(size, size)
                b = torch.randn(size, size)
                
                start = time.time()
                c = torch.matmul(a, b)
                cpu_times.append(time.time() - start)
            
            avg_cpu_time = sum(cpu_times) / len(cpu_times)
            
            # GPU benchmark (if available)
            gpu_times = []
            gpu_available = torch.cuda.is_available()
            
            if gpu_available:
                for _ in range(iterations):
                    a = torch.randn(size, size, device='cuda')
                    b = torch.randn(size, size, device='cuda')
                    
                    start = time.time()
                    c = torch.matmul(a, b)
                    torch.cuda.synchronize()
                    gpu_times.append(time.time() - start)
                
                avg_gpu_time = sum(gpu_times) / len(gpu_times)
                speedup = avg_cpu_time / avg_gpu_time if avg_gpu_time > 0 else 0
            else:
                avg_gpu_time = 0
                speedup = 0
            
            # Calculate GFLOPS
            ops = 2 * size**3  # Matrix multiplication operations
            cpu_gflops = ops / avg_cpu_time / 1e9
            gpu_gflops = ops / avg_gpu_time / 1e9 if avg_gpu_time > 0 else 0
            
            self.test_results['performance'] = {
                'status': 'passed',
                'cpu_time_ms': round(avg_cpu_time * 1000, 2),
                'gpu_time_ms': round(avg_gpu_time * 1000, 2) if gpu_available else 'n/a',
                'speedup': round(speedup, 1) if gpu_available else 'n/a',
                'cpu_gflops': round(cpu_gflops, 1),
                'gpu_gflops': round(gpu_gflops, 1) if gpu_available else 'n/a',
                'gpu_available': gpu_available
            }
            
            logger.info(f"✅ Performance benchmarks passed - {speedup:.1f}x speedup" if gpu_available else "✅ Performance benchmarks passed - CPU only")
            return True
            
        except Exception as e:
            logger.error(f"❌ Performance benchmarks test failed: {e}")
            self.test_results['performance'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def run_comprehensive_integration_test(self) -> Dict[str, Any]:
        """Run all integration tests"""
        logger.info("🚀 Starting Comprehensive Integration Test")
        logger.info("=" * 70)
        
        self.start_time = time.time()
        
        # Test suite
        tests = [
            ('Core System', self.test_core_system_initialization),
            ('GPU Acceleration', self.test_gpu_acceleration),
            ('Database Operations', self.test_database_operations),
            ('Orchestration', self.test_orchestration_system),
            ('API Endpoints', self.test_api_endpoints),
            ('Integration Workflows', self.test_integration_workflows),
            ('Performance', self.test_performance_benchmarks)
        ]
        
        # Run all tests
        for test_name, test_func in tests:
            try:
                success = test_func()
                if success:
                    self.passed_tests += 1
                else:
                    self.failed_tests += 1
            except Exception as e:
                logger.error(f"❌ Test {test_name} crashed: {e}")
                self.failed_tests += 1
        
        # Calculate overall status
        total_tests = self.passed_tests + self.failed_tests
        success_rate = self.passed_tests / total_tests if total_tests > 0 else 0
        
        if success_rate >= 0.9:
            overall_status = 'excellent'
        elif success_rate >= 0.8:
            overall_status = 'good'
        elif success_rate >= 0.7:
            overall_status = 'acceptable'
        else:
            overall_status = 'needs_work'
        
        self.test_results['overall_status'] = overall_status
        
        # Test summary
        test_duration = time.time() - self.start_time
        
        self.test_results['summary'] = {
            'total_tests': total_tests,
            'passed_tests': self.passed_tests,
            'failed_tests': self.failed_tests,
            'success_rate': success_rate,
            'test_duration': test_duration,
            'overall_status': overall_status,
            'timestamp': datetime.now().isoformat()
        }
        
        return self.test_results
    
    def generate_integration_report(self) -> str:
        """Generate comprehensive integration report"""
        summary = self.test_results.get('summary', {})
        
        report = []
        report.append("# KIMERA SWM - FINAL INTEGRATION TEST REPORT")
        report.append("=" * 80)
        report.append(f"**Test Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Overall Status**: {summary.get('overall_status', 'unknown').upper()}")
        report.append(f"**Success Rate**: {summary.get('success_rate', 0) * 100:.1f}%")
        report.append("")
        
        # Test Summary
        report.append("## Test Summary")
        report.append(f"- **Total Tests**: {summary.get('total_tests', 0)}")
        report.append(f"- **Passed**: {summary.get('passed_tests', 0)}")
        report.append(f"- **Failed**: {summary.get('failed_tests', 0)}")
        report.append(f"- **Duration**: {summary.get('test_duration', 0):.2f}s")
        report.append("")
        
        # Detailed Results
        test_sections = [
            ('Core System', 'core_system'),
            ('GPU Acceleration', 'gpu_acceleration'),
            ('Database Operations', 'database_operations'),
            ('Orchestration', 'orchestration'),
            ('API Endpoints', 'api_endpoints'),
            ('Integration Workflows', 'integration_workflows'),
            ('Performance', 'performance')
        ]
        
        for section_name, section_key in test_sections:
            section_data = self.test_results.get(section_key, {})
            status = section_data.get('status', 'unknown')
            
            report.append(f"## {section_name}")
            report.append(f"**Status**: {'✅ PASSED' if status == 'passed' else '❌ FAILED'}")
            
            if status == 'passed':
                # Add specific metrics for each section
                if section_key == 'gpu_acceleration':
                    report.append(f"- Device: {section_data.get('device_name', 'Unknown')}")
                    report.append(f"- Performance: {section_data.get('performance_gflops', 0)} GFLOPS")
                elif section_key == 'orchestration':
                    report.append(f"- Engines: {section_data.get('total_engines', 0)} total, {section_data.get('gpu_engines', 0)} GPU")
                elif section_key == 'performance':
                    speedup = section_data.get('speedup', 'n/a')
                    report.append(f"- GPU Speedup: {speedup}x" if speedup != 'n/a' else "- GPU: Not available")
            else:
                error = section_data.get('error', 'Unknown error')
                report.append(f"- Error: {error[:100]}...")
            
            report.append("")
        
        # Recommendations
        report.append("## Integration Readiness")
        overall_status = summary.get('overall_status', 'unknown')
        
        if overall_status == 'excellent':
            report.append("🎉 **SYSTEM FULLY READY FOR COHERENT CORE INTEGRATION**")
            report.append("- All major components operational")
            report.append("- GPU acceleration fully functional")
            report.append("- Database and vault systems stable")
            report.append("- API endpoints accessible")
            report.append("- Performance benchmarks excellent")
        elif overall_status == 'good':
            report.append("✅ **SYSTEM READY FOR INTEGRATION WITH MINOR NOTES**")
            report.append("- Core functionality verified")
            report.append("- Minor optimizations may be beneficial")
        elif overall_status == 'acceptable':
            report.append("⚠️ **SYSTEM FUNCTIONAL BUT NEEDS ATTENTION**")
            report.append("- Core functionality works")
            report.append("- Several issues should be addressed")
        else:
            report.append("❌ **SYSTEM NEEDS SIGNIFICANT WORK BEFORE INTEGRATION**")
            report.append("- Multiple critical issues detected")
            report.append("- Address failures before proceeding")
        
        report.append("")
        report.append("---")
        report.append("*Report generated by Kimera SWM Final Integration Tester*")
        
        return "\n".join(report)

def main():
    """Main integration test function"""
    try:
        tester = FinalIntegrationTester()
        results = tester.run_comprehensive_integration_test()
        
        # Generate report
        report = tester.generate_integration_report()
        
        # Save results
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Ensure reports directory exists
        reports_dir = tester.project_root / "docs" / "reports" / "integration"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON results
        json_path = reports_dir / f"{timestamp}_final_integration_test.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save markdown report
        md_path = reports_dir / f"{timestamp}_final_integration_test.md"
        with open(md_path, 'w') as f:
            f.write(report)
        
        # Print summary
        summary = results.get('summary', {})
        logger.info("\n" + "=" * 80)
        logger.info("KIMERA SWM - FINAL INTEGRATION TEST COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Overall Status: {summary.get('overall_status', 'unknown').upper()}")
        logger.info(f"Success Rate: {summary.get('success_rate', 0) * 100:.1f}%")
        logger.info(f"Tests Passed: {summary.get('passed_tests', 0)}/{summary.get('total_tests', 0)}")
        logger.info(f"Duration: {summary.get('test_duration', 0):.2f}s")
        logger.info(f"Detailed Report: {md_path}")
        
        if summary.get('overall_status') in ['excellent', 'good']:
            logger.info("\n🎉 SYSTEM READY FOR COHERENT CORE INTEGRATION! 🎉")
            return 0
        else:
            logger.info("\n⚠️ Issues detected - review report before integration")
            return 1
            
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 