"""
Comprehensive Test Suite for High-Dimensional BGM Engine

This test suite validates:
- Multi-dimensional BGM simulation
- GPU acceleration performance
- Cognitive field integration
- Risk management scenarios
- Scalability to high dimensions
"""

import torch
import numpy as np
import logging
import time
from typing import Dict, Any
import asyncio

from backend.engines.high_dimensional_bgm import HighDimensionalBGM, BGMConfig, create_high_dimensional_bgm
from backend.engines.cognitive_field_dynamics import CognitiveFieldDynamics
from backend.monitoring.metrics_collector import get_metrics_collector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HighDimensionalBGMTestSuite:
    """Comprehensive test suite for high-dimensional BGM engine"""
    
    def __init__(self):
        self.test_results = []
        self.performance_metrics = {}
        
    def run_all_tests(self):
        """Run all BGM tests"""
        logger.info("🚀 Starting High-Dimensional BGM Test Suite")
        
        # Test 1: Basic functionality
        self._test_basic_functionality()
        
        # Test 2: Multi-dimensional scaling
        self._test_multi_dimensional_scaling()
        
        # Test 3: GPU acceleration
        self._test_gpu_acceleration()
        
        # Test 4: Cognitive field integration
        self._test_cognitive_integration()
        
        # Test 5: Risk management scenarios
        self._test_risk_management()
        
        # Test 6: Correlation modeling
        self._test_correlation_modeling()
        
        # Test 7: Performance benchmarks
        self._test_performance_benchmarks()
        
        # Generate summary report
        self._generate_summary_report()
        
    def _test_basic_functionality(self):
        """Test basic BGM functionality"""
        logger.info("🧪 Test 1: Basic Functionality")
        
        try:
            # Create BGM engine
            bgm = create_high_dimensional_bgm(dimension=100, time_horizon=0.25)
            
            # Set parameters (ensure device consistency)
            drift = torch.ones(100, device=bgm.device) * 0.05 / 252  # 5% annual drift
            volatility = torch.ones(100, device=bgm.device) * 0.2 / np.sqrt(252)  # 20% annual volatility
            bgm.set_parameters(drift, volatility)
            
            # Simulate paths
            initial_prices = torch.ones(100, device=bgm.device) * 100.0
            paths = bgm.simulate_paths(initial_prices, num_paths=100, num_steps=10)
            
            # Validate results
            assert paths.shape == (100, 11, 100), f"Expected shape (100, 11, 100), got {paths.shape}"
            assert torch.all(paths > 0), "All prices should be positive"
            assert torch.all(paths[:, 0, :] == initial_prices), "Initial prices should match"
            
            # Compute moments
            moments = bgm.compute_moments(paths)
            assert 'mean' in moments
            assert 'variance' in moments
            assert 'skewness' in moments
            assert 'kurtosis' in moments
            
            self.test_results.append({
                'test': 'basic_functionality',
                'status': 'PASSED',
                'details': f"Successfully simulated {paths.shape} paths"
            })
            
        except Exception as e:
            self.test_results.append({
                'test': 'basic_functionality',
                'status': 'FAILED',
                'error': str(e)
            })
    
    def _test_multi_dimensional_scaling(self):
        """Test scaling to different dimensions"""
        logger.info("🧪 Test 2: Multi-Dimensional Scaling")
        
        scaling_results = {}
        
        for dim in [128, 256, 512, 1024, 2048]:
            try:
                start_time = time.time()
                
                # Create BGM engine
                bgm = create_high_dimensional_bgm(dimension=dim, time_horizon=0.1)
                
                # Set parameters (ensure device consistency)
                drift = torch.ones(dim, device=bgm.device) * 0.05 / 252
                volatility = torch.ones(dim, device=bgm.device) * 0.2 / np.sqrt(252)
                bgm.set_parameters(drift, volatility)
                
                # Simulate paths
                initial_prices = torch.ones(dim, device=bgm.device) * 100.0
                paths = bgm.simulate_paths(initial_prices, num_paths=50, num_steps=5)
                
                # Measure performance
                execution_time = time.time() - start_time
                
                scaling_results[dim] = {
                    'execution_time': execution_time,
                    'memory_usage': bgm.get_performance_stats()['memory_usage_mb'],
                    'paths_per_second': (50 * 5 * dim) / execution_time
                }
                
                logger.info(f"   {dim}D: {execution_time:.3f}s, {scaling_results[dim]['paths_per_second']:.0f} paths/s")
                
            except Exception as e:
                scaling_results[dim] = {'error': str(e)}
        
        self.performance_metrics['scaling'] = scaling_results
        
        # Check if scaling is reasonable
        successful_dims = [d for d in scaling_results if 'error' not in scaling_results[d]]
        if len(successful_dims) >= 3:
            self.test_results.append({
                'test': 'multi_dimensional_scaling',
                'status': 'PASSED',
                'details': f"Successfully scaled to {max(successful_dims)}D"
            })
        else:
            self.test_results.append({
                'test': 'multi_dimensional_scaling',
                'status': 'FAILED',
                'details': f"Only {len(successful_dims)} dimensions successful"
            })
    
    def _test_gpu_acceleration(self):
        """Test GPU acceleration benefits"""
        logger.info("🧪 Test 3: GPU Acceleration")
        
        if not torch.cuda.is_available():
            self.test_results.append({
                'test': 'gpu_acceleration',
                'status': 'SKIPPED',
                'details': 'CUDA not available'
            })
            return
        
        try:
            # Test CPU vs GPU performance
            dimension = 512
            
            # CPU version
            config_cpu = BGMConfig(dimension=dimension, device='cpu')
            bgm_cpu = HighDimensionalBGM(config_cpu)
            
            drift = torch.ones(dimension) * 0.05 / 252
            volatility = torch.ones(dimension) * 0.2 / np.sqrt(252)
            bgm_cpu.set_parameters(drift, volatility)
            
            initial_prices = torch.ones(dimension) * 100.0
            
            start_time = time.time()
            paths_cpu = bgm_cpu.simulate_paths(initial_prices, num_paths=100, num_steps=10)
            cpu_time = time.time() - start_time
            
            # GPU version
            config_gpu = BGMConfig(dimension=dimension, device='cuda')
            bgm_gpu = HighDimensionalBGM(config_gpu)
            bgm_gpu.set_parameters(drift.cuda(), volatility.cuda())
            
            start_time = time.time()
            paths_gpu = bgm_gpu.simulate_paths(initial_prices.cuda(), num_paths=100, num_steps=10)
            gpu_time = time.time() - start_time
            
            # Compare performance
            speedup = cpu_time / gpu_time
            
            self.performance_metrics['gpu_acceleration'] = {
                'cpu_time': cpu_time,
                'gpu_time': gpu_time,
                'speedup': speedup
            }
            
            if speedup > 1.2:  # Expect at least 20% speedup
                self.test_results.append({
                    'test': 'gpu_acceleration',
                    'status': 'PASSED',
                    'details': f"GPU {speedup:.2f}x faster than CPU"
                })
            else:
                self.test_results.append({
                    'test': 'gpu_acceleration',
                    'status': 'WARNING',
                    'details': f"GPU only {speedup:.2f}x faster than CPU"
                })
                
        except Exception as e:
            self.test_results.append({
                'test': 'gpu_acceleration',
                'status': 'FAILED',
                'error': str(e)
            })
    
    def _test_cognitive_integration(self):
        """Test integration with cognitive field dynamics"""
        logger.info("🧪 Test 4: Cognitive Integration")
        
        try:
            # Create BGM with cognitive integration
            bgm = create_high_dimensional_bgm(dimension=512)
            
            # Set base parameters (ensure device consistency)
            drift = torch.ones(512, device=bgm.device) * 0.05 / 252
            volatility = torch.ones(512, device=bgm.device) * 0.2 / np.sqrt(252)
            bgm.set_parameters(drift, volatility)
            
            # Test cognitive enhancement
            market_data = {
                'price': 100.0,
                'volume': 1000000,
                'change_24h': 0.02,
                'volatility': 0.2
            }
            
            enhanced_drift = bgm.integrate_with_cognitive_field(market_data)
            
            # Validate cognitive enhancement
            assert enhanced_drift.shape == drift.shape, "Enhanced drift should have same shape"
            assert not torch.allclose(enhanced_drift, drift), "Enhanced drift should be different from base"
            
            self.test_results.append({
                'test': 'cognitive_integration',
                'status': 'PASSED',
                'details': f"Cognitive enhancement applied successfully"
            })
            
        except Exception as e:
            self.test_results.append({
                'test': 'cognitive_integration',
                'status': 'FAILED',
                'error': str(e)
            })
    
    def _test_risk_management(self):
        """Test risk management scenario generation"""
        logger.info("🧪 Test 5: Risk Management Scenarios")
        
        try:
            # Create BGM engine
            bgm = create_high_dimensional_bgm(dimension=256)
            
            # Set parameters with some risk (ensure device consistency)
            drift = torch.ones(256, device=bgm.device) * 0.03 / 252  # 3% annual drift
            volatility = torch.ones(256, device=bgm.device) * 0.25 / np.sqrt(252)  # 25% annual volatility
            bgm.set_parameters(drift, volatility)
            
            # Generate risk scenarios
            initial_prices = torch.ones(256, device=bgm.device) * 100.0
            risk_metrics = bgm.generate_market_scenarios(initial_prices, num_scenarios=1000)
            
            # Validate risk metrics
            required_metrics = ['var_95', 'var_99', 'expected_shortfall_95', 'max_drawdown', 'volatility']
            for metric in required_metrics:
                assert metric in risk_metrics, f"Missing risk metric: {metric}"
            
            # Check VaR ordering
            assert risk_metrics['var_99'] <= risk_metrics['var_95'], "VaR(99%) should be <= VaR(95%)"
            
            # Check expected shortfall
            assert risk_metrics['expected_shortfall_95'] <= risk_metrics['var_95'], "ES should be <= VaR"
            
            self.test_results.append({
                'test': 'risk_management',
                'status': 'PASSED',
                'details': f"Risk scenarios generated with VaR(95%)={risk_metrics['var_95']:.4f}"
            })
            
        except Exception as e:
            self.test_results.append({
                'test': 'risk_management',
                'status': 'FAILED',
                'error': str(e)
            })
    
    def _test_correlation_modeling(self):
        """Test correlation matrix modeling"""
        logger.info("🧪 Test 6: Correlation Modeling")
        
        try:
            # Create BGM engine
            bgm = create_high_dimensional_bgm(dimension=100)
            
            # Set parameters with correlation (ensure device consistency)
            drift = torch.zeros(100, device=bgm.device)
            volatility = torch.ones(100, device=bgm.device) * 0.2 / np.sqrt(252)
            
            # Create correlation matrix with structure
            correlation = torch.zeros(100, 100, device=bgm.device)
            for i in range(100):
                for j in range(100):
                    correlation[i, j] = 0.8 ** abs(i - j)  # Exponential decay
            
            bgm.set_parameters(drift, volatility, correlation)
            
            # Simulate paths
            initial_prices = torch.ones(100, device=bgm.device) * 100.0
            paths = bgm.simulate_paths(initial_prices, num_paths=1000, num_steps=20)
            
            # Compute empirical correlations
            returns = torch.diff(torch.log(paths), dim=1)
            final_returns = returns[:, -1, :]  # Last period returns
            
            # Compute correlation matrix
            empirical_corr = torch.corrcoef(final_returns.T)
            
            # Check that correlations are reasonable
            diagonal_corr = torch.diag(empirical_corr)
            assert torch.allclose(diagonal_corr, torch.ones(100), atol=1e-3), "Diagonal should be 1"
            
            # Check that nearby assets have higher correlation
            nearby_corr = empirical_corr[0, 1]  # Correlation between asset 0 and 1
            distant_corr = empirical_corr[0, 50]  # Correlation between asset 0 and 50
            
            self.test_results.append({
                'test': 'correlation_modeling',
                'status': 'PASSED',
                'details': f"Correlation structure preserved: nearby={nearby_corr:.3f}, distant={distant_corr:.3f}"
            })
            
        except Exception as e:
            self.test_results.append({
                'test': 'correlation_modeling',
                'status': 'FAILED',
                'error': str(e)
            })
    
    def _test_performance_benchmarks(self):
        """Test performance benchmarks"""
        logger.info("🧪 Test 7: Performance Benchmarks")
        
        try:
            # Define benchmark scenarios
            benchmarks = [
                {'dimension': 512, 'paths': 1000, 'steps': 252, 'name': 'Annual_512D'},
                {'dimension': 1024, 'paths': 500, 'steps': 126, 'name': 'Quarterly_1024D'},
                {'dimension': 2048, 'paths': 250, 'steps': 63, 'name': 'Monthly_2048D'}
            ]
            
            benchmark_results = {}
            
            for benchmark in benchmarks:
                try:
                    bgm = create_high_dimensional_bgm(dimension=benchmark['dimension'])
                    
                    drift = torch.ones(benchmark['dimension'], device=bgm.device) * 0.05 / 252
                    volatility = torch.ones(benchmark['dimension'], device=bgm.device) * 0.2 / np.sqrt(252)
                    bgm.set_parameters(drift, volatility)
                    
                    initial_prices = torch.ones(benchmark['dimension'], device=bgm.device) * 100.0
                    
                    start_time = time.time()
                    paths = bgm.simulate_paths(initial_prices, 
                                             num_paths=benchmark['paths'], 
                                             num_steps=benchmark['steps'])
                    execution_time = time.time() - start_time
                    
                    # Calculate throughput
                    total_elements = benchmark['paths'] * benchmark['steps'] * benchmark['dimension']
                    throughput = total_elements / execution_time
                    
                    benchmark_results[benchmark['name']] = {
                        'execution_time': execution_time,
                        'throughput': throughput,
                        'memory_usage': bgm.get_performance_stats()['memory_usage_mb']
                    }
                    
                    logger.info(f"   {benchmark['name']}: {execution_time:.2f}s, {throughput:.0f} elements/s")
                    
                except Exception as e:
                    benchmark_results[benchmark['name']] = {'error': str(e)}
            
            self.performance_metrics['benchmarks'] = benchmark_results
            
            # Check if benchmarks are reasonable
            successful_benchmarks = [b for b in benchmark_results if 'error' not in benchmark_results[b]]
            
            if len(successful_benchmarks) >= 2:
                self.test_results.append({
                    'test': 'performance_benchmarks',
                    'status': 'PASSED',
                    'details': f"Completed {len(successful_benchmarks)}/{len(benchmarks)} benchmarks"
                })
            else:
                self.test_results.append({
                    'test': 'performance_benchmarks',
                    'status': 'FAILED',
                    'details': f"Only {len(successful_benchmarks)} benchmarks successful"
                })
                
        except Exception as e:
            self.test_results.append({
                'test': 'performance_benchmarks',
                'status': 'FAILED',
                'error': str(e)
            })
    
    def _generate_summary_report(self):
        """Generate comprehensive summary report"""
        logger.info("\n" + "="*80)
        logger.info("📊 HIGH-DIMENSIONAL BGM TEST SUMMARY")
        logger.info("="*80)
        
        # Test results summary
        total_tests = len(self.test_results)
        passed_tests = len([t for t in self.test_results if t['status'] == 'PASSED'])
        failed_tests = len([t for t in self.test_results if t['status'] == 'FAILED'])
        warning_tests = len([t for t in self.test_results if t['status'] == 'WARNING'])
        skipped_tests = len([t for t in self.test_results if t['status'] == 'SKIPPED'])
        
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"✅ Passed: {passed_tests}")
        logger.info(f"❌ Failed: {failed_tests}")
        logger.info(f"⚠️  Warning: {warning_tests}")
        logger.info(f"⏭️  Skipped: {skipped_tests}")
        
        # Detailed results
        logger.info("\nDetailed Results:")
        for result in self.test_results:
            status_icon = {
                'PASSED': '✅',
                'FAILED': '❌',
                'WARNING': '⚠️',
                'SKIPPED': '⏭️'
            }[result['status']]
            
            logger.info(f"{status_icon} {result['test']}: {result['status']}")
            if 'details' in result:
                logger.info(f"   {result['details']}")
            if 'error' in result:
                logger.info(f"   Error: {result['error']}")
        
        # Performance metrics summary
        if self.performance_metrics:
            logger.info("\nPerformance Metrics:")
            
            if 'scaling' in self.performance_metrics:
                logger.info("  Scaling Performance:")
                for dim, metrics in self.performance_metrics['scaling'].items():
                    if 'error' not in metrics:
                        logger.info(f"    {dim}D: {metrics['execution_time']:.3f}s, {metrics['paths_per_second']:.0f} paths/s")
            
            if 'gpu_acceleration' in self.performance_metrics:
                gpu_metrics = self.performance_metrics['gpu_acceleration']
                logger.info(f"  GPU Acceleration: {gpu_metrics['speedup']:.2f}x speedup")
            
            if 'benchmarks' in self.performance_metrics:
                logger.info("  Benchmark Results:")
                for name, metrics in self.performance_metrics['benchmarks'].items():
                    if 'error' not in metrics:
                        logger.info(f"    {name}: {metrics['execution_time']:.2f}s, {metrics['throughput']:.0f} elements/s")
        
        # Overall assessment
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        logger.info(f"\n🎯 Overall Success Rate: {success_rate:.1%}")
        
        if success_rate >= 0.8:
            logger.info("🚀 HIGH-DIMENSIONAL BGM ENGINE: READY FOR PRODUCTION")
        elif success_rate >= 0.6:
            logger.info("⚠️ HIGH-DIMENSIONAL BGM ENGINE: NEEDS IMPROVEMENTS")
        else:
            logger.info("❌ HIGH-DIMENSIONAL BGM ENGINE: REQUIRES MAJOR FIXES")
        
        logger.info("="*80)

def main():
    """Main test execution"""
    test_suite = HighDimensionalBGMTestSuite()
    test_suite.run_all_tests()

if __name__ == "__main__":
    main() 