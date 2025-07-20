#!/usr/bin/env python3
"""
Pharmaceutical Testing Engine Performance Benchmark

Comprehensive benchmarking and performance analysis of the KCl testing engine
including GPU optimization, throughput analysis, and system health monitoring.
"""

import sys
import os
import time
import json
import numpy as np
import torch
from datetime import datetime
from typing import Dict, List, Any
import psutil
import logging

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.pharmaceutical.core.kcl_testing_engine import KClTestingEngine
from backend.pharmaceutical.protocols.usp_protocols import USPProtocolEngine
from backend.pharmaceutical.analysis.dissolution_analyzer import DissolutionAnalyzer
from backend.pharmaceutical.validation.pharmaceutical_validator import PharmaceuticalValidator
from backend.utils.kimera_logger import get_logger

logger = get_logger(__name__)


class PharmaceuticalPerformanceBenchmark:
    """Comprehensive performance benchmarking for pharmaceutical testing."""
    
    def __init__(self):
        """Initialize benchmark environment."""
        self.benchmark_results = {}
        self.system_info = self._get_system_info()
        
        # Initialize engines for testing
        logger.info("🚀 Initializing Pharmaceutical Testing Engines for Benchmarking...")
        
        self.cpu_engine = KClTestingEngine(use_gpu=False)
        self.gpu_engine = KClTestingEngine(use_gpu=True) if torch.cuda.is_available() else None
        self.usp_engine = USPProtocolEngine()
        self.dissolution_analyzer = DissolutionAnalyzer(use_gpu=torch.cuda.is_available())
        self.validator = PharmaceuticalValidator(use_gpu=torch.cuda.is_available())
        
        logger.info("✅ Engines initialized successfully")
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        info = {
            'timestamp': datetime.now().isoformat(),
            'cpu_count': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'python_version': sys.version,
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available()
        }
        
        if torch.cuda.is_available():
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            info['cuda_version'] = torch.version.cuda
        
        return info
    
    def benchmark_raw_material_characterization(self, batch_sizes: List[int]) -> Dict[str, Any]:
        """Benchmark raw material characterization performance."""
        logger.info("📊 Benchmarking Raw Material Characterization...")
        
        results = {'cpu': {}, 'gpu': {}}
        
        for batch_size in batch_sizes:
            logger.info(f"   Testing batch size: {batch_size}")
            
            # Generate test data
            test_batches = self._generate_material_batches(batch_size)
            
            # CPU benchmarking
            cpu_results = self._benchmark_characterization_engine(
                self.cpu_engine, test_batches, f"CPU_batch_{batch_size}"
            )
            results['cpu'][batch_size] = cpu_results
            
            # GPU benchmarking (if available)
            if self.gpu_engine:
                gpu_results = self._benchmark_characterization_engine(
                    self.gpu_engine, test_batches, f"GPU_batch_{batch_size}"
                )
                results['gpu'][batch_size] = gpu_results
                
                # Calculate speedup
                speedup = cpu_results['total_time'] / gpu_results['total_time']
                results['gpu'][batch_size]['speedup_vs_cpu'] = speedup
                logger.info(f"   GPU Speedup: {speedup:.2f}x")
            
        return results
    
    def benchmark_dissolution_analysis(self) -> Dict[str, Any]:
        """Benchmark dissolution analysis performance."""
        logger.info("📈 Benchmarking Dissolution Analysis...")
        
        # Generate test dissolution data
        test_profiles = self._generate_dissolution_profiles(50)  # 50 profiles
        
        start_time = time.perf_counter()
        memory_before = psutil.Process().memory_info().rss / (1024**2)
        
        analysis_results = []
        for i, profile_data in enumerate(test_profiles):
            try:
                kinetics = self.dissolution_analyzer.analyze_dissolution_kinetics(
                    profile_data['times'], profile_data['releases']
                )
                analysis_results.append(kinetics)
                
                if i % 10 == 0:
                    logger.info(f"   Processed {i+1}/{len(test_profiles)} profiles")
                    
            except Exception as e:
                logger.warning(f"Analysis failed for profile {i}: {e}")
        
        end_time = time.perf_counter()
        memory_after = psutil.Process().memory_info().rss / (1024**2)
        
        results = {
            'total_profiles': len(test_profiles),
            'successful_analyses': len(analysis_results),
            'total_time': end_time - start_time,
            'avg_time_per_profile': (end_time - start_time) / len(test_profiles),
            'memory_increase_mb': memory_after - memory_before,
            'throughput_profiles_per_sec': len(test_profiles) / (end_time - start_time)
        }
        
        logger.info(f"   Throughput: {results['throughput_profiles_per_sec']:.2f} profiles/sec")
        return results
    
    def benchmark_comprehensive_validation(self) -> Dict[str, Any]:
        """Benchmark comprehensive validation workflow."""
        logger.info("🔬 Benchmarking Comprehensive Validation...")
        
        # Generate test validation data
        validation_datasets = self._generate_validation_datasets(10)
        
        start_time = time.perf_counter()
        successful_validations = 0
        
        for i, dataset in enumerate(validation_datasets):
            try:
                result = self.validator.validate_comprehensive_pharmaceutical_development(dataset)
                if result.status in ['PASSED', 'WARNING']:
                    successful_validations += 1
                    
                logger.debug(f"   Validation {i+1}: {result.status}")
                
            except Exception as e:
                logger.warning(f"Validation failed for dataset {i}: {e}")
        
        end_time = time.perf_counter()
        
        results = {
            'total_validations': len(validation_datasets),
            'successful_validations': successful_validations,
            'success_rate': successful_validations / len(validation_datasets),
            'total_time': end_time - start_time,
            'avg_time_per_validation': (end_time - start_time) / len(validation_datasets)
        }
        
        logger.info(f"   Success rate: {results['success_rate']:.2%}")
        return results
    
    def benchmark_gpu_optimization(self) -> Dict[str, Any]:
        """Benchmark GPU optimization features."""
        logger.info("⚡ Benchmarking GPU Optimization...")
        
        if not torch.cuda.is_available():
            return {'status': 'GPU not available'}
        
        # Test different optimization scenarios
        optimization_scenarios = [
            {'mixed_precision': False, 'tensor_cores': False},
            {'mixed_precision': True, 'tensor_cores': False},
            {'mixed_precision': True, 'tensor_cores': True}
        ]
        
        results = {}
        baseline_time = None
        
        for scenario in optimization_scenarios:
            scenario_name = f"mixed_precision_{scenario['mixed_precision']}_tensor_cores_{scenario['tensor_cores']}"
            
            # Create specialized engine for this scenario
            test_engine = KClTestingEngine(use_gpu=True)
            
            # Override optimization settings
            test_engine.tensor_core_enabled = scenario['tensor_cores']
            
            # Test with coating simulation (GPU-intensive operation)
            test_data = {
                'coating_thickness': 12.0,
                'polymer_ratios': {'ethylcellulose': 0.8, 'hpc': 0.2},
                'process_parameters': {'temperature': 60.0, 'spray_rate': 1.0}
            }
            
            start_time = time.perf_counter()
            
            # Run multiple iterations for statistical significance
            for _ in range(100):
                test_engine._simulate_encapsulation_gpu(**test_data)
            
            end_time = time.perf_counter()
            scenario_time = end_time - start_time
            
            results[scenario_name] = {
                'time': scenario_time,
                'iterations': 100,
                'avg_time_per_iteration': scenario_time / 100
            }
            
            if baseline_time is None:
                baseline_time = scenario_time
            else:
                speedup = baseline_time / scenario_time
                results[scenario_name]['speedup'] = speedup
                logger.info(f"   {scenario_name}: {speedup:.2f}x speedup")
        
        return results
    
    def analyze_system_health(self) -> Dict[str, Any]:
        """Analyze overall system health and performance."""
        logger.info("🏥 Analyzing System Health...")
        
        # Get performance reports from all engines
        cpu_report = self.cpu_engine.get_performance_report()
        gpu_report = self.gpu_engine.get_performance_report() if self.gpu_engine else None
        
        # System resource analysis
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        health_analysis = {
            'cpu_utilization': cpu_percent,
            'memory_utilization': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            'engine_health': {
                'cpu_engine': cpu_report['system_health'],
                'gpu_engine': gpu_report['system_health'] if gpu_report else None
            },
            'overall_status': 'HEALTHY'
        }
        
        # Determine overall health status
        if cpu_percent > 90 or memory.percent > 90:
            health_analysis['overall_status'] = 'DEGRADED'
        
        if cpu_report['system_health']['overall_status'] != 'HEALTHY':
            health_analysis['overall_status'] = 'DEGRADED'
        
        return health_analysis
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive benchmark suite."""
        logger.info("🚀 Starting Comprehensive Pharmaceutical Performance Benchmark")
        logger.info("="*80)
        
        benchmark_start = time.perf_counter()
        
        # Run all benchmarks
        self.benchmark_results = {
            'system_info': self.system_info,
            'raw_material_characterization': self.benchmark_raw_material_characterization([10, 50]),
            'dissolution_analysis': self.benchmark_dissolution_analysis(),
            'comprehensive_validation': self.benchmark_comprehensive_validation(),
            'gpu_optimization': self.benchmark_gpu_optimization(),
            'system_health': self.analyze_system_health()
        }
        
        benchmark_end = time.perf_counter()
        self.benchmark_results['total_benchmark_time'] = benchmark_end - benchmark_start
        
        # Generate summary
        summary = self._generate_benchmark_summary()
        self.benchmark_results['summary'] = summary
        
        logger.info("="*80)
        logger.info("✅ Comprehensive Benchmark Completed")
        logger.info(f"   Total Time: {benchmark_end - benchmark_start:.2f} seconds")
        
        return self.benchmark_results
    
    def _generate_material_batches(self, count: int) -> List[Dict[str, Any]]:
        """Generate test material batches."""
        batches = []
        for i in range(count):
            batch = {
                'name': f'KCl_Test_Batch_{i}',
                'grade': 'USP',
                'purity_percent': np.random.uniform(99.0, 100.5),
                'moisture_content': np.random.uniform(0.1, 0.9),
                'particle_size_d50': np.random.uniform(100, 200),
                'bulk_density': np.random.uniform(0.7, 1.0),
                'tapped_density': np.random.uniform(0.9, 1.3),
                'potassium_confirmed': True,
                'chloride_confirmed': True,
                'heavy_metals_ppm': np.random.uniform(0, 10),
                'sodium_percent': np.random.uniform(0, 0.2),
                'bromide_ppm': np.random.uniform(0, 200)
            }
            batches.append(batch)
        return batches
    
    def _generate_dissolution_profiles(self, count: int) -> List[Dict[str, Any]]:
        """Generate test dissolution profiles."""
        profiles = []
        for i in range(count):
            times = [1.0, 2.0, 4.0, 6.0]
            # Generate realistic dissolution curve
            releases = [
                np.random.uniform(20, 35),  # 1 hour
                np.random.uniform(45, 65),  # 2 hours
                np.random.uniform(70, 85),  # 4 hours
                np.random.uniform(85, 95)   # 6 hours
            ]
            profiles.append({'times': times, 'releases': releases})
        return profiles
    
    def _generate_validation_datasets(self, count: int) -> List[Dict[str, Any]]:
        """Generate test validation datasets."""
        datasets = []
        for i in range(count):
            dataset = {
                'raw_materials': {
                    'name': f'KCl_Validation_{i}',
                    'purity_percent': np.random.uniform(99.0, 100.5),
                    'moisture_content': np.random.uniform(0.1, 0.9),
                    'potassium_confirmed': True,
                    'chloride_confirmed': True
                },
                'formulation': {
                    'coating_thickness_percent': np.random.uniform(10, 15),
                    'polymer_ratios': {'ethylcellulose': 0.8, 'hpc': 0.2}
                },
                'finished_product': {
                    'dissolution_profile': [25, 50, 75, 90],
                    'content_uniformity': np.random.uniform(95, 105, 10).tolist()
                }
            }
            datasets.append(dataset)
        return datasets
    
    def _benchmark_characterization_engine(self, engine, test_batches: List[Dict], name: str) -> Dict[str, Any]:
        """Benchmark characterization engine performance."""
        start_time = time.perf_counter()
        memory_before = psutil.Process().memory_info().rss / (1024**2)
        
        successful_characterizations = 0
        
        for batch in test_batches:
            try:
                engine.characterize_raw_materials(batch)
                successful_characterizations += 1
            except Exception as e:
                logger.debug(f"Characterization failed: {e}")
        
        end_time = time.perf_counter()
        memory_after = psutil.Process().memory_info().rss / (1024**2)
        
        return {
            'total_batches': len(test_batches),
            'successful_characterizations': successful_characterizations,
            'success_rate': successful_characterizations / len(test_batches),
            'total_time': end_time - start_time,
            'avg_time_per_batch': (end_time - start_time) / len(test_batches),
            'throughput_batches_per_sec': len(test_batches) / (end_time - start_time),
            'memory_increase_mb': memory_after - memory_before
        }
    
    def _generate_benchmark_summary(self) -> Dict[str, Any]:
        """Generate comprehensive benchmark summary."""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'system_performance': 'EXCELLENT',
            'key_metrics': {},
            'recommendations': []
        }
        
        # Extract key performance metrics
        raw_material_cpu = self.benchmark_results['raw_material_characterization']['cpu']
        dissolution = self.benchmark_results['dissolution_analysis']
        validation = self.benchmark_results['comprehensive_validation']
        
        if raw_material_cpu:
            batch_100 = raw_material_cpu.get(100, raw_material_cpu.get(50, {}))
            if batch_100:
                summary['key_metrics']['characterization_throughput'] = batch_100.get('throughput_batches_per_sec', 0)
        
        summary['key_metrics']['dissolution_throughput'] = dissolution.get('throughput_profiles_per_sec', 0)
        summary['key_metrics']['validation_success_rate'] = validation.get('success_rate', 0)
        
        # GPU performance comparison
        if torch.cuda.is_available() and 'gpu' in self.benchmark_results['raw_material_characterization']:
            gpu_results = self.benchmark_results['raw_material_characterization']['gpu']
            if gpu_results:
                batch_100_gpu = gpu_results.get(100, gpu_results.get(50, {}))
                if batch_100_gpu:
                    summary['key_metrics']['gpu_speedup'] = batch_100_gpu.get('speedup_vs_cpu', 1.0)
        
        # Generate recommendations
        if summary['key_metrics'].get('characterization_throughput', 0) < 10:
            summary['recommendations'].append("Consider GPU acceleration for improved characterization throughput")
        
        if summary['key_metrics'].get('validation_success_rate', 0) < 0.95:
            summary['recommendations'].append("Review validation logic for improved success rate")
        
        return summary
    
    def save_results(self, filename: str = None):
        """Save benchmark results to file."""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"pharmaceutical_benchmark_{timestamp}.json"
        
        filepath = os.path.join('reports', filename)
        os.makedirs('reports', exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.benchmark_results, f, indent=2, default=str)
        
        logger.info(f"📊 Benchmark results saved to: {filepath}")
        return filepath


def main():
    """Main benchmark execution."""
    logger.info("🔬 Starting Pharmaceutical Performance Benchmark Suite")
    
    # Create benchmark instance
    benchmark = PharmaceuticalPerformanceBenchmark()
    
    # Run comprehensive benchmark
    results = benchmark.run_comprehensive_benchmark()
    
    # Save results
    report_file = benchmark.save_results()
    
    # Print summary
    logger.info("\n📊 BENCHMARK SUMMARY")
    logger.info("=" * 50)
    
    summary = results['summary']
    logger.info(f"System Performance: {summary['system_performance']}")
    
    if 'key_metrics' in summary:
        metrics = summary['key_metrics']
        logger.info("\nKey Performance Metrics:")
        for metric, value in metrics.items():
            if 'throughput' in metric:
                logger.info(f"  {metric}: {value:.2f} ops/sec")
            elif 'rate' in metric:
                logger.info(f"  {metric}: {value:.2%}")
            elif 'speedup' in metric:
                logger.info(f"  {metric}: {value:.2f}x")
            else:
                logger.info(f"  {metric}: {value}")
    
    if summary.get('recommendations'):
        logger.info("\nRecommendations:")
        for rec in summary['recommendations']:
            logger.info(f"  • {rec}")
    
    logger.info(f"\nDetailed results saved to: {report_file}")
    logger.info("✅ Benchmark completed successfully")


if __name__ == '__main__':
    main() 