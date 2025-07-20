#!/usr/bin/env python3
"""
Advanced Pharmaceutical Testing Benchmark

Comprehensive benchmarking of new advanced pharmaceutical features.
"""

import time
import json
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
import sys
import logging

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.pharmaceutical.core.kcl_testing_engine import KClTestingEngine
from backend.pharmaceutical.analysis.dissolution_analyzer import DissolutionAnalyzer
from backend.pharmaceutical.validation.pharmaceutical_validator import PharmaceuticalValidator
from backend.utils.kimera_logger import get_logger

logger = get_logger(__name__)


class AdvancedPharmaceuticalBenchmark:
    """Comprehensive benchmark for advanced pharmaceutical capabilities."""
    
    def __init__(self):
        """Initialize benchmark environment."""
        logger.info("🚀 Initializing Advanced Pharmaceutical Benchmark...")
        
        self.kcl_engine = KClTestingEngine(use_gpu=True)
        self.dissolution_analyzer = DissolutionAnalyzer(use_gpu=True)
        
        logger.info("✅ Advanced engines initialized successfully")
    
    def benchmark_batch_processing(self) -> dict:
        """Benchmark advanced batch processing capabilities."""
        logger.info("📊 Benchmarking Advanced Batch Processing...")
        
        results = {'timestamp': datetime.now().isoformat()}
        batch_sizes = [10, 25, 50, 100]
        
        for batch_size in batch_sizes:
            logger.info(f"   Testing batch size: {batch_size}")
            
            # Generate test data
            test_batches = []
            for i in range(batch_size):
                batch = {
                    'name': f'Batch_{i:04d}',
                    'grade': 'USP',
                    'purity_percent': np.random.normal(99.5, 0.3),
                    'moisture_content': np.random.exponential(0.5),
                    'particle_size_d50': np.random.normal(150, 20),
                    'bulk_density': np.random.normal(1.1, 0.1),
                    'tapped_density': np.random.normal(1.3, 0.1),
                    'potassium_confirmed': True,
                    'chloride_confirmed': True
                }
                test_batches.append(batch)
            
            # Benchmark batch processing
            start_time = time.perf_counter()
            batch_results = self.kcl_engine.characterize_raw_materials_batch(test_batches)
            batch_time = time.perf_counter() - start_time
            
            throughput = len(batch_results) / batch_time
            success_rate = len(batch_results) / batch_size
            
            results[f'batch_size_{batch_size}'] = {
                'processing_time': batch_time,
                'throughput': throughput,
                'success_rate': success_rate
            }
            
            logger.info(f"     Throughput: {throughput:.1f} samples/sec")
        
        return results
    
    def benchmark_ml_dissolution(self) -> dict:
        """Benchmark ML dissolution prediction."""
        logger.info("🤖 Benchmarking ML Dissolution Prediction...")
        
        if not torch.cuda.is_available():
            return {'status': 'CUDA_NOT_AVAILABLE'}
        
        results = {'timestamp': datetime.now().isoformat()}
        
        # Test prediction performance
        test_params = {
            'coating_thickness': 15.0,
            'ethylcellulose_ratio': 0.8,
            'hpc_ratio': 0.2,
            'drug_loading': 50.0,
            'particle_size': 150.0,
            'tablet_hardness': 8.0,
            'porosity': 0.15,
            'surface_area': 2.5,
            'ph_media': 6.8,
            'temperature': 37.0,
            'agitation_speed': 100.0,
            'ionic_strength': 0.1
        }
        
        if self.dissolution_analyzer.ml_model is not None:
            prediction_times = []
            for i in range(50):
                start_time = time.perf_counter()
                try:
                    prediction = self.dissolution_analyzer.predict_dissolution_ml(
                        test_params, [1, 6, 12, 24]
                    )
                    prediction_time = time.perf_counter() - start_time
                    prediction_times.append(prediction_time)
                except:
                    pass
            
            if prediction_times:
                results['prediction_performance'] = {
                    'average_time': np.mean(prediction_times),
                    'predictions_per_second': 1.0 / np.mean(prediction_times),
                    'fastest_prediction': min(prediction_times)
                }
        
        return results
    
    def run_comprehensive_benchmark(self) -> dict:
        """Run comprehensive benchmark."""
        logger.info("🚀 Starting Comprehensive Advanced Benchmark")
        
        start_time = time.perf_counter()
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'system_info': {
                'cuda_available': torch.cuda.is_available(),
                'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            },
            'batch_processing': self.benchmark_batch_processing(),
            'ml_dissolution': self.benchmark_ml_dissolution()
        }
        
        total_time = time.perf_counter() - start_time
        results['total_time'] = total_time
        
        logger.info(f"✅ Benchmark completed in {total_time:.2f}s")
        
        return results


def main():
    """Run the benchmark."""
    try:
        benchmark = AdvancedPharmaceuticalBenchmark()
        results = benchmark.run_comprehensive_benchmark()
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"reports/advanced_benchmark_{timestamp}.json"
        
        Path("reports").mkdir(exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"📊 Results saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"❌ Benchmark failed: {e}")
        raise


if __name__ == "__main__":
    main() 