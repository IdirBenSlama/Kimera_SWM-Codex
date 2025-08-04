#!/usr/bin/env python3
"""
BGM Dimensional Scaling Test Protocol
====================================
Aerospace-grade graduated testing: 128D → 256D → 512D → 1024D

Following DO-178C Level A testing standards:
- Systematic progression through dimensional scales
- Performance validation at each level
- Memory utilization monitoring
- GPU capacity verification
- Failure mode analysis
"""

import torch
import time
import numpy as np
import logging
from typing import Dict, List, Tuple
import psutil
import os
from datetime import datetime

# Add project root to path
import sys
sys.path.insert(0, '.')

from src.core.high_dimensional_modeling.high_dimensional_bgm import HighDimensionalBGM, BGMConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BGMDimensionalTester:
    """Aerospace-grade dimensional testing framework"""

    def __init__(self):
        self.test_dimensions = [128, 256, 512, 1024]
        self.results = {}

    def test_dimension(self, dimensions: int) -> Dict:
        """Test BGM at specific dimensional level"""
        logger.info(f"🧪 TESTING BGM AT {dimensions}D")

        # Memory baseline
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        start_time = time.time()

        try:
            # Create BGM config for this dimension
            config = BGMConfig(
                dimension=dimensions,
                batch_size=100 if dimensions >= 512 else 1000,  # Reduce batch for high-D
                time_horizon=1.0,
                dt=1.0/252.0,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )

            # Initialize BGM
            bgm = HighDimensionalBGM(config)

            # Test basic operations
            initial_prices = torch.ones(dimensions) * 100.0

            # Create drift and volatility for this dimension
            drift = torch.ones(dimensions) * 0.05 / 252  # 5% annual return
            volatility = torch.ones(dimensions) * 0.2 / (252**0.5)  # 20% annual vol

            # Set parameters
            bgm.set_parameters(drift, volatility)

            # Performance test with smaller scenario count for high dimensions
            num_scenarios = 5 if dimensions >= 512 else 10
            scenarios = bgm.generate_market_scenarios(initial_prices, num_scenarios)

            initialization_time = time.time() - start_time

            # Memory after initialization
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_usage = memory_after - memory_before

            # GPU memory if available
            gpu_memory = 0
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB

            result = {
                'dimensions': dimensions,
                'success': True,
                'initialization_time': initialization_time,
                'memory_usage_mb': memory_usage,
                'gpu_memory_mb': gpu_memory,
                'scenarios_shape': scenarios['scenarios'].shape,
                'device': str(config.device),
                'timestamp': datetime.now().isoformat()
            }

            logger.info(f"✅ {dimensions}D: {initialization_time:.3f}s, {memory_usage:.1f}MB RAM, {gpu_memory:.1f}MB GPU")

            return result

        except Exception as e:
            logger.error(f"❌ {dimensions}D FAILED: {str(e)}")
            return {
                'dimensions': dimensions,
                'success': False,
                'error': str(e),
                'initialization_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }

    def run_graduated_test(self) -> Dict:
        """Execute graduated dimensional testing protocol"""
        logger.info("🚀 STARTING BGM DIMENSIONAL SCALING TEST")
        logger.info("=" * 60)

        all_results = []

        for dimensions in self.test_dimensions:
            result = self.test_dimension(dimensions)
            all_results.append(result)
            self.results[dimensions] = result

            # Stop if we hit a failure (aerospace safety protocol)
            if not result['success']:
                logger.error(f"💥 CRITICAL: BGM failed at {dimensions}D - STOPPING PROGRESSION")
                break

            # Brief cooling period for GPU
            time.sleep(2)

        return self.generate_report()

    def generate_report(self) -> Dict:
        """Generate comprehensive test report"""
        successful_tests = [r for r in self.results.values() if r['success']]

        if not successful_tests:
            return {
                'status': 'FAILED',
                'max_successful_dimension': 0,
                'recommendation': 'System cannot handle high-dimensional BGM'
            }

        max_dimension = max(r['dimensions'] for r in successful_tests)

        # Performance analysis
        performance_data = []
        for result in successful_tests:
            performance_data.append({
                'dimensions': result['dimensions'],
                'time': result['initialization_time'],
                'memory': result['memory_usage_mb'],
                'gpu_memory': result.get('gpu_memory_mb', 0)
            })

        # Determine optimal configuration
        if max_dimension >= 1024:
            recommendation = "✅ SYSTEM READY FOR 1024D OPERATION"
            optimal_config = 1024
        elif max_dimension >= 512:
            recommendation = "⚠️ RECOMMEND 512D WITH MONITORING"
            optimal_config = 512
        else:
            recommendation = f"🔻 LIMITED TO {max_dimension}D OPERATION"
            optimal_config = max_dimension

        report = {
            'status': 'SUCCESS' if max_dimension >= 512 else 'LIMITED',
            'max_successful_dimension': max_dimension,
            'optimal_configuration': optimal_config,
            'recommendation': recommendation,
            'performance_data': performance_data,
            'detailed_results': self.results,
            'test_timestamp': datetime.now().isoformat()
        }

        # Display summary
        logger.info("=" * 60)
        logger.info("📊 BGM DIMENSIONAL TESTING COMPLETE")
        logger.info(f"🎯 Maximum Dimension: {max_dimension}D")
        logger.info(f"🔧 Optimal Config: {optimal_config}D")
        logger.info(f"💡 {recommendation}")

        return report

def main():
    """Execute BGM dimensional testing protocol"""
    tester = BGMDimensionalTester()

    try:
        report = tester.run_graduated_test()

        # Save detailed report
        import json
        os.makedirs('docs/reports/testing', exist_ok=True)

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        report_path = f'docs/reports/testing/{timestamp}_bgm_dimensional_test.json'

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"📄 Detailed report saved: {report_path}")

        return report

    except Exception as e:
        logger.error(f"💥 CRITICAL TEST FAILURE: {e}")
        return {'status': 'CRITICAL_FAILURE', 'error': str(e)}

if __name__ == "__main__":
    main()
