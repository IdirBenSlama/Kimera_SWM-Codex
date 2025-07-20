#!/usr/bin/env python3
"""
Test script for Kimera SWM Monitoring System

This script tests the core monitoring functionality to ensure everything works correctly.
"""

import sys
import os
import numpy as np
from datetime import datetime

# Initialize structured logger
from backend.utils.kimera_logger import get_system_logger
logger = get_system_logger(__name__)


# Add the backend to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.core.geoid import GeoidState
from backend.core.models import LinguisticGeoid
from backend.monitoring.entropy_monitor import EntropyMonitor
from backend.monitoring.semantic_metrics import SemanticMetricsCollector
from backend.monitoring.thermodynamic_analyzer import ThermodynamicAnalyzer
from backend.monitoring.system_observer import SystemObserver
from backend.monitoring.benchmarking_suite import benchmark_runner

def create_test_geoids(count=10):
    """Create test geoids for monitoring"""
    geoids = []
    for i in range(count):
        semantic_state = {
            f'feature_{j}': np.random.random() 
            for j in range(np.random.randint(3, 8))
        }
        
        geoid = GeoidState(
            geoid_id=f'test_geoid_{i}',
            semantic_state=semantic_state,
            symbolic_state={'type': 'test', 'index': i},
            metadata={'created': datetime.now().isoformat()}
        )
        geoids.append(geoid)
    
    return geoids

def create_test_linguistic_geoids(geoids):
    """Create test linguistic geoids"""
    linguistic_geoids = []
    
    for i, geoid in enumerate(geoids[:5]):  # Only first 5
        lg = LinguisticGeoid(
            primary_statement=f"Test statement for geoid {geoid.geoid_id}",
            confidence_score=np.random.uniform(0.6, 1.0),
            source_geoid_id=geoid.geoid_id,
            supporting_scars=[],
            potential_ambiguities=[f'ambiguity_{j}' for j in range(np.random.randint(0, 3))],
            explanation_lineage=f'test_lineage_{i}'
        )
        linguistic_geoids.append(lg)
    
    return linguistic_geoids

def test_entropy_monitor():
    """Test the entropy monitoring system"""
    logger.info("🧪 Testing Entropy Monitor...")
    
    entropy_monitor = EntropyMonitor()
    geoids = create_test_geoids(20)
    
    vault_info = {
        'vault_a_scars': 15,
        'vault_b_scars': 12,
        'active_geoids': len(geoids)
    }
    
    # Set baseline
    entropy_monitor.set_baseline(geoids[:10])
    
    # Calculate entropy
    measurement = entropy_monitor.calculate_system_entropy(geoids, vault_info)
    
    logger.info(f"  ✅ Shannon Entropy: {measurement.shannon_entropy:.4f}")
    logger.info(f"  ✅ Thermodynamic Entropy: {measurement.thermodynamic_entropy:.4f}")
    logger.info(f"  ✅ Relative Entropy: {measurement.relative_entropy:.4f}")
    logger.info(f"  ✅ System Complexity: {measurement.system_complexity:.4f}")
    
    # Test trends
    for _ in range(5):
        measurement = entropy_monitor.calculate_system_entropy(geoids, vault_info)
    
    trends = entropy_monitor.get_entropy_trends(window_size=5)
    logger.info(f"  ✅ Trend data points: {len(trends.get('shannon_entropy', [])
    
    # Test anomaly detection
    anomalies = entropy_monitor.detect_entropy_anomalies()
    logger.info(f"  ✅ Anomalies detected: {len(anomalies)
    
    assert measurement.shannon_entropy > 0
    assert len(trends.get('shannon_entropy', [])) == 5
    assert len(anomalies) == 0
    
    return True

def test_semantic_metrics():
    """Test the semantic metrics collection"""
    logger.info("🧠 Testing Semantic Metrics Collector...")
    
    semantic_collector = SemanticMetricsCollector()
    geoids = create_test_geoids(15)
    linguistic_geoids = create_test_linguistic_geoids(geoids)
    
    # Collect metrics
    measurement = semantic_collector.collect_semantic_metrics(
        geoids, linguistic_geoids, 'analytical'
    )
    
    logger.info(f"  ✅ Semantic Entropy: {measurement.semantic_entropy:.4f}")
    logger.info(f"  ✅ Meaning Density: {measurement.meaning_density:.4f}")
    logger.info(f"  ✅ Context Coherence: {measurement.context_coherence:.4f}")
    logger.info(f"  ✅ Semantic Efficiency: {measurement.semantic_efficiency:.4f}")
    logger.info(f"  ✅ Information Utility: {measurement.information_utility:.4f}")
    
    # Test trends
    for context in ['analytical', 'creative', 'balanced']:
        measurement = semantic_collector.collect_semantic_metrics(
            geoids, linguistic_geoids, context
        )
    
    trends = semantic_collector.get_semantic_trends(window_size=3)
    logger.info(f"  ✅ Trend data points: {len(trends.get('semantic_entropy', [])
    
    # Test anomaly detection
    anomalies = semantic_collector.detect_semantic_anomalies()
    logger.info(f"  ✅ Anomalies detected: {len(anomalies)
    
    assert measurement.semantic_entropy > 0
    assert len(trends.get('semantic_entropy', [])) == 3
    assert len(anomalies) == 0
    
    return True

def test_thermodynamic_analyzer():
    """Test the thermodynamic analysis"""
    logger.info("🔥 Testing Thermodynamic Analyzer...")
    
    thermodynamic_analyzer = ThermodynamicAnalyzer()
    geoids = create_test_geoids(25)
    
    vault_info = {
        'vault_a_scars': 20,
        'vault_b_scars': 18,
        'active_geoids': len(geoids)
    }
    
    # Analyze state
    state = thermodynamic_analyzer.analyze_thermodynamic_state(
        geoids, vault_info, 2.5  # system entropy
    )
    
    logger.info(f"  ✅ Total Energy: {state.total_energy:.4f}")
    logger.info(f"  ✅ Temperature: {state.temperature:.4f}")
    logger.info(f"  ✅ Pressure: {state.pressure:.4f}")
    logger.info(f"  ✅ Free Energy: {state.free_energy:.4f}")
    logger.info(f"  ✅ Efficiency: {state.efficiency:.4f}")
    
    # Test multiple states for trends
    for _ in range(3):
        # Modify geoids slightly
        for geoid in geoids:
            for feature in geoid.semantic_state:
                geoid.semantic_state[feature] += np.random.normal(0, 0.05)
            geoid.__post_init__()
        
        state = thermodynamic_analyzer.analyze_thermodynamic_state(
            geoids, vault_info, 2.5
        )
    
    # Test efficiency calculation
    efficiency = thermodynamic_analyzer.calculate_thermodynamic_efficiency()
    logger.info(f"  ✅ Efficiency metrics: {len(efficiency)
    
    # Test constraint checking
    violations = thermodynamic_analyzer.check_thermodynamic_constraints(state)
    logger.info(f"  ✅ Constraint violations: {len(violations)
    
    assert state.total_energy > 0
    assert len(efficiency) > 0
    assert len(violations) == 0
    
    return True

def test_system_observer():
    """Test the integrated system observer"""
    logger.info("👁️ Testing System Observer...")
    
    system_observer = SystemObserver()
    geoids = create_test_geoids(30)
    linguistic_geoids = create_test_linguistic_geoids(geoids)
    
    vault_info = {
        'vault_a_scars': 25,
        'vault_b_scars': 22,
        'active_geoids': len(geoids)
    }
    
    # Test different observer contexts
    for context in ['analytical', 'operational', 'research', 'default']:
        system_observer.set_observer_context(context)
        snapshot = system_observer.observe_system(geoids, linguistic_geoids, vault_info)
        
        logger.info(f"  ✅ {context.capitalize()
        logger.info(f"    - Overall Health: {snapshot.system_health['overall_health']:.3f}")
        logger.info(f"    - Shannon Entropy: {snapshot.entropy_measurement.shannon_entropy:.3f}")
        logger.info(f"    - Semantic Efficiency: {snapshot.semantic_measurement.semantic_efficiency:.3f}")
    
    # Test summary generation
    summary = system_observer.get_system_summary()
    logger.info(f"  ✅ System summary generated with {summary.get('observation_count', 0)
    
    # Test report generation
    for report_type in ['comprehensive', 'executive', 'technical']:
        report = system_observer.generate_report(report_type)
        logger.info(f"  ✅ {report_type.capitalize()
    
    assert summary.get('observation_count', 0) == 4
    assert len(system_observer.generate_report('comprehensive')) > 0
    
    return True

def test_benchmarking_suite():
    """Test the benchmarking system"""
    logger.info("🧪 Testing Benchmarking Suite...")
    
    # Test individual benchmark functions
    try:
        # Test entropy estimator comparison
        result = benchmark_runner.run_test('entropy_estimator_comparison')
        logger.info(f"  ✅ Entropy estimator test: {'PASSED' if result.success else 'FAILED'}")
        if result.success:
            logger.info(f"    - Duration: {result.duration:.3f}s")
            logger.info(f"    - Metrics: {len(result.metrics)
        
        # Test thermodynamic consistency
        result = benchmark_runner.run_test('thermodynamic_consistency')
        logger.info(f"  ✅ Thermodynamic test: {'PASSED' if result.success else 'FAILED'}")
        if result.success:
            logger.info(f"    - Duration: {result.duration:.3f}s")
            logger.error(f"    - Energy conservation error: {result.metrics.get('energy_conservation_error', 'N/A')
        
        # Test semantic processing
        text_samples = [
            "This is a test of semantic processing capabilities",
            "Analyzing information content and meaning density",
            "Evaluating semantic thermodynamic principles"
        ]
        result = benchmark_runner.run_test('semantic_processing', {'text_samples': text_samples})
        logger.info(f"  ✅ Semantic processing test: {'PASSED' if result.success else 'FAILED'}")
        if result.success:
            logger.info(f"    - Duration: {result.duration:.3f}s")
            logger.info(f"    - Throughput: {result.metrics.get('throughput', 'N/A')
        
        # Test benchmark suite
        suite_results = benchmark_runner.run_suite('entropy_suite')
        passed = sum(1 for r in suite_results.values() if r.success)
        total = len(suite_results)
        logger.info(f"  ✅ Entropy suite: {passed}/{total} tests passed")
        
        assert all(r.success for r in suite_results.values())
        
        return True
        
    except Exception as e:
        logger.error(f"  ❌ Benchmarking test failed: {e}")
        return False

def main():
    """Run all monitoring system tests"""
    logger.info("🧠 Kimera SWM Monitoring System Test Suite")
    logger.info("=" * 50)
    
    tests = [
        ("Entropy Monitor", test_entropy_monitor),
        ("Semantic Metrics", test_semantic_metrics),
        ("Thermodynamic Analyzer", test_thermodynamic_analyzer),
        ("System Observer", test_system_observer),
        ("Benchmarking Suite", test_benchmarking_suite)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            logger.info(f"\n{test_name}")
            logger.info("-" * len(test_name)
            success = test_func()
            results.append((test_name, success))
            if success:
                logger.info(f"✅ {test_name} - PASSED")
            else:
                logger.error(f"❌ {test_name} - FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} - ERROR: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 Test Results Summary")
    logger.info("=" * 50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name:.<30} {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)
    
    if passed == total:
        logger.info("\n🎉 All tests passed! The monitoring system is ready to use.")
        logger.info("\nNext steps:")
        logger.info("1. Start the API server: python -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8001")
        logger.info("2. Launch the monitoring dashboard: python launch_monitoring.py")
        logger.info("3. Access the dashboard at: file:///path/to/monitoring_dashboard.html")
        return 0
    else:
        logger.warning(f"\n⚠️  {total - passed} tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
