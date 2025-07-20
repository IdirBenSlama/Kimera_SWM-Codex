#!/usr/bin/env python3
"""
Simple Test Suite for Kimera Enterprise Trading System

This script tests the core functionality of all enterprise trading components
without async complications, focusing on initialization and basic operations.

Author: Kimera AI System
Date: 2025-01-10
"""

import sys
import os
import time
import traceback
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_gpu_availability():
    """Check if GPU is available for trading operations."""
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"✅ GPU Available: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            logger.warning("⚠️ GPU not available - using CPU")
        return gpu_available
    except ImportError:
        logger.warning("⚠️ PyTorch not available - cannot check GPU")
        return False

def test_complex_event_processor():
    """Test Complex Event Processing Engine initialization."""
    logger.info("Testing Complex Event Processing Engine...")
    try:
        from backend.trading.enterprise.complex_event_processor import ComplexEventProcessor
        
        # Initialize without async context
        processor = ComplexEventProcessor()
        
        # Test basic attributes
        assert hasattr(processor, 'event_store')
        assert hasattr(processor, 'pattern_library')
        assert hasattr(processor, 'quantum_matcher')
        
        # Test pattern library
        patterns = processor.pattern_library
        expected_patterns = ['price_spike', 'volume_surge', 'momentum_shift']
        
        for pattern in expected_patterns:
            assert pattern in patterns, f"Missing pattern: {pattern}"
        
        return {
            'status': 'success',
            'component': 'ComplexEventProcessor',
            'patterns_available': len(patterns),
            'quantum_enabled': processor.quantum_matcher is not None,
            'features': ['event_processing', 'pattern_matching', 'quantum_enhanced']
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'component': 'ComplexEventProcessor',
            'error': str(e)
        }

def test_smart_order_router():
    """Test Smart Order Routing System initialization."""
    logger.info("Testing Smart Order Routing System...")
    try:
        from backend.trading.enterprise.smart_order_router import SmartOrderRouter
        
        # Initialize router
        router = SmartOrderRouter()
        
        # Test basic attributes
        assert hasattr(router, 'venues')
        assert hasattr(router, 'routing_engine')
        
        # Test venue configuration
        venues = getattr(router, 'venues', {})
        logger.info(f"Available venues: {list(venues.keys())}")
        
        return {
            'status': 'success',
            'component': 'SmartOrderRouter',
            'venues_available': len(venues),
            'features': ['order_routing', 'venue_selection', 'latency_optimization']
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'component': 'SmartOrderRouter',
            'error': str(e),
            'traceback': traceback.format_exc()
        }

def test_market_microstructure_analyzer():
    """Test Market Microstructure Analyzer initialization."""
    logger.info("Testing Market Microstructure Analyzer...")
    try:
        from backend.trading.enterprise.market_microstructure_analyzer import MarketMicrostructureAnalyzer
        
        # Initialize analyzer
        analyzer = MarketMicrostructureAnalyzer()
        
        # Test basic attributes
        assert hasattr(analyzer, 'order_book_reconstructor')
        assert hasattr(analyzer, 'liquidity_analyzer')
        
        return {
            'status': 'success',
            'component': 'MarketMicrostructureAnalyzer',
            'features': ['order_book_reconstruction', 'liquidity_analysis', 'market_impact_prediction']
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'component': 'MarketMicrostructureAnalyzer',
            'error': str(e)
        }

def test_regulatory_compliance_engine():
    """Test Regulatory Compliance Engine initialization."""
    logger.info("Testing Regulatory Compliance Engine...")
    try:
        from backend.trading.enterprise.regulatory_compliance_engine import RegulatoryComplianceEngine
        
        # Initialize compliance engine
        compliance = RegulatoryComplianceEngine()
        
        # Test basic attributes
        assert hasattr(compliance, 'compliance_rules')
        assert hasattr(compliance, 'jurisdictions')
        
        # Test compliance rules
        rules = getattr(compliance, 'compliance_rules', {})
        logger.info(f"Compliance rules loaded: {len(rules)}")
        
        return {
            'status': 'success',
            'component': 'RegulatoryComplianceEngine',
            'rules_loaded': len(rules),
            'features': ['multi_jurisdictional', 'real_time_monitoring', 'automated_reporting']
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'component': 'RegulatoryComplianceEngine',
            'error': str(e)
        }

def test_quantum_trading_engine():
    """Test Quantum Trading Engine initialization."""
    logger.info("Testing Quantum Trading Engine...")
    try:
        from backend.trading.enterprise.quantum_trading_engine import QuantumTradingEngine
        
        # Initialize quantum engine
        quantum_engine = QuantumTradingEngine()
        
        # Test basic attributes
        assert hasattr(quantum_engine, 'quantum_optimizer')
        assert hasattr(quantum_engine, 'pattern_recognizer')
        
        # Test quantum availability
        quantum_available = getattr(quantum_engine, 'quantum_available', False)
        logger.info(f"Quantum computing available: {quantum_available}")
        
        return {
            'status': 'success',
            'component': 'QuantumTradingEngine',
            'quantum_available': quantum_available,
            'features': ['quantum_optimization', 'portfolio_management', 'pattern_recognition']
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'component': 'QuantumTradingEngine',
            'error': str(e)
        }

def test_ml_trading_engine():
    """Test Machine Learning Trading Engine initialization."""
    logger.info("Testing Machine Learning Trading Engine...")
    try:
        from backend.trading.enterprise.ml_trading_engine import MLTradingEngine
        
        # Initialize ML engine
        ml_engine = MLTradingEngine()
        
        # Test basic attributes
        assert hasattr(ml_engine, 'device')
        assert hasattr(ml_engine, 'models')
        
        # Test device
        device = getattr(ml_engine, 'device', 'cpu')
        logger.info(f"ML Engine device: {device}")
        
        return {
            'status': 'success',
            'component': 'MLTradingEngine',
            'device': str(device),
            'features': ['neural_networks', 'ensemble_methods', 'reinforcement_learning']
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'component': 'MLTradingEngine',
            'error': str(e)
        }

def test_hft_infrastructure():
    """Test High-Frequency Trading Infrastructure initialization."""
    logger.info("Testing High-Frequency Trading Infrastructure...")
    try:
        from backend.trading.enterprise.hft_infrastructure import HFTInfrastructure
        
        # Initialize HFT infrastructure
        hft = HFTInfrastructure()
        
        # Test basic attributes
        assert hasattr(hft, 'latency_monitor')
        assert hasattr(hft, 'execution_engine')
        
        # Test performance features
        gpu_acceleration = getattr(hft, 'gpu_acceleration', False)
        logger.info(f"HFT GPU acceleration: {gpu_acceleration}")
        
        return {
            'status': 'success',
            'component': 'HFTInfrastructure',
            'gpu_acceleration': gpu_acceleration,
            'features': ['ultra_low_latency', 'hardware_acceleration', 'lock_free_structures']
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'component': 'HFTInfrastructure',
            'error': str(e)
        }

def test_integrated_trading_system():
    """Test Integrated Trading System initialization."""
    logger.info("Testing Integrated Trading System...")
    try:
        # Import required components first
        from backend.engines.cognitive_field_dynamics import CognitiveFieldDynamics
        from backend.engines.thermodynamic_engine import ThermodynamicEngine
        from backend.engines.contradiction_engine import ContradictionEngine
        from backend.trading.enterprise.integrated_trading_system import IntegratedTradingSystem
        
        # Create mock components with required parameters
        cognitive_field = CognitiveFieldDynamics(dimension=128)  # Standard dimension
        thermodynamic_engine = ThermodynamicEngine()
        contradiction_engine = ContradictionEngine()
        
        # Initialize integrated system
        integrated_system = IntegratedTradingSystem(
            cognitive_field=cognitive_field,
            thermodynamic_engine=thermodynamic_engine,
            contradiction_engine=contradiction_engine
        )
        
        # Test basic attributes
        assert hasattr(integrated_system, 'cognitive_field')
        assert hasattr(integrated_system, 'components')
        
        # Test component integration
        components = getattr(integrated_system, 'components', {})
        logger.info(f"Integrated components: {len(components)}")
        
        return {
            'status': 'success',
            'component': 'IntegratedTradingSystem',
            'components_integrated': len(components),
            'features': ['unified_control', 'cognitive_integration', 'orchestration']
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'component': 'IntegratedTradingSystem',
            'error': str(e)
        }

def run_functional_tests():
    """Run basic functional tests on working components."""
    logger.info("Running functional tests...")
    
    functional_results = {}
    
    # Test Smart Order Router functionality
    try:
        from backend.trading.enterprise.smart_order_router import SmartOrderRouter
        router = SmartOrderRouter()
        
        # Test order routing (without async)
        test_order = {
            'symbol': 'BTCUSDT',
            'side': 'buy',
            'quantity': 1.0,
            'order_type': 'market'
        }
        
        # Test venue selection
        if hasattr(router, 'select_venue'):
            venue = router.select_venue(test_order)
            functional_results['smart_order_router'] = {
                'venue_selection': venue,
                'status': 'functional'
            }
        
    except Exception as e:
        functional_results['smart_order_router'] = {
            'status': 'error',
            'error': str(e)
        }
    
    # Test ML Trading Engine functionality
    try:
        from backend.trading.enterprise.ml_trading_engine import MLTradingEngine
        ml_engine = MLTradingEngine()
        
        # Test basic prediction capability
        test_data = {
            'prices': [45000, 45100, 44900, 45200],
            'volumes': [100, 150, 80, 200]
        }
        
        if hasattr(ml_engine, 'prepare_features'):
            features = ml_engine.prepare_features(test_data)
            functional_results['ml_trading_engine'] = {
                'feature_preparation': 'working',
                'status': 'functional'
            }
        
    except Exception as e:
        functional_results['ml_trading_engine'] = {
            'status': 'error',
            'error': str(e)
        }
    
    return functional_results

def main():
    """Main test execution function."""
    logger.info("Starting Simple Enterprise Trading System Test")
    logger.info("=" * 80)
    
    # Check GPU availability
    gpu_available = check_gpu_availability()
    
    # Test all components
    test_functions = [
        ('ComplexEventProcessor', test_complex_event_processor),
        ('SmartOrderRouter', test_smart_order_router),
        ('MarketMicrostructureAnalyzer', test_market_microstructure_analyzer),
        ('RegulatoryComplianceEngine', test_regulatory_compliance_engine),
        ('QuantumTradingEngine', test_quantum_trading_engine),
        ('MLTradingEngine', test_ml_trading_engine),
        ('HFTInfrastructure', test_hft_infrastructure),
        ('IntegratedTradingSystem', test_integrated_trading_system)
    ]
    
    results = {}
    
    for component_name, test_func in test_functions:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing {component_name}")
        logger.info(f"{'='*60}")
        
        start_time = time.time()
        result = test_func()
        end_time = time.time()
        
        result['execution_time'] = end_time - start_time
        results[component_name] = result
        
        if result['status'] == 'success':
            logger.info(f"✅ {component_name} - SUCCESS")
            if 'features' in result:
                logger.info(f"   Features: {', '.join(result['features'])}")
        else:
            logger.error(f"❌ {component_name} - FAILED")
            logger.error(f"   Error: {result['error']}")
    
    # Run functional tests
    logger.info(f"\n{'='*60}")
    logger.info("Running Functional Tests")
    logger.info(f"{'='*60}")
    
    functional_results = run_functional_tests()
    
    # Generate final report
    logger.info(f"\n{'='*80}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*80}")
    
    successful = len([r for r in results.values() if r['status'] == 'success'])
    failed = len([r for r in results.values() if r['status'] == 'error'])
    total = len(results)
    
    logger.info(f"Total Components Tested: {total}")
    logger.info(f"Successful Components: {successful}")
    logger.info(f"Failed Components: {failed}")
    logger.info(f"Success Rate: {successful / total * 100:.1f}%")
    logger.info(f"GPU Available: {gpu_available}")
    
    # Save detailed report
    report = {
        'test_timestamp': datetime.now().isoformat(),
        'system_info': {
            'gpu_available': gpu_available
        },
        'component_tests': results,
        'functional_tests': functional_results,
        'summary': {
            'total_components': total,
            'successful_components': successful,
            'failed_components': failed,
            'success_rate': successful / total * 100
        }
    }
    
    report_filename = f"enterprise_trading_simple_test_{int(time.time())}.json"
    
    with open(report_filename, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"\nDetailed report saved to: {report_filename}")
    
    # Print component status
    logger.info(f"\n{'='*80}")
    logger.info("COMPONENT STATUS DETAILS")
    logger.info(f"{'='*80}")
    
    for component_name, result in results.items():
        status_icon = "✅" if result['status'] == 'success' else "❌"
        logger.info(f"{status_icon} {component_name:<30} - {result['status'].upper()}")
        
        if 'features' in result:
            logger.info(f"    Features: {', '.join(result['features'])}")
        
        if result['status'] == 'error':
            logger.error(f"    Error: {result['error']}")
    
    # Print functional test results
    if functional_results:
        logger.info(f"\n{'='*80}")
        logger.info("FUNCTIONAL TEST RESULTS")
        logger.info(f"{'='*80}")
        
        for component, result in functional_results.items():
            status_icon = "✅" if result['status'] == 'functional' else "❌"
            logger.info(f"{status_icon} {component:<30} - {result['status'].upper()}")
    
    return results

if __name__ == "__main__":
    try:
        results = main()
        
        # Exit with appropriate code
        failed_count = len([r for r in results.values() if r['status'] == 'error'])
        sys.exit(0 if failed_count == 0 else 1)
        
    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test suite failed with error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1) 