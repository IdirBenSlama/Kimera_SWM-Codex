#!/usr/bin/env python3
"""
Comprehensive Objective Test Suite for Kimera Trading System
Tests all components without requiring API keys or making real trades
"""

import asyncio
import sys
import json
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any
import warnings

# Initialize structured logger
from backend.utils.kimera_logger import get_system_logger
logger = get_system_logger(__name__)

warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append('.')

logger.info("=" * 70)
logger.info("🧪 KIMERA TRADING SYSTEM - COMPREHENSIVE OBJECTIVE TEST")
logger.info("=" * 70)

# Test Results Storage
test_results = {
    "timestamp": datetime.now().isoformat(),
    "environment": {
        "python_version": sys.version,
        "platform": sys.platform,
        "working_directory": os.getcwd()
    },
    "dependencies": {},
    "component_tests": {},
    "integration_tests": {},
    "performance_tests": {},
    "overall_status": "UNKNOWN"
}

def log_test(category: str, component: str, test_name: str, status: str, details: str = ""):
    """Log test results"""
    if category not in test_results:
        test_results[category] = {}
    if component not in test_results[category]:
        test_results[category][component] = {}
    
    test_results[category][component][test_name] = {
        "status": status,
        "details": details,
        "timestamp": datetime.now().isoformat()
    }
    
    status_icon = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
    logger.info(f"{status_icon} {category}.{component}.{test_name}: {status} {details}")

def test_environment():
    """Test environment and dependencies"""
    logger.info("\n🌍 Testing Environment & Dependencies...")
    
    # Core dependencies
    core_deps = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("sklearn", "sklearn"),
        ("asyncio", "asyncio"),
    ]
    
    # Test core dependencies
    for name, import_name in core_deps:
        try:
            module = __import__(import_name)
            version = getattr(module, '__version__', 'unknown')
            test_results["dependencies"][name] = f"AVAILABLE - {version}"
            log_test("environment", "dependencies", name, "PASS", f"v{version}")
        except ImportError as e:
            test_results["dependencies"][name] = f"MISSING: {str(e)}"
            log_test("environment", "dependencies", name, "FAIL", f"Missing: {str(e)}")

def test_integrated_trading_engine():
    """Test the Integrated Kimera Engine"""
    logger.info("\n🧠 Testing Integrated Kimera Engine...")
    
    try:
        from backend.trading.core.integrated_trading_engine import (
            IntegratedTradingEngine, 
            create_integrated_trading_engine
        )
        log_test("component_tests", "IntegratedTradingEngine", "import", "PASS", "Module imported successfully")
        
        # Test engine creation
        try:
            engine = create_integrated_trading_engine(
                initial_balance=1000.0,
                risk_tolerance=0.05
            )
            log_test("component_tests", "IntegratedTradingEngine", "creation", "PASS", 
                    f"Engine created successfully")
            
            # Test market data processing
            sample_market_data = {
                'close': 50000.0,
                'volume': 1500.0,
                'bid': 49950.0,
                'ask': 50050.0,
                'volatility': 0.025,
                'news_sentiment': 0.1,
                'social_sentiment': 0.05,
                'price_history': [49800, 49900, 49950, 50000],
                'volume_history': [1400, 1450, 1500, 1550]
            }
            
            try:
                intelligence = engine.process_market_data(sample_market_data)
                log_test("component_tests", "IntegratedTradingEngine", "market_processing", "PASS", 
                        f"Market intelligence generated successfully")
                
                # Test signal generation
                try:
                    signal = engine.generate_enhanced_signal(sample_market_data, "BTC/USDT")
                    log_test("component_tests", "IntegratedTradingEngine", "signal_generation", "PASS", 
                            f"Signal: {signal.action} (conf: {signal.confidence:.2f})")
                    return True
                        
                except Exception as e:
                    log_test("component_tests", "IntegratedTradingEngine", "signal_generation", "FAIL", str(e))
                    return False
                    
            except Exception as e:
                log_test("component_tests", "IntegratedTradingEngine", "market_processing", "FAIL", str(e))
                return False
                
        except Exception as e:
            log_test("component_tests", "IntegratedTradingEngine", "creation", "FAIL", str(e))
            return False
            
    except ImportError as e:
        log_test("component_tests", "IntegratedTradingEngine", "import", "FAIL", str(e))
        return False

def generate_comprehensive_report():
    """Generate comprehensive test report"""
    logger.info("\n" + "=" * 70)
    logger.info("📋 COMPREHENSIVE TEST REPORT")
    logger.info("=" * 70)
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    
    # Count tests by category
    for category, components in test_results.items():
        if category in ["component_tests", "integration_tests", "performance_tests"]:
            for component, tests in components.items():
                for test_name, result in tests.items():
                    total_tests += 1
                    if result["status"] == "PASS":
                        passed_tests += 1
                    elif result["status"] == "FAIL":
                        failed_tests += 1
    
    # Overall assessment
    overall_success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    if overall_success_rate >= 85:
        status = "EXCELLENT"
        emoji = "🟢"
    elif overall_success_rate >= 70:
        status = "GOOD"
        emoji = "🟡"
    else:
        status = "POOR"
        emoji = "🔴"
    
    test_results["overall_status"] = status
    
    logger.info(f"{emoji} OVERALL SYSTEM STATUS: {status}")
    logger.info(f"Total Tests: {total_tests}")
    logger.info(f"Passed: {passed_tests} ({passed_tests/total_tests*100:.1f}%)
    logger.error(f"Failed: {failed_tests} ({failed_tests/total_tests*100:.1f}%)
    logger.info("=" * 70)
    
    return status

async def main():
    """Run comprehensive test suite"""
    logger.info("🚀 Starting Comprehensive Kimera Trading System Test...\n")
    
    try:
        # Environment tests
        test_environment()
        
        # Component tests
        test_integrated_trading_engine()
        
        # Generate final report
        final_status = generate_comprehensive_report()
        
        return final_status
        
    except Exception as e:
        logger.error(f"\n❌ Critical test failure: {e}")
        return "CRITICAL_FAILURE"

if __name__ == "__main__":
    # Run the comprehensive test suite
    result = asyncio.run(main())
    
    # Exit with appropriate code
    if result in ["EXCELLENT", "GOOD"]:
        logger.info(f"\n✅ Test completed successfully: {result}")
        sys.exit(0)
    else:
        logger.error(f"\n❌ Test failed: {result}")
        sys.exit(1) 