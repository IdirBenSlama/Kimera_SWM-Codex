#!/usr/bin/env python3
"""
Objective Test Suite for Kimera Trading System
Tests all components without requiring API keys or making real trades
"""

import asyncio
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any
import warnings

# Initialize structured logger
from backend.utils.kimera_logger import get_system_logger
logger = get_system_logger(__name__)

warnings.filterwarnings('ignore')

logger.info("=" * 60)
logger.info("🧪 KIMERA TRADING SYSTEM - OBJECTIVE TEST SUITE")
logger.info("=" * 60)

# Test Results Storage
test_results = {
    "timestamp": datetime.now().isoformat(),
    "environment": {
        "python_version": sys.version,
        "platform": sys.platform
    },
    "dependencies": {},
    "component_tests": {},
    "overall_status": "UNKNOWN"
}

def log_test(component: str, test_name: str, status: str, details: str = ""):
    """Log test results"""
    if component not in test_results["component_tests"]:
        test_results["component_tests"][component] = {}
    
    test_results["component_tests"][component][test_name] = {
        "status": status,
        "details": details,
        "timestamp": datetime.now().isoformat()
    }
    
    status_icon = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
    logger.info(f"{status_icon} {component}.{test_name}: {status} {details}")

def test_dependencies():
    """Test all required dependencies"""
    logger.info("\n📦 Testing Dependencies...")
    
    dependencies = [
        ("numpy", "np"),
        ("pandas", "pd"),
        ("sklearn", "sklearn"),
        ("asyncio", "asyncio"),
        ("datetime", "datetime"),
        ("decimal", "decimal"),
        ("dataclasses", "dataclasses"),
        ("enum", "enum"),
        ("typing", "typing"),
        ("json", "json"),
        ("logging", "logging")
    ]
    
    # Test core dependencies
    passed = 0
    total = len(dependencies)
    
    for name, import_name in dependencies:
        try:
            __import__(import_name)
            test_results["dependencies"][name] = "AVAILABLE"
            log_test("Dependencies", name, "PASS", "Core dependency available")
            passed += 1
        except ImportError as e:
            test_results["dependencies"][name] = f"MISSING: {str(e)}"
            log_test("Dependencies", name, "FAIL", f"Missing: {str(e)}")
    
    return passed == total

def test_integrated_trading_engine():
    """Test the Integrated Kimera Engine"""
    logger.info("\n🧠 Testing Integrated Kimera Engine...")
    
    try:
        from core.integrated_trading_engine import IntegratedTradingEngine, create_integrated_trading_engine
        log_test("IntegratedTradingEngine", "import", "PASS", "Module imported successfully")
        
        # Test engine creation
        try:
            engine = create_integrated_trading_engine(
                initial_balance=1000.0,
                risk_tolerance=0.05
            )
            log_test("IntegratedTradingEngine", "creation", "PASS", "Engine created successfully")
            
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
                log_test("IntegratedTradingEngine", "market_processing", "PASS", 
                        f"Processed market data successfully. Price: {intelligence.price}")
                
                # Test signal generation
                try:
                    signal = engine.generate_enhanced_signal(sample_market_data, "BTC/USDT")
                    log_test("IntegratedTradingEngine", "signal_generation", "PASS", 
                            f"Generated signal: {signal.action} (conf: {signal.confidence:.2f})")
                    
                    return True
                        
                except Exception as e:
                    log_test("IntegratedTradingEngine", "signal_generation", "FAIL", str(e))
                    return False
                    
            except Exception as e:
                log_test("IntegratedTradingEngine", "market_processing", "FAIL", str(e))
                return False
                
        except Exception as e:
            log_test("IntegratedTradingEngine", "creation", "FAIL", str(e))
            return False
            
    except ImportError as e:
        log_test("IntegratedTradingEngine", "import", "FAIL", str(e))
        return False

def test_anomaly_detector():
    """Test the Enhanced Anomaly Detector"""
    logger.debug("\n🔍 Testing Enhanced Anomaly Detector...")
    
    try:
        from intelligence.enhanced_anomaly_detector import create_enhanced_detector
        log_test("AnomalyDetector", "import", "PASS", "Module imported successfully")
        
        try:
            detector = create_enhanced_detector()
            if detector:
                log_test("AnomalyDetector", "creation", "PASS", "Detector created successfully")
                
                # Test market anomaly detection
                sample_data = pd.DataFrame({
                    'close': np.random.randn(50).cumsum() + 100,
                    'volume': np.random.exponential(1000, 50),
                    'bid': np.random.randn(50).cumsum() + 99.5,
                    'ask': np.random.randn(50).cumsum() + 100.5
                })
                
                # Introduce anomalies
                sample_data.loc[25, 'close'] = sample_data.loc[25, 'close'] * 1.2  # Price spike
                sample_data.loc[40, 'volume'] = sample_data.loc[40, 'volume'] * 15  # Volume spike
                
                try:
                    anomalies = detector.detect_market_anomalies(sample_data)
                    log_test("AnomalyDetector", "market_detection", "PASS", 
                            f"Detected {len(anomalies)} market anomalies")
                    
                    # Test execution anomaly detection
                    sample_execution = {
                        'execution_time': 0.8,  # High execution time
                        'expected_price': 100.0,
                        'actual_price': 100.3,  # High slippage
                        'order_size': 1000
                    }
                    
                    try:
                        exec_anomalies = detector.detect_execution_anomalies(sample_execution)
                        log_test("AnomalyDetector", "execution_detection", "PASS", 
                                f"Detected {len(exec_anomalies)} execution anomalies")
                        
                        # Test portfolio anomaly detection
                        sample_portfolio = {
                            'total_exposure': 1.2,  # Over-exposure
                            'concentration_ratio': 0.7,  # High concentration
                            'positions': {'BTC': 0.7, 'ETH': 0.3}
                        }
                        
                        try:
                            portfolio_anomalies = detector.detect_portfolio_anomalies(sample_portfolio)
                            log_test("AnomalyDetector", "portfolio_detection", "PASS", 
                                    f"Detected {len(portfolio_anomalies)} portfolio anomalies")
                            
                            # Test summary
                            try:
                                summary = detector.get_anomaly_summary()
                                log_test("AnomalyDetector", "summary", "PASS", 
                                        f"System status: {summary['system_status']}")
                                return True
                            except Exception as e:
                                log_test("AnomalyDetector", "summary", "FAIL", str(e))
                                return False
                                
                        except Exception as e:
                            log_test("AnomalyDetector", "portfolio_detection", "FAIL", str(e))
                            return False
                            
                    except Exception as e:
                        log_test("AnomalyDetector", "execution_detection", "FAIL", str(e))
                        return False
                        
                except Exception as e:
                    log_test("AnomalyDetector", "market_detection", "FAIL", str(e))
                    return False
                    
            else:
                log_test("AnomalyDetector", "creation", "FAIL", "Detector creation returned None (missing deps)")
                return False
                
        except Exception as e:
            log_test("AnomalyDetector", "creation", "FAIL", str(e))
            return False
            
    except ImportError as e:
        log_test("AnomalyDetector", "import", "FAIL", str(e))
        return False

def test_portfolio_optimizer():
    """Test the Portfolio Optimizer"""
    logger.info("\n📊 Testing Portfolio Optimizer...")
    
    try:
        from optimization.portfolio_optimizer import create_portfolio_optimizer
        log_test("PortfolioOptimizer", "import", "PASS", "Module imported successfully")
        
        try:
            optimizer = create_portfolio_optimizer()
            log_test("PortfolioOptimizer", "creation", "PASS", "Optimizer created successfully")
            
            # Test price data update
            try:
                test_prices = {
                    'BTC/USDT': 50000.0,
                    'ETH/USDT': 3000.0,
                    'ADA/USDT': 1.5
                }
                
                optimizer.update_price_data(test_prices, pd.Timestamp.now())
                log_test("PortfolioOptimizer", "price_update", "PASS", "Price data updated successfully")
                
                # Add more price points for proper optimization
                for i in range(10):
                    test_prices_variant = {
                        'BTC/USDT': 50000.0 + np.random.normal(0, 1000),
                        'ETH/USDT': 3000.0 + np.random.normal(0, 100),
                        'ADA/USDT': 1.5 + np.random.normal(0, 0.1)
                    }
                    optimizer.update_price_data(test_prices_variant, 
                                              pd.Timestamp.now() + pd.Timedelta(days=i))
                
                # Test portfolio optimization
                try:
                    assets = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT']
                    result = optimizer.optimize_portfolio(assets)
                    
                    log_test("PortfolioOptimizer", "optimization", "PASS", 
                            f"Optimized portfolio - Sharpe: {result.sharpe_ratio:.2f}")
                    return True
                    
                except Exception as e:
                    log_test("PortfolioOptimizer", "optimization", "FAIL", str(e))
                    return False
                    
            except Exception as e:
                log_test("PortfolioOptimizer", "price_update", "FAIL", str(e))
                return False
                
        except Exception as e:
            log_test("PortfolioOptimizer", "creation", "FAIL", str(e))
            return False
            
    except ImportError as e:
        log_test("PortfolioOptimizer", "import", "FAIL", str(e))
        return False

def test_sentiment_analyzer():
    """Test the Sentiment Analyzer"""
    logger.info("\n💭 Testing Sentiment Analyzer...")
    
    try:
        from intelligence.sentiment_analyzer import AdvancedSentimentAnalyzer, CryptoSpecificSentiment
        log_test("SentimentAnalyzer", "import", "PASS", "Module imported successfully")
        
        try:
            analyzer = AdvancedSentimentAnalyzer()
            log_test("SentimentAnalyzer", "creation", "PASS", "Analyzer created successfully")
            
            # Test text analysis
            test_texts = [
                "Bitcoin is going to the moon! 🚀 Diamond hands!",
                "Market crash incoming, time to sell everything",
                "Steady growth in crypto markets, looking bullish",
                "FUD spreading everywhere, but HODL strong",
                "New ATH for BTC, incredible gains this week!"
            ]
            
            try:
                results = []
                for text in test_texts:
                    result = analyzer.analyze_text(text, "social_media")
                    results.append(result)
                
                log_test("SentimentAnalyzer", "text_analysis", "PASS", 
                        f"Analyzed {len(results)} texts successfully")
                
                # Test multiple text analysis
                try:
                    text_dicts = [{"text": text, "source": "twitter"} for text in test_texts]
                    multi_result = analyzer.analyze_multiple_texts(text_dicts)
                    
                    log_test("SentimentAnalyzer", "multi_text_analysis", "PASS", 
                            f"Multi-text analysis: {multi_result['market_signal']}")
                    
                    # Test crypto-specific sentiment
                    try:
                        crypto_result = CryptoSpecificSentiment.analyze_crypto_sentiment(
                            "HODL strong! To the moon! 💎🙌 Buy the dip!"
                        )
                        
                        log_test("SentimentAnalyzer", "crypto_sentiment", "PASS", 
                                f"Crypto sentiment: {crypto_result['emotion']}")
                        return True
                        
                    except Exception as e:
                        log_test("SentimentAnalyzer", "crypto_sentiment", "FAIL", str(e))
                        return False
                        
                except Exception as e:
                    log_test("SentimentAnalyzer", "multi_text_analysis", "FAIL", str(e))
                    return False
                    
            except Exception as e:
                log_test("SentimentAnalyzer", "text_analysis", "FAIL", str(e))
                return False
                
        except Exception as e:
            log_test("SentimentAnalyzer", "creation", "FAIL", str(e))
            return False
            
    except ImportError as e:
        log_test("SentimentAnalyzer", "import", "FAIL", str(e))
        return False

def test_data_collector():
    """Test the Live Data Collector (simulated mode)"""
    logger.info("\n📡 Testing Data Collector...")
    
    try:
        from intelligence.live_data_collector import SimulatedDataCollector
        log_test("DataCollector", "import", "PASS", "Module imported successfully")
        
        try:
            # Test simulated data collection
            simulated_data = SimulatedDataCollector.get_simulated_intelligence()
            
            # Since this is async, we need to run it properly
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(simulated_data)
            loop.close()
            
            log_test("DataCollector", "simulation", "PASS", 
                    f"Simulated data generated - Market: {result['market_condition']}")
            return True
            
        except Exception as e:
            log_test("DataCollector", "simulation", "FAIL", str(e))
            return False
            
    except ImportError as e:
        log_test("DataCollector", "import", "FAIL", str(e))
        return False

def test_api_connectors():
    """Test API Connectors (without actual connections)"""
    logger.info("\n🔌 Testing API Connectors...")
    
    try:
        from api.binance_connector import BinanceConnector
        from api.phemex_connector import PhemexConnector
        log_test("APIConnectors", "import", "PASS", "Connector modules imported successfully")
        
        try:
            # Test connector initialization (testnet mode)
            binance = BinanceConnector("test_key", "test_secret", testnet=True)
            phemex = PhemexConnector("test_key", "test_secret", testnet=True)
            
            log_test("APIConnectors", "initialization", "PASS", "Connectors initialized successfully")
            
            # Test helper methods
            try:
                # Test Binance helper methods
                formatted_qty = binance._format_quantity("BTCUSDT", 0.001234)
                formatted_price = binance._format_price("BTCUSDT", 50000.12)
                
                # Test Phemex helper methods
                scaled_price = phemex.scale_price(50000.0)
                unscaled_price = phemex.unscale_price(scaled_price)
                
                log_test("APIConnectors", "helper_methods", "PASS", 
                        f"Helper methods working - Scaled price: {scaled_price}")
                return True
                
            except Exception as e:
                log_test("APIConnectors", "helper_methods", "FAIL", str(e))
                return False
                
        except Exception as e:
            log_test("APIConnectors", "initialization", "FAIL", str(e))
            return False
            
    except ImportError as e:
        log_test("APIConnectors", "import", "FAIL", str(e))
        return False

def test_performance_tracker():
    """Test the Performance Tracker"""
    logger.info("\n📈 Testing Performance Tracker...")
    
    try:
        from monitoring.performance_tracker import PerformanceTracker
        log_test("PerformanceTracker", "import", "PASS", "Module imported successfully")
        
        try:
            tracker = PerformanceTracker()
            log_test("PerformanceTracker", "creation", "PASS", "Tracker created successfully")
            
            # Test performance calculation (with empty data)
            try:
                metrics = tracker.calculate_metrics()
                summary = tracker.get_performance_summary()
                
                log_test("PerformanceTracker", "calculations", "PASS", 
                        f"Calculated metrics - Total trades: {metrics.total_trades}")
                return True
                
            except Exception as e:
                log_test("PerformanceTracker", "calculations", "FAIL", str(e))
                return False
                
        except Exception as e:
            log_test("PerformanceTracker", "creation", "FAIL", str(e))
            return False
            
    except ImportError as e:
        log_test("PerformanceTracker", "import", "FAIL", str(e))
        return False

async def test_trading_engine():
    """Test the Core Trading Engine"""
    logger.info("\n⚡ Testing Core Trading Engine...")
    
    try:
        from core.trading_engine import KimeraTradingEngine, MarketState, TradingDecision
        log_test("TradingEngine", "import", "PASS", "Module imported successfully")
        
        try:
            config = {"testnet": True, "initial_balance": 1000.0}
            engine = KimeraTradingEngine(config)
            log_test("TradingEngine", "creation", "PASS", "Engine created successfully")
            
            # Test market analysis
            try:
                sample_market_data = {
                    "price": 50000.0,
                    "volume": 1500.0,
                    "bid": 49950.0,
                    "ask": 50050.0,
                    "price_history": [49000, 49500, 49800, 50000],
                    "volume_history": [1200, 1300, 1400, 1500],
                    "sentiment": "neutral",
                    "order_book_imbalance": 0.1
                }
                
                market_state = await engine.analyze_market("BTCUSDT", sample_market_data)
                log_test("TradingEngine", "market_analysis", "PASS", 
                        f"Market analyzed - Cognitive pressure: {market_state.cognitive_pressure:.2f}")
                
                # Test trading decision
                try:
                    portfolio_state = {
                        "free_balance": 1000.0,
                        "total_value": 1000.0,
                        "positions": {},
                        "daily_pnl": 0.0,
                        "margin_used": 0.0
                    }
                    
                    decision = await engine.make_trading_decision("BTCUSDT", market_state, portfolio_state)
                    log_test("TradingEngine", "decision_making", "PASS", 
                            f"Decision made: {decision.action} (confidence: {decision.confidence:.2f})")
                    return True
                    
                except Exception as e:
                    log_test("TradingEngine", "decision_making", "FAIL", str(e))
                    return False
                    
            except Exception as e:
                log_test("TradingEngine", "market_analysis", "FAIL", str(e))
                return False
                
        except Exception as e:
            log_test("TradingEngine", "creation", "FAIL", str(e))
            return False
            
    except ImportError as e:
        log_test("TradingEngine", "import", "FAIL", str(e))
        return False

def test_strategic_components():
    """Test Strategic Components"""
    logger.info("\n🎯 Testing Strategic Components...")
    
    try:
        from strategies.small_balance_optimizer import SmallBalanceOptimizer
        from strategies.strategic_warfare_engine import ProfitTradingEngine
        log_test("StrategicComponents", "import", "PASS", "Strategy modules imported successfully")
        
        try:
            # Test Small Balance Optimizer
            sbo = SmallBalanceOptimizer(0.001)  # Small BTC amount
            
            growth_potential = sbo.calculate_growth_potential(30)
            inspiration = sbo.get_inspiration()
            
            log_test("StrategicComponents", "small_balance_optimizer", "PASS", 
                    f"SBO working - Growth potential: {growth_potential['realistic_target']}")
            
            # Test Strategic Warfare Engine
            swe = ProfitTradingEngine(1000.0)
            
            sample_market = {
                "price": 50000.0,
                "volume": 1500.0,
                "trend": "bullish",
                "volatility": 0.025
            }
            
            # Run async market analysis
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            analysis = loop.run_until_complete(swe.analyze_market(sample_market))
            loop.close()
            
            log_test("StrategicComponents", "strategic_warfare_engine", "PASS", 
                    f"SWE working - Trend strength: {analysis.trend_strength:.2f}")
            return True
            
        except Exception as e:
            log_test("StrategicComponents", "strategy_execution", "FAIL", str(e))
            return False
            
    except ImportError as e:
        log_test("StrategicComponents", "import", "FAIL", str(e))
        return False

def generate_final_report():
    """Generate final test report"""
    logger.info("\n" + "=" * 60)
    logger.info("📋 FINAL TEST REPORT")
    logger.info("=" * 60)
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    
    for component, tests in test_results["component_tests"].items():
        component_passed = 0
        component_total = len(tests)
        
        for test_name, result in tests.items():
            total_tests += 1
            if result["status"] == "PASS":
                passed_tests += 1
                component_passed += 1
            elif result["status"] == "FAIL":
                failed_tests += 1
        
        success_rate = (component_passed / component_total * 100) if component_total > 0 else 0
        status_icon = "✅" if success_rate >= 80 else "⚠️" if success_rate >= 50 else "❌"
        
        logger.info(f"{status_icon} {component}: {component_passed}/{component_total} tests passed ({success_rate:.1f}%)
    
    overall_success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    if overall_success_rate >= 80:
        test_results["overall_status"] = "EXCELLENT"
        status_emoji = "🟢"
    elif overall_success_rate >= 60:
        test_results["overall_status"] = "GOOD"
        status_emoji = "🟡"
    else:
        test_results["overall_status"] = "POOR"
        status_emoji = "🔴"
    
    logger.info("\n" + "=" * 60)
    logger.info(f"{status_emoji} OVERALL SYSTEM STATUS: {test_results['overall_status']}")
    logger.info(f"Total Tests: {total_tests}")
    logger.info(f"Passed: {passed_tests} ({passed_tests/total_tests*100:.1f}%)
    logger.error(f"Failed: {failed_tests} ({failed_tests/total_tests*100:.1f}%)
    logger.info("=" * 60)
    
    return test_results["overall_status"]

async def main():
    """Run comprehensive test suite"""
    logger.info("Starting comprehensive test suite...\n")
    
    # Test 1: Dependencies
    deps_ok = test_dependencies()
    
    # Test 2: Integrated Kimera Engine
    kimera_ok = test_integrated_trading_engine()
    
    # Test 3: Anomaly Detector
    anomaly_ok = test_anomaly_detector()
    
    # Test 4: Portfolio Optimizer
    portfolio_ok = test_portfolio_optimizer()
    
    # Test 5: Sentiment Analyzer
    sentiment_ok = test_sentiment_analyzer()
    
    # Test 6: Data Collector
    data_ok = test_data_collector()
    
    # Test 7: API Connectors
    api_ok = test_api_connectors()
    
    # Test 8: Performance Tracker
    perf_ok = test_performance_tracker()
    
    # Test 9: Trading Engine
    engine_ok = await test_trading_engine()
    
    # Test 10: Strategic Components
    strategy_ok = test_strategic_components()
    
    # Generate final report
    final_status = generate_final_report()
    
    return final_status

if __name__ == "__main__":
    # Run the test suite
    result = asyncio.run(main())
    
    # Exit with appropriate code
    if result in ["EXCELLENT", "GOOD"]:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure 