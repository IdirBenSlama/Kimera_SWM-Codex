#!/usr/bin/env python3
"""
Complete System Test for Kimera Trading System
Comprehensive objective evaluation without real trading
"""

import asyncio
import sys
import json
import os
import time
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

logger.info("=" * 80)
logger.info("🚀 KIMERA TRADING SYSTEM - COMPLETE SYSTEM TEST")
logger.info("=" * 80)

# Global test results
results = {
    "test_start": datetime.now().isoformat(),
    "environment": {
        "python_version": sys.version.split()[0],
        "platform": sys.platform,
        "working_dir": os.getcwd()
    },
    "tests": {},
    "performance": {},
    "summary": {}
}

def test_log(category: str, test: str, status: str, details: str = "", duration: float = 0):
    """Log test results"""
    if category not in results["tests"]:
        results["tests"][category] = {}
    
    results["tests"][category][test] = {
        "status": status,
        "details": details,
        "duration": duration,
        "timestamp": datetime.now().isoformat()
    }
    
    icon = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
    time_str = f"({duration:.3f}s)" if duration > 0 else ""
    logger.info(f"{icon} [{category}] {test}: {status} {details} {time_str}")

async def test_integrated_trading_engine():
    """Test Integrated Kimera Engine"""
    logger.info("\n🧠 Testing Integrated Kimera Engine...")
    
    try:
        from backend.trading.core.integrated_trading_engine import create_integrated_trading_engine
        logger.info("✅ [KimeraEngine] import: PASS")
        
        # Test engine creation
        engine = create_integrated_trading_engine(initial_balance=1000.0, risk_tolerance=0.05)
        logger.info("✅ [KimeraEngine] creation: PASS")
        
        # Test market scenarios
        scenarios = [
            {
                "name": "bullish_market",
                "data": {
                    'close': 50000.0, 'volume': 2000.0, 'bid': 49950.0, 'ask': 50050.0,
                    'volatility': 0.02, 'news_sentiment': 0.8, 'social_sentiment': 0.7,
                    'price_history': [48000, 49000, 49500, 50000], 
                    'volume_history': [1800, 1900, 1950, 2000]
                }
            },
            {
                "name": "bearish_market", 
                "data": {
                    'close': 45000.0, 'volume': 2500.0, 'bid': 44950.0, 'ask': 45050.0,
                    'volatility': 0.05, 'news_sentiment': -0.6, 'social_sentiment': -0.4,
                    'price_history': [50000, 48000, 46000, 45000], 
                    'volume_history': [2000, 2200, 2400, 2500]
                }
            }
        ]
        
        for scenario in scenarios:
            try:
                intelligence = engine.process_market_data(scenario["data"])
                signal = engine.generate_enhanced_signal(scenario["data"], "BTC/USDT")
                logger.info(f"✅ [KimeraEngine] {scenario['name']}: PASS - Signal: {signal.action} (conf: {signal.confidence:.2f})
            except Exception as e:
                logger.error(f"❌ [KimeraEngine] {scenario['name']}: FAIL - {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ [KimeraEngine] import: FAIL - {e}")
        return False

def test_api_connectors():
    """Test API connector functionality"""
    logger.info("\n🔌 Testing API Connectors...")
    
    try:
        from backend.trading.api.binance_connector import BinanceConnector
        from backend.trading.api.phemex_connector import PhemexConnector
        test_log("API", "import", "PASS", "Connectors imported")
        
        # Test Binance connector
        binance_start = time.time()
        try:
            binance = BinanceConnector("test_key", "test_secret", testnet=True)
            
            # Test formatting functions
            qty = binance._format_quantity("BTCUSDT", 0.001234)
            price = binance._format_price("BTCUSDT", 50000.12345)
            
            test_log("API", "binance_formatting", "PASS", 
                    f"Qty: {qty}, Price: {price}", time.time() - binance_start)
            
        except Exception as e:
            test_log("API", "binance_formatting", "FAIL", str(e))
        
        # Test Phemex connector
        phemex_start = time.time()
        try:
            phemex = PhemexConnector("test_key", "test_secret", testnet=True)
            
            # Test scaling functions
            scaled_price = phemex.scale_price(50000.0)
            unscaled = phemex.unscale_price(scaled_price)
            
            test_log("API", "phemex_scaling", "PASS", 
                    f"Price: {50000.0} -> {scaled_price} -> {unscaled}", 
                    time.time() - phemex_start)
            
        except Exception as e:
            test_log("API", "phemex_scaling", "FAIL", str(e))
        
        return True
        
    except Exception as e:
        test_log("API", "import", "FAIL", str(e))
        return False

def test_intelligence_system():
    """Test intelligence gathering components"""
    logger.info("\n🧠 Testing Intelligence System...")
    
    # Test sentiment analyzer
    try:
        from backend.trading.intelligence.sentiment_analyzer import AdvancedSentimentAnalyzer
        test_log("Intelligence", "sentiment_import", "PASS", "Sentiment analyzer imported")
        
        analyzer_start = time.time()
        analyzer = AdvancedSentimentAnalyzer()
        
        # Test sentiment analysis on crypto-specific text
        test_texts = [
            "Bitcoin to the moon! 🚀 HODL diamond hands!",
            "Bear market confirmed. Time to sell everything.",
            "Steady growth in DeFi. Looking bullish long term.",
            "FUD everywhere but staying strong. Diamond hands!",
            "New ATH incoming! Best investment decision ever!"
        ]
        
        sentiment_results = []
        for text in test_texts:
            result = analyzer.analyze_text(text, "social_media")
            sentiment_results.append(result)
        
        avg_confidence = np.mean([r['confidence'] for r in sentiment_results])
        test_log("Intelligence", "sentiment_analysis", "PASS", 
                f"Analyzed {len(test_texts)} texts, avg confidence: {avg_confidence:.2f}",
                time.time() - analyzer_start)
        
    except Exception as e:
        test_log("Intelligence", "sentiment_analysis", "FAIL", str(e))
    
    # Test data collector
    try:
        from backend.trading.intelligence.live_data_collector import SimulatedDataCollector
        test_log("Intelligence", "data_collector_import", "PASS", "Data collector imported")
        
        data_start = time.time()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        simulated_data = loop.run_until_complete(SimulatedDataCollector.get_simulated_intelligence())
        loop.close()
        
        test_log("Intelligence", "data_simulation", "PASS", 
                f"Market: {simulated_data['market_condition']}", 
                time.time() - data_start)
        
    except Exception as e:
        test_log("Intelligence", "data_simulation", "FAIL", str(e))

async def test_trading_engine():
    """Test core trading engine"""
    logger.info("\n⚡ Testing Core Trading Engine...")
    
    try:
        from backend.trading.core.trading_engine import KimeraTradingEngine
        test_log("TradingEngine", "import", "PASS", "Trading engine imported")
        
        engine_start = time.time()
        config = {"testnet": True, "initial_balance": 1000.0}
        engine = KimeraTradingEngine(config)
        
        # Test market analysis
        market_data = {
            "price": 50000.0, "volume": 1500.0, "bid": 49950.0, "ask": 50050.0,
            "price_history": [49000, 49500, 49800, 50000],
            "volume_history": [1200, 1300, 1400, 1500],
            "sentiment": "neutral", "order_book_imbalance": 0.1
        }
        
        analysis_start = time.time()
        market_state = await engine.analyze_market("BTCUSDT", market_data)
        
        test_log("TradingEngine", "market_analysis", "PASS", 
                f"Cognitive pressure: {market_state.cognitive_pressure:.3f}",
                time.time() - analysis_start)
        
        # Test decision making
        portfolio_state = {
            "free_balance": 1000.0, "total_value": 1000.0, "positions": {},
            "daily_pnl": 0.0, "margin_used": 0.0
        }
        
        decision_start = time.time()
        decision = await engine.make_trading_decision("BTCUSDT", market_state, portfolio_state)
        
        test_log("TradingEngine", "decision_making", "PASS", 
                f"Decision: {decision.action} (conf: {decision.confidence:.2f})",
                time.time() - decision_start)
        
        test_log("TradingEngine", "creation", "PASS", "Engine operational", 
                time.time() - engine_start)
        
        return True
        
    except Exception as e:
        test_log("TradingEngine", "import", "FAIL", str(e))
        return False

def test_strategy_components():
    """Test strategy components"""
    logger.info("\n🎯 Testing Strategy Components...")
    
    # Test Small Balance Optimizer
    try:
        from backend.trading.strategies.small_balance_optimizer import SmallBalanceOptimizer
        test_log("Strategy", "sbo_import", "PASS", "Small Balance Optimizer imported")
        
        sbo_start = time.time()
        sbo = SmallBalanceOptimizer(0.001)  # 0.001 BTC
        
        growth_potential = sbo.calculate_growth_potential(30)
        inspiration = sbo.get_inspiration()
        
        test_log("Strategy", "sbo_functionality", "PASS", 
                f"Growth target: {growth_potential['realistic_target']}", 
                time.time() - sbo_start)
        
    except Exception as e:
        test_log("Strategy", "sbo_functionality", "FAIL", str(e))
    
    # Test Strategic Warfare Engine
    try:
        from backend.trading.strategies.strategic_warfare_engine import ProfitTradingEngine
        test_log("Strategy", "swe_import", "PASS", "Strategic Warfare Engine imported")
        
        swe_start = time.time()
        swe = ProfitTradingEngine(1000.0)
        
        sample_market = {
            "price": 50000.0, "volume": 1500.0, "trend": "bullish", "volatility": 0.025
        }
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        analysis = loop.run_until_complete(swe.analyze_market(sample_market))
        loop.close()
        
        test_log("Strategy", "swe_functionality", "PASS", 
                f"Trend strength: {analysis.trend_strength:.2f}", 
                time.time() - swe_start)
        
    except Exception as e:
        test_log("Strategy", "swe_functionality", "FAIL", str(e))

def performance_benchmark():
    """Run performance benchmarks"""
    logger.info("\n⚡ Running Performance Benchmarks...")
    
    try:
        from backend.trading.core.integrated_trading_engine import create_integrated_trading_engine
        
        benchmark_start = time.time()
        engine = create_integrated_trading_engine(initial_balance=1000.0)
        
        # Benchmark signal generation speed
        sample_data = {
            'close': 50000.0, 'volume': 1500.0, 'bid': 49950.0, 'ask': 50050.0,
            'volatility': 0.025, 'news_sentiment': 0.1, 'social_sentiment': 0.05,
            'price_history': [49800, 49900, 49950, 50000],
            'volume_history': [1400, 1450, 1500, 1550]
        }
        
        # Time signal generation
        signal_times = []
        signals = []
        
        for i in range(50):  # 50 signal generations
            varied_data = sample_data.copy()
            varied_data['close'] = 50000 + np.random.normal(0, 100)
            varied_data['volume'] = 1500 + np.random.normal(0, 200)
            
            signal_start = time.time()
            signal = engine.generate_enhanced_signal(varied_data, "BTC/USDT")
            signal_time = time.time() - signal_start
            
            signal_times.append(signal_time)
            signals.append(signal)
        
        # Calculate performance metrics
        avg_signal_time = np.mean(signal_times)
        max_signal_time = np.max(signal_times)
        signals_per_second = 1.0 / avg_signal_time
        
        results["performance"]["benchmark"] = {
            "avg_signal_time": avg_signal_time,
            "max_signal_time": max_signal_time,
            "signals_per_second": signals_per_second,
            "total_signals": len(signals)
        }
        
        test_log("Performance", "signal_speed", "PASS", 
                f"{signals_per_second:.1f} signals/sec (avg: {avg_signal_time:.3f}s)", 
                time.time() - benchmark_start)
        
        # Analyze signal quality
        actions = [s.action for s in signals]
        confidences = [s.confidence for s in signals]
        
        action_distribution = dict(pd.Series(actions).value_counts())
        avg_confidence = np.mean(confidences)
        
        test_log("Performance", "signal_quality", "PASS", 
                f"Avg confidence: {avg_confidence:.2f}, Distribution: {action_distribution}")
        
        return True
        
    except Exception as e:
        test_log("Performance", "benchmark", "FAIL", str(e))
        return False

async def integration_test():
    """Test full system integration"""
    logger.info("\n🔗 Testing System Integration...")
    
    try:
        from backend.trading.core.integrated_trading_engine import create_integrated_trading_engine
        from backend.trading.intelligence.live_data_collector import SimulatedDataCollector
        
        integration_start = time.time()
        
        # Create engine
        engine = create_integrated_trading_engine(initial_balance=1000.0)
        
        # Get simulated market data
        market_data = await SimulatedDataCollector.get_simulated_intelligence()
        
        # Create trading-compatible data
        trading_data = {
            'close': 50000.0, 'volume': 1500.0, 'bid': 49950.0, 'ask': 50050.0,
            'volatility': 0.025, 
            'news_sentiment': market_data['intelligence']['news_sentiment'],
            'social_sentiment': market_data['intelligence']['reddit_activity']['sentiment'],
            'price_history': [49800, 49900, 49950, 50000],
            'volume_history': [1400, 1450, 1500, 1550]
        }
        
        # Process full pipeline
        intelligence = engine.process_market_data(trading_data)
        signal = engine.generate_enhanced_signal(trading_data, "BTC/USDT")
        
        test_log("Integration", "full_pipeline", "PASS", 
                f"Market: {market_data['market_condition']}, Signal: {signal.action}",
                time.time() - integration_start)
        
        return True
        
    except Exception as e:
        test_log("Integration", "full_pipeline", "FAIL", str(e))
        return False

def generate_final_report():
    """Generate comprehensive final report"""
    logger.info("\n" + "=" * 80)
    logger.info("📊 FINAL SYSTEM EVALUATION REPORT")
    logger.info("=" * 80)
    
    # Calculate overall statistics
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    warned_tests = 0
    
    for category, tests in results["tests"].items():
        for test_name, result in tests.items():
            total_tests += 1
            if result["status"] == "PASS":
                passed_tests += 1
            elif result["status"] == "FAIL":
                failed_tests += 1
            elif result["status"] == "WARN":
                warned_tests += 1
    
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    # Overall system status
    if success_rate >= 90:
        status = "EXCELLENT"
        emoji = "🟢"
        verdict = "System is production-ready with outstanding performance"
    elif success_rate >= 75:
        status = "GOOD"
        emoji = "🟡"
        verdict = "System is functional with minor issues"
    elif success_rate >= 50:
        status = "ACCEPTABLE"
        emoji = "🟠"
        verdict = "System has issues but core functionality works"
    else:
        status = "POOR"
        emoji = "🔴"
        verdict = "System has critical issues requiring attention"
    
    results["summary"] = {
        "overall_status": status,
        "success_rate": success_rate,
        "total_tests": total_tests,
        "passed": passed_tests,
        "failed": failed_tests,
        "warned": warned_tests,
        "verdict": verdict
    }
    
    # Print summary
    logger.info(f"{emoji} OVERALL STATUS: {status} ({success_rate:.1f}% success rate)
    logger.error(f"📋 TEST RESULTS: {passed_tests} passed, {failed_tests} failed, {warned_tests} warnings")
    logger.info(f"🎯 VERDICT: {verdict}")
    
    # Component breakdown
    logger.info(f"\n📋 Component Breakdown:")
    for category, tests in results["tests"].items():
        cat_passed = sum(1 for t in tests.values() if t["status"] == "PASS")
        cat_total = len(tests)
        cat_rate = (cat_passed / cat_total * 100) if cat_total > 0 else 0
        
        cat_icon = "✅" if cat_rate >= 80 else "⚠️" if cat_rate >= 50 else "❌"
        logger.info(f"  {cat_icon} {category}: {cat_passed}/{cat_total} ({cat_rate:.1f}%)
    
    # Performance metrics
    if "performance" in results and "benchmark" in results["performance"]:
        perf = results["performance"]["benchmark"]
        logger.info(f"\n⚡ Performance Metrics:")
        logger.info(f"  • Signal Generation: {perf['signals_per_second']:.1f} signals/sec")
        logger.info(f"  • Average Latency: {perf['avg_signal_time']:.3f} seconds")
        logger.info(f"  • Peak Latency: {perf['max_signal_time']:.3f} seconds")
    
    # Save results
    results["test_end"] = datetime.now().isoformat()
    try:
        with open("kimera_complete_test_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"\n📄 Detailed results saved to: kimera_complete_test_results.json")
    except Exception as e:
        logger.warning(f"\n⚠️ Could not save results: {e}")
    
    logger.info("=" * 80)
    return status

async def main():
    """Run complete system test"""
    logger.info("🚀 Kimera Trading System - Complete Objective Evaluation")
    logger.info(f"📅 Started: {datetime.now()
    logger.info(f"🐍 Python: {sys.version.split()
    logger.info(f"💻 Platform: {sys.platform}")
    
    try:
        # Core component tests
        await test_integrated_trading_engine()
        test_api_connectors()
        test_intelligence_system()
        await test_trading_engine()
        test_strategy_components()
        
        # Performance benchmarks
        performance_benchmark()
        
        # Integration test
        await integration_test()
        
        # Generate final report
        final_status = generate_final_report()
        
        return final_status
        
    except Exception as e:
        logger.error(f"\n❌ Critical system failure: {e}")
        test_log("System", "critical_failure", "FAIL", str(e))
        return "CRITICAL_FAILURE"

if __name__ == "__main__":
    start_time = time.time()
    result = asyncio.run(main())
    total_time = time.time() - start_time
    
    logger.info(f"\n🏁 Test completed in {total_time:.2f} seconds")
    logger.info(f"🎯 Final Result: {result}")
    
    # Exit with appropriate code
    if result in ["EXCELLENT", "GOOD"]:
        sys.exit(0)
    elif result == "ACCEPTABLE":
        sys.exit(1)
    else:
        sys.exit(2) 