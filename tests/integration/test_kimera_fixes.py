#!/usr/bin/env python3
"""
Test script to verify Kimera trading system fixes
"""

import os
import sys
import tempfile
import time
from pathlib import Path


def test_logging_system():
    """Test that the logging system works without permission errors"""
    print("🔍 Testing process-safe logging system...")

    try:
        # Test importing the logger
        from src.utils.kimera_logger import get_system_logger, get_trading_logger

        # Create test loggers
        trading_logger = get_trading_logger("test_trading")
        system_logger = get_system_logger("test_system")

        # Test basic logging
        trading_logger.info("Test trading log message")
        system_logger.info("Test system log message")

        # Test structured logging
        trading_logger.info("Test with context", symbol="BTCUSDT", action="buy")

        print("✅ Logging system test passed")
        return True

    except Exception as e:
        print(f"❌ Logging system test failed: {e}")
        return False


def test_trading_configuration():
    """Test that trading system defaults to real trading"""
    print("🔍 Testing trading system configuration...")

    try:
        # Test LiveTradingConfig
        from src.trading.core.live_trading_manager import LiveTradingConfig

        config = LiveTradingConfig()

        # Check if real trading is enabled by default
        real_trading_enabled = not config.use_testnet
        print(f"Real trading enabled: {real_trading_enabled}")

        if real_trading_enabled:
            print("✅ Trading system defaults to real trading")
        else:
            print("⚠️ Trading system still in testnet mode")

        # Test environment variable override
        os.environ["KIMERA_USE_TESTNET"] = "true"
        config_testnet = LiveTradingConfig()
        testnet_enabled = config_testnet.use_testnet

        if testnet_enabled:
            print("✅ Environment variable override works")
        else:
            print("❌ Environment variable override failed")

        # Reset environment
        del os.environ["KIMERA_USE_TESTNET"]

        return real_trading_enabled and testnet_enabled

    except Exception as e:
        print(f"❌ Trading configuration test failed: {e}")
        return False


def test_exchange_connectors():
    """Test that exchange connectors default to real trading"""
    print("🔍 Testing exchange connector configuration...")

    try:
        # Test BinanceConnector
        from src.trading.api.binance_connector import BinanceConnector

        # Create test key path
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pem", delete=False) as f:
            f.write("test_key_content")
            test_key_path = f.name

        try:
            # Test default configuration (should be real trading)
            connector = BinanceConnector("test_api_key", test_key_path)

            # Check if it's using real URLs
            real_trading = connector.BASE_URL == "https://api.binance.com"

            if real_trading:
                print("✅ BinanceConnector defaults to real trading")
            else:
                print("❌ BinanceConnector still using testnet")

            return real_trading

        finally:
            # Clean up test file
            os.unlink(test_key_path)

    except Exception as e:
        print(f"❌ Exchange connector test failed: {e}")
        return False


def test_environment_control():
    """Test environment variable control"""
    print("🔍 Testing environment variable control...")

    try:
        # Test with KIMERA_USE_TESTNET=true
        os.environ["KIMERA_USE_TESTNET"] = "true"

        from src.trading.core.live_trading_manager import LiveTradingConfig

        config = LiveTradingConfig()

        testnet_enabled = config.use_testnet

        if testnet_enabled:
            print("✅ KIMERA_USE_TESTNET=true forces testnet mode")
        else:
            print("❌ Environment variable not working")

        # Test with KIMERA_USE_TESTNET=false
        os.environ["KIMERA_USE_TESTNET"] = "false"
        config = LiveTradingConfig()

        real_trading = not config.use_testnet

        if real_trading:
            print("✅ KIMERA_USE_TESTNET=false enables real trading")
        else:
            print("❌ Environment variable not working")

        # Clean up
        del os.environ["KIMERA_USE_TESTNET"]

        return testnet_enabled and real_trading

    except Exception as e:
        print(f"❌ Environment control test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🧪 KIMERA TRADING SYSTEM - FIXES VERIFICATION")
    print("=" * 50)

    # Run all tests
    tests = [
        ("Logging System", test_logging_system),
        ("Trading Configuration", test_trading_configuration),
        ("Exchange Connectors", test_exchange_connectors),
        ("Environment Control", test_environment_control),
    ]

    results = []
    for name, test_func in tests:
        print(f"\n📋 {name}:")
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ {name} test crashed: {e}")
            results.append(False)

    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY:")

    passed = sum(results)
    total = len(results)

    for i, (name, _) in enumerate(tests):
        status = "✅ PASSED" if results[i] else "❌ FAILED"
        print(f"  {name}: {status}")

    print(f"\n🎯 Overall: {passed}/{total} tests passed")

    if passed == total:
        print("🚀 ALL TESTS PASSED - SYSTEM READY FOR REAL TRADING!")
        return True
    else:
        print("⚠️ SOME TESTS FAILED - PLEASE REVIEW ISSUES")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
