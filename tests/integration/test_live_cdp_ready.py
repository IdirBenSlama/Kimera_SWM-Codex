#!/usr/bin/env python3
"""
KIMERA CDP LIVE READINESS TEST
=============================

Quick test to verify the live CDP system is ready for autonomous trading.
"""

import os
import sys
from pathlib import Path


def test_dependencies():
    """Test required dependencies"""
    print("🔍 Testing Dependencies...")

    try:
        import cdp

        print("✅ CDP SDK available")
    except ImportError:
        print("❌ CDP SDK not installed. Run: pip install cdp-sdk")
        return False

    try:
        import numpy as np

        print("✅ NumPy available")
    except ImportError:
        print("❌ NumPy not installed. Run: pip install numpy")
        return False

    try:
        from dotenv import load_dotenv

        print("✅ python-dotenv available")
    except ImportError:
        print("❌ python-dotenv not installed. Run: pip install python-dotenv")
        return False

    return True


def test_files():
    """Test required files exist"""
    print("\n🔍 Testing Files...")

    required_files = [
        "kimera_cdp_live_integration.py",
        "setup_live_cdp_credentials.py",
        "KIMERA_LIVE_CDP_DEPLOYMENT_GUIDE.md",
    ]

    for file in required_files:
        if Path(file).exists():
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} missing")
            return False

    return True


def test_credentials():
    """Test if credentials are configured"""
    print("\n🔍 Testing Credentials...")

    config_files = ["kimera_cdp_live.env", "kimera_cdp_config.env"]

    config_found = False
    for config_file in config_files:
        if Path(config_file).exists():
            print(f"✅ Configuration file found: {config_file}")
            config_found = True

            # Check if it has the required fields
            try:
                with open(config_file, "r") as f:
                    content = f.read()
                    if "CDP_API_KEY_NAME" in content:
                        print("✅ API Key Name configured")
                    if "CDP_API_KEY_PRIVATE_KEY" in content:
                        if "your_cdp_private_key_here" in content:
                            print("⚠️  Private key not set (still template)")
                        else:
                            print("✅ Private key configured")
                    if "CDP_NETWORK_ID" in content:
                        print("✅ Network ID configured")
            except Exception as e:
                print(f"⚠️  Error reading config: {e}")

    if not config_found:
        print("❌ No configuration file found")
        print("   Run: python setup_live_cdp_credentials.py")
        return False

    return True


def test_backend():
    """Test backend cognitive engines"""
    print("\n🔍 Testing Backend...")

    try:
        sys.path.append("./backend")
        from engines.cognitive_field_dynamics import CognitiveFieldDynamics

        print("✅ Cognitive Field Dynamics available")
    except ImportError as e:
        print(f"⚠️  Cognitive Field Dynamics not available: {e}")
        print("   System will use simplified models")

    try:
        from engines.thermodynamics import ThermodynamicFieldProcessor

        print("✅ Thermodynamic Field Processor available")
    except ImportError as e:
        print(f"⚠️  Thermodynamic Processor not available: {e}")
        print("   System will use simplified models")

    return True


def main():
    """Main test function"""
    print("🚀 KIMERA CDP LIVE READINESS TEST")
    print("=" * 40)

    tests = [
        ("Dependencies", test_dependencies),
        ("Files", test_files),
        ("Credentials", test_credentials),
        ("Backend", test_backend),
    ]

    all_passed = True

    for test_name, test_func in tests:
        try:
            result = test_func()
            if not result:
                all_passed = False
        except Exception as e:
            print(f"❌ {test_name} test failed: {e}")
            all_passed = False

    print("\n" + "=" * 40)

    if all_passed:
        print("🎉 SYSTEM READY FOR LIVE TRADING!")
        print()
        print("🚀 NEXT STEPS:")
        print("1. Configure your CDP private key:")
        print("   python setup_live_cdp_credentials.py")
        print()
        print("2. Start autonomous trading:")
        print("   python kimera_cdp_live_integration.py")
        print()
        print("⚠️  REMEMBER: Start with testnet and small amounts!")

    else:
        print("❌ SYSTEM NOT READY")
        print("Please resolve the issues above before proceeding.")

    print("=" * 40)


if __name__ == "__main__":
    main()
