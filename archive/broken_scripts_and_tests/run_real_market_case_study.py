#!/usr/bin/env python3
"""
KIMERA Real Market Case Study Launcher
Simplified launcher for the comprehensive market analysis case study
"""

import asyncio
import sys
import os
from pathlib import Path
import subprocess
import time

# Initialize structured logger
from backend.utils.kimera_logger import get_system_logger
logger = get_system_logger(__name__)


def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = ['aiohttp', 'pandas', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error("❌ Missing required packages:")
        for package in missing_packages:
            logger.info(f"   - {package}")
        logger.info("\n📦 Install missing packages with:")
        logger.info(f"   pip install {' '.join(missing_packages)
        return False
    
    return True

def check_kimera_api():
    """Check if KIMERA API is running"""
    try:
        import requests
        response = requests.get("http://localhost:8001/system/status", timeout=5)
        if response.status_code == 200:
            logger.info("✅ KIMERA API is running")
            return True
        else:
            logger.warning(f"⚠️  KIMERA API responded with status {response.status_code}")
            return False
    except ImportError:
        logger.warning("⚠️  Cannot check KIMERA API (requests not installed)
        return True  # Assume it's running
    except Exception as e:
        logger.error(f"❌ KIMERA API not accessible: {e}")
        logger.info("💡 Make sure to run 'python run_kimera.py' first")
        return False

def print_banner():
    """Print case study banner"""
    logger.info("=" * 70)
    logger.info("🧠 KIMERA SWM REAL MARKET CASE STUDY")
    logger.info("=" * 70)
    logger.info("📊 Real-time financial market contradiction detection")
    logger.info("🔑 Using Alpha Vantage API: X5GCHUJSY20536ID")
    logger.info("🎯 Analyzing: AAPL, GOOGL, MSFT, NVDA, AMD, TSLA, META, AMZN, NFLX, INTC")
    logger.info("=" * 70)
    logger.info()

def print_pre_run_info():
    """Print information before running the case study"""
    logger.info("📋 CASE STUDY OVERVIEW:")
    logger.info("   • Collect real market data from Alpha Vantage")
    logger.info("   • Analyze cross-asset correlations")
    logger.info("   • Detect 5 types of market contradictions:")
    logger.info("     - Sector divergence (tech stocks moving opposite directions)
    logger.info("     - Volume-price mismatches (high volume, low price movement)
    logger.info("     - Technical indicator divergences (RSI vs price contradictions)
    logger.info("     - Correlation breakdowns (historically correlated assets diverging)
    logger.info("     - Cross-asset anomalies (market-wide volatility spikes)
    logger.info("   • Send significant contradictions to KIMERA for processing")
    logger.info("   • Generate comprehensive analysis report")
    logger.info()
    logger.info("⏱️  Expected runtime: 5-10 minutes (due to API rate limits)
    logger.info("📁 Results will be saved to: real_market_case_study_results_[timestamp].json")
    logger.info()

async def run_case_study():
    """Run the actual case study"""
    try:
        # Import and run the case study
        from real_market_case_study import run_real_market_case_study
        await run_real_market_case_study()
        return True
    except ImportError as e:
        logger.error(f"❌ Cannot import case study module: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Case study execution failed: {e}")
        return False

def main():
    """Main launcher function"""
    print_banner()
    
    # Check dependencies
    logger.debug("🔍 Checking dependencies...")
    if not check_dependencies():
        logger.error("\n❌ Please install missing dependencies and try again.")
        return 1
    
    logger.info("✅ All dependencies are installed")
    
    # Check KIMERA API
    logger.debug("\n🔍 Checking KIMERA API connection...")
    kimera_running = check_kimera_api()
    
    if not kimera_running:
        logger.warning("\n⚠️  KIMERA API is not running. The case study will still collect and analyze")
        logger.info("   market data, but won't be able to send contradictions to KIMERA.")
        logger.info("\n💡 To enable full KIMERA integration:")
        logger.info("   1. Open a new terminal")
        logger.info("   2. Run: python run_kimera.py")
        logger.info("   3. Wait for 'Server will start on http://localhost:XXXX'")
        logger.info("   4. Re-run this case study")
        
        response = input("\n❓ Continue without KIMERA integration? (y/N): ").strip().lower()
        if response != 'y':
            logger.info("👋 Exiting. Start KIMERA API first for full integration.")
            return 0
    
    print_pre_run_info()
    
    # Confirm execution
    response = input("🚀 Ready to start the case study? (Y/n): ").strip().lower()
    if response == 'n':
        logger.info("👋 Case study cancelled.")
        return 0
    
    logger.info("\n🎬 Starting case study execution...")
    logger.info("📊 This will take several minutes due to API rate limiting...")
    logger.info("📝 Watch the logs for real-time progress updates")
    logger.info("-" * 70)
    
    # Run the case study
    try:
        success = asyncio.run(run_case_study())
        
        if success:
            logger.info("\n" + "=" * 70)
            logger.info("🎉 CASE STUDY COMPLETED SUCCESSFULLY!")
            logger.info("=" * 70)
            logger.info("📁 Check the generated JSON file for detailed results")
            logger.info("📊 Review the console output above for key findings")
            logger.info("🧠 If KIMERA was running, check the dashboard for new geoids")
            logger.info("\n💡 Next steps:")
            logger.info("   • Analyze the generated report")
            logger.info("   • Review detected contradictions")
            logger.info("   • Explore KIMERA dashboard for semantic insights")
            return 0
        else:
            logger.error("\n❌ Case study execution failed. Check the logs above for details.")
            return 1
            
    except KeyboardInterrupt:
        logger.info("\n\n⏹️  Case study interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"\n❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    
    if exit_code == 0:
        logger.info("\n🎯 Case study launcher completed successfully!")
    else:
        logger.critical("\n💥 Case study launcher failed!")
    
    sys.exit(exit_code)