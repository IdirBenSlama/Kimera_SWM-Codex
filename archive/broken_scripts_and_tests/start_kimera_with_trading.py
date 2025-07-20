"""
Start KIMERA with Trading Module
================================

This script starts the KIMERA system with the semantic trading module integrated.
It ensures all components are properly initialized and connected.
"""

import asyncio
import logging
import subprocess
import sys
import time
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if required dependencies are installed"""
    logger.info("Checking dependencies...")
    
    required_packages = [
        'fastapi',
        'uvicorn',
        'numpy',
        'pandas',
        'sqlalchemy',
        'prometheus-client'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        logger.error(f"Missing required packages: {', '.join(missing)}")
        logger.info("Please install them with: pip install " + ' '.join(missing))
        return False
    
    logger.info("✅ All core dependencies are installed")
    return True


def check_trading_dependencies():
    """Check if trading module dependencies are installed"""
    logger.info("Checking trading module dependencies...")
    
    trading_packages = [
        'ccxt',
        'plotly',
        'dash'
    ]
    
    missing = []
    for package in trading_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        logger.warning(f"Missing trading packages: {', '.join(missing)}")
        logger.info("Some features may be limited. Install with: pip install " + ' '.join(missing))
        # Don't return False - these are optional
    else:
        logger.info("✅ All trading dependencies are installed")
    
    return True


async def start_kimera_api():
    """Start the KIMERA API server"""
    logger.info("Starting KIMERA API server...")
    
    # Change to the project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Command to start the API
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "backend.api.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
    ]
    
    # Start the API server as a subprocess
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    # Wait for the server to start
    logger.info("Waiting for KIMERA to initialize...")
    await asyncio.sleep(5)
    
    # Check if the process is still running
    if process.poll() is None:
        logger.info("✅ KIMERA API server is running on http://localhost:8000")
        return process
    else:
        logger.error("❌ Failed to start KIMERA API server")
        stdout, stderr = process.communicate()
        logger.error(f"stdout: {stdout}")
        logger.error(f"stderr: {stderr}")
        return None


async def test_kimera_health():
    """Test if KIMERA is responding"""
    import aiohttp
    
    logger.info("Testing KIMERA health...")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:8000/system/health') as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"✅ KIMERA is healthy: {data}")
                    return True
                else:
                    logger.error(f"❌ KIMERA health check failed: {response.status}")
                    return False
    except Exception as e:
        logger.error(f"❌ Could not connect to KIMERA: {e}")
        return False


async def initialize_trading_module():
    """Initialize the trading module within KIMERA"""
    logger.info("\nInitializing Trading Module...")
    
    try:
        # Import the trading module
        from backend.trading import create_kimera_trading_system
        
        # Create trading system with default configuration
        config = {
            'tension_threshold': 0.4,
            'max_position_size': 1000,
            'risk_per_trade': 0.02,
            'enable_paper_trading': True,
            'enable_sentiment_analysis': True,
            'enable_news_processing': True,
            'dashboard_port': 8050
        }
        
        trading_system = create_kimera_trading_system(config)
        
        # Start the trading system
        await trading_system.start()
        
        logger.info("✅ Trading module initialized successfully")
        logger.info(f"📊 Trading dashboard will be available at http://localhost:{config['dashboard_port']}")
        
        return trading_system
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize trading module: {e}")
        return None


async def run_trading_demo():
    """Run a simple trading demonstration"""
    logger.info("\nRunning Trading Demonstration...")
    
    try:
        from backend.trading import process_trading_opportunity
        
        # Create a test market event with contradiction
        test_event = {
            'market_data': {
                'symbol': 'BTC-USD',
                'price': 50000,
                'volume': 2500,
                'momentum': 0.03,  # Positive price momentum
                'volatility': 0.02,
                'trend': 'bullish'
            },
            'context': {
                'news_sentiment': -0.6,  # Negative news (contradiction!)
                'social_sentiment': 0.2
            }
        }
        
        logger.info("📊 Market Event:")
        logger.info(f"   Symbol: {test_event['market_data']['symbol']}")
        logger.info(f"   Price: ${test_event['market_data']['price']:,.2f}")
        logger.info(f"   Momentum: {test_event['market_data']['momentum']:.1%} (Bullish)")
        logger.info(f"   News Sentiment: {test_event['context']['news_sentiment']} (Bearish)")
        logger.info("   ⚡ Contradiction detected: Bullish price action vs Bearish news!")
        
        # Process through trading system
        result = await process_trading_opportunity(test_event)
        
        logger.info("\n📈 Trading Analysis Result:")
        logger.info(f"   Status: {result['status']}")
        
        if 'analysis' in result:
            analysis = result['analysis']
            logger.info(f"   Action: {analysis.action_taken}")
            logger.info(f"   Confidence: {analysis.confidence:.1%}")
            logger.info(f"   Contradictions Found: {len(analysis.contradiction_map)}")
            
            if analysis.semantic_analysis:
                logger.info("   Semantic Metrics:")
                for key, value in analysis.semantic_analysis.items():
                    logger.info(f"     - {key}: {value:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Trading demo failed: {e}")
        return False


async def main():
    """Main function to start KIMERA with trading module"""
    logger.info("="*80)
    logger.info("KIMERA WITH SEMANTIC TRADING MODULE")
    logger.info("="*80)
    
    # Step 1: Check dependencies
    if not check_dependencies():
        return
    
    check_trading_dependencies()
    
    # Step 2: Start KIMERA API
    api_process = await start_kimera_api()
    if not api_process:
        return
    
    try:
        # Step 3: Test KIMERA health
        health_ok = await test_kimera_health()
        if not health_ok:
            logger.error("KIMERA is not responding. Please check the logs.")
            return
        
        # Step 4: Initialize trading module
        trading_system = await initialize_trading_module()
        if not trading_system:
            logger.warning("Trading module initialization failed, but KIMERA is running.")
        
        # Step 5: Run a demo
        if trading_system:
            await run_trading_demo()
        
        # Keep running
        logger.info("\n" + "="*80)
        logger.info("✅ KIMERA is running with Trading Module")
        logger.info("="*80)
        logger.info("🌐 API: http://localhost:8000")
        logger.info("📊 API Docs: http://localhost:8000/docs")
        logger.info("📈 Trading Dashboard: http://localhost:8050")
        logger.info("\nPress Ctrl+C to stop...")
        
        # Keep the process running
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("\n🛑 Shutting down...")
        
        # Stop trading system
        if 'trading_system' in locals() and trading_system:
            await trading_system.stop()
        
        # Terminate API process
        if api_process:
            api_process.terminate()
            api_process.wait()
        
        logger.info("✅ KIMERA stopped gracefully")
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if api_process:
            api_process.terminate()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user") 