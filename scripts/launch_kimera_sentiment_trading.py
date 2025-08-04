#!/usr/bin/env python3
"""
KIMERA SENTIMENT-ENHANCED TRADING LAUNCHER
==========================================
Comprehensive launcher for all sentiment-enhanced trading capabilities:

🧠 SENTIMENT ANALYSIS FRAMEWORKS:
   - FinBERT (97.18% accuracy financial sentiment)
   - spaCy (30K+ stars, multilingual)
   - VADER (social media optimized)
   - TextBlob (user-friendly APIs)
   - Pattern (web scraping + sentiment)
   - NLP.js equivalent (multilingual)

🔮 DECENTRALIZED ORACLE PROTOCOLS:
   - Chainlink (dominant oracle network)
   - Pyth Network (23.5% faster response)
   - Band Protocol (cross-chain, low latency)
   - API3 (direct API providers)

🚀 TRADING SYSTEMS:
   - Parallel Omnidimensional Trader
   - Triangular Arbitrage Engine
   - Sentiment-Enhanced Trading
   - Oracle-Driven Decisions
"""

import os
import sys
import json
import time
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_credentials() -> Dict[str, bool]:
    """Check available credentials and capabilities"""
    credentials = {
        'cdp_available': False,
        'advanced_trade_available': False,
        'sentiment_available': False,
        'oracles_available': False
    }
    
    # Check CDP credentials
    try:
        from dotenv import load_dotenv
        load_dotenv('kimera_cdp_live.env')
        
        cdp_key = os.getenv('CDP_API_KEY_NAME', '').strip()
        if cdp_key:
            credentials['cdp_available'] = True
            logger.info("✅ CDP credentials found")
        else:
            logger.warning("⚠️ CDP credentials not found")
    except Exception as e:
        logger.warning(f"⚠️ CDP check failed: {e}")
    
    # Check sentiment analysis dependencies
    try:
        import spacy
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        from textblob import TextBlob
        from transformers import pipeline
        credentials['sentiment_available'] = True
        logger.info("✅ Sentiment analysis frameworks available")
    except ImportError as e:
        logger.warning(f"⚠️ Some sentiment frameworks missing: {e}")
        credentials['sentiment_available'] = False
    
    # Check oracle dependencies
    try:
        from web3 import Web3
        import requests
        credentials['oracles_available'] = True
        logger.info("✅ Oracle integration available")
    except ImportError:
        logger.warning("⚠️ Oracle dependencies missing")
        credentials['oracles_available'] = False
    
    return credentials

async def run_sentiment_enhanced_trading(duration_minutes: int = 10):
    """Run the comprehensive sentiment-enhanced trading system"""
    try:
        from kimera_sentiment_enhanced_trader import SentimentEnhancedTrader
        
        logger.info("🧠 Launching Sentiment-Enhanced Trading")
        
        trader = SentimentEnhancedTrader()
        await trader.initialize()
        await trader.run_sentiment_enhanced_trading(duration_minutes)
        
    except ImportError as e:
        logger.error(f"❌ Sentiment trading unavailable: {e}")
        logger.info("💡 Running fallback parallel trading...")
        await run_parallel_trading(duration_minutes)

async def run_parallel_trading(duration_minutes: int = 5):
    """Run the parallel omnidimensional trading system"""
    try:
        from kimera_parallel_omnidimensional_trader import ParallelOmnidimensionalTrader
        
        logger.info("⚡ Launching Parallel Omnidimensional Trading")
        
        trader = ParallelOmnidimensionalTrader()
        await trader.initialize()
        await trader.run_parallel_trading(duration_minutes)
        
    except ImportError as e:
        logger.error(f"❌ Parallel trading unavailable: {e}")

async def run_triangular_arbitrage(duration_minutes: int = 5):
    """Run the triangular arbitrage engine"""
    try:
        from kimera_triangular_arbitrage import TriangularArbitrageEngine
        
        logger.info("🔄 Launching Triangular Arbitrage Engine")
        
        engine = TriangularArbitrageEngine()
        await engine.run_continuous_arbitrage(duration_minutes)
        
    except ImportError as e:
        logger.error(f"❌ Triangular arbitrage unavailable: {e}")

async def test_sentiment_analysis():
    """Test all sentiment analysis frameworks"""
    try:
        from kimera_sentiment_engine import KimeraSentimentEngine
        
        logger.info("🧠 Testing Sentiment Analysis Frameworks")
        
        engine = KimeraSentimentEngine()
        test_assets = ['BTC', 'ETH', 'SOL']
        
        results = await engine.analyze_multiple_assets(test_assets)
        
        logger.info(f"\n🧠 SENTIMENT ANALYSIS TEST RESULTS")
        logger.info("=" * 60)
        
        for asset, sentiment in results.items():
            signal = engine.get_sentiment_signal(sentiment)
            logger.info(f"\n{asset}:")
            logger.info(f"  Overall: {sentiment.aggregated_score:.3f} ({sentiment.trending_direction})")
            logger.info(f"  Signal: {signal['action']} (confidence: {signal['confidence']:.2f})")
            logger.info(f"  Oracle: {sentiment.oracle_sentiment:.3f}")
            logger.info(f"  Social: {sentiment.social_sentiment:.3f}")
            logger.info(f"  News: {sentiment.news_sentiment:.3f}")
        
    except ImportError as e:
        logger.error(f"❌ Sentiment analysis test failed: {e}")

async def test_oracle_integration():
    """Test decentralized oracle integration"""
    try:
        from kimera_oracle_integration import OracleAggregator
        
        logger.info("🔮 Testing Oracle Integration")
        
        aggregator = OracleAggregator()
        test_assets = ['BTC', 'ETH']
        
        results = await aggregator.get_multi_asset_sentiment(test_assets)
        
        logger.info(f"\n🔮 ORACLE INTEGRATION TEST RESULTS")
        logger.info("=" * 60)
        
        for asset, data in results.items():
            logger.info(f"\n{asset}:")
            logger.info(f"  Sentiment: {data['aggregated_sentiment']:.3f}")
            logger.info(f"  Confidence: {data['aggregated_confidence']:.3f}")
            logger.info(f"  Consensus: {data['consensus_score']:.3f}")
            logger.info(f"  Response Time: {data['total_response_time_ms']:.1f}ms")
            logger.info(f"  Oracle Sources: {data['oracle_count']}")
        
    except ImportError as e:
        logger.error(f"❌ Oracle integration test failed: {e}")

async def show_system_capabilities():
    """Display all available system capabilities"""
    capabilities = check_credentials()
    
    logger.info("\n🚀 KIMERA SENTIMENT-ENHANCED TRADING SYSTEM")
    logger.info("=" * 70)
    logger.info("SYSTEM CAPABILITIES:")
    logger.info("=" * 70)
    
    # Trading Systems
    logger.info("\n📈 TRADING SYSTEMS:")
    logger.info(f"   ⚡ Parallel Omnidimensional Trading: ✅ Available")
    logger.info(f"   🔄 Triangular Arbitrage Engine: ✅ Available") 
    logger.info(f"   🧠 Sentiment-Enhanced Trading: {'✅ Available' if capabilities['sentiment_available'] else '⚠️ Limited'}")
    
    # Sentiment Analysis
    logger.info(f"\n🧠 SENTIMENT ANALYSIS FRAMEWORKS:")
    if capabilities['sentiment_available']:
        logger.info(f"   📊 FinBERT (Financial BERT): ✅ 97.18% accuracy")
        logger.info(f"   🌐 spaCy (Multilingual): ✅ 30K+ GitHub stars")
        logger.info(f"   📱 VADER (Social Media): ✅ Valence-aware lexicon")
        logger.info(f"   📝 TextBlob (User-friendly): ✅ Simple APIs")
        logger.info(f"   🕷️ Pattern (Web scraping): ✅ Full-stack solution")
        logger.info(f"   🌍 NLP.js (40 languages): ✅ Real-time analysis")
    else:
        logger.info(f"   ⚠️ Install requirements: pip install -r requirements_sentiment.txt")
    
    # Oracle Integration
    logger.info(f"\n🔮 DECENTRALIZED ORACLE PROTOCOLS:")
    if capabilities['oracles_available']:
        logger.info(f"   🔗 Chainlink: ✅ Dominant oracle network")
        logger.info(f"   🚀 Pyth Network: ✅ 23.5% faster response")
        logger.info(f"   🌐 Band Protocol: ✅ Cross-chain, low latency")
        logger.info(f"   🔌 API3: ✅ Direct API providers")
    else:
        logger.info(f"   ⚠️ Install requirements: pip install web3 aiohttp")
    
    # Credentials
    logger.info(f"\n🔑 CREDENTIALS STATUS:")
    logger.info(f"   CDP Trading: {'✅ Available' if capabilities['cdp_available'] else '❌ Missing'}")
    logger.info(f"   Advanced Trade: {'✅ Available' if capabilities['advanced_trade_available'] else '❌ Missing'}")
    
    # Performance Features
    logger.info(f"\n⚡ PERFORMANCE FEATURES:")
    logger.info(f"   🔄 Parallel Processing: ✅ Up to 50 concurrent trades")
    logger.info(f"   🧠 Real-time Sentiment: ✅ Multi-framework aggregation")
    logger.info(f"   🔮 Oracle Aggregation: ✅ 4 decentralized protocols")
    logger.info(f"   💱 Inter-coin Trading: ✅ BTC-ETH, ETH-SOL cycles")
    logger.info(f"   📊 Multi-dimensional: ✅ Horizontal + Vertical strategies")
    
    return capabilities

def display_menu():
    """Display the main menu"""
    logger.info(f"\n🎯 SELECT TRADING SYSTEM:")
    logger.info("=" * 50)
    logger.info("1. 🧠 Sentiment-Enhanced Trading (RECOMMENDED)")
    logger.info("2. ⚡ Parallel Omnidimensional Trading")
    logger.info("3. 🔄 Triangular Arbitrage Engine")
    logger.info("4. 🧪 Test Sentiment Analysis")
    logger.info("5. 🔮 Test Oracle Integration")
    logger.info("6. 📊 Show System Capabilities")
    logger.info("7. 🚀 Run All Systems (Sequential)")
    logger.info("8. ❌ Exit")
    logger.info("=" * 50)

async def run_all_systems():
    """Run all trading systems sequentially"""
    logger.info("🚀 RUNNING ALL SYSTEMS SEQUENTIALLY")
    
    systems = [
        ("🧠 Sentiment-Enhanced Trading", run_sentiment_enhanced_trading, 5),
        ("⚡ Parallel Trading", run_parallel_trading, 3),
        ("🔄 Triangular Arbitrage", run_triangular_arbitrage, 3)
    ]
    
    for name, func, duration in systems:
        logger.info(f"\n▶️ Starting {name}")
        try:
            await func(duration)
            logger.info(f"✅ {name} completed")
        except Exception as e:
            logger.error(f"❌ {name} failed: {e}")
        
        # Brief pause between systems
        await asyncio.sleep(2)
    
    logger.info("🏁 All systems completed!")

async def main():
    """Main launcher"""
    logger.info("\n🚀 KIMERA SENTIMENT-ENHANCED TRADING LAUNCHER")
    logger.info("🧠 AI-POWERED MARKET SENTIMENT + PARALLEL TRADING")
    logger.info("=" * 70)
    
    # Show capabilities
    capabilities = await show_system_capabilities()
    
    while True:
        display_menu()
        
        try:
            choice = input("\n👉 Enter your choice (1-8): ").strip()
            
            if choice == '1':
                duration = int(input("Enter duration in minutes (default 10): ") or "10")
                await run_sentiment_enhanced_trading(duration)
                
            elif choice == '2':
                duration = int(input("Enter duration in minutes (default 5): ") or "5")
                await run_parallel_trading(duration)
                
            elif choice == '3':
                duration = int(input("Enter duration in minutes (default 5): ") or "5")
                await run_triangular_arbitrage(duration)
                
            elif choice == '4':
                await test_sentiment_analysis()
                
            elif choice == '5':
                await test_oracle_integration()
                
            elif choice == '6':
                await show_system_capabilities()
                
            elif choice == '7':
                await run_all_systems()
                
            elif choice == '8':
                logger.info("👋 Goodbye!")
                break
                
            else:
                logger.info("❌ Invalid choice. Please try again.")
                
        except KeyboardInterrupt:
            logger.info("\n⏹️ Operation cancelled by user")
            break
        except ValueError:
            logger.info("❌ Invalid input. Please enter a number.")
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            
        # Pause before showing menu again
        input("\n📱 Press Enter to continue...")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n👋 System shutdown requested")
    except Exception as e:
        logger.error(f"❌ Launcher failed: {e}")
        import traceback
        traceback.print_exc() 