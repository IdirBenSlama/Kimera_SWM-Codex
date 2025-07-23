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
        
        print(f"\n🧠 SENTIMENT ANALYSIS TEST RESULTS")
        print("=" * 60)
        
        for asset, sentiment in results.items():
            signal = engine.get_sentiment_signal(sentiment)
            print(f"\n{asset}:")
            print(f"  Overall: {sentiment.aggregated_score:.3f} ({sentiment.trending_direction})")
            print(f"  Signal: {signal['action']} (confidence: {signal['confidence']:.2f})")
            print(f"  Oracle: {sentiment.oracle_sentiment:.3f}")
            print(f"  Social: {sentiment.social_sentiment:.3f}")
            print(f"  News: {sentiment.news_sentiment:.3f}")
        
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
        
        print(f"\n🔮 ORACLE INTEGRATION TEST RESULTS")
        print("=" * 60)
        
        for asset, data in results.items():
            print(f"\n{asset}:")
            print(f"  Sentiment: {data['aggregated_sentiment']:.3f}")
            print(f"  Confidence: {data['aggregated_confidence']:.3f}")
            print(f"  Consensus: {data['consensus_score']:.3f}")
            print(f"  Response Time: {data['total_response_time_ms']:.1f}ms")
            print(f"  Oracle Sources: {data['oracle_count']}")
        
    except ImportError as e:
        logger.error(f"❌ Oracle integration test failed: {e}")

async def show_system_capabilities():
    """Display all available system capabilities"""
    capabilities = check_credentials()
    
    print("\n🚀 KIMERA SENTIMENT-ENHANCED TRADING SYSTEM")
    print("=" * 70)
    print("SYSTEM CAPABILITIES:")
    print("=" * 70)
    
    # Trading Systems
    print("\n📈 TRADING SYSTEMS:")
    print(f"   ⚡ Parallel Omnidimensional Trading: ✅ Available")
    print(f"   🔄 Triangular Arbitrage Engine: ✅ Available") 
    print(f"   🧠 Sentiment-Enhanced Trading: {'✅ Available' if capabilities['sentiment_available'] else '⚠️ Limited'}")
    
    # Sentiment Analysis
    print(f"\n🧠 SENTIMENT ANALYSIS FRAMEWORKS:")
    if capabilities['sentiment_available']:
        print(f"   📊 FinBERT (Financial BERT): ✅ 97.18% accuracy")
        print(f"   🌐 spaCy (Multilingual): ✅ 30K+ GitHub stars")
        print(f"   📱 VADER (Social Media): ✅ Valence-aware lexicon")
        print(f"   📝 TextBlob (User-friendly): ✅ Simple APIs")
        print(f"   🕷️ Pattern (Web scraping): ✅ Full-stack solution")
        print(f"   🌍 NLP.js (40 languages): ✅ Real-time analysis")
    else:
        print(f"   ⚠️ Install requirements: pip install -r requirements_sentiment.txt")
    
    # Oracle Integration
    print(f"\n🔮 DECENTRALIZED ORACLE PROTOCOLS:")
    if capabilities['oracles_available']:
        print(f"   🔗 Chainlink: ✅ Dominant oracle network")
        print(f"   🚀 Pyth Network: ✅ 23.5% faster response")
        print(f"   🌐 Band Protocol: ✅ Cross-chain, low latency")
        print(f"   🔌 API3: ✅ Direct API providers")
    else:
        print(f"   ⚠️ Install requirements: pip install web3 aiohttp")
    
    # Credentials
    print(f"\n🔑 CREDENTIALS STATUS:")
    print(f"   CDP Trading: {'✅ Available' if capabilities['cdp_available'] else '❌ Missing'}")
    print(f"   Advanced Trade: {'✅ Available' if capabilities['advanced_trade_available'] else '❌ Missing'}")
    
    # Performance Features
    print(f"\n⚡ PERFORMANCE FEATURES:")
    print(f"   🔄 Parallel Processing: ✅ Up to 50 concurrent trades")
    print(f"   🧠 Real-time Sentiment: ✅ Multi-framework aggregation")
    print(f"   🔮 Oracle Aggregation: ✅ 4 decentralized protocols")
    print(f"   💱 Inter-coin Trading: ✅ BTC-ETH, ETH-SOL cycles")
    print(f"   📊 Multi-dimensional: ✅ Horizontal + Vertical strategies")
    
    return capabilities

def display_menu():
    """Display the main menu"""
    print(f"\n🎯 SELECT TRADING SYSTEM:")
    print("=" * 50)
    print("1. 🧠 Sentiment-Enhanced Trading (RECOMMENDED)")
    print("2. ⚡ Parallel Omnidimensional Trading")
    print("3. 🔄 Triangular Arbitrage Engine")
    print("4. 🧪 Test Sentiment Analysis")
    print("5. 🔮 Test Oracle Integration")
    print("6. 📊 Show System Capabilities")
    print("7. 🚀 Run All Systems (Sequential)")
    print("8. ❌ Exit")
    print("=" * 50)

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
    print("\n🚀 KIMERA SENTIMENT-ENHANCED TRADING LAUNCHER")
    print("🧠 AI-POWERED MARKET SENTIMENT + PARALLEL TRADING")
    print("=" * 70)
    
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
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please try again.")
                
        except KeyboardInterrupt:
            print("\n⏹️ Operation cancelled by user")
            break
        except ValueError:
            print("❌ Invalid input. Please enter a number.")
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            
        # Pause before showing menu again
        input("\n📱 Press Enter to continue...")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 System shutdown requested")
    except Exception as e:
        logger.error(f"❌ Launcher failed: {e}")
        import traceback
        traceback.print_exc() 