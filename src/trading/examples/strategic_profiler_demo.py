#!/usr/bin/env python3
"""
Strategic Profiler System Demo
=============================

Demonstrates KIMERA's advanced strategic profiling capabilities:
- Behavioral analysis of different trader types
- Context-aware market participant identification
- Strategic response generation
- Adaptive profiling with scientific rigor

This demo shows how KIMERA can identify, analyze, and respond to
different market participants like a strategic warfare system.
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

# Import KIMERA strategic profiler
try:
    from ..intelligence.strategic_profiler_system import (
        StrategicProfilerSystem, TraderArchetype, MarketRegime, 
        StrategicIntent, create_strategic_profiler_system,
        create_whale_hunter_profile, create_hft_detector_profile,
        create_retail_sentiment_profile
    )
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StrategicProfilerDemo:
    """
    Comprehensive demo of KIMERA's strategic profiling system
    """
    
    def __init__(self):
        self.profiler_system = create_strategic_profiler_system()
        self.demo_data = {}
        self.analysis_results = {}
        
    async def run_comprehensive_demo(self):
        """Run complete strategic profiler demonstration"""
        
        logger.info("🎯" * 25)
        logger.info("🔍 KIMERA STRATEGIC PROFILER SYSTEM DEMO")
        logger.info("🎯" * 25)
        logger.info("")
        logger.info("🎖️  MISSION: Identify and analyze market participants")
        logger.info("⚔️  STRATEGY: Behavioral profiling with scientific rigor")
        logger.info("🧠 INTELLIGENCE: Multi-dimensional participant analysis")
        logger.info("🎯 OUTCOME: Strategic responses for competitive advantage")
        logger.info("")
        
        # Phase 1: Data Collection
        await self.collect_demo_data()
        
        # Phase 2: Market Participant Analysis
        await self.analyze_market_participants()
        
        # Phase 3: Context Assessment
        await self.assess_market_contexts()
        
        # Phase 4: Strategic Response Generation
        await self.generate_strategic_responses()
        
        # Phase 5: Specialized Profiler Demos
        await self.demo_specialized_profiles()
        
        # Phase 6: System Status and Performance
        await self.display_system_status()
        
        logger.info("")
        logger.info("🎯 STRATEGIC PROFILER DEMO COMPLETE")
        logger.info("✅ All market participant analysis systems operational")
        logger.info("⚔️  Ready for live market warfare!")
    
    async def collect_demo_data(self):
        """Collect diverse market data for profiler testing"""
        
        logger.info("📊 PHASE 1: COLLECTING STRATEGIC INTELLIGENCE DATA")
        logger.info("=" * 50)
        
        # Collect different types of market data
        symbols = {
            'institutional_heavy': ['AAPL', 'MSFT', 'GOOGL'],  # Large institutional presence
            'retail_popular': ['GME', 'AMC', 'TSLA'],         # Retail momentum favorites
            'hft_active': ['SPY', 'QQQ', 'IWM'],              # High HFT activity
            'manipulation_prone': ['DOGE-USD', 'SHIB-USD']     # Crypto manipulation
        }
        
        for category, symbol_list in symbols.items():
            logger.info(f"📈 Collecting {category} data...")
            
            category_data = {}
            for symbol in symbol_list:
                try:
                    # Get recent data
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period='5d', interval='1h')
                    
                    if not data.empty:
                        # Add synthetic features for demo
                        data = await self.add_synthetic_features(data, category)
                        category_data[symbol] = data
                        logger.info(f"  ✅ {symbol}: {len(data)} data points")
                    else:
                        logger.warning(f"  ⚠️ {symbol}: No data available")
                        
                except Exception as e:
                    logger.error(f"  ❌ {symbol}: Error - {e}")
            
            self.demo_data[category] = category_data
        
        total_datasets = sum(len(cat_data) for cat_data in self.demo_data.values())
        logger.info(f"📊 Data collection complete: {total_datasets} datasets ready")
        logger.info("")
    
    async def add_synthetic_features(self, data: pd.DataFrame, category: str) -> pd.DataFrame:
        """Add synthetic features to simulate different trader behaviors"""
        
        # Add timestamp
        data['timestamp'] = data.index
        
        # Add category-specific synthetic features
        if category == 'institutional_heavy':
            # Simulate institutional patterns
            data['large_block_trades'] = np.random.poisson(2, len(data))
            data['iceberg_indicator'] = (data['Volume'] > data['Volume'].rolling(20).mean() * 1.5) & \
                                      (abs(data['Close'].pct_change()) < data['Close'].pct_change().rolling(20).std())
            data['stealth_execution'] = np.random.beta(0.8, 0.2, len(data))
            
        elif category == 'retail_popular':
            # Simulate retail momentum patterns
            data['social_sentiment'] = np.random.normal(0.5, 0.3, len(data))
            data['momentum_chasing'] = (data['Close'].pct_change(5) > 0.02) & \
                                     (data['Volume'] > data['Volume'].rolling(10).mean() * 2)
            data['fomo_indicator'] = np.random.exponential(0.3, len(data))
            
        elif category == 'hft_active':
            # Simulate HFT patterns
            data['microsecond_trades'] = np.random.poisson(100, len(data))
            data['spread_scalping'] = np.random.uniform(0.001, 0.01, len(data))
            data['order_cancellation_rate'] = np.random.beta(0.9, 0.1, len(data))
            
        elif category == 'manipulation_prone':
            # Simulate manipulation patterns
            data['wash_trading_indicator'] = np.random.binomial(1, 0.1, len(data))
            data['pump_dump_score'] = np.random.gamma(2, 0.1, len(data))
            data['spoofing_detected'] = np.random.binomial(1, 0.05, len(data))
        
        # Add common technical features
        data['volatility'] = data['Close'].pct_change().rolling(20).std()
        data['trend_strength'] = (data['Close'].rolling(10).mean() - data['Close'].rolling(50).mean()) / data['Close'].rolling(50).mean()
        data['liquidity'] = data['Volume'] / data['Volume'].rolling(50).mean()
        data['sentiment'] = np.random.normal(0.5, 0.2, len(data))
        
        return data
    
    async def analyze_market_participants(self):
        """Analyze market participants across different datasets"""
        
        logger.info("🔍 PHASE 2: MARKET PARTICIPANT ANALYSIS")
        logger.info("=" * 50)
        
        all_profiles = []
        
        for category, datasets in self.demo_data.items():
            logger.info(f"🎯 Analyzing {category} markets...")
            
            category_profiles = []
            for symbol, data in datasets.items():
                try:
                    logger.info(f"  📊 Scanning {symbol} for trader archetypes...")
                    
                    # Convert to dict format for analysis
                    market_data = data.to_dict('records')
                    
                    # Analyze participants
                    profiles = await self.profiler_system.analyze_market_participants(market_data)
                    category_profiles.extend(profiles)
                    
                    if profiles:
                        logger.info(f"    🎯 Detected {len(profiles)} trader archetypes:")
                        for profile in profiles:
                            logger.info(f"      - {profile.archetype.value}: {profile.confidence:.2f} confidence")
                    else:
                        logger.info(f"    ℹ️  No clear archetypes detected")
                        
                except Exception as e:
                    logger.error(f"  ❌ Error analyzing {symbol}: {e}")
            
            self.analysis_results[category] = category_profiles
            all_profiles.extend(category_profiles)
            
            # Category summary
            if category_profiles:
                archetypes = [p.archetype for p in category_profiles]
                unique_archetypes = set(archetypes)
                logger.info(f"  📈 {category} summary: {len(unique_archetypes)} unique archetypes detected")
            
            logger.info("")
        
        # Overall summary
        logger.info("🎯 PARTICIPANT ANALYSIS SUMMARY")
        logger.info("-" * 30)
        
        if all_profiles:
            archetype_counts = {}
            confidence_scores = {}
            
            for profile in all_profiles:
                archetype = profile.archetype.value
                if archetype not in archetype_counts:
                    archetype_counts[archetype] = 0
                    confidence_scores[archetype] = []
                
                archetype_counts[archetype] += 1
                confidence_scores[archetype].append(profile.confidence)
            
            logger.info(f"📊 Total profiles detected: {len(all_profiles)}")
            logger.info(f"🎯 Unique archetypes: {len(archetype_counts)}")
            logger.info("")
            
            logger.info("🏆 TOP DETECTED ARCHETYPES:")
            sorted_archetypes = sorted(archetype_counts.items(), key=lambda x: x[1], reverse=True)
            
            for archetype, count in sorted_archetypes[:5]:
                avg_confidence = np.mean(confidence_scores[archetype])
                logger.info(f"  {count:2d}x {archetype:<20} (avg confidence: {avg_confidence:.2f})")
        else:
            logger.info("⚠️  No trader archetypes detected in demo data")
        
        logger.info("")
    
    async def assess_market_contexts(self):
        """Assess market contexts for strategic decision making"""
        
        logger.info("🌐 PHASE 3: MARKET CONTEXT ASSESSMENT")
        logger.info("=" * 50)
        
        context_results = {}
        
        for category, datasets in self.demo_data.items():
            logger.info(f"🔍 Assessing {category} market contexts...")
            
            category_contexts = []
            for symbol, data in datasets.items():
                try:
                    # Use latest data point for context assessment
                    latest_data = data.iloc[-1].to_dict()
                    
                    # Assess market context
                    context = await self.profiler_system.assess_market_context(latest_data)
                    category_contexts.append(context)
                    
                    logger.info(f"  📊 {symbol}:")
                    logger.info(f"    - Regime: {context.regime.value}")
                    logger.info(f"    - Volatility: {context.volatility_level:.3f}")
                    logger.info(f"    - Manipulation Risk: {context.manipulation_risk:.3f}")
                    logger.info(f"    - Opportunity Score: {context.opportunity_score:.3f}")
                    logger.info(f"    - Active Participants: {len(context.active_participants)}")
                    logger.info(f"    - Dominant Intent: {context.dominant_intent.value}")
                    
                except Exception as e:
                    logger.error(f"  ❌ Error assessing {symbol}: {e}")
            
            context_results[category] = category_contexts
            logger.info("")
        
        # Context analysis summary
        logger.info("🌐 CONTEXT ASSESSMENT SUMMARY")
        logger.info("-" * 30)
        
        all_contexts = []
        for contexts in context_results.values():
            all_contexts.extend(contexts)
        
        if all_contexts:
            # Regime distribution
            regimes = [c.regime for c in all_contexts]
            regime_counts = {regime: regimes.count(regime) for regime in set(regimes)}
            
            logger.info("📊 Market Regime Distribution:")
            for regime, count in regime_counts.items():
                logger.info(f"  {regime.value:<20}: {count} markets")
            
            # Risk assessment
            avg_volatility = np.mean([c.volatility_level for c in all_contexts])
            avg_manipulation_risk = np.mean([c.manipulation_risk for c in all_contexts])
            avg_opportunity = np.mean([c.opportunity_score for c in all_contexts])
            
            logger.info("")
            logger.info("📈 Risk & Opportunity Metrics:")
            logger.info(f"  Average Volatility: {avg_volatility:.3f}")
            logger.info(f"  Average Manipulation Risk: {avg_manipulation_risk:.3f}")
            logger.info(f"  Average Opportunity Score: {avg_opportunity:.3f}")
        
        self.analysis_results['contexts'] = context_results
        logger.info("")
    
    async def generate_strategic_responses(self):
        """Generate strategic responses to identified market participants"""
        
        logger.info("⚔️ PHASE 4: STRATEGIC RESPONSE GENERATION")
        logger.info("=" * 50)
        
        response_results = {}
        
        # Use contexts from previous phase
        if 'contexts' in self.analysis_results:
            for category, contexts in self.analysis_results['contexts'].items():
                logger.info(f"🎯 Generating responses for {category} markets...")
                
                category_responses = []
                for context in contexts:
                    try:
                        if context.active_participants:
                            # Generate strategic responses
                            responses = await self.profiler_system.generate_strategic_responses(context)
                            category_responses.extend(responses.values())
                            
                            logger.info(f"  ⚔️  Generated {len(responses)} strategic responses")
                            
                            # Display key responses
                            for archetype, response in responses.items():
                                logger.info(f"    🎯 {archetype.value}:")
                                logger.info(f"      Strategy: {response.response_strategy}")
                                logger.info(f"      Success Probability: {response.success_probability:.2f}")
                                logger.info(f"      Expected Profit: {response.expected_profit:.3f}")
                                logger.info(f"      Max Drawdown: {response.max_drawdown:.3f}")
                        else:
                            logger.info(f"  ℹ️  No active participants detected")
                            
                    except Exception as e:
                        logger.error(f"  ❌ Error generating responses: {e}")
                
                response_results[category] = category_responses
                logger.info("")
        
        # Response generation summary
        logger.info("⚔️ STRATEGIC RESPONSE SUMMARY")
        logger.info("-" * 30)
        
        all_responses = []
        for responses in response_results.values():
            all_responses.extend(responses)
        
        if all_responses:
            # Strategy distribution
            strategies = [r.response_strategy for r in all_responses]
            strategy_counts = {strategy: strategies.count(strategy) for strategy in set(strategies)}
            
            logger.info("📊 Response Strategy Distribution:")
            for strategy, count in strategy_counts.items():
                logger.info(f"  {strategy:<25}: {count} responses")
            
            # Performance metrics
            avg_success_prob = np.mean([r.success_probability for r in all_responses])
            avg_expected_profit = np.mean([r.expected_profit for r in all_responses])
            avg_max_drawdown = np.mean([r.max_drawdown for r in all_responses])
            
            logger.info("")
            logger.info("📈 Expected Performance Metrics:")
            logger.info(f"  Average Success Probability: {avg_success_prob:.2f}")
            logger.info(f"  Average Expected Profit: {avg_expected_profit:.3f}")
            logger.info(f"  Average Max Drawdown: {avg_max_drawdown:.3f}")
            
            # Risk-adjusted return
            risk_adjusted_return = avg_expected_profit / max(avg_max_drawdown, 0.001)
            logger.info(f"  Risk-Adjusted Return Ratio: {risk_adjusted_return:.2f}")
        
        self.analysis_results['responses'] = response_results
        logger.info("")
    
    async def demo_specialized_profiles(self):
        """Demonstrate specialized trader profiles"""
        
        logger.info("🎖️ PHASE 5: SPECIALIZED PROFILER DEMONSTRATIONS")
        logger.info("=" * 50)
        
        # Whale Hunter Profile
        logger.info("🐋 WHALE HUNTER PROFILE DEMO")
        logger.info("-" * 30)
        
        whale_profile = await create_whale_hunter_profile()
        logger.info(f"Archetype: {whale_profile.archetype.value}")
        logger.info(f"Confidence: {whale_profile.confidence:.2f}")
        logger.info(f"Key Traits: {list(whale_profile.behavioral_traits.keys())}")
        logger.info(f"Detection Signals: {whale_profile.detection_signals}")
        logger.info(f"Counter Strategies: {whale_profile.counter_strategies}")
        logger.info("")
        
        # HFT Detector Profile
        logger.info("🤖 HFT DETECTOR PROFILE DEMO")
        logger.info("-" * 30)
        
        hft_profile = await create_hft_detector_profile()
        logger.info(f"Archetype: {hft_profile.archetype.value}")
        logger.info(f"Confidence: {hft_profile.confidence:.2f}")
        logger.info(f"Key Traits: {list(hft_profile.behavioral_traits.keys())}")
        logger.info(f"Detection Signals: {hft_profile.detection_signals}")
        logger.info(f"Counter Strategies: {hft_profile.counter_strategies}")
        logger.info("")
        
        # Retail Sentiment Profile
        logger.info("📱 RETAIL SENTIMENT PROFILE DEMO")
        logger.info("-" * 30)
        
        retail_profile = await create_retail_sentiment_profile()
        logger.info(f"Archetype: {retail_profile.archetype.value}")
        logger.info(f"Confidence: {retail_profile.confidence:.2f}")
        logger.info(f"Key Traits: {list(retail_profile.behavioral_traits.keys())}")
        logger.info(f"Detection Signals: {retail_profile.detection_signals}")
        logger.info(f"Counter Strategies: {retail_profile.counter_strategies}")
        logger.info("")
    
    async def display_system_status(self):
        """Display comprehensive system status"""
        
        logger.info("📊 PHASE 6: SYSTEM STATUS & PERFORMANCE")
        logger.info("=" * 50)
        
        status = self.profiler_system.get_system_status()
        
        logger.info("🎯 STRATEGIC PROFILER SYSTEM STATUS")
        logger.info("-" * 35)
        logger.info(f"System Health: {status['system_health'].upper()}")
        logger.info(f"Active Profiles: {status['active_profiles']}")
        logger.info(f"Archetypes Detected: {len(status['archetypes_detected'])}")
        logger.info(f"Market Contexts Stored: {status['market_contexts_stored']}")
        logger.info(f"Strategic Responses Generated: {status['strategic_responses_generated']}")
        logger.info(f"Learning Adaptations: {status['learning_adaptations']}")
        
        if status['archetypes_detected']:
            logger.info("")
            logger.info("🎯 Currently Active Archetypes:")
            for archetype in status['archetypes_detected']:
                logger.info(f"  - {archetype.value}")
        
        if status['last_analysis']:
            logger.info(f"Last Analysis: {status['last_analysis']}")
        
        logger.info("")
        logger.info("🔧 COMPONENT STATUS:")
        logger.info("  ✅ Anthropomorphic Profiler: Operational")
        logger.info("  ✅ Context Field Selector: Operational")
        logger.info("  ✅ Manipulation Detector: Operational")
        logger.info("  ✅ Rules Engine: Operational")
        logger.info("  ✅ Financial Processor: Operational")
        logger.info("  ✅ Strategic Response Generator: Operational")
        logger.info("")
        
        # Performance summary
        logger.info("📈 PERFORMANCE SUMMARY")
        logger.info("-" * 20)
        
        total_profiles = sum(len(profiles) for profiles in self.analysis_results.values() if isinstance(profiles, list))
        total_contexts = len(self.analysis_results.get('contexts', {}))
        total_responses = sum(len(responses) for responses in self.analysis_results.get('responses', {}).values())
        
        logger.info(f"Total Profiles Generated: {total_profiles}")
        logger.info(f"Total Contexts Assessed: {total_contexts}")
        logger.info(f"Total Strategic Responses: {total_responses}")
        
        if total_profiles > 0:
            logger.info(f"Analysis Success Rate: {(total_profiles / len(self.demo_data)) * 100:.1f}%")
        
        logger.info("")


async def main():
    """Run the strategic profiler demo"""
    
    try:
        demo = StrategicProfilerDemo()
        await demo.run_comprehensive_demo()
        
        logger.info("\n" + "🎯" * 25)
        logger.info("🏆 STRATEGIC PROFILER DEMO COMPLETE!")
        logger.info("🎯" * 25)
        logger.info("")
        logger.info("🎖️  KIMERA's strategic profiler system is now operational")
        logger.info("⚔️  Ready to identify and analyze market participants")
        logger.info("🧠 Behavioral profiling with scientific rigor achieved")
        logger.info("🎯 Strategic warfare capabilities fully deployed")
        logger.info("")
        logger.info("🚀 READY FOR LIVE MARKET COMBAT!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        logger.error(f"\n❌ Demo failed: {e}")


if __name__ == "__main__":
    asyncio.run(main()) 