"""
KIMERA Semantic Trading Demo
============================

Demonstrates how to use the semantic trading module with Kimera's reactor.
This example shows the complete flow from market data ingestion to trade execution.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import the trading module
from backend.trading import (
    create_kimera_trading_system,
    process_trading_opportunity,
    KimeraTradingConfig
)


async def simulate_market_event() -> Dict[str, Any]:
    """Simulate a market event with price data and context"""
    import random
    
    # Simulate market data
    base_price = 50000  # BTC price
    volatility = random.uniform(0.01, 0.03)
    momentum = random.uniform(-0.02, 0.02)
    
    return {
        'market_data': {
            'symbol': 'BTC-USD',
            'price': base_price * (1 + random.uniform(-volatility, volatility)),
            'volume': random.uniform(1000, 5000),
            'bid': base_price * 0.999,
            'ask': base_price * 1.001,
            'momentum': momentum,
            'volatility': volatility,
            'trend': 'bullish' if momentum > 0 else 'bearish',
            'support': base_price * 0.95,
            'resistance': base_price * 1.05,
            'technical_indicators': {
                'rsi': random.uniform(30, 70),
                'macd': random.uniform(-100, 100),
                'bollinger_position': random.uniform(0, 1)
            }
        },
        'context': {
            'market_phase': random.choice(['accumulation', 'distribution', 'trending']),
            'global_sentiment': random.uniform(-1, 1)
        },
        'timestamp': datetime.now().isoformat()
    }


async def run_trading_demo():
    """Run the semantic trading demonstration"""
    
    logger.info("=" * 60)
    logger.info("KIMERA Semantic Trading Demo")
    logger.info("=" * 60)
    
    # Configuration for the trading system
    config = KimeraTradingConfig(
        tension_threshold=0.4,  # Sensitivity to contradictions
        max_position_size=1000,  # Maximum position size in USD
        risk_per_trade=0.02,  # 2% risk per trade
        enable_paper_trading=True,  # Use paper trading for demo
        enable_sentiment_analysis=True,
        enable_news_processing=True,
        dashboard_port=8050
    )
    
    # Create the trading system
    logger.info("\n🚀 Initializing KIMERA Trading System...")
    trading_system = create_kimera_trading_system(config.__dict__)
    
    # Start the system
    await trading_system.start()
    logger.info("✅ Trading system started successfully")
    
    # Run the dashboard in background
    logger.info(f"\n📊 Dashboard available at http://localhost:{config.dashboard_port}")
    
    # Simulate trading for a period
    logger.info("\n📈 Starting trading simulation...")
    logger.info("-" * 60)
    
    for i in range(20):  # Simulate 20 market events
        # Generate market event
        market_event = await simulate_market_event()
        
        logger.info(f"\n[Event {i+1}] Processing market data:")
        logger.info(f"  Symbol: {market_event['market_data']['symbol']}")
        logger.info(f"  Price: ${market_event['market_data']['price']:,.2f}")
        logger.info(f"  Momentum: {market_event['market_data']['momentum']:.3f}")
        logger.info(f"  Volatility: {market_event['market_data']['volatility']:.3f}")
        
        # Process through trading system
        result = await trading_system.process_market_event(market_event)
        
        # Display results
        if result['status'] == 'executed':
            logger.info(f"  ✅ TRADE EXECUTED:")
            logger.info(f"     Action: {result['analysis'].action_taken}")
            logger.info(f"     Confidence: {result['analysis'].confidence:.2%}")
            logger.info(f"     Contradictions: {len(result['analysis'].contradiction_map)
            
            if 'execution' in result:
                logger.info(f"     Fill Price: ${result['execution'].average_price:,.2f}")
                logger.info(f"     Quantity: {result['execution'].filled_quantity}")
                logger.info(f"     Latency: {result['execution'].execution_time*1000:.1f}ms")
        
        elif result['status'] == 'monitored':
            logger.info(f"  👁️  MONITORING - No action taken")
            logger.info(f"     Reason: {result.get('reason', 'unknown')
            logger.info(f"     Confidence: {result['analysis'].confidence:.2%}")
        
        else:
            logger.error(f"  ❌ ERROR: {result.get('error', 'unknown error')
        
        # Show semantic analysis
        if 'analysis' in result and result['analysis'].semantic_analysis:
            logger.info(f"  📊 Semantic Analysis:")
            for key, value in result['analysis'].semantic_analysis.items():
                logger.info(f"     {key}: {value:.3f}")
        
        # Wait before next event
        await asyncio.sleep(2)
    
    # Get final performance summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 PERFORMANCE SUMMARY")
    logger.info("=" * 60)
    
    status = trading_system.get_status()
    perf_summary = status.get('performance_summary', {})
    
    if perf_summary:
        current_metrics = perf_summary.get('current_metrics', {})
        logger.info(f"Total P&L: ${current_metrics.get('total_pnl', 0)
        logger.info(f"Win Rate: {current_metrics.get('win_rate', 0)
        logger.info(f"Sharpe Ratio: {current_metrics.get('sharpe_ratio', 0)
        logger.info(f"Max Drawdown: {current_metrics.get('max_drawdown', 0)
        logger.info(f"Active Positions: {current_metrics.get('active_positions', 0)
        
        trade_stats = perf_summary.get('trade_statistics', {})
        logger.info(f"\nTotal Trades: {trade_stats.get('total_trades', 0)
        logger.info(f"Avg Execution Time: {trade_stats.get('avg_execution_time', 0)
        
        contradiction_analysis = perf_summary.get('contradiction_analysis', {})
        logger.info(f"\nContradictions Detected: {contradiction_analysis.get('total_detected', 0)
        logger.info(f"Avg Tension Score: {contradiction_analysis.get('avg_tension_score', 0)
    
    # Stop the system
    logger.info("\n🛑 Stopping trading system...")
    await trading_system.stop()
    logger.info("✅ Trading system stopped")


async def simple_trading_example():
    """Simple example using the convenience function"""
    logger.info("\n" + "=" * 60)
    logger.info("Simple Trading Example")
    logger.info("=" * 60)
    
    # Create a market event
    market_event = {
        'market_data': {
            'symbol': 'ETH-USD',
            'price': 3000,
            'volume': 5000,
            'momentum': 0.02,
            'volatility': 0.015
        },
        'context': {
            'news_sentiment': 0.7,  # Positive news
            'social_sentiment': -0.3  # Negative social sentiment
        }
    }
    
    logger.debug("\n🔍 Analyzing contradiction between positive news and negative social sentiment...")
    
    # Process through the convenience function
    result = await process_trading_opportunity(market_event)
    
    logger.info(f"\nResult: {result['status']}")
    if 'analysis' in result:
        logger.info(f"Contradictions found: {len(result['analysis'].contradiction_map)
        logger.info(f"Recommended action: {result['analysis'].action_taken}")
        logger.info(f"Confidence: {result['analysis'].confidence:.2%}")


if __name__ == "__main__":
    # Run the full demo
    asyncio.run(run_trading_demo())
    
    # Run the simple example
    # asyncio.run(simple_trading_example()) 