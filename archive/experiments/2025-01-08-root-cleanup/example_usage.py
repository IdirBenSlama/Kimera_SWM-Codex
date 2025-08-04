#!/usr/bin/env python3
"""
KIMERA SEMANTIC TRADING SYSTEM - EXAMPLE USAGE
==============================================

This script demonstrates various ways to use the Kimera Semantic Trading System,
including different configurations, monitoring, and advanced features.

Usage:
    python example_usage.py --mode development
    python example_usage.py --mode production --duration 3600
    python example_usage.py --mode demo
"""

import asyncio
import argparse
import logging
import sys
import os
from datetime import datetime, timedelta

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from kimera_trading_system import create_kimera_trading_system, TradingConfig
from kimera_trading_config import get_trading_config, validate_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('KIMERA_EXAMPLE')

async def development_example():
    """
    Development example - Safe paper trading with limited functionality
    """
    logger.info("🚀 Starting Kimera Trading System - Development Mode")
    logger.info("=" * 60)
    
    # Get development configuration
    config = get_trading_config('development')
    
    # Validate configuration
    issues = validate_config(config)
    if issues:
        logger.error("Configuration validation failed:")
        for issue in issues:
            logger.error(f"  - {issue}")
        return
    
    logger.info("✅ Configuration validated successfully")
    logger.info(f"   Starting Capital: ${config.starting_capital}")
    logger.info(f"   Paper Trading: {config.enable_paper_trading}")
    logger.info(f"   Max Position Size: {config.risk_management.max_position_size * 100}%")
    logger.info(f"   Trading Symbols: {config.trading_symbols}")
    
    # Create trading system
    trading_system = create_kimera_trading_system(config.__dict__)
    
    try:
        # Start the system
        logger.info("\n🎯 Starting trading operations...")
        
        # Start system in background
        system_task = asyncio.create_task(trading_system.start())
        
        # Monitor for 5 minutes
        monitor_duration = 300  # 5 minutes
        start_time = datetime.now()
        
        while (datetime.now() - start_time).total_seconds() < monitor_duration:
            # Get system status
            status = trading_system.get_status()
            
            logger.info(f"\n📊 System Status ({datetime.now().strftime('%H:%M:%S')})")
            logger.info(f"   Portfolio Value: ${status['portfolio']['total_value']:.2f}")
            logger.info(f"   Active Positions: {status['positions']['active_count']}")
            logger.info(f"   Daily PnL: ${status['portfolio']['daily_pnl']:.2f}")
            logger.info(f"   Active Signals: {status['active_signals']}")
            logger.info(f"   Thermodynamic State: T={status['thermodynamic_state']['temperature']:.2f}")
            
            # Wait 30 seconds
            await asyncio.sleep(30)
        
        logger.info("\n✅ Development example completed successfully")
        
    except KeyboardInterrupt:
        logger.info("\n👋 Graceful shutdown requested...")
    except Exception as e:
        logger.error(f"\n❌ Error in development example: {e}")
    finally:
        await trading_system.stop()

async def production_example(duration: int = 3600):
    """
    Production example - Real trading with full functionality
    
    Args:
        duration: How long to run (seconds)
    """
    logger.info("🚀 Starting Kimera Trading System - Production Mode")
    logger.info("=" * 60)
    logger.warning("⚠️  WARNING: This will use real money for trading!")
    
    # Get production configuration
    config = get_trading_config('production')
    
    # Validate configuration
    issues = validate_config(config)
    if issues:
        logger.error("Configuration validation failed:")
        for issue in issues:
            logger.error(f"  - {issue}")
        return
    
    logger.info("✅ Configuration validated successfully")
    logger.info(f"   Starting Capital: ${config.starting_capital}")
    logger.info(f"   Paper Trading: {config.enable_paper_trading}")
    logger.info(f"   Real Trading: {not config.enable_paper_trading}")
    logger.info(f"   Max Position Size: {config.risk_management.max_position_size * 100}%")
    logger.info(f"   Max Daily Trades: {config.risk_management.max_daily_trades}")
    
    # Confirm before proceeding
    if not config.enable_paper_trading:
        response = input("\n🚨 Proceed with REAL MONEY trading? (type 'YES' to confirm): ")
        if response != 'YES':
            logger.info("❌ Production trading cancelled")
            return
    
    # Create trading system
    trading_system = create_kimera_trading_system(config.__dict__)
    
    try:
        # Start the system
        logger.info(f"\n🎯 Starting production trading for {duration} seconds...")
        
        # Start system in background
        system_task = asyncio.create_task(trading_system.start())
        
        # Monitor for specified duration
        start_time = datetime.now()
        last_report = start_time
        
        while (datetime.now() - start_time).total_seconds() < duration:
            current_time = datetime.now()
            
            # Report every 5 minutes
            if (current_time - last_report).total_seconds() >= 300:
                status = trading_system.get_status()
                
                logger.info(f"\n📊 Production Status Report ({current_time.strftime('%H:%M:%S')})")
                logger.info(f"   Portfolio Value: ${status['portfolio']['total_value']:.2f}")
                logger.info(f"   Realized PnL: ${status['portfolio']['realized_pnl']:.2f}")
                logger.info(f"   Unrealized PnL: ${status['portfolio']['unrealized_pnl']:.2f}")
                logger.info(f"   Active Positions: {status['positions']['active_count']}")
                logger.info(f"   Daily Trades: {status['risk_metrics']['daily_trades']}")
                logger.info(f"   Win Rate: {status['performance']['win_rate']:.1%}")
                logger.info(f"   Max Drawdown: {status['performance']['max_drawdown']:.1%}")
                
                last_report = current_time
            
            # Wait 60 seconds
            await asyncio.sleep(60)
        
        logger.info("\n✅ Production trading session completed")
        
        # Final report
        final_status = trading_system.get_status()
        logger.info("\n📈 Final Performance Report")
        logger.info("=" * 40)
        logger.info(f"Final Portfolio Value: ${final_status['portfolio']['total_value']:.2f}")
        logger.info(f"Total PnL: ${final_status['portfolio']['realized_pnl'] + final_status['portfolio']['unrealized_pnl']:.2f}")
        logger.info(f"Total Trades: {final_status['performance']['total_trades']}")
        logger.info(f"Win Rate: {final_status['performance']['win_rate']:.1%}")
        logger.info(f"Sharpe Ratio: {final_status['performance']['sharpe_ratio']:.2f}")
        
    except KeyboardInterrupt:
        logger.info("\n👋 Graceful shutdown requested...")
    except Exception as e:
        logger.error(f"\n❌ Error in production example: {e}")
    finally:
        await trading_system.stop()

async def demo_example():
    """
    Demo example - Showcase all features without real trading
    """
    logger.info("🚀 Starting Kimera Trading System - Demo Mode")
    logger.info("=" * 60)
    
    # Custom demo configuration
    config = get_trading_config('development', {
        'starting_capital': 10000.0,
        'trading_symbols': ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'SOLUSDT', 'XRPUSDT'],
        'risk_management': {
            'max_position_size': 0.20,
            'max_daily_trades': 20
        }
    })
    
    logger.info("✅ Demo configuration loaded")
    logger.info(f"   Starting Capital: ${config.starting_capital}")
    logger.info(f"   Trading Symbols: {len(config.trading_symbols)} pairs")
    logger.info(f"   Max Position Size: {config.risk_management.max_position_size * 100}%")
    
    # Create trading system
    trading_system = create_kimera_trading_system(config.__dict__)
    
    try:
        # Start the system
        logger.info("\n🎯 Starting demo operations...")
        
        # Start system in background
        system_task = asyncio.create_task(trading_system.start())
        
        # Demo various features
        await demo_features(trading_system)
        
        logger.info("\n✅ Demo completed successfully")
        
    except KeyboardInterrupt:
        logger.info("\n👋 Demo interrupted...")
    except Exception as e:
        logger.error(f"\n❌ Error in demo: {e}")
    finally:
        await trading_system.stop()

async def demo_features(trading_system):
    """
    Demonstrate various system features
    """
    logger.info("\n🔍 Demonstrating System Features")
    logger.info("=" * 40)
    
    # Wait for system to initialize
    await asyncio.sleep(10)
    
    # 1. Show real-time market data
    logger.info("\n📊 Real-time Market Data:")
    try:
        btc_data = await trading_system.exchange_connector.get_market_data('BTCUSDT')
        logger.info(f"   BTC/USDT: ${btc_data.price:,.2f} ({btc_data.change_pct_24h:+.2f}%)")
        
        eth_data = await trading_system.exchange_connector.get_market_data('ETHUSDT')
        logger.info(f"   ETH/USDT: ${eth_data.price:,.2f} ({eth_data.change_pct_24h:+.2f}%)")
    except Exception as e:
        logger.warning(f"   Could not fetch market data: {e}")
    
    # 2. Show semantic analysis
    logger.info("\n🧠 Semantic Analysis:")
    thermo_state = trading_system.contradiction_engine.thermodynamic_state
    logger.info(f"   Market Temperature: {thermo_state['temperature']:.3f}")
    logger.info(f"   Market Pressure: {thermo_state['pressure']:.3f}")
    logger.info(f"   Market Entropy: {thermo_state['entropy']:.3f}")
    
    # 3. Show active signals
    logger.info("\n🎯 Active Trading Signals:")
    if trading_system.active_signals:
        for symbol, signal in trading_system.active_signals.items():
            logger.info(f"   {symbol}: {signal.action.upper()} (confidence: {signal.confidence:.2f})")
            logger.info(f"     Strategy: {signal.strategy.value}")
            logger.info(f"     Reasoning: {signal.reasoning[0] if signal.reasoning else 'N/A'}")
    else:
        logger.info("   No active signals at this time")
    
    # 4. Show portfolio status
    logger.info("\n💼 Portfolio Status:")
    status = trading_system.get_status()
    logger.info(f"   Total Value: ${status['portfolio']['total_value']:.2f}")
    logger.info(f"   Daily PnL: ${status['portfolio']['daily_pnl']:.2f}")
    logger.info(f"   Active Positions: {status['positions']['active_count']}")
    logger.info(f"   Available for Trading: {status['positions']['max_allowed'] - status['positions']['active_count']}")
    
    # 5. Show performance metrics
    logger.info("\n📈 Performance Metrics:")
    perf = status['performance']
    logger.info(f"   Total Trades: {perf['total_trades']}")
    logger.info(f"   Win Rate: {perf['win_rate']:.1%}")
    logger.info(f"   Max Drawdown: {perf['max_drawdown']:.1%}")
    logger.info(f"   Sharpe Ratio: {perf['sharpe_ratio']:.2f}")
    
    # 6. Show risk metrics
    logger.info("\n🛡️ Risk Management:")
    risk = status['risk_metrics']
    logger.info(f"   Daily Trades Used: {risk['daily_trades']}/{risk['max_daily_trades']}")
    logger.info(f"   Current Drawdown: {risk['max_drawdown']:.1%}")
    logger.info(f"   Risk Limits: {'Active' if trading_system.config.enable_risk_limits else 'Disabled'}")
    
    # Wait and show updates
    logger.info("\n⏱️ Monitoring for 2 minutes...")
    for i in range(4):
        await asyncio.sleep(30)
        status = trading_system.get_status()
        logger.info(f"   Update {i+1}: Portfolio=${status['portfolio']['total_value']:.2f}, "
                   f"Positions={status['positions']['active_count']}, "
                   f"Signals={status['active_signals']}")

async def custom_strategy_example():
    """
    Example of implementing custom trading strategies
    """
    logger.info("🚀 Custom Strategy Example")
    logger.info("=" * 40)
    
    config = get_trading_config('testing')
    trading_system = create_kimera_trading_system(config.__dict__)
    
    # Custom signal generation function
    async def generate_custom_signal(symbol: str):
        """Generate a custom trading signal"""
        try:
            # Get market data
            market_data = await trading_system.exchange_connector.get_market_data(symbol)
            
            # Simple custom strategy: Buy on strong positive momentum
            if market_data.change_pct_24h > 5.0:
                logger.info(f"🎯 Custom signal: BUY {symbol} (momentum: {market_data.change_pct_24h:.1f}%)")
                return {
                    'symbol': symbol,
                    'action': 'buy',
                    'confidence': min(market_data.change_pct_24h / 10.0, 1.0),
                    'reasoning': f"Strong positive momentum: {market_data.change_pct_24h:.1f}%"
                }
            elif market_data.change_pct_24h < -5.0:
                logger.info(f"🎯 Custom signal: SELL {symbol} (momentum: {market_data.change_pct_24h:.1f}%)")
                return {
                    'symbol': symbol,
                    'action': 'sell',
                    'confidence': min(abs(market_data.change_pct_24h) / 10.0, 1.0),
                    'reasoning': f"Strong negative momentum: {market_data.change_pct_24h:.1f}%"
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating custom signal for {symbol}: {e}")
            return None
    
    try:
        # Start system
        system_task = asyncio.create_task(trading_system.start())
        await asyncio.sleep(5)  # Let system initialize
        
        # Run custom strategy for 2 minutes
        for _ in range(4):
            for symbol in ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']:
                signal = await generate_custom_signal(symbol)
                if signal:
                    logger.info(f"   Generated signal: {signal}")
            
            await asyncio.sleep(30)
        
        logger.info("✅ Custom strategy example completed")
        
    except Exception as e:
        logger.error(f"❌ Error in custom strategy example: {e}")
    finally:
        await trading_system.stop()

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description='Kimera Trading System Examples')
    parser.add_argument('--mode', choices=['development', 'production', 'demo', 'custom'], 
                       default='development', help='Example mode to run')
    parser.add_argument('--duration', type=int, default=3600, 
                       help='Duration to run (seconds, for production mode)')
    
    args = parser.parse_args()
    
    logger.info("🌟 KIMERA SEMANTIC TRADING SYSTEM")
    logger.info("🌟 Advanced Autonomous Trading Platform")
    logger.info("=" * 60)
    logger.info(f"Mode: {args.mode.upper()}")
    logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        if args.mode == 'development':
            asyncio.run(development_example())
        elif args.mode == 'production':
            asyncio.run(production_example(args.duration))
        elif args.mode == 'demo':
            asyncio.run(demo_example())
        elif args.mode == 'custom':
            asyncio.run(custom_strategy_example())
    except KeyboardInterrupt:
        logger.info("\n👋 Example terminated by user")
    except Exception as e:
        logger.error(f"\n❌ Example failed: {e}")
        sys.exit(1)
    
    logger.info(f"\n✅ Example completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 