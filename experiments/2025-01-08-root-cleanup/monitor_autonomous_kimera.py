#!/usr/bin/env python3
"""
KIMERA AUTONOMOUS TRADER MONITOR
================================

Real-time monitoring dashboard for autonomous Kimera trading.
Track portfolio growth, active positions, and AI decision-making.
"""

import os
import sys
import json
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any
import logging
logger = logging.getLogger(__name__)

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def clear_screen():
    """Clear terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def load_autonomous_state() -> Dict[str, Any]:
    """Load current autonomous trading state"""
    try:
        if os.path.exists('data/autonomous_state.json'):
            with open('data/autonomous_state.json', 'r') as f:
                return json.load(f)
        return {}
    except Exception:
        return {}

def calculate_performance_metrics(state: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate performance metrics"""
    portfolio_value = state.get('portfolio_value', 5.0)
    start_capital = 5.0
    target = 100.0
    
    # Growth metrics
    total_growth = ((portfolio_value / start_capital) - 1) * 100
    progress_to_target = (portfolio_value / target) * 100
    
    # Trade metrics
    total_trades = state.get('total_trades', 0)
    wins = state.get('wins', 0)
    losses = state.get('losses', 0)
    win_rate = (wins / max(total_trades, 1)) * 100
    
    # Strategy performance
    strategy_perf = state.get('strategy_performance', {})
    
    return {
        'portfolio_value': portfolio_value,
        'total_growth_pct': total_growth,
        'progress_to_target_pct': progress_to_target,
        'total_trades': total_trades,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'current_strategy': state.get('current_strategy', 'unknown'),
        'market_regime': state.get('market_regime', 'unknown'),
        'strategy_performance': strategy_perf
    }

def format_currency(amount: float) -> str:
    """Format currency with color coding"""
    if amount > 0:
        return f"\033[92m€{amount:.2f}\033[0m"  # Green
    elif amount < 0:
        return f"\033[91m€{amount:.2f}\033[0m"  # Red
    else:
        return f"€{amount:.2f}"

def format_percentage(pct: float) -> str:
    """Format percentage with color coding"""
    if pct > 0:
        return f"\033[92m+{pct:.2f}%\033[0m"  # Green
    elif pct < 0:
        return f"\033[91m{pct:.2f}%\033[0m"   # Red
    else:
        return f"{pct:.2f}%"

def display_banner():
    """Display monitoring banner"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║               KIMERA AUTONOMOUS TRADER MONITOR               ║
║                                                              ║
║                🧠 REAL-TIME AI MONITORING 🧠                ║
╚══════════════════════════════════════════════════════════════╝
    """
    logger.info(banner)

def display_portfolio_overview(metrics: Dict[str, Any]):
    """Display portfolio overview"""
    logger.info("┌─────────────────────────────────────────────────────────────┐")
    logger.info("│                    PORTFOLIO OVERVIEW                       │")
    logger.info("├─────────────────────────────────────────────────────────────┤")
    
    portfolio_value = metrics['portfolio_value']
    growth = metrics['total_growth_pct']
    progress = metrics['progress_to_target_pct']
    
    logger.info(f"│ Current Value:    {format_currency(portfolio_value):<20} │")
    logger.info(f"│ Total Growth:     {format_percentage(growth):<20} │")
    logger.info(f"│ Target Progress:  {format_percentage(progress):<20} │")
    logger.info(f"│ Remaining:        {format_currency(100 - portfolio_value):<20} │")
    
    # Progress bar
    bar_length = 40
    filled = int((progress / 100) * bar_length)
    bar = "█" * filled + "░" * (bar_length - filled)
    logger.info(f"│ Progress Bar:     [{bar}] │")
    
    logger.info("└─────────────────────────────────────────────────────────────┘")

def display_trading_stats(metrics: Dict[str, Any]):
    """Display trading statistics"""
    logger.info("┌─────────────────────────────────────────────────────────────┐")
    logger.info("│                    TRADING STATISTICS                       │")
    logger.info("├─────────────────────────────────────────────────────────────┤")
    
    total_trades = metrics['total_trades']
    wins = metrics['wins']
    losses = metrics['losses']
    win_rate = metrics['win_rate']
    
    logger.info(f"│ Total Trades:     {total_trades:<30} │")
    logger.info(f"│ Wins:             \033[92m{wins}\033[0m{'':<27} │")
    logger.info(f"│ Losses:           \033[91m{losses}\033[0m{'':<27} │")
    logger.info(f"│ Win Rate:         {format_percentage(win_rate):<20} │")
    
    logger.info("└─────────────────────────────────────────────────────────────┘")

def display_ai_status(metrics: Dict[str, Any]):
    """Display AI decision-making status"""
    logger.info("┌─────────────────────────────────────────────────────────────┐")
    logger.info("│                      AI STATUS                              │")
    logger.info("├─────────────────────────────────────────────────────────────┤")
    
    strategy = metrics['current_strategy'].replace('_', ' ').title()
    regime = metrics['market_regime'].replace('_', ' ').title()
    
    logger.info(f"│ Current Strategy: {strategy:<30} │")
    logger.info(f"│ Market Regime:    {regime:<30} │")
    logger.info(f"│ AI Mode:          \033[93mFULLY AUTONOMOUS\033[0m{'':<18} │")
    logger.info(f"│ Safety Limits:    \033[91mNONE\033[0m{'':<26} │")
    
    logger.info("└─────────────────────────────────────────────────────────────┘")

def display_strategy_performance(metrics: Dict[str, Any]):
    """Display strategy performance breakdown"""
    logger.info("┌─────────────────────────────────────────────────────────────┐")
    logger.info("│                  STRATEGY PERFORMANCE                       │")
    logger.info("├─────────────────────────────────────────────────────────────┤")
    
    strategy_perf = metrics['strategy_performance']
    
    if strategy_perf:
        for strategy, performance in strategy_perf.items():
            strategy_name = strategy.replace('_', ' ').title()[:20]
            perf_str = format_percentage(performance * 100)
            logger.info(f"│ {strategy_name:<20}: {perf_str:<15} │")
    else:
        logger.info("│ No strategy performance data available yet              │")
    
    logger.info("└─────────────────────────────────────────────────────────────┘")

def display_recent_log_entries():
    """Display recent log entries"""
    logger.info("┌─────────────────────────────────────────────────────────────┐")
    logger.info("│                     RECENT ACTIVITY                         │")
    logger.info("├─────────────────────────────────────────────────────────────┤")
    
    try:
        log_file = 'logs/autonomous_kimera.log'
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            # Get last 5 lines
            recent_lines = lines[-5:] if len(lines) >= 5 else lines
            
            for line in recent_lines:
                if line.strip():
                    # Extract timestamp and message
                    parts = line.strip().split(' - ', 3)
                    if len(parts) >= 4:
                        timestamp = parts[0]
                        level = parts[2]
                        message = parts[3]
                        
                        # Color code by level
                        if level == 'INFO':
                            level_colored = f"\033[94m{level}\033[0m"
                        elif level == 'ERROR':
                            level_colored = f"\033[91m{level}\033[0m"
                        elif level == 'WARNING':
                            level_colored = f"\033[93m{level}\033[0m"
                        else:
                            level_colored = level
                        
                        # Truncate message to fit
                        if len(message) > 45:
                            message = message[:42] + "..."
                        
                        time_part = timestamp.split(' ')[1][:8]  # HH:MM:SS
                        logger.info(f"│ {time_part} {level_colored} {message:<35} │")
        else:
            logger.info("│ No log file found                                       │")
    
    except Exception as e:
        logger.info(f"│ Error reading logs: {str(e):<35} │")
    
    logger.info("└─────────────────────────────────────────────────────────────┘")

def display_time_info():
    """Display current time and runtime info"""
    now = datetime.now()
    logger.info(f"\nLast Updated: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("Press Ctrl+C to exit monitoring")

async def monitor_loop():
    """Main monitoring loop"""
    logger.info("🚀 Starting Kimera Autonomous Trader Monitor...")
    logger.info("   Monitoring autonomous trading activity...")
    
    try:
        while True:
            clear_screen()
            
            # Load current state
            state = load_autonomous_state()
            metrics = calculate_performance_metrics(state)
            
            # Display dashboard
            display_banner()
            display_portfolio_overview(metrics)
            logger.info()
            display_trading_stats(metrics)
            logger.info()
            display_ai_status(metrics)
            logger.info()
            display_strategy_performance(metrics)
            logger.info()
            display_recent_log_entries()
            
            display_time_info()
            
            # Check if target reached
            if metrics['portfolio_value'] >= 100.0:
                logger.info("\n🎉 TARGET REACHED! Kimera has achieved the €100 goal!")
                break
            
            # Wait 10 seconds before refresh
            await asyncio.sleep(10)
    
    except KeyboardInterrupt:
        logger.info("\n👋 Monitoring stopped by user")
    except Exception as e:
        logger.info(f"\n❌ Monitoring error: {e}")

def main():
    """Main function"""
    try:
        asyncio.run(monitor_loop())
    except KeyboardInterrupt:
        logger.info("\n👋 Goodbye!")
    except Exception as e:
        logger.info(f"❌ Monitor failed: {e}")

if __name__ == "__main__":
    main() 