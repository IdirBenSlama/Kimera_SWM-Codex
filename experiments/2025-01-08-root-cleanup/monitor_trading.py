"""
KIMERA TRADING MONITOR
======================

Real-time monitoring dashboard for micro-trading system.
Displays current status, positions, P&L, and safety metrics.
"""

import json
import time
import os
from datetime import datetime
from pathlib import Path

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def load_trading_state():
    """Load current trading state"""
    try:
        if os.path.exists('data/trading_state.json'):
            with open('data/trading_state.json', 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error in monitor_trading.py: {e}", exc_info=True)
        raise  # Re-raise for proper error handling
    return {}

def load_micro_trades():
    """Load micro-trading history"""
    trades = []
    try:
        if os.path.exists('data/micro_trades.jsonl'):
            with open('data/micro_trades.jsonl', 'r') as f:
                for line in f:
                    trades.append(json.loads(line.strip()))
    except Exception as e:
        logger.error(f"Error in monitor_trading.py: {e}", exc_info=True)
        raise  # Re-raise for proper error handling
    return trades

def get_latest_log_entries():
    """Get latest log entries"""
    try:
        if os.path.exists('logs/micro_trading.log'):
            with open('logs/micro_trading.log', 'r') as f:
                lines = f.readlines()
                return lines[-10:]  # Last 10 lines
    except Exception as e:
        logger.error(f"Error in monitor_trading.py: {e}", exc_info=True)
        raise  # Re-raise for proper error handling
    return []

def display_dashboard():
    """Display real-time trading dashboard"""
    clear_screen()
    
    print("🎯 KIMERA MICRO-TRADING MONITOR")
    print("=" * 60)
    print(f"⏰ Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load current state
    state = load_trading_state()
    trades = load_micro_trades()
    logs = get_latest_log_entries()
    
    # System Status
    print("🔐 SYSTEM STATUS:")
    print(f"   Mode: {'ACTIVE' if state else 'INITIALIZING'}")
    print(f"   Daily P&L: €{state.get('daily_pnl', 0.0):.2f}")
    print(f"   Consecutive Losses: {state.get('consecutive_losses', 0)}")
    print(f"   Emergency Stop: {'🚨 ACTIVE' if state.get('emergency_stop') else '✅ NORMAL'}")
    print()
    
    # Trading Summary
    print("📊 TRADING SUMMARY:")
    if trades:
        total_trades = len(trades)
        total_amount = sum(t.get('amount_eur', 0) for t in trades)
        latest_trade = trades[-1] if trades else None
        
        print(f"   Total Trades: {total_trades}")
        print(f"   Total Amount: €{total_amount:.2f}")
        print(f"   Average Size: €{total_amount/max(total_trades, 1):.3f}")
        
        if latest_trade:
            print(f"   Latest Trade: {latest_trade.get('side', 'N/A')} {latest_trade.get('symbol', 'N/A')}")
            print(f"   Trade Amount: €{latest_trade.get('amount_eur', 0):.2f}")
    else:
        print("   No trades executed yet")
    print()
    
    # Safety Metrics
    print("🛡️ SAFETY METRICS:")
    print(f"   Max Position: €0.10")
    print(f"   Daily Limit: €0.50")
    print(f"   Min Balance: €4.50")
    print(f"   Confidence Req: 80%")
    print()
    
    # Recent Activity
    print("📝 RECENT ACTIVITY:")
    if logs:
        for log in logs[-5:]:  # Last 5 log entries
            if log.strip():
                # Parse log entry
                try:
                    parts = log.split(' - ', 2)
                    if len(parts) >= 3:
                        timestamp = parts[0].split(', ')[-1] if ', ' in parts[0] else parts[0]
                        level = parts[1]
                        message = parts[2].strip()
                        
                        # Color code by level
                        if level == 'INFO':
                            color = '💙'
                        elif level == 'WARNING':
                            color = '⚠️'
                        elif level == 'ERROR':
                            color = '❌'
                        else:
                            color = '📝'
                        
                        print(f"   {color} {timestamp} - {message[:60]}...")
                    else:
                        print(f"   📝 {log.strip()[:70]}...")
                except Exception as e:
                    logger.error(f"Error in monitor_trading.py: {e}", exc_info=True)
                    raise  # Re-raise for proper error handling
                    print(f"   📝 {log.strip()[:70]}...")
    else:
        print("   System starting up...")
    print()
    
    # Instructions
    print("🎮 CONTROLS:")
    print("   Press Ctrl+C to stop monitoring")
    print("   Emergency stop: python emergency_stop.py")
    print("   View full logs: tail -f logs/micro_trading.log")

def main():
    """Main monitoring loop"""
    print("🚀 Starting Kimera Trading Monitor...")
    print("Press Ctrl+C to exit")
    time.sleep(2)
    
    try:
        while True:
            display_dashboard()
            time.sleep(5)  # Update every 5 seconds
    except KeyboardInterrupt:
        print("\n\n👋 Monitor stopped by user")
    except Exception as e:
        print(f"\n\n❌ Monitor error: {e}")

if __name__ == "__main__":
    main() 