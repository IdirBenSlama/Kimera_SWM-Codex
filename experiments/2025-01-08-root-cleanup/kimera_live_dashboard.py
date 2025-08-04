#!/usr/bin/env python3
"""
KIMERA LIVE TRADING DASHBOARD
============================

Real-time monitoring of all active trading systems
- Portfolio value tracking
- Profit/loss in real-time
- Trade execution monitoring
- System status dashboard

Shows live updates of all autonomous trading systems
"""

import asyncio
import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List
from binance.client import Client
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KimeraLiveDashboard:
    """
    Real-time dashboard for monitoring all Kimera trading systems
    """
    
    def __init__(self):
        self.client = Client(
            os.environ.get('BINANCE_API_KEY'),
            os.environ.get('BINANCE_API_SECRET')
        )
        
        self.start_time = datetime.now()
        self.initial_balance = 0.0
        self.update_interval = 15  # Update every 15 seconds
        
        # System status tracking
        self.systems_status = {
            'portfolio_aware_trader': 'UNKNOWN',
            'kimera_server': 'UNKNOWN',
            'autonomous_trader': 'UNKNOWN',
            'profit_trader': 'UNKNOWN'
        }
        
        logger.info("🎯 KIMERA LIVE DASHBOARD INITIALIZED")
    
    async def get_current_portfolio(self) -> Dict:
        """Get current portfolio value"""
        try:
            account = self.client.get_account()
            portfolio = {
                'assets': {},
                'total_value': 0.0,
                'timestamp': datetime.now().isoformat()
            }
            
            for balance in account['balances']:
                asset = balance['asset']
                free = float(balance['free'])
                locked = float(balance['locked'])
                total = free + locked
                
                if total > 0:
                    if asset == 'USDT':
                        value = total
                    else:
                        try:
                            ticker = self.client.get_avg_price(symbol=f"{asset}USDT")
                            price = float(ticker['price'])
                            value = total * price
                        except Exception as e:
                            logger.error(f"Error in kimera_live_dashboard.py: {e}", exc_info=True)
                            raise  # Re-raise for proper error handling
                            value = 0.0
                    
                    portfolio['assets'][asset] = {
                        'amount': total,
                        'free': free,
                        'locked': locked,
                        'value_usd': value
                    }
                    portfolio['total_value'] += value
            
            return portfolio
            
        except Exception as e:
            logger.error(f"Failed to get portfolio: {e}")
            return {'assets': {}, 'total_value': 0.0}
    
    async def check_system_status(self) -> Dict:
        """Check status of all systems"""
        status = {}
        
        # Check Kimera server
        try:
            response = requests.get('http://localhost:8000/health', timeout=5)
            if response.status_code == 200:
                status['kimera_server'] = 'ONLINE'
            else:
                status['kimera_server'] = 'ERROR'
        except Exception as e:
            logger.error(f"Error in kimera_live_dashboard.py: {e}", exc_info=True)
            raise  # Re-raise for proper error handling
            status['kimera_server'] = 'OFFLINE'
        
        # Check for log files to infer system status
        log_files = [
            'kimera_30min_aggressive_*.log',
            'portfolio_aware_report_*.json',
            'autonomous_trading_*.log'
        ]
        
        for log_pattern in log_files:
            # This is a simplified check - in reality you'd scan for actual files
            pass
        
        return status
    
    async def get_recent_trades(self) -> List[Dict]:
        """Get recent trades from Binance"""
        try:
            trades = []
            symbols = ['TRXUSDT', 'ADAUSDT', 'BNBUSDT']
            
            for symbol in symbols:
                try:
                    recent_trades = self.client.get_my_trades(symbol=symbol, limit=10)
                    for trade in recent_trades:
                        trades.append({
                            'symbol': symbol,
                            'side': 'BUY' if trade['isBuyer'] else 'SELL',
                            'quantity': float(trade['qty']),
                            'price': float(trade['price']),
                            'value': float(trade['quoteQty']),
                            'time': datetime.fromtimestamp(trade['time'] / 1000),
                            'commission': float(trade['commission'])
                        })
                except Exception as e:
                    logger.error(f"Error in kimera_live_dashboard.py: {e}", exc_info=True)
                    raise  # Re-raise for proper error handling
                    continue
            
            # Sort by time
            trades.sort(key=lambda x: x['time'], reverse=True)
            return trades[:20]  # Return last 20 trades
            
        except Exception as e:
            logger.error(f"Failed to get recent trades: {e}")
            return []
    
    def print_dashboard(self, portfolio: Dict, trades: List[Dict], profit_data: Dict):
        """Print the live dashboard"""
        os.system('cls' if os.name == 'nt' else 'clear')  # Clear screen
        
        current_time = datetime.now()
        elapsed = current_time - self.start_time
        
        logger.info("🚀" + "="*70 + "🚀")
        logger.info("🎯          KIMERA LIVE AUTONOMOUS TRADING DASHBOARD           🎯")
        logger.info("🚀" + "="*70 + "🚀")
        logger.info(f"⏱️  Session Time: {elapsed}")
        logger.info(f"🕐 Current Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info()
        
        # Portfolio Overview
        logger.info("💰 PORTFOLIO OVERVIEW:")
        logger.info("-" * 50)
        total_value = portfolio.get('total_value', 0)
        if self.initial_balance == 0:
            self.initial_balance = total_value
        
        profit = total_value - self.initial_balance
        profit_pct = (profit / self.initial_balance * 100) if self.initial_balance > 0 else 0
        
        logger.info(f"💵 Total Value: ${total_value:.2f}")
        logger.info(f"📈 Profit/Loss: ${profit:.2f} ({profit_pct:+.2f}%)")
        logger.info(f"🎯 Initial: ${self.initial_balance:.2f}")
        logger.info()
        
        # Asset Breakdown
        logger.info("🪙 ASSET BREAKDOWN:")
        logger.info("-" * 50)
        for asset, data in portfolio.get('assets', {}).items():
            if data['value_usd'] > 0.1:  # Only show assets worth more than $0.10
                logger.info(f"   {asset}: {data['amount']:.8f} = ${data['value_usd']:.2f}")
        logger.info()
        
        # Recent Trades
        logger.info("📊 RECENT TRADES:")
        logger.info("-" * 50)
        if trades:
            for trade in trades[:5]:  # Show last 5 trades
                side_emoji = "🟢" if trade['side'] == 'BUY' else "🔴"
                logger.info(f"   {side_emoji} {trade['symbol']}: {trade['side']} {trade['quantity']:.4f} @ ${trade['price']:.6f}")
        else:
            logger.info("   No recent trades found")
        logger.info()
        
        # System Status
        logger.info("🔧 SYSTEM STATUS:")
        logger.info("-" * 50)
        for system, status in self.systems_status.items():
            status_emoji = "🟢" if status == 'ONLINE' else "🔴" if status == 'ERROR' else "🟡"
            logger.info(f"   {status_emoji} {system}: {status}")
        logger.info()
        
        # Performance Metrics
        logger.info("📈 PERFORMANCE METRICS:")
        logger.info("-" * 50)
        hours_elapsed = elapsed.total_seconds() / 3600
        profit_per_hour = profit / hours_elapsed if hours_elapsed > 0 else 0
        logger.info(f"   💰 Profit/Hour: ${profit_per_hour:.2f}")
        logger.info(f"   📊 Total Trades: {len(trades)}")
        logger.info(f"   ⚡ ROI: {profit_pct:.2f}%")
        logger.info()
        
        logger.info("🚀" + "="*70 + "🚀")
        logger.info("🤖 KIMERA AUTONOMOUS TRADING SYSTEMS ACTIVE")
        logger.info("🎯 Maximum Profit Mode - Full Autonomy Granted")
        logger.info("🚀" + "="*70 + "🚀")
    
    async def run_dashboard(self):
        """Run the live dashboard"""
        logger.info("🎯 Starting live dashboard...")
        
        while True:
            try:
                # Get current data
                portfolio = await self.get_current_portfolio()
                trades = await self.get_recent_trades()
                system_status = await self.check_system_status()
                
                # Update system status
                self.systems_status.update(system_status)
                
                # Calculate profit data
                profit_data = {
                    'current_value': portfolio.get('total_value', 0),
                    'initial_value': self.initial_balance,
                    'profit': portfolio.get('total_value', 0) - self.initial_balance
                }
                
                # Print dashboard
                self.print_dashboard(portfolio, trades, profit_data)
                
                # Wait for next update
                await asyncio.sleep(self.update_interval)
                
            except KeyboardInterrupt:
                logger.info("Dashboard stopped by user")
                break
            except Exception as e:
                logger.error(f"Dashboard error: {e}")
                await asyncio.sleep(5)  # Wait before retrying

async def main():
    """Main dashboard execution"""
    try:
        logger.info("🎯 KIMERA LIVE TRADING DASHBOARD")
        logger.info("=" * 50)
        logger.info("🚀 Real-time monitoring of all trading systems")
        logger.info("💰 Live portfolio tracking")
        logger.info("📊 Trade execution monitoring")
        logger.info("🔧 System status dashboard")
        logger.info("=" * 50)
        logger.info("Press Ctrl+C to stop")
        logger.info()
        
        dashboard = KimeraLiveDashboard()
        await dashboard.run_dashboard()
        
    except Exception as e:
        logger.error(f"Dashboard failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Set environment variables if not already set
    if not os.environ.get('BINANCE_API_KEY'):
        os.environ['BINANCE_API_KEY'] = os.getenv("BINANCE_API_KEY", "")
        os.environ['BINANCE_API_SECRET'] = 'qUn5JqSpYz1GDxFj2X3UF23TYgtxKrTsCbDZEoBMYCPbYZgP4siVLyspkB5HAPl7'
    
    asyncio.run(main()) 