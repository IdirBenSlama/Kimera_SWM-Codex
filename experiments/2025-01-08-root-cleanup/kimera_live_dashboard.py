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
                        except:
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
        except:
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
                except:
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
        
        print("🚀" + "="*70 + "🚀")
        print("🎯          KIMERA LIVE AUTONOMOUS TRADING DASHBOARD           🎯")
        print("🚀" + "="*70 + "🚀")
        print(f"⏱️  Session Time: {elapsed}")
        print(f"🕐 Current Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Portfolio Overview
        print("💰 PORTFOLIO OVERVIEW:")
        print("-" * 50)
        total_value = portfolio.get('total_value', 0)
        if self.initial_balance == 0:
            self.initial_balance = total_value
        
        profit = total_value - self.initial_balance
        profit_pct = (profit / self.initial_balance * 100) if self.initial_balance > 0 else 0
        
        print(f"💵 Total Value: ${total_value:.2f}")
        print(f"📈 Profit/Loss: ${profit:.2f} ({profit_pct:+.2f}%)")
        print(f"🎯 Initial: ${self.initial_balance:.2f}")
        print()
        
        # Asset Breakdown
        print("🪙 ASSET BREAKDOWN:")
        print("-" * 50)
        for asset, data in portfolio.get('assets', {}).items():
            if data['value_usd'] > 0.1:  # Only show assets worth more than $0.10
                print(f"   {asset}: {data['amount']:.8f} = ${data['value_usd']:.2f}")
        print()
        
        # Recent Trades
        print("📊 RECENT TRADES:")
        print("-" * 50)
        if trades:
            for trade in trades[:5]:  # Show last 5 trades
                side_emoji = "🟢" if trade['side'] == 'BUY' else "🔴"
                print(f"   {side_emoji} {trade['symbol']}: {trade['side']} {trade['quantity']:.4f} @ ${trade['price']:.6f}")
        else:
            print("   No recent trades found")
        print()
        
        # System Status
        print("🔧 SYSTEM STATUS:")
        print("-" * 50)
        for system, status in self.systems_status.items():
            status_emoji = "🟢" if status == 'ONLINE' else "🔴" if status == 'ERROR' else "🟡"
            print(f"   {status_emoji} {system}: {status}")
        print()
        
        # Performance Metrics
        print("📈 PERFORMANCE METRICS:")
        print("-" * 50)
        hours_elapsed = elapsed.total_seconds() / 3600
        profit_per_hour = profit / hours_elapsed if hours_elapsed > 0 else 0
        print(f"   💰 Profit/Hour: ${profit_per_hour:.2f}")
        print(f"   📊 Total Trades: {len(trades)}")
        print(f"   ⚡ ROI: {profit_pct:.2f}%")
        print()
        
        print("🚀" + "="*70 + "🚀")
        print("🤖 KIMERA AUTONOMOUS TRADING SYSTEMS ACTIVE")
        print("🎯 Maximum Profit Mode - Full Autonomy Granted")
        print("🚀" + "="*70 + "🚀")
    
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
        print("🎯 KIMERA LIVE TRADING DASHBOARD")
        print("=" * 50)
        print("🚀 Real-time monitoring of all trading systems")
        print("💰 Live portfolio tracking")
        print("📊 Trade execution monitoring")
        print("🔧 System status dashboard")
        print("=" * 50)
        print("Press Ctrl+C to stop")
        print()
        
        dashboard = KimeraLiveDashboard()
        await dashboard.run_dashboard()
        
    except Exception as e:
        logger.error(f"Dashboard failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Set environment variables if not already set
    if not os.environ.get('BINANCE_API_KEY'):
        os.environ['BINANCE_API_KEY'] = 'Y9WyflPyK1tVXnET3CTMvSdCbPia3Nhtd89VYWjS9RaAbQ0KEhHezkcGSCySQ8cL'
        os.environ['BINANCE_API_SECRET'] = 'qUn5JqSpYz1GDxFj2X3UF23TYgtxKrTsCbDZEoBMYCPbYZgP4siVLyspkB5HAPl7'
    
    asyncio.run(main()) 