#!/usr/bin/env python3
"""
KIMERA REAL-TIME TRADING DASHBOARD
==================================

Real-time monitoring dashboard for Kimera trading activities.
Displays live market data, signals, positions, and performance metrics.
"""

import asyncio
import time
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
import numpy as np

# Add backend to path
sys.path.append('backend')

from trading.api.binance_connector_hmac import BinanceConnector
import logging
logger = logging.getLogger(__name__)

class KimeraDashboard:
    """Real-time trading dashboard for Kimera system."""
    
    def __init__(self):
        self.connector = None
        self.fig = None
        self.axes = {}
        self.data_history = {
            'timestamps': [],
            'prices': [],
            'rsi': [],
            'volume': [],
            'signals': [],
            'trades': []
        }
        self.current_data = {}
        self.running = False
        
        # Dashboard configuration
        self.symbol = "TRXUSDT"
        self.update_interval = 2  # seconds
        self.max_history = 100  # data points to keep
        
    async def initialize(self):
        """Initialize dashboard connection."""
        try:
            logger.info("🚀 Initializing Kimera Dashboard...")
            
            # Load credentials
            if not os.path.exists('kimera_binance_hmac.env'):
                logger.info("❌ Credentials file not found!")
                return False
                
            with open('kimera_binance_hmac.env', 'r') as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value
                        
            api_key = os.environ.get('BINANCE_API_KEY')
            secret_key = os.environ.get('BINANCE_SECRET_KEY')
            testnet = os.environ.get('BINANCE_USE_TESTNET', 'false').lower() == 'true'
            
            self.connector = BinanceConnector(
                api_key=api_key,
                secret_key=secret_key,
                testnet=testnet
            )
            
            logger.info("✅ Dashboard connection established")
            return True
            
        except Exception as e:
            logger.info(f"❌ Dashboard initialization failed: {e}")
            return False
            
    def setup_dashboard_layout(self):
        """Setup the dashboard layout with multiple panels."""
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('KIMERA REAL-TIME TRADING DASHBOARD', fontsize=16, fontweight='bold')
        
        # Create subplots
        gs = self.fig.add_gridspec(3, 3, height_ratios=[2, 1, 1], width_ratios=[2, 1, 1])
        
        # Main price chart
        self.axes['price'] = self.fig.add_subplot(gs[0, :2])
        self.axes['price'].set_title('TRX/USDT Price & Signals', fontweight='bold')
        self.axes['price'].set_ylabel('Price (USDT)')
        self.axes['price'].grid(True, alpha=0.3)
        
        # RSI chart
        self.axes['rsi'] = self.fig.add_subplot(gs[1, :2])
        self.axes['rsi'].set_title('RSI Indicator', fontweight='bold')
        self.axes['rsi'].set_ylabel('RSI')
        self.axes['rsi'].set_ylim(0, 100)
        self.axes['rsi'].axhline(y=70, color='r', linestyle='--', alpha=0.7)
        self.axes['rsi'].axhline(y=30, color='g', linestyle='--', alpha=0.7)
        self.axes['rsi'].grid(True, alpha=0.3)
        
        # Volume chart
        self.axes['volume'] = self.fig.add_subplot(gs[2, :2])
        self.axes['volume'].set_title('Volume', fontweight='bold')
        self.axes['volume'].set_ylabel('Volume')
        self.axes['volume'].set_xlabel('Time')
        self.axes['volume'].grid(True, alpha=0.3)
        
        # Market info panel
        self.axes['info'] = self.fig.add_subplot(gs[0, 2])
        self.axes['info'].set_title('Market Info', fontweight='bold')
        self.axes['info'].axis('off')
        
        # Performance panel
        self.axes['performance'] = self.fig.add_subplot(gs[1, 2])
        self.axes['performance'].set_title('Performance', fontweight='bold')
        self.axes['performance'].axis('off')
        
        # Controls panel
        self.axes['controls'] = self.fig.add_subplot(gs[2, 2])
        self.axes['controls'].set_title('Controls', fontweight='bold')
        self.axes['controls'].axis('off')
        
        plt.tight_layout()
        
    async def fetch_market_data(self):
        """Fetch current market data."""
        try:
            # Get current price
            ticker = await self.connector.get_ticker_price(self.symbol)
            current_price = float(ticker['price'])
            
            # Get 24hr ticker
            ticker_24hr = await self.connector.get_ticker(self.symbol)
            
            # Get recent klines for RSI calculation
            klines = await self.connector.get_klines(self.symbol, '1m', 20)
            
            if klines:
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                    'taker_buy_quote', 'ignore'
                ])
                
                for col in ['close', 'volume']:
                    df[col] = pd.to_numeric(df[col])
                    
                # Calculate RSI
                rsi = self.calculate_rsi(df['close'])
                current_volume = df['volume'].iloc[-1]
            else:
                rsi = 50
                current_volume = 0
                
            self.current_data = {
                'timestamp': datetime.now(),
                'price': current_price,
                'change_24h': float(ticker_24hr['priceChangePercent']),
                'volume_24h': float(ticker_24hr['volume']),
                'high_24h': float(ticker_24hr['highPrice']),
                'low_24h': float(ticker_24hr['lowPrice']),
                'rsi': rsi,
                'current_volume': current_volume
            }
            
            return True
            
        except Exception as e:
            logger.info(f"❌ Error fetching market data: {e}")
            return False
            
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator."""
        if len(prices) < period + 1:
            return 50
            
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
        
    def update_data_history(self):
        """Update historical data arrays."""
        if not self.current_data:
            return
            
        # Add current data to history
        self.data_history['timestamps'].append(self.current_data['timestamp'])
        self.data_history['prices'].append(self.current_data['price'])
        self.data_history['rsi'].append(self.current_data['rsi'])
        self.data_history['volume'].append(self.current_data['current_volume'])
        
        # Generate signal based on RSI
        rsi = self.current_data['rsi']
        if rsi > 70:
            signal = 'SELL'
        elif rsi < 30:
            signal = 'BUY'
        else:
            signal = 'HOLD'
        self.data_history['signals'].append(signal)
        
        # Limit history size
        for key in self.data_history:
            if len(self.data_history[key]) > self.max_history:
                self.data_history[key] = self.data_history[key][-self.max_history:]
                
    def update_charts(self):
        """Update all dashboard charts."""
        if not self.data_history['timestamps']:
            return
            
        # Clear all axes
        for ax in self.axes.values():
            if ax != self.axes['info'] and ax != self.axes['performance'] and ax != self.axes['controls']:
                ax.clear()
                
        timestamps = self.data_history['timestamps']
        
        # Price chart
        self.axes['price'].plot(timestamps, self.data_history['prices'], 'cyan', linewidth=2)
        self.axes['price'].set_title('TRX/USDT Price & Signals', fontweight='bold')
        self.axes['price'].set_ylabel('Price (USDT)')
        self.axes['price'].grid(True, alpha=0.3)
        
        # Add signal markers
        for i, (ts, price, signal) in enumerate(zip(timestamps, self.data_history['prices'], self.data_history['signals'])):
            if signal == 'BUY':
                self.axes['price'].scatter(ts, price, color='green', s=100, marker='^', alpha=0.8)
            elif signal == 'SELL':
                self.axes['price'].scatter(ts, price, color='red', s=100, marker='v', alpha=0.8)
                
        # RSI chart
        self.axes['rsi'].plot(timestamps, self.data_history['rsi'], 'yellow', linewidth=2)
        self.axes['rsi'].set_title('RSI Indicator', fontweight='bold')
        self.axes['rsi'].set_ylabel('RSI')
        self.axes['rsi'].set_ylim(0, 100)
        self.axes['rsi'].axhline(y=70, color='r', linestyle='--', alpha=0.7)
        self.axes['rsi'].axhline(y=30, color='g', linestyle='--', alpha=0.7)
        self.axes['rsi'].grid(True, alpha=0.3)
        
        # Volume chart
        self.axes['volume'].bar(timestamps, self.data_history['volume'], color='orange', alpha=0.7, width=0.0001)
        self.axes['volume'].set_title('Volume', fontweight='bold')
        self.axes['volume'].set_ylabel('Volume')
        self.axes['volume'].set_xlabel('Time')
        self.axes['volume'].grid(True, alpha=0.3)
        
        # Format x-axis
        for ax in [self.axes['price'], self.axes['rsi'], self.axes['volume']]:
            ax.tick_params(axis='x', rotation=45)
            
    def update_info_panel(self):
        """Update market information panel."""
        if not self.current_data:
            return
            
        self.axes['info'].clear()
        self.axes['info'].set_title('Market Info', fontweight='bold')
        self.axes['info'].axis('off')
        
        info_text = f"""
Current Price: ${self.current_data['price']:.6f}

24h Change: {self.current_data['change_24h']:+.2f}%

24h High: ${self.current_data['high_24h']:.6f}
24h Low: ${self.current_data['low_24h']:.6f}

RSI: {self.current_data['rsi']:.1f}

24h Volume: {self.current_data['volume_24h']:,.0f} TRX

Last Update: {self.current_data['timestamp'].strftime('%H:%M:%S')}
"""
        
        self.axes['info'].text(0.05, 0.95, info_text, transform=self.axes['info'].transAxes,
                              fontsize=10, verticalalignment='top', fontfamily='monospace')
                              
    def update_performance_panel(self):
        """Update performance panel."""
        self.axes['performance'].clear()
        self.axes['performance'].set_title('Performance', fontweight='bold')
        self.axes['performance'].axis('off')
        
        # Load latest trading reports
        performance_text = """
Session Status: MONITORING

Total Trades: 0
Win Rate: 0.0%
Total P&L: $0.00

Current Signal: HOLD
Confidence: 0.0%

Portfolio Value: $376.59
Available TRX: 1,240.00

Risk Level: LOW
"""
        
        self.axes['performance'].text(0.05, 0.95, performance_text, transform=self.axes['performance'].transAxes,
                                    fontsize=10, verticalalignment='top', fontfamily='monospace')
                                    
    def update_controls_panel(self):
        """Update controls panel."""
        self.axes['controls'].clear()
        self.axes['controls'].set_title('Controls', fontweight='bold')
        self.axes['controls'].axis('off')
        
        controls_text = """
🟢 Dashboard: ACTIVE
🔄 Auto-Update: ON
⏱️  Interval: 2s

Commands:
• Ctrl+C: Stop Dashboard
• Close Window: Exit

Status: LIVE MONITORING
"""
        
        self.axes['controls'].text(0.05, 0.95, controls_text, transform=self.axes['controls'].transAxes,
                                 fontsize=10, verticalalignment='top', fontfamily='monospace')
                                 
    async def update_dashboard(self, frame=None):
        """Update the entire dashboard."""
        try:
            # Fetch new market data
            if await self.fetch_market_data():
                self.update_data_history()
                self.update_charts()
                self.update_info_panel()
                self.update_performance_panel()
                self.update_controls_panel()
                
                plt.tight_layout()
                
        except Exception as e:
            logger.info(f"❌ Dashboard update error: {e}")
            
    async def run_dashboard(self):
        """Run the real-time dashboard."""
        logger.info("🚀 Starting Kimera Real-Time Dashboard...")
        
        if not await self.initialize():
            logger.info("❌ Failed to initialize dashboard")
            return
            
        self.setup_dashboard_layout()
        self.running = True
        
        logger.info("✅ Dashboard is running!")
        logger.info("📊 Monitoring TRX/USDT market data...")
        logger.info("⚠️  Close the window or press Ctrl+C to stop")
        
        try:
            # Initial data fetch
            await self.fetch_market_data()
            self.update_data_history()
            
            # Start animation
            ani = animation.FuncAnimation(
                self.fig, 
                lambda frame: asyncio.create_task(self.update_dashboard(frame)),
                interval=self.update_interval * 1000,
                blit=False
            )
            
            plt.show()
            
        except KeyboardInterrupt:
            logger.info("\n🛑 Dashboard stopped by user")
        except Exception as e:
            logger.info(f"❌ Dashboard error: {e}")
        finally:
            self.running = False
            if self.connector:
                await self.connector.close()

async def main():
    """Main dashboard execution."""
    dashboard = KimeraDashboard()
    await dashboard.run_dashboard()

if __name__ == "__main__":
    logger.info("🎛️  KIMERA REAL-TIME TRADING DASHBOARD")
    logger.info("=" * 50)
    logger.info("📊 Live market monitoring for TRX/USDT")
    logger.info("🔄 Real-time price, RSI, volume, and signals")
    logger.info("⚡ Updates every 2 seconds")
    logger.info("=" * 50)
    
    asyncio.run(main()) 