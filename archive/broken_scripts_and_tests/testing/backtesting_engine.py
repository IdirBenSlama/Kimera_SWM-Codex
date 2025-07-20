#!/usr/bin/env python3
"""
📊 KIMERA HISTORICAL BACKTESTING ENGINE 📊

This script provides a professional-grade backtesting environment to simulate
trading strategies against historical market data. It processes data sequentially,
ensuring strategies cannot "see the future," providing a realistic assessment
of performance.

Core Components:
- DataLoader: Loads historical data (CSV) or generates realistic synthetic data.
- Strategy (ABC): A template for creating new trading strategies.
- BacktestingEngine: The orchestrator that runs the simulation and reports results.
- Example Strategy: A simple SMA Crossover to demonstrate functionality.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from datetime import datetime
import os

# Initialize structured logger
from backend.utils.kimera_logger import get_system_logger
logger = get_system_logger(__name__)


# --- 1. Data Loader & Generator ---
class DataLoader:
    """Handles loading historical data or generating synthetic data for testing."""

    def __init__(self, file_path: str = None, symbol: str = "SYNTHETIC_BTC"):
        self.file_path = file_path
        self.symbol = symbol

    def load_data(self) -> pd.DataFrame:
        """Loads data from a CSV file or generates it if no file is provided."""
        if self.file_path and os.path.exists(self.file_path):
            logger.info(f"Loading data from {self.file_path}...")
            df = pd.read_csv(
                self.file_path,
                index_col='timestamp',
                parse_dates=True
            )
            df.sort_index(inplace=True)
            return df
        else:
            logger.info("No data file found. Generating realistic synthetic data...")
            return self._generate_synthetic_data()

    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Generates a DataFrame with realistic-looking OHLCV data."""
        dates = pd.date_range(start="2022-01-01", periods=1500, freq='H')
        price = 20000
        prices = []
        
        # Geometric Brownian Motion for price simulation
        drift = 0.0001
        volatility = 0.02
        for _ in range(len(dates)):
            price += price * (drift + volatility * np.random.randn())
            prices.append(price)

        df = pd.DataFrame(index=dates, data={'close': prices})
        df['open'] = df['close'].shift(1) * (1 + np.random.uniform(-0.005, 0.005, len(df)))
        df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0, 0.005, len(df)))
        df['low'] = df[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0, 0.005, len(df)))
        df['volume'] = np.random.randint(100, 5000, len(df))
        df.dropna(inplace=True)
        return df

# --- 2. Strategy Interface (Abstract Base Class) ---
class Strategy(ABC):
    """Abstract base class for all trading strategies."""
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.signals = self._initialize_signals()

    def _initialize_signals(self) -> pd.DataFrame:
        """Creates a DataFrame to store buy/sell signals."""
        signals = pd.DataFrame(index=self.data.index)
        signals['signal'] = 0.0 # 1.0 for buy, -1.0 for sell
        return signals
    
    @abstractmethod
    def generate_signals(self):
        """The core logic of the strategy to generate buy/sell signals."""
        raise NotImplementedError("Should implement generate_signals()!")

# --- 3. Example Strategy: SMA Crossover ---
class SmaCrossStrategy(Strategy):
    """A simple strategy based on the crossover of two moving averages."""
    def __init__(self, data: pd.DataFrame, short_window: int = 40, long_window: int = 100):
        self.short_window = short_window
        self.long_window = long_window
        super().__init__(data)

    def generate_signals(self):
        """Generate signals when the short SMA crosses the long SMA."""
        logger.info(f"Generating signals for SMA Crossover ({self.short_window}/{self.long_window})
        # Calculate SMAs
        self.signals['short_mavg'] = self.data['close'].rolling(window=self.short_window, min_periods=1).mean()
        self.signals['long_mavg'] = self.data['close'].rolling(window=self.long_window, min_periods=1).mean()
        
        # Generate signal when short SMA crosses above long SMA (buy)
        self.signals['signal'][self.short_window:] = np.where(
            self.signals['short_mavg'][self.short_window:] > self.signals['long_mavg'][self.short_window:],
            1.0, 0.0
        )
        
        # Take the difference to find the exact crossover point
        self.signals['positions'] = self.signals['signal'].diff()
        
        logger.info("Signals generated.")
        return self.signals

# --- 4. The Backtesting Engine ---
class BacktestingEngine:
    """Orchestrates the backtest from start to finish."""

    def __init__(self, data: pd.DataFrame, strategy: Strategy, initial_capital: float = 100000.0):
        self.data = data
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.portfolio = self._setup_portfolio()

    def _setup_portfolio(self) -> pd.DataFrame:
        """Initializes the portfolio DataFrame to track value over time."""
        portfolio = pd.DataFrame(index=self.data.index)
        portfolio['holdings'] = 0.0  # Total value of asset held
        portfolio['cash'] = self.initial_capital
        portfolio['total'] = self.initial_capital
        portfolio['returns'] = 0.0
        return portfolio

    def run_backtest(self):
        """Executes the core backtesting loop."""
        logger.info("Running backtest...")
        signals = self.strategy.generate_signals()
        
        position = 0.0 # Number of shares/units held

        for i, (timestamp, row) in enumerate(signals.iterrows()):
            if i == 0: continue # Skip first row

            # Execute buy signal
            if row['positions'] == 1.0 and self.portfolio.iloc[i-1]['cash'] > 0:
                # Simple position sizing: use all cash
                investment = self.portfolio.iloc[i-1]['cash']
                position = investment / self.data['close'].iloc[i]
                self.portfolio.loc[timestamp, 'cash'] = 0.0
            
            # Execute sell signal
            elif row['positions'] == -1.0 and position > 0:
                self.portfolio.loc[timestamp, 'cash'] = position * self.data['close'].iloc[i]
                position = 0.0
            
            # If no trade, carry over cash from previous day
            else:
                 self.portfolio.loc[timestamp, 'cash'] = self.portfolio.iloc[i-1]['cash']

            # Update total portfolio value
            self.portfolio.loc[timestamp, 'holdings'] = position * self.data['close'].iloc[i]
            self.portfolio.loc[timestamp, 'total'] = self.portfolio.loc[timestamp, 'cash'] + self.portfolio.loc[timestamp, 'holdings']
            
            # Calculate daily returns
            prev_total = self.portfolio.iloc[i-1]['total']
            if prev_total != 0:
                self.portfolio.loc[timestamp, 'returns'] = (self.portfolio.loc[timestamp, 'total'] / prev_total) - 1

        logger.info("Backtest complete.")
        return self.portfolio

    def display_results(self, portfolio: pd.DataFrame):
        """Prints a performance report and plots the equity curve."""
        logger.info("\n" + "---" * 20)
        logger.info("📊 BACKTESTING PERFORMANCE REPORT 📊")
        logger.info("---" * 20)

        # Performance Metrics
        total_return = (portfolio['total'].iloc[-1] / self.initial_capital - 1) * 100
        sharpe_ratio = np.sqrt(252) * (portfolio['returns'].mean() / portfolio['returns'].std()) if portfolio['returns'].std() != 0 else 0
        
        logger.info(f"Total Return: {total_return:.2f}%")
        logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")

        # Plotting
        fig, ax = plt.subplots(figsize=(16, 9))
        ax.plot(portfolio['total'], label='Strategy Equity Curve', color='cyan')
        ax.set_title(f'Strategy Performance | Final Value: ${portfolio["total"].iloc[-1]:,.2f}', fontsize=16)
        ax.set_ylabel('Portfolio Value ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plot_filename = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_filename)
        logger.info(f"\n📈 Results plot saved to: {plot_filename}")

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load Data (uses synthetic data if no file is present)
    data_loader = DataLoader()
    market_data = data_loader.load_data()

    # 2. Initialize Strategy
    sma_strategy = SmaCrossStrategy(market_data, short_window=50, long_window=200)

    # 3. Setup and Run Engine
    engine = BacktestingEngine(market_data, sma_strategy)
    results_portfolio = engine.run_backtest()

    # 4. Display Results
    engine.display_results(results_portfolio) 