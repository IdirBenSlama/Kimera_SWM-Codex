"""
Advanced Financial Data Processor for KIMERA Trading
Implements trajectory analysis, technical indicators, and movement data processing
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
import yfinance as yf
import ccxt
from finta import TA
import pandas_ta as ta
from stockstats import StockDataFrame
import warnings
from src.utils.kimera_logger import get_logger, LogCategory
from src.core.exception_handling import safe_operation

warnings.filterwarnings('ignore')

logger = get_logger(__name__, category=LogCategory.TRADING)

@dataclass
class FinancialTrajectory:
    """Financial trajectory analysis result"""
    symbol: str
    start_time: datetime
    end_time: datetime
    trajectory_type: str  # 'uptrend', 'downtrend', 'sideways', 'volatile'
    strength: float  # 0-1 scale
    key_points: List[Dict[str, Any]]
    indicators: Dict[str, Any]
    risk_metrics: Dict[str, float]

@dataclass
class TechnicalSignal:
    """Technical analysis signal"""
    timestamp: datetime
    symbol: str
    indicator: str
    signal_type: str  # 'buy', 'sell', 'hold'
    strength: float
    price: float
    confidence: float
    metadata: Dict[str, Any]

class AdvancedFinancialProcessor:
    """
    Advanced financial data processor combining:
    - yfinance for historical data
    - Multiple technical analysis libraries
    - Trajectory analysis for price movements
    - Advanced statistical analysis
    - Multi-timeframe analysis
    """
    
    def __init__(self):
        self.exchanges = {}
        self.data_cache = {}
        self.indicator_cache = {}
        
        # Initialize crypto exchanges
        self._initialize_exchanges()
        
        # Technical analysis parameters
        self.ta_params = {
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bb_period': 20,
            'bb_std': 2,
            'sma_periods': [10, 20, 50, 200],
            'ema_periods': [12, 26, 50],
            'stoch_k': 14,
            'stoch_d': 3,
            'adx_period': 14,
            'cci_period': 20,
            'williams_r_period': 14
        }
        
    def _initialize_exchanges(self):
        """Initialize cryptocurrency exchanges"""
        try:
            # Initialize major exchanges
            self.exchanges['binance'] = ccxt.binance({'enableRateLimit': True})
            self.exchanges['coinbase'] = ccxt.coinbasepro({'enableRateLimit': True})
            logger.info("Initialized crypto exchanges successfully")
        except Exception as e:
            logger.warning(f"Could not initialize some exchanges: {e}")
    
    @safe_operation("get_comprehensive_data", fallback=pd.DataFrame())
    async def get_comprehensive_data(self, symbol: str, 
                                   period: str = '1y',
                                   interval: str = '1d',
                                   data_source: str = 'yahoo') -> pd.DataFrame:
        """
        Get comprehensive financial data from multiple sources
        
        Args:
            symbol: Trading symbol (e.g., 'AAPL', 'BTC/USDT')
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            data_source: 'yahoo' for stocks, 'binance' for crypto
            
        Returns:
            Comprehensive DataFrame with OHLCV and additional data
        """
        if data_source == 'yahoo':
            return await self._get_yahoo_data(symbol, period, interval)
        elif data_source == 'binance' and 'binance' in self.exchanges:
            return await self._get_binance_data(symbol, period, interval)
        else:
            logger.error(f"Unsupported data source: {data_source}")
            return pd.DataFrame()
    
    @safe_operation("get_yahoo_data", fallback=pd.DataFrame())
    async def _get_yahoo_data(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        """Get data from Yahoo Finance"""
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)
        
        if data.empty:
            logger.warning(f"No data found for {symbol}")
            return pd.DataFrame()
        
        # Standardize column names
        data.columns = [col.lower() for col in data.columns]
        data.reset_index(inplace=True)
        
        # Add symbol column
        data['symbol'] = symbol
        
        # Add timestamp if not present
        if 'date' in data.columns:
            data['timestamp'] = pd.to_datetime(data['date'])
        elif 'datetime' in data.columns:
            data['timestamp'] = pd.to_datetime(data['datetime'])
        else:
            data['timestamp'] = data.index
        
        return data
    
    @safe_operation("get_binance_data", fallback=pd.DataFrame())
    async def _get_binance_data(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        """Get data from Binance exchange"""
        exchange = self.exchanges['binance']
        
        # Convert period to milliseconds
        timeframe_map = {
            '1m': 60000, '5m': 300000, '15m': 900000, '30m': 1800000,
            '1h': 3600000, '4h': 14400000, '1d': 86400000, '1w': 604800000
        }
        
        if interval not in timeframe_map:
            interval = '1d'  # Default to daily
        
        # Get OHLCV data
        ohlcv = await exchange.fetch_ohlcv(symbol, interval, limit=1000)
        
        if not ohlcv:
            logger.warning(f"No data found for {symbol}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        data['symbol'] = symbol
        
        return data
    
    @safe_operation("calculate_comprehensive_indicators", fallback=None)
    async def calculate_comprehensive_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive technical indicators using multiple libraries
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            DataFrame with all technical indicators
        """
        if data.empty or len(data) < 50:
            logger.warning("Insufficient data for indicator calculation")
            return data
        
        result = data.copy()
        
        # Use different libraries for different indicators
        result = await self._add_finta_indicators(result)
        result = await self._add_pandas_ta_indicators(result)
        result = await self._add_stockstats_indicators(result)
        result = await self._add_custom_indicators(result)
        
        logger.info(f"Calculated {len(result.columns) - len(data.columns)} technical indicators")
        return result
    
    @safe_operation("add_finta_indicators", reraise=True)
    async def _add_finta_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add indicators from FinTA library"""
        ohlc = data[['open', 'high', 'low', 'close']].copy()
        
        # Trend indicators
        data['sma_10'] = TA.SMA(ohlc, 10)
        data['sma_20'] = TA.SMA(ohlc, 20)
        data['sma_50'] = TA.SMA(ohlc, 50)
        data['ema_12'] = TA.EMA(ohlc, 12)
        data['ema_26'] = TA.EMA(ohlc, 26)
        
        # Momentum indicators
        data['rsi'] = TA.RSI(ohlc, self.ta_params['rsi_period'])
        data['macd'] = TA.MACD(ohlc)['MACD']
        data['macd_signal'] = TA.MACD(ohlc)['SIGNAL']
        data['macd_histogram'] = TA.MACD(ohlc)['HISTOGRAM']
        
        # Volatility indicators
        bb = TA.BBANDS(ohlc, self.ta_params['bb_period'])
        data['bb_upper'] = bb['BB_UPPER']
        data['bb_middle'] = bb['BB_MIDDLE']
        data['bb_lower'] = bb['BB_LOWER']
        
        # Volume indicators
        if 'volume' in data.columns:
            data['obv'] = TA.OBV(ohlc, data['volume'])
            data['ad'] = TA.AD(ohlc, data['volume'])
        
        return data
    
    @safe_operation("add_pandas_ta_indicators", reraise=True)
    async def _add_pandas_ta_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add indicators from pandas-ta library"""
        data.ta.cores = 0  # Use all available cores
        
        # Add various indicator categories. They are appended in place.
        data.ta.cdl_pattern(name="all")
        data.ta.momentum(append=True)
        data.ta.overlap(append=True)
        data.ta.volatility(append=True)
        data.ta.volume(append=True)
        data.ta.trend(append=True)
        
        return data
    
    @safe_operation("add_stockstats_indicators", reraise=True)
    async def _add_stockstats_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add indicators from stockstats library"""
        stock = StockDataFrame.retype(data)
        
        # Add various indicators
        data['kdjk'] = stock['kdjk']
        data['kdjd'] = stock['kdjd']
        data['kdjj'] = stock['kdjj']
        data['boll_ub'] = stock['boll_ub']
        data['boll_lb'] = stock['boll_lb']
        data['cr'] = stock['cr']
        data['wr_10'] = stock['wr_10']
        
        return data
    
    @safe_operation("add_custom_indicators", reraise=True)
    async def _add_custom_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add custom financial indicators"""
        # Price-based indicators
        data['price_change'] = data['close'].pct_change()
        data['price_change_5'] = data['close'].pct_change(5)
        data['price_volatility'] = data['price_change'].rolling(20).std()
        
        # Volume-based indicators
        if 'volume' in data.columns:
            data['volume_change'] = data['volume'].pct_change()
            data['volume_sma'] = data['volume'].rolling(20).mean()
            data['volume_ratio'] = data['volume'] / data['volume_sma']
            data['price_volume'] = data['close'] * data['volume']
        
        # Support and resistance levels
        data['resistance'] = data['high'].rolling(20).max()
        data['support'] = data['low'].rolling(20).min()
        
        # Trend strength
        data['trend_strength'] = abs(data['close'].rolling(10).apply(
            lambda x: (x[-1] - x[0]) / x[0] if x[0] != 0 else 0
        ))
        
        # Market regime indicators
        data['market_regime'] = self._calculate_market_regime(data)
        
        return data
    
    def _calculate_market_regime(self, data: pd.DataFrame) -> pd.Series:
        """Calculate market regime (trending vs ranging)"""
        try:
            # Use ADX-like calculation
            high_low = data['high'] - data['low']
            high_close = abs(data['high'] - data['close'].shift())
            low_close = abs(data['low'] - data['close'].shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(14).mean()
            
            # Simplified regime classification
            regime = pd.Series(index=data.index, dtype=str)
            regime[atr > atr.rolling(50).mean()] = 'trending'
            regime[atr <= atr.rolling(50).mean()] = 'ranging'
            
            return regime
            
        except Exception as e:
            logger.error(f"Error calculating market regime: {e}")
            return pd.Series(index=data.index, dtype=str)
    
    @safe_operation("analyze_price_trajectory", fallback=[])
    async def analyze_price_trajectory(self, data: pd.DataFrame, 
                                     window: int = 50) -> List[FinancialTrajectory]:
        """
        Analyze price trajectories using movement analysis concepts
        
        Args:
            data: OHLCV DataFrame with indicators
            window: Analysis window size
            
        Returns:
            List of trajectory analysis results
        """
        if data.empty or len(data) < window:
            logger.warning("Insufficient data for trajectory analysis")
            return []
        
        trajectories = []
        
        # Sliding window analysis
        for i in range(window, len(data), window // 2):
            window_data = data.iloc[i-window:i].copy()
            
            trajectory = await self._analyze_trajectory_segment(window_data)
            if trajectory:
                trajectories.append(trajectory)
        
        logger.info(f"Analyzed {len(trajectories)} price trajectories")
        return trajectories
    
    async def _analyze_trajectory_segment(self, segment_data: pd.DataFrame) -> Optional[FinancialTrajectory]:
        """Analyze a single trajectory segment"""
        try:
            if segment_data.empty:
                return None
            
            symbol = segment_data['symbol'].iloc[0] if 'symbol' in segment_data.columns else 'UNKNOWN'
            start_time = segment_data['timestamp'].iloc[0] if 'timestamp' in segment_data.columns else datetime.now()
            end_time = segment_data['timestamp'].iloc[-1] if 'timestamp' in segment_data.columns else datetime.now()
            
            # Calculate trajectory characteristics
            price_start = segment_data['close'].iloc[0]
            price_end = segment_data['close'].iloc[-1]
            price_change = (price_end - price_start) / price_start
            
            # Determine trajectory type
            if price_change > 0.05:
                trajectory_type = 'uptrend'
                strength = min(1.0, abs(price_change) * 5)
            elif price_change < -0.05:
                trajectory_type = 'downtrend'
                strength = min(1.0, abs(price_change) * 5)
            else:
                volatility = segment_data['close'].std() / segment_data['close'].mean()
                if volatility > 0.02:
                    trajectory_type = 'volatile'
                    strength = min(1.0, volatility * 20)
                else:
                    trajectory_type = 'sideways'
                    strength = 1.0 - min(1.0, volatility * 20)
            
            # Find key points (peaks and troughs)
            key_points = self._find_key_points(segment_data)
            
            # Calculate indicators summary
            indicators = {}
            if 'rsi' in segment_data.columns:
                indicators['avg_rsi'] = segment_data['rsi'].mean()
            if 'macd' in segment_data.columns:
                indicators['macd_trend'] = 'bullish' if segment_data['macd'].iloc[-1] > segment_data['macd'].iloc[0] else 'bearish'
            if 'volume' in segment_data.columns:
                indicators['avg_volume'] = segment_data['volume'].mean()
            
            # Calculate risk metrics
            risk_metrics = {
                'volatility': segment_data['close'].std() / segment_data['close'].mean(),
                'max_drawdown': self._calculate_max_drawdown(segment_data['close']),
                'sharpe_ratio': self._calculate_sharpe_ratio(segment_data['close'])
            }
            
            return FinancialTrajectory(
                symbol=symbol,
                start_time=start_time,
                end_time=end_time,
                trajectory_type=trajectory_type,
                strength=strength,
                key_points=key_points,
                indicators=indicators,
                risk_metrics=risk_metrics
            )
            
        except Exception as e:
            logger.error(f"Error analyzing trajectory segment: {e}")
            return None
    
    def _find_key_points(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Find key points (peaks and troughs) in price data"""
        try:
            key_points = []
            prices = data['close'].values
            
            # Simple peak/trough detection
            for i in range(1, len(prices) - 1):
                if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                    # Peak
                    key_points.append({
                        'type': 'peak',
                        'timestamp': data['timestamp'].iloc[i] if 'timestamp' in data.columns else None,
                        'price': prices[i],
                        'index': i
                    })
                elif prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                    # Trough
                    key_points.append({
                        'type': 'trough',
                        'timestamp': data['timestamp'].iloc[i] if 'timestamp' in data.columns else None,
                        'price': prices[i],
                        'index': i
                    })
            
            return key_points
            
        except Exception as e:
            logger.error(f"Error finding key points: {e}")
            return []
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate max drawdown"""
        try:
            peak = prices.cummax()
            drawdown = (prices - peak) / peak
            return drawdown.min()
        except Exception as e:
            logger.warning("Could not calculate max drawdown", error=e)
            return -1.0
    
    def _calculate_sharpe_ratio(self, prices: pd.Series) -> float:
        """Calculate Sharpe ratio (simplified)"""
        try:
            returns = prices.pct_change().dropna()
            if returns.std() == 0:
                return 0.0
            return np.sqrt(252) * (returns.mean() / returns.std())
        except Exception as e:
            logger.warning("Could not calculate Sharpe ratio", error=e)
            return 0.0
    
    @safe_operation("generate_trading_signals", fallback=[])
    async def generate_trading_signals(self, data: pd.DataFrame) -> List[TechnicalSignal]:
        """
        Generate comprehensive trading signals from technical indicators
        
        Args:
            data: DataFrame with OHLCV and technical indicators
            
        Returns:
            List of trading signals
        """
        if data.empty or len(data) < 20:
            logger.warning("Insufficient data for signal generation")
            return []
        
        signals = []
        
        # Generate signals from different indicators
        rsi_signals = await self._generate_rsi_signals(data)
        macd_signals = await self._generate_macd_signals(data)
        bb_signals = await self._generate_bollinger_signals(data)
        ma_signals = await self._generate_moving_average_signals(data)
        volume_signals = await self._generate_volume_signals(data)
        
        # Combine all signals
        all_signals = rsi_signals + macd_signals + bb_signals + ma_signals + volume_signals
        
        # Remove duplicates and sort by timestamp
        signals = sorted(all_signals, key=lambda x: x.timestamp)
        
        logger.info(f"Generated {len(signals)} trading signals")
        return signals
    
    async def _generate_rsi_signals(self, data: pd.DataFrame) -> List[TechnicalSignal]:
        """Generate RSI-based signals"""
        signals = []
        
        if 'rsi' not in data.columns:
            return signals
        
        try:
            for i in range(1, len(data)):
                rsi_current = data['rsi'].iloc[i]
                rsi_previous = data['rsi'].iloc[i-1]
                
                signal_type = None
                strength = 0.0
                confidence = 0.0
                
                # Oversold condition
                if rsi_previous >= 30 and rsi_current < 30:
                    signal_type = 'buy'
                    strength = (30 - rsi_current) / 30
                    confidence = 0.7
                
                # Overbought condition
                elif rsi_previous <= 70 and rsi_current > 70:
                    signal_type = 'sell'
                    strength = (rsi_current - 70) / 30
                    confidence = 0.7
                
                if signal_type:
                    signal = TechnicalSignal(
                        timestamp=data['timestamp'].iloc[i] if 'timestamp' in data.columns else datetime.now(),
                        symbol=data['symbol'].iloc[i] if 'symbol' in data.columns else 'UNKNOWN',
                        indicator='RSI',
                        signal_type=signal_type,
                        strength=strength,
                        price=data['close'].iloc[i],
                        confidence=confidence,
                        metadata={'rsi_value': rsi_current}
                    )
                    signals.append(signal)
        
        except Exception as e:
            logger.error(f"Error generating RSI signals: {e}")
        
        return signals
    
    async def _generate_macd_signals(self, data: pd.DataFrame) -> List[TechnicalSignal]:
        """Generate MACD-based signals"""
        signals = []
        
        if 'macd' not in data.columns or 'macd_signal' not in data.columns:
            return signals
        
        try:
            for i in range(1, len(data)):
                macd_current = data['macd'].iloc[i]
                macd_signal_current = data['macd_signal'].iloc[i]
                macd_previous = data['macd'].iloc[i-1]
                macd_signal_previous = data['macd_signal'].iloc[i-1]
                
                signal_type = None
                strength = 0.0
                confidence = 0.0
                
                # Bullish crossover
                if macd_previous <= macd_signal_previous and macd_current > macd_signal_current:
                    signal_type = 'buy'
                    strength = abs(macd_current - macd_signal_current) / abs(macd_current + 0.001)
                    confidence = 0.8
                
                # Bearish crossover
                elif macd_previous >= macd_signal_previous and macd_current < macd_signal_current:
                    signal_type = 'sell'
                    strength = abs(macd_current - macd_signal_current) / abs(macd_current + 0.001)
                    confidence = 0.8
                
                if signal_type:
                    signal = TechnicalSignal(
                        timestamp=data['timestamp'].iloc[i] if 'timestamp' in data.columns else datetime.now(),
                        symbol=data['symbol'].iloc[i] if 'symbol' in data.columns else 'UNKNOWN',
                        indicator='MACD',
                        signal_type=signal_type,
                        strength=min(1.0, strength),
                        price=data['close'].iloc[i],
                        confidence=confidence,
                        metadata={
                            'macd': macd_current,
                            'macd_signal': macd_signal_current
                        }
                    )
                    signals.append(signal)
        
        except Exception as e:
            logger.error(f"Error generating MACD signals: {e}")
        
        return signals
    
    async def _generate_bollinger_signals(self, data: pd.DataFrame) -> List[TechnicalSignal]:
        """Generate Bollinger Bands signals"""
        signals = []
        
        bb_cols = ['bb_upper', 'bb_lower', 'bb_middle']
        if not all(col in data.columns for col in bb_cols):
            return signals
        
        try:
            for i in range(len(data)):
                close_price = data['close'].iloc[i]
                bb_upper = data['bb_upper'].iloc[i]
                bb_lower = data['bb_lower'].iloc[i]
                bb_middle = data['bb_middle'].iloc[i]
                
                signal_type = None
                strength = 0.0
                confidence = 0.0
                
                # Price touches lower band (oversold)
                if close_price <= bb_lower:
                    signal_type = 'buy'
                    strength = (bb_lower - close_price) / (bb_upper - bb_lower)
                    confidence = 0.6
                
                # Price touches upper band (overbought)
                elif close_price >= bb_upper:
                    signal_type = 'sell'
                    strength = (close_price - bb_upper) / (bb_upper - bb_lower)
                    confidence = 0.6
                
                if signal_type:
                    signal = TechnicalSignal(
                        timestamp=data['timestamp'].iloc[i] if 'timestamp' in data.columns else datetime.now(),
                        symbol=data['symbol'].iloc[i] if 'symbol' in data.columns else 'UNKNOWN',
                        indicator='Bollinger_Bands',
                        signal_type=signal_type,
                        strength=min(1.0, strength),
                        price=close_price,
                        confidence=confidence,
                        metadata={
                            'bb_upper': bb_upper,
                            'bb_lower': bb_lower,
                            'bb_position': (close_price - bb_lower) / (bb_upper - bb_lower)
                        }
                    )
                    signals.append(signal)
        
        except Exception as e:
            logger.error(f"Error generating Bollinger signals: {e}")
        
        return signals
    
    async def _generate_moving_average_signals(self, data: pd.DataFrame) -> List[TechnicalSignal]:
        """Generate moving average crossover signals"""
        signals = []
        
        if 'sma_10' not in data.columns or 'sma_20' not in data.columns:
            return signals
        
        try:
            for i in range(1, len(data)):
                sma_fast_current = data['sma_10'].iloc[i]
                sma_slow_current = data['sma_20'].iloc[i]
                sma_fast_previous = data['sma_10'].iloc[i-1]
                sma_slow_previous = data['sma_20'].iloc[i-1]
                
                signal_type = None
                strength = 0.0
                confidence = 0.0
                
                # Golden cross (bullish)
                if sma_fast_previous <= sma_slow_previous and sma_fast_current > sma_slow_current:
                    signal_type = 'buy'
                    strength = abs(sma_fast_current - sma_slow_current) / sma_slow_current
                    confidence = 0.75
                
                # Death cross (bearish)
                elif sma_fast_previous >= sma_slow_previous and sma_fast_current < sma_slow_current:
                    signal_type = 'sell'
                    strength = abs(sma_fast_current - sma_slow_current) / sma_slow_current
                    confidence = 0.75
                
                if signal_type:
                    signal = TechnicalSignal(
                        timestamp=data['timestamp'].iloc[i] if 'timestamp' in data.columns else datetime.now(),
                        symbol=data['symbol'].iloc[i] if 'symbol' in data.columns else 'UNKNOWN',
                        indicator='MA_Crossover',
                        signal_type=signal_type,
                        strength=min(1.0, strength * 100),
                        price=data['close'].iloc[i],
                        confidence=confidence,
                        metadata={
                            'sma_fast': sma_fast_current,
                            'sma_slow': sma_slow_current
                        }
                    )
                    signals.append(signal)
        
        except Exception as e:
            logger.error(f"Error generating MA signals: {e}")
        
        return signals
    
    async def _generate_volume_signals(self, data: pd.DataFrame) -> List[TechnicalSignal]:
        """Generate volume-based signals"""
        signals = []
        
        if 'volume' not in data.columns or 'volume_ratio' not in data.columns:
            return signals
        
        try:
            for i in range(len(data)):
                volume_ratio = data['volume_ratio'].iloc[i]
                price_change = data['price_change'].iloc[i] if 'price_change' in data.columns else 0
                
                signal_type = None
                strength = 0.0
                confidence = 0.0
                
                # High volume with price increase
                if volume_ratio > 2.0 and price_change > 0.02:
                    signal_type = 'buy'
                    strength = min(1.0, volume_ratio / 5.0)
                    confidence = 0.5
                
                # High volume with price decrease
                elif volume_ratio > 2.0 and price_change < -0.02:
                    signal_type = 'sell'
                    strength = min(1.0, volume_ratio / 5.0)
                    confidence = 0.5
                
                if signal_type:
                    signal = TechnicalSignal(
                        timestamp=data['timestamp'].iloc[i] if 'timestamp' in data.columns else datetime.now(),
                        symbol=data['symbol'].iloc[i] if 'symbol' in data.columns else 'UNKNOWN',
                        indicator='Volume',
                        signal_type=signal_type,
                        strength=strength,
                        price=data['close'].iloc[i],
                        confidence=confidence,
                        metadata={
                            'volume_ratio': volume_ratio,
                            'price_change': price_change
                        }
                    )
                    signals.append(signal)
        
        except Exception as e:
            logger.error(f"Error generating volume signals: {e}")
        
        return signals

# Factory function
def create_financial_processor() -> AdvancedFinancialProcessor:
    """Create and return a configured financial processor"""
    return AdvancedFinancialProcessor() 