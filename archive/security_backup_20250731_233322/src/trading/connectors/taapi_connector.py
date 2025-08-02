"""
TAAPI.io API Connector for KIMERA Trading Module
================================================

Integrates TAAPI.io's technical analysis API for real-time
indicator calculations to enhance contradiction detection.

API Documentation: https://taapi.io/documentation
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import time

logger = logging.getLogger(__name__)


class Indicator(Enum):
    """Available technical indicators from TAAPI"""
    # Trend Indicators
    SMA = "sma"  # Simple Moving Average
    EMA = "ema"  # Exponential Moving Average
    WMA = "wma"  # Weighted Moving Average
    DEMA = "dema"  # Double Exponential Moving Average
    TEMA = "tema"  # Triple Exponential Moving Average
    
    # Momentum Indicators
    RSI = "rsi"  # Relative Strength Index
    MACD = "macd"  # Moving Average Convergence Divergence
    STOCH = "stoch"  # Stochastic Oscillator
    STOCHRSI = "stochrsi"  # Stochastic RSI
    MOM = "mom"  # Momentum
    ROC = "roc"  # Rate of Change
    
    # Volatility Indicators
    BBands = "bbands"  # Bollinger Bands
    ATR = "atr"  # Average True Range
    NATR = "natr"  # Normalized ATR
    
    # Volume Indicators
    OBV = "obv"  # On Balance Volume
    AD = "ad"  # Accumulation/Distribution
    ADOSC = "adosc"  # Chaikin A/D Oscillator
    
    # Other Indicators
    ADX = "adx"  # Average Directional Index
    AROON = "aroon"  # Aroon
    CCI = "cci"  # Commodity Channel Index
    DX = "dx"  # Directional Movement Index
    MFI = "mfi"  # Money Flow Index
    PSAR = "psar"  # Parabolic SAR


class Timeframe(Enum):
    """Available timeframes"""
    ONE_MIN = "1m"
    FIVE_MIN = "5m"
    FIFTEEN_MIN = "15m"
    THIRTY_MIN = "30m"
    ONE_HOUR = "1h"
    TWO_HOUR = "2h"
    FOUR_HOUR = "4h"
    ONE_DAY = "1d"
    ONE_WEEK = "1w"
    ONE_MONTH = "1M"


@dataclass
class TechnicalAnalysis:
    """Structured technical analysis result"""
    symbol: str
    indicator: str
    timeframe: str
    value: Union[float, Dict[str, float]]
    timestamp: datetime
    metadata: Dict[str, Any]
    
    @property
    def is_bullish(self) -> Optional[bool]:
        """Determine if indicator suggests bullish sentiment"""
        if self.indicator == "rsi":
            if isinstance(self.value, (int, float)):
                return self.value < 30  # Oversold = potential bullish
        elif self.indicator == "macd":
            if isinstance(self.value, dict):
                return self.value.get('macd', 0) > self.value.get('signal', 0)
        return None
    
    @property
    def is_bearish(self) -> Optional[bool]:
        """Determine if indicator suggests bearish sentiment"""
        if self.indicator == "rsi":
            if isinstance(self.value, (int, float)):
                return self.value > 70  # Overbought = potential bearish
        elif self.indicator == "macd":
            if isinstance(self.value, dict):
                return self.value.get('macd', 0) < self.value.get('signal', 0)
        return None


class TAAPIConnector:
    """
    Connector for TAAPI.io Technical Analysis API
    
    Features:
    - Real-time technical indicator calculations
    - Multiple timeframe analysis
    - Bulk indicator requests
    - Contradiction detection between indicators
    """
    
    BASE_URL = "https://api.taapi.io"
    
    def __init__(self, api_key: str):
        """
        Initialize TAAPI connector
        
        Args:
            api_key: Your TAAPI.io API key
        """
        self.api_key = api_key
        self.session: Optional[aiohttp.ClientSession] = None
        self.last_request_time = 0
        self.min_request_interval = 3.0  # Minimum 3 seconds between requests for free tier
        
        logger.info(f"ðŸ“Š TAAPI connector initialized")
        logger.info(f"   API Endpoint: {self.BASE_URL}")
        logger.info(f"   Rate limit: {self.min_request_interval}s between requests")
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make API request with error handling and rate limiting"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            wait_time = self.min_request_interval - time_since_last
            await asyncio.sleep(wait_time)
        
        self.last_request_time = time.time()
        
        # Add API key to params
        params['secret'] = self.api_key
        
        url = f"{self.BASE_URL}{endpoint}"
        
        try:
            async with self.session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                
                if 'error' in data:
                    raise Exception(f"API Error: {data['error']}")
                
                return data
                
        except aiohttp.ClientError as e:
            logger.error(f"API request failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise
    
    async def get_indicator(
        self,
        indicator: Indicator,
        symbol: str,
        exchange: str = "binance",
        timeframe: Timeframe = Timeframe.ONE_HOUR,
        **kwargs
    ) -> TechnicalAnalysis:
        """
        Get a single technical indicator
        
        Args:
            indicator: Technical indicator to calculate
            symbol: Trading pair (e.g., "BTC/USDT")
            exchange: Exchange name
            timeframe: Timeframe for calculation
            **kwargs: Additional indicator parameters
            
        Returns:
            TechnicalAnalysis object
        """
        params = {
            'exchange': exchange,
            'symbol': symbol,
            'interval': timeframe.value
        }
        
        # Add any additional parameters
        params.update(kwargs)
        
        try:
            data = await self._make_request(f"/{indicator.value}", params)
            
            # Parse the response
            value = data.get('value')
            if value is None:
                # Some indicators return multiple values
                value = {k: v for k, v in data.items() if k not in ['request_id', 'timestamp']}
            
            return TechnicalAnalysis(
                symbol=symbol,
                indicator=indicator.value,
                timeframe=timeframe.value,
                value=value,
                timestamp=datetime.now(),
                metadata={
                    'exchange': exchange,
                    'raw_response': data
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to get {indicator.value} for {symbol}: {e}")
            raise
    
    async def get_bulk_indicators(
        self,
        indicators: List[Indicator],
        symbol: str,
        exchange: str = "binance",
        timeframe: Timeframe = Timeframe.ONE_HOUR
    ) -> Dict[str, TechnicalAnalysis]:
        """
        Get multiple indicators in parallel
        
        Args:
            indicators: List of indicators to calculate
            symbol: Trading pair
            exchange: Exchange name
            timeframe: Timeframe for calculations
            
        Returns:
            Dictionary mapping indicator names to TechnicalAnalysis objects
        """
        tasks = []
        for indicator in indicators:
            task = self.get_indicator(indicator, symbol, exchange, timeframe)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        output = {}
        for indicator, result in zip(indicators, results):
            if isinstance(result, Exception):
                logger.warning(f"Failed to get {indicator.value}: {result}")
            else:
                output[indicator.value] = result
        
        return output
    
    async def analyze_trend(
        self,
        symbol: str,
        exchange: str = "binance",
        timeframe: Timeframe = Timeframe.ONE_HOUR
    ) -> Dict[str, Any]:
        """
        Comprehensive trend analysis using multiple indicators
        
        Args:
            symbol: Trading pair
            exchange: Exchange name
            timeframe: Timeframe for analysis
            
        Returns:
            Trend analysis results
        """
        # Get key trend indicators
        trend_indicators = [
            Indicator.SMA,
            Indicator.EMA,
            Indicator.MACD,
            Indicator.ADX,
            Indicator.AROON
        ]
        
        results = await self.get_bulk_indicators(
            trend_indicators,
            symbol,
            exchange,
            timeframe
        )
        
        # Analyze trend strength
        bullish_signals = 0
        bearish_signals = 0
        
        # MACD analysis
        if 'macd' in results:
            macd_data = results['macd']
            if isinstance(macd_data.value, dict):
                if macd_data.value.get('macd', 0) > macd_data.value.get('signal', 0):
                    bullish_signals += 1
                else:
                    bearish_signals += 1
        
        # ADX for trend strength
        trend_strength = 0
        if 'adx' in results and isinstance(results['adx'].value, (int, float)):
            trend_strength = results['adx'].value / 100  # Normalize to 0-1
        
        # Aroon for trend direction
        if 'aroon' in results and isinstance(results['aroon'].value, dict):
            aroon_up = results['aroon'].value.get('aroonUp', 50)
            aroon_down = results['aroon'].value.get('aroonDown', 50)
            if aroon_up > aroon_down:
                bullish_signals += 1
            else:
                bearish_signals += 1
        
        # Determine overall trend
        trend_score = (bullish_signals - bearish_signals) / max(bullish_signals + bearish_signals, 1)
        
        return {
            'symbol': symbol,
            'timeframe': timeframe.value,
            'trend_direction': 'bullish' if trend_score > 0 else 'bearish',
            'trend_score': trend_score,
            'trend_strength': trend_strength,
            'bullish_signals': bullish_signals,
            'bearish_signals': bearish_signals,
            'indicators': {k: v.value for k, v in results.items()},
            'timestamp': datetime.now().isoformat()
        }
    
    async def detect_indicator_contradictions(
        self,
        symbol: str,
        exchange: str = "binance",
        timeframes: List[Timeframe] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect contradictions between indicators or across timeframes
        
        Args:
            symbol: Trading pair
            exchange: Exchange name
            timeframes: List of timeframes to analyze
            
        Returns:
            List of detected contradictions
        """
        if timeframes is None:
            timeframes = [Timeframe.FIFTEEN_MIN, Timeframe.ONE_HOUR, Timeframe.FOUR_HOUR]
        
        contradictions = []
        
        # Get momentum indicators across timeframes
        momentum_indicators = [Indicator.RSI, Indicator.MACD, Indicator.STOCH]
        
        timeframe_results = {}
        for tf in timeframes:
            results = await self.get_bulk_indicators(
                momentum_indicators,
                symbol,
                exchange,
                tf
            )
            timeframe_results[tf.value] = results
        
        # Check for timeframe contradictions
        for i, tf1 in enumerate(timeframes):
            for tf2 in timeframes[i+1:]:
                results1 = timeframe_results[tf1.value]
                results2 = timeframe_results[tf2.value]
                
                # Check RSI contradiction
                if 'rsi' in results1 and 'rsi' in results2:
                    rsi1 = results1['rsi'].value
                    rsi2 = results2['rsi'].value
                    
                    if isinstance(rsi1, (int, float)) and isinstance(rsi2, (int, float)):
                        # One oversold, other overbought
                        if (rsi1 < 30 and rsi2 > 70) or (rsi1 > 70 and rsi2 < 30):
                            contradictions.append({
                                'type': 'rsi_timeframe_contradiction',
                                'symbol': symbol,
                                'indicator': 'rsi',
                                'timeframe_1': tf1.value,
                                'value_1': rsi1,
                                'timeframe_2': tf2.value,
                                'value_2': rsi2,
                                'severity': abs(rsi1 - rsi2) / 100,
                                'description': f"RSI shows {('oversold' if rsi1 < 30 else 'overbought')} on {tf1.value} but {('oversold' if rsi2 < 30 else 'overbought')} on {tf2.value}"
                            })
        
        # Check for indicator contradictions within same timeframe
        for tf, results in timeframe_results.items():
            # RSI vs MACD contradiction
            if 'rsi' in results and 'macd' in results:
                rsi_bullish = results['rsi'].is_bullish
                macd_bullish = results['macd'].is_bullish
                
                if rsi_bullish is not None and macd_bullish is not None:
                    if rsi_bullish != macd_bullish:
                        contradictions.append({
                            'type': 'indicator_disagreement',
                            'symbol': symbol,
                            'timeframe': tf,
                            'indicators': ['rsi', 'macd'],
                            'rsi_signal': 'bullish' if rsi_bullish else 'bearish',
                            'macd_signal': 'bullish' if macd_bullish else 'bearish',
                            'severity': 0.6,
                            'description': f"RSI and MACD show conflicting signals on {tf}"
                        })
        
        return contradictions
    
    async def get_market_overview(
        self,
        symbols: List[str],
        exchange: str = "binance",
        timeframe: Timeframe = Timeframe.ONE_HOUR
    ) -> Dict[str, Any]:
        """
        Get market overview for multiple symbols
        
        Args:
            symbols: List of trading pairs
            exchange: Exchange name
            timeframe: Timeframe for analysis
            
        Returns:
            Market overview with trend analysis
        """
        overview = {
            'timestamp': datetime.now().isoformat(),
            'timeframe': timeframe.value,
            'symbols': {}
        }
        
        for symbol in symbols:
            try:
                # Get key indicators
                indicators = await self.get_bulk_indicators(
                    [Indicator.RSI, Indicator.MACD, Indicator.BBands, Indicator.ADX],
                    symbol,
                    exchange,
                    timeframe
                )
                
                # Simple trend determination
                trend = 'neutral'
                if 'rsi' in indicators and isinstance(indicators['rsi'].value, (int, float)):
                    rsi_value = indicators['rsi'].value
                    if rsi_value < 40:
                        trend = 'bullish'
                    elif rsi_value > 60:
                        trend = 'bearish'
                
                overview['symbols'][symbol] = {
                    'trend': trend,
                    'rsi': indicators.get('rsi', {}).value if 'rsi' in indicators else None,
                    'indicators': {k: v.value for k, v in indicators.items()}
                }
                
            except Exception as e:
                logger.error(f"Failed to analyze {symbol}: {e}")
                overview['symbols'][symbol] = {'error': str(e)}
        
        return overview


# Factory function
def create_taapi_connector(api_key: str) -> TAAPIConnector:
    """
    Create TAAPI connector instance
    
    Args:
        api_key: Your TAAPI.io API key
        
    Returns:
        TAAPIConnector instance
    """
    return TAAPIConnector(api_key) 