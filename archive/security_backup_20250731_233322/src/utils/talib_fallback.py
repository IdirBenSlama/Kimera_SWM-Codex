#!/usr/bin/env python3
"""
ðŸ”§ TA-LIB FALLBACK IMPLEMENTATION ðŸ”§
Pure Python/NumPy/Pandas implementation of TA-Lib functions
Eliminates external C library dependencies
"""

import numpy as np
import pandas as pd
from typing import Tuple, Union, Optional
import warnings

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore')

def RSI(close: np.ndarray, timeperiod: int = 14) -> np.ndarray:
    """
    Relative Strength Index (RSI)
    
    Args:
        close: Close prices array
        timeperiod: Period for RSI calculation
        
    Returns:
        RSI values array
    """
    if len(close) < timeperiod + 1:
        return np.full(len(close), np.nan)
    
    # Calculate price changes
    delta = np.diff(close)
    
    # Separate gains and losses
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    # Calculate initial averages
    avg_gain = np.mean(gains[:timeperiod])
    avg_loss = np.mean(losses[:timeperiod])
    
    # Initialize RSI array
    rsi = np.full(len(close), np.nan)
    
    # Calculate RSI for each period
    for i in range(timeperiod, len(close)):
        if i == timeperiod:
            # First RSI calculation
            rs = avg_gain / avg_loss if avg_loss != 0 else 0
        else:
            # Smoothed averages (Wilder's smoothing)
            avg_gain = (avg_gain * (timeperiod - 1) + gains[i-1]) / timeperiod
            avg_loss = (avg_loss * (timeperiod - 1) + losses[i-1]) / timeperiod
            rs = avg_gain / avg_loss if avg_loss != 0 else 0
        
        rsi[i] = 100 - (100 / (1 + rs))
    
    return rsi

def MACD(close: np.ndarray, fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Moving Average Convergence Divergence (MACD)
    
    Args:
        close: Close prices array
        fastperiod: Fast EMA period
        slowperiod: Slow EMA period  
        signalperiod: Signal line EMA period
        
    Returns:
        Tuple of (MACD line, Signal line, Histogram)
    """
    # Calculate EMAs
    ema_fast = EMA(close, fastperiod)
    ema_slow = EMA(close, slowperiod)
    
    # MACD line
    macd_line = ema_fast - ema_slow
    
    # Signal line (EMA of MACD)
    signal_line = EMA(macd_line, signalperiod)
    
    # Histogram
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

def BBANDS(close: np.ndarray, timeperiod: int = 20, nbdevup: float = 2.0, nbdevdn: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bollinger Bands
    
    Args:
        close: Close prices array
        timeperiod: Period for moving average
        nbdevup: Number of standard deviations for upper band
        nbdevdn: Number of standard deviations for lower band
        
    Returns:
        Tuple of (Upper band, Middle band, Lower band)
    """
    # Calculate Simple Moving Average (middle band)
    middle_band = SMA(close, timeperiod)
    
    # Calculate standard deviation
    std_dev = np.full(len(close), np.nan)
    for i in range(timeperiod - 1, len(close)):
        std_dev[i] = np.std(close[i - timeperiod + 1:i + 1])
    
    # Calculate bands
    upper_band = middle_band + (nbdevup * std_dev)
    lower_band = middle_band - (nbdevdn * std_dev)
    
    return upper_band, middle_band, lower_band

def ROC(close: np.ndarray, timeperiod: int = 10) -> np.ndarray:
    """
    Rate of Change (ROC)
    
    Args:
        close: Close prices array
        timeperiod: Period for ROC calculation
        
    Returns:
        ROC values array
    """
    if len(close) < timeperiod + 1:
        return np.full(len(close), np.nan)
    
    roc = np.full(len(close), np.nan)
    
    for i in range(timeperiod, len(close)):
        if close[i - timeperiod] != 0:
            roc[i] = ((close[i] - close[i - timeperiod]) / close[i - timeperiod]) * 100
    
    return roc

def SMA(close: np.ndarray, timeperiod: int) -> np.ndarray:
    """
    Simple Moving Average (SMA)
    
    Args:
        close: Close prices array
        timeperiod: Period for moving average
        
    Returns:
        SMA values array
    """
    if len(close) < timeperiod:
        return np.full(len(close), np.nan)
    
    sma = np.full(len(close), np.nan)
    
    for i in range(timeperiod - 1, len(close)):
        sma[i] = np.mean(close[i - timeperiod + 1:i + 1])
    
    return sma

def EMA(close: np.ndarray, timeperiod: int) -> np.ndarray:
    """
    Exponential Moving Average (EMA)
    
    Args:
        close: Close prices array
        timeperiod: Period for EMA calculation
        
    Returns:
        EMA values array
    """
    if len(close) < timeperiod:
        return np.full(len(close), np.nan)
    
    ema = np.full(len(close), np.nan)
    multiplier = 2.0 / (timeperiod + 1)
    
    # Initialize with SMA
    ema[timeperiod - 1] = np.mean(close[:timeperiod])
    
    # Calculate EMA
    for i in range(timeperiod, len(close)):
        ema[i] = (close[i] * multiplier) + (ema[i - 1] * (1 - multiplier))
    
    return ema

def WMA(close: np.ndarray, timeperiod: int) -> np.ndarray:
    """
    Weighted Moving Average (WMA)
    
    Args:
        close: Close prices array
        timeperiod: Period for WMA calculation
        
    Returns:
        WMA values array
    """
    if len(close) < timeperiod:
        return np.full(len(close), np.nan)
    
    wma = np.full(len(close), np.nan)
    weights = np.arange(1, timeperiod + 1)
    
    for i in range(timeperiod - 1, len(close)):
        wma[i] = np.average(close[i - timeperiod + 1:i + 1], weights=weights)
    
    return wma

def STOCH(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
          fastk_period: int = 14, slowk_period: int = 3, slowd_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stochastic Oscillator
    
    Args:
        high: High prices array
        low: Low prices array
        close: Close prices array
        fastk_period: Fast %K period
        slowk_period: Slow %K period
        slowd_period: Slow %D period
        
    Returns:
        Tuple of (%K, %D)
    """
    if len(close) < fastk_period:
        return np.full(len(close), np.nan), np.full(len(close), np.nan)
    
    # Calculate %K
    fastk = np.full(len(close), np.nan)
    
    for i in range(fastk_period - 1, len(close)):
        period_high = np.max(high[i - fastk_period + 1:i + 1])
        period_low = np.min(low[i - fastk_period + 1:i + 1])
        
        if period_high != period_low:
            fastk[i] = ((close[i] - period_low) / (period_high - period_low)) * 100
        else:
            fastk[i] = 50  # Neutral value when no range
    
    # Calculate slow %K (SMA of fast %K)
    slowk = SMA(fastk, slowk_period)
    
    # Calculate %D (SMA of slow %K)
    slowd = SMA(slowk, slowd_period)
    
    return slowk, slowd

def ATR(high: np.ndarray, low: np.ndarray, close: np.ndarray, timeperiod: int = 14) -> np.ndarray:
    """
    Average True Range (ATR)
    
    Args:
        high: High prices array
        low: Low prices array
        close: Close prices array
        timeperiod: Period for ATR calculation
        
    Returns:
        ATR values array
    """
    if len(close) < 2:
        return np.full(len(close), np.nan)
    
    # Calculate True Range
    tr = np.full(len(close), np.nan)
    
    for i in range(1, len(close)):
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i - 1])
        tr3 = abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    # Calculate ATR (SMA of True Range)
    atr = SMA(tr, timeperiod)
    
    return atr

def ADX(high: np.ndarray, low: np.ndarray, close: np.ndarray, timeperiod: int = 14) -> np.ndarray:
    """
    Average Directional Index (ADX)
    
    Args:
        high: High prices array
        low: Low prices array
        close: Close prices array
        timeperiod: Period for ADX calculation
        
    Returns:
        ADX values array
    """
    if len(close) < timeperiod + 1:
        return np.full(len(close), np.nan)
    
    # Calculate True Range and Directional Movement
    tr = np.full(len(close), np.nan)
    dm_plus = np.full(len(close), np.nan)
    dm_minus = np.full(len(close), np.nan)
    
    for i in range(1, len(close)):
        # True Range
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i - 1])
        tr3 = abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
        
        # Directional Movement
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        
        if up_move > down_move and up_move > 0:
            dm_plus[i] = up_move
        else:
            dm_plus[i] = 0
            
        if down_move > up_move and down_move > 0:
            dm_minus[i] = down_move
        else:
            dm_minus[i] = 0
    
    # Calculate smoothed values
    atr = EMA(tr, timeperiod)
    di_plus = 100 * EMA(dm_plus, timeperiod) / atr
    di_minus = 100 * EMA(dm_minus, timeperiod) / atr
    
    # Calculate DX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    
    # Calculate ADX
    adx = EMA(dx, timeperiod)
    
    return adx

# Additional utility functions
def TRANGE(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """True Range"""
    if len(close) < 2:
        return np.full(len(close), np.nan)
    
    tr = np.full(len(close), np.nan)
    
    for i in range(1, len(close)):
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i - 1])
        tr3 = abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    return tr

def MAMA(close: np.ndarray, fastlimit: float = 0.5, slowlimit: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """MESA Adaptive Moving Average"""
    # Simplified implementation
    mama = EMA(close, 10)
    fama = EMA(close, 20)
    return mama, fama

def WILLR(high: np.ndarray, low: np.ndarray, close: np.ndarray, timeperiod: int = 14) -> np.ndarray:
    """Williams %R"""
    if len(close) < timeperiod:
        return np.full(len(close), np.nan)
    
    willr = np.full(len(close), np.nan)
    
    for i in range(timeperiod - 1, len(close)):
        period_high = np.max(high[i - timeperiod + 1:i + 1])
        period_low = np.min(low[i - timeperiod + 1:i + 1])
        
        if period_high != period_low:
            willr[i] = ((period_high - close[i]) / (period_high - period_low)) * -100
        else:
            willr[i] = -50  # Neutral value
    
    return willr

# Function availability check
def get_available_functions():
    """Return list of available TA-Lib fallback functions"""
    return [
        'RSI', 'MACD', 'BBANDS', 'ROC', 'SMA', 'EMA', 'WMA', 
        'STOCH', 'ATR', 'ADX', 'TRANGE', 'MAMA', 'WILLR'
    ]

def is_function_available(func_name: str) -> bool:
    """Check if a TA-Lib function is available in fallback"""
    return func_name in get_available_functions()

# Module info
__version__ = "1.0.0"
__author__ = "Kimera SWM"
__description__ = "Pure Python TA-Lib fallback implementation"

print(f"[OK] TA-Lib Fallback v{__version__} loaded - {len(get_available_functions())} functions available") 