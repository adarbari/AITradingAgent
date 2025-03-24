"""
Trend Features Module

Contains feature generators for trend-based technical indicators.
"""
import numpy as np
import pandas as pd
from typing import Optional

from ..registry import FeatureRegistry


@FeatureRegistry.register(name="sma_5", category="trend")
def calculate_sma_5(data: pd.DataFrame) -> pd.Series:
    """
    Calculate the 5-day Simple Moving Average ratio to close price.
    
    Args:
        data (pd.DataFrame): OHLCV data
        
    Returns:
        pd.Series: SMA5 to close price ratio
    """
    close = data['Close']
    sma = close.rolling(window=5).mean()
    sma = sma.fillna(method='bfill').fillna(close)
    
    # SMA/Price ratio (> 1 means price is below SMA, < 1 means price is above SMA)
    ratio = sma / close
    
    result = np.nan_to_num(ratio.values, nan=1.0, posinf=1.0, neginf=1.0)
    return pd.Series(result, index=data.index)


@FeatureRegistry.register(name="sma_10", category="trend")
def calculate_sma_10(data: pd.DataFrame) -> pd.Series:
    """
    Calculate the 10-day Simple Moving Average ratio to close price.
    
    Args:
        data (pd.DataFrame): OHLCV data
        
    Returns:
        pd.Series: SMA10 to close price ratio
    """
    close = data['Close']
    sma = close.rolling(window=10).mean()
    sma = sma.fillna(method='bfill').fillna(close)
    
    # SMA/Price ratio (> 1 means price is below SMA, < 1 means price is above SMA)
    ratio = sma / close
    
    result = np.nan_to_num(ratio.values, nan=1.0, posinf=1.0, neginf=1.0)
    return pd.Series(result, index=data.index)


@FeatureRegistry.register(name="sma_20", category="trend")
def calculate_sma_20(data: pd.DataFrame) -> pd.Series:
    """
    Calculate the 20-day Simple Moving Average (SMA) ratio to current price.
    
    Args:
        data (pd.DataFrame): OHLCV data
        
    Returns:
        pd.Series: SMA/Price ratio values
    """
    close = data['Close']
    sma = close.rolling(window=20).mean()
    sma = sma.fillna(method='bfill').fillna(close)
    
    # SMA/Price ratio (> 1 means price is below SMA, < 1 means price is above SMA)
    ratio = sma / close
    
    result = np.nan_to_num(ratio.values, nan=1.0, posinf=1.0, neginf=1.0)
    return pd.Series(result, index=data.index)


@FeatureRegistry.register(name="sma_50", category="trend")
def calculate_sma_50(data: pd.DataFrame) -> pd.Series:
    """
    Calculate the 50-day Simple Moving Average ratio to close price.
    
    Args:
        data (pd.DataFrame): OHLCV data
        
    Returns:
        pd.Series: SMA50 to close price ratio
    """
    close = data['Close']
    sma = close.rolling(window=50).mean()
    sma = sma.fillna(method='bfill').fillna(close)
    
    # SMA/Price ratio (> 1 means price is below SMA, < 1 means price is above SMA)
    ratio = sma / close
    
    result = np.nan_to_num(ratio.values, nan=1.0, posinf=1.0, neginf=1.0)
    return pd.Series(result, index=data.index)


@FeatureRegistry.register(name="sma_200", category="trend")
def calculate_sma_200(data: pd.DataFrame) -> pd.Series:
    """
    Calculate the 200-day Simple Moving Average ratio to close price.
    
    Args:
        data (pd.DataFrame): OHLCV data
        
    Returns:
        pd.Series: SMA200 to close price ratio
    """
    close = data['Close']
    sma = close.rolling(window=200).mean()
    sma = sma.fillna(method='bfill').fillna(close)
    
    # SMA/Price ratio (> 1 means price is below SMA, < 1 means price is above SMA)
    ratio = sma / close
    
    result = np.nan_to_num(ratio.values, nan=1.0, posinf=1.0, neginf=1.0)
    return pd.Series(result, index=data.index)


@FeatureRegistry.register(name="ema_12", category="trend")
def calculate_ema_12(data: pd.DataFrame) -> pd.Series:
    """
    Calculate the 12-day Exponential Moving Average ratio to close price.
    
    Args:
        data (pd.DataFrame): OHLCV data
        
    Returns:
        pd.Series: EMA12 to close price ratio
    """
    close = data['Close']
    ema = close.ewm(span=12, adjust=False).mean()
    
    # EMA/Price ratio (> 1 means price is below EMA, < 1 means price is above EMA)
    ratio = ema / close
    
    result = np.nan_to_num(ratio.values, nan=1.0, posinf=1.0, neginf=1.0)
    return pd.Series(result, index=data.index)


@FeatureRegistry.register(name="ema_26", category="trend")
def calculate_ema_26(data: pd.DataFrame) -> pd.Series:
    """
    Calculate the 26-day Exponential Moving Average ratio to close price.
    
    Args:
        data (pd.DataFrame): OHLCV data
        
    Returns:
        pd.Series: EMA26 to close price ratio
    """
    close = data['Close']
    ema = close.ewm(span=26, adjust=False).mean()
    
    # EMA/Price ratio (> 1 means price is below EMA, < 1 means price is above EMA)
    ratio = ema / close
    
    result = np.nan_to_num(ratio.values, nan=1.0, posinf=1.0, neginf=1.0)
    return pd.Series(result, index=data.index)


@FeatureRegistry.register(name="ema_20", category="trend")
def calculate_ema_20(data: pd.DataFrame) -> pd.Series:
    """
    Calculate the 20-day Exponential Moving Average (EMA) ratio to current price.
    
    Args:
        data (pd.DataFrame): OHLCV data
        
    Returns:
        pd.Series: EMA/Price ratio values
    """
    close = data['Close']
    ema = close.ewm(span=20, adjust=False).mean()
    
    # EMA/Price ratio (> 1 means price is below EMA, < 1 means price is above EMA)
    ratio = ema / close
    
    result = np.nan_to_num(ratio.values, nan=1.0, posinf=1.0, neginf=1.0)
    return pd.Series(result, index=data.index)


@FeatureRegistry.register(name="ma_crossover", category="trend")
def calculate_ma_crossover(data: pd.DataFrame) -> pd.Series:
    """
    Calculate the crossover signal between short-term and long-term moving averages.
    
    Args:
        data (pd.DataFrame): OHLCV data
        
    Returns:
        pd.Series: Crossover signal (-1 to 1 range)
    """
    close = data['Close']
    
    # Calculate short and long-term moving averages
    short_ma = close.rolling(window=10).mean()
    long_ma = close.rolling(window=50).mean()
    
    # Ensure we have values from the beginning
    short_ma = short_ma.fillna(method='bfill').fillna(close)
    long_ma = long_ma.fillna(method='bfill').fillna(close)
    
    # Calculate relative difference
    diff = (short_ma - long_ma) / long_ma
    
    # Normalize to -1 to 1 range (a 10% difference is considered significant)
    signal = np.clip(diff * 10, -1, 1)
    
    result = np.nan_to_num(signal.values, nan=0.0)
    return pd.Series(result, index=data.index)


@FeatureRegistry.register(name="price_trend", category="trend")
def calculate_price_trend(data: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Calculate the price trend direction and strength.
    
    Args:
        data (pd.DataFrame): OHLCV data
        window (int): Window size for trend calculation
        
    Returns:
        pd.Series: Trend strength (-1 to 1 range)
    """
    close = data['Close'].values
    
    # Convert to pandas Series
    close_s = pd.Series(close)
    
    # Calculate linear regression slope
    x = np.arange(window)
    
    # Rolling window calculations
    slopes = []
    for i in range(len(close_s) - window + 1):
        y = close_s.iloc[i:i+window]
        slope, _ = np.polyfit(x, y, 1)
        slopes.append(slope)
    
    # Pad beginning with initial value
    slopes = [slopes[0]] * (window - 1) + slopes
    
    # Normalize by price level to get percentage trend
    normalized_slopes = np.array(slopes) / np.maximum(close, 1e-8) * window
    
    # Clip to -1 to 1 range (a trend of 10% over the window period is considered significant)
    normalized_slopes = np.clip(normalized_slopes * 10, -1, 1)
    
    result = np.nan_to_num(normalized_slopes, nan=0.0)
    return pd.Series(result, index=data.index)


@FeatureRegistry.register(name="adx", category="trend")
def calculate_adx(data: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Calculate the Average Directional Index (ADX) for trend strength.
    
    Args:
        data (pd.DataFrame): OHLCV data
        window (int): Window size for ADX calculation
        
    Returns:
        pd.Series: ADX values (0-1 range)
    """
    high = data['High'].values
    low = data['Low'].values
    close = data['Close'].values
    
    # Calculate True Range
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.insert(tr, 0, tr1[0])  # Set first value
    
    # Calculate +DM and -DM
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    
    # +DM and -DM
    plus_dm = np.zeros_like(up_move)
    minus_dm = np.zeros_like(down_move)
    
    plus_dm[(up_move > down_move) & (up_move > 0)] = up_move[(up_move > down_move) & (up_move > 0)]
    minus_dm[(down_move > up_move) & (down_move > 0)] = down_move[(down_move > up_move) & (down_move > 0)]
    
    # Pad the first value
    plus_dm = np.insert(plus_dm, 0, 0)
    minus_dm = np.insert(minus_dm, 0, 0)
    
    # Convert to pandas for rolling calculations
    tr_s = pd.Series(tr)
    plus_dm_s = pd.Series(plus_dm)
    minus_dm_s = pd.Series(minus_dm)
    
    # Smooth the indicators
    atr = tr_s.rolling(window=window).mean()
    plus_di = 100 * (plus_dm_s.rolling(window=window).mean() / atr.clip(lower=1e-8))
    minus_di = 100 * (minus_dm_s.rolling(window=window).mean() / atr.clip(lower=1e-8))
    
    # Calculate ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).clip(lower=1e-8)
    adx = dx.rolling(window=window).mean()
    
    # Normalize to 0-1 range (ADX > 25 is considered strong trend)
    adx_normalized = adx / 100.0
    
    result = np.nan_to_num(adx_normalized.values, nan=0.0)
    return pd.Series(result, index=data.index) 