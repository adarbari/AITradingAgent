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
    # Ensure we're working with 1D arrays
    close_values = data['Close'].values
    if len(close_values.shape) > 1:
        close_values = close_values.flatten()
    
    close = pd.Series(close_values, index=data.index)
    sma = close.rolling(window=5).mean()
    sma = sma.bfill().fillna(close)
    
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
    # Ensure we're working with 1D arrays
    close_values = data['Close'].values
    if len(close_values.shape) > 1:
        close_values = close_values.flatten()
    
    close = pd.Series(close_values, index=data.index)
    sma = close.rolling(window=10).mean()
    sma = sma.bfill().fillna(close)
    
    # SMA/Price ratio (> 1 means price is below SMA, < 1 means price is above SMA)
    ratio = sma / close
    
    result = np.nan_to_num(ratio.values, nan=1.0, posinf=1.0, neginf=1.0)
    return pd.Series(result, index=data.index)


@FeatureRegistry.register(name="sma_20", category="trend")
def calculate_sma_20(data: pd.DataFrame) -> pd.Series:
    """
    Calculate the 20-day Simple Moving Average ratio to current price.
    
    Args:
        data (pd.DataFrame): OHLCV data
        
    Returns:
        pd.Series: SMA-20 to price ratio
    """
    # Ensure we're working with 1D arrays
    close_values = data['Close'].values
    if len(close_values.shape) > 1:
        close_values = close_values.flatten()
    
    close = pd.Series(close_values, index=data.index)
    sma = close.rolling(window=20).mean().bfill().fillna(close)
    
    # Calculate ratio and handle potential division by zero
    ratio = sma / close.clip(lower=1e-8)
    
    # Replace NaNs and infinities with 1.0
    ratio = ratio.fillna(1.0).replace([np.inf, -np.inf], 1.0)
    
    # Ensure we return a Series, not a DataFrame
    if isinstance(ratio, pd.DataFrame):
        ratio = pd.Series(ratio.values.flatten(), index=ratio.index)
    
    return ratio


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
    sma = sma.bfill().fillna(close)
    
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
    sma = sma.bfill().fillna(close)
    
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
    # Ensure we're working with 1D arrays
    close_values = data['Close'].values
    if len(close_values.shape) > 1:
        close_values = close_values.flatten()
    
    close = pd.Series(close_values, index=data.index)
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
    # Ensure we're working with 1D arrays
    close_values = data['Close'].values
    if len(close_values.shape) > 1:
        close_values = close_values.flatten()
    
    close = pd.Series(close_values, index=data.index)
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


@FeatureRegistry.register(name="ema_10", category="trend")
def calculate_ema_10(data: pd.DataFrame) -> pd.Series:
    """
    Calculate the 10-day Exponential Moving Average ratio to current price.
    
    Args:
        data (pd.DataFrame): OHLCV data
        
    Returns:
        pd.Series: EMA-10 to price ratio
    """
    close = data['Close'].values
    
    # Calculate EMA
    alpha = 2 / (10 + 1)
    ema = np.zeros_like(close)
    
    # First value is the first close
    ema[0] = close[0]
    
    # Calculate EMA
    for i in range(1, len(close)):
        ema[i] = close[i] * alpha + ema[i-1] * (1 - alpha)
    
    # Calculate ratio
    ratio = ema / np.maximum(close, 1e-8)
    
    # Replace NaNs and infinities with 1
    result = np.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=1.0)
    return pd.Series(result, index=data.index)


@FeatureRegistry.register(name="ma_crossover", category="trend")
def calculate_ma_crossover(data: pd.DataFrame, 
                          short_window: int = 10, 
                          long_window: int = 50) -> pd.Series:
    """
    Calculate moving average crossover signal.
    
    Args:
        data (pd.DataFrame): OHLCV data
        short_window (int): Window size for short MA
        long_window (int): Window size for long MA
        
    Returns:
        pd.Series: Crossover signal (-1 to 1)
    """
    close = data['Close']
    
    # Calculate short and long MAs
    short_ma = close.rolling(window=short_window).mean()
    long_ma = close.rolling(window=long_window).mean()
    
    # Use backfill for missing values
    short_ma = short_ma.bfill().fillna(close)
    long_ma = long_ma.bfill().fillna(close)
    
    # Calculate crossover signal
    signal = (short_ma - long_ma) / close.clip(lower=1e-8)
    
    # Normalize to -1 to 1 range
    normalized = signal.clip(lower=-1, upper=1)  # Scale and clip
    
    # Return the result as a Series with the same index as the input data
    return normalized


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
    Calculate the Average Directional Index (ADX).
    
    Args:
        data (pd.DataFrame): OHLCV data
        window (int): Window size for ADX calculation
        
    Returns:
        pd.Series: ADX values (normalized to 0-1 range)
    """
    high = data['High'].values
    low = data['Low'].values
    close = data['Close'].values
    
    # Get previous highs and lows
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    
    # Set initial values to the current values
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    # Calculate +DM, -DM
    plus_dm = np.maximum(high - prev_high, 0)
    minus_dm = np.maximum(prev_low - low, 0)
    
    # When +DM > -DM, set -DM to zero
    plus_dm = np.where(plus_dm > minus_dm, plus_dm, 0)
    
    # When -DM > +DM, set +DM to zero
    minus_dm = np.where(minus_dm > plus_dm, minus_dm, 0)
    
    # Calculate true range
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Calculate smoothed values
    tr_smooth = np.zeros_like(tr)
    plus_smooth = np.zeros_like(plus_dm)
    minus_smooth = np.zeros_like(minus_dm)
    
    # First values
    tr_smooth[0] = tr[0]
    plus_smooth[0] = plus_dm[0]
    minus_smooth[0] = minus_dm[0]
    
    # Calculate smoothed values
    for i in range(1, len(tr)):
        tr_smooth[i] = tr_smooth[i-1] - (tr_smooth[i-1] / window) + tr[i]
        plus_smooth[i] = plus_smooth[i-1] - (plus_smooth[i-1] / window) + plus_dm[i]
        minus_smooth[i] = minus_smooth[i-1] - (minus_smooth[i-1] / window) + minus_dm[i]
    
    # Calculate directional indicators
    plus_di = 100 * plus_smooth / np.maximum(tr_smooth, 1e-8)
    minus_di = 100 * minus_smooth / np.maximum(tr_smooth, 1e-8)
    
    # Calculate directional index
    dx = 100 * np.abs(plus_di - minus_di) / np.maximum(plus_di + minus_di, 1e-8)
    
    # Calculate ADX with smoothing
    adx = np.zeros_like(dx)
    adx[0] = dx[0]
    
    for i in range(1, len(dx)):
        adx[i] = adx[i-1] + (dx[i] - adx[i-1]) / window
    
    # Normalize to 0-1 range
    adx_normalized = adx / 100.0
    
    # Replace NaNs and Infs
    result = np.nan_to_num(adx_normalized, nan=0.0, posinf=1.0, neginf=0.0)
    return pd.Series(result, index=data.index)


@FeatureRegistry.register(name="aroon_oscillator", category="trend")
def calculate_aroon_oscillator(data: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Calculate the Aroon Oscillator.
    
    Args:
        data (pd.DataFrame): OHLCV data
        window (int): Window size for Aroon calculation
        
    Returns:
        pd.Series: Aroon Oscillator values (-1 to 1 range)
    """
    high = data['High'].values
    low = data['Low'].values
    
    # Initialize arrays
    aroon_up = np.zeros_like(high)
    aroon_down = np.zeros_like(low)
    
    # Calculate Aroon Up and Down
    for i in range(len(high)):
        if i < window:
            # For the first few points, use available data
            period_high = high[0:i+1]
            period_low = low[0:i+1]
            
            if len(period_high) > 0:  # Check if we have any data
                high_idx = np.argmax(period_high)
                low_idx = np.argmin(period_low)
                
                aroon_up[i] = (i - high_idx) / i if i > 0 else 0
                aroon_down[i] = (i - low_idx) / i if i > 0 else 0
            else:
                aroon_up[i] = 0
                aroon_down[i] = 0
        else:
            # Use full window
            period_high = high[i-window+1:i+1]
            period_low = low[i-window+1:i+1]
            
            high_idx = np.argmax(period_high)
            low_idx = np.argmin(period_low)
            
            aroon_up[i] = (window - 1 - high_idx) / (window - 1)
            aroon_down[i] = (window - 1 - low_idx) / (window - 1)
    
    # Calculate oscillator (up - down, normalized to -1 to 1)
    oscillator = aroon_up - aroon_down
    
    # Replace NaNs
    result = np.nan_to_num(oscillator, nan=0.0)
    return pd.Series(result, index=data.index) 