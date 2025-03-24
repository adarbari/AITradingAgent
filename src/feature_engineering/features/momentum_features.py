"""
Momentum Features Module

Contains feature generators for momentum-based indicators.
"""
import numpy as np
import pandas as pd
from typing import Optional

from ..registry import FeatureRegistry


@FeatureRegistry.register(name="rsi_14", category="momentum")
def calculate_rsi(data: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Calculate the Relative Strength Index (RSI).
    
    Args:
        data (pd.DataFrame): OHLCV data
        window (int): Window size for RSI calculation
        
    Returns:
        pd.Series: RSI values (normalized to 0-1 range)
    """
    # Ensure we're working with a 1D array
    close = data['Close'].values
    if len(close.shape) > 1:
        close = close.flatten()
        
    delta = np.diff(close, prepend=close[0])
    
    # Separate gains and losses
    gains = np.maximum(delta, 0)
    losses = np.maximum(-delta, 0)
    
    # Initialize arrays for avg_gain and avg_loss
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    # Calculate first average
    if len(gains) >= window:
        avg_gain[window-1] = np.mean(gains[:window])
        avg_loss[window-1] = np.mean(losses[:window])
    
    # Calculate remaining averages using smoothing
    for i in range(window, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (window-1) + gains[i]) / window
        avg_loss[i] = (avg_loss[i-1] * (window-1) + losses[i]) / window
    
    # Calculate RS and RSI
    rs = np.divide(avg_gain, np.maximum(avg_loss, 1e-8))
    rsi = 1.0 - (1.0 / (1.0 + rs))
    
    # Fill initial values
    rsi[:window-1] = 0.5  # Neutral value for initial points
    
    # Replace NaNs and Infs with 0.5 (neutral value)
    result = np.nan_to_num(rsi, nan=0.5, posinf=1.0, neginf=0.0)
    return pd.Series(result, index=data.index)


@FeatureRegistry.register(name="rsi_2", category="momentum")
def calculate_rsi_2(data: pd.DataFrame) -> pd.Series:
    """
    Calculate the 2-day Relative Strength Index (RSI).
    
    Args:
        data (pd.DataFrame): OHLCV data
        
    Returns:
        pd.Series: RSI values (0-100 range, normalized to 0-1)
    """
    close = data['Close'].values
    period = 2
    
    # Calculate price changes
    delta = pd.Series(close).diff().fillna(0)
    
    # Separate gains and losses
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # Calculate average gain and loss
    avg_gain = gain.rolling(window=period).mean().fillna(0)
    avg_loss = loss.rolling(window=period).mean().fillna(0)
    
    # Calculate RS (Relative Strength)
    rs = np.where(avg_loss < 1e-8, 100, avg_gain / np.maximum(avg_loss, 1e-8))
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    
    # Normalize to 0-1 range
    rsi_normalized = rsi / 100.0
    
    result = np.nan_to_num(rsi_normalized, nan=0.5)  # Default to middle value if NaN
    return pd.Series(result, index=data.index)


@FeatureRegistry.register(name="macd", category="momentum")
def calculate_macd(data: pd.DataFrame, 
                   fast_period: int = 12, 
                   slow_period: int = 26) -> pd.Series:
    """
    Calculate the Moving Average Convergence Divergence (MACD) line.
    
    Args:
        data (pd.DataFrame): OHLCV data
        fast_period (int): Period for the fast EMA
        slow_period (int): Period for the slow EMA
        
    Returns:
        pd.Series: MACD line values normalized by price
    """
    # Ensure we're working with 1D arrays
    close_values = data['Close'].values
    if len(close_values.shape) > 1:
        close_values = close_values.flatten()
    
    close = pd.Series(close_values, index=data.index)
    
    # Calculate EMAs directly using pandas
    ema_fast = close.ewm(span=fast_period, adjust=False).mean()
    ema_slow = close.ewm(span=slow_period, adjust=False).mean()
    
    # Calculate MACD line and normalize by price
    macd_line = (ema_fast - ema_slow) / close
    
    # Ensure we return a Series, not a DataFrame
    if isinstance(macd_line, pd.DataFrame):
        macd_line = pd.Series(macd_line.values.flatten(), index=macd_line.index)
    
    return macd_line


@FeatureRegistry.register(name="macd_signal", category="momentum")
def calculate_macd_signal(data: pd.DataFrame, 
                         fast_period: int = 12, 
                         slow_period: int = 26,
                         signal_period: int = 9) -> pd.Series:
    """
    Calculate the MACD signal line.
    
    Args:
        data (pd.DataFrame): OHLCV data
        fast_period (int): Period for the fast EMA
        slow_period (int): Period for the slow EMA
        signal_period (int): Period for the signal line
        
    Returns:
        pd.Series: MACD signal line values normalized by price
    """
    # Ensure we're working with a 1D array
    close = data['Close'].values
    if len(close.shape) > 1:
        close = close.flatten()
    
    # Calculate fast and slow EMAs
    fast_alpha = 2 / (fast_period + 1)
    slow_alpha = 2 / (slow_period + 1)
    
    ema_fast = np.zeros_like(close)
    ema_slow = np.zeros_like(close)
    
    # Calculate first value
    ema_fast[0] = close[0]
    ema_slow[0] = close[0]
    
    # Calculate EMA values
    for i in range(1, len(close)):
        ema_fast[i] = close[i] * fast_alpha + ema_fast[i-1] * (1 - fast_alpha)
        ema_slow[i] = close[i] * slow_alpha + ema_slow[i-1] * (1 - slow_alpha)
    
    # Calculate MACD line
    macd_line = ema_fast - ema_slow
    
    # Calculate signal line (EMA of MACD line)
    signal_alpha = 2 / (signal_period + 1)
    signal_line = np.zeros_like(close)
    signal_line[0] = macd_line[0]
    
    for i in range(1, len(macd_line)):
        signal_line[i] = macd_line[i] * signal_alpha + signal_line[i-1] * (1 - signal_alpha)
    
    # Normalize by price
    signal_normalized = signal_line / np.maximum(close, 1e-8)
    
    # Replace NaNs and Infs with zeros
    result = np.nan_to_num(signal_normalized, nan=0.0, posinf=0.0, neginf=0.0)
    return pd.Series(result, index=data.index)


@FeatureRegistry.register(name="macd_histogram", category="momentum")
def calculate_macd_histogram(data: pd.DataFrame, 
                            fast_period: int = 12, 
                            slow_period: int = 26,
                            signal_period: int = 9) -> pd.Series:
    """
    Calculate the MACD histogram (difference between MACD line and signal line).
    
    Args:
        data (pd.DataFrame): OHLCV data
        fast_period (int): Period for the fast EMA
        slow_period (int): Period for the slow EMA
        signal_period (int): Period for the signal line
        
    Returns:
        pd.Series: MACD histogram values normalized by price
    """
    # Ensure we're working with a 1D array
    close = data['Close'].values
    if len(close.shape) > 1:
        close = close.flatten()
    
    # Calculate fast and slow EMAs
    fast_alpha = 2 / (fast_period + 1)
    slow_alpha = 2 / (slow_period + 1)
    
    ema_fast = np.zeros_like(close)
    ema_slow = np.zeros_like(close)
    
    # Calculate first value
    ema_fast[0] = close[0]
    ema_slow[0] = close[0]
    
    # Calculate EMA values
    for i in range(1, len(close)):
        ema_fast[i] = close[i] * fast_alpha + ema_fast[i-1] * (1 - fast_alpha)
        ema_slow[i] = close[i] * slow_alpha + ema_slow[i-1] * (1 - slow_alpha)
    
    # Calculate MACD line
    macd_line = ema_fast - ema_slow
    
    # Calculate signal line (EMA of MACD line)
    signal_alpha = 2 / (signal_period + 1)
    signal_line = np.zeros_like(close)
    signal_line[0] = macd_line[0]
    
    for i in range(1, len(macd_line)):
        signal_line[i] = macd_line[i] * signal_alpha + signal_line[i-1] * (1 - signal_alpha)
    
    # Calculate histogram
    histogram = macd_line - signal_line
    
    # Normalize by price
    hist_normalized = histogram / np.maximum(close, 1e-8)
    
    # Replace NaNs and Infs with zeros
    result = np.nan_to_num(hist_normalized, nan=0.0, posinf=0.0, neginf=0.0)
    return pd.Series(result, index=data.index)


@FeatureRegistry.register(name="momentum_5", category="momentum")
def calculate_momentum_5(data: pd.DataFrame) -> pd.Series:
    """
    Calculate the 5-day momentum indicator.
    
    Args:
        data (pd.DataFrame): OHLCV data
        
    Returns:
        pd.Series: Momentum values
    """
    # Ensure we're working with a 1D array
    close = data['Close'].values
    if len(close.shape) > 1:
        close = close.flatten()
        
    period = 5
    
    # Calculate momentum (current price / price n periods ago - 1)
    momentum = pd.Series(close).pct_change(periods=period).fillna(0).values
    
    result = np.nan_to_num(momentum, nan=0.0, posinf=0.0, neginf=0.0)
    return pd.Series(result, index=data.index)


@FeatureRegistry.register(name="stoch_k", category="momentum")
def calculate_stochastic_k(data: pd.DataFrame, 
                          window: int = 14) -> pd.Series:
    """
    Calculate the Stochastic Oscillator %K.
    
    Args:
        data (pd.DataFrame): OHLCV data
        window (int): Window size for stochastic calculation
        
    Returns:
        pd.Series: %K values normalized to 0-1 range
    """
    # Ensure we're working with 1D arrays
    high = data['High'].values
    low = data['Low'].values
    close = data['Close'].values
    
    if len(high.shape) > 1:
        high = high.flatten()
    if len(low.shape) > 1:
        low = low.flatten()
    if len(close.shape) > 1:
        close = close.flatten()
    
    # Calculate rolling highest high and lowest low
    highest_high = np.zeros_like(close)
    lowest_low = np.zeros_like(close)
    
    for i in range(len(close)):
        if i < window:
            # Use available data for first few points
            highest_high[i] = np.max(high[0:i+1])
            lowest_low[i] = np.min(low[0:i+1])
        else:
            # Use full window for later points
            highest_high[i] = np.max(high[i-window+1:i+1])
            lowest_low[i] = np.min(low[i-window+1:i+1])
    
    # Calculate %K
    k_values = (close - lowest_low) / np.maximum(highest_high - lowest_low, 1e-8)
    
    # Replace NaNs and Infs with 0.5 (neutral value)
    result = np.nan_to_num(k_values, nan=0.5, posinf=1.0, neginf=0.0)
    return pd.Series(result, index=data.index)


@FeatureRegistry.register(name="stoch_d", category="momentum")
def calculate_stochastic_d(data: pd.DataFrame, 
                          k_window: int = 14,
                          d_window: int = 3) -> pd.Series:
    """
    Calculate the Stochastic Oscillator %D.
    
    Args:
        data (pd.DataFrame): OHLCV data
        k_window (int): Window size for %K calculation
        d_window (int): Window size for %D calculation (moving average of %K)
        
    Returns:
        pd.Series: %D values normalized to 0-1 range
    """
    # Calculate %K (already handles flattening internally)
    k_values = calculate_stochastic_k(data, window=k_window).values
    
    # Calculate %D (moving average of %K)
    d_values = np.zeros_like(k_values)
    
    for i in range(len(k_values)):
        if i < d_window:
            # Use available data for first few points
            d_values[i] = np.mean(k_values[0:i+1])
        else:
            # Use full window for later points
            d_values[i] = np.mean(k_values[i-d_window+1:i+1])
    
    # Replace NaNs with 0.5 (neutral value)
    result = np.nan_to_num(d_values, nan=0.5)
    return pd.Series(result, index=data.index) 