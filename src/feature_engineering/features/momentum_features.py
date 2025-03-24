"""
Momentum Features Module

Contains feature generators for momentum-based indicators.
"""
import numpy as np
import pandas as pd
from typing import Optional

from ..registry import FeatureRegistry


@FeatureRegistry.register(name="rsi_14", category="momentum")
def calculate_rsi_14(data: pd.DataFrame) -> np.ndarray:
    """
    Calculate the 14-day Relative Strength Index (RSI).
    
    Args:
        data (pd.DataFrame): OHLCV data
        
    Returns:
        np.ndarray: RSI values (0-100 range, normalized to 0-1)
    """
    close = data['Close'].values
    period = 14
    
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
    
    return np.nan_to_num(rsi_normalized, nan=0.5)  # Default to middle value if NaN


@FeatureRegistry.register(name="rsi_2", category="momentum")
def calculate_rsi_2(data: pd.DataFrame) -> np.ndarray:
    """
    Calculate the 2-day Relative Strength Index (RSI).
    
    Args:
        data (pd.DataFrame): OHLCV data
        
    Returns:
        np.ndarray: RSI values (0-100 range, normalized to 0-1)
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
    
    return np.nan_to_num(rsi_normalized, nan=0.5)  # Default to middle value if NaN


@FeatureRegistry.register(name="macd", category="momentum")
def calculate_macd(data: pd.DataFrame) -> np.ndarray:
    """
    Calculate the MACD (Moving Average Convergence Divergence) line.
    
    Args:
        data (pd.DataFrame): OHLCV data
        
    Returns:
        np.ndarray: MACD line values (normalized by price)
    """
    close = data['Close'].values
    
    # Calculate EMAs
    ema12 = pd.Series(close).ewm(span=12, adjust=False).mean()
    ema26 = pd.Series(close).ewm(span=26, adjust=False).mean()
    
    # Calculate MACD line
    macd_line = ema12 - ema26
    
    # Normalize by price
    macd_normalized = macd_line / np.maximum(close, 1e-8)
    
    return np.nan_to_num(macd_normalized.values, nan=0.0, posinf=0.0, neginf=0.0)


@FeatureRegistry.register(name="macd_signal", category="momentum")
def calculate_macd_signal(data: pd.DataFrame) -> np.ndarray:
    """
    Calculate the MACD signal line (9-day EMA of MACD line).
    
    Args:
        data (pd.DataFrame): OHLCV data
        
    Returns:
        np.ndarray: MACD signal line values (normalized by price)
    """
    close = data['Close'].values
    
    # Calculate EMAs
    ema12 = pd.Series(close).ewm(span=12, adjust=False).mean()
    ema26 = pd.Series(close).ewm(span=26, adjust=False).mean()
    
    # Calculate MACD line
    macd_line = ema12 - ema26
    
    # Calculate signal line (9-day EMA of MACD line)
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    
    # Normalize by price
    signal_normalized = signal_line / np.maximum(close, 1e-8)
    
    return np.nan_to_num(signal_normalized.values, nan=0.0, posinf=0.0, neginf=0.0)


@FeatureRegistry.register(name="macd_histogram", category="momentum")
def calculate_macd_histogram(data: pd.DataFrame) -> np.ndarray:
    """
    Calculate the MACD histogram (MACD line - Signal line).
    
    Args:
        data (pd.DataFrame): OHLCV data
        
    Returns:
        np.ndarray: MACD histogram values (normalized by price)
    """
    close = data['Close'].values
    
    # Calculate EMAs
    ema12 = pd.Series(close).ewm(span=12, adjust=False).mean()
    ema26 = pd.Series(close).ewm(span=26, adjust=False).mean()
    
    # Calculate MACD line
    macd_line = ema12 - ema26
    
    # Calculate signal line (9-day EMA of MACD line)
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    
    # Calculate histogram
    histogram = macd_line - signal_line
    
    # Normalize by price
    histogram_normalized = histogram / np.maximum(close, 1e-8)
    
    return np.nan_to_num(histogram_normalized.values, nan=0.0, posinf=0.0, neginf=0.0)


@FeatureRegistry.register(name="momentum_5", category="momentum")
def calculate_momentum_5(data: pd.DataFrame) -> np.ndarray:
    """
    Calculate the 5-day momentum indicator.
    
    Args:
        data (pd.DataFrame): OHLCV data
        
    Returns:
        np.ndarray: Momentum values
    """
    close = data['Close'].values
    period = 5
    
    # Calculate momentum (current price / price n periods ago - 1)
    momentum = pd.Series(close).pct_change(periods=period).fillna(0).values
    
    return np.nan_to_num(momentum, nan=0.0, posinf=0.0, neginf=0.0)


@FeatureRegistry.register(name="stoch_k", category="momentum")
def calculate_stochastic_k(data: pd.DataFrame, window: int = 14) -> np.ndarray:
    """
    Calculate the Stochastic Oscillator %K.
    
    Args:
        data (pd.DataFrame): OHLCV data
        window (int): Window size for calculation
        
    Returns:
        np.ndarray: Stochastic %K values (0-1 range)
    """
    high = data['High'].values
    low = data['Low'].values
    close = data['Close'].values
    
    # Convert to Series for rolling operations
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # Calculate highest high and lowest low over window
    highest_high = high_s.rolling(window=window).max().fillna(high_s)
    lowest_low = low_s.rolling(window=window).min().fillna(low_s)
    
    # Calculate %K
    k = 100 * (close_s - lowest_low) / np.maximum(highest_high - lowest_low, 1e-8)
    
    # Normalize to 0-1 range
    k_normalized = k / 100.0
    
    return np.nan_to_num(k_normalized.values, nan=0.5)  # Default to middle if NaN 