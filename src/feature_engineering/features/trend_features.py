"""
Trend Features Module

Contains feature generators for trend-based metrics such as moving averages.
"""
import numpy as np
import pandas as pd
from typing import Optional

from ..registry import FeatureRegistry


@FeatureRegistry.register(name="sma_5", category="trend")
def calculate_sma_5(data: pd.DataFrame) -> np.ndarray:
    """
    Calculate the 5-day Simple Moving Average ratio to close price.
    
    Args:
        data (pd.DataFrame): OHLCV data
        
    Returns:
        np.ndarray: SMA5 to close price ratio
    """
    close = data['Close'].values
    window = 5
    
    # Calculate the SMA
    sma = pd.Series(close).rolling(window=window).mean().fillna(method='bfill').values
    
    # Calculate ratio to current price
    ratio = sma / np.maximum(close, 1e-8)
    
    return np.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=1.0)


@FeatureRegistry.register(name="sma_10", category="trend")
def calculate_sma_10(data: pd.DataFrame) -> np.ndarray:
    """
    Calculate the 10-day Simple Moving Average ratio to close price.
    
    Args:
        data (pd.DataFrame): OHLCV data
        
    Returns:
        np.ndarray: SMA10 to close price ratio
    """
    close = data['Close'].values
    window = 10
    
    # Calculate the SMA
    sma = pd.Series(close).rolling(window=window).mean().fillna(method='bfill').values
    
    # Calculate ratio to current price
    ratio = sma / np.maximum(close, 1e-8)
    
    return np.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=1.0)


@FeatureRegistry.register(name="sma_20", category="trend")
def calculate_sma_20(data: pd.DataFrame) -> np.ndarray:
    """
    Calculate the 20-day Simple Moving Average ratio to close price.
    
    Args:
        data (pd.DataFrame): OHLCV data
        
    Returns:
        np.ndarray: SMA20 to close price ratio
    """
    close = data['Close'].values
    window = 20
    
    # Calculate the SMA
    sma = pd.Series(close).rolling(window=window).mean().fillna(method='bfill').values
    
    # Calculate ratio to current price
    ratio = sma / np.maximum(close, 1e-8)
    
    return np.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=1.0)


@FeatureRegistry.register(name="sma_50", category="trend")
def calculate_sma_50(data: pd.DataFrame) -> np.ndarray:
    """
    Calculate the 50-day Simple Moving Average ratio to close price.
    
    Args:
        data (pd.DataFrame): OHLCV data
        
    Returns:
        np.ndarray: SMA50 to close price ratio
    """
    close = data['Close'].values
    window = 50
    
    # Calculate the SMA
    sma = pd.Series(close).rolling(window=window).mean().fillna(method='bfill').values
    
    # Calculate ratio to current price
    ratio = sma / np.maximum(close, 1e-8)
    
    return np.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=1.0)


@FeatureRegistry.register(name="sma_200", category="trend")
def calculate_sma_200(data: pd.DataFrame) -> np.ndarray:
    """
    Calculate the 200-day Simple Moving Average ratio to close price.
    
    Args:
        data (pd.DataFrame): OHLCV data
        
    Returns:
        np.ndarray: SMA200 to close price ratio
    """
    close = data['Close'].values
    window = 200
    
    # Calculate the SMA (use bfill to avoid NaNs at the beginning)
    sma = pd.Series(close).rolling(window=min(window, len(close))).mean().fillna(method='bfill').values
    
    # Calculate ratio to current price
    ratio = sma / np.maximum(close, 1e-8)
    
    return np.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=1.0)


@FeatureRegistry.register(name="ema_12", category="trend")
def calculate_ema_12(data: pd.DataFrame) -> np.ndarray:
    """
    Calculate the 12-day Exponential Moving Average ratio to close price.
    
    Args:
        data (pd.DataFrame): OHLCV data
        
    Returns:
        np.ndarray: EMA12 to close price ratio
    """
    close = data['Close'].values
    window = 12
    
    # Calculate the EMA
    ema = pd.Series(close).ewm(span=window, adjust=False).mean().values
    
    # Calculate ratio to current price
    ratio = ema / np.maximum(close, 1e-8)
    
    return np.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=1.0)


@FeatureRegistry.register(name="ema_26", category="trend")
def calculate_ema_26(data: pd.DataFrame) -> np.ndarray:
    """
    Calculate the 26-day Exponential Moving Average ratio to close price.
    
    Args:
        data (pd.DataFrame): OHLCV data
        
    Returns:
        np.ndarray: EMA26 to close price ratio
    """
    close = data['Close'].values
    window = 26
    
    # Calculate the EMA
    ema = pd.Series(close).ewm(span=window, adjust=False).mean().values
    
    # Calculate ratio to current price
    ratio = ema / np.maximum(close, 1e-8)
    
    return np.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=1.0)


@FeatureRegistry.register(name="ma_crossover", category="trend")
def calculate_ma_crossover(data: pd.DataFrame, fast_window: int = 5, slow_window: int = 20) -> np.ndarray:
    """
    Calculate a moving average crossover signal.
    
    Args:
        data (pd.DataFrame): OHLCV data
        fast_window (int): Window for fast moving average
        slow_window (int): Window for slow moving average
        
    Returns:
        np.ndarray: Crossover signal (-1 to 1 range)
    """
    close = data['Close'].values
    
    # Calculate fast and slow MAs
    fast_ma = pd.Series(close).rolling(window=fast_window).mean().fillna(method='bfill').values
    slow_ma = pd.Series(close).rolling(window=slow_window).mean().fillna(method='bfill').values
    
    # Calculate difference between MAs (normalized by price)
    difference = (fast_ma - slow_ma) / np.maximum(close, 1e-8)
    
    # Scale to -1 to 1 range (sigmoid-like transformation)
    scaled_diff = 2 / (1 + np.exp(-10 * difference)) - 1
    
    return np.nan_to_num(scaled_diff, nan=0.0) 