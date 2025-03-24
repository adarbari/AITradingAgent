"""
Volatility Features Module

Contains feature generators for volatility-based metrics.
"""
import numpy as np
import pandas as pd
from typing import Optional

from ..registry import FeatureRegistry


@FeatureRegistry.register(name="volatility", category="volatility")
def calculate_volatility(data: pd.DataFrame, window: int = 5) -> np.ndarray:
    """
    Calculate the rolling standard deviation of returns (volatility).
    
    Args:
        data (pd.DataFrame): OHLCV data
        window (int): Window size for volatility calculation
        
    Returns:
        np.ndarray: Volatility values
    """
    close_prices = data['Close'].values
    returns = np.diff(close_prices, prepend=close_prices[0]) / np.maximum(close_prices, 1e-8)
    
    # Calculate rolling standard deviation
    rolling_std = pd.Series(returns).rolling(window=window).std().fillna(0).values
    
    return np.nan_to_num(rolling_std, nan=0.0)


@FeatureRegistry.register(name="bollinger_bandwidth", category="volatility")
def calculate_bollinger_bandwidth(data: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> np.ndarray:
    """
    Calculate Bollinger Bandwidth, which measures relative width of Bollinger Bands.
    
    Args:
        data (pd.DataFrame): OHLCV data
        window (int): Window size for moving average
        num_std (float): Number of standard deviations for the bands
        
    Returns:
        np.ndarray: Bollinger Bandwidth values
    """
    # Calculate the rolling mean and standard deviation
    rolling_mean = data['Close'].rolling(window=window).mean()
    rolling_std = data['Close'].rolling(window=window).std()
    
    # Calculate upper and lower bands
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    
    # Calculate bandwidth
    bandwidth = (upper_band - lower_band) / np.maximum(rolling_mean, 1e-8)
    
    return np.nan_to_num(bandwidth.values, nan=0.0, posinf=0.0, neginf=0.0)


@FeatureRegistry.register(name="bollinger_position", category="volatility")
def calculate_bollinger_position(data: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> np.ndarray:
    """
    Calculate the position of price within the Bollinger Bands (0 to 1).
    
    Args:
        data (pd.DataFrame): OHLCV data
        window (int): Window size for moving average
        num_std (float): Number of standard deviations for the bands
        
    Returns:
        np.ndarray: Position within Bollinger Bands (0=lower band, 0.5=middle, 1=upper band)
    """
    # Calculate the rolling mean and standard deviation
    rolling_mean = data['Close'].rolling(window=window).mean()
    rolling_std = data['Close'].rolling(window=window).std()
    
    # Calculate upper and lower bands
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    
    # Calculate position within bands (scaled 0 to 1)
    position = (data['Close'] - lower_band) / np.maximum(upper_band - lower_band, 1e-8)
    
    # Clip values to [0, 1] range
    position = np.clip(position, 0, 1)
    
    return np.nan_to_num(position.values, nan=0.5)  # Default to middle if NaN


@FeatureRegistry.register(name="atr_14", category="volatility")
def calculate_atr(data: pd.DataFrame, window: int = 14) -> np.ndarray:
    """
    Calculate the Average True Range (ATR), a measure of volatility.
    
    Args:
        data (pd.DataFrame): OHLCV data
        window (int): Window size for ATR calculation
        
    Returns:
        np.ndarray: ATR values scaled by price
    """
    high = data['High'].values
    low = data['Low'].values
    close = data['Close'].values
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]  # Set first value to avoid using the rolled last value
    
    # Calculate true range
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    
    # True range is the maximum of the three
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Calculate ATR using simple moving average
    atr = pd.Series(tr).rolling(window=window).mean().fillna(tr).values
    
    # Scale by price to get relative ATR
    atr_scaled = atr / np.maximum(close, 1e-8)
    
    return np.nan_to_num(atr_scaled, nan=0.0)


@FeatureRegistry.register(name="atr_ratio", category="volatility")
def calculate_atr_ratio(data: pd.DataFrame, window1: int = 5, window2: int = 20) -> np.ndarray:
    """
    Calculate the ratio of short-term ATR to long-term ATR.
    
    Args:
        data (pd.DataFrame): OHLCV data
        window1 (int): Window size for short-term ATR
        window2 (int): Window size for long-term ATR
        
    Returns:
        np.ndarray: Ratio of short-term to long-term ATR
    """
    # Calculate short-term ATR
    atr_short = calculate_atr(data, window=window1)
    
    # Calculate long-term ATR
    atr_long = calculate_atr(data, window=window2)
    
    # Calculate ratio
    ratio = atr_short / np.maximum(atr_long, 1e-8)
    
    return np.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=1.0) 