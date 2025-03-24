"""
Volume Features Module

Contains feature generators for volume-based indicators.
"""
import numpy as np
import pandas as pd
from typing import Optional

from ..registry import FeatureRegistry


@FeatureRegistry.register(name="volume_change", category="volume")
def calculate_volume_change(data: pd.DataFrame) -> np.ndarray:
    """
    Calculate the percentage change in volume from the previous day.
    
    Args:
        data (pd.DataFrame): OHLCV data
        
    Returns:
        np.ndarray: Volume change values
    """
    volume = np.maximum(data['Volume'].values, 1)  # Ensure no zeros
    volume_changes = np.diff(volume, prepend=volume[0]) / volume
    return np.nan_to_num(volume_changes, nan=0.0, posinf=0.0, neginf=0.0)


@FeatureRegistry.register(name="volume_sma_ratio", category="volume")
def calculate_volume_sma_ratio(data: pd.DataFrame, window: int = 20) -> np.ndarray:
    """
    Calculate the ratio of current volume to its moving average.
    
    Args:
        data (pd.DataFrame): OHLCV data
        window (int): Window size for moving average
        
    Returns:
        np.ndarray: Volume to SMA ratio
    """
    volume = data['Volume'].values
    volume_sma = pd.Series(volume).rolling(window=window).mean().fillna(volume).values
    ratio = volume / np.maximum(volume_sma, 1)
    
    # Cap extreme values
    ratio = np.clip(ratio, 0, 5)
    
    return np.nan_to_num(ratio, nan=1.0)


@FeatureRegistry.register(name="obv", category="volume")
def calculate_obv(data: pd.DataFrame) -> np.ndarray:
    """
    Calculate On-Balance Volume (OBV) normalized.
    
    Args:
        data (pd.DataFrame): OHLCV data
        
    Returns:
        np.ndarray: Normalized OBV values
    """
    close = data['Close'].values
    volume = data['Volume'].values
    
    # Calculate price direction
    direction = np.zeros_like(close)
    direction[1:] = np.sign(np.diff(close))
    
    # Calculate OBV
    obv = np.zeros_like(volume, dtype=float)
    for i in range(1, len(obv)):
        if direction[i] > 0:
            obv[i] = obv[i-1] + volume[i]
        elif direction[i] < 0:
            obv[i] = obv[i-1] - volume[i]
        else:
            obv[i] = obv[i-1]
    
    # Normalize OBV using a rolling window
    obv_series = pd.Series(obv)
    obv_min = obv_series.rolling(window=20).min().fillna(obv_series)
    obv_max = obv_series.rolling(window=20).max().fillna(obv_series)
    
    # Avoid division by zero
    obv_range = np.maximum(obv_max - obv_min, 1)
    
    # Normalize to -1 to 1 range
    obv_normalized = 2 * (obv_series - obv_min) / obv_range - 1
    
    return np.nan_to_num(obv_normalized.values, nan=0.0)


@FeatureRegistry.register(name="price_volume_trend", category="volume")
def calculate_pvt(data: pd.DataFrame) -> np.ndarray:
    """
    Calculate Price Volume Trend (PVT) indicator normalized.
    
    Args:
        data (pd.DataFrame): OHLCV data
        
    Returns:
        np.ndarray: Normalized PVT values
    """
    close = data['Close'].values
    volume = data['Volume'].values
    
    # Calculate percentage price change
    price_change = np.zeros_like(close, dtype=float)
    price_change[1:] = np.diff(close) / np.maximum(close[:-1], 1e-8)
    
    # Calculate PVT
    pvt = np.zeros_like(volume, dtype=float)
    for i in range(1, len(pvt)):
        pvt[i] = pvt[i-1] + volume[i] * price_change[i]
    
    # Normalize PVT
    pvt_series = pd.Series(pvt)
    pvt_std = pvt_series.rolling(window=20).std().fillna(1)
    pvt_mean = pvt_series.rolling(window=20).mean().fillna(0)
    
    # Z-score normalization
    pvt_normalized = (pvt_series - pvt_mean) / np.maximum(pvt_std, 1e-8)
    
    # Clip to reasonable range (-3 to 3)
    pvt_normalized = np.clip(pvt_normalized, -3, 3) / 3
    
    return np.nan_to_num(pvt_normalized.values, nan=0.0)


@FeatureRegistry.register(name="volume_price_confirm", category="volume")
def calculate_volume_price_confirm(data: pd.DataFrame, window: int = 5) -> np.ndarray:
    """
    Calculate if volume confirms price movement (positive when both price and volume increase).
    
    Args:
        data (pd.DataFrame): OHLCV data
        window (int): Window for moving average
        
    Returns:
        np.ndarray: Confirmation signal (-1 to 1 range)
    """
    close = data['Close'].values
    volume = data['Volume'].values
    
    # Calculate price and volume changes
    price_change = pd.Series(close).pct_change().fillna(0)
    volume_change = pd.Series(volume).pct_change().fillna(0)
    
    # Smooth with moving average
    smooth_price_change = price_change.rolling(window=window).mean().fillna(price_change)
    smooth_volume_change = volume_change.rolling(window=window).mean().fillna(volume_change)
    
    # Normalize to -1 to 1 range
    norm_price_change = np.clip(smooth_price_change / 0.05, -1, 1)  # 5% move = full signal
    norm_volume_change = np.clip(smooth_volume_change / 0.2, -1, 1)  # 20% move = full signal
    
    # Calculate confirmation (product of price and volume direction)
    # A high positive value means strong confirmation (price up + volume up or price down + volume down)
    # A high negative value means divergence
    confirmation = norm_price_change * np.sign(norm_volume_change) * np.abs(norm_volume_change)
    
    return np.nan_to_num(confirmation.values, nan=0.0) 