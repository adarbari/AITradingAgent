"""
Price Features Module

Contains feature generators for price-based features.
"""
import numpy as np
import pandas as pd
from typing import Optional

from ..registry import FeatureRegistry


@FeatureRegistry.register(name="price_change", category="price")
def calculate_price_change(data: pd.DataFrame) -> pd.Series:
    """
    Calculate the percentage change in price from the previous day.
    
    Args:
        data (pd.DataFrame): OHLCV data
        
    Returns:
        pd.Series: Daily price changes
    """
    # Ensure we're working with a 1D array
    close_prices = data['Close'].values
    if len(close_prices.shape) > 1:
        close_prices = close_prices.flatten()
        
    # Calculate as (current close - previous close) / previous close
    price_changes = np.zeros_like(close_prices, dtype=float)
    price_changes[1:] = (close_prices[1:] - close_prices[:-1]) / close_prices[:-1]
    
    # Replace NaNs, infinities with zeros
    result = np.nan_to_num(price_changes, nan=0.0, posinf=0.0, neginf=0.0)
    return pd.Series(result, index=data.index)


@FeatureRegistry.register(name="high_low_range", category="price")
def calculate_high_low_range(data: pd.DataFrame) -> pd.Series:
    """
    Calculate the high-low range as a percentage of closing price.
    
    Args:
        data (pd.DataFrame): OHLCV data
        
    Returns:
        pd.Series: High-low range values
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
        
    # Calculate high-low range
    result = (high - low) / np.maximum(close, 1e-8)
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return pd.Series(result, index=data.index)


@FeatureRegistry.register(name="gap", category="price")
def calculate_gap(data: pd.DataFrame) -> pd.Series:
    """
    Calculate the overnight gap (open price vs previous close).
    
    Args:
        data (pd.DataFrame): OHLCV data
        
    Returns:
        pd.Series: Overnight gap values
    """
    # Ensure we're working with 1D arrays
    open_prices = data['Open'].values
    close_prices = data['Close'].values
    
    if len(open_prices.shape) > 1:
        open_prices = open_prices.flatten()
    if len(close_prices.shape) > 1:
        close_prices = close_prices.flatten()
    
    shifted_close = np.roll(close_prices, 1)
    shifted_close[0] = open_prices[0]  # First value has no previous close
    
    gap = (open_prices - shifted_close) / np.maximum(shifted_close, 1e-8)
    result = np.nan_to_num(gap, nan=0.0, posinf=0.0, neginf=0.0)
    return pd.Series(result, index=data.index)


@FeatureRegistry.register(name="vwap_distance", category="price")
def calculate_vwap_distance(data: pd.DataFrame) -> pd.Series:
    """
    Calculate the distance of close price from Volume Weighted Average Price (VWAP).
    
    Args:
        data (pd.DataFrame): OHLCV data
        
    Returns:
        pd.Series: Distance from VWAP
    """
    # Ensure we're working with 1D arrays
    high = data['High'].values
    low = data['Low'].values
    close = data['Close'].values
    volume = data['Volume'].values
    
    if len(high.shape) > 1:
        high = high.flatten()
    if len(low.shape) > 1:
        low = low.flatten()
    if len(close.shape) > 1:
        close = close.flatten()
    if len(volume.shape) > 1:
        volume = volume.flatten()
    
    # Calculate typical price
    typical_price = (high + low + close) / 3
    
    # Calculate VWAP
    cum_tp_vol = np.cumsum(typical_price * volume)
    cum_vol = np.cumsum(volume)
    vwap = cum_tp_vol / np.maximum(cum_vol, 1e-8)
    
    # Calculate distance from VWAP
    distance = (close - vwap) / np.maximum(vwap, 1e-8)
    
    result = np.nan_to_num(distance, nan=0.0, posinf=0.0, neginf=0.0)
    return pd.Series(result, index=data.index)


@FeatureRegistry.register(name="price_dispersion", category="price")
def calculate_price_dispersion(data: pd.DataFrame, window: int = 5) -> pd.Series:
    """
    Calculate price dispersion (difference between high and low over a window).
    
    Args:
        data (pd.DataFrame): OHLCV data
        window (int): Window size for rolling calculation
        
    Returns:
        pd.Series: Price dispersion values
    """
    # Handle potential multi-dimensional data
    high = data['High']
    low = data['Low']
    close = data['Close']
    
    if isinstance(high, pd.Series) and high.values.ndim > 1:
        high = pd.Series(high.values.flatten(), index=high.index)
    if isinstance(low, pd.Series) and low.values.ndim > 1:
        low = pd.Series(low.values.flatten(), index=low.index)
    if isinstance(close, pd.Series) and close.values.ndim > 1:
        close = pd.Series(close.values.flatten(), index=close.index)
    
    rolling_high = high.rolling(window=window).max()
    rolling_low = low.rolling(window=window).min()
    
    dispersion = (rolling_high - rolling_low) / np.maximum(close, 1e-8)
    
    result = np.nan_to_num(dispersion.values, nan=0.0, posinf=0.0, neginf=0.0)
    return pd.Series(result, index=data.index) 