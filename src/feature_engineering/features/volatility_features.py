"""
Volatility Features Module

Contains feature generators for volatility-based indicators.
"""
import numpy as np
import pandas as pd
from typing import Optional

from ..registry import FeatureRegistry


@FeatureRegistry.register(name="volatility", category="volatility")
def calculate_volatility(data: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Calculate price volatility as standard deviation of returns.
    
    Args:
        data (pd.DataFrame): OHLCV data
        window (int): Window size for rolling calculation
        
    Returns:
        pd.Series: Volatility values
    """
    # Calculate daily returns
    close = data['Close'].values
    daily_returns = np.diff(close, prepend=close[0]) / np.maximum(close, 1e-8)
    
    # Calculate rolling standard deviation of returns
    returns_series = pd.Series(daily_returns, index=data.index)
    volatility = returns_series.rolling(window=window).std()
    
    # Fill NaNs with a reasonable default
    volatility = volatility.fillna(0.01)  # 1% volatility as default
    
    result = np.nan_to_num(volatility.values, nan=0.01)
    return pd.Series(result, index=data.index)


@FeatureRegistry.register(name="atr_14", category="volatility")
def calculate_atr(data: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR).
    
    Args:
        data (pd.DataFrame): OHLCV data
        window (int): Window size for ATR calculation
        
    Returns:
        pd.Series: ATR values
    """
    high = data['High'].values
    low = data['Low'].values
    close = data['Close'].values
    
    # Calculate true range
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    
    tr1 = high - low  # Current high - current low
    tr2 = np.abs(high - prev_close)  # Current high - previous close
    tr3 = np.abs(low - prev_close)  # Current low - previous close
    
    true_range = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Calculate ATR as moving average of true range
    tr_series = pd.Series(true_range, index=data.index)
    atr = tr_series.rolling(window=window).mean()
    
    # Normalize by current price
    atr_normalized = atr / np.maximum(close, 1e-8)
    
    # Fill NaNs with a reasonable default
    atr_normalized = atr_normalized.fillna(0.02)  # 2% ATR as default
    
    result = np.nan_to_num(atr_normalized.values, nan=0.02)
    return pd.Series(result, index=data.index)


@FeatureRegistry.register(name="bollinger_bandwidth", category="volatility")
def calculate_bollinger_bandwidth(data: pd.DataFrame, window: int = 20, 
                                 num_std: float = 2.0) -> pd.Series:
    """
    Calculate Bollinger Bandwidth (relative width of Bollinger Bands).
    
    Args:
        data (pd.DataFrame): OHLCV data
        window (int): Window size for Bollinger Bands calculation
        num_std (float): Number of standard deviations for band width
        
    Returns:
        pd.Series: Bollinger Bandwidth values
    """
    close = data['Close'].values
    
    # Calculate middle band (SMA)
    close_series = pd.Series(close, index=data.index)
    middle_band = close_series.rolling(window=window).mean()
    
    # Fill NaNs with current price
    middle_band = middle_band.fillna(close_series)
    
    # Calculate standard deviation
    std_dev = close_series.rolling(window=window).std()
    std_dev = std_dev.fillna(close_series * 0.01)  # Default to 1% of price
    
    # Calculate bandwidth: (upper band - lower band) / middle band
    bandwidth = (2 * num_std * std_dev) / np.maximum(middle_band, 1e-8)
    
    # Normalize to a reasonable range
    result = np.clip(bandwidth.values, 0, 0.5)
    result = np.nan_to_num(result, nan=0.05)  # Default to 5% bandwidth
    
    return pd.Series(result, index=data.index)


@FeatureRegistry.register(name="keltner_channel_width", category="volatility")
def calculate_keltner_width(data: pd.DataFrame, window: int = 20, atr_mult: float = 2.0) -> pd.Series:
    """
    Calculate the Keltner Channel width normalized by price.
    
    Args:
        data (pd.DataFrame): OHLCV data
        window (int): Window size for calculation
        atr_mult (float): ATR multiplier for channel width
        
    Returns:
        pd.Series: Normalized Keltner Channel width
    """
    high = data['High'].values
    low = data['Low'].values
    close = data['Close'].values
    
    # Calculate True Range
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    
    true_range = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Calculate ATR
    atr = pd.Series(true_range).rolling(window=window).mean().fillna(true_range).values
    
    # Calculate EMA
    ema = pd.Series(close).ewm(span=window, adjust=False).mean().values
    
    # Calculate channel width
    width = 2 * atr_mult * atr / np.maximum(ema, 1e-8)
    
    result = np.nan_to_num(width, nan=0.0)
    return pd.Series(result, index=data.index)


@FeatureRegistry.register(name="vix_proxy", category="volatility")
def calculate_vix_proxy(data: pd.DataFrame, window: int = 30) -> pd.Series:
    """
    Calculate a simple VIX proxy (implied volatility estimation) using historical volatility.
    
    Args:
        data (pd.DataFrame): OHLCV data
        window (int): Window size for volatility calculation
        
    Returns:
        pd.Series: VIX proxy values (normalized)
    """
    close_prices = data['Close'].values
    
    # Calculate log returns
    log_returns = np.diff(np.log(np.maximum(close_prices, 1e-8)), prepend=0)
    
    # Calculate rolling volatility
    rolling_std = pd.Series(log_returns).rolling(window=window).std().fillna(0).values
    
    # Annualize volatility (approx. 252 trading days in a year)
    annualized_vol = rolling_std * np.sqrt(252)
    
    # Normalize to 0-1 range (VIX of 30+ is considered high)
    normalized_vol = np.minimum(annualized_vol / 0.3, 1.0)
    
    result = np.nan_to_num(normalized_vol, nan=0.0)
    return pd.Series(result, index=data.index)


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