"""
Seasonal Features Module

Contains feature generators for date/time and seasonal patterns.
"""
import numpy as np
import pandas as pd
from typing import Optional
import calendar
from datetime import datetime, timedelta

from ..registry import FeatureRegistry


@FeatureRegistry.register(name="day_of_week", category="seasonal")
def calculate_day_of_week(data: pd.DataFrame) -> np.ndarray:
    """
    Calculate the day of week as a normalized feature.
    
    Args:
        data (pd.DataFrame): OHLCV data
        
    Returns:
        np.ndarray: Day of week values (0=Monday, 1=Friday, normalized to 0-1)
    """
    # Extract day of week (0=Monday, 6=Sunday)
    day_of_week = pd.Series(data.index).dt.dayofweek
    
    # Normalize to 0-1 range (assuming trading days are 0-4, Monday to Friday)
    normalized = day_of_week / 4.0
    
    return normalized.values


@FeatureRegistry.register(name="month", category="seasonal")
def calculate_month(data: pd.DataFrame) -> np.ndarray:
    """
    Calculate the month as a normalized feature.
    
    Args:
        data (pd.DataFrame): OHLCV data
        
    Returns:
        np.ndarray: Month values (normalized to 0-1)
    """
    # Extract month (1-12)
    month = pd.Series(data.index).dt.month
    
    # Normalize to 0-1 range
    normalized = (month - 1) / 11.0
    
    return normalized.values


@FeatureRegistry.register(name="quarter", category="seasonal")
def calculate_quarter(data: pd.DataFrame) -> np.ndarray:
    """
    Calculate the quarter as a one-hot encoded feature.
    
    Args:
        data (pd.DataFrame): OHLCV data
        
    Returns:
        np.ndarray: Quarter values (0-1)
    """
    # Extract quarter (1-4)
    quarter = pd.Series(data.index).dt.quarter
    
    # Normalize to 0-1 range
    normalized = (quarter - 1) / 3.0
    
    return normalized.values


@FeatureRegistry.register(name="day_of_month", category="seasonal")
def calculate_day_of_month(data: pd.DataFrame) -> np.ndarray:
    """
    Calculate the day of month as a normalized feature.
    
    Args:
        data (pd.DataFrame): OHLCV data
        
    Returns:
        np.ndarray: Day of month values (normalized to 0-1)
    """
    # Extract day of month (1-31)
    day = pd.Series(data.index).dt.day
    
    # Get the last day of each month
    last_day = pd.Series(data.index).apply(lambda x: calendar.monthrange(x.year, x.month)[1])
    
    # Normalize to 0-1 range based on the last day of the month
    normalized = (day - 1) / (last_day - 1)
    
    return normalized.values


@FeatureRegistry.register(name="weekday_indicator", category="seasonal")
def calculate_weekday_indicator(data: pd.DataFrame) -> np.ndarray:
    """
    Calculate a binary indicator for whether it's a weekday (Monday-Thursday) vs Friday.
    
    Args:
        data (pd.DataFrame): OHLCV data
        
    Returns:
        np.ndarray: Binary indicator (1=Mon-Thu, 0=Fri)
    """
    # Extract day of week (0=Monday, 6=Sunday)
    day_of_week = pd.Series(data.index).dt.dayofweek
    
    # Create indicator (1 for Mon-Thu, 0 for Fri)
    weekday_indicator = (day_of_week < 4).astype(float)
    
    return weekday_indicator.values


@FeatureRegistry.register(name="month_start", category="seasonal")
def calculate_month_start(data: pd.DataFrame, days: int = 5) -> np.ndarray:
    """
    Calculate a month start indicator (higher at beginning of month, tapering to 0).
    
    Args:
        data (pd.DataFrame): OHLCV data
        days (int): Number of days at the start of the month to consider
        
    Returns:
        np.ndarray: Month start indicator (0-1)
    """
    # Extract day of month (1-31)
    day = pd.Series(data.index).dt.day
    
    # Calculate indicator (1 on day 1, decreasing to 0 by specified days)
    indicator = np.maximum(0, 1 - (day - 1) / days)
    
    return indicator.values


@FeatureRegistry.register(name="month_end", category="seasonal")
def calculate_month_end(data: pd.DataFrame, days: int = 5) -> np.ndarray:
    """
    Calculate a month end indicator (higher at end of month, tapering from 0).
    
    Args:
        data (pd.DataFrame): OHLCV data
        days (int): Number of days at the end of the month to consider
        
    Returns:
        np.ndarray: Month end indicator (0-1)
    """
    # Extract day of month and last day of each month
    day = pd.Series(data.index).dt.day
    last_day = pd.Series(data.index).apply(lambda x: calendar.monthrange(x.year, x.month)[1])
    
    # Calculate indicator (1 on last day, increasing from 0 over specified days)
    days_to_end = last_day - day
    indicator = np.maximum(0, 1 - days_to_end / days)
    
    return indicator.values 