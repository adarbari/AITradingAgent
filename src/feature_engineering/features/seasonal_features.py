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
def calculate_day_of_week(data: pd.DataFrame) -> pd.Series:
    """
    Calculate the day of week as a normalized feature.
    
    Args:
        data (pd.DataFrame): OHLCV data
        
    Returns:
        pd.Series: Day of week values (0=Monday, 1=Friday, normalized to 0-1)
    """
    # Extract day of week (0=Monday, 6=Sunday)
    days = pd.Series(data.index).dt.dayofweek
    
    # Normalize to 0-1 range (assuming trading days are 0-4, Monday to Friday)
    # Ensure we don't exceed 1.0 by capping at 4 (Friday)
    capped_days = np.minimum(days, 4)  
    normalized = capped_days / 4.0
    
    result = np.nan_to_num(normalized.values, nan=0.0)
    return pd.Series(result, index=data.index)


@FeatureRegistry.register(name="month", category="seasonal")
def calculate_month(data: pd.DataFrame) -> pd.Series:
    """
    Calculate the month as a normalized feature.
    
    Args:
        data (pd.DataFrame): OHLCV data
        
    Returns:
        pd.Series: Month values (normalized to 0-1)
    """
    # Extract month (1-12)
    month = pd.Series(data.index).dt.month
    
    # Normalize to 0-1 range
    normalized = (month - 1) / 11.0
    
    result = np.nan_to_num(normalized.values, nan=0.0)
    return pd.Series(result, index=data.index)


@FeatureRegistry.register(name="quarter", category="seasonal")
def calculate_quarter(data: pd.DataFrame) -> pd.Series:
    """
    Calculate the quarter as a one-hot encoded feature.
    
    Args:
        data (pd.DataFrame): OHLCV data
        
    Returns:
        pd.Series: Quarter values (0-1)
    """
    # Extract quarter (1-4)
    quarter = pd.Series(data.index).dt.quarter
    
    # Normalize to 0-1 range
    normalized = (quarter - 1) / 3.0
    
    result = np.nan_to_num(normalized.values, nan=0.0)
    return pd.Series(result, index=data.index)


@FeatureRegistry.register(name="day_of_month", category="seasonal")
def calculate_day_of_month(data: pd.DataFrame) -> pd.Series:
    """
    Calculate the day of month as a normalized feature.
    
    Args:
        data (pd.DataFrame): OHLCV data
        
    Returns:
        pd.Series: Day of month values (normalized to 0-1)
    """
    # Extract day of month (1-31)
    day = pd.Series(data.index).dt.day
    
    # Get the last day of each month
    last_day = pd.Series(data.index).apply(lambda x: calendar.monthrange(x.year, x.month)[1])
    
    # Normalize to 0-1 range based on the last day of the month
    normalized = (day - 1) / (last_day - 1)
    
    result = np.nan_to_num(normalized.values, nan=0.0)
    return pd.Series(result, index=data.index)


@FeatureRegistry.register(name="weekday_indicator", category="seasonal")
def calculate_weekday_indicator(data: pd.DataFrame) -> pd.Series:
    """
    Calculate a binary indicator for whether it's a weekday (Monday-Thursday) vs Friday.
    
    Args:
        data (pd.DataFrame): OHLCV data
        
    Returns:
        pd.Series: Binary indicator (1=Mon-Thu, 0=Fri)
    """
    # Extract day of week (0=Monday, 6=Sunday)
    day_of_week = pd.Series(data.index).dt.dayofweek
    
    # Create indicator (1 for Mon-Thu, 0 for Fri)
    weekday_indicator = (day_of_week < 4).astype(float)
    
    result = np.nan_to_num(weekday_indicator.values, nan=0.0)
    return pd.Series(result, index=data.index)


@FeatureRegistry.register(name="month_start", category="seasonal")
def calculate_month_start(data: pd.DataFrame, days: int = 5) -> pd.Series:
    """
    Calculate a month start indicator (higher at beginning of month, tapering to 0).
    
    Args:
        data (pd.DataFrame): OHLCV data
        days (int): Number of days at the start of the month to consider
        
    Returns:
        pd.Series: Month start indicator (0-1)
    """
    # Extract day of month (1-31)
    day = pd.Series(data.index).dt.day
    
    # Calculate indicator (1 on day 1, decreasing to 0 by specified days)
    indicator = np.maximum(0, 1 - (day - 1) / days)
    
    result = np.nan_to_num(indicator.values, nan=0.0)
    return pd.Series(result, index=data.index)


@FeatureRegistry.register(name="month_end", category="seasonal")
def calculate_month_end(data: pd.DataFrame, days: int = 5) -> pd.Series:
    """
    Calculate a month end indicator (higher at end of month, tapering from 0).
    
    Args:
        data (pd.DataFrame): OHLCV data
        days (int): Number of days at the end of the month to consider
        
    Returns:
        pd.Series: Month end indicator (0-1)
    """
    # Extract day of month and last day of each month
    day = pd.Series(data.index).dt.day
    last_day = pd.Series(data.index).apply(lambda x: calendar.monthrange(x.year, x.month)[1])
    
    # Calculate indicator (1 on last day, increasing from 0 over specified days)
    days_to_end = last_day - day
    indicator = np.maximum(0, 1 - days_to_end / days)
    
    result = np.nan_to_num(indicator.values, nan=0.0)
    return pd.Series(result, index=data.index)


@FeatureRegistry.register(name="year_progress", category="seasonal")
def calculate_year_progress(data: pd.DataFrame) -> pd.Series:
    """
    Calculate a year progress feature (0 at beginning of year, 1 at end).
    
    Args:
        data (pd.DataFrame): OHLCV data
        
    Returns:
        pd.Series: Year progress values (0-1 range)
    """
    # Get index as datetime
    if isinstance(data.index, pd.DatetimeIndex):
        dates = data.index
    else:
        # If index is not datetime, try to find a Date column
        if 'Date' in data.columns:
            dates = pd.to_datetime(data['Date'])
        else:
            # If no Date column, create synthetic dates
            dates = pd.date_range(start='2020-01-01', periods=len(data), freq='B')
    
    # Calculate day of year
    day_of_year = dates.dayofyear
    
    # Get days in year (accounting for leap years)
    days_in_year = np.ones_like(day_of_year, dtype=float) * 365
    leap_years = dates.is_leap_year.values
    days_in_year[leap_years] = 366
    
    # Normalize to 0-1 range
    normalized = (day_of_year - 1) / (days_in_year - 1)
    
    result = np.nan_to_num(normalized, nan=0.0)
    return pd.Series(result, index=data.index) 