"""
Utilities for feature preparation and data handling.
"""
import numpy as np
import pandas as pd
import warnings
import sys
import os

# Add parent directory to path if necessary
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import feature engineering module
from src.feature_engineering import process_features, FeatureRegistry
from src.feature_engineering.pipeline import FeaturePipeline
from src.feature_engineering.cache import FeatureCache
from src.data.yahoo_data_fetcher import YahooDataFetcher

# Import all feature categories to ensure they are registered
from src.feature_engineering.features import (
    price_features, 
    trend_features, 
    volatility_features, 
    momentum_features, 
    volume_features,
    seasonal_features
)


def prepare_features_from_indicators(features_df, expected_feature_count=21, verbose=False):
    """
    Prepare features from a DataFrame that already contains technical indicators.
    Handles column conversion, NaN values, normalization, and feature count matching.
    
    Args:
        features_df (pd.DataFrame): DataFrame containing technical indicators
        expected_feature_count (int): Expected number of features
        verbose (bool): Whether to print information about the transformation
        
    Returns:
        pd.DataFrame: Processed features DataFrame
    """
    # Create a feature pipeline manually to handle the pre-computed features
    pipeline = FeaturePipeline(
        feature_list=list(features_df.columns), 
        feature_count=expected_feature_count,
        verbose=verbose
    )
    
    # Skip feature generation and just do normalization and cleanup
    features = features_df.copy()
    
    if pipeline.normalize:
        features = pipeline._normalize_features(features)
    
    # Handle feature count
    features = pipeline._handle_feature_count(features)
    
    # Final checks and cleanup
    features = pipeline._final_cleanup(features)
    
    return features


def prepare_robust_features(data, feature_count=21, verbose=False):
    """
    Prepare features for the trading agent with robust error handling.
    
    Args:
        data (pd.DataFrame): Raw price data with OHLCV columns
        feature_count (int): Expected number of features
        verbose (bool): Whether to print information about the transformation
        
    Returns:
        np.array: Processed features with shape (n_samples, feature_count)
    """
    if verbose:
        print("Using feature engineering pipeline")
    
    # Use standard feature set
    pipeline = FeaturePipeline(
        feature_list=None,  # Use default from config
        feature_count=feature_count,
        verbose=verbose
    )
    
    # Process features using the pipeline
    features_df = pipeline.process(data)
    
    # Convert to numpy array if needed
    if isinstance(features_df, pd.DataFrame):
        return features_df.values
    return features_df


def get_data(symbol, start_date, end_date, data_source='yahoo', synthetic_params=None):
    """
    Unified function to fetch financial data from different sources.
    Handles Yahoo Finance, CSV files, and synthetic data generation.
    
    Args:
        symbol (str): Stock symbol or identifier
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        data_source (str): Source of data ('yahoo' or 'synthetic')
        synthetic_params (dict): Parameters for synthetic data generation
        
    Returns:
        pd.DataFrame: OHLCV data
    """
    if data_source == 'yahoo':
        try:
            # Fetch data from Yahoo Finance using the data fetcher
            data_fetcher = YahooDataFetcher()
            data = data_fetcher.fetch_data_simple(symbol, start_date, end_date)
                
            if data is None or data.empty:
                print(f"Error: No data returned for {symbol}")
                print("Falling back to synthetic data generation")
                data_source = 'synthetic'  # Fall back to synthetic data
            else:
                return data
                
        except Exception as e:
            print(f"Error fetching data from Yahoo Finance: {e}")
            print("Falling back to synthetic data generation")
            data_source = 'synthetic'  # Fall back to synthetic data
    
    if data_source == 'synthetic':
        # Generate synthetic data
        return _generate_synthetic_data(
            symbol, 
            start_date, 
            end_date, 
            params=synthetic_params
        )
    
    # If we get here, we didn't get any data
    print(f"Warning: Could not fetch data for {symbol} from {start_date} to {end_date}")
    return None


def _generate_synthetic_data(symbol, start_date, end_date, params=None):
    """
    Generate synthetic price data for testing when real data is unavailable.
    
    Args:
        symbol (str): Symbol name (used only for reference)
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        params (dict): Parameters for data generation
            - initial_price: Starting price
            - volatility: Daily volatility
            - drift: Daily drift
    
    Returns:
        pd.DataFrame: Synthetic OHLCV data
    """
    # Default parameters
    if params is None:
        params = {
            "initial_price": 100.0,
            "volatility": 0.01,
            "drift": 0.0001
        }
    
    initial_price = params.get("initial_price", 100.0)
    volatility = params.get("volatility", 0.01)
    drift = params.get("drift", 0.0001)
    
    # Create date range
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    date_range = pd.date_range(start=start, end=end, freq='B')  # Business days
    
    # Generate synthetic prices
    np.random.seed(42)  # For reproducibility
    returns = np.random.normal(drift, volatility, size=len(date_range))
    prices = initial_price * (1 + returns).cumprod()
    
    # Create synthetic OHLCV data
    data = pd.DataFrame(index=date_range)
    data['Close'] = prices
    
    # Simulate daily price action
    daily_range_factor = 0.01
    data['High'] = data['Close'] * (1 + np.random.uniform(0, daily_range_factor, size=len(date_range)))
    data['Low'] = data['Close'] * (1 - np.random.uniform(0, daily_range_factor, size=len(date_range)))
    data['Open'] = data['Low'] + np.random.uniform(0, 1, size=len(date_range)) * (data['High'] - data['Low'])
    
    # Simulate volume
    volume_base = 1000000
    volume_volatility = 0.3
    data['Volume'] = volume_base * np.exp(np.random.normal(0, volume_volatility, size=len(date_range)))
    
    # Add columns to match Yahoo Finance format
    data['Adj Close'] = data['Close']
    data['Dividends'] = 0
    data['Stock Splits'] = 0
    
    # Add the date as a column too
    data['Date'] = data.index
    
    # Add some technical indicators
    # SMA
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    
    # Add RSI (simplified)
    delta = data['Close'].diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = -delta.clip(upper=0).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)  # Replace 0 with NaN to avoid division by zero
    data['RSI_14'] = 100 - (100 / (1 + rs))
    
    # Fill NAs
    data = data.fillna(method='bfill').fillna(method='ffill')
    
    print(f"Generated {len(data)} days of synthetic data from {start_date} to {end_date}")
    
    return data 