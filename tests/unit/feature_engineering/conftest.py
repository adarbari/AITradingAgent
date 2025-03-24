"""
Fixtures for feature engineering tests
"""
import os
import sys
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tempfile
import shutil

# Import feature engineering components
from src.feature_engineering import FeatureRegistry, FEATURE_CONFIGS
from src.feature_engineering.pipeline import FeaturePipeline
from src.feature_engineering.cache import FeatureCache


@pytest.fixture
def sample_ohlcv_data():
    """
    Generate a sample OHLCV dataset with realistic price patterns.
    """
    # Create date range
    dates = pd.date_range(start='2022-01-01', end='2022-03-01')
    n = len(dates)
    
    # Generate synthetic price data with trend, seasonality, and noise
    base_price = 100
    trend = np.linspace(0, 20, n)  # Upward trend
    seasonality = 5 * np.sin(np.linspace(0, 4 * np.pi, n))  # Seasonal pattern
    noise = np.random.normal(0, 3, n)  # Random noise
    
    # Combine components to create price series
    close_prices = base_price + trend + seasonality + noise
    
    # Create related price series with realistic relationships
    open_prices = close_prices - np.random.normal(0, 2, n)
    high_prices = np.maximum(close_prices, open_prices) + np.random.normal(1, 1, n)
    low_prices = np.minimum(close_prices, open_prices) - np.random.normal(1, 1, n)
    
    # Generate volume with some correlation to price movements
    price_diff = np.diff(close_prices, prepend=close_prices[0])
    volume_base = np.random.normal(1000000, 200000, n)
    volume = volume_base * (1 + 0.2 * np.abs(price_diff) / np.mean(np.abs(price_diff)))
    
    # Create the DataFrame
    data = pd.DataFrame({
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volume
    }, index=dates)
    
    return data


@pytest.fixture
def empty_ohlcv_data():
    """
    Generate an empty OHLCV dataset for testing edge cases.
    """
    return pd.DataFrame({
        'Open': [],
        'High': [],
        'Low': [],
        'Close': [],
        'Volume': []
    })


@pytest.fixture
def small_ohlcv_data():
    """
    Generate a very small OHLCV dataset (just a few days) for testing edge cases.
    """
    dates = pd.date_range(start='2022-01-01', periods=3)
    data = pd.DataFrame({
        'Open': [100, 101, 102],
        'High': [105, 106, 107],
        'Low': [95, 96, 97],
        'Close': [101, 102, 103],
        'Volume': [1000000, 1100000, 900000]
    }, index=dates)
    return data


@pytest.fixture
def abnormal_ohlcv_data():
    """
    Generate OHLCV data with abnormal values (NaN, zero, negative) for testing robustness.
    """
    dates = pd.date_range(start='2022-01-01', periods=10)
    data = pd.DataFrame({
        'Open': [100, 101, np.nan, 103, 104, 105, 106, 107, 108, 109],
        'High': [105, 106, 107, np.nan, 109, 110, 111, 112, 113, 114],
        'Low': [95, 96, 97, 98, np.nan, 100, 101, 102, 103, 104],
        'Close': [101, 102, 103, 104, 105, np.nan, 107, 108, 109, 110],
        'Volume': [1000000, 0, 1200000, 1300000, 1400000, 1500000, -100, 1700000, 1800000, np.nan]
    }, index=dates)
    return data


@pytest.fixture
def feature_registry():
    """
    Return the FeatureRegistry for testing.
    """
    return FeatureRegistry


@pytest.fixture
def temp_cache_dir():
    """
    Create a temporary directory for cache testing and clean it up afterwards.
    """
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def feature_cache(temp_cache_dir):
    """
    Create a FeatureCache instance for testing.
    """
    return FeatureCache(cache_dir=temp_cache_dir, enable_cache=True, verbose=False)


@pytest.fixture
def feature_pipeline_minimal():
    """
    Create a minimal feature pipeline for testing.
    """
    feature_list = ['price_change', 'volatility', 'volume_change']
    return FeaturePipeline(feature_list=feature_list, feature_count=3, verbose=False)


@pytest.fixture
def feature_pipeline_standard():
    """
    Create a standard feature pipeline for testing.
    """
    if 'standard' in FEATURE_CONFIGS:
        feature_list = FEATURE_CONFIGS['standard']
    else:
        feature_list = ['price_change', 'volatility', 'volume_change', 'rsi_14', 'sma_20']
    
    return FeaturePipeline(feature_list=feature_list, feature_count=len(feature_list), verbose=False)


@pytest.fixture
def custom_feature():
    """
    Register a custom feature for testing and remove it after the test.
    """
    # Function to register
    @FeatureRegistry.register(name="test_custom_feature", category="test")
    def calculate_test_feature(data):
        """Test feature for unit testing"""
        return data['Close'] / data['Open']
    
    yield calculate_test_feature
    
    # Cleanup - remove the feature from registry
    if "test_custom_feature" in FeatureRegistry._features:
        del FeatureRegistry._features["test_custom_feature"]
    if "test" in FeatureRegistry._categories and "test_custom_feature" in FeatureRegistry._categories["test"]:
        FeatureRegistry._categories["test"].remove("test_custom_feature")
    if "test_custom_feature" in FeatureRegistry._metadata:
        del FeatureRegistry._metadata["test_custom_feature"] 