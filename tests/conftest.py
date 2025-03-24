"""
Fixtures and utilities for tests
"""
import os
import sys
import pytest
import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timedelta

# Suppress specific deprecation warnings from external libraries
warnings.filterwarnings("ignore", message="distutils Version classes are deprecated")
warnings.filterwarnings("ignore", message="Box low's precision lowered by casting to float32")
warnings.filterwarnings("ignore", message="Box high's precision lowered by casting to float32")
warnings.filterwarnings("ignore", message="__array__ implementation doesn't accept a copy keyword")

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import SyntheticDataFetcher, YahooDataFetcher, DataFetcherFactory
from src.models import ModelTrainer
from src.backtest import Backtester
from src.agent.trading_env import TradingEnvironment


@pytest.fixture
def sample_price_data():
    """Generate a sample price dataset"""
    # Create a simple price series with a clear trend
    dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
    prices = np.linspace(100, 150, 100) + np.random.normal(0, 5, 100)
    
    # Create a DataFrame with standard OHLCV structure
    data = pd.DataFrame({
        'Date': dates,
        'Open': prices,
        'High': prices * 1.02,
        'Low': prices * 0.98,
        'Close': prices,
        'Volume': np.random.randint(1000, 10000, 100)
    })
    
    return data


@pytest.fixture
def sample_features():
    """Generate sample features for the trading environment"""
    # Create 100 days of feature data with 10 features each
    features = np.random.normal(0, 1, (100, 10)).astype(np.float32)
    return features


@pytest.fixture
def synthetic_data_fetcher():
    """Create a synthetic data fetcher"""
    return SyntheticDataFetcher()


@pytest.fixture
def data_fetcher_factory():
    """Create a data fetcher factory"""
    return DataFetcherFactory


@pytest.fixture
def trading_env_class():
    """Return the TradingEnvironment class"""
    return TradingEnvironment


@pytest.fixture
def trading_env(sample_price_data, sample_features):
    """Create a trading environment with sample data"""
    # Convert to format expected by TradingEnvironment
    prices = sample_price_data['Close'].values
    features = sample_features
    
    env = TradingEnvironment(
        prices=prices,
        features=features,
        initial_balance=10000,
        transaction_fee_percent=0.001
    )
    
    return env


@pytest.fixture
def model_trainer():
    """Create a model trainer"""
    # Use a temporary directory for test models
    test_models_dir = os.path.join('tests', 'test_models')
    os.makedirs(test_models_dir, exist_ok=True)
    
    return ModelTrainer(models_dir=test_models_dir, verbose=0)


@pytest.fixture
def backtester():
    """Create a backtester"""
    # Use a temporary directory for test results
    test_results_dir = os.path.join('tests', 'test_results')
    os.makedirs(test_results_dir, exist_ok=True)
    
    return Backtester(results_dir=test_results_dir)


@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Clean up test files and directories after tests"""
    yield
    
    # Clean up test model and results directories
    import shutil
    test_models_dir = os.path.join('tests', 'test_models')
    test_results_dir = os.path.join('tests', 'test_results')
    
    if os.path.exists(test_models_dir):
        shutil.rmtree(test_models_dir)
    
    if os.path.exists(test_results_dir):
        shutil.rmtree(test_results_dir) 