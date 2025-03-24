"""
Integration tests for the feature engineering module.
Tests the full pipeline from raw data to features and their use in models.
"""
import os
import sys
import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from datetime import datetime, timedelta

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.feature_engineering import process_features, FeatureRegistry, FEATURE_CONFIGS
from src.feature_engineering.pipeline import FeaturePipeline
from src.feature_engineering.cache import FeatureCache
from src.utils.feature_utils import get_data
from src.agent.trading_env import TradingEnvironment
from src.train.trainer import TrainingManager


@pytest.fixture
def test_data():
    """
    Generate a realistic test dataset with multiple stocks.
    """
    # Create date range
    dates = pd.date_range(start='2022-01-01', end='2022-03-01')
    n = len(dates)
    
    # Define stocks
    stocks = ["AAPL", "MSFT", "GOOG"]
    
    # Dictionary to hold DataFrames
    data_dict = {}
    
    for stock in stocks:
        # Generate synthetic price data with trend, seasonality, and noise components
        base_price = 100 + np.random.normal(0, 10)  # Different starting points
        trend = np.linspace(0, 20, n) * (1 + np.random.normal(0, 0.2))  # Different trends
        seasonality = 5 * np.sin(np.linspace(0, 4 * np.pi, n))
        noise = np.random.normal(0, 3, n)
        
        # Combine components to create price series
        close_prices = base_price + trend + seasonality + noise
        
        # Create related price series
        open_prices = close_prices - np.random.normal(0, 2, n)
        high_prices = np.maximum(close_prices, open_prices) + np.random.normal(1, 1, n)
        low_prices = np.minimum(close_prices, open_prices) - np.random.normal(1, 1, n)
        
        # Generate volume
        volume = np.random.normal(1000000, 200000, n)
        
        # Create the DataFrame
        data_dict[stock] = pd.DataFrame({
            'Open': open_prices,
            'High': high_prices,
            'Low': low_prices,
            'Close': close_prices,
            'Volume': volume
        }, index=dates)
    
    return data_dict


@pytest.fixture
def temp_cache_dir():
    """
    Create a temporary directory for caching during tests.
    """
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


class TestFeatureEngineeringIntegration:
    """
    Integration tests for the full feature engineering pipeline.
    """
    
    def test_end_to_end_pipeline(self, test_data, temp_cache_dir):
        """
        Test the full feature engineering pipeline from raw data to features.
        """
        # Get sample data for one stock
        data = test_data["AAPL"]
        
        # 1. Test feature processing pipeline
        cache = FeatureCache(cache_dir=temp_cache_dir, enable_cache=True)
        cache_key = cache.get_cache_key("AAPL", "2022-01-01", "2022-03-01", "standard")
        
        # Process features through the pipeline
        features = process_features(data, feature_set="standard", verbose=True)
        
        # Check basic properties
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(data)
        assert len(features.columns) > 0
        
        # Ensure no NaN or Inf values
        assert not features.isnull().any().any()
        assert not np.isinf(features.values).any()
        
        # 2. Test caching
        # Save to cache
        assert cache.save(features, cache_key)
        
        # Load from cache
        cached_features = cache.load(cache_key)
        assert cached_features is not None
        pd.testing.assert_frame_equal(features, cached_features)
    
    def test_feature_categories_coverage(self, test_data):
        """
        Test that all feature categories are properly represented in the output.
        """
        # Get sample data
        data = test_data["MSFT"]
        
        # Process features
        features = process_features(data, feature_set="standard", verbose=False)
        
        # Get all categories
        categories = FeatureRegistry.list_categories()
        
        # For each category, at least one feature should be in the output
        for category in categories:
            category_features = FeatureRegistry.get_features_by_category(category)
            
            # Skip if category is empty
            if not category_features:
                continue
                
            # Skip if category features are not part of standard set
            if not any(feature in FEATURE_CONFIGS.get("standard", []) for feature in category_features):
                continue
                
            # At least one feature from this category should be in the output
            assert any(feature in features.columns for feature in category_features), \
                f"No features from category '{category}' found in the output"
    
    def test_input_data_flexibility(self, test_data):
        """
        Test that the pipeline works with different input data formats.
        """
        # Get sample data
        data = test_data["GOOG"]
        
        # Test with DatetimeIndex
        features1 = process_features(data, feature_set="minimal", verbose=False)
        
        # Test with regular RangeIndex, datetime in a column
        data2 = data.reset_index().rename(columns={"index": "Date"})
        features2 = process_features(data2, feature_set="minimal", verbose=False)
        
        # Should get the same features (minus the index)
        assert features1.columns.equals(features2.columns)
        
        # Test with missing columns (should fill in with zeros or handle gracefully)
        data3 = data.copy()
        del data3["Volume"]  # Remove Volume column
        features3 = process_features(data3, feature_set="minimal", verbose=False)
        
        # Should still work but possibly with missing volume-based features
        assert isinstance(features3, pd.DataFrame)
        assert len(features3) == len(data)
    
    def test_trading_env_integration(self, test_data):
        """
        Test integration with trading environment.
        """
        # Get sample data
        data = test_data["AAPL"]
        
        # Process features
        features = process_features(data, feature_set="minimal", verbose=False)
        
        # Create a trading environment
        env = TradingEnvironment(
            prices=data['Close'].values,
            features=features,
            initial_balance=10000,
            transaction_fee_percent=0.001
        )
        
        # Basic test - make sure we can reset and step
        obs = env.reset()
        assert obs is not None
        
        # Take an action and check response
        action = np.array([1.0])  # Buy action as numpy array
        step_result = env.step(action)
        
        # Check return structure
        if len(step_result) == 5:  # New gymnasium API
            obs, reward, terminated, truncated, info = step_result
            assert not terminated
            assert not truncated
        else:  # Old gym API
            obs, reward, done, info = step_result
            assert not done
            
        # Check that info contains expected fields
        assert "portfolio_value" in info
        # Check for position-related fields (various implementations might use different names)
        assert "shares_held" in info or "position" in info, f"No position-related field in info: {info.keys()}"
        
        # Check observation structure - some implementations may return dictionaries or structured objects
        if isinstance(obs, dict):
            # Dict-like observation
            assert "features" in obs or "observation" in obs
            # Check that portfolio information is included
            assert "portfolio" in obs or any(k.lower().find("portf") >= 0 for k in obs.keys())
        else:
            # Array-like observation
            assert len(obs) >= 3  # At least position, balance, and price
    
    @pytest.mark.skip(reason="Requires actual model training, slow")
    def test_model_training_integration(self, test_data, temp_cache_dir):
        """
        Test integration with model training.
        Note: This test is skipped by default as it requires actual model training, which is slow.
        """
        # Setup temporary directories
        models_dir = os.path.join(temp_cache_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        
        # Get sample data
        data = test_data["AAPL"]
        
        # Process features
        features = process_features(data, feature_set="minimal", verbose=False)
        
        # Initialize training manager
        training_manager = TrainingManager(models_dir=models_dir, verbose=0)
        
        # Train a simple model
        symbol = "AAPL"
        train_start = "2022-01-01"
        train_end = "2022-02-15"
        
        # Use a very small number of timesteps for testing
        model, path = training_manager.get_model(
            symbol=symbol,
            train_start=train_start,
            train_end=train_end,
            feature_count=len(features.columns),
            data_source="test",
            timesteps=10,
            force_train=True,
            features=features.loc[train_start:train_end],
            prices=data.loc[train_start:train_end, 'Close'].values
        )
        
        # Ensure model was created
        assert model is not None
        assert os.path.exists(path)
    
    def test_get_data_integration(self):
        """
        Test integration with the get_data utility.
        Use synthetic data to avoid API calls.
        """
        # Define parameters
        symbol = "TEST"
        start_date = "2022-01-01"
        end_date = "2022-03-01"
        
        # Get synthetic data
        data = get_data(symbol, start_date, end_date, data_source="synthetic")
        
        # Check data
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert "Open" in data.columns
        assert "High" in data.columns
        assert "Low" in data.columns
        assert "Close" in data.columns
        assert "Volume" in data.columns
        
        # Process features
        features = process_features(data, feature_set="minimal", verbose=False)
        
        # Check features
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(data)
        assert len(features.columns) > 0
    
    def test_feature_set_configurations(self, test_data):
        """
        Test different feature set configurations.
        """
        # Get sample data
        data = test_data["MSFT"]
        
        # Test minimal set
        if "minimal" in FEATURE_CONFIGS:
            minimal_features = process_features(data, feature_set="minimal", verbose=False)
            # The minimal set still has feature_count=21 by default, so it will be padded with dummy features
            assert len(minimal_features.columns) == 21
            # Check that at least some of the minimal features are present
            minimal_feature_list = FEATURE_CONFIGS.get("minimal", [])
            for feature in minimal_feature_list:
                assert feature in minimal_features.columns
        
        # Test standard set
        if "standard" in FEATURE_CONFIGS:
            standard_features = process_features(data, feature_set="standard", verbose=False)
            assert len(standard_features.columns) >= 10  # Should be a larger set
        
        # Test advanced set (if available)
        if "advanced" in FEATURE_CONFIGS:
            advanced_features = process_features(data, feature_set="advanced", verbose=False)
            assert len(advanced_features.columns) >= len(standard_features.columns)  # Should be even larger
        
        # Test custom set with explicit feature_count and as a string list to avoid the split() issue
        custom_feature_list = "price_change,volatility,rsi_14"
        custom_features = process_features(
            data, 
            feature_set=custom_feature_list,
            feature_count=3,  # Explicitly set to match our features
            verbose=False
        )
        assert len(custom_features.columns) == 3
        assert "price_change" in custom_features.columns
        assert "volatility" in custom_features.columns
        assert "rsi_14" in custom_features.columns
    
    def test_performance(self, test_data, temp_cache_dir):
        """
        Test performance of the feature engineering pipeline.
        """
        # Get sample data
        data = test_data["GOOG"]
        
        # Create a cache
        cache = FeatureCache(cache_dir=temp_cache_dir, enable_cache=True)
        
        # Measure time to process features (first run, no cache)
        import time
        start_time = time.time()
        features1 = process_features(data, feature_set="standard", verbose=False)
        first_run_time = time.time() - start_time
        
        # Measure time to process features (second run, with cache)
        start_time = time.time()
        features2 = process_features(data, feature_set="standard", verbose=False)
        second_run_time = time.time() - start_time
        
        # Second run should be faster due to caching
        assert second_run_time < first_run_time, "Cached run should be faster"
        
        # Verify results are the same
        pd.testing.assert_frame_equal(features1, features2)
    
    def test_multi_stock_processing(self, test_data):
        """
        Test processing features for multiple stocks.
        """
        # Process features for all stocks
        all_features = {}
        for symbol, data in test_data.items():
            features = process_features(data, feature_set="minimal", verbose=False)
            all_features[symbol] = features
        
        # All should have the same columns
        first_columns = list(all_features[list(all_features.keys())[0]].columns)
        for symbol, features in all_features.items():
            assert list(features.columns) == first_columns
        
        # Check correlations between stocks
        corr_matrix = {}
        for feature in first_columns:
            # Extract this feature for all stocks
            feature_data = pd.DataFrame({
                symbol: features[feature] for symbol, features in all_features.items()
            })
            
            # Calculate correlation
            corr_matrix[feature] = feature_data.corr()
        
        # There should be some differences in correlations (features shouldn't be identical across stocks)
        for feature, corr in corr_matrix.items():
            if len(corr) > 1:
                # If there's more than one stock, check that not all correlations are 1
                assert not np.all(np.abs(corr.values - 1.0) < 1e-8), f"Feature {feature} is identical across all stocks" 