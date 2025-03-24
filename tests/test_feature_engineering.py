#!/usr/bin/env python3
"""
Tests for the feature engineering module.
"""
import os
import sys
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import shutil

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import feature engineering components
from src.feature_engineering import process_features, FeatureRegistry, FEATURE_CONFIGS
from src.feature_engineering.pipeline import FeaturePipeline
from src.feature_engineering.cache import FeatureCache


class TestFeatureEngineering(unittest.TestCase):
    """Test cases for the feature engineering module."""

    def setUp(self):
        """Set up test data for feature engineering tests."""
        # Create a synthetic price dataset
        dates = pd.date_range(start='2022-01-01', end='2022-03-01')
        n = len(dates)
        
        # Generate synthetic price data with some trend and seasonality
        base_price = 100
        trend = np.linspace(0, 20, n)
        seasonality = 5 * np.sin(np.linspace(0, 4 * np.pi, n))
        noise = np.random.normal(0, 3, n)
        
        close_prices = base_price + trend + seasonality + noise
        open_prices = close_prices - np.random.normal(0, 2, n)
        high_prices = np.maximum(close_prices, open_prices) + np.random.normal(1, 1, n)
        low_prices = np.minimum(close_prices, open_prices) - np.random.normal(1, 1, n)
        volume = np.random.normal(1000000, 200000, n)
        
        # Create the DataFrame
        self.data = pd.DataFrame({
            'Open': open_prices,
            'High': high_prices,
            'Low': low_prices,
            'Close': close_prices,
            'Volume': volume
        }, index=dates)
        
        # Create temporary directory for cache tests
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up after tests."""
        # Remove temp directory
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_feature_registry(self):
        """Test that the feature registry correctly registers and manages features."""
        # Get all registered features
        all_features = FeatureRegistry.list_features()
        
        # Check that we have a reasonable number of features
        self.assertGreater(len(all_features), 10, "Should have at least 10 registered features")
        
        # Test feature categories
        categories = FeatureRegistry.list_categories()
        self.assertGreater(len(categories), 3, "Should have at least 3 feature categories")
        
        # Test getting features by category
        for category in categories:
            features = FeatureRegistry.get_features_by_category(category)
            self.assertGreater(len(features), 0, f"Category {category} should have at least one feature")
        
        # Test feature documentation
        for feature_name in all_features:
            feature_func = FeatureRegistry.get_feature_function(feature_name)
            self.assertIsNotNone(feature_func.__doc__, f"Feature {feature_name} should have documentation")

    def test_feature_computation(self):
        """Test that individual features can be computed correctly."""
        # Test a few basic features
        basic_features = ['price_change', 'volatility', 'rsi_14', 'volume_change']
        
        for feature_name in basic_features:
            # Skip if not registered
            if feature_name not in FeatureRegistry.list_features():
                continue
                
            # Compute the feature
            result = FeatureRegistry.compute_feature(feature_name, self.data)
            
            # Check that the result is a Series
            self.assertIsInstance(result, pd.Series, f"Feature {feature_name} should return a Series")
            
            # Check that the result has the same length as the input
            self.assertEqual(len(result), len(self.data), 
                            f"Feature {feature_name} should return a Series with the same length as input")
            
            # Check that the result doesn't have too many NaN values
            nan_count = result.isna().sum()
            self.assertLess(nan_count / len(result), 0.5, 
                           f"Feature {feature_name} has too many NaN values: {nan_count}/{len(result)}")

    def test_feature_pipeline(self):
        """Test the feature pipeline end-to-end."""
        # Test with a predefined feature set
        if 'minimal' in FEATURE_CONFIGS:
            feature_set = 'minimal'
        else:
            # Create a minimal feature set if not defined
            feature_set = ['price_change', 'volatility', 'volume_change']
            
        # Process features
        pipeline = FeaturePipeline(
            feature_list=feature_set if isinstance(feature_set, list) else FEATURE_CONFIGS.get(feature_set, []),
            feature_count=10,
            verbose=True
        )
        features = pipeline.process(self.data)
        
        # Check that features were computed
        self.assertIsInstance(features, pd.DataFrame, "Pipeline should return a DataFrame")
        self.assertGreater(len(features.columns), 0, "Pipeline should return features")
        
        # Process with convenience function
        if isinstance(feature_set, str) and feature_set in FEATURE_CONFIGS:
            features2 = process_features(self.data, feature_set=feature_set)
            self.assertIsInstance(features2, pd.DataFrame, "process_features should return a DataFrame")
            self.assertGreater(len(features2.columns), 0, "process_features should return features")

    def test_feature_cache(self):
        """Test the feature caching functionality."""
        # Initialize cache
        cache = FeatureCache(cache_dir=self.temp_dir, enable_cache=True, verbose=True)
        
        # Generate a cache key
        cache_key = cache.get_cache_key("TEST", "2022-01-01", "2022-03-01", "standard")
        
        # Create test data to cache
        test_data = pd.DataFrame({
            'feature1': np.random.randn(10),
            'feature2': np.random.randn(10)
        })
        
        # Save to cache
        cache.save(test_data, cache_key)
        
        # Load from cache
        loaded_data = cache.load(cache_key)
        
        # Check that data was cached correctly
        self.assertIsNotNone(loaded_data, "Cached data should be retrieved")
        pd.testing.assert_frame_equal(test_data, loaded_data, "Cached data should match original")
        
        # Test cache stats
        stats = cache.get_cache_stats()
        self.assertTrue('count' in stats, "Cache stats should include count")
        self.assertTrue('size_mb' in stats, "Cache stats should include size")
        
        # Test manually invalidating by removing the file
        cache_path = cache.get_cache_path(cache_key)
        os.remove(cache_path)
        
        invalidated_data = cache.load(cache_key)
        self.assertIsNone(invalidated_data, "Data should be None after cache invalidation")

    def test_feature_correlations(self):
        """Test feature correlations and dependencies."""
        # Get a set of features to test
        features_to_test = [
            'price_change', 'volatility', 'sma_5', 'sma_10',
            'rsi_14', 'macd', 'volume_change'
        ]
        
        # Compute only the features that are registered
        available_features = [f for f in features_to_test if f in FeatureRegistry.list_features()]
        
        if not available_features:
            self.skipTest("No test features available in registry")
            return
            
        # Compute features
        feature_data = FeatureRegistry.compute_features(available_features, self.data)
        
        # Check for correlation (features shouldn't be perfectly correlated)
        if len(available_features) > 1:
            correlation = feature_data.corr()
            
            # Check that we don't have perfect correlation between all features
            for i in range(len(correlation.columns)):
                for j in range(i+1, len(correlation.columns)):
                    col_i = correlation.columns[i]
                    col_j = correlation.columns[j]
                    
                    # Skip if correlation is NaN
                    if pd.isna(correlation.loc[col_i, col_j]):
                        continue
                        
                    # Features shouldn't be perfectly correlated
                    self.assertNotEqual(
                        abs(correlation.loc[col_i, col_j]), 
                        1.0, 
                        f"Features {col_i} and {col_j} are perfectly correlated"
                    )


if __name__ == '__main__':
    unittest.main() 