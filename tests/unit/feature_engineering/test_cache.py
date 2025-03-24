"""
Unit tests for the feature caching module.
"""
import pytest
import pandas as pd
import numpy as np
import os
import json
import time
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, mock_open

# Add parent directory to path so we can import from src
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.feature_engineering.cache import FeatureCache


class TestFeatureCache:
    """Test cases for the FeatureCache class."""
    
    def test_init(self, temp_cache_dir):
        """Test initialization with default and custom parameters."""
        # Default initialization
        cache = FeatureCache()
        assert cache.cache_dir == ".feature_cache"
        assert cache.max_age_days == 30
        assert cache.enable_cache is True
        
        # Custom initialization
        cache = FeatureCache(
            cache_dir=temp_cache_dir,
            max_age_days=10,
            enable_cache=False,
            verbose=True
        )
        assert cache.cache_dir == temp_cache_dir
        assert cache.max_age_days == 10
        assert cache.enable_cache is False
        assert cache.verbose is True
        
        # Directory should be created if enable_cache is True
        cache = FeatureCache(cache_dir=os.path.join(temp_cache_dir, "new_cache"))
        assert os.path.exists(os.path.join(temp_cache_dir, "new_cache"))
    
    def test_get_cache_key(self, feature_cache):
        """Test generating cache keys for different inputs."""
        # Test with string feature set
        key1 = feature_cache.get_cache_key("AAPL", "2022-01-01", "2022-03-01", "standard")
        assert isinstance(key1, str)
        assert len(key1) > 0
        
        # Test with list feature set
        key2 = feature_cache.get_cache_key("AAPL", "2022-01-01", "2022-03-01", ["price_change", "volatility"])
        assert isinstance(key2, str)
        assert len(key2) > 0
        
        # Same parameters should give same key
        key3 = feature_cache.get_cache_key("AAPL", "2022-01-01", "2022-03-01", ["price_change", "volatility"])
        assert key2 == key3
        
        # Different parameters should give different keys
        key4 = feature_cache.get_cache_key("MSFT", "2022-01-01", "2022-03-01", ["price_change", "volatility"])
        assert key2 != key4
        
        # Order of features shouldn't matter
        key5 = feature_cache.get_cache_key("AAPL", "2022-01-01", "2022-03-01", ["volatility", "price_change"])
        assert key2 == key5
        
        # Test with config
        key6 = feature_cache.get_cache_key(
            "AAPL", "2022-01-01", "2022-03-01", "standard", 
            feature_config={"normalize": True, "method": "zscore"}
        )
        assert key1 != key6
    
    def test_get_cache_path(self, feature_cache):
        """Test getting cache file paths."""
        # Get a cache key
        key = feature_cache.get_cache_key("AAPL", "2022-01-01", "2022-03-01", "standard")
        
        # Get cache path
        path = feature_cache.get_cache_path(key)
        
        # Should be in the cache directory
        assert os.path.dirname(path) == feature_cache.cache_dir
        
        # Should have the key in the filename
        assert key in os.path.basename(path)
        
        # Should have a .pkl extension
        assert path.endswith(".pkl")
    
    def test_is_cached(self, feature_cache, sample_ohlcv_data):
        """Test checking if data is cached."""
        # Get a cache key
        key = feature_cache.get_cache_key("AAPL", "2022-01-01", "2022-03-01", "standard")
        
        # Initially should not be cached
        assert not feature_cache.is_cached(key)
        
        # Save some data
        test_data = pd.DataFrame({
            'feature1': np.random.randn(10),
            'feature2': np.random.randn(10)
        })
        feature_cache.save(test_data, key)
        
        # Now should be cached
        assert feature_cache.is_cached(key)
    
    def test_load_save(self, feature_cache, sample_ohlcv_data):
        """Test saving and loading from cache."""
        # Get a cache key
        key = feature_cache.get_cache_key("AAPL", "2022-01-01", "2022-03-01", "standard")
        
        # Create test data
        test_data = pd.DataFrame({
            'feature1': np.random.randn(10),
            'feature2': np.random.randn(10)
        })
        
        # Save to cache
        assert feature_cache.save(test_data, key)
        
        # Load from cache
        loaded_data = feature_cache.load(key)
        
        # Should get the same data back
        assert loaded_data is not None
        pd.testing.assert_frame_equal(test_data, loaded_data)
    
    def test_cache_disabled(self, feature_cache, sample_ohlcv_data):
        """Test behavior when cache is disabled."""
        # Disable cache
        feature_cache.enable_cache = False
        
        # Get a cache key
        key = feature_cache.get_cache_key("AAPL", "2022-01-01", "2022-03-01", "standard")
        
        # Create test data
        test_data = pd.DataFrame({
            'feature1': np.random.randn(10),
            'feature2': np.random.randn(10)
        })
        
        # Save to cache should return False
        assert not feature_cache.save(test_data, key)
        
        # is_cached should return False
        assert not feature_cache.is_cached(key)
        
        # Load should return None
        assert feature_cache.load(key) is None
    
    def test_cache_expiration(self, feature_cache, sample_ohlcv_data):
        """Test cache expiration handling."""
        # Set a short max age
        feature_cache.max_age_days = 5
        
        # Get a cache key
        key = feature_cache.get_cache_key("AAPL", "2022-01-01", "2022-03-01", "standard")
        
        # Create test data
        test_data = pd.DataFrame({
            'feature1': np.random.randn(10),
            'feature2': np.random.randn(10)
        })
        
        # Save to cache
        feature_cache.save(test_data, key)
        
        # Should be cached now
        assert feature_cache.is_cached(key)
        
        # Modify file timestamp to be older than max age
        cache_path = feature_cache.get_cache_path(key)
        old_time = datetime.now() - timedelta(days=10)
        os.utime(cache_path, (old_time.timestamp(), old_time.timestamp()))
        
        # Should no longer be considered cached
        assert not feature_cache.is_cached(key)
    
    def test_invalidate(self, feature_cache, sample_ohlcv_data):
        """Test manually invalidating a cache entry."""
        # Get a cache key
        key = feature_cache.get_cache_key("AAPL", "2022-01-01", "2022-03-01", "standard")
        
        # Create test data
        test_data = pd.DataFrame({
            'feature1': np.random.randn(10),
            'feature2': np.random.randn(10)
        })
        
        # Save to cache
        feature_cache.save(test_data, key)
        
        # Should be cached
        assert feature_cache.is_cached(key)
        
        # Manually invalidate by removing the file
        cache_path = feature_cache.get_cache_path(key)
        os.remove(cache_path)
        
        # Should no longer be cached
        assert not feature_cache.is_cached(key)
        
        # Load should return None
        assert feature_cache.load(key) is None
    
    def test_clear(self, feature_cache):
        """Test clearing the cache."""
        # Create multiple cache entries
        for symbol in ["AAPL", "MSFT", "GOOG"]:
            key = feature_cache.get_cache_key(symbol, "2022-01-01", "2022-03-01", "standard")
            test_data = pd.DataFrame({
                'feature1': np.random.randn(10),
                'feature2': np.random.randn(10)
            })
            feature_cache.save(test_data, key)
        
        # Clear all
        cleared = feature_cache.clear()
        
        # Should return the number of files cleared
        assert cleared == 3
        
        # Cache should be empty
        for symbol in ["AAPL", "MSFT", "GOOG"]:
            key = feature_cache.get_cache_key(symbol, "2022-01-01", "2022-03-01", "standard")
            assert not feature_cache.is_cached(key)
    
    def test_clear_with_age(self, feature_cache):
        """Test clearing cache entries based on age."""
        # Create multiple cache entries
        for symbol in ["AAPL", "MSFT", "GOOG"]:
            key = feature_cache.get_cache_key(symbol, "2022-01-01", "2022-03-01", "standard")
            test_data = pd.DataFrame({
                'feature1': np.random.randn(10),
                'feature2': np.random.randn(10)
            })
            feature_cache.save(test_data, key)
        
        # Modify one file to be old
        old_key = feature_cache.get_cache_key("AAPL", "2022-01-01", "2022-03-01", "standard")
        old_path = feature_cache.get_cache_path(old_key)
        old_time = datetime.now() - timedelta(days=10)
        os.utime(old_path, (old_time.timestamp(), old_time.timestamp()))
        
        # Clear old entries
        cleared = feature_cache.clear(older_than_days=5)
        
        # Should have cleared only the old entry
        assert cleared == 1
        assert not feature_cache.is_cached(old_key)
        
        # Other entries should still be cached
        assert feature_cache.is_cached(feature_cache.get_cache_key("MSFT", "2022-01-01", "2022-03-01", "standard"))
        assert feature_cache.is_cached(feature_cache.get_cache_key("GOOG", "2022-01-01", "2022-03-01", "standard"))
    
    def test_get_cache_stats(self, feature_cache):
        """Test retrieving cache statistics."""
        # Initially stats should show empty cache
        stats = feature_cache.get_cache_stats()
        assert stats["enabled"] is True
        assert stats["count"] == 0
        
        # Add some cache entries
        for symbol in ["AAPL", "MSFT", "GOOG"]:
            key = feature_cache.get_cache_key(symbol, "2022-01-01", "2022-03-01", "standard")
            test_data = pd.DataFrame({
                'feature1': np.random.randn(10),
                'feature2': np.random.randn(10)
            })
            feature_cache.save(test_data, key)
        
        # Check updated stats
        stats = feature_cache.get_cache_stats()
        assert stats["count"] == 3
        assert stats["size_mb"] > 0
    
    def test_error_handling(self, feature_cache, temp_cache_dir):
        """Test error handling during load and save operations."""
        # Test load with invalid data
        key = "invalid_key"
        cache_path = feature_cache.get_cache_path(key)
        
        # Create an invalid pickle file
        with open(cache_path, "w") as f:
            f.write("not a pickle file")
        
        # Load should return None and not raise an exception
        with patch('builtins.print') as mock_print:
            result = feature_cache.load(key)
            assert result is None
            assert mock_print.call_count >= 1
        
        # Test save with problematic dataframe (using a mock)
        test_data = pd.DataFrame({
            'feature1': np.random.randn(10),
            'feature2': np.random.randn(10)
        })
        
        with patch('pandas.DataFrame.to_pickle', side_effect=Exception("Test error")):
            with patch('builtins.print') as mock_print:
                result = feature_cache.save(test_data, "test_key")
                assert result is False
                assert mock_print.call_count >= 1 