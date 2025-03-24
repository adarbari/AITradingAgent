"""
Unit tests for the feature registry module.
"""
import pytest
import pandas as pd
import numpy as np
import inspect
from unittest.mock import patch, MagicMock

from src.feature_engineering.registry import FeatureRegistry


class TestFeatureRegistry:
    """Test cases for the FeatureRegistry class."""
    
    def test_register_feature(self, custom_feature):
        """Test that features can be registered correctly."""
        # Feature should be in the registry now
        assert "test_custom_feature" in FeatureRegistry.list_features()
        assert "test" in FeatureRegistry.list_categories()
        assert "test_custom_feature" in FeatureRegistry.get_features_by_category("test")
        
        # Metadata should be stored
        metadata = FeatureRegistry.get_feature_metadata("test_custom_feature")
        assert metadata["name"] == "test_custom_feature"
        assert metadata["category"] == "test"
        assert "Test feature for unit testing" in metadata["doc"]
    
    def test_register_duplicate_feature(self):
        """Test that registering a duplicate feature raises an error."""
        # Register a feature
        @FeatureRegistry.register(name="test_duplicate", category="test")
        def feature1(data):
            """First feature"""
            return data
        
        # Try to register another with the same name
        with pytest.raises(ValueError):
            @FeatureRegistry.register(name="test_duplicate", category="test")
            def feature2(data):
                """Second feature"""
                return data
        
        # Clean up
        if "test_duplicate" in FeatureRegistry._features:
            del FeatureRegistry._features["test_duplicate"]
        if "test" in FeatureRegistry._categories and "test_duplicate" in FeatureRegistry._categories["test"]:
            FeatureRegistry._categories["test"].remove("test_duplicate")
        if "test_duplicate" in FeatureRegistry._metadata:
            del FeatureRegistry._metadata["test_duplicate"]
    
    def test_get_feature(self, custom_feature):
        """Test that features can be retrieved correctly."""
        # Get the feature function
        feature_func = FeatureRegistry.get_feature("test_custom_feature")
        
        # Check that it's the same function
        assert feature_func == custom_feature
        
        # Test with a non-existent feature
        with pytest.raises(KeyError, match="Feature 'non_existent_feature' not found"):
            FeatureRegistry.get_feature("non_existent_feature")
    
    def test_list_features(self):
        """Test listing all features."""
        features = FeatureRegistry.list_features()
        
        # Should be a non-empty list
        assert isinstance(features, list)
        assert len(features) > 0
        
        # Price change should be in the list
        assert "price_change" in features
    
    def test_list_categories(self):
        """Test listing all categories."""
        categories = FeatureRegistry.list_categories()
        
        # Should be a non-empty list
        assert isinstance(categories, list)
        assert len(categories) > 0
        
        # Some standard categories should be there
        standard_categories = ["price", "volume", "momentum", "trend", "volatility", "seasonal"]
        for category in standard_categories:
            assert category in categories
    
    def test_get_features_by_category(self):
        """Test getting features by category."""
        # Check price features
        price_features = FeatureRegistry.get_features_by_category("price")
        assert isinstance(price_features, list)
        assert len(price_features) > 0
        assert "price_change" in price_features
        
        # Check non-existent category
        assert FeatureRegistry.get_features_by_category("non_existent") == []
    
    def test_get_feature_metadata(self):
        """Test getting feature metadata."""
        # Get metadata for an existing feature
        metadata = FeatureRegistry.get_feature_metadata("price_change")
        assert metadata["name"] == "price_change"
        assert metadata["category"] == "price"
        assert "doc" in metadata
        
        # Test with a non-existent feature
        with pytest.raises(KeyError, match="Feature 'non_existent_feature' not found"):
            FeatureRegistry.get_feature_metadata("non_existent_feature")
    
    def test_compute_feature(self, sample_ohlcv_data):
        """Test computing a single feature."""
        # Compute a basic feature
        result = FeatureRegistry.compute_feature("price_change", sample_ohlcv_data)
        
        # Should return a Series
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv_data)
        
        # Test with a non-existent feature
        with pytest.raises(KeyError, match="Feature 'non_existent_feature' not found"):
            FeatureRegistry.compute_feature("non_existent_feature", sample_ohlcv_data)
    
    def test_compute_features(self, sample_ohlcv_data):
        """Test computing multiple features."""
        # Compute multiple features
        features_to_compute = ["price_change", "volatility", "volume_change"]
        result = FeatureRegistry.compute_features(features_to_compute, sample_ohlcv_data)
        
        # Should return a DataFrame with the requested columns
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_ohlcv_data)
        for feature in features_to_compute:
            assert feature in result.columns
    
    def test_compute_features_with_missing(self, sample_ohlcv_data):
        """Test computing features with some missing."""
        # Try to compute a mix of existing and non-existing features
        features_to_compute = ["price_change", "non_existent_1", "volume_change", "non_existent_2"]
        
        # This should not raise an error but print warnings
        with patch('builtins.print') as mock_print:
            result = FeatureRegistry.compute_features(features_to_compute, sample_ohlcv_data)
            
            # Print should be called with error messages
            mock_print.assert_called()
        
        # Should return a DataFrame with only the valid features
        assert isinstance(result, pd.DataFrame)
        assert "price_change" in result.columns
        assert "volume_change" in result.columns
        assert "non_existent_1" not in result.columns
        assert "non_existent_2" not in result.columns
    
    def test_compute_feature_with_kwargs(self, sample_ohlcv_data):
        """Test computing a feature with additional kwargs."""
        # Create a mock feature that accepts kwargs
        @FeatureRegistry.register(name="test_kwargs_feature", category="test")
        def calculate_test_kwargs(data, window=5, alpha=0.5):
            """Test feature with kwargs"""
            return pd.Series(np.ones(len(data)) * window * alpha)
        
        # Compute with default kwargs
        result1 = FeatureRegistry.compute_feature("test_kwargs_feature", sample_ohlcv_data)
        assert result1.iloc[0] == 5 * 0.5  # default values
        
        # Compute with custom kwargs
        result2 = FeatureRegistry.compute_feature("test_kwargs_feature", sample_ohlcv_data, window=10, alpha=0.2)
        assert result2.iloc[0] == 10 * 0.2  # custom values
        
        # Clean up
        if "test_kwargs_feature" in FeatureRegistry._features:
            del FeatureRegistry._features["test_kwargs_feature"]
        if "test" in FeatureRegistry._categories and "test_kwargs_feature" in FeatureRegistry._categories["test"]:
            FeatureRegistry._categories["test"].remove("test_kwargs_feature")
        if "test_kwargs_feature" in FeatureRegistry._metadata:
            del FeatureRegistry._metadata["test_kwargs_feature"]
    
    def test_feature_error_handling(self, sample_ohlcv_data):
        """Test error handling during feature computation."""
        # Register a feature that raises an error
        @FeatureRegistry.register(name="error_feature", category="test")
        def calculate_error_feature(data):
            """Feature that raises an error"""
            raise ValueError("Test error")
        
        # Computing this feature directly should raise the error
        with pytest.raises(ValueError):
            FeatureRegistry.compute_feature("error_feature", sample_ohlcv_data)
        
        # Computing as part of multiple features should skip it and print warning
        with patch('builtins.print') as mock_print:
            result = FeatureRegistry.compute_features(["price_change", "error_feature"], sample_ohlcv_data)
            
            # Print should be called with error message
            mock_print.assert_called()
            
            # Result should only have the good feature
            assert "price_change" in result.columns
            assert "error_feature" not in result.columns
        
        # Clean up
        if "error_feature" in FeatureRegistry._features:
            del FeatureRegistry._features["error_feature"]
        if "test" in FeatureRegistry._categories and "error_feature" in FeatureRegistry._categories["test"]:
            FeatureRegistry._categories["test"].remove("error_feature")
        if "error_feature" in FeatureRegistry._metadata:
            del FeatureRegistry._metadata["error_feature"] 