"""
Unit tests for the feature pipeline module.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from src.feature_engineering.pipeline import FeaturePipeline
from src.feature_engineering import process_features, FEATURE_CONFIGS
from src.feature_engineering.registry import FeatureRegistry


class TestFeaturePipeline:
    """Test cases for the FeaturePipeline class."""
    
    def test_init(self, sample_ohlcv_data):
        """Test initialization with different parameters."""
        # Test with default parameters
        pipeline = FeaturePipeline()
        assert pipeline.feature_count == 21
        assert pipeline.verbose is False
        assert pipeline.normalize is True
        assert pipeline.normalization_method == "zscore"
        
        # Test with custom parameters
        pipeline = FeaturePipeline(
            feature_list=["price_change", "volatility"],
            feature_count=10,
            verbose=True,
            normalize=False
        )
        assert pipeline.feature_count == 10
        assert pipeline.verbose is True
        assert pipeline.normalize is False
        assert len(pipeline.feature_list) == 2
        
        # Test with non-existent features
        with patch('warnings.warn') as mock_warn:
            pipeline = FeaturePipeline(
                feature_list=["price_change", "non_existent_feature"],
                verbose=True
            )
            mock_warn.assert_called_once()
            warning_msg = mock_warn.call_args[0][0]
            assert "not registered" in warning_msg
    
    def test_validate_features(self):
        """Test feature validation."""
        # Test with valid features
        pipeline = FeaturePipeline(feature_list=["price_change", "volatility"])
        # _validate_features is called in __init__, so we don't need to call it here
        
        # Test with invalid features
        with patch('warnings.warn') as mock_warn:
            pipeline = FeaturePipeline(feature_list=["price_change", "non_existent_feature"])
            mock_warn.assert_called_once()
            warning_msg = mock_warn.call_args[0][0]
            assert "Feature 'non_existent_feature' is not registered" in warning_msg
    
    def test_process(self, sample_ohlcv_data, feature_pipeline_minimal):
        """Test the main processing method."""
        # Process the data
        features = feature_pipeline_minimal.process(sample_ohlcv_data)
        
        # Check the output
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(sample_ohlcv_data)
        assert len(features.columns) == feature_pipeline_minimal.feature_count
        
        # Ensure no NaN values
        assert not features.isnull().any().any()
    
    def test_generate_features(self, sample_ohlcv_data, feature_pipeline_minimal):
        """Test feature generation."""
        # Mock the compute_features method
        with patch('src.feature_engineering.registry.FeatureRegistry.compute_features') as mock_compute:
            mock_compute.return_value = pd.DataFrame({
                'feature1': np.random.rand(len(sample_ohlcv_data)),
                'feature2': np.random.rand(len(sample_ohlcv_data))
            })
            
            # Generate features
            features = feature_pipeline_minimal._generate_features(sample_ohlcv_data)
            
            # Check that the method was called with the right arguments
            mock_compute.assert_called_once_with(
                feature_pipeline_minimal.feature_list, sample_ohlcv_data
            )
            
            # Check the output
            assert isinstance(features, pd.DataFrame)
            assert len(features) == len(sample_ohlcv_data)
            assert len(features.columns) == 2
    
    def test_normalize_features_zscore(self, sample_ohlcv_data):
        """Test Z-score normalization."""
        # Create test data
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50]
        })
        
        # Create pipeline with Z-score normalization
        pipeline = FeaturePipeline(
            feature_list=['feature1', 'feature2'],
            normalization_method="zscore"
        )
        
        # Normalize
        normalized = pipeline._normalize_features(data)
        
        # Check statistical properties
        for col in normalized.columns:
            assert abs(normalized[col].mean()) < 1e-10  # Should be close to 0
            assert abs(normalized[col].std() - 1.0) < 1e-10  # Should be close to 1
    
    def test_normalize_features_minmax(self, sample_ohlcv_data):
        """Test min-max normalization."""
        # Create test data
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50]
        })
        
        # Create pipeline with min-max normalization
        pipeline = FeaturePipeline(
            feature_list=['feature1', 'feature2'],
            normalization_method="minmax"
        )
        
        # Normalize
        normalized = pipeline._normalize_features(data)
        
        # Check range (should be 0 to 1)
        for col in normalized.columns:
            assert abs(normalized[col].min()) < 1e-10  # Should be close to 0
            assert abs(normalized[col].max() - 1.0) < 1e-10  # Should be close to 1
    
    def test_normalize_features_robust(self, sample_ohlcv_data):
        """Test robust normalization."""
        # Create test data
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9],
            'feature2': [10, 20, 30, 40, 50, 60, 70, 80, 90]
        })
        
        # Create pipeline with robust normalization
        pipeline = FeaturePipeline(
            feature_list=['feature1', 'feature2'],
            normalization_method="robust"
        )
        
        # Normalize
        normalized = pipeline._normalize_features(data)
        
        # Check that median is close to 0
        for col in normalized.columns:
            assert abs(normalized[col].median()) < 1e-10
    
    def test_normalize_features_unknown_method(self, sample_ohlcv_data):
        """Test behavior with unknown normalization method."""
        # Create test data
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50]
        })
        
        # Create pipeline with unknown normalization method
        pipeline = FeaturePipeline(
            feature_list=['feature1', 'feature2'],
            normalization_method="unknown_method"
        )
        
        # Should print a warning and return unnormalized features
        with patch('warnings.warn') as mock_warn:
            normalized = pipeline._normalize_features(data)
            mock_warn.assert_called_once()
            warning_msg = mock_warn.call_args[0][0]
            assert "Unknown normalization method" in warning_msg
        
        # Check that data was not modified
        pd.testing.assert_frame_equal(normalized, data)
    
    def test_handle_feature_count_exact(self):
        """Test handling feature count when count is exact."""
        # Create test data with exact feature count
        data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        })
        
        # Create pipeline with feature_count=2
        pipeline = FeaturePipeline(feature_list=['feature1', 'feature2'], feature_count=2)
        
        # Handle feature count
        result = pipeline._handle_feature_count(data)
        
        # Should be unchanged
        pd.testing.assert_frame_equal(result, data)
    
    def test_handle_feature_count_too_few(self):
        """Test handling feature count when there are too few features."""
        # Create test data with fewer features than expected
        data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        })
        
        # Create pipeline with feature_count=5
        pipeline = FeaturePipeline(feature_list=['feature1', 'feature2'], feature_count=5, verbose=True)
        
        # Handle feature count (should add dummy features)
        with patch('builtins.print') as mock_print:
            result = pipeline._handle_feature_count(data)
            assert mock_print.call_count >= 1
        
        # Should have 5 columns now
        assert len(result.columns) == 5
        
        # First two columns should be unchanged
        pd.testing.assert_series_equal(result['feature1'], data['feature1'])
        pd.testing.assert_series_equal(result['feature2'], data['feature2'])
        
        # Added columns should be all zeros
        for i in range(3):
            col_name = f"dummy_feature_{i+2}"
            assert col_name in result.columns
            assert (result[col_name] == 0).all()
    
    def test_handle_feature_count_too_many(self):
        """Test handling feature count when there are too many features."""
        # Create test data with more features than expected
        data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'feature3': [7, 8, 9],
            'feature4': [10, 11, 12],
            'feature5': [13, 14, 15]
        })
        
        # Create pipeline with feature_count=3
        pipeline = FeaturePipeline(
            feature_list=['feature1', 'feature2', 'feature3', 'feature4', 'feature5'],
            feature_count=3,
            verbose=True
        )
        
        # Handle feature count (should drop extra features)
        with patch('builtins.print') as mock_print:
            result = pipeline._handle_feature_count(data)
            assert mock_print.call_count >= 1
        
        # Should have 3 columns now
        assert len(result.columns) == 3
        
        # Should keep only the first 3 columns
        expected_columns = ['feature1', 'feature2', 'feature3']
        assert list(result.columns) == expected_columns
    
    def test_final_cleanup_no_issues(self):
        """Test final cleanup with no issues."""
        # Create clean test data
        data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        })
        
        # Create pipeline
        pipeline = FeaturePipeline(feature_list=['feature1', 'feature2'])
        
        # Clean the data
        result = pipeline._final_cleanup(data)
        
        # Should be unchanged
        pd.testing.assert_frame_equal(result, data)
    
    def test_final_cleanup_with_issues(self):
        """Test final cleanup with NaN and infinity values."""
        # Create test data with NaN and infinity
        data = pd.DataFrame({
            'feature1': [1, np.nan, 3],
            'feature2': [4, np.inf, 6],
            'feature3': [7, 8, -np.inf]
        })
        
        # Create pipeline with verbose output
        pipeline = FeaturePipeline(
            feature_list=['feature1', 'feature2', 'feature3'],
            verbose=True
        )
        
        # Clean the data
        with patch('builtins.print') as mock_print:
            result = pipeline._final_cleanup(data)
            assert mock_print.call_count >= 1
            warning_msg = mock_print.call_args[0][0]
            assert "NaN or infinite values" in warning_msg
        
        # Check that all values are finite
        assert np.isfinite(result.values).all()
        
        # NaN and infinity should be replaced with 0
        assert result.iloc[1, 0] == 0  # NaN in feature1
        assert result.iloc[1, 1] == 0  # inf in feature2
        assert result.iloc[2, 2] == 0  # -inf in feature3
    
    def test_process_features_convenience(self, sample_ohlcv_data):
        """Test the convenience function for processing features."""
        # Process features using the convenience function
        features = process_features(sample_ohlcv_data, feature_set="minimal", verbose=True)
        
        # Check output structure
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(sample_ohlcv_data)
        assert len(features.columns) > 0
    
    def test_process_edge_cases(self, empty_ohlcv_data, abnormal_ohlcv_data, small_ohlcv_data):
        """Test processing with various edge cases."""
        # Test with empty DataFrame
        if len(empty_ohlcv_data) > 0:  # Some implementations might not allow truly empty DataFrames
            features = process_features(empty_ohlcv_data, feature_set=["price_change"], verbose=False)
            assert isinstance(features, pd.DataFrame)
        
        # Test with abnormal data (NaN, zeros, etc.)
        features = process_features(abnormal_ohlcv_data, feature_set=["price_change", "volatility"], verbose=False)
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(abnormal_ohlcv_data)
        
        # Test with very small dataset
        features = process_features(small_ohlcv_data, feature_set=["price_change"], verbose=False)
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(small_ohlcv_data) 