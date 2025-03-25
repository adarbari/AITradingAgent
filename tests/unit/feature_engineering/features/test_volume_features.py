#!/usr/bin/env python3
"""
Unit tests for volume features 
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytest

from src.feature_engineering.features.volume_features import (
    calculate_volume_change,
    calculate_volume_sma_ratio,
    calculate_obv,
    calculate_pvt,
    calculate_volume_price_confirm,
    calculate_relative_volume
)


class TestVolumeFeatures:
    """Test case for volume features"""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing"""
        dates = [datetime.now() + timedelta(days=i) for i in range(30)]
        
        # Create sample data with some patterns for proper testing
        np.random.seed(42)  # For reproducibility
        df = pd.DataFrame({
            'Open': np.random.normal(100, 5, 30),
            'High': np.random.normal(105, 5, 30),
            'Low': np.random.normal(95, 5, 30),
            'Close': np.random.normal(100, 5, 30),
            'Volume': np.random.randint(1000, 5000, 30),
        }, index=dates)
        
        # Ensure high > low for all rows without using max()
        for i in range(len(df)):
            max_val = max(df.iloc[i]['Open'], df.iloc[i]['High'], df.iloc[i]['Low'], df.iloc[i]['Close'])
            min_val = min(df.iloc[i]['Open'], df.iloc[i]['High'], df.iloc[i]['Low'], df.iloc[i]['Close'])
            df.iloc[i, df.columns.get_loc('High')] = max_val + 1
            df.iloc[i, df.columns.get_loc('Low')] = min_val - 1
        
        # Create some price trends
        for i in range(5, 15):
            df.iloc[i, df.columns.get_loc('Close')] = 100 + i
        
        # Create some volume patterns - increasing volume trend
        for i in range(5, 15):
            df.iloc[i, df.columns.get_loc('Volume')] = 1000 + i * 200
        
        return df

    @pytest.fixture
    def empty_data(self):
        """Create empty dataframe for edge case testing"""
        return pd.DataFrame({
            'Open': [],
            'High': [],
            'Low': [],
            'Close': [],
            'Volume': [],
        })

    @pytest.fixture
    def single_row_data(self):
        """Create dataframe with a single row for edge case testing"""
        return pd.DataFrame({
            'Open': [100],
            'High': [105],
            'Low': [95],
            'Close': [101],
            'Volume': [1000],
        }, index=[datetime.now()])

    def test_volume_change_calculation(self, sample_data):
        """Test volume change calculation"""
        result = calculate_volume_change(sample_data)
        
        # Test return type is correct
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
        
        # First value should be 0 (no previous day)
        assert result.iloc[0] == 0
        
        # Manual calculation for subsequent values
        for i in range(1, len(sample_data)):
            prev_vol = sample_data['Volume'].iloc[i-1]
            curr_vol = sample_data['Volume'].iloc[i]
            expected = (curr_vol - prev_vol) / max(prev_vol, 1e-8)
            np.testing.assert_almost_equal(result.iloc[i], expected)
    
    def test_volume_change_empty(self, empty_data):
        """Test volume change with empty data"""
        result = calculate_volume_change(empty_data)
        assert len(result) == 0
    
    def test_volume_change_single_row(self, single_row_data):
        """Test volume change with a single row"""
        result = calculate_volume_change(single_row_data)
        assert len(result) == 1
        assert result.iloc[0] == 0  # First value should be 0
    
    def test_volume_sma_ratio_calculation(self, sample_data):
        """Test volume to SMA ratio calculation"""
        window = 10
        result = calculate_volume_sma_ratio(sample_data, window=window)
        
        # Test return type is correct
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
        
        # Just verify the results are finite
        assert np.isfinite(result).all()
    
    def test_volume_sma_ratio_default_window(self, sample_data):
        """Test volume SMA ratio with default window"""
        result = calculate_volume_sma_ratio(sample_data)  # Default window=20
        assert len(result) == len(sample_data)
    
    def test_volume_sma_ratio_empty(self, empty_data):
        """Test volume SMA ratio with empty data"""
        result = calculate_volume_sma_ratio(empty_data)
        assert len(result) == 0
    
    def test_volume_sma_ratio_single_row(self, single_row_data):
        """Test volume SMA ratio with a single row"""
        result = calculate_volume_sma_ratio(single_row_data)
        assert len(result) == 1
        assert result.iloc[0] == 1.0  # Should be 1 when only one value
    
    def test_obv_calculation(self, sample_data):
        """Test OBV calculation"""
        result = calculate_obv(sample_data)
        
        # Test return type is correct
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
        
        # Verify values are finite
        assert np.isfinite(result).all()
    
    def test_obv_empty(self, empty_data):
        """Test OBV with empty data"""
        result = calculate_obv(empty_data)
        assert len(result) == 0
    
    def test_obv_single_row(self, single_row_data):
        """Test OBV with a single row"""
        result = calculate_obv(single_row_data)
        assert len(result) == 1
        # The implementation starts with a normalized value in the range [-1, 1]
        assert -1.0 <= result.iloc[0] <= 1.0
    
    def test_pvt_calculation(self, sample_data):
        """Test Price Volume Trend (PVT) calculation"""
        result = calculate_pvt(sample_data)
        
        # Test return type is correct
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
        
        # Verify first value is 0
        np.testing.assert_almost_equal(result.iloc[0], 0.0)
        
        # Verify values are finite
        assert np.isfinite(result).all()
    
    def test_pvt_empty(self, empty_data):
        """Test PVT with empty data"""
        result = calculate_pvt(empty_data)
        assert len(result) == 0
    
    def test_pvt_single_row(self, single_row_data):
        """Test PVT with a single row"""
        result = calculate_pvt(single_row_data)
        assert len(result) == 1
        assert result.iloc[0] == 0.0  # First value should be 0
    
    def test_volume_price_confirm_calculation(self, sample_data):
        """Test volume price confirmation calculation"""
        window = 5
        result = calculate_volume_price_confirm(sample_data, window=window)
        
        # Test return type is correct
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
        
        # Verify values are finite
        assert np.isfinite(result).all()
    
    def test_volume_price_confirm_default_window(self, sample_data):
        """Test volume price confirmation with default window"""
        result = calculate_volume_price_confirm(sample_data)  # Default window=5
        assert len(result) == len(sample_data)
    
    def test_volume_price_confirm_empty(self, empty_data):
        """Test volume price confirmation with empty data"""
        result = calculate_volume_price_confirm(empty_data)
        assert len(result) == 0
    
    def test_volume_price_confirm_single_row(self, single_row_data):
        """Test volume price confirmation with a single row"""
        result = calculate_volume_price_confirm(single_row_data)
        assert len(result) == 1
        assert result.iloc[0] == 0.0  # First value should be 0
    
    def test_relative_volume_calculation(self, sample_data):
        """Test relative volume calculation"""
        window = 10
        result = calculate_relative_volume(sample_data, window=window)
        
        # Test return type is correct
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
        
        # First value should be 1.0 (relative to itself)
        assert result.iloc[0] == 1.0
        
        # Manual calculation for subsequent values
        volumes = sample_data['Volume'].values
        for i in range(1, len(sample_data)):
            if i < window:
                # Use available data
                avg_vol = np.mean(volumes[0:i+1])
            else:
                # Use full window
                avg_vol = np.mean(volumes[i-window+1:i+1])
            
            expected = volumes[i] / max(avg_vol, 1e-8)
            np.testing.assert_almost_equal(result.iloc[i], expected)
    
    def test_relative_volume_default_window(self, sample_data):
        """Test relative volume with default window"""
        result = calculate_relative_volume(sample_data)  # Default window=20
        assert len(result) == len(sample_data)
    
    def test_relative_volume_empty(self, empty_data):
        """Test relative volume with empty data"""
        result = calculate_relative_volume(empty_data)
        assert len(result) == 0
    
    def test_relative_volume_single_row(self, single_row_data):
        """Test relative volume with a single row"""
        result = calculate_relative_volume(single_row_data)
        assert len(result) == 1
        assert result.iloc[0] == 1.0  # Should be 1 when only one value 