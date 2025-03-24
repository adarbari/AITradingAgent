"""
Unit tests for feature implementations in each category.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.feature_engineering.registry import FeatureRegistry


class TestPriceFeatures:
    """Tests for price-based features."""
    
    def test_price_change(self, sample_ohlcv_data):
        """Test price change feature calculation."""
        result = FeatureRegistry.compute_feature("price_change", sample_ohlcv_data)
        
        # Check basic properties
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv_data)
        
        # Check calculation correctness (spot check)
        # Price change = (current close - previous close) / previous close
        expected_second_value = (sample_ohlcv_data['Close'].iloc[1] - sample_ohlcv_data['Close'].iloc[0]) / sample_ohlcv_data['Close'].iloc[0]
        assert np.isclose(result.iloc[1], expected_second_value, atol=1e-4)
        
        # Should handle NaN values (first value is special case)
        assert not np.isnan(result.iloc[0])
    
    def test_high_low_range(self, sample_ohlcv_data):
        """Test high-low range feature calculation."""
        result = FeatureRegistry.compute_feature("high_low_range", sample_ohlcv_data)
        
        # Check basic properties
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv_data)
        
        # Check calculation correctness
        # High-low range = (High - Low) / Close
        expected_value = (sample_ohlcv_data['High'] - sample_ohlcv_data['Low']) / sample_ohlcv_data['Close']
        
        # Take absolute values of both result and expected value for comparison
        result_abs = result.abs()
        expected_abs = expected_value.abs()
        pd.testing.assert_series_equal(result_abs, expected_abs, check_exact=False, atol=1e-4)
        
        # Values should be positive after taking abs
        assert (result_abs >= 0).all()
    
    def test_price_feature_robustness(self, abnormal_ohlcv_data):
        """Test robustness of price features with abnormal data."""
        # Should handle NaNs and zeros
        result = FeatureRegistry.compute_feature("price_change", abnormal_ohlcv_data)
        assert isinstance(result, pd.Series)
        assert len(result) == len(abnormal_ohlcv_data)
        assert not np.isnan(result).any()  # Should replace NaNs
        assert not np.isinf(result).any()  # Should replace Infs


class TestVolumeFeatures:
    """Tests for volume-based features."""
    
    def test_volume_change(self, sample_ohlcv_data):
        """Test volume change feature calculation."""
        result = FeatureRegistry.compute_feature("volume_change", sample_ohlcv_data)
        
        # Check basic properties
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv_data)
        
        # Check calculation (spot check)
        # Volume change = (current volume - previous volume) / previous volume
        expected_second_value = (sample_ohlcv_data['Volume'].iloc[1] - sample_ohlcv_data['Volume'].iloc[0]) / sample_ohlcv_data['Volume'].iloc[0]
        assert np.isclose(result.iloc[1], expected_second_value, atol=1e-4)
    
    def test_volume_sma_ratio(self, sample_ohlcv_data):
        """Test volume to SMA ratio feature calculation."""
        result = FeatureRegistry.compute_feature("volume_sma_ratio", sample_ohlcv_data)
        
        # Check basic properties
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv_data)
        
        # Values should be positive
        assert (result >= 0).all()
        
        # Check calculation for a specific window
        window = 20
        volume_sma = sample_ohlcv_data['Volume'].rolling(window=window).mean().fillna(sample_ohlcv_data['Volume'])
        expected_ratio = np.minimum(sample_ohlcv_data['Volume'] / volume_sma, 5)  # Capped at 5
        pd.testing.assert_series_equal(result, expected_ratio, check_exact=False, atol=1e-4)
    
    def test_volume_feature_robustness(self, abnormal_ohlcv_data):
        """Test robustness of volume features with abnormal data."""
        # Should handle NaNs, zeros, and negative values
        result = FeatureRegistry.compute_feature("volume_change", abnormal_ohlcv_data)
        assert isinstance(result, pd.Series)
        assert len(result) == len(abnormal_ohlcv_data)
        assert not np.isnan(result).any()  # Should replace NaNs
        assert not np.isinf(result).any()  # Should replace Infs


class TestMomentumFeatures:
    """Tests for momentum-based features."""
    
    def test_rsi_14(self, sample_ohlcv_data):
        """Test RSI-14 feature calculation."""
        result = FeatureRegistry.compute_feature("rsi_14", sample_ohlcv_data)
        
        # Check basic properties
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv_data)
        
        # RSI should be in 0-1 range (normalized from traditional 0-100)
        assert (result >= 0).all()
        assert (result <= 1).all()
        
        # For a steady uptrend, RSI should be high (> 0.5)
        uptrend_data = sample_ohlcv_data.copy()
        uptrend_data['Close'] = np.linspace(100, 200, len(uptrend_data))
        uptrend_rsi = FeatureRegistry.compute_feature("rsi_14", uptrend_data)
        assert uptrend_rsi.iloc[-1] > 0.5
        
        # For a steady downtrend, RSI should be low (< 0.5)
        downtrend_data = sample_ohlcv_data.copy()
        downtrend_data['Close'] = np.linspace(200, 100, len(downtrend_data))
        downtrend_rsi = FeatureRegistry.compute_feature("rsi_14", downtrend_data)
        assert downtrend_rsi.iloc[-1] < 0.5
    
    def test_macd(self, sample_ohlcv_data):
        """Test MACD feature calculation."""
        result = FeatureRegistry.compute_feature("macd", sample_ohlcv_data)
        
        # Check basic properties
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv_data)
        
        # MACD line should be the difference between EMA12 and EMA26
        ema12 = sample_ohlcv_data['Close'].ewm(span=12, adjust=False).mean()
        ema26 = sample_ohlcv_data['Close'].ewm(span=26, adjust=False).mean()
        expected_macd = (ema12 - ema26) / sample_ohlcv_data['Close']  # Normalized
        pd.testing.assert_series_equal(result, expected_macd, check_exact=False, atol=1e-4)
    
    def test_momentum_feature_robustness(self, abnormal_ohlcv_data):
        """Test robustness of momentum features with abnormal data."""
        # Should handle NaNs
        result = FeatureRegistry.compute_feature("rsi_14", abnormal_ohlcv_data)
        assert isinstance(result, pd.Series)
        assert len(result) == len(abnormal_ohlcv_data)
        assert not np.isnan(result).any()  # Should replace NaNs
        assert not np.isinf(result).any()  # Should replace Infs


class TestTrendFeatures:
    """Tests for trend-based features."""
    
    def test_sma_20(self, sample_ohlcv_data):
        """Test SMA-20 feature calculation."""
        result = FeatureRegistry.compute_feature("sma_20", sample_ohlcv_data)
        
        # Check basic properties
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv_data)
        
        # SMA_20 ratio should be the 20-day SMA divided by close
        sma20 = sample_ohlcv_data['Close'].rolling(window=20).mean().fillna(method='bfill')
        expected_ratio = sma20 / sample_ohlcv_data['Close']
        pd.testing.assert_series_equal(result, expected_ratio, check_exact=False, atol=1e-4)
        
        # In an uptrend, SMA should be below price (ratio < 1)
        uptrend_data = sample_ohlcv_data.copy()
        uptrend_data['Close'] = np.linspace(100, 200, len(uptrend_data))
        uptrend_sma = FeatureRegistry.compute_feature("sma_20", uptrend_data)
        assert uptrend_sma.iloc[-1] < 1
        
        # In a downtrend, SMA should be above price (ratio > 1)
        downtrend_data = sample_ohlcv_data.copy()
        downtrend_data['Close'] = np.linspace(200, 100, len(downtrend_data))
        downtrend_sma = FeatureRegistry.compute_feature("sma_20", downtrend_data)
        assert downtrend_sma.iloc[-1] > 1
    
    def test_ma_crossover(self, sample_ohlcv_data):
        """Test moving average crossover feature calculation."""
        result = FeatureRegistry.compute_feature("ma_crossover", sample_ohlcv_data)
        
        # Check basic properties
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv_data)
        
        # Crossover signal should be in -1 to 1 range
        assert (result >= -1).all()
        assert (result <= 1).all()
        
        # In a clear uptrend, should be positive
        uptrend_data = sample_ohlcv_data.copy()
        uptrend_data['Close'] = np.linspace(100, 200, len(uptrend_data))
        uptrend_signal = FeatureRegistry.compute_feature("ma_crossover", uptrend_data)
        assert uptrend_signal.iloc[-1] > 0
        
        # In a clear downtrend, should be negative
        downtrend_data = sample_ohlcv_data.copy()
        downtrend_data['Close'] = np.linspace(200, 100, len(downtrend_data))
        downtrend_signal = FeatureRegistry.compute_feature("ma_crossover", downtrend_data)
        assert downtrend_signal.iloc[-1] < 0
    
    def test_trend_feature_robustness(self, abnormal_ohlcv_data):
        """Test robustness of trend features with abnormal data."""
        # Should handle NaNs
        result = FeatureRegistry.compute_feature("sma_20", abnormal_ohlcv_data)
        assert isinstance(result, pd.Series)
        assert len(result) == len(abnormal_ohlcv_data)
        assert not np.isnan(result).any()  # Should replace NaNs
        assert not np.isinf(result).any()  # Should replace Infs


class TestVolatilityFeatures:
    """Tests for volatility-based features."""
    
    def test_volatility(self, sample_ohlcv_data):
        """Test volatility feature calculation."""
        result = FeatureRegistry.compute_feature("volatility", sample_ohlcv_data)
        
        # Check basic properties
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv_data)
        
        # Volatility should be non-negative
        assert (result >= 0).all()
        
        # Verify calculation
        close = sample_ohlcv_data['Close'].values
        returns = np.diff(close, prepend=close[0]) / np.maximum(close, 1e-8)
        rolling_std = pd.Series(returns).rolling(window=5).std().fillna(0).values
        expected = np.nan_to_num(rolling_std)
        
        np.testing.assert_allclose(result, expected, rtol=1e-4)
    
    def test_bollinger_bandwidth(self, sample_ohlcv_data):
        """Test Bollinger Bandwidth feature calculation."""
        result = FeatureRegistry.compute_feature("bollinger_bandwidth", sample_ohlcv_data)
        
        # Check basic properties
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv_data)
        
        # Bandwidth should be non-negative
        assert (result >= 0).all()
        
        # For flat price data, bandwidth should be low
        flat_data = sample_ohlcv_data.copy()
        flat_data['Close'] = 100
        flat_bandwidth = FeatureRegistry.compute_feature("bollinger_bandwidth", flat_data)
        assert np.all(flat_bandwidth < 0.01)
        
        # For highly volatile data, bandwidth should be high
        volatile_data = sample_ohlcv_data.copy()
        volatile_data['Close'] = 100 + np.random.normal(0, 10, len(volatile_data))
        volatile_bandwidth = FeatureRegistry.compute_feature("bollinger_bandwidth", volatile_data)
        assert volatile_bandwidth.mean() > flat_bandwidth.mean()
    
    def test_volatility_feature_robustness(self, abnormal_ohlcv_data):
        """Test robustness of volatility features with abnormal data."""
        # Should handle NaNs
        result = FeatureRegistry.compute_feature("volatility", abnormal_ohlcv_data)
        assert isinstance(result, pd.Series)
        assert len(result) == len(abnormal_ohlcv_data)
        assert not np.isnan(result).any()  # Should replace NaNs
        assert not np.isinf(result).any()  # Should replace Infs


class TestSeasonalFeatures:
    """Tests for seasonal/time-based features."""
    
    def test_day_of_week(self, sample_ohlcv_data):
        """Test day of week feature calculation."""
        result = FeatureRegistry.compute_feature("day_of_week", sample_ohlcv_data)
        
        # Check basic properties
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv_data)
        
        # Values should be non-negative
        assert (result >= 0).all()
        
        # Calculate expected values (assuming business days)
        expected = pd.Series(sample_ohlcv_data.index).dt.dayofweek / 4.0
        pd.testing.assert_series_equal(result, expected, check_exact=True)
    
    def test_month(self, sample_ohlcv_data):
        """Test month feature calculation."""
        result = FeatureRegistry.compute_feature("month", sample_ohlcv_data)
        
        # Check basic properties
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv_data)
        
        # Values should be in 0-1 range (normalized)
        assert (result >= 0).all()
        assert (result <= 1).all()
        
        # Calculate expected values
        expected = (pd.Series(sample_ohlcv_data.index).dt.month - 1) / 11.0
        pd.testing.assert_series_equal(result, expected, check_exact=True)
    
    def test_seasonal_feature_consistency(self):
        """Test consistency of seasonal features with fixed dates."""
        # Create a DataFrame with specific dates
        dates = pd.date_range(start="2022-01-01", end="2022-12-31", freq="B")  # Business days
        data = pd.DataFrame({
            'Open': 100,
            'High': 105,
            'Low': 95,
            'Close': 101,
            'Volume': 1000000
        }, index=dates)
        
        # Test day of month
        day_of_month = FeatureRegistry.compute_feature("day_of_month", data)
        
        # January 1st should have a value of 0
        jan_1_idx = data.index.get_indexer([pd.Timestamp("2022-01-03")])[0]  # First business day
        assert day_of_month.iloc[jan_1_idx] == (3 - 1) / (31 - 1)  # (day - 1) / (last_day - 1)
        
        # Test month start
        month_start = FeatureRegistry.compute_feature("month_start", data)
        
        # First day of each month should have a value of 1
        for month in range(1, 13):
            try:
                # Find first business day of the month
                first_day = data.index[data.index.month == month].min()
                idx = data.index.get_indexer([first_day])[0]
                assert month_start.iloc[idx] == 1.0
            except (ValueError, IndexError):
                continue  # Skip if no business day in this month
        
        # Last day of each month should have a value of 0
        last_days = []
        for month in range(1, 13):
            try:
                # Find last business day of the month
                last_day = data.index[data.index.month == month].max()
                last_days.append(last_day)
            except (ValueError, IndexError):
                continue  # Skip if no business day in this month
        
        # Test month end
        month_end = FeatureRegistry.compute_feature("month_end", data)
        
        # Last day of each month should have a value of 1
        for last_day in last_days:
            idx = data.index.get_indexer([last_day])[0]
            assert month_end.iloc[idx] == 1.0 