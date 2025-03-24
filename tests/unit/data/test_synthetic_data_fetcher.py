"""
Tests for the SyntheticDataFetcher class
"""
import pytest
import pandas as pd
import numpy as np
import os
from datetime import datetime
from src.data import SyntheticDataFetcher


class TestSyntheticDataFetcher:
    """Test cases for the SyntheticDataFetcher class"""

    def test_initialization(self):
        """Test initialization of the fetcher"""
        fetcher = SyntheticDataFetcher()
        assert fetcher.base_dir == "data/synthetic"
        assert os.path.exists(fetcher.base_dir)

    def test_fetch_data(self):
        """Test fetching synthetic data"""
        fetcher = SyntheticDataFetcher()
        
        symbol = "TEST_SYMBOL"
        start_date = "2020-01-01"
        end_date = "2020-01-31"
        
        data = fetcher.fetch_data(symbol, start_date, end_date)
        
        # Check that the data is a pandas DataFrame
        assert isinstance(data, pd.DataFrame)
        
        # Check that the date range is correct
        assert data['Date'].min().strftime('%Y-%m-%d') == start_date
        assert data['Date'].max().strftime('%Y-%m-%d') <= end_date
        
        # Check that the data has the expected columns
        expected_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        assert all(col in data.columns for col in expected_columns)
        
        # Check that the data has at least one row
        assert len(data) > 0
        
        # Check that Open, High, Low, Close columns have sensible values
        assert (data['High'] >= data['Open']).all()
        assert (data['High'] >= data['Close']).all()
        assert (data['Low'] <= data['Open']).all()
        assert (data['Low'] <= data['Close']).all()
        assert (data['Volume'] >= 0).all()
        
        # Clean up the test file
        os.remove(os.path.join(fetcher.base_dir, f"{symbol}.csv"))

    def test_add_technical_indicators(self):
        """Test adding technical indicators to data"""
        fetcher = SyntheticDataFetcher()
        
        # Generate some test data
        symbol = "TEST_INDICATORS"
        start_date = "2020-01-01"
        end_date = "2020-02-28"  # Need enough data for the indicators
        
        data = fetcher.fetch_data(symbol, start_date, end_date)
        
        # Add technical indicators
        data_with_indicators = fetcher.add_technical_indicators(data)
        
        # Check that the data still has the original columns
        expected_original_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        assert all(col in data_with_indicators.columns for col in expected_original_columns)
        
        # Check that the technical indicators are added
        indicator_columns = [
            'SMA_20', 'EMA_20', 'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist',
            'Upper_Band', 'Middle_Band', 'Lower_Band', 'ATR_14', 'ADX_14'
        ]
        assert all(col in data_with_indicators.columns for col in indicator_columns)
        
        # Check that the indicators have non-NaN values (after initial periods)
        # SMA_20 needs 20 days of data, so we check after 20 days
        assert not data_with_indicators['SMA_20'].iloc[20:].isna().any()
        
        # RSI_14 needs 14+1 days of data
        assert not data_with_indicators['RSI_14'].iloc[15:].isna().any()
        
        # Clean up the test file
        os.remove(os.path.join(fetcher.base_dir, f"{symbol}.csv"))

    def test_prepare_data_for_agent(self):
        """Test preparing data for the agent"""
        fetcher = SyntheticDataFetcher()
        
        # Generate some test data
        symbol = "TEST_AGENT_DATA"
        start_date = "2020-01-01"
        end_date = "2020-03-31"  # Need enough data for the indicators and preparation
        
        data = fetcher.fetch_data(symbol, start_date, end_date)
        data_with_indicators = fetcher.add_technical_indicators(data)
        
        # Prepare data for agent
        prices, features = fetcher.prepare_data_for_agent(data_with_indicators)
        
        # Check that prices is a numpy array of the close prices
        assert isinstance(prices, np.ndarray)
        assert len(prices) == len(data_with_indicators.dropna())
        
        # Check that features is a numpy array with the right shape
        assert isinstance(features, np.ndarray)
        assert features.shape[0] == len(data_with_indicators.dropna())
        assert features.shape[1] > 0  # Should have at least one feature
        
        # Check that features are normalized
        # Most normalized features should be between -3 and 3 (allowing for some outliers)
        assert np.percentile(features, 1) > -5
        assert np.percentile(features, 99) < 5
        
        # Clean up the test file
        os.remove(os.path.join(fetcher.base_dir, f"{symbol}.csv"))

    def test_caching_via_csv(self):
        """Test that data is cached correctly via CSV files"""
        fetcher = SyntheticDataFetcher()
        
        symbol = "TEST_CACHE"
        start_date = "2020-01-01"
        end_date = "2020-01-31"
        
        # First fetch should create a CSV file
        data1 = fetcher.fetch_data(symbol, start_date, end_date)
        
        # Verify the CSV file exists
        csv_path = os.path.join(fetcher.base_dir, f"{symbol}.csv")
        assert os.path.exists(csv_path)
        
        # Second fetch should use the CSV file
        data2 = fetcher.fetch_data(symbol, start_date, end_date)
        
        # Data should be similar (not exactly identical due to technical indicators calculation)
        pd.testing.assert_frame_equal(
            data1[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']], 
            data2[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        )
        
        # Clean up the test file
        os.remove(csv_path) 