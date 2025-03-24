"""
Tests for the Yahoo data fetcher module.
"""
import os
import json
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, mock_open
from src.data.yahoo_data_fetcher import YahooDataFetcher


class TestYahooDataFetcher:
    """Test cases for the YahooDataFetcher class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Create temp directory for cached data
        os.makedirs('tests/test_cache', exist_ok=True)
        
        # Init test parameters
        self.symbol = 'AAPL'
        self.start_date = '2020-01-01'
        self.end_date = '2020-01-31'
        
        # Create fetcher with custom cache directory
        self.fetcher = YahooDataFetcher()
        self.fetcher.cache_dir = 'tests/test_cache'
        
        # Sample data for tests
        self.sample_dates = pd.date_range(start=self.start_date, end=self.end_date)
        self.sample_data = pd.DataFrame({
            'Date': self.sample_dates,
            'Open': np.random.uniform(100, 150, len(self.sample_dates)),
            'High': np.random.uniform(120, 170, len(self.sample_dates)),
            'Low': np.random.uniform(90, 140, len(self.sample_dates)),
            'Close': np.random.uniform(100, 160, len(self.sample_dates)),
            'Volume': np.random.randint(1000000, 10000000, len(self.sample_dates)),
            'Adj Close': np.random.uniform(100, 160, len(self.sample_dates))
        })
    
    def teardown_method(self):
        """Clean up after tests"""
        # Clean up cache files
        cache_file = os.path.join('tests/test_cache', f"{self.symbol}_{self.start_date.replace('-', '')}_{self.end_date.replace('-', '')}.csv")
        if os.path.exists(cache_file):
            os.remove(cache_file)
    
    def test_initialization(self):
        """Test fetcher initialization"""
        assert self.fetcher.cache_dir == 'tests/test_cache'
        assert self.fetcher.max_retries == 3
        assert self.fetcher.retry_delay == 2
    
    @patch('yfinance.Ticker')
    def test_fetch_data_success(self, mock_ticker):
        """Test successful data fetching"""
        # Mock yfinance direct approach
        mock_ticker_instance = MagicMock()
        ticker_data = self.sample_data.copy().set_index('Date')
        mock_ticker_instance.history.return_value = ticker_data
        mock_ticker.return_value = mock_ticker_instance
        
        # Call fetch_data
        result = self.fetcher.fetch_data(
            symbol=self.symbol,
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        # Check that yfinance.Ticker was called
        mock_ticker.assert_called_once_with(self.symbol)
        mock_ticker_instance.history.assert_called_once()
        
        # Check returned data
        assert isinstance(result, pd.DataFrame)
        assert 'Date' in result.columns or result.index.name == 'Date'
        assert 'Open' in result.columns
        assert 'Close' in result.columns
    
    @patch('yfinance.Ticker')
    @patch('pandas.DataFrame.to_csv')
    @patch('pandas.read_csv')
    def test_fetch_data_caching(self, mock_read_csv, mock_to_csv, mock_ticker):
        """Test data caching mechanism"""
        # Setup for direct yfinance approach
        mock_ticker_instance = MagicMock()
        ticker_data = self.sample_data.copy().set_index('Date')
        mock_ticker_instance.history.return_value = ticker_data
        mock_ticker.return_value = mock_ticker_instance
        
        # Mock os.path.exists to return False initially, then True
        with patch('os.path.exists') as mock_exists:
            mock_exists.side_effect = [False, True]  # First call returns False, second call returns True
            
            # Mock read_csv for second call
            cache_data = self.sample_data.copy()
            mock_read_csv.return_value = cache_data
            
            # First call - should fetch from yfinance
            result1 = self.fetcher.fetch_data(
                symbol=self.symbol,
                start_date=self.start_date,
                end_date=self.end_date
            )
            
            # Check that yfinance was called
            mock_ticker.assert_called_once()
            mock_ticker_instance.history.assert_called_once()
            
            # Reset mocks to check second call
            mock_ticker.reset_mock()
            mock_ticker_instance.history.reset_mock()
            
            # Second call - should use cache
            result2 = self.fetcher.fetch_data(
                symbol=self.symbol,
                start_date=self.start_date,
                end_date=self.end_date
            )
            
            # Check that yfinance was not called again
            mock_ticker.assert_not_called()
            
            # Check read_csv was called
            mock_read_csv.assert_called_once()
    
    @patch('yfinance.Ticker')
    @patch('src.data.synthetic_data_fetcher.SyntheticDataFetcher')
    @patch('time.sleep')  # Mock sleep to avoid waiting
    def test_fetch_data_error_handling(self, mock_sleep, mock_synthetic_fetcher, mock_ticker):
        """Test error handling during data fetching"""
        # Mock yfinance.Ticker to fail
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.side_effect = Exception("YFinance Error")
        mock_ticker.return_value = mock_ticker_instance
        
        # Mock synthetic data fetcher
        mock_synthetic_instance = MagicMock()
        mock_synthetic_instance.fetch_data.return_value = self.sample_data
        mock_synthetic_fetcher.return_value = mock_synthetic_instance
        
        # Mock os.path.exists to return False for all calls
        with patch('os.path.exists', return_value=False):
            # Call fetch_data (should fall back to synthetic data)
            result = self.fetcher.fetch_data(
                symbol=self.symbol,
                start_date=self.start_date,
                end_date=self.end_date
            )
            
            # Check that ticker.history was attempted at least once
            mock_ticker_instance.history.assert_called()
            
            # Check that sleep was called for retries
            assert mock_sleep.call_count > 0
            
            # Check that synthetic data was generated as a fallback
            mock_synthetic_instance.fetch_data.assert_called_once_with(
                self.symbol, self.start_date, self.end_date
            )
            
            # Check result contains data
            assert isinstance(result, pd.DataFrame)
            assert not result.empty
    
    @patch('yfinance.Ticker')
    @patch('time.sleep')  # Mock sleep to avoid waiting
    def test_fetch_data_retries(self, mock_sleep, mock_ticker):
        """Test retry mechanism during data fetching"""
        # Mock yfinance.Ticker with success
        mock_ticker_instance = MagicMock()
        ticker_data = self.sample_data.copy().set_index('Date')
        mock_ticker_instance.history.return_value = ticker_data
        mock_ticker.return_value = mock_ticker_instance
        
        # Call fetch_data
        result = self.fetcher.fetch_data(
            symbol=self.symbol,
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        # Check that sleep was called at least once (for retries)
        assert mock_sleep.call_count > 0
        
        # Check that yfinance direct approach was successful
        mock_ticker.assert_called_once_with(self.symbol)
        mock_ticker_instance.history.assert_called_once()
        
        # Check result contains data
        assert not result.empty
        assert 'Open' in result.columns
        assert 'Close' in result.columns
    
    def test_add_technical_indicators(self):
        """Test adding technical indicators"""
        # Create a sample dataframe
        df = self.sample_data.copy()
        
        # Call the method directly
        result = self.fetcher.add_technical_indicators(df)
        
        # Verify technical indicators were added
        assert 'SMA_5' in result.columns
        assert 'SMA_20' in result.columns
        assert 'EMA_5' in result.columns
        assert 'EMA_20' in result.columns
        assert 'RSI_14' in result.columns 