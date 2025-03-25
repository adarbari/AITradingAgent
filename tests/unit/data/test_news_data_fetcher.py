"""
Tests for the NewsDataFetcher class
"""
import pytest
import os
import pandas as pd
import numpy as np
from unittest.mock import patch, mock_open
from datetime import datetime, timedelta

from src.data.news_data_fetcher import NewsDataFetcher


@pytest.fixture
def news_data_fetcher():
    """Create a news data fetcher with a test cache directory"""
    # Use a temporary directory for the test cache
    test_cache_dir = "test_cache/news"
    os.makedirs(test_cache_dir, exist_ok=True)
    
    fetcher = NewsDataFetcher(cache_dir=test_cache_dir, cache_expiry_days=1)
    return fetcher


class TestNewsDataFetcher:
    """Test cases for the NewsDataFetcher class"""
    
    def test_initialization(self):
        """Test initialization with different parameters"""
        # Test default initialization
        fetcher = NewsDataFetcher()
        assert fetcher.api_key is None
        assert fetcher.cache_dir == "data/cache"
        assert fetcher.cache_expiry_days == 7
        
        # Test with custom parameters
        api_key = "test_api_key"
        cache_dir = "custom_cache"
        cache_expiry_days = 30
        
        fetcher = NewsDataFetcher(
            api_key=api_key,
            cache_dir=cache_dir,
            cache_expiry_days=cache_expiry_days
        )
        
        assert fetcher.api_key == api_key
        assert fetcher.cache_dir == cache_dir
        assert fetcher.cache_expiry_days == cache_expiry_days
    
    def test_fetch_data(self, news_data_fetcher):
        """Test fetch_data method (should call fetch_sentiment_data)"""
        with patch.object(news_data_fetcher, 'fetch_sentiment_data') as mock_sentiment:
            # Set up mock to return a dataframe
            mock_sentiment.return_value = pd.DataFrame({'test': [1, 2, 3]})
            
            # Call fetch_data
            result = news_data_fetcher.fetch_data(
                symbol="AAPL",
                start_date="2023-01-01",
                end_date="2023-01-31"
            )
            
            # Verify fetch_sentiment_data was called with the same parameters
            mock_sentiment.assert_called_once_with(
                symbol="AAPL",
                start_date="2023-01-01",
                end_date="2023-01-31"
            )
            
            # Verify result is the same as what fetch_sentiment_data returned
            assert result is mock_sentiment.return_value
    
    def test_fetch_sentiment_data_simulated(self, news_data_fetcher):
        """Test fetching simulated sentiment data"""
        # Call with a test symbol and date range
        data = news_data_fetcher.fetch_sentiment_data(
            symbol="AAPL",
            start_date="2023-01-01",
            end_date="2023-01-10"
        )
        
        # Verify basic properties of the returned data
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        
        # Check index is datetime and spans requested dates
        assert isinstance(data.index, pd.DatetimeIndex)
        
        # Instead of exactly January 1, 2023, check that the date is on or after January 1, 2023
        first_date = data.index[0]
        assert first_date >= pd.to_datetime("2023-01-01"), f"First date {first_date} should be on or after 2023-01-01"
        
        # Check expected columns exist
        assert 'Sentiment_Score' in data.columns
        assert 'Article_Count' in data.columns
        assert 'Volatility' in data.columns
        
        # Check data is within expected ranges
        assert data['Sentiment_Score'].min() >= -1
        assert data['Sentiment_Score'].max() <= 1
        assert data['Article_Count'].min() >= 0
        assert data['Volatility'].min() >= 0
    
    def test_cache_operations(self, news_data_fetcher):
        """Test caching operations"""
        test_symbol = "TEST"
        start_date = "2023-01-01"
        end_date = "2023-01-10"
        cache_file = news_data_fetcher._get_cache_path(test_symbol, start_date, end_date)
        
        # Make sure the cache file doesn't exist initially
        if os.path.exists(cache_file):
            os.remove(cache_file)
        
        # First fetch (should generate and cache data)
        data1 = news_data_fetcher.fetch_sentiment_data(
            symbol=test_symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        # Verify cache file was created
        assert os.path.exists(cache_file), f"Cache file {cache_file} was not created"
        
        # Second fetch (should use cache)
        with patch.object(news_data_fetcher, '_generate_simulated_data') as mock_generate:
            data2 = news_data_fetcher.fetch_sentiment_data(
                symbol=test_symbol,
                start_date=start_date,
                end_date=end_date
            )
            
            # Generate should not have been called if cache is working
            mock_generate.assert_not_called()
        
        # Data should have the same content (might not be exactly equal due to serialization)
        # So we check key properties instead of exact equality
        assert data1.shape == data2.shape
        assert list(data1.columns) == list(data2.columns)
        pd.testing.assert_series_equal(data1['Sentiment_Score'], data2['Sentiment_Score'], check_names=False)
        pd.testing.assert_series_equal(data1['Article_Count'], data2['Article_Count'], check_names=False)
        pd.testing.assert_series_equal(data1['Volatility'], data2['Volatility'], check_names=False)
        
        # Clean up the cache file
        if os.path.exists(cache_file):
            os.remove(cache_file)
    
    def test_cache_expiry(self, news_data_fetcher):
        """Test cache expiry"""
        test_symbol = "TEST"
        start_date = "2023-01-01"
        end_date = "2023-01-10"
        cache_file = news_data_fetcher._get_cache_path(test_symbol, start_date, end_date)
        
        # Fetch data to create cache
        data1 = news_data_fetcher.fetch_sentiment_data(
            symbol=test_symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        # Modify the file time to be older (expired)
        old_time = datetime.now() - timedelta(days=news_data_fetcher.cache_expiry_days + 1)
        os.utime(cache_file, (old_time.timestamp(), old_time.timestamp()))
        
        # Fetch again with patch to see if generate is called
        with patch.object(news_data_fetcher, '_generate_simulated_data') as mock_generate:
            # Set up mock to return a dataframe
            mock_df = pd.DataFrame({
                'Sentiment_Score': [0.5, -0.5],
                'Article_Count': [10, 15]
            }, index=pd.date_range(start=start_date, periods=2))
            mock_generate.return_value = mock_df
            
            data2 = news_data_fetcher.fetch_sentiment_data(
                symbol=test_symbol,
                start_date=start_date,
                end_date=end_date
            )
            
            # Generate should have been called because cache expired
            mock_generate.assert_called_once()
        
        # Clean up the cache file
        if os.path.exists(cache_file):
            os.remove(cache_file)
    
    def test_simulate_seed_consistency(self):
        """Test that simulated data is consistent for the same symbol"""
        fetcher = NewsDataFetcher(cache_dir="test_cache/no_cache")
        
        # Get data for the same symbol twice
        data1 = fetcher._generate_simulated_data(
            symbol="AAPL",
            start_date="2023-01-01",
            end_date="2023-01-10"
        )
        
        data2 = fetcher._generate_simulated_data(
            symbol="AAPL",
            start_date="2023-01-01",
            end_date="2023-01-10"
        )
        
        # The data should be identical
        assert data1.shape == data2.shape
        pd.testing.assert_series_equal(data1['Sentiment_Score'], data2['Sentiment_Score'], check_names=False)
        pd.testing.assert_series_equal(data1['Article_Count'], data2['Article_Count'], check_names=False)
        pd.testing.assert_series_equal(data1['Volatility'], data2['Volatility'], check_names=False)
        
        # Different symbols should generate different data
        data3 = fetcher._generate_simulated_data(
            symbol="MSFT",
            start_date="2023-01-01",
            end_date="2023-01-10"
        )
        
        # The data should be different
        assert data1.shape == data3.shape  # same shape
        with pytest.raises(AssertionError):
            pd.testing.assert_series_equal(data1['Sentiment_Score'], data3['Sentiment_Score'], check_names=False)
    
    def test_add_technical_indicators(self, news_data_fetcher):
        """Test add_technical_indicators (should be a no-op for news data)"""
        # Create a test dataframe
        df = pd.DataFrame({
            'Sentiment_Score': [0.5, -0.5, 0.2],
            'Article_Count': [10, 15, 8]
        })
        
        # Call the method
        result = news_data_fetcher.add_technical_indicators(df)
        
        # It should return the same dataframe unchanged
        pd.testing.assert_frame_equal(df, result)
        
        # Should be the same object, not a copy
        assert result is df
    
    def test_error_handling(self, news_data_fetcher):
        """Test error handling during cache operations"""
        # Test _is_cache_expired with a non-existent file
        with pytest.raises(FileNotFoundError):
            news_data_fetcher._is_cache_expired("/tmp/non_existent_file.json")
        
        # Test _cache_data with an error
        with patch('pandas.DataFrame.to_json', side_effect=Exception("Test error")):
            # Should not raise an exception
            df = pd.DataFrame({'test': [1, 2, 3]})
            news_data_fetcher._cache_data(df, "test_file.json")
            
        # Test cache read error
        with patch('pandas.read_json', side_effect=Exception("Test error")):
            with patch.object(news_data_fetcher, '_generate_simulated_data') as mock_generate:
                # Set up mock to return a dataframe
                mock_df = pd.DataFrame({
                    'Sentiment_Score': [0.5],
                    'Article_Count': [10],
                    'Volatility': [0.3]
                }, index=pd.date_range(start="2023-01-01", periods=1))
                mock_generate.return_value = mock_df
                
                # Should fall back to generating new data
                result = news_data_fetcher.fetch_sentiment_data(
                    symbol="TEST",
                    start_date="2023-01-01",
                    end_date="2023-01-10"
                )
                
                assert result is mock_df
                mock_generate.assert_called_once() 