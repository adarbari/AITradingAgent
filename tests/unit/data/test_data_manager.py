"""
Tests for the DataManager class
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, Mock, MagicMock
from datetime import datetime, timedelta

from src.data import DataManager


@pytest.fixture
def mock_data_fetcher():
    """Create a mock data fetcher"""
    fetcher = MagicMock()
    
    # Create sample market data
    dates = pd.date_range(start="2023-01-01", periods=20, freq="D")
    prices = np.linspace(100, 150, 20) + np.random.normal(0, 5, 20)
    
    market_data = pd.DataFrame({
        'Open': prices,
        'High': prices * 1.02,
        'Low': prices * 0.98,
        'Close': prices,
        'Volume': np.random.randint(1000, 10000, 20)
    }, index=dates)
    
    # Configure the mock to return the sample data
    fetcher.fetch_data.return_value = market_data
    
    # Configure add_technical_indicators to add some indicators
    def add_indicators(df):
        df = df.copy()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['RSI_14'] = 50 + np.random.normal(0, 10, len(df))  # Dummy RSI values
        return df
    
    fetcher.add_technical_indicators.side_effect = add_indicators
    
    # Configure sentiment data
    sentiment_data = pd.DataFrame({
        'Sentiment_Score': np.random.uniform(-1, 1, 20),
        'Article_Count': np.random.randint(1, 20, 20)
    }, index=dates)
    
    fetcher.fetch_sentiment_data.return_value = sentiment_data
    
    # Configure economic data
    economic_data = pd.DataFrame({
        'GDP': np.random.normal(3, 0.5, 20),
        'CPI': np.random.normal(2, 0.3, 20),
        'Unemployment': np.random.normal(5, 0.5, 20)
    }, index=dates)
    
    fetcher.fetch_economic_data.return_value = economic_data
    
    # Configure social sentiment data
    social_data = pd.DataFrame({
        'Social_Score': np.random.uniform(-1, 1, 20),
        'Tweet_Count': np.random.randint(10, 200, 20)
    }, index=dates)
    
    fetcher.fetch_social_sentiment.return_value = social_data
    
    return fetcher


@pytest.fixture
def mock_data_fetcher_factory():
    """Create a mock data fetcher factory"""
    factory = MagicMock()
    return factory


class TestDataManager:
    """Test cases for the DataManager class"""
    
    @patch('src.data.data_manager.DataFetcherFactory')
    def test_initialization(self, mock_factory, mock_data_fetcher):
        """Test initialization with various sources"""
        mock_factory.create_data_fetcher.return_value = mock_data_fetcher
        
        # Initialize with only market data
        data_manager = DataManager(market_data_source="yahoo")
        
        assert "market" in data_manager.data_sources
        assert mock_factory.create_data_fetcher.call_count == 1
        
        # Reset mock for next test
        mock_factory.reset_mock()
        
        # Initialize with all data sources
        data_manager = DataManager(
            market_data_source="yahoo",
            news_data_source="news",
            economic_data_source="economic",
            social_data_source="social",
            cache_data=True,
            verbose=2
        )
        
        assert "market" in data_manager.data_sources
        assert "news" in data_manager.data_sources
        assert "economic" in data_manager.data_sources
        assert "social" in data_manager.data_sources
        assert data_manager.cache is not None
        assert data_manager.verbose == 2
        assert mock_factory.create_data_fetcher.call_count == 4
    
    @patch('src.data.data_manager.DataFetcherFactory')
    def test_get_market_data(self, mock_factory, mock_data_fetcher):
        """Test getting market data"""
        mock_factory.create_data_fetcher.return_value = mock_data_fetcher
        
        data_manager = DataManager(market_data_source="yahoo")
        
        # Get data without indicators
        data = data_manager.get_market_data(
            symbol="AAPL",
            start_date="2023-01-01",
            end_date="2023-01-31",
            include_indicators=False
        )
        
        assert data is not None
        assert isinstance(data, pd.DataFrame)
        assert mock_data_fetcher.fetch_data.called
        assert not mock_data_fetcher.add_technical_indicators.called
        
        # Reset counters
        mock_data_fetcher.fetch_data.reset_mock()
        mock_data_fetcher.add_technical_indicators.reset_mock()
        
        # Get data with indicators
        data = data_manager.get_market_data(
            symbol="AAPL",
            start_date="2023-01-01",
            end_date="2023-01-31",
            include_indicators=True
        )
        
        assert data is not None
        assert isinstance(data, pd.DataFrame)
        assert mock_data_fetcher.fetch_data.called
        assert mock_data_fetcher.add_technical_indicators.called
    
    @patch('src.data.data_manager.DataFetcherFactory')
    def test_get_sentiment_data(self, mock_factory, mock_data_fetcher):
        """Test getting sentiment data"""
        mock_factory.create_data_fetcher.return_value = mock_data_fetcher
        
        # Initialize with news data source
        data_manager = DataManager(
            market_data_source="yahoo",
            news_data_source="news"
        )
        
        # Get sentiment data
        data = data_manager.get_sentiment_data(
            symbol="AAPL",
            start_date="2023-01-01",
            end_date="2023-01-31"
        )
        
        assert data is not None
        assert isinstance(data, pd.DataFrame)
        assert mock_data_fetcher.fetch_sentiment_data.called
        
        # Test when no news source is configured
        data_manager = DataManager(market_data_source="yahoo")
        
        data = data_manager.get_sentiment_data(
            symbol="AAPL",
            start_date="2023-01-01",
            end_date="2023-01-31"
        )
        
        assert data is None
    
    @patch('src.data.data_manager.DataFetcherFactory')
    def test_get_economic_data(self, mock_factory, mock_data_fetcher):
        """Test getting economic data"""
        mock_factory.create_data_fetcher.return_value = mock_data_fetcher
        
        # Initialize with economic data source
        data_manager = DataManager(
            market_data_source="yahoo",
            economic_data_source="economic"
        )
        
        # Get economic data
        data = data_manager.get_economic_data(
            indicators=["GDP", "CPI", "Unemployment"],
            start_date="2023-01-01",
            end_date="2023-01-31"
        )
        
        assert data is not None
        assert isinstance(data, pd.DataFrame)
        assert mock_data_fetcher.fetch_economic_data.called
        
        # Test when no economic source is configured
        data_manager = DataManager(market_data_source="yahoo")
        
        data = data_manager.get_economic_data(
            indicators=["GDP", "CPI", "Unemployment"],
            start_date="2023-01-01",
            end_date="2023-01-31"
        )
        
        assert data is None
    
    @patch('src.data.data_manager.DataFetcherFactory')
    def test_get_social_sentiment(self, mock_factory, mock_data_fetcher):
        """Test getting social sentiment data"""
        mock_factory.create_data_fetcher.return_value = mock_data_fetcher
        
        # Initialize with social data source
        data_manager = DataManager(
            market_data_source="yahoo",
            social_data_source="social"
        )
        
        # Get social sentiment data
        data = data_manager.get_social_sentiment(
            symbol="AAPL",
            start_date="2023-01-01",
            end_date="2023-01-31"
        )
        
        assert data is not None
        assert isinstance(data, pd.DataFrame)
        assert mock_data_fetcher.fetch_social_sentiment.called
        
        # Test when no social source is configured
        data_manager = DataManager(market_data_source="yahoo")
        
        data = data_manager.get_social_sentiment(
            symbol="AAPL",
            start_date="2023-01-01",
            end_date="2023-01-31"
        )
        
        assert data is None
    
    @patch('src.data.data_manager.DataFetcherFactory')
    def test_get_correlation_data(self, mock_factory, mock_data_fetcher):
        """Test getting correlation data"""
        mock_factory.create_data_fetcher.return_value = mock_data_fetcher
        
        # Sample correlation matrix
        corr_matrix = pd.DataFrame({
            'AAPL': [1.0, 0.7, 0.5],
            'MSFT': [0.7, 1.0, 0.6],
            'GOOG': [0.5, 0.6, 1.0]
        }, index=['AAPL', 'MSFT', 'GOOG'])
        
        mock_data_fetcher.fetch_correlation_data.return_value = corr_matrix
        
        # Initialize with market data source
        data_manager = DataManager(market_data_source="yahoo")
        
        # Get correlation data using specialized method
        data = data_manager.get_correlation_data(
            symbols=["AAPL", "MSFT", "GOOG"],
            start_date="2023-01-01",
            end_date="2023-01-31"
        )
        
        assert data is not None
        assert isinstance(data, pd.DataFrame)
        assert mock_data_fetcher.fetch_correlation_data.called
        
        # Test when specialized method is not available
        mock_data_fetcher.fetch_correlation_data.return_value = None
        
        # Configure get_market_data to return different data for each symbol
        def mock_get_market_data(symbol, *args, **kwargs):
            dates = pd.date_range(start="2023-01-01", periods=20, freq="D")
            if symbol == "AAPL":
                prices = np.linspace(100, 150, 20)
            elif symbol == "MSFT":
                prices = np.linspace(200, 250, 20)
            else:
                prices = np.linspace(150, 200, 20)
            
            return pd.DataFrame({'Close': prices}, index=dates)
        
        # Replace the real method with our mock
        data_manager.get_market_data = mock_get_market_data
        
        # Get correlation data using manual calculation
        data = data_manager.get_correlation_data(
            symbols=["AAPL", "MSFT", "GOOG"],
            start_date="2023-01-01",
            end_date="2023-01-31"
        )
        
        assert data is not None
        assert isinstance(data, pd.DataFrame)
        assert data.shape == (3, 3)  # 3x3 correlation matrix
    
    @patch('src.data.data_manager.DataFetcherFactory')
    def test_prepare_data_for_agent(self, mock_factory, mock_data_fetcher):
        """Test preparing data for agent"""
        mock_factory.create_data_fetcher.return_value = mock_data_fetcher
        
        # Initialize with all data sources
        data_manager = DataManager(
            market_data_source="yahoo",
            news_data_source="news",
            economic_data_source="economic",
            social_data_source="social"
        )
        
        # Prepare data with all sources
        data = data_manager.prepare_data_for_agent(
            symbol="AAPL",
            start_date="2023-01-01",
            end_date="2023-01-31",
            include_sentiment=True,
            include_economic=True,
            include_social=True
        )
        
        assert data is not None
        assert isinstance(data, dict)
        assert "market" in data
        assert "sentiment" in data
        assert "economic" in data
        assert "social" in data
        
        # Test when market data is not available
        # Create a fresh mock for this test case to ensure clean state
        new_mock = MagicMock()
        new_mock.fetch_data.return_value = None
        mock_factory.create_data_fetcher.return_value = new_mock
        
        # Create a new data manager with the new mock
        new_data_manager = DataManager(
            market_data_source="yahoo"
        )
        
        # This should now return None because market data is not available
        data = new_data_manager.prepare_data_for_agent(
            symbol="AAPL",
            start_date="2023-01-01",
            end_date="2023-01-31"
        )
        
        assert data is None
    
    @patch('src.data.data_manager.DataFetcherFactory')
    def test_caching(self, mock_factory, mock_data_fetcher):
        """Test data caching"""
        mock_factory.create_data_fetcher.return_value = mock_data_fetcher
        
        # Initialize with caching enabled
        data_manager = DataManager(
            market_data_source="yahoo",
            cache_data=True
        )
        
        # Get data first time (should cache)
        data_manager.get_market_data(
            symbol="AAPL",
            start_date="2023-01-01",
            end_date="2023-01-31",
            include_indicators=True
        )
        
        assert mock_data_fetcher.fetch_data.call_count == 1
        
        # Get data second time (should use cache)
        data_manager.get_market_data(
            symbol="AAPL",
            start_date="2023-01-01",
            end_date="2023-01-31",
            include_indicators=True
        )
        
        # Fetcher should not be called again
        assert mock_data_fetcher.fetch_data.call_count == 1
        
        # Test with caching disabled
        data_manager = DataManager(
            market_data_source="yahoo",
            cache_data=False
        )
        
        mock_data_fetcher.fetch_data.reset_mock()
        
        # Get data first time
        data_manager.get_market_data(
            symbol="AAPL",
            start_date="2023-01-01",
            end_date="2023-01-31"
        )
        
        assert mock_data_fetcher.fetch_data.call_count == 1
        
        # Get data second time (should fetch again)
        data_manager.get_market_data(
            symbol="AAPL",
            start_date="2023-01-01",
            end_date="2023-01-31"
        )
        
        assert mock_data_fetcher.fetch_data.call_count == 2
    
    @patch('src.data.data_manager.DataFetcherFactory')
    def test_error_handling(self, mock_factory, mock_data_fetcher):
        """Test error handling"""
        mock_factory.create_data_fetcher.return_value = mock_data_fetcher
        
        # Configure fetcher to raise an exception
        mock_data_fetcher.fetch_data.side_effect = Exception("Test error")
        
        data_manager = DataManager(
            market_data_source="yahoo",
            verbose=1
        )
        
        # Get data should handle the exception
        data = data_manager.get_market_data(
            symbol="AAPL",
            start_date="2023-01-01",
            end_date="2023-01-31"
        )
        
        assert data is None 