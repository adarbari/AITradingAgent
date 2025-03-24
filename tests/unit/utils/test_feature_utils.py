"""
Tests for the utility functions in feature_utils.py
"""
import pytest
import pandas as pd
import numpy as np
from src.utils.feature_utils import (
    prepare_features_from_indicators,
    prepare_robust_features,
    get_data,
    _generate_synthetic_data
)
from src.data.synthetic_data_fetcher import SyntheticDataFetcher
from unittest.mock import patch, MagicMock

# Sample data for tests
@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing"""
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'Open': np.random.normal(100, 5, 100),
        'High': np.random.normal(105, 5, 100),
        'Low': np.random.normal(95, 5, 100),
        'Close': np.random.normal(102, 5, 100),
        'Volume': np.random.randint(1000, 10000, 100),
        'Adj Close': np.random.normal(102, 5, 100),
    }, index=dates)
    data['Date'] = dates
    return data

@pytest.fixture
def sample_indicators(sample_data):
    """Add sample indicators to the data"""
    data = sample_data.copy()
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['RSI_14'] = 50 + np.random.normal(0, 10, 100)  # Simplified RSI
    data['MACD'] = np.random.normal(0, 1, 100)
    data['MACD_Signal'] = np.random.normal(0, 1, 100)
    data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']
    # Fill NaN values using newer methods (avoiding deprecation warnings)
    return data.bfill().ffill()

def test_prepare_features_from_indicators(sample_indicators):
    """Test that features can be prepared from pre-computed indicators"""
    # Remove the Date column which causes issues with normalization (Timedelta comparisons)
    indicators_without_date = sample_indicators.drop(columns=['Date'])
    
    features = prepare_features_from_indicators(indicators_without_date, expected_feature_count=10)
    
    # Check shape
    assert isinstance(features, pd.DataFrame)
    assert features.shape[0] == len(sample_indicators)
    assert features.shape[1] == 10  # Expected feature count
    
    # Check normalization
    assert features.min().min() >= -5
    assert features.max().max() <= 5

def test_prepare_robust_features(sample_data):
    """Test the robust feature preparation pipeline"""
    features = prepare_robust_features(sample_data, feature_count=15)
    
    # Check shape
    assert isinstance(features, np.ndarray)
    assert features.shape[0] == len(sample_data)
    assert features.shape[1] == 15  # Expected feature count
    
    # Check normalization
    assert np.min(features) >= -5
    assert np.max(features) <= 5

def test_get_data_yahoo():
    """Test the get_data function with yahoo source"""
    with patch('src.utils.feature_utils.YahooDataFetcher') as mock_fetcher_class:
        mock_fetcher = MagicMock()
        mock_fetcher_class.return_value = mock_fetcher
        
        # Mock the fetch_data_simple method
        mock_data = pd.DataFrame({
            'Date': pd.date_range(start='2020-01-01', periods=10),
            'Open': np.random.rand(10),
            'High': np.random.rand(10),
            'Low': np.random.rand(10),
            'Close': np.random.rand(10),
            'Volume': np.random.rand(10),
            'Adj Close': np.random.rand(10),
        })
        mock_fetcher.fetch_data_simple.return_value = mock_data
        
        result = get_data('AAPL', '2020-01-01', '2020-01-10', 'yahoo')
        
        # Check that the Yahoo fetcher was used
        mock_fetcher_class.assert_called_once()
        mock_fetcher.fetch_data_simple.assert_called_once_with('AAPL', '2020-01-01', '2020-01-10')
        
        # Check result
        assert result is mock_data

def test_get_data_synthetic():
    """Test the get_data function with synthetic source"""
    with patch('src.utils.feature_utils._generate_synthetic_data') as mock_generate:
        # Mock the generate function
        mock_data = pd.DataFrame({
            'Date': pd.date_range(start='2020-01-01', periods=10),
            'Open': np.random.rand(10),
            'High': np.random.rand(10),
            'Low': np.random.rand(10),
            'Close': np.random.rand(10),
            'Volume': np.random.rand(10),
            'Adj Close': np.random.rand(10),
        })
        mock_generate.return_value = mock_data
        
        params = {'initial_price': 200.0, 'volatility': 0.02}
        result = get_data('SYNTH', '2020-01-01', '2020-01-10', 'synthetic', params)
        
        # Check that the synthetic generator was used
        mock_generate.assert_called_once_with('SYNTH', '2020-01-01', '2020-01-10', params=params)
        
        # Check result
        assert result is mock_data

def test_get_data_fallback():
    """Test the fallback to synthetic when Yahoo fails"""
    with patch('src.utils.feature_utils.YahooDataFetcher') as mock_fetcher_class, \
         patch('src.utils.feature_utils._generate_synthetic_data') as mock_generate:
            
        # Mock Yahoo fetcher to raise exception
        mock_fetcher = MagicMock()
        mock_fetcher_class.return_value = mock_fetcher
        mock_fetcher.fetch_data_simple.side_effect = Exception("Yahoo API error")
        
        # Mock synthetic data generation
        mock_data = pd.DataFrame({
            'Date': pd.date_range(start='2020-01-01', periods=10),
            'Open': np.random.rand(10),
            'High': np.random.rand(10),
            'Low': np.random.rand(10),
            'Close': np.random.rand(10),
            'Volume': np.random.rand(10),
            'Adj Close': np.random.rand(10),
        })
        mock_generate.return_value = mock_data
        
        result = get_data('AAPL', '2020-01-01', '2020-01-10', 'yahoo')
        
        # Check that Yahoo was attempted
        mock_fetcher.fetch_data_simple.assert_called_once()
        
        # Check that synthetic generation was used as fallback
        mock_generate.assert_called_once()
        
        # Check result
        assert result is mock_data

@patch('src.data.synthetic_data_fetcher.SyntheticDataFetcher')
def test_generate_synthetic_data(mock_fetcher_class):
    """Test the _generate_synthetic_data function"""
    # Mock fetcher instance and its methods
    mock_fetcher = MagicMock()
    mock_fetcher_class.return_value = mock_fetcher
    
    # Create mock data to return
    mock_data = pd.DataFrame({
        'Date': pd.date_range(start='2020-01-01', periods=10),
        'Open': np.random.rand(10),
        'High': np.random.rand(10),
        'Low': np.random.rand(10),
        'Close': np.random.rand(10),
        'Volume': np.random.rand(10),
        'Adj Close': np.random.rand(10),
    })
    
    # Config for the mock tech indicators
    mock_data_with_indicators = mock_data.copy()
    mock_data_with_indicators['SMA_5'] = mock_data['Close'].rolling(window=5).mean()
    mock_data_with_indicators.fillna(0, inplace=True)
    
    # Set up the returns for mock methods
    mock_fetcher.fetch_data.return_value = mock_data
    mock_fetcher.add_technical_indicators.return_value = mock_data_with_indicators
    
    # Call the function we're testing
    result = _generate_synthetic_data('TEST', '2020-01-01', '2020-01-10')
    
    # Verify the SyntheticDataFetcher was instantiated
    mock_fetcher_class.assert_called_once()
    
    # Verify fetch_data was called with correct parameters
    mock_fetcher.fetch_data.assert_called_once_with('TEST', '2020-01-01', '2020-01-10')
    
    # Check that technical indicators were checked and added
    assert mock_fetcher.add_technical_indicators.called
    
    # Check the result
    assert result.equals(mock_data_with_indicators) 