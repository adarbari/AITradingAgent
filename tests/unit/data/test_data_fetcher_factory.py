"""
Tests for the DataFetcherFactory class
"""
import pytest
from src.data import DataFetcherFactory, SyntheticDataFetcher, YahooDataFetcher


class TestDataFetcherFactory:
    """Test cases for the DataFetcherFactory class"""

    def test_create_data_fetcher_synthetic(self):
        """Test creating a synthetic data fetcher"""
        fetcher = DataFetcherFactory.create_data_fetcher("synthetic")
        assert isinstance(fetcher, SyntheticDataFetcher)

    def test_create_data_fetcher_yahoo(self):
        """Test creating a Yahoo data fetcher"""
        fetcher = DataFetcherFactory.create_data_fetcher("yahoo")
        assert isinstance(fetcher, YahooDataFetcher)

    def test_create_data_fetcher_invalid(self):
        """Test creating an invalid data fetcher"""
        with pytest.raises(ValueError) as excinfo:
            DataFetcherFactory.create_data_fetcher("invalid_source")
        
        assert "Invalid data source" in str(excinfo.value)

    def test_create_data_fetcher_case_insensitive(self):
        """Test that data source is case insensitive"""
        fetcher1 = DataFetcherFactory.create_data_fetcher("SYNTHETIC")
        fetcher2 = DataFetcherFactory.create_data_fetcher("synthetic")
        
        assert isinstance(fetcher1, SyntheticDataFetcher)
        assert isinstance(fetcher2, SyntheticDataFetcher) 