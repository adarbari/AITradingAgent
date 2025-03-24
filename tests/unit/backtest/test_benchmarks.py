"""
Tests for the benchmarks module.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.backtest.benchmarks import (
    Benchmark, 
    BuyAndHoldBenchmark, 
    MarketIndexBenchmark,
    BenchmarkFactory
)

class TestBenchmark:
    """Test cases for the base Benchmark class"""
    
    def test_abstract_methods(self):
        """Test that abstract methods need to be implemented"""
        with pytest.raises(TypeError):
            Benchmark("Test", "TEST", 10000)
    
    def test_calculate_metrics(self):
        """Test calculation of performance metrics"""
        # Create a concrete subclass for testing
        class TestBenchmark(Benchmark):
            def calculate_returns(self, start_date, end_date):
                pass
        
        benchmark = TestBenchmark("Test", "TEST", 10000)
        
        # Create sample returns data
        dates = pd.date_range(start="2023-01-01", end="2023-01-31")
        values = np.array([10000 * (1 + i * 0.01) for i in range(len(dates))])
        returns_df = pd.DataFrame({
            'value': values
        }, index=dates)
        
        # Calculate metrics
        metrics = benchmark.calculate_metrics(returns_df)
        
        # Check metric values
        assert metrics['name'] == "Test"
        assert metrics['total_return'] > 0
        assert isinstance(metrics['sharpe_ratio'], float)
        assert metrics['max_drawdown'] <= 0
        assert isinstance(metrics['returns_df'], pd.DataFrame)

class TestBuyAndHoldBenchmark:
    """Test cases for the BuyAndHoldBenchmark class"""
    
    @patch('src.data.yahoo_data_fetcher.YahooDataFetcher.fetch_ticker_data')
    def test_calculate_returns(self, mock_fetch_ticker_data):
        """Test calculating returns for buy and hold strategy"""
        # Mock data fetcher
        dates = pd.date_range(start="2023-01-01", end="2023-01-31")
        mock_data = pd.DataFrame({
            'Close': np.linspace(100, 150, len(dates))
        }, index=dates)
        
        mock_fetch_ticker_data.return_value = mock_data
        
        # Create benchmark
        benchmark = BuyAndHoldBenchmark("Buy & Hold AAPL", "AAPL", 10000)
        
        # Calculate returns
        returns_df = benchmark.calculate_returns("2023-01-01", "2023-01-31")
        
        # Verify correct data fetching
        mock_fetch_ticker_data.assert_called_once_with("AAPL", "2023-01-01", "2023-01-31")
        
        # Check results
        assert not returns_df.empty
        assert returns_df.index.equals(dates)
        assert 'value' in returns_df.columns
        
        # Initial value should be close to initial_investment
        initial_value = returns_df['value'].iloc[0]
        assert initial_value == pytest.approx(10000, abs=1)
        
        # Final value should reflect price increase
        final_value = returns_df['value'].iloc[-1]
        assert final_value > initial_value

class TestMarketIndexBenchmark:
    """Test cases for the MarketIndexBenchmark class"""
    
    @patch('src.data.yahoo_data_fetcher.YahooDataFetcher.fetch_ticker_data')
    def test_calculate_returns(self, mock_fetch_ticker_data):
        """Test calculating returns for market index benchmark"""
        # Mock data fetcher
        dates = pd.date_range(start="2023-01-01", end="2023-01-31")
        mock_data = pd.DataFrame({
            'Close': np.linspace(4000, 4500, len(dates))
        }, index=dates)
        
        mock_fetch_ticker_data.return_value = mock_data
        
        # Create benchmark
        benchmark = MarketIndexBenchmark("S&P 500", "^GSPC", 10000)
        
        # Calculate returns
        returns_df = benchmark.calculate_returns("2023-01-01", "2023-01-31")
        
        # Verify correct data fetching
        mock_fetch_ticker_data.assert_called_once_with("^GSPC", "2023-01-01", "2023-01-31")
        
        # Check results
        assert not returns_df.empty
        assert returns_df.index.equals(dates)
        assert 'value' in returns_df.columns
        
        # Initial value should be close to initial_investment
        initial_value = returns_df['value'].iloc[0]
        assert initial_value == pytest.approx(10000, abs=1)
        
        # Final value should reflect price increase
        final_value = returns_df['value'].iloc[-1]
        assert final_value > initial_value
    
    @patch('src.data.yahoo_data_fetcher.YahooDataFetcher.fetch_ticker_data')
    def test_error_handling(self, mock_fetch_ticker_data):
        """Test error handling in market index benchmark"""
        # Mock data fetcher to return empty DataFrame
        mock_fetch_ticker_data.return_value = pd.DataFrame()
        
        # Create benchmark
        benchmark = MarketIndexBenchmark("S&P 500", "^GSPC", 10000)
        
        # Calculate returns
        returns_df = benchmark.calculate_returns("2023-01-01", "2023-01-31")
        
        # Verify correct data fetching
        mock_fetch_ticker_data.assert_called_once()
        
        # Check returns an empty DataFrame on error
        assert isinstance(returns_df, pd.DataFrame)
        assert returns_df.empty

class TestBenchmarkFactory:
    """Test cases for the BenchmarkFactory class"""
    
    def test_create_buy_and_hold_benchmark(self):
        """Test creating a buy and hold benchmark"""
        benchmark = BenchmarkFactory.create_benchmark("buy_and_hold", "AAPL", 10000)
        
        assert isinstance(benchmark, BuyAndHoldBenchmark)
        assert benchmark.symbol == "AAPL"
        assert benchmark.name == "Buy & Hold AAPL"
        assert benchmark.initial_investment == 10000
    
    def test_create_sp500_benchmark(self):
        """Test creating an S&P 500 benchmark"""
        benchmark = BenchmarkFactory.create_benchmark("sp500", "AAPL", 10000)
        
        assert isinstance(benchmark, MarketIndexBenchmark)
        assert benchmark.symbol == "^GSPC"
        assert benchmark.name == "S&P 500"
        assert benchmark.initial_investment == 10000
    
    def test_create_nasdaq_benchmark(self):
        """Test creating a NASDAQ benchmark"""
        benchmark = BenchmarkFactory.create_benchmark("nasdaq", "AAPL", 10000)
        
        assert isinstance(benchmark, MarketIndexBenchmark)
        assert benchmark.symbol == "^IXIC"
        assert benchmark.name == "NASDAQ"
        assert benchmark.initial_investment == 10000
    
    def test_invalid_benchmark_type(self):
        """Test error handling for invalid benchmark type"""
        with pytest.raises(ValueError):
            BenchmarkFactory.create_benchmark("invalid_type", "AAPL", 10000) 