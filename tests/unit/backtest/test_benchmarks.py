"""
Tests for the benchmark functionality.
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
    
    @patch('yfinance.Ticker')
    def test_calculate_returns(self, mock_ticker):
        """Test calculating returns for buy and hold strategy"""
        # Mock yfinance data
        dates = pd.date_range(start="2023-01-01", end="2023-01-31")
        mock_data = pd.DataFrame({
            'Close': np.linspace(100, 150, len(dates))
        }, index=dates)
        
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = mock_data
        mock_ticker.return_value = mock_ticker_instance
        
        # Create benchmark
        benchmark = BuyAndHoldBenchmark("Buy & Hold AAPL", "AAPL", 10000)
        
        # Calculate returns
        returns_df = benchmark.calculate_returns("2023-01-01", "2023-01-31")
        
        # Check results
        assert isinstance(returns_df, pd.DataFrame)
        assert 'value' in returns_df.columns
        assert len(returns_df) == len(dates)
        
        # Initial value should be 10000
        assert returns_df['value'].iloc[0] == pytest.approx(10000)
        
        # Final value should reflect price change
        expected_return = 10000 * (150/100)
        assert returns_df['value'].iloc[-1] == pytest.approx(expected_return)

class TestMarketIndexBenchmark:
    """Test cases for the MarketIndexBenchmark class"""
    
    @patch('yfinance.Ticker')
    def test_calculate_returns(self, mock_ticker):
        """Test calculating returns for market index"""
        # Mock yfinance data
        dates = pd.date_range(start="2023-01-01", end="2023-01-31")
        mock_data = pd.DataFrame({
            'Close': np.linspace(4000, 4200, len(dates))
        }, index=dates)
        
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = mock_data
        mock_ticker.return_value = mock_ticker_instance
        
        # Create benchmark
        benchmark = MarketIndexBenchmark("S&P 500", "^GSPC", 10000)
        
        # Calculate returns
        returns_df = benchmark.calculate_returns("2023-01-01", "2023-01-31")
        
        # Check results
        assert isinstance(returns_df, pd.DataFrame)
        assert 'value' in returns_df.columns
        assert len(returns_df) == len(dates)
        
        # Initial value should be 10000
        assert returns_df['value'].iloc[0] == pytest.approx(10000)
        
        # Final value should reflect index change
        expected_return = 10000 * (4200/4000)
        assert returns_df['value'].iloc[-1] == pytest.approx(expected_return)
    
    @patch('yfinance.Ticker')
    def test_error_handling(self, mock_ticker):
        """Test error handling when data fetch fails"""
        # Mock yfinance error
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.side_effect = Exception("API Error")
        mock_ticker.return_value = mock_ticker_instance
        
        # Create benchmark
        benchmark = MarketIndexBenchmark("S&P 500", "^GSPC", 10000)
        
        # Calculate returns should handle error gracefully
        returns_df = benchmark.calculate_returns("2023-01-01", "2023-01-31")
        assert returns_df.empty

class TestBenchmarkFactory:
    """Test cases for the BenchmarkFactory class"""
    
    def test_create_buy_and_hold_benchmark(self):
        """Test creating buy and hold benchmark"""
        benchmark = BenchmarkFactory.create_benchmark("buy_and_hold", "AAPL", 10000)
        assert isinstance(benchmark, BuyAndHoldBenchmark)
        assert benchmark.name == "Buy & Hold AAPL"
        assert benchmark.symbol == "AAPL"
        assert benchmark.initial_investment == 10000
    
    def test_create_sp500_benchmark(self):
        """Test creating S&P 500 benchmark"""
        benchmark = BenchmarkFactory.create_benchmark("sp500", "AAPL", 10000)
        assert isinstance(benchmark, MarketIndexBenchmark)
        assert benchmark.name == "S&P 500"
        assert benchmark.symbol == "^GSPC"
        assert benchmark.initial_investment == 10000
    
    def test_create_nasdaq_benchmark(self):
        """Test creating NASDAQ benchmark"""
        benchmark = BenchmarkFactory.create_benchmark("nasdaq", "AAPL", 10000)
        assert isinstance(benchmark, MarketIndexBenchmark)
        assert benchmark.name == "NASDAQ"
        assert benchmark.symbol == "^IXIC"
        assert benchmark.initial_investment == 10000
    
    def test_invalid_benchmark_type(self):
        """Test error handling for invalid benchmark type"""
        with pytest.raises(ValueError):
            BenchmarkFactory.create_benchmark("invalid", "AAPL", 10000) 