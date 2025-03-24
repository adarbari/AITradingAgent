"""
Tests for the Backtester class
"""
import os
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt

from src.backtest import Backtester
from src.data import SyntheticDataFetcher
from src.agent.trading_env import TradingEnvironment


class TestBacktester:
    """Test cases for the Backtester class"""

    def test_initialization(self):
        """Test initialization of the backtester"""
        # Test with default results directory
        backtester = Backtester()
        assert backtester.results_dir == "results"
        
        # Test with custom results directory
        custom_dir = "custom_results"
        backtester = Backtester(results_dir=custom_dir)
        assert backtester.results_dir == custom_dir
        
        # Verify the directory is created
        assert os.path.exists(custom_dir)
        
        # Clean up the test directory
        os.rmdir(custom_dir)

    @patch('stable_baselines3.PPO.load')
    @patch('src.data.DataFetcherFactory.create_data_fetcher')
    def test_backtest_model(self, mock_create_data_fetcher, mock_load):
        """Test backtesting a model"""
        # Mock dependencies
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        
        # Create a synthetic data fetcher for testing
        data_fetcher = SyntheticDataFetcher()
        mock_create_data_fetcher.return_value = data_fetcher
        
        # Sample data for testing
        symbol = "TEST"
        test_start = "2023-01-01"
        test_end = "2023-01-31"
        
        # Create a temporary directory for test results
        test_results_dir = "test_results"
        os.makedirs(test_results_dir, exist_ok=True)
        
        # Initialize backtester
        backtester = Backtester(results_dir=test_results_dir)
        
        # Set up expected returns from mocked predict method
        # For each step, return an action value and "done" status
        actions = np.linspace(-1, 1, 31)  # Different actions for each day
        side_effects = [(np.array([action]), None) for action in actions]
        mock_model.predict.side_effect = side_effects
        
        # Execute backtest
        results = backtester.backtest_model(
            model_path="mock_model_path",
            symbol=symbol,
            test_start=test_start,
            test_end=test_end,
            data_source="synthetic",
            env_class=TradingEnvironment
        )
        
        # Verify model was loaded
        mock_load.assert_called_once_with("mock_model_path")
        
        # Verify correct data fetcher was created
        mock_create_data_fetcher.assert_called_once_with("synthetic")
        
        # Verify backtest results
        assert 'total_return' in results
        assert 'sharpe_ratio' in results
        assert 'max_drawdown' in results
        assert 'returns' in results
        
        # Verify returns dataframe
        returns_df = results['returns']
        assert isinstance(returns_df, pd.DataFrame)
        assert 'portfolio_value' in returns_df.columns
        assert len(returns_df) > 0
        
        # Clean up
        import shutil
        if os.path.exists(test_results_dir):
            shutil.rmtree(test_results_dir)

    @patch('pandas_datareader.data.DataReader')
    def test_get_market_performance(self, mock_data_reader):
        """Test getting market performance data"""
        # Create mock market data
        dates = pd.date_range(start="2023-01-01", end="2023-01-31")
        market_data = pd.DataFrame({
            'Close': np.linspace(100, 150, len(dates))
        }, index=dates)
        mock_data_reader.return_value = market_data
        
        # Initialize backtester
        backtester = Backtester()
        
        # Get market performance
        result = backtester.get_market_performance(
            test_start="2023-01-01",
            test_end="2023-01-31"
        )
        
        # Verify data reader was called correctly
        mock_data_reader.assert_called_once()
        
        # Verify result
        assert isinstance(result, pd.DataFrame)
        assert 'Close' in result.columns
        assert 'Normalized' in result.columns
        assert len(result) == len(dates)
        
        # Verify normalization (first value should be 1.0)
        assert result['Normalized'].iloc[0] == 1.0

    @patch('matplotlib.pyplot.savefig')
    def test_plot_comparison(self, mock_savefig):
        """Test plotting comparison between model and market"""
        # Create sample returns data
        dates = pd.date_range(start="2023-01-01", end="2023-01-31")
        returns_df = pd.DataFrame({
            'date': dates,
            'portfolio_value': np.linspace(10000, 12000, len(dates))
        })
        returns_df.set_index('date', inplace=True)
        
        # Create sample market data
        market_data = pd.DataFrame({
            'Close': np.linspace(100, 120, len(dates)),
            'Normalized': np.linspace(1, 1.2, len(dates))
        }, index=dates)
        
        # Initialize backtester
        test_results_dir = "test_results"
        os.makedirs(test_results_dir, exist_ok=True)
        backtester = Backtester(results_dir=test_results_dir)
        
        # Plot comparison
        symbol = "TEST"
        plot_path = backtester.plot_comparison(
            returns_df=returns_df,
            market_data=market_data,
            symbol=symbol
        )
        
        # Verify savefig was called
        mock_savefig.assert_called_once()
        
        # Verify plot path
        expected_path = os.path.join(test_results_dir, f"{symbol}/market_comparison.png")
        assert plot_path == expected_path
        
        # Clean up
        import shutil
        if os.path.exists(test_results_dir):
            shutil.rmtree(test_results_dir) 