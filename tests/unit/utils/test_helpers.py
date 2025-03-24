"""
Tests for the helpers utility module.
"""
import os
import json
import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock, mock_open
from src.utils.helpers import (
    create_directories, calculate_returns, calculate_sharpe_ratio,
    calculate_drawdown, plot_portfolio_performance, plot_trade_history,
    save_performance_metrics
)


class TestHelperFunctions:
    """Test cases for helper utility functions"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Create sample data for tests
        self.dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        
        # Portfolio values (starting at 10000, with some ups and downs)
        self.portfolio_values = pd.Series(
            10000 * (1 + np.cumsum(np.random.normal(0.001, 0.02, 100))),
            index=self.dates
        )
        
        # Sample trades data
        self.trades = pd.DataFrame({
            'date': self.dates[np.random.choice(len(self.dates), 20, replace=False)].sort_values(),
            'action': np.random.choice(['buy', 'sell'], 20),
            'price': np.random.uniform(100, 200, 20),
            'shares': np.random.randint(1, 10, 20)
        })
        
        # Temp directories for testing file operations
        self.test_dirs = ['tests/test_models', 'tests/test_results', 'tests/test_logs']
        
        # Create test directories
        for d in self.test_dirs:
            os.makedirs(d, exist_ok=True)
    
    def teardown_method(self):
        """Clean up after tests"""
        # Clean up test files if needed
        test_file = 'tests/test_results/metrics.json'
        if os.path.exists(test_file):
            os.remove(test_file)
    
    def test_create_directories(self):
        """Test creating directories"""
        # Define test directories
        test_paths = [
            'tests/temp/models',
            'tests/temp/data',
            'tests/temp/results'
        ]
        
        # Call the function for each directory
        for directory in test_paths:
            create_directories(directory)
        
        # Check directories exist
        for path in test_paths:
            assert os.path.exists(path)
            assert os.path.isdir(path)
            
        # Clean up
        import shutil
        shutil.rmtree('tests/temp')
    
    def test_calculate_returns(self):
        """Test calculating returns from portfolio values"""
        # Calculate returns between two values
        initial_value = 10000
        final_value = 12500
        
        # Call the function
        returns = calculate_returns(initial_value, final_value)
        
        # Expected return is (12500 - 10000) / 10000 * 100 = 25.0%
        expected_return = 25.0
        
        # Check calculation is correct
        assert returns == expected_return
    
    def test_calculate_sharpe_ratio_positive(self):
        """Test calculating Sharpe ratio with positive returns"""
        # Create predictable returns series with positive mean
        returns = np.array([0.01, 0.02, 0.015, 0.025, 0.02])  # Mean = 0.018, std = 0.00570
        
        # Calculate Sharpe ratio with risk-free rate of 0
        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0)
        
        # Check that Sharpe ratio is positive (since returns > risk-free rate)
        assert sharpe > 0
        
        # Manually compute to make sure logic is correct
        # Note: The formula in the actual code multiplies by sqrt(252) for annualization
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        manual_sharpe = (mean_return / std_return) * np.sqrt(252)
        
        # Use a more relaxed precision for this test
        np.testing.assert_almost_equal(sharpe, manual_sharpe, decimal=2)
    
    def test_calculate_sharpe_ratio_negative(self):
        """Test calculating Sharpe ratio with negative returns"""
        # Create predictable returns series with negative mean
        returns = np.array([-0.01, -0.02, -0.015, -0.025, -0.02])  # Mean = -0.018
        
        # Calculate Sharpe ratio with risk-free rate of 0
        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0)
        
        # Sharpe should be negative since mean return < risk-free rate
        assert sharpe < 0
    
    def test_calculate_drawdown(self):
        """Test calculating drawdown"""
        # Create a portfolio series with a clear drawdown
        portfolio = np.array([100, 110, 105, 95, 90, 85, 90, 100, 110])
        
        # Calculate drawdown
        max_drawdown = calculate_drawdown(portfolio)
        
        # The max drawdown should be (85-110)/110 = -0.2273 as a percentage
        expected_max_drawdown = (85 - 110) / 110 * 100
        
        # Check calculation is approximately correct
        np.testing.assert_almost_equal(max_drawdown, expected_max_drawdown, decimal=2)
    
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.plot')
    @patch('matplotlib.pyplot.title')
    @patch('matplotlib.pyplot.xlabel')
    @patch('matplotlib.pyplot.ylabel')
    @patch('matplotlib.pyplot.grid')
    @patch('matplotlib.pyplot.legend')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.tight_layout')
    @patch('matplotlib.pyplot.show')
    def test_plot_portfolio_performance(self, mock_show, mock_tight_layout, mock_savefig, 
                                        mock_legend, mock_grid, mock_ylabel, mock_xlabel, 
                                        mock_title, mock_plot, mock_figure):
        """Test plotting portfolio performance"""
        # Convert portfolio values to list
        portfolio_list = self.portfolio_values.tolist()
        dates_list = [d.strftime('%Y-%m-%d') for d in self.dates]
        benchmark_list = [val * 0.9 for val in portfolio_list]
        
        # Call the function
        plot_portfolio_performance(
            dates=dates_list,
            portfolio_values=portfolio_list,
            benchmark_values=benchmark_list,
            save_path='tests/test_results/performance.png'
        )
        
        # Check figure was created
        mock_figure.assert_called_once()
        
        # Check plot was called with portfolio and benchmark
        assert mock_plot.call_count >= 2
        
        # Check figure was saved
        mock_savefig.assert_called_once_with('tests/test_results/performance.png')
    
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.plot')
    @patch('matplotlib.pyplot.scatter')
    @patch('matplotlib.pyplot.title')
    @patch('matplotlib.pyplot.xlabel')
    @patch('matplotlib.pyplot.ylabel')
    @patch('matplotlib.pyplot.grid')
    @patch('matplotlib.pyplot.legend')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.tight_layout')
    @patch('matplotlib.pyplot.show')
    def test_plot_trade_history(self, mock_show, mock_tight_layout, mock_savefig, 
                               mock_legend, mock_grid, mock_ylabel, mock_xlabel, 
                               mock_title, mock_scatter, mock_plot, mock_figure):
        """Test plotting trade history"""
        # Convert to lists for function
        dates_list = [d.strftime('%Y-%m-%d') for d in self.dates]
        actions_list = self.trades['action'].tolist()
        prices_list = self.trades['price'].tolist()
        
        # Call the function
        plot_trade_history(
            dates=dates_list,
            actions=actions_list,
            prices=prices_list,
            save_path='tests/test_results/trades.png'
        )
        
        # Check figure was created
        mock_figure.assert_called_once()
        
        # Check plot was called for price history
        mock_plot.assert_called_once()
        
        # Check scatter was called for buys and sells
        assert mock_scatter.call_count == 2
        
        # Check figure was saved
        mock_savefig.assert_called_once_with('tests/test_results/trades.png')
    
    @patch('builtins.open', new_callable=mock_open)
    def test_save_performance_metrics(self, mock_file):
        """Test saving performance metrics to file"""
        # Create sample metrics
        metrics = {
            'total_return': 0.25,
            'sharpe_ratio': 0.8,
            'max_drawdown': -0.15,
            'win_rate': 0.6
        }
        
        # Call the function
        save_performance_metrics(metrics, 'tests/test_results/metrics.json')
        
        # Check file was opened for appending (not writing)
        mock_file.assert_called_once_with('tests/test_results/metrics.json', 'a')
        
        # Check something was written to the file
        file_handle = mock_file()
        assert file_handle.write.call_count >= 1 