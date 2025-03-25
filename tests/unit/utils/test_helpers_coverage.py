#!/usr/bin/env python3
"""
Additional unit tests for the helper functions in src/utils/helpers.py
specifically focused on increasing code coverage
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from unittest.mock import patch, mock_open, MagicMock, call

import pytest
from src.utils.helpers import (
    create_directories,
    calculate_returns,
    calculate_sharpe_ratio,
    calculate_drawdown,
    plot_portfolio_performance,
    plot_trade_history,
    save_performance_metrics
)


class TestHelperFunctionsCoverage:
    """Additional tests to increase coverage of helper functions"""
    
    @patch('os.makedirs')
    def test_create_directories_multiple(self, mock_makedirs):
        """Test creating multiple directories"""
        create_directories('dir1', 'dir2', 'dir3')
        expected_calls = [
            call('dir1', exist_ok=True),
            call('dir2', exist_ok=True),
            call('dir3', exist_ok=True)
        ]
        mock_makedirs.assert_has_calls(expected_calls)
    
    def test_calculate_returns_positive(self):
        """Test calculating positive returns"""
        initial_value = 100
        final_value = 120
        expected_return = 20.0  # 20%
        assert calculate_returns(initial_value, final_value) == expected_return
    
    def test_calculate_returns_negative(self):
        """Test calculating negative returns"""
        initial_value = 100
        final_value = 90
        expected_return = -10.0  # -10%
        assert calculate_returns(initial_value, final_value) == expected_return
    
    def test_calculate_returns_zero(self):
        """Test calculating returns with zero initial value"""
        initial_value = 0
        final_value = 100
        with pytest.raises(ZeroDivisionError):
            calculate_returns(initial_value, final_value)
    
    def test_calculate_sharpe_ratio_positive(self):
        """Test calculating sharpe ratio with positive returns"""
        returns = [0.01, 0.02, 0.015, 0.025, 0.01]
        # Expected value: (mean(returns) - risk_free_rate) / std(returns) * sqrt(252)
        expected_sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        assert calculate_sharpe_ratio(returns) == pytest.approx(expected_sharpe)
    
    def test_calculate_sharpe_ratio_with_risk_free_rate(self):
        """Test calculating sharpe ratio with a risk-free rate"""
        returns = [0.01, 0.02, 0.015, 0.025, 0.01]
        risk_free_rate = 0.005
        # Expected value: (mean(returns) - risk_free_rate) / std(returns) * sqrt(252)
        expected_sharpe = (np.mean(returns) - risk_free_rate) / np.std(returns) * np.sqrt(252)
        assert calculate_sharpe_ratio(returns, risk_free_rate) == pytest.approx(expected_sharpe)
    
    def test_calculate_sharpe_ratio_zero_std(self):
        """Test calculating sharpe ratio with zero standard deviation (all returns the same)"""
        returns = [0.01, 0.01, 0.01, 0.01, 0.01]
        # Expected value should be 0 when std is 0
        assert calculate_sharpe_ratio(returns) == 0
    
    def test_calculate_sharpe_ratio_empty(self):
        """Test calculating sharpe ratio with empty returns list"""
        returns = []
        # When given an empty list, we should get nan
        result = calculate_sharpe_ratio(returns)
        assert np.isnan(result)
    
    def test_calculate_drawdown_no_drawdown(self):
        """Test calculating drawdown with continuously increasing values"""
        portfolio_values = [100, 105, 110, 115, 120]
        expected_drawdown = 0  # No drawdown
        assert calculate_drawdown(portfolio_values) == expected_drawdown
    
    def test_calculate_drawdown_with_drawdown(self):
        """Test calculating drawdown with fluctuating values"""
        portfolio_values = [100, 110, 105, 95, 100]
        # Drawdown calculation:
        # Running max: [100, 110, 110, 110, 110]
        # Drawdown: [0, 0, -4.55%, -13.64%, -9.09%]
        # Min drawdown: -13.64%
        expected_drawdown = -13.64  # Approximately
        assert calculate_drawdown(portfolio_values) == pytest.approx(expected_drawdown, abs=0.01)
    
    def test_calculate_drawdown_empty(self):
        """Test calculating drawdown with empty list"""
        portfolio_values = []
        with pytest.raises(Exception):
            calculate_drawdown(portfolio_values)
    
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
    def test_plot_portfolio_performance_without_benchmark(self, mock_show, mock_tight_layout,
                                                         mock_savefig, mock_legend, mock_grid,
                                                         mock_ylabel, mock_xlabel, mock_title,
                                                         mock_plot, mock_figure):
        """Test plotting portfolio performance without benchmark values"""
        dates = pd.date_range(start='2020-01-01', periods=5)
        portfolio_values = [100, 105, 103, 110, 115]
        
        plot_portfolio_performance(dates, portfolio_values)
        
        mock_figure.assert_called_once()
        mock_plot.assert_called_once()
        mock_title.assert_called_once()
        mock_xlabel.assert_called_once()
        mock_ylabel.assert_called_once()
        mock_grid.assert_called_once()
        mock_legend.assert_called_once()
        mock_tight_layout.assert_called_once()
        mock_show.assert_called_once()
        mock_savefig.assert_not_called()  # No save path
    
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
    def test_plot_portfolio_performance_with_benchmark(self, mock_show, mock_tight_layout,
                                                     mock_savefig, mock_legend, mock_grid,
                                                     mock_ylabel, mock_xlabel, mock_title,
                                                     mock_plot, mock_figure):
        """Test plotting portfolio performance with benchmark values"""
        dates = pd.date_range(start='2020-01-01', periods=5)
        portfolio_values = [100, 105, 103, 110, 115]
        benchmark_values = [100, 102, 104, 106, 108]
        
        plot_portfolio_performance(dates, portfolio_values, benchmark_values)
        
        mock_figure.assert_called_once()
        assert mock_plot.call_count == 2  # Called twice, for portfolio and benchmark
        mock_title.assert_called_once()
        mock_xlabel.assert_called_once()
        mock_ylabel.assert_called_once()
        mock_grid.assert_called_once()
        mock_legend.assert_called_once()
        mock_tight_layout.assert_called_once()
        mock_show.assert_called_once()
        mock_savefig.assert_not_called()  # No save path
    
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
    def test_plot_portfolio_performance_with_save_path(self, mock_show, mock_tight_layout,
                                                     mock_savefig, mock_legend, mock_grid,
                                                     mock_ylabel, mock_xlabel, mock_title,
                                                     mock_plot, mock_figure):
        """Test plotting portfolio performance with a save path"""
        dates = pd.date_range(start='2020-01-01', periods=5)
        portfolio_values = [100, 105, 103, 110, 115]
        save_path = 'test_plot.png'
        
        plot_portfolio_performance(dates, portfolio_values, save_path=save_path)
        
        mock_figure.assert_called_once()
        mock_plot.assert_called_once()
        mock_title.assert_called_once()
        mock_xlabel.assert_called_once()
        mock_ylabel.assert_called_once()
        mock_grid.assert_called_once()
        mock_legend.assert_called_once()
        mock_tight_layout.assert_called_once()
        mock_show.assert_called_once()
        mock_savefig.assert_called_once_with(save_path)
    
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
    def test_plot_trade_history_without_save(self, mock_show, mock_tight_layout,
                                           mock_savefig, mock_legend, mock_grid,
                                           mock_ylabel, mock_xlabel, mock_title,
                                           mock_scatter, mock_plot, mock_figure):
        """Test plotting trade history without saving"""
        dates = pd.date_range(start='2020-01-01', periods=5)
        actions = ['buy', 'hold', 'sell', 'hold', 'buy']
        prices = [100, 105, 110, 105, 100]
        
        plot_trade_history(dates, actions, prices)
        
        mock_figure.assert_called_once()
        mock_plot.assert_called_once()
        assert mock_scatter.call_count == 2  # Called for both buy and sell points
        mock_title.assert_called_once()
        mock_xlabel.assert_called_once()
        mock_ylabel.assert_called_once()
        mock_grid.assert_called_once()
        mock_legend.assert_called_once()
        mock_tight_layout.assert_called_once()
        mock_show.assert_called_once()
        mock_savefig.assert_not_called()
    
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
    def test_plot_trade_history_with_save(self, mock_show, mock_tight_layout,
                                        mock_savefig, mock_legend, mock_grid,
                                        mock_ylabel, mock_xlabel, mock_title,
                                        mock_scatter, mock_plot, mock_figure):
        """Test plotting trade history with save path"""
        dates = pd.date_range(start='2020-01-01', periods=5)
        actions = ['hold', 'buy', 'hold', 'sell', 'hold']
        prices = [100, 105, 110, 105, 100]
        save_path = 'test_trade_history.png'
        
        plot_trade_history(dates, actions, prices, save_path=save_path)
        
        mock_figure.assert_called_once()
        mock_plot.assert_called_once()
        assert mock_scatter.call_count == 2  # Called for both buy and sell points
        mock_title.assert_called_once()
        mock_xlabel.assert_called_once()
        mock_ylabel.assert_called_once()
        mock_grid.assert_called_once()
        mock_legend.assert_called_once()
        mock_tight_layout.assert_called_once()
        mock_show.assert_called_once()
        mock_savefig.assert_called_once_with(save_path)
    
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
    def test_plot_trade_history_no_buys_sells(self, mock_show, mock_tight_layout,
                                            mock_savefig, mock_legend, mock_grid,
                                            mock_ylabel, mock_xlabel, mock_title,
                                            mock_scatter, mock_plot, mock_figure):
        """Test plotting trade history with no buy or sell actions"""
        dates = pd.date_range(start='2020-01-01', periods=5)
        actions = ['hold', 'hold', 'hold', 'hold', 'hold']
        prices = [100, 105, 110, 105, 100]
        
        plot_trade_history(dates, actions, prices)
        
        mock_figure.assert_called_once()
        mock_plot.assert_called_once()
        assert mock_scatter.call_count == 2  # Called for both buy and sell points, with empty data
        mock_title.assert_called_once()
        mock_xlabel.assert_called_once()
        mock_ylabel.assert_called_once()
        mock_grid.assert_called_once()
        mock_legend.assert_called_once()
        mock_tight_layout.assert_called_once()
        mock_show.assert_called_once()
    
    @patch('src.utils.helpers.datetime')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.makedirs')
    def test_save_performance_metrics(self, mock_makedirs, mock_file, mock_datetime):
        """Test saving performance metrics to a file"""
        # Mock datetime.now() to return a fixed datetime
        mock_now = MagicMock()
        mock_now.return_value = datetime(2023, 1, 1, 12, 0, 0)
        mock_datetime.now = mock_now
        
        # Test metrics with different value types
        metrics = {
            'Total Returns': 15.5,
            'Sharpe Ratio': 1.2,
            'Max Drawdown': -10.3,
            'Number of Trades': 42,
            'Strategy': 'PPO'
        }
        
        file_path = 'test_dir/metrics.txt'
        save_performance_metrics(metrics, file_path)
        
        # Check that the directory was created
        mock_makedirs.assert_called_once_with('test_dir', exist_ok=True)
        
        # Check that the file was opened for writing
        mock_file.assert_called_once_with(file_path, 'a')
        
        # Check that the correct content was written
        expected_content = (
            '\nTimestamp: 2023-01-01 12:00:00\n'
            'Total Returns: 15.50\n'
            'Sharpe Ratio: 1.20\n'
            'Max Drawdown: -10.30\n'
            'Number of Trades: 42\n'
            'Strategy: PPO\n'
            '--------------------------------------------------\n'
        )
        mock_file().write.assert_called_once_with(expected_content)
    
    @patch('os.makedirs')
    @patch('builtins.open', new_callable=mock_open)
    def test_save_performance_metrics_empty(self, mock_file, mock_makedirs):
        """Test saving empty performance metrics"""
        metrics = {}
        file_path = 'test_dir/metrics.txt'
        
        save_performance_metrics(metrics, file_path)
        
        # Check that the directory was created
        mock_makedirs.assert_called_once_with('test_dir', exist_ok=True)
        
        # Check that the file was opened for writing
        mock_file.assert_called_once_with(file_path, 'a')
        
        # Check that only the timestamp and separator were written
        mock_file().write.assert_called_once()
        content = mock_file().write.call_args[0][0]
        assert 'Timestamp:' in content
        assert '--------------------------------------------------' in content 