"""
Tests for the Backtester class.
"""
import os
import shutil
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import pytest
from stable_baselines3 import PPO

from src.backtest.backtester import Backtester
from src.agent.trading_env import TradingEnvironment
from src.data import SyntheticDataFetcher

class TestBacktester:
    """Test cases for the Backtester class"""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock trained model"""
        model = MagicMock()
        model.predict.return_value = (np.array([0]), None)  # Action, _
        return model
    
    @pytest.fixture
    def mock_env(self):
        """Create a mock trading environment"""
        env = MagicMock()
        env.reset.return_value = (np.array([0, 0, 0]), {})  # State, info
        env.step.return_value = (np.array([0, 0, 0]), 0, False, False, {})  # State, reward, terminated, truncated, info
        return env
    
    @pytest.fixture
    def sample_prices(self):
        """Create sample price data"""
        dates = pd.date_range(start="2023-01-01", end="2023-01-31")
        return pd.DataFrame({
            'Close': np.linspace(100, 150, len(dates))
        }, index=dates)
    
    def test_initialization(self):
        """Test initialization of the backtester"""
        # Create a temporary directory for test results
        test_results_dir = "test_results"
        os.makedirs(test_results_dir, exist_ok=True)
        
        # Initialize backtester
        backtester = Backtester(results_dir=test_results_dir)
        
        # Verify initialization
        assert backtester.results_dir == test_results_dir
        assert len(backtester.benchmarks) == 3  # buy_and_hold, sp500, nasdaq
        
        # Clean up
        if os.path.exists(test_results_dir):
            shutil.rmtree(test_results_dir)
    
    @patch('src.data.yahoo_data_fetcher.YahooDataFetcher.fetch_ticker_data')
    def test_backtest_model(self, mock_fetch_ticker_data):
        """Test backtesting a model on historical data"""
        # Mock yfinance data
        dates = pd.date_range(start="2023-01-01", end="2023-01-10")
        test_data = pd.DataFrame({
            'Open': np.linspace(100, 110, len(dates)),
            'High': np.linspace(105, 115, len(dates)),
            'Low': np.linspace(95, 105, len(dates)),
            'Close': np.linspace(102, 112, len(dates)),
            'Volume': np.random.randint(1000000, 2000000, size=len(dates))
        }, index=dates)
        
        # Mock data fetcher
        mock_fetch_ticker_data.return_value = test_data
        
        # Create a temporary directory for test results
        test_results_dir = "test_results"
        os.makedirs(test_results_dir, exist_ok=True)
        
        # Create features for testing
        features = np.random.randn(len(test_data), 5)  # 5 random features
        
        # Mock the predict method to return actions and values
        mock_action = np.array([1])  # Buy action as array
        
        # Mock the PPO.load method using a context manager
        with patch('stable_baselines3.PPO.load') as mock_load:
            # Create a mock model with a predict method
            mock_model = MagicMock()
            mock_model.predict.return_value = (mock_action, None)
            mock_load.return_value = mock_model
            
            # Initialize backtester
            backtester = Backtester(results_dir=test_results_dir)
            
            # Mock prepare_data_for_agent to return fixed features
            backtester.prepare_data_for_agent = MagicMock(return_value=features)
            
            # Run backtest
            results = backtester.backtest_model(
                model_path="test_results/test_model",
                symbol="TEST",
                test_start="2023-01-01",
                test_end="2023-01-31",
                data_source="yahoo",
                env_class=TradingEnvironment
            )
        
        # Verify the test was successful
        assert isinstance(results, dict)
        assert 'total_return' in results
        assert 'sharpe_ratio' in results
        assert 'max_drawdown' in results
        
        # Clean up
        if os.path.exists(test_results_dir):
            shutil.rmtree(test_results_dir)

    @patch('matplotlib.pyplot.savefig')
    def test_plot_comparison(self, mock_savefig, mock_model, mock_env, sample_prices):
        """Test plotting performance comparison"""
        # Create a temporary directory for test results
        test_results_dir = "test_results"
        os.makedirs(test_results_dir, exist_ok=True)
        
        # Initialize backtester
        backtester = Backtester(results_dir=test_results_dir)
        
        # Create sample results
        returns_df = pd.DataFrame({
            'returns': np.random.randn(len(sample_prices.index))
        }, index=sample_prices.index)
        
        benchmark_results = {
            'buy_and_hold': {
                'name': 'Buy & Hold TEST',
                'returns_df': pd.DataFrame({
                    'returns': np.random.randn(len(sample_prices.index))
                }, index=sample_prices.index),
                'total_return': 0.05,
                'sharpe_ratio': 1.0,
                'max_drawdown': -0.03
            }
        }
        
        results = {
            'returns': returns_df,
            'total_return': 0.1,
            'sharpe_ratio': 1.5,
            'max_drawdown': -0.05,
            'benchmark_results': benchmark_results,
            'portfolio_values': np.linspace(10000, 11000, len(sample_prices.index)),
            'actions': np.random.randn(len(sample_prices.index))
        }
        
        # Plot comparison
        backtester.plot_comparison("TEST", results)
        
        # Verify that savefig was called
        mock_savefig.assert_called_once()

        # Cleanup
        shutil.rmtree(test_results_dir)
    
    def test_save_and_load_results(self, mock_model, mock_env, sample_prices):
        """Test saving and loading backtest results"""
        # Create a temporary directory for test results
        test_results_dir = "test_results"
        os.makedirs(test_results_dir, exist_ok=True)
    
        # Initialize backtester
        backtester = Backtester(results_dir=test_results_dir)
    
        # Create sample results
        returns_df = pd.DataFrame({
            'portfolio_value': [10000, 11000]
        }, index=pd.date_range(start="2023-01-01", periods=2))
    
        benchmark_df = pd.DataFrame({
            'portfolio_value': [10000, 10500]
        }, index=pd.date_range(start="2023-01-01", periods=2))
    
        results = {
            'returns': returns_df,
            'final_value': 11000,
            'initial_value': 10000,
            'total_return': 0.1,  # (11000 - 10000) / 10000
            'sharpe_ratio': 1.0,
            'max_drawdown': -0.05,
            'benchmark_results': {
                'buy_and_hold': {
                    'name': 'Buy & Hold TEST',
                    'returns_df': benchmark_df,
                    'total_return': 0.05,
                    'sharpe_ratio': 1.0,
                    'max_drawdown': -0.03
                }
            }
        }
    
        # Save results
        file_path = os.path.join(test_results_dir, "test_results.json")
        backtester.save_results(results, file_path)
    
        # Load results
        loaded_results = backtester.load_results(file_path)
    
        # Verify loaded results
        assert loaded_results['returns'].equals(results['returns'])
        assert loaded_results['final_value'] == results['final_value']
        assert loaded_results['initial_value'] == results['initial_value']
        assert loaded_results['total_return'] == results['total_return']
        assert loaded_results['sharpe_ratio'] == results['sharpe_ratio']
        assert loaded_results['max_drawdown'] == results['max_drawdown']
        assert 'buy_and_hold' in loaded_results['benchmark_results']
        assert loaded_results['benchmark_results']['buy_and_hold']['name'] == results['benchmark_results']['buy_and_hold']['name']
        assert loaded_results['benchmark_results']['buy_and_hold']['total_return'] == results['benchmark_results']['buy_and_hold']['total_return']
        assert loaded_results['benchmark_results']['buy_and_hold']['sharpe_ratio'] == results['benchmark_results']['buy_and_hold']['sharpe_ratio']
        assert loaded_results['benchmark_results']['buy_and_hold']['max_drawdown'] == results['benchmark_results']['buy_and_hold']['max_drawdown']

        # Cleanup
        shutil.rmtree(test_results_dir) 