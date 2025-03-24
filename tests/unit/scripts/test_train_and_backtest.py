"""
Tests for the train_and_backtest.py script
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.scripts.train_and_backtest import backtest_model

class TestTrainAndBacktest:
    """Test cases for the train_and_backtest.py script"""
    
    @patch('src.data.synthetic_data_fetcher.SyntheticDataFetcher')
    @patch('src.scripts.train_and_backtest.PPO')
    @patch('src.scripts.train_and_backtest.TradingEnvironment')
    @patch('src.scripts.train_and_backtest.process_features')
    @patch('src.scripts.train_and_backtest.FeatureCache')
    def test_backtest_model_uses_synthetic_data_fetcher(self, 
                                                       mock_cache, 
                                                       mock_process_features, 
                                                       mock_env, 
                                                       mock_ppo, 
                                                       mock_fetcher_class):
        """Test that backtest_model uses SyntheticDataFetcher for synthetic data"""
        # Set up the SyntheticDataFetcher mock
        mock_fetcher = MagicMock()
        mock_fetcher_class.return_value = mock_fetcher
        
        # Create mock data
        mock_data = pd.DataFrame({
            'Date': pd.date_range(start='2020-01-01', periods=10),
            'Open': np.random.rand(10),
            'High': np.random.rand(10),
            'Low': np.random.rand(10),
            'Close': np.random.rand(10),
            'Volume': np.random.rand(10),
            'Adj Close': np.random.rand(10),
        })
        mock_fetcher.fetch_data.return_value = mock_data
        
        # Mock feature cache
        mock_cache_instance = MagicMock()
        mock_cache.return_value = mock_cache_instance
        mock_cache_instance.get_cache_key.return_value = "test_key"
        mock_cache_instance.load.return_value = None  # No cached features
        
        # Mock process_features
        mock_features = pd.DataFrame(np.random.rand(10, 5))
        mock_process_features.return_value = mock_features
        
        # Mock TradingEnvironment
        mock_env_instance = MagicMock()
        mock_env.return_value = mock_env_instance
        mock_env_instance.reset.return_value = np.zeros(5)  # Mock observation
        mock_env_instance.step.return_value = (
            np.zeros(5),  # observation
            0.0,         # reward
            True,        # done
            {'portfolio_value': 10000.0},  # info with portfolio value
        )
        
        # Mock PPO
        mock_model = MagicMock()
        mock_ppo.load.return_value = mock_model
        mock_model.predict.return_value = (np.zeros(1), None)  # (action, _)
        
        # Call the function we're testing with synthetic data source
        backtest_model(
            model_path="models/test_model",
            symbol="TEST",
            test_start="2020-01-01",
            test_end="2020-01-10",
            data_source="synthetic",
            feature_set="standard"
        )
        
        # Verify SyntheticDataFetcher was instantiated
        mock_fetcher_class.assert_called_once()
        
        # Verify fetch_data was called with correct parameters
        mock_fetcher.fetch_data.assert_called_once_with("TEST", "2020-01-01", "2020-01-10") 