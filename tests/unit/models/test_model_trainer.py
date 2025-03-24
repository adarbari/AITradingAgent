"""
Tests for the ModelTrainer class
"""
import os
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from src.models import ModelTrainer
from src.agent.trading_env import TradingEnvironment


class TestModelTrainer:
    """Test cases for the ModelTrainer class"""

    def test_initialization(self):
        """Test initialization of the trainer"""
        # Test with default models directory
        trainer = ModelTrainer()
        assert trainer.models_dir == "models"
        assert trainer.verbose == 1
        
        # Test with custom models directory and verbosity
        custom_dir = "custom_models"
        trainer = ModelTrainer(models_dir=custom_dir, verbose=0)
        assert trainer.models_dir == custom_dir
        assert trainer.verbose == 0
        
        # Verify the directory is created
        assert os.path.exists(custom_dir)
        
        # Clean up the test directory
        os.rmdir(custom_dir)

    def test_train_model(self):
        """Test training a model"""
        # Mock PPO class
        mock_ppo = MagicMock()
        mock_model = MagicMock()
        mock_ppo.return_value = mock_model
        
        # Sample data for testing
        symbol = "TEST"
        train_start = "2022-01-01"
        train_end = "2022-12-31"
        
        # Create sample price and feature data
        prices = np.linspace(100, 200, 252)  # Approximately 1 year of trading days
        features = np.random.normal(0, 1, (252, 10))
        
        # Create a temporary directory for test models
        test_models_dir = "test_models"
        os.makedirs(test_models_dir, exist_ok=True)
        
        # Initialize trainer with mock PPO
        trainer = ModelTrainer(models_dir=test_models_dir, verbose=0, ppo_constructor=mock_ppo)
        
        # Train model
        model_path = trainer.train_model(
            env_class=TradingEnvironment,
            prices=prices,
            features=features,
            symbol=symbol,
            train_start=train_start,
            train_end=train_end,
            total_timesteps=10000
        )
        
        # Verify PPO was called correctly
        mock_ppo.assert_called_once()
        
        # Verify learn was called
        mock_model.learn.assert_called_once()
        args, kwargs = mock_model.learn.call_args
        assert kwargs['total_timesteps'] == 10000
        
        # Verify save was called
        mock_model.save.assert_called_once()
        
        # Verify model path
        expected_path = os.path.join(test_models_dir, f"ppo_{symbol}_{train_start.split('-')[0]}_{train_end.split('-')[0]}")
        assert model_path == expected_path
        
        # Clean up
        import shutil
        if os.path.exists(test_models_dir):
            shutil.rmtree(test_models_dir)

    def test_train_model_custom_params(self):
        """Test training a model with custom parameters"""
        # Mock PPO class
        mock_ppo = MagicMock()
        mock_model = MagicMock()
        mock_ppo.return_value = mock_model
        
        # Sample data for testing
        prices = np.linspace(100, 200, 100)
        features = np.random.normal(0, 1, (100, 5))
        
        # Create a temporary directory for test models
        test_models_dir = "test_models"
        os.makedirs(test_models_dir, exist_ok=True)
        
        # Initialize trainer with mock PPO
        trainer = ModelTrainer(models_dir=test_models_dir, verbose=0, ppo_constructor=mock_ppo)
        
        # Custom training parameters
        custom_params = {
            'learning_rate': 0.0001,
            'n_steps': 512,
            'batch_size': 64,
            'gamma': 0.95
        }
        
        # Train model with custom parameters
        model_path = trainer.train_model(
            env_class=TradingEnvironment,
            prices=prices,
            features=features,
            symbol="CUSTOM",
            train_start="2022-01-01",
            train_end="2022-03-31",
            total_timesteps=5000,
            model_params=custom_params
        )
        
        # Verify PPO was called with custom parameters
        _, kwargs = mock_ppo.call_args
        for param, value in custom_params.items():
            assert kwargs[param] == value
        
        # Verify learn was called with custom timesteps
        mock_model.learn.assert_called_once()
        args, kwargs = mock_model.learn.call_args
        assert kwargs['total_timesteps'] == 5000
        
        # Clean up
        import shutil
        if os.path.exists(test_models_dir):
            shutil.rmtree(test_models_dir)

    @patch('stable_baselines3.PPO.load')
    def test_load_model(self, mock_load):
        """Test loading a model"""
        # Mock dependencies
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        
        # Initialize trainer
        trainer = ModelTrainer(verbose=0)
        
        # Load model
        model = trainer.load_model("mock_model_path")
        
        # Verify load was called
        mock_load.assert_called_once_with("mock_model_path")
        
        # Verify returned model
        assert model == mock_model 