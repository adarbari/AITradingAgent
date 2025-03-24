#!/usr/bin/env python3
"""
Unit tests for edge cases of the TrainingManager class in src/train/trainer.py
"""
import os
import json
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from src.train.trainer import TrainingManager
from src.models import ModelTrainer
from src.data import DataFetcherFactory

class TestTrainingManagerEdgeCases:
    """Edge case tests for the TrainingManager class"""
    
    @patch('src.data.DataFetcherFactory.create_data_fetcher')
    @patch('src.train.trainer.ModelTrainer')
    def test_get_model_with_data_fetcher_error(self, mock_trainer_class, mock_factory, tmpdir):
        """Test handling of errors in data fetching during model training"""
        # Setup
        models_dir = tmpdir.mkdir("models")
        
        # Create mock objects
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_data.side_effect = Exception("Data fetching error")
        
        mock_trainer = MagicMock()
        mock_trainer_class.return_value = mock_trainer
        
        # Set up mocks
        mock_factory.return_value = mock_fetcher
        
        # Initialize the manager
        tm = TrainingManager(models_dir=str(models_dir), verbose=0)
        
        # Get a new model - should handle the error
        with pytest.raises(Exception) as exc_info:
            tm.get_model(
                symbol="AAPL",
                train_start="2020-01-01",
                train_end="2020-12-31",
                feature_count=21,
                data_source="synthetic",
                timesteps=100
            )
        
        # Verify the error was propagated
        assert "Data fetching error" in str(exc_info.value)
        
        # Verify that the cache still exists but no model was added
        assert len(tm.cache["models"]) == 0
    
    @patch('src.data.DataFetcherFactory.create_data_fetcher')
    @patch('src.train.trainer.ModelTrainer')
    def test_get_model_with_model_training_error(self, mock_trainer_class, mock_factory, tmpdir):
        """Test handling of errors in model training"""
        # Setup
        models_dir = tmpdir.mkdir("models")
        
        # Create mock objects
        mock_fetcher = MagicMock()
        
        # Generate sample data
        dates = pd.date_range(start='2020-01-01', periods=30)
        data = pd.DataFrame({
            'Open': np.random.normal(100, 5, 30),
            'High': np.random.normal(105, 5, 30),
            'Low': np.random.normal(95, 5, 30),
            'Close': np.random.normal(100, 5, 30),
            'Volume': np.random.randint(1000, 10000, 30),
            'SMA_20': np.random.normal(100, 2, 30),
            'RSI': np.random.normal(50, 10, 30)
        }, index=dates)
        
        mock_fetcher.fetch_data.return_value = data
        mock_fetcher.add_technical_indicators.return_value = data
        
        mock_trainer = MagicMock()
        mock_trainer.train_model.side_effect = Exception("Training failed")
        mock_trainer_class.return_value = mock_trainer
        
        # Set up mocks
        mock_factory.return_value = mock_fetcher
        
        # Initialize the manager
        tm = TrainingManager(models_dir=str(models_dir), verbose=0)
        
        # Get a new model - should handle the error
        with pytest.raises(Exception) as exc_info:
            tm.get_model(
                symbol="AAPL",
                train_start="2020-01-01",
                train_end="2020-12-31",
                feature_count=21,
                data_source="synthetic",
                timesteps=100
            )
        
        # Verify the error was propagated
        assert "Training failed" in str(exc_info.value)
        
        # Verify that the cache still exists but no model was added
        assert len(tm.cache["models"]) == 0
    
    def test_load_corrupted_cache(self, tmpdir):
        """Test handling of corrupted cache file"""
        # Setup
        models_dir = tmpdir.mkdir("models")
        cache_file = os.path.join(str(models_dir), "model_cache.json")
        
        # Create a corrupted cache file
        with open(cache_file, 'w') as f:
            f.write("{This is not valid JSON")
        
        # Initialize the manager, which should handle the corrupted cache
        tm = TrainingManager(models_dir=str(models_dir))
        
        # Check that the cache was initialized with defaults
        assert tm.cache == {"models": {}}
    
    @patch('src.data.DataFetcherFactory.create_data_fetcher')
    @patch('src.train.trainer.ModelTrainer')
    def test_get_model_with_force_retrain(self, mock_trainer_class, mock_factory, tmpdir):
        """Test forcing retraining even when a cached model exists"""
        # Setup
        models_dir = tmpdir.mkdir("models")
        cache_file = os.path.join(str(models_dir), "model_cache.json")
        
        # Create a fake model file
        model_path = os.path.join(str(models_dir), "test_model")
        with open(f"{model_path}.zip", 'w') as f:
            f.write("mock model data")
        
        # Create a cache file with the model
        config_hash = "abcdef1234567890"
        cache_data = {"models": {config_hash: {
            "model_path": model_path, 
            "symbol": "AAPL", 
            "train_start": "2020-01-01", 
            "train_end": "2020-12-31",
            "feature_count": 21
        }}}
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
        
        # Set up mock data fetcher
        mock_fetcher = MagicMock()
        
        # Generate sample data
        dates = pd.date_range(start='2020-01-01', periods=30)
        data = pd.DataFrame({
            'Open': np.random.normal(100, 5, 30),
            'High': np.random.normal(105, 5, 30),
            'Low': np.random.normal(95, 5, 30),
            'Close': np.random.normal(100, 5, 30),
            'Volume': np.random.randint(1000, 10000, 30),
            'SMA_20': np.random.normal(100, 2, 30),
            'RSI': np.random.normal(50, 10, 30)
        }, index=dates)
        
        mock_fetcher.fetch_data.return_value = data
        mock_fetcher.add_technical_indicators.return_value = data
        
        # Mock the trainer
        mock_trainer = MagicMock()
        new_model_path = os.path.join(str(models_dir), "new_test_model")
        mock_trainer.train_model.return_value = new_model_path
        mock_trainer.load_model.return_value = "mock_model"
        mock_trainer_class.return_value = mock_trainer
        
        # Set up factory mock
        mock_factory.return_value = mock_fetcher
        
        # Initialize the manager
        tm = TrainingManager(models_dir=str(models_dir))
        
        # Patch the hash generation to return our predefined hash
        with patch.object(tm, '_generate_config_hash', return_value=config_hash):
            # Get the model with force_train=True
            model, path = tm.get_model(
                symbol="AAPL",
                train_start="2020-01-01",
                train_end="2020-12-31",
                force_train=True
            )
        
        # Verify that train_model was called despite the cached model
        mock_trainer.train_model.assert_called_once()
        
        # Verify that the model path is the new one
        assert path == new_model_path

if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 