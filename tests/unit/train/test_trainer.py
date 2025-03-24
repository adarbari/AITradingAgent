#!/usr/bin/env python3
"""
Unit tests for the TrainingManager class in src/train/trainer.py
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

class TestTrainingManager:
    """Tests for the TrainingManager class"""
    
    @pytest.fixture
    def mock_data_fetcher(self):
        """Create a mock data fetcher that returns predefined data"""
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
        
        return mock_fetcher
    
    @pytest.fixture
    def mock_model_trainer(self):
        """Create a mock model trainer"""
        mock_trainer = MagicMock()
        mock_trainer.train_model.return_value = "models/test_model"
        mock_trainer.load_model.return_value = MagicMock()
        return mock_trainer
    
    def test_init(self, tmpdir):
        """Test TrainingManager initialization"""
        models_dir = tmpdir.mkdir("models")
        tm = TrainingManager(models_dir=str(models_dir))
        
        # The cache is initialized in memory but not saved to disk yet
        assert tm.cache == {"models": {}}
        
        # Save the cache to disk
        tm._save_cache()
        
        # Now check that the cache file was created in the right place
        assert os.path.exists(os.path.join(str(models_dir), "model_cache.json"))
    
    @patch('src.train.trainer.ModelTrainer')
    def test_load_and_save_cache(self, mock_trainer_class, tmpdir):
        """Test loading and saving the cache"""
        models_dir = tmpdir.mkdir("models")
        cache_file = os.path.join(str(models_dir), "model_cache.json")
        
        # Create a cache file with some content
        cache_data = {"models": {"test_hash": {"model_path": "test_model"}}}
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
        
        # Initialize the manager, which should load the cache
        tm = TrainingManager(models_dir=str(models_dir))
        
        # Check that the cache was loaded correctly
        assert "test_hash" in tm.cache["models"]
        
        # Add a new entry to the cache
        tm.cache["models"]["new_hash"] = {"model_path": "new_model"}
        tm._save_cache()
        
        # Check that the cache was saved correctly
        with open(cache_file, 'r') as f:
            saved_cache = json.load(f)
        
        assert "new_hash" in saved_cache["models"]
    
    @patch('src.data.DataFetcherFactory.create_data_fetcher')
    @patch('src.train.trainer.ModelTrainer')
    def test_get_model_cached(self, mock_trainer_class, mock_factory, tmpdir):
        """Test getting a model that's already in the cache"""
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
        
        # Mock the trainer
        mock_trainer = MagicMock()
        mock_trainer.load_model.return_value = "mock_model"
        mock_trainer_class.return_value = mock_trainer
        
        # Initialize the manager
        tm = TrainingManager(models_dir=str(models_dir))
        
        # Patch the hash generation to return our predefined hash
        with patch.object(tm, '_generate_config_hash', return_value=config_hash):
            # Get the model
            model, path = tm.get_model(
                symbol="AAPL",
                train_start="2020-01-01",
                train_end="2020-12-31"
            )
        
        # Verify that the cached model was returned
        assert model == "mock_model"
        assert path == model_path
        
        # Verify that load_model was called with the correct path
        mock_trainer.load_model.assert_called_once_with(model_path)
        
        # Verify that train_model was not called
        mock_trainer.train_model.assert_not_called()
    
    @patch('src.data.DataFetcherFactory.create_data_fetcher')
    @patch('src.train.trainer.ModelTrainer')
    def test_get_model_new(self, mock_trainer_class, mock_factory, tmpdir):
        """Test getting a model that's not in the cache"""
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
        mock_trainer.train_model.return_value = "models/test_model"
        mock_trainer.load_model.return_value = MagicMock()
        
        # Set up mocks
        mock_factory.return_value = mock_fetcher
        mock_trainer_class.return_value = mock_trainer
        
        # Initialize the manager
        tm = TrainingManager(models_dir=str(models_dir), verbose=0)
        
        # Get a new model
        model, path = tm.get_model(
            symbol="AAPL",
            train_start="2020-01-01",
            train_end="2020-12-31",
            feature_count=21,
            data_source="synthetic",
            timesteps=100
        )
        
        # Verify that train_model was called
        mock_trainer.train_model.assert_called_once()
        
        # Verify that the model was added to the cache
        assert len(tm.cache["models"]) == 1
    
    @patch('src.train.trainer.ModelTrainer')
    def test_list_cached_models(self, mock_trainer_class, tmpdir):
        """Test listing cached models"""
        # Setup
        models_dir = tmpdir.mkdir("models")
        cache_file = os.path.join(str(models_dir), "model_cache.json")
        
        # Create a cache file with multiple models
        cache_data = {"models": {
            "hash1": {
                "model_path": "model1", 
                "symbol": "AAPL", 
                "train_start": "2020-01-01", 
                "train_end": "2020-12-31",
                "feature_count": 21,
                "data_source": "yahoo",
                "created_at": "2023-01-01 12:00:00"
            },
            "hash2": {
                "model_path": "model2", 
                "symbol": "MSFT", 
                "train_start": "2020-01-01", 
                "train_end": "2020-12-31",
                "feature_count": 21,
                "data_source": "synthetic",
                "created_at": "2023-01-02 12:00:00"
            }
        }}
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
        
        # Initialize the manager
        tm = TrainingManager(models_dir=str(models_dir))
        
        # List all models
        models = tm.list_cached_models()
        assert len(models) == 2
        
        # List models for a specific symbol
        aapl_models = tm.list_cached_models(symbol="AAPL")
        assert len(aapl_models) == 1
        assert aapl_models[0]["symbol"] == "AAPL"
    
    @patch('src.train.trainer.ModelTrainer')
    def test_clear_cache(self, mock_trainer_class, tmpdir):
        """Test clearing the cache"""
        # Setup
        models_dir = tmpdir.mkdir("models")
        cache_file = os.path.join(str(models_dir), "model_cache.json")
        
        # Create a cache file with multiple models
        cache_data = {"models": {
            "hash1": {
                "model_path": "model1", 
                "symbol": "AAPL", 
                "train_start": "2020-01-01", 
                "train_end": "2020-12-31",
                "created_at": "2023-01-01 12:00:00"
            },
            "hash2": {
                "model_path": "model2", 
                "symbol": "MSFT", 
                "train_start": "2020-01-01", 
                "train_end": "2020-12-31",
                "created_at": "2023-01-02 12:00:00"
            }
        }}
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
        
        # Create dummy model files
        for path in ["model1", "model2"]:
            model_path = os.path.join(str(models_dir), f"{path}.zip")
            with open(model_path, 'w') as f:
                f.write("mock model data")
        
        # Initialize the manager
        tm = TrainingManager(models_dir=str(models_dir))
        
        # Clear cache for a specific symbol
        num_removed = tm.clear_cache(symbol="AAPL")
        assert num_removed == 1
        
        # Check that only AAPL model was removed
        assert "hash1" not in tm.cache["models"]
        assert "hash2" in tm.cache["models"]
        
        # Clear all remaining models
        num_removed = tm.clear_cache()
        assert num_removed == 1
        
        # Check that all models were removed
        assert len(tm.cache["models"]) == 0

if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 