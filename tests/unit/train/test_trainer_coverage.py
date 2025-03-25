#!/usr/bin/env python3
"""
Additional unit tests for the TrainingManager class in src/train/trainer.py
specifically focused on increasing code coverage
"""
import os
import json
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, mock_open, call

from src.train.trainer import TrainingManager
from src.models import ModelTrainer
from src.data import DataFetcherFactory


class TestTrainingManagerCoverage:
    """Additional tests for the TrainingManager class to increase code coverage"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing"""
        dates = pd.date_range(start='2020-01-01', periods=30)
        data = pd.DataFrame({
            'Open': np.random.normal(100, 5, 30),
            'High': np.random.normal(105, 5, 30),
            'Low': np.random.normal(95, 5, 30),
            'Close': np.random.normal(100, 5, 30),
            'Volume': np.random.randint(1000, 10000, 30),
        }, index=dates)
        return data
    
    @pytest.fixture
    def sample_features(self):
        """Generate sample features for testing"""
        return np.random.normal(0, 1, (30, 21))
    
    @patch('os.makedirs')
    def test_init_directory_creation(self, mock_makedirs):
        """Test directory creation in initialization"""
        TrainingManager(models_dir="test_models_dir")
        # The directory is created both in __init__ and potentially in other methods
        expected_calls = [
            call('test_models_dir', exist_ok=True),
            call('test_models_dir', exist_ok=True)
        ]
        mock_makedirs.assert_has_calls(expected_calls)
    
    @patch('builtins.open', new_callable=mock_open, read_data='{"models": {"test": "data"}}')
    def test_load_cache_existing(self, mock_file):
        """Test loading an existing cache file"""
        with patch('os.path.exists', return_value=True):
            tm = TrainingManager()
            assert tm.cache == {"models": {"test": "data"}}
    
    @patch('builtins.open', new_callable=mock_open)
    def test_load_cache_json_error(self, mock_file):
        """Test handling JSON errors when loading cache"""
        with patch('os.path.exists', return_value=True):
            with patch('json.load', side_effect=json.JSONDecodeError("Test error", "", 0)):
                tm = TrainingManager(verbose=1)
                assert tm.cache == {"models": {}}
    
    @patch('builtins.open', new_callable=mock_open)
    def test_load_cache_io_error(self, mock_file):
        """Test handling IO errors when loading cache"""
        with patch('os.path.exists', return_value=True):
            mock_file.side_effect = IOError("Test IO error")
            tm = TrainingManager(verbose=1)
            assert tm.cache == {"models": {}}
    
    @patch('builtins.open', new_callable=mock_open)
    def test_save_cache_io_error(self, mock_file):
        """Test handling IO errors when saving cache"""
        mock_file.side_effect = IOError("Test IO error")
        tm = TrainingManager(verbose=1)
        # Should not raise an exception, just print an error
        tm._save_cache()  
    
    def test_generate_config_hash(self):
        """Test generating a hash for configuration"""
        tm = TrainingManager()
        config1 = {"a": 1, "b": 2}
        config2 = {"b": 2, "a": 1}  # Same content, different order
        
        # Should generate the same hash for equivalent configs
        hash1 = tm._generate_config_hash(config1)
        hash2 = tm._generate_config_hash(config2)
        assert hash1 == hash2
        
        # Different configs should have different hashes
        config3 = {"a": 1, "b": 3}
        hash3 = tm._generate_config_hash(config3)
        assert hash1 != hash3
    
    @patch('os.path.exists')
    @patch('src.train.trainer.ModelTrainer')
    def test_get_model_cached_file_not_found(self, mock_trainer_class, mock_exists, tmpdir):
        """Test get_model when the cached model file is not found"""
        models_dir = str(tmpdir.mkdir("models"))
        cache_file = os.path.join(models_dir, "model_cache.json")
        
        # Create a cache file with a model that doesn't exist
        config_hash = "abcdef1234567890"
        cache_data = {"models": {config_hash: {
            "model_path": os.path.join(models_dir, "missing_model"),
            "symbol": "AAPL", 
            "train_start": "2020-01-01", 
            "train_end": "2020-12-31"
        }}}
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
        
        # Mock ModelTrainer
        mock_trainer = MagicMock()
        mock_trainer_class.return_value = mock_trainer
        
        # Mock os.path.exists to return False for the model file
        mock_exists.side_effect = lambda path: not path.endswith(".zip")
        
        # Initialize TrainingManager
        tm = TrainingManager(models_dir=models_dir, verbose=1)
        
        # Patch necessary methods for training a new model
        with patch.object(tm, '_generate_config_hash', return_value=config_hash):
            with patch('src.data.DataFetcherFactory.create_data_fetcher') as mock_factory:
                # Mock the data fetcher
                mock_fetcher = MagicMock()
                mock_factory.return_value = mock_fetcher
                
                # Mock fetch_data
                mock_fetcher.fetch_data.return_value = pd.DataFrame({
                    'Close': np.random.normal(100, 5, 30)
                })
                mock_fetcher.add_technical_indicators.return_value = mock_fetcher.fetch_data.return_value
                
                # Mock prepare_robust_features
                with patch('src.train.trainer.prepare_robust_features', return_value=np.random.normal(0, 1, (30, 21))):
                    # Mock _save_cache to avoid JSON serialization issues with MagicMock
                    with patch.object(tm, '_save_cache'):
                        # Get the model
                        tm.get_model(
                            symbol="AAPL",
                            train_start="2020-01-01",
                            train_end="2020-12-31"
                        )
        
        # Verify that train_model was called
        mock_trainer.train_model.assert_called_once()
    
    @patch('src.train.trainer.ModelTrainer')
    def test_get_model_cache_load_error(self, mock_trainer_class, tmpdir):
        """Test get_model when there's an error loading the cached model"""
        models_dir = str(tmpdir.mkdir("models"))
        cache_file = os.path.join(models_dir, "model_cache.json")
        
        # Create a fake model file
        model_path = os.path.join(models_dir, "test_model")
        with open(f"{model_path}.zip", 'w') as f:
            f.write("mock model data")
        
        # Create a cache file with the model
        config_hash = "abcdef1234567890"
        cache_data = {"models": {config_hash: {
            "model_path": model_path, 
            "symbol": "AAPL", 
            "train_start": "2020-01-01", 
            "train_end": "2020-12-31"
        }}}
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
        
        # Mock the trainer to raise an exception when loading the model
        mock_trainer = MagicMock()
        mock_trainer.load_model.side_effect = Exception("Test error loading model")
        mock_trainer_class.return_value = mock_trainer
        
        # Mock train_model to return a path and avoid os.rename error
        mock_trainer.train_model.return_value = os.path.join(models_dir, "trained_model")
        
        # Initialize TrainingManager
        tm = TrainingManager(models_dir=models_dir, verbose=1)
        
        # Create a real mock model output for load_model to return after fallback training
        mock_model = MagicMock()
        # Set up load_model to return mock_model on second call (after retraining)
        mock_trainer.load_model.side_effect = [Exception("Test error loading model"), mock_model]
        
        # Patch necessary methods
        with patch.object(tm, '_generate_config_hash', return_value=config_hash):
            with patch('src.data.DataFetcherFactory.create_data_fetcher') as mock_factory:
                # Mock the data fetcher
                mock_fetcher = MagicMock()
                mock_factory.return_value = mock_fetcher
                
                # Mock fetch_data
                mock_fetcher.fetch_data.return_value = pd.DataFrame({
                    'Close': np.random.normal(100, 5, 30)
                })
                mock_fetcher.add_technical_indicators.return_value = mock_fetcher.fetch_data.return_value
                
                # Mock prepare_robust_features
                with patch('src.train.trainer.prepare_robust_features', return_value=np.random.normal(0, 1, (30, 21))):
                    # Mock _save_cache to avoid JSON serialization issues with MagicMock
                    with patch.object(tm, '_save_cache'):
                        with patch('os.rename'):  # Mock os.rename to avoid error
                            # Get the model - should retrain
                            model, path = tm.get_model(
                                symbol="AAPL",
                                train_start="2020-01-01",
                                train_end="2020-12-31"
                            )
        
        # Verify that train_model was called after load_model failed
        mock_trainer.train_model.assert_called_once()
    
    @patch('src.data.DataFetcherFactory.create_data_fetcher')
    @patch('src.train.trainer.ModelTrainer')
    def test_get_model_data_fetcher_error(self, mock_trainer_class, mock_factory, tmpdir, sample_data):
        """Test get_model when the data fetcher raises an error"""
        models_dir = str(tmpdir.mkdir("models"))
        
        # Mock the data fetcher to raise an exception
        primary_fetcher = MagicMock()
        primary_fetcher.fetch_data.side_effect = Exception("Test data fetcher error")
        
        # Mock the synthetic data fetcher (fallback)
        synthetic_fetcher = MagicMock()
        synthetic_fetcher.fetch_data.return_value = sample_data
        synthetic_fetcher.add_technical_indicators.return_value = sample_data
        
        # Configure the factory to return different fetchers
        mock_factory.side_effect = lambda source: primary_fetcher if source == "yahoo" else synthetic_fetcher
        
        # Mock the trainer
        mock_trainer = MagicMock()
        mock_trainer.train_model.return_value = "models/test_model"
        mock_trainer_class.return_value = mock_trainer
        
        # Initialize TrainingManager
        tm = TrainingManager(models_dir=models_dir, verbose=1)
        
        # Mock prepare_robust_features
        with patch('src.train.trainer.prepare_robust_features', return_value=np.random.normal(0, 1, (30, 21))):
            # Get the model - should use fallback data fetcher
            tm.get_model(
                symbol="AAPL",
                train_start="2020-01-01",
                train_end="2020-12-31",
                data_source="yahoo"
            )
        
        # Verify that primary fetcher was tried
        primary_fetcher.fetch_data.assert_called_once()
        
        # Verify that synthetic fetcher was used as fallback
        synthetic_fetcher.fetch_data.assert_called_once()
        
        # Verify that train_model was called
        mock_trainer.train_model.assert_called_once()
    
    @patch('src.data.DataFetcherFactory.create_data_fetcher')
    @patch('src.train.trainer.ModelTrainer')
    def test_get_model_no_training_data(self, mock_trainer_class, mock_factory, tmpdir):
        """Test get_model when no training data is available"""
        models_dir = str(tmpdir.mkdir("models"))
        
        # Mock the data fetcher to return empty data
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_data.return_value = None
        mock_factory.return_value = mock_fetcher
        
        # Mock the trainer
        mock_trainer = MagicMock()
        mock_trainer_class.return_value = mock_trainer
        
        # Initialize TrainingManager
        tm = TrainingManager(models_dir=models_dir, verbose=1)
        
        # Get model - should return None, None
        model, path = tm.get_model(
            symbol="AAPL",
            train_start="2020-01-01",
            train_end="2020-12-31"
        )
        
        # Verify that fetch_data was called (might be called twice in implementation)
        assert mock_fetcher.fetch_data.call_count >= 1
        
        # Verify that train_model was not called
        mock_trainer.train_model.assert_not_called()
        
        # Verify return values
        assert model is None
        assert path is None
    
    @patch('os.rename')
    @patch('src.data.DataFetcherFactory.create_data_fetcher')
    @patch('src.train.trainer.ModelTrainer')
    def test_get_model_rename_error(self, mock_trainer_class, mock_factory, mock_rename, tmpdir, sample_data):
        """Test get_model when there's an error renaming the model file"""
        models_dir = str(tmpdir.mkdir("models"))
        
        # Mock the data fetcher
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_data.return_value = sample_data
        mock_fetcher.add_technical_indicators.return_value = sample_data
        mock_factory.return_value = mock_fetcher
        
        # Mock the trainer
        mock_trainer = MagicMock()
        # Return a different path than the one that will be generated
        mock_trainer.train_model.return_value = os.path.join(models_dir, "original_model")
        mock_trainer.load_model.return_value = MagicMock()
        mock_trainer_class.return_value = mock_trainer
        
        # Mock os.rename to raise an error
        mock_rename.side_effect = OSError("Test rename error")
        
        # Initialize TrainingManager
        tm = TrainingManager(models_dir=models_dir, verbose=1)
        
        # Mock prepare_robust_features
        with patch('src.train.trainer.prepare_robust_features', return_value=np.random.normal(0, 1, (30, 21))):
            # Get the model - should handle the rename error gracefully
            model, path = tm.get_model(
                symbol="AAPL",
                train_start="2020-01-01",
                train_end="2020-12-31"
            )
        
        # Verify that rename was attempted
        mock_rename.assert_called_once()
        
        # Verify that the original path was returned
        assert path == os.path.join(models_dir, "original_model")
    
    @patch('src.train.trainer.ModelTrainer')
    def test_list_cached_models_empty(self, mock_trainer_class):
        """Test listing cached models when the cache is empty"""
        # Create a temporary directory for models
        with patch('os.path.exists', return_value=False):  # Force empty cache by saying file doesn't exist
            with patch.object(TrainingManager, '_load_cache', return_value={"models": {}}):  # Ensure empty cache
                tm = TrainingManager()
                models = tm.list_cached_models()
                assert models == []
    
    @patch('src.train.trainer.ModelTrainer')
    def test_list_cached_models_no_match(self, mock_trainer_class, tmpdir):
        """Test listing cached models with a symbol that doesn't match anything"""
        models_dir = str(tmpdir.mkdir("models"))
        cache_file = os.path.join(models_dir, "model_cache.json")
        
        # Create a cache file with models for AAPL only
        cache_data = {"models": {
            "hash1": {
                "model_path": "model1", 
                "symbol": "AAPL", 
                "train_start": "2020-01-01", 
                "train_end": "2020-12-31"
            },
            "hash2": {
                "model_path": "model2", 
                "symbol": "AAPL", 
                "train_start": "2021-01-01", 
                "train_end": "2021-12-31"
            }
        }}
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
        
        # Initialize TrainingManager
        tm = TrainingManager(models_dir=models_dir)
        
        # List models for a non-existent symbol
        models = tm.list_cached_models(symbol="MSFT")
        assert models == []
    
    @patch('os.remove')
    @patch('src.train.trainer.ModelTrainer')
    def test_clear_cache_all(self, mock_trainer_class, mock_remove, tmpdir):
        """Test clearing the entire cache"""
        models_dir = str(tmpdir.mkdir("models"))
        cache_file = os.path.join(models_dir, "model_cache.json")
        
        # Create model files
        model1_path = os.path.join(models_dir, "model1.zip")
        model2_path = os.path.join(models_dir, "model2.zip")
        
        with open(model1_path, 'w') as f:
            f.write("model1 data")
        
        with open(model2_path, 'w') as f:
            f.write("model2 data")
        
        # Create a cache file with models
        cache_data = {"models": {
            "hash1": {
                "model_path": os.path.join(models_dir, "model1"), 
                "symbol": "AAPL", 
                "train_start": "2020-01-01", 
                "train_end": "2020-12-31",
                "created_at": "2023-01-01 12:00:00"
            },
            "hash2": {
                "model_path": os.path.join(models_dir, "model2"), 
                "symbol": "MSFT", 
                "train_start": "2020-01-01", 
                "train_end": "2020-12-31",
                "created_at": "2023-01-02 12:00:00"
            }
        }}
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
        
        # Initialize TrainingManager
        tm = TrainingManager(models_dir=models_dir)
        
        # Clear the entire cache
        num_cleared = tm.clear_cache()
        
        # Verify that all models were cleared
        assert num_cleared == 2
        assert tm.cache["models"] == {}
        
        # Verify that remove was called for each model
        expected_calls = [
            call(f"{os.path.join(models_dir, 'model1')}.zip"),
            call(f"{os.path.join(models_dir, 'model2')}.zip")
        ]
        mock_remove.assert_has_calls(expected_calls, any_order=True)
    
    @patch('os.remove')
    @patch('src.train.trainer.ModelTrainer')
    def test_clear_cache_by_symbol(self, mock_trainer_class, mock_remove, tmpdir):
        """Test clearing the cache for a specific symbol"""
        models_dir = str(tmpdir.mkdir("models"))
        cache_file = os.path.join(models_dir, "model_cache.json")
        
        # Create model files
        model1_path = os.path.join(models_dir, "model1.zip")
        model2_path = os.path.join(models_dir, "model2.zip")
        
        with open(model1_path, 'w') as f:
            f.write("model1 data")
        
        with open(model2_path, 'w') as f:
            f.write("model2 data")
        
        # Create a cache file with models for different symbols
        cache_data = {"models": {
            "hash1": {
                "model_path": os.path.join(models_dir, "model1"), 
                "symbol": "AAPL", 
                "train_start": "2020-01-01", 
                "train_end": "2020-12-31",
                "created_at": "2023-01-01 12:00:00"
            },
            "hash2": {
                "model_path": os.path.join(models_dir, "model2"), 
                "symbol": "MSFT", 
                "train_start": "2020-01-01", 
                "train_end": "2020-12-31",
                "created_at": "2023-01-02 12:00:00"
            }
        }}
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
        
        # Initialize TrainingManager
        tm = TrainingManager(models_dir=models_dir)
        
        # Clear the cache for AAPL
        num_cleared = tm.clear_cache(symbol="AAPL")
        
        # Verify that only AAPL models were cleared
        assert num_cleared == 1
        assert len(tm.cache["models"]) == 1
        assert "hash2" in tm.cache["models"]
        
        # Verify that remove was called only for the AAPL model
        mock_remove.assert_called_once_with(f"{os.path.join(models_dir, 'model1')}.zip")
    
    @patch('os.remove')
    @patch('src.train.trainer.ModelTrainer')
    def test_clear_cache_by_age(self, mock_trainer_class, mock_remove, tmpdir):
        """Test clearing the cache based on age"""
        models_dir = str(tmpdir.mkdir("models"))
        cache_file = os.path.join(models_dir, "model_cache.json")
        
        # Create model files
        model1_path = os.path.join(models_dir, "model1.zip")
        model2_path = os.path.join(models_dir, "model2.zip")
        
        with open(model1_path, 'w') as f:
            f.write("model1 data")
        
        with open(model2_path, 'w') as f:
            f.write("model2 data")
        
        # Create a date in the past
        old_date = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d %H:%M:%S")
        
        # Create a recent date
        recent_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
        
        # Create a cache file with models of different ages
        cache_data = {"models": {
            "hash1": {
                "model_path": os.path.join(models_dir, "model1"), 
                "symbol": "AAPL", 
                "train_start": "2020-01-01", 
                "train_end": "2020-12-31",
                "created_at": old_date
            },
            "hash2": {
                "model_path": os.path.join(models_dir, "model2"), 
                "symbol": "MSFT", 
                "train_start": "2020-01-01", 
                "train_end": "2020-12-31",
                "created_at": recent_date
            }
        }}
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
        
        # Initialize TrainingManager
        tm = TrainingManager(models_dir=models_dir)
        
        # Clear models older than 5 days
        num_cleared = tm.clear_cache(older_than=5)
        
        # Verify that only old models were cleared
        assert num_cleared == 1
        assert len(tm.cache["models"]) == 1
        assert "hash2" in tm.cache["models"]
        
        # Verify that remove was called only for the old model
        mock_remove.assert_called_once_with(f"{os.path.join(models_dir, 'model1')}.zip")
    
    @patch('os.remove')
    @patch('src.train.trainer.ModelTrainer')
    def test_clear_cache_file_not_found(self, mock_trainer_class, mock_remove, tmpdir):
        """Test clearing the cache when a model file is not found"""
        models_dir = str(tmpdir.mkdir("models"))
        cache_file = os.path.join(models_dir, "model_cache.json")
        
        # Create a cache file with models, but don't create the actual files
        cache_data = {"models": {
            "hash1": {
                "model_path": os.path.join(models_dir, "model1"), 
                "symbol": "AAPL", 
                "train_start": "2020-01-01", 
                "train_end": "2020-12-31",
                "created_at": "2023-01-01 12:00:00"
            }
        }}
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
        
        # Mock os.remove to raise FileNotFoundError
        mock_remove.side_effect = FileNotFoundError("Test file not found")
        
        # Initialize TrainingManager
        tm = TrainingManager(models_dir=models_dir, verbose=1)
        
        # Clear the cache
        num_cleared = tm.clear_cache()
        
        # Verify that the model was still removed from the cache
        assert num_cleared == 1
        assert tm.cache["models"] == {}
    
    @patch('os.remove')
    @patch('src.train.trainer.ModelTrainer')
    def test_clear_cache_other_error(self, mock_trainer_class, mock_remove, tmpdir):
        """Test clearing the cache when an unexpected error occurs"""
        models_dir = str(tmpdir.mkdir("models"))
        cache_file = os.path.join(models_dir, "model_cache.json")
        
        # Create a cache file with models
        cache_data = {"models": {
            "hash1": {
                "model_path": os.path.join(models_dir, "model1"), 
                "symbol": "AAPL", 
                "train_start": "2020-01-01", 
                "train_end": "2020-12-31",
                "created_at": "2023-01-01 12:00:00"
            }
        }}
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
        
        # Mock os.remove to raise a different kind of error
        mock_remove.side_effect = Exception("Test unexpected error")
        
        # Initialize TrainingManager
        tm = TrainingManager(models_dir=models_dir, verbose=1)
        
        # Clear the cache - should handle the error gracefully
        num_cleared = tm.clear_cache()
        
        # Verify that the model was still removed from the cache
        assert num_cleared == 1
        assert tm.cache["models"] == {} 