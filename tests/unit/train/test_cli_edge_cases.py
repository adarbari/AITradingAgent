#!/usr/bin/env python3
"""
Unit tests for edge cases of the CLI interface in src/train/cli.py
"""
import os
import json
import pytest
from unittest.mock import patch, MagicMock
import argparse
from datetime import datetime
import sys

from src.train.cli import train_model, list_models, clear_models, main

class TestCLIEdgeCases:
    """Edge case tests for the CLI module"""
    
    @patch('src.train.cli.TrainingManager')
    def test_train_model_with_invalid_data_source(self, mock_training_manager_class):
        """Test train_model with invalid data source"""
        # Setup mock args
        args = argparse.Namespace(
            symbols=["AAPL"],
            train_start="2020-01-01",
            train_end="2020-12-31",
            force=False,
            data_source="invalid_source",
            timesteps=100,
            feature_count=21,
            models_dir="models",
            verbose=1,
            synthetic_initial_price=100.0,
            synthetic_volatility=0.02,
            synthetic_drift=0.001,
            learning_rate=None,
            n_steps=None,
            batch_size=None,
            gamma=None
        )
        
        # Setup mock training manager
        mock_training_manager = MagicMock()
        mock_training_manager_class.return_value = mock_training_manager
        
        # Execute with a patch to prevent sys.exit
        with patch('sys.stdout'), patch('sys.stderr'):
            try:
                train_model(args)
            except Exception:
                pass  # Expected to raise an exception
        
        # Verify that get_model was called with the expected parameters
        mock_training_manager.get_model.assert_called_once()
    
    @patch('src.train.cli.TrainingManager')
    def test_train_model_with_future_dates(self, mock_training_manager_class):
        """Test train_model with future dates"""
        # Get a future date
        future_year = datetime.now().year + 1
        future_date = f"{future_year}-01-01"
        
        # Setup mock args
        args = argparse.Namespace(
            symbols=["AAPL"],
            train_start="2020-01-01",
            train_end=future_date,
            force=False,
            data_source="synthetic",
            timesteps=100,
            feature_count=21,
            models_dir="models",
            verbose=1,
            synthetic_initial_price=100.0,
            synthetic_volatility=0.02,
            synthetic_drift=0.001,
            learning_rate=None,
            n_steps=None,
            batch_size=None,
            gamma=None
        )
        
        # Setup mock training manager
        mock_training_manager = MagicMock()
        mock_training_manager.get_model.return_value = (MagicMock(), "mock_model_path")  # Return mock values
        mock_training_manager_class.return_value = mock_training_manager
        
        # Execute with a patch to capture stdout
        with patch('sys.stdout'):
            train_model(args)
        
        # Verify that get_model was called with the expected parameters
        mock_training_manager.get_model.assert_called_once()
        
        # Check that the call was made with the expected parameters
        call_args = mock_training_manager.get_model.call_args[1]
        assert call_args['symbol'] == "AAPL"
        assert call_args['train_start'] == "2020-01-01"
        assert call_args['train_end'] == future_date
        assert call_args['data_source'] == "synthetic"
    
    @patch('src.train.cli.TrainingManager')
    def test_list_models_with_nonexistent_models(self, mock_training_manager_class):
        """Test list_models when no models exist"""
        # Setup mock args
        args = argparse.Namespace(
            symbol=None,
            models_dir="models",
            verbose=1
        )
        
        # Setup mock training manager
        mock_training_manager = MagicMock()
        mock_training_manager.list_cached_models.return_value = []
        mock_training_manager_class.return_value = mock_training_manager
        
        # Execute with a patch to capture stdout
        with patch('sys.stdout') as mock_stdout:
            list_models(args)
        
        # Verify that list_cached_models was called
        mock_training_manager.list_cached_models.assert_called_once_with(symbol=None)
    
    @patch('src.train.cli.TrainingManager')
    def test_clear_models_with_confirmation(self, mock_training_manager_class):
        """Test clear_models with user confirmation"""
        # Setup mock args
        args = argparse.Namespace(
            symbol="AAPL",
            older_than=None,
            force=False,
            models_dir="models",
            verbose=1
        )
        
        # Setup mock training manager
        mock_training_manager = MagicMock()
        mock_training_manager.clear_cache.return_value = 1  # 1 model removed
        mock_training_manager_class.return_value = mock_training_manager
        
        # Mock user input for confirmation
        with patch('builtins.input', return_value='y'), patch('sys.stdout'):
            clear_models(args)
        
        # Verify that clear_cache was called with the right parameters
        mock_training_manager.clear_cache.assert_called_once_with(symbol="AAPL", older_than=None)
    
    @patch('src.train.cli.TrainingManager')
    def test_clear_models_with_rejected_confirmation(self, mock_training_manager_class):
        """Test clear_models when user rejects confirmation"""
        # Setup mock args
        args = argparse.Namespace(
            symbol="AAPL",
            older_than=None,
            force=False,
            models_dir="models",
            verbose=1
        )
        
        # Setup mock training manager
        mock_training_manager = MagicMock()
        mock_training_manager_class.return_value = mock_training_manager
        
        # Mock user input to reject confirmation
        with patch('builtins.input', return_value='n'), patch('sys.stdout'):
            clear_models(args)
        
        # Verify that clear_cache was not called
        mock_training_manager.clear_cache.assert_not_called()
    
    @patch('sys.argv', ['cli.py', 'train', '--symbols', 'AAPL', 'MSFT', 'GOOGL'])
    @patch('src.train.cli.train_model')
    def test_main_with_train_command(self, mock_train_model):
        """Test main function with train command"""
        main()
        mock_train_model.assert_called_once()
        args = mock_train_model.call_args[0][0]
        assert args.symbols == ['AAPL', 'MSFT', 'GOOGL']
        assert args.command == 'train'
    
    @patch('sys.argv', ['cli.py', 'list'])
    @patch('src.train.cli.list_models')
    def test_main_with_list_command(self, mock_list_models):
        """Test main function with list command"""
        main()
        mock_list_models.assert_called_once()
        args = mock_list_models.call_args[0][0]
        assert args.command == 'list'
    
    @patch('sys.argv', ['cli.py', 'clear', '--force'])
    @patch('src.train.cli.clear_models')
    def test_main_with_clear_command(self, mock_clear_models):
        """Test main function with clear command"""
        main()
        mock_clear_models.assert_called_once()
        args = mock_clear_models.call_args[0][0]
        assert args.command == 'clear'
        assert args.force
    
    @patch('sys.argv', ['cli.py', 'invalid_command'])
    def test_main_with_invalid_command(self):
        """Test main function with invalid command"""
        # We expect the main function to exit with a non-zero status code when an invalid command is provided
        with pytest.raises(SystemExit):
            main()

if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 