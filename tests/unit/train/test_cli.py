#!/usr/bin/env python3
"""
Unit tests for the CLI module in src/train/cli.py
"""
import os
import pytest
from unittest.mock import patch, MagicMock

from src.train.cli import train_model, list_models, clear_models, main

class TestTrainingCLI:
    """Tests for the training CLI module"""
    
    @patch('argparse.ArgumentParser.parse_args')
    def test_parse_args_train(self, mock_parse_args):
        """Test parsing arguments for the train command"""
        # Create a mock args object
        mock_args = MagicMock()
        mock_args.command = 'train'
        mock_args.symbols = ['AAPL', 'MSFT']
        mock_args.train_start = '2020-01-01'
        mock_args.train_end = '2020-12-31'
        mock_args.timesteps = 1000
        mock_args.force = True
        mock_parse_args.return_value = mock_args
        
        # Call main which will use our mocked parse_args
        with patch('src.train.cli.train_model') as mock_train:
            main()
            mock_train.assert_called_once_with(mock_args)
    
    @patch('argparse.ArgumentParser.parse_args')
    def test_parse_args_list(self, mock_parse_args):
        """Test parsing arguments for the list command"""
        # Create a mock args object
        mock_args = MagicMock()
        mock_args.command = 'list'
        mock_args.symbol = 'AAPL'
        mock_parse_args.return_value = mock_args
        
        # Call main which will use our mocked parse_args
        with patch('src.train.cli.list_models') as mock_list:
            main()
            mock_list.assert_called_once_with(mock_args)
    
    @patch('argparse.ArgumentParser.parse_args')
    def test_parse_args_clear(self, mock_parse_args):
        """Test parsing arguments for the clear command"""
        # Create a mock args object
        mock_args = MagicMock()
        mock_args.command = 'clear'
        mock_args.symbol = 'AAPL'
        mock_args.older_than = '2020-01-01'
        mock_parse_args.return_value = mock_args
        
        # Call main which will use our mocked parse_args
        with patch('src.train.cli.clear_models') as mock_clear:
            main()
            mock_clear.assert_called_once_with(mock_args)
    
    @patch('src.train.cli.TrainingManager')
    def test_train_model(self, mock_tm_class):
        """Test the train_model function"""
        # Setup the mock
        mock_tm = MagicMock()
        mock_tm.get_model.return_value = (MagicMock(), 'model_path')
        mock_tm_class.return_value = mock_tm
        
        # Create args object
        args = MagicMock()
        args.symbols = ['AAPL', 'MSFT']
        args.train_start = '2020-01-01'
        args.train_end = '2020-12-31'
        args.timesteps = 1000
        args.feature_count = 21
        args.data_source = 'yfinance'
        args.models_dir = 'models'
        args.force = False
        args.verbose = 1
        args.synthetic_initial_price = 100.0
        args.synthetic_volatility = 0.02
        args.synthetic_drift = 0.001
        args.learning_rate = None
        args.n_steps = None
        args.batch_size = None
        args.gamma = None
        
        # Call function
        train_model(args)
        
        # Check TrainingManager was initialized correctly
        mock_tm_class.assert_called_once_with(models_dir='models', verbose=1)
        
        # Check get_model was called for each symbol
        assert mock_tm.get_model.call_count == 2
    
    @patch('src.train.cli.TrainingManager')
    def test_list_models(self, mock_tm_class):
        """Test the list_models function"""
        # Setup the mock
        mock_tm = MagicMock()
        mock_tm.list_cached_models.return_value = [
            {
                'symbol': 'AAPL',
                'train_start': '2020-01-01',
                'train_end': '2020-12-31',
                'feature_count': 21,
                'data_source': 'yahoo',
                'created_at': '2023-01-01 12:00:00',
                'model_path': 'models/aapl_model',
                'hash': 'abcd1234efgh5678'
            }
        ]
        mock_tm_class.return_value = mock_tm
        
        # Create args object
        args = MagicMock()
        args.symbol = 'AAPL'
        args.models_dir = 'models'
        args.verbose = 1
        
        # Call function with capturing stdout
        with patch('builtins.print') as mock_print:
            list_models(args)
            
            # Verify TrainingManager was initialized correctly
            mock_tm_class.assert_called_once_with(models_dir='models', verbose=1)
            
            # Verify list_cached_models was called with the correct symbol
            mock_tm.list_cached_models.assert_called_once_with(symbol='AAPL')
    
    @patch('src.train.cli.TrainingManager')
    def test_clear_models(self, mock_tm_class):
        """Test the clear_models function"""
        # Setup the mock
        mock_tm = MagicMock()
        mock_tm.clear_cache.return_value = 2
        mock_tm_class.return_value = mock_tm
        
        # Create args object
        args = MagicMock()
        args.symbol = 'AAPL'
        args.older_than = '2020-01-01'
        args.models_dir = 'models'
        args.verbose = 1
        args.force = True  # Skip confirmation prompt
        
        # Call function with capturing stdout
        with patch('builtins.print') as mock_print:
            clear_models(args)
            
            # Verify TrainingManager was initialized correctly
            mock_tm_class.assert_called_once_with(models_dir='models', verbose=1)
            
            # Verify clear_cache was called with the correct parameters
            mock_tm.clear_cache.assert_called_once_with(symbol='AAPL', older_than='2020-01-01')
    
    @patch('src.train.cli.train_model')
    @patch('src.train.cli.list_models')
    @patch('src.train.cli.clear_models')
    @patch('argparse.ArgumentParser.parse_args')
    def test_main_train(self, mock_parse_args, mock_clear, mock_list, mock_train):
        """Test the main function with train command"""
        # Setup mock
        args = MagicMock()
        args.command = 'train'
        mock_parse_args.return_value = args
        
        # Call main
        main()
        
        # Verify correct function was called
        mock_train.assert_called_once_with(args)
        mock_list.assert_not_called()
        mock_clear.assert_not_called()
    
    @patch('src.train.cli.train_model')
    @patch('src.train.cli.list_models')
    @patch('src.train.cli.clear_models')
    @patch('argparse.ArgumentParser.parse_args')
    def test_main_list(self, mock_parse_args, mock_clear, mock_list, mock_train):
        """Test the main function with list command"""
        # Setup mock
        args = MagicMock()
        args.command = 'list'
        mock_parse_args.return_value = args
        
        # Call main
        main()
        
        # Verify correct function was called
        mock_list.assert_called_once_with(args)
        mock_train.assert_not_called()
        mock_clear.assert_not_called()
    
    @patch('src.train.cli.train_model')
    @patch('src.train.cli.list_models')
    @patch('src.train.cli.clear_models')
    @patch('argparse.ArgumentParser.parse_args')
    def test_main_clear(self, mock_parse_args, mock_clear, mock_list, mock_train):
        """Test the main function with clear command"""
        # Setup mock
        args = MagicMock()
        args.command = 'clear'
        mock_parse_args.return_value = args
        
        # Call main
        main()
        
        # Verify correct function was called
        mock_clear.assert_called_once_with(args)
        mock_train.assert_not_called()
        mock_list.assert_not_called()

if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 