"""
Integration test for training a model on AMZN data and backtesting it.
"""
import os
import pytest
import subprocess
import json
from pathlib import Path
import pandas as pd
import numpy as np
import shutil
from datetime import datetime

from src.data.yahoo_data_fetcher import YahooDataFetcher
from src.feature_engineering.cache import FeatureCache

@pytest.mark.integration
class TestAMZNTrainBacktest:
    """
    Integration tests for training and backtesting with AMZN data.
    This validates the complete workflow from training to validation.
    """
    
    @pytest.fixture(scope="class", autouse=True)
    def setup_test_data(self):
        """
        Setup test data by fetching real market data once and storing it in cache.
        This ensures the test can run in CI without repeatedly calling Yahoo Finance APIs.
        """
        # Test parameters
        symbol = "AMZN"
        train_start = "2023-01-01"
        train_end = "2024-12-31"
        test_start = "2025-01-01"
        test_end = "2025-03-15"
        
        # Create directories for cache
        yahoo_cache_dir = "data/cache"
        feature_cache_dir = ".feature_cache"
        os.makedirs(yahoo_cache_dir, exist_ok=True)
        os.makedirs(feature_cache_dir, exist_ok=True)
        
        # Initialize Yahoo data fetcher
        yahoo_fetcher = YahooDataFetcher()
        yahoo_fetcher.cache_dir = yahoo_cache_dir
        
        # Check if data is already cached
        train_cache_file = yahoo_fetcher._get_cache_path(symbol, train_start, train_end)
        test_cache_file = yahoo_fetcher._get_cache_path(symbol, test_start, test_end)
        train_ticker_cache_file = yahoo_fetcher._get_cache_path(symbol, train_start, train_end, suffix="ticker")
        test_ticker_cache_file = yahoo_fetcher._get_cache_path(symbol, test_start, test_end, suffix="ticker")
        
        # Check if all cache files exist before skipping data fetching
        cache_files_exist = all([
            os.path.exists(train_cache_file),
            os.path.exists(test_cache_file),
            os.path.exists(train_ticker_cache_file),
            os.path.exists(test_ticker_cache_file)
        ])
        
        # Fetch data if not in cache
        if not cache_files_exist:
            print(f"Fetching real market data for {symbol} for training and testing periods")
            # Get training data
            train_data = yahoo_fetcher.fetch_data(symbol, train_start, train_end)
            # Get test data
            test_data = yahoo_fetcher.fetch_data(symbol, test_start, test_end)
            # Get ticker data (used by backtest functions)
            train_ticker_data = yahoo_fetcher.fetch_ticker_data(symbol, train_start, train_end)
            test_ticker_data = yahoo_fetcher.fetch_ticker_data(symbol, test_start, test_end)
        else:
            print(f"Using cached market data for {symbol}")
        
        # Pre-compute and cache features if they don't exist yet
        feature_cache = FeatureCache(cache_dir=feature_cache_dir, enable_cache=True)
        
        # Import here to avoid circular imports
        from src.feature_engineering import process_features
        
        # Check if training features are cached
        train_cache_key = feature_cache.get_cache_key(symbol, train_start, train_end, "standard")
        if not feature_cache.is_cached(train_cache_key):
            print(f"Computing features for {symbol} training data")
            # Load the data from cache
            train_data = pd.read_csv(train_cache_file, parse_dates=['Date'])
            # Process and cache training features
            train_features = process_features(train_data, feature_set="standard", verbose=False)
            feature_cache.save(train_features, train_cache_key)
        
        # Check if test features are cached
        test_cache_key = feature_cache.get_cache_key(symbol, test_start, test_end, "standard")
        if not feature_cache.is_cached(test_cache_key):
            print(f"Computing features for {symbol} test data")
            # Load the data from cache
            test_data = pd.read_csv(test_cache_file, parse_dates=['Date'])
            # Process and cache test features
            test_features = process_features(test_data, feature_set="standard", verbose=False)
            feature_cache.save(test_features, test_cache_key)
        
        yield
        
        # We don't clean up the cache here since we want to preserve the data
        # for future test runs
    
    @pytest.mark.slow
    def test_amzn_train_2023_2024_test_2025(self):
        """
        Test training a model for AMZN with 2023-2024 data and testing it for early 2025.
        
        This test:
        1. Trains a model on AMZN stock data from 2023-01-01 to 2024-12-31
        2. Backtests the model on data from 2025-01-01 to 2025-03-15
        3. Validates that the model runs and produces reasonable results
        """
        # Setup test parameters
        symbol = "AMZN"
        train_start = "2023-01-01"
        train_end = "2024-12-31"
        test_start = "2025-01-01"
        test_end = "2025-03-15"
        models_dir = "test_models"
        results_dir = "test_results"
        
        # Create directories if they don't exist
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        model_name = f"ppo_{symbol}_{train_start.split('-')[0]}_{train_end.split('-')[0]}"
        model_path = os.path.join(models_dir, model_name)
        
        try:
            # Run training command
            train_cmd = [
                "python", "../src/scripts/train_and_backtest.py",
                "--symbol", symbol,
                "--train",
                "--train-start", train_start,
                "--train-end", train_end,
                "--model-path", model_path,
                "--models-dir", models_dir,
                "--results-dir", results_dir,
                "--timesteps", "10000",  # Reduced for testing
                "--feature-set", "standard"
            ]
            
            print(f"Running command: {' '.join(train_cmd)}")
            train_result = subprocess.run(train_cmd, capture_output=True, text=True)
            assert train_result.returncode == 0, f"Training failed with exit code {train_result.returncode}:\n{train_result.stderr}"
            
            print(f"Training output: {train_result.stdout}")
            
            # Find the actual model file (which will have a hash added to the name)
            model_files = list(Path(models_dir).glob(f"ppo_{symbol}_*.zip"))
            assert len(model_files) > 0, f"No model file found for {symbol}"
            
            # Use the most recent model file
            actual_model_path = str(max(model_files, key=lambda x: x.stat().st_mtime))
            print(f"Using model: {actual_model_path}")
            
            # Run backtest command
            backtest_cmd = [
                "python", "src/scripts/train_and_backtest.py",
                "--symbol", symbol,
                "--backtest",
                "--test-start", test_start,
                "--test-end", test_end,
                "--model-path", actual_model_path,
                "--models-dir", models_dir,
                "--results-dir", results_dir,
                "--feature-set", "standard"
            ]
            
            print(f"Running command: {' '.join(backtest_cmd)}")
            backtest_result = subprocess.run(backtest_cmd, capture_output=True, text=True)
            assert backtest_result.returncode == 0, f"Backtesting failed with exit code {backtest_result.returncode}:\n{backtest_result.stderr}"
            
            print(f"Backtest output: {backtest_result.stdout}")
            
            # Get the results from the output
            # The backtest script outputs a summary of results in the stdout
            backtest_lines = backtest_result.stdout.strip().split('\n')
            results_summary = {}
            
            # Look for results summary section
            summary_start = False
            for line in backtest_lines:
                if "Backtest Results Summary:" in line:
                    summary_start = True
                    continue
                    
                if summary_start and ":" in line:
                    key, value = line.split(":", 1)
                    results_summary[key.strip()] = value.strip()
            
            # Validate the results from the output even if no file was created
            assert symbol in results_summary.get("Symbol", ""), f"Symbol not found in results summary"
            
            # Check if Sharpe ratio is present in the summary
            if "Sharpe Ratio" in results_summary:
                sharpe_ratio = float(results_summary["Sharpe Ratio"])
                print(f"Sharpe ratio from output: {sharpe_ratio}")
                # We're only validating that the backtest runs and produces a Sharpe ratio within reasonable bounds
                # In a real test, we'd want a positive Sharpe, but for an integration test we're just making sure
                # the workflow completes successfully
                assert -2.0 < sharpe_ratio < 2.0, f"Sharpe ratio ({sharpe_ratio}) is outside reasonable bounds for a test"
            else:
                # Try to find the results file as a fallback
                # Make sure results directory exists for this symbol
                symbol_results_dir = Path(f"{results_dir}/{symbol}")
                os.makedirs(symbol_results_dir, exist_ok=True)
                
                # Check results directory for the most recent backtest results
                results_files = list(symbol_results_dir.glob("backtest_*.json"))
                
                if not results_files:
                    # Try fallback to check for other result file patterns
                    results_files = list(Path(results_dir).glob(f"*{symbol}*backtest*.json"))
                
                if results_files:
                    # Get the most recent results file
                    latest_results_file = max(results_files, key=lambda x: x.stat().st_mtime)
                    
                    # Load and validate the results
                    with open(latest_results_file, 'r') as f:
                        results = json.load(f)
                    
                    assert results["symbol"] == symbol, f"Wrong symbol in results: {results['symbol']}"
                    assert "sharpe_ratio" in results, "Sharpe ratio not in results"
                    # Print results
                    print(f"Test results from file: Final value: ${results['final_value']:.2f}, " +
                        f"Return: {results['strategy_return']*100:.2f}%, " + 
                        f"Sharpe: {results['sharpe_ratio']:.2f}")
                    
                    # Test that the model produces results in a reasonable range
                    # Rather than requiring a specific performance metric
                    assert -2.0 < results["sharpe_ratio"] < 2.0, f"Sharpe ratio ({results['sharpe_ratio']}) is outside reasonable bounds for a test"
                else:
                    # If we couldn't find results in stdout or result files, fail the test
                    assert False, "Could not find backtest results in output or result files"
            
        finally:
            # Clean up (uncomment to delete test artifacts)
            # if os.path.exists(models_dir):
            #     import shutil
            #     shutil.rmtree(models_dir)
            # if os.path.exists(results_dir):
            #     import shutil
            #     shutil.rmtree(results_dir)
            pass 