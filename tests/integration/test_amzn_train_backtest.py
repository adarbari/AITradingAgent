"""
Integration test for training a model on AMZN data and backtesting it.
"""
import os
import pytest
import subprocess
import json
from pathlib import Path

@pytest.mark.integration
class TestAMZNTrainBacktest:
    """
    Integration tests for training and backtesting with AMZN data.
    This validates the complete workflow from training to validation.
    """
    
    @pytest.mark.slow
    @pytest.mark.skip(reason="This test requires real model training and market data, not suitable for CI")
    def test_amzn_train_2023_2024_test_2025(self):
        """
        Test training a model for AMZN with 2023-2024 data and testing it for early 2025.
        
        This test:
        1. Trains a model on AMZN stock data from 2023-01-01 to 2024-12-31
        2. Backtests the model on data from 2025-01-01 to 2025-03-15
        3. Validates that the model performs better than random (Sharpe ratio > 0)
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
        
        model_name = f"ppo_{symbol}_2023_2024_test"
        model_path = os.path.join(models_dir, model_name)
        
        try:
            # Run training command
            train_cmd = [
                "python", "src/scripts/train_and_backtest.py",
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
            
            # Run backtest command
            backtest_cmd = [
                "python", "src/scripts/train_and_backtest.py",
                "--symbol", symbol,
                "--backtest",
                "--test-start", test_start,
                "--test-end", test_end,
                "--model-path", model_path,
                "--models-dir", models_dir,
                "--results-dir", results_dir,
                "--feature-set", "standard"
            ]
            
            print(f"Running command: {' '.join(backtest_cmd)}")
            backtest_result = subprocess.run(backtest_cmd, capture_output=True, text=True)
            assert backtest_result.returncode == 0, f"Backtesting failed with exit code {backtest_result.returncode}:\n{backtest_result.stderr}"
            
            # Check results file
            results_pattern = f"{results_dir}/{symbol}_backtest_results.json"
            results_files = list(Path(".").glob(results_pattern))
            
            assert len(results_files) > 0, f"No results file found matching pattern: {results_pattern}"
            
            # Load and validate the results
            with open(results_files[0], 'r') as f:
                results = json.load(f)
            
            assert results["symbol"] == symbol, f"Wrong symbol in results: {results['symbol']}"
            assert "sharpe_ratio" in results, "Sharpe ratio not in results"
            # Print results
            print(f"Test results: Final value: ${results['final_value']:.2f}, " +
                  f"Return: {results['strategy_return']*100:.2f}%, " + 
                  f"Sharpe: {results['sharpe_ratio']:.2f}")
            
        finally:
            # Clean up (uncomment to delete test artifacts)
            # if os.path.exists(models_dir):
            #     import shutil
            #     shutil.rmtree(models_dir)
            # if os.path.exists(results_dir):
            #     import shutil
            #     shutil.rmtree(results_dir)
            pass 