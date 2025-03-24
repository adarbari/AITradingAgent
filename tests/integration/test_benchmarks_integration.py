"""
Integration tests for the benchmark functionality.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.backtest.backtester import Backtester
from src.backtest.benchmarks import BenchmarkFactory
from src.agent.trading_env import TradingEnvironment
from stable_baselines3 import PPO
import os
import shutil

class TestBenchmarksIntegration:
    """Integration tests for benchmarks with backtester"""
    
    @pytest.fixture
    def test_data(self):
        """Create sample test data"""
        # Use past dates instead of future dates
        dates = pd.date_range(start="2023-01-01", end="2023-01-31", freq='D')
        data = pd.DataFrame(index=dates)
        data['Close'] = np.linspace(100, 150, len(dates))  # Simple linear price trend
        data['Open'] = data['Close'] - 1
        data['High'] = data['Close'] + 2
        data['Low'] = data['Close'] - 2
        data['Volume'] = np.random.randint(1000000, 2000000, size=len(dates))
        return data
    
    @pytest.fixture
    def trained_model(self, test_data):
        """Create and train a simple model for testing"""
        # Create environment
        env = TradingEnvironment(
            prices=test_data['Close'].values,
            features=np.random.randn(len(test_data), 5),  # 5 random features
            initial_balance=10000,
            transaction_fee_percent=0.001  # Changed from commission to transaction_fee_percent
        )
        
        # Train model
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=0,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=None,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            use_sde=False,
            sde_sample_freq=-1,
            target_kl=None,
            tensorboard_log=None,
            policy_kwargs=dict(
                net_arch=[dict(pi=[64, 64], vf=[64, 64])]
            )
        )
        
        model.learn(total_timesteps=1000)
        return model
    
    def test_full_backtest_with_benchmarks(self, test_data, trained_model):
        """Test full backtesting process with benchmarks"""
        # Create test results directory
        test_results_dir = "test_results"
        os.makedirs(test_results_dir, exist_ok=True)
    
        # Create synthetic features for testing
        features = np.random.randn(len(test_data), 5)  # 5 random features
    
        # Create environment for testing
        env = TradingEnvironment(
            prices=test_data['Close'].values,
            features=features,
            initial_balance=10000,
            transaction_fee_percent=0.001
        )
    
        # Create a new model with the correct policy
        model = PPO("MultiInputPolicy", env, policy_kwargs=dict(
            net_arch=dict(pi=[64, 64], vf=[64, 64])
        ))
        model.learn(total_timesteps=100)  # Train for a few steps
    
        # Save the model
        model_path = os.path.join(test_results_dir, "test_model")
        model.save(model_path)
    
        # Initialize backtester
        backtester = Backtester(results_dir=test_results_dir)
    
        # Run backtest
        start_date = test_data.index[0].strftime('%Y-%m-%d')
        end_date = test_data.index[-1].strftime('%Y-%m-%d')
    
        results = backtester.backtest_model(
            model_path=model_path,
            symbol="AAPL",  # Use a real stock symbol
            test_start=start_date,
            test_end=end_date,
            data_source="yfinance",
            env_class=TradingEnvironment
        )
    
        # Verify results structure
        assert isinstance(results, dict)
        assert 'returns' in results
        assert 'total_return' in results
        assert 'sharpe_ratio' in results
        assert 'max_drawdown' in results
        assert 'benchmark_results' in results
    
        # Verify benchmark results
        assert 'buy_and_hold' in results['benchmark_results']
        assert 'sp500' in results['benchmark_results']
        assert 'nasdaq' in results['benchmark_results']

        # Clean up
        if os.path.exists(test_results_dir):
            shutil.rmtree(test_results_dir) 