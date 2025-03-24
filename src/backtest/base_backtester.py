"""
Base class for backtesting trading strategies.
"""
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class BaseBacktester(ABC):
    """
    Abstract base class for backtesting trading strategies.
    """
    
    def __init__(self, results_dir='results'):
        """
        Initialize the backtester.
        
        Args:
            results_dir (str): Directory to store backtest results
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
    
    @abstractmethod
    def backtest_model(self, model_path, symbol, test_start, test_end, data_source, env_class):
        """
        Backtest a trained model on historical data.
        
        Args:
            model_path (str): Path to the trained model.
            symbol (str): Symbol to backtest on.
            test_start (str): Start date for testing (YYYY-MM-DD).
            test_end (str): End date for testing (YYYY-MM-DD).
            data_source (str): Source of data (e.g., 'yahoo', 'synthetic').
            env_class (class): Environment class to use for backtesting.
            
        Returns:
            dict: Results of the backtest including returns and performance metrics.
        """
        pass
    
    @abstractmethod
    def plot_comparison(self, returns_df, benchmark_results, symbol):
        """
        Plot comparison between model and benchmark performances.
        
        Args:
            returns_df (pd.DataFrame): DataFrame with portfolio values.
            benchmark_results (dict): Dictionary of benchmark results.
            symbol (str): Symbol being backtested.
            
        Returns:
            str: Path to the saved plot.
        """
        pass
    
    @abstractmethod
    def save_results(self, results, file_path):
        """
        Save backtest results to a file.
        
        Args:
            results (dict): Results from the backtest.
            file_path (str): Path to save the results to.
            
        Returns:
            str: Path to the saved file.
        """
        pass
    
    @abstractmethod
    def load_results(self, file_path):
        """
        Load backtest results from a file.
        
        Args:
            file_path (str): Path to the file.
            
        Returns:
            dict: Loaded results.
        """
        pass 