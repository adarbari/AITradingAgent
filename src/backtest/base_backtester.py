"""
Base interface for backtesting systems
"""
from abc import ABC, abstractmethod


class BaseBacktester(ABC):
    """
    Abstract base class for backtesting systems
    
    This defines the interface that all backtesters must implement
    to ensure interchangeability.
    """
    
    @abstractmethod
    def backtest_model(self, model_name, test_start_date, test_end_date, data_source, 
                       data_fetcher_factory, trading_env_class, **kwargs):
        """
        Backtest a trained model on test data
        
        Args:
            model_name (str): Name of the model to test
            test_start_date (str): Start date for test data in YYYY-MM-DD format
            test_end_date (str): End date for test data in YYYY-MM-DD format
            data_source (str): Source of data (e.g., 'synthetic', 'yahoo')
            data_fetcher_factory: Factory to create data fetchers
            trading_env_class: Class to use for the trading environment
            **kwargs: Additional keyword arguments specific to the implementation
            
        Returns:
            dict: Dictionary of backtest results or None if backtest failed
        """
        pass
    
    @abstractmethod
    def get_market_performance(self, symbol, test_start_date, test_end_date, data_source, 
                              data_fetcher_factory, **kwargs):
        """
        Get market performance for comparison
        
        Args:
            symbol (str): Market symbol to use (e.g., "^IXIC" for NASDAQ)
            test_start_date (str): Start date for comparison in YYYY-MM-DD format
            test_end_date (str): End date for comparison in YYYY-MM-DD format
            data_source (str): Source of data (e.g., 'synthetic', 'yahoo')
            data_fetcher_factory: Factory to create data fetchers
            **kwargs: Additional keyword arguments specific to the implementation
            
        Returns:
            numpy.ndarray: Normalized market price data or None if not available
        """
        pass
    
    @abstractmethod
    def plot_comparison(self, results_dict, market_performance, test_start_date, test_end_date, **kwargs):
        """
        Plot comparison of model performances against market
        
        Args:
            results_dict (dict): Dictionary of backtest results for each symbol
            market_performance (numpy.ndarray): Normalized market price data
            test_start_date (str): Start date for test data in YYYY-MM-DD format
            test_end_date (str): End date for test data in YYYY-MM-DD format
            **kwargs: Additional keyword arguments specific to the implementation
            
        Returns:
            str: Path to the saved chart or None if plotting failed
        """
        pass 