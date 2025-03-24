"""
Base interface for model trainers
"""
from abc import ABC, abstractmethod


class BaseModelTrainer(ABC):
    """
    Abstract base class for model trainers
    
    This defines the interface that all model trainers must implement
    to ensure interchangeability.
    """
    
    @abstractmethod
    def train_model(self, symbol, train_start_date, train_end_date, data_source, 
                   data_fetcher_factory, trading_env_class, **kwargs):
        """
        Train a model for a given symbol and date range
        
        Args:
            symbol (str): Stock symbol to train on
            train_start_date (str): Start date for training data in YYYY-MM-DD format
            train_end_date (str): End date for training data in YYYY-MM-DD format
            data_source (str): Source of data (e.g., 'synthetic', 'yahoo')
            data_fetcher_factory: Factory to create data fetchers
            trading_env_class: Class to use for the trading environment
            **kwargs: Additional keyword arguments specific to the implementation
        
        Returns:
            Trained model or None if training failed
        """
        pass
    
    @abstractmethod
    def load_model(self, model_name, **kwargs):
        """
        Load a previously trained model
        
        Args:
            model_name (str): Name of the model to load
            **kwargs: Additional keyword arguments specific to the implementation
            
        Returns:
            The loaded model or None if loading failed
        """
        pass
    
    @abstractmethod
    def save_model(self, model, model_name, **kwargs):
        """
        Save a trained model
        
        Args:
            model: The model to save
            model_name (str): Name to save the model under
            **kwargs: Additional keyword arguments specific to the implementation
            
        Returns:
            bool: True if saving was successful, False otherwise
        """
        pass 