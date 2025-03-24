"""
Factory class for creating data fetchers
"""
from .base_data_fetcher import BaseDataFetcher
from .synthetic_data_fetcher import SyntheticDataFetcher
from .yahoo_data_fetcher import YahooDataFetcher

class DataFetcherFactory:
    """Factory class for creating data fetchers"""
    
    @staticmethod
    def create_data_fetcher(data_source):
        """
        Create a data fetcher based on the specified data source.
        
        Args:
            data_source (str): The data source to use ('yahoo' or 'synthetic').
            
        Returns:
            BaseDataFetcher: An instance of a data fetcher.
            
        Raises:
            ValueError: If an invalid data source is specified.
        """
        if data_source.lower() == 'yahoo':
            return YahooDataFetcher()
        elif data_source.lower() == 'synthetic':
            return SyntheticDataFetcher()
        else:
            raise ValueError(f"Invalid data source: {data_source}. Valid options are 'yahoo' or 'synthetic'.") 