"""
Factory class for creating data fetchers
"""
from .base_data_fetcher import BaseDataFetcher
from .synthetic_data_fetcher import SyntheticDataFetcher
from .yahoo_data_fetcher import YahooDataFetcher
from .news_data_fetcher import NewsDataFetcher

class DataFetcherFactory:
    """Factory class for creating data fetchers"""
    
    @staticmethod
    def create_data_fetcher(data_source):
        """
        Create a data fetcher based on the specified data source.
        
        Args:
            data_source (str): The data source to use.
            
        Returns:
            BaseDataFetcher: An instance of a data fetcher.
            
        Raises:
            ValueError: If an invalid data source is specified.
        """
        if data_source is None:
            return None
            
        if isinstance(data_source, str):
            data_source = data_source.lower()
            
            if data_source == 'yahoo':
                return YahooDataFetcher()
            elif data_source == 'synthetic':
                return SyntheticDataFetcher()
            elif data_source == 'news':
                return NewsDataFetcher()
            elif data_source == 'economic':
                # Economic data fetcher not implemented yet, fallback to dummy implementation
                from .news_data_fetcher import NewsDataFetcher
                return NewsDataFetcher()  # Placeholder until EconomicDataFetcher is implemented
            elif data_source == 'social':
                # Social data fetcher not implemented yet, fallback to dummy implementation
                from .news_data_fetcher import NewsDataFetcher
                return NewsDataFetcher()  # Placeholder until SocialDataFetcher is implemented
            else:
                raise ValueError(f"Invalid data source: {data_source}. Valid options are 'yahoo', 'synthetic', 'news', 'economic', or 'social'.")
        elif isinstance(data_source, BaseDataFetcher):
            # If an instance is passed directly, return it
            return data_source
        else:
            raise ValueError(f"Data source must be a string or BaseDataFetcher instance, got {type(data_source)}.") 