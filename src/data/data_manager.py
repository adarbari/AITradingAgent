"""
Data manager for handling all data access in the system.
Acts as a unified interface for the multi-agent system to access different types of data.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from .data_fetcher_factory import DataFetcherFactory

class DataManager:
    """
    Manages data access for all agents in the system.
    Provides a unified interface for accessing different types of data.
    """
    
    def __init__(self, market_data_source="yahoo", news_data_source=None, 
                 economic_data_source=None, social_data_source=None,
                 cache_data=True, verbose=1):
        """
        Initialize the DataManager with different data sources.
        
        Args:
            market_data_source (str): Source for market data ('yahoo', 'synthetic', etc.)
            news_data_source (str): Source for news data (optional)
            economic_data_source (str): Source for economic data (optional)
            social_data_source (str): Source for social sentiment data (optional)
            cache_data (bool): Whether to cache data to avoid repeated API calls
            verbose (int): Verbosity level (0: silent, 1: normal, 2: detailed)
        """
        self.data_sources = {}
        self.cache = {} if cache_data else None
        self.verbose = verbose
        
        # Set up data fetchers for each data type
        self.data_sources["market"] = DataFetcherFactory.create_data_fetcher(market_data_source)
        
        if news_data_source:
            self.data_sources["news"] = DataFetcherFactory.create_data_fetcher(news_data_source)
        
        if economic_data_source:
            self.data_sources["economic"] = DataFetcherFactory.create_data_fetcher(economic_data_source)
            
        if social_data_source:
            self.data_sources["social"] = DataFetcherFactory.create_data_fetcher(social_data_source)
    
    def _get_from_cache(self, cache_key):
        """Get data from cache if available"""
        if self.cache is not None and cache_key in self.cache:
            if self.verbose > 1:
                print(f"Using cached data for {cache_key}")
            return self.cache[cache_key]
        return None
    
    def _cache_data(self, cache_key, data):
        """Cache data for future use"""
        if self.cache is not None and data is not None:
            self.cache[cache_key] = data
    
    def get_market_data(self, symbol, start_date, end_date, include_indicators=True):
        """
        Get market data with optional technical indicators.
        
        Args:
            symbol (str): Stock symbol
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            include_indicators (bool): Whether to include technical indicators
            
        Returns:
            pd.DataFrame: DataFrame with market data
            None: If data fetching failed
        """
        cache_key = f"market_{symbol}_{start_date}_{end_date}_{include_indicators}"
        cached_data = self._get_from_cache(cache_key)
        
        if cached_data is not None:
            return cached_data
            
        if "market" not in self.data_sources or self.data_sources["market"] is None:
            if self.verbose > 0:
                print("No market data source configured")
            return None
            
        try:
            data = self.data_sources["market"].fetch_data(symbol, start_date, end_date)
            
            if include_indicators and data is not None:
                data = self.data_sources["market"].add_technical_indicators(data)
                
            self._cache_data(cache_key, data)
            return data
        except Exception as e:
            if self.verbose > 0:
                print(f"Error fetching market data: {e}")
            return None
    
    def get_sentiment_data(self, symbol, start_date, end_date):
        """
        Get news sentiment data if available.
        
        Args:
            symbol (str): Stock symbol
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            
        Returns:
            pd.DataFrame: DataFrame with sentiment data
            None: If not available or error occurred
        """
        cache_key = f"sentiment_{symbol}_{start_date}_{end_date}"
        cached_data = self._get_from_cache(cache_key)
        
        if cached_data is not None:
            return cached_data
            
        if "news" not in self.data_sources or self.data_sources["news"] is None:
            if self.verbose > 1:
                print("No news data source configured")
            return None
            
        try:
            data = self.data_sources["news"].fetch_sentiment_data(symbol, start_date, end_date)
            self._cache_data(cache_key, data)
            return data
        except Exception as e:
            if self.verbose > 0:
                print(f"Error fetching sentiment data: {e}")
            return None
    
    def get_economic_data(self, indicators, start_date, end_date):
        """
        Get economic indicators if available.
        
        Args:
            indicators (list): List of economic indicator names
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            
        Returns:
            pd.DataFrame: DataFrame with economic data
            None: If not available or error occurred
        """
        cache_key = f"economic_{'_'.join(indicators)}_{start_date}_{end_date}"
        cached_data = self._get_from_cache(cache_key)
        
        if cached_data is not None:
            return cached_data
            
        if "economic" not in self.data_sources or self.data_sources["economic"] is None:
            if self.verbose > 1:
                print("No economic data source configured")
            return None
            
        try:
            data = self.data_sources["economic"].fetch_economic_data(indicators, start_date, end_date)
            self._cache_data(cache_key, data)
            return data
        except Exception as e:
            if self.verbose > 0:
                print(f"Error fetching economic data: {e}")
            return None
    
    def get_social_sentiment(self, symbol, start_date, end_date):
        """
        Get social media sentiment data if available.
        
        Args:
            symbol (str): Stock symbol
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            
        Returns:
            pd.DataFrame: DataFrame with social sentiment data
            None: If not available or error occurred
        """
        cache_key = f"social_{symbol}_{start_date}_{end_date}"
        cached_data = self._get_from_cache(cache_key)
        
        if cached_data is not None:
            return cached_data
            
        if "social" not in self.data_sources or self.data_sources["social"] is None:
            if self.verbose > 1:
                print("No social sentiment data source configured")
            return None
            
        try:
            data = self.data_sources["social"].fetch_social_sentiment(symbol, start_date, end_date)
            self._cache_data(cache_key, data)
            return data
        except Exception as e:
            if self.verbose > 0:
                print(f"Error fetching social sentiment data: {e}")
            return None
    
    def get_correlation_data(self, symbols, start_date, end_date):
        """
        Get correlation data between multiple symbols.
        
        Args:
            symbols (list): List of stock symbols
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            
        Returns:
            pd.DataFrame: DataFrame with correlation matrix
            None: If not available or error occurred
        """
        if len(symbols) < 2:
            if self.verbose > 0:
                print("Need at least 2 symbols for correlation analysis")
            return None
            
        cache_key = f"correlation_{'_'.join(sorted(symbols))}_{start_date}_{end_date}"
        cached_data = self._get_from_cache(cache_key)
        
        if cached_data is not None:
            return cached_data
            
        if "market" not in self.data_sources or self.data_sources["market"] is None:
            if self.verbose > 0:
                print("No market data source configured")
            return None
            
        try:
            # First try to use a specialized method if available
            if hasattr(self.data_sources["market"], "fetch_correlation_data"):
                data = self.data_sources["market"].fetch_correlation_data(symbols, start_date, end_date)
                if data is not None:
                    self._cache_data(cache_key, data)
                    return data
            
            # If not available, calculate correlations manually
            symbol_data = {}
            for symbol in symbols:
                df = self.get_market_data(symbol, start_date, end_date, include_indicators=False)
                if df is not None:
                    symbol_data[symbol] = df['Close']
            
            if len(symbol_data) < 2:
                if self.verbose > 0:
                    print("Not enough valid data for correlation analysis")
                return None
                
            # Create a dataframe with close prices for each symbol
            price_df = pd.DataFrame(symbol_data)
            
            # Calculate correlation matrix
            corr_matrix = price_df.corr()
            
            self._cache_data(cache_key, corr_matrix)
            return corr_matrix
        except Exception as e:
            if self.verbose > 0:
                print(f"Error calculating correlation data: {e}")
            return None
    
    def prepare_data_for_agent(self, symbol, start_date, end_date, include_sentiment=False, 
                              include_economic=False, include_social=False):
        """
        Prepare complete data set for agent, combining multiple data sources if available.
        
        Args:
            symbol (str): Stock symbol
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            include_sentiment (bool): Whether to include news sentiment
            include_economic (bool): Whether to include economic indicators
            include_social (bool): Whether to include social sentiment
            
        Returns:
            dict: Dictionary with different types of data
            None: If market data fetching failed
        """
        # Get market data as base
        market_data = self.get_market_data(symbol, start_date, end_date)
        
        if market_data is None:
            if self.verbose > 0:
                print(f"Failed to fetch market data for {symbol}")
            return None
            
        # Initialize dictionary for combined data
        combined_data = {"market": market_data}
        
        # Add sentiment data if requested and available
        if include_sentiment:
            sentiment_data = self.get_sentiment_data(symbol, start_date, end_date)
            if sentiment_data is not None:
                combined_data["sentiment"] = sentiment_data
        
        # Add economic data if requested and available
        if include_economic:
            # Default economic indicators
            default_indicators = ["GDP", "CPI", "Unemployment", "Interest_Rate"]
            economic_data = self.get_economic_data(default_indicators, start_date, end_date)
            if economic_data is not None:
                combined_data["economic"] = economic_data
                
        # Add social sentiment if requested and available
        if include_social:
            social_data = self.get_social_sentiment(symbol, start_date, end_date)
            if social_data is not None:
                combined_data["social"] = social_data
                
        return combined_data
    
    def get_features_and_prices(self, symbol, start_date, end_date, feature_count=21):
        """
        Get prepared features and prices for the reinforcement learning agent.
        
        Args:
            symbol (str): Stock symbol
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            feature_count (int): Number of features to include
            
        Returns:
            tuple: (features, prices) arrays for the RL agent
            None: If data fetching or preparation failed
        """
        from src.utils.feature_utils import prepare_robust_features
        
        try:
            # Get market data
            market_data = self.get_market_data(symbol, start_date, end_date)
            
            if market_data is None:
                return None, None
                
            # Prepare features using the utility function
            features = prepare_robust_features(market_data, feature_count)
            prices = market_data['Close'].values
            
            return features, prices
        except Exception as e:
            if self.verbose > 0:
                print(f"Error preparing features: {e}")
            return None, None 