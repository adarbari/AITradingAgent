"""
News data fetcher for retrieving news sentiment data.
"""

import os
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.data.base_data_fetcher import BaseDataFetcher


class NewsDataFetcher(BaseDataFetcher):
    """
    Fetches news sentiment data related to a stock.
    """
    
    def __init__(self, api_key=None, cache_dir="data/cache", cache_expiry_days=7):
        """
        Initialize the news data fetcher.
        
        Args:
            api_key (str): API key for news API (not needed for simulated data)
            cache_dir (str): Directory to store cached data
            cache_expiry_days (int): Number of days before cache expires
        """
        super().__init__()  # BaseDataFetcher takes no args
        self.api_key = api_key
        self.cache_dir = cache_dir
        self.cache_expiry_days = cache_expiry_days
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
    
    def fetch_data(self, symbol, start_date, end_date, **kwargs):
        """
        Fetch news data for a stock. This is a wrapper around fetch_sentiment_data.
        
        Args:
            symbol (str): Stock ticker symbol
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            **kwargs: Additional parameters
            
        Returns:
            pd.DataFrame: News sentiment data
        """
        # Forward the request to fetch_sentiment_data with keyword arguments
        return self.fetch_sentiment_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            **kwargs
        )
    
    def fetch_sentiment_data(self, symbol, start_date, end_date, **kwargs):
        """
        Fetch sentiment data for a stock.
        
        Args:
            symbol (str): Stock ticker symbol
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            **kwargs: Additional parameters
            
        Returns:
            pd.DataFrame: News sentiment data
        """
        # Check cache first
        cache_data = self._check_cache(symbol, start_date, end_date)
        if cache_data is not None:
            return cache_data
        
        # If no API key is provided, generate simulated data
        if self.api_key is None:
            data = self._generate_simulated_data(symbol, start_date, end_date)
        else:
            # TODO: Implement real API call
            data = self._generate_simulated_data(symbol, start_date, end_date)
        
        # Cache the data
        cache_path = self._get_cache_path(symbol, start_date, end_date)
        self._cache_data(data, cache_path)
        
        return data
    
    def _generate_simulated_data(self, symbol, start_date, end_date):
        """
        Generate simulated sentiment data.
        
        Args:
            symbol (str): Stock ticker symbol
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            
        Returns:
            pd.DataFrame: Simulated news sentiment data
        """
        # Parse dates
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Create date range including the start_date (always include start_date even if weekend)
        date_range = []
        
        # Always include the start date
        date_range.append(start_dt)
        
        # For remaining dates, include only weekdays
        current = start_dt + timedelta(days=1)
        while current <= end_dt:
            # Only include weekdays (Mon-Fri)
            if current.weekday() < 5:
                date_range.append(current)
            current += timedelta(days=1)
        
        # Use symbol as seed for numpy to create deterministic data for the same stock
        # Hash the symbol string to an integer
        symbol_hash = sum(ord(c) for c in symbol)
        np.random.seed(symbol_hash)
        
        # Generate sentiment scores
        sentiment_scores = np.random.normal(0, 0.5, size=len(date_range))
        # Clip to reasonable range
        sentiment_scores = np.clip(sentiment_scores, -1, 1)
        
        # Generate article counts (1-20)
        article_counts = np.random.randint(1, 20, size=len(date_range))
        
        # Generate volatility (0-1)
        volatility = np.abs(sentiment_scores) * 0.5 + np.random.uniform(0, 0.3, len(date_range))
        
        # Create DataFrame with proper date index
        data = pd.DataFrame({
            'Sentiment_Score': sentiment_scores,
            'Article_Count': article_counts,
            'Volatility': volatility
        }, index=date_range)
        
        # Set index name for consistency
        data.index.name = 'Date'
        
        return data
    
    def _get_cache_path(self, symbol, start_date, end_date):
        """
        Get the path to the cache file.
        
        Args:
            symbol (str): Stock ticker symbol
            start_date (str): Start date
            end_date (str): End date
            
        Returns:
            str: Path to cache file
        """
        cache_dir = self.cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        
        cache_filename = f"news_{symbol}_{start_date}_{end_date}.json"
        return os.path.join(cache_dir, cache_filename)
    
    def _check_cache(self, symbol, start_date, end_date):
        """
        Check if data is in cache and not expired.
        
        Args:
            symbol (str): Stock ticker symbol
            start_date (str): Start date
            end_date (str): End date
            
        Returns:
            pd.DataFrame or None: Cached data if valid, None otherwise
        """
        cache_path = self._get_cache_path(symbol, start_date, end_date)
        
        if os.path.exists(cache_path):
            # Check if cache is expired
            if not self._is_cache_expired(cache_path):
                try:
                    # Load cached data
                    data = pd.read_json(cache_path)
                    return data
                except Exception:
                    # If there's any issue reading the cache, ignore it
                    pass
        
        return None
    
    def _is_cache_expired(self, cache_file):
        """
        Check if the cache file is expired.
        
        Args:
            cache_file (str): Path to the cache file
            
        Returns:
            bool: True if expired, False otherwise
        """
        cache_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        expiry_time = datetime.now() - timedelta(days=self.cache_expiry_days)
        return cache_time < expiry_time
    
    def _cache_data(self, data, cache_file):
        """
        Cache data to file.
        
        Args:
            data (pd.DataFrame): Data to cache
            cache_file (str): Path to the cache file
        """
        try:
            # Save to JSON
            data.to_json(cache_file)
        except Exception as e:
            print(f"Error caching data: {e}")
    
    def add_technical_indicators(self, data):
        """
        As this is sentiment data, no technical indicators are added.
        This method is implemented to maintain compatibility with the interface.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Same data without modifications
        """
        return data 