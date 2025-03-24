"""
Yahoo Finance data fetcher for retrieving real stock data
"""
import pandas as pd
import numpy as np
import os
import time
import random
import yfinance as yf
from pandas_datareader import data as pdr
from datetime import datetime, timedelta
from functools import wraps
from .base_data_fetcher import BaseDataFetcher

class YahooDataFetcher(BaseDataFetcher):
    """Class for fetching stock data from Yahoo Finance"""
    
    def __init__(self):
        """Initialize the Yahoo Finance data fetcher"""
        self.cache_dir = "data/cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.max_retries = 3
        self.retry_delay = 0.25  # seconds
        self.required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
    
    def _parse_dates(self, start_date, end_date):
        """
        Parse date strings into datetime objects.
        
        Args:
            start_date (str or datetime): Start date in YYYY-MM-DD format or datetime object.
            end_date (str or datetime): End date in YYYY-MM-DD format or datetime object.
            
        Returns:
            tuple: (start_datetime, end_datetime)
        """
        start = start_date if isinstance(start_date, datetime) else datetime.strptime(start_date, "%Y-%m-%d")
        end = end_date if isinstance(end_date, datetime) else datetime.strptime(end_date, "%Y-%m-%d")
        return start, end
    
    def _get_cache_path(self, symbol, start_date, end_date, suffix=""):
        """
        Generate cache file path for a given symbol and date range.
        
        Args:
            symbol (str): The stock symbol.
            start_date (str or datetime): Start date.
            end_date (str or datetime): End date.
            suffix (str): Optional suffix to add to the filename.
            
        Returns:
            str: Path to the cache file.
        """
        if isinstance(start_date, datetime):
            start_str = start_date.strftime("%Y%m%d")
        else:
            start_str = start_date.replace("-", "")
            
        if isinstance(end_date, datetime):
            end_str = end_date.strftime("%Y%m%d")
        else:
            end_str = end_date.replace("-", "")
            
        suffix = f"_{suffix}" if suffix else ""
        return os.path.join(self.cache_dir, f"{symbol}{suffix}_{start_str}_{end_str}.csv")
    
    def _load_from_cache(self, cache_file, symbol, index_col=None, parse_dates=True):
        """
        Load data from cache file if it exists.
        
        Args:
            cache_file (str): Path to the cache file.
            symbol (str): The stock symbol (for logging).
            index_col (int, optional): Column to set as index.
            parse_dates (bool or list): Whether to parse dates.
            
        Returns:
            pd.DataFrame or None: DataFrame if cache exists, None otherwise.
        """
        if os.path.exists(cache_file):
            try:
                if index_col is not None:
                    df = pd.read_csv(cache_file, index_col=index_col, parse_dates=parse_dates)
                else:
                    date_cols = ['Date'] if parse_dates is True else parse_dates
                    df = pd.read_csv(cache_file, parse_dates=date_cols)
                print(f"Loaded cached data for {symbol} from {cache_file}")
                return df
            except Exception as e:
                print(f"Error reading cache file: {e}")
        return None
    
    def _save_to_cache(self, df, cache_file, symbol, index=False):
        """
        Save DataFrame to cache file.
        
        Args:
            df (pd.DataFrame): DataFrame to save.
            cache_file (str): Path to the cache file.
            symbol (str): The stock symbol (for logging).
            index (bool): Whether to include index in the CSV.
            
        Returns:
            pd.DataFrame: The input DataFrame (for method chaining).
        """
        try:
            df.to_csv(cache_file, index=index)
            print(f"Successfully cached {len(df)} rows of data for {symbol}")
        except Exception as e:
            print(f"Error caching data: {e}")
        return df
    
    def _validate_dataframe(self, df, symbol, required_columns=None):
        """
        Validate that DataFrame is not empty and has required columns.
        
        Args:
            df (pd.DataFrame): DataFrame to validate.
            symbol (str): The stock symbol (for error messages).
            required_columns (list, optional): List of required columns.
            
        Returns:
            pd.DataFrame: The validated DataFrame.
            
        Raises:
            ValueError: If validation fails.
        """
        if df.empty:
            raise ValueError(f"Empty dataframe returned for {symbol}")
        
        cols_to_check = required_columns or self.required_columns
        missing_columns = [col for col in cols_to_check if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        return df
    
    def _retry_with_backoff(self, func, symbol, *args, **kwargs):
        """
        Execute a function with retry logic and exponential backoff.
        
        Args:
            func (callable): The function to execute.
            symbol (str): Stock symbol for logging.
            *args, **kwargs: Arguments to pass to the function.
            
        Returns:
            The return value of the function if successful.
            
        Raises:
            Exception: If all retries fail.
        """
        attempts = 0
        last_exception = None
        
        while attempts < self.max_retries:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"Error in attempt {attempts + 1} for {symbol}: {str(e)}")
                last_exception = e
                attempts += 1
                
                # Exponential backoff
                sleep_time = self.retry_delay * (2 ** attempts) + random.uniform(0, 1)
                print(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
        
        # If we get here, all retries failed
        raise last_exception
    
    def _standardize_columns(self, df):
        """
        Standardize column names in the DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame to standardize.
            
        Returns:
            pd.DataFrame: DataFrame with standardized column names.
        """
        # Make sure date column is capitalized
        if 'date' in map(str.lower, df.columns):
            for col in df.columns:
                if col.lower() == 'date':
                    df.rename(columns={col: 'Date'}, inplace=True)
        
        # Standardize OHLCV column names
        rename_map = {}
        for col in df.columns:
            if col.lower() in ['open', 'high', 'low', 'close', 'volume'] and col not in self.required_columns:
                rename_map[col] = col.title()
                
        # Handle special cases
        if 'Stock Splits' in df.columns:
            rename_map['Stock Splits'] = 'Splits'
        if 'Dividends' in df.columns and 'Dividends' not in self.required_columns:
            rename_map['Dividends'] = 'Dividends'
            
        if rename_map:
            df.rename(columns=rename_map, inplace=True)
            
        return df
        
    def fetch_data(self, symbol, start_date, end_date):
        """
        Fetch stock data from Yahoo Finance for a given symbol and date range.
        Implements a caching mechanism and retry logic with fallback to synthetic data.
        
        Args:
            symbol (str): The stock symbol.
            start_date (str): Start date in YYYY-MM-DD format.
            end_date (str): End date in YYYY-MM-DD format.
            
        Returns:
            pd.DataFrame: Dataframe with stock data.
        """
        # Parse dates
        start, end = self._parse_dates(start_date, end_date)
        
        # Define cache file path
        cache_file = self._get_cache_path(symbol, start_date, end_date)
        
        # Check if cached data exists
        cached_df = self._load_from_cache(cache_file, symbol)
        if cached_df is not None:
            return cached_df
        
        # Try to fetch data using pandas_datareader
        try:
            # Define the data fetching function to be used with retry logic
            def fetch_with_pandas_datareader():
                print(f"Fetching {symbol} data using pandas_datareader")
                yf.pdr_override()
                df = pdr.get_data_yahoo(symbol, start=start, end=end)
                
                # Reset index to make Date a column
                df = df.reset_index()
                
                # Validate and standardize
                self._validate_dataframe(df, symbol)
                df = self._standardize_columns(df)
                
                # Cache and add indicators
                self._save_to_cache(df, cache_file, symbol)
                return self.add_technical_indicators(df)
            
            # Try with retry logic
            return self._retry_with_backoff(fetch_with_pandas_datareader, symbol)
            
        except Exception as e:
            print(f"All pandas_datareader attempts failed: {str(e)}")
            
            # Try yfinance directly as a fallback
            try:
                def fetch_with_yfinance():
                    print(f"Trying direct yfinance for {symbol}")
                    ticker = yf.Ticker(symbol)
                    df = ticker.history(start=start, end=end)
                    
                    # Validate
                    self._validate_dataframe(df, symbol, ['Open', 'High', 'Low', 'Close', 'Volume'])
                    
                    # Reset index to make Date a column
                    df = df.reset_index()
                    
                    # Standardize
                    df = self._standardize_columns(df)
                    
                    # Cache and add indicators
                    self._save_to_cache(df, cache_file, symbol)
                    return self.add_technical_indicators(df)
                
                return self._retry_with_backoff(fetch_with_yfinance, symbol)
                
            except Exception as e2:
                print(f"All yfinance direct attempts failed: {str(e2)}")
        
        # If all attempts fail, try to use synthetic data
        print(f"All attempts to fetch {symbol} failed. Trying to use synthetic data...")
        
        # Check if synthetic data exists
        synthetic_file = os.path.join("data/synthetic", f"{symbol}.csv")
        if os.path.exists(synthetic_file):
            try:
                df = pd.read_csv(synthetic_file, parse_dates=['Date'])
                df = df[(df['Date'] >= start) & (df['Date'] <= end)]
                if not df.empty:
                    print(f"Using synthetic data for {symbol}")
                    self._save_to_cache(df, cache_file, symbol)
                    return self.add_technical_indicators(df)
            except Exception as e:
                print(f"Error loading synthetic data: {e}")
        
        # Generate synthetic data as a last resort
        print(f"Generating synthetic data for {symbol}")
        from .synthetic_data_fetcher import SyntheticDataFetcher
        synthetic_fetcher = SyntheticDataFetcher()
        df = synthetic_fetcher.fetch_data(symbol, start_date, end_date)
        self._save_to_cache(df, cache_file, symbol)
        
        return df
        
    def fetch_ticker_data(self, symbol, start_date, end_date, cache=True):
        """
        Fetch data directly using Ticker.history() method for specific use cases like benchmarks.
        
        Args:
            symbol (str): The stock symbol.
            start_date (str): Start date in YYYY-MM-DD format.
            end_date (str): End date in YYYY-MM-DD format.
            cache (bool): Whether to cache the results.
            
        Returns:
            pd.DataFrame: Dataframe with stock data with Date as index.
        """
        # Parse dates
        start, end = self._parse_dates(start_date, end_date)
        
        # Define cache file path if caching is enabled
        cache_file = None
        if cache:
            cache_file = self._get_cache_path(symbol, start, end, suffix="ticker")
            
            # Check if cached data exists
            cached_df = self._load_from_cache(cache_file, symbol, index_col=0, parse_dates=True)
            if cached_df is not None:
                return cached_df
        
        # Try to fetch data using yfinance Ticker.history()
        try:
            def fetch_ticker_history():
                print(f"Fetching {symbol} ticker data")
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start, end=end)
                
                # Validate
                self._validate_dataframe(df, symbol, ['Open', 'High', 'Low', 'Close', 'Volume'])
                
                # If caching is enabled, cache the data
                if cache and cache_file:
                    self._save_to_cache(df, cache_file, symbol, index=True)
                
                return df
            
            return self._retry_with_backoff(fetch_ticker_history, symbol)
                
        except Exception as e:
            print(f"All attempts to fetch {symbol} ticker data failed: {str(e)}")
            return pd.DataFrame()
    
    def fetch_data_simple(self, symbol, start_date, end_date):
        """
        A simpler version of fetch_data that uses yf.download() directly.
        Useful for cases where the full fetch_data functionality isn't needed.
        
        Args:
            symbol (str): The stock symbol.
            start_date (str): Start date in YYYY-MM-DD format.
            end_date (str): End date in YYYY-MM-DD format.
            
        Returns:
            pd.DataFrame: Dataframe with stock data.
        """
        try:
            # Parse dates
            start, end = self._parse_dates(start_date, end_date)
            
            # Define the download function for retry
            def download_data():
                print(f"Downloading simple data for {symbol}")
                data = yf.download(
                    symbol, 
                    start=start_date, 
                    end=end_date, 
                    progress=False
                )
                
                # Validate basic columns
                self._validate_dataframe(data, symbol, ['Open', 'High', 'Low', 'Close', 'Volume'])
                return data
                
            # Use retry logic
            return self._retry_with_backoff(download_data, symbol)
                
        except Exception as e:
            print(f"Error fetching simple data for {symbol}: {e}")
            return None 