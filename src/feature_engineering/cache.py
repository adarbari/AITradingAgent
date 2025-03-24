"""
Feature Caching Module

Implements caching mechanisms for feature data to avoid recomputation.
"""
import os
import hashlib
import pandas as pd
from typing import List, Dict, Any, Optional, Union
import json
from datetime import datetime


class FeatureCache:
    """
    Cache for storing and retrieving pre-computed features.
    
    This class handles:
    1. Checking if features are already cached
    2. Saving computed features to cache
    3. Loading cached features
    4. Managing cache lifetime and invalidation
    """
    
    def __init__(self, cache_dir: str = ".feature_cache", 
                 max_age_days: int = 30, 
                 enable_cache: bool = True,
                 verbose: bool = False):
        """
        Initialize the feature cache.
        
        Args:
            cache_dir (str): Directory to store cached features
            max_age_days (int): Maximum age of cached features in days
            enable_cache (bool): Whether to use caching
            verbose (bool): Whether to print information about cache operations
        """
        self.cache_dir = cache_dir
        self.max_age_days = max_age_days
        self.enable_cache = enable_cache
        self.verbose = verbose
        
        # Create cache directory if it doesn't exist
        if self.enable_cache:
            os.makedirs(cache_dir, exist_ok=True)
    
    def get_cache_key(self, 
                      symbol: str, 
                      start_date: str, 
                      end_date: str, 
                      feature_set: Union[str, List[str]],
                      feature_config: Dict[str, Any] = None) -> str:
        """
        Generate a cache key for the given parameters.
        
        Args:
            symbol (str): Stock symbol
            start_date (str): Start date
            end_date (str): End date
            feature_set (str or List[str]): Feature set name or list of features
            feature_config (Dict[str, Any]): Additional configuration parameters
            
        Returns:
            str: Cache key
        """
        # If feature_set is a list, convert to sorted string
        if isinstance(feature_set, list):
            feature_set_str = ",".join(sorted(feature_set))
        else:
            feature_set_str = feature_set
        
        # Create a canonical representation of the parameters
        cache_params = {
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "feature_set": feature_set_str,
            "config": feature_config or {}
        }
        
        # Convert to a deterministic string
        cache_str = json.dumps(cache_params, sort_keys=True)
        
        # Generate hash
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def get_cache_path(self, cache_key: str) -> str:
        """
        Get the file path for a cache key.
        
        Args:
            cache_key (str): Cache key
            
        Returns:
            str: File path
        """
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
    
    def is_cached(self, cache_key: str) -> bool:
        """
        Check if features are cached for the given key.
        
        Args:
            cache_key (str): Cache key
            
        Returns:
            bool: Whether features are cached
        """
        if not self.enable_cache:
            return False
        
        cache_path = self.get_cache_path(cache_key)
        
        # Check if cache file exists
        if not os.path.exists(cache_path):
            return False
        
        # Check cache age if max_age_days is set
        if self.max_age_days > 0:
            file_time = os.path.getmtime(cache_path)
            file_age_days = (datetime.now().timestamp() - file_time) / (60 * 60 * 24)
            
            if file_age_days > self.max_age_days:
                if self.verbose:
                    print(f"Cache expired (age: {file_age_days:.1f} days > max: {self.max_age_days} days)")
                return False
        
        return True
    
    def load(self, cache_key: str) -> Optional[pd.DataFrame]:
        """
        Load cached features.
        
        Args:
            cache_key (str): Cache key
            
        Returns:
            pd.DataFrame or None: Cached features or None if not cached
        """
        if not self.enable_cache:
            return None
        
        if not self.is_cached(cache_key):
            return None
        
        cache_path = self.get_cache_path(cache_key)
        
        try:
            if self.verbose:
                print(f"Loading cached features from {cache_path}")
            
            # Load from pickle
            features = pd.read_pickle(cache_path)
            
            # Update access time
            os.utime(cache_path, None)
            
            return features
            
        except Exception as e:
            if self.verbose:
                print(f"Error loading cached features: {e}")
            return None
    
    def save(self, features: pd.DataFrame, cache_key: str) -> bool:
        """
        Save features to cache.
        
        Args:
            features (pd.DataFrame): Features to cache
            cache_key (str): Cache key
            
        Returns:
            bool: Whether features were saved
        """
        if not self.enable_cache:
            return False
        
        cache_path = self.get_cache_path(cache_key)
        
        try:
            if self.verbose:
                print(f"Saving features to cache: {cache_path}")
            
            # Save to pickle
            features.to_pickle(cache_path)
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"Error saving features to cache: {e}")
            return False
    
    def clear(self, older_than_days: Optional[int] = None) -> int:
        """
        Clear cached features.
        
        Args:
            older_than_days (int, optional): Only clear cache entries older than this many days
            
        Returns:
            int: Number of cache entries cleared
        """
        if not self.enable_cache:
            return 0
        
        # Get list of all cache files
        try:
            cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith(".pkl")]
        except FileNotFoundError:
            return 0
        
        cleared_count = 0
        current_time = datetime.now().timestamp()
        
        for cache_file in cache_files:
            cache_path = os.path.join(self.cache_dir, cache_file)
            
            # Check age if older_than_days is set
            if older_than_days is not None:
                file_time = os.path.getmtime(cache_path)
                file_age_days = (current_time - file_time) / (60 * 60 * 24)
                
                if file_age_days <= older_than_days:
                    continue
            
            # Remove file
            try:
                os.remove(cache_path)
                cleared_count += 1
            except Exception as e:
                if self.verbose:
                    print(f"Error removing cache file {cache_path}: {e}")
        
        if self.verbose:
            print(f"Cleared {cleared_count} cache entries")
        
        return cleared_count
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache.
        
        Returns:
            Dict[str, Any]: Cache statistics
        """
        if not self.enable_cache:
            return {"enabled": False}
        
        try:
            cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith(".pkl")]
        except FileNotFoundError:
            return {"enabled": True, "count": 0, "size_mb": 0}
        
        # Calculate total size
        total_size = sum(os.path.getsize(os.path.join(self.cache_dir, f)) for f in cache_files)
        
        # Get age statistics
        current_time = datetime.now().timestamp()
        ages = []
        
        for cache_file in cache_files:
            cache_path = os.path.join(self.cache_dir, cache_file)
            file_time = os.path.getmtime(cache_path)
            file_age_days = (current_time - file_time) / (60 * 60 * 24)
            ages.append(file_age_days)
        
        return {
            "enabled": True,
            "count": len(cache_files),
            "size_mb": total_size / (1024 * 1024),
            "oldest_days": max(ages) if ages else 0,
            "newest_days": min(ages) if ages else 0,
            "avg_age_days": sum(ages) / len(ages) if ages else 0
        } 