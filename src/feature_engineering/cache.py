"""
Feature Caching Module

This module implements a caching system for feature computations.
Features can be cached to disk to avoid redundant calculations.
"""
import os
import time
import pickle
import glob
import numpy as np
import pandas as pd
import hashlib
import json
from typing import Optional, Dict, List, Union, Tuple, Any


class FeatureCache:
    """
    Cache for computed features to avoid redundant calculations.
    
    Features are stored as Pandas DataFrames on disk using pickle.
    Cache keys are generated based on features, stock symbols, and parameters.
    """
    
    def __init__(self, cache_dir: str = ".feature_cache", 
                 enable_cache: bool = True,
                 max_age_days: int = 30,
                 verbose: bool = False):
        """
        Initialize the feature cache.
        
        Args:
            cache_dir (str): Directory to store cached features
            enable_cache (bool): Whether caching is enabled
            max_age_days (int): Maximum age of cache files in days
            verbose (bool): Whether to print cache operations
        """
        self.cache_dir = cache_dir
        self.enable_cache = enable_cache
        self.max_age_days = max_age_days
        self.verbose = verbose
        self.stats = {
            "hits": 0,
            "misses": 0,
            "saves": 0,
            "failures": 0
        }
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
    
    def get_cache_key(self, symbol: str, 
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      feature_set: Union[str, List[str]] = None,
                      **kwargs) -> str:
        """
        Generate a cache key based on features and parameters.
        
        Args:
            symbol (str): Stock symbol
            start_date (Optional[str]): Start date for data
            end_date (Optional[str]): End date for data
            feature_set (Union[str, List[str]]): Feature set name or list of features
            **kwargs: Additional parameters
            
        Returns:
            str: Cache key as an MD5 hash
        """
        # Convert features to a list if it's a string
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
            "config": kwargs or {}
        }
        
        # Convert to a deterministic string
        cache_str = json.dumps(cache_params, sort_keys=True)
        
        # Generate hash
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def get_cache_path(self, key: str) -> str:
        """
        Get the file path for a cache key.
        
        Args:
            key (str): Cache key
            
        Returns:
            str: Path to the cache file
        """
        return os.path.join(self.cache_dir, f"{key}.pkl")
    
    def is_cached(self, key: str) -> bool:
        """
        Check if data for a key is cached.
        
        Args:
            key (str): Cache key
            
        Returns:
            bool: Whether the data is cached
        """
        if not self.enable_cache:
            return False
        
        cache_path = self.get_cache_path(key)
        
        # Check if file exists
        if not os.path.exists(cache_path):
            return False
        
        # Check file age if max_age_days is set
        if self.max_age_days is not None:
            file_time = os.path.getmtime(cache_path)
            max_age_seconds = self.max_age_days * 24 * 60 * 60
            if time.time() - file_time > max_age_seconds:
                return False
        
        return True
    
    def load(self, key: str) -> Optional[pd.DataFrame]:
        """
        Load cached features for a key.
        
        Args:
            key (str): Cache key
            
        Returns:
            Optional[pd.DataFrame]: Cached feature DataFrame or None if not cached
        """
        if not self.enable_cache:
            if self.verbose:
                print("Cache is disabled")
            return None
        
        if not self.is_cached(key):
            if self.verbose:
                print(f"Cache miss for key: {key}")
            self.stats["misses"] += 1
            return None
        
        try:
            cache_path = self.get_cache_path(key)
            data = pd.read_pickle(cache_path)
            
            if self.verbose:
                print(f"Loaded cached features for key: {key}")
            
            self.stats["hits"] += 1
            return data
        except Exception as e:
            # Always print errors for testing purposes, not just when verbose is True
            print(f"Error loading cached features: {e}")
            self.stats["failures"] += 1
            return None
    
    def save(self, data: pd.DataFrame, key: str) -> bool:
        """
        Save features to cache.
        
        Args:
            data (pd.DataFrame): Feature DataFrame
            key (str): Cache key
            
        Returns:
            bool: Whether the save was successful
        """
        if not self.enable_cache:
            if self.verbose:
                print("Cache is disabled")
            return False
        
        try:
            cache_path = self.get_cache_path(key)
            data.to_pickle(cache_path)
            
            if self.verbose:
                print(f"Saved features to cache for key: {key}")
            
            self.stats["saves"] += 1
            return True
        except Exception as e:
            # Always print errors for testing purposes, not just when verbose is True
            print(f"Error saving cached features: {e}")
            self.stats["failures"] += 1
            return False
    
    def invalidate(self, key: str) -> bool:
        """
        Invalidate a cached item.
        
        Args:
            key (str): Cache key
            
        Returns:
            bool: Whether the invalidation was successful
        """
        cache_path = self.get_cache_path(key)
        
        if os.path.exists(cache_path):
            try:
                os.remove(cache_path)
                if self.verbose:
                    print(f"Invalidated cache for key: {key}")
                return True
            except Exception as e:
                if self.verbose:
                    print(f"Error invalidating cache: {e}")
                return False
        
        return False
    
    def clear(self, max_age_days: Optional[int] = None, older_than_days: Optional[int] = None) -> int:
        """
        Clear all cached items or those older than max_age_days or older_than_days.
        
        Args:
            max_age_days (Optional[int]): Only clear items older than this many days
            older_than_days (Optional[int]): Alias for max_age_days for backwards compatibility
            
        Returns:
            int: Number of items cleared
        """
        if not os.path.exists(self.cache_dir):
            return 0
        
        # For backwards compatibility
        if older_than_days is not None and max_age_days is None:
            max_age_days = older_than_days
        
        count = 0
        
        # Get all pkl files in the cache directory
        cache_files = glob.glob(os.path.join(self.cache_dir, "*.pkl"))
        
        for file_path in cache_files:
            should_remove = True
            
            # Check file age if max_age_days is set
            if max_age_days is not None:
                file_time = os.path.getmtime(file_path)
                max_age_seconds = max_age_days * 24 * 60 * 60
                if time.time() - file_time <= max_age_seconds:
                    should_remove = False
            
            if should_remove:
                try:
                    os.remove(file_path)
                    count += 1
                except Exception as e:
                    if self.verbose:
                        print(f"Error removing cache file {file_path}: {e}")
        
        if self.verbose:
            print(f"Cleared {count} cached items")
        
        return count
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache usage statistics.
        
        Returns:
            Dict[str, Any]: Cache statistics
        """
        total_size = 0
        
        # Calculate total size of cache files
        for file_path in glob.glob(os.path.join(self.cache_dir, "*.pkl")):
            try:
                total_size += os.path.getsize(file_path)
            except Exception:
                pass
        
        return {
            "enabled": self.enable_cache,
            "count": len(glob.glob(os.path.join(self.cache_dir, "*.pkl"))),
            "size_mb": total_size / (1024 * 1024) if total_size > 0 else 0,
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "saves": self.stats["saves"],
            "failures": self.stats["failures"]
        } 