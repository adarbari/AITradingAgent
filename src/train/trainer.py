"""
Training manager for creating and caching models for various stocks.
"""
import os
import json
from datetime import datetime
import hashlib
import numpy as np
import pandas as pd

from src.models import ModelTrainer
from src.data import DataFetcherFactory
from src.agent.trading_env import TradingEnvironment
from src.utils.feature_utils import prepare_robust_features


class TrainingManager:
    """
    Manager for training and caching models for various stocks.
    
    This class handles:
    1. Checking if a model with the same parameters exists in cache
    2. Training new models when needed
    3. Managing the model cache
    4. Providing models for backtesting
    """
    
    def __init__(self, models_dir="models", cache_file="model_cache.json", verbose=1):
        """
        Initialize the training manager.
        
        Args:
            models_dir (str): Directory to store trained models
            cache_file (str): JSON file to store cache metadata
            verbose (int): Verbosity level (0: silent, 1: normal, 2: detailed)
        """
        self.models_dir = models_dir
        self.cache_file = os.path.join(models_dir, cache_file)
        self.verbose = verbose
        self.model_trainer = ModelTrainer(models_dir=models_dir, verbose=verbose)
        
        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
        
        # Load or initialize the cache
        self.cache = self._load_cache()
    
    def _load_cache(self):
        """Load the model cache from disk or initialize a new one."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                if self.verbose > 0:
                    print(f"Error loading cache file: {e}. Creating new cache.")
                return {"models": {}}
        return {"models": {}}
    
    def _save_cache(self):
        """Save the current cache to disk."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except IOError as e:
            if self.verbose > 0:
                print(f"Error saving cache file: {e}")
    
    def _generate_config_hash(self, config):
        """
        Generate a hash for the training configuration.
        
        Args:
            config (dict): Training configuration
            
        Returns:
            str: Hash string representing the configuration
        """
        # Create a sorted string representation of the config for consistent hashing
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def get_model(self, symbol, train_start, train_end, feature_count=21, 
                 data_source="yahoo", timesteps=100000, force_train=False,
                 synthetic_params=None, model_params=None):
        """
        Get a trained model for the given parameters, using cache if available.
        
        Args:
            symbol (str): Stock symbol
            train_start (str): Start date for training data
            train_end (str): End date for training data
            feature_count (int): Number of features to use
            data_source (str): Source of data ("yahoo", "synthetic")
            timesteps (int): Number of timesteps to train for
            force_train (bool): Force retraining even if cached model exists
            synthetic_params (dict): Parameters for synthetic data generation
            model_params (dict): Additional parameters for the model
            
        Returns:
            tuple: (model, model_path)
        """
        # Create a config dictionary for this training run
        config = {
            "symbol": symbol,
            "train_start": train_start,
            "train_end": train_end,
            "feature_count": feature_count,
            "data_source": data_source,
            "timesteps": timesteps,
            "synthetic_params": synthetic_params if synthetic_params else {},
            "model_params": model_params if model_params else {}
        }
        
        # Generate a hash for this configuration
        config_hash = self._generate_config_hash(config)
        
        # Generate a model name
        model_name = f"ppo_{symbol}_{train_start.split('-')[0]}_{train_end.split('-')[0]}_{config_hash[:8]}"
        model_path = os.path.join(self.models_dir, model_name)
        
        # Check if we already have this model in the cache
        if not force_train and config_hash in self.cache["models"]:
            cached_info = self.cache["models"][config_hash]
            cached_path = cached_info["model_path"]
            
            # Check if the model file actually exists
            if os.path.exists(f"{cached_path}.zip"):
                if self.verbose > 0:
                    print(f"Using cached model for {symbol} with hash {config_hash[:8]}")
                # Load and return the cached model
                try:
                    model = self.model_trainer.load_model(cached_path)
                    return model, cached_path
                except Exception as e:
                    if self.verbose > 0:
                        print(f"Error loading cached model: {e}. Retraining...")
                    # If loading fails, we'll retrain
            else:
                if self.verbose > 0:
                    print(f"Cached model file not found at {cached_path}.zip. Retraining...")
        elif self.verbose > 0 and force_train:
            print(f"Forced retraining for {symbol}")
        elif self.verbose > 0:
            print(f"No cached model found for {symbol} with given parameters. Training new model.")
        
        # If we get here, we need to train a new model
        # Get training data using the appropriate data fetcher
        if self.verbose > 0:
            print(f"Fetching training data for {symbol} from {train_start} to {train_end}...")
        
        # Normalize data source string to match what DataFetcherFactory expects
        fetcher_source = "yahoo" if data_source.lower() == "yahoo" else data_source.lower()
        
        try:
            # Create the appropriate data fetcher using the factory
            data_fetcher = DataFetcherFactory.create_data_fetcher(fetcher_source)
            
            # Fetch the data
            training_data = data_fetcher.fetch_data(symbol, train_start, train_end)
            
            # Add technical indicators if not already present
            if 'SMA_20' not in training_data.columns:
                training_data = data_fetcher.add_technical_indicators(training_data)
        except Exception as e:
            if self.verbose > 0:
                print(f"Error using data fetcher: {e}. Falling back to synthetic data.")
            # Fall back to synthetic data fetcher
            data_fetcher = DataFetcherFactory.create_data_fetcher('synthetic')
            training_data = data_fetcher.fetch_data(symbol, train_start, train_end)
            training_data = data_fetcher.add_technical_indicators(training_data)
        
        if training_data is None or len(training_data) == 0:
            if self.verbose > 0:
                print(f"Error: No training data available for {symbol} from {train_start} to {train_end}.")
            return None, None
        
        if self.verbose > 0:
            print(f"Fetched {len(training_data)} data points for training.")
        
        # Prepare features for the model
        if self.verbose > 0:
            print("Preparing data for the agent...")
        
        # Option 1: Use the data fetcher's prepare_data_for_agent method
        # This may return a different format than we need for our trading environment
        # prices, features = data_fetcher.prepare_data_for_agent(training_data)
        
        # Option 2: Use our custom feature preparation to ensure compatibility
        # with our specific trading environment requirements
        features = prepare_robust_features(training_data, feature_count)
        prices = training_data['Close'].values
        
        if self.verbose > 0:
            print(f"Prepared {len(features)} data points with {features.shape[1]} features.")
        
        # Train the model
        if self.verbose > 0:
            print(f"Training model with {timesteps} timesteps...")
        
        # Train the model using the model trainer
        path = self.model_trainer.train_model(
            env_class=TradingEnvironment,
            prices=prices,
            features=features,
            symbol=symbol,
            train_start=train_start,
            train_end=train_end,
            total_timesteps=timesteps,
            model_params=model_params
        )
        
        # Rename the model file to include the hash
        if path != model_path:
            try:
                os.rename(f"{path}.zip", f"{model_path}.zip")
                path = model_path
            except OSError as e:
                if self.verbose > 0:
                    print(f"Error renaming model file: {e}")
        
        # Load the trained model
        model = self.model_trainer.load_model(path)
        
        # Update the cache
        self.cache["models"][config_hash] = {
            "model_path": path,
            "symbol": symbol,
            "train_start": train_start,
            "train_end": train_end,
            "feature_count": feature_count,
            "data_source": data_source,
            "timesteps": timesteps,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "hash": config_hash
        }
        
        # Save the updated cache
        self._save_cache()
        
        if self.verbose > 0:
            print(f"Model trained and saved to {path}")
        
        return model, path
    
    def list_cached_models(self, symbol=None):
        """
        List all cached models or models for a specific symbol.
        
        Args:
            symbol (str, optional): Filter by symbol
            
        Returns:
            list: List of cached model info dictionaries
        """
        models = []
        for model_hash, info in self.cache["models"].items():
            if symbol is None or info["symbol"] == symbol:
                models.append({
                    "hash": model_hash,
                    "symbol": info["symbol"],
                    "train_start": info["train_start"],
                    "train_end": info["train_end"],
                    "data_source": info.get("data_source", "unknown"),
                    "feature_count": info.get("feature_count", 0),
                    "created_at": info.get("created_at", "unknown"),
                    "model_path": info["model_path"]
                })
        return models
    
    def clear_cache(self, symbol=None, older_than=None):
        """
        Clear the cache for a specific symbol or all symbols.
        
        Args:
            symbol (str, optional): Symbol to clear cache for, or None for all
            older_than (str, optional): Clear models older than this date (YYYY-MM-DD)
            
        Returns:
            int: Number of models removed from cache
        """
        models_to_remove = []
        
        # Filter models to remove
        for model_hash, info in self.cache["models"].items():
            should_remove = False
            
            # Check symbol filter
            if symbol is not None and info["symbol"] != symbol:
                continue
            
            # Check date filter
            if older_than is not None and "created_at" in info:
                try:
                    model_date = datetime.strptime(info["created_at"], "%Y-%m-%d %H:%M:%S")
                    filter_date = datetime.strptime(older_than, "%Y-%m-%d")
                    if model_date >= filter_date:
                        continue
                except ValueError:
                    # If date parsing fails, include it in removal if symbol matches
                    pass
            
            # If we got here, the model should be removed
            models_to_remove.append(model_hash)
        
        # Remove models from cache and disk
        for model_hash in models_to_remove:
            info = self.cache["models"][model_hash]
            model_path = info["model_path"]
            
            # Try to remove the model file
            try:
                if os.path.exists(f"{model_path}.zip"):
                    os.remove(f"{model_path}.zip")
            except OSError as e:
                if self.verbose > 0:
                    print(f"Error removing model file {model_path}.zip: {e}")
            
            # Remove from cache
            del self.cache["models"][model_hash]
        
        # Save the updated cache
        self._save_cache()
        
        return len(models_to_remove) 