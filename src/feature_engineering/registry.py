"""
Feature Registry Module

Implements a registry pattern for managing feature generators.
Features can be registered, accessed, and computed via this central registry.
"""
import inspect
import pandas as pd
import numpy as np
from typing import Callable, Dict, List, Any, Union


class FeatureRegistry:
    """
    Registry for feature transformers that can be applied to financial data.
    
    This class provides methods to:
    - Register feature generator functions
    - Access registered features
    - Compute features from data
    - Group features by category
    """
    
    _features: Dict[str, Callable] = {}
    _categories: Dict[str, List[str]] = {}
    _metadata: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def register(cls, name: str, category: str = "other", **metadata):
        """
        Decorator to register a feature generator function.
        
        Args:
            name (str): Name of the feature
            category (str): Category for grouping features
            **metadata: Additional metadata about the feature
        
        Returns:
            Callable: Decorator function
        """
        def decorator(func):
            if name in cls._features:
                raise ValueError(f"Feature '{name}' is already registered.")
            
            cls._features[name] = func
            
            # Add to category
            if category not in cls._categories:
                cls._categories[category] = []
            cls._categories[category].append(name)
            
            # Store metadata
            cls._metadata[name] = {
                "name": name,
                "category": category,
                "doc": func.__doc__,
                "parameters": inspect.signature(func).parameters,
                **metadata
            }
            
            return func
        return decorator
    
    @classmethod
    def get_feature(cls, name: str) -> Callable:
        """
        Get a feature generator by name.
        
        Args:
            name (str): Name of the feature
            
        Returns:
            Callable: Feature generator function
            
        Raises:
            KeyError: If feature does not exist
        """
        if name not in cls._features:
            raise KeyError(f"Feature '{name}' not found")
        return cls._features.get(name)
    
    @classmethod
    def list_features(cls) -> List[str]:
        """
        List all available features.
        
        Returns:
            List[str]: List of feature names
        """
        return list(cls._features.keys())
    
    @classmethod
    def list_categories(cls) -> List[str]:
        """
        List all available feature categories.
        
        Returns:
            List[str]: List of category names
        """
        return list(cls._categories.keys())
    
    @classmethod
    def get_features_by_category(cls, category: str) -> List[str]:
        """
        Get all feature names in a specific category.
        
        Args:
            category (str): Category name
            
        Returns:
            List[str]: List of feature names in the category (empty list if category doesn't exist)
        """
        return cls._categories.get(category, [])
    
    @classmethod
    def get_feature_metadata(cls, name: str) -> Dict[str, Any]:
        """
        Get metadata for a feature.
        
        Args:
            name (str): Feature name
            
        Returns:
            Dict[str, Any]: Feature metadata
            
        Raises:
            KeyError: If feature does not exist
        """
        if name not in cls._metadata:
            raise KeyError(f"Feature '{name}' not found")
        return cls._metadata.get(name, {})
    
    @classmethod
    def compute_feature(cls, name: str, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Compute a single feature.
        
        Args:
            name (str): Feature name
            data (pd.DataFrame): Data to compute feature from
            **kwargs: Additional parameters to pass to the feature function
            
        Returns:
            pd.Series: Computed feature values
            
        Raises:
            KeyError: If feature does not exist
        """
        if name not in cls._features:
            raise KeyError(f"Feature '{name}' not found")
        return cls._features[name](data, **kwargs)
    
    @classmethod
    def compute_features(cls, feature_list: List[str], data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute multiple features.
        
        Args:
            feature_list (List[str]): List of feature names
            data (pd.DataFrame): Data to compute features from
            
        Returns:
            pd.DataFrame: DataFrame containing computed features
        """
        features = {}
        
        for feature_name in feature_list:
            try:
                feature_values = cls.compute_feature(feature_name, data)
                features[feature_name] = feature_values
            except Exception as e:
                print(f"Error computing feature '{feature_name}': {e}")
                continue
        
        return pd.DataFrame(features, index=data.index)
    
    @classmethod
    def get_feature_function(cls, name: str) -> Callable:
        """
        Get a feature generator function by name.
        This is an alias for get_feature, provided for backward compatibility.
        
        Args:
            name (str): Name of the feature
            
        Returns:
            Callable: Feature generator function
            
        Raises:
            KeyError: If feature does not exist
        """
        return cls.get_feature(name) 