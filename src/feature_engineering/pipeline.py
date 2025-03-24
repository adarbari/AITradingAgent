"""
Feature Pipeline Module

Implements a pipeline architecture for feature engineering.
The pipeline provides consistent normalization, scaling, and preprocessing.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union
import warnings

from .registry import FeatureRegistry


class FeaturePipeline:
    """
    Pipeline for feature engineering on financial data.
    
    This class handles:
    1. Feature generation using the FeatureRegistry
    2. Normalization and scaling of features
    3. Ensuring feature count consistency
    4. Final cleanup and validation
    """
    
    def __init__(self, feature_list: Optional[List[str]] = None, feature_count: int = 21, 
                 verbose: bool = False, normalize: bool = True, 
                 normalization_method: str = "zscore"):
        """
        Initialize the feature pipeline.
        
        Args:
            feature_list (List[str], optional): List of feature names to include
            feature_count (int): Expected number of features
            verbose (bool): Whether to print information about the transformation
            normalize (bool): Whether to normalize features
            normalization_method (str): Method for normalization ('zscore', 'minmax')
        """
        from .config import FEATURE_CONFIGS
        
        # Default to standard feature set if none provided
        self.feature_list = feature_list or FEATURE_CONFIGS.get("standard", [])
        self.feature_count = feature_count
        self.verbose = verbose
        self.normalize = normalize
        self.normalization_method = normalization_method
        
        # Validate features
        self._validate_features()
    
    def _validate_features(self) -> None:
        """
        Validate that all requested features are registered.
        
        Raises:
            ValueError: If any feature is not registered
        """
        registered_features = FeatureRegistry.list_features()
        
        for feature in self.feature_list:
            if feature not in registered_features:
                warnings.warn(f"Feature '{feature}' is not registered and will be skipped.")
    
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process the data through the pipeline.
        
        Args:
            data (pd.DataFrame): Raw price data
            
        Returns:
            pd.DataFrame: Processed features
        """
        if self.verbose:
            print(f"Starting feature engineering pipeline with {len(self.feature_list)} features")
        
        # Generate features
        features = self._generate_features(data)
        
        # Normalize if requested
        if self.normalize:
            features = self._normalize_features(features)
        
        # Handle feature count
        features = self._handle_feature_count(features)
        
        # Final checks and cleanup
        features = self._final_cleanup(features)
        
        if self.verbose:
            print(f"Completed feature engineering: {len(features)} samples with {len(features.columns)} features")
        
        return features
    
    def _generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all requested features.
        
        Args:
            data (pd.DataFrame): Raw price data
            
        Returns:
            pd.DataFrame: Generated features
        """
        if self.verbose:
            print(f"Generating {len(self.feature_list)} features...")
        
        # Use the registry to compute all requested features
        features = FeatureRegistry.compute_features(self.feature_list, data)
        
        if self.verbose:
            print(f"Generated {len(features.columns)} features")
        
        return features
    
    def _normalize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize features using the specified method.
        
        Args:
            features (pd.DataFrame): Feature data
            
        Returns:
            pd.DataFrame: Normalized features
        """
        if self.verbose:
            print(f"Normalizing features using {self.normalization_method} method")
        
        normalized_features = features.copy()
        
        # Apply chosen normalization method
        if self.normalization_method == "zscore":
            # Z-score normalization (mean=0, std=1)
            for col in normalized_features.columns:
                if normalized_features[col].std() > 0:
                    normalized_features[col] = (normalized_features[col] - normalized_features[col].mean()) / normalized_features[col].std()
                else:
                    normalized_features[col] = 0  # If std is 0, set all values to 0
        
        elif self.normalization_method == "minmax":
            # Min-max normalization (range 0 to 1)
            for col in normalized_features.columns:
                min_val = normalized_features[col].min()
                max_val = normalized_features[col].max()
                if max_val > min_val:
                    normalized_features[col] = (normalized_features[col] - min_val) / (max_val - min_val)
                else:
                    normalized_features[col] = 0  # If min==max, set all values to 0
        
        elif self.normalization_method == "robust":
            # Robust normalization using median and interquartile range
            for col in normalized_features.columns:
                median = normalized_features[col].median()
                q1 = normalized_features[col].quantile(0.25)
                q3 = normalized_features[col].quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    normalized_features[col] = (normalized_features[col] - median) / iqr
                else:
                    normalized_features[col] = 0  # If IQR is 0, set all values to 0
        
        else:
            warnings.warn(f"Unknown normalization method: {self.normalization_method}. Features not normalized.")
        
        return normalized_features
    
    def _handle_feature_count(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure we have the expected number of features.
        
        Args:
            features (pd.DataFrame): Feature data
            
        Returns:
            pd.DataFrame: Features with adjusted column count
        """
        # Ensure we have the expected number of features
        if len(features.columns) < self.feature_count:
            if self.verbose:
                print(f"Warning: Expected {self.feature_count} features but only {len(features.columns)} are available.")
                print("Adding dummy features to match the expected count...")
            
            # Add missing features with zeros
            for i in range(len(features.columns), self.feature_count):
                feature_name = f"dummy_feature_{i}"
                features[feature_name] = 0.0
        
        elif len(features.columns) > self.feature_count:
            if self.verbose:
                print(f"Warning: More features than expected ({len(features.columns)} vs {self.feature_count}).")
                print(f"Keeping only the first {self.feature_count} features...")
            features = features.iloc[:, :self.feature_count]
        
        return features
    
    def _final_cleanup(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Final checks and cleanup of features.
        
        Args:
            features (pd.DataFrame): Feature data
            
        Returns:
            pd.DataFrame: Clean feature data
        """
        # Check for and replace NaN or infinite values
        if features.isnull().any().any() or np.isinf(features.values).any():
            if self.verbose:
                print("Warning: NaN or infinite values detected. Replacing with zeros.")
            features = features.replace([np.inf, -np.inf, np.nan], 0)
        
        return features 