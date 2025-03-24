"""
Unit tests for the feature configuration module.
"""
import pytest
import os
import json
import tempfile
import hashlib
from unittest.mock import patch, MagicMock, mock_open

from src.feature_engineering.config import FeatureConfig, FEATURE_CONFIGS


class TestFeatureConfig:
    """Test cases for the FeatureConfig class."""
    
    def test_current_version(self):
        """Test that the current version is defined."""
        assert hasattr(FeatureConfig, 'CURRENT_VERSION')
        assert isinstance(FeatureConfig.CURRENT_VERSION, str)
        assert len(FeatureConfig.CURRENT_VERSION.split('.')) == 3  # Should be in semantic versioning format
    
    def test_get_version_hash(self):
        """Test generating version hashes for feature configurations."""
        # Test with minimal feature list
        feature_list = ["price_change", "volatility", "volume_change"]
        hash1 = FeatureConfig.get_version_hash(feature_list)
        
        # Should return a string
        assert isinstance(hash1, str)
        assert len(hash1) > 0
        
        # Same inputs should produce same hash
        hash2 = FeatureConfig.get_version_hash(feature_list)
        assert hash1 == hash2
        
        # Different feature lists should produce different hashes
        hash3 = FeatureConfig.get_version_hash(feature_list + ["rsi_14"])
        assert hash1 != hash3
        
        # Order shouldn't matter
        hash4 = FeatureConfig.get_version_hash(["volume_change", "price_change", "volatility"])
        assert hash1 == hash4
        
        # Test with normalization params
        hash5 = FeatureConfig.get_version_hash(
            feature_list, 
            normalization_params={"method": "zscore", "clip": True}
        )
        assert hash1 != hash5
        
        # Ensure hash changes when version changes
        with patch.object(FeatureConfig, 'CURRENT_VERSION', new="2.0.0"):
            hash6 = FeatureConfig.get_version_hash(feature_list)
            assert hash1 != hash6
    
    def test_save_config(self):
        """Test saving feature configurations to file."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tf:
            temp_file = tf.name
        
        try:
            # Save a configuration
            feature_list = ["price_change", "volatility", "volume_change"]
            normalization_params = {"method": "zscore", "clip": True}
            
            # Test the save function
            FeatureConfig.save_config(
                feature_list=feature_list,
                normalization_params=normalization_params,
                filename=temp_file
            )
            
            # Verify the file was created
            assert os.path.exists(temp_file)
            
            # Read the file and check contents
            with open(temp_file, 'r') as f:
                config = json.load(f)
            
            # Check structure
            assert "features" in config
            assert "normalization" in config
            assert "version" in config
            assert "hash" in config
            
            # Check values
            assert set(config["features"]) == set(feature_list)
            assert config["normalization"] == normalization_params
            assert config["version"] == FeatureConfig.CURRENT_VERSION
            
            # Verify hash matches what we'd generate
            expected_hash = FeatureConfig.get_version_hash(feature_list, normalization_params)
            assert config["hash"] == expected_hash
        
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_load_config(self):
        """Test loading feature configurations from file."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tf:
            temp_file = tf.name
        
        try:
            # Create a test configuration
            test_config = {
                "features": ["price_change", "volatility", "volume_change"],
                "normalization": {"method": "zscore", "clip": True},
                "version": FeatureConfig.CURRENT_VERSION,
                "hash": "test_hash"
            }
            
            # Write to file
            with open(temp_file, 'w') as f:
                json.dump(test_config, f)
            
            # Load the configuration
            loaded_config = FeatureConfig.load_config(temp_file)
            
            # Verify loaded config matches original
            assert loaded_config == test_config
        
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_load_config_error(self):
        """Test error handling when loading configurations."""
        # Non-existent file
        with pytest.raises(FileNotFoundError):
            FeatureConfig.load_config("non_existent_file.json")
        
        # Invalid JSON
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tf:
            temp_file = tf.name
            tf.write(b"not valid json")
        
        try:
            with pytest.raises(json.JSONDecodeError):
                FeatureConfig.load_config(temp_file)
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


class TestFeatureConfigs:
    """Test cases for the predefined feature configurations."""
    
    def test_feature_configs_structure(self):
        """Test that FEATURE_CONFIGS has the expected structure."""
        assert isinstance(FEATURE_CONFIGS, dict)
        assert len(FEATURE_CONFIGS) > 0
        
        # Check each config set
        for set_name, feature_list in FEATURE_CONFIGS.items():
            # Should be a string key with a list value
            assert isinstance(set_name, str)
            assert isinstance(feature_list, list)
            assert len(feature_list) > 0
            
            # All items in the list should be strings
            for feature in feature_list:
                assert isinstance(feature, str)
    
    def test_minimal_set(self):
        """Test the minimal feature set."""
        if "minimal" not in FEATURE_CONFIGS:
            pytest.skip("No 'minimal' feature set defined")
        
        minimal_set = FEATURE_CONFIGS["minimal"]
        
        # Should be a small set with core features
        assert isinstance(minimal_set, list)
        assert len(minimal_set) <= 10  # Should be relatively small
        
        # Should include some basic features
        basic_features = ["price_change", "volatility", "high_low_range"]
        for feature in basic_features:
            assert feature in minimal_set
    
    def test_standard_set(self):
        """Test the standard feature set."""
        if "standard" not in FEATURE_CONFIGS:
            pytest.skip("No 'standard' feature set defined")
        
        standard_set = FEATURE_CONFIGS["standard"]
        
        # Should be a comprehensive set
        assert isinstance(standard_set, list)
        assert len(standard_set) >= 10  # Should have a good number of features
        
        # Should include features from different categories
        categories = {
            "trend": ["sma_5", "sma_20", "ema_12"],
            "volatility": ["volatility", "bollinger_bandwidth"],
            "momentum": ["rsi_14", "macd"],
            "volume": ["volume_change"],
            "seasonal": ["day_of_week"]
        }
        
        for category, features in categories.items():
            found = False
            for feature in features:
                if feature in standard_set:
                    found = True
                    break
            assert found, f"Standard set should include at least one feature from {category} category"
    
    def test_advanced_set(self):
        """Test the advanced feature set."""
        if "advanced" not in FEATURE_CONFIGS:
            pytest.skip("No 'advanced' feature set defined")
        
        advanced_set = FEATURE_CONFIGS["advanced"]
        standard_set = FEATURE_CONFIGS.get("standard", [])
        
        # Should be larger than the standard set
        assert isinstance(advanced_set, list)
        assert len(advanced_set) > len(standard_set)
        
        # Should include all standard features
        for feature in standard_set:
            assert feature in advanced_set
        
        # Should include additional advanced features
        advanced_features = [
            "sma_50", "sma_200",  # Longer-term moving averages
            "adx_14", "cci_20"     # More complex indicators
        ]
        
        found_advanced = False
        for feature in advanced_features:
            if feature in advanced_set:
                found_advanced = True
                break
        
        assert found_advanced, "Advanced set should include some advanced features not in standard set" 