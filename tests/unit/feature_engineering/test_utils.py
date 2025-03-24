"""
Test utilities for verifying the feature engineering module API.
"""
import os
import sys
import inspect
import pytest

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.feature_engineering import registry, pipeline, cache


class TestFeatureEngineeringAPI:
    """
    Tests to verify that the feature engineering module API is consistent with what the tests expect.
    This helps catch API changes that would break the tests.
    """
    
    def test_registry_api(self):
        """Test that the FeatureRegistry class has the expected methods."""
        # Check essential methods used in tests
        expected_methods = [
            'register',
            'get_feature',
            'list_features',
            'list_categories',
            'get_features_by_category',
            'get_feature_metadata',
            'compute_feature',
            'compute_features'
        ]
        
        # Verify all expected methods exist
        for method_name in expected_methods:
            assert hasattr(registry.FeatureRegistry, method_name), f"FeatureRegistry missing method: {method_name}"
            
        # Check method signatures
        register_sig = inspect.signature(registry.FeatureRegistry.register)
        assert len(register_sig.parameters) >= 3, "register() should have at least 3 parameters"
        
        compute_feature_sig = inspect.signature(registry.FeatureRegistry.compute_feature)
        assert len(compute_feature_sig.parameters) >= 2, "compute_feature() should have at least 2 parameters"
    
    def test_pipeline_api(self):
        """Test that the FeaturePipeline class has the expected methods."""
        # Check essential methods used in tests
        expected_methods = [
            '__init__',
            'process',
            '_generate_features',
            '_normalize_features',
            '_handle_feature_count',
            '_final_cleanup',
            '_validate_features'
        ]
        
        # Verify all expected methods exist
        for method_name in expected_methods:
            assert hasattr(pipeline.FeaturePipeline, method_name), f"FeaturePipeline missing method: {method_name}"
        
        # Check for process_features function
        assert hasattr(pipeline, 'process_features') or hasattr(sys.modules['src.feature_engineering'], 'process_features'), "Missing process_features function"
    
    def test_cache_api(self):
        """Test that the FeatureCache class has the expected methods."""
        # Check essential methods used in tests
        expected_methods = [
            '__init__',
            'get_cache_key',
            'get_cache_path',
            'is_cached',
            'load',
            'save',
            'clear',
            'get_cache_stats'
        ]
        
        # Verify all expected methods exist
        for method_name in expected_methods:
            assert hasattr(cache.FeatureCache, method_name), f"FeatureCache missing method: {method_name}"
        
        # Check method signatures
        init_sig = inspect.signature(cache.FeatureCache.__init__)
        assert 'cache_dir' in init_sig.parameters, "__init__() should have a cache_dir parameter"
        assert 'enable_cache' in init_sig.parameters, "__init__() should have an enable_cache parameter"
        
        get_cache_key_sig = inspect.signature(cache.FeatureCache.get_cache_key)
        assert len(get_cache_key_sig.parameters) >= 2, "get_cache_key() should have at least 2 parameters" 