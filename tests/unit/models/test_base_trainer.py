"""
Tests for the BaseModelTrainer abstract class
"""
import pytest
from unittest.mock import MagicMock
from src.models.base_trainer import BaseModelTrainer


class TestBaseModelTrainer:
    """Test cases for the BaseModelTrainer abstract class"""
    
    def test_abstract_methods_implementation(self):
        """Test that abstract methods need to be implemented"""
        # Verify that we cannot instantiate the abstract class directly
        with pytest.raises(TypeError):
            BaseModelTrainer()
            
    def test_inheritance(self):
        """Test that the BaseModelTrainer can be inherited from and extended"""
        # Create a concrete subclass with all methods implemented
        class CompleteTrainer(BaseModelTrainer):
            def train_model(self, symbol, train_start_date, train_end_date, data_source, 
                           data_fetcher_factory, trading_env_class, **kwargs):
                return MagicMock()
                
            def load_model(self, model_name, **kwargs):
                return MagicMock()
                
            def save_model(self, model, model_name, **kwargs):
                return True
        
        # Create an instance
        trainer = CompleteTrainer()
        
        # Test that the implementation can be used without errors
        model = trainer.train_model("AAPL", "2020-01-01", "2020-12-31", "yahoo", MagicMock(), MagicMock())
        assert model is not None
        
        loaded_model = trainer.load_model("model_name")
        assert loaded_model is not None
        
        save_result = trainer.save_model(MagicMock(), "model_name")
        assert save_result is True 