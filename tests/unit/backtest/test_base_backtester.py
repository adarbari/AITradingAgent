"""
Tests for the BaseBacktester abstract class
"""
import pytest
from unittest.mock import MagicMock
from src.backtest.base_backtester import BaseBacktester


class TestBaseBacktester:
    """Test cases for the BaseBacktester abstract class"""
    
    def test_abstract_methods_implementation(self):
        """Test that abstract methods need to be implemented"""
        # Verify that we cannot instantiate the abstract class directly
        with pytest.raises(TypeError):
            BaseBacktester()
            
    def test_inheritance(self):
        """Test that the BaseBacktester can be inherited from and extended"""
        # Create a concrete subclass with all methods implemented
        class CompleteBacktester(BaseBacktester):
            def backtest_model(self, model_path, symbol, test_start, test_end, data_source, env_class):
                return {"result": "mocked"}
                
            def plot_comparison(self, returns_df, benchmark_results, symbol):
                return "/path/to/chart.png"

            def save_results(self, results, file_path):
                return file_path

            def load_results(self, file_path):
                return {"loaded": "results"}
        
        # Create an instance
        backtester = CompleteBacktester()
        
        # Test that the implementation can be used without errors
        result = backtester.backtest_model("model", "AAPL", "2020-01-01", "2020-12-31", "yahoo", MagicMock())
        assert isinstance(result, dict)
        
        chart_path = backtester.plot_comparison({}, {}, "AAPL")
        assert isinstance(chart_path, str)

        # Test save and load results
        save_path = backtester.save_results({}, "test.json")
        assert isinstance(save_path, str)

        loaded_results = backtester.load_results("test.json")
        assert isinstance(loaded_results, dict) 