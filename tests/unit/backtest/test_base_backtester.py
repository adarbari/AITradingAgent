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
            def backtest_model(self, model_name, test_start_date, test_end_date, data_source, 
                               data_fetcher_factory, trading_env_class, **kwargs):
                return {"result": "mocked"}
                
            def get_market_performance(self, symbol, test_start_date, test_end_date, data_source, 
                                       data_fetcher_factory, **kwargs):
                return MagicMock()
                
            def plot_comparison(self, results_dict, market_performance, test_start_date, test_end_date, **kwargs):
                return "/path/to/chart.png"
        
        # Create an instance
        backtester = CompleteBacktester()
        
        # Test that the implementation can be used without errors
        result = backtester.backtest_model("model", "2020-01-01", "2020-12-31", "yahoo", MagicMock(), MagicMock())
        assert isinstance(result, dict)
        
        market_perf = backtester.get_market_performance("AAPL", "2020-01-01", "2020-12-31", "yahoo", MagicMock())
        assert market_perf is not None
        
        chart_path = backtester.plot_comparison({}, MagicMock(), "2020-01-01", "2020-12-31")
        assert isinstance(chart_path, str) 