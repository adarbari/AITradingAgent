"""
Unit tests for the PortfolioManagementAgent class, focusing on portfolio optimization.
"""
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.agent.multi_agent.portfolio_management_agent import PortfolioManagementAgent
from src.agent.multi_agent.portfolio_optimizer import PortfolioOptimizer
from src.agent.multi_agent.base_agent import AgentInput, AgentOutput
from src.data import DataManager

class TestPortfolioManagementAgent(unittest.TestCase):
    """Test cases for the PortfolioManagementAgent class."""
    
    def setUp(self):
        """Set up test environment before each test method."""
        # Create mock DataManager
        self.mock_data_manager = MagicMock(spec=DataManager)
        
        # Setup sample market data responses
        self.setup_mock_market_data()
        
        # Setup sample sentiment data responses
        self.setup_mock_sentiment_data()
        
        # Initialize the agent with the mock data manager
        self.agent = PortfolioManagementAgent(data_manager=self.mock_data_manager, verbose=0)
        
        # Create a list of test symbols
        self.test_symbols = ['AAPL', 'MSFT', 'GOOGL']
        
        # Mock the portfolio optimizer to isolate testing
        self.agent.portfolio_optimizer = MagicMock(spec=PortfolioOptimizer)
        self.setup_mock_optimizer()
    
    def setup_mock_market_data(self):
        """Set up mock market data responses."""
        # Create sample price data
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        
        # Create price data for each test symbol
        self.price_data = {}
        for symbol in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']:
            start_price = np.random.uniform(50, 200)
            prices = start_price + np.cumsum(np.random.normal(0, 1, size=100))
            self.price_data[symbol] = pd.DataFrame({
                'Close': prices,
                'Open': prices * 0.99,
                'High': prices * 1.01,
                'Low': prices * 0.98,
                'Volume': np.random.randint(1000000, 5000000, size=100)
            }, index=dates)
        
        # Configure the mock to return the price data
        def mock_get_market_data(symbol, *args, **kwargs):
            return self.price_data.get(symbol, None)
        
        self.mock_data_manager.get_market_data.side_effect = mock_get_market_data
    
    def setup_mock_sentiment_data(self):
        """Set up mock sentiment data responses."""
        # Create sample sentiment data
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        
        # News sentiment data
        aapl_sentiment = pd.DataFrame({
            'Sentiment_Score': np.random.uniform(-0.5, 0.8, size=30),
            'Article_Count': np.random.randint(10, 100, size=30)
        }, index=dates)
        
        msft_sentiment = pd.DataFrame({
            'Sentiment_Score': np.random.uniform(0.1, 0.9, size=30),  # More positive
            'Article_Count': np.random.randint(10, 80, size=30)
        }, index=dates)
        
        googl_sentiment = pd.DataFrame({
            'Sentiment_Score': np.random.uniform(-0.8, -0.1, size=30),  # More negative
            'Article_Count': np.random.randint(10, 70, size=30)
        }, index=dates)
        
        self.sentiment_data = {
            'AAPL': aapl_sentiment,
            'MSFT': msft_sentiment,
            'GOOGL': googl_sentiment
        }
        
        # Configure the mock to return the sentiment data
        def mock_get_sentiment_data(symbol, *args, **kwargs):
            return self.sentiment_data.get(symbol, None)
        
        self.mock_data_manager.get_sentiment_data.side_effect = mock_get_sentiment_data
        self.mock_data_manager.get_social_sentiment.return_value = None  # No social sentiment data
    
    def setup_mock_optimizer(self):
        """Set up mock portfolio optimizer responses."""
        # Mock optimization results
        mock_optimization_result = {
            'weights': {'AAPL': 0.4, 'MSFT': 0.35, 'GOOGL': 0.25},
            'return': 0.12,
            'volatility': 0.18,
            'sharpe_ratio': 0.56,
            'objective': 'sharpe',
            'risk_free_rate': 0.02
        }
        
        # Mock efficient frontier
        mock_frontier = {
            'returns': np.array([0.08, 0.10, 0.12, 0.14]),
            'volatilities': np.array([0.12, 0.15, 0.18, 0.22])
        }
        
        # Mock risk parity portfolio
        mock_risk_parity = {
            'weights': {'AAPL': 0.3, 'MSFT': 0.4, 'GOOGL': 0.3},
            'return': 0.10,
            'volatility': 0.16,
            'sharpe_ratio': 0.50,
            'objective': 'risk_parity',
            'risk_free_rate': 0.02
        }
        
        # Configure the mocks
        self.agent.portfolio_optimizer.optimize_portfolio.return_value = mock_optimization_result
        self.agent.portfolio_optimizer.generate_efficient_frontier.return_value = mock_frontier
        self.agent.portfolio_optimizer.calculate_risk_parity_portfolio.return_value = mock_risk_parity
    
    def test_optimize_multi_asset_portfolio(self):
        """Test the multi-asset portfolio optimization method."""
        # Test parameters
        symbols = self.test_symbols
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")
        risk_tolerance = "moderate"
        
        # Call the method
        result = self.agent.optimize_multi_asset_portfolio(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            risk_tolerance=risk_tolerance,
            optimization_objective="sharpe",
            include_sentiment=True
        )
        
        # Assertions
        self.assertIn('weights', result)
        self.assertIn('return', result)
        self.assertIn('volatility', result)
        self.assertIn('sharpe_ratio', result)
        self.assertIn('formatted_weights', result)
        
        # Verify the optimizer was called with correct parameters
        self.agent.portfolio_optimizer.optimize_portfolio.assert_called_once()
        call_args = self.agent.portfolio_optimizer.optimize_portfolio.call_args[1]
        self.assertEqual(call_args['objective'], 'sharpe')
        self.assertIsInstance(call_args['constraints'], dict)
    
    def test_process_portfolio_optimization_request(self):
        """Test processing of portfolio optimization request."""
        # Create agent input
        request = "Optimize a portfolio of AAPL, MSFT, and GOOGL with moderate risk"
        agent_input = AgentInput(
            request=request,
            context={
                "symbols": self.test_symbols,
                "date_range": {
                    "start_date": "2022-01-01",
                    "end_date": "2023-01-01"
                },
                "risk_tolerance": "moderate"
            }
        )
        
        # Call the method
        result = self.agent.process_portfolio_optimization_request(agent_input)
        
        # Assertions
        self.assertIsInstance(result, AgentOutput)
        self.assertIsNotNone(result.response)
        self.assertIsNotNone(result.data)
        self.assertEqual(result.confidence, 0.9)
        
        # Check that the response contains key metrics
        self.assertIn("Expected Annual Return", result.response)
        self.assertIn("Expected Annual Volatility", result.response)
        self.assertIn("Sharpe Ratio", result.response)
    
    def test_sentiment_integration(self):
        """Test integration of sentiment data into portfolio optimization."""
        # Test with explicit sentiment request
        request = "Optimize a portfolio of AAPL, MSFT, and GOOGL using sentiment data"
        agent_input = AgentInput(
            request=request,
            context={
                "symbols": self.test_symbols
            }
        )
        
        # Patch the _get_sentiment_scores method to track calls
        with patch.object(self.agent, '_get_sentiment_scores', return_value={
            'AAPL': 0.2,
            'MSFT': 0.5,
            'GOOGL': -0.3
        }) as mock_get_sentiment:
            result = self.agent.process_portfolio_optimization_request(agent_input)
            
            # Verify sentiment data was requested
            mock_get_sentiment.assert_called_once()
            
            # Verify sentiment data was passed to optimizer
            optimizer_call_args = self.agent.portfolio_optimizer.optimize_portfolio.call_args[1]
            self.assertIn('sentiment_scores', optimizer_call_args)
            
            # Check that the response mentions sentiment
            self.assertIn("Sentiment Analysis Integration", result.response)
    
    def test_extract_symbols_from_request(self):
        """Test extraction of symbols from request text."""
        # Test with symbols in text
        request = "Can you optimize a portfolio with AAPL, MSFT, and GOOGL?"
        agent_input = AgentInput(request=request)
        
        symbols = self.agent._extract_symbols_from_request(agent_input)
        
        # Verify symbols were extracted correctly
        self.assertEqual(len(symbols), 3)
        self.assertIn('AAPL', symbols)
        self.assertIn('MSFT', symbols)
        self.assertIn('GOOGL', symbols)
        
        # Test with symbols in context
        context_symbols = ['TSLA', 'AMZN', 'META']
        agent_input = AgentInput(
            request="Optimize my portfolio please",
            context={"symbols": context_symbols}
        )
        
        symbols = self.agent._extract_symbols_from_request(agent_input)
        
        # Verify symbols from context were used
        self.assertEqual(symbols, context_symbols)
    
    def test_detect_optimization_request(self):
        """Test detection of optimization requests in process method."""
        # Test specific optimization request patterns that should be detected
        test_cases = [
            "Optimize portfolio for AAPL, MSFT, GOOGL",  # Has "optimize portfolio"
            "Run portfolio optimization for my stocks",  # Has "portfolio optimization"
            "Show me the efficient frontier for my investments",  # Has "efficient frontier"
            "Apply modern portfolio theory to my assets"  # Has "modern portfolio theory"
        ]
        
        for request in test_cases:
            # Create a new agent for each test to avoid side effects
            test_agent = PortfolioManagementAgent(data_manager=self.mock_data_manager, verbose=0)
            
            # Create a mock for the process_portfolio_optimization_request method
            mock_output = AgentOutput(response="Mocked optimization response", confidence=0.9)
            
            # Install the mock
            original_method = test_agent.process_portfolio_optimization_request
            test_agent.process_portfolio_optimization_request = MagicMock(return_value=mock_output)
            
            try:
                # Create input with the test request
                agent_input = AgentInput(
                    request=request,
                    context={"symbols": self.test_symbols}
                )
                
                # Call the process method
                result = test_agent.process(agent_input)
                
                # Verify the mock was called and the result is our mock output
                test_agent.process_portfolio_optimization_request.assert_called_once()
                self.assertEqual(result, mock_output)
                
            finally:
                # Restore the original method to avoid side effects
                test_agent.process_portfolio_optimization_request = original_method
    
    def test_different_risk_tolerances(self):
        """Test optimization with different risk tolerance levels."""
        risk_levels = ["conservative", "moderate", "aggressive"]
        
        for risk in risk_levels:
            # Create agent input with specific risk tolerance
            agent_input = AgentInput(
                request=f"Optimize portfolio with {risk} risk tolerance",
                context={
                    "symbols": self.test_symbols,
                    "risk_tolerance": risk
                }
            )
            
            # Call the method
            result = self.agent.process_portfolio_optimization_request(agent_input)
            
            # Check that the risk tolerance is included in the response
            self.assertIn(f"{risk.capitalize()} Risk Profile", result.response)
            
            # For conservative, check for specific text about risk
            if risk == "conservative":
                self.assertIn("capital preservation", result.response.lower())
            # For aggressive, check for specific text about risk
            elif risk == "aggressive":
                self.assertIn("maximizing returns", result.response.lower())

if __name__ == '__main__':
    unittest.main() 