"""
Tests for the RiskAssessmentAgent class
"""
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.agent.multi_agent.base_agent import AgentInput, AgentOutput
from src.agent.multi_agent.risk_assessment_agent import RiskAssessmentAgent


@pytest.fixture
def mock_data_manager():
    """Create a mock data manager"""
    data_manager = MagicMock()
    
    # Create sample market data with volatility pattern
    dates = pd.date_range(start="2023-01-01", periods=20, freq="D")
    
    # Create a price series with some volatility
    np.random.seed(42)  # For reproducible results
    base_prices = np.linspace(100, 120, 20)  # Uptrend
    volatility = np.random.normal(0, 3, 20)  # Add noise
    prices = base_prices + volatility
    
    # Calculate returns for proper risk metrics
    returns = np.diff(prices) / prices[:-1]
    returns = np.insert(returns, 0, 0)  # Add 0 for the first day
    
    market_data = pd.DataFrame({
        'Open': prices * 0.99,
        'High': prices * 1.02,
        'Low': prices * 0.98,
        'Close': prices,
        'Volume': np.random.randint(1000, 10000, 20),
        'SMA_20': prices - 2,
        'SMA_50': prices - 5,
        'RSI_14': np.clip(50 + volatility * 5, 30, 70),  # RSI fluctuating
        'MACD': np.random.normal(0, 1, 20),
        'MACD_Signal': np.random.normal(0, 1, 20),
        'Upper_Band': prices + 10,
        'Lower_Band': prices - 10,
        'Middle_Band': prices
    }, index=dates)
    
    # Configure the mock to return the sample data
    data_manager.get_market_data.return_value = market_data
    
    return data_manager


@pytest.fixture
def risk_agent(mock_data_manager):
    """Create a risk assessment agent for testing"""
    return RiskAssessmentAgent(data_manager=mock_data_manager, verbose=0)


class TestRiskAssessmentAgent:
    """Test cases for the RiskAssessmentAgent"""
    
    def test_initialization(self, mock_data_manager):
        """Test agent initialization"""
        agent = RiskAssessmentAgent(data_manager=mock_data_manager, verbose=1)
        
        assert agent.name == "Risk Assessment Agent"
        assert "risk" in agent.description.lower()
        assert agent.data_manager == mock_data_manager
        assert agent.verbose == 1
    
    def test_extract_symbol(self, risk_agent):
        """Test symbol extraction from text"""
        # Test various formats of symbol representation
        assert risk_agent._extract_symbol("Assess risk for AAPL stock") == "AAPL"
        assert risk_agent._extract_symbol("What's the risk level of $MSFT?") == "MSFT"
        assert risk_agent._extract_symbol("Evaluate GOOG and AMZN risk levels") in ["GOOG", "AMZN"]
        assert risk_agent._extract_symbol("How risky is the market today?") is None
        assert risk_agent._extract_symbol("Let's look at AMAZON") is None  # Not a valid ticker format
    
    def test_extract_date_range(self, risk_agent):
        """Test date range extraction from text"""
        # Test absolute date range
        date_range = risk_agent._extract_date_range("Assess risk for AAPL from 2023-01-01 to 2023-12-31")
        assert date_range is not None
        assert date_range["start_date"] == "2023-01-01"
        assert date_range["end_date"] == "2023-12-31"
        
        # Test MM/DD/YYYY format
        date_range = risk_agent._extract_date_range("Assess risk for AAPL from 01/01/2023 to 12/31/2023")
        assert date_range is not None
        assert date_range["start_date"] == "2023-01-01"
        assert date_range["end_date"] == "2023-12-31"
        
        # Test relative date range
        date_range = risk_agent._extract_date_range("Assess risk for AAPL for the last 30 days")
        assert date_range is not None
        assert "start_date" in date_range
        assert "end_date" in date_range
        
        # Calculate expected dates
        today = datetime.now()
        expected_end = today.strftime("%Y-%m-%d")
        expected_start = (today - timedelta(days=30)).strftime("%Y-%m-%d")
        
        # Allow for test execution time differences by checking approximately
        actual_end = datetime.strptime(date_range["end_date"], "%Y-%m-%d")
        actual_start = datetime.strptime(date_range["start_date"], "%Y-%m-%d")
        
        assert (today - actual_end).days <= 1  # Allow a day of difference
        assert abs((today - timedelta(days=30) - actual_start).days) <= 1
    
    def test_process_with_valid_data(self, risk_agent, mock_data_manager):
        """Test processing a request with valid data"""
        input_data = AgentInput(request="Assess risk for AAPL stock")
        output = risk_agent.process(input_data)
        
        assert isinstance(output, AgentOutput)
        assert output.response is not None
        assert len(output.response) > 0
        assert output.data is not None
        assert "symbol" in output.data
        assert "risk_score" in output.data
        assert "risk_rating" in output.data
        assert "volatility" in output.data
        assert "market_condition" in output.data
        assert output.confidence is not None
        
        # Verify data manager was called correctly
        mock_data_manager.get_market_data.assert_called_once()
    
    def test_process_with_missing_symbol(self, risk_agent, mock_data_manager):
        """Test processing a request with no symbol"""
        input_data = AgentInput(request="How risky is the market?")
        output = risk_agent.process(input_data)
        
        assert isinstance(output, AgentOutput)
        assert "need a specific stock symbol" in output.response.lower()
        assert output.confidence == 0.0
        
        # Verify data manager was not called
        mock_data_manager.get_market_data.assert_not_called()
    
    def test_process_with_symbol_in_context(self, risk_agent, mock_data_manager):
        """Test processing a request with symbol in context"""
        context = {"symbol": "AAPL", "date_range": {"start_date": "2023-01-01", "end_date": "2023-12-31"}}
        input_data = AgentInput(request="Assess the risk level", context=context)
        output = risk_agent.process(input_data)
        
        assert isinstance(output, AgentOutput)
        assert output.response is not None
        assert len(output.response) > 0
        
        # Verify data manager was called with context parameters
        mock_data_manager.get_market_data.assert_called_once_with(
            symbol="AAPL",
            start_date="2023-01-01",
            end_date="2023-12-31",
            include_indicators=True
        )
    
    def test_process_with_no_data(self, risk_agent, mock_data_manager):
        """Test processing when data manager returns no data"""
        mock_data_manager.get_market_data.return_value = None
        
        input_data = AgentInput(request="Assess risk for AAPL stock")
        output = risk_agent.process(input_data)
        
        assert isinstance(output, AgentOutput)
        assert "couldn't retrieve market data" in output.response.lower()
        assert output.confidence == 0.0
    
    def test_calculate_risk_metrics(self, risk_agent, mock_data_manager):
        """Test risk metric calculation"""
        market_data = mock_data_manager.get_market_data.return_value
        
        # Calculate risk metrics
        risk_data = risk_agent._calculate_risk_metrics(market_data, "AAPL")
        
        # Verify the risk metrics were calculated correctly
        assert "symbol" in risk_data
        assert risk_data["symbol"] == "AAPL"
        assert "volatility" in risk_data
        assert "value_at_risk_95" in risk_data
        assert "max_drawdown" in risk_data
        assert "risk_score" in risk_data
        assert 0 <= risk_data["risk_score"] <= 1
        assert "risk_rating" in risk_data
        assert risk_data["risk_rating"] in ["Low", "Medium", "High"]
        assert "latest_rsi" in risk_data
        assert "trend" in risk_data
        assert "market_condition" in risk_data
    
    def test_assess_market_condition(self, risk_agent, mock_data_manager):
        """Test market condition assessment"""
        market_data = mock_data_manager.get_market_data.return_value
        
        # Assess market condition
        condition = risk_agent._assess_market_condition(market_data)
        
        # Verify the condition assessment
        assert "rsi_condition" in condition
        assert condition["rsi_condition"] in ["oversold", "neutral", "overbought"]
        assert "macd_condition" in condition
        assert condition["macd_condition"] in ["bullish", "neutral", "bearish"]
        assert "bollinger_condition" in condition
        assert condition["bollinger_condition"] in ["oversold", "neutral", "overbought"]
        assert "overall_condition" in condition
        assert condition["overall_condition"] in ["bullish", "neutral", "bearish"]
        assert "bullish_signals" in condition
        assert "bearish_signals" in condition
    
    def test_generate_risk_assessment(self, risk_agent):
        """Test risk assessment text generation"""
        risk_data = {
            "symbol": "AAPL",
            "volatility": 0.015,
            "value_at_risk_95": -2.5,
            "max_drawdown": -0.12,
            "risk_score": 0.65,
            "risk_rating": "Medium",
            "current_price": 150.25,
            "trend": "Uptrend",
            "market_condition": {
                "rsi_condition": "neutral",
                "macd_condition": "bullish",
                "overall_condition": "bullish"
            }
        }
        
        # Generate assessment text
        assessment = risk_agent._generate_risk_assessment(risk_data)
        
        # Verify the assessment text
        assert "AAPL" in assessment
        assert "$150.25" in assessment
        assert "Uptrend" in assessment
        assert "Volatility" in assessment
        assert "1.50%" in assessment  # 0.015 * 100
        assert "Value at Risk" in assessment
        assert "Medium" in assessment
        assert "Risk Management Recommendations" in assessment
    
    def test_calculate_position_risk(self, risk_agent):
        """Test position risk calculation"""
        portfolio = {
            "total_value": 100000.0,
            "positions": [
                {
                    "symbol": "AAPL",
                    "quantity": 100,
                    "cost_basis": 140.0
                },
                {
                    "symbol": "MSFT",
                    "quantity": 50,
                    "cost_basis": 250.0
                }
            ]
        }
        
        risk_data = {
            "symbol": "AAPL",
            "current_price": 150.0,
            "risk_rating": "Medium"
        }
        
        # Calculate position risk
        position_risk = risk_agent._calculate_position_risk("AAPL", portfolio, risk_data)
        
        # Verify the position risk text
        assert "Position Risk Assessment" in position_risk
        assert "100 shares" in position_risk
        assert "$15000.00" in position_risk  # 100 * 150.0
        assert "15.00%" in position_risk  # (15000/100000) * 100
        assert "$1000.00" in position_risk  # (150-140) * 100
        assert "7.14%" in position_risk  # ((150/140)-1) * 100
    
    def test_process_with_portfolio_context(self, risk_agent, mock_data_manager):
        """Test processing a request with portfolio data in context"""
        portfolio = {
            "total_value": 100000.0,
            "positions": [
                {
                    "symbol": "AAPL",
                    "quantity": 100,
                    "cost_basis": 140.0
                }
            ]
        }
        
        context = {
            "symbol": "AAPL", 
            "date_range": {"start_date": "2023-01-01", "end_date": "2023-12-31"},
            "portfolio": portfolio
        }
        
        input_data = AgentInput(request="Assess risk for my AAPL position", context=context)
        output = risk_agent.process(input_data)
        
        assert isinstance(output, AgentOutput)
        assert "Position Risk Assessment" in output.response
        assert "Position Size" in output.response 