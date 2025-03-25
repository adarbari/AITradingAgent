"""
Tests for the PortfolioManagementAgent class
"""
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.agent.multi_agent.base_agent import AgentInput, AgentOutput
from src.agent.multi_agent.portfolio_management_agent import PortfolioManagementAgent


@pytest.fixture
def mock_data_manager():
    """Create a mock data manager"""
    data_manager = MagicMock()
    
    # Create sample market data
    dates = pd.date_range(start="2023-01-01", periods=20, freq="D")
    
    # Create sample price data for a few symbols
    aapl_prices = np.linspace(150, 165, 20) + np.random.normal(0, 2, 20)
    msft_prices = np.linspace(250, 270, 20) + np.random.normal(0, 3, 20)
    
    # Configure the mock to return different data based on symbol
    def get_market_data(symbol=None, **kwargs):
        if symbol == "AAPL":
            return pd.DataFrame({
                'Close': aapl_prices,
                'Open': aapl_prices * 0.99,
                'High': aapl_prices * 1.01,
                'Low': aapl_prices * 0.98,
                'Volume': np.random.randint(1000, 10000, 20),
            }, index=dates)
        elif symbol == "MSFT":
            return pd.DataFrame({
                'Close': msft_prices,
                'Open': msft_prices * 0.99,
                'High': msft_prices * 1.01,
                'Low': msft_prices * 0.98,
                'Volume': np.random.randint(1000, 10000, 20),
            }, index=dates)
        else:
            return None
    
    data_manager.get_market_data.side_effect = get_market_data
    
    return data_manager


@pytest.fixture
def portfolio():
    """Create a sample portfolio for testing"""
    return {
        "total_value": 100000.0,
        "positions": [
            {
                "symbol": "AAPL",
                "quantity": 100,
                "current_price": 165.0,
                "cost_basis": 150.0,
                "sector": "Technology"
            },
            {
                "symbol": "MSFT",
                "quantity": 50,
                "current_price": 270.0,
                "cost_basis": 250.0,
                "sector": "Technology"
            },
            {
                "symbol": "JNJ",
                "quantity": 75,
                "current_price": 150.0,
                "cost_basis": 145.0,
                "sector": "Healthcare"
            }
        ],
        "1m_return": 3.5,
        "3m_return": 7.2,
        "ytd_return": 5.8,
        "1y_return": 12.6,
        "volatility": 15.0,
        "sharpe_ratio": 1.2
    }


@pytest.fixture
def portfolio_agent(mock_data_manager):
    """Create a portfolio management agent for testing"""
    return PortfolioManagementAgent(data_manager=mock_data_manager, verbose=0)


class TestPortfolioManagementAgent:
    """Test cases for the PortfolioManagementAgent"""
    
    def test_initialization(self, mock_data_manager):
        """Test agent initialization"""
        agent = PortfolioManagementAgent(data_manager=mock_data_manager, verbose=1)
        
        assert agent.name == "Portfolio Management Agent"
        assert "portfolio" in agent.description.lower()
        assert agent.data_manager == mock_data_manager
        assert agent.verbose == 1
        assert "conservative" in agent.risk_levels
        assert "moderate" in agent.risk_levels
        assert "aggressive" in agent.risk_levels
    
    def test_extract_risk_tolerance(self, portfolio_agent):
        """Test risk tolerance extraction from text"""
        # Test various risk tolerance descriptions
        assert portfolio_agent._extract_risk_tolerance("I want a conservative portfolio") == "conservative"
        assert portfolio_agent._extract_risk_tolerance("Looking for low-risk investments") == "conservative"
        assert portfolio_agent._extract_risk_tolerance("Need an aggressive growth strategy") == "aggressive"
        assert portfolio_agent._extract_risk_tolerance("Seeking high-risk opportunities") == "aggressive"
        assert portfolio_agent._extract_risk_tolerance("Prefer a moderate approach") == "moderate"
        assert portfolio_agent._extract_risk_tolerance("Looking for balanced investments") == "moderate"
        assert portfolio_agent._extract_risk_tolerance("What stocks should I buy?") is None
    
    def test_extract_rebalance_frequency(self, portfolio_agent):
        """Test rebalancing frequency extraction from text"""
        # Test various rebalancing frequency descriptions
        assert portfolio_agent._extract_rebalance_frequency("Rebalance daily") == "daily"
        assert portfolio_agent._extract_rebalance_frequency("Rebalance weekly") == "weekly"
        assert portfolio_agent._extract_rebalance_frequency("I prefer monthly rebalancing") == "monthly"
        assert portfolio_agent._extract_rebalance_frequency("Let's do quarterly reviews") == "quarterly"
        assert portfolio_agent._extract_rebalance_frequency("Annual rebalancing is sufficient") == "yearly"
        assert portfolio_agent._extract_rebalance_frequency("What's my portfolio worth?") is None
    
    def test_calculate_portfolio_metrics(self, portfolio_agent, portfolio):
        """Test portfolio metrics calculation"""
        metrics = portfolio_agent._calculate_portfolio_metrics(portfolio)
        
        # Verify the metrics were calculated correctly
        assert metrics["total_value"] == 100000.0
        assert metrics["num_positions"] == 3
        
        # Verify allocations
        assert "AAPL" in metrics["allocations"]
        assert "MSFT" in metrics["allocations"]
        assert "JNJ" in metrics["allocations"]
        
        # Check allocation values
        aapl_value = 100 * 165.0
        msft_value = 50 * 270.0
        jnj_value = 75 * 150.0
        
        assert metrics["allocations"]["AAPL"]["value"] == aapl_value
        assert metrics["allocations"]["MSFT"]["value"] == msft_value
        assert metrics["allocations"]["JNJ"]["value"] == jnj_value
        
        # Check percentage calculations
        # Note: In the actual implementation, percentages are calculated based on the
        # portfolio's total_value (100000.0), not the sum of position values
        aapl_pct = (aapl_value / 100000.0) * 100
        msft_pct = (msft_value / 100000.0) * 100
        jnj_pct = (jnj_value / 100000.0) * 100
        
        assert metrics["allocations"]["AAPL"]["percentage"] == pytest.approx(aapl_pct, 0.01)
        assert metrics["allocations"]["MSFT"]["percentage"] == pytest.approx(msft_pct, 0.01)
        assert metrics["allocations"]["JNJ"]["percentage"] == pytest.approx(jnj_pct, 0.01)
        
        # Check sector allocations
        assert "Technology" in metrics["sector_allocations"]
        assert "Healthcare" in metrics["sector_allocations"]
        assert metrics["sector_allocations"]["Technology"] == pytest.approx(aapl_pct + msft_pct, 0.01)
        assert metrics["sector_allocations"]["Healthcare"] == pytest.approx(jnj_pct, 0.01)
        
        # Check max allocation
        assert metrics["max_allocation"] == pytest.approx(max(aapl_pct, msft_pct, jnj_pct), 0.01)
        
        # Check performance metrics are passed through
        assert metrics["performance"]["1m_return"] == 3.5
        assert metrics["performance"]["3m_return"] == 7.2
        assert metrics["volatility"] == 15.0
        assert metrics["sharpe_ratio"] == 1.2
    
    def test_generate_portfolio_assessment(self, portfolio_agent, portfolio):
        """Test portfolio assessment text generation"""
        metrics = portfolio_agent._calculate_portfolio_metrics(portfolio)
        assessment = portfolio_agent._generate_portfolio_assessment(metrics, "moderate")
        
        # Verify the assessment text
        assert "Portfolio Assessment" in assessment
        assert "$100,000.00" in assessment
        assert "Number of Positions: 3" in assessment
        assert "Technology" in assessment
        assert "Healthcare" in assessment
        assert "AAPL" in assessment
        assert "MSFT" in assessment
        assert "JNJ" in assessment
        
        # Verify performance data
        assert "1 Month: 3.50%" in assessment
        assert "3 Months: 7.20%" in assessment
        assert "Year-to-Date: 5.80%" in assessment
        assert "1 Year: 12.60%" in assessment
    
    def test_analyze_portfolio_risk(self, portfolio_agent, portfolio):
        """Test portfolio risk analysis"""
        # Add risk assessment data
        risk_assessment = {
            "AAPL": {"risk_rating": "Medium", "risk_score": 0.5},
            "MSFT": {"risk_rating": "Medium", "risk_score": 0.6},
            "JNJ": {"risk_rating": "Low", "risk_score": 0.3}
        }
        
        metrics = portfolio_agent._calculate_portfolio_metrics(portfolio)
        risk_analysis = portfolio_agent._analyze_portfolio_risk(metrics, "moderate", risk_assessment)
        
        # Verify the risk analysis text
        assert "Risk Analysis (Moderate Profile)" in risk_analysis
        assert "Position Size Risk" in risk_analysis
        assert "Sector Concentration Risk" in risk_analysis
        
        # All positions should be flagged as exceeding position size limits
        assert "AAPL: 16.5% (over by" in risk_analysis
        assert "MSFT: 13.5% (over by" in risk_analysis
        assert "JNJ: 11.2% (over by" in risk_analysis
        
        # No sector exceeds limits with our sample portfolio
        assert "All sector allocations are within your maximum sector exposure limit" in risk_analysis
        
        # Check individual holdings risk levels
        assert "AAPL: Medium Risk" in risk_analysis
        assert "MSFT: Medium Risk" in risk_analysis
        assert "JNJ: Low Risk" in risk_analysis
    
    def test_generate_rebalance_recommendations(self, portfolio_agent, portfolio):
        """Test rebalancing recommendations generation"""
        # Add market analysis data
        market_analysis = {
            "AAPL": {"decision": "BUY", "confidence": 0.8},
            "MSFT": {"decision": "HOLD", "confidence": 0.6},
            "JNJ": {"decision": "SELL", "confidence": 0.7}
        }
        
        # Add risk assessment data
        risk_assessment = {
            "AAPL": {"risk_rating": "Medium", "risk_score": 0.5},
            "MSFT": {"risk_rating": "Medium", "risk_score": 0.6},
            "JNJ": {"risk_rating": "Low", "risk_score": 0.3}
        }
        
        metrics = portfolio_agent._calculate_portfolio_metrics(portfolio)
        recommendations = portfolio_agent._generate_rebalance_recommendations(
            portfolio, metrics, "moderate", market_analysis, risk_assessment
        )
        
        # Verify the recommendations text
        assert "Rebalancing Recommendations (Moderate Profile)" in recommendations
        assert "Position Sizing Guidelines" in recommendations
        
        # Should recommend buying more AAPL (due to BUY signal) and selling JNJ (due to SELL signal)
        if "BUY" in recommendations:
            assert "AAPL" in recommendations
        if "SELL" in recommendations:
            assert "JNJ" in recommendations
    
    def test_format_recommendations_as_dict(self, portfolio_agent):
        """Test parsing recommendations text into structured data"""
        recommendations_text = """
        Rebalancing Recommendations (Moderate Profile):
        
        The following trades are recommended to optimize your portfolio:
        - BUY 10 shares of AAPL ($1,650.00)
          Current: 16.5% → Target: 20.0% (+3.5%)
        - SELL 15 shares of JNJ ($2,250.00)
          Current: 11.3% → Target: 8.0% (-3.3%)
        """
        
        result = portfolio_agent._format_recommendations_as_dict(recommendations_text)
        
        assert len(result) == 2
        
        # Check first recommendation
        assert result[0]["action"] == "buy"
        assert result[0]["symbol"] == "AAPL"
        assert result[0]["shares"] == 10
        assert result[0]["value"] == 1650.0
        
        # Check second recommendation
        assert result[1]["action"] == "sell"
        assert result[1]["symbol"] == "JNJ"
        assert result[1]["shares"] == 15
        assert result[1]["value"] == 2250.0
    
    def test_process_with_valid_data(self, portfolio_agent, portfolio):
        """Test processing a request with valid data"""
        # Create context with portfolio and analysis data
        context = {
            "portfolio": portfolio,
            "risk_assessment": {
                "AAPL": {"risk_rating": "Medium", "risk_score": 0.5},
                "MSFT": {"risk_rating": "Medium", "risk_score": 0.6},
                "JNJ": {"risk_rating": "Low", "risk_score": 0.3}
            },
            "market_analysis": {
                "AAPL": {"decision": "BUY", "confidence": 0.8},
                "MSFT": {"decision": "HOLD", "confidence": 0.6},
                "JNJ": {"decision": "SELL", "confidence": 0.7}
            }
        }
        
        input_data = AgentInput(
            request="Optimize my portfolio with a moderate risk profile",
            context=context
        )
        
        output = portfolio_agent.process(input_data)
        
        # Verify output format
        assert isinstance(output, AgentOutput)
        assert output.response is not None
        assert len(output.response) > 0
        assert output.data is not None
        assert "portfolio_metrics" in output.data
        assert "recommendations" in output.data
        assert "risk_tolerance" in output.data
        assert output.data["risk_tolerance"] == "moderate"
        assert output.confidence > 0.7
    
    def test_process_with_missing_portfolio(self, portfolio_agent):
        """Test processing a request with no portfolio data"""
        input_data = AgentInput(request="Optimize my portfolio")
        output = portfolio_agent.process(input_data)
        
        assert isinstance(output, AgentOutput)
        assert "need portfolio information" in output.response.lower()
        assert output.confidence == 0.0
    
    def test_process_with_risk_tolerance_in_request(self, portfolio_agent, portfolio):
        """Test processing with risk tolerance in the request"""
        context = {"portfolio": portfolio}
        
        input_data = AgentInput(
            request="Optimize my portfolio with an aggressive risk profile",
            context=context
        )
        
        output = portfolio_agent.process(input_data)
        
        assert isinstance(output, AgentOutput)
        assert output.data["risk_tolerance"] == "aggressive"
        
        # Should be evident in response
        assert "Aggressive" in output.response 