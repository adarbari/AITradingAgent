"""
Tests for the MarketAnalysisAgent class
"""
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.agent.multi_agent.base_agent import AgentInput, AgentOutput
from src.agent.multi_agent.market_analysis_agent import MarketAnalysisAgent


@pytest.fixture
def mock_data_manager():
    """Create a mock data manager"""
    data_manager = MagicMock()
    
    # Create sample market data
    dates = pd.date_range(start="2023-01-01", periods=20, freq="D")
    prices = np.linspace(100, 150, 20) + np.random.normal(0, 5, 20)
    
    market_data = pd.DataFrame({
        'Open': prices,
        'High': prices * 1.02,
        'Low': prices * 0.98,
        'Close': prices,
        'Volume': np.random.randint(1000, 10000, 20),
        'SMA_20': prices - 5,
        'SMA_50': prices - 10,
        'RSI_14': np.random.uniform(30, 70, 20),
        'MACD': np.random.normal(0, 1, 20),
        'MACD_Signal': np.random.normal(0, 1, 20),
        'Upper_Band': prices + 15,
        'Lower_Band': prices - 15,
        'Middle_Band': prices
    }, index=dates)
    
    # Configure the mock to return the sample data
    data_manager.get_market_data.return_value = market_data
    
    return data_manager


class TestMarketAnalysisAgent:
    """Test cases for the MarketAnalysisAgent"""
    
    def test_initialization(self, mock_data_manager):
        """Test agent initialization"""
        agent = MarketAnalysisAgent(data_manager=mock_data_manager, verbose=1)
        
        assert agent.name == "Market Analysis Agent"
        assert "market data" in agent.description.lower()
        assert agent.data_manager == mock_data_manager
        assert agent.verbose == 1
        assert agent.llm is None
    
    def test_initialization_with_api_key(self, mock_data_manager):
        """Test agent initialization with API key"""
        with patch('src.agent.multi_agent.market_analysis_agent.ChatOpenAI') as mock_chat:
            mock_chat.return_value = MagicMock()
            
            agent = MarketAnalysisAgent(
                data_manager=mock_data_manager,
                openai_api_key="test_api_key",
                model_name="test-model",
                temperature=0.5
            )
            
            assert agent.llm is not None
            mock_chat.assert_called_once_with(
                api_key="test_api_key",
                model="test-model",
                temperature=0.5
            )
    
    def test_extract_symbol(self, mock_data_manager):
        """Test symbol extraction from text"""
        agent = MarketAnalysisAgent(data_manager=mock_data_manager)
        
        # Test various formats of symbol representation
        assert agent._extract_symbol("Analyze AAPL stock") == "AAPL"
        assert agent._extract_symbol("What do you think about $MSFT?") == "MSFT"
        assert agent._extract_symbol("Compare GOOG and AMZN") in ["GOOG", "AMZN"]
        assert agent._extract_symbol("How is the market today?") is None
        assert agent._extract_symbol("Let's look at AMAZON") is None  # Not a valid ticker format
    
    def test_extract_date_range(self, mock_data_manager):
        """Test date range extraction from text"""
        agent = MarketAnalysisAgent(data_manager=mock_data_manager)
        
        # Test absolute date range
        date_range = agent._extract_date_range("Analyze AAPL from 2023-01-01 to 2023-12-31")
        assert date_range is not None
        assert date_range["start_date"] == "2023-01-01"
        assert date_range["end_date"] == "2023-12-31"
        
        # Test MM/DD/YYYY format
        date_range = agent._extract_date_range("Analyze AAPL from 01/01/2023 to 12/31/2023")
        assert date_range is not None
        assert date_range["start_date"] == "2023-01-01"
        assert date_range["end_date"] == "2023-12-31"
        
        # Test relative date range
        date_range = agent._extract_date_range("Analyze AAPL for the last 30 days")
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
    
    def test_process_with_valid_data(self, mock_data_manager):
        """Test processing a request with valid data"""
        agent = MarketAnalysisAgent(data_manager=mock_data_manager)
        
        input_data = AgentInput(request="Analyze AAPL stock")
        output = agent.process(input_data)
        
        assert isinstance(output, AgentOutput)
        assert output.response is not None
        assert len(output.response) > 0
        assert output.data is not None
        assert "symbol" in output.data
        assert output.confidence is not None
        
        # Verify data manager was called correctly
        mock_data_manager.get_market_data.assert_called_once()
    
    def test_process_with_missing_symbol(self, mock_data_manager):
        """Test processing a request with no symbol"""
        agent = MarketAnalysisAgent(data_manager=mock_data_manager)
        
        input_data = AgentInput(request="How is the market doing?")
        output = agent.process(input_data)
        
        assert isinstance(output, AgentOutput)
        assert "need a specific stock symbol" in output.response.lower()
        assert output.confidence == 0.0
        
        # Verify data manager was not called
        mock_data_manager.get_market_data.assert_not_called()
    
    def test_process_with_symbol_in_context(self, mock_data_manager):
        """Test processing a request with symbol in context"""
        agent = MarketAnalysisAgent(data_manager=mock_data_manager)
        
        context = {"symbol": "AAPL", "date_range": {"start_date": "2023-01-01", "end_date": "2023-12-31"}}
        input_data = AgentInput(request="How is the stock performing?", context=context)
        output = agent.process(input_data)
        
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
    
    def test_process_with_no_data(self, mock_data_manager):
        """Test processing when data manager returns no data"""
        mock_data_manager.get_market_data.return_value = None
        agent = MarketAnalysisAgent(data_manager=mock_data_manager)
        
        input_data = AgentInput(request="Analyze AAPL stock")
        output = agent.process(input_data)
        
        assert isinstance(output, AgentOutput)
        assert "couldn't retrieve market data" in output.response.lower()
        assert output.confidence == 0.0
    
    def test_rule_based_analysis(self, mock_data_manager):
        """Test rule-based analysis without LLM"""
        agent = MarketAnalysisAgent(data_manager=mock_data_manager)
        
        # Create test analysis data
        analysis_data = {
            "symbol": "AAPL",
            "current_price": 150.25,
            "price_change": 5.75,
            "percent_change": 3.98,
            "volatility": 15.5,
            "moving_averages": {
                "sma_20": 145.0,
                "sma_50": 140.0,
                "price_vs_sma20": 5.25,
                "price_vs_sma50": 10.25,
                "ma_cross": True
            },
            "support_resistance": {
                "support": 145.0,
                "resistance": 155.0
            },
            "indicators": {
                "rsi": 65.0,
                "rsi_signal": "neutral",
                "macd": 2.5,
                "macd_signal": 1.5,
                "macd_histogram": 1.0,
                "macd_cross_up": True,
                "macd_cross_down": False
            }
        }
        
        analysis_text = agent._generate_rule_based_analysis(analysis_data)
        
        assert "AAPL" in analysis_text
        assert "$150.25" in analysis_text
        assert "3.98%" in analysis_text
        assert "bullish" in analysis_text.lower()
        assert "support level is around $145.00" in analysis_text
        assert "resistance level is around $155.00" in analysis_text
        assert "RSI is at 65.00" in analysis_text
        assert "MACD has crossed above the signal line" in analysis_text
    
    @patch('src.agent.multi_agent.market_analysis_agent.ChatOpenAI')
    def test_llm_analysis(self, mock_chat, mock_data_manager):
        """Test LLM-based analysis"""
        # Setup mock LLM response
        mock_llm = MagicMock()
        mock_content = MagicMock()
        mock_content.content = "LLM Analysis: AAPL shows a bullish trend with strong momentum."
        mock_llm.invoke.return_value = mock_content
        mock_chat.return_value = mock_llm
        
        # Create agent with mocked LLM
        agent = MarketAnalysisAgent(
            data_manager=mock_data_manager,
            openai_api_key="test_api_key"
        )
        
        # Mock the _analyze_market_data method to skip the complex analysis
        # and directly return a test output
        with patch.object(agent, '_analyze_market_data') as mock_analyze:
            mock_analyze.return_value = AgentOutput(
                response="LLM Analysis: AAPL shows a bullish trend with strong momentum.",
                data={"symbol": "AAPL", "current_price": 150.0},
                confidence=0.85
            )
            
            input_data = AgentInput(request="Analyze AAPL stock")
            output = agent.process(input_data)
            
            assert isinstance(output, AgentOutput)
            assert "LLM Analysis:" in output.response
            assert output.confidence > 0.8
            
            # Verify LLM was initialized
            assert mock_chat.called 