"""
Integration tests for the Sentiment Analysis Agent functionality
"""
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.agent.multi_agent import TradingAgentOrchestrator
from src.agent.multi_agent.sentiment_analysis_agent import SentimentAnalysisAgent
from src.agent.multi_agent.base_agent import AgentInput, AgentOutput
from src.data import DataManager


@pytest.fixture
def mock_data_manager():
    """Create a mock data manager for tests"""
    data_manager = MagicMock()
    
    # Create sample date range
    dates = pd.date_range(start="2023-01-01", periods=30, freq="D")
    
    # Mock sentiment data
    def get_sentiment_data(symbol, start_date, end_date):
        if symbol == "AAPL":
            # Create sample sentiment data for AAPL
            return pd.DataFrame({
                'Sentiment_Score': np.linspace(-0.2, 0.7, 30) + np.random.normal(0, 0.1, 30),
                'Article_Count': np.random.randint(1, 20, 30),
                'Volatility': np.random.uniform(0.1, 0.5, 30)
            }, index=dates)
        elif symbol == "META":
            # Create sample sentiment data for META with negative trend
            return pd.DataFrame({
                'Sentiment_Score': np.linspace(0.3, -0.5, 30) + np.random.normal(0, 0.1, 30),
                'Article_Count': np.random.randint(5, 30, 30),
                'Volatility': np.random.uniform(0.2, 0.6, 30)
            }, index=dates)
        elif symbol == "NONEXISTENT":
            return None
        else:
            return None
    
    data_manager.get_sentiment_data.side_effect = get_sentiment_data
    
    # Mock social sentiment data
    def get_social_sentiment(symbol, start_date, end_date):
        if symbol == "AAPL":
            # Create sample social sentiment data for AAPL
            return pd.DataFrame({
                'Sentiment_Score': np.linspace(-0.1, 0.6, 30) + np.random.normal(0, 0.15, 30),
                'Engagement': np.random.randint(100, 1000, 30),
                'Reach': np.random.randint(1000, 10000, 30)
            }, index=dates)
        elif symbol == "META":
            # Create sample social sentiment data for META
            return pd.DataFrame({
                'Sentiment_Score': np.linspace(0.2, -0.3, 30) + np.random.normal(0, 0.15, 30),
                'Engagement': np.random.randint(200, 2000, 30),
                'Reach': np.random.randint(2000, 20000, 30)
            }, index=dates)
        else:
            return None
    
    data_manager.get_social_sentiment.side_effect = get_social_sentiment
    
    # Mock get_market_data method
    def get_market_data(symbol, start_date, end_date, **kwargs):
        if symbol in ["AAPL", "META"]:
            return pd.DataFrame({
                'Open': np.random.uniform(150, 170, 30),
                'High': np.random.uniform(160, 180, 30),
                'Low': np.random.uniform(140, 160, 30),
                'Close': np.random.uniform(145, 175, 30),
                'Volume': np.random.randint(1000000, 5000000, 30)
            }, index=dates)
        return None
    
    data_manager.get_market_data.side_effect = get_market_data
    
    return data_manager


class TestSentimentIntegration:
    """Integration tests for sentiment analysis agent functionality"""
    
    def test_sentiment_agent_standalone(self, mock_data_manager):
        """Test the sentiment analysis agent directly"""
        sentiment_agent = SentimentAnalysisAgent(data_manager=mock_data_manager, verbose=1)
        
        # Test with AAPL
        from src.agent.multi_agent.base_agent import AgentInput
        input_data = AgentInput(
            request="Analyze sentiment for AAPL",
            context={
                "symbol": "AAPL",
                "date_range": {
                    "start_date": "2023-01-01",
                    "end_date": "2023-01-30"
                }
            }
        )
        
        output = sentiment_agent.process(input_data)
        
        # Verify output
        assert output.response is not None
        assert len(output.response) > 0
        assert output.data is not None
        assert output.data.get("symbol") == "AAPL"
        assert "sentiment_score" in output.data
        assert "sentiment_rating" in output.data
        assert "trading_signal" in output.data
        assert output.confidence > 0.5
        
        # Test with META (negative sentiment trend)
        input_data = AgentInput(
            request="Analyze sentiment for META",
            context={
                "symbol": "META",
                "date_range": {
                    "start_date": "2023-01-01",
                    "end_date": "2023-01-30"
                }
            }
        )
        
        output = sentiment_agent.process(input_data)
        
        # Verify output has different sentiment
        assert output.data.get("symbol") == "META"
        assert "sentiment_score" in output.data
        assert output.data.get("sentiment_trend", 0) < 0  # Should have negative trend
    
    @patch('src.agent.multi_agent.market_analysis_agent.ChatOpenAI')
    @patch('src.agent.multi_agent.orchestrator.StateGraph')
    def test_sentiment_integration(self, mock_state_graph, mock_chat, mock_data_manager):
        """Test integration of sentiment agent with orchestrator"""
        # Configure mock graph
        mock_graph = MagicMock()
        mock_state_graph.return_value = mock_graph
        mock_graph.compile.return_value = mock_graph
        
        # Configure result from invoke
        mock_result = MagicMock()
        mock_result.symbol = "AAPL"
        mock_result.start_date = "2023-01-01"
        mock_result.end_date = "2023-01-30"
        mock_result.agent_outputs = {
            "market_analysis": {
                "response": "Analysis of AAPL shows a bullish trend.",
                "data": {"current_price": 165.0, "trend": "bullish"},
                "confidence": 0.8
            },
            "sentiment_analysis": {
                "response": "Sentiment analysis for AAPL is positive.",
                "data": {
                    "sentiment_score": 0.65,
                    "sentiment_rating": "positive",
                    "sentiment_trend": 0.12,
                    "trading_signal": {"action": "buy", "impact": "medium"}
                },
                "confidence": 0.75
            },
            "risk_assessment": {
                "response": "Risk assessment for AAPL shows medium risk.",
                "data": {"risk_score": 0.5, "risk_rating": "Medium"},
                "confidence": 0.7
            }
        }
        mock_result.decision = "BUY"
        mock_result.confidence = 0.75
        mock_result.explanation = "Bullish signals with positive sentiment and acceptable risk"
        mock_result.recommended_actions = [
            {
                "action": "buy",
                "symbol": "AAPL",
                "position_size": "moderate"
            }
        ]
        mock_result.analysis_data = {"current_price": 165.0, "trend": "bullish"}
        mock_result.sentiment_data = {
            "sentiment_score": 0.65,
            "sentiment_rating": "positive", 
            "sentiment_trend": 0.12
        }
        mock_result.risk_assessment = {"risk_score": 0.5, "risk_rating": "Medium"}
        
        mock_graph.invoke.return_value = mock_result
        
        # Configure chat mock
        mock_chat_instance = MagicMock()
        mock_chat.return_value = mock_chat_instance
        mock_chat_instance.invoke.return_value.content = "This is a sample analysis."
        
        # Initialize orchestrator with the mocked components
        orchestrator = TradingAgentOrchestrator(
            data_manager=mock_data_manager,
            verbose=1
        )
        orchestrator.workflow = mock_graph
        
        # Process a request that includes sentiment analysis
        result = orchestrator.process_request(
            request="Analyze AAPL including sentiment analysis",
            symbol="AAPL",
            start_date="2023-01-01",
            end_date="2023-01-30"
        )
        
        # Verify the result structure
        assert "request" in result
        assert "symbol" in result
        assert "decision" in result
        assert "confidence" in result
        assert "explanation" in result
        assert "analysis" in result
        assert "sentiment" in result
        assert "risk_assessment" in result
        
        # Verify that sentiment data is included
        assert "sentiment_data" in result
        assert result["sentiment_data"]["sentiment_score"] == 0.65
        assert result["sentiment_data"]["sentiment_rating"] == "positive"
    
    def test_extract_symbol_functionality(self, mock_data_manager):
        """Test the symbol extraction functionality"""
        sentiment_agent = SentimentAnalysisAgent(data_manager=mock_data_manager, verbose=0)
        
        # Test with explicit symbol
        symbol = sentiment_agent._extract_symbol("Analyze sentiment for AAPL stock")
        assert symbol == "AAPL"
        
        # Test with multiple symbols (should return first one)
        symbol = sentiment_agent._extract_symbol("Compare AAPL and MSFT sentiment")
        assert symbol == "AAPL"
        
        # Test with common words that could be mistaken for tickers
        symbol = sentiment_agent._extract_symbol("The CEO announced AI initiatives")
        assert symbol is None
    
    def test_missing_data_handling(self, mock_data_manager):
        """Test how the agent handles missing data"""
        # Configure mock to return None for sentiment
        mock_data_manager.get_sentiment_data.return_value = None
        mock_data_manager.get_social_sentiment.return_value = None
        
        sentiment_agent = SentimentAnalysisAgent(data_manager=mock_data_manager, verbose=0)
        
        input_data = AgentInput(
            request="Analyze sentiment for NONEXISTENT",
            context={
                "symbol": "NONEXISTENT",
                "date_range": {
                    "start_date": "2023-01-01",
                    "end_date": "2023-01-30"
                }
            }
        )
        
        output = sentiment_agent.process(input_data)
        
        # Should return an error response with low confidence
        assert "No sentiment data available" in output.response
        assert output.confidence == 0.0 