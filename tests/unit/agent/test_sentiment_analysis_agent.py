"""
Unit tests for the Sentiment Analysis Agent
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

from src.agent.multi_agent.sentiment_analysis_agent import SentimentAnalysisAgent
from src.agent.multi_agent.base_agent import AgentInput, AgentOutput


@pytest.fixture
def mock_data_manager():
    """Create a mock data manager for tests"""
    data_manager = MagicMock()
    
    # Create sample date range
    dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
    
    # Mock news sentiment data
    news_data = pd.DataFrame({
        'Sentiment_Score': [0.2, 0.3, 0.1, -0.1, -0.2, 0.4, 0.5, 0.3, 0.2, 0.1],
        'Article_Count': [10, 12, 8, 15, 20, 18, 22, 14, 12, 10],
        'Volatility': [0.2, 0.25, 0.3, 0.28, 0.35, 0.3, 0.22, 0.21, 0.2, 0.18]
    }, index=dates)
    
    # Mock social media sentiment data
    social_data = pd.DataFrame({
        'Sentiment_Score': [0.1, 0.2, 0.3, 0.1, -0.3, -0.1, 0.4, 0.3, 0.2, 0.0],
        'Engagement': [500, 700, 800, 600, 1200, 900, 1000, 800, 600, 500],
        'Reach': [5000, 7000, 8000, 6000, 12000, 9000, 10000, 8000, 6000, 5000]
    }, index=dates)
    
    # Configure mocks to return different data based on symbol
    def get_sentiment_data(symbol, start_date, end_date):
        if symbol == "AAPL":
            return news_data
        elif symbol == "NEGATIVE":
            # Return negative sentiment data
            negative_data = news_data.copy()
            negative_data['Sentiment_Score'] = -negative_data['Sentiment_Score']
            return negative_data
        elif symbol == "NODATA":
            return None
        else:
            return news_data.copy() * 0.5  # Return different data for other symbols
    
    def get_social_sentiment(symbol, start_date, end_date):
        if symbol == "AAPL":
            return social_data
        elif symbol == "NEGATIVE":
            # Return negative sentiment data
            negative_data = social_data.copy()
            negative_data['Sentiment_Score'] = -negative_data['Sentiment_Score']
            return negative_data
        elif symbol == "NODATA":
            return None
        else:
            return social_data.copy() * 0.7  # Return different data for other symbols
    
    data_manager.get_sentiment_data.side_effect = get_sentiment_data
    data_manager.get_social_sentiment.side_effect = get_social_sentiment
    
    return data_manager


class TestSentimentAnalysisAgent:
    """Unit tests for the SentimentAnalysisAgent class"""
    
    def test_initialization(self, mock_data_manager):
        """Test the initialization of the agent"""
        agent = SentimentAnalysisAgent(data_manager=mock_data_manager, verbose=1)
        assert agent.name == "sentiment_analysis"
        assert agent.data_manager is mock_data_manager
        assert agent.verbose == 1
    
    def test_extract_symbol(self, mock_data_manager):
        """Test the symbol extraction method"""
        agent = SentimentAnalysisAgent(data_manager=mock_data_manager, verbose=0)
        
        # Test with explicit symbol
        assert agent._extract_symbol("Analyze sentiment for AAPL") == "AAPL"
        assert agent._extract_symbol("What's the sentiment for MSFT?") == "MSFT"
        assert agent._extract_symbol("AAPL sentiment analysis needed") == "AAPL"
        
        # Test with ticker in company name
        assert agent._extract_symbol("Analyze Apple Inc. (AAPL)") == "AAPL"
        
        # Test with multiple symbols
        assert agent._extract_symbol("Compare AAPL and MSFT") == "AAPL"
        
        # Test with no valid symbol
        assert agent._extract_symbol("What's the market sentiment today?") is None
        assert agent._extract_symbol("Analyze the top tech companies") is None
    
    def test_analyze_news_sentiment(self, mock_data_manager):
        """Test the news sentiment analysis method"""
        agent = SentimentAnalysisAgent(data_manager=mock_data_manager, verbose=0)
        
        # Test with valid data
        news_result = agent._analyze_news_sentiment(
            "AAPL", 
            "2023-01-01", 
            "2023-01-10"
        )
        
        assert news_result is not None
        assert "score" in news_result
        assert "trend" in news_result
        assert "rating" in news_result
        assert "article_count" in news_result
        
        # Calculated score should be average of Sentiment_Score column
        expected_score = mock_data_manager.get_sentiment_data("AAPL", "", "")["Sentiment_Score"].mean()
        assert abs(news_result["score"] - expected_score) < 0.01
        
        # Test with negative sentiment
        negative_result = agent._analyze_news_sentiment(
            "NEGATIVE", 
            "2023-01-01", 
            "2023-01-10"
        )
        
        assert negative_result["rating"] in ["negative", "very negative"]
        assert negative_result["score"] < 0
        
        # Test with no data
        no_data_result = agent._analyze_news_sentiment(
            "NODATA", 
            "2023-01-01", 
            "2023-01-10"
        )
        
        assert no_data_result is None
    
    def test_analyze_social_sentiment(self, mock_data_manager):
        """Test the social sentiment analysis method"""
        agent = SentimentAnalysisAgent(data_manager=mock_data_manager, verbose=0)
        
        # Test with valid data
        social_result = agent._analyze_social_sentiment(
            "AAPL", 
            "2023-01-01", 
            "2023-01-10"
        )
        
        assert social_result is not None
        assert "score" in social_result
        assert "trend" in social_result
        assert "rating" in social_result
        assert "engagement" in social_result
        
        # Calculated score should be average of Sentiment_Score column
        expected_score = mock_data_manager.get_social_sentiment("AAPL", "", "")["Sentiment_Score"].mean()
        assert abs(social_result["score"] - expected_score) < 0.01
        
        # Test with negative sentiment
        negative_result = agent._analyze_social_sentiment(
            "NEGATIVE", 
            "2023-01-01", 
            "2023-01-10"
        )
        
        assert negative_result["rating"] in ["negative", "very negative"]
        assert negative_result["score"] < 0
        
        # Test with no data
        no_data_result = agent._analyze_social_sentiment(
            "NODATA", 
            "2023-01-01", 
            "2023-01-10"
        )
        
        assert no_data_result is None
    
    def test_combine_sentiment_results(self, mock_data_manager):
        """Test the sentiment combination method"""
        agent = SentimentAnalysisAgent(data_manager=mock_data_manager, verbose=0)
        
        # Test with both news and social data
        news_data = {
            "score": 0.3,
            "rating": "positive",
            "trend": 0.05,
            "article_count": 15
        }
        
        social_data = {
            "score": 0.2,
            "rating": "positive",
            "trend": 0.03,
            "engagement": 800
        }
        
        combined = agent._combine_sentiment_results(news_data, social_data, "AAPL")
        
        assert combined is not None
        assert "sentiment_score" in combined
        assert "sentiment_rating" in combined
        assert "sentiment_trend" in combined
        assert "trading_signal" in combined
        assert "source_data" in combined
        
        # The combined score should be weighted average of news and social scores
        expected_score = (news_data["score"] * 0.6) + (social_data["score"] * 0.4)
        assert abs(combined["sentiment_score"] - expected_score) < 0.01
        
        # Test with only news data
        news_only = agent._combine_sentiment_results(news_data, None, "AAPL")
        
        assert news_only is not None
        assert news_only["sentiment_score"] == news_data["score"]
        assert news_only["source_data"]["has_news_data"] is True
        assert news_only["source_data"]["has_social_data"] is False
        
        # Test with only social data
        social_only = agent._combine_sentiment_results(None, social_data, "AAPL")
        
        assert social_only is not None
        assert social_only["sentiment_score"] == social_data["score"]
        assert social_only["source_data"]["has_news_data"] is False
        assert social_only["source_data"]["has_social_data"] is True
        
        # Test with no data
        no_data = agent._combine_sentiment_results(None, None, "AAPL")
        
        assert no_data is None
    
    def test_generate_trading_signal(self, mock_data_manager):
        """Test the trading signal generation"""
        agent = SentimentAnalysisAgent(data_manager=mock_data_manager, verbose=0)
        
        # Test with strong positive sentiment
        strong_positive = agent._generate_trading_signal(0.7, 0.1, "AAPL")
        assert strong_positive["action"] == "buy"
        assert strong_positive["impact"] in ["high", "medium"]
        
        # Test with moderate positive sentiment
        moderate_positive = agent._generate_trading_signal(0.4, 0.05, "AAPL")
        assert moderate_positive["action"] == "buy"
        assert moderate_positive["impact"] in ["medium", "low"]
        
        # Test with neutral sentiment
        neutral = agent._generate_trading_signal(0.1, 0.0, "AAPL")
        assert neutral["action"] == "hold"
        assert neutral["impact"] == "low"
        
        # Test with negative sentiment
        negative = agent._generate_trading_signal(-0.4, -0.05, "AAPL")
        assert negative["action"] == "sell"
        assert negative["impact"] in ["medium", "low"]
        
        # Test with strong negative sentiment
        strong_negative = agent._generate_trading_signal(-0.7, -0.1, "AAPL")
        assert strong_negative["action"] == "sell"
        assert strong_negative["impact"] in ["high", "medium"]
    
    def test_process_valid_input(self, mock_data_manager):
        """Test the process method with valid input"""
        agent = SentimentAnalysisAgent(data_manager=mock_data_manager, verbose=0)
        
        # Create valid input
        input_data = AgentInput(
            request="Analyze sentiment for AAPL",
            context={
                "symbol": "AAPL",
                "date_range": {
                    "start_date": "2023-01-01",
                    "end_date": "2023-01-10"
                }
            }
        )
        
        output = agent.process(input_data)
        
        # Verify the output
        assert isinstance(output, AgentOutput)
        assert output.response is not None
        assert len(output.response) > 0
        assert output.data is not None
        assert output.data.get("symbol") == "AAPL"
        assert "sentiment_score" in output.data
        assert "sentiment_rating" in output.data
        assert "trading_signal" in output.data
        assert output.confidence > 0.5
    
    def test_process_no_symbol(self, mock_data_manager):
        """Test the process method with no symbol"""
        agent = SentimentAnalysisAgent(data_manager=mock_data_manager, verbose=0)
        
        # Create input with no symbol
        input_data = AgentInput(
            request="Analyze market sentiment",
            context={
                "date_range": {
                    "start_date": "2023-01-01",
                    "end_date": "2023-01-10"
                }
            }
        )
        
        output = agent.process(input_data)
        
        # Verify error response
        assert "No valid stock symbol" in output.response
        assert output.data is None
        assert output.confidence == 0.0
    
    def test_process_missing_date_range(self, mock_data_manager):
        """Test the process method with missing date range"""
        agent = SentimentAnalysisAgent(data_manager=mock_data_manager, verbose=0)
        
        # Create input with no date range
        input_data = AgentInput(
            request="Analyze sentiment for AAPL",
            context={
                "symbol": "AAPL"
            }
        )
        
        output = agent.process(input_data)
        
        # Should still work by using default date range
        assert output.response is not None
        assert len(output.response) > 0
        assert output.data is not None
        assert output.data.get("symbol") == "AAPL"
        assert "sentiment_score" in output.data
        assert "sentiment_rating" in output.data
        assert "trading_signal" in output.data
        assert output.confidence > 0.0
    
    def test_process_extract_symbol_from_request(self, mock_data_manager):
        """Test that the agent can extract symbol from request"""
        agent = SentimentAnalysisAgent(data_manager=mock_data_manager, verbose=0)
        
        # Create input without symbol in context but with symbol in request
        input_data = AgentInput(
            request="What's the sentiment for AAPL?",
            context={
                "date_range": {
                    "start_date": "2023-01-01",
                    "end_date": "2023-01-10"
                }
            }
        )
        
        output = agent.process(input_data)
        
        # Verify the output
        assert output.data is not None
        assert output.data.get("symbol") == "AAPL"
        assert "sentiment_score" in output.data
        assert "sentiment_rating" in output.data
        assert output.confidence > 0.0
    
    def test_format_response_positive(self, mock_data_manager):
        """Test response formatting with positive sentiment"""
        agent = SentimentAnalysisAgent(data_manager=mock_data_manager, verbose=0)
        
        sentiment_data = {
            "symbol": "AAPL",
            "sentiment_score": 0.65,
            "sentiment_rating": "positive",
            "sentiment_trend": 0.08,
            "trading_signal": {"action": "buy", "impact": "medium"},
            "source_data": {
                "has_news_data": True,
                "has_social_data": True,
                "news_article_count": 20,
                "social_engagement": 1200
            }
        }
        
        response = agent._format_response(sentiment_data)
        
        assert "AAPL" in response
        assert "positive" in response.lower()
        assert "buy" in response.lower()
        assert "news sources" in response.lower()
        assert "social media" in response.lower()
    
    def test_format_response_negative(self, mock_data_manager):
        """Test response formatting with negative sentiment"""
        agent = SentimentAnalysisAgent(data_manager=mock_data_manager, verbose=0)
        
        sentiment_data = {
            "symbol": "AAPL",
            "sentiment_score": -0.45,
            "sentiment_rating": "negative",
            "sentiment_trend": -0.12,
            "trading_signal": {"action": "sell", "impact": "medium"},
            "source_data": {
                "has_news_data": True,
                "has_social_data": True,
                "news_article_count": 25,
                "social_engagement": 1500
            }
        }
        
        response = agent._format_response(sentiment_data)
        
        assert "AAPL" in response
        assert "negative" in response.lower()
        assert "sell" in response.lower()
        assert "declining" in response.lower() or "downward" in response.lower()
        assert "news sources" in response.lower()
        assert "social media" in response.lower() 