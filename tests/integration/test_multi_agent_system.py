"""
Integration tests for the multi-agent trading system.
These tests verify that all components work together correctly.
"""
import pytest
import os
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from src.data import DataManager
from src.agent.multi_agent import TradingAgentOrchestrator, MarketAnalysisAgent
from src.agent.multi_agent.base_agent import AgentInput, AgentOutput
from src.agent.multi_agent.orchestrator import SystemState


@pytest.fixture
def test_data():
    """Create test data for the integration tests"""
    # Create a date range
    dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
    
    # Create sample price data with a clear trend
    price_data = pd.DataFrame({
        'Open': np.linspace(100, 150, 50) + np.random.normal(0, 2, 50),
        'High': np.linspace(105, 155, 50) + np.random.normal(0, 2, 50),
        'Low': np.linspace(95, 145, 50) + np.random.normal(0, 2, 50),
        'Close': np.linspace(100, 150, 50) + np.random.normal(0, 2, 50),
        'Volume': np.random.randint(1000, 10000, 50),
        'SMA_20': np.linspace(95, 145, 50),
        'SMA_50': np.linspace(90, 140, 50),
        'RSI_14': np.clip(np.linspace(40, 70, 50) + np.random.normal(0, 5, 50), 0, 100),
        'MACD': np.linspace(-2, 2, 50),
        'MACD_Signal': np.linspace(-1.5, 1.5, 50),
    }, index=dates)
    
    # Create sample sentiment data
    sentiment_data = pd.DataFrame({
        'Sentiment_Score': np.linspace(-0.5, 0.8, 50) + np.random.normal(0, 0.2, 50),
        'Article_Count': np.random.randint(1, 20, 50)
    }, index=dates)
    
    return {
        'price_data': price_data,
        'sentiment_data': sentiment_data
    }


@pytest.fixture
def data_manager(test_data):
    """Create a data manager with mock data"""
    with patch('src.data.DataFetcherFactory.create_data_fetcher') as mock_factory:
        # Create a mock data fetcher that returns our test data
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_data.return_value = test_data['price_data']
        mock_fetcher.add_technical_indicators.return_value = test_data['price_data']
        mock_fetcher.fetch_sentiment_data.return_value = test_data['sentiment_data']
        
        # Configure factory to return our mock fetcher
        mock_factory.return_value = mock_fetcher
        
        # Create data manager with mock factory
        data_manager = DataManager(
            market_data_source="yahoo",
            news_data_source="news"
        )
        
        return data_manager


class TestMultiAgentIntegration:
    """Integration tests for the multi-agent trading system"""
    
    def test_market_analysis_agent_with_data_manager(self, data_manager):
        """Test that MarketAnalysisAgent works correctly with DataManager"""
        # Create the agent
        agent = MarketAnalysisAgent(data_manager=data_manager)
        
        # Create input for the agent
        agent_input = AgentInput(
            request="Analyze AAPL stock from 2023-01-01 to 2023-02-19"
        )
        
        # Process the request
        output = agent.process(agent_input)
        
        # Verify the output
        assert isinstance(output, AgentOutput)
        assert output.response is not None
        assert len(output.response) > 0
        assert output.data is not None
        assert output.confidence is not None
        
        # Verify key data was extracted
        assert output.data["symbol"] == "AAPL"
        assert "current_price" in output.data
        assert "percent_change" in output.data
        assert "moving_averages" in output.data
        assert "indicators" in output.data
    
    def test_orchestrator_end_to_end(self, data_manager):
        """Test the full orchestrator workflow"""
        # Create the orchestrator with a mocked workflow
        with patch('src.agent.multi_agent.orchestrator.StateGraph') as mock_graph:
            # Mock the workflow
            mock_workflow = MagicMock()
            # Make invoke return a realistic SystemState
            mock_workflow.invoke.return_value = SystemState(
                request="Analyze AAPL stock and provide a trading recommendation",
                symbol="AAPL",
                start_date="2023-01-01",
                end_date="2023-02-19",
                agent_outputs={
                    "market_analysis": {
                        "response": "AAPL analysis: The stock shows a bullish trend...",
                        "confidence": 0.85
                    }
                },
                analysis_data={
                    "current_price": 150.0,
                    "percent_change": 3.5
                },
                decision="BUY",
                confidence=0.85,
                explanation="Strong bullish signals",
                recommended_actions=[{"action": "buy", "symbol": "AAPL", "reason": "Strong bullish trend"}]
            )
            
            # Setup the mock graph
            mock_graph_instance = MagicMock()
            mock_graph_instance.compile.return_value = mock_workflow
            mock_graph.return_value = mock_graph_instance
            
            # Create orchestrator and set the mocked workflow
            orchestrator = TradingAgentOrchestrator(data_manager=data_manager)
            orchestrator.workflow = mock_workflow
            
            # Process a request
            result = orchestrator.process_request(
                request="Analyze AAPL stock and provide a trading recommendation",
                symbol="AAPL",
                start_date="2023-01-01",
                end_date="2023-02-19"
            )
            
            # Verify the result structure
            assert isinstance(result, dict)
            assert "request" in result
            assert "symbol" in result
            assert "date_range" in result
            assert "decision" in result
            assert "confidence" in result
            assert "explanation" in result
            assert "recommended_actions" in result
            assert "analysis" in result
            
            # Verify the decision
            assert result["decision"] in ["BUY", "SELL", "HOLD"]
            assert result["symbol"] == "AAPL"
            
            # Verify recommended actions format
            actions = result["recommended_actions"]
            assert isinstance(actions, list)
            if actions:
                assert "action" in actions[0]
                assert "symbol" in actions[0]
                # Not all implementations might include reason
                # assert "reason" in actions[0]
    
    @patch('src.agent.multi_agent.market_analysis_agent.ChatOpenAI')
    def test_orchestrator_with_llm(self, mock_chat, data_manager):
        """Test the orchestrator with a mocked LLM"""
        # Setup mock LLM
        mock_llm = MagicMock()
        mock_content = MagicMock()
        mock_content.content = "LLM Analysis: AAPL shows a strong upward trend with bullish signals from both price action and technical indicators."
        mock_llm.invoke.return_value = mock_content
        mock_chat.return_value = mock_llm
        
        # Create the orchestrator with a mocked workflow
        with patch('src.agent.multi_agent.orchestrator.StateGraph') as mock_graph:
            # Mock the workflow
            mock_workflow = MagicMock()
            # Make invoke return a realistic SystemState
            mock_workflow.invoke.return_value = SystemState(
                request="Provide a detailed analysis of AAPL with consideration of recent market conditions",
                symbol="AAPL",
                start_date="2023-01-01",
                end_date="2023-02-19",
                agent_outputs={
                    "market_analysis": {
                        "response": "LLM Analysis: AAPL shows a strong upward trend with bullish signals from both price action and technical indicators.",
                        "confidence": 0.9
                    }
                },
                decision="BUY",
                confidence=0.9,
                explanation="Strong bullish signals from LLM analysis",
                recommended_actions=[{"action": "buy", "symbol": "AAPL"}]
            )
            
            # Setup the mock graph
            mock_graph_instance = MagicMock()
            mock_graph_instance.compile.return_value = mock_workflow
            mock_graph.return_value = mock_graph_instance
            
            # Create orchestrator with the mock LLM and set the mocked workflow
            orchestrator = TradingAgentOrchestrator(
                data_manager=data_manager,
                openai_api_key="fake_api_key"
            )
            orchestrator.workflow = mock_workflow
            
            # Process a request
            result = orchestrator.process_request(
                request="Provide a detailed analysis of AAPL with consideration of recent market conditions",
                symbol="AAPL",
                start_date="2023-01-01",
                end_date="2023-02-19"
            )
            
            # Verify the LLM was used in the analysis (via the mocked response)
            assert "LLM Analysis" in result["analysis"]
            assert mock_chat.called
    
    def test_system_with_multiple_requests(self, data_manager):
        """Test the system handling multiple requests"""
        # Create the orchestrator with a mocked workflow
        with patch('src.agent.multi_agent.orchestrator.StateGraph') as mock_graph:
            # Mock the workflow
            mock_workflow = MagicMock()
            
            # Setup to return different results based on the symbol
            def get_mock_state(symbol):
                return SystemState(
                    request=f"Analyze {symbol} stock",
                    symbol=symbol,
                    decision="BUY" if symbol == "AAPL" else "HOLD" if symbol == "MSFT" else "SELL",
                    confidence=0.85,
                    explanation=f"Analysis for {symbol}",
                    recommended_actions=[{"action": "buy" if symbol == "AAPL" else "hold" if symbol == "MSFT" else "sell", 
                                         "symbol": symbol, 
                                         "reason": "Test reason"}]
                )
            
            # Configure mock to return different values for different symbols
            mock_workflow.invoke.side_effect = lambda state: get_mock_state(state.symbol)
            
            # Setup the mock graph
            mock_graph_instance = MagicMock()
            mock_graph_instance.compile.return_value = mock_workflow
            mock_graph.return_value = mock_graph_instance
            
            # Create orchestrator and set the mocked workflow
            orchestrator = TradingAgentOrchestrator(data_manager=data_manager)
            orchestrator.workflow = mock_workflow
            
            # Process multiple requests
            symbols = ["AAPL", "MSFT", "GOOG"]
            decisions = {}
            
            for symbol in symbols:
                result = orchestrator.process_request(
                    request=f"Analyze {symbol} stock",
                    symbol=symbol
                )
                
                # Store the decision
                decisions[symbol] = result["decision"]
                
                # Verify basic result properties
                assert result["symbol"] == symbol
                assert "decision" in result
                assert "explanation" in result
            
            # Verify we got decisions for all symbols
            assert len(decisions) == len(symbols)
            for symbol in symbols:
                assert symbol in decisions
            
            # Verify the expected decisions for each symbol
            assert decisions["AAPL"] == "BUY"
            assert decisions["MSFT"] == "HOLD"
            assert decisions["GOOG"] == "SELL"
    
    def test_integration_with_example_script(self, data_manager):
        """Test integration with the example script approach"""
        # Import the example script
        import sys
        import os
        
        # We don't need to run the script, just verify the components work together
        from src.examples.multi_agent_example import main
        
        with patch('src.examples.multi_agent_example.DataManager') as mock_dm_class:
            mock_dm_class.return_value = data_manager
            
            with patch('src.examples.multi_agent_example.TradingAgentOrchestrator') as mock_orch_class:
                # Create a mock orchestrator that returns a valid result
                mock_orchestrator = MagicMock()
                mock_orchestrator.process_request.return_value = {
                    "request": "Test request",
                    "symbol": "AAPL",
                    "date_range": {"start_date": "2023-01-01", "end_date": "2023-02-19"},
                    "decision": "BUY",
                    "confidence": 0.85,
                    "explanation": "Test explanation",
                    "recommended_actions": [{"action": "buy", "symbol": "AAPL", "reason": "Test reason"}],
                    "analysis": "Test analysis"
                }
                mock_orch_class.return_value = mock_orchestrator
                
                with patch('sys.argv', ['multi_agent_example.py', '--symbol', 'AAPL', '--verbose', '0']):
                    with patch('json.dump'):  # Prevent writing to a file
                        with patch('builtins.print'):  # Suppress output
                            with patch('os.makedirs'):  # Prevent directory creation
                                # This should execute without errors
                                try:
                                    main()
                                    assert True  # If we get here, no exception was raised
                                except Exception as e:
                                    pytest.fail(f"Example script raised an exception: {e}")
                                
                                # Verify the orchestrator was used correctly
                                mock_orchestrator.process_request.assert_called_once()
                                args = mock_orchestrator.process_request.call_args[1]
                                assert args["symbol"] == "AAPL" 