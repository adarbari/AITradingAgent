"""
Integration tests for the Execution Agent functionality
"""
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.agent.multi_agent import TradingAgentOrchestrator
from src.agent.multi_agent.execution_agent import ExecutionAgent
from src.data import DataManager


@pytest.fixture
def mock_data_manager():
    """Create a mock data manager for tests"""
    data_manager = MagicMock()
    
    # Create sample market data
    dates = pd.date_range(start="2023-01-01", periods=30, freq="D")
    
    # Create price series with different trends
    aapl_prices = np.linspace(150, 165, 30) + np.random.normal(0, 2, 30)  # Uptrend
    msft_prices = np.linspace(270, 285, 30) + np.random.normal(0, 3, 30)  # Uptrend
    
    # Configure mock to return data based on symbol
    def get_market_data(symbol=None, **kwargs):
        if symbol == "AAPL":
            data = pd.DataFrame({
                'Open': aapl_prices * 0.99,
                'High': aapl_prices * 1.02,
                'Low': aapl_prices * 0.98,
                'Close': aapl_prices,
                'Volume': np.random.randint(5000000, 10000000, 30),
                'SMA_20': aapl_prices * 0.97,
                'SMA_50': aapl_prices * 0.95,
                'RSI_14': np.clip(60 + np.random.normal(0, 5, 30), 30, 70),
                'MACD': np.random.normal(0.5, 0.2, 30),
                'MACD_Signal': np.random.normal(0.3, 0.1, 30),
                'Upper_Band': aapl_prices * 1.05,
                'Lower_Band': aapl_prices * 0.95
            }, index=dates)
            return data
            
        elif symbol == "MSFT":
            data = pd.DataFrame({
                'Open': msft_prices * 0.99,
                'High': msft_prices * 1.02,
                'Low': msft_prices * 0.98,
                'Close': msft_prices,
                'Volume': np.random.randint(5000000, 10000000, 30),
                'SMA_20': msft_prices * 0.97,
                'SMA_50': msft_prices * 0.95,
                'RSI_14': np.clip(65 + np.random.normal(0, 5, 30), 30, 70),
                'MACD': np.random.normal(0.6, 0.2, 30),
                'MACD_Signal': np.random.normal(0.4, 0.1, 30),
                'Upper_Band': msft_prices * 1.05,
                'Lower_Band': msft_prices * 0.95
            }, index=dates)
            return data
            
        return None
    
    data_manager.get_market_data.side_effect = get_market_data
    
    # Configure prepare_data_for_agent to return a simple dict with market data
    def prepare_data_for_agent(symbol, **kwargs):
        market_data = get_market_data(symbol=symbol, **kwargs)
        if market_data is not None:
            return {"market": market_data}
        return None
    
    data_manager.prepare_data_for_agent.side_effect = prepare_data_for_agent
    
    return data_manager


@pytest.fixture
def sample_portfolio():
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
            }
        ],
        "1m_return": 3.5,
        "3m_return": 7.2,
        "ytd_return": 5.8,
        "1y_return": 12.6,
        "volatility": 15.0,
        "sharpe_ratio": 1.2
    }


class TestExecutionIntegration:
    """Integration tests for execution agent functionality"""
    
    @patch('src.agent.multi_agent.market_analysis_agent.ChatOpenAI')
    @patch('src.agent.multi_agent.orchestrator.StateGraph')
    def test_execution_integration(self, mock_state_graph, mock_chat, mock_data_manager, sample_portfolio):
        """Test integration of execution agent with orchestrator"""
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
            "risk_assessment": {
                "response": "Risk assessment for AAPL shows medium risk.",
                "data": {"risk_score": 0.5, "risk_rating": "Medium"},
                "confidence": 0.7
            },
            "portfolio_management": {
                "response": "Portfolio optimization complete with moderate risk profile.",
                "data": {
                    "portfolio_metrics": {
                        "total_value": 100000.0,
                        "allocations": {
                            "AAPL": {"percentage": 16.5, "value": 16500.0}
                        }
                    },
                    "risk_tolerance": "moderate",
                    "recommendations": [
                        {"action": "buy", "symbol": "AAPL", "shares": 20, "value": 3300.0}
                    ]
                },
                "confidence": 0.85
            },
            "execution": {
                "response": "Execution plan for BUY 20 shares of AAPL prepared.",
                "data": {
                    "symbol": "AAPL",
                    "action": "buy",
                    "quantity": 20,
                    "order_type": "limit",
                    "order_params": {"limit_price": 165.5},
                    "execution_strategy": "Staged Execution (2 stages)",
                    "estimated_costs": 0.00065,
                    "market_impact": 0.00023
                },
                "confidence": 0.82
            }
        }
        mock_result.decision = "BUY"
        mock_result.confidence = 0.75
        mock_result.explanation = "Bullish signals with acceptable risk"
        mock_result.recommended_actions = [
            {
                "action": "buy",
                "symbol": "AAPL",
                "position_size": "moderate",
                "execution": {
                    "order_type": "limit",
                    "execution_strategy": "Staged Execution (2 stages)",
                    "estimated_costs": 0.00065,
                    "market_impact": 0.00023,
                    "parameters": {"limit_price": 165.5}
                }
            }
        ]
        mock_result.analysis_data = {"current_price": 165.0, "trend": "bullish"}
        mock_result.risk_assessment = {"risk_score": 0.5, "risk_rating": "Medium"}
        mock_result.portfolio_recommendations = {
            "portfolio_metrics": {
                "total_value": 100000.0,
                "allocations": {"AAPL": {"percentage": 16.5, "value": 16500.0}}
            },
            "risk_tolerance": "moderate",
            "recommendations": [{"action": "buy", "symbol": "AAPL", "shares": 20, "value": 3300.0}]
        }
        mock_result.execution_plan = {
            "symbol": "AAPL",
            "action": "buy",
            "quantity": 20,
            "order_type": "limit",
            "order_params": {"limit_price": 165.5},
            "execution_strategy": "Staged Execution (2 stages)",
            "estimated_costs": 0.00065,
            "market_impact": 0.00023
        }
        mock_result.trade_details = {
            "symbol": "AAPL",
            "action": "buy",
            "quantity": 20,
            "price": 165.0,
            "urgency": "normal"
        }
        mock_graph.invoke.return_value = mock_result
        
        # Configure chat mock
        mock_chat_instance = MagicMock()
        mock_chat.return_value = mock_chat_instance
        
        # Set up a simple response from the chat
        mock_chat_instance.invoke.return_value.content = "This is a sample analysis of the market trends."
        
        # Initialize orchestrator with the mocked components
        orchestrator = TradingAgentOrchestrator(
            data_manager=mock_data_manager,
            verbose=1
        )
        orchestrator.workflow = mock_graph
        
        # Process a request that includes execution instructions
        result = orchestrator.process_request(
            request="Analyze AAPL, optimize my portfolio, and execute any recommended trades",
            symbol="AAPL",
            start_date="2023-01-01",
            end_date="2023-01-30",
            portfolio=sample_portfolio,
            risk_tolerance="moderate",
            execution_urgency="normal"
        )
        
        # Verify the result structure
        assert "request" in result
        assert "symbol" in result
        assert "decision" in result
        assert "confidence" in result
        assert "explanation" in result
        assert "recommended_actions" in result
        assert "analysis" in result
        assert "risk_assessment" in result
        assert "portfolio_management" in result
        assert "execution" in result
        
        # Verify that execution details exist
        assert "execution_data" in result
        assert "order_type" in result["execution_data"]
        assert "execution_strategy" in result["execution_data"]
        
        # Verify that recommendations contain execution information
        for action in result["recommended_actions"]:
            if action["symbol"] == "AAPL" and action["action"] == "buy":
                assert "execution" in action
                assert "order_type" in action["execution"]
                assert "execution_strategy" in action["execution"]
                assert "estimated_costs" in action["execution"]
                assert "market_impact" in action["execution"]
    
    def test_execution_agent_standalone(self, mock_data_manager):
        """Test the execution agent directly without going through orchestrator"""
        # Create the execution agent
        execution_agent = ExecutionAgent(data_manager=mock_data_manager, verbose=1)
        
        # Prepare sample trade details
        trade_details = {
            "symbol": "AAPL",
            "action": "buy",
            "quantity": 100,
            "price": 165.0,
            "urgency": "normal"
        }
        
        # Prepare sample market analysis data
        market_analysis = {
            "trend": "bullish",
            "volatility": "medium",
            "current_price": 165.0,
            "support_levels": [160.0, 155.0],
            "resistance_levels": [170.0, 175.0]
        }
        
        # Create input with trade details
        from src.agent.multi_agent.base_agent import AgentInput
        input_data = AgentInput(
            request="Execute BUY order for 100 shares of AAPL",
            context={
                "trade_details": trade_details,
                "market_analysis": market_analysis
            }
        )
        
        # Process the request
        output = execution_agent.process(input_data)
        
        # Verify the output
        assert output.response is not None
        assert len(output.response) > 0
        assert output.data is not None
        assert "symbol" in output.data
        assert output.data["symbol"] == "AAPL"
        assert "action" in output.data
        assert output.data["action"] == "buy"
        assert "quantity" in output.data
        assert output.data["quantity"] == 100
        assert "order_type" in output.data
        assert "order_params" in output.data
        assert "execution_strategy" in output.data
        assert "estimated_costs" in output.data
        assert "market_impact" in output.data
        
        # Verify market conditions analysis
        assert "market_conditions" in output.data
        assert "volatility" in output.data["market_conditions"]
        assert "liquidity" in output.data["market_conditions"]
        assert "position_size_ratio" in output.data["market_conditions"]
        
        # Verify execution strategy is appropriate for the order size and urgency
        assert output.data["execution_strategy"] in [
            "Standard Market Order", 
            "Limit Order Strategy", 
            "Staged Execution (2 stages)", 
            "VWAP (Half Day)"
        ]
    
    @patch('src.agent.multi_agent.market_analysis_agent.ChatOpenAI')
    @patch('src.agent.multi_agent.orchestrator.StateGraph')
    def test_execution_not_triggered(self, mock_state_graph, mock_chat, mock_data_manager, sample_portfolio):
        """Test that execution is not triggered for non-execution requests"""
        # Configure mock graph
        mock_graph = MagicMock()
        mock_state_graph.return_value = mock_graph
        mock_graph.compile.return_value = mock_graph
        
        # Configure result from invoke with no execution
        mock_result = MagicMock()
        mock_result.symbol = "AAPL"
        mock_result.agent_outputs = {
            "market_analysis": {
                "response": "Analysis of AAPL shows a neutral trend.",
                "data": {"current_price": 165.0, "trend": "neutral"},
                "confidence": 0.8
            },
            "risk_assessment": {
                "response": "Risk assessment for AAPL shows medium risk.",
                "data": {"risk_score": 0.5, "risk_rating": "Medium"},
                "confidence": 0.7
            },
            "portfolio_management": {
                "response": "Portfolio optimization complete with moderate risk profile.",
                "data": {
                    "portfolio_metrics": {
                        "total_value": 100000.0,
                        "allocations": {"AAPL": {"percentage": 16.5}}
                    },
                    "risk_tolerance": "moderate",
                    "recommendations": []  # No recommendations
                },
                "confidence": 0.85
            }
        }
        mock_result.decision = "HOLD"
        mock_result.confidence = 0.7
        mock_result.explanation = "Not enough signals to act"
        mock_result.recommended_actions = [{"action": "hold", "symbol": "AAPL"}]
        mock_result.execution_plan = None  # No execution plan
        mock_result.trade_details = None   # No trade details
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
        
        # Process a request that does not include execution instructions
        result = orchestrator.process_request(
            request="Analyze AAPL and optimize my portfolio",
            symbol="AAPL",
            portfolio=sample_portfolio,
            risk_tolerance="moderate"
        )
        
        # Verify the result structure
        assert "decision" in result
        assert result["decision"] == "HOLD"
        
        # In our mock setup, "execution" may be None or have a placeholder value
        # What's important is that no execution data was generated
        assert "execution_data" not in result 