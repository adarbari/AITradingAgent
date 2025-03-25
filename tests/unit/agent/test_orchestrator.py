"""
Tests for the TradingAgentOrchestrator class
"""
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.agent.multi_agent.base_agent import AgentInput, AgentOutput
from src.agent.multi_agent.market_analysis_agent import MarketAnalysisAgent
from src.agent.multi_agent.orchestrator import TradingAgentOrchestrator, SystemState


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


@pytest.fixture
def mock_market_analysis_agent():
    """Create a mock market analysis agent"""
    agent = MagicMock()
    
    # Configure the mock to return a realistic output
    def mock_process(input_data):
        return AgentOutput(
            response="AAPL analysis: The stock shows a bullish trend...",
            data={
                "symbol": "AAPL",
                "current_price": 150.0,
                "price_change": 5.0,
                "percent_change": 3.5,
                "volatility": 15.0,
                "moving_averages": {
                    "sma_20": 145.0,
                    "sma_50": 140.0,
                    "price_vs_sma20": 5.0,
                    "price_vs_sma50": 10.0,
                    "ma_cross": True
                },
                "indicators": {
                    "rsi": 65.0,
                    "rsi_signal": "neutral",
                    "macd": 2.0,
                    "macd_signal": 1.0,
                    "macd_cross_up": True,
                    "macd_cross_down": False
                },
                "start_date": "2023-01-01",
                "end_date": "2023-01-31"
            },
            confidence=0.85
        )
    
    agent.process.side_effect = mock_process
    return agent


class TestSystemState:
    """Test cases for the SystemState model"""
    
    def test_init_with_required_fields(self):
        """Test initialization with only required fields"""
        state = SystemState(request="Analyze AAPL stock")
        
        assert state.request == "Analyze AAPL stock"
        assert state.symbol is None
        assert state.start_date is None
        assert state.end_date is None
        assert state.agent_outputs == {}
        assert state.current_agent is None
        assert state.history == []
        assert state.analysis_data is None
        assert state.decision is None
    
    def test_init_with_all_fields(self):
        """Test initialization with all fields"""
        state = SystemState(
            request="Analyze AAPL stock",
            symbol="AAPL",
            start_date="2023-01-01",
            end_date="2023-01-31",
            agent_outputs={"market_analysis": {"response": "Analysis..."}},
            current_agent="market_analysis",
            history=[{"agent": "market_analysis", "input": {}, "output": {}}],
            analysis_data={"price": 150.0},
            decision="BUY",
            confidence=0.85,
            explanation="Strong bullish signals",
            recommended_actions=[{"action": "buy", "symbol": "AAPL"}]
        )
        
        assert state.request == "Analyze AAPL stock"
        assert state.symbol == "AAPL"
        assert state.start_date == "2023-01-01"
        assert state.end_date == "2023-01-31"
        assert state.agent_outputs == {"market_analysis": {"response": "Analysis..."}}
        assert state.current_agent == "market_analysis"
        assert len(state.history) == 1
        assert state.analysis_data == {"price": 150.0}
        assert state.decision == "BUY"
        assert state.confidence == 0.85
        assert state.explanation == "Strong bullish signals"
        assert state.recommended_actions == [{"action": "buy", "symbol": "AAPL"}]


class TestTradingAgentOrchestrator:
    """Test cases for the TradingAgentOrchestrator"""
    
    def test_initialization(self, mock_data_manager):
        """Test orchestra initialization"""
        orchestrator = TradingAgentOrchestrator(data_manager=mock_data_manager, verbose=1)
        
        assert orchestrator.data_manager == mock_data_manager
        assert orchestrator.openai_api_key is None
        assert orchestrator.verbose == 1
        assert "market_analysis" in orchestrator.agents
        assert orchestrator.workflow is not None
    
    def test_initialization_with_api_key(self, mock_data_manager):
        """Test orchestrator initialization with API key"""
        with patch('src.agent.multi_agent.market_analysis_agent.ChatOpenAI'):
            orchestrator = TradingAgentOrchestrator(
                data_manager=mock_data_manager,
                openai_api_key="test_api_key",
                verbose=1
            )
            
            assert orchestrator.openai_api_key == "test_api_key"
            assert "market_analysis" in orchestrator.agents
    
    @patch('src.agent.multi_agent.orchestrator.MarketAnalysisAgent')
    def test_initialize_agents(self, mock_agent_class, mock_data_manager):
        """Test agent initialization"""
        mock_agent = MagicMock()
        mock_agent_class.return_value = mock_agent
        
        orchestrator = TradingAgentOrchestrator(data_manager=mock_data_manager)
        
        # Verify the agent was created correctly
        mock_agent_class.assert_called_once_with(
            data_manager=mock_data_manager,
            openai_api_key=None,
            verbose=0
        )
        
        # Verify the agent was added to the dict
        assert orchestrator.agents["market_analysis"] == mock_agent
    
    def test_run_market_analysis_agent(self, mock_data_manager, mock_market_analysis_agent):
        """Test running the market analysis agent"""
        # Create orchestrator with mock agent
        orchestrator = TradingAgentOrchestrator(data_manager=mock_data_manager)
        orchestrator.agents["market_analysis"] = mock_market_analysis_agent
        
        # Initial state
        state = SystemState(
            request="Analyze AAPL stock",
            symbol="AAPL",
            start_date="2023-01-01",
            end_date="2023-01-31"
        )
        
        # Run the agent
        updated_state = orchestrator._run_market_analysis_agent(state)
        
        # Verify state was updated correctly
        assert updated_state.current_agent == "market_analysis"
        assert "market_analysis" in updated_state.agent_outputs
        assert updated_state.analysis_data is not None
        assert len(updated_state.history) == 1
        assert updated_state.history[0]["agent"] == "market_analysis"
        
        # Verify agent was called correctly
        mock_market_analysis_agent.process.assert_called_once()
        
        # Check input passed to agent
        call_args = mock_market_analysis_agent.process.call_args[0][0]
        assert isinstance(call_args, AgentInput)
        assert call_args.request == "Analyze AAPL stock"
        assert call_args.context["symbol"] == "AAPL"
        assert call_args.context["date_range"]["start_date"] == "2023-01-01"
        assert call_args.context["date_range"]["end_date"] == "2023-01-31"
    
    def test_finalize_workflow(self, mock_data_manager):
        """Test finalizing the workflow"""
        orchestrator = TradingAgentOrchestrator(data_manager=mock_data_manager)
        
        # Create a state with analysis data
        state = SystemState(
            request="Analyze AAPL stock",
            symbol="AAPL",
            agent_outputs={
                "market_analysis": {
                    "response": "AAPL analysis: The stock shows a bullish trend...",
                    "confidence": 0.85,
                    "data": {
                        "current_price": 150.0,
                        "percent_change": 3.5,
                        "moving_averages": {
                            "sma_20": 145.0, 
                            "sma_50": 140.0
                        },
                        "indicators": {
                            "rsi": 65.0,
                            "macd_cross_up": True
                        }
                    }
                }
            },
            analysis_data={
                "current_price": 150.0,
                "percent_change": 3.5,
                "moving_averages": {
                    "sma_20": 145.0, 
                    "sma_50": 140.0
                },
                "indicators": {
                    "rsi": 65.0,
                    "macd_cross_up": True
                }
            }
        )
        
        # Finalize the workflow
        final_state = orchestrator._finalize_workflow(state)
        
        # Verify decision was made
        assert final_state.decision in ["BUY", "SELL", "HOLD"]
        assert final_state.confidence is not None
        assert final_state.explanation is not None
        assert final_state.recommended_actions is not None
        assert len(final_state.recommended_actions) > 0
        
        # For this specific bullish setup, should recommend BUY
        assert final_state.decision == "BUY"
    
    def test_process_request(self, mock_data_manager, mock_market_analysis_agent):
        """Test processing a complete request"""
        # Create orchestrator with mock components
        with patch('src.agent.multi_agent.orchestrator.StateGraph') as mock_graph:
            # Mock the workflow
            mock_workflow = MagicMock()
            # Make invoke return a realistic state
            mock_workflow.invoke.return_value = SystemState(
                request="Analyze AAPL stock",
                symbol="AAPL",
                start_date="2023-01-01",
                end_date="2023-01-31",
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
                recommended_actions=[{"action": "buy", "symbol": "AAPL"}]
            )
            
            # Setup the mock graph
            mock_graph_instance = MagicMock()
            mock_graph_instance.compile.return_value = mock_workflow
            mock_graph.return_value = mock_graph_instance
            
            # Create orchestrator
            orchestrator = TradingAgentOrchestrator(data_manager=mock_data_manager)
            orchestrator.agents["market_analysis"] = mock_market_analysis_agent
            orchestrator.workflow = mock_workflow
            
            # Process a request
            result = orchestrator.process_request(
                request="Analyze AAPL stock",
                symbol="AAPL",
                start_date="2023-01-01",
                end_date="2023-01-31"
            )
            
            # Verify result structure
            assert "request" in result
            assert "symbol" in result
            assert "date_range" in result
            assert "decision" in result
            assert "confidence" in result
            assert "explanation" in result
            assert "recommended_actions" in result
            assert "analysis" in result
            
            # Verify workflow was invoked
            mock_workflow.invoke.assert_called_once()
    
    def test_process_request_with_error(self, mock_data_manager):
        """Test processing a request that encounters an error"""
        # Create orchestrator with mock components that raises an exception
        with patch('src.agent.multi_agent.orchestrator.StateGraph') as mock_graph:
            # Mock the workflow to raise an exception
            mock_workflow = MagicMock()
            mock_workflow.invoke.side_effect = Exception("Test error")
            
            # Setup the mock graph
            mock_graph_instance = MagicMock()
            mock_graph_instance.compile.return_value = mock_workflow
            mock_graph.return_value = mock_graph_instance
            
            # Create orchestrator
            orchestrator = TradingAgentOrchestrator(data_manager=mock_data_manager)
            orchestrator.workflow = mock_workflow
            
            # Process a request
            result = orchestrator.process_request(
                request="Analyze AAPL stock",
                symbol="AAPL"
            )
            
            # Verify error handling
            assert "request" in result
            assert "error" in result
            assert "Test error" in result["error"]
    
    def test_get_available_agents(self, mock_data_manager):
        """Test getting the list of available agents"""
        orchestrator = TradingAgentOrchestrator(data_manager=mock_data_manager)
        
        agents = orchestrator.get_available_agents()
        
        assert isinstance(agents, list)
        assert "market_analysis" in agents 