"""
Orchestrator for the multi-agent trading system.
Coordinates the workflow between different agents using LangGraph.
"""
from typing import Dict, Any, List, Optional, Type
import json
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

# The MessageContext import was causing an error, so we'll remove it since it's not actually used in the code
# from langgraph.graph.message import MessageContext

from .base_agent import BaseAgent, AgentInput, AgentOutput
from .market_analysis_agent import MarketAnalysisAgent
from src.data import DataManager

class SystemState(BaseModel):
    """
    Shared state for the multi-agent system.
    Tracks the workflow progress and data flow between agents.
    """
    # Request information
    request: str = Field(..., description="Original user request")
    symbol: Optional[str] = Field(None, description="Stock symbol being analyzed")
    start_date: Optional[str] = Field(None, description="Start date for analysis")
    end_date: Optional[str] = Field(None, description="End date for analysis")
    
    # Agent interaction tracking
    agent_outputs: Dict[str, Any] = Field(default_factory=dict, description="Outputs from each agent")
    current_agent: Optional[str] = Field(None, description="Currently active agent")
    history: List[Dict[str, Any]] = Field(default_factory=list, description="History of agent interactions")
    
    # Decision-making data
    analysis_data: Optional[Dict[str, Any]] = Field(None, description="Market analysis data")
    sentiment_data: Optional[Dict[str, Any]] = Field(None, description="Sentiment analysis data")
    risk_assessment: Optional[Dict[str, Any]] = Field(None, description="Risk assessment data")
    
    # Final output
    decision: Optional[str] = Field(None, description="Final trading decision")
    confidence: Optional[float] = Field(None, description="Confidence in the decision (0-1)")
    explanation: Optional[str] = Field(None, description="Explanation for the decision")
    recommended_actions: Optional[List[Dict[str, Any]]] = Field(None, description="Recommended trading actions")

class TradingAgentOrchestrator:
    """
    Orchestrator for the multi-agent trading system.
    
    Manages the workflow between specialized agents using LangGraph.
    """
    
    def __init__(self, data_manager: DataManager, openai_api_key: Optional[str] = None, 
                verbose: int = 0):
        """
        Initialize the orchestrator.
        
        Args:
            data_manager (DataManager): Data manager for accessing market and other data
            openai_api_key (str, optional): OpenAI API key for LLM-based agents
            verbose (int): Verbosity level (0: silent, 1: normal, 2: detailed)
        """
        self.data_manager = data_manager
        self.openai_api_key = openai_api_key
        self.verbose = verbose
        
        # Initialize agents
        self.agents = self._initialize_agents()
        
        # Create the workflow graph
        self.workflow = self._build_workflow()
    
    def _initialize_agents(self) -> Dict[str, BaseAgent]:
        """Initialize all agents in the system"""
        agents = {}
        
        # Market Analysis Agent
        agents["market_analysis"] = MarketAnalysisAgent(
            data_manager=self.data_manager,
            openai_api_key=self.openai_api_key,
            verbose=self.verbose
        )
        
        # TODO: Add more agents as they are implemented
        # agents["sentiment_analysis"] = SentimentAnalysisAgent(...)
        # agents["strategy"] = StrategyAgent(...)
        # agents["risk"] = RiskManagementAgent(...)
        # agents["execution"] = ExecutionAgent(...)
        
        return agents
    
    def _build_workflow(self) -> StateGraph:
        """
        Build the LangGraph workflow for agent orchestration.
        
        Returns:
            StateGraph: LangGraph workflow
        """
        # Create a new graph
        workflow = StateGraph(SystemState)
        
        # Add nodes for each agent
        workflow.add_node("market_analysis", self._run_market_analysis_agent)
        
        # TODO: Add more nodes as more agents are implemented
        # workflow.add_node("sentiment_analysis", self._run_sentiment_analysis_agent)
        # workflow.add_node("strategy", self._run_strategy_agent)
        # workflow.add_node("risk", self._run_risk_agent)
        # workflow.add_node("execution", self._run_execution_agent)
        
        # Add an end node for final processing
        workflow.add_node("finalize", self._finalize_workflow)
        
        # Define the edges (flow between agents)
        # In the initial implementation with just one agent, go straight to finalize
        workflow.add_edge("market_analysis", "finalize")
        
        # TODO: Update edges as more agents are implemented
        # workflow.add_edge("market_analysis", "sentiment_analysis")
        # workflow.add_edge("sentiment_analysis", "strategy")
        # workflow.add_edge("strategy", "risk")
        # workflow.add_conditional_edges(
        #     "risk",
        #     self._should_execute,
        #     {
        #         "execute": "execution",
        #         "abort": "finalize"
        #     }
        # )
        # workflow.add_edge("execution", "finalize")
        
        # Set the entry point
        workflow.set_entry_point("market_analysis")
        
        # Add edge from finalize to END
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    def _run_market_analysis_agent(self, state: SystemState) -> SystemState:
        """
        Run the market analysis agent on the current state.
        
        Args:
            state (SystemState): Current system state
            
        Returns:
            SystemState: Updated system state
        """
        # Track the current agent
        state.current_agent = "market_analysis"
        
        # Create input for the agent
        agent_input = AgentInput(
            request=state.request,
            context={
                "symbol": state.symbol,
                "date_range": {
                    "start_date": state.start_date,
                    "end_date": state.end_date
                } if state.start_date and state.end_date else None
            }
        )
        
        # Get the agent
        agent = self.agents["market_analysis"]
        
        # Process the request
        output = agent.process(agent_input)
        
        # Store the output
        state.agent_outputs["market_analysis"] = output.dict()
        
        # Update state with analysis data
        if output.data:
            state.analysis_data = output.data
            
            # Extract symbol if not already set
            if not state.symbol and "symbol" in output.data:
                state.symbol = output.data["symbol"]
                
            # Extract date range if not already set
            if (not state.start_date or not state.end_date) and "start_date" in output.data and "end_date" in output.data:
                state.start_date = output.data["start_date"]
                state.end_date = output.data["end_date"]
        
        # Add to history
        state.history.append({
            "agent": "market_analysis",
            "input": agent_input.dict(),
            "output": output.dict()
        })
        
        return state
    
    def _finalize_workflow(self, state: SystemState) -> SystemState:
        """
        Finalize the workflow and generate the final output.
        
        Args:
            state (SystemState): Current system state
            
        Returns:
            SystemState: Final system state with decision and recommendations
        """
        # Extract market analysis data
        market_analysis = state.agent_outputs.get("market_analysis", {})
        analysis_response = market_analysis.get("response", "No market analysis available.")
        confidence = market_analysis.get("confidence", 0.0)
        
        # In a more complete system, we would combine insights from multiple agents here
        # For now, just use the market analysis as our final output
        
        # Generate a simple decision based on market analysis
        if state.analysis_data:
            percent_change = state.analysis_data.get("percent_change", 0)
            moving_averages = state.analysis_data.get("moving_averages", {})
            indicators = state.analysis_data.get("indicators", {})
            
            # Simple decision logic based on available data
            bullish_signals = 0
            bearish_signals = 0
            
            # Price above/below moving averages
            if moving_averages.get("sma_20") and moving_averages.get("sma_50"):
                if state.analysis_data["current_price"] > moving_averages["sma_20"]:
                    bullish_signals += 1
                else:
                    bearish_signals += 1
                    
                if state.analysis_data["current_price"] > moving_averages["sma_50"]:
                    bullish_signals += 1
                else:
                    bearish_signals += 1
            
            # MA crossover
            if moving_averages.get("ma_cross"):
                bullish_signals += 1
                
            # RSI signals
            if indicators.get("rsi"):
                if indicators["rsi"] < 30:
                    bullish_signals += 1  # Oversold, potential buy
                elif indicators["rsi"] > 70:
                    bearish_signals += 1  # Overbought, potential sell
            
            # MACD signals
            if indicators.get("macd_cross_up"):
                bullish_signals += 1
            elif indicators.get("macd_cross_down"):
                bearish_signals += 1
            
            # Make a decision
            if bullish_signals > bearish_signals:
                state.decision = "BUY"
                action_type = "buy"
                reason = "Bullish technical signals outweigh bearish signals."
            elif bearish_signals > bullish_signals:
                state.decision = "SELL"
                action_type = "sell"
                reason = "Bearish technical signals outweigh bullish signals."
            else:
                state.decision = "HOLD"
                action_type = "hold"
                reason = "Technical signals are mixed or neutral."
                
            # Generate a recommendation
            state.recommended_actions = [{
                "action": action_type,
                "symbol": state.symbol,
                "reason": reason,
                "confidence": confidence
            }]
            
            # Generate an explanation
            state.explanation = f"Based on the market analysis, the recommendation is to {action_type.upper()} {state.symbol}. {reason}"
        else:
            # Default if no analysis data
            state.decision = "INSUFFICIENT_DATA"
            state.explanation = "Insufficient data to make a trading decision."
            state.recommended_actions = []
            
        # Set confidence from the market analysis
        state.confidence = confidence
            
        return state
    
    def _should_execute(self, state: SystemState) -> str:
        """
        Determine whether to proceed with execution or abort based on risk assessment.
        
        Args:
            state (SystemState): Current system state
            
        Returns:
            str: Next node - 'execute' or 'abort'
        """
        # This would include more sophisticated logic in a complete implementation
        # For now, it's a placeholder
        if state.risk_assessment and "risk_score" in state.risk_assessment:
            risk_score = state.risk_assessment["risk_score"]
            if risk_score <= 0.7:  # Threshold for acceptable risk
                return "execute"
            else:
                return "abort"
        return "abort"  # Default to abort if no risk assessment
    
    def process_request(self, request: str, symbol: Optional[str] = None, 
                       start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a trading analysis request through the multi-agent system.
        
        Args:
            request (str): User request text
            symbol (str, optional): Stock symbol to analyze
            start_date (str, optional): Start date for analysis (YYYY-MM-DD)
            end_date (str, optional): End date for analysis (YYYY-MM-DD)
            
        Returns:
            Dict[str, Any]: Results from the multi-agent analysis
        """
        # Initialize system state
        initial_state = SystemState(
            request=request,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        # Run the workflow
        try:
            if self.verbose > 0:
                print(f"Processing request: {request}")
                
            final_state = self.workflow.invoke(initial_state)
            
            # Format the result
            result = {
                "request": request,
                "symbol": final_state.symbol,
                "date_range": {
                    "start_date": final_state.start_date,
                    "end_date": final_state.end_date
                },
                "decision": final_state.decision,
                "confidence": final_state.confidence,
                "explanation": final_state.explanation,
                "recommended_actions": final_state.recommended_actions,
                "analysis": final_state.agent_outputs.get("market_analysis", {}).get("response", "")
            }
            
            return result
            
        except Exception as e:
            if self.verbose > 0:
                print(f"Error processing request: {e}")
            
            # Return error information
            return {
                "request": request,
                "error": str(e),
                "symbol": symbol,
                "date_range": {
                    "start_date": start_date,
                    "end_date": end_date
                }
            }
    
    def get_available_agents(self) -> List[str]:
        """Get list of available agent names"""
        return list(self.agents.keys()) 