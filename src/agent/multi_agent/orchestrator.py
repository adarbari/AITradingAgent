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
from .risk_assessment_agent import RiskAssessmentAgent
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
        
        # Risk Assessment Agent
        agents["risk_assessment"] = RiskAssessmentAgent(
            data_manager=self.data_manager,
            verbose=self.verbose
        )
        
        # TODO: Add more agents as they are implemented
        # agents["sentiment_analysis"] = SentimentAnalysisAgent(...)
        # agents["strategy"] = StrategyAgent(...)
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
        workflow.add_node("risk_assessment", self._run_risk_assessment_agent)
        
        # TODO: Add more nodes as more agents are implemented
        # workflow.add_node("sentiment_analysis", self._run_sentiment_analysis_agent)
        # workflow.add_node("strategy", self._run_strategy_agent)
        # workflow.add_node("execution", self._run_execution_agent)
        
        # Add an end node for final processing
        workflow.add_node("finalize", self._finalize_workflow)
        
        # Define the edges (flow between agents)
        # Market analysis -> Risk assessment -> Finalize
        workflow.add_edge("market_analysis", "risk_assessment")
        workflow.add_edge("risk_assessment", "finalize")
        
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
        Run the market analysis agent with the current state.
        
        Args:
            state (SystemState): Current system state
            
        Returns:
            SystemState: Updated system state
        """
        if self.verbose > 0:
            print("Running Market Analysis Agent...")
        
        # Get the agent
        agent = self.agents["market_analysis"]
        
        # Prepare input for the agent
        agent_input = AgentInput(
            request=state.request,
            context={
                "symbol": state.symbol,
                "date_range": {
                    "start_date": state.start_date,
                    "end_date": state.end_date
                }
            }
        )
        
        # Run the agent
        output = agent.process(agent_input)
        
        # Store the output in the state
        state.agent_outputs["market_analysis"] = {
            "response": output.response,
            "data": output.data,
            "confidence": output.confidence
        }
        
        # Update the agent name in state
        state.current_agent = "market_analysis"
        
        # Extract and store analysis data
        if output.data:
            state.analysis_data = output.data
        
        # Save interaction to history
        state.history.append({
            "agent": "market_analysis",
            "input": agent_input.dict(),
            "output": output.dict()
        })
        
        return state
    
    def _run_risk_assessment_agent(self, state: SystemState) -> SystemState:
        """
        Run the risk assessment agent with the current state.
        
        Args:
            state (SystemState): Current system state
            
        Returns:
            SystemState: Updated system state
        """
        if self.verbose > 0:
            print("Running Risk Assessment Agent...")
        
        # Get the agent
        agent = self.agents["risk_assessment"]
        
        # Prepare input for the agent
        context = {
            "symbol": state.symbol,
            "date_range": {
                "start_date": state.start_date,
                "end_date": state.end_date
            }
        }
        
        # Add market analysis data to the context if available
        if state.analysis_data:
            context["market_analysis"] = state.analysis_data
        
        # Create agent input
        agent_input = AgentInput(
            request=f"Assess risk for {state.symbol} based on market analysis",
            context=context
        )
        
        # Run the agent
        output = agent.process(agent_input)
        
        # Store the output in the state
        state.agent_outputs["risk_assessment"] = {
            "response": output.response,
            "data": output.data,
            "confidence": output.confidence
        }
        
        # Update the agent name in state
        state.current_agent = "risk_assessment"
        
        # Extract and store risk assessment data
        if output.data:
            state.risk_assessment = output.data
        
        # Save interaction to history
        state.history.append({
            "agent": "risk_assessment",
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
        # Extract agent outputs
        market_analysis = state.agent_outputs.get("market_analysis", {})
        risk_assessment = state.agent_outputs.get("risk_assessment", {})
        
        # Get confidence values
        market_confidence = market_analysis.get("confidence", 0.0)
        risk_confidence = risk_assessment.get("confidence", 0.0)
        
        # In a more complete system, we would combine insights from multiple agents here
        
        # Generate a decision based on market analysis and risk assessment
        if state.analysis_data and state.risk_assessment:
            # Market signals
            percent_change = state.analysis_data.get("percent_change", 0)
            moving_averages = state.analysis_data.get("moving_averages", {})
            indicators = state.analysis_data.get("indicators", {})
            
            # Risk signals
            risk_score = state.risk_assessment.get("risk_score", 0.5)
            risk_rating = state.risk_assessment.get("risk_rating", "Medium")
            market_condition = state.risk_assessment.get("market_condition", {})
            
            # Count signals
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
                
            # Risk adjustment
            # If risk is high, increase the threshold for buying
            if risk_rating == "High":
                buy_threshold = 3  # Need more bullish signals to overcome high risk
                sell_threshold = 1  # Lower threshold for selling in high risk
            elif risk_rating == "Medium":
                buy_threshold = 2
                sell_threshold = 2
            else:  # Low risk
                buy_threshold = 1  # More willing to buy in low risk
                sell_threshold = 3  # Need more bearish signals to sell in low risk
                
            # Make decision
            if bullish_signals >= buy_threshold and bullish_signals > bearish_signals:
                decision = "BUY"
                explanation = f"Bullish signals ({bullish_signals}) exceed the threshold ({buy_threshold}) for {risk_rating} risk."
            elif bearish_signals >= sell_threshold and bearish_signals > bullish_signals:
                decision = "SELL"
                explanation = f"Bearish signals ({bearish_signals}) exceed the threshold ({sell_threshold}) for {risk_rating} risk."
            else:
                decision = "HOLD"
                explanation = f"Not enough clear signals for action at {risk_rating} risk level."
            
            # Confidence calculation
            # Base confidence on signal strength and risk level
            signal_diff = abs(bullish_signals - bearish_signals)
            signal_confidence = min(0.3 + (signal_diff * 0.1), 0.7)  # 0.3-0.7 based on signal strength
            
            # Adjust confidence based on risk (higher risk = lower confidence)
            risk_adjustment = 1.0 - (risk_score * 0.3)  # 0.7-1.0 based on risk
            
            # Combine confidences
            combined_confidence = signal_confidence * risk_adjustment
            
            # Recommended actions based on decision
            recommended_actions = []
            
            if decision == "BUY":
                position_size = "small" if risk_rating == "High" else "moderate" if risk_rating == "Medium" else "standard"
                stop_loss = "5-8%" if risk_rating == "High" else "8-12%" if risk_rating == "Medium" else "10-15%"
                
                recommended_actions = [
                    {
                        "action": "buy",
                        "symbol": state.symbol,
                        "position_size": position_size,
                        "stop_loss": stop_loss,
                        "reason": explanation
                    }
                ]
            elif decision == "SELL":
                recommended_actions = [
                    {
                        "action": "sell",
                        "symbol": state.symbol,
                        "reason": explanation
                    }
                ]
            else:  # HOLD
                recommended_actions = [
                    {
                        "action": "hold",
                        "symbol": state.symbol,
                        "reason": explanation
                    }
                ]
            
            # Update state with decision
            state.decision = decision
            state.confidence = combined_confidence
            state.explanation = explanation
            state.recommended_actions = recommended_actions
            
        else:
            # Default decision if analysis data is missing
            state.decision = "HOLD"
            state.confidence = 0.3
            state.explanation = "Insufficient data for a confident decision."
            state.recommended_actions = [
                {
                    "action": "hold",
                    "symbol": state.symbol,
                    "reason": "Insufficient data"
                }
            ]
        
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
        
        try:
            # Run the workflow
            if self.verbose > 0:
                print(f"Processing request: {request}")
            
            final_state = self.workflow.invoke(initial_state)
            
            # Extract market analysis text from the output
            market_analysis = final_state.agent_outputs.get("market_analysis", {})
            analysis_text = market_analysis.get("response", "No market analysis available.")
            
            # Extract risk assessment text from the output
            risk_assessment = final_state.agent_outputs.get("risk_assessment", {})
            risk_text = risk_assessment.get("response", "No risk assessment available.")
            
            # Prepare the result
            result = {
                "request": request,
                "symbol": final_state.symbol,
                "date_range": {
                    "start_date": final_state.start_date,
                    "end_date": final_state.end_date
                } if final_state.start_date and final_state.end_date else None,
                "decision": final_state.decision,
                "confidence": final_state.confidence,
                "explanation": final_state.explanation,
                "recommended_actions": final_state.recommended_actions if final_state.recommended_actions else [],
                "analysis": analysis_text,
                "risk_assessment": risk_text
            }
            
            # Include additional data if available
            if final_state.analysis_data:
                result["analysis_data"] = final_state.analysis_data
            
            if final_state.risk_assessment:
                result["risk_data"] = final_state.risk_assessment
            
            return result
        except Exception as e:
            # Handle any errors
            if self.verbose > 0:
                print(f"Error processing request: {e}")
            
            return {
                "request": request,
                "error": f"Error in trading agent system: {str(e)}"
            }
    
    def get_available_agents(self) -> List[str]:
        """Get list of available agent names"""
        return list(self.agents.keys()) 