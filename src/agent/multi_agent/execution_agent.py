"""
Execution Agent for the multi-agent trading system.
"""
from typing import Dict, Any, Optional, List, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from .base_agent import BaseAgent, AgentInput, AgentOutput
from src.data import DataManager

class ExecutionAgent(BaseAgent):
    """
    Agent specialized in trade execution strategies and order placement.
    
    This agent can:
    1. Determine optimal execution timing based on market conditions
    2. Recommend order types (market, limit, stop, etc.)
    3. Suggest execution strategies (e.g., VWAP, TWAP, Implementation Shortfall)
    4. Calculate optimal order sizes for large positions
    5. Estimate execution costs and market impact
    """
    
    def __init__(self, data_manager: DataManager, verbose: int = 0):
        """
        Initialize the execution agent.
        
        Args:
            data_manager (DataManager): Data manager for accessing market data
            verbose (int): Verbosity level (0: silent, 1: normal, 2: detailed)
        """
        super().__init__(
            name="Execution Agent",
            description="Optimizes trade execution strategies and order placement",
            verbose=verbose
        )
        self.data_manager = data_manager
        
        # Default execution parameters
        self.execution_params = {
            "market": {
                "slippage_estimate": 0.0005,  # 5 basis points for market orders
                "price_improvement": 0.0,      # No price improvement for market orders
                "immediacy": 1.0               # Maximum immediacy
            },
            "limit": {
                "slippage_estimate": 0.0,      # No slippage if filled
                "price_improvement": 0.001,    # 10 basis points potential improvement
                "fill_probability": 0.8,       # 80% chance of fill for reasonable limits
                "immediacy": 0.6               # Medium immediacy
            },
            "stop": {
                "slippage_estimate": 0.001,    # 10 basis points for stop orders
                "price_improvement": 0.0,      # No price improvement
                "immediacy": 0.9               # High immediacy once triggered
            },
            "vwap": {
                "slippage_estimate": 0.0002,   # 2 basis points for VWAP
                "market_impact": 0.0001,       # 1 basis point impact
                "immediacy": 0.4               # Lower immediacy (throughout the day)
            },
            "twap": {
                "slippage_estimate": 0.0003,   # 3 basis points for TWAP
                "market_impact": 0.0002,       # 2 basis points impact
                "immediacy": 0.3               # Lower immediacy (throughout the day)
            }
        }
        
        # Trading hours for major markets
        self.market_hours = {
            "US": {
                "open": "09:30",
                "close": "16:00",
                "timezone": "America/New_York",
                "high_volume_periods": ["09:30-10:30", "15:30-16:00"]
            },
            "EU": {
                "open": "09:00",
                "close": "17:30",
                "timezone": "Europe/London",
                "high_volume_periods": ["09:00-10:00", "16:30-17:30"]
            }
        }
    
    def process(self, input_data: AgentInput) -> AgentOutput:
        """
        Process the input and generate execution recommendations.
        
        Args:
            input_data (AgentInput): Input data containing request and context
            
        Returns:
            AgentOutput: Execution recommendations
        """
        if self.verbose > 0:
            print(f"Processing execution request: {input_data.request}")
        
        # Extract trade details from context
        trade_details = None
        if input_data.context and "trade_details" in input_data.context:
            trade_details = input_data.context["trade_details"]
        
        if not trade_details:
            return AgentOutput(
                response="I need trade details to provide execution recommendations.",
                confidence=0.0
            )
        
        # Extract relevant data for execution decision
        symbol = trade_details.get("symbol")
        action = trade_details.get("action", "").lower()
        quantity = trade_details.get("quantity", 0)
        price = trade_details.get("price", 0.0)
        urgency = trade_details.get("urgency", "normal")
        
        # Get market analysis if available
        market_analysis = None
        if input_data.context and "market_analysis" in input_data.context:
            market_analysis = input_data.context["market_analysis"]
        
        # Get risk assessment if available
        risk_assessment = None
        if input_data.context and "risk_assessment" in input_data.context:
            risk_assessment = input_data.context["risk_assessment"]
            
        # Get portfolio data if available
        portfolio = None
        if input_data.context and "portfolio" in input_data.context:
            portfolio = input_data.context["portfolio"]
            
        # Analyze volatility and liquidity
        volatility, liquidity, avg_volume = self._analyze_market_conditions(symbol)
        
        # Calculate position size relative to average volume
        position_size_ratio = self._calculate_position_size_ratio(quantity, avg_volume)
        
        # Determine if this is a large order that needs special handling
        is_large_order = position_size_ratio > 0.01  # More than 1% of avg volume
        
        # Get market hours and determine if we're in a high-volume period
        current_time = datetime.now().strftime("%H:%M")
        market_info = self.market_hours.get("US", {})
        is_high_volume_period = self._is_high_volume_period(current_time, market_info)
        
        # Get execution strategy recommendation
        if is_large_order:
            execution_strategy = self._recommend_large_order_strategy(
                action, quantity, volatility, liquidity, urgency
            )
        else:
            execution_strategy = self._recommend_standard_execution(
                action, quantity, volatility, liquidity, urgency
            )
        
        # Get order type recommendation
        order_type, order_params = self._recommend_order_type(
            action, price, volatility, urgency, is_high_volume_period, market_analysis
        )
        
        # Estimate costs and market impact
        estimated_costs, market_impact = self._estimate_execution_costs(
            action, quantity, price, order_type, execution_strategy, position_size_ratio, volatility
        )
        
        # Format the execution recommendation response
        response = self._format_execution_recommendation(
            symbol, action, quantity, order_type, order_params,
            execution_strategy, estimated_costs, market_impact, is_high_volume_period
        )
        
        # Format data output
        output_data = {
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "order_type": order_type,
            "order_params": order_params,
            "execution_strategy": execution_strategy,
            "estimated_costs": estimated_costs,
            "market_impact": market_impact,
            "market_conditions": {
                "volatility": volatility,
                "liquidity": liquidity,
                "position_size_ratio": position_size_ratio,
                "is_high_volume_period": is_high_volume_period
            }
        }
        
        # Return the execution recommendations
        return AgentOutput(
            response=response,
            data=output_data,
            confidence=0.85
        )
    
    def _analyze_market_conditions(self, symbol: str) -> Tuple[float, str, float]:
        """
        Analyze market conditions for a given symbol.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            Tuple[float, str, float]: 
                - Volatility score (0-1)
                - Liquidity assessment ("high", "medium", "low")
                - Average daily volume
        """
        try:
            # Get recent market data
            market_data = self.data_manager.get_market_data(
                symbol=symbol, 
                limit=20  # Last 20 trading days
            )
            
            if market_data is None or market_data.empty:
                return 0.5, "medium", 1000000  # Default values if no data
            
            # Calculate volatility (normalized)
            if 'Close' in market_data.columns:
                returns = market_data['Close'].pct_change().dropna()
                volatility = min(returns.std() * 15.87, 1.0)  # Annualized and capped at 1.0
            else:
                volatility = 0.5  # Default
            
            # Assess liquidity based on volume
            avg_volume = market_data['Volume'].mean() if 'Volume' in market_data.columns else 1000000
            
            if avg_volume > 5000000:
                liquidity = "high"
            elif avg_volume > 1000000:
                liquidity = "medium"
            else:
                liquidity = "low"
            
            return volatility, liquidity, avg_volume
            
        except Exception as e:
            if self.verbose > 0:
                print(f"Error analyzing market conditions: {e}")
            return 0.5, "medium", 1000000  # Default values on error
    
    def _calculate_position_size_ratio(self, quantity: int, avg_volume: float) -> float:
        """
        Calculate the position size as a ratio of average daily volume.
        
        Args:
            quantity (int): Number of shares to trade
            avg_volume (float): Average daily trading volume
            
        Returns:
            float: Position size ratio (0-1)
        """
        if avg_volume <= 0:
            return 1.0  # Default to maximum if no volume data
        
        return min(quantity / avg_volume, 1.0)
    
    def _is_high_volume_period(self, current_time: str, market_info: Dict[str, Any]) -> bool:
        """
        Determine if the current time is within a high-volume trading period.
        
        Args:
            current_time (str): Current time in HH:MM format
            market_info (Dict[str, Any]): Market hours information
            
        Returns:
            bool: True if in high volume period, False otherwise
        """
        high_volume_periods = market_info.get("high_volume_periods", [])
        
        for period in high_volume_periods:
            start, end = period.split('-')
            if start <= current_time <= end:
                return True
                
        return False
    
    def _recommend_large_order_strategy(self, action: str, quantity: int, 
                                      volatility: float, liquidity: str, urgency: str) -> str:
        """
        Recommend an execution strategy for large orders.
        
        Args:
            action (str): Trade action (buy/sell)
            quantity (int): Number of shares
            volatility (float): Market volatility score
            liquidity (str): Market liquidity assessment
            urgency (str): Execution urgency
            
        Returns:
            str: Recommended execution strategy
        """
        # For large orders, consider advanced strategies
        if urgency == "high":
            # High urgency - faster execution but potentially higher impact
            if volatility > 0.7:
                return "Implementation Shortfall"  # Balances urgency with impact
            else:
                return "TWAP (2 hours)"  # Relatively fast but spread over time
        elif urgency == "low":
            # Low urgency - minimize impact
            if liquidity == "high":
                return "VWAP (Full Day)"
            else:
                return "VWAP (2 Days)"  # Spread very thin for illiquid stocks
        else:
            # Normal urgency
            if volatility > 0.7:
                return "TWAP (3 hours)"
            else:
                return "VWAP (Half Day)"
    
    def _recommend_standard_execution(self, action: str, quantity: int,
                                    volatility: float, liquidity: str, urgency: str) -> str:
        """
        Recommend an execution strategy for standard-sized orders.
        
        Args:
            action (str): Trade action (buy/sell)
            quantity (int): Number of shares
            volatility (float): Market volatility score
            liquidity (str): Market liquidity assessment
            urgency (str): Execution urgency
            
        Returns:
            str: Recommended execution strategy
        """
        # For standard orders, simpler strategies
        if urgency == "high":
            return "Immediate Execution"
        elif urgency == "low":
            if volatility > 0.7:
                return "Staged Execution (3 stages)"
            else:
                return "Passive Limit Order"
        else:
            # Normal urgency
            if liquidity == "low":
                return "Staged Execution (2 stages)"
            else:
                return "Standard Market Order"
    
    def _recommend_order_type(self, action: str, price: float, volatility: float,
                            urgency: str, is_high_volume_period: bool,
                            market_analysis: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Recommend order type and parameters.
        
        Args:
            action (str): Trade action (buy/sell)
            price (float): Current price or target price
            volatility (float): Market volatility score
            urgency (str): Execution urgency
            is_high_volume_period (bool): Whether it's a high volume period
            market_analysis (Dict[str, Any], optional): Market analysis data
            
        Returns:
            Tuple[str, Dict[str, Any]]: Order type and parameters
        """
        market_trend = "neutral"
        if market_analysis and "trend" in market_analysis:
            market_trend = market_analysis["trend"]
        
        # Parameters for the order
        params = {"base_price": price}
        
        if urgency == "high":
            # High urgency typically means market orders
            order_type = "market"
            
        elif volatility > 0.7:
            # High volatility - use limit orders with room
            order_type = "limit"
            if action == "buy":
                # For buys, set limit slightly above market to ensure fill
                buffer = price * 0.002  # 20 basis points
                params["limit_price"] = price + buffer
            else:  # sell
                # For sells, set limit slightly below market to ensure fill
                buffer = price * 0.002  # 20 basis points
                params["limit_price"] = price - buffer
                
        elif urgency == "low" and not is_high_volume_period:
            # Low urgency, not high volume - can be more passive
            order_type = "limit"
            if action == "buy":
                # For buys, try to get a better price
                improvement = price * 0.003  # 30 basis points
                params["limit_price"] = price - improvement
            else:  # sell
                # For sells, try to get a better price
                improvement = price * 0.003  # 30 basis points
                params["limit_price"] = price + improvement
                
        else:
            # Normal conditions - use market orders with potential limits
            if market_trend == "bullish" and action == "buy":
                # Bullish and buying - market might run away, use market order
                order_type = "market"
            elif market_trend == "bearish" and action == "sell":
                # Bearish and selling - market might drop, use market order
                order_type = "market"
            else:
                # Otherwise use a tight limit
                order_type = "limit"
                if action == "buy":
                    buffer = price * 0.001  # 10 basis points
                    params["limit_price"] = price + buffer
                else:  # sell
                    buffer = price * 0.001  # 10 basis points
                    params["limit_price"] = price - buffer
        
        return order_type, params
    
    def _estimate_execution_costs(self, action: str, quantity: int, price: float,
                                order_type: str, execution_strategy: str,
                                position_size_ratio: float, volatility: float) -> Tuple[float, float]:
        """
        Estimate execution costs and market impact.
        
        Args:
            action (str): Trade action (buy/sell)
            quantity (int): Number of shares
            price (float): Current or target price
            order_type (str): Recommended order type
            execution_strategy (str): Recommended execution strategy
            position_size_ratio (float): Position size relative to avg volume
            volatility (float): Market volatility score
            
        Returns:
            Tuple[float, float]: Estimated costs (as percentage), Market impact (as percentage)
        """
        # Base costs based on order type
        if order_type in self.execution_params:
            base_slippage = self.execution_params[order_type]["slippage_estimate"]
        else:
            base_slippage = 0.0005  # Default to 5 basis points
        
        # Adjust for strategy
        strategy_factor = 1.0
        if "VWAP" in execution_strategy:
            strategy_factor = 0.8  # VWAP typically reduces costs
        elif "TWAP" in execution_strategy:
            strategy_factor = 0.9  # TWAP also reduces but slightly less
        elif "Implementation Shortfall" in execution_strategy:
            strategy_factor = 1.1  # May increase costs slightly
        
        # Adjust for position size and volatility
        size_impact = position_size_ratio * 0.01  # 1 basis point per 1% of ADV
        volatility_impact = volatility * 0.005  # Up to 50 basis points for extreme volatility
        
        # Calculate total costs (slippage + fees)
        total_costs = (base_slippage * strategy_factor) + 0.0003  # Add 3 basis points for fees
        
        # Calculate market impact
        market_impact = (size_impact + volatility_impact) * strategy_factor
        
        return total_costs, market_impact
    
    def _format_execution_recommendation(self, symbol: str, action: str, quantity: int,
                                       order_type: str, order_params: Dict[str, Any],
                                       execution_strategy: str, estimated_costs: float,
                                       market_impact: float, is_high_volume_period: bool) -> str:
        """
        Format execution recommendations into readable text.
        
        Args:
            symbol (str): Stock symbol
            action (str): Trade action (buy/sell)
            quantity (int): Number of shares
            order_type (str): Recommended order type
            order_params (Dict[str, Any]): Order parameters
            execution_strategy (str): Recommended execution strategy
            estimated_costs (float): Estimated execution costs
            market_impact (float): Estimated market impact
            is_high_volume_period (bool): Whether it's a high volume period
            
        Returns:
            str: Formatted recommendation text
        """
        # Format order type description
        if order_type == "market":
            order_description = "Market Order (immediate execution at best available price)"
        elif order_type == "limit":
            limit_price = order_params.get("limit_price", order_params.get("base_price", 0))
            order_description = f"Limit Order at ${limit_price:.2f}"
        elif order_type == "stop":
            stop_price = order_params.get("stop_price", 0)
            order_description = f"Stop Order at ${stop_price:.2f}"
        else:
            order_description = f"{order_type.capitalize()} Order"
        
        # Format costs for display
        costs_bps = estimated_costs * 10000  # Convert to basis points
        impact_bps = market_impact * 10000  # Convert to basis points
        
        # Build the recommendation text
        recommendation = f"""
Execution Recommendation for {action.upper()} {quantity} shares of {symbol}:

Order Type: {order_description}
Execution Strategy: {execution_strategy}

Market Conditions:
- {'High' if is_high_volume_period else 'Normal'} trading volume period
- Estimated execution costs: {costs_bps:.1f} basis points
- Estimated market impact: {impact_bps:.1f} basis points

Execution Instructions:
"""

        # Add specific instructions based on strategy
        if "VWAP" in execution_strategy:
            recommendation += "- Execute trade according to Volume-Weighted Average Price strategy\n"
            recommendation += f"- Divide order into smaller child orders throughout the specified time period\n"
            recommendation += "- Target execution in line with historical volume patterns\n"
        elif "TWAP" in execution_strategy:
            recommendation += "- Execute trade according to Time-Weighted Average Price strategy\n"
            recommendation += f"- Divide order into equal-sized child orders over the specified time period\n"
        elif "Implementation Shortfall" in execution_strategy:
            recommendation += "- Execute with Implementation Shortfall algorithm\n"
            recommendation += "- Begin with larger initial execution followed by smaller tranches\n"
            recommendation += "- Optimize between market impact and opportunity cost\n"
        elif "Staged Execution" in execution_strategy:
            stages = int(execution_strategy.split("(")[1].split()[0])
            size_per_stage = quantity // stages
            recommendation += f"- Divide order into {stages} equal parts of approximately {size_per_stage} shares each\n"
            recommendation += "- Execute each part with a 30-minute interval between orders\n"
        elif execution_strategy == "Passive Limit Order":
            if action == "buy":
                recommendation += "- Place a passive limit order slightly below current market price\n"
                recommendation += "- Be prepared to adjust the limit price if market moves away\n"
            else:
                recommendation += "- Place a passive limit order slightly above current market price\n"
                recommendation += "- Be prepared to adjust the limit price if market moves away\n"
        else:
            recommendation += "- Execute standard order according to the recommended order type\n"

        # Add timing recommendations
        if is_high_volume_period:
            recommendation += "\nTiming: Current high-volume period is suitable for execution."
        else:
            recommendation += "\nTiming: Consider waiting for high-volume periods for better execution if urgency is low."
        
        return recommendation 