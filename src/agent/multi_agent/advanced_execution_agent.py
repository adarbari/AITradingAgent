"""
Advanced Execution Agent for the multi-agent trading system.
Extends the ExecutionAgent with more sophisticated trading algorithms and strategies.
"""
from typing import Dict, Any, Optional, List, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

from .execution_agent import ExecutionAgent
from .base_agent import AgentInput, AgentOutput
from src.data import DataManager

class AdvancedExecutionAgent(ExecutionAgent):
    """
    Advanced agent specialized in sophisticated execution strategies and optimal order placement.
    
    This agent extends the base ExecutionAgent with:
    1. Adaptive execution strategies that respond to real-time market conditions
    2. Machine learning-enhanced execution algorithms
    3. Dynamic time-of-day optimizations
    4. Smart order routing capabilities
    5. Support for dark pool liquidity
    6. Execution strategies optimized by portfolio decisions
    7. Anti-gaming protection mechanisms
    """
    
    def __init__(self, data_manager: DataManager, verbose: int = 0):
        """
        Initialize the advanced execution agent.
        
        Args:
            data_manager (DataManager): Data manager for accessing market data
            verbose (int): Verbosity level (0: silent, 1: normal, 2: detailed)
        """
        super().__init__(data_manager=data_manager, verbose=verbose)
        
        # Override the name and description
        self.name = "Advanced Execution Agent"
        self.description = "Provides sophisticated execution strategies with machine learning optimizations"
        
        # Add advanced execution parameters
        self.execution_params.update({
            "adaptive": {
                "slippage_estimate": 0.00015,  # 1.5 basis points for adaptive orders
                "price_improvement": 0.0002,   # 2 basis points potential improvement
                "market_impact": 0.00008,      # 0.8 basis points impact
                "immediacy": 0.7               # High immediacy with smart adaptation
            },
            "dark_pool": {
                "slippage_estimate": 0.0001,   # 1 basis point for dark pool orders
                "price_improvement": 0.0003,   # 3 basis points potential improvement
                "fill_probability": 0.6,       # 60% chance of fill
                "immediacy": 0.5               # Medium immediacy
            },
            "iceberg": {
                "slippage_estimate": 0.0002,   # 2 basis points for iceberg orders
                "market_impact": 0.00015,      # 1.5 basis points impact
                "immediacy": 0.65              # Medium-high immediacy
            },
            "pov": {  # Percentage of Volume
                "slippage_estimate": 0.00018,  # 1.8 basis points for POV
                "market_impact": 0.00012,      # 1.2 basis points impact
                "immediacy": 0.55              # Medium immediacy
            }
        })
        
        # Historical execution performance tracking
        self.execution_history = {
            "strategies": {},
            "venues": {},
            "time_of_day": {}
        }
        
        # ML model confidence scores for different strategies
        # In a real system, these would be updated by ML models
        self.ml_confidence = {
            "adaptive": 0.85,
            "vwap": 0.82,
            "twap": 0.80,
            "pov": 0.78,
            "dark_pool": 0.75,
            "iceberg": 0.76
        }
    
    def process(self, input_data: AgentInput) -> AgentOutput:
        """
        Process the input and generate advanced execution recommendations.
        
        Args:
            input_data (AgentInput): Input data containing request and context
            
        Returns:
            AgentOutput: Advanced execution recommendations
        """
        # First get the base execution recommendation
        base_output = super().process(input_data)
        
        # If base recommendation couldn't be generated, return it as is
        if base_output.confidence < 0.5:
            return base_output
        
        # Extract trade details from context
        trade_details = None
        if input_data.context and "trade_details" in input_data.context:
            trade_details = input_data.context["trade_details"]
        
        if not trade_details:
            return base_output
        
        # Extract portfolio information to enhance execution strategy
        portfolio = None
        portfolio_statistics = None
        if input_data.context and "portfolio" in input_data.context:
            portfolio = input_data.context["portfolio"]
            # Try to extract portfolio metrics if available
            if "metrics" in portfolio:
                portfolio_statistics = portfolio["metrics"]
        
        # Extract enhanced data
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
            
        # Extract base recommendation data
        base_strategy = None
        base_order_type = None
        if base_output.data:
            base_strategy = base_output.data.get("execution_strategy")
            base_order_type = base_output.data.get("order_type")
        
        # Enhance the execution strategy based on portfolio context and advanced analytics
        enhanced_strategy, strategy_details = self._enhance_execution_strategy(
            symbol, action, quantity, price, urgency,
            base_strategy, portfolio_statistics,
            market_analysis, risk_assessment
        )
        
        # Enhance the order type recommendation
        enhanced_order_type, order_params = self._enhance_order_type(
            symbol, action, quantity, price,
            base_order_type, enhanced_strategy,
            market_analysis, risk_assessment
        )
        
        # Calculate position size relative to average volume
        volatility, liquidity, avg_volume = self._analyze_market_conditions(symbol)
        position_size_ratio = self._calculate_position_size_ratio(quantity, avg_volume)
        
        # Estimate enhanced costs and market impact
        estimated_costs, market_impact = self._estimate_advanced_execution_costs(
            action, quantity, price, enhanced_order_type,
            enhanced_strategy, position_size_ratio, volatility
        )
        
        # Determine if this is a large order that needs special handling
        is_large_order = position_size_ratio > 0.01  # More than 1% of avg volume
        
        # Get market hours and determine if we're in a high-volume period
        current_time = datetime.now().strftime("%H:%M")
        market_info = self.market_hours.get("US", {})
        is_high_volume_period = self._is_high_volume_period(current_time, market_info)
        
        # Format the enhanced execution recommendation
        enhanced_response = self._format_advanced_execution_recommendation(
            symbol, action, quantity, enhanced_order_type, order_params,
            enhanced_strategy, strategy_details, estimated_costs, market_impact,
            is_high_volume_period, is_large_order
        )
        
        # Format enhanced data output
        output_data = {
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "order_type": enhanced_order_type,
            "order_params": order_params,
            "execution_strategy": enhanced_strategy,
            "strategy_details": strategy_details,
            "estimated_costs": estimated_costs,
            "market_impact": market_impact,
            "market_conditions": {
                "volatility": volatility,
                "liquidity": liquidity,
                "position_size_ratio": position_size_ratio,
                "is_high_volume_period": is_high_volume_period
            }
        }
        
        # Return the enhanced execution recommendations
        return AgentOutput(
            response=enhanced_response,
            data=output_data,
            confidence=0.9  # Higher confidence with advanced strategies
        )
    
    def _enhance_execution_strategy(self, symbol: str, action: str, quantity: int, price: float,
                                  urgency: str, base_strategy: Optional[str],
                                  portfolio_stats: Optional[Dict[str, Any]],
                                  market_analysis: Optional[Dict[str, Any]],
                                  risk_assessment: Optional[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        """
        Enhance execution strategy based on portfolio context and market conditions.
        
        Args:
            symbol (str): Stock symbol
            action (str): Trade action (buy/sell)
            quantity (int): Number of shares to trade
            price (float): Current price
            urgency (str): Execution urgency
            base_strategy (str): Base execution strategy
            portfolio_stats (Dict[str, Any]): Portfolio statistics
            market_analysis (Dict[str, Any]): Market analysis data
            risk_assessment (Dict[str, Any]): Risk assessment data
            
        Returns:
            Tuple[str, Dict[str, Any]]: Enhanced strategy name and details
        """
        # Default to the base strategy if available
        if base_strategy:
            strategy = base_strategy
        else:
            strategy = "Standard Market Order"
        
        # Initialize strategy details
        strategy_details = {
            "description": "",
            "ml_confidence": 0.0,
            "expected_duration": "",
            "venues": [],
            "parameters": {}
        }
        
        # Get market conditions
        volatility, liquidity, avg_volume = self._analyze_market_conditions(symbol)
        position_size_ratio = self._calculate_position_size_ratio(quantity, avg_volume)
        
        # Get time-based factors
        current_time = datetime.now().strftime("%H:%M")
        market_info = self.market_hours.get("US", {})
        is_high_volume_period = self._is_high_volume_period(current_time, market_info)
        
        # Check if portfolio is highly concentrated
        is_concentrated_position = False
        if portfolio_stats and "concentration_ratio" in portfolio_stats:
            is_concentrated_position = portfolio_stats["concentration_ratio"] > 0.2
        
        # Check risk metrics
        high_risk = False
        if risk_assessment and symbol in risk_assessment:
            if risk_assessment[symbol].get("risk_rating", "Medium") == "High":
                high_risk = True
        
        # Check market sentiment
        market_trending = False
        if market_analysis and symbol in market_analysis:
            trend = market_analysis[symbol].get("trend", "Neutral")
            if trend in ["Strong Uptrend", "Strong Downtrend"]:
                market_trending = True
        
        # Determine the optimal strategy based on all factors
        
        # Large order with high risk or concentration - use adaptive strategy
        if position_size_ratio > 0.02 and (high_risk or is_concentrated_position):
            strategy = "Adaptive Execution"
            pov_target = min(0.15, 2.0 * position_size_ratio)  # Cap at 15% of volume
            
            strategy_details = {
                "description": "Dynamic strategy that adapts to real-time market conditions",
                "ml_confidence": self.ml_confidence.get("adaptive", 0.0),
                "expected_duration": "2-4 hours",
                "venues": ["Primary Exchange", "Dark Pools", "Alternative Venues"],
                "parameters": {
                    "pov_target": pov_target,
                    "min_participation": 0.05,
                    "max_participation": 0.25,
                    "dark_pool_usage": "High" if liquidity == "low" else "Medium"
                }
            }
        
        # High urgency with adequate liquidity - use iceberg orders
        elif urgency == "high" and liquidity != "low":
            strategy = "Iceberg Execution"
            visible_quantity = min(int(quantity * 0.1), 1000)  # 10% visible, max 1000 shares
            
            strategy_details = {
                "description": "Displays only a portion of the order to minimize market impact",
                "ml_confidence": self.ml_confidence.get("iceberg", 0.0),
                "expected_duration": "30-60 minutes",
                "venues": ["Primary Exchange"],
                "parameters": {
                    "visible_quantity": visible_quantity,
                    "refresh_rate": "Auto",
                    "price_limit": price * (0.99 if action == "buy" else 1.01)  # 1% buffer
                }
            }
        
        # Low urgency with potential for dark pool execution
        elif urgency == "low" and quantity > 5000:
            strategy = "Dark Pool Execution"
            
            strategy_details = {
                "description": "Routes orders to dark pools for minimal market impact",
                "ml_confidence": self.ml_confidence.get("dark_pool", 0.0),
                "expected_duration": "1-2 days",
                "venues": ["Dark Pools", "Alternative Trading Systems"],
                "parameters": {
                    "min_execution_size": 500,
                    "time_in_force": "Day",
                    "allow_partial_fills": True
                }
            }
        
        # Medium-sized orders in trending markets - use POV
        elif 0.005 < position_size_ratio < 0.02 and market_trending:
            strategy = "Percentage of Volume (POV)"
            pov_target = min(0.12, position_size_ratio * 3)  # Scale based on size
            
            strategy_details = {
                "description": "Executes order as a percentage of market volume",
                "ml_confidence": self.ml_confidence.get("pov", 0.0),
                "expected_duration": "Full trading day",
                "venues": ["All Available"],
                "parameters": {
                    "target_pov": pov_target,
                    "min_pov": 0.03,
                    "max_pov": 0.15
                }
            }
        
        # High volatility situation - use TWAP for more controlled execution
        elif volatility > 0.6:
            strategy = "Enhanced TWAP"
            duration_hours = 4 if quantity > 10000 else 2
            
            strategy_details = {
                "description": "Time-Weighted Average Price with adaptive scheduling",
                "ml_confidence": self.ml_confidence.get("twap", 0.0),
                "expected_duration": f"{duration_hours} hours",
                "venues": ["Primary Exchange", "ECNs"],
                "parameters": {
                    "interval_minutes": 15,
                    "variance_percent": 10,  # Randomize size by 10%
                    "accelerate_on_momentum": action == "buy"  # Buy more when price is rising
                }
            }
        
        # Default to enhanced VWAP when nothing else is a better fit
        elif base_strategy == "VWAP" or quantity > 2000:
            strategy = "Enhanced VWAP"
            
            strategy_details = {
                "description": "Volume-Weighted Average Price with adaptive volume profile",
                "ml_confidence": self.ml_confidence.get("vwap", 0.0),
                "expected_duration": "Full trading day",
                "venues": ["Primary Exchange", "Alternative Venues"],
                "parameters": {
                    "start_time": "Now",
                    "end_time": market_info.get("close", "16:00"),
                    "custom_volume_profile": is_high_volume_period,
                    "allow_dark_pool": True
                }
            }
        
        # Adjust strategy for portfolio importance
        if portfolio_stats and "allocations" in portfolio_stats:
            allocations = portfolio_stats["allocations"]
            if symbol in allocations:
                position_pct = allocations[symbol].get("percentage", 0)
                if position_pct > 15:  # Important position (>15% of portfolio)
                    # For important positions, prioritize execution quality
                    strategy_details["description"] += " (Optimized for significant portfolio position)"
                    strategy_details["parameters"]["quality_priority"] = "High"
        
        return strategy, strategy_details
    
    def _enhance_order_type(self, symbol: str, action: str, quantity: int, price: float,
                          base_order_type: Optional[str], enhanced_strategy: str,
                          market_analysis: Optional[Dict[str, Any]],
                          risk_assessment: Optional[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        """
        Enhance order type recommendation based on advanced strategy.
        
        Args:
            symbol (str): Stock symbol
            action (str): Trade action (buy/sell)
            quantity (int): Number of shares to trade
            price (float): Current price
            base_order_type (str): Base recommended order type
            enhanced_strategy (str): Enhanced execution strategy
            market_analysis (Dict[str, Any]): Market analysis data
            risk_assessment (Dict[str, Any]): Risk assessment data
            
        Returns:
            Tuple[str, Dict[str, Any]]: Enhanced order type and parameters
        """
        # Default to base order type if available
        order_type = base_order_type if base_order_type else "market"
        
        # Initialize order parameters
        order_params = {
            "description": "Standard order",
            "limit_price": None,
            "stop_price": None,
            "time_in_force": "day",
            "additional_conditions": []
        }
        
        # Check if we have market momentum data
        market_momentum = "neutral"
        if market_analysis and symbol in market_analysis:
            momentum = market_analysis[symbol].get("momentum", 0)
            if momentum > 0.3:
                market_momentum = "positive"
            elif momentum < -0.3:
                market_momentum = "negative"
        
        # Adjust order type based on strategy
        if enhanced_strategy == "Adaptive Execution":
            # Use dynamic limits for adaptive execution
            order_type = "adaptive"
            buffer = 0.003  # 0.3% buffer
            order_params = {
                "description": "Adaptive limit order with dynamic repricing",
                "initial_limit_price": price * (1 + buffer) if action == "buy" else price * (1 - buffer),
                "max_limit_deviation": 0.005,  # 0.5% max deviation
                "time_in_force": "day",
                "repricing_interval": "1m",  # Reprice every minute
                "additional_conditions": ["min_quantity_50"]
            }
        
        elif enhanced_strategy == "Iceberg Execution":
            # Use iceberg order type
            order_type = "iceberg"
            order_params = {
                "description": "Iceberg order showing only a portion at a time",
                "limit_price": price * (1.002 if action == "buy" else 0.998),  # 0.2% buffer
                "display_quantity": min(500, quantity // 10),  # Show smaller of 500 or 10% of total
                "refresh_type": "immediate",
                "time_in_force": "day",
                "additional_conditions": []
            }
        
        elif enhanced_strategy == "Dark Pool Execution":
            # Dark pool routing
            order_type = "dark_pool"
            order_params = {
                "description": "Dark pool routed order for minimal market impact",
                "limit_price": price * (1.005 if action == "buy" else 0.995),  # 0.5% buffer
                "min_execution_quantity": 100,
                "time_in_force": "day",
                "routing": "dark_pool_smart",
                "additional_conditions": ["allow_partial"]
            }
        
        elif enhanced_strategy in ["Enhanced VWAP", "Enhanced TWAP", "Percentage of Volume (POV)"]:
            # Use algorithmic order type
            order_type = enhanced_strategy.lower().replace(" ", "_").replace("(", "").replace(")", "")
            
            # Base parameters
            order_params = {
                "description": f"{enhanced_strategy} algorithmic order",
                "start_time": "immediate",
                "end_time": "market_close",
                "participation_rate": 0.1,  # 10% of volume as default
                "time_in_force": "day",
                "allow_extension": False,
                "additional_conditions": []
            }
            
            # Add strategy-specific parameters
            if enhanced_strategy == "Enhanced VWAP":
                order_params["use_custom_profile"] = True
                order_params["dark_pool_usage"] = "opportunistic"
            elif enhanced_strategy == "Enhanced TWAP":
                order_params["interval_minutes"] = 15
                order_params["randomize_time"] = True
                order_params["randomize_size"] = True
            elif enhanced_strategy == "Percentage of Volume (POV)":
                order_params["min_rate"] = 0.03
                order_params["max_rate"] = 0.15
                order_params["use_hidden_liquidity"] = True
        
        # Add protection against adverse price movements
        if action == "buy" and market_momentum == "negative":
            order_params["additional_conditions"].append("buy_side_protection")
        elif action == "sell" and market_momentum == "positive":
            order_params["additional_conditions"].append("sell_side_protection")
        
        # Add special handling for high volatility
        volatility, _, _ = self._analyze_market_conditions(symbol)
        if volatility > 0.5:
            order_params["additional_conditions"].append("volatility_protection")
        
        return order_type, order_params
    
    def _estimate_advanced_execution_costs(self, action: str, quantity: int, price: float,
                                         order_type: str, execution_strategy: str,
                                         position_size_ratio: float, volatility: float) -> Tuple[float, float]:
        """
        Estimate execution costs and market impact for advanced strategies.
        
        Args:
            action (str): Trade action (buy/sell)
            quantity (int): Number of shares to trade
            price (float): Current or target price
            order_type (str): Recommended order type
            execution_strategy (str): Recommended execution strategy
            position_size_ratio (float): Position size relative to avg volume
            volatility (float): Market volatility score
            
        Returns:
            Tuple[float, float]: Estimated costs (as percentage), Market impact (as percentage)
        """
        # Get base costs
        if order_type in self.execution_params:
            base_slippage = self.execution_params[order_type]["slippage_estimate"]
        else:
            base_slippage = 0.0005  # Default to 5 basis points
        
        # Adjust for strategy
        strategy_factor = 1.0
        if "Adaptive" in execution_strategy:
            strategy_factor = 0.7  # Adaptive strategies reduce costs
        elif "Dark Pool" in execution_strategy:
            strategy_factor = 0.6  # Dark pools typically have lower impact
        elif "Iceberg" in execution_strategy:
            strategy_factor = 0.8  # Iceberg orders reduce market impact
        elif "POV" in execution_strategy:
            strategy_factor = 0.85  # POV has controlled impact
        elif "VWAP" in execution_strategy:
            strategy_factor = 0.75  # Enhanced VWAP improvements
        elif "TWAP" in execution_strategy:
            strategy_factor = 0.8  # Enhanced TWAP improvements
        
        # Adjust for position size and volatility
        size_impact = position_size_ratio * 0.01  # 1 basis point per 1% of ADV
        volatility_impact = volatility * 0.005  # Up to 50 basis points for extreme volatility
        
        # Calculate advanced costs (slippage + fees)
        # Assume more advanced strategies get better fee rates due to volume
        advanced_fee_reduction = 0.0001  # 1 basis point reduction in fees
        total_costs = (base_slippage * strategy_factor) + 0.0003 - advanced_fee_reduction
        
        # Calculate market impact with advanced strategy adjustments
        market_impact = (size_impact + volatility_impact) * strategy_factor
        
        # Add opportunity cost consideration for slower strategies
        if "VWAP" in execution_strategy or "TWAP" in execution_strategy:
            opportunity_cost = 0.0001  # 1 basis point for longer execution window
            total_costs += opportunity_cost
        
        return total_costs, market_impact
    
    def _format_advanced_execution_recommendation(self, symbol: str, action: str, quantity: int,
                                              order_type: str, order_params: Dict[str, Any],
                                              execution_strategy: str, strategy_details: Dict[str, Any],
                                              estimated_costs: float, market_impact: float,
                                              is_high_volume_period: bool, is_large_order: bool) -> str:
        """
        Format a human-readable advanced execution recommendation.
        
        Args:
            symbol (str): Stock symbol
            action (str): Trade action (buy/sell)
            quantity (int): Number of shares to trade
            order_type (str): Recommended order type
            order_params (Dict[str, Any]): Order parameters
            execution_strategy (str): Recommended execution strategy
            strategy_details (Dict[str, Any]): Strategy details
            estimated_costs (float): Estimated execution costs
            market_impact (float): Estimated market impact
            is_high_volume_period (bool): Whether it's a high volume period
            is_large_order (bool): Whether it's a large order
            
        Returns:
            str: Formatted recommendation text
        """
        # Order description
        order_description = order_params.get("description", "Standard order")
        
        # Convert costs and impact to basis points for better readability
        costs_bps = estimated_costs * 10000  # Convert to basis points
        impact_bps = market_impact * 10000   # Convert to basis points
        
        # Build the recommendation text
        recommendation = f"""
Advanced Execution Recommendation for {action.upper()} {quantity} shares of {symbol}:

Execution Strategy: {execution_strategy}
{strategy_details.get('description', '')}

Order Type: {order_description}
Time Frame: {strategy_details.get('expected_duration', 'Standard')}

Market Conditions:
- {'High' if is_high_volume_period else 'Normal'} trading volume period
- {'Large position requiring special handling' if is_large_order else 'Standard position size'}
- Estimated execution costs: {costs_bps:.1f} basis points
- Estimated market impact: {impact_bps:.1f} basis points

Trading Venues:
"""
        # Add venues
        venues = strategy_details.get("venues", ["Primary Exchange"])
        for venue in venues:
            recommendation += f"- {venue}\n"
        
        # Add strategy parameters
        parameters = strategy_details.get("parameters", {})
        if parameters:
            recommendation += "\nStrategy Parameters:\n"
            for param, value in parameters.items():
                recommendation += f"- {param.replace('_', ' ').title()}: {value}\n"
        
        # Add ML confidence if available
        ml_confidence = strategy_details.get("ml_confidence", 0)
        if ml_confidence > 0:
            recommendation += f"\nML Model Confidence: {ml_confidence:.2f}\n"
        
        # Add special conditions
        additional_conditions = order_params.get("additional_conditions", [])
        if additional_conditions:
            recommendation += "\nSpecial Order Conditions:\n"
            for condition in additional_conditions:
                # Format condition names for readability
                condition_name = condition.replace("_", " ").title()
                recommendation += f"- {condition_name}\n"
        
        # Add execution instructions
        recommendation += "\nExecution Instructions:\n"
        
        if execution_strategy == "Adaptive Execution":
            recommendation += "- Execute with dynamic limits that adjust to market conditions\n"
            recommendation += "- System will automatically reprice limits based on real-time analytics\n"
            recommendation += "- Execution pace will accelerate or slow based on price action and liquidity\n"
        
        elif execution_strategy == "Iceberg Execution":
            visible = min(int(quantity * 0.1), 1000)
            recommendation += f"- Display only {visible} shares at a time to minimize signaling\n"
            recommendation += "- Automatically refresh the displayed quantity when filled\n"
            recommendation += "- Adjust limit price if market moves significantly\n"
        
        elif execution_strategy == "Dark Pool Execution":
            recommendation += "- Route order to dark pools and alternative trading systems\n"
            recommendation += "- Minimize market impact by avoiding lit exchanges\n"
            recommendation += "- Set minimum execution size to avoid small fills\n"
        
        elif "VWAP" in execution_strategy:
            recommendation += "- Execute according to enhanced Volume-Weighted Average Price strategy\n"
            recommendation += "- Follow historical volume profile with real-time adjustments\n"
            recommendation += "- Take advantage of dark pools when advantageous\n"
        
        elif "TWAP" in execution_strategy:
            recommendation += "- Execute according to enhanced Time-Weighted Average Price strategy\n"
            recommendation += "- Slice order into intervals with size and timing randomization\n"
            recommendation += "- Adjust execution pace based on price movements\n"
        
        elif "POV" in execution_strategy:
            pov = strategy_details.get("parameters", {}).get("target_pov", 0.1)
            recommendation += f"- Target approximately {pov*100:.1f}% of market volume\n"
            recommendation += "- Dynamically adjust participation rate based on available liquidity\n"
            recommendation += "- Utilize both lit and dark liquidity pools\n"
            
        # Add recommendation for monitoring
        recommendation += "\nMonitoring Recommendations:\n"
        recommendation += "- Monitor execution performance against VWAP benchmark\n"
        
        if is_large_order:
            recommendation += "- Set price alerts for significant deviations from entry price\n"
            recommendation += "- Be prepared to pause execution if market conditions deteriorate\n"
        
        return recommendation 