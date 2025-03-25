"""
Portfolio Management Agent for the multi-agent trading system.
"""
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from .base_agent import BaseAgent, AgentInput, AgentOutput
from .portfolio_optimizer import PortfolioOptimizer
from src.data import DataManager

class PortfolioManagementAgent(BaseAgent):
    """
    Agent specialized in portfolio management and asset allocation.
    
    This agent can:
    1. Optimize portfolio allocations based on risk/reward profiles
    2. Generate rebalancing recommendations
    3. Provide position sizing strategies based on risk metrics
    4. Monitor portfolio diversification and concentration risk
    5. Perform multi-asset portfolio optimization using Modern Portfolio Theory
    6. Incorporate sentiment analysis into portfolio optimization
    """
    
    def __init__(self, data_manager: DataManager, verbose: int = 0):
        """
        Initialize the portfolio management agent.
        
        Args:
            data_manager (DataManager): Data manager for accessing market data
            verbose (int): Verbosity level (0: silent, 1: normal, 2: detailed)
        """
        super().__init__(
            name="Portfolio Management Agent",
            description="Optimizes portfolio allocations and generates rebalancing recommendations",
            verbose=verbose
        )
        self.data_manager = data_manager
        
        # Default risk tolerance levels
        self.risk_levels = {
            "conservative": {"max_position_size": 0.05, "max_sector_exposure": 0.20, "volatility_target": 0.10},
            "moderate": {"max_position_size": 0.08, "max_sector_exposure": 0.30, "volatility_target": 0.15},
            "aggressive": {"max_position_size": 0.12, "max_sector_exposure": 0.40, "volatility_target": 0.25}
        }
        
        # Initialize portfolio optimizer with default risk-free rate
        self.portfolio_optimizer = PortfolioOptimizer(risk_free_rate=0.02, verbose=verbose)
    
    def process(self, input_data: AgentInput) -> AgentOutput:
        """
        Process the input and generate portfolio recommendations.
        
        Args:
            input_data (AgentInput): Input data containing request and context
            
        Returns:
            AgentOutput: Portfolio recommendations
        """
        if self.verbose > 0:
            print(f"Processing portfolio management request: {input_data.request}")
        
        # Check if this is a portfolio optimization request
        request_lower = input_data.request.lower()
        if any(keyword in request_lower for keyword in [
            "optimize portfolio", "portfolio optimization", "efficient frontier", 
            "multi-asset", "multi asset", "sharpe ratio", "modern portfolio theory"
        ]):
            return self.process_portfolio_optimization_request(input_data)
        
        # Otherwise continue with original portfolio management logic
        # Extract portfolio from context if available
        portfolio = None
        if input_data.context and "portfolio" in input_data.context:
            portfolio = input_data.context["portfolio"]
        
        if not portfolio:
            return AgentOutput(
                response="I need portfolio information to provide recommendations.",
                confidence=0.0
            )
        
        # Extract risk assessment from context if available
        risk_assessment = None
        if input_data.context and "risk_assessment" in input_data.context:
            risk_assessment = input_data.context["risk_assessment"]
            
        # Extract market analysis from context if available
        market_analysis = None
        if input_data.context and "market_analysis" in input_data.context:
            market_analysis = input_data.context["market_analysis"]
            
        # Determine user's risk tolerance (extract from request or use default)
        risk_tolerance = self._extract_risk_tolerance(input_data.request)
        if not risk_tolerance and input_data.context and "risk_tolerance" in input_data.context:
            risk_tolerance = input_data.context["risk_tolerance"]
        
        if not risk_tolerance:
            risk_tolerance = "moderate"  # Default risk tolerance
            
        # Extract rebalancing frequency from context or request
        rebalance_frequency = self._extract_rebalance_frequency(input_data.request)
        if not rebalance_frequency and input_data.context and "rebalance_frequency" in input_data.context:
            rebalance_frequency = input_data.context["rebalance_frequency"]
            
        if not rebalance_frequency:
            rebalance_frequency = "monthly"  # Default rebalancing frequency
        
        # Calculate current portfolio metrics
        portfolio_metrics = self._calculate_portfolio_metrics(portfolio)
        
        # Generate portfolio assessment
        assessment = self._generate_portfolio_assessment(portfolio_metrics, risk_tolerance)
        
        # Analyze portfolio against risk tolerance
        risk_analysis = self._analyze_portfolio_risk(portfolio_metrics, risk_tolerance, risk_assessment)
        
        # Generate rebalancing recommendations
        rebalance_recommendations = self._generate_rebalance_recommendations(
            portfolio, portfolio_metrics, risk_tolerance, market_analysis, risk_assessment
        )
        
        # Combine all recommendations
        response = f"{assessment}\n\n{risk_analysis}\n\n{rebalance_recommendations}"
        
        # Prepare data output
        output_data = {
            "portfolio_metrics": portfolio_metrics,
            "risk_tolerance": risk_tolerance,
            "rebalance_frequency": rebalance_frequency,
            "recommendations": self._format_recommendations_as_dict(rebalance_recommendations)
        }
        
        # Return the portfolio recommendations
        return AgentOutput(
            response=response,
            data=output_data,
            confidence=0.85
        )
    
    def _extract_risk_tolerance(self, request: str) -> Optional[str]:
        """
        Extract risk tolerance from the user request.
        
        Args:
            request (str): User request
            
        Returns:
            str or None: Extracted risk tolerance or None if not found
        """
        request_lower = request.lower()
        
        # Check for risk tolerance keywords
        if "conservative" in request_lower:
            return "conservative"
        elif "aggressive" in request_lower:
            return "aggressive"
        elif "moderate" in request_lower or "balanced" in request_lower:
            return "moderate"
        
        return None
    
    def _extract_rebalance_frequency(self, request: str) -> Optional[str]:
        """
        Extract rebalancing frequency from the user request.
        
        Args:
            request (str): User request
            
        Returns:
            str or None: Extracted rebalancing frequency or None if not found
        """
        request_lower = request.lower()
        
        # Check for frequency keywords
        if "daily" in request_lower:
            return "daily"
        elif "weekly" in request_lower:
            return "weekly"
        elif "monthly" in request_lower:
            return "monthly"
        elif "quarterly" in request_lower:
            return "quarterly"
        elif "annually" in request_lower or "yearly" in request_lower:
            return "annually"
        
        return None
    
    def _calculate_portfolio_metrics(self, portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate portfolio metrics from portfolio data.
        
        Args:
            portfolio (Dict[str, Any]): Portfolio data
            
        Returns:
            Dict[str, Any]: Portfolio metrics
        """
        # Extract basic portfolio data
        total_value = portfolio.get("total_value", 0)
        positions = portfolio.get("positions", [])
        
        # Calculate allocation percentages
        allocations = {}
        for position in positions:
            symbol = position.get("symbol", "Unknown")
            quantity = position.get("quantity", 0)
            current_price = position.get("current_price", 0)
            
            # If current price not available, try to get it from data manager
            if current_price == 0:
                try:
                    market_data = self.data_manager.get_market_data(symbol=symbol, limit=1)
                    if market_data is not None and len(market_data) > 0:
                        current_price = market_data['Close'].iloc[-1]
                except:
                    pass
            
            position_value = quantity * current_price
            allocation_pct = (position_value / total_value * 100) if total_value > 0 else 0
            
            allocations[symbol] = {
                "value": position_value,
                "percentage": allocation_pct,
                "quantity": quantity,
                "current_price": current_price
            }
        
        # Group by sector if available
        sector_allocations = {}
        for position in positions:
            symbol = position.get("symbol", "Unknown")
            sector = position.get("sector", "Unknown")
            
            if symbol in allocations:
                if sector not in sector_allocations:
                    sector_allocations[sector] = 0
                
                sector_allocations[sector] += allocations[symbol]["percentage"]
        
        # Calculate diversification metrics
        num_positions = len(positions)
        max_allocation = max(pos["percentage"] for pos in allocations.values()) if allocations else 0
        concentration_ratio = max_allocation / 100  # Higher means more concentrated
        
        # Calculate effective number of positions (1/Herfindahl-Hirschman Index)
        hhi = sum((pos["percentage"]/100)**2 for pos in allocations.values()) if allocations else 0
        effective_positions = 1/hhi if hhi > 0 else 0
        
        # Calculate recent portfolio performance if available
        portfolio_performance = {
            "1d_return": portfolio.get("1d_return", 0),
            "1w_return": portfolio.get("1w_return", 0),
            "1m_return": portfolio.get("1m_return", 0),
            "3m_return": portfolio.get("3m_return", 0),
            "ytd_return": portfolio.get("ytd_return", 0),
            "1y_return": portfolio.get("1y_return", 0)
        }
        
        # Calculate volatility and Sharpe ratio if historical data is available
        volatility = portfolio.get("volatility", 0)
        sharpe_ratio = portfolio.get("sharpe_ratio", 0)
        
        # Compile portfolio metrics
        return {
            "total_value": total_value,
            "num_positions": num_positions,
            "allocations": allocations,
            "sector_allocations": sector_allocations,
            "max_allocation": max_allocation,
            "concentration_ratio": concentration_ratio,
            "effective_positions": effective_positions,
            "performance": portfolio_performance,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio
        }
    
    def _generate_portfolio_assessment(self, portfolio_metrics: Dict[str, Any], risk_tolerance: str) -> str:
        """
        Generate a human-readable portfolio assessment.
        
        Args:
            portfolio_metrics (Dict[str, Any]): Portfolio metrics
            risk_tolerance (str): Risk tolerance level
            
        Returns:
            str: Portfolio assessment text
        """
        total_value = portfolio_metrics.get("total_value", 0)
        num_positions = portfolio_metrics.get("num_positions", 0)
        max_allocation = portfolio_metrics.get("max_allocation", 0)
        concentration_ratio = portfolio_metrics.get("concentration_ratio", 0)
        effective_positions = portfolio_metrics.get("effective_positions", 0)
        
        # Get allocations and sort by percentage
        allocations = portfolio_metrics.get("allocations", {})
        sorted_allocations = sorted(
            [(symbol, data) for symbol, data in allocations.items()],
            key=lambda x: x[1]["percentage"],
            reverse=True
        )
        
        # Get sector allocations and sort
        sector_allocations = portfolio_metrics.get("sector_allocations", {})
        sorted_sectors = sorted(
            [(sector, allocation) for sector, allocation in sector_allocations.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Evaluate diversification
        diversification_rating = "Poor"
        if effective_positions >= 10:
            diversification_rating = "Excellent"
        elif effective_positions >= 7:
            diversification_rating = "Good"
        elif effective_positions >= 4:
            diversification_rating = "Fair"
        
        # Evaluate concentration
        concentration_rating = "High"
        if concentration_ratio <= 0.1:
            concentration_rating = "Low"
        elif concentration_ratio <= 0.2:
            concentration_rating = "Moderate"
        
        # Build the assessment text
        assessment = f"""
Portfolio Assessment:

Overview:
- Total Value: ${total_value:,.2f}
- Number of Positions: {num_positions}
- Effective Diversification: {effective_positions:.1f} positions ({diversification_rating})
- Concentration Risk: {concentration_rating} (max allocation: {max_allocation:.1f}%)

Top Holdings:"""
        
        # Add top 5 holdings or all if less than 5
        for i, (symbol, data) in enumerate(sorted_allocations[:5]):
            assessment += f"\n- {symbol}: ${data['value']:,.2f} ({data['percentage']:.1f}%)"
        
        if len(sorted_allocations) > 5:
            assessment += f"\n- Others: {len(sorted_allocations) - 5} positions"
        
        # Add sector breakdown if available
        if sector_allocations:
            assessment += "\n\nSector Allocation:"
            for sector, allocation in sorted_sectors[:5]:
                assessment += f"\n- {sector}: {allocation:.1f}%"
            
            if len(sorted_sectors) > 5:
                assessment += f"\n- Others: {len(sorted_sectors) - 5} sectors"
        
        # Add performance data if available
        performance = portfolio_metrics.get("performance", {})
        if performance:
            assessment += "\n\nPerformance:"
            if performance.get("1m_return") is not None:
                assessment += f"\n- 1 Month: {performance.get('1m_return', 0):.2f}%"
            if performance.get("3m_return") is not None:
                assessment += f"\n- 3 Months: {performance.get('3m_return', 0):.2f}%"
            if performance.get("ytd_return") is not None:
                assessment += f"\n- Year-to-Date: {performance.get('ytd_return', 0):.2f}%"
            if performance.get("1y_return") is not None:
                assessment += f"\n- 1 Year: {performance.get('1y_return', 0):.2f}%"
        
        return assessment
    
    def _analyze_portfolio_risk(self, portfolio_metrics: Dict[str, Any], 
                               risk_tolerance: str, risk_assessment: Optional[Dict[str, Any]]) -> str:
        """
        Analyze portfolio risk against risk tolerance.
        
        Args:
            portfolio_metrics (Dict[str, Any]): Portfolio metrics
            risk_tolerance (str): Risk tolerance level
            risk_assessment (Dict[str, Any], optional): Risk assessment data
            
        Returns:
            str: Risk analysis text
        """
        # Get risk parameters for the specified tolerance level
        risk_params = self.risk_levels.get(risk_tolerance, self.risk_levels["moderate"])
        
        # Extract metrics
        max_allocation = portfolio_metrics.get("max_allocation", 0) / 100
        allocations = portfolio_metrics.get("allocations", {})
        sector_allocations = portfolio_metrics.get("sector_allocations", {})
        volatility = portfolio_metrics.get("volatility", 0)
        
        # Check position size violations
        position_violations = []
        for symbol, data in allocations.items():
            if data["percentage"] / 100 > risk_params["max_position_size"]:
                position_violations.append((symbol, data["percentage"]))
        
        # Check sector exposure violations
        sector_violations = []
        for sector, allocation in sector_allocations.items():
            if allocation / 100 > risk_params["max_sector_exposure"]:
                sector_violations.append((sector, allocation))
        
        # Check overall volatility
        volatility_status = "Unknown"
        if volatility > 0:
            if volatility > risk_params["volatility_target"] * 1.2:
                volatility_status = "Too High"
            elif volatility < risk_params["volatility_target"] * 0.8:
                volatility_status = "Too Low"
            else:
                volatility_status = "Appropriate"
        
        # Build risk analysis text
        analysis = f"""
Risk Analysis ({risk_tolerance.capitalize()} Profile):

Position Size Risk:"""
        
        if position_violations:
            analysis += "\nThe following positions exceed your maximum position size limit:"
            for symbol, percentage in position_violations:
                over_by = percentage - (risk_params["max_position_size"] * 100)
                analysis += f"\n- {symbol}: {percentage:.1f}% (over by {over_by:.1f}%)"
        else:
            analysis += "\nAll positions are within your maximum position size limit."
        
        analysis += "\n\nSector Concentration Risk:"
        if sector_violations:
            analysis += "\nThe following sectors exceed your maximum sector exposure limit:"
            for sector, percentage in sector_violations:
                over_by = percentage - (risk_params["max_sector_exposure"] * 100)
                analysis += f"\n- {sector}: {percentage:.1f}% (over by {over_by:.1f}%)"
        else:
            analysis += "\nAll sector allocations are within your maximum sector exposure limit."
        
        if volatility > 0:
            analysis += f"\n\nPortfolio Volatility: {volatility:.2f}% ({volatility_status} for your risk profile)"
            
        # Include specific risk assessment info if available
        if risk_assessment:
            analysis += "\n\nRisk Metrics for Individual Holdings:"
            
            for symbol, data in allocations.items():
                if symbol in risk_assessment:
                    symbol_risk = risk_assessment[symbol]
                    risk_rating = symbol_risk.get("risk_rating", "Unknown")
                    analysis += f"\n- {symbol}: {risk_rating} Risk"
        
        return analysis
    
    def _generate_rebalance_recommendations(self, portfolio: Dict[str, Any], 
                                           portfolio_metrics: Dict[str, Any],
                                           risk_tolerance: str,
                                           market_analysis: Optional[Dict[str, Any]],
                                           risk_assessment: Optional[Dict[str, Any]]) -> str:
        """
        Generate rebalancing recommendations.
        
        Args:
            portfolio (Dict[str, Any]): Portfolio data
            portfolio_metrics (Dict[str, Any]): Portfolio metrics
            risk_tolerance (str): Risk tolerance level
            market_analysis (Dict[str, Any], optional): Market analysis data
            risk_assessment (Dict[str, Any], optional): Risk assessment data
            
        Returns:
            str: Rebalancing recommendations text
        """
        # Get risk parameters for the specified tolerance level
        risk_params = self.risk_levels.get(risk_tolerance, self.risk_levels["moderate"])
        
        # Extract metrics
        total_value = portfolio_metrics.get("total_value", 0)
        allocations = portfolio_metrics.get("allocations", {})
        
        # Determine target allocations
        target_allocations = self._calculate_target_allocations(
            allocations, risk_tolerance, market_analysis, risk_assessment
        )
        
        # Calculate rebalancing trades
        rebalance_trades = []
        for symbol, target_pct in target_allocations.items():
            current_pct = allocations.get(symbol, {}).get("percentage", 0)
            current_value = allocations.get(symbol, {}).get("value", 0)
            current_price = allocations.get(symbol, {}).get("current_price", 0)
            
            target_value = total_value * (target_pct / 100)
            value_diff = target_value - current_value
            
            # Only suggest trades if difference is significant (>1% of portfolio or >$1000)
            if abs(value_diff) > max(0.01 * total_value, 1000):
                if current_price > 0:
                    shares_diff = int(value_diff / current_price)
                    
                    if shares_diff > 0:
                        rebalance_trades.append((symbol, "BUY", shares_diff, value_diff, target_pct, current_pct))
                    else:
                        rebalance_trades.append((symbol, "SELL", abs(shares_diff), value_diff, target_pct, current_pct))
        
        # Build recommendations text
        recommendations = f"""
Rebalancing Recommendations ({risk_tolerance.capitalize()} Profile):
"""
        
        if not rebalance_trades:
            recommendations += "\nYour portfolio is properly balanced. No trades recommended at this time."
        else:
            recommendations += "\nThe following trades are recommended to optimize your portfolio:"
            
            # Sort trades by absolute value difference (largest first)
            rebalance_trades.sort(key=lambda x: abs(x[3]), reverse=True)
            
            for symbol, action, shares, value_diff, target_pct, current_pct in rebalance_trades:
                diff = target_pct - current_pct
                recommendations += f"\n- {action} {shares} shares of {symbol} (${abs(value_diff):,.2f})"
                recommendations += f"\n  Current: {current_pct:.1f}% → Target: {target_pct:.1f}% ({diff:+.1f}%)"
        
        # Add position sizing guidance
        recommendations += "\n\nPosition Sizing Guidelines:"
        recommendations += f"\n- Conservative positions: up to {(risk_params['max_position_size'] * 100) / 2:.1f}% of portfolio"
        recommendations += f"\n- Standard positions: up to {(risk_params['max_position_size'] * 100) * 0.75:.1f}% of portfolio"
        recommendations += f"\n- Maximum position size: {risk_params['max_position_size'] * 100:.1f}% of portfolio"
        
        return recommendations
    
    def _calculate_target_allocations(self, current_allocations: Dict[str, Dict[str, Any]],
                                     risk_tolerance: str,
                                     market_analysis: Optional[Dict[str, Any]],
                                     risk_assessment: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate target allocations based on risk profile and market conditions.
        
        Args:
            current_allocations (Dict[str, Dict[str, Any]]): Current allocations
            risk_tolerance (str): Risk tolerance level
            market_analysis (Dict[str, Any], optional): Market analysis data
            risk_assessment (Dict[str, Any], optional): Risk assessment data
            
        Returns:
            Dict[str, float]: Target allocations as percentages
        """
        # Start with current allocations as baseline
        target_allocations = {symbol: data["percentage"] for symbol, data in current_allocations.items()}
        
        # Adjust based on risk assessment if available
        if risk_assessment:
            for symbol in target_allocations:
                if symbol in risk_assessment:
                    risk_rating = risk_assessment[symbol].get("risk_rating", "Medium")
                    risk_score = risk_assessment[symbol].get("risk_score", 0.5)
                    
                    # Adjust allocation based on risk rating and tolerance
                    if risk_tolerance == "conservative":
                        if risk_rating == "High":
                            target_allocations[symbol] *= 0.7  # Reduce high-risk positions for conservative portfolios
                        elif risk_rating == "Low":
                            target_allocations[symbol] *= 1.2  # Increase low-risk positions
                    elif risk_tolerance == "aggressive":
                        if risk_rating == "High":
                            target_allocations[symbol] *= 1.2  # Increase high-risk positions for aggressive portfolios
                        elif risk_rating == "Low":
                            target_allocations[symbol] *= 0.9  # Reduce low-risk positions
        
        # Adjust based on market analysis if available
        if market_analysis:
            for symbol in target_allocations:
                if symbol in market_analysis:
                    decision = market_analysis[symbol].get("decision", "HOLD")
                    confidence = market_analysis[symbol].get("confidence", 0.5)
                    
                    # Adjust allocation based on market prediction
                    if decision == "BUY":
                        adjustment = 1.0 + (0.3 * confidence)  # Up to 30% increase based on confidence
                        target_allocations[symbol] *= adjustment
                    elif decision == "SELL":
                        adjustment = 1.0 - (0.5 * confidence)  # Up to 50% decrease based on confidence
                        target_allocations[symbol] *= adjustment
        
        # Normalize allocations to sum to 100%
        total = sum(target_allocations.values())
        if total > 0:
            target_allocations = {symbol: (pct / total) * 100 for symbol, pct in target_allocations.items()}
        
        return target_allocations
    
    def _format_recommendations_as_dict(self, recommendations_text: str) -> List[Dict[str, Any]]:
        """
        Parse recommendations text into structured data.
        
        Args:
            recommendations_text (str): Recommendations text
            
        Returns:
            List[Dict[str, Any]]: Structured recommendations
        """
        import re
        
        recommendations = []
        
        # Extract trade recommendations using regex
        trade_pattern = r'- (BUY|SELL) (\d+) shares of ([A-Z]+) \(\$([0-9,.]+)\)'
        matches = re.findall(trade_pattern, recommendations_text)
        
        for match in matches:
            action, shares, symbol, value = match
            recommendations.append({
                "action": action.lower(),
                "symbol": symbol,
                "shares": int(shares),
                "value": float(value.replace(',', ''))
            })
        
        return recommendations
    
    def optimize_multi_asset_portfolio(self, 
                                      symbols: List[str], 
                                      start_date: str, 
                                      end_date: str,
                                      risk_tolerance: str = "moderate",
                                      optimization_objective: str = "sharpe",
                                      include_sentiment: bool = True) -> Dict[str, Any]:
        """
        Optimize a multi-asset portfolio using Modern Portfolio Theory.
        
        Args:
            symbols (List[str]): List of symbols to include in the portfolio
            start_date (str): Start date for historical data
            end_date (str): End date for historical data
            risk_tolerance (str): Risk tolerance level
            optimization_objective (str): Optimization objective
            include_sentiment (bool): Whether to incorporate sentiment data
            
        Returns:
            Dict[str, Any]: Optimized portfolio details
        """
        if self.verbose > 0:
            print(f"Optimizing portfolio for {len(symbols)} assets with {optimization_objective} objective")
        
        # Map risk tolerance to constraints
        risk_params = self.risk_levels.get(risk_tolerance, self.risk_levels["moderate"])
        
        # Fetch historical price data for all symbols
        price_data = {}
        for symbol in symbols:
            try:
                data = self.data_manager.get_market_data(symbol, start_date, end_date, include_indicators=False)
                if data is not None and not data.empty:
                    price_data[symbol] = data['Close']
            except Exception as e:
                if self.verbose > 0:
                    print(f"Error fetching data for {symbol}: {e}")
        
        # Check if we have enough data
        if len(price_data) < 2:
            if self.verbose > 0:
                print("Not enough valid symbols with data for optimization")
            return {
                "error": "Insufficient data for portfolio optimization",
                "weights": {symbol: 1.0 / len(symbols) for symbol in symbols}  # Equal weights as fallback
            }
        
        # Create dataframe with all price data
        prices_df = pd.DataFrame(price_data)
        
        # Calculate returns
        returns_df = prices_df.pct_change().dropna()
        
        # Get sentiment data if requested
        sentiment_scores = None
        if include_sentiment:
            sentiment_scores = self._get_sentiment_scores(symbols, start_date, end_date)
        
        # Set up constraints based on risk tolerance
        constraints = {
            'max_position': risk_params['max_position_size'],
            'target_volatility': risk_params['volatility_target']
        }
        
        # If we have sector data, add sector constraints
        if self._has_sector_information(symbols):
            sector_constraints = self._create_sector_constraints(symbols, risk_params['max_sector_exposure'])
            constraints['sector_constraints'] = sector_constraints
        
        # Set optimization objective based on risk tolerance
        if optimization_objective == 'auto':
            if risk_tolerance == 'conservative':
                optimization_objective = 'min_volatility'
            elif risk_tolerance == 'aggressive':
                optimization_objective = 'return'
            else:
                optimization_objective = 'sharpe'
        
        # Perform optimization
        optimization_result = self.portfolio_optimizer.optimize_portfolio(
            returns_df, 
            constraints=constraints,
            objective=optimization_objective,
            sentiment_scores=sentiment_scores
        )
        
        # Generate efficient frontier for visualization
        try:
            frontier_data = self.portfolio_optimizer.generate_efficient_frontier(returns_df)
            optimization_result['efficient_frontier'] = frontier_data
        except Exception as e:
            if self.verbose > 0:
                print(f"Could not generate efficient frontier: {e}")
        
        # Add risk parity portfolio for comparison
        try:
            risk_parity = self.portfolio_optimizer.calculate_risk_parity_portfolio(returns_df)
            optimization_result['risk_parity'] = risk_parity
        except Exception as e:
            if self.verbose > 0:
                print(f"Could not calculate risk parity portfolio: {e}")
        
        # Format results for better readability
        formatted_weights = {
            symbol: round(weight * 100, 2) 
            for symbol, weight in optimization_result['weights'].items()
        }
        optimization_result['formatted_weights'] = formatted_weights
        
        return optimization_result
    
    def _get_sentiment_scores(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, float]:
        """
        Get sentiment scores for the given symbols.
        
        Args:
            symbols (List[str]): List of symbols
            start_date (str): Start date
            end_date (str): End date
            
        Returns:
            Dict[str, float]: Dictionary mapping symbols to sentiment scores
        """
        sentiment_scores = {}
        
        for symbol in symbols:
            # Try to get sentiment data from sentiment analysis agent context
            sentiment_data = None
            
            # Check if we have any recent memory with sentiment data for this symbol
            relevant_memories = self.get_relevant_memories(f"sentiment {symbol}", limit=1)
            
            if relevant_memories and "data" in relevant_memories[0]:
                data = relevant_memories[0]["data"]
                if isinstance(data, dict) and symbol in data:
                    sentiment_data = data[symbol]
            
            # If not in memory, try to fetch from data manager
            if sentiment_data is None:
                try:
                    # First try news sentiment
                    news_sentiment = self.data_manager.get_sentiment_data(symbol, start_date, end_date)
                    if news_sentiment is not None and not news_sentiment.empty:
                        # Calculate average sentiment score
                        avg_sentiment = news_sentiment['Sentiment_Score'].mean()
                        sentiment_scores[symbol] = avg_sentiment
                        continue
                    
                    # Then try social sentiment
                    social_sentiment = self.data_manager.get_social_sentiment(symbol, start_date, end_date)
                    if social_sentiment is not None and not social_sentiment.empty:
                        # Calculate average sentiment score
                        avg_sentiment = social_sentiment['Sentiment_Score'].mean()
                        sentiment_scores[symbol] = avg_sentiment
                except Exception as e:
                    if self.verbose > 1:
                        print(f"Error getting sentiment data for {symbol}: {e}")
        
        return sentiment_scores
    
    def _has_sector_information(self, symbols: List[str]) -> bool:
        """
        Check if sector information is available for the symbols.
        
        Args:
            symbols (List[str]): List of symbols
            
        Returns:
            bool: True if sector information is available
        """
        # In a real implementation, we would check if we have sector data
        # For now, return False for simplicity
        return False
    
    def _create_sector_constraints(self, symbols: List[str], max_sector_exposure: float) -> Dict[str, Tuple[List[int], float]]:
        """
        Create sector constraints for portfolio optimization.
        
        Args:
            symbols (List[str]): List of symbols
            max_sector_exposure (float): Maximum exposure to any single sector
            
        Returns:
            Dict[str, Tuple[List[int], float]]: Sector constraints
        """
        # In a real implementation, we would fetch sector data and create constraints
        # For now, return an empty dict
        return {}
    
    def format_multi_asset_optimization_response(self, optimization_result: Dict[str, Any], 
                                               risk_tolerance: str) -> str:
        """
        Format multi-asset portfolio optimization results into human-readable text.
        
        Args:
            optimization_result (Dict[str, Any]): Optimization results
            risk_tolerance (str): Risk tolerance level
            
        Returns:
            str: Formatted response
        """
        if "error" in optimization_result:
            return f"Error in portfolio optimization: {optimization_result['error']}"
        
        # Extract key metrics
        expected_return = optimization_result.get("return", 0) * 100  # Convert to percentage
        volatility = optimization_result.get("volatility", 0) * 100  # Convert to percentage
        sharpe_ratio = optimization_result.get("sharpe_ratio", 0)
        weights = optimization_result.get("formatted_weights", {})
        
        # Sort weights by allocation (descending)
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        
        # Build response
        response = f"""
## Optimized Portfolio ({risk_tolerance.capitalize()} Risk Profile)

### Key Metrics
- Expected Annual Return: {expected_return:.2f}%
- Expected Annual Volatility: {volatility:.2f}%
- Sharpe Ratio: {sharpe_ratio:.2f}
- Optimization Method: {optimization_result.get('objective', 'Sharpe Ratio Maximization').capitalize()}

### Recommended Asset Allocation
"""
        
        # Add allocation table
        for symbol, allocation in sorted_weights:
            response += f"- {symbol}: {allocation:.2f}%\n"
        
        # Add portfolio characteristics based on risk tolerance
        response += f"\n### Portfolio Characteristics\n"
        
        if risk_tolerance == "conservative":
            response += "- Focus on capital preservation with moderate growth\n"
            response += "- Lower volatility with more stable returns\n"
            response += "- Greater diversification across assets\n"
        elif risk_tolerance == "aggressive":
            response += "- Focus on maximizing returns with higher risk tolerance\n"
            response += "- Higher volatility with potential for greater returns\n"
            response += "- More concentrated positions in high-growth assets\n"
        else:  # moderate
            response += "- Balanced approach between growth and capital preservation\n"
            response += "- Moderate volatility with good return potential\n"
            response += "- Diversified but with targeted allocation to growth assets\n"
        
        # Add note about sentiment data if used
        if "sentiment_adjusted" in optimization_result and optimization_result["sentiment_adjusted"]:
            response += "\n### Sentiment Analysis Integration\n"
            response += "- This portfolio incorporates recent sentiment data for the assets\n"
            response += "- Allocations are adjusted based on positive/negative sentiment\n"
            
            # Add specific sentiment adjustments if available
            if "sentiment_scores" in optimization_result:
                response += "- Sentiment Adjustments:\n"
                for symbol, score in optimization_result["sentiment_scores"].items():
                    sentiment_text = "positive" if score > 0.1 else "negative" if score < -0.1 else "neutral"
                    response += f"  - {symbol}: {sentiment_text.capitalize()} ({score:.2f})\n"
        
        return response
        
    def process_portfolio_optimization_request(self, input_data: AgentInput) -> AgentOutput:
        """
        Process portfolio optimization request.
        
        Args:
            input_data (AgentInput): Input data containing request and context
            
        Returns:
            AgentOutput: Portfolio optimization results
        """
        if self.verbose > 0:
            print(f"Processing portfolio optimization request: {input_data.request}")
        
        # Extract parameters from request and context
        symbols = self._extract_symbols_from_request(input_data)
        if not symbols and input_data.context and "portfolio" in input_data.context:
            # Extract symbols from portfolio
            portfolio = input_data.context["portfolio"]
            symbols = [position.get("symbol") for position in portfolio.get("positions", [])]
        
        if not symbols:
            return AgentOutput(
                response="Please provide the symbols you want to include in your portfolio.",
                confidence=0.0
            )
        
        # Extract date range
        start_date = None
        end_date = None
        if input_data.context and "date_range" in input_data.context:
            date_range = input_data.context["date_range"]
            start_date = date_range.get("start_date")
            end_date = date_range.get("end_date")
        
        # If dates not provided, default to 1 year of data
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            # Default to 1 year of historical data
            start_dt = datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=365)
            start_date = start_dt.strftime("%Y-%m-%d")
        
        # Extract risk tolerance
        risk_tolerance = self._extract_risk_tolerance(input_data.request)
        if not risk_tolerance and input_data.context and "risk_tolerance" in input_data.context:
            risk_tolerance = input_data.context["risk_tolerance"]
        if not risk_tolerance:
            risk_tolerance = "moderate"  # Default
        
        # Extract optimization objective
        optimization_objective = "auto"  # Default
        request_lower = input_data.request.lower()
        if "sharpe" in request_lower:
            optimization_objective = "sharpe"
        elif "minimum volatility" in request_lower or "min volatility" in request_lower:
            optimization_objective = "min_volatility"
        elif "maximum return" in request_lower or "max return" in request_lower:
            optimization_objective = "return"
        elif "risk parity" in request_lower:
            optimization_objective = "risk_parity"
        
        # Check if sentiment data should be included
        include_sentiment = "sentiment" in request_lower or "news" in request_lower
        
        # Run optimization
        optimization_result = self.optimize_multi_asset_portfolio(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            risk_tolerance=risk_tolerance,
            optimization_objective=optimization_objective,
            include_sentiment=include_sentiment
        )
        
        # If sentiment was used, mark it in the result
        if include_sentiment:
            optimization_result["sentiment_adjusted"] = True
        
        # Generate response
        response = self.format_multi_asset_optimization_response(
            optimization_result=optimization_result,
            risk_tolerance=risk_tolerance
        )
        
        # Store the optimization in memory
        self.add_to_memory({
            "input": input_data.request,
            "symbols": symbols,
            "risk_tolerance": risk_tolerance,
            "optimization_result": optimization_result
        })
        
        return AgentOutput(
            response=response,
            data=optimization_result,
            confidence=0.9
        )
    
    def _extract_symbols_from_request(self, input_data: AgentInput) -> List[str]:
        """
        Extract stock symbols from the request.
        
        Args:
            input_data (AgentInput): Input data
            
        Returns:
            List[str]: List of stock symbols
        """
        # Extract from context if available
        if input_data.context and "symbols" in input_data.context:
            return input_data.context["symbols"]
        
        # Otherwise extract from request text
        import re
        
        # Look for ticker symbols (common format: 1-5 uppercase letters)
        request = input_data.request
        matches = re.findall(r'\b[A-Z]{1,5}\b', request)
        
        # Filter out common words that might be mistaken for tickers
        common_words = {"A", "I", "CEO", "CFO", "USA", "GDP", "IPO", "AI", "ML"}
        filtered_matches = [m for m in matches if m not in common_words]
        
        return filtered_matches 