"""
Risk Assessment Agent for the multi-agent trading system.
"""
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from .base_agent import BaseAgent, AgentInput, AgentOutput
from src.data import DataManager

class RiskAssessmentAgent(BaseAgent):
    """
    Agent specialized in assessing trading risks.
    
    This agent can:
    1. Calculate market risk metrics (volatility, VaR, etc.)
    2. Evaluate position risk relative to portfolio
    3. Assess risk based on market conditions
    4. Make risk-adjusted recommendations
    """
    
    def __init__(self, data_manager: DataManager, verbose: int = 0):
        """
        Initialize the risk assessment agent.
        
        Args:
            data_manager (DataManager): Data manager for accessing market data
            verbose (int): Verbosity level (0: silent, 1: normal, 2: detailed)
        """
        super().__init__(
            name="Risk Assessment Agent",
            description="Evaluates trading risks based on market conditions and portfolio data",
            verbose=verbose
        )
        self.data_manager = data_manager
    
    def process(self, input_data: AgentInput) -> AgentOutput:
        """
        Process the input and generate a risk assessment.
        
        Args:
            input_data (AgentInput): Input data containing request and context
            
        Returns:
            AgentOutput: Risk assessment results
        """
        if self.verbose > 0:
            print(f"Processing risk assessment for: {input_data.request}")
        
        # Extract symbol from context or request
        symbol = None
        if input_data.context and "symbol" in input_data.context:
            symbol = input_data.context["symbol"]
        else:
            symbol = self._extract_symbol(input_data.request)
        
        if not symbol:
            return AgentOutput(
                response="I need a specific stock symbol to assess risk.",
                confidence=0.0
            )
        
        # Extract date range from context or request
        date_range = None
        if input_data.context and "date_range" in input_data.context:
            if isinstance(input_data.context["date_range"], dict):
                if "start_date" in input_data.context["date_range"] and "end_date" in input_data.context["date_range"]:
                    date_range = {
                        "start_date": input_data.context["date_range"]["start_date"],
                        "end_date": input_data.context["date_range"]["end_date"]
                    }
            else:
                date_range = self._extract_date_range(input_data.request)
        else:
            date_range = self._extract_date_range(input_data.request)
        
        # If no date range specified, use a default range (last 90 days)
        if not date_range:
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
            date_range = {"start_date": start_date, "end_date": end_date}
        
        # Get market data
        market_data = self.data_manager.get_market_data(
            symbol=symbol,
            start_date=date_range["start_date"],
            end_date=date_range["end_date"],
            include_indicators=True
        )
        
        if market_data is None or len(market_data) == 0:
            return AgentOutput(
                response=f"Couldn't retrieve market data for {symbol}.",
                confidence=0.0
            )
        
        # Calculate risk metrics
        risk_data = self._calculate_risk_metrics(market_data, symbol)
        
        # Generate risk assessment response
        response = self._generate_risk_assessment(risk_data)
        
        # Check if portfolio data exists in context to include position risk
        portfolio = input_data.context.get("portfolio") if input_data.context else None
        if portfolio:
            position_risk = self._calculate_position_risk(symbol, portfolio, risk_data)
            response += f"\n\n{position_risk}"
            risk_data["position_risk"] = position_risk
        
        # Return the risk assessment
        return AgentOutput(
            response=response,
            data=risk_data,
            confidence=0.8 if risk_data.get("confidence") is None else risk_data.get("confidence")
        )
    
    def _extract_symbol(self, text: str) -> Optional[str]:
        """
        Extract stock symbol from text.
        
        Args:
            text (str): Text to extract from
            
        Returns:
            Optional[str]: Extracted symbol or None
        """
        # Look for common patterns of stock symbols
        import re
        
        # Look for $SYMBOL pattern
        dollar_match = re.search(r'\$([A-Z]{1,5})', text)
        if dollar_match:
            return dollar_match.group(1)
        
        # Look for "SYMBOL stock" or "SYMBOL shares" patterns
        stock_match = re.search(r'([A-Z]{1,5})\s+(?:stock|shares|ticker)', text, re.IGNORECASE)
        if stock_match:
            return stock_match.group(1)
        
        # Look for standalone ticker symbols (less reliable)
        words = text.split()
        for word in words:
            if re.match(r'^[A-Z]{1,5}$', word) and word not in ['I', 'A', 'THE', 'FOR', 'OF']:
                return word
        
        return None
    
    def _extract_date_range(self, text: str) -> Optional[Dict[str, str]]:
        """
        Extract date range from text.
        
        Args:
            text (str): Text to extract from
            
        Returns:
            Optional[Dict[str, str]]: Dictionary with start_date and end_date keys
        """
        import re
        from datetime import datetime, timedelta
        
        # Look for explicit date range patterns like "from 2023-01-01 to 2023-12-31"
        date_pattern = r'from\s+(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4})\s+to\s+(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4})'
        date_match = re.search(date_pattern, text, re.IGNORECASE)
        
        if date_match:
            start_date_str = date_match.group(1)
            end_date_str = date_match.group(2)
            
            # Normalize date format
            start_date = self._normalize_date(start_date_str)
            end_date = self._normalize_date(end_date_str)
            
            return {"start_date": start_date, "end_date": end_date}
        
        # Look for relative date ranges like "last 30 days" or "past 6 months"
        relative_match = re.search(r'(?:last|past)\s+(\d+)\s+(day|days|week|weeks|month|months|year|years)', text, re.IGNORECASE)
        
        if relative_match:
            number = int(relative_match.group(1))
            unit = relative_match.group(2).lower()
            
            end_date = datetime.now()
            
            if unit in ['day', 'days']:
                start_date = end_date - timedelta(days=number)
            elif unit in ['week', 'weeks']:
                start_date = end_date - timedelta(weeks=number)
            elif unit in ['month', 'months']:
                start_date = end_date - timedelta(days=number*30)  # Approximate
            elif unit in ['year', 'years']:
                start_date = end_date - timedelta(days=number*365)  # Approximate
            
            return {
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d")
            }
        
        return None
    
    def _normalize_date(self, date_str: str) -> str:
        """
        Normalize date to YYYY-MM-DD format.
        
        Args:
            date_str (str): Date string to normalize
            
        Returns:
            str: Normalized date string
        """
        # Check if it's already in YYYY-MM-DD format
        if len(date_str) == 10 and date_str[4] == '-' and date_str[7] == '-':
            return date_str
        
        # If it's in MM/DD/YYYY format, convert it
        parts = date_str.split('/')
        if len(parts) == 3 and len(parts[2]) == 4:
            return f"{parts[2]}-{parts[0].zfill(2)}-{parts[1].zfill(2)}"
        
        # Return original string if we can't parse it
        return date_str
    
    def _calculate_risk_metrics(self, market_data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Calculate risk metrics from market data.
        
        Args:
            market_data (pd.DataFrame): Market data with price history
            symbol (str): Stock symbol
            
        Returns:
            Dict[str, Any]: Risk metrics
        """
        # Calculate basic risk metrics
        # Daily returns
        returns = market_data['Close'].pct_change().dropna()
        
        # Volatility (annualized)
        volatility = returns.std() * np.sqrt(252)
        
        # Value at Risk (VaR) at 95% confidence level
        var_95 = np.percentile(returns, 5) * market_data['Close'].iloc[-1]
        
        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        max_return = cumulative_returns.cummax()
        drawdown = (cumulative_returns - max_return) / max_return
        max_drawdown = drawdown.min()
        
        # Calculate risk score (0-1, where 1 is highest risk)
        # This is a simplified calculation that could be expanded
        vol_score = min(1.0, volatility / 0.5)  # Normalize volatility (assuming 50% annual vol is max)
        drawdown_score = min(1.0, abs(max_drawdown) / 0.5)  # Normalize drawdown (assuming 50% drawdown is max)
        
        # RSI as an oversold/overbought indicator
        latest_rsi = market_data.get('RSI_14', pd.Series()).iloc[-1] if 'RSI_14' in market_data.columns else None
        rsi_risk = 0
        if latest_rsi is not None:
            if latest_rsi > 70:  # Overbought
                rsi_risk = (latest_rsi - 70) / 30  # Scale from 0-1
            elif latest_rsi < 30:  # Oversold
                rsi_risk = (30 - latest_rsi) / 30  # Scale from 0-1
        
        # Combined risk score
        risk_score = (vol_score * 0.4) + (drawdown_score * 0.4) + (rsi_risk * 0.2)
        
        # Risk rating
        risk_rating = "Low"
        if risk_score > 0.7:
            risk_rating = "High"
        elif risk_score > 0.4:
            risk_rating = "Medium"
        
        # Current market trend
        current_price = market_data['Close'].iloc[-1]
        sma_20 = market_data.get('SMA_20', pd.Series()).iloc[-1] if 'SMA_20' in market_data.columns else None
        sma_50 = market_data.get('SMA_50', pd.Series()).iloc[-1] if 'SMA_50' in market_data.columns else None
        
        trend = "Undefined"
        if sma_20 is not None and sma_50 is not None:
            if current_price > sma_20 and current_price > sma_50 and sma_20 > sma_50:
                trend = "Strong Uptrend"
            elif current_price > sma_20 and current_price > sma_50:
                trend = "Uptrend"
            elif current_price < sma_20 and current_price < sma_50 and sma_20 < sma_50:
                trend = "Strong Downtrend"
            elif current_price < sma_20 and current_price < sma_50:
                trend = "Downtrend"
            else:
                trend = "Sideways"
        
        # Compile risk metrics
        return {
            "symbol": symbol,
            "volatility": volatility,
            "volatility_annualized": volatility * np.sqrt(252),
            "value_at_risk_95": var_95,
            "max_drawdown": max_drawdown,
            "risk_score": risk_score,
            "risk_rating": risk_rating,
            "latest_rsi": latest_rsi,
            "trend": trend,
            "current_price": current_price,
            "analysis_date": datetime.now().strftime("%Y-%m-%d"),
            # Add market condition assessment
            "market_condition": self._assess_market_condition(market_data)
        }
    
    def _assess_market_condition(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Assess market conditions based on technical indicators.
        
        Args:
            market_data (pd.DataFrame): Market data with indicators
            
        Returns:
            Dict[str, Any]: Market condition assessment
        """
        # Get latest values
        current_price = market_data['Close'].iloc[-1]
        
        # Check for indicators
        rsi = market_data.get('RSI_14', pd.Series()).iloc[-1] if 'RSI_14' in market_data.columns else None
        macd = market_data.get('MACD', pd.Series()).iloc[-1] if 'MACD' in market_data.columns else None
        macd_signal = market_data.get('MACD_Signal', pd.Series()).iloc[-1] if 'MACD_Signal' in market_data.columns else None
        upper_band = market_data.get('Upper_Band', pd.Series()).iloc[-1] if 'Upper_Band' in market_data.columns else None
        lower_band = market_data.get('Lower_Band', pd.Series()).iloc[-1] if 'Lower_Band' in market_data.columns else None
        
        # Assess RSI conditions
        rsi_condition = "neutral"
        if rsi is not None:
            if rsi > 70:
                rsi_condition = "overbought"
            elif rsi < 30:
                rsi_condition = "oversold"
        
        # Assess MACD conditions
        macd_condition = "neutral"
        if macd is not None and macd_signal is not None:
            if macd > macd_signal:
                macd_condition = "bullish"
            else:
                macd_condition = "bearish"
        
        # Assess Bollinger Band conditions
        bb_condition = "neutral"
        if upper_band is not None and lower_band is not None:
            if current_price > upper_band:
                bb_condition = "overbought"
            elif current_price < lower_band:
                bb_condition = "oversold"
        
        # Overall market condition
        bullish_signals = 0
        bearish_signals = 0
        
        if rsi_condition == "oversold":
            bullish_signals += 1
        elif rsi_condition == "overbought":
            bearish_signals += 1
        
        if macd_condition == "bullish":
            bullish_signals += 1
        elif macd_condition == "bearish":
            bearish_signals += 1
        
        if bb_condition == "oversold":
            bullish_signals += 1
        elif bb_condition == "overbought":
            bearish_signals += 1
        
        overall_condition = "neutral"
        if bullish_signals > bearish_signals:
            overall_condition = "bullish"
        elif bearish_signals > bullish_signals:
            overall_condition = "bearish"
        
        return {
            "rsi_condition": rsi_condition,
            "macd_condition": macd_condition,
            "bollinger_condition": bb_condition,
            "overall_condition": overall_condition,
            "bullish_signals": bullish_signals,
            "bearish_signals": bearish_signals
        }
    
    def _calculate_position_risk(self, symbol: str, portfolio: Dict[str, Any], risk_data: Dict[str, Any]) -> str:
        """
        Calculate position-specific risk based on portfolio context.
        
        Args:
            symbol (str): Stock symbol
            portfolio (Dict[str, Any]): Portfolio data
            risk_data (Dict[str, Any]): Risk metrics already calculated
            
        Returns:
            str: Position risk assessment text
        """
        # Extract position data
        position = None
        for pos in portfolio.get("positions", []):
            if pos.get("symbol") == symbol:
                position = pos
                break
        
        if not position:
            return "No position data available in the portfolio for this symbol."
        
        # Get position size and cost basis
        position_size = position.get("quantity", 0)
        cost_basis = position.get("cost_basis", 0)
        
        # Calculate position metrics
        current_price = risk_data.get("current_price", 0)
        position_value = position_size * current_price
        
        # Calculate unrealized P&L
        if cost_basis > 0:
            unrealized_pl = (current_price - cost_basis) * position_size
            unrealized_pl_pct = ((current_price / cost_basis) - 1) * 100
        else:
            unrealized_pl = 0
            unrealized_pl_pct = 0
        
        # Calculate position as % of portfolio
        portfolio_value = portfolio.get("total_value", 0)
        position_pct = (position_value / portfolio_value * 100) if portfolio_value > 0 else 0
        
        # Position risk assessment based on concentration
        concentration_risk = "Low"
        if position_pct > 20:
            concentration_risk = "High"
        elif position_pct > 10:
            concentration_risk = "Medium"
        
        # Combine with overall risk rating
        overall_position_risk = "Low"
        risk_rating = risk_data.get("risk_rating", "Low")
        
        if risk_rating == "High" and concentration_risk in ["Medium", "High"]:
            overall_position_risk = "High"
        elif risk_rating == "High" or concentration_risk == "High":
            overall_position_risk = "Medium-High"
        elif risk_rating == "Medium" and concentration_risk in ["Medium", "High"]:
            overall_position_risk = "Medium"
        
        # Generate position risk text
        return f"""
Position Risk Assessment:
- Position Size: {position_size} shares (${position_value:.2f})
- Position Weight: {position_pct:.2f}% of portfolio
- Unrealized P&L: ${unrealized_pl:.2f} ({unrealized_pl_pct:.2f}%)
- Concentration Risk: {concentration_risk}
- Overall Position Risk: {overall_position_risk}
        """
    
    def _generate_risk_assessment(self, risk_data: Dict[str, Any]) -> str:
        """
        Generate a human-readable risk assessment.
        
        Args:
            risk_data (Dict[str, Any]): Risk metrics
            
        Returns:
            str: Risk assessment text
        """
        symbol = risk_data.get("symbol", "Unknown")
        volatility = risk_data.get("volatility", 0) * 100
        var_95 = risk_data.get("value_at_risk_95", 0)
        max_drawdown = risk_data.get("max_drawdown", 0) * 100
        risk_rating = risk_data.get("risk_rating", "Unknown")
        current_price = risk_data.get("current_price", 0)
        trend = risk_data.get("trend", "Unknown")
        
        # Market condition data
        market_condition = risk_data.get("market_condition", {})
        rsi_condition = market_condition.get("rsi_condition", "neutral")
        macd_condition = market_condition.get("macd_condition", "neutral")
        overall_condition = market_condition.get("overall_condition", "neutral")
        
        assessment = f"""
Risk Assessment for {symbol}:

Current Status:
- Price: ${current_price:.2f}
- Trend: {trend}
- Market Condition: {overall_condition.capitalize()}

Risk Metrics:
- Volatility (Daily): {volatility:.2f}%
- Value at Risk (95%): ${abs(var_95):.2f}
- Maximum Historical Drawdown: {abs(max_drawdown):.2f}%
- Overall Risk Rating: {risk_rating}

Technical Indicators:
- RSI Condition: {rsi_condition.capitalize()}
- MACD Condition: {macd_condition.capitalize()}
        """
        
        # Add recommendations based on risk level
        assessment += "\nRisk Management Recommendations:"
        
        if risk_rating == "High":
            assessment += """
- Consider smaller position sizes
- Set tight stop losses (5-8% from entry)
- Monitor the position closely
- Consider hedging strategies
            """
        elif risk_rating == "Medium":
            assessment += """
- Use moderate position sizes
- Set standard stop losses (8-12% from entry)
- Review the position weekly
            """
        else:  # Low risk
            assessment += """
- Standard position sizing is appropriate
- Set wider stop losses if desired (10-15% from entry)
- Regular position review schedule
            """
        
        return assessment 