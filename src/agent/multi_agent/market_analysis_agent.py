"""
Market Analysis Agent for the multi-agent trading system.
"""
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from .base_agent import BaseAgent, AgentInput, AgentOutput
from src.data import DataManager

class MarketAnalysisAgent(BaseAgent):
    """
    Agent specialized in market data analysis.
    
    This agent can:
    1. Analyze price trends and patterns
    2. Identify support and resistance levels
    3. Evaluate technical indicators
    4. Detect potential market regime changes
    """
    
    def __init__(self, data_manager: DataManager, openai_api_key: Optional[str] = None, 
                model_name: str = "gpt-4o", temperature: float = 0.1, verbose: int = 0):
        """
        Initialize the market analysis agent.
        
        Args:
            data_manager (DataManager): Data manager instance
            openai_api_key (str, optional): OpenAI API key
            model_name (str): Name of the OpenAI model to use
            temperature (float): Temperature for LLM (0-1)
            verbose (int): Verbosity level (0: silent, 1: normal, 2: detailed)
        """
        super().__init__(
            name="Market Analysis Agent",
            description="Analyzes market data and technical indicators to identify trends and patterns",
            verbose=verbose
        )
        self.data_manager = data_manager
        
        # Set up LLM if API key is provided
        self.llm = None
        if openai_api_key:
            self.llm = ChatOpenAI(
                api_key=openai_api_key,
                model=model_name,
                temperature=temperature
            )
    
    def process(self, input_data: AgentInput) -> AgentOutput:
        """
        Process the input and generate a market analysis.
        
        Args:
            input_data (AgentInput): Input data for the agent
            
        Returns:
            AgentOutput: Agent's analysis response
        """
        # Extract parameters from the request
        symbol = self._extract_symbol(input_data.request)
        date_range = self._extract_date_range(input_data.request)
        
        if not symbol:
            # Check if symbol is in the context
            if input_data.context and "symbol" in input_data.context:
                symbol = input_data.context["symbol"]
            
        if not symbol:
            return AgentOutput(
                response="I need a specific stock symbol to analyze. Please provide a symbol like 'AAPL' or 'MSFT'.",
                confidence=0.0
            )
        
        # If dates weren't specified in the request, check if they're in the context
        if not date_range and input_data.context and "date_range" in input_data.context:
            context_date_range = input_data.context["date_range"]
            # Check if date_range in context is a dictionary with start_date and end_date
            if isinstance(context_date_range, dict):
                if "start_date" in context_date_range and "end_date" in context_date_range:
                    date_range = {
                        "start_date": context_date_range["start_date"],
                        "end_date": context_date_range["end_date"]
                    }
        
        # Default date range if none provided
        if not date_range:
            from datetime import datetime, timedelta
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
            date_range = {
                "start_date": start_date,
                "end_date": end_date
            }
        
        # Get market data using the data manager
        market_data = self.data_manager.get_market_data(
            symbol=symbol,
            start_date=date_range["start_date"],
            end_date=date_range["end_date"],
            include_indicators=True
        )
        
        if market_data is None or len(market_data) == 0:
            return AgentOutput(
                response=f"I couldn't retrieve market data for {symbol} in the specified date range. Please try a different symbol or date range.",
                confidence=0.0
            )
            
        # Perform market analysis
        analysis_result = self._analyze_market_data(symbol, market_data, input_data.request)
        
        # Record this interaction in memory
        self.add_to_memory({
            "input": input_data.request,
            "symbol": symbol,
            "date_range": date_range,
            "output": analysis_result.response
        })
        
        return analysis_result
    
    def _extract_symbol(self, text: str) -> Optional[str]:
        """Extract stock symbol from text"""
        # Simple approach: look for common stock symbol patterns
        import re
        
        # Look for stock symbol patterns (all caps 1-5 letters)
        matches = re.findall(r'\b[A-Z]{1,5}\b', text)
        
        # Also look for "$SYMBOL" format
        dollar_matches = re.findall(r'\$([A-Z]{1,5})\b', text)
        
        all_matches = matches + dollar_matches
        
        if all_matches:
            # Prioritize symbols that look more like stock tickers
            for match in all_matches:
                # Ignore common words that might be in all caps
                if match not in ["I", "A", "AT", "THE", "FOR", "AND", "OR"]:
                    return match
        
        return None
    
    def _extract_date_range(self, text: str) -> Optional[Dict[str, str]]:
        """Extract date range from text"""
        import re
        from datetime import datetime, timedelta
        
        # Look for date patterns like "from 2023-01-01 to 2023-01-31"
        date_pattern = r'(?:from|between)?\s*(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{2}/\d{2}/\d{2})\s*(?:to|and|through|until)\s*(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{2}/\d{2}/\d{2})'
        
        match = re.search(date_pattern, text, re.IGNORECASE)
        
        if match:
            start_date_str, end_date_str = match.groups()
            
            # Convert to YYYY-MM-DD format if needed
            start_date = self._normalize_date(start_date_str)
            end_date = self._normalize_date(end_date_str)
            
            return {
                "start_date": start_date,
                "end_date": end_date
            }
        
        # Look for relative time periods like "last 30 days"
        period_pattern = r'(?:last|past)\s+(\d+)\s+(day|days|week|weeks|month|months|year|years)'
        
        match = re.search(period_pattern, text, re.IGNORECASE)
        
        if match:
            amount, unit = match.groups()
            amount = int(amount)
            
            end_date = datetime.now()
            
            if 'day' in unit:
                start_date = end_date - timedelta(days=amount)
            elif 'week' in unit:
                start_date = end_date - timedelta(weeks=amount)
            elif 'month' in unit:
                start_date = end_date - timedelta(days=amount * 30)  # Approximate
            elif 'year' in unit:
                start_date = end_date - timedelta(days=amount * 365)  # Approximate
            
            return {
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d")
            }
        
        return None
    
    def _normalize_date(self, date_str: str) -> str:
        """Convert various date formats to YYYY-MM-DD"""
        from datetime import datetime
        
        # Try different formats
        formats = [
            "%Y-%m-%d",       # 2023-01-31
            "%m/%d/%Y",       # 01/31/2023
            "%m/%d/%y"        # 01/31/23
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue
        
        # If all parsing attempts fail, return the original string
        return date_str
    
    def _analyze_market_data(self, symbol: str, market_data: pd.DataFrame, request: str) -> AgentOutput:
        """
        Analyze market data using a combination of traditional techniques and LLM.
        
        Args:
            symbol (str): Stock symbol
            market_data (pd.DataFrame): Market data including indicators
            request (str): Original request for context
            
        Returns:
            AgentOutput: Analysis results
        """
        # Calculate key metrics using traditional methods
        analysis_data = {}
        
        # Price trend analysis
        close_prices = market_data['Close']
        current_price = close_prices.iloc[-1]
        price_change = close_prices.iloc[-1] - close_prices.iloc[0]
        percent_change = (price_change / close_prices.iloc[0]) * 100
        
        # Volatility
        daily_returns = close_prices.pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252)  # Annualized
        
        # Moving average analysis
        if 'SMA_20' in market_data.columns and 'SMA_50' in market_data.columns:
            sma_20 = market_data['SMA_20'].iloc[-1]
            sma_50 = market_data['SMA_50'].iloc[-1]
            price_vs_sma20 = current_price - sma_20
            price_vs_sma50 = current_price - sma_50
            ma_cross = (market_data['SMA_20'].iloc[-2] <= market_data['SMA_50'].iloc[-2] and 
                        market_data['SMA_20'].iloc[-1] > market_data['SMA_50'].iloc[-1])
        else:
            sma_20, sma_50, price_vs_sma20, price_vs_sma50, ma_cross = None, None, None, None, None
            
        # Support and resistance levels (simplistic approach)
        if len(market_data) >= 20:
            # Local minima and maxima in the last 20 days
            rolling_min = market_data['Low'].rolling(window=10, center=True).min().iloc[-20:]
            rolling_max = market_data['High'].rolling(window=10, center=True).max().iloc[-20:]
            
            # Find points that are local extrema
            supports = rolling_min[rolling_min == market_data['Low'].iloc[-20:]]
            resistances = rolling_max[rolling_max == market_data['High'].iloc[-20:]]
            
            # Take the average of the closest 2 points if available
            if len(supports) >= 2:
                support_level = supports.iloc[-2:].mean()
            elif len(supports) == 1:
                support_level = supports.iloc[-1]
            else:
                support_level = None
                
            if len(resistances) >= 2:
                resistance_level = resistances.iloc[-2:].mean()
            elif len(resistances) == 1:
                resistance_level = resistances.iloc[-1]
            else:
                resistance_level = None
        else:
            support_level, resistance_level = None, None
            
        # RSI & MACD analysis
        if 'RSI_14' in market_data.columns:
            rsi = market_data['RSI_14'].iloc[-1]
            rsi_signal = "oversold" if rsi < 30 else "overbought" if rsi > 70 else "neutral"
        else:
            rsi, rsi_signal = None, None
            
        if all(col in market_data.columns for col in ['MACD', 'MACD_Signal']):
            macd = market_data['MACD'].iloc[-1]
            macd_signal = market_data['MACD_Signal'].iloc[-1]
            macd_histogram = macd - macd_signal
            macd_cross_up = (market_data['MACD'].iloc[-2] <= market_data['MACD_Signal'].iloc[-2] and 
                             market_data['MACD'].iloc[-1] > market_data['MACD_Signal'].iloc[-1])
            macd_cross_down = (market_data['MACD'].iloc[-2] >= market_data['MACD_Signal'].iloc[-2] and 
                               market_data['MACD'].iloc[-1] < market_data['MACD_Signal'].iloc[-1])
        else:
            macd, macd_signal, macd_histogram, macd_cross_up, macd_cross_down = None, None, None, None, None
        
        # Put all calculated metrics into the analysis data
        analysis_data = {
            "symbol": symbol,
            "current_price": round(current_price, 2),
            "price_change": round(price_change, 2),
            "percent_change": round(percent_change, 2),
            "volatility": round(volatility * 100, 2),
            "moving_averages": {
                "sma_20": round(sma_20, 2) if sma_20 is not None else None,
                "sma_50": round(sma_50, 2) if sma_50 is not None else None,
                "price_vs_sma20": round(price_vs_sma20, 2) if price_vs_sma20 is not None else None,
                "price_vs_sma50": round(price_vs_sma50, 2) if price_vs_sma50 is not None else None,
                "ma_cross": ma_cross
            },
            "support_resistance": {
                "support": round(support_level, 2) if support_level is not None else None,
                "resistance": round(resistance_level, 2) if resistance_level is not None else None
            },
            "indicators": {
                "rsi": round(rsi, 2) if rsi is not None else None,
                "rsi_signal": rsi_signal,
                "macd": round(macd, 4) if macd is not None else None,
                "macd_signal": round(macd_signal, 4) if macd_signal is not None else None,
                "macd_histogram": round(macd_histogram, 4) if macd_histogram is not None else None,
                "macd_cross_up": macd_cross_up,
                "macd_cross_down": macd_cross_down
            },
            "start_date": market_data.index[0].strftime("%Y-%m-%d"),
            "end_date": market_data.index[-1].strftime("%Y-%m-%d"),
            "data_points": len(market_data)
        }
        
        # Use the LLM for interpretation if available
        if self.llm is not None:
            try:
                # Format the market data for the LLM context
                condensed_data = market_data.iloc[::5]  # Sample every 5th row to keep context manageable
                data_summary = condensed_data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(10).to_string()
                
                # Create a prompt for the LLM
                prompt = ChatPromptTemplate.from_template("""
                You are a market analysis expert. Analyze the following market data for {symbol} and provide key insights.
                
                User Request: {request}
                
                Data Summary:
                {data_summary}
                
                Technical Analysis Metrics:
                - Current Price: ${current_price}
                - Price Change: ${price_change} ({percent_change}%)
                - Volatility: {volatility}%
                - SMA 20: ${sma_20}
                - SMA 50: ${sma_50}
                - RSI: {rsi}
                - MACD: {macd}
                - MACD Signal: {macd_signal}
                - Support Level: ${support}
                - Resistance Level: ${resistance}
                
                Based on this data, provide a comprehensive analysis including:
                1. Price trend analysis
                2. Support and resistance analysis
                3. Technical indicator interpretation
                4. Overall market sentiment
                5. Key levels to watch
                
                Format your response in a clear, professional manner. 
                Return ONLY your analysis without restating the prompt or data provided.
                """)
                
                # Set up the LLM chain
                chain = prompt | self.llm
                
                # Prepare input by filtering out None values
                llm_input = {
                    "symbol": symbol,
                    "request": request,
                    "data_summary": data_summary,
                    "current_price": analysis_data["current_price"],
                    "price_change": analysis_data["price_change"],
                    "percent_change": analysis_data["percent_change"],
                    "volatility": analysis_data["volatility"],
                    "sma_20": analysis_data["moving_averages"]["sma_20"] or "N/A",
                    "sma_50": analysis_data["moving_averages"]["sma_50"] or "N/A",
                    "rsi": analysis_data["indicators"]["rsi"] or "N/A",
                    "macd": analysis_data["indicators"]["macd"] or "N/A",
                    "macd_signal": analysis_data["indicators"]["macd_signal"] or "N/A",
                    "support": analysis_data["support_resistance"]["support"] or "N/A",
                    "resistance": analysis_data["support_resistance"]["resistance"] or "N/A",
                }
                
                # Get the LLM response
                llm_response = chain.invoke(llm_input)
                
                # Create the final response
                response_text = llm_response.content
                confidence = 0.85  # Arbitrary confidence value
                
            except Exception as e:
                if self.verbose > 0:
                    print(f"Error using LLM for market analysis: {e}")
                # Fallback to rule-based analysis
                response_text = self._generate_rule_based_analysis(analysis_data)
                confidence = 0.6
        else:
            # Use rule-based analysis if no LLM is available
            response_text = self._generate_rule_based_analysis(analysis_data)
            confidence = 0.6
        
        return AgentOutput(
            response=response_text,
            data=analysis_data,
            confidence=confidence
        )
    
    def _generate_rule_based_analysis(self, analysis_data: Dict[str, Any]) -> str:
        """
        Generate a rule-based market analysis without using an LLM.
        
        Args:
            analysis_data (Dict): Analysis data dictionary
            
        Returns:
            str: Analysis text
        """
        symbol = analysis_data["symbol"]
        current_price = analysis_data["current_price"]
        percent_change = analysis_data["percent_change"]
        
        # Start with basic price information
        analysis = [
            f"Market Analysis for {symbol}:",
            f"The current price is ${current_price:.2f}, which represents a {percent_change:.2f}% change over the analyzed period."
        ]
        
        # Add trend analysis
        if percent_change > 0:
            trend = "upward"
            strength = "strong" if percent_change > 5 else "moderate" if percent_change > 2 else "mild"
        elif percent_change < 0:
            trend = "downward"
            strength = "strong" if percent_change < -5 else "moderate" if percent_change < -2 else "mild"
        else:
            trend = "sideways"
            strength = "neutral"
            
        analysis.append(f"The stock shows a {strength} {trend} trend.")
        
        # Moving average analysis
        ma_data = analysis_data["moving_averages"]
        if ma_data["sma_20"] is not None and ma_data["sma_50"] is not None:
            if current_price > ma_data["sma_20"] and current_price > ma_data["sma_50"]:
                analysis.append(f"Price is above both the 20-day and 50-day moving averages, suggesting bullish momentum.")
            elif current_price < ma_data["sma_20"] and current_price < ma_data["sma_50"]:
                analysis.append(f"Price is below both the 20-day and 50-day moving averages, suggesting bearish momentum.")
            elif current_price > ma_data["sma_20"] and current_price < ma_data["sma_50"]:
                analysis.append(f"Price is above the 20-day MA but below the 50-day MA, suggesting a potential shift from bearish to bullish.")
            else:
                analysis.append(f"Price is below the 20-day MA but above the 50-day MA, suggesting a potential shift from bullish to bearish.")
                
            if ma_data["ma_cross"]:
                analysis.append(f"A bullish crossover has occurred with the 20-day MA crossing above the 50-day MA.")
        
        # Support and resistance
        sr_data = analysis_data["support_resistance"]
        if sr_data["support"] is not None:
            analysis.append(f"The nearest support level is around ${sr_data['support']:.2f}.")
        if sr_data["resistance"] is not None:
            analysis.append(f"The nearest resistance level is around ${sr_data['resistance']:.2f}.")
            
        # RSI analysis
        indicators = analysis_data["indicators"]
        if indicators["rsi"] is not None:
            rsi = indicators["rsi"]
            if rsi < 30:
                analysis.append(f"RSI is at {rsi:.2f}, indicating the stock may be oversold.")
            elif rsi > 70:
                analysis.append(f"RSI is at {rsi:.2f}, indicating the stock may be overbought.")
            else:
                analysis.append(f"RSI is at {rsi:.2f}, within a neutral range.")
                
        # MACD analysis
        if indicators["macd"] is not None and indicators["macd_signal"] is not None:
            if indicators["macd_cross_up"]:
                analysis.append("MACD has crossed above the signal line, generating a bullish signal.")
            elif indicators["macd_cross_down"]:
                analysis.append("MACD has crossed below the signal line, generating a bearish signal.")
            elif indicators["macd"] > indicators["macd_signal"]:
                analysis.append("MACD is above the signal line, indicating bullish momentum.")
            else:
                analysis.append("MACD is below the signal line, indicating bearish momentum.")
        
        # Volatility assessment
        volatility = analysis_data["volatility"]
        if volatility > 30:
            analysis.append(f"The stock shows high volatility at {volatility:.2f}%, suggesting significant price swings.")
        elif volatility > 15:
            analysis.append(f"The stock shows moderate volatility at {volatility:.2f}%.")
        else:
            analysis.append(f"The stock shows relatively low volatility at {volatility:.2f}%.")
            
        # Overall summary
        if percent_change > 0 and (indicators["rsi"] or 50) < 70 and current_price > ma_data.get("sma_50", 0):
            analysis.append("\nOverall assessment: The technical indicators suggest a bullish outlook with potential for continued upward movement.")
        elif percent_change < 0 and (indicators["rsi"] or 50) > 30 and current_price < ma_data.get("sma_50", float('inf')):
            analysis.append("\nOverall assessment: The technical indicators suggest a bearish outlook with potential for continued downward movement.")
        else:
            analysis.append("\nOverall assessment: The technical indicators show mixed signals, suggesting a neutral or consolidating market.")
        
        return "\n".join(analysis) 