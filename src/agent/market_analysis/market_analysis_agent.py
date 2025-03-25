"""Market analysis agent for analyzing market conditions and trends."""

from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from ..base_trading_env import BaseAgent, AgentInput, AgentOutput


class MarketAnalysisAgent(BaseAgent):
    """Agent responsible for analyzing market conditions and trends."""
    
    def __init__(self, name: str = "market_analysis", config: Optional[Dict[str, Any]] = None):
        """Initialize the market analysis agent.
        
        Args:
            name: Unique identifier for the agent
            config: Optional configuration dictionary
        """
        super().__init__(name, config)
        self.ma_periods = config.get("ma_periods", [20, 50, 200])
        self.rsi_period = config.get("rsi_period", 14)
        self.volume_ma_period = config.get("volume_ma_period", 20)
        
    def process(self, input_data: AgentInput) -> AgentOutput:
        """Process market data and analyze conditions.
        
        Args:
            input_data: Input data containing market prices and volumes
            
        Returns:
            AgentOutput containing market analysis results
        """
        data = input_data.data
        df = pd.DataFrame(data)
        
        # Calculate technical indicators
        analysis = self._calculate_indicators(df)
        
        # Analyze market conditions
        conditions = self._analyze_conditions(analysis)
        
        # Calculate confidence score
        confidence = self._calculate_confidence(conditions)
        
        return AgentOutput(
            result={
                "conditions": conditions,
                "indicators": analysis,
                "timestamp": df.index[-1]
            },
            confidence=confidence,
            metadata={
                "ma_periods": self.ma_periods,
                "rsi_period": self.rsi_period
            }
        )
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators.
        
        Args:
            df: DataFrame with price and volume data
            
        Returns:
            Dictionary of calculated indicators
        """
        indicators = {}
        
        # Calculate moving averages
        for period in self.ma_periods:
            indicators[f"ma_{period}"] = df["close"].rolling(window=period).mean()
        
        # Calculate RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        indicators["rsi"] = 100 - (100 / (1 + rs))
        
        # Calculate volume indicators
        indicators["volume_ma"] = df["volume"].rolling(window=self.volume_ma_period).mean()
        indicators["volume_std"] = df["volume"].rolling(window=self.volume_ma_period).std()
        
        return indicators
    
    def _analyze_conditions(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market conditions based on indicators.
        
        Args:
            indicators: Dictionary of technical indicators
            
        Returns:
            Dictionary of market conditions
        """
        conditions = {
            "trend": self._analyze_trend(indicators),
            "momentum": self._analyze_momentum(indicators),
            "volatility": self._analyze_volatility(indicators),
            "volume": self._analyze_volume(indicators)
        }
        return conditions
    
    def _analyze_trend(self, indicators: Dict[str, Any]) -> str:
        """Analyze price trend.
        
        Args:
            indicators: Dictionary of technical indicators
            
        Returns:
            Trend description
        """
        ma_20 = indicators["ma_20"].iloc[-1]
        ma_50 = indicators["ma_50"].iloc[-1]
        ma_200 = indicators["ma_200"].iloc[-1]
        
        if ma_20 > ma_50 > ma_200:
            return "strong_uptrend"
        elif ma_20 > ma_50:
            return "uptrend"
        elif ma_20 < ma_50 < ma_200:
            return "strong_downtrend"
        elif ma_20 < ma_50:
            return "downtrend"
        else:
            return "sideways"
    
    def _analyze_momentum(self, indicators: Dict[str, Any]) -> str:
        """Analyze price momentum.
        
        Args:
            indicators: Dictionary of technical indicators
            
        Returns:
            Momentum description
        """
        rsi = indicators["rsi"].iloc[-1]
        
        if rsi > 70:
            return "overbought"
        elif rsi < 30:
            return "oversold"
        else:
            return "neutral"
    
    def _analyze_volatility(self, indicators: Dict[str, Any]) -> str:
        """Analyze price volatility.
        
        Args:
            indicators: Dictionary of technical indicators
            
        Returns:
            Volatility description
        """
        volume_std = indicators["volume_std"].iloc[-1]
        volume_ma = indicators["volume_ma"].iloc[-1]
        
        if volume_std > volume_ma * 1.5:
            return "high"
        elif volume_std < volume_ma * 0.5:
            return "low"
        else:
            return "normal"
    
    def _analyze_volume(self, indicators: Dict[str, Any]) -> str:
        """Analyze trading volume.
        
        Args:
            indicators: Dictionary of technical indicators
            
        Returns:
            Volume description
        """
        volume = indicators["volume"].iloc[-1]
        volume_ma = indicators["volume_ma"].iloc[-1]
        
        if volume > volume_ma * 1.5:
            return "high"
        elif volume < volume_ma * 0.5:
            return "low"
        else:
            return "normal"
    
    def _calculate_confidence(self, conditions: Dict[str, Any]) -> float:
        """Calculate confidence score for the analysis.
        
        Args:
            conditions: Dictionary of market conditions
            
        Returns:
            Confidence score between 0 and 1
        """
        # Simple confidence calculation based on condition consistency
        trend_confidence = {
            "strong_uptrend": 1.0,
            "uptrend": 0.8,
            "sideways": 0.5,
            "downtrend": 0.8,
            "strong_downtrend": 1.0
        }
        
        momentum_confidence = {
            "overbought": 0.9,
            "oversold": 0.9,
            "neutral": 0.5
        }
        
        volume_confidence = {
            "high": 0.8,
            "normal": 0.6,
            "low": 0.4
        }
        
        confidence = (
            trend_confidence[conditions["trend"]] +
            momentum_confidence[conditions["momentum"]] +
            volume_confidence[conditions["volume"]]
        ) / 3
        
        return min(max(confidence, 0.0), 1.0) 