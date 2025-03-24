"""
The agent module contains trading environment implementations
for reinforcement learning agents.
"""

from src.agent.base_trading_env import BaseTradingEnvironment
from src.agent.trading_env import TradingEnvironment, LegacyTradingEnvironment

# For backward compatibility
SafeTradingEnvironment = TradingEnvironment

__all__ = [
    'BaseTradingEnvironment',
    'LegacyTradingEnvironment',  # Deprecated, use TradingEnvironment instead
    'TradingEnvironment',        # Recommended implementation with improved safety
    'SafeTradingEnvironment',    # For backward compatibility
] 