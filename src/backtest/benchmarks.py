"""
Benchmark calculations for backtesting performance comparison.
"""
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import yfinance as yf

class Benchmark(ABC):
    """Base class for benchmark calculations."""
    
    def __init__(self, name, symbol, initial_investment=10000):
        self.name = name
        self.symbol = symbol
        self.initial_investment = initial_investment
    
    @abstractmethod
    def calculate_returns(self, start_date, end_date):
        """Calculate returns for the benchmark."""
        pass
    
    def calculate_metrics(self, returns_df):
        """Calculate performance metrics for the benchmark."""
        if returns_df.empty:
            return None
            
        # Calculate total return
        final_value = returns_df['value'].iloc[-1]
        total_return = (final_value / self.initial_investment - 1) * 100
        
        # Calculate daily returns
        returns_df['daily_return'] = returns_df['value'].pct_change()
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0)
        sharpe_ratio = np.sqrt(252) * returns_df['daily_return'].mean() / returns_df['daily_return'].std()
        
        # Calculate maximum drawdown
        returns_df['cumulative_max'] = returns_df['value'].cummax()
        returns_df['drawdown'] = (returns_df['value'] / returns_df['cumulative_max'] - 1) * 100
        max_drawdown = returns_df['drawdown'].min()
        
        return {
            'name': self.name,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'returns_df': returns_df
        }

class BuyAndHoldBenchmark(Benchmark):
    """Buy and hold benchmark for a specific stock."""
    
    def calculate_returns(self, start_date, end_date):
        try:
            # Get stock data
            stock = yf.Ticker(self.symbol)
            data = stock.history(start=start_date, end=end_date)
            
            if data.empty:
                print(f"No data available for {self.symbol}")
                return pd.DataFrame()
            
            # Calculate portfolio value
            initial_price = data['Close'].iloc[0]
            shares = self.initial_investment / initial_price
            
            # Create returns DataFrame
            returns_df = pd.DataFrame(index=data.index)
            returns_df['value'] = shares * data['Close']
            
            return returns_df
            
        except Exception as e:
            print(f"Error calculating {self.name} benchmark: {e}")
            return pd.DataFrame()

class MarketIndexBenchmark(Benchmark):
    """Benchmark for market indices (S&P 500, NASDAQ)."""
    
    def calculate_returns(self, start_date, end_date):
        try:
            # Get index data
            index = yf.Ticker(self.symbol)
            data = index.history(start=start_date, end=end_date)
            
            if data.empty:
                print(f"No data available for {self.symbol}")
                return pd.DataFrame()
            
            # Calculate portfolio value
            initial_price = data['Close'].iloc[0]
            shares = self.initial_investment / initial_price
            
            # Create returns DataFrame
            returns_df = pd.DataFrame(index=data.index)
            returns_df['value'] = shares * data['Close']
            
            return returns_df
            
        except Exception as e:
            print(f"Error calculating {self.name} benchmark: {e}")
            return pd.DataFrame()

# Factory for creating benchmark instances
class BenchmarkFactory:
    """Factory for creating benchmark instances."""
    
    @staticmethod
    def create_benchmark(benchmark_type, symbol, initial_investment=10000):
        if benchmark_type == "buy_and_hold":
            return BuyAndHoldBenchmark(f"Buy & Hold {symbol}", symbol, initial_investment)
        elif benchmark_type == "sp500":
            return MarketIndexBenchmark("S&P 500", "^GSPC", initial_investment)
        elif benchmark_type == "nasdaq":
            return MarketIndexBenchmark("NASDAQ", "^IXIC", initial_investment)
        else:
            raise ValueError(f"Unknown benchmark type: {benchmark_type}") 