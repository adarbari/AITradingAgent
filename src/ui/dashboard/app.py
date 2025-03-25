"""
Trading System Dashboard.
Main UI application for interacting with the AI trading system.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import sys
import json

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.data import DataManager
from src.agent.multi_agent.orchestrator import TradingAgentOrchestrator
from src.agent.multi_agent.base_agent import AgentInput

# Configure page
st.set_page_config(
    page_title="AI Trading System Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Page styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #424242;
        margin-bottom: 1rem;
    }
    .info-text {
        font-size: 1rem;
        color: #616161;
    }
    .highlight {
        background-color: #f0f7ff;
        padding: 0.5rem;
        border-radius: 0.3rem;
        border-left: 0.3rem solid #1E88E5;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 0.15rem 0.3rem rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .buy {color: #4CAF50; font-weight: bold;}
    .sell {color: #F44336; font-weight: bold;}
    .hold {color: #FF9800; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_manager' not in st.session_state:
    st.session_state.data_manager = DataManager()

if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = TradingAgentOrchestrator(
        data_manager=st.session_state.data_manager,
        verbose=1
    )

if 'portfolio' not in st.session_state:
    # Default empty portfolio
    st.session_state.portfolio = {
        "total_value": 100000.0,
        "cash": 100000.0,
        "positions": [],
        "performance": {
            "1d_return": 0.0,
            "1w_return": 0.0,
            "1m_return": 0.0,
            "3m_return": 0.0,
            "ytd_return": 0.0,
            "1y_return": 0.0
        }
    }

if 'trading_history' not in st.session_state:
    st.session_state.trading_history = []

if 'analysis_cache' not in st.session_state:
    st.session_state.analysis_cache = {}


# App header
st.markdown('<div class="main-header">AI Trading System Dashboard</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<div class="sub-header">Navigation</div>', unsafe_allow_html=True)
    page = st.radio("", ["Dashboard", "Market Analysis", "Portfolio Management", "Trade Execution", "System Performance"])
    
    st.markdown('<div class="sub-header">Settings</div>', unsafe_allow_html=True)
    risk_tolerance = st.select_slider(
        "Risk Tolerance",
        options=["Conservative", "Moderate", "Aggressive"],
        value="Moderate"
    )
    
    # Time settings
    st.markdown('<div class="sub-header">Time Range</div>', unsafe_allow_html=True)
    time_range = st.selectbox(
        "Select Time Range",
        ["1 Day", "1 Week", "1 Month", "3 Months", "6 Months", "1 Year", "YTD"]
    )
    
    # Symbol selection
    st.markdown('<div class="sub-header">Symbols</div>', unsafe_allow_html=True)
    default_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    symbol = st.selectbox("Select Symbol", default_symbols)
    
    # Action buttons
    st.markdown('<div class="sub-header">Actions</div>', unsafe_allow_html=True)
    if st.button("Refresh Data"):
        st.success("Data refreshed!")


# Dashboard page
if page == "Dashboard":
    # Create three columns
    col1, col2, col3 = st.columns([1, 1, 1])
    
    # Portfolio summary in first column
    with col1:
        st.markdown('<div class="sub-header">Portfolio Summary</div>', unsafe_allow_html=True)
        
        # Portfolio value
        st.metric(
            label="Total Value",
            value=f"${st.session_state.portfolio['total_value']:,.2f}",
            delta=f"{st.session_state.portfolio['performance']['1d_return']:.2f}%"
        )
        
        # Cash position
        st.metric(
            label="Cash Position",
            value=f"${st.session_state.portfolio['cash']:,.2f}",
            delta=f"{st.session_state.portfolio['cash'] / st.session_state.portfolio['total_value'] * 100:.1f}% of portfolio"
        )
        
        # Positions count
        st.metric(
            label="Open Positions",
            value=len(st.session_state.portfolio['positions'])
        )
    
    # Latest recommendations in second column
    with col2:
        st.markdown('<div class="sub-header">Latest Recommendations</div>', unsafe_allow_html=True)
        
        if 'analysis_cache' not in st.session_state or not st.session_state.analysis_cache:
            st.info("No recent recommendations available. Run market analysis first.")
        else:
            for sym, analysis in list(st.session_state.analysis_cache.items())[:3]:
                decision = analysis.get('decision', 'HOLD')
                confidence = analysis.get('confidence', 0.0)
                
                decision_color = "hold"
                if decision == "BUY":
                    decision_color = "buy"
                elif decision == "SELL":
                    decision_color = "sell"
                
                st.markdown(f"""
                <div class="highlight">
                    <b>{sym}:</b> <span class="{decision_color}">{decision}</span> with {confidence:.1%} confidence<br>
                    <small>Based on analysis at {analysis.get('timestamp', 'N/A')}</small>
                </div>
                """, unsafe_allow_html=True)
    
    # System status in third column
    with col3:
        st.markdown('<div class="sub-header">System Status</div>', unsafe_allow_html=True)
        
        # Risk tolerance setting
        st.info(f"Risk Tolerance: {risk_tolerance}")
        
        # Active models
        st.markdown("**Active Models:**")
        st.markdown("- Market Analysis Model: Active")
        st.markdown("- Risk Assessment Model: Active")
        st.markdown("- Portfolio Optimization: Active")
        st.markdown("- Advanced Execution: Active")
    
    # Portfolio performance chart
    st.markdown('<div class="sub-header">Portfolio Performance</div>', unsafe_allow_html=True)
    
    # Create sample data for demo
    dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
    portfolio_values = 100000 * (1 + np.cumsum(np.random.normal(0.001, 0.01, size=len(dates))))
    benchmark_values = 100000 * (1 + np.cumsum(np.random.normal(0.0005, 0.01, size=len(dates))))
    
    # Create dataframe
    df_performance = pd.DataFrame({
        'Date': dates,
        'Portfolio': portfolio_values,
        'Benchmark': benchmark_values
    })
    
    # Create plotly chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_performance['Date'], 
        y=df_performance['Portfolio'],
        mode='lines',
        name='Portfolio',
        line=dict(color='#1E88E5', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=df_performance['Date'], 
        y=df_performance['Benchmark'],
        mode='lines',
        name='S&P 500',
        line=dict(color='#FFC107', width=2, dash='dot')
    ))
    
    fig.update_layout(
        title="30-Day Performance",
        xaxis_title="Date",
        yaxis_title="Value ($)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=400,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Latest trades and activity
    st.markdown('<div class="sub-header">Recent Activity</div>', unsafe_allow_html=True)
    
    if not st.session_state.trading_history:
        st.info("No recent trading activity.")
    else:
        # Convert trading history to DataFrame
        df_trades = pd.DataFrame(st.session_state.trading_history)
        st.dataframe(df_trades, use_container_width=True)

# Market Analysis page
elif page == "Market Analysis":
    st.markdown('<div class="sub-header">Market Analysis</div>', unsafe_allow_html=True)
    
    # Symbol selection for analysis
    analysis_symbols = st.multiselect(
        "Select Symbols for Analysis",
        default_symbols,
        default=[symbol]
    )
    
    if st.button("Run Analysis"):
        progress_bar = st.progress(0)
        
        for i, sym in enumerate(analysis_symbols):
            # Update progress
            progress_bar.progress((i + 0.5) / len(analysis_symbols))
            
            # Create input for the orchestrator
            agent_input = AgentInput(
                request=f"Analyze {sym} and provide trading recommendation",
                context={
                    "symbols": [sym],
                    "risk_tolerance": risk_tolerance.lower()
                }
            )
            
            # Run the analysis
            with st.spinner(f"Analyzing {sym}..."):
                try:
                    # In a real implementation, this would use the actual orchestrator
                    # Here we're just simulating results
                    result = simulate_market_analysis(sym, risk_tolerance)
                    
                    # Store in session state
                    st.session_state.analysis_cache[sym] = result
                except Exception as e:
                    st.error(f"Error analyzing {sym}: {str(e)}")
                    continue
            
            # Update progress again
            progress_bar.progress((i + 1) / len(analysis_symbols))
        
        progress_bar.empty()
        st.success(f"Analysis completed for {len(analysis_symbols)} symbols!")
    
    # Display analysis results
    for sym in analysis_symbols:
        if sym in st.session_state.analysis_cache:
            result = st.session_state.analysis_cache[sym]
            decision = result.get('decision', 'HOLD')
            confidence = result.get('confidence', 0.0)
            
            # Create an expander for each symbol
            with st.expander(f"{sym}: {decision} ({confidence:.1%} confidence)", expanded=True):
                # Create columns for metrics
                metrics_cols = st.columns(4)
                
                with metrics_cols[0]:
                    st.metric(
                        label="Current Price",
                        value=f"${result.get('price', 0.0):.2f}",
                        delta=f"{result.get('price_change_pct', 0.0):.2f}%"
                    )
                
                with metrics_cols[1]:
                    st.metric(
                        label="Volume",
                        value=f"{result.get('volume', 0)/1000000:.1f}M",
                        delta=f"{result.get('volume_change_pct', 0.0):.2f}%"
                    )
                
                with metrics_cols[2]:
                    st.metric(
                        label="RSI",
                        value=f"{result.get('rsi', 0.0):.1f}",
                        delta=None
                    )
                
                with metrics_cols[3]:
                    st.metric(
                        label="Trend",
                        value=result.get('trend', 'Neutral'),
                        delta=None
                    )
                
                # Create price chart
                if 'price_history' in result:
                    price_df = pd.DataFrame(result['price_history'])
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(
                        x=price_df['date'],
                        open=price_df['open'],
                        high=price_df['high'],
                        low=price_df['low'],
                        close=price_df['close'],
                        name='Price'
                    ))
                    
                    fig.update_layout(
                        title=f"{sym} Price Chart",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        height=400,
                        margin=dict(l=20, r=20, t=50, b=20),
                        xaxis_rangeslider_visible=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Display detailed analysis
                st.markdown("### Analysis Details")
                st.markdown(result.get('analysis_text', 'No detailed analysis available.'))
        else:
            st.info(f"No analysis data available for {sym}. Run analysis first.")

# Function to simulate market analysis
def simulate_market_analysis(symbol, risk_tolerance):
    """
    Simulate market analysis for demo purposes.
    In a real implementation, this would call the actual market analysis agent.
    """
    current_price = np.random.uniform(100, 1000)
    price_change = np.random.uniform(-5, 5)
    volume = np.random.uniform(1000000, 10000000)
    
    # Generate random decision
    decision_options = ["BUY", "HOLD", "SELL"]
    decision_weights = [0.4, 0.3, 0.3]  # Slightly bias toward BUY for demo
    decision = np.random.choice(decision_options, p=decision_weights)
    
    confidence = np.random.uniform(0.6, 0.95)
    
    # RSI (30 = oversold, 70 = overbought)
    rsi = np.random.uniform(30, 70)
    
    # Trend options
    trend_options = ["Strong Uptrend", "Uptrend", "Neutral", "Downtrend", "Strong Downtrend"]
    trend = np.random.choice(trend_options)
    
    # Generate simulated price history
    days = 30
    dates = pd.date_range(end=datetime.now(), periods=days)
    
    # Start with a base price
    base_price = current_price * 0.9
    
    # Generate daily changes
    daily_changes = np.random.normal(0.001, 0.015, days)
    
    # Accumulate changes
    cumulative_changes = np.exp(np.cumsum(daily_changes))
    
    # Calculate daily prices
    closes = base_price * cumulative_changes
    
    # Generate open, high, low values
    opens = closes * np.random.uniform(0.99, 1.01, days)
    highs = np.maximum(opens, closes) * np.random.uniform(1.001, 1.02, days)
    lows = np.minimum(opens, closes) * np.random.uniform(0.98, 0.999, days)
    
    # Create price history data
    price_history = []
    for i in range(days):
        price_history.append({
            'date': dates[i],
            'open': opens[i],
            'high': highs[i],
            'low': lows[i],
            'close': closes[i],
            'volume': int(np.random.uniform(0.5, 1.5) * volume)
        })
    
    # Create analysis text
    if decision == "BUY":
        analysis_text = f"""
        Technical indicators suggest a favorable entry point for {symbol}. The RSI of {rsi:.1f} indicates the stock 
        is {'oversold' if rsi < 40 else 'in a balanced position'} with room for upward movement. The current 
        {trend.lower()} pattern suggests momentum that could drive further price increases.
        
        Volume trends show {'increasing' if np.random.random() > 0.5 else 'consistent'} investor interest, 
        and the recent price action has formed a {'bullish' if np.random.random() > 0.5 else 'supportive'} pattern.
        
        For a {risk_tolerance.lower()} risk profile, entering a position at the current price of ${current_price:.2f}
        offers a favorable risk/reward ratio with potential upside of 10-15% over the next 3-6 months.
        """
    elif decision == "SELL":
        analysis_text = f"""
        Technical indicators suggest an exit point for {symbol}. The RSI of {rsi:.1f} indicates the stock 
        is {'overbought' if rsi > 60 else 'reaching resistance levels'}. The current {trend.lower()} pattern 
        shows signs of weakening momentum.
        
        Volume trends show {'decreasing' if np.random.random() > 0.5 else 'concerning'} patterns, and the 
        price action has formed a {'bearish' if np.random.random() > 0.5 else 'weakening'} structure.
        
        For a {risk_tolerance.lower()} risk profile, exiting the position at the current price of ${current_price:.2f}
        would lock in profits and protect against potential downside risk of 8-12% in the near term.
        """
    else:  # HOLD
        analysis_text = f"""
        Technical indicators suggest maintaining current positions in {symbol}. The RSI of {rsi:.1f} indicates 
        the stock is in a balanced position without strong overbought or oversold signals.
        
        The current {trend.lower()} shows stable momentum, and volume patterns indicate normal trading activity.
        Price action remains within expected ranges with no significant breakout or breakdown signals.
        
        For a {risk_tolerance.lower()} risk profile, holding positions at the current price of ${current_price:.2f}
        aligns with longer-term objectives while waiting for clearer directional signals.
        """
    
    return {
        'symbol': symbol,
        'decision': decision,
        'confidence': confidence,
        'price': current_price,
        'price_change': price_change,
        'price_change_pct': (price_change / current_price) * 100,
        'volume': volume,
        'volume_change_pct': np.random.uniform(-10, 10),
        'rsi': rsi,
        'trend': trend,
        'analysis_text': analysis_text,
        'price_history': price_history,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    } 