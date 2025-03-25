"""
Tests for the Streamlit dashboard application.
"""
import pytest
from unittest.mock import MagicMock, patch, Mock
import pandas as pd
import numpy as np
import os
import sys
import tempfile
import json
from datetime import datetime, timedelta
import importlib.util

# Create a mock for streamlit since it's not designed for unit testing
class MockSt:
    """Mock implementation of essential streamlit functions for testing"""
    
    def __init__(self):
        self.sidebar_elements = []
        self.main_elements = []
        self.session_state = {}
        self.metrics = []
        self.charts = []
        self.containers = []
        self.columns_created = []
        self.expanders = []
        self.progress_bars = []
        self.dataframes = []
        self.markdown_texts = []
        
    def set_page_config(self, **kwargs):
        self.page_config = kwargs
        
    def markdown(self, text, unsafe_allow_html=False):
        self.markdown_texts.append({"text": text, "unsafe_allow_html": unsafe_allow_html})
    
    def sidebar(self):
        sidebar = MagicMock()
        self.sidebar_elements.append(sidebar)
        return sidebar
    
    def columns(self, widths):
        cols = [MagicMock() for _ in range(len(widths))]
        self.columns_created.append(cols)
        return cols
    
    def metric(self, label, value, delta=None):
        self.metrics.append({"label": label, "value": value, "delta": delta})
    
    def pyplot(self, fig):
        self.charts.append({"type": "pyplot", "fig": fig})
    
    def plotly_chart(self, fig, use_container_width=False):
        self.charts.append({"type": "plotly", "fig": fig, "use_container_width": use_container_width})
    
    def container(self):
        container = MagicMock()
        self.containers.append(container)
        return container
    
    def expander(self, title, expanded=False):
        expander = MagicMock()
        self.expanders.append({"title": title, "expanded": expanded, "expander": expander})
        return expander
    
    def progress(self, value):
        progress_bar = MagicMock()
        self.progress_bars.append({"value": value, "progress_bar": progress_bar})
        return progress_bar
    
    def dataframe(self, df, use_container_width=False):
        self.dataframes.append({"df": df, "use_container_width": use_container_width})
    
    def button(self, label):
        # Return True sometimes to simulate button clicks
        import random
        return random.choice([True, False])
    
    def selectbox(self, label, options, index=0):
        return options[index]
    
    def multiselect(self, label, options, default=None):
        return default if default is not None else options[:1]
    
    def radio(self, label, options):
        return options[0]
    
    def select_slider(self, label, options, value=None):
        return value if value is not None else options[0]
    
    def success(self, text):
        pass
    
    def info(self, text):
        pass
    
    def error(self, text):
        pass
    
    def warning(self, text):
        pass
    
    def spinner(self, text):
        class SpinnerContextManager:
            def __enter__(self):
                pass
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
        return SpinnerContextManager()


@pytest.fixture
def mock_streamlit():
    """Create a mock for streamlit"""
    return MockSt()


@pytest.fixture
def mock_data_manager():
    """Create a mock DataManager"""
    data_manager = MagicMock()
    
    # Configure the mock to return sample data
    def get_market_data(symbol=None, **kwargs):
        # Generate sample market data
        dates = pd.date_range(start="2023-01-01", periods=30, freq="D")
        close_prices = np.linspace(150, 165, 30) + np.random.normal(0, 2, 30)
        
        return pd.DataFrame({
            'Close': close_prices,
            'Open': close_prices * 0.99,
            'High': close_prices * 1.01,
            'Low': close_prices * 0.98,
            'Volume': np.random.randint(5000000, 10000000, 30),
        }, index=dates)
    
    data_manager.get_market_data.side_effect = get_market_data
    return data_manager


@pytest.fixture
def mock_orchestrator(mock_data_manager):
    """Create a mock TradingAgentOrchestrator"""
    orchestrator = MagicMock()
    
    # Configure the mock to return sample outputs
    def process(input_data):
        # Generate a sample response based on the agent_input
        if "market_analysis" in input_data.request.lower():
            return {
                "response": f"Analysis for {input_data.context.get('symbols', ['AAPL'])[0]}",
                "data": {
                    "symbol": input_data.context.get('symbols', ['AAPL'])[0],
                    "decision": np.random.choice(["BUY", "HOLD", "SELL"]),
                    "confidence": np.random.uniform(0.6, 0.95),
                    "price": np.random.uniform(100, 1000),
                    "analysis_text": "Sample analysis text."
                },
                "confidence": np.random.uniform(0.7, 0.9)
            }
        elif "portfolio" in input_data.request.lower():
            return {
                "response": "Portfolio optimization completed",
                "data": {
                    "allocations": {
                        "AAPL": 0.25,
                        "MSFT": 0.25,
                        "GOOGL": 0.20,
                        "AMZN": 0.15,
                        "TSLA": 0.15
                    },
                    "expected_return": 0.12,
                    "expected_risk": 0.18,
                    "sharpe_ratio": 0.67
                },
                "confidence": 0.85
            }
        else:
            return {
                "response": "Generic response",
                "data": {},
                "confidence": 0.5
            }
    
    orchestrator.process.side_effect = process
    orchestrator.data_manager = mock_data_manager
    return orchestrator


@pytest.fixture
def sample_portfolio():
    """Create a sample portfolio for testing"""
    return {
        "total_value": 100000.0,
        "cash": 25000.0,
        "positions": [
            {
                "symbol": "AAPL",
                "shares": 100,
                "avg_price": 150.0,
                "current_price": 165.0,
                "market_value": 16500.0,
                "unrealized_pl": 1500.0,
                "allocation": 0.165
            },
            {
                "symbol": "MSFT",
                "shares": 50,
                "avg_price": 280.0,
                "current_price": 290.0,
                "market_value": 14500.0,
                "unrealized_pl": 500.0,
                "allocation": 0.145
            }
        ],
        "performance": {
            "1d_return": 0.8,
            "1w_return": 2.1,
            "1m_return": -0.5,
            "3m_return": 5.2,
            "ytd_return": 12.3,
            "1y_return": 18.7
        }
    }


@pytest.fixture
def sample_trading_history():
    """Create a sample trading history for testing"""
    return [
        {
            "date": "2023-03-15",
            "symbol": "AAPL",
            "action": "BUY",
            "quantity": 50,
            "price": 145.0,
            "total": 7250.0,
            "strategy": "Limit Order",
            "status": "Filled"
        },
        {
            "date": "2023-03-10",
            "symbol": "TSLA",
            "action": "SELL",
            "quantity": 10,
            "price": 180.0,
            "total": 1800.0,
            "strategy": "Market Order",
            "status": "Filled"
        }
    ]


def test_ui_initialization(mock_streamlit, mock_data_manager, mock_orchestrator):
    """Test dashboard initialization"""
    # Setup the test
    with patch.dict(sys.modules, {'streamlit': mock_streamlit}):
        # Need to patch these before importing the dashboard module
        with patch('sys.path.append'):
            with patch.dict(sys.modules, {
                'src.data': MagicMock(),
                'src.agent.multi_agent.orchestrator': MagicMock(),
                'src.agent.multi_agent.base_agent': MagicMock()
            }):
                # Create temporary module
                dashboard_spec = importlib.util.spec_from_file_location(
                    "dashboard", "src/ui/dashboard/app.py"
                )
                dashboard = importlib.util.module_from_spec(dashboard_spec)
                
                # Mock dependencies
                dashboard.DataManager = MagicMock(return_value=mock_data_manager)
                dashboard.TradingAgentOrchestrator = MagicMock(return_value=mock_orchestrator)
                dashboard.AgentInput = MagicMock()
                
                # Execute the module (this will run the dashboard initialization code)
                with patch('streamlit.set_page_config'):
                    dashboard_spec.loader.exec_module(dashboard)
                
                # Check page configuration
                assert hasattr(mock_streamlit, 'page_config')
                assert mock_streamlit.page_config.get('page_title') == "AI Trading System Dashboard"
                
                # Check session state initialization
                assert 'data_manager' in dashboard.st.session_state
                assert 'orchestrator' in dashboard.st.session_state
                assert 'portfolio' in dashboard.st.session_state
                assert 'trading_history' in dashboard.st.session_state
                assert 'analysis_cache' in dashboard.st.session_state


def test_simulate_market_analysis():
    """Test the market analysis simulation function"""
    # Import the module with the function
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src/ui/dashboard')))
    from app import simulate_market_analysis
    
    # Test with different risk tolerance levels
    risk_levels = ["Conservative", "Moderate", "Aggressive"]
    symbols = ["AAPL", "MSFT", "GOOGL"]
    
    for risk in risk_levels:
        for symbol in symbols:
            result = simulate_market_analysis(symbol, risk)
            
            # Check structure of returned data
            assert "symbol" in result
            assert "decision" in result
            assert "confidence" in result
            assert "price" in result
            assert "price_change_pct" in result
            assert "volume" in result
            assert "rsi" in result
            assert "trend" in result
            assert "analysis_text" in result
            assert "price_history" in result
            assert "timestamp" in result
            
            # Check data types
            assert isinstance(result["decision"], str)
            assert isinstance(result["confidence"], float)
            assert 0 <= result["confidence"] <= 1
            assert isinstance(result["price"], float)
            assert isinstance(result["rsi"], float)
            assert 0 <= result["rsi"] <= 100
            
            # Check price history
            assert len(result["price_history"]) == 30  # 30 days
            for day_data in result["price_history"]:
                assert "date" in day_data
                assert "open" in day_data
                assert "high" in day_data
                assert "low" in day_data
                assert "close" in day_data
                assert "volume" in day_data
                assert day_data["high"] >= day_data["close"]
                assert day_data["high"] >= day_data["open"]
                assert day_data["low"] <= day_data["close"]
                assert day_data["low"] <= day_data["open"]


@patch('streamlit.set_page_config')
def test_dashboard_page(mock_set_page_config, mock_streamlit, sample_portfolio, sample_trading_history):
    """Test the dashboard page rendering"""
    # Setup mocks
    st = mock_streamlit
    
    # Create module namespace
    module_namespace = {
        'st': st,
        'pd': pd,
        'np': np,
        'go': MagicMock(),
        'px': MagicMock(),
        'datetime': datetime,
        'timedelta': timedelta,
        'os': os,
        'sys': sys,
        'json': json,
        'DataManager': MagicMock(),
        'TradingAgentOrchestrator': MagicMock(),
        'AgentInput': MagicMock()
    }
    
    # Initialize session state
    st.session_state = {
        'data_manager': MagicMock(),
        'orchestrator': MagicMock(),
        'portfolio': sample_portfolio,
        'trading_history': sample_trading_history,
        'analysis_cache': {
            'AAPL': {
                'decision': 'BUY',
                'confidence': 0.85,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }
    }
    
    # Execute dashboard page code
    with open('src/ui/dashboard/app.py', 'r') as f:
        dashboard_code = f.read()
    
    # Patch dashboard-specific matplotlib and plotly functions
    with patch('plotly.graph_objects.Figure', return_value=MagicMock()):
        with patch('plotly.graph_objects.Scatter', return_value=MagicMock()):
            # Execute only the dashboard page section
            dashboard_sections = dashboard_code.split("# Dashboard page")
            if len(dashboard_sections) > 1:
                dashboard_page_code = dashboard_sections[1].split("# Market Analysis page")[0]
                exec(dashboard_page_code, module_namespace)
    
    # Check that the dashboard page rendered the expected components
    
    # Check portfolio metrics
    portfolio_metrics = [m for m in st.metrics if m["label"] == "Total Value"]
    assert len(portfolio_metrics) > 0
    assert str(sample_portfolio["total_value"]) in portfolio_metrics[0]["value"]
    
    # Check cash position display
    cash_metrics = [m for m in st.metrics if m["label"] == "Cash Position"]
    assert len(cash_metrics) > 0
    assert str(sample_portfolio["cash"]) in cash_metrics[0]["value"]
    
    # Check positions count
    positions_metrics = [m for m in st.metrics if m["label"] == "Open Positions"]
    assert len(positions_metrics) > 0
    assert positions_metrics[0]["value"] == len(sample_portfolio["positions"])
    
    # Check plotly charts (portfolio performance)
    assert len(st.charts) > 0
    assert any(chart["type"] == "plotly" for chart in st.charts)


@patch('streamlit.set_page_config')
def test_market_analysis_page(mock_set_page_config, mock_streamlit):
    """Test the market analysis page rendering"""
    # Setup mocks
    st = mock_streamlit
    
    # Create module namespace
    module_namespace = {
        'st': st,
        'pd': pd,
        'np': np,
        'go': MagicMock(),
        'px': MagicMock(),
        'datetime': datetime,
        'timedelta': timedelta,
        'os': os,
        'sys': sys,
        'json': json,
        'DataManager': MagicMock(),
        'TradingAgentOrchestrator': MagicMock(),
        'AgentInput': MagicMock(),
        'simulate_market_analysis': MagicMock(return_value={
            'symbol': 'AAPL',
            'decision': 'BUY',
            'confidence': 0.85,
            'price': 150.0,
            'price_change': 2.5,
            'price_change_pct': 1.7,
            'volume': 8500000,
            'volume_change_pct': 3.2,
            'rsi': 65.4,
            'trend': 'Uptrend',
            'analysis_text': 'Sample analysis text',
            'price_history': [
                {
                    'date': datetime.now() - timedelta(days=i),
                    'open': 150.0 - i * 0.5,
                    'high': 152.0 - i * 0.5,
                    'low': 148.0 - i * 0.5,
                    'close': 151.0 - i * 0.5,
                    'volume': 8000000 - i * 100000
                } for i in range(30)
            ],
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    }
    
    # Initialize session state
    st.session_state = {
        'data_manager': MagicMock(),
        'orchestrator': MagicMock(),
        'portfolio': {},
        'trading_history': [],
        'analysis_cache': {}
    }
    
    # Execute market analysis page code
    with open('src/ui/dashboard/app.py', 'r') as f:
        dashboard_code = f.read()
    
    # Extract and execute the market analysis page section
    dashboard_sections = dashboard_code.split("# Market Analysis page")
    if len(dashboard_sections) > 1:
        market_analysis_code = dashboard_sections[1].split("# Function to simulate market analysis")[0]
        with patch('plotly.graph_objects.Candlestick', return_value=MagicMock()):
            with patch('plotly.graph_objects.Figure', return_value=MagicMock()):
                exec(market_analysis_code, module_namespace)
    
    # Verify that the page has analysis UI elements
    assert any("Select Symbols for Analysis" in str(call) for call in st.mock_calls if hasattr(call, '__str__'))
    
    # Verify that progress handling is included
    assert len(st.progress_bars) > 0


def test_integrate_portfolio_with_analysis(mock_streamlit, sample_portfolio):
    """Test integration between portfolio and analysis components"""
    st = mock_streamlit
    
    # Set up session state with portfolio and analysis data
    st.session_state = {
        'portfolio': sample_portfolio,
        'analysis_cache': {
            'AAPL': {
                'decision': 'BUY',
                'confidence': 0.85,
                'price': 165.0,
                'price_change_pct': 1.2,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            'MSFT': {
                'decision': 'HOLD',
                'confidence': 0.75,
                'price': 290.0,
                'price_change_pct': -0.3,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }
    }
    
    # Check if portfolio positions match analysis data
    for position in sample_portfolio['positions']:
        symbol = position['symbol']
        if symbol in st.session_state['analysis_cache']:
            analysis = st.session_state['analysis_cache'][symbol]
            # Verify price consistency
            assert position['current_price'] == analysis['price']


def test_dashboard_styling(mock_streamlit):
    """Test dashboard styling and CSS is properly applied"""
    st = mock_streamlit
    
    # Create module namespace
    module_namespace = {
        'st': st
    }
    
    # Extract and execute just the CSS styling part
    with open('src/ui/dashboard/app.py', 'r') as f:
        dashboard_code = f.read()
    
    style_section = None
    if "st.markdown(\"\"\"" in dashboard_code:
        style_start = dashboard_code.find("st.markdown(\"\"\"")
        style_end = dashboard_code.find("\"\"\"", style_start + 14)
        style_section = dashboard_code[style_start:style_end+3]
    
    if style_section:
        exec(style_section, module_namespace)
    
    # Check if styling was applied
    assert len(st.markdown_texts) > 0
    
    # Check if any CSS classes were defined
    css_text = None
    for item in st.markdown_texts:
        if item.get("unsafe_allow_html", False) and "<style>" in item.get("text", ""):
            css_text = item.get("text")
            break
    
    assert css_text is not None
    
    # Verify essential CSS classes are defined
    assert ".main-header" in css_text
    assert ".sub-header" in css_text
    assert ".metric-card" in css_text
    assert ".buy" in css_text
    assert ".sell" in css_text
    assert ".hold" in css_text 