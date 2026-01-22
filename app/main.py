"""
TradeSense Modern Dashboard

A complete redesign with:
- Dark theme with glassmorphism
- Interactive candlestick charts with indicator overlays
- Real-time prediction vs actual visualization
- Modern signal action cards
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import stock_fetcher, crypto_fetcher, watchlist
from src.signals import signal_generator
from src.ml import FeatureEngineer, signal_model


# ============================================================================
# PAGE CONFIG & CUSTOM CSS
# ============================================================================

st.set_page_config(
    page_title="TradeSense",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Dark Theme CSS
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Root Variables */
    :root {
        --bg-primary: #0f0f1a;
        --bg-secondary: #1a1a2e;
        --bg-card: rgba(30, 30, 50, 0.7);
        --accent-purple: #7c3aed;
        --accent-blue: #3b82f6;
        --accent-green: #10b981;
        --accent-red: #ef4444;
        --text-primary: #ffffff;
        --text-secondary: #a0a0b0;
        --border-color: rgba(255, 255, 255, 0.1);
    }
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu, footer, header {visibility: hidden;}
    
    /* Glassmorphism Cards */
    .glass-card {
        background: var(--bg-card);
        backdrop-filter: blur(20px);
        border-radius: 16px;
        border: 1px solid var(--border-color);
        padding: 20px;
        margin-bottom: 16px;
    }
    
    /* Signal Cards */
    .signal-buy {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(16, 185, 129, 0.05) 100%);
        border-left: 4px solid var(--accent-green);
    }
    .signal-sell {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(239, 68, 68, 0.05) 100%);
        border-left: 4px solid var(--accent-red);
    }
    .signal-hold {
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.2) 0%, rgba(124, 58, 237, 0.05) 100%);
        border-left: 4px solid var(--accent-purple);
    }
    
    /* Metric Cards */
    .metric-card {
        background: var(--bg-card);
        backdrop-filter: blur(20px);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: var(--text-primary);
    }
    .metric-label {
        font-size: 12px;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Watchlist Items */
    .watchlist-item {
        background: var(--bg-card);
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .watchlist-item:hover {
        background: rgba(124, 58, 237, 0.2);
        transform: translateX(4px);
    }
    
    /* Score Bars */
    .score-bar-container {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        height: 8px;
        margin: 8px 0;
        overflow: hidden;
    }
    .score-bar {
        height: 100%;
        border-radius: 8px;
        transition: width 0.5s ease;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-purple) 0%, var(--accent-blue) 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(124, 58, 237, 0.4);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: var(--bg-secondary);
        border-right: 1px solid var(--border-color);
    }
    
    /* Input Fields */
    .stTextInput > div > div > input {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 10px;
        color: var(--text-primary);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: var(--text-primary) !important;
        font-weight: 600;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-card);
        border-radius: 12px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: var(--text-secondary);
        border-radius: 8px;
    }
    .stTabs [aria-selected="true"] {
        background: var(--accent-purple) !important;
        color: white !important;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .metric-value { font-size: 20px; }
        .metric-label { font-size: 10px; }
        .glass-card { padding: 12px; }
        .signal-card h2 { font-size: 24px; }
    }
    
    /* Settings Panel */
    .settings-panel {
        background: var(--bg-card);
        border-radius: 12px;
        padding: 16px;
        margin-top: 16px;
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background: var(--bg-card);
        border-radius: 10px;
        color: var(--text-primary);
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# CHART COMPONENTS
# ============================================================================

def create_candlestick_chart(df: pd.DataFrame, ticker: str, show_rsi: bool = True, 
                              show_macd: bool = True, show_bb: bool = False) -> go.Figure:
    """Create interactive candlestick chart with indicator overlays."""
    
    # Determine number of rows based on indicators
    rows = 1
    row_heights = [0.6]
    if show_rsi:
        rows += 1
        row_heights.append(0.2)
    if show_macd:
        rows += 1
        row_heights.append(0.2)
    
    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name=ticker,
            increasing_line_color='#10b981',
            decreasing_line_color='#ef4444'
        ),
        row=1, col=1
    )
    
    # Bollinger Bands
    if show_bb and 'BBL_20_2.0' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['BBU_20_2.0'], name='BB Upper',
                                  line=dict(color='rgba(124, 58, 237, 0.3)')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BBL_20_2.0'], name='BB Lower',
                                  line=dict(color='rgba(124, 58, 237, 0.3)'),
                                  fill='tonexty', fillcolor='rgba(124, 58, 237, 0.1)'), row=1, col=1)
    
    current_row = 2
    
    # RSI
    if show_rsi:
        import pandas_ta as ta
        rsi = ta.rsi(df['Close'], length=14)
        fig.add_trace(
            go.Scatter(x=df.index, y=rsi, name='RSI', 
                       line=dict(color='#7c3aed', width=2)),
            row=current_row, col=1
        )
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="rgba(239, 68, 68, 0.5)", row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="rgba(16, 185, 129, 0.5)", row=current_row, col=1)
        fig.update_yaxes(title_text="RSI", row=current_row, col=1)
        current_row += 1
    
    # MACD
    if show_macd:
        import pandas_ta as ta
        macd = ta.macd(df['Close'])
        if macd is not None:
            fig.add_trace(
                go.Scatter(x=df.index, y=macd['MACD_12_26_9'], name='MACD',
                           line=dict(color='#3b82f6', width=2)),
                row=current_row, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=macd['MACDs_12_26_9'], name='Signal',
                           line=dict(color='#f59e0b', width=2)),
                row=current_row, col=1
            )
            colors = ['#10b981' if v > 0 else '#ef4444' for v in macd['MACDh_12_26_9']]
            fig.add_trace(
                go.Bar(x=df.index, y=macd['MACDh_12_26_9'], name='Histogram',
                       marker_color=colors),
                row=current_row, col=1
            )
            fig.update_yaxes(title_text="MACD", row=current_row, col=1)
    
    # Layout
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,15,26,0.8)',
        height=600,
        margin=dict(l=50, r=50, t=30, b=30),
        xaxis_rangeslider_visible=False,
        showlegend=False,
        font=dict(family="Inter", color="#a0a0b0")
    )
    
    return fig


def create_prediction_chart(df: pd.DataFrame) -> go.Figure:
    """Create prediction vs actual price chart."""
    
    # Generate ML predictions for historical data
    engineer = FeatureEngineer()
    features_df = engineer.calculate_features(df.copy(), include_targets=False)
    
    if features_df.empty or not signal_model.is_trained:
        # Fallback: show just actual prices
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Actual',
                                  line=dict(color='#3b82f6', width=2)))
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15,15,26,0.8)',
            height=200,
            margin=dict(l=50, r=50, t=20, b=20)
        )
        return fig
    
    # Get predictions
    X = features_df[engineer.feature_columns]
    probs = signal_model.predict_proba(X)
    
    # Normalize price and prediction for comparison
    close_norm = (df['Close'].loc[features_df.index] - df['Close'].min()) / (df['Close'].max() - df['Close'].min())
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=features_df.index, y=close_norm, name='Actual (Normalized)',
                              line=dict(color='#3b82f6', width=2)))
    fig.add_trace(go.Scatter(x=features_df.index, y=probs, name='ML Prediction',
                              line=dict(color='#7c3aed', width=2, dash='dash')))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,15,26,0.8)',
        height=200,
        margin=dict(l=50, r=50, t=20, b=20),
        legend=dict(orientation="h", y=1.1),
        font=dict(family="Inter", color="#a0a0b0")
    )
    
    return fig


# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_signal_card(signal):
    """Render a modern signal action card."""
    signal_class = f"signal-{signal.signal_type.value.lower()}"
    signal_color = {
        "BUY": "#10b981",
        "SELL": "#ef4444", 
        "HOLD": "#7c3aed"
    }.get(signal.signal_type.value, "#7c3aed")
    
    st.markdown(f"""
    <div class="glass-card {signal_class}">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h2 style="margin: 0; color: {signal_color};">{signal.ticker}</h2>
                <p style="color: #a0a0b0; margin: 4px 0;">Entry: ${signal.entry_price:.2f}</p>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 36px; font-weight: 700; color: {signal_color};">
                    {signal.signal_type.value}
                </div>
                <div style="color: #a0a0b0;">Confidence: {signal.confidence:.0%}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_score_bar(label: str, score: float, color: str):
    """Render a horizontal score bar."""
    width = int(score * 100)
    st.markdown(f"""
    <div style="margin: 12px 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
            <span style="color: #a0a0b0; font-size: 13px;">{label}</span>
            <span style="color: white; font-weight: 600;">{score:.0%}</span>
        </div>
        <div class="score-bar-container">
            <div class="score-bar" style="width: {width}%; background: linear-gradient(90deg, {color}, {color}88);"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_metric(value: str, label: str, color: str = "#ffffff"):
    """Render a metric card."""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color: {color};">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Sidebar
    with st.sidebar:
        st.markdown("## 📈 TradeSense")
        st.markdown("---")
        
        # Ticker Search
        ticker = st.text_input("🔍 Search Ticker", value="AAPL", placeholder="Enter symbol...")
        
        if st.button("Analyze", use_container_width=True):
            st.session_state['analyze_ticker'] = ticker
        
        st.markdown("---")
        
        # Watchlist
        st.markdown("### 📋 Watchlist")
        watchlist_tickers = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"]
        
        for t in watchlist_tickers:
            col1, col2 = st.columns([2, 1])
            with col1:
                if st.button(t, key=f"wl_{t}", use_container_width=True):
                    st.session_state['analyze_ticker'] = t
            with col2:
                st.markdown(f"<span style='color: #10b981;'>↑ 1.2%</span>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Indicator Toggles
        st.markdown("### ⚙️ Indicators")
        show_rsi = st.checkbox("RSI", value=True)
        show_macd = st.checkbox("MACD", value=True)
        show_bb = st.checkbox("Bollinger Bands", value=False)
        
        # Advanced Settings
        with st.expander("🔧 Advanced Settings"):
            chart_period = st.selectbox(
                "Chart Period",
                options=["1mo", "3mo", "6mo", "1y", "2y"],
                index=2  # Default: 6mo
            )
            st.session_state['chart_period'] = chart_period
            
            st.markdown("**Indicator Parameters**")
            rsi_length = st.slider("RSI Length", 7, 21, 14)
            rsi_oversold = st.slider("RSI Oversold", 20, 40, 30)
            rsi_overbought = st.slider("RSI Overbought", 60, 80, 70)
            
            st.session_state['rsi_params'] = {
                'length': rsi_length,
                'oversold': rsi_oversold,
                'overbought': rsi_overbought
            }
    
    # Main Content
    current_ticker = st.session_state.get('analyze_ticker', 'AAPL')
    
    # Header
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    # Fetch data
    try:
        df = stock_fetcher.get_historical(current_ticker, period="6mo")
        current_price = df['Close'].iloc[-1]
        price_change = ((current_price - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
        
        with col1:
            st.markdown(f"# {current_ticker}")
        with col2:
            render_metric(f"${current_price:.2f}", "Price", "#ffffff")
        with col3:
            change_color = "#10b981" if price_change > 0 else "#ef4444"
            render_metric(f"{price_change:+.2f}%", "Change", change_color)
        with col4:
            render_metric(f"{df['Volume'].iloc[-1]/1e6:.1f}M", "Volume", "#7c3aed")
        
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return
    
    st.markdown("---")
    
    # Main Layout
    chart_col, signal_col = st.columns([3, 1])
    
    with chart_col:
        # Price Chart with Indicators
        st.markdown("### 📊 Price Chart")
        fig = create_candlestick_chart(df, current_ticker, show_rsi, show_macd, show_bb)
        st.plotly_chart(fig, use_container_width=True)
        
        # Prediction vs Actual
        st.markdown("### 🔮 Prediction vs Actual")
        pred_fig = create_prediction_chart(df)
        st.plotly_chart(pred_fig, use_container_width=True)
    
    with signal_col:
        # Generate Signal
        st.markdown("### 🎯 Signal Analysis")
        try:
            signal = signal_generator.generate_signal(current_ticker)
            render_signal_card(signal)
            
            # Score Breakdown
            st.markdown("#### Score Breakdown")
            render_score_bar("Technical", signal.technical_score, "#3b82f6")
            render_score_bar("Fundamental", signal.fundamental_score, "#10b981")
            render_score_bar("Sentiment", signal.sentiment_score, "#f59e0b")
            render_score_bar("Options", signal.options_score, "#ef4444")
            render_score_bar("ML Model", signal.ml_score, "#7c3aed")
            
            # Reasons
            st.markdown("#### Signal Reasons")
            for reason in signal.reasons:
                st.markdown(f"• {reason}")
                
        except Exception as e:
            st.error(f"Signal error: {e}")


if __name__ == "__main__":
    main()
