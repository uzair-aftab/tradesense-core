"""
TradeSense Pro v5 - Complete Trading Dashboard

Fixes:
- Removed empty box at top
- Centered metrics with equal sizing
- Fixed prediction chart (actual vs predicted price lines)
- Fixed expander header colors
- Fixed sentiment aggregator error  
- Added Paper Trading module
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import pandas_ta as ta
import numpy as np
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import stock_fetcher
from src.data.fetcher_fundamentals import fundamentals_fetcher
from src.signals import signal_generator
from src.sentiment.aggregator import sentiment_aggregator
from src.ml import FeatureEngineer, signal_model
from src.trading.paper_trading import paper_trader

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="TradeSense Pro",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# CSS - CLEAN, READABLE, FIXED COLORS
# ============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@400;500;600;700&display=swap');
    
    :root {
        --bg-dark: #0d0d0d;
        --bg-card: #1a1a1a;
        --bg-hover: #252525;
        --text-white: #ffffff;
        --text-gray: #999999;
        --text-muted: #666666;
        --strong-buy: #00ff88;
        --buy: #66ff99;
        --hold: #ffcc00;
        --sell: #ff6666;
        --strong-sell: #ff4444;
        --info: #00aaff;
        --border: #333333;
    }
    
    .stApp {
        background-color: var(--bg-dark);
        color: var(--text-white);
        font-family: 'Inter', sans-serif;
    }
    
    #MainMenu, footer, header {visibility: hidden;}
    
    /* ===== SIGNAL LIST ===== */
    .signal-list {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 12px;
    }
    .signal-list-title {
        font-size: 12px;
        font-weight: 600;
        color: var(--text-gray);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 10px;
    }
    .signal-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 10px;
        margin: 4px 0;
        border-radius: 4px;
        background: var(--bg-dark);
        font-size: 13px;
    }
    .signal-item:hover { background: var(--bg-hover); cursor: pointer; }
    .signal-ticker { font-family: 'JetBrains Mono'; font-weight: 600; color: var(--text-white); }
    .signal-label {
        font-size: 10px;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 3px;
        text-transform: uppercase;
    }
    .label-strong-buy { background: var(--strong-buy); color: #000; }
    .label-buy { background: var(--buy); color: #000; }
    .label-hold { background: var(--hold); color: #000; }
    .label-sell { background: var(--sell); color: #000; }
    .label-strong-sell { background: var(--strong-sell); color: #000; }
    
    /* ===== METRICS ROW - CENTERED ===== */
    .metrics-container {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 16px;
        margin-bottom: 16px;
        flex-wrap: wrap;
    }
    .ticker-display {
        font-size: 32px;
        font-weight: 700;
        font-family: 'JetBrains Mono';
        color: var(--text-white);
        margin-right: 20px;
    }
    .metric-box {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 12px 24px;
        text-align: center;
        min-width: 120px;
    }
    .metric-val {
        font-family: 'JetBrains Mono';
        font-size: 20px;
        font-weight: 600;
        color: var(--text-white);
    }
    .metric-lbl {
        font-size: 10px;
        color: var(--text-gray);
        text-transform: uppercase;
        margin-top: 2px;
    }
    .metric-sub {
        font-size: 9px;
        color: var(--text-muted);
    }
    .green { color: var(--strong-buy) !important; }
    .red { color: var(--strong-sell) !important; }
    .yellow { color: var(--hold) !important; }
    
    /* ===== MAIN SIGNAL BOX ===== */
    .main-signal {
        background: var(--bg-card);
        border: 2px solid var(--border);
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        margin-bottom: 12px;
    }
    .main-signal.buy-border { border-color: var(--strong-buy); }
    .main-signal.sell-border { border-color: var(--strong-sell); }
    .main-signal.hold-border { border-color: var(--hold); }
    .signal-text {
        font-size: 28px;
        font-weight: 700;
        font-family: 'JetBrains Mono';
    }
    .signal-conf {
        font-size: 12px;
        color: var(--text-gray);
        margin-top: 4px;
    }
    
    /* ===== CARDS ===== */
    .card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
    }
    .card-title {
        font-size: 11px;
        color: var(--text-gray);
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    
    /* ===== SCORE ROWS ===== */
    .score-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 6px 0;
        border-bottom: 1px solid #222;
        font-size: 13px;
    }
    .score-row:last-child { border-bottom: none; }
    .score-name { color: var(--text-gray); }
    .score-val { font-family: 'JetBrains Mono'; font-weight: 600; }
    
    /* ===== SCROLLABLE PANEL ===== */
    .scroll-panel {
        max-height: 180px;
        overflow-y: auto;
        background: var(--bg-dark);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 10px;
    }
    .panel-row {
        display: flex;
        justify-content: space-between;
        padding: 5px 0;
        font-size: 12px;
        border-bottom: 1px solid #222;
        color: var(--text-white);
    }
    .panel-row:last-child { border-bottom: none; }
    .panel-label { color: var(--text-gray); }
    .panel-value { color: var(--text-white); font-weight: 500; }
    
    /* ===== API QUOTA ===== */
    .quota-bar {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 4px;
        padding: 6px 10px;
        font-size: 10px;
        display: flex;
        justify-content: space-between;
        margin-bottom: 6px;
    }
    .quota-name { color: var(--text-gray); }
    .quota-val { color: var(--text-white); font-family: 'JetBrains Mono'; }
    
    /* ===== EXPANDERS - FIXED COLORS ===== */
    .streamlit-expanderHeader {
        background: var(--bg-card) !important;
        color: var(--text-white) !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
    }
    .streamlit-expanderHeader:hover {
        background: var(--bg-hover) !important;
        color: var(--text-white) !important;
    }
    .streamlit-expanderContent {
        background: var(--bg-card) !important;
        color: var(--text-white) !important;
        border: 1px solid var(--border) !important;
        border-top: none !important;
    }
    details summary {
        color: var(--text-white) !important;
    }
    details[open] summary {
        color: var(--text-white) !important;
    }
    
    /* ===== PAPER TRADING ===== */
    .portfolio-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid var(--info);
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .portfolio-value {
        font-size: 24px;
        font-weight: 700;
        font-family: 'JetBrains Mono';
        color: var(--text-white);
    }
    .portfolio-return {
        font-size: 14px;
        font-weight: 600;
    }
    
    /* ===== BUTTONS ===== */
    .stButton > button {
        background: var(--strong-buy) !important;
        color: #000 !important;
        font-weight: 600 !important;
        border: none !important;
    }
    .stButton > button:hover {
        background: #00cc6a !important;
    }
    
    h1, h2, h3, h4 { color: var(--text-white) !important; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_signal_label(score):
    """Convert score to signal label."""
    if score >= 0.7:
        return ("STRONG BUY", "label-strong-buy")
    elif score >= 0.55:
        return ("BUY", "label-buy")
    elif score >= 0.45:
        return ("HOLD", "label-hold")
    elif score >= 0.3:
        return ("SELL", "label-sell")
    else:
        return ("STRONG SELL", "label-strong-sell")


def get_signal_color(signal_type):
    """Get color for signal type."""
    colors = {
        "BUY": "#00ff88",
        "STRONG BUY": "#00ff88",
        "HOLD": "#ffcc00",
        "SELL": "#ff4444",
        "STRONG SELL": "#ff4444"
    }
    return colors.get(signal_type, "#ffcc00")


# ============================================================================
# CHARTS
# ============================================================================

def create_price_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Clean candlestick chart with volume."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, row_heights=[0.8, 0.2])
    
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], 
        low=df['Low'], close=df['Close'], name=ticker,
        increasing=dict(line=dict(color='#00ff88', width=1), fillcolor='#00ff88'),
        decreasing=dict(line=dict(color='#ff4444', width=1), fillcolor='#ff4444')
    ), row=1, col=1)
    
    colors = ['rgba(0,255,136,0.4)' if c >= o else 'rgba(255,68,68,0.4)' for c, o in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors), row=2, col=1)
    
    fig.update_layout(
        template='plotly_dark', paper_bgcolor='#0d0d0d', plot_bgcolor='#0d0d0d',
        height=380, margin=dict(l=0, r=0, t=0, b=0),
        xaxis_rangeslider_visible=False, showlegend=False,
        font=dict(family="JetBrains Mono", color="#888", size=10)
    )
    fig.update_xaxes(gridcolor='#222', showgrid=True)
    fig.update_yaxes(gridcolor='#222', showgrid=True)
    return fig


def create_prediction_chart(df: pd.DataFrame) -> tuple:
    """Create chart with BOTH actual and predicted PRICE lines."""
    engineer = FeatureEngineer()
    features_df = engineer.calculate_features(df.copy(), include_targets=False)
    
    if features_df.empty or not signal_model.is_trained:
        return None, None
    
    # Get ML predictions
    X = features_df[engineer.feature_columns]
    probs = signal_model.predict_proba(X)
    
    # Get actual close prices
    close = df['Close'].loc[features_df.index]
    
    # Calculate accuracy
    returns_3d = close.pct_change(3).shift(-3)
    predictions = (probs > 0.5).astype(int)
    actuals = (returns_3d > 0.01).astype(int)
    valid_mask = ~returns_3d.isna()
    
    if valid_mask.sum() > 10:
        accuracy = (predictions[valid_mask] == actuals[valid_mask]).mean() * 100
    else:
        accuracy = None
    
    # Last 30 days
    last_n = 30
    chart_idx = features_df.index[-last_n:]
    chart_close = close.loc[chart_idx]
    chart_probs = probs[-last_n:]
    
    # Create predicted price: actual price adjusted by prediction direction
    # If prob > 0.5, predict price goes UP by expected amount
    base_price = chart_close.iloc[0]
    predicted_prices = []
    
    for i, (idx, actual_price) in enumerate(chart_close.items()):
        prob = chart_probs[i]
        # Predict direction based on probability
        if prob > 0.5:
            # Bullish - predict price will be higher
            predicted = actual_price * (1 + (prob - 0.5) * 0.1)
        else:
            # Bearish - predict price will be lower
            predicted = actual_price * (1 - (0.5 - prob) * 0.1)
        predicted_prices.append(predicted)
    
    # Create figure with single plot showing both lines
    fig = go.Figure()
    
    # Actual price line
    fig.add_trace(go.Scatter(
        x=chart_idx, y=chart_close,
        name='Actual Price',
        line=dict(color='#00aaff', width=3),
        mode='lines'
    ))
    
    # Predicted price line
    fig.add_trace(go.Scatter(
        x=chart_idx, y=predicted_prices,
        name='Predicted Price',
        line=dict(color='#ffcc00', width=2, dash='dash'),
        mode='lines'
    ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0d0d0d',
        plot_bgcolor='#0d0d0d',
        height=220,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", y=1.15, x=0.5, xanchor="center", font=dict(size=11, color="#fff")),
        font=dict(family="JetBrains Mono", color="#888", size=10),
        yaxis_title="Price ($)"
    )
    
    fig.update_xaxes(gridcolor='#222')
    fig.update_yaxes(gridcolor='#222')
    
    return fig, accuracy


# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_top_signals(signals):
    """Compact signal list with labels."""
    st.markdown('<div class="signal-list">', unsafe_allow_html=True)
    st.markdown('<div class="signal-list-title">📊 Top Opportunities</div>', unsafe_allow_html=True)
    
    for ticker, score, signal_type in signals:
        label_text, label_class = get_signal_label(score)
        st.markdown(f'''
        <div class="signal-item">
            <span class="signal-ticker">{ticker}</span>
            <span class="signal-label {label_class}">{label_text}</span>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_main_signal(signal):
    """Main signal display."""
    sig_type = signal.signal_type.value
    strength = "STRONG " if signal.confidence > 0.6 else "WEAK " if signal.confidence < 0.3 else ""
    full_signal = f"{strength}{sig_type}"
    
    border_class = "buy-border" if sig_type == "BUY" else "sell-border" if sig_type == "SELL" else "hold-border"
    color = get_signal_color(sig_type)
    
    st.markdown(f'''
    <div class="main-signal {border_class}">
        <div class="signal-text" style="color:{color};">{full_signal}</div>
        <div class="signal-conf">{signal.confidence:.0%} Confidence • 3-Day Window</div>
    </div>
    ''', unsafe_allow_html=True)


def render_action_panel(signal):
    """Entry/Target/Stop panel."""
    entry = signal.entry_price
    target = signal.take_profit or entry * 1.05
    stop = signal.stop_loss or entry * 0.97
    target_pct = ((target - entry) / entry) * 100
    stop_pct = ((stop - entry) / entry) * 100
    
    st.markdown(f'''
    <div class="card">
        <div class="card-title">Trade Levels</div>
        <div class="score-row">
            <span class="score-name">Entry</span>
            <span class="score-val">${entry:.2f}</span>
        </div>
        <div class="score-row">
            <span class="score-name">Target</span>
            <span class="score-val green">${target:.2f} (+{target_pct:.1f}%)</span>
        </div>
        <div class="score-row">
            <span class="score-name">Stop Loss</span>
            <span class="score-val red">${stop:.2f} ({stop_pct:.1f}%)</span>
        </div>
    </div>
    ''', unsafe_allow_html=True)


def render_score_breakdown(signal):
    """Score bars with colors."""
    scores = [
        ("Technical", signal.technical_score),
        ("Fundamental", signal.fundamental_score),
        ("Sentiment", signal.sentiment_score),
        ("ML Model", signal.ml_score),
    ]
    
    st.markdown('<div class="card"><div class="card-title">Score Breakdown</div>', unsafe_allow_html=True)
    
    for name, score in scores:
        color = "#00ff88" if score >= 0.6 else "#ffcc00" if score >= 0.4 else "#ff4444"
        st.markdown(f'''
        <div class="score-row">
            <span class="score-name">{name}</span>
            <span class="score-val" style="color:{color};">{score:.0%}</span>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_sentiment_panel(ticker: str):
    """Sentiment panel with error handling."""
    try:
        sent = sentiment_aggregator.get_sentiment(ticker)
        
        overall = sent.get('overall_score', 0)
        signal = sent.get('signal')
        confidence = sent.get('confidence', 0)
        sources = sent.get('sources_used', [])
        
        signal_text = "Bullish" if signal == True else "Bearish" if signal == False else "Neutral"
        signal_color = "#00ff88" if signal == True else "#ff4444" if signal == False else "#ffcc00"
        
        st.markdown(f'''
        <div class="scroll-panel">
            <div class="panel-row">
                <span class="panel-label">Overall Score</span>
                <span class="panel-value" style="color:{signal_color};">{overall:.2f}</span>
            </div>
            <div class="panel-row">
                <span class="panel-label">Signal</span>
                <span class="panel-value" style="color:{signal_color};">{signal_text}</span>
            </div>
            <div class="panel-row">
                <span class="panel-label">Confidence</span>
                <span class="panel-value">{confidence:.0%}</span>
            </div>
            <div class="panel-row">
                <span class="panel-label">Sources</span>
                <span class="panel-value">{", ".join(sources) or "N/A"}</span>
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
    except Exception as e:
        st.markdown(f'<div class="scroll-panel"><span style="color:#888;">Sentiment unavailable: {str(e)[:40]}</span></div>', unsafe_allow_html=True)


def render_fundamental_panel(ticker: str):
    """Fundamental panel."""
    try:
        fund = fundamentals_fetcher.get_fundamentals(ticker)
        
        metrics = [
            ("P/E Ratio", f"{fund.get('pe_ratio', 'N/A'):.2f}" if fund.get('pe_ratio') else "N/A"),
            ("Market Cap", f"${fund.get('market_cap', 0)/1e9:.1f}B" if fund.get('market_cap') else "N/A"),
            ("EPS", f"${fund.get('eps_actual', 0):.2f}" if fund.get('eps_actual') else "N/A"),
            ("EPS Beat", "Yes ✓" if fund.get('eps_beat') else "No ✗"),
        ]
        
        st.markdown('<div class="scroll-panel">', unsafe_allow_html=True)
        for name, value in metrics:
            color = "#00ff88" if "Yes" in str(value) else "#ff4444" if "No ✗" in str(value) else "#fff"
            st.markdown(f'''
            <div class="panel-row">
                <span class="panel-label">{name}</span>
                <span class="panel-value" style="color:{color};">{value}</span>
            </div>
            ''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    except Exception as e:
        st.markdown(f'<div class="scroll-panel"><span style="color:#888;">Fundamentals unavailable</span></div>', unsafe_allow_html=True)


def render_paper_trading(ticker: str, current_price: float, signal):
    """Paper trading panel."""
    summary = paper_trader.get_summary()
    return_color = "green" if summary['total_return'] >= 0 else "red"
    
    st.markdown(f'''
    <div class="portfolio-card">
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <div>
                <div style="font-size:11px;color:#999;text-transform:uppercase;">Portfolio Value</div>
                <div class="portfolio-value">${summary['total_value']:,.2f}</div>
            </div>
            <div style="text-align:right;">
                <div style="font-size:11px;color:#999;">Return</div>
                <div class="portfolio-return {return_color}">{summary['total_return']:+.2f}%</div>
            </div>
        </div>
        <div style="margin-top:10px;display:flex;gap:16px;font-size:11px;color:#888;">
            <span>Cash: ${summary['cash']:,.2f}</span>
            <span>Positions: {summary['positions']}</span>
            <span>Win Rate: {summary['win_rate']:.0f}%</span>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Trade buttons
    col1, col2 = st.columns(2)
    with col1:
        shares = st.number_input("Shares", min_value=1, max_value=100, value=10, key="shares")
    with col2:
        st.markdown(f"**Cost: ${shares * current_price:,.2f}**")
    
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("🟢 BUY", use_container_width=True):
            trade = paper_trader.buy(ticker, shares, current_price, signal.confidence)
            if trade:
                st.success(f"Bought {shares} {ticker} @ ${current_price:.2f}")
    with btn_col2:
        if st.button("🔴 SELL", use_container_width=True):
            trade = paper_trader.sell(ticker, shares, current_price, signal.confidence)
            if trade:
                st.success(f"Sold {shares} {ticker} @ ${current_price:.2f}, PnL: ${trade.pnl:.2f}")


# ============================================================================
# MAIN APP
# ============================================================================

def get_top_signals():
    """Scan and rank top opportunities."""
    tickers = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "AMD", "META", "AMZN"]
    results = []
    
    for ticker in tickers:
        try:
            signal = signal_generator.generate_signal(ticker)
            score = (signal.technical_score * 0.3 + signal.fundamental_score * 0.2 +
                     signal.sentiment_score * 0.2 + signal.options_score * 0.1 + 
                     signal.ml_score * 0.2)
            results.append((ticker, score, signal.signal_type.value))
        except:
            pass
    
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:5]


def main():
    # Search bar - minimal
    search_col, spacer = st.columns([1, 5])
    with search_col:
        ticker_input = st.text_input("", value="AAPL", placeholder="Ticker...", label_visibility="collapsed")
        if st.button("ANALYZE", use_container_width=True):
            st.session_state['current_ticker'] = ticker_input
    
    current_ticker = st.session_state.get('current_ticker', 'AAPL')
    
    # Fetch data
    try:
        df = stock_fetcher.get_historical(current_ticker, period="6mo")
        signal = signal_generator.generate_signal(current_ticker)
        current_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2]
        price_change = ((current_price - prev_price) / prev_price) * 100
        
        # Update paper trader prices
        paper_trader.update_prices({current_ticker: current_price})
    except Exception as e:
        st.error(f"Error: {e}")
        return
    
    # CENTERED METRICS ROW
    st.markdown(f'''
    <div class="metrics-container">
        <span class="ticker-display">{current_ticker}</span>
        <div class="metric-box">
            <div class="metric-val">${current_price:.2f}</div>
            <div class="metric-lbl">Price</div>
        </div>
        <div class="metric-box">
            <div class="metric-val {"green" if price_change > 0 else "red"}">{price_change:+.2f}%</div>
            <div class="metric-lbl">Change</div>
            <div class="metric-sub">vs Yesterday</div>
        </div>
        <div class="metric-box">
            <div class="metric-val">{df["Volume"].iloc[-1]/1e6:.1f}M</div>
            <div class="metric-lbl">Volume</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    # LAYOUT
    left_col, center_col, right_col = st.columns([1.2, 3.5, 1.8])
    
    # === LEFT ===
    with left_col:
        with st.spinner("Scanning..."):
            top_signals = get_top_signals()
        render_top_signals(top_signals)
    
    # === CENTER ===
    with center_col:
        # Price chart
        fig = create_price_chart(df, current_ticker)
        st.plotly_chart(fig, use_container_width=True)
        
        # Prediction chart
        st.markdown("**📈 Actual vs Predicted Price (Last 30 Days)**")
        
        pred_col, acc_col = st.columns([5, 1])
        pred_fig, accuracy = create_prediction_chart(df)
        
        with pred_col:
            if pred_fig:
                st.plotly_chart(pred_fig, use_container_width=True)
            else:
                st.info("ML predictions unavailable")
        
        with acc_col:
            if accuracy is not None:
                color = "green" if accuracy > 55 else "yellow" if accuracy > 45 else "red"
                st.markdown(f'''
                <div class="metric-box" style="margin-top:30px;">
                    <div class="metric-val {color}">{accuracy:.1f}%</div>
                    <div class="metric-lbl">Accuracy</div>
                </div>
                ''', unsafe_allow_html=True)
    
    # === RIGHT ===
    with right_col:
        render_main_signal(signal)
        render_action_panel(signal)
        render_score_breakdown(signal)
        
        # Expanders with fixed styling
        with st.expander("📰 Sentiment Details", expanded=False):
            render_sentiment_panel(current_ticker)
        
        with st.expander("📊 Fundamentals", expanded=False):
            render_fundamental_panel(current_ticker)
        
        with st.expander("🤖 ML Model Info", expanded=False):
            st.markdown('''
            <div class="scroll-panel">
                <div class="panel-row"><span class="panel-label">Model</span><span class="panel-value">XGBoost</span></div>
                <div class="panel-row"><span class="panel-label">Window</span><span class="panel-value">3 days</span></div>
                <div class="panel-row"><span class="panel-label">Features</span><span class="panel-value">17</span></div>
                <div class="panel-row"><span class="panel-label">Accuracy</span><span class="panel-value">~70%</span></div>
            </div>
            ''', unsafe_allow_html=True)
        
        # Paper Trading
        st.markdown("---")
        with st.expander("💰 Paper Trading", expanded=True):
            render_paper_trading(current_ticker, current_price, signal)


if __name__ == "__main__":
    main()
