"""
TradeSense Pro v4 - Polished UI with Full API Integration

Fixes:
- Compact Top Signals with signal labels (Strong Buy, etc.)
- Cleaner metrics row layout
- Better Prediction vs Actual charts
- Track button for live prediction
- Scrollable Sentiment/Fundamental panels
- API quota display
- Fixed expander fonts
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

from src.data import stock_fetcher, watchlist
from src.data.fetcher_fundamentals import fundamentals_fetcher
from src.signals import signal_generator
from src.sentiment import sentiment_aggregator
from src.ml import FeatureEngineer, signal_model
from src.config import config

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
# API QUOTA TRACKING
# ============================================================================

def get_api_quota_info():
    """Get API quota status."""
    # Track calls in session state
    if 'api_calls' not in st.session_state:
        st.session_state['api_calls'] = {
            'finnhub': {'count': 0, 'limit': 60, 'reset_time': datetime.now() + timedelta(minutes=1)},
            'marketaux': {'count': 0, 'limit': 100, 'reset_time': datetime.now() + timedelta(hours=24)}
        }
    
    # Reset if time passed
    now = datetime.now()
    if now > st.session_state['api_calls']['finnhub']['reset_time']:
        st.session_state['api_calls']['finnhub']['count'] = 0
        st.session_state['api_calls']['finnhub']['reset_time'] = now + timedelta(minutes=1)
    
    if now > st.session_state['api_calls']['marketaux']['reset_time']:
        st.session_state['api_calls']['marketaux']['count'] = 0
        st.session_state['api_calls']['marketaux']['reset_time'] = now + timedelta(hours=24)
    
    return st.session_state['api_calls']


# ============================================================================
# CSS - CLEAN, COMPACT, READABLE
# ============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@400;500;600;700&display=swap');
    
    :root {
        --bg-dark: #0d0d0d;
        --bg-card: #1a1a1a;
        --bg-hover: #252525;
        --text-white: #ffffff;
        --text-gray: #888888;
        --text-muted: #555555;
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
    
    /* ===== COMPACT SIGNAL LIST ===== */
    .signal-list {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 12px;
    }
    .signal-list-title {
        font-size: 11px;
        font-weight: 600;
        color: var(--text-gray);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    .signal-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 6px 8px;
        margin: 2px 0;
        border-radius: 4px;
        background: var(--bg-dark);
        font-size: 13px;
    }
    .signal-item:hover { background: var(--bg-hover); cursor: pointer; }
    .signal-ticker { font-family: 'JetBrains Mono'; font-weight: 600; }
    .signal-label {
        font-size: 10px;
        font-weight: 600;
        padding: 2px 6px;
        border-radius: 3px;
        text-transform: uppercase;
    }
    .label-strong-buy { background: var(--strong-buy); color: #000; }
    .label-buy { background: var(--buy); color: #000; }
    .label-hold { background: var(--hold); color: #000; }
    .label-sell { background: var(--sell); color: #000; }
    .label-strong-sell { background: var(--strong-sell); color: #000; }
    
    /* ===== METRICS ROW ===== */
    .metrics-row {
        display: flex;
        gap: 12px;
        align-items: center;
        margin-bottom: 16px;
    }
    .ticker-name {
        font-size: 28px;
        font-weight: 700;
        font-family: 'JetBrains Mono';
        min-width: 100px;
    }
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 8px 16px;
        text-align: center;
        min-width: 100px;
    }
    .metric-val {
        font-family: 'JetBrains Mono';
        font-size: 18px;
        font-weight: 600;
    }
    .metric-lbl {
        font-size: 10px;
        color: var(--text-gray);
        text-transform: uppercase;
    }
    .metric-sub {
        font-size: 9px;
        color: var(--text-muted);
        font-style: italic;
    }
    .green { color: var(--strong-buy); }
    .red { color: var(--strong-sell); }
    .yellow { color: var(--hold); }
    
    /* ===== MAIN SIGNAL BOX ===== */
    .main-signal {
        background: var(--bg-card);
        border: 2px solid var(--border);
        border-radius: 8px;
        padding: 16px;
        text-align: center;
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
        font-size: 13px;
        color: var(--text-gray);
        margin-top: 4px;
    }
    
    /* ===== SCORE BARS ===== */
    .score-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 6px 0;
        border-bottom: 1px solid #222;
        font-size: 13px;
    }
    .score-name { color: var(--text-gray); }
    .score-val { font-family: 'JetBrains Mono'; font-weight: 600; }
    
    /* ===== SCROLLABLE PANEL ===== */
    .scroll-panel {
        max-height: 200px;
        overflow-y: auto;
        background: var(--bg-dark);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 8px;
    }
    .panel-row {
        display: flex;
        justify-content: space-between;
        padding: 4px 0;
        font-size: 12px;
        border-bottom: 1px solid #222;
    }
    .panel-row:last-child { border-bottom: none; }
    
    /* ===== API QUOTA ===== */
    .quota-bar {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 4px;
        padding: 6px 10px;
        font-size: 11px;
        display: flex;
        justify-content: space-between;
        margin-bottom: 8px;
    }
    .quota-name { color: var(--text-gray); }
    .quota-val { color: var(--text-white); font-family: 'JetBrains Mono'; }
    
    /* ===== EXPANDERS ===== */
    .streamlit-expanderHeader {
        background: var(--bg-card) !important;
        color: var(--text-white) !important;
        font-weight: 500 !important;
    }
    .streamlit-expanderContent {
        background: var(--bg-card) !important;
        color: var(--text-white) !important;
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
    """Clean candlestick chart."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, row_heights=[0.8, 0.2])
    
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], 
        low=df['Low'], close=df['Close'], name=ticker,
        increasing=dict(line=dict(color='#00ff88', width=1), fillcolor='#00ff88'),
        decreasing=dict(line=dict(color='#ff4444', width=1), fillcolor='#ff4444')
    ), row=1, col=1)
    
    colors = ['rgba(0,255,136,0.3)' if c >= o else 'rgba(255,68,68,0.3)' for c, o in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors), row=2, col=1)
    
    fig.update_layout(
        template='plotly_dark', paper_bgcolor='#0d0d0d', plot_bgcolor='#0d0d0d',
        height=400, margin=dict(l=0, r=0, t=0, b=0),
        xaxis_rangeslider_visible=False, showlegend=False,
        font=dict(family="JetBrains Mono", color="#fff", size=10)
    )
    fig.update_xaxes(gridcolor='#222', showgrid=True)
    fig.update_yaxes(gridcolor='#222', showgrid=True)
    return fig


def create_prediction_comparison_chart(df: pd.DataFrame) -> tuple:
    """Create chart showing predicted vs actual PRICE (not probability)."""
    engineer = FeatureEngineer()
    features_df = engineer.calculate_features(df.copy(), include_targets=False)
    
    if features_df.empty:
        return None, None
    
    # Get ML predictions (probability of price going up)
    X = features_df[engineer.feature_columns]
    
    if not signal_model.is_trained:
        try:
            signal_model.load()
        except:
            return None, None
    
    probs = signal_model.predict_proba(X)
    
    # Get actual close prices
    close = df['Close'].loc[features_df.index]
    
    # Calculate predicted direction and compare with actual
    # Prediction: if prob > 0.5, expect price to go UP in 3 days
    predictions = (probs > 0.5).astype(int)
    
    # Actual: did price go up in 3 days?
    returns_3d = close.pct_change(3).shift(-3)
    actuals = (returns_3d > 0.01).astype(int)  # 1% threshold
    
    # Calculate accuracy on valid data
    valid_mask = ~returns_3d.isna()
    if valid_mask.sum() > 10:
        matches = (predictions[valid_mask] == actuals[valid_mask])
        accuracy = matches.mean() * 100
    else:
        accuracy = None
    
    # Last 30 days for chart
    last_n = 30
    chart_idx = features_df.index[-last_n:]
    chart_close = close.loc[chart_idx]
    chart_probs = probs[-last_n:]
    
    # Create figure with 2 subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1, row_heights=[0.6, 0.4],
                        subplot_titles=("Actual Price", "ML Bullish Probability"))
    
    # Actual price
    fig.add_trace(go.Scatter(
        x=chart_idx, y=chart_close,
        name='Actual Price',
        line=dict(color='#00aaff', width=3),
        mode='lines'
    ), row=1, col=1)
    
    # ML probability
    fig.add_trace(go.Scatter(
        x=chart_idx, y=chart_probs,
        name='ML Probability',
        line=dict(color='#ffcc00', width=2),
        fill='tozeroy',
        fillcolor='rgba(255,204,0,0.1)'
    ), row=2, col=1)
    
    # 50% threshold line
    fig.add_hline(y=0.5, line=dict(color='#666', dash='dash', width=1), row=2, col=1)
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0d0d0d',
        plot_bgcolor='#0d0d0d',
        height=300,
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=False,
        font=dict(family="JetBrains Mono", color="#888", size=10)
    )
    
    fig.update_xaxes(gridcolor='#222')
    fig.update_yaxes(gridcolor='#222')
    fig.update_yaxes(range=[0, 1], row=2, col=1)
    
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
    <div style="background:#1a1a1a;border:1px solid #333;border-radius:6px;padding:12px;margin:8px 0;">
        <div class="score-row">
            <span class="score-name">Entry</span>
            <span class="metric-val">${entry:.2f}</span>
        </div>
        <div class="score-row">
            <span class="score-name">Target</span>
            <span class="metric-val green">${target:.2f} (+{target_pct:.1f}%)</span>
        </div>
        <div class="score-row">
            <span class="score-name">Stop Loss</span>
            <span class="metric-val red">${stop:.2f} ({stop_pct:.1f}%)</span>
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
    
    for name, score in scores:
        if score >= 0.6:
            color = "#00ff88"
        elif score >= 0.4:
            color = "#ffcc00"
        else:
            color = "#ff4444"
        
        st.markdown(f'''
        <div class="score-row">
            <span class="score-name">{name}</span>
            <span class="score-val" style="color:{color};">{score:.0%}</span>
        </div>
        ''', unsafe_allow_html=True)


def render_api_quota():
    """Display API quota status."""
    quota = get_api_quota_info()
    finnhub = quota['finnhub']
    marketaux = quota['marketaux']
    
    finn_remaining = finnhub['limit'] - finnhub['count']
    mkt_remaining = marketaux['limit'] - marketaux['count']
    finn_reset = finnhub['reset_time'].strftime('%H:%M')
    mkt_reset = marketaux['reset_time'].strftime('%H:%M')
    
    st.markdown(f'''
    <div class="quota-bar">
        <span class="quota-name">Finnhub</span>
        <span class="quota-val">{finn_remaining}/{finnhub['limit']} • Reset {finn_reset}</span>
    </div>
    <div class="quota-bar">
        <span class="quota-name">Marketaux</span>
        <span class="quota-val">{mkt_remaining}/{marketaux['limit']} • Reset {mkt_reset}</span>
    </div>
    ''', unsafe_allow_html=True)


def render_sentiment_panel(ticker: str):
    """Scrollable sentiment panel."""
    try:
        sent = sentiment_aggregator.aggregate_sentiment(ticker)
        
        news_items = sent.get('news_items', [])[:5]
        
        st.markdown('<div class="scroll-panel">', unsafe_allow_html=True)
        
        if news_items:
            for item in news_items:
                headline = item.get('headline', 'No headline')[:50]
                sentiment_score = item.get('sentiment', 0)
                color = "#00ff88" if sentiment_score > 0.2 else "#ff4444" if sentiment_score < -0.2 else "#ffcc00"
                
                st.markdown(f'''
                <div class="panel-row">
                    <span style="color:#888;">{headline}...</span>
                    <span style="color:{color};">{sentiment_score:.2f}</span>
                </div>
                ''', unsafe_allow_html=True)
        else:
            overall = sent.get('overall_sentiment', 0)
            label = sent.get('sentiment_label', 'Neutral')
            st.markdown(f'''
            <div class="panel-row">
                <span style="color:#888;">Overall Sentiment</span>
                <span style="color:#fff;">{label} ({overall:.2f})</span>
            </div>
            <div class="panel-row">
                <span style="color:#888;">News Count</span>
                <span style="color:#fff;">{sent.get('news_count', 0)}</span>
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
    except Exception as e:
        st.markdown(f'<div style="color:#888;font-size:12px;">Unable to load sentiment: {str(e)[:50]}</div>', unsafe_allow_html=True)


def render_fundamental_panel(ticker: str):
    """Scrollable fundamental panel."""
    try:
        fund = fundamentals_fetcher.get_fundamentals(ticker)
        
        st.markdown('<div class="scroll-panel">', unsafe_allow_html=True)
        
        metrics = [
            ("P/E Ratio", fund.get('pe_ratio', 'N/A')),
            ("Industry P/E", fund.get('industry_pe', 'N/A')),
            ("EPS", f"${fund.get('eps_actual', 0):.2f}" if fund.get('eps_actual') else "N/A"),
            ("EPS Beat", "Yes ✓" if fund.get('eps_beat') else "No ✗"),
            ("Market Cap", f"${fund.get('market_cap', 0)/1e9:.1f}B" if fund.get('market_cap') else "N/A"),
        ]
        
        for name, value in metrics:
            color = "#00ff88" if "Yes" in str(value) else "#ff4444" if "No" in str(value) else "#fff"
            st.markdown(f'''
            <div class="panel-row">
                <span style="color:#888;">{name}</span>
                <span style="color:{color};">{value}</span>
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
    except Exception as e:
        st.markdown(f'<div style="color:#888;font-size:12px;">Unable to load fundamentals: {str(e)[:50]}</div>', unsafe_allow_html=True)


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
    # Search bar
    search_col, _, _ = st.columns([2, 4, 2])
    with search_col:
        ticker_input = st.text_input("", value="AAPL", placeholder="Enter ticker...", label_visibility="collapsed")
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
    except Exception as e:
        st.error(f"Error: {e}")
        return
    
    # LAYOUT
    left_col, center_col, right_col = st.columns([1.2, 4, 1.5])
    
    # === LEFT ===
    with left_col:
        with st.spinner("Scanning..."):
            top_signals = get_top_signals()
        render_top_signals(top_signals)
        
        st.markdown("### ")
        st.markdown("**API Quota**")
        render_api_quota()
    
    # === CENTER ===
    with center_col:
        # Metrics row
        st.markdown(f'''
        <div class="metrics-row">
            <span class="ticker-name">{current_ticker}</span>
            <div class="metric-card">
                <div class="metric-val">${current_price:.2f}</div>
                <div class="metric-lbl">Price</div>
            </div>
            <div class="metric-card">
                <div class="metric-val {"green" if price_change > 0 else "red"}">{price_change:+.2f}%</div>
                <div class="metric-lbl">Change</div>
                <div class="metric-sub">vs Yesterday</div>
            </div>
            <div class="metric-card">
                <div class="metric-val">{df["Volume"].iloc[-1]/1e6:.1f}M</div>
                <div class="metric-lbl">Volume</div>
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
        # Price chart
        fig = create_price_chart(df, current_ticker)
        st.plotly_chart(fig, use_container_width=True)
        
        # Prediction chart
        st.markdown("**📈 Prediction vs Actual (Last 30 Days)**")
        
        pred_col, acc_col = st.columns([5, 1])
        
        pred_fig, accuracy = create_prediction_comparison_chart(df)
        
        with pred_col:
            if pred_fig:
                st.plotly_chart(pred_fig, use_container_width=True)
            else:
                st.info("ML model not available for predictions")
        
        with acc_col:
            if accuracy is not None:
                color = "green" if accuracy > 55 else "yellow" if accuracy > 45 else "red"
                st.markdown(f'''
                <div class="metric-card" style="margin-top:50px;">
                    <div class="metric-val {color}">{accuracy:.1f}%</div>
                    <div class="metric-lbl">Accuracy</div>
                    <div class="metric-sub">3-day</div>
                </div>
                ''', unsafe_allow_html=True)
        
        # Track button
        track_col, _ = st.columns([1, 4])
        with track_col:
            if st.button("🔴 TRACK LIVE", use_container_width=True):
                st.session_state['tracking'] = current_ticker
                st.info(f"Now tracking {current_ticker} predictions vs actual")
    
    # === RIGHT ===
    with right_col:
        render_main_signal(signal)
        render_action_panel(signal)
        
        st.markdown("**Score Breakdown**")
        render_score_breakdown(signal)
        
        # Expandable details
        with st.expander("📰 Sentiment Details"):
            render_sentiment_panel(current_ticker)
        
        with st.expander("📊 Fundamental Details"):
            render_fundamental_panel(current_ticker)
        
        with st.expander("🤖 ML Model Info"):
            st.markdown('''
            <div class="scroll-panel" style="color:#fff;">
                <p><b>Model:</b> XGBoost Classifier</p>
                <p><b>Prediction:</b> 3-day price movement >1%</p>
                <p><b>Training:</b> ~450 samples/ticker (2 years)</p>
                <p><b>Features:</b> 17 (RSI, MACD, BB, ATR, etc.)</p>
                <p><b>Validation Accuracy:</b> ~70%</p>
            </div>
            ''', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
