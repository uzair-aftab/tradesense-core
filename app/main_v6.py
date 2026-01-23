"""
TradeSense Pro v6 - Professional Trading Workstation

Features:
- Non-blocking live chart updates
- Timeframe-aware signal generation
- One-click trade execution
- Real-time position tracking
- Portfolio analytics
"""

import streamlit as st
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml import Timeframe, mtf_engine
from src.signals.timeframe_generator import timeframe_signal_generator, TradeSetup
from src.trading.trading_station import trading_station
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="TradeSense Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# STYLES
# ============================================================================
st.markdown("""
<style>
    /* Dark theme */
    .stApp { background-color: #0e1117; }
    
    /* Header */
    .main-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 10px 20px;
        background: linear-gradient(90deg, #1a1f2e 0%, #0e1117 100%);
        border-bottom: 1px solid #2d3748;
        margin-bottom: 15px;
    }
    
    .header-title {
        font-size: 24px;
        font-weight: bold;
        color: #00d4aa;
    }
    
    .header-ticker {
        font-size: 28px;
        font-weight: bold;
        color: white;
    }
    
    .header-price {
        font-size: 24px;
        color: white;
        margin-left: 15px;
    }
    
    .header-change.positive { color: #00d4aa; }
    .header-change.negative { color: #ff4757; }
    
    .header-portfolio {
        text-align: right;
        color: #a0a0a0;
    }
    
    .header-portfolio-value {
        font-size: 20px;
        font-weight: bold;
        color: white;
    }
    
    /* Signal Card */
    .signal-card {
        background: #1a1f2e;
        border-radius: 12px;
        padding: 20px;
        border: 2px solid #2d3748;
    }
    
    .signal-card.long { border-color: #00d4aa; }
    .signal-card.short { border-color: #ff4757; }
    .signal-card.hold { border-color: #ffa502; }
    
    .signal-direction {
        font-size: 28px;
        font-weight: bold;
        text-align: center;
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    
    .signal-direction.long { background: rgba(0, 212, 170, 0.2); color: #00d4aa; }
    .signal-direction.short { background: rgba(255, 71, 87, 0.2); color: #ff4757; }
    .signal-direction.hold { background: rgba(255, 165, 2, 0.2); color: #ffa502; }
    
    .signal-level {
        display: flex;
        justify-content: space-between;
        padding: 8px 0;
        border-bottom: 1px solid #2d3748;
    }
    
    .signal-label { color: #a0a0a0; }
    .signal-value { color: white; font-weight: bold; }
    .signal-value.target { color: #00d4aa; }
    .signal-value.stop { color: #ff4757; }
    
    /* Positions Table */
    .positions-header {
        font-size: 18px;
        font-weight: bold;
        color: white;
        margin: 15px 0 10px;
    }
    
    .position-row {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr 1fr 1fr 100px;
        padding: 10px;
        border-bottom: 1px solid #2d3748;
        align-items: center;
    }
    
    .position-row:hover { background: #1a1f2e; }
    
    .pnl-positive { color: #00d4aa; }
    .pnl-negative { color: #ff4757; }
    
    /* Analytics Footer */
    .analytics-bar {
        display: flex;
        justify-content: space-around;
        padding: 15px;
        background: #1a1f2e;
        border-radius: 8px;
        margin-top: 15px;
    }
    
    .analytics-item {
        text-align: center;
    }
    
    .analytics-value {
        font-size: 18px;
        font-weight: bold;
        color: white;
    }
    
    .analytics-label {
        font-size: 12px;
        color: #a0a0a0;
    }
    
    /* Live indicator */
    .live-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        background: #ff4757;
        border-radius: 50%;
        animation: pulse 1.5s infinite;
        margin-right: 5px;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }
    
    /* Hide default elements */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# CHART FUNCTIONS
# ============================================================================
def create_candlestick_chart(df: pd.DataFrame, setup: TradeSetup = None) -> go.Figure:
    """Create professional candlestick chart with entry/target/stop lines."""
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25]
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            increasing_line_color='#00d4aa',
            decreasing_line_color='#ff4757',
        ),
        row=1, col=1
    )
    
    # Add entry/target/stop lines if we have a setup
    if setup and setup.direction != "HOLD":
        # Entry line
        fig.add_hline(
            y=setup.entry_price, 
            line_dash="solid", 
            line_color="white",
            annotation_text=f"Entry: ${setup.entry_price:.2f}",
            row=1
        )
        # Target line
        fig.add_hline(
            y=setup.target_price, 
            line_dash="dash", 
            line_color="#00d4aa",
            annotation_text=f"Target: ${setup.target_price:.2f}",
            row=1
        )
        # Stop line
        fig.add_hline(
            y=setup.stop_loss, 
            line_dash="dash", 
            line_color="#ff4757",
            annotation_text=f"Stop: ${setup.stop_loss:.2f}",
            row=1
        )
    
    # Volume
    colors = ['#00d4aa' if c >= o else '#ff4757' 
              for c, o in zip(df['Close'], df['Open'])]
    
    fig.add_trace(
        go.Bar(
            x=df.index, 
            y=df['Volume'],
            marker_color=colors,
            opacity=0.7,
            name='Volume'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0e1117',
        plot_bgcolor='#0e1117',
        height=450,
        margin=dict(l=50, r=50, t=10, b=30),
        xaxis_rangeslider_visible=False,
        showlegend=False,
    )
    
    fig.update_xaxes(gridcolor='#1a1f2e')
    fig.update_yaxes(gridcolor='#1a1f2e')
    
    return fig


# ============================================================================
# UI COMPONENTS
# ============================================================================
def render_header(ticker: str, price: float, change_pct: float, portfolio_value: float):
    """Render the top header bar."""
    change_class = "positive" if change_pct >= 0 else "negative"
    return_pct = ((portfolio_value - 10000) / 10000) * 100
    
    st.markdown(f"""
    <div class="main-header">
        <div style="display: flex; align-items: center;">
            <span class="header-title">📈 TradeSense Pro</span>
        </div>
        <div style="display: flex; align-items: center;">
            <span class="header-ticker">{ticker}</span>
            <span class="header-price">${price:.2f}</span>
            <span class="header-change {change_class}" style="margin-left: 10px;">{change_pct:+.2f}%</span>
        </div>
        <div class="header-portfolio">
            <div>Portfolio</div>
            <div class="header-portfolio-value">${portfolio_value:,.2f}</div>
            <div class="header-change {change_class}">{return_pct:+.2f}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_signal_card(setup: TradeSetup):
    """Render the signal card with trade details."""
    
    direction_class = setup.direction.lower()
    
    st.markdown(f"""
    <div class="signal-card {direction_class}">
        <div class="signal-direction {direction_class}">
            {'🟢' if setup.direction == 'LONG' else '🔴' if setup.direction == 'SHORT' else '⚪'} 
            {setup.direction}
        </div>
        
        <div class="signal-level">
            <span class="signal-label">Entry Price</span>
            <span class="signal-value">${setup.entry_price:.2f}</span>
        </div>
        
        <div class="signal-level">
            <span class="signal-label">Target</span>
            <span class="signal-value target">${setup.target_price:.2f} ({setup.target_pct:+.1f}%)</span>
        </div>
        
        <div class="signal-level">
            <span class="signal-label">Stop Loss</span>
            <span class="signal-value stop">${setup.stop_loss:.2f} ({setup.stop_pct:+.1f}%)</span>
        </div>
        
        <div class="signal-level">
            <span class="signal-label">Risk/Reward</span>
            <span class="signal-value">1:{setup.risk_reward_ratio:.1f}</span>
        </div>
        
        <div class="signal-level">
            <span class="signal-label">Confidence</span>
            <span class="signal-value">{setup.confidence:.0%}</span>
        </div>
        
        <div class="signal-level">
            <span class="signal-label">Duration</span>
            <span class="signal-value">{setup.duration}</span>
        </div>
        
        <div class="signal-level">
            <span class="signal-label">Position Size</span>
            <span class="signal-value">{setup.position_size} shares</span>
        </div>
        
        <div class="signal-level">
            <span class="signal-label">Risk Amount</span>
            <span class="signal-value">${setup.risk_amount:.2f}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_positions_table():
    """Render open positions table."""
    positions = trading_station.positions
    
    st.markdown('<div class="positions-header">📊 Open Positions</div>', unsafe_allow_html=True)
    
    if not positions:
        st.info("No open positions. Generate a signal and click BUY to open a position.")
        return
    
    for pos in positions:
        pnl_class = "pnl-positive" if pos.unrealized_pnl >= 0 else "pnl-negative"
        
        cols = st.columns([2, 1, 1, 1, 1, 1])
        with cols[0]:
            st.write(f"**{pos.ticker}** ({pos.direction})")
        with cols[1]:
            st.write(f"{pos.quantity} shares")
        with cols[2]:
            st.write(f"${pos.entry_price:.2f}")
        with cols[3]:
            st.write(f"${pos.current_price:.2f}")
        with cols[4]:
            color = "#00d4aa" if pos.unrealized_pnl >= 0 else "#ff4757"
            st.markdown(f'<span style="color: {color}; font-weight: bold;">${pos.unrealized_pnl:+.2f}</span>', unsafe_allow_html=True)
        with cols[5]:
            if st.button("CLOSE", key=f"close_{pos.id}"):
                trading_station.close_position(pos.id, "MANUAL")
                st.rerun()


def render_analytics_bar():
    """Render portfolio analytics footer."""
    analytics = trading_station.get_analytics()
    
    st.markdown(f"""
    <div class="analytics-bar">
        <div class="analytics-item">
            <div class="analytics-value">${analytics['closed_pnl']:+.2f}</div>
            <div class="analytics-label">Realized P&L</div>
        </div>
        <div class="analytics-item">
            <div class="analytics-value">${analytics['open_pnl']:+.2f}</div>
            <div class="analytics-label">Unrealized P&L</div>
        </div>
        <div class="analytics-item">
            <div class="analytics-value">{analytics['win_rate']:.0f}%</div>
            <div class="analytics-label">Win Rate</div>
        </div>
        <div class="analytics-item">
            <div class="analytics-value">{analytics['total_trades']}</div>
            <div class="analytics-label">Total Trades</div>
        </div>
        <div class="analytics-item">
            <div class="analytics-value">{analytics['profit_factor']:.1f}</div>
            <div class="analytics-label">Profit Factor</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# MAIN APP
# ============================================================================
def main():
    # Non-blocking auto-refresh (runs in background)
    st_autorefresh(interval=15000, limit=None, key="live_refresh")
    
    # Initialize session state
    if 'current_ticker' not in st.session_state:
        st.session_state['current_ticker'] = 'AAPL'
    if 'timeframe' not in st.session_state:
        st.session_state['timeframe'] = Timeframe.H1
    
    # === TOP CONTROLS ===
    control_cols = st.columns([2, 2, 1, 5])
    
    with control_cols[0]:
        ticker_input = st.text_input(
            "Ticker", 
            value=st.session_state['current_ticker'],
            label_visibility="collapsed",
            placeholder="Enter ticker..."
        )
        if ticker_input != st.session_state['current_ticker']:
            st.session_state['current_ticker'] = ticker_input.upper()
    
    with control_cols[1]:
        timeframe_map = {
            "5 Min": Timeframe.M5,
            "15 Min": Timeframe.M15,
            "1 Hour": Timeframe.H1,
            "4 Hour": Timeframe.H4,
            "Daily": Timeframe.D1,
        }
        tf_label = st.selectbox(
            "Timeframe",
            list(timeframe_map.keys()),
            index=2,
            label_visibility="collapsed"
        )
        st.session_state['timeframe'] = timeframe_map[tf_label]
    
    with control_cols[2]:
        st.markdown('<span class="live-dot"></span> LIVE', unsafe_allow_html=True)
    
    ticker = st.session_state['current_ticker']
    timeframe = st.session_state['timeframe']
    
    # === FETCH DATA & GENERATE SIGNAL ===
    try:
        df = mtf_engine.get_data(ticker, timeframe, use_cache=False)
        
        if df.empty or len(df) < 10:
            st.error(f"No data available for {ticker}")
            return
        
        current_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2] if len(df) > 1 else current_price
        change_pct = ((current_price - prev_price) / prev_price) * 100
        
        # Generate timeframe-aware signal
        setup = timeframe_signal_generator.generate(ticker, timeframe)
        setup.set_portfolio_context(trading_station.total_value)
        
        # Update position prices
        trading_station.update_prices({ticker: current_price})
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        logger.error(f"Data error: {e}")
        return
    
    # === HEADER ===
    render_header(ticker, current_price, change_pct, trading_station.total_value)
    
    # === MAIN CONTENT: CHART + SIGNAL ===
    chart_col, signal_col = st.columns([7, 3])
    
    with chart_col:
        fig = create_candlestick_chart(df, setup)
        st.plotly_chart(fig, use_container_width=True)
    
    with signal_col:
        render_signal_card(setup)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Trade execution buttons
        btn_cols = st.columns(2)
        with btn_cols[0]:
            if st.button("🟢 BUY", use_container_width=True, disabled=setup.direction == "HOLD"):
                position = trading_station.execute(setup)
                if position:
                    st.success(f"Opened {position.direction} {position.quantity} {position.ticker}")
                    st.rerun()
        
        with btn_cols[1]:
            if st.button("🔴 SELL ALL", use_container_width=True):
                for pos in trading_station.positions[:]:
                    trading_station.close_position(pos.id, "MANUAL")
                st.rerun()
        
        # Reasons
        if setup.reasons:
            st.markdown("---")
            st.markdown("**Signal Reasoning:**")
            for reason in setup.reasons[:5]:
                st.markdown(f"• {reason}")
    
    # === POSITIONS TABLE ===
    st.markdown("---")
    render_positions_table()
    
    # === ANALYTICS BAR ===
    render_analytics_bar()


if __name__ == "__main__":
    main()
