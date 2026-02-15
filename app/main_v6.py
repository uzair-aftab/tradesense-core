"""
TradeSense Pro v6 - Professional Trading Workstation

Features:
- Non-blocking live chart updates
- Timeframe-aware signal generation
- One-click trade execution
- Real-time position tracking
- Portfolio analytics
- Prediction chart (actual vs predicted)
"""

import streamlit as st
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml import (
    Timeframe,
    mtf_engine,
    ensemble_model,
    FeatureEngineer,
    unified_predictor,
    model_registry,
)
from src.ml.models import signal_model
from src.ml.lstm_model import lstm_model
from src.ml.transformer_model import transformer_model
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
    .stApp { background-color: #0e1117; }
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 1rem; }
    
    .metric-card {
        background: #1a1f2e;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
    }
    
    .live-indicator {
        color: #ff4757;
        font-weight: bold;
    }
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
        fig.add_hline(y=setup.entry_price, line_dash="solid", line_color="white", row=1)
        fig.add_hline(y=setup.target_price, line_dash="dash", line_color="#00d4aa", row=1)
        fig.add_hline(y=setup.stop_loss, line_dash="dash", line_color="#ff4757", row=1)
    
    # Volume
    colors = ['#00d4aa' if c >= o else '#ff4757' for c, o in zip(df['Close'], df['Open'])]
    fig.add_trace(
        go.Bar(x=df.index, y=df['Volume'], marker_color=colors, opacity=0.7, name='Volume'),
        row=2, col=1
    )
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0e1117',
        plot_bgcolor='#0e1117',
        height=400,
        margin=dict(l=50, r=50, t=10, b=30),
        xaxis_rangeslider_visible=False,
        showlegend=False,
    )
    
    fig.update_xaxes(gridcolor='#1a1f2e')
    fig.update_yaxes(gridcolor='#1a1f2e')
    
    return fig


def create_prediction_chart(df: pd.DataFrame, predictions: np.ndarray = None) -> go.Figure:
    """Create chart showing actual prices vs ML predictions."""
    
    fig = go.Figure()
    
    # Actual prices (last 50 points)
    display_df = df.tail(50)
    
    fig.add_trace(go.Scatter(
        x=list(range(len(display_df))),
        y=display_df['Close'].values,
        mode='lines',
        name='Actual Price',
        line=dict(color='#00d4aa', width=2)
    ))
    
    # If we have predictions array, show them
    if predictions is not None:
        try:
            # Handle both scalar and array predictions
            if isinstance(predictions, (int, float, np.floating)):
                # Single prediction value - create a constant line
                pred_value = float(predictions)
                price_range = float(display_df['Close'].max() - display_df['Close'].min())
                pred_level = float(display_df['Close'].mean()) + (pred_value - 0.5) * price_range
                
                fig.add_hline(
                    y=pred_level, 
                    line_dash="dash", 
                    line_color="#ffa502",
                    annotation_text=f"ML: {pred_value:.0%} bullish"
                )
            elif hasattr(predictions, '__len__') and len(predictions) > 0:
                # Array of predictions
                price_range = display_df['Close'].max() - display_df['Close'].min()
                pred_scaled = display_df['Close'].mean() + (np.array(predictions)[-len(display_df):] - 0.5) * price_range
                
                fig.add_trace(go.Scatter(
                    x=list(range(len(pred_scaled))),
                    y=pred_scaled,
                    mode='lines',
                    name='ML Prediction Trend',
                    line=dict(color='#ffa502', width=2, dash='dash')
                ))
        except Exception as e:
            pass  # Silently ignore prediction errors
    
    # Add prediction arrow for next period
    last_price = display_df['Close'].iloc[-1]
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0e1117',
        plot_bgcolor='#0e1117',
        height=250,
        margin=dict(l=50, r=50, t=30, b=30),
        title=dict(text='Price vs ML Prediction', font=dict(color='white', size=14)),
        showlegend=True,
        legend=dict(orientation='h', y=1.1)
    )
    
    fig.update_xaxes(gridcolor='#1a1f2e', title='Periods')
    fig.update_yaxes(gridcolor='#1a1f2e', title='Price')
    
    return fig


# ============================================================================
# UI COMPONENTS - Using Streamlit native components
# ============================================================================
def render_signal_card_native(setup: TradeSetup):
    """Render signal card using native Streamlit components."""
    
    # Signal direction with color
    if setup.direction == "LONG":
        st.success(f"🟢 **{setup.direction}** Signal")
    elif setup.direction == "SHORT":
        st.error(f"🔴 **{setup.direction}** Signal")
    else:
        st.warning(f"⚪ **{setup.direction}** Signal")
    
    # Metrics in columns
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Entry", f"${setup.entry_price:.2f}")
        st.metric("Stop Loss", f"${setup.stop_loss:.2f}", f"{setup.stop_pct:+.1f}%")
        st.metric("R:R Ratio", f"1:{setup.risk_reward_ratio:.1f}")
    
    with col2:
        st.metric("Target", f"${setup.target_price:.2f}", f"{setup.target_pct:+.1f}%")
        st.metric("Confidence", f"{setup.confidence:.0%}")
        st.metric("Duration", setup.duration)
    
    st.divider()
    
    # Position sizing
    st.write(f"**Position Size:** {setup.position_size} shares")
    st.write(f"**Risk Amount:** ${setup.risk_amount:.2f}")


def render_positions_table():
    """Render open positions using native Streamlit."""
    st.subheader("📊 Open Positions")
    
    positions = trading_station.positions
    
    if not positions:
        st.info("No open positions. Generate a signal and click BUY to open a position.")
        return
    
    for pos in positions:
        with st.container():
            cols = st.columns([2, 1, 1, 1, 1, 1])
            with cols[0]:
                st.write(f"**{pos.ticker}** ({pos.direction})")
            with cols[1]:
                st.write(f"{pos.quantity} shares")
            with cols[2]:
                st.write(f"Entry: ${pos.entry_price:.2f}")
            with cols[3]:
                st.write(f"Current: ${pos.current_price:.2f}")
            with cols[4]:
                pnl_color = "green" if pos.unrealized_pnl >= 0 else "red"
                st.markdown(f":{pnl_color}[**${pos.unrealized_pnl:+.2f}**]")
            with cols[5]:
                if st.button("CLOSE", key=f"close_{pos.id}"):
                    trading_station.close_position(pos.id, "MANUAL")
                    st.rerun()


def render_analytics():
    """Render portfolio analytics."""
    analytics = trading_station.get_analytics()
    
    cols = st.columns(5)
    with cols[0]:
        st.metric("Portfolio Value", f"${analytics['total_value']:,.2f}")
    with cols[1]:
        st.metric("Realized P&L", f"${analytics['closed_pnl']:+.2f}")
    with cols[2]:
        st.metric("Open P&L", f"${analytics['open_pnl']:+.2f}")
    with cols[3]:
        st.metric("Win Rate", f"{analytics['win_rate']:.0f}%")
    with cols[4]:
        st.metric("Profit Factor", f"{analytics['profit_factor']:.1f}")


def render_trade_thesis_native(setup: TradeSetup):
    """Render a concise narrative explaining why this trade exists."""
    components = [
        ("Technical", setup.technical_score),
        ("Fundamental", setup.fundamental_score),
        ("Sentiment", setup.sentiment_score),
        ("ML", setup.ml_score),
    ]

    active_components = [(name, score) for name, score in components if score is not None]
    active_scores = [score for _, score in active_components]
    coverage = len(active_components) / len(components) if components else 0.0
    composite = float(np.mean(active_scores)) if active_scores else 0.5
    decision_strength = abs(composite - 0.5) * 2

    regime = "Unknown"
    for reason in setup.reasons:
        if reason.startswith("Regime:"):
            regime = reason.replace("Regime:", "").strip()
            break

    top_drivers = sorted(
        active_components,
        key=lambda item: abs(item[1] - 0.5),
        reverse=True
    )[:2]
    driver_text = ", ".join(f"{name} {score:.0%}" for name, score in top_drivers) if top_drivers else "N/A"

    thesis = "Neutral bias due to mixed factors"
    if setup.direction == "LONG":
        thesis = "Bullish bias: upside factors dominate"
    elif setup.direction == "SHORT":
        thesis = "Bearish bias: downside pressure dominates"

    st.subheader("🧠 Why This Trade")
    st.write(thesis)

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Regime", regime)
        st.metric("Composite", f"{composite:.0%}")
        st.metric("Coverage", f"{coverage:.0%}")
    with c2:
        st.metric("Decision Strength", f"{decision_strength:.0%}")
        st.metric("Risk:Reward", f"1:{setup.risk_reward_ratio:.2f}")
        st.metric("Duration", setup.duration)

    st.caption(f"Top drivers: {driver_text}")


def render_ml_metadata_panel(ml_info: dict):
    """Render model availability, ensemble composition, and registry context."""
    st.subheader("🤖 ML Metadata")

    prediction_source = ml_info.get("source", "N/A")
    probability = ml_info.get("probability")
    confidence = ml_info.get("confidence")

    pcol1, pcol2 = st.columns(2)
    with pcol1:
        st.metric("Prediction Source", prediction_source.upper())
        st.metric("ML Probability", f"{probability:.0%}" if probability is not None else "N/A")
    with pcol2:
        st.metric("ML Confidence", f"{confidence:.0%}" if confidence is not None else "N/A")
        st.metric("Registry Models", str(len(model_registry.models)))

    active_models = []
    if signal_model.is_trained:
        active_models.append("XGBoost")
    if lstm_model.is_trained:
        active_models.append("LSTM")
    if transformer_model.is_trained:
        active_models.append("Transformer")

    st.write("**Active models:** " + (", ".join(active_models) if active_models else "None"))

    try:
        best = model_registry.get_best_model(metric="accuracy")
        if best:
            st.caption(f"Best registered model: {best.model_name} v{best.version} • acc={best.accuracy:.1%}")
        else:
            st.caption("No registered model metrics yet.")
    except Exception:
        st.caption("Model registry metadata unavailable.")


# ============================================================================
# MAIN APP
# ============================================================================
def main():
    # Non-blocking auto-refresh
    st_autorefresh(interval=15000, limit=None, key="live_refresh")
    
    # Initialize session state
    if 'current_ticker' not in st.session_state:
        st.session_state['current_ticker'] = 'AAPL'
    if 'timeframe' not in st.session_state:
        st.session_state['timeframe'] = Timeframe.H1
    
    # === HEADER ===
    st.title("📈 TradeSense Pro")
    
    # === TOP CONTROLS ===
    col1, col2, col3, col4 = st.columns([2, 2, 1, 5])
    
    with col1:
        ticker_input = st.text_input(
            "Ticker", 
            value=st.session_state['current_ticker'],
            placeholder="Enter ticker..."
        )
        if ticker_input != st.session_state['current_ticker']:
            st.session_state['current_ticker'] = ticker_input.upper()
    
    with col2:
        timeframe_map = {
            "5 Min": Timeframe.M5,
            "15 Min": Timeframe.M15,
            "1 Hour": Timeframe.H1,
            "4 Hour": Timeframe.H4,
            "Daily": Timeframe.D1,
        }
        tf_label = st.selectbox("Timeframe", list(timeframe_map.keys()), index=2)
        st.session_state['timeframe'] = timeframe_map[tf_label]
    
    with col3:
        st.markdown('<p class="live-indicator">🔴 LIVE</p>', unsafe_allow_html=True)
    
    ticker = st.session_state['current_ticker']
    timeframe = st.session_state['timeframe']
    
    # === FETCH DATA & GENERATE SIGNAL ===
    try:
        df = mtf_engine.get_data(ticker, timeframe, use_cache=False)
        
        if df.empty or len(df) < 10:
            st.error(f"No data available for {ticker}")
            return
        
        current_price = float(df['Close'].iloc[-1])
        prev_price = float(df['Close'].iloc[-2]) if len(df) > 1 else current_price
        change_pct = ((current_price - prev_price) / prev_price) * 100
        
        # Generate timeframe-aware signal
        setup = timeframe_signal_generator.generate(ticker, timeframe)
        setup.set_portfolio_context(trading_station.total_value)
        
        # Get ML predictions for chart
        try:
            feature_engineer = FeatureEngineer()
            features_df = feature_engineer.calculate_features(df, include_targets=False)
            if not features_df.empty:
                predictions = ensemble_model.predict(df).get('probability', 0.5)
            else:
                predictions = None
        except:
            predictions = None

        # Unified predictor context for UI metadata
        ml_info = {"source": "none", "probability": None, "confidence": None}
        try:
            pred = unified_predictor.predict_with_fallback(df, preferred_models=["ensemble", "xgboost", "lstm"])
            if pred is not None:
                ml_info = {
                    "source": pred.model_type,
                    "probability": pred.probability,
                    "confidence": pred.confidence,
                }
        except Exception as e:
            logger.warning(f"Unified predictor UI context failed: {e}")
        
        # Update position prices
        trading_station.update_prices({ticker: current_price})
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        logger.error(f"Data error: {e}")
        return
    
    # === PRICE INFO ===
    price_cols = st.columns([1, 1, 1, 4])
    with price_cols[0]:
        st.metric("Current Price", f"${current_price:.2f}", f"{change_pct:+.2f}%")
    with price_cols[1]:
        st.metric("Portfolio", f"${trading_station.total_value:,.2f}")
    
    st.divider()
    
    # === MAIN CONTENT: CHART + SIGNAL ===
    chart_col, signal_col = st.columns([7, 3])
    
    with chart_col:
        # Tab for Price Chart and Prediction Chart
        tab1, tab2 = st.tabs(["📊 Price Chart", "🤖 ML Prediction"])
        
        with tab1:
            fig = create_candlestick_chart(df, setup)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            pred_fig = create_prediction_chart(df, predictions)
            st.plotly_chart(pred_fig, use_container_width=True)
            st.caption("Orange dashed line shows ML model's bullish/bearish trend prediction.")
    
    with signal_col:
        render_signal_card_native(setup)
        render_trade_thesis_native(setup)
        st.divider()
        render_ml_metadata_panel(ml_info)
        
        st.divider()
        
        # Trade execution buttons
        btn_cols = st.columns(2)
        with btn_cols[0]:
            buy_disabled = setup.direction == "HOLD"
            if st.button("🟢 BUY", use_container_width=True, disabled=buy_disabled, type="primary"):
                position = trading_station.execute(setup)
                if position:
                    st.success(f"Opened {position.direction} {position.quantity} {position.ticker}")
                    st.rerun()
                else:
                    st.warning("Could not execute trade. Check logs.")
        
        with btn_cols[1]:
            if st.button("🔴 SELL ALL", use_container_width=True):
                for pos in trading_station.positions[:]:
                    trading_station.close_position(pos.id, "MANUAL")
                st.rerun()
        
        # Reasons
        if setup.reasons:
            st.divider()
            st.write("**Signal Reasoning:**")
            for reason in setup.reasons[:5]:
                st.write(f"• {reason}")
    
    # === POSITIONS TABLE ===
    st.divider()
    render_positions_table()
    
    # === ANALYTICS BAR ===
    st.divider()
    render_analytics()


if __name__ == "__main__":
    main()
