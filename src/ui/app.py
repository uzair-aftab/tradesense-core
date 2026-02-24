"""
HFP War Room — The Trading Cockpit
====================================

Institutional-grade Streamlit dashboard for the HFP system.

Layout
------
┌─────────────────────────────────────────────────────────┐
│  Sidebar: Watchlist | Ticker | Timeframe | Health       │
├─────────────────────────────────────────────────────────┤
│  Top Bar: Equity | P&L | Sharpe | Drawdown | Win Rate  │
├──────────────────────────────┬──────────────────────────┤
│                              │  Prediction Panel        │
│  Plotly Candlestick +        │  P10 / P50 / P90         │
│  Confidence Cloud +          │  Confidence %            │
│  Predicted Path              │  Direction               │
│                              ├──────────────────────────┤
│                              │  Trade Controls          │
│                              │  Buy/Sell + Stop/TP      │
├──────────────────────────────┴──────────────────────────┤
│  Positions | Trade History | Feature Importance | Health │
└─────────────────────────────────────────────────────────┘

Usage
-----
    streamlit run src/ui/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.config import config
from src.data.data_feed import DataLake
from src.data.watchlists import WatchlistManager, AssetCategory
from src.models.ai_engine import AIEngine, QuantilePrediction
from src.models.trader import Gauntlet, TradeOrder, Side
from src.utils.health import HealthCheck

# ---------------------------------------------------------------------------
#  Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="HFP War Room",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
#  Premium CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global */
    .stApp {
        background: linear-gradient(180deg, #0a0a1a 0%, #0e1117 50%, #0a0e1a 100%);
        font-family: 'Inter', sans-serif;
    }

    /* Metric cards — glassmorphism */
    div[data-testid="stMetric"] {
        background: rgba(26, 26, 46, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(99, 110, 250, 0.15);
        border-radius: 12px;
        padding: 14px 18px;
        transition: all 0.3s ease;
    }
    div[data-testid="stMetric"]:hover {
        border-color: rgba(99, 110, 250, 0.4);
        box-shadow: 0 0 20px rgba(99, 110, 250, 0.1);
    }
    div[data-testid="stMetricValue"] { font-size: 1.3rem; font-weight: 600; }

    /* Health badges */
    .health-green { color: #00ff88; font-weight: 700; }
    .health-yellow { color: #ffa500; font-weight: 700; }
    .health-red { color: #ff4444; font-weight: 700; }

    /* Section headers */
    .section-header {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #6b7280;
        margin-bottom: 8px;
        font-weight: 600;
    }

    /* Live badge */
    .live-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255, 68, 68, 0.15);
        border: 1px solid rgba(255, 68, 68, 0.3);
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 0.7rem;
        color: #ff4444;
        font-weight: 600;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
        border-right: 1px solid rgba(48, 54, 61, 0.5);
    }

    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
#  Session State Initialisation
# ---------------------------------------------------------------------------
def _init_state() -> None:
    """Ensure all session-state keys exist."""
    defaults = {
        "lake": DataLake(sentiment_backend=config.SENTIMENT_BACKEND),
        "engine": AIEngine(
            lstm_epochs=config.LSTM_EPOCHS,
            xgb_estimators=config.XGB_ESTIMATORS,
            seq_len=config.LSTM_SEQ_LEN,
        ),
        "gauntlet": Gauntlet(starting_capital=config.STARTING_CAPITAL, seed=42),
        "watchlist": WatchlistManager(),
        "trained": False,
        "raw_df": None,
        "feat_df": None,
        "prediction": None,
        "ticker": "AAPL",
        "timeframe": config.DEFAULT_TIMEFRAME,
        "lookback": config.DEFAULT_LOOKBACK,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ---------------------------------------------------------------------------
#  Sidebar
# ---------------------------------------------------------------------------
def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 🎯 HFP War Room")
        st.caption("High-Frequency Prediction Engine")

        st.divider()

        # ── Watchlist selector ─────────────────────────────────────
        st.markdown('<p class="section-header">📋 Watchlist</p>', unsafe_allow_html=True)
        wm: WatchlistManager = st.session_state.watchlist
        category = st.selectbox(
            "Category",
            [c.value for c in AssetCategory],
            format_func=lambda x: x.replace("_", " ").title(),
            label_visibility="collapsed",
        )
        cat_enum = AssetCategory(category)
        tickers = wm.get_tickers(cat_enum)

        selected = st.selectbox("Ticker", tickers, label_visibility="collapsed")
        if selected:
            st.session_state.ticker = selected

        # Manual ticker override
        custom = st.text_input(
            "Or type ticker",
            value="",
            placeholder="e.g. TSLA",
        ).upper().strip()
        if custom:
            st.session_state.ticker = custom

        st.divider()

        # ── Timeframe & Lookback ───────────────────────────────────
        st.markdown('<p class="section-header">⏱ Settings</p>', unsafe_allow_html=True)
        tf_options = {"12 Hours": "1h", "1 Day": "1d"}
        tf_label = st.selectbox("Prediction Horizon", list(tf_options.keys()), index=1)
        st.session_state.timeframe = tf_options[tf_label]

        lb_options = {"3 Months": "3mo", "6 Months": "6mo", "1 Year": "1y", "2 Years": "2y"}
        lb_label = st.selectbox("Training Lookback", list(lb_options.keys()), index=2)
        st.session_state.lookback = lb_options[lb_label]

        sniper = st.toggle("🎯 Sniper Mode", value=False, help="Only show predictions with >80% confidence")
        st.session_state["sniper_mode"] = sniper

        st.divider()

        # ── Health Dashboard ───────────────────────────────────────
        st.markdown('<p class="section-header">💊 System Health</p>', unsafe_allow_html=True)

        engine: AIEngine = st.session_state.engine
        health = engine.health
        color_class = f"health-{health}"
        health_label = {"green": "● HEALTHY", "yellow": "● CAUTION", "red": "● OFFLINE"}
        st.markdown(f'<p class="{color_class}">{health_label.get(health, "● UNKNOWN")}</p>', unsafe_allow_html=True)

        if engine.last_train_result:
            r = engine.last_train_result
            st.caption(f"Meta MAE: {r.meta_mae:.6f}" if r.meta_mae else "No training data")
            st.caption(f"Trained in {r.duration_sec:.1f}s")

        # Compact health indicators
        try:
            health_summary = HealthCheck.summary_emoji()
            st.caption(health_summary)
        except Exception:
            pass

        st.divider()

        # ── Action Buttons ─────────────────────────────────────────
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Fetch Data", use_container_width=True):
                _fetch_data()
        with col2:
            if st.button("🧠 Train Model", use_container_width=True):
                _train_model()

        if st.button("🎯 Predict", use_container_width=True):
            _predict()

        st.divider()

        # ── Account Summary ────────────────────────────────────────
        st.markdown('<p class="section-header">💰 Account</p>', unsafe_allow_html=True)
        sim: Gauntlet = st.session_state.gauntlet
        st.metric("Cash", f"${sim.cash:,.2f}")
        st.metric("Total Equity", f"${sim.total_equity:,.2f}")

        if st.button("🔄 Reset Paper Account", use_container_width=True):
            sim.reset()
            st.rerun()


# ---------------------------------------------------------------------------
#  Actions
# ---------------------------------------------------------------------------
def _fetch_data() -> None:
    lake: DataLake = st.session_state.lake
    ticker = st.session_state.ticker
    tf = st.session_state.timeframe
    lb = st.session_state.lookback

    with st.spinner(f"Fetching {ticker} ({tf}, {lb})…"):
        raw = lake.fetch(ticker, timeframe=tf, lookback=lb)
        feat = lake.engineer_features(raw, ticker=ticker, include_sentiment=True)
        feat.ffill(inplace=True)
        feat.bfill(inplace=True)
        feat.dropna(inplace=True)

    st.session_state.raw_df = raw
    st.session_state.feat_df = feat
    st.toast(f"✅ {len(raw)} bars fetched, {feat.shape[1]} features", icon="📊")


def _train_model() -> None:
    feat = st.session_state.feat_df
    if feat is None or feat.empty:
        st.error("Fetch data first!")
        return

    engine: AIEngine = st.session_state.engine
    with st.spinner("Training LSTM + XGBoost ensemble…"):
        result = engine.train(feat, target_col="Close")

    st.session_state.trained = True
    st.toast(
        f"✅ Trained in {result.duration_sec:.1f}s | MAE={result.meta_mae:.6f} | Health={result.health_color}",
        icon="🧠",
    )


def _predict() -> None:
    engine: AIEngine = st.session_state.engine
    feat = st.session_state.feat_df

    if not engine._trained:
        st.error("Train the model first!")
        return
    if feat is None or feat.empty:
        st.error("No feature data — fetch first!")
        return

    pred = engine.predict(feat)

    # Sniper mode filter
    if st.session_state.get("sniper_mode") and pred.confidence < 0.80:
        st.session_state.prediction = None
        st.warning(f"🎯 Sniper Mode: Confidence {pred.confidence:.0%} < 80% — no trade signal")
        return

    st.session_state.prediction = pred
    st.toast(f"P50=${pred.p50:.2f} | Confidence={pred.confidence:.0%}", icon="🎯")


# ---------------------------------------------------------------------------
#  Chart — Candlestick + Confidence Cloud
# ---------------------------------------------------------------------------
def _render_chart() -> None:
    raw = st.session_state.raw_df
    if raw is None or raw.empty:
        st.info("👈 Select a ticker and click **Fetch Data** to begin")
        return

    pred: QuantilePrediction = st.session_state.prediction
    ticker = st.session_state.ticker

    # Live badge
    st.markdown(
        f'<span class="live-badge">● LIVE</span> <strong>{ticker}</strong> — HFP Analysis',
        unsafe_allow_html=True,
    )

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.03,
    )

    # ── Candlestick ───────────────────────────────────────────────────
    fig.add_trace(
        go.Candlestick(
            x=raw.index,
            open=raw["Open"],
            high=raw["High"],
            low=raw["Low"],
            close=raw["Close"],
            name="Price",
            increasing_line_color="#00ff88",
            decreasing_line_color="#ff4444",
        ),
        row=1,
        col=1,
    )

    # ── Moving Averages ───────────────────────────────────────────────
    if len(raw) >= 20:
        ma20 = raw["Close"].rolling(20).mean()
        fig.add_trace(
            go.Scatter(x=raw.index, y=ma20, name="MA20",
                       line=dict(color="#636efa", width=1, dash="dot"), opacity=0.7),
            row=1, col=1,
        )
    if len(raw) >= 50:
        ma50 = raw["Close"].rolling(50).mean()
        fig.add_trace(
            go.Scatter(x=raw.index, y=ma50, name="MA50",
                       line=dict(color="#ffa500", width=1, dash="dot"), opacity=0.7),
            row=1, col=1,
        )

    # ── Confidence Cloud ──────────────────────────────────────────────
    if pred is not None:
        last_date = raw.index[-1]

        # Shaded region between P10 and P90
        fig.add_shape(
            type="rect",
            x0=raw.index[-5],
            x1=last_date,
            y0=pred.p10,
            y1=pred.p90,
            fillcolor="rgba(99, 110, 250, 0.15)",
            line=dict(width=0),
            row=1,
            col=1,
        )

        # P10 / P50 / P90 lines
        for val, color, dash, label in [
            (pred.p10, "#ff6b6b", "dot", "P10 (Low)"),
            (pred.p50, "#ffd93d", "dash", "P50 (Most Likely)"),
            (pred.p90, "#6bcb77", "dot", "P90 (High)"),
        ]:
            fig.add_hline(
                y=val, line_dash=dash, line_color=color,
                annotation_text=f"{label}: ${val:.2f}",
                annotation_position="right",
                row=1, col=1,
            )

    # ── Volume bars ───────────────────────────────────────────────────
    colors = [
        "#00ff88" if c >= o else "#ff4444"
        for c, o in zip(raw["Close"], raw["Open"])
    ]
    fig.add_trace(
        go.Bar(x=raw.index, y=raw["Volume"], marker_color=colors, name="Volume", opacity=0.5),
        row=2,
        col=1,
    )

    # ── Whale flags ───────────────────────────────────────────────────
    feat = st.session_state.feat_df
    if feat is not None and "Whale_Flag" in feat.columns:
        whales = feat[feat["Whale_Flag"] == 1]
        if not whales.empty and "Close" in whales.columns:
            fig.add_trace(
                go.Scatter(
                    x=whales.index,
                    y=whales["Close"],
                    mode="markers",
                    marker=dict(symbol="triangle-up", size=14, color="#00d4ff"),
                    name="🐋 Whale",
                ),
                row=1,
                col=1,
            )

    # ── Layout ────────────────────────────────────────────────────────
    fig.update_layout(
        template="plotly_dark",
        height=650,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", y=1.12),
        margin=dict(l=60, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,10,26,0.8)",
    )
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
#  Prediction Panel
# ---------------------------------------------------------------------------
def _render_prediction_panel() -> None:
    pred: QuantilePrediction = st.session_state.prediction

    if pred is None:
        st.markdown("### 🔮 Prediction")
        st.caption("Train model and click **Predict** to see quantile forecast")
        return

    st.markdown("### 🔮 Quantile Prediction")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("P10 (Low)", f"${pred.p10:.2f}", help="90% chance price stays above this")
    with c2:
        st.metric("P50 (Likely)", f"${pred.p50:.2f}", help="Most probable price")
    with c3:
        st.metric("P90 (High)", f"${pred.p90:.2f}", help="90% chance price stays below this")

    col1, col2 = st.columns(2)
    with col1:
        conf_color = "🟢" if pred.confidence > 0.8 else "🟡" if pred.confidence > 0.5 else "🔴"
        st.metric("Confidence", f"{conf_color} {pred.confidence:.0%}")
    with col2:
        dir_icon = "📈" if pred.direction == "BULLISH" else "📉" if pred.direction == "BEARISH" else "➡️"
        st.metric("Direction", f"{dir_icon} {pred.direction}")

    st.metric("Spread (P90−P10)", f"${pred.spread:.2f}")


# ---------------------------------------------------------------------------
#  Trade Controls
# ---------------------------------------------------------------------------
def _render_trade_controls() -> None:
    st.markdown("### ⚡ Trade Controls")

    sim: Gauntlet = st.session_state.gauntlet
    pred: QuantilePrediction = st.session_state.prediction
    raw = st.session_state.raw_df
    ticker = st.session_state.ticker

    if raw is None or raw.empty:
        st.caption("Fetch data to enable trading")
        return

    current_price = float(raw["Close"].iloc[-1])
    st.caption(f"Current Price: **${current_price:.2f}**")

    # Position sizer
    st.markdown("##### 📐 Position Sizer")
    risk_pct = st.slider("Risk Tolerance (%)", 0.5, 5.0, 2.0, 0.5) / 100

    stop_price = pred.p10 if pred else current_price * 0.97
    suggested_shares = sim.calculate_position_size(
        price=current_price,
        stop_price=stop_price,
        risk_pct=risk_pct,
    )
    st.info(f"Suggested: **{suggested_shares} shares** (stop @ ${stop_price:.2f})")

    # Manual quantity override
    qty = st.number_input("Shares", min_value=1, value=max(1, suggested_shares), step=1)

    # ── Stop-Loss / Take-Profit ─────────────────────────────────────
    st.markdown("##### 🛡 Risk Management")
    sl_col, tp_col = st.columns(2)
    with sl_col:
        stop_loss = st.number_input(
            "Stop-Loss ($)",
            min_value=0.0,
            value=round(stop_price, 2),
            step=0.50,
            format="%.2f",
        )
    with tp_col:
        take_profit = st.number_input(
            "Take-Profit ($)",
            min_value=0.0,
            value=round(current_price * 1.05, 2),
            step=0.50,
            format="%.2f",
        )

    # Buy / Sell
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🟢 BUY", use_container_width=True, type="primary"):
            try:
                fill = sim.submit_order(
                    TradeOrder(ticker=ticker, side=Side.BUY, quantity=int(qty), price=current_price)
                )
                # Set stop-loss and take-profit
                if stop_loss > 0:
                    sim.set_stop_loss(ticker, stop_loss)
                if take_profit > 0:
                    sim.set_take_profit(ticker, take_profit)
                st.success(f"BUY {fill.quantity} @ ${fill.fill_price:.2f} (comm=${fill.commission:.2f})")
            except ValueError as e:
                st.error(str(e))

    with c2:
        if st.button("🔴 SELL", use_container_width=True):
            try:
                pos = sim.positions.get(ticker)
                sell_qty = pos.quantity if pos else int(qty)
                fill = sim.submit_order(
                    TradeOrder(ticker=ticker, side=Side.SELL, quantity=sell_qty, price=current_price)
                )
                st.success(f"SELL {fill.quantity} @ ${fill.fill_price:.2f} (P&L=${fill.pnl:+.2f})")
            except ValueError as e:
                st.error(str(e))

    # Update mark-to-market
    sim.mark_to_market({ticker: current_price})


# ---------------------------------------------------------------------------
#  Metrics Bar
# ---------------------------------------------------------------------------
def _render_metrics_bar() -> None:
    sim: Gauntlet = st.session_state.gauntlet
    m = sim.metrics

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        delta = m.total_equity - sim.starting_capital
        st.metric("Total Equity", f"${m.total_equity:,.2f}", f"${delta:+,.2f}")
    with c2:
        st.metric("Realised P&L", f"${m.realised_pnl:+,.2f}")
    with c3:
        st.metric("Sharpe Ratio", f"{m.sharpe_ratio:.2f}")
    with c4:
        st.metric("Max Drawdown", f"{m.max_drawdown_pct:.2f}%")
    with c5:
        st.metric("Win Rate", f"{m.win_rate:.0%}" if m.total_trades > 0 else "N/A")


# ---------------------------------------------------------------------------
#  Positions & History
# ---------------------------------------------------------------------------
def _render_positions() -> None:
    sim: Gauntlet = st.session_state.gauntlet

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Positions", "📜 Trade History", "📈 Feature Importance", "💊 System Health"
    ])

    with tab1:
        pos_df = sim.positions_summary()
        if pos_df.empty:
            st.caption("No open positions")
        else:
            st.dataframe(pos_df, use_container_width=True, hide_index=True)

    with tab2:
        hist_df = sim.trade_history()
        if hist_df.empty:
            st.caption("No trades yet")
        else:
            st.dataframe(hist_df, use_container_width=True, hide_index=True)

    with tab3:
        engine: AIEngine = st.session_state.engine
        imp = engine.feature_importance()
        if imp is not None:
            fig = go.Figure(
                go.Bar(
                    x=imp["importance"].values[:15],
                    y=imp.index[:15],
                    orientation="h",
                    marker_color="#636efa",
                )
            )
            fig.update_layout(
                title="Top 15 Feature Importance (XGBoost P50)",
                template="plotly_dark",
                height=400,
                yaxis=dict(autorange="reversed"),
                margin=dict(l=150),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(10,10,26,0.8)",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Train model to see feature importance")

    with tab4:
        try:
            report = HealthCheck.run_all()
            h1, h2, h3 = st.columns(3)
            with h1:
                gpu = report["gpu"]
                st.metric("GPU", "✅ Available" if gpu["available"] else "❌ N/A")
                if gpu["available"]:
                    st.caption(f'{gpu["device_name"]}')
                    st.caption(f'VRAM: {gpu["memory_allocated"]}')
            with h2:
                st.metric("Internet", "✅ Connected" if report["internet"] else "❌ Offline")
                mem = report["memory"]
                st.caption(f'RAM: {mem["available"]} / {mem["total"]}')
            with h3:
                models = report["models"]
                ready = sum(1 for v in models.values() if v)
                st.metric("Models", f"{ready}/{len(models)} loaded")
                cache = report["cache"]
                st.caption(f'Cache: {cache["size"]}')
        except Exception as e:
            st.error(f"Health check error: {e}")


# ---------------------------------------------------------------------------
#  Equity Curve
# ---------------------------------------------------------------------------
def _render_equity_curve() -> None:
    sim: Gauntlet = st.session_state.gauntlet
    curve = sim.equity_curve

    if len(curve) <= 1:
        return

    st.markdown("### 💰 Equity Curve")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            y=curve.values,
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(99, 110, 250, 0.1)",
            line=dict(color="#636efa", width=2),
            name="Equity",
        )
    )
    fig.add_hline(
        y=sim.starting_capital,
        line_dash="dash",
        line_color="#888",
        annotation_text=f"Starting: ${sim.starting_capital:,.0f}",
    )
    fig.update_layout(
        template="plotly_dark",
        height=250,
        margin=dict(l=60, r=20, t=20, b=20),
        yaxis_title="Equity ($)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,10,26,0.8)",
    )
    st.plotly_chart(fig, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Main Layout                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main() -> None:
    _render_sidebar()

    # ── Top Metrics Bar ───────────────────────────────────────────────
    _render_metrics_bar()

    st.divider()

    # ── Main content: Chart + Prediction/Controls ─────────────────────
    chart_col, panel_col = st.columns([2, 1])

    with chart_col:
        _render_chart()

    with panel_col:
        _render_prediction_panel()
        st.divider()
        _render_trade_controls()

    st.divider()

    # ── Bottom section ────────────────────────────────────────────────
    _render_equity_curve()
    _render_positions()


if __name__ == "__main__":
    main()
