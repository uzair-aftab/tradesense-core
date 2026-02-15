"""
HFP War Room — The Trading Cockpit
====================================

Institutional-grade Streamlit dashboard for the HFP system.

Layout
------
┌─────────────────────────────────────────────────────────┐
│  Sidebar: Ticker | Timeframe | Sniper Toggle | Health  │
├─────────────────────────────────────────────────────────┤
│  Top Bar: Equity | P&L | Sharpe | Drawdown | Win Rate  │
├──────────────────────────────┬──────────────────────────┤
│                              │  Prediction Panel       │
│  Plotly Candlestick +        │  P10 / P50 / P90        │
│  Confidence Cloud +          │  Confidence %            │
│  Predicted Path              │  Direction               │
│                              ├──────────────────────────┤
│                              │  Trade Controls          │
│                              │  Buy/Sell + Pos Sizer    │
├──────────────────────────────┴──────────────────────────┤
│  Positions | Trade History | Feature Importance          │
└─────────────────────────────────────────────────────────┘

Usage
-----
    streamlit run src/ui/app.py -- --port 8501

    or from project root:
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

from src.data.data_feed import DataLake
from src.models.ai_engine import AIEngine, QuantilePrediction
from src.models.trader import Gauntlet, TradeOrder, Side

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
#  Dark theme CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Dark background */
    .stApp { background-color: #0e1117; }
    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 12px 16px;
    }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; }
    /* Health indicator */
    .health-green { color: #00ff88; font-weight: 700; }
    .health-yellow { color: #ffa500; font-weight: 700; }
    .health-red { color: #ff4444; font-weight: 700; }
    /* Confidence Cloud label */
    .cloud-label {
        font-size: 0.85rem; color: #8b949e;
        text-transform: uppercase; letter-spacing: 1px;
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
        "lake": DataLake(sentiment_backend="vader"),
        "engine": AIEngine(lstm_epochs=30, xgb_estimators=300, seq_len=20),
        "gauntlet": Gauntlet(starting_capital=10_000, seed=42),
        "trained": False,
        "raw_df": None,
        "feat_df": None,
        "prediction": None,
        "ticker": "AAPL",
        "timeframe": "1d",
        "lookback": "1y",
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
        st.title("🎯 HFP War Room")
        st.caption("High-Frequency Prediction Engine")

        st.divider()

        # Ticker
        st.session_state.ticker = st.text_input(
            "Ticker Symbol",
            value=st.session_state.ticker,
            placeholder="AAPL",
        ).upper().strip()

        # Timeframe selector
        tf_options = {"12 Hours": "1h", "1 Day": "1d"}
        tf_label = st.selectbox("Prediction Horizon", list(tf_options.keys()), index=1)
        st.session_state.timeframe = tf_options[tf_label]

        # Lookback
        lb_options = {"3 Months": "3mo", "6 Months": "6mo", "1 Year": "1y", "2 Years": "2y"}
        lb_label = st.selectbox("Training Lookback", list(lb_options.keys()), index=2)
        st.session_state.lookback = lb_options[lb_label]

        st.divider()

        # Sniper mode toggle
        sniper = st.toggle("🎯 Sniper Mode", value=False, help="Only show predictions with >80% confidence")
        st.session_state["sniper_mode"] = sniper

        st.divider()

        # Model Health Indicator
        engine: AIEngine = st.session_state.engine
        health = engine.health
        color_class = f"health-{health}"
        health_label = {"green": "● HEALTHY", "yellow": "● CAUTION", "red": "● OFFLINE"}
        st.markdown(f'<p class="{color_class}">{health_label.get(health, "● UNKNOWN")}</p>', unsafe_allow_html=True)

        if engine.last_train_result:
            r = engine.last_train_result
            st.caption(f"Meta MAE: {r.meta_mae:.6f}" if r.meta_mae else "No training data")
            st.caption(f"Trained in {r.duration_sec:.1f}s")

        st.divider()

        # Action buttons
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

        # Account summary
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

    # ── Confidence Cloud ──────────────────────────────────────────────
    if pred is not None:
        last_date = raw.index[-1]
        # Extend a few bars into the future for the prediction zone
        cloud_x = [last_date, last_date]

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
        title=f"{ticker} — HFP Analysis",
        template="plotly_dark",
        height=650,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", y=1.12),
        margin=dict(l=60, r=20, t=80, b=20),
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

    # Buy / Sell
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🟢 BUY", use_container_width=True, type="primary"):
            try:
                fill = sim.submit_order(
                    TradeOrder(ticker=ticker, side=Side.BUY, quantity=int(qty), price=current_price)
                )
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

    tab1, tab2, tab3 = st.tabs(["📊 Positions", "📜 Trade History", "📈 Feature Importance"])

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
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Train model to see feature importance")


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
