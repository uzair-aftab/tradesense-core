"""
TradeSense Interactive Chart Manipulation Suite
================================================
Plotly glass modebar, dynamic scroll-zoom, auto-scaled Y-axis,
Heikin-Ashi/Line/Candle renderers, multi-indicator overlays, and forecast fan.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PLOTLY_GLASS_CONFIG: Dict[str, Any] = {
    "displayModeBar": True,
    "displaylogo": False,
    "scrollZoom": True,
    "doubleClick": "reset+autosize",
    "modeBarButtonsToAdd": ["drawline", "drawrect", "eraseshape"],
    "toImageButtonOptions": {"format": "png", "filename": "tradesense_chart", "scale": 2},
}


def calculate_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """Computes Heikin-Ashi smoothed candles from standard OHLCV."""
    ha = pd.DataFrame(index=df.index)
    ha["Close"] = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4.0

    ha_open = [float(df["Open"].iloc[0])]
    for i in range(1, len(df)):
        ha_open.append((ha_open[i - 1] + ha["Close"].iloc[i - 1]) / 2.0)
    ha["Open"] = ha_open
    ha["High"] = [max(h, o, c) for h, o, c in zip(df["High"], ha["Open"], ha["Close"])]
    ha["Low"] = [min(l, o, c) for l, o, c in zip(df["Low"], ha["Open"], ha["Close"])]
    if "Volume" in df.columns:
        ha["Volume"] = df["Volume"]
    return ha


def _generate_time_labels(dt_index: Any) -> List[str]:
    """Generates unique categorical string labels for the x-axis."""
    labels: List[str] = []
    seen: Dict[str, int] = {}
    for dt in dt_index:
        ts = pd.to_datetime(dt)
        if hasattr(ts, "hour") and (ts.hour != 0 or ts.minute != 0):
            base = ts.strftime("%b %d %H:%M")
        else:
            base = ts.strftime("%b %d, %Y")
        if base in seen:
            seen[base] += 1
            labels.append(f"{base} ({seen[base]})")
        else:
            seen[base] = 0
            labels.append(base)
    return labels


def build_interactive_chart(
    df: pd.DataFrame,
    symbol: str = "BTC-CAD",
    forecast: Optional[Dict[str, Any]] = None,
    visible_bars: int = 80,
    chart_type: str = "Candles",
    overlays: Optional[List[str]] = None,
    trades_history: Optional[List[Dict[str, Any]]] = None,
    order_levels: Optional[Dict[str, float]] = None,
) -> go.Figure:
    """
    Builds the interactive frosted-glass financial chart with Plotly modebar and auto-scaling.
    """
    if df is None or len(df) == 0:
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            annotations=[dict(text="No Price Data Available", showarrow=False, font=dict(color="#94a3b8", size=14))],
        )
        return fig

    overlays = overlays or []
    show_volume = "Volume" in overlays or "Volume Bars" in overlays
    visible_bars = max(15, min(visible_bars, len(df)))

    # Slice visible window
    df_visible = df.iloc[-visible_bars:].copy()

    # Apply chart type transformations
    if chart_type == "Heikin-Ashi":
        plot_df = calculate_heikin_ashi(df_visible)
    else:
        plot_df = df_visible

    x_labels = _generate_time_labels(plot_df.index)

    # Set up Subplots (with Volume row if active)
    if show_volume and "Volume" in plot_df.columns:
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.82, 0.18],
        )
    else:
        fig = make_subplots(rows=1, cols=1)

    # 1. Main Price Series
    if chart_type == "Line":
        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=plot_df["Close"],
                mode="lines",
                name=f"{symbol} Close",
                line=dict(color="#38bdf8", width=2.0),
                fill="tozeroy",
                fillcolor="rgba(56, 189, 248, 0.05)",
                hovertemplate="%{x}<br>Close: $%{y:,.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    else:
        fig.add_trace(
            go.Candlestick(
                x=x_labels,
                open=plot_df["Open"],
                high=plot_df["High"],
                low=plot_df["Low"],
                close=plot_df["Close"],
                name=symbol,
                increasing=dict(line=dict(color="#34d399", width=1.2), fillcolor="rgba(52, 211, 153, 0.85)"),
                decreasing=dict(line=dict(color="#fb7185", width=1.2), fillcolor="rgba(251, 113, 133, 0.85)"),
            ),
            row=1,
            col=1,
        )

    # 2. Indicators & Overlays
    # EMA 9 & 21
    if any("EMA" in ov for ov in overlays):
        if len(df) >= 9:
            ema9_full = df["Close"].ewm(span=9, adjust=False).mean()
            ema9_vis = ema9_full.iloc[-visible_bars:].values
            fig.add_trace(
                go.Scatter(
                    x=x_labels,
                    y=ema9_vis,
                    mode="lines",
                    name="EMA 9",
                    line=dict(color="#38bdf8", width=1.2),
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )
        if len(df) >= 21:
            ema21_full = df["Close"].ewm(span=21, adjust=False).mean()
            ema21_vis = ema21_full.iloc[-visible_bars:].values
            fig.add_trace(
                go.Scatter(
                    x=x_labels,
                    y=ema21_vis,
                    mode="lines",
                    name="EMA 21",
                    line=dict(color="#fbbf24", width=1.2),
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )

    # VWAP
    if any("VWAP" in ov for ov in overlays) and "Volume" in df.columns:
        hlc3 = (df["High"] + df["Low"] + df["Close"]) / 3.0
        vwap_full = (hlc3 * df["Volume"]).cumsum() / df["Volume"].cumsum().replace(0, 1)
        vwap_vis = vwap_full.iloc[-visible_bars:].values
        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=vwap_vis,
                mode="lines",
                name="VWAP",
                line=dict(color="#a78bfa", width=1.4, dash="dash"),
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )

    # Bollinger Bands (20, 2)
    if any("Bollinger" in ov or "BB" in ov for ov in overlays):
        if len(df) >= 20:
            mid = df["Close"].rolling(20).mean()
            std = df["Close"].rolling(20).std()
            upper_vis = (mid + 2 * std).iloc[-visible_bars:].values
            lower_vis = (mid - 2 * std).iloc[-visible_bars:].values

            fig.add_trace(
                go.Scatter(
                    x=x_labels,
                    y=upper_vis,
                    mode="lines",
                    name="BB Upper",
                    line=dict(color="rgba(148, 163, 184, 0.4)", width=1.0, dash="dot"),
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=x_labels,
                    y=lower_vis,
                    mode="lines",
                    name="BB Lower",
                    line=dict(color="rgba(148, 163, 184, 0.4)", width=1.0, dash="dot"),
                    fill="tonexty",
                    fillcolor="rgba(148, 163, 184, 0.04)",
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )

    # Support & Resistance Zones
    if any("Support" in ov or "S&R" in ov for ov in overlays):
        sup_val = float(plot_df["Low"].min())
        res_val = float(plot_df["High"].max())
        fig.add_hline(
            y=res_val,
            line=dict(color="rgba(244, 63, 94, 0.45)", width=1.0, dash="dash"),
            annotation_text=f"Res ${res_val:,.2f}",
            annotation_position="top left",
            annotation_font=dict(color="#f43f5e", size=10),
            row=1,
            col=1,
        )
        fig.add_hline(
            y=sup_val,
            line=dict(color="rgba(16, 185, 129, 0.45)", width=1.0, dash="dash"),
            annotation_text=f"Sup ${sup_val:,.2f}",
            annotation_position="bottom left",
            annotation_font=dict(color="#10b981", size=10),
            row=1,
            col=1,
        )

    # 3. Forecast Fan & Quantile Cones
    p10_val: Optional[float] = None
    p50_val: Optional[float] = None
    p90_val: Optional[float] = None

    if forecast:
        p10_val = forecast.get("consensus_p10") or forecast.get("p10")
        p50_val = forecast.get("consensus_p50") or forecast.get("p50")
        p90_val = forecast.get("consensus_p90") or forecast.get("p90")
        horizon_bars = int(forecast.get("horizon_bars", 5))

        if p10_val is not None and p50_val is not None and p90_val is not None and len(x_labels) > 0:
            last_x = x_labels[-1]
            last_close = float(plot_df["Close"].iloc[-1])

            fan_x = [last_x] + [f"+{i}B" for i in range(1, horizon_bars + 1)]
            fan_p10 = [last_close] + list(np.linspace(last_close, p10_val, horizon_bars))
            fan_p50 = [last_close] + list(np.linspace(last_close, p50_val, horizon_bars))
            fan_p90 = [last_close] + list(np.linspace(last_close, p90_val, horizon_bars))

            # P90 upper bound
            fig.add_trace(
                go.Scatter(
                    x=fan_x,
                    y=fan_p90,
                    mode="lines",
                    name="P90 Target (+)",
                    line=dict(color="rgba(56, 189, 248, 0.4)", width=1.0, dash="dash"),
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )
            # P10 lower bound with filled cone
            fig.add_trace(
                go.Scatter(
                    x=fan_x,
                    y=fan_p10,
                    mode="lines",
                    name="P10 Downside (-)",
                    line=dict(color="rgba(56, 189, 248, 0.4)", width=1.0, dash="dash"),
                    fill="tonexty",
                    fillcolor="rgba(56, 189, 248, 0.12)",
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )
            # P50 consensus line
            fig.add_trace(
                go.Scatter(
                    x=fan_x,
                    y=fan_p50,
                    mode="lines+markers",
                    name="P50 Consensus Target",
                    line=dict(color="#38bdf8", width=2.4, dash="dot"),
                    marker=dict(size=4.5, color="#38bdf8"),
                    hovertemplate="Forecast %{x}: $%{y:,.2f}<extra></extra>",
                ),
                row=1,
                col=1,
            )

    # 4. Order Level Price Lines (Limit Entry, Take Profit, Stop Loss)
    if order_levels:
        if "limit_entry" in order_levels:
            le = order_levels["limit_entry"]
            fig.add_hline(
                y=le,
                line=dict(color="#38bdf8", width=1.4, dash="dash"),
                annotation_text=f"Entry ${le:,.2f}",
                annotation_position="right",
                annotation_font=dict(color="#38bdf8", size=10),
                row=1,
                col=1,
            )
        if "take_profit" in order_levels:
            tp = order_levels["take_profit"]
            fig.add_hline(
                y=tp,
                line=dict(color="#34d399", width=1.4, dash="dash"),
                annotation_text=f"TP ${tp:,.2f}",
                annotation_position="right",
                annotation_font=dict(color="#34d399", size=10),
                row=1,
                col=1,
            )
        if "stop_loss" in order_levels:
            sl = order_levels["stop_loss"]
            fig.add_hline(
                y=sl,
                line=dict(color="#fb7185", width=1.4, dash="dash"),
                annotation_text=f"SL ${sl:,.2f}",
                annotation_position="right",
                annotation_font=dict(color="#fb7185", size=10),
                row=1,
                col=1,
            )

    # 5. Virtual Bot Trade Markers (Walk-Forward & Live)
    if trades_history:
        buy_x, buy_y = [], []
        exit_x, exit_y = [], []
        vis_start_idx = len(df) - visible_bars
        for t in trades_history:
            e_idx = t.get("entry_idx")
            x_idx = t.get("exit_idx")
            if e_idx is not None and e_idx >= vis_start_idx:
                rel_idx = e_idx - vis_start_idx
                if 0 <= rel_idx < len(x_labels):
                    buy_x.append(x_labels[rel_idx])
                    buy_y.append(t["entry_price"])
            if x_idx is not None and x_idx >= vis_start_idx:
                rel_idx = x_idx - vis_start_idx
                if 0 <= rel_idx < len(x_labels):
                    exit_x.append(x_labels[rel_idx])
                    exit_y.append(t["exit_price"])

        if buy_x:
            fig.add_trace(
                go.Scatter(
                    x=buy_x,
                    y=buy_y,
                    mode="markers",
                    name="Virtual Buy",
                    marker=dict(symbol="triangle-up", size=10, color="#34d399", line=dict(width=1, color="#ffffff")),
                ),
                row=1,
                col=1,
            )
        if exit_x:
            fig.add_trace(
                go.Scatter(
                    x=exit_x,
                    y=exit_y,
                    mode="markers",
                    name="Virtual Exit",
                    marker=dict(symbol="triangle-down", size=10, color="#fb7185", line=dict(width=1, color="#ffffff")),
                ),
                row=1,
                col=1,
            )

    # 6. Volume Row
    if show_volume and "Volume" in plot_df.columns:
        vol_colors = [
            "#34d399" if c >= o else "#fb7185"
            for c, o in zip(plot_df["Close"], plot_df["Open"])
        ]
        fig.add_trace(
            go.Bar(
                x=x_labels,
                y=plot_df["Volume"],
                name="Volume",
                marker=dict(color=vol_colors, opacity=0.7),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=2,
            col=1,
        )

    # 7. Dynamic Y-Axis Scaling on Visible Slice + Forecast
    all_prices = [float(plot_df["Low"].min()), float(plot_df["High"].max())]
    if p10_val is not None:
        all_prices.append(float(p10_val))
    if p90_val is not None:
        all_prices.append(float(p90_val))
    if order_levels:
        for v in order_levels.values():
            if v and not math.isnan(v):
                all_prices.append(float(v))

    y_min, y_max = min(all_prices), max(all_prices)
    padding = max((y_max - y_min) * 0.06, y_min * 0.005)
    y_range = [y_min - padding, y_max + padding]

    # Layout & Modebar Styling
    chart_height = 420 if show_volume else 385
    fig.update_layout(
        height=chart_height,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            font=dict(color="#94a3b8", size=11),
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.04)",
            color="#64748b",
            rangeslider=dict(visible=False),
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikethickness=1,
            spikecolor="rgba(56, 189, 248, 0.3)",
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.04)",
            color="#64748b",
            side="right",
            range=y_range,
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikethickness=1,
            spikecolor="rgba(56, 189, 248, 0.3)",
        ),
    )

    if show_volume:
        fig.update_layout(
            yaxis2=dict(
                showgrid=False,
                color="#64748b",
                side="right",
                showticklabels=False,
            )
        )

    return fig
