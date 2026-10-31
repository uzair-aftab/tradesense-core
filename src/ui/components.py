"""
TradeSense Institutional UI Components & Mathematical Engines
=============================================================
Provides:
  - Enterprise Controlled Table Suite (live search, sort, pagination, CSV export)
  - Mathematical Chart Feature Suite (Heikin-Ashi, Timeframe Resampler, Indicators)
  - Hardware Telemetry SVG Dials (GPU VRAM, System RAM, CPU)
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from src.ui.icons import svg_icon


# ─────────────────────────────────────────────────────────────────────────────
# 1. Enterprise Controlled Table Engine
# ─────────────────────────────────────────────────────────────────────────────
def render_controlled_table(
    df: pd.DataFrame,
    key_prefix: str,
    title: str = "",
    default_sort_col: Optional[str] = None,
    default_sort_asc: bool = False,
    csv_filename_base: str = "tradesense_export",
) -> None:
    """
    Renders an institutional, interactive data table equipped with:
      - Live search filtering across all columns
      - Sizing / pagination controls ([ 10 ], [ 25 ], [ 50 ], [ All ])
      - Multi-column sort controls
      - 1-Click CSV export
      - Formatted tabular layouts
    """
    if df is None or df.empty:
        st.markdown(
            f"""
            <div style="background: rgba(11, 17, 30, 0.6); border: 1px dashed rgba(255, 255, 255, 0.1); 
                        border-radius: 10px; padding: 24px; text-align: center; color: #64748b; font-size: 0.9rem;">
                No records recorded yet in this view.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    work_df = df.copy()

    # Toolbar row: Search bar, Sort selector, Row limit, CSV Download
    col_search, col_sort, col_limit, col_export = st.columns([2.5, 1.5, 1.2, 1.2])

    with col_search:
        search_query = st.text_input(
            "Search table...",
            placeholder="Search symbols, IDs, sides, dates...",
            key=f"{key_prefix}_search",
            label_visibility="collapsed",
        ).strip().lower()

    with col_sort:
        sort_cols = list(work_df.columns)
        default_idx = sort_cols.index(default_sort_col) if default_sort_col in sort_cols else 0
        sort_by = st.selectbox(
            "Sort Column",
            options=sort_cols,
            index=default_idx,
            key=f"{key_prefix}_sort_col",
            label_visibility="collapsed",
        )

    with col_limit:
        limit_choice = st.selectbox(
            "Show Rows",
            options=["10 Rows", "25 Rows", "50 Rows", "All Rows"],
            index=1,
            key=f"{key_prefix}_limit",
            label_visibility="collapsed",
        )

    # 1. Apply Search Filter
    if search_query:
        mask = work_df.astype(str).apply(lambda row: row.str.lower().str.contains(search_query, regex=False)).any(axis=1)
        work_df = work_df[mask]

    # 2. Apply Sorting
    if sort_by and sort_by in work_df.columns:
        try:
            # Check if column is numeric or currency
            col_series = work_df[sort_by].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False).str.replace("%", "", regex=False).str.replace("+", "", regex=False)
            num_check = pd.to_numeric(col_series, errors="coerce")
            if num_check.notna().sum() > len(work_df) * 0.5:
                work_df["_sort_tmp"] = num_check
                work_df = work_df.sort_values(by="_sort_tmp", ascending=default_sort_asc).drop(columns=["_sort_tmp"])
            else:
                work_df = work_df.sort_values(by=sort_by, ascending=default_sort_asc)
        except Exception:
            work_df = work_df.sort_values(by=sort_by, ascending=default_sort_asc)

    # 3. Apply Pagination / Sizing
    total_records = len(df)
    filtered_records = len(work_df)

    if limit_choice == "10 Rows":
        display_df = work_df.head(10)
    elif limit_choice == "25 Rows":
        display_df = work_df.head(25)
    elif limit_choice == "50 Rows":
        display_df = work_df.head(50)
    else:
        display_df = work_df

    with col_export:
        csv_buffer = io.StringIO()
        work_df.to_csv(csv_buffer, index=False)
        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        st.download_button(
            label="Download CSV",
            data=csv_buffer.getvalue(),
            file_name=f"{csv_filename_base}_{timestamp_str}.csv",
            mime="text/csv",
            key=f"{key_prefix}_dl_csv",
            use_container_width=True,
        )

    # Metadata Summary Bar
    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center; 
                    font-size: 0.78rem; color: #94a3b8; margin: 4px 0 8px 2px;">
            <div>
                Showing <b>{len(display_df)}</b> of <b>{filtered_records}</b> filtered records 
                (Total dataset: <b>{total_records}</b>)
            </div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #64748b;">
                Auto-indexed • Monospace aligned
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Render Table with customized Streamlit view
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Mathematical Chart Engines: Heikin-Ashi & Technical Overlays
# ─────────────────────────────────────────────────────────────────────────────
def compute_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform standard OHLC bars into smoothed Heikin-Ashi candles:
      HA_Close = (Open + High + Low + Close) / 4
      HA_Open_t = (HA_Open_{t-1} + HA_Close_{t-1}) / 2
      HA_High = max(High, HA_Open, HA_Close)
      HA_Low  = min(Low, HA_Open, HA_Close)
    """
    if df is None or len(df) == 0:
        return df

    ha_df = df.copy()
    opens = df["Open"].values
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    n = len(df)

    ha_close = (opens + highs + lows + closes) / 4.0
    ha_open = np.zeros(n, dtype=np.float64)
    ha_high = np.zeros(n, dtype=np.float64)
    ha_low = np.zeros(n, dtype=np.float64)

    ha_open[0] = (opens[0] + closes[0]) / 2.0
    ha_high[0] = max(highs[0], ha_open[0], ha_close[0])
    ha_low[0] = min(lows[0], ha_open[0], ha_close[0])

    for i in range(1, n):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0
        ha_high[i] = max(highs[i], ha_open[i], ha_close[i])
        ha_low[i] = min(lows[i], ha_open[i], ha_close[i])

    ha_df["Open"] = ha_open
    ha_df["High"] = ha_high
    ha_df["Low"] = ha_low
    ha_df["Close"] = ha_close
    ha_df["HA_Open"] = ha_open
    ha_df["HA_High"] = ha_high
    ha_df["HA_Low"] = ha_low
    ha_df["HA_Close"] = ha_close
    return ha_df


def compute_chart_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate full mathematical indicator overlays for institutional charting:
      - EMA 9, EMA 20, EMA 50, EMA 200
      - Dynamic VWAP with ±1σ and ±2σ envelopes
      - Bollinger Bands (20, 2σ)
      - ATR (14) & Trailing Ratchet Stops
      - RSI (14)
      - MACD (12, 26, 9)
    """
    if df is None or len(df) == 0:
        return df

    res = df.copy()
    close = res["Close"]
    high = res["High"]
    low = res["Low"]
    volume = res["Volume"] if "Volume" in res else pd.Series(1000.0, index=res.index)

    # 1. Exponential Moving Averages
    res["EMA_9"] = close.ewm(span=9, adjust=False).mean()
    res["EMA_20"] = close.ewm(span=20, adjust=False).mean()
    res["EMA_50"] = close.ewm(span=50, adjust=False).mean()
    res["EMA_200"] = close.ewm(span=200, adjust=False).mean()

    # 2. Dynamic VWAP + Standard Deviation Bands
    typical_price = (high + low + close) / 3.0
    cum_vol = volume.cumsum().replace(0, 1.0)
    cum_pv = (typical_price * volume).cumsum()
    vwap = cum_pv / cum_vol
    res["VWAP"] = vwap

    # VWAP Variance & Standard Deviation Bands
    dev_sq = (typical_price - vwap) ** 2
    cum_dev_sq_vol = (dev_sq * volume).cumsum()
    vwap_std = np.sqrt(np.maximum(0.0, cum_dev_sq_vol / cum_vol))
    res["VWAP_Upper_1"] = vwap + vwap_std
    res["VWAP_Lower_1"] = vwap - vwap_std
    res["VWAP_Upper_2"] = vwap + (2.0 * vwap_std)
    res["VWAP_Lower_2"] = vwap - (2.0 * vwap_std)

    # 3. Bollinger Bands (20, 2σ)
    sma_20 = close.rolling(20, min_periods=1).mean()
    std_20 = close.rolling(20, min_periods=1).std().fillna(0.0)
    res["BB_Middle"] = sma_20
    res["BB_Upper"] = sma_20 + (2.0 * std_20)
    res["BB_Lower"] = sma_20 - (2.0 * std_20)

    # 4. Average True Range (14) & Ratchet Trailing Floor
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    res["ATR_14"] = tr.rolling(14, min_periods=1).mean()
    res["ATR_Trailing_Stop"] = np.maximum.accumulate(close - (2.0 * res["ATR_14"]))

    # 5. Relative Strength Index (RSI 14)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(14, min_periods=1).mean()
    avg_loss = loss.rolling(14, min_periods=1).mean().replace(0, 1e-9)
    rs = avg_gain / avg_loss
    res["RSI_14"] = 100.0 - (100.0 / (1.0 + rs))

    # 6. MACD (12, 26, 9)
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    res["MACD_Line"] = macd_line
    res["MACD_Signal"] = signal_line
    res["MACD_Hist"] = macd_line - signal_line

    return res


# ─────────────────────────────────────────────────────────────────────────────
# 3. Hardware & Telemetry SVG Gauges
# ─────────────────────────────────────────────────────────────────────────────
def render_svg_gauge(
    percentage: float,
    label: str,
    value_text: str,
    color: str = "#10b981",
    size: int = 110,
    sub_title: str = "",
    sub_desc: str = "",
) -> str:
    """
    Renders an institutional circular SVG progress dial gauge.
    """
    pct = max(0.0, min(100.0, float(percentage)))
    radius = 42
    circumference = 2 * np.pi * radius
    stroke_offset = circumference - (pct / 100.0) * circumference

    html = (
        f'<div class="glass-panel" style="text-align: center; padding: 16px 12px; margin-bottom: 8px;">'
        f'<svg width="{size}" height="{size}" viewBox="0 0 100 100">'
        f'<circle cx="50" cy="50" r="{radius}" fill="none" stroke="rgba(255,255,255,0.08)" stroke-width="8"/>'
        f'<circle cx="50" cy="50" r="{radius}" fill="none" stroke="{color}" stroke-width="8" '
        f'stroke-dasharray="{circumference}" stroke-dashoffset="{stroke_offset}" stroke-linecap="round" '
        f'transform="rotate(-90 50 50)" style="transition: stroke-dashoffset 0.6s ease;"/>'
        f'<text x="50" y="47" font-family="\'JetBrains Mono\', monospace" font-size="14" font-weight="700" fill="#f8fafc" text-anchor="middle" dominant-baseline="middle">{pct:.0f}%</text>'
        f'<text x="50" y="62" font-family="\'Plus Jakarta Sans\', sans-serif" font-size="8" font-weight="600" fill="#94a3b8" text-anchor="middle">{label}</text>'
        f'</svg>'
        f'<div style="font-family: \'JetBrains Mono\', monospace; font-size: 0.82rem; color: #cbd5e1; margin-top: 6px;">{value_text}</div>'
    )
    if sub_title:
        html += f'<div style="font-weight: 700; color: {color}; font-size: 0.85rem; margin-top: 6px;">{sub_title}</div>'
    if sub_desc:
        html += f'<div style="font-size: 0.72rem; color: #94a3b8; margin-top: 2px;">{sub_desc}</div>'
    html += '</div>'
    return html
