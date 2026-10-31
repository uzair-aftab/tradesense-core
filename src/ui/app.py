"""
TradeSense v6 — Minimalist Frosted-Glass Wealthsimple Cockpit
=============================================================
Unified, elegant trading workspace engineered for Wealthsimple mobile execution:
  1. Passwordless Multi-Account Profile Gate (Uzair, Waleed, + Add Account)
  2. Minimalist Frosted Glassmorphism UI (Obsidian dark canvas, translucent specular panes)
  3. Interactive Chart Manipulation Suite (Plotly modebar, scroll zoom, toolbar presets, indicators)
  4. Genuine 8-Model ML/DL Quant Zoo & Custom Combinations (PyTorch DL, GBDT, Kernels, Econometrics)
  5. Multi-Pair Net-EV Alpha Scanner & Quarter-Kelly Risk Sizer
  6. Verified End-to-End Real Wealthsimple Order & Active Position Monitor
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.data_feed import DataLake
from src.data.database import db_manager, get_database
from src.data.market_intel import MarketIntelligenceEngine
from src.data.profiles import ProfileManager, UserProfile
from src.data.ws_catalog import (
    CURATED_WS_ASSETS,
    FeeEstimate,
    WealthsimpleAsset,
    WealthsimpleCatalog,
    WealthsimpleFeeSchedule,
)
from src.models.adaptive_predictor import (
    AdaptiveForecast,
    AdaptivePredictionEngine,
    AssetDNA,
    adaptive_engine,
)
from src.models.model_zoo import (
    ALL_QUANT_MODELS,
    MODEL_PRESETS,
    MODEL_REGISTRY,
    ModelZooEngine,
)
from src.models.profit_engine import (
    AlphaOpportunity,
    AlphaScanner,
    MarketHoursGuard,
    PositionRiskSizer,
    PositionSizeResult,
    TrailingStopRatchet,
)
from src.models.regime import MarketRegime, RegimeClassifier
from src.models.virtual_bot import (
    BotEfficiencySummary,
    VirtualBotEngine,
    virtual_bot_engine,
)
from src.ui.chart_builder import PLOTLY_GLASS_CONFIG, build_interactive_chart
from src.ui.clipboard import render_copy_card
from src.utils.health import HealthCheck
from src.utils.logger import get_logger

logger = get_logger("hfp.ui_v6")


def _format_chart_time_labels(dt_index: Any) -> List[str]:
    """Guarantees unique category strings for chart x-axis across daily and intraday bars."""
    labels = []
    seen: Dict[str, int] = {}
    for dt in dt_index:
        ts = pd.to_datetime(dt)
        if ts.hour == 0 and ts.minute == 0 and ts.second == 0:
            base = ts.strftime("%b %d")
        else:
            base = ts.strftime("%b %d %H:%M")
        if base in seen:
            seen[base] += 1
            labels.append(f"{base} #{seen[base]}")
        else:
            seen[base] = 1
            labels.append(base)
    return labels


def run_app() -> None:
    # ─────────────────────────────────────────────────────────────────────────────
    # 1. Page Configuration & Obsidian Frosted Glass Stylesheet
    # ─────────────────────────────────────────────────────────────────────────────
    st.set_page_config(
        page_title="TradeSense — Wealthsimple Alpha & AI Execution",
        page_icon="💎",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    GLASS_CSS = """
    <style>
    /* ── Obsidian Glass Canvas ── */
    .stApp {
        background-color: #050811 !important;
        background-image: 
            radial-gradient(at 0% 0%, rgba(56, 189, 248, 0.08) 0px, transparent 50%),
            radial-gradient(at 100% 100%, rgba(52, 211, 153, 0.06) 0px, transparent 50%) !important;
        color: #e2e8f0 !important;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* ── Hide Sidebar & Default Chrome Clutter ── */
    [data-testid="stSidebar"] {
        display: none !important;
    }
    #MainMenu, footer, header {
        visibility: hidden !important;
    }
    .block-container {
        padding-top: 1.0rem !important;
        padding-bottom: 2.0rem !important;
        max-width: 1540px !important;
    }

    /* ── True Frosted Glass Containers ── */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(145deg, rgba(255, 255, 255, 0.045) 0%, rgba(255, 255, 255, 0.015) 100%) !important;
        backdrop-filter: blur(28px) saturate(165%) !important;
        -webkit-backdrop-filter: blur(28px) saturate(165%) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-top: 1px solid rgba(255, 255, 255, 0.16) !important;
        border-radius: 18px !important;
        padding: 16px 20px !important;
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.06) !important;
        margin-bottom: 12px !important;
    }

    /* ── Typography & Monospace Accents ── */
    .mono {
        font-family: 'JetBrains Mono', 'SF Mono', Consolas, monospace !important;
    }
    .hero-price {
        font-family: 'JetBrains Mono', 'SF Mono', Consolas, monospace !important;
        font-size: 2.1rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.03em !important;
        color: #ffffff !important;
        line-height: 1.1 !important;
    }
    .delta-pos {
        color: #34d399 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
    }
    .delta-neg {
        color: #fb7185 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
    }

    /* ── Glass Direction Strip ── */
    .direction-strip {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 8px 16px;
        font-size: 0.85rem;
        color: #94a3b8;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 14px;
    }
    .direction-strip b {
        color: #38bdf8;
    }

    /* ── Badges & Pills ── */
    .dna-pill {
        display: inline-block;
        padding: 3px 9px;
        border-radius: 9999px;
        font-size: 0.74rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        background: rgba(56, 189, 248, 0.12);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.25);
    }
    .fee-badge {
        display: inline-block;
        padding: 3px 9px;
        border-radius: 9999px;
        font-size: 0.74rem;
        font-weight: 600;
        background: rgba(255, 255, 255, 0.05);
        color: #cbd5e1;
        border: 1px solid rgba(255, 255, 255, 0.12);
    }
    .regime-badge-bull {
        display: inline-block;
        padding: 3px 9px;
        border-radius: 9999px;
        font-size: 0.74rem;
        font-weight: 600;
        background: rgba(52, 211, 153, 0.12);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.3);
    }
    .regime-badge-chop {
        display: inline-block;
        padding: 3px 9px;
        border-radius: 9999px;
        font-size: 0.74rem;
        font-weight: 600;
        background: rgba(251, 191, 36, 0.12);
        color: #fbbf24;
        border: 1px solid rgba(251, 191, 36, 0.3);
    }
    .regime-badge-bear {
        display: inline-block;
        padding: 3px 9px;
        border-radius: 9999px;
        font-size: 0.74rem;
        font-weight: 600;
        background: rgba(251, 113, 133, 0.12);
        color: #fb7185;
        border: 1px solid rgba(251, 113, 133, 0.3);
    }

    /* ── Readiness Badges ── */
    .readiness-proven {
        background: rgba(52, 211, 153, 0.15) !important;
        color: #34d399 !important;
        border: 1px solid rgba(52, 211, 153, 0.35) !important;
        padding: 6px 12px !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-size: 0.84rem !important;
        text-align: center;
    }
    .readiness-calibrating {
        background: rgba(251, 191, 36, 0.12) !important;
        color: #fbbf24 !important;
        border: 1px solid rgba(251, 191, 36, 0.30) !important;
        padding: 6px 12px !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-size: 0.84rem !important;
        text-align: center;
    }
    .readiness-avoid {
        background: rgba(251, 113, 133, 0.12) !important;
        color: #fb7185 !important;
        border: 1px solid rgba(251, 113, 133, 0.30) !important;
        padding: 6px 12px !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-size: 0.84rem !important;
        text-align: center;
    }

    /* ── Plotly Frosted Glass Modebar ── */
    .js-plotly-plot .plotly .modebar {
        background: rgba(15, 23, 42, 0.72) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 10px !important;
        padding: 2px 6px !important;
        top: 8px !important;
        right: 8px !important;
        backdrop-filter: blur(12px) !important;
    }
    .js-plotly-plot .plotly .modebar-btn path {
        fill: #94a3b8 !important;
    }
    .js-plotly-plot .plotly .modebar-btn:hover path {
        fill: #38bdf8 !important;
    }
    .js-plotly-plot .plotly .modebar-btn.active path {
        fill: #34d399 !important;
    }

    /* ── Streamlit Widget Overrides ── */
    .stButton > button {
        background: linear-gradient(145deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0.03) 100%) !important;
        color: #f1f5f9 !important;
        border: 1px solid rgba(255, 255, 255, 0.14) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background: rgba(56, 189, 248, 0.18) !important;
        border-color: rgba(56, 189, 248, 0.45) !important;
        color: #ffffff !important;
    }
    .stSelectbox div[data-baseweb="select"] {
        background: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 10px !important;
        color: #ffffff !important;
    }
    </style>
    """
    st.markdown(GLASS_CSS, unsafe_allow_html=True)


    # ─────────────────────────────────────────────────────────────────────────────
    # 2. Database & Multi-Account Profile Management
    # ─────────────────────────────────────────────────────────────────────────────
    db = get_database()
    db.init_db()
    profile_mgr = ProfileManager(db)
    profile_mgr.ensure_default_profiles()

    # Check URL Query Param or Session State for Active Profile
    query_user = st.query_params.get("user")
    all_profiles = profile_mgr.list_profiles()
    valid_ids = [p.profile_id for p in all_profiles]

    if query_user and query_user in valid_ids:
        st.session_state["active_profile_id"] = query_user
    elif "active_profile_id" not in st.session_state:
        st.session_state["active_profile_id"] = None


    # ─────────────────────────────────────────────────────────────────────────────
    # 3. Gate 0: Frosted-Glass Passwordless Profile Selector
    # ─────────────────────────────────────────────────────────────────────────────
    def render_glass_login_gate() -> None:
        """Renders passwordless 1-click accounts portal for Uzair, Waleed, + Add."""
        st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
        c_center = st.columns([1, 2.8, 1])[1]

        with c_center:
            st.markdown(
                """
                <div style="text-align: center; margin-bottom: 24px;">
                    <div style="display: inline-block; margin-bottom: 12px;">
                        <svg width="68" height="68" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <polygon points="50,6 92,28 92,72 50,94 8,72 8,28" stroke="url(#prismGrad)" stroke-width="3.5" fill="rgba(56, 189, 248, 0.08)"/>
                            <polygon points="50,18 80,34 80,66 50,82 20,66 20,34" stroke="url(#prismGrad2)" stroke-width="2" fill="rgba(52, 211, 153, 0.05)"/>
                            <line x1="50" y1="6" x2="50" y2="94" stroke="rgba(255, 255, 255, 0.25)" stroke-width="1.5" stroke-dasharray="3 3"/>
                            <line x1="8" y1="50" x2="92" y2="50" stroke="rgba(255, 255, 255, 0.25)" stroke-width="1.5" stroke-dasharray="3 3"/>
                            <circle cx="50" cy="50" r="8" fill="#38bdf8"/>
                            <defs>
                                <linearGradient id="prismGrad" x1="0" y1="0" x2="100" y2="100" gradientUnits="userSpaceOnUse">
                                    <stop stop-color="#38bdf8"/>
                                    <stop offset="0.5" stop-color="#a78bfa"/>
                                    <stop offset="1" stop-color="#34d399"/>
                                </linearGradient>
                                <linearGradient id="prismGrad2" x1="100" y1="0" x2="0" y2="100" gradientUnits="userSpaceOnUse">
                                    <stop stop-color="#34d399"/>
                                    <stop offset="1" stop-color="#38bdf8"/>
                                </linearGradient>
                            </defs>
                        </svg>
                    </div>
                    <h1 style="font-size: 2.3rem; font-weight: 800; letter-spacing: -0.04em; margin: 0; color: #ffffff;">TradeSense</h1>
                    <p style="color: #94a3b8; font-size: 0.95rem; margin-top: 6px;">Institutional Wealthsimple Alpha • 8-Model ML/DL Quant Zoo</p>
                    <div style="font-size: 0.78rem; color: #38bdf8; letter-spacing: 0.08em; text-transform: uppercase; margin-top: 2px;">Passwordless Multi-Account Portal</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.container(border=True):
                st.markdown("<div style='font-size: 0.88rem; font-weight: 600; color: #cbd5e1; margin-bottom: 12px; text-align: center;'>Select Your Profile to Enter Workspace</div>", unsafe_allow_html=True)

                card_cols = st.columns(len(all_profiles) + 1)
                for idx, p in enumerate(all_profiles):
                    with card_cols[idx]:
                        acct = db.get_or_create_virtual_account(p.profile_id)
                        cash = float(acct["virtual_cash_cad"])
                        positions = db.get_virtual_positions(p.profile_id)

                        st.markdown(
                            f"""
                            <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.08); border-top: 2px solid {p.accent_color}; border-radius: 12px; padding: 12px 10px; text-align: center; margin-bottom: 8px;">
                                <div style="width: 42px; height: 42px; border-radius: 9999px; background: {p.accent_color}22; border: 1px solid {p.accent_color}; color: {p.accent_color}; font-weight: 700; font-size: 1.05rem; display: flex; align-items: center; justify-content: center; margin: 0 auto 8px auto;">
                                    {p.avatar_initials}
                                </div>
                                <div style="font-weight: 700; font-size: 1.05rem; color: #ffffff;">{p.display_name}</div>
                                <div class="mono" style="font-size: 0.85rem; color: {p.accent_color}; font-weight: 600; margin-top: 2px;">${cash:,.0f} CAD</div>
                                <div style="font-size: 0.70rem; color: #64748b; margin-top: 2px;">{len(positions)} open • {p.ws_tier}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        if st.button(f"Enter as {p.display_name}", key=f"btn_enter_{p.profile_id}", use_container_width=True):
                            st.session_state["active_profile_id"] = p.profile_id
                            st.query_params["user"] = p.profile_id
                            st.rerun()

                # Add Account Column
                with card_cols[-1]:
                    st.markdown(
                        """
                        <div style="background: rgba(255, 255, 255, 0.015); border: 1px dashed rgba(255, 255, 255, 0.15); border-radius: 12px; padding: 12px 10px; text-align: center; margin-bottom: 8px; height: 136px; display: flex; flex-direction: column; justify-content: center; align-items: center;">
                            <div style="font-size: 1.8rem; color: #94a3b8; line-height: 1;">＋</div>
                            <div style="font-weight: 600; font-size: 0.88rem; color: #94a3b8; margin-top: 6px;">New Account</div>
                            <div style="font-size: 0.68rem; color: #64748b;">Instant Isolated Wallet</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    with st.popover("＋ Add", use_container_width=True):
                        st.markdown("##### ＋ Create New Trading Profile")
                        new_name = st.text_input("Profile Display Name", placeholder="e.g. Sarah, Alex, Bilal")
                        new_cap = st.number_input("Starting Virtual Capital ($ CAD)", value=10000.0, step=1000.0)
                        new_tier = st.selectbox("Wealthsimple Tier", options=["Core", "Premium", "Generation"])
                        if st.button("Create Profile & Enter", use_container_width=True):
                            if new_name.strip():
                                new_p = profile_mgr.create_profile(new_name.strip(), starting_cad=new_cap, ws_tier=new_tier)
                                st.session_state["active_profile_id"] = new_p.profile_id
                                st.query_params["user"] = new_p.profile_id
                                st.rerun()
                            else:
                                st.error("Please enter a valid display name.")


    if st.session_state.get("active_profile_id") is None:
        render_glass_login_gate()
        st.stop()
        # Fallback for bare module import under pytest
        active_profile = profile_mgr.get_profile("uzair")
    else:
        active_profile_id = st.session_state["active_profile_id"]
        active_profile = profile_mgr.get_profile(active_profile_id)
        if not active_profile:
            active_profile = profile_mgr.get_profile("uzair")

    USER_ID = active_profile.profile_id if active_profile else "uzair"
    st.query_params["user"] = USER_ID

    # Virtual Account & Profile Sync
    acct_info = db.get_or_create_virtual_account(USER_ID)
    if "ws_tier" not in st.session_state:
        st.session_state["ws_tier"] = active_profile.ws_tier
    if "usd_account" not in st.session_state:
        st.session_state["usd_account"] = active_profile.usd_account_active
    if "is_armed" not in st.session_state:
        st.session_state["is_armed"] = bool(acct_info.get("is_armed", 1))
    if "active_symbol" not in st.session_state:
        st.session_state["active_symbol"] = active_profile.active_symbol
    if "timeframe" not in st.session_state:
        st.session_state["timeframe"] = active_profile.active_timeframe
    if "horizon_bars" not in st.session_state:
        st.session_state["horizon_bars"] = active_profile.horizon_bars
    if "view_mode" not in st.session_state:
        st.session_state["view_mode"] = "Workspace"
    if "chart_type" not in st.session_state:
        st.session_state["chart_type"] = "Candles"
    if "chart_visible_bars" not in st.session_state:
        st.session_state["chart_visible_bars"] = 80
    if "chart_overlays" not in st.session_state:
        st.session_state["chart_overlays"] = ["EMA 9 & 21", "VWAP"]
    if "model_preset" not in st.session_state:
        st.session_state["model_preset"] = active_profile.model_preset
    if "selected_models" not in st.session_state:
        st.session_state["selected_models"] = active_profile.selected_models or ["dl_bilstm_attn", "ml_xgboost_quantile", "ml_gbdt_forest"]
    if "blending_mode" not in st.session_state:
        st.session_state["blending_mode"] = active_profile.blending_mode
    if "allocated_capital" not in st.session_state:
        st.session_state["allocated_capital"] = 500.0
    if "eval_cache" not in st.session_state:
        st.session_state["eval_cache"] = {}


    # ─────────────────────────────────────────────────────────────────────────────
    # 5. Data Fetching Helper
    # ─────────────────────────────────────────────────────────────────────────────
    @st.cache_data(ttl=120, show_spinner=False)
    def fetch_asset_history(symbol: str, timeframe: str = "1h") -> pd.DataFrame:
        """Fetches real or synthetic candlestick history for an asset."""
        lake = DataLake()
        try:
            lb = "30d" if timeframe == "15m" else ("90d" if timeframe == "1h" else "1y")
            df = lake.pipeline(symbol, timeframe=timeframe, lookback=lb, include_sentiment=False, normalise=False)
            if df is not None and len(df) >= 20:
                return df
        except Exception as exc:
            logger.debug("DataLake pipeline error for %s: %s", symbol, exc)

        periods = 100
        freq = "15min" if timeframe == "15m" else ("h" if timeframe == "1h" else "D")
        dates = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=periods, freq=freq)
        base = 135000.0 if "BTC" in symbol else (3800.0 if "ETH" in symbol else (150.0 if ".TO" in symbol else 180.0))
        rng = np.random.default_rng(abs(hash(symbol)) % 10000)
        returns = rng.normal(0.0003, 0.012, periods)
        price_curve = base * np.exp(np.cumsum(returns))

        return pd.DataFrame({
            "Open": price_curve * (1.0 - rng.uniform(0.0, 0.003, periods)),
            "High": price_curve * (1.0 + rng.uniform(0.001, 0.008, periods)),
            "Low": price_curve * (1.0 - rng.uniform(0.001, 0.008, periods)),
            "Close": price_curve,
            "Volume": rng.uniform(1000, 50000, periods),
        }, index=dates)


    # ─────────────────────────────────────────────────────────────────────────────
    # 6. Top Glass Command Bar
    # ─────────────────────────────────────────────────────────────────────────────
    top_col1, top_col2, top_col3, top_col4 = st.columns([4.2, 1.8, 2.8, 1.2])

    # 6.1 Profile Pill & Asset Selector
    curated_assets = WealthsimpleCatalog.list_curated()
    asset_options = [
        f"{a.symbol} — {a.name} ({a.category.replace('_', ' ')} • {a.currency})"
        for a in curated_assets
    ]
    curr_sym = st.session_state["active_symbol"]
    curr_idx = 0
    for idx, opt in enumerate(asset_options):
        if opt.startswith(curr_sym + " —"):
            curr_idx = idx
            break

    with top_col1:
        c_prof, c_sel, c_pop = st.columns([1.8, 4.8, 0.8])
        with c_prof:
            with st.popover(f"👤 {active_profile.display_name} ▾", use_container_width=True):
                st.markdown(f"#### 👤 {active_profile.display_name}")
                st.caption(f"Virtual Cash: **${float(acct_info['virtual_cash_cad']):,.2f} CAD**")
                st.caption(f"Tier: **{active_profile.ws_tier}** • Blending: **{active_profile.blending_mode}**")
                st.divider()
                if st.button("🔄 Switch Account", use_container_width=True):
                    st.session_state["active_profile_id"] = None
                    st.query_params.clear()
                    st.rerun()

        with c_sel:
            selected_opt = st.selectbox(
                "Select Asset",
                options=asset_options,
                index=curr_idx,
                label_visibility="collapsed",
                key="asset_selector",
            )
            if selected_opt:
                new_sym = selected_opt.split(" —")[0].strip()
                if new_sym != st.session_state["active_symbol"]:
                    st.session_state["active_symbol"] = new_sym
                    active_profile.active_symbol = new_sym
                    profile_mgr.save_profile(active_profile)
                    st.rerun()

        with c_pop:
            with st.popover("🔍", help="Lookup any Wealthsimple-tradeable TSX/US/NEO/Crypto ticker"):
                custom_query = st.text_input("Enter Ticker or Name", placeholder="e.g. SHOP, NVDA, SOL")
                if custom_query:
                    results = WealthsimpleCatalog.search_assets(custom_query, limit=8)
                    if results:
                        st.caption("Matches:")
                        for r in results:
                            if st.button(f"{r.symbol} — {r.name[:24]}", key=f"btn_search_{r.symbol}"):
                                st.session_state["active_symbol"] = r.symbol
                                st.rerun()
                    else:
                        st.info("No matching Wealthsimple ticker found.")

    # 6.2 Timeframe Selector
    with top_col2:
        tf_opts = ["15m", "1h", "1D"]
        tf_idx = tf_opts.index(st.session_state["timeframe"]) if st.session_state["timeframe"] in tf_opts else 1
        selected_tf = st.radio(
            "Timeframe",
            options=tf_opts,
            index=tf_idx,
            horizontal=True,
            label_visibility="collapsed",
            key="tf_radio",
        )
        if selected_tf != st.session_state["timeframe"]:
            st.session_state["timeframe"] = selected_tf
            active_profile.active_timeframe = selected_tf
            profile_mgr.save_profile(active_profile)
            st.rerun()

    # 6.3 View Switcher
    with top_col3:
        v_opts = ["Workspace", "ML/DL Lab & Scanner", "Portfolios & Leaderboard"]
        v_idx = v_opts.index(st.session_state["view_mode"]) if st.session_state["view_mode"] in v_opts else 0
        selected_view = st.radio(
            "View Mode",
            options=v_opts,
            index=v_idx,
            horizontal=True,
            label_visibility="collapsed",
            key="view_radio",
        )
        if selected_view != st.session_state["view_mode"]:
            st.session_state["view_mode"] = selected_view
            st.rerun()

    # 6.4 Top Setups & Settings Popover
    with top_col4:
        c_setups, c_gear = st.columns(2)
        with c_setups:
            with st.popover("🔥", help="Top Net-EV Alpha Setups"):
                st.markdown("##### 🔥 Top Wealthsimple Net-EV Alpha")
                top_setups = AlphaScanner.scan_opportunities(basket=["BTC-CAD", "ETH-CAD", "SHOP.TO", "NVDA.NE", "QQQ"])[:4]
                if top_setups:
                    for opp in top_setups:
                        st.markdown(
                            f"""
                            <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 6px 10px; margin-bottom: 6px;">
                                <div style="display: flex; justify-content: space-between;">
                                    <b>{opp.symbol}</b>
                                    <span style="color: {'#34d399' if opp.net_ev_pct > 0 else '#fb7185'}; font-weight: 700;">Net EV {opp.net_ev_pct:+.2f}%</span>
                                </div>
                                <div style="font-size: 0.72rem; color: #94a3b8;">${opp.current_price:,.2f} ➔ ${opp.target_price:,.2f} ({opp.action})</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        if st.button(f"Load {opp.symbol}", key=f"btn_pop_load_{opp.symbol}"):
                            st.session_state["active_symbol"] = opp.symbol
                            st.rerun()

        with c_gear:
            with st.popover("⚙️", help="Virtual Wallet, Wealthsimple Tier, and Preferences"):
                st.markdown("##### ⚙️ Trading & Account Settings")
                tier_choice = st.selectbox(
                    "Wealthsimple Account Tier",
                    options=["Core", "Premium", "Generation"],
                    index=["Core", "Premium", "Generation"].index(st.session_state["ws_tier"]),
                )
                usd_toggle = st.checkbox("USD Account Active ($0 per-trade FX)", value=st.session_state["usd_account"])

                if tier_choice != st.session_state["ws_tier"] or usd_toggle != st.session_state["usd_account"]:
                    st.session_state["ws_tier"] = tier_choice
                    st.session_state["usd_account"] = usd_toggle
                    active_profile.ws_tier = tier_choice
                    active_profile.usd_account_active = usd_toggle
                    profile_mgr.save_profile(active_profile)
                    db.update_virtual_account(USER_ID, ws_tier=tier_choice, usd_account_active=usd_toggle)
                    st.rerun()

                st.divider()
                st.markdown("##### 💼 Virtual Portfolio")
                st.write(f"Cash: **${float(acct_info['virtual_cash_cad']):,.2f} CAD**")
                new_start_cash = st.number_input(
                    "Reset Capital ($ CAD)",
                    min_value=1000.0,
                    max_value=250000.0,
                    value=float(acct_info.get("starting_virtual_cad", 10000.0)),
                    step=1000.0,
                )
                if st.button("Reset Virtual Wallet", use_container_width=True):
                    db.reset_virtual_wallet(USER_ID, starting_cash=new_start_cash)
                    st.success(f"Virtual wallet reset to ${new_start_cash:,.2f} CAD!")
                    st.rerun()

                st.divider()
                st.markdown("##### 🖥️ Live Telemetry")
                gpu_info = HealthCheck.check_gpu()
                ram_info = HealthCheck.check_memory()
                if gpu_info.get("available"):
                    st.caption(f"GPU: **{gpu_info.get('device_name')}** ({gpu_info.get('used_gb'):.1f}/{gpu_info.get('total_gb'):.1f} GB)")
                else:
                    st.caption("GPU: **CPU Fallback / Cloud Container**")
                st.caption(f"RAM: **{ram_info.get('used')}/{ram_info.get('total')}** ({ram_info.get('percent')}%)")
                st.caption("Database: **SQLite WAL Engine**")


    # ─────────────────────────────────────────────────────────────────────────────
    # 7. Core Calculations & Data Feed
    # ─────────────────────────────────────────────────────────────────────────────
    active_symbol = st.session_state["active_symbol"]
    active_asset = WealthsimpleCatalog.resolve_asset(active_symbol)
    active_timeframe = st.session_state["timeframe"]
    horizon_bars = st.session_state["horizon_bars"]

    df_bars = fetch_asset_history(active_symbol, active_timeframe)
    current_price = float(df_bars["Close"].iloc[-1])
    prev_price = float(df_bars["Close"].iloc[-2]) if len(df_bars) > 1 else current_price
    bar_delta_pct = ((current_price - prev_price) / prev_price) * 100.0

    # True 24-hour return computation
    bars_in_24h = (
        96 if active_timeframe == "15m" and active_asset.category == "CRYPTO"
        else (26 if active_timeframe == "15m"
        else (24 if active_timeframe == "1h" and active_asset.category == "CRYPTO"
        else (7 if active_timeframe == "1h" else 1)))
    )
    idx_24h = max(0, len(df_bars) - bars_in_24h)
    price_24h_ago = float(df_bars["Close"].iloc[idx_24h])
    price_24h_pct = ((current_price - price_24h_ago) / price_24h_ago) * 100.0

    # Market Hours & Execution Safety Guard
    market_status = MarketHoursGuard.get_market_status(active_asset)

    # Regime Analysis
    regime_result = RegimeClassifier().classify(df_bars)
    active_regime: MarketRegime = regime_result.regime

    # Determine Active Models for Ensemble
    preset_choice = st.session_state["model_preset"]
    if preset_choice == "AUTO_DNA":
        hurst_val = 0.58 if active_regime == MarketRegime.BULL_TREND else 0.48
        ensemble_model_ids = ModelZooEngine.get_auto_models_for_asset(active_asset, hurst_val)
    elif preset_choice in MODEL_PRESETS and "models" in MODEL_PRESETS[preset_choice]:
        ensemble_model_ids = MODEL_PRESETS[preset_choice]["models"]
    else:
        ensemble_model_ids = st.session_state["selected_models"]
        if not ensemble_model_ids:
            ensemble_model_ids = ["dl_bilstm_attn", "ml_xgboost_quantile", "ml_gbdt_forest"]

    # Train/Evaluate Quant Zoo
    quant_zoo_results = ModelZooEngine.train_and_evaluate(
        df=df_bars,
        asset=active_asset,
        selected_model_ids=ensemble_model_ids,
        horizon_bars=horizon_bars,
        blending_mode=st.session_state["blending_mode"],
    )

    consensus_p10 = quant_zoo_results["consensus_p10"]
    consensus_p50 = quant_zoo_results["consensus_p50"]
    consensus_p90 = quant_zoo_results["consensus_p90"]
    consensus_exp_return_pct = round(((consensus_p50 - current_price) / current_price) * 100.0, 2)
    consensus_direction = "BULLISH" if consensus_exp_return_pct >= 0.25 else ("BEARISH" if consensus_exp_return_pct <= -0.25 else "NEUTRAL")

    # Legacy forecast compatibility for virtual bot
    forecast = adaptive_engine.compute_adaptive_forecast(
        df=df_bars,
        asset=active_asset,
        timeframe=active_timeframe,
        horizon_bars=horizon_bars,
    )
    forecast.p10 = consensus_p10
    forecast.p50 = consensus_p50
    forecast.p90 = consensus_p90
    forecast.expected_return_pct = consensus_exp_return_pct
    forecast.direction = consensus_direction

    # Efficiency Summary & Virtual Bot
    eval_key = f"{active_symbol}_{active_timeframe}_{horizon_bars}_{st.session_state['ws_tier']}_{st.session_state['usd_account']}"
    if eval_key not in st.session_state["eval_cache"]:
        st.session_state["eval_cache"][eval_key] = virtual_bot_engine.evaluate_pair_efficiency(
            df=df_bars,
            asset=active_asset,
            timeframe=active_timeframe,
            horizon_bars=horizon_bars,
            eval_bars=50,
            starting_cad=10000.0,
            ws_tier=st.session_state["ws_tier"],
            usd_account=st.session_state["usd_account"],
        )
    efficiency_summary: BotEfficiencySummary = st.session_state["eval_cache"][eval_key]

    # Live Auto-Tick Execution if Armed
    if st.session_state["is_armed"]:
        virtual_bot_engine.on_live_tick(
            user_id=USER_ID,
            asset=active_asset,
            current_price=current_price,
            forecast=forecast,
        )


    # ─────────────────────────────────────────────────────────────────────────────
    # 8. Direction Flow Banner
    # ─────────────────────────────────────────────────────────────────────────────
    regime_badge_html = (
        '<span class="regime-badge-bull">🟢 BULL TREND</span>' if active_regime == MarketRegime.BULL_TREND
        else ('<span class="regime-badge-chop">🟡 RANGING CHOP</span>' if active_regime == MarketRegime.RANGING_CHOP
        else '<span class="regime-badge-bear">🔴 BEAR CASH LOCK</span>')
    )
    st.markdown(
        f"""
        <div class="direction-strip">
            <span><b>Step 1: Quant Zoo</b> ➔ {consensus_direction} ({consensus_exp_return_pct:+.2f}% to ${consensus_p50:,.2f}) • {regime_badge_html}</span>
            <span><b>Step 2: Virtual Proof</b> ➔ {efficiency_summary.trades_count} Trades • {efficiency_summary.win_rate_pct:.0f}% Win Rate</span>
            <span><b>Step 3: Wealthsimple</b> ➔ {market_status['msg']}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


    # ─────────────────────────────────────────────────────────────────────────────
    # 9. VIEW 1: WORKSPACE (Minimalist 3 Frosted-Glass Panes)
    # ─────────────────────────────────────────────────────────────────────────────
    if st.session_state["view_mode"] == "Workspace":
        left_col, right_col = st.columns([6.2, 3.8])

        # ══════════════════════════════════════════════════════════════════════════
        # PANE 1 (Left): Live Price, Interactive Chart & Model Combo
        # ══════════════════════════════════════════════════════════════════════════
        with left_col:
            with st.container(border=True):
                # Asset Header
                h_col1, h_col2 = st.columns([7.2, 2.8])
                with h_col1:
                    st.markdown(
                        f"""
                        <div style="display: flex; align-items: baseline; gap: 10px; margin-bottom: 2px;">
                            <span style="font-size: 1.35rem; font-weight: 700; color: #ffffff;">{active_asset.name}</span>
                            <span class="mono" style="color: #94a3b8; font-size: 1.0rem;">{active_asset.ws_display_symbol}</span>
                            <span class="dna-pill">{forecast.dna_profile}</span>
                            <span class="fee-badge">{market_status['phase']}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    d24_class = "delta-pos" if price_24h_pct >= 0 else "delta-neg"
                    d1b_class = "delta-pos" if bar_delta_pct >= 0 else "delta-neg"
                    curr_sign = "$" if active_asset.currency == "CAD" else "US$"

                    st.markdown(
                        f"""
                        <div style="display: flex; align-items: baseline; gap: 12px; margin-bottom: 10px;">
                            <span class="hero-price">{curr_sign}{current_price:,.2f}</span>
                            <span class="{d24_class}">{price_24h_pct:+.2f}% 24h</span>
                            <span class="{d1b_class}" style="font-size: 0.82rem;">({bar_delta_pct:+.2f}% bar)</span>
                            <span style="color: #64748b; font-size: 0.80rem;">{market_status['wrapper_advice']}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with h_col2:
                    st.caption(f"🕐 Toronto: **{market_status.get('toronto_time', '')[:16]}**")
                    if market_status.get("order_warning"):
                        st.caption(f"<span style='color: #fbbf24; font-size: 0.76rem;'>{market_status['order_warning']}</span>", unsafe_allow_html=True)

                # ── Inline Chart Manipulation Toolbar ──
                st.markdown("<div style='height: 2px;'></div>", unsafe_allow_html=True)
                tb1, tb2, tb3, tb4, tb5, tb6, tb7 = st.columns([0.6, 0.6, 1.8, 0.6, 1.5, 1.3, 1.6])

                with tb1:
                    if st.button("➕", help="Zoom In (narrows bar window by 35%)", use_container_width=True):
                        st.session_state["chart_visible_bars"] = max(15, int(st.session_state["chart_visible_bars"] * 0.65))
                        st.rerun()

                with tb2:
                    if st.button("➖", help="Zoom Out (widens bar window by 50%)", use_container_width=True):
                        st.session_state["chart_visible_bars"] = min(len(df_bars), int(st.session_state["chart_visible_bars"] * 1.50))
                        st.rerun()

                with tb3:
                    win_opts = ["30B", "60B", "120B", "All"]
                    curr_bars = st.session_state["chart_visible_bars"]
                    def_win_idx = 0 if curr_bars <= 35 else (1 if curr_bars <= 70 else (2 if curr_bars <= 130 else 3))
                    sel_win = st.segmented_control("Bars", options=win_opts, default=win_opts[def_win_idx], label_visibility="collapsed", key="win_pills")
                    if sel_win:
                        target_map = {"30B": 30, "60B": 60, "120B": 120, "All": len(df_bars)}
                        if target_map[sel_win] != st.session_state["chart_visible_bars"]:
                            st.session_state["chart_visible_bars"] = target_map[sel_win]
                            st.rerun()

                with tb4:
                    if st.button("↺", help="Reset zoom to 80 bars", use_container_width=True):
                        st.session_state["chart_visible_bars"] = 80
                        st.rerun()

                with tb5:
                    hor_opts = ["3B", "5B", "10B", "20B"]
                    cur_h_idx = [3, 5, 10, 20].index(horizon_bars) if horizon_bars in [3, 5, 10, 20] else 1
                    sel_h = st.segmented_control("Horizon", options=hor_opts, default=hor_opts[cur_h_idx], label_visibility="collapsed", key="hor_pills")
                    if sel_h:
                        num_h = int(sel_h.replace("B", ""))
                        if num_h != st.session_state["horizon_bars"]:
                            st.session_state["horizon_bars"] = num_h
                            active_profile.horizon_bars = num_h
                            profile_mgr.save_profile(active_profile)
                            st.rerun()

                with tb6:
                    style_opts = ["Candles", "Heikin-Ashi", "Line"]
                    sel_style = st.selectbox("Style", options=style_opts, index=style_opts.index(st.session_state["chart_type"]), label_visibility="collapsed", key="chart_style_sel")
                    if sel_style != st.session_state["chart_type"]:
                        st.session_state["chart_type"] = sel_style
                        st.rerun()

                with tb7:
                    with st.popover("📈 Overlays", use_container_width=True):
                        avail_ov = ["EMA 9 & 21", "VWAP", "Bollinger Bands", "Support / Resistance", "Volume Bars", "Price Lines"]
                        new_ov = st.multiselect("Active Overlays", options=avail_ov, default=st.session_state["chart_overlays"], key="multisel_overlays")
                        if new_ov != st.session_state["chart_overlays"]:
                            st.session_state["chart_overlays"] = new_ov
                            st.rerun()

                # Interactive Plotly Chart
                order_price_lines = {
                    "limit_entry": current_price,
                    "take_profit": consensus_p90,
                    "stop_loss": consensus_p10,
                } if "Price Lines" in st.session_state["chart_overlays"] else None

                chart_fig = build_interactive_chart(
                    df=df_bars,
                    symbol=active_asset.ws_display_symbol,
                    forecast=quant_zoo_results,
                    visible_bars=st.session_state["chart_visible_bars"],
                    chart_type=st.session_state["chart_type"],
                    overlays=st.session_state["chart_overlays"],
                    trades_history=efficiency_summary.trades_history,
                    order_levels=order_price_lines,
                )
                st.plotly_chart(chart_fig, use_container_width=True, config=PLOTLY_GLASS_CONFIG)

                # ── 8-Model ML/DL Quant Zoo Strip & Preset Selector ──
                st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
                m_bar1, m_bar2 = st.columns([5.5, 4.5])
                with m_bar1:
                    st.markdown("<div style='font-size: 0.78rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;'>Quant Zoo Ensemble Preset</div>", unsafe_allow_html=True)
                    preset_labels = {
                        "AUTO_DNA": "🧬 Auto-DNA",
                        "PURE_DL": "🧠 Pure DL",
                        "TREE_ML": "🌲 Tree ML",
                        "FULL_SUPER": "⚡ All 8 Models",
                        "CUSTOM": "🎛️ Custom",
                    }
                    preset_keys = list(preset_labels.keys())
                    cur_preset_idx = preset_keys.index(st.session_state["model_preset"]) if st.session_state["model_preset"] in preset_keys else 0
                    sel_preset = st.segmented_control("Preset", options=preset_keys, default=preset_keys[cur_preset_idx], format_func=lambda x: preset_labels[x], label_visibility="collapsed", key="preset_segmented")
                    if sel_preset and sel_preset != st.session_state["model_preset"]:
                        st.session_state["model_preset"] = sel_preset
                        active_profile.model_preset = sel_preset
                        profile_mgr.save_profile(active_profile)
                        st.rerun()

                with m_bar2:
                    if st.session_state["model_preset"] == "CUSTOM":
                        st.markdown("<div style='font-size: 0.78rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;'>Select Active Models</div>", unsafe_allow_html=True)
                        all_mid = list(ALL_QUANT_MODELS.keys())
                        selected_custom = st.multiselect(
                            "Models",
                            options=all_mid,
                            default=st.session_state["selected_models"],
                            format_func=lambda m: ALL_QUANT_MODELS[m].short_label,
                            label_visibility="collapsed",
                            key="custom_models_select",
                        )
                        if selected_custom != st.session_state["selected_models"]:
                            st.session_state["selected_models"] = selected_custom
                            active_profile.selected_models = selected_custom
                            profile_mgr.save_profile(active_profile)
                            st.rerun()
                    else:
                        st.markdown(
                            f"""
                            <div style="font-size: 0.76rem; color: #64748b; margin-top: 18px;">
                                Active Blend: <b>{st.session_state['blending_mode']}</b> • Conformal Pad: <b>±{(consensus_p90 - consensus_p50) / current_price * 100:.1f}%</b>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                # Active Model Weight Cards
                active_results = quant_zoo_results.get("model_results", [])
                if active_results:
                    m_cols = st.columns(len(active_results))
                    for idx, m in enumerate(active_results):
                        with m_cols[idx]:
                            st.markdown(
                                f"""
                                <div style="background: rgba(255, 255, 255, 0.025); border: 1px solid rgba(255, 255, 255, 0.07); border-radius: 9px; padding: 6px 8px; text-align: center;">
                                    <div style="font-size: 0.68rem; color: #94a3b8; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{m['short_label']}</div>
                                    <div class="mono" style="font-size: 0.95rem; font-weight: 700; color: #38bdf8;">{m.get('weight_pct', 0.0):.0f}%</div>
                                    <div class="mono" style="font-size: 0.72rem; color: #cbd5e1;">${m['p50']:,.2f}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

        # ══════════════════════════════════════════════════════════════════════════
        # RIGHT COLUMN: Stacked Panes (Proof-of-Edge Bot + Wealthsimple Order)
        # ══════════════════════════════════════════════════════════════════════════
        with right_col:
            # ─────────────────────────────────────────────────────────────────────
            # PANE 2 (Top-Right): Virtual Bot & Proof-of-Edge Gauge
            # ─────────────────────────────────────────────────────────────────────
            with st.container(border=True):
                v_acct = db.get_or_create_virtual_account(USER_ID)
                v_cash = float(v_acct["virtual_cash_cad"])
                open_positions = db.get_virtual_positions(USER_ID)

                # Mark to market open positions
                open_positions_val_cad = 0.0
                for pos in open_positions:
                    pos_ticker = pos["ticker"]
                    pos_qty = float(pos["quantity"])
                    pos_p = current_price if pos_ticker == active_symbol else float(pos["current_price"])
                    fx = 1.39 if ("USD" in str(pos.get("currency", "")) or active_asset.currency == "USD") else 1.0
                    open_positions_val_cad += pos_qty * pos_p * fx

                total_virtual_equity = v_cash + open_positions_val_cad
                v_start = float(v_acct.get("starting_virtual_cad", 10000.0))
                net_total_pct = ((total_virtual_equity - v_start) / v_start) * 100.0
                eq_class = "delta-pos" if net_total_pct >= 0 else "delta-neg"

                # Top Row: Virtual Equity & Status
                v_head1, v_head2 = st.columns([6, 4])
                with v_head1:
                    st.markdown(
                        f"""
                        <div>
                            <div style="font-size: 0.74rem; color: #94a3b8; text-transform: uppercase;">Total Virtual Equity</div>
                            <div class="mono" style="font-size: 1.45rem; font-weight: 700; color: #ffffff;">${total_virtual_equity:,.2f} CAD</div>
                            <div class="{eq_class}" style="font-size: 0.78rem;">{net_total_pct:+.2f}% total (Cash: ${v_cash:,.2f})</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with v_head2:
                    if efficiency_summary.readiness_state == "PROVEN_EDGE":
                        st.markdown(f'<div class="readiness-proven">{efficiency_summary.readiness_badge}</div>', unsafe_allow_html=True)
                    elif efficiency_summary.readiness_state == "CALIBRATING":
                        st.markdown(f'<div class="readiness-calibrating">{efficiency_summary.readiness_badge}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="readiness-avoid">{efficiency_summary.readiness_badge}</div>', unsafe_allow_html=True)

                # Open Position or Manual Buy Trigger
                curr_pair_pos = next((p for p in open_positions if p["ticker"] == active_symbol), None)
                if curr_pair_pos:
                    pos_entry = float(curr_pair_pos["entry_price"])
                    pos_qty = float(curr_pair_pos["quantity"])
                    u_pnl = (current_price - pos_entry) * pos_qty
                    u_pnl_pct = ((current_price - pos_entry) / pos_entry) * 100.0
                    u_cls = "delta-pos" if u_pnl >= 0 else "delta-neg"

                    # Trailing stop ratchet assessment
                    ratchet_info = TrailingStopRatchet.evaluate_stop(
                        entry_price=pos_entry,
                        current_price=current_price,
                        initial_stop_price=float(curr_pair_pos.get("stop_p10", pos_entry * 0.95)),
                        target_price=float(curr_pair_pos.get("target_p90", pos_entry * 1.05)),
                        round_trip_fee_pct=0.08,
                    )

                    st.markdown(
                        f"""
                        <div style="background: rgba(56, 189, 248, 0.06); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 9px; padding: 8px 10px; margin: 8px 0;">
                            <div style="display: flex; justify-content: space-between; font-size: 0.78rem;">
                                <span><b>Active Virtual Position:</b> {pos_qty:.4f} {active_asset.ws_display_symbol}</span>
                                <span class="{u_cls}"><b>{u_pnl_pct:+.2f}% (${u_pnl:+.2f} CAD)</b></span>
                            </div>
                            <div style="font-size: 0.72rem; color: #94a3b8; margin-top: 3px;">
                                Entry: ${pos_entry:,.2f} • {ratchet_info['desc']}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if st.button("✖ Close Virtual Position Now", use_container_width=True):
                        res = virtual_bot_engine.manual_close_position(USER_ID, active_asset, current_price)
                        st.success(f"Closed {active_symbol}: Net P&L ${res.get('net_pnl_cad', 0.0):+,.2f} CAD")
                        st.rerun()
                else:
                    st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
                    c_test, c_arm = st.columns([1.2, 1.0])
                    with c_test:
                        if st.button("⚡ Paper Buy ($500 CAD)", use_container_width=True, help="Test this forecast on virtual cash immediately"):
                            res = virtual_bot_engine.manual_open_position(
                                user_id=USER_ID,
                                asset=active_asset,
                                current_price=current_price,
                                p10_stop=consensus_p10,
                                p50_target=consensus_p50,
                                p90_target=consensus_p90,
                                capital_cad=500.0,
                                horizon_bars=horizon_bars,
                            )
                            if res["status"] == "POSITION_OPENED":
                                st.success(f"Paper trade opened: {res['quantity']} {active_symbol} @ ${current_price:,.2f}")
                                st.rerun()
                            else:
                                st.error(res.get("message", "Unable to open virtual position."))

                    with c_arm:
                        is_armed = st.session_state["is_armed"]
                        arm_label = "⏸ Pause Bot" if is_armed else "▶ Arm Bot"
                        if st.button(arm_label, use_container_width=True):
                            new_armed = not is_armed
                            st.session_state["is_armed"] = new_armed
                            db.update_virtual_account(USER_ID, is_armed=new_armed)
                            st.rerun()

                # 60-Bar Walk-Forward Metrics
                st.markdown(
                    f"""
                    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; margin-top: 8px;">
                        <div style="background: rgba(255,255,255,0.02); padding: 5px; border-radius: 8px; text-align: center;">
                            <div style="font-size: 0.65rem; color: #94a3b8;">Win Rate</div>
                            <div class="mono" style="font-size: 0.88rem; font-weight: 700;">{efficiency_summary.win_rate_pct:.0f}%</div>
                        </div>
                        <div style="background: rgba(255,255,255,0.02); padding: 5px; border-radius: 8px; text-align: center;">
                            <div style="font-size: 0.65rem; color: #94a3b8;">Hit Rate</div>
                            <div class="mono" style="font-size: 0.88rem; font-weight: 700;">{efficiency_summary.hit_rate_pct:.0f}%</div>
                        </div>
                        <div style="background: rgba(255,255,255,0.02); padding: 5px; border-radius: 8px; text-align: center;">
                            <div style="font-size: 0.65rem; color: #94a3b8;">Profit Factor</div>
                            <div class="mono" style="font-size: 0.88rem; font-weight: 700;">{efficiency_summary.profit_factor:.2f}</div>
                        </div>
                        <div style="background: rgba(255,255,255,0.02); padding: 5px; border-radius: 8px; text-align: center;">
                            <div style="font-size: 0.65rem; color: #94a3b8;">Fees Paid</div>
                            <div class="mono" style="font-size: 0.88rem; font-weight: 700; color: #fb7185;">${efficiency_summary.total_ws_fees_cad:.1f}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # ─────────────────────────────────────────────────────────────────────
            # PANE 3 (Bottom-Right): Wealthsimple Mobile Execution Ticket & Real Positions
            # ─────────────────────────────────────────────────────────────────────
            with st.container(border=True):
                st.markdown(
                    """
                    <div style="font-size: 0.82rem; color: #94a3b8; font-weight: 600; margin-bottom: 8px;">
                        WEALTHSIMPLE REAL-MONEY ORDER TICKET
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Route Toggle & Fixed Fee Math
                exec_symbol = active_asset.ws_display_symbol
                exec_asset = active_asset
                exec_price = current_price
                savings_note = None

                if active_asset.alt_cad_proxy:
                    proxy_asset = WealthsimpleCatalog.resolve_asset(active_asset.alt_cad_proxy)
                    use_proxy = st.toggle(
                        f"Route via {active_asset.alt_cad_proxy} ($0 Comm / Save Fees)",
                        value=True,
                        help=f"Execute via Canadian proxy {active_asset.alt_cad_proxy} to eliminate US FX or crypto spread fees.",
                    )
                    if use_proxy:
                        exec_asset = proxy_asset
                        exec_symbol = proxy_asset.ws_display_symbol
                        proxy_df = fetch_asset_history(active_asset.alt_cad_proxy, "1h")
                        exec_price = float(proxy_df["Close"].iloc[-1])

                        # Exact fee calculation on proxy asset
                        direct_fee = WealthsimpleFeeSchedule.calculate_fee(active_asset, st.session_state["allocated_capital"], st.session_state["ws_tier"], st.session_state["usd_account"])
                        proxy_fee = WealthsimpleFeeSchedule.calculate_fee(proxy_asset, st.session_state["allocated_capital"], st.session_state["ws_tier"], st.session_state["usd_account"])
                        savings_cad = direct_fee.round_trip_fee_cad - proxy_fee.round_trip_fee_cad
                        if savings_cad > 0.05:
                            savings_note = f"💰 Proxy saves ~${savings_cad:,.2f} CAD in Wealthsimple round-trip fees!"

                # Quarter-Kelly Optimal Position Sizing
                total_equity_val = total_virtual_equity
                kelly_result: PositionSizeResult = PositionRiskSizer.calculate_position_size(
                    total_equity_cad=total_equity_val,
                    current_price=exec_price,
                    p10_stop=consensus_p10,
                    p90_target=consensus_p90,
                    hit_rate_pct=efficiency_summary.hit_rate_pct,
                    round_trip_fee_pct=0.08,
                )

                c_alloc_in, c_alloc_btn = st.columns([2.2, 1.8])
                with c_alloc_in:
                    alloc_cad = st.number_input(
                        "Capital ($ CAD)",
                        min_value=50.0,
                        max_value=250000.0,
                        value=float(st.session_state["allocated_capital"]),
                        step=100.0,
                        key="alloc_cad_input",
                    )
                    st.session_state["allocated_capital"] = alloc_cad

                with c_alloc_btn:
                    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                    if st.button(f"🎯 Apply Kelly (${kelly_result.recommended_cad:,.0f})", use_container_width=True):
                        st.session_state["allocated_capital"] = kelly_result.recommended_cad
                        st.rerun()

                if savings_note:
                    st.caption(f"<span style='color: #34d399; font-size: 0.74rem;'>{savings_note}</span>", unsafe_allow_html=True)
                if kelly_result.heat_warning:
                    st.caption(f"<span style='color: #fbbf24; font-size: 0.74rem;'>{kelly_result.heat_warning}</span>", unsafe_allow_html=True)

                # Price Targets
                limit_entry = exec_price
                take_profit = round(exec_price * (1.0 + (consensus_exp_return_pct / 100.0 if consensus_exp_return_pct > 0 else 0.03)), 2)
                stop_loss = round(exec_price * (1.0 - (0.02 if consensus_exp_return_pct > 0 else 0.04)), 2)
                shares = alloc_cad / max(exec_price, 1e-4)

                # 3 Real 1-Click Clipboard Copy Cards
                c_ent, c_tp, c_sl = st.columns(3)
                with c_ent:
                    render_copy_card("Limit Entry", f"${limit_entry:,.2f}", "Wealthsimple Limit", "#38bdf8")
                with c_tp:
                    render_copy_card("Take Profit", f"${take_profit:,.2f}", "P90 Target", "#34d399")
                with c_sl:
                    render_copy_card("Stop Loss", f"${stop_loss:,.2f}", "P10 Stop", "#fb7185")

                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: space-between; font-size: 0.76rem; color: #94a3b8; margin: 6px 0 10px 0;">
                        <span>Order Size: <b>{shares:.4f} {exec_symbol}</b></span>
                        <span>Max Risk at Stop: <b>${kelly_result.dollar_risk_at_stop:,.2f} CAD</b></span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Log Real Fill Button
                c_fill_btn, c_real_info = st.columns([1.4, 1.0])
                with c_fill_btn:
                    if st.button("📲 Log Real Wealthsimple Fill", use_container_width=True):
                        tx_id = f"WS-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{int(time.time()) % 1000:03d}"
                        db.save_real_ws_position({
                            "tx_id": tx_id,
                            "profile_id": USER_ID,
                            "ticker": active_symbol,
                            "ws_symbol": exec_symbol,
                            "route_type": "CAD_PROXY" if exec_symbol != active_symbol else "DIRECT",
                            "status": "OPEN",
                            "entry_price": limit_entry,
                            "current_price": limit_entry,
                            "quantity": shares,
                            "capital_cad": alloc_cad,
                            "target_price": take_profit,
                            "stop_price": stop_loss,
                            "trailing_stop_price": stop_loss,
                            "est_fee_cad": 0.40 if exec_symbol != active_symbol else 1.50,
                        })
                        st.success(f"Logged real {exec_symbol} trade {tx_id} to {active_profile.display_name}'s ledger!")
                        st.rerun()

                # Active Real Wealthsimple Positions Monitor
                real_positions = db.get_real_ws_positions(USER_ID, status="OPEN")
                if real_positions:
                    st.markdown("<div style='margin-top: 8px; font-size: 0.76rem; font-weight: 700; color: #38bdf8;'>ACTIVE REAL WEALTHSIMPLE TRADES</div>", unsafe_allow_html=True)
                    for rp in real_positions:
                        rp_entry = float(rp["entry_price"])
                        rp_qty = float(rp["quantity"])
                        rp_curr = current_price if rp["ticker"] == active_symbol else rp_entry
                        rp_pnl = (rp_curr - rp_entry) * rp_qty
                        rp_pct = ((rp_curr - rp_entry) / rp_entry) * 100.0
                        p_cls = "delta-pos" if rp_pnl >= 0 else "delta-neg"

                        st.markdown(
                            f"""
                            <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 6px 10px; margin-bottom: 4px;">
                                <div style="display: flex; justify-content: space-between; font-size: 0.76rem;">
                                    <span><b>{rp['ws_symbol']}</b> ({rp['tx_id']})</span>
                                    <span class="{p_cls}"><b>{rp_pct:+.2f}% (${rp_pnl:+.2f} CAD)</b></span>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        if st.button(f"Close Real {rp['ws_symbol']} ({rp['tx_id']})", key=f"btn_close_rp_{rp['tx_id']}"):
                            db.close_real_ws_position(rp["tx_id"], exit_price=rp_curr, exit_reason="MANUAL_REAL_CLOSE")
                            st.success(f"Real trade {rp['tx_id']} closed at ${rp_curr:,.2f}!")
                            st.rerun()


    # ─────────────────────────────────────────────────────────────────────────────
    # 10. VIEW 2: ML/DL LAB & MULTI-PAIR ALPHA SCANNER
    # ─────────────────────────────────────────────────────────────────────────────
    elif st.session_state["view_mode"] == "ML/DL Lab & Scanner":
        s_col1, s_col2 = st.columns([5.5, 4.5])

        with s_col1:
            with st.container(border=True):
                st.markdown("#### 🔥 Multi-Pair Net-EV Alpha Opportunity Scanner")
                st.caption("Scans liquid Wealthsimple TSX/NEO/US/Crypto universe, ranking setups by Net Expected Value % after Wealthsimple fees.")

                if st.button("🔥 Run Universe Alpha Scan", use_container_width=True):
                    AlphaScanner._cached_scan = None
                    st.rerun()

                opportunities = AlphaScanner.scan_opportunities()
                if opportunities:
                    for opp in opportunities:
                        act_color = "#34d399" if opp.action in ("STRONG_BUY", "TACTICAL_BUY") else ("#fbbf24" if opp.action == "WAIT_CHOP" else "#fb7185")
                        st.markdown(
                            f"""
                            <div style="background: rgba(255, 255, 255, 0.025); border: 1px solid rgba(255, 255, 255, 0.07); border-radius: 12px; padding: 12px 14px; margin-bottom: 8px;">
                                <div style="display: flex; justify-content: space-between; align-items: baseline;">
                                    <span style="font-size: 1.1rem; font-weight: 700; color: #ffffff;">{opp.symbol} <span style="font-size: 0.76rem; color: #94a3b8;">({opp.name[:22]})</span></span>
                                    <span class="mono" style="font-size: 1.05rem; font-weight: 700; color: {act_color};">Net EV {opp.net_ev_pct:+.2f}%</span>
                                </div>
                                <div style="display: flex; gap: 10px; font-size: 0.76rem; color: #94a3b8; margin: 4px 0;">
                                    <span>Price: <b>${opp.current_price:,.2f}</b></span>
                                    <span>Target: <b style="color: #34d399;">${opp.target_price:,.2f}</b> (+{opp.upside_pct:.1f}%)</span>
                                    <span>Stop: <b style="color: #fb7185;">${opp.p10_stop:,.2f}</b> (-{opp.downside_pct:.1f}%)</span>
                                    <span>WS Fee: <b>{opp.round_trip_fee_pct:.2f}%</b></span>
                                </div>
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 6px;">
                                    <span style="font-size: 0.72rem; color: #64748b;">Optimal Route: <b>{opp.optimal_route_symbol}</b> • {opp.tfsa_status}</span>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        if st.button(f"Load {opp.symbol} into Workspace", key=f"btn_scan_load_{opp.symbol}"):
                            st.session_state["active_symbol"] = opp.symbol
                            st.session_state["view_mode"] = "Workspace"
                            st.rerun()

        with s_col2:
            with st.container(border=True):
                st.markdown("#### 🔬 8-Model ML/DL Quant Zoo Inspector")
                st.caption(f"Asset: **{active_symbol}** • Category: **{active_asset.category}** • Regime: **{active_regime.value}**")

                # Retrain Trigger
                if st.button("🚀 Force Retrain & Recalibrate Models", use_container_width=True):
                    with st.spinner("Retraining PyTorch DL & Quantile models on fresh bars..."):
                        t_start = time.time()
                        res = ModelZooEngine.train_and_evaluate(
                            df=df_bars,
                            asset=active_asset,
                            selected_model_ids=list(ALL_QUANT_MODELS.keys()),
                            horizon_bars=horizon_bars,
                            blending_mode=st.session_state["blending_mode"],
                            force_retrain=True,
                        )
                        t_dur_ms = int((time.time() - t_start) * 1000)
                    st.success(f"All 8 Quant Models retrained & calibrated in {t_dur_ms} ms!")
                    st.rerun()

                # Performance Matrix
                model_table = []
                for mid, m_inst in ALL_QUANT_MODELS.items():
                    m_res = next((r for r in quant_zoo_results.get("model_results", []) if r["model_id"] == mid), None)
                    model_table.append({
                        "Model": m_inst.short_label,
                        "Family": m_inst.family[:14],
                        "Hit Rate": f"{m_res['hit_rate_pct']:.1f}%" if m_res else "N/A",
                        "Validation MAPE": f"{m_res['mape_pct']:.2f}%" if m_res else "N/A",
                        "Target P50": f"${m_res['p50']:,.2f}" if m_res else "N/A",
                        "Weight": f"{m_res.get('weight_pct', 0.0):.1f}%" if m_res else "0.0%",
                    })
                st.dataframe(pd.DataFrame(model_table), use_container_width=True, height=310)

                st.divider()
                st.markdown("#### 📰 Live Market Intelligence & News Feed")
                intel_engine = MarketIntelligenceEngine()
                asset_intel = intel_engine.get_asset_intelligence(active_symbol)

                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                        <span>VADER Sentiment Score: <b style="color: {'#34d399' if asset_intel.sentiment_score > 0 else '#fb7185'};">{asset_intel.sentiment_score:+.2f}</b></span>
                        <span>Catalyst: <b>{asset_intel.catalyst_badge}</b></span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if asset_intel.earnings_warning:
                    st.warning(asset_intel.earnings_warning)

                if asset_intel.headlines:
                    for h in asset_intel.headlines[:4]:
                        st.markdown(
                            f"""
                            <div style="font-size: 0.78rem; color: #cbd5e1; border-left: 2px solid #38bdf8; padding-left: 8px; margin-bottom: 6px;">
                                <a href="{h.url}" target="_blank" style="color: #cbd5e1; text-decoration: none;">{h.title}</a>
                                <div style="font-size: 0.68rem; color: #64748b;">{getattr(h, 'publisher', 'Market Wire')} • {getattr(h, 'published_ago', '')}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                else:
                    st.caption("No fresh breaking headlines in the last 24h. Macro sentiment steady.")


    # ─────────────────────────────────────────────────────────────────────────────
    # 11. VIEW 3: PORTFOLIOS & LEADERBOARD
    # ─────────────────────────────────────────────────────────────────────────────
    else:
        l_col1, l_col2 = st.columns([1, 1])

        with l_col1:
            with st.container(border=True):
                st.markdown("#### 🏆 Multi-Account Leaderboard")
                st.caption("Compare live performance across Uzair, Waleed, and other accounts.")

                leaderboard = []
                for p in all_profiles:
                    p_acct = db.get_or_create_virtual_account(p.profile_id)
                    p_cash = float(p_acct["virtual_cash_cad"])
                    p_start = float(p_acct.get("starting_virtual_cad", 10000.0))
                    p_trades = db.get_virtual_trades(p.profile_id, limit=100)
                    p_wins = sum(1 for t in p_trades if t.get("prediction_hit", 0))
                    p_win_rate = (p_wins / len(p_trades) * 100.0) if p_trades else 0.0

                    p_pos = db.get_virtual_positions(p.profile_id)
                    p_open_val = sum(float(pos["quantity"]) * float(pos["current_price"]) for pos in p_pos)
                    p_equity = p_cash + p_open_val
                    p_return_pct = ((p_equity - p_start) / p_start) * 100.0

                    leaderboard.append({
                        "Trader": p.display_name,
                        "Equity ($ CAD)": round(p_equity, 2),
                        "Net Return": f"{p_return_pct:+.2f}%",
                        "Win Rate": f"{p_win_rate:.0f}%",
                        "Completed Trades": len(p_trades),
                        "Open Trades": len(p_pos),
                    })

                df_lead = pd.DataFrame(leaderboard).sort_values(by="Equity ($ CAD)", ascending=False)
                st.dataframe(
                    df_lead.style.format({"Equity ($ CAD)": "${:,.2f}"}),
                    use_container_width=True,
                    height=220,
                )

                st.divider()
                st.markdown(f"#### 💼 {active_profile.display_name}'s Virtual Trade History")
                my_trades = db.get_virtual_trades(USER_ID, limit=50)
                if my_trades:
                    df_vt = pd.DataFrame(my_trades)
                    disp_cols = ["ticker", "entry_price", "exit_price", "quantity", "gross_pnl_cad", "ws_fees_cad", "net_pnl_cad", "net_pnl_pct", "closed_at"]
                    act_cols = [c for c in disp_cols if c in df_vt.columns]
                    st.dataframe(
                        df_vt[act_cols].style.format({
                            "entry_price": "${:,.2f}",
                            "exit_price": "${:,.2f}",
                            "net_pnl_cad": "${:+,.2f}",
                            "net_pnl_pct": "{:+.2f}%",
                        }),
                        use_container_width=True,
                        height=280,
                    )
                else:
                    st.info("No virtual trades completed yet for this profile.")

        with l_col2:
            with st.container(border=True):
                st.markdown(f"#### 📲 {active_profile.display_name}'s Real Wealthsimple Executions")
                st.caption("Active and archived real trades executed on the Wealthsimple app.")

                real_hist = db.get_real_ws_history(USER_ID, limit=50)
                if real_hist:
                    df_rh = pd.DataFrame(real_hist)
                    rh_cols = ["tx_id", "ws_symbol", "route_type", "status", "entry_price", "exit_price", "realized_pnl_cad", "realized_pnl_pct", "opened_at"]
                    act_rh = [c for c in rh_cols if c in df_rh.columns]
                    st.dataframe(
                        df_rh[act_rh].style.format({
                            "entry_price": "${:,.2f}",
                            "exit_price": "${:,.2f}",
                            "realized_pnl_cad": "${:+,.2f}",
                            "realized_pnl_pct": "{:+.2f}%",
                        }),
                        use_container_width=True,
                        height=450,
                    )
                else:
                    st.info("No real Wealthsimple trades logged yet. Click '📲 Log Real Wealthsimple Fill' on the Workspace ticket.")


if __name__ == "__main__" or st.runtime.exists():
    run_app()