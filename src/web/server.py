"""
TradeSense v7 — FastAPI Application Server & REST API
=====================================================
Powers the modern Slate-950 financial terminal:
  - Cookie-based authentication (zero ?user= URL clutter)
  - TradingView Lightweight Charts v4 backend feeds
  - Autonomous 60-second background trading bot task
  - Exact Wealthsimple fee schedule & CAD proxy routing
  - 8-Model ML/DL Quant Zoo evaluation & Split Conformal calibration
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from src.config import config
from src.data.data_feed import DataLake
from src.data.database import db_manager
from src.data.market_intel import MarketIntelligenceEngine
from src.data.profiles import UserProfile, profile_manager
from src.data.ws_catalog import WealthsimpleCatalog, WealthsimpleFeeSchedule
from src.models.model_zoo import MODEL_PRESETS, ModelZooEngine
from src.models.profit_engine import AlphaScanner, MarketHoursGuard, PositionRiskSizer, TrailingStopRatchet
from src.models.virtual_bot import virtual_bot_engine
from src.utils.health import health_check
from src.utils.logger import get_logger
from src.web.auth_cookie import (
    SESSION_COOKIE_NAME,
    SESSION_TTL_SECONDS,
    create_session_token,
    get_authenticated_profile,
    verify_session_token,
)

logger = get_logger("hfp.web")

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Background Bot Execution Task (Every 60s)
# ─────────────────────────────────────────────────────────────────────────────
async def _autonomous_bot_task():
    """Runs autonomous bot mark-to-market and order execution in the background."""
    logger.info("Autonomous Virtual Bot background runner started (60s cycle).")
    while True:
        try:
            await asyncio.sleep(60)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, virtual_bot_engine.run_autonomous_bot_cycle)
        except asyncio.CancelledError:
            logger.info("Autonomous Virtual Bot background runner stopped.")
            break
        except Exception as exc:
            logger.warning("Error in autonomous bot background runner: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup & shutdown lifespan."""
    # Ensure database initialized and default profiles created
    db_manager.init_db()
    profile_manager.ensure_default_profiles()

    bot_task = asyncio.create_task(_autonomous_bot_task())
    yield
    bot_task.cancel()
    try:
        await bot_task
    except asyncio.CancelledError:
        pass


app = FastAPI(
    title="TradeSense",
    version="7.0.0",
    description="Wealthsimple AI Alpha & Execution Platform",
    lifespan=lifespan,
)

# CORS Middleware (allows single-url gateway probe)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static and Templates mounting
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Dependency: Get Authenticated Profile
# ─────────────────────────────────────────────────────────────────────────────
def get_current_profile(request: Request) -> UserProfile:
    token = request.cookies.get(SESSION_COOKIE_NAME)
    prof = get_authenticated_profile(token)
    if not prof:
        # Fallback to default uzair profile if none exists
        prof = profile_manager.get_profile("uzair")
        if not prof:
            profiles = profile_manager.list_profiles()
            if profiles:
                prof = profiles[0]
    if not prof:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not logged in")
    return prof


def get_current_profile_optional(request: Request) -> Optional[UserProfile]:
    token = request.cookies.get(SESSION_COOKIE_NAME)
    return get_authenticated_profile(token)


# ─────────────────────────────────────────────────────────────────────────────
# 3. HTML Pages & Gateway Probe
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/favicon.ico")
async def favicon_ico():
    """
    Serves favicon with full CORS and no-store headers.
    This enables the GitHub Pages gateway (uzair-aftab.github.io/trade-sense)
    to probe the laptop server online status with 100% accuracy!
    """
    fav_path = STATIC_DIR / "img" / "favicon.svg"
    if fav_path.exists():
        content = fav_path.read_text(encoding="utf-8")
        return Response(
            content=content,
            media_type="image/svg+xml",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate",
                "Access-Control-Allow-Origin": "*",
            },
        )
    return Response(status_code=204)


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Clean Slate-950 passwordless profile selection portal."""
    profile_manager.ensure_default_profiles()
    profiles_raw = profile_manager.list_profiles()
    profiles_view = []

    for p in profiles_raw:
        acct = db_manager.get_or_create_virtual_account(p.profile_id)
        open_pos = db_manager.get_virtual_positions(p.profile_id)
        cash = float(acct.get("virtual_cash_cad", 10000.0))
        pos_val = sum(float(pos["quantity"]) * float(pos["current_price"]) for pos in open_pos)
        equity = cash + pos_val

        profiles_view.append({
            "profile_id": p.profile_id,
            "display_name": p.display_name,
            "avatar_initials": p.avatar_initials,
            "accent_color": p.accent_color,
            "ws_tier": p.ws_tier,
            "active_symbol": p.active_symbol,
            "active_timeframe": p.active_timeframe,
            "virtual_equity_cad": equity,
            "open_positions_count": len(open_pos),
        })

    toronto_time = MarketHoursGuard.get_market_status(
        WealthsimpleCatalog.resolve_asset("SHOP.TO")
    ).get("toronto_time", "Toronto Active")

    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={
            "profiles": profiles_view,
            "toronto_time": toronto_time,
        },
    )


@app.get("/", response_class=HTMLResponse)
async def index_page(request: Request):
    """Main institutional terminal page (protected by session cookie)."""
    prof = get_current_profile_optional(request)
    if not prof:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "profile": prof,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Authentication Endpoints
# ─────────────────────────────────────────────────────────────────────────────
class SelectProfileRequest(BaseModel):
    profile_id: str


@app.post("/api/auth/select-profile")
async def select_profile(payload: SelectProfileRequest, response: Response):
    prof = profile_manager.get_profile(payload.profile_id)
    if not prof:
        raise HTTPException(status_code=404, detail="Profile not found")

    token = create_session_token(prof.profile_id)
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        max_age=SESSION_TTL_SECONDS,
        httponly=True,
        samesite="lax",
        secure=False,
    )
    return {"status": "SUCCESS", "profile_id": prof.profile_id, "display_name": prof.display_name}


class CreateProfileRequest(BaseModel):
    display_name: str
    starting_cad: float = 10000.0
    ws_tier: str = "Core"


@app.post("/api/auth/create-profile")
async def create_profile(payload: CreateProfileRequest, response: Response):
    clean_name = payload.display_name.strip()
    if not clean_name:
        raise HTTPException(status_code=400, detail="Display name is required")

    prof = profile_manager.create_profile(
        display_name=clean_name,
        starting_cad=payload.starting_cad,
        ws_tier=payload.ws_tier,
    )
    token = create_session_token(prof.profile_id)
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        max_age=SESSION_TTL_SECONDS,
        httponly=True,
        samesite="lax",
        secure=False,
    )
    return {"status": "SUCCESS", "profile_id": prof.profile_id, "display_name": prof.display_name}


@app.post("/api/auth/logout")
async def logout(response: Response):
    response.delete_cookie(key=SESSION_COOKIE_NAME)
    return {"status": "SUCCESS"}


# ─────────────────────────────────────────────────────────────────────────────
# 5. Asset Catalog & Preferences Endpoints
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/catalog")
async def search_catalog(q: str = "", category: Optional[str] = None):
    results = WealthsimpleCatalog.search_assets(query=q, limit=20)
    if category and category != "ALL":
        results = [a for a in results if a.category == category]
    return {"assets": [a.to_dict() for a in results]}


class PreferencesUpdate(BaseModel):
    active_symbol: Optional[str] = None
    active_timeframe: Optional[str] = None
    horizon_bars: Optional[int] = None
    model_preset: Optional[str] = None
    selected_models: Optional[List[str]] = None
    blending_mode: Optional[str] = None
    ws_tier: Optional[str] = None
    usd_account_active: Optional[bool] = None
    use_proxy_route: Optional[bool] = None


@app.post("/api/preferences")
async def update_preferences(
    payload: PreferencesUpdate,
    prof: UserProfile = Depends(get_current_profile),
):
    updates = {k: v for k, v in payload.model_dump().items() if v is not None}
    updated = profile_manager.update_profile(prof.profile_id, **updates)
    return {"status": "SUCCESS", "profile": updated.to_dict() if updated else prof.to_dict()}


# ─────────────────────────────────────────────────────────────────────────────
# 6. Core Workspace Payload Endpoint (0ms Reactivity)
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/workspace")
async def get_workspace(
    retrain: bool = False,
    prof: UserProfile = Depends(get_current_profile),
):
    sym = prof.active_symbol
    primary_asset, exec_asset, is_proxied = WealthsimpleCatalog.get_execution_route(sym, use_proxy=prof.use_proxy_route)

    # 1. Market clock & wrapper rules
    market_clock = MarketHoursGuard.get_market_status(primary_asset)

    # 2. Fee estimate
    alloc_cad = 500.0
    direct_fee = WealthsimpleFeeSchedule.calculate_fee(primary_asset, alloc_cad, prof.ws_tier, prof.usd_account_active)
    exec_fee = WealthsimpleFeeSchedule.calculate_fee(exec_asset, alloc_cad, prof.ws_tier, prof.usd_account_active)
    proxy_savings = direct_fee.fee_cad_on_notional - exec_fee.fee_cad_on_notional

    # 3. Ingest OHLCV bars
    tf = prof.active_timeframe
    lookback = "60d" if tf == "1h" else ("15d" if tf == "15m" else "1y")
    df = DataLake().pipeline(sym, timeframe=tf, lookback=lookback, include_sentiment=False, normalise=False)

    if df is None or len(df) < 25:
        # Emergency fallback to daily if intraday unavailable off-hours
        df = DataLake().pipeline(sym, timeframe="1d", lookback="1y", include_sentiment=False, normalise=False)

    if df is None or len(df) < 5:
        raise HTTPException(status_code=500, detail=f"Failed to fetch market data for {sym}")

    curr_price = float(df["Close"].iloc[-1])
    prev_price = float(df["Close"].iloc[-2]) if len(df) > 1 else curr_price
    change_pct = ((curr_price - prev_price) / max(prev_price, 1e-4)) * 100.0

    # 4. Indicators computation for chart overlays
    closes = df["Close"].astype(float)
    highs = df["High"].astype(float)
    lows = df["Low"].astype(float)
    volumes = df["Volume"].astype(float)

    # Generate Unix seconds times
    chart_times = []
    for ts in df.index:
        if hasattr(ts, "timestamp"):
            chart_times.append(int(ts.timestamp()))
        else:
            chart_times.append(int(len(chart_times)))

    bars_list = []
    for i in range(len(df)):
        bars_list.append({
            "time": chart_times[i],
            "open": round(float(df["Open"].iloc[i]), 4),
            "high": round(float(highs.iloc[i]), 4),
            "low": round(float(lows.iloc[i]), 4),
            "close": round(float(closes.iloc[i]), 4),
            "volume": float(volumes.iloc[i]),
        })

    # Indicators
    ema9_vals = closes.ewm(span=9, adjust=False).mean()
    ema21_vals = closes.ewm(span=21, adjust=False).mean()
    roll20_vol = volumes.rolling(20, min_periods=3).sum().replace(0, 1.0)
    roll20_pv = (closes * volumes).rolling(20, min_periods=3).sum()
    vwap_vals = roll20_pv / roll20_vol
    bb_mean = closes.rolling(20, min_periods=5).mean()
    bb_std = closes.rolling(20, min_periods=5).std().replace(0, 1e-6)
    bb_up = bb_mean + 2 * bb_std
    bb_low = bb_mean - 2 * bb_std

    indicators = {
        "ema9": [{"time": chart_times[i], "value": round(float(ema9_vals.iloc[i]), 4)} for i in range(len(df)) if not np.isnan(ema9_vals.iloc[i])],
        "ema21": [{"time": chart_times[i], "value": round(float(ema21_vals.iloc[i]), 4)} for i in range(len(df)) if not np.isnan(ema21_vals.iloc[i])],
        "vwap": [{"time": chart_times[i], "value": round(float(vwap_vals.iloc[i]), 4)} for i in range(len(df)) if not np.isnan(vwap_vals.iloc[i])],
        "bb_upper": [{"time": chart_times[i], "value": round(float(bb_up.iloc[i]), 4)} for i in range(len(df)) if not np.isnan(bb_up.iloc[i])],
        "bb_lower": [{"time": chart_times[i], "value": round(float(bb_low.iloc[i]), 4)} for i in range(len(df)) if not np.isnan(bb_low.iloc[i])],
    }

    # 5. 8-Model ML/DL Quant Zoo evaluation
    active_models = prof.selected_models if (prof.model_preset == "CUSTOM" and prof.selected_models) else MODEL_PRESETS.get(prof.model_preset, {}).get("models")
    zoo_res = ModelZooEngine.train_and_evaluate(
        df=df,
        asset=primary_asset,
        active_model_ids=active_models,
        horizon_bars=prof.horizon_bars,
        blending_mode=prof.blending_mode,
        force_retrain=retrain,
    )

    cons_p10 = zoo_res["consensus_p10"]
    cons_p50 = zoo_res["consensus_p50"]
    cons_p90 = zoo_res["consensus_p90"]
    exp_ret_pct = zoo_res["exp_return_pct"]
    direction = "BULLISH" if exp_ret_pct > 0.15 else ("BEARISH" if exp_ret_pct < -0.15 else "NEUTRAL")

    # Forecast path projection for chart
    last_t = chart_times[-1]
    step_sec = (chart_times[-1] - chart_times[-2]) if len(chart_times) > 1 else 3600
    forecast_path = [
        {"time": last_t, "value": curr_price},
        {"time": last_t + step_sec * prof.horizon_bars, "value": cons_p50},
    ]

    # 6. Order ticket & Quarter-Kelly sizing
    acct = db_manager.get_or_create_virtual_account(prof.profile_id)
    virtual_cash = float(acct["virtual_cash_cad"])
    open_pos = db_manager.get_virtual_positions(prof.profile_id)
    pos_val = sum(float(p["quantity"]) * float(p["current_price"]) for p in open_pos)
    total_equity = virtual_cash + pos_val

    # Scale ticket prices to executed asset
    if is_proxied:
        df_exec = DataLake().pipeline(exec_asset.symbol, timeframe=tf, lookback="15d", include_sentiment=False, normalise=False)
        exec_price = float(df_exec["Close"].iloc[-1]) if (df_exec is not None and len(df_exec)) else curr_price
        exec_p10 = exec_price * (1.0 + (cons_p10 - curr_price) / curr_price)
        exec_p50 = exec_price * (1.0 + (cons_p50 - curr_price) / curr_price)
        exec_p90 = exec_price * (1.0 + (cons_p90 - curr_price) / curr_price)
    else:
        exec_price = curr_price
        exec_p10 = cons_p10
        exec_p50 = cons_p50
        exec_p90 = cons_p90

    active_hit_rates = [m["hit_rate_pct"] for m in zoo_res["all_model_results"] if m.get("is_active")]
    consensus_hit_rate = float(np.mean(active_hit_rates)) if active_hit_rates else 60.0

    sizer_res = PositionRiskSizer.calculate_position_size(
        total_equity_cad=total_equity,
        current_price=exec_price,
        p10_stop=exec_p10,
        p90_target=exec_p90,
        hit_rate_pct=consensus_hit_rate,
        round_trip_fee_pct=exec_fee.round_trip_total_pct,
    )

    ticket_alloc = 500.0
    ticket_shares = ticket_alloc / exec_price if exec_price > 0 else 0.0

    # 7. Walk-forward efficiency evaluation
    efficiency_summary = virtual_bot_engine.evaluate_pair_efficiency(
        df=df,
        asset=primary_asset,
        timeframe=tf,
        horizon_bars=prof.horizon_bars,
        use_proxy=prof.use_proxy_route,
        starting_cad=10000.0,
    )

    # 8. Active open position for this ticker if any
    active_virtual_pos = next((p for p in open_pos if p["ticker"] in (sym, exec_asset.symbol)), None)
    active_pos_dict = None
    if active_virtual_pos:
        pos_entry = float(active_virtual_pos["entry_price"])
        pos_qty = float(active_virtual_pos["quantity"])
        pos_mark = curr_price if active_virtual_pos["ticker"] == sym else exec_price
        unrealized_cad = (pos_mark - pos_entry) * pos_qty
        unrealized_pct = ((pos_mark - pos_entry) / max(pos_entry, 1e-4)) * 100.0

        ratchet_info = TrailingStopRatchet.evaluate_stop(
            entry_price=pos_entry,
            current_price=pos_mark,
            initial_stop_price=float(active_virtual_pos["stop_p10"]),
            target_price=float(active_virtual_pos["target_p90"]),
        )

        active_pos_dict = {
            "ticker": active_virtual_pos["ticker"],
            "ws_symbol": active_virtual_pos["ws_symbol"],
            "quantity": pos_qty,
            "entry_price": pos_entry,
            "current_price": pos_mark,
            "stop_p10": ratchet_info["stop_price"],
            "target_p90": float(active_virtual_pos["target_p90"]),
            "unrealized_pnl_cad": round(unrealized_cad, 2),
            "unrealized_pnl_pct": round(unrealized_pct, 2),
            "stop_status_desc": ratchet_info["desc"],
        }

    return {
        "symbol": primary_asset.symbol,
        "profile": prof.to_dict(),
        "asset": primary_asset.to_dict(),
        "exec_asset": exec_asset.to_dict(),
        "is_proxied": is_proxied,
        "market_clock": market_clock,
        "quote": {
            "price": curr_price,
            "prev_price": prev_price,
            "change_pct": round(change_pct, 2),
            "time": chart_times[-1],
        },
        "bars": bars_list,
        "indicators": indicators,
        "consensus": {
            "direction": direction,
            "exp_return_pct": exp_ret_pct,
            "p10": cons_p10,
            "p50": cons_p50,
            "p90": cons_p90,
            "conformal_pad_pct": zoo_res.get("conformal_pad_pct", 1.8),
        },
        "forecast_cone": {
            "path": forecast_path,
        },
        "model_results": zoo_res["all_model_results"],
        "feature_importance": zoo_res.get("feature_importance", []),
        "order_ticket": {
            "exec_symbol": exec_asset.ws_display_symbol,
            "exec_price": round(exec_price, 4),
            "limit_entry": round(exec_price, 4),
            "take_profit_p90": round(exec_p90, 4),
            "stop_loss_p10": round(exec_p10, 4),
            "recommended_shares": round(ticket_shares, 4),
            "quarter_kelly_cad": round(sizer_res.recommended_cad, 2),
            "dollar_risk_at_stop": round(sizer_res.dollar_risk_at_stop, 2),
            "fee_estimate": exec_fee.to_dict(),
            "proxy_savings_badge": f"Saves ~${proxy_savings:,.2f} CAD round-trip vs direct trading" if proxy_savings > 0.05 else None,
        },
        "virtual_bot": {
            "is_armed": bool(acct.get("is_armed", 1)),
            "virtual_cash_cad": round(virtual_cash, 2),
            "total_equity_cad": round(total_equity, 2),
            "active_position": active_pos_dict,
            "efficiency_summary": efficiency_summary.to_dict(),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. Alpha Scanner & News Wire Endpoints
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/scanner")
async def get_scanner_opportunities(
    force: bool = False,
    prof: UserProfile = Depends(get_current_profile),
):
    opps = AlphaScanner.scan_opportunities(
        force_refresh=force,
        ws_tier=prof.ws_tier,
        usd_account_active=prof.usd_account_active,
    )

    intel = MarketIntelligenceEngine()
    intel_rep = intel.get_asset_intelligence(prof.active_symbol)
    articles = [
        {"title": a.title, "source": getattr(a, "publisher", "Wire"), "url": a.url, "catalyst": a.catalyst_tag}
        for a in (intel_rep.articles or [])
    ]

    return {
        "opportunities": [o.to_dict() for o in opps],
        "news_articles": articles,
        "news_sentiment": intel_rep.sentiment_score,
        "dominant_catalyst": intel_rep.dominant_catalyst,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8. Virtual Bot & Real Fills Actions
# ─────────────────────────────────────────────────────────────────────────────
class BotArmRequest(BaseModel):
    is_armed: bool


@app.post("/api/bot/arm")
async def toggle_bot_arm(
    payload: BotArmRequest,
    prof: UserProfile = Depends(get_current_profile),
):
    db_manager.update_virtual_account(prof.profile_id, is_armed=1 if payload.is_armed else 0)
    return {"status": "SUCCESS", "is_armed": payload.is_armed}


class PaperBuyRequest(BaseModel):
    symbol: str
    capital_cad: float = 500.0
    use_proxy: bool = True


@app.post("/api/bot/paper-buy")
async def execute_paper_buy_endpoint(
    payload: PaperBuyRequest,
    prof: UserProfile = Depends(get_current_profile),
):
    res = virtual_bot_engine.execute_paper_buy(
        user_id=prof.profile_id,
        symbol=payload.symbol,
        capital_cad=payload.capital_cad,
        use_proxy=payload.use_proxy,
    )
    return res


class PaperCloseRequest(BaseModel):
    ticker: str


@app.post("/api/bot/paper-close")
async def execute_paper_close_endpoint(
    payload: PaperCloseRequest,
    prof: UserProfile = Depends(get_current_profile),
):
    res = virtual_bot_engine.execute_paper_close(
        user_id=prof.profile_id,
        ticker=payload.ticker,
        exit_reason="MANUAL_CLOSE",
    )
    return res


class ResetWalletRequest(BaseModel):
    starting_cad: float = 10000.0


@app.post("/api/bot/reset-wallet")
async def reset_wallet(
    payload: ResetWalletRequest,
    prof: UserProfile = Depends(get_current_profile),
):
    db_manager.reset_virtual_portfolio(prof.profile_id, starting_cash=payload.starting_cad)
    return {"status": "SUCCESS", "virtual_cash_cad": payload.starting_cad}


# Real Wealthsimple mobile fill logger
class RealTradeOpenRequest(BaseModel):
    ticker: str
    entry_price: float
    quantity: float
    target_price: float
    stop_price: float


@app.post("/api/real-trades/open")
async def log_real_trade_open(
    payload: RealTradeOpenRequest,
    prof: UserProfile = Depends(get_current_profile),
):
    tx_id = f"WS_{prof.profile_id[:2].upper()}_{int(asyncio.get_event_loop().time() * 1000) % 1000000}"
    capital = payload.entry_price * payload.quantity
    db_manager.save_real_ws_position({
        "tx_id": tx_id,
        "profile_id": prof.profile_id,
        "ticker": payload.ticker,
        "ws_symbol": payload.ticker.split(".")[0],
        "route_type": "DIRECT",
        "status": "OPEN",
        "entry_price": payload.entry_price,
        "current_price": payload.entry_price,
        "quantity": payload.quantity,
        "capital_cad": capital,
        "target_price": payload.target_price,
        "stop_price": payload.stop_price,
    })
    return {"status": "SUCCESS", "tx_id": tx_id}


class RealTradeCloseRequest(BaseModel):
    tx_id: str
    exit_price: float


@app.post("/api/real-trades/close")
async def log_real_trade_close(payload: RealTradeCloseRequest):
    db_manager.close_real_ws_position(
        tx_id=payload.tx_id,
        exit_price=payload.exit_price,
        exit_reason="MANUAL_CLOSE",
    )
    return {"status": "SUCCESS"}


# ─────────────────────────────────────────────────────────────────────────────
# 9. Portfolios & Multi-Account Leaderboard Endpoint
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/portfolios")
async def get_portfolios(prof: UserProfile = Depends(get_current_profile)):
    all_profiles = profile_manager.list_profiles()
    leaderboard = []

    for p in all_profiles:
        acct = db_manager.get_or_create_virtual_account(p.profile_id)
        open_pos = db_manager.get_virtual_positions(p.profile_id)
        closed_trades = db_manager.get_virtual_trades(p.profile_id, limit=500)
        real_trades = db_manager.get_real_ws_positions(p.profile_id, status="OPEN") + db_manager.get_real_ws_positions(p.profile_id, status="CLOSED")

        starting = float(acct.get("starting_virtual_cad", 10000.0))
        cash = float(acct.get("virtual_cash_cad", 10000.0))
        pos_val = sum(float(pos["quantity"]) * float(pos["current_price"]) for pos in open_pos)
        equity = cash + pos_val
        net_ret = ((equity - starting) / max(starting, 1.0)) * 100.0

        leaderboard.append({
            "profile_id": p.profile_id,
            "display_name": p.display_name,
            "ws_tier": p.ws_tier,
            "starting_cad": starting,
            "current_equity_cad": round(equity, 2),
            "net_return_pct": round(net_ret, 2),
            "open_virtual_count": len(open_pos),
            "completed_trades_count": len(closed_trades),
            "real_ws_trades_count": len(real_trades),
        })

    leaderboard.sort(key=lambda x: x["net_return_pct"], reverse=True)

    # Active user's positions & history
    user_open_pos = db_manager.get_virtual_positions(prof.profile_id)
    open_pos_views = []
    for pos in user_open_pos:
        entry = float(pos["entry_price"])
        curr = float(pos["current_price"])
        qty = float(pos["quantity"])
        pnl = (curr - entry) * qty
        pct = ((curr - entry) / max(entry, 1e-4)) * 100.0
        open_pos_views.append({
            "ticker": pos["ticker"],
            "ws_symbol": pos["ws_symbol"],
            "quantity": qty,
            "entry_price": entry,
            "current_price": curr,
            "stop_p10": float(pos["stop_p10"]),
            "target_p90": float(pos["target_p90"]),
            "unrealized_pnl_cad": round(pnl, 2),
            "unrealized_pnl_pct": round(pct, 2),
        })

    completed_trades = db_manager.get_virtual_trades(prof.profile_id, limit=100)
    real_positions = db_manager.get_real_ws_history(prof.profile_id, limit=50)

    # Telemetry
    telemetry = health_check.run_all()

    return {
        "leaderboard": leaderboard,
        "open_virtual_positions": open_pos_views,
        "completed_virtual_trades": completed_trades,
        "real_ws_positions": real_positions,
        "telemetry": telemetry,
    }
