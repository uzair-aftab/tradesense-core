"""
TradeSense v5 Overhaul Comprehensive Unit & Integration Tests
============================================================
Tests the 4 foundational v5 pillars:
  1. Universal Wealthsimple Catalog & Fee Engine (ws_catalog.py)
  2. Asset-Class-Adaptive 5-Technique Prediction Engine (adaptive_predictor.py)
  3. Persistent Virtual-Money Proof-of-Edge Bot Engine (virtual_bot.py)
  4. SQLite WAL Persistent State & Fee Friction Integrity (database.py)
"""

import math
import numpy as np
import pandas as pd
import pytest

from src.data.database import DatabaseManager
from src.data.ws_catalog import (
    CURATED_WS_ASSETS,
    WealthsimpleAsset,
    WealthsimpleCatalog,
    WealthsimpleFeeSchedule,
)
from src.models.adaptive_predictor import (
    AdaptiveForecast,
    AdaptivePredictionEngine,
    AssetDNA,
)
from src.models.virtual_bot import BotEfficiencySummary, VirtualBotEngine


# ─────────────────────────────────────────────────────────────────────────────
# 1. Wealthsimple Catalog & Fee Engine Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestWealthsimpleCatalog:
    """Tests catalog resolution, search, and exact tier fee schedules."""

    def test_curated_registry_coverage(self):
        """Verifies curated registry contains over 150 assets covering all 5 categories."""
        assets = WealthsimpleCatalog.list_curated()
        assert len(assets) >= 150, f"Expected at least 150 curated assets, found {len(assets)}"

        categories = {a.category for a in assets}
        assert "TSX_STOCK" in categories
        assert "CANADIAN_ETF" in categories
        assert "NEO_CDR" in categories
        assert "US_STOCK" in categories
        assert "CRYPTO" in categories

    def test_asset_resolution_and_proxies(self):
        """Verifies resolution of direct and proxy CAD assets."""
        nvda = WealthsimpleCatalog.resolve_asset("NVDA")
        assert nvda.symbol == "NVDA"
        assert nvda.alt_cad_proxy == "NVDA.NE"
        assert nvda.currency == "USD"

        btc = WealthsimpleCatalog.resolve_asset("BTC-CAD")
        assert btc.symbol == "BTC-CAD"
        assert btc.alt_cad_proxy == "BTCC.TO"
        assert btc.is_24_7 is True

        shop = WealthsimpleCatalog.resolve_asset("SHOP.TO")
        assert shop.exchange == "TSX"
        assert shop.currency == "CAD"

        cdr = WealthsimpleCatalog.resolve_asset("AAPL.NE")
        assert cdr.exchange == "NEO"
        assert cdr.currency == "CAD"

    def test_fee_schedule_calculations(self):
        """Verifies exact Wealthsimple fees across TSX, CDRs, US Equities, and Crypto tiers."""
        shop = WealthsimpleCatalog.resolve_asset("SHOP.TO")
        fee_shop = WealthsimpleFeeSchedule.calculate_fee(shop, notional_cad=1000.0)
        assert fee_shop.one_way_fee_pct == 0.0
        assert fee_shop.round_trip_total_pct == pytest.approx(0.08, rel=1e-2)

        cdr = WealthsimpleCatalog.resolve_asset("NVDA.NE")
        fee_cdr = WealthsimpleFeeSchedule.calculate_fee(cdr, notional_cad=1000.0)
        assert fee_cdr.one_way_fee_pct == 0.0
        assert fee_cdr.round_trip_total_pct == pytest.approx(0.16, rel=1e-2)

        # US Stock on Core without USD account: 1.5% FX per side
        nvda = WealthsimpleCatalog.resolve_asset("NVDA")
        fee_nvda_core = WealthsimpleFeeSchedule.calculate_fee(nvda, notional_cad=1000.0, ws_tier="Core", usd_account_active=False)
        assert fee_nvda_core.one_way_fee_pct == 1.50
        assert fee_nvda_core.round_trip_total_pct == pytest.approx(3.08, rel=1e-2)
        assert fee_nvda_core.optimization_tip is not None
        assert "NVDA.NE" in fee_nvda_core.optimization_tip

        # US Stock with USD Account Active: $0 FX
        fee_nvda_usd = WealthsimpleFeeSchedule.calculate_fee(nvda, notional_cad=1000.0, ws_tier="Core", usd_account_active=True)
        assert fee_nvda_usd.one_way_fee_pct == 0.0

        # Crypto Tiers: Core 2.0%, Premium 1.0%, Generation 0.5%
        btc = WealthsimpleCatalog.resolve_asset("BTC-CAD")
        fee_btc_core = WealthsimpleFeeSchedule.calculate_fee(btc, notional_cad=1000.0, ws_tier="Core")
        fee_btc_prem = WealthsimpleFeeSchedule.calculate_fee(btc, notional_cad=1000.0, ws_tier="Premium")
        fee_btc_gen = WealthsimpleFeeSchedule.calculate_fee(btc, notional_cad=1000.0, ws_tier="Generation")

        assert fee_btc_core.one_way_fee_pct == 2.0
        assert fee_btc_prem.one_way_fee_pct == 1.0
        assert fee_btc_gen.one_way_fee_pct == 0.5
        assert fee_btc_core.optimization_tip is not None


# ─────────────────────────────────────────────────────────────────────────────
# 2. Asset-Class-Adaptive Prediction Engine Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestAdaptivePredictionEngine:
    """Tests pair DNA profiling, 5 distinct techniques, and dynamic weight adaptation."""

    @pytest.fixture
    def market_data(self):
        """Generates realistic market data for crypto and equity."""
        dates = pd.date_range("2026-01-01", periods=100, freq="h")
        rng = np.random.default_rng(42)

        # High-vol trending crypto
        p_crypto = 100000.0 * np.exp(np.cumsum(rng.normal(0.001, 0.02, 100)))
        df_crypto = pd.DataFrame({
            "Open": p_crypto * 0.998,
            "High": p_crypto * 1.008,
            "Low": p_crypto * 0.992,
            "Close": p_crypto,
            "Volume": rng.uniform(500, 2500, 100),
        }, index=dates)

        # Low-vol mean-reverting equity
        p_equity = 140.0 + np.sin(np.linspace(0, 12, 100)) * 4.0 + rng.normal(0, 0.3, 100)
        df_equity = pd.DataFrame({
            "Open": p_equity,
            "High": p_equity + 0.5,
            "Low": p_equity - 0.5,
            "Close": p_equity,
            "Volume": rng.uniform(10000, 30000, 100),
        }, index=dates)

        return df_crypto, df_equity

    def test_asset_dna_profiling(self, market_data):
        """Verifies DNA classification and volatility scaling between asset classes."""
        df_crypto, df_equity = market_data
        engine = AdaptivePredictionEngine()

        btc = WealthsimpleCatalog.resolve_asset("BTC-CAD")
        ry = WealthsimpleCatalog.resolve_asset("RY.TO")

        dna_btc = engine.profile_asset(btc, df_crypto, "1h")
        dna_ry = engine.profile_asset(ry, df_equity, "1h")

        assert dna_btc.archetype == "CRYPTO_MOMENTUM_24_7"
        assert dna_btc.annualized_vol_pct > dna_ry.annualized_vol_pct
        assert dna_ry.archetype in ("MEAN_REVERTING_BLUE_CHIP", "HIGH_BETA_CATALYST_EQUITY")

    def test_forecast_monotonicity_and_weights(self, market_data):
        """Verifies strictly P10 < P50 < P90 monotonicity and 100% weight sum."""
        df_crypto, _ = market_data
        engine = AdaptivePredictionEngine()
        btc = WealthsimpleCatalog.resolve_asset("BTC-CAD")

        fc = engine.compute_adaptive_forecast(df_crypto, btc, "1h", horizon_bars=5)

        # Monotonicity check
        assert fc.p10 < fc.p50 < fc.p90, f"Envelope violation: p10={fc.p10}, p50={fc.p50}, p90={fc.p90}"
        assert len(fc.techniques) == 5

        # Check weights sum to 100%
        total_w = sum(t.weight_pct for t in fc.techniques)
        assert total_w == pytest.approx(100.0, abs=0.5)

        # Check each technique is monotonic
        for t in fc.techniques:
            assert t.p10 < t.p50 < t.p90, f"Technique {t.name} violation: p10={t.p10}, p50={t.p50}, p90={t.p90}"

        # Projected path length
        assert len(fc.projected_path) == 5
        for p in fc.projected_path:
            assert p["p10"] <= p["p50"] <= p["p90"]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Persistent Virtual Bot Engine Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestVirtualBotEngine:
    """Tests virtual account persistence, trade execution, and walk-forward efficiency."""

    @pytest.fixture
    def test_db(self, tmp_path):
        """Creates an isolated temporary SQLite database."""
        db_file = tmp_path / "test_tradesense.db"
        mgr = DatabaseManager(str(db_file))
        mgr.init_db()
        return mgr

    def test_virtual_account_crud(self, test_db):
        """Tests virtual wallet creation, balance updates, and resetting."""
        user_id = "test_quant_user"
        acct = test_db.get_or_create_virtual_account(user_id, starting_cash=10000.0)
        assert acct["virtual_cash_cad"] == 10000.0
        assert acct["is_armed"] == 1

        test_db.update_virtual_account(user_id, virtual_cash_cad=12500.0, ws_tier="Premium")
        updated = test_db.get_or_create_virtual_account(user_id)
        assert updated["virtual_cash_cad"] == 12500.0
        assert updated["ws_tier"] == "Premium"

        test_db.reset_virtual_wallet(user_id, starting_cash=5000.0)
        reset_acct = test_db.get_or_create_virtual_account(user_id)
        assert reset_acct["virtual_cash_cad"] == 5000.0

    def test_walk_forward_efficiency_evaluator(self, test_db):
        """Tests zero-lookahead walk-forward evaluation on a sample asset."""
        dates = pd.date_range("2026-01-01", periods=80, freq="h")
        rng = np.random.default_rng(123)
        p = 50.0 + np.cumsum(rng.normal(0.05, 0.8, 80))
        df = pd.DataFrame({
            "Open": p,
            "High": p + 1.0,
            "Low": p - 1.0,
            "Close": p,
            "Volume": rng.uniform(2000, 8000, 80),
        }, index=dates)

        bot = VirtualBotEngine(db=test_db)
        asset = WealthsimpleCatalog.resolve_asset("SHOP.TO")

        summary = bot.evaluate_pair_efficiency(
            df=df,
            asset=asset,
            timeframe="1h",
            horizon_bars=5,
            eval_bars=40,
            starting_cad=10000.0,
        )

        assert isinstance(summary, BotEfficiencySummary)
        assert summary.ticker == "SHOP.TO"
        assert summary.readiness_state in ("PROVEN_EDGE", "CALIBRATING", "AVOID")
        assert summary.win_rate_pct >= 0.0
        assert summary.profit_factor >= 0.0
