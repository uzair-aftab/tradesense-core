"""
Unit tests for Wealthsimple Mobile Copier Lifecycle Engine
"""

import tempfile
from pathlib import Path
import pytest

from src.data.database import DatabaseManager
from src.data.auth import AuthManager
from src.models.wealthsimple_copier import WealthsimpleCopier
from src.models.strategies import SignalType, StrategySignal


@pytest.fixture
def temp_copier():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    db = DatabaseManager(db_path=db_path)
    copier = WealthsimpleCopier(db=db)
    auth = AuthManager(db=db)
    _, _, user_id = auth.register_user("ws_trader", "Pass123456")
    yield copier, user_id, db
    if db_path.exists():
        try:
            db_path.unlink()
        except Exception:
            pass


def test_canadian_proxy_lookup(temp_copier):
    copier, _, _ = temp_copier
    proxy_btc = copier.get_proxy_info("BTC-CAD")
    assert proxy_btc["ws_symbol"] == "BTCC.TO"
    assert "Purpose Bitcoin" in proxy_btc["name"]

    proxy_eth = copier.get_proxy_info("ETH-CAD")
    assert proxy_eth["ws_symbol"] == "ETHX.TO"

    proxy_shop = copier.get_proxy_info("SHOP.TO")
    assert proxy_shop["ws_symbol"] == "SHOP.TO"


def test_signal_gating_and_alert_evaluation(temp_copier):
    copier, _, _ = temp_copier
    # 1. HOLD signal -> Rejected
    hold_sig = StrategySignal("Ensemble", "BTC-CAD", SignalType.HOLD, 0.0)
    assert copier.evaluate_signal_for_alert(hold_sig, 120000.0, 10000.0) is None

    # 2. Low confidence -> Rejected
    weak_sig = StrategySignal("Ensemble", "BTC-CAD", SignalType.BUY, 0.50)
    assert copier.evaluate_signal_for_alert(weak_sig, 120000.0, 10000.0) is None

    # 3. High confidence BUY -> Alert generated
    strong_sig = StrategySignal(
        "Ensemble",
        "BTC-CAD",
        SignalType.BUY,
        0.85,
        target_price=124000.0,
        stop_price=117000.0,
        reason="Whale surge + bull regime",
    )
    alert = copier.evaluate_signal_for_alert(strong_sig, 120000.0, 10000.0)
    assert alert is not None
    assert alert["ws_symbol"] == "BTCC.TO"
    assert alert["side"] == "BUY"
    assert alert["confidence"] == 85.0
    assert alert["upside_pct"] > 1.5
    assert alert["suggested_capital"] > 0


def test_trade_lifecycle_with_linked_closing_id(temp_copier):
    copier, user_id, _ = temp_copier

    strong_sig = StrategySignal(
        "Whale Flow",
        "BTC-CAD",
        SignalType.BUY,
        0.90,
        target_price=125000.0,
        stop_price=116000.0,
        reason="Whale block accumulation",
    )
    alert = copier.evaluate_signal_for_alert(strong_sig, 120000.0, 10000.0)
    assert alert is not None

    # Step 1: User confirms execution on Wealthsimple
    tx_id = copier.confirm_entry(
        user_id=user_id,
        alert=alert,
        actual_price=120100.0,
        actual_capital=600.0,
    )
    assert tx_id.startswith("WS-")
    assert tx_id.endswith("-001")

    # Verify trade is OPEN
    active = copier.db.get_active_ws_trades(user_id)
    assert len(active) == 1
    assert active[0]["tx_id"] == tx_id

    # Step 2: Price has not reached target yet (e.g. 122,000)
    no_exits = copier.check_open_trades_for_exit(user_id, current_price=122000.0, current_regime="BULL_TREND")
    assert len(no_exits) == 0

    # Step 3: Price hits Take-Profit ($125,100 >= $125,000)
    exit_alerts = copier.check_open_trades_for_exit(user_id, current_price=125100.0, current_regime="BULL_TREND")
    assert len(exit_alerts) == 1
    exit_alert = exit_alerts[0]
    assert exit_alert["opening_tx_id"] == tx_id
    assert exit_alert["pnl_cad"] > 0
    assert "Take-Profit" in exit_alert["exit_reason"]

    # Step 4: User confirms exit on Wealthsimple
    close_summary = copier.confirm_exit(
        tx_id=tx_id,
        actual_exit_price=125150.0,
        exit_reason="Take Profit Target Reached",
    )
    assert close_summary["tx_id"] == tx_id
    assert close_summary["pnl_cad"] > 0
    assert close_summary["pnl_pct"] > 4.0

    # Verify trade moved to CLOSED
    assert len(copier.db.get_active_ws_trades(user_id)) == 0
    closed = copier.db.get_closed_ws_trades(user_id)
    assert len(closed) == 1

    # Check analytics
    analytics = copier.get_user_analytics(user_id)
    assert analytics["closed_trades"] == 1
    assert analytics["win_rate_pct"] == 100.0
    assert analytics["total_realized_pnl_cad"] > 0
