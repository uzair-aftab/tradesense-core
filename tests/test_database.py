"""
Tests for TradeSense SQLite Database Engine & Auth Security
"""

import tempfile
from pathlib import Path
import pytest

from src.data.database import DatabaseManager
from src.data.auth import AuthManager


@pytest.fixture
def temp_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    db = DatabaseManager(db_path=db_path)
    yield db
    if db_path.exists():
        try:
            db_path.unlink()
        except Exception:
            pass


def test_user_registration_and_authentication(temp_db):
    auth = AuthManager(db=temp_db)
    # 1. Register
    success, msg, user_id = auth.register_user("trader_northyork", "SecretPass123", starting_capital=15000.0)
    assert success is True
    assert user_id is not None

    # Duplicate registration fails
    dup_success, dup_msg, _ = auth.register_user("trader_northyork", "AnotherPass")
    assert dup_success is False
    assert "already taken" in dup_msg

    # 2. Authentication with valid password
    user = auth.authenticate_user("trader_northyork", "SecretPass123")
    assert user is not None
    assert user["username"] == "trader_northyork"
    assert user["cash_balance"] == 15000.0
    assert "session_token" in user
    token = user["session_token"]

    # 3. Authentication with invalid password
    bad_user = auth.authenticate_user("trader_northyork", "WrongPass")
    assert bad_user is None

    # 4. Session validation (F5 refresh immunity)
    session_user = auth.get_user_by_session(token)
    assert session_user is not None
    assert session_user["id"] == user["id"]

    # 5. Logout
    auth.logout_user(token)
    assert auth.get_user_by_session(token) is None


def test_portfolio_and_position_persistence(temp_db):
    auth = AuthManager(db=temp_db)
    _, _, user_id = auth.register_user("alice", "PassAlice123")

    # Save active position
    temp_db.save_user_position(
        user_id=user_id,
        ticker="BTC-CAD",
        quantity=0.5,
        avg_cost=118000.0,
        current_price=120000.0,
        stop_loss=115000.0,
        take_profit=125000.0,
    )

    positions = temp_db.get_user_positions(user_id)
    assert "BTC-CAD" in positions
    assert positions["BTC-CAD"]["quantity"] == 0.5
    assert positions["BTC-CAD"]["avg_cost"] == 118000.0

    # Update balance
    temp_db.update_user_balance(user_id, 8500.0)
    user = temp_db.get_user_by_id(user_id)
    assert user["cash_balance"] == 8500.0

    # Record trade
    trade_id = temp_db.record_user_trade(
        user_id=user_id,
        order_id="ORD-101",
        ticker="BTC-CAD",
        side="BUY",
        quantity=0.5,
        fill_price=118000.0,
        total_cost=59000.0,
        pnl=0.0,
    )
    assert trade_id is not None
    trades = temp_db.get_user_trades(user_id)
    assert len(trades) == 1
    assert trades[0]["order_id"] == "ORD-101"

    # Delete position
    temp_db.delete_user_position(user_id, "BTC-CAD")
    assert len(temp_db.get_user_positions(user_id)) == 0


def test_wealthsimple_trade_lifecycle(temp_db):
    auth = AuthManager(db=temp_db)
    _, _, user_id = auth.register_user("bob", "PassBob123")

    # Initial trade count
    count = temp_db.get_today_ws_trade_count(user_id)
    assert count == 0

    # Create WS Trade
    tx_id = f"WS-20260924-{count + 1:03d}"
    temp_db.create_ws_trade(
        tx_id=tx_id,
        user_id=user_id,
        ticker="BTC-CAD",
        ws_symbol="BTCC.TO",
        side="BUY",
        status="OPEN",
        suggested_entry=119000.0,
        actual_entry=119100.0,
        quantity=38.0,
        capital_cad=500.0,
        target_price=124000.0,
        stop_price=116000.0,
    )

    assert temp_db.get_today_ws_trade_count(user_id) == 1
    active_trades = temp_db.get_active_ws_trades(user_id)
    assert len(active_trades) == 1
    assert active_trades[0]["tx_id"] == tx_id
    assert active_trades[0]["status"] == "OPEN"

    # Close WS Trade
    temp_db.close_ws_trade(
        tx_id=tx_id,
        actual_exit_price=124200.0,
        realized_pnl_cad=21.40,
        realized_pnl_pct=4.28,
        closing_reason="Take Profit Reached",
    )

    assert len(temp_db.get_active_ws_trades(user_id)) == 0
    closed_trades = temp_db.get_closed_ws_trades(user_id)
    assert len(closed_trades) == 1
    assert closed_trades[0]["realized_pnl_cad"] == 21.40
    assert closed_trades[0]["closing_reason"] == "Take Profit Reached"


def test_gauntlet_with_user_db_integration(temp_db, monkeypatch):
    import src.data.database as db_mod
    monkeypatch.setattr(db_mod, "_GLOBAL_DB", temp_db)

    from src.models.trader import Gauntlet, TradeOrder, Side

    auth = AuthManager(db=temp_db)
    _, _, user_id = auth.register_user("charlie", "PassCharlie123", starting_capital=20000.0)

    # Initialize Gauntlet with user_id
    sim = Gauntlet(user_id=user_id, persist=True)
    assert sim.cash == 20000.0

    # Execute BUY
    fill = sim.submit_order(TradeOrder(ticker="ETH-CAD", side=Side.BUY, quantity=2.0, price=4000.0))
    assert "ETH-CAD" in sim.positions
    assert sim.cash < 20000.0

    # Verify positions and trade fills were saved to DB
    db_positions = temp_db.get_user_positions(user_id)
    assert "ETH-CAD" in db_positions
    assert db_positions["ETH-CAD"]["quantity"] == 2.0

    db_trades = temp_db.get_user_trades(user_id)
    assert len(db_trades) == 1
    assert db_trades[0]["ticker"] == "ETH-CAD"

    # Now simulate webpage refresh: instantiate a brand new Gauntlet instance with user_id
    sim_refreshed = Gauntlet(user_id=user_id, persist=True)
    assert sim_refreshed.cash == pytest.approx(sim.cash, abs=0.01)
    assert "ETH-CAD" in sim_refreshed.positions
    assert sim_refreshed.positions["ETH-CAD"].quantity == 2.0
    assert len(sim_refreshed.fills) == 1

