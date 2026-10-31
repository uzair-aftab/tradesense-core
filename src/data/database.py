"""
TradeSense Persistent Database Engine
======================================
High-concurrency SQLite storage using Write-Ahead Logging (WAL) mode.
Guarantees permanent persistence of:
  1. Users & Authentication credentials
  2. Persistent Web Sessions (F5 refresh immunity)
  3. User Paper Portfolios, Cash, & Active Positions
  4. Real-World Wealthsimple Copier Trades (with linked transaction IDs)
  5. User Settings & Risk Parameters
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

from src.utils.logger import get_logger

logger = get_logger("hfp.database")

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "tradesense.db"


class DatabaseManager:
    """Centralized, thread-safe database interface."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager yielding a thread-safe connection with Row factory."""
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=10.0,
        )
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("PRAGMA busy_timeout = 5000;")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_db(self) -> None:
        """Create all tables and indices if they do not exist."""
        with self.get_connection() as conn:
            conn.executescript(
                """
                -- Users
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL COLLATE NOCASE,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    cash_balance REAL DEFAULT 10000.0,
                    starting_capital REAL DEFAULT 10000.0,
                    risk_per_trade REAL DEFAULT 0.02,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP
                );

                -- Persistent Web Sessions (F5 Refresh Immunity)
                CREATE TABLE IF NOT EXISTS user_sessions (
                    session_token TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                -- Paper Trading Active Positions
                CREATE TABLE IF NOT EXISTS user_positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    ticker TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    avg_cost REAL NOT NULL,
                    current_price REAL NOT NULL,
                    stop_loss REAL,
                    take_profit REAL,
                    trailing_stop_pct REAL,
                    trailing_stop_peak REAL,
                    opened_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
                    UNIQUE(user_id, ticker)
                );

                -- Paper Trading Fills Ledger
                CREATE TABLE IF NOT EXISTS user_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    order_id TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    fill_price REAL NOT NULL,
                    total_cost REAL NOT NULL,
                    pnl REAL DEFAULT 0.0,
                    exit_reason TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                -- Wealthsimple Mobile Copier Lifecycle Table
                CREATE TABLE IF NOT EXISTS wealthsimple_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tx_id TEXT UNIQUE NOT NULL,
                    user_id INTEGER NOT NULL,
                    ticker TEXT NOT NULL,
                    ws_symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    status TEXT NOT NULL, -- 'OPEN', 'CLOSED', 'DISMISSED'
                    suggested_entry REAL NOT NULL,
                    actual_entry REAL,
                    quantity REAL NOT NULL,
                    capital_cad REAL NOT NULL,
                    target_price REAL,
                    stop_price REAL,
                    suggested_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    executed_time TIMESTAMP,
                    exit_price REAL,
                    actual_exit REAL,
                    exit_time TIMESTAMP,
                    realized_pnl_cad REAL DEFAULT 0.0,
                    realized_pnl_pct REAL DEFAULT 0.0,
                    closing_reason TEXT,
                    notes TEXT,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                -- User Application Settings
                CREATE TABLE IF NOT EXISTS user_settings (
                    user_id INTEGER PRIMARY KEY,
                    active_ticker TEXT DEFAULT 'BTC-CAD',
                    active_strategy TEXT DEFAULT 'Ensemble Multi-Strategy Alpha',
                    timeframe TEXT DEFAULT '1d',
                    lookback TEXT DEFAULT '6mo',
                    sound_alerts_enabled INTEGER DEFAULT 1,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                                -- TradeSense v5 Persistent Virtual Bot Tables
                CREATE TABLE IF NOT EXISTS virtual_bot_accounts (
                    user_id TEXT PRIMARY KEY,
                    starting_virtual_cad REAL DEFAULT 10000.0,
                    virtual_cash_cad REAL DEFAULT 10000.0,
                    is_armed INTEGER DEFAULT 1,
                    ws_tier TEXT DEFAULT 'Core',
                    usd_account_active INTEGER DEFAULT 0,
                    risk_per_trade REAL DEFAULT 0.02,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS virtual_bot_positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    ws_symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    current_price REAL NOT NULL,
                    predicted_p50 REAL NOT NULL,
                    target_p90 REAL NOT NULL,
                    stop_p10 REAL NOT NULL,
                    bars_held INTEGER DEFAULT 0,
                    horizon_bars INTEGER DEFAULT 5,
                    entry_fee_cad REAL DEFAULT 0.0,
                    dna_profile TEXT,
                    opened_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, ticker)
                );

                CREATE TABLE IF NOT EXISTS virtual_bot_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    ws_symbol TEXT NOT NULL,
                    mode TEXT DEFAULT 'LIVE_VIRTUAL',
                    entry_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    predicted_p50 REAL NOT NULL,
                    quantity REAL NOT NULL,
                    gross_pnl_cad REAL NOT NULL,
                    ws_fees_cad REAL NOT NULL,
                    net_pnl_cad REAL NOT NULL,
                    net_pnl_pct REAL NOT NULL,
                    prediction_hit INTEGER DEFAULT 0,
                    exit_reason TEXT,
                    opened_at TEXT,
                    closed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_vbot_pos_user ON virtual_bot_positions(user_id);
                CREATE INDEX IF NOT EXISTS idx_vbot_trades_user ON virtual_bot_trades(user_id, closed_at);

                -- TradeSense v6 Passwordless Profiles & Real Wealthsimple Positions
                CREATE TABLE IF NOT EXISTS user_profiles (
                    profile_id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    avatar_initials TEXT NOT NULL,
                    accent_color TEXT DEFAULT '#38bdf8',
                    ws_tier TEXT DEFAULT 'Core',
                    usd_account_active INTEGER DEFAULT 0,
                    active_symbol TEXT DEFAULT 'BTC-CAD',
                    active_timeframe TEXT DEFAULT '1h',
                    horizon_bars INTEGER DEFAULT 5,
                    model_preset TEXT DEFAULT 'AUTO_DNA',
                    selected_models_json TEXT DEFAULT '[]',
                    blending_mode TEXT DEFAULT 'WALK_FORWARD',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_active_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS real_ws_positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tx_id TEXT UNIQUE NOT NULL,
                    profile_id TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    ws_symbol TEXT NOT NULL,
                    route_type TEXT DEFAULT 'DIRECT',
                    status TEXT DEFAULT 'OPEN',
                    entry_price REAL NOT NULL,
                    current_price REAL NOT NULL,
                    quantity REAL NOT NULL,
                    capital_cad REAL NOT NULL,
                    target_price REAL NOT NULL,
                    stop_price REAL NOT NULL,
                    trailing_stop_price REAL,
                    est_fee_cad REAL DEFAULT 0.0,
                    exit_price REAL,
                    realized_pnl_cad REAL DEFAULT 0.0,
                    realized_pnl_pct REAL DEFAULT 0.0,
                    exit_reason TEXT,
                    opened_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    closed_at TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_user_profiles_active ON user_profiles(last_active_at);
                CREATE INDEX IF NOT EXISTS idx_real_ws_profile_status ON real_ws_positions(profile_id, status);

                -- Indices
                CREATE INDEX IF NOT EXISTS idx_sessions_token ON user_sessions(session_token);
                CREATE INDEX IF NOT EXISTS idx_ws_trades_user_status ON wealthsimple_trades(user_id, status);
                CREATE INDEX IF NOT EXISTS idx_user_positions_lookup ON user_positions(user_id, ticker);
                """
            )
        logger.info("Database initialized successfully at %s", self.db_path)

    # ══════════════════════════════════════════════════════════════════════
    #  User Management
    # ══════════════════════════════════════════════════════════════════════

    def create_user(
        self,
        username: str,
        password_hash: str,
        salt: str,
        starting_capital: float = 10000.0,
    ) -> int:
        """Create new user with isolated default portfolio and settings."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO users (username, password_hash, salt, cash_balance, starting_capital)
                VALUES (?, ?, ?, ?, ?)
                """,
                (username.strip(), password_hash, salt, starting_capital, starting_capital),
            )
            user_id = cursor.lastrowid
            # Default settings
            conn.execute(
                """
                INSERT INTO user_settings (user_id, active_ticker, active_strategy)
                VALUES (?, 'BTC-CAD', 'Ensemble Multi-Strategy Alpha')
                """,
                (user_id,),
            )
            return user_id

    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Lookup user by username."""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE username = ? COLLATE NOCASE",
                (username.strip(),),
            ).fetchone()
            return dict(row) if row else None

    def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Lookup user by ID."""
        with self.get_connection() as conn:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            return dict(row) if row else None

    def update_user_last_login(self, user_id: int) -> None:
        """Update last login timestamp."""
        with self.get_connection() as conn:
            conn.execute(
                "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?",
                (user_id,),
            )

    def update_user_balance(self, user_id: int, cash_balance: float) -> None:
        """Update persistent cash balance for user."""
        with self.get_connection() as conn:
            conn.execute(
                "UPDATE users SET cash_balance = ? WHERE id = ?",
                (round(cash_balance, 2), user_id),
            )

    def reset_user_account(self, user_id: int) -> None:
        """Reset user account cash to starting capital and wipe positions/trades."""
        with self.get_connection() as conn:
            conn.execute(
                """
                UPDATE users SET cash_balance = starting_capital WHERE id = ?
                """,
                (user_id,),
            )
            conn.execute("DELETE FROM user_positions WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM user_trades WHERE user_id = ?", (user_id,))

    # ══════════════════════════════════════════════════════════════════════
    #  Session Management (F5 Refresh Immunity)
    # ══════════════════════════════════════════════════════════════════════

    def create_session(self, user_id: int, session_token: str, expires_at: datetime) -> None:
        """Insert or replace active user session."""
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO user_sessions (session_token, user_id, expires_at)
                VALUES (?, ?, ?)
                """,
                (session_token, user_id, expires_at.isoformat()),
            )

    def get_session(self, session_token: str) -> Optional[Dict[str, Any]]:
        """Verify session token and return user row if valid."""
        with self.get_connection() as conn:
            row = conn.execute(
                """
                SELECT s.session_token, s.expires_at, u.*
                FROM user_sessions s
                JOIN users u ON s.user_id = u.id
                WHERE s.session_token = ?
                """,
                (session_token,),
            ).fetchone()
            if not row:
                return None
            data = dict(row)
            expires = datetime.fromisoformat(data["expires_at"])
            if expires.tzinfo is None:
                expires = expires.replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc) > expires:
                # Expired
                self.delete_session(session_token)
                return None
            return data

    def delete_session(self, session_token: str) -> None:
        """Delete user session on logout."""
        with self.get_connection() as conn:
            conn.execute("DELETE FROM user_sessions WHERE session_token = ?", (session_token,))

    # ══════════════════════════════════════════════════════════════════════
    #  Paper Positions & Trades Persistence
    # ══════════════════════════════════════════════════════════════════════

    def get_user_positions(self, user_id: int) -> Dict[str, Dict[str, Any]]:
        """Retrieve all active paper positions for user."""
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM user_positions WHERE user_id = ?",
                (user_id,),
            ).fetchall()
            return {r["ticker"]: dict(r) for r in rows}

    def save_user_position(
        self,
        user_id: int,
        ticker: str,
        quantity: float,
        avg_cost: float,
        current_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        trailing_stop_pct: Optional[float] = None,
        trailing_stop_peak: Optional[float] = None,
    ) -> None:
        """Insert or update user's active paper position."""
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO user_positions (
                    user_id, ticker, quantity, avg_cost, current_price,
                    stop_loss, take_profit, trailing_stop_pct, trailing_stop_peak, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id, ticker) DO UPDATE SET
                    quantity = excluded.quantity,
                    avg_cost = excluded.avg_cost,
                    current_price = excluded.current_price,
                    stop_loss = excluded.stop_loss,
                    take_profit = excluded.take_profit,
                    trailing_stop_pct = excluded.trailing_stop_pct,
                    trailing_stop_peak = excluded.trailing_stop_peak,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    user_id,
                    ticker,
                    quantity,
                    avg_cost,
                    current_price,
                    stop_loss,
                    take_profit,
                    trailing_stop_pct,
                    trailing_stop_peak,
                ),
            )

    def delete_user_position(self, user_id: int, ticker: str) -> None:
        """Delete closed position."""
        with self.get_connection() as conn:
            conn.execute(
                "DELETE FROM user_positions WHERE user_id = ? AND ticker = ?",
                (user_id, ticker),
            )

    def record_user_trade(
        self,
        user_id: int,
        order_id: str,
        ticker: str,
        side: str,
        quantity: float,
        fill_price: float,
        total_cost: float,
        pnl: float = 0.0,
        exit_reason: Optional[str] = None,
    ) -> int:
        """Record paper trade fill."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO user_trades (
                    user_id, order_id, ticker, side, quantity, fill_price, total_cost, pnl, exit_reason
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    order_id,
                    ticker,
                    side,
                    quantity,
                    fill_price,
                    total_cost,
                    pnl,
                    exit_reason,
                ),
            )
            return cursor.lastrowid

    def get_user_trades(self, user_id: int, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent paper trade fills."""
        with self.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM user_trades WHERE user_id = ?
                ORDER BY timestamp DESC LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    # ══════════════════════════════════════════════════════════════════════
    #  Wealthsimple Mobile Copier Lifecycle Methods
    # ══════════════════════════════════════════════════════════════════════

    def get_today_ws_trade_count(self, user_id: int) -> int:
        """Get total trades created today for sequential ID numbering."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self.get_connection() as conn:
            val = conn.execute(
                """
                SELECT COUNT(*) FROM wealthsimple_trades
                WHERE user_id = ? AND DATE(suggested_time) = DATE(?)
                """,
                (user_id, today),
            ).fetchone()[0]
            return int(val)

    def create_ws_trade(
        self,
        tx_id: str,
        user_id: int,
        ticker: str,
        ws_symbol: str,
        side: str,
        status: str,
        suggested_entry: float,
        actual_entry: float,
        quantity: float,
        capital_cad: float,
        target_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        notes: Optional[str] = None,
    ) -> int:
        """Insert new Wealthsimple trade."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO wealthsimple_trades (
                    tx_id, user_id, ticker, ws_symbol, side, status,
                    suggested_entry, actual_entry, quantity, capital_cad,
                    target_price, stop_price, executed_time, notes
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
                """,
                (
                    tx_id,
                    user_id,
                    ticker,
                    ws_symbol,
                    side,
                    status,
                    suggested_entry,
                    actual_entry,
                    quantity,
                    capital_cad,
                    target_price,
                    stop_price,
                    notes,
                ),
            )
            return cursor.lastrowid

    def get_active_ws_trades(self, user_id: int) -> List[Dict[str, Any]]:
        """Get all open Wealthsimple trades for user."""
        with self.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM wealthsimple_trades
                WHERE user_id = ? AND status = 'OPEN'
                ORDER BY executed_time DESC
                """,
                (user_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_closed_ws_trades(self, user_id: int, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all closed Wealthsimple trades for user."""
        with self.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM wealthsimple_trades
                WHERE user_id = ? AND status = 'CLOSED'
                ORDER BY exit_time DESC LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_ws_trade_by_tx_id(self, tx_id: str) -> Optional[Dict[str, Any]]:
        """Lookup Wealthsimple trade by transaction ID."""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM wealthsimple_trades WHERE tx_id = ?",
                (tx_id,),
            ).fetchone()
            return dict(row) if row else None

    def close_ws_trade(
        self,
        tx_id: str,
        actual_exit_price: float,
        realized_pnl_cad: float,
        realized_pnl_pct: float,
        closing_reason: str,
    ) -> None:
        """Mark Wealthsimple trade as closed and record final PnL."""
        with self.get_connection() as conn:
            conn.execute(
                """
                UPDATE wealthsimple_trades SET
                    status = 'CLOSED',
                    actual_exit = ?,
                    exit_price = ?,
                    exit_time = CURRENT_TIMESTAMP,
                    realized_pnl_cad = ?,
                    realized_pnl_pct = ?,
                    closing_reason = ?
                WHERE tx_id = ?
                """,
                (
                    actual_exit_price,
                    actual_exit_price,
                    realized_pnl_cad,
                    realized_pnl_pct,
                    closing_reason,
                    tx_id,
                ),
            )

    # ══════════════════════════════════════════════════════════════════════
    #  Settings Persistence
    # ══════════════════════════════════════════════════════════════════════

    def get_user_settings(self, user_id: int) -> Dict[str, Any]:
        """Fetch user settings."""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM user_settings WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            if row:
                return dict(row)
            return {
                "active_ticker": "BTC-CAD",
                "active_strategy": "Ensemble Multi-Strategy Alpha",
                "timeframe": "1d",
                "lookback": "6mo",
                "sound_alerts_enabled": 1,
            }

    def save_user_settings(
        self,
        user_id: int,
        active_ticker: Optional[str] = None,
        active_strategy: Optional[str] = None,
        timeframe: Optional[str] = None,
        lookback: Optional[str] = None,
        sound_alerts_enabled: Optional[int] = None,
    ) -> None:
        """Save user settings."""
        current = self.get_user_settings(user_id)
        ticker = active_ticker or current.get("active_ticker", "BTC-CAD")
        strat = active_strategy or current.get("active_strategy", "Ensemble Multi-Strategy Alpha")
        tf = timeframe or current.get("timeframe", "1d")
        lb = lookback or current.get("lookback", "6mo")
        sound = sound_alerts_enabled if sound_alerts_enabled is not None else current.get("sound_alerts_enabled", 1)

        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO user_settings (user_id, active_ticker, active_strategy, timeframe, lookback, sound_alerts_enabled)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    active_ticker = excluded.active_ticker,
                    active_strategy = excluded.active_strategy,
                    timeframe = excluded.timeframe,
                    lookback = excluded.lookback,
                    sound_alerts_enabled = excluded.sound_alerts_enabled
                """,
                (user_id, ticker, strat, tf, lb, sound),
            )



    # ======================================================================
    #  TradeSense v5 Virtual-Money Bot Engine Persistence
    # ======================================================================

    def get_or_create_virtual_account(
        self,
        user_id: str = "apex_quant",
        starting_cash: float = 10000.0,
        ws_tier: str = "Core",
        usd_account_active: bool = False,
    ) -> Dict[str, Any]:
        """Fetch or initialize the user's persistent virtual trading account."""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM virtual_bot_accounts WHERE user_id = ?",
                (str(user_id),),
            ).fetchone()
            if row:
                return dict(row)

            conn.execute(
                """
                INSERT INTO virtual_bot_accounts (user_id, starting_virtual_cad, virtual_cash_cad, is_armed, ws_tier, usd_account_active)
                VALUES (?, ?, ?, 1, ?, ?)
                """,
                (str(user_id), starting_cash, starting_cash, ws_tier, 1 if usd_account_active else 0),
            )
            return {
                "user_id": str(user_id),
                "starting_virtual_cad": starting_cash,
                "virtual_cash_cad": starting_cash,
                "is_armed": 1,
                "ws_tier": ws_tier,
                "usd_account_active": 1 if usd_account_active else 0,
                "risk_per_trade": 0.02,
            }

    def update_virtual_account(
        self,
        user_id: str,
        virtual_cash_cad: Optional[float] = None,
        is_armed: Optional[bool] = None,
        ws_tier: Optional[str] = None,
        usd_account_active: Optional[bool] = None,
    ) -> None:
        """Update virtual account balance, armed status, or fee tier."""
        curr = self.get_or_create_virtual_account(user_id)
        cash = virtual_cash_cad if virtual_cash_cad is not None else curr["virtual_cash_cad"]
        armed = (1 if is_armed else 0) if is_armed is not None else curr["is_armed"]
        tier = ws_tier if ws_tier is not None else curr["ws_tier"]
        usd_acc = (1 if usd_account_active else 0) if usd_account_active is not None else curr["usd_account_active"]

        with self.get_connection() as conn:
            conn.execute(
                """
                UPDATE virtual_bot_accounts
                SET virtual_cash_cad = ?, is_armed = ?, ws_tier = ?, usd_account_active = ?, updated_at = CURRENT_TIMESTAMP
                WHERE user_id = ?
                """,
                (cash, armed, tier, usd_acc, str(user_id)),
            )

    def reset_virtual_wallet(self, user_id: str, starting_cash: float = 10000.0) -> None:
        """Reset virtual wallet to starting cash and wipe open positions."""
        with self.get_connection() as conn:
            conn.execute(
                """
                UPDATE virtual_bot_accounts
                SET starting_virtual_cad = ?, virtual_cash_cad = ?, updated_at = CURRENT_TIMESTAMP
                WHERE user_id = ?
                """,
                (starting_cash, starting_cash, str(user_id)),
            )
            conn.execute("DELETE FROM virtual_bot_positions WHERE user_id = ?", (str(user_id),))

    def get_virtual_positions(self, user_id: str) -> List[Dict[str, Any]]:
        """Retrieve all active open virtual positions."""
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM virtual_bot_positions WHERE user_id = ? ORDER BY opened_at DESC",
                (str(user_id),),
            ).fetchall()
            return [dict(r) for r in rows]

    def save_virtual_position(
        self,
        user_id: str,
        ticker: str,
        ws_symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        current_price: float,
        predicted_p50: float,
        target_p90: float,
        stop_p10: float,
        entry_fee_cad: float,
        dna_profile: str,
        horizon_bars: int = 5,
    ) -> None:
        """Insert or update an active virtual position."""
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO virtual_bot_positions (
                    user_id, ticker, ws_symbol, side, quantity, entry_price, current_price,
                    predicted_p50, target_p90, stop_p10, entry_fee_cad, dna_profile, horizon_bars
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, ticker) DO UPDATE SET
                    current_price = excluded.current_price,
                    predicted_p50 = excluded.predicted_p50,
                    target_p90 = excluded.target_p90,
                    stop_p10 = excluded.stop_p10,
                    bars_held = bars_held + 1
                """,
                (
                    str(user_id), ticker, ws_symbol, side, quantity, entry_price, current_price,
                    predicted_p50, target_p90, stop_p10, entry_fee_cad, dna_profile, horizon_bars,
                ),
            )

    def delete_virtual_position(self, user_id: str, ticker: str) -> None:
        """Remove a closed position from active positions."""
        with self.get_connection() as conn:
            conn.execute(
                "DELETE FROM virtual_bot_positions WHERE user_id = ? AND ticker = ?",
                (str(user_id), ticker),
            )

    def record_virtual_trade(
        self,
        user_id: str,
        ticker: str,
        ws_symbol: str,
        entry_price: float,
        exit_price: float,
        predicted_p50: float,
        quantity: float,
        gross_pnl_cad: float,
        ws_fees_cad: float,
        net_pnl_cad: float,
        net_pnl_pct: float,
        prediction_hit: bool,
        exit_reason: str,
        mode: str = "LIVE_VIRTUAL",
        opened_at: Optional[str] = None,
    ) -> None:
        """Record a completed virtual trade into the proof-of-edge ledger."""
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO virtual_bot_trades (
                    user_id, ticker, ws_symbol, mode, entry_price, exit_price,
                    predicted_p50, quantity, gross_pnl_cad, ws_fees_cad,
                    net_pnl_cad, net_pnl_pct, prediction_hit, exit_reason, opened_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(user_id), ticker, ws_symbol, mode, entry_price, exit_price,
                    predicted_p50, quantity, gross_pnl_cad, ws_fees_cad,
                    net_pnl_cad, net_pnl_pct, 1 if prediction_hit else 0, exit_reason,
                    opened_at,
                ),
            )

    def get_virtual_trades(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve historical virtual trades."""
        with self.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM virtual_bot_trades
                WHERE user_id = ?
                ORDER BY closed_at DESC
                LIMIT ?
                """,
                (str(user_id), limit),
            ).fetchall()
            return [dict(r) for r in rows]



    # ======================================================================
    #  TradeSense v6 Multi-Profile & Real Wealthsimple Trade Engine
    # ======================================================================

    def get_user_profile(self, profile_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve profile attributes by slug (e.g. 'uzair', 'waleed')."""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM user_profiles WHERE profile_id = ?",
                (str(profile_id).lower(),),
            ).fetchone()
            return dict(row) if row else None

    def list_user_profiles(self) -> List[Dict[str, Any]]:
        """List all registered user profiles ordered by creation time."""
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM user_profiles ORDER BY created_at ASC"
            ).fetchall()
            return [dict(r) for r in rows]

    def save_user_profile(self, data: Dict[str, Any]) -> None:
        """Create or update a user profile."""
        pid = str(data["profile_id"]).lower()
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO user_profiles (
                    profile_id, display_name, avatar_initials, accent_color,
                    ws_tier, usd_account_active, active_symbol, active_timeframe,
                    horizon_bars, model_preset, selected_models_json, blending_mode,
                    last_active_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(profile_id) DO UPDATE SET
                    display_name = excluded.display_name,
                    avatar_initials = excluded.avatar_initials,
                    accent_color = excluded.accent_color,
                    ws_tier = excluded.ws_tier,
                    usd_account_active = excluded.usd_account_active,
                    active_symbol = excluded.active_symbol,
                    active_timeframe = excluded.active_timeframe,
                    horizon_bars = excluded.horizon_bars,
                    model_preset = excluded.model_preset,
                    selected_models_json = excluded.selected_models_json,
                    blending_mode = excluded.blending_mode,
                    last_active_at = CURRENT_TIMESTAMP
                """,
                (
                    pid,
                    data.get("display_name", pid.capitalize()),
                    data.get("avatar_initials", pid[:2].upper()),
                    data.get("accent_color", "#38bdf8"),
                    data.get("ws_tier", "Core"),
                    1 if data.get("usd_account_active") else 0,
                    data.get("active_symbol", "BTC-CAD"),
                    data.get("active_timeframe", "1h"),
                    int(data.get("horizon_bars", 5)),
                    data.get("model_preset", "AUTO_DNA"),
                    data.get("selected_models_json", "[]"),
                    data.get("blending_mode", "WALK_FORWARD"),
                ),
            )

    def delete_user_profile(self, profile_id: str) -> None:
        """Delete a profile and associated accounts."""
        pid = str(profile_id).lower()
        with self.get_connection() as conn:
            conn.execute("DELETE FROM user_profiles WHERE profile_id = ?", (pid,))
            conn.execute("DELETE FROM virtual_bot_accounts WHERE user_id = ?", (pid,))
            conn.execute("DELETE FROM virtual_bot_positions WHERE user_id = ?", (pid,))
            conn.execute("DELETE FROM virtual_bot_trades WHERE user_id = ?", (pid,))
            conn.execute("DELETE FROM real_ws_positions WHERE profile_id = ?", (pid,))

    def get_real_ws_positions(self, profile_id: Optional[str] = None, status: str = "OPEN") -> List[Dict[str, Any]]:
        """Retrieve real logged Wealthsimple positions (OPEN or CLOSED)."""
        with self.get_connection() as conn:
            if profile_id:
                rows = conn.execute(
                    "SELECT * FROM real_ws_positions WHERE profile_id = ? AND status = ? ORDER BY opened_at DESC",
                    (str(profile_id).lower(), status),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM real_ws_positions WHERE status = ? ORDER BY opened_at DESC",
                    (status,),
                ).fetchall()
            return [dict(r) for r in rows]

    def save_real_ws_position(self, data: Dict[str, Any]) -> None:
        """Log a real Wealthsimple execution fill."""
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO real_ws_positions (
                    tx_id, profile_id, ticker, ws_symbol, route_type, status,
                    entry_price, current_price, quantity, capital_cad,
                    target_price, stop_price, trailing_stop_price, est_fee_cad
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tx_id) DO UPDATE SET
                    current_price = excluded.current_price,
                    trailing_stop_price = excluded.trailing_stop_price
                """,
                (
                    data["tx_id"],
                    str(data["profile_id"]).lower(),
                    data["ticker"],
                    data["ws_symbol"],
                    data.get("route_type", "DIRECT"),
                    data.get("status", "OPEN"),
                    float(data["entry_price"]),
                    float(data.get("current_price", data["entry_price"])),
                    float(data["quantity"]),
                    float(data["capital_cad"]),
                    float(data["target_price"]),
                    float(data["stop_price"]),
                    float(data.get("trailing_stop_price", data["stop_price"])),
                    float(data.get("est_fee_cad", 0.0)),
                ),
            )

    def close_real_ws_position(
        self,
        tx_id: str,
        exit_price: float,
        exit_reason: str = "MANUAL_CLOSE",
        realized_pnl_cad: Optional[float] = None,
        realized_pnl_pct: Optional[float] = None,
    ) -> None:
        """Close a real Wealthsimple position and record realized CAD outcome."""
        with self.get_connection() as conn:
            if realized_pnl_cad is None or realized_pnl_pct is None:
                row = conn.execute("SELECT entry_price, quantity FROM real_ws_positions WHERE tx_id = ?", (tx_id,)).fetchone()
                if row:
                    entry = float(row["entry_price"])
                    qty = float(row["quantity"])
                    realized_pnl_cad = round((exit_price - entry) * qty, 2)
                    realized_pnl_pct = round(((exit_price - entry) / max(entry, 1e-4)) * 100.0, 2)
                else:
                    realized_pnl_cad = 0.0
                    realized_pnl_pct = 0.0

            conn.execute(
                """
                UPDATE real_ws_positions
                SET status = 'CLOSED',
                    exit_price = ?,
                    exit_reason = ?,
                    realized_pnl_cad = ?,
                    realized_pnl_pct = ?,
                    closed_at = CURRENT_TIMESTAMP
                WHERE tx_id = ?
                """,
                (exit_price, exit_reason, realized_pnl_cad, realized_pnl_pct, tx_id),
            )

    def get_real_ws_history(self, profile_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve complete real trade history for audit/leaderboard."""
        with self.get_connection() as conn:
            if profile_id:
                rows = conn.execute(
                    "SELECT * FROM real_ws_positions WHERE profile_id = ? ORDER BY opened_at DESC LIMIT ?",
                    (str(profile_id).lower(), limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM real_ws_positions ORDER BY opened_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]


# Global singleton instance
db_manager = DatabaseManager()
_GLOBAL_DB = db_manager


def get_database() -> DatabaseManager:
    """Return the global DatabaseManager singleton instance."""
    global _GLOBAL_DB
    return _GLOBAL_DB
