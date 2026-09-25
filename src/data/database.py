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


# Global singleton instance
db_manager = DatabaseManager()
_GLOBAL_DB = db_manager


def get_database() -> DatabaseManager:
    """Return the global DatabaseManager singleton instance."""
    global _GLOBAL_DB
    return _GLOBAL_DB
