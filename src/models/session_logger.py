"""
TradeSense Institutional Session Logger & Telemetry Engine
==========================================================

Provides institutional session recording, tick-by-tick microstructural event capture,
configurable timeslot management, and automated diagnostic summaries for post-trade
AI analysis and decision enhancement.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger("hfp.session_logger")

SESSIONS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "bot_sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class SessionTradeRecord:
    trade_id: str
    ticker: str
    side: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    return_pct: float
    exit_reason: str
    holding_ticks: int
    slippage_cost: float
    commission_cost: float


class BotSessionLogger:
    """
    Manages structured, streaming tick-by-tick telemetry for a single timed bot session.
    Writes every market tick, technical indicator, AI forecast, sub-strategy decision,
    and execution fill into a high-density JSON Lines (.jsonl) file and companion summary.
    """

    def __init__(
        self,
        symbol: str,
        target_duration_minutes: float,
        initial_equity: float,
        initial_price: float,
        config_snapshot: Dict[str, Any],
        session_tag: Optional[str] = None,
    ):
        self.symbol = symbol
        self.target_duration_minutes = target_duration_minutes
        self.initial_equity = initial_equity
        self.initial_price = initial_price
        self.config_snapshot = config_snapshot
        self.session_tag = session_tag or "session"

        # Timestamp identifiers
        now = datetime.now(timezone.utc)
        self.start_time = now
        time_str = now.strftime("%Y%m%d_%H%M%S")
        dur_label = f"{int(target_duration_minutes)}m" if target_duration_minutes else "indefinite"
        clean_sym = symbol.replace("-", "").replace("/", "")
        self.session_id = f"session_{time_str}_{clean_sym}_{dur_label}"

        # File paths
        self.jsonl_path = SESSIONS_DIR / f"{self.session_id}.jsonl"
        self.summary_path = SESSIONS_DIR / f"{self.session_id}_summary.json"
        self.report_path = SESSIONS_DIR / f"{self.session_id}_report.md"

        self._file_handle = open(self.jsonl_path, "a", encoding="utf-8")

        # Running session telemetry
        self.ticks_processed = 0
        self.last_price = initial_price
        self.peak_equity = initial_equity
        self.min_equity = initial_equity
        self.max_drawdown_pct = 0.0

        # Trade tracking
        self.trades: List[SessionTradeRecord] = []
        self._active_entry_events: Dict[str, Dict[str, Any]] = {}
        self.winning_trades = 0
        self.losing_trades = 0
        self.gross_profit = 0.0
        self.gross_loss = 0.0

        logger.info(
            "SessionLogger initialised: %s | Target: %s min | Output: %s",
            self.session_id,
            target_duration_minutes,
            self.jsonl_path.name,
        )

    def log_tick_snapshot(
        self,
        tick: Any,
        df_indicators: Optional[Dict[str, Any]],
        prediction: Optional[Any],
        decision: Dict[str, Any],
        execution_event: Dict[str, Any],
        portfolio: Dict[str, Any],
    ) -> None:
        """
        Record a comprehensive, fully deserializable telemetry record for this tick.
        """
        self.ticks_processed += 1
        cur_price = float(tick.price)
        self.last_price = cur_price

        # Update equity telemetry
        cur_equity = float(portfolio.get("equity", self.initial_equity))
        if cur_equity > self.peak_equity:
            self.peak_equity = cur_equity
        if cur_equity < self.min_equity:
            self.min_equity = cur_equity

        dd = (self.peak_equity - cur_equity) / (self.peak_equity + 1e-9) * 100.0
        if dd > self.max_drawdown_pct:
            self.max_drawdown_pct = dd

        # Format indicators snapshot safely
        indicators = {}
        if df_indicators:
            for k in [
                "EMA_9",
                "EMA_20",
                "EMA_50",
                "BB_Upper",
                "BB_Mid",
                "BB_Lower",
                "RSI_14",
                "ATR_14",
                "VWAP",
                "Whale_Flag",
                "Whale_Z",
            ]:
                if k in df_indicators:
                    val = df_indicators[k]
                    indicators[k] = round(float(val), 4) if val is not None else None

        # Format prediction snapshot
        pred_dict = None
        if prediction is not None:
            pred_dict = {
                "p10": round(float(prediction.p10), 4),
                "p50": round(float(prediction.p50), 4),
                "p90": round(float(prediction.p90), 4),
                "spread": round(float(prediction.spread), 4),
                "direction": str(prediction.direction),
                "confidence": round(float(prediction.confidence), 4),
            }

        # Market micro context
        market_context = {
            "symbol": tick.ticker,
            "price": round(cur_price, 4),
            "bid": round(float(tick.bid), 4),
            "ask": round(float(tick.ask), 4),
            "spread": round(float(tick.spread), 4),
            "spread_bps": round(float(tick.spread / max(0.01, cur_price) * 10000.0), 2),
            "volume": round(float(tick.volume), 4),
            "is_whale": bool(tick.is_whale),
            "whale_z": round(float(tick.whale_z), 2),
            "indicators": indicators,
        }

        # Construct single self-contained JSONL record
        record = {
            "tick_index": self.ticks_processed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": self.session_id,
            "market": market_context,
            "prediction": pred_dict,
            "decision": decision,
            "execution": execution_event,
            "portfolio": portfolio,
        }

        try:
            line = json.dumps(record, default=str)
            self._file_handle.write(line + "\n")
            self._file_handle.flush()
        except Exception as e:
            logger.error("Failed to write session record: %s", e)

    def record_entry(self, fill: Any) -> None:
        """Record entry fill to calculate complete round-trip trade statistics on exit."""
        self._active_entry_events[fill.ticker] = {
            "entry_time": datetime.now(timezone.utc).isoformat(),
            "entry_price": fill.fill_price,
            "quantity": fill.quantity,
            "commission": fill.commission,
            "slippage_pct": fill.slippage_pct,
            "tick_index": self.ticks_processed,
            "order_id": fill.order_id,
        }

    def record_exit(self, fill: Any) -> None:
        """Record trade completion and calculate trade performance metrics."""
        entry = self._active_entry_events.pop(fill.ticker, None)
        entry_price = entry["entry_price"] if entry else fill.fill_price
        entry_time = entry["entry_time"] if entry else datetime.now(timezone.utc).isoformat()
        hold_ticks = self.ticks_processed - entry["tick_index"] if entry else 1

        pnl = float(fill.pnl)
        ret_pct = ((fill.fill_price - entry_price) / max(0.01, entry_price)) * 100.0 if fill.side.value == "SELL" else 0.0

        if pnl > 0:
            self.winning_trades += 1
            self.gross_profit += pnl
        elif pnl < 0:
            self.losing_trades += 1
            self.gross_loss += abs(pnl)

        trade = SessionTradeRecord(
            trade_id=fill.order_id,
            ticker=fill.ticker,
            side=fill.side.value,
            entry_time=entry_time,
            exit_time=datetime.now(timezone.utc).isoformat(),
            entry_price=round(entry_price, 4),
            exit_price=round(fill.fill_price, 4),
            quantity=round(fill.quantity, 6),
            pnl=round(pnl, 2),
            return_pct=round(ret_pct, 2),
            exit_reason=fill.exit_reason or "EXIT",
            holding_ticks=hold_ticks,
            slippage_cost=round(fill.fill_price * (fill.slippage_pct / 100.0) * fill.quantity, 2),
            commission_cost=round(fill.commission, 2),
        )
        self.trades.append(trade)

    def finalize(self, current_equity: float, reason: str = "COMPLETED") -> Dict[str, Any]:
        """
        Close session file and generate machine-readable summary JSON and Markdown report.
        """
        try:
            self._file_handle.close()
        except Exception:
            pass

        now = datetime.now(timezone.utc)
        duration_sec = (now - self.start_time).total_seconds()
        duration_min = round(duration_sec / 60.0, 2)

        net_pnl = current_equity - self.initial_equity
        net_ret_pct = (net_pnl / max(1.0, self.initial_equity)) * 100.0

        mkt_ret_pct = ((self.last_price - self.initial_price) / max(0.01, self.initial_price)) * 100.0
        alpha_pct = net_ret_pct - mkt_ret_pct

        total_trades = len(self.trades)
        win_rate = (self.winning_trades / total_trades * 100.0) if total_trades > 0 else 0.0
        profit_factor = (self.gross_profit / max(0.01, self.gross_loss)) if self.gross_loss > 0 else (99.0 if self.gross_profit > 0 else 1.0)

        summary = {
            "session_id": self.session_id,
            "symbol": self.symbol,
            "tag": self.session_tag,
            "start_time": self.start_time.isoformat(),
            "end_time": now.isoformat(),
            "target_duration_minutes": self.target_duration_minutes,
            "actual_duration_minutes": duration_min,
            "actual_duration_seconds": round(duration_sec, 1),
            "disarm_reason": reason,
            "initial_equity": round(self.initial_equity, 2),
            "final_equity": round(current_equity, 2),
            "net_pnl": round(net_pnl, 2),
            "net_return_pct": round(net_ret_pct, 2),
            "initial_price": round(self.initial_price, 4),
            "final_price": round(self.last_price, 4),
            "market_return_pct": round(mkt_ret_pct, 2),
            "alpha_vs_market_pct": round(alpha_pct, 2),
            "total_ticks_processed": self.ticks_processed,
            "total_trades": total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate_pct": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "peak_equity": round(self.peak_equity, 2),
            "config": self.config_snapshot,
            "trades": [asdict(t) for t in self.trades],
            "jsonl_file": str(self.jsonl_path.resolve()),
            "summary_file": str(self.summary_path.resolve()),
            "report_file": str(self.report_path.resolve()),
        }

        # Write summary JSON
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # Write Markdown Report
        md_report = self._build_markdown_report(summary)
        with open(self.report_path, "w", encoding="utf-8") as f:
            f.write(md_report)

        logger.info(
            "Session %s finalized: Net PnL=$%.2f (%.2f%%) | Trades=%d | WinRate=%.1f%%",
            self.session_id,
            net_pnl,
            net_ret_pct,
            total_trades,
            win_rate,
        )
        return summary

    def _build_markdown_report(self, s: Dict[str, Any]) -> str:
        """Create a professional, actionable Markdown debrief for AI post-trade review."""
        pnl_color = "🟢" if s["net_pnl"] >= 0 else "🔴"
        lines = [
            f"# TradeSense Autonomous Trading Session Report",
            f"**Session ID**: `{s['session_id']}`  ",
            f"**Asset**: `{s['symbol']}` | **Duration**: {s['actual_duration_minutes']} min (Target: {s['target_duration_minutes']} min) | **Disarm Reason**: {s['disarm_reason']}  ",
            f"**Timestamp**: {s['start_time']} -> {s['end_time']}",
            "",
            "## 1. Executive Performance Summary",
            "",
            f"| Metric | Value |",
            f"| :--- | :--- |",
            f"| **Net PnL** | **{pnl_color} ${s['net_pnl']:,.2f} ({s['net_return_pct']:+.2f}%)** |",
            f"| **Starting Capital** | ${s['initial_equity']:,.2f} |",
            f"| **Ending Equity** | ${s['final_equity']:,.2f} |",
            f"| **Asset Price Change** | ${s['initial_price']:,.2f} -> ${s['final_price']:,.2f} ({s['market_return_pct']:+.2f}%) |",
            f"| **Alpha vs Market** | **{s['alpha_vs_market_pct']:+.2f}%** |",
            f"| **Total Closed Trades** | {s['total_trades']} |",
            f"| **Win Rate** | {s['win_rate_pct']:.1f}% ({s['winning_trades']}W / {s['losing_trades']}L) |",
            f"| **Profit Factor** | {s['profit_factor']:.2f} |",
            f"| **Max Drawdown** | {s['max_drawdown_pct']:.2f}% |",
            f"| **Ticks Processed** | {s['total_ticks_processed']:,} |",
            "",
            "## 2. Executed Trades Breakdown",
            "",
        ]

        if not s["trades"]:
            lines.append("_No completed round-trip trades executed during this session._\n")
        else:
            lines.append("| Trade ID | Side | Qty | Entry Price | Exit Price | Net PnL | Return % | Exit Trigger | Hold Ticks |")
            lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
            for t in s["trades"]:
                p_sym = "+" if t["pnl"] >= 0 else ""
                lines.append(
                    f"| `{t['trade_id'][:8]}` | {t['side']} | {t['quantity']} | ${t['entry_price']:,.2f} | ${t['exit_price']:,.2f} | **{p_sym}${t['pnl']:,.2f}** | {t['return_pct']:+.2f}% | `{t['exit_reason']}` | {t['holding_ticks']} |"
                )
            lines.append("")

        lines.extend(
            [
                "## 3. Bot Risk Configuration",
                f"- **Risk Per Trade**: `{s['config'].get('risk_per_trade', 0.02) * 100:.1f}%`",
                f"- **Min Confidence**: `{s['config'].get('min_confidence', 0.70) * 100:.0f}%`",
                f"- **Min RRR**: `{s['config'].get('min_rrr', 1.5):.2f}x`",
                f"- **Trailing Stop Pct**: `{s['config'].get('trailing_stop_pct', 0.015) * 100:.1f}%`",
                f"- **Strategy**: `{s['config'].get('strategy', 'EnsembleAlpha')}`",
                "",
                "## 4. Telemetry File Locations",
                f"- **Full JSONL Tick Stream**: `{s['jsonl_file']}`",
                f"- **Summary JSON**: `{s['summary_file']}`",
                f"- **Markdown Report**: `{s['report_file']}`",
            ]
        )
        return "\n".join(lines)


class SessionLogManager:
    """
    Utility class for searching, loading, and performing deep-dive AI diagnostics
    on past bot sessions.
    """

    @staticmethod
    def list_sessions() -> List[Dict[str, Any]]:
        """List all completed session summaries sorted newest first."""
        if not SESSIONS_DIR.exists():
            return []

        summaries = []
        for file in SESSIONS_DIR.glob("*_summary.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    summaries.append(data)
            except Exception as e:
                logger.warning("Could not read session summary %s: %s", file, e)

        summaries.sort(key=lambda x: x.get("start_time", ""), reverse=True)
        return summaries

    @staticmethod
    def load_session_summary(session_id: str) -> Optional[Dict[str, Any]]:
        """Load summary dictionary by session_id."""
        path = SESSIONS_DIR / f"{session_id}_summary.json"
        if not path.exists():
            # Check if file has session prefix
            matches = list(SESSIONS_DIR.glob(f"*{session_id}*_summary.json"))
            if matches:
                path = matches[0]
            else:
                return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def load_session_records(session_id: str, max_records: int = 1000) -> List[Dict[str, Any]]:
        """Load JSONL tick stream records for deep-dive analysis."""
        path = SESSIONS_DIR / f"{session_id}.jsonl"
        if not path.exists():
            matches = list(SESSIONS_DIR.glob(f"*{session_id}*.jsonl"))
            if matches:
                path = matches[0]
            else:
                return []

        records = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_records:
                    break
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass
        return records

    @staticmethod
    def generate_ai_analysis_prompt(session_id: str) -> str:
        """
        Produce a comprehensive diagnostic prompt containing market history, indicator
        extremes, sub-strategy breakdown, and trade exits for AI post-trade learning.
        """
        summary = SessionLogManager.load_session_summary(session_id)
        if not summary:
            return f"Session `{session_id}` not found."

        records = SessionLogManager.load_session_records(session_id, max_records=200)

        # Extract market price movements and indicator ranges
        prices = [r["market"]["price"] for r in records if "market" in r]
        p_min = min(prices) if prices else 0.0
        p_max = max(prices) if prices else 0.0

        # Sub-strategy vote distributions
        strategy_votes: Dict[str, Dict[str, int]] = {}
        for r in records:
            dec = r.get("decision", {})
            subs = dec.get("sub_strategy_signals", [])
            for s in subs:
                s_name = s.get("strategy", "Unknown")
                sig = s.get("signal", "HOLD")
                if s_name not in strategy_votes:
                    strategy_votes[s_name] = {"BUY": 0, "SELL": 0, "HOLD": 0}
                strategy_votes[s_name][sig] = strategy_votes[s_name].get(sig, 0) + 1

        prompt = [
            f"# Post-Trade Learning & Optimization Diagnostic",
            f"**Session ID**: `{summary['session_id']}`",
            f"**Asset**: `{summary['symbol']}` | **Duration**: {summary['actual_duration_minutes']} min",
            f"**Net Result**: ${summary['net_pnl']:,.2f} ({summary['net_return_pct']:+.2f}%) | Max DD: {summary['max_drawdown_pct']:.2f}%",
            f"**Market Price Range**: ${p_min:,.2f} - ${p_max:,.2f} ({summary['market_return_pct']:+.2f}%)",
            "",
            "## 1. Trade Results & Exit Triggers",
        ]

        if summary.get("trades"):
            for t in summary["trades"]:
                prompt.append(
                    f"- **Trade {t['trade_id'][:8]}** ({t['side']} {t['quantity']}): Entry @ ${t['entry_price']:,.2f} -> Exit @ ${t['exit_price']:,.2f} | PnL: ${t['pnl']:,.2f} ({t['return_pct']:+.2f}%) | Trigger: `{t['exit_reason']}` (Hold: {t['holding_ticks']} ticks)"
                )
        else:
            prompt.append("- _No trades executed._")

        prompt.extend(
            [
                "",
                "## 2. Sub-Strategy Signal Distribution During Session",
            ]
        )
        for s_name, counts in strategy_votes.items():
            prompt.append(f"- **{s_name}**: BUY={counts.get('BUY', 0)} | SELL={counts.get('SELL', 0)} | HOLD={counts.get('HOLD', 0)}")

        prompt.extend(
            [
                "",
                "## 3. Analysis Objective",
                "Based on this telemetry:",
                "1. Analyze why losing trades occurred (e.g. was trailing stop too tight for current ATR, was entry too late, or was consensus premature?).",
                "2. Determine if the market was trending or range-bound and whether the active strategy aligned with that regime.",
                "3. Provide concrete adjustments to risk parameters (trailing stop pct, min confidence threshold, RRR) to maximize net profitability.",
            ]
        )
        return "\n".join(prompt)
