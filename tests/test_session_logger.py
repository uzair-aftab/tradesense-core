"""
Unit tests for BotSessionLogger, Timed Arming, and SessionLogManager
"""

import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pytest

from src.data.live_feed import LiveMarketFeed, StreamMode
from src.models.bot import AutonomousSniperBot, BotConfig, BotState
from src.models.session_logger import BotSessionLogger, SessionLogManager, SESSIONS_DIR
from src.models.strategies import BaseStrategy, SignalType, StrategySignal
from tests.test_bot import DummyBullishStrategy, _create_sample_df


def test_bot_timed_session_arming_and_auto_disarm(tmp_path):
    """Verify arming with configurable timeslot and automatic disarming upon expiration."""
    df = _create_sample_df(30, start_price=100.0)
    feed = LiveMarketFeed(ticker="BTC-CAD", mode=StreamMode.HISTORICAL_REPLAY, base_df=df)
    bot = AutonomousSniperBot(feed=feed, strategy=DummyBullishStrategy())

    # Arm for 15 minutes
    session_id = bot.arm(duration_minutes=15.0, session_tag="test_15m")
    assert bot.state == BotState.ARMED
    assert session_id is not None
    assert bot.session_logger is not None
    assert bot.session_duration_minutes == 15.0
    assert bot.session_end_time is not None

    status = bot.get_session_status()
    assert status["active"] is True
    assert status["is_timed"] is True
    assert status["duration_minutes"] == 15.0
    assert status["remaining_seconds"] > 0
    assert "14:" in status["remaining_formatted"] or "15:00" in status["remaining_formatted"]

    # Step bot to produce ticks and log records
    for _ in range(3):
        bot.step()

    assert bot.session_logger.ticks_processed >= 3
    jsonl_file = bot.session_logger.jsonl_path
    assert jsonl_file.exists()

    # Verify JSONL lines have all required rich telemetry fields
    with open(jsonl_file, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    assert len(lines) >= 3
    first_record = lines[0]
    assert "tick_index" in first_record
    assert "market" in first_record
    assert "price" in first_record["market"]
    assert "indicators" in first_record["market"]
    assert "decision" in first_record
    assert "portfolio" in first_record
    assert "cash" in first_record["portfolio"]

    # Now simulate time expiration (fast-forward clock past end_time)
    bot.session_end_time = datetime.now(timezone.utc) - timedelta(seconds=1)

    # Next step should detect expiration and disarm automatically
    bot.step()
    assert bot.state == BotState.DISARMED
    assert bot.session_logger is None

    # Verify summary JSON and Markdown report files exist
    summary_file = SESSIONS_DIR / f"{session_id}_summary.json"
    report_file = SESSIONS_DIR / f"{session_id}_report.md"
    assert summary_file.exists()
    assert report_file.exists()

    # Verify summary structure
    with open(summary_file, "r", encoding="utf-8") as f:
        summary = json.load(f)
    assert summary["session_id"] == session_id
    assert summary["target_duration_minutes"] == 15.0
    assert "net_pnl" in summary
    assert "total_ticks_processed" in summary
    assert "win_rate_pct" in summary


def test_session_log_manager_analysis():
    """Verify SessionLogManager can list sessions, load details, and generate AI analysis prompt."""
    sessions = SessionLogManager.list_sessions()
    assert isinstance(sessions, list)
    assert len(sessions) > 0

    latest = sessions[0]
    session_id = latest["session_id"]

    summary = SessionLogManager.load_session_summary(session_id)
    assert summary is not None
    assert summary["session_id"] == session_id

    records = SessionLogManager.load_session_records(session_id, max_records=50)
    assert isinstance(records, list)
    assert len(records) > 0

    # Generate AI diagnostic prompt
    prompt = SessionLogManager.generate_ai_analysis_prompt(session_id)
    assert "Post-Trade Learning" in prompt
    assert session_id in prompt
    assert "Sub-Strategy Signal Distribution" in prompt
