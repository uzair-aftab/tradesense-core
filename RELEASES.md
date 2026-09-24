# TradeSense Institutional System — Release History & Version Control Log (`RELEASES.md`)

This document records the state of the TradeSense project across major releases, detailing implemented features, architectural trade-offs, advantages, and drawbacks for each milestone.

---

## Release `v2.3.0` — Regime-Adaptive Strategy Gating, Circuit Breakers, RTX 4070 Acceleration & Wealthsimple Real-World Mobile Copier Engine

- **Date**: 2026-09-24
- **Branch**: `overhaul/tradesense-best`
- **Scope**: Market Regime Engine, Wilder ADX/DMI Integration, Circuit Breakers, RTX 4070 PyTorch CUDA Acceleration, Wealthsimple Mobile Trade Copier Studio, and 41 Passing Unit Tests.

### Project State
Directly addresses the forensic findings from the 6 completed live sessions (2,320 ticks, 3 hours, 21 executed trades, 15 losses) and bridges the simulator with the real-world operational realities of trading on the **Wealthsimple mobile app in North York, Ontario, Canada**. Eliminates range-ceiling buy traps through a brand-new multi-factor Market Regime Classifier, locks capital in 100% Cash Preservation during downtrends, protects drawdowns with a 2-consecutive-loss Circuit Breaker, accelerates deep learning models on the NVIDIA GeForce RTX 4070 Laptop GPU, and equips the cockpit with a dedicated 📲 Wealthsimple Mobile Trade Copier Card.

### Key Features Added
1. **Multi-Factor Market Regime Engine (`src/models/regime.py`)**:
   - `MarketRegime` enum: `BULL_TREND`, `BEAR_TREND`, `RANGING_CHOP`, `HIGH_VOLATILITY`.
   - `RegimeClassifier`: Computes true Wilder ADX-14, +DI, -DI, Bollinger %B, and EMA slopes.
   - `RegimeContext`: Emits execution permissions (`allow_breakout_buys`, `allow_mean_reversion_buys`, `cash_preservation_active`).
2. **Real-Time Dynamic Indicators in Streaming Feed (`src/data/live_feed.py`)**:
   - Computes dynamic Wilder ADX-14, +DI, and -DI on streaming bars, replacing static defaults.
3. **Regime-Adaptive Sub-Strategies & Ensemble Alpha (`src/models/strategies.py`)**:
   - `DonchianVolatilityBreakout`: Gated against false breakouts in ranging chop or at Bollinger upper ceilings (%B > 90% without ADX > 28).
   - `StatisticalMeanReversion`: Gated to buy only on true oversold channel floors (Price <= BB_Lower and RSI <= 38).
   - `EnsembleAlpha`: Enforces 100% Cash Preservation Lock in `BEAR_TREND`; dynamically weights sub-strategies based on regime; suppresses breakout buys in `RANGING_CHOP`.
4. **Autonomous Bot Hardening & Circuit Breakers (`src/models/bot.py`)**:
   - **Consecutive Loss Breaker**: 2 consecutive stop-outs trigger an automated 60-tick (2-minute) execution freeze.
   - **Wealthsimple Target Hurdle**: Enforces minimum target distance ($\ge 1.0\%$) to beat crypto spread friction and human execution latency.
   - **Canadian Proxy Mapping**: Auto-maps `BTC-USD`/`BTC-CAD` to Canadian TSX ETF `BTCC.TO` ($0 commission on Wealthsimple).
   - **Continuous AI Telemetry**: Auto-feeds forward quantile projections into `DeepQuantileSniper` so the bot never operates blindly.
5. **NVIDIA RTX 4070 Laptop GPU Acceleration (`src/models/ai_engine.py`)**:
   - Configured PyTorch mixed precision (`torch.amp.autocast`) and CUDA pinned memory (`pin_memory=True`).
   - Non-blocking host-to-device transfers (`.to(DEVICE, non_blocking=True)`).
   - XGBoost GPU acceleration (`device="cuda"`, `tree_method="hist"`).
6. **Cockpit UI & Wealthsimple Mobile Copier Studio (`src/ui/app.py`)**:
   - **Live Market Regime Badge** (`🟢 BULL TREND`, `🟡 RANGING CHOP`, `🔴 BEAR TREND - CASH LOCK`).
   - **📲 Wealthsimple Mobile Trade Copier Card**: High-visibility card with Canadian TSX proxy, Limit Entry, TP, SL, RRR, and 1-click clipboard copy formatted for phone execution.
   - Circuit breaker active countdown banner.
7. **Comprehensive Automated Test Suite**:
   - 41/41 passing unit tests across all models, strategies, regime classifier, and UI components.

### Pros & Cons
- **Pros**:
  - Direct real-world alignment with Wealthsimple mobile trading in Ontario (zero-commission Canadian TSX ETFs, actionable swing horizons, spread hurdle protection).
  - Eliminates the primary root cause of historical losses (buying range ceilings during chop).
  - Rapid GPU training and inference on RTX 4070 with CUDA mixed precision.
  - Bulletproof drawdown protection via automated circuit breakers.
- **Cons & Trade-offs**:
  - Stricter entry gating reduces trade frequency during choppy markets (a deliberate quant trade-off favoring high-conviction win rate over overtrading).

---

## Release `v2.2.0` — Configurable Timed Bot Sessions & AI Post-Trade Telemetry Diagnostics

- **Date**: 2026-09-24
- **Branch**: `overhaul/tradesense-best`
- **Scope**: Timed Bot Arming, Tick-by-Tick JSONL Session Logging, and AI Post-Trade Optimization Studio

### Project State
Introduces full lifecycle session tracking for algorithmic bot execution. Users can arm the Sniper Bot for configurable timeslots (15m, 20m, 30m, 60m, custom, or continuous), during which every microstructural market tick, technical indicator snapshot, AI forecast, sub-strategy consensus vote, order execution, and portfolio mark-to-market equity is streamed to a dedicated JSON Lines (`.jsonl`) file. Upon timer expiration, the bot automatically disarms, finalizes execution, and generates executive Markdown reports and machine-readable JSON summaries ready for future AI learning and post-trade performance enhancement.

### Key Features Added
1. **Configurable Timed Arming & Auto-Disarm Engine (`src/models/bot.py`)**:
   - `AutonomousSniperBot.arm(duration_minutes=...)` supports configurable execution slots with automated expiration checking on every tick.
   - Automatically disarms, closes active files, and logs session completion when the target time is reached.
   - Real-time countdown timer (`MM:SS left`) with progress bar in the cockpit top ribbon.
2. **High-Density JSONL Telemetry & Session Logger (`src/models/session_logger.py`)**:
   - Dedicated directory `data/bot_sessions/` with files named `session_YYYYMMDD_HHMMSS_<SYMBOL>_<DURATION>m.jsonl`.
   - Captures comprehensive market context (price, bid, ask, spread bps, volume, whale Z-score), indicators (EMA 9/20/50, BB, RSI, ATR, VWAP), predictions (P10/P50/P90, direction, confidence), sub-strategy voting breakdown, order execution details, and portfolio equity.
   - Generates companion `_summary.json` and human-readable executive `_report.md` debrief files.
3. **Sub-Strategy Consensus Observability (`src/models/strategies.py`)**:
   - `EnsembleAlpha` tracks `last_sub_signals`, `last_buy_votes`, and `last_sell_votes` so the session logger records the exact individual votes of every underlying quant strategy.
4. **Session Archives & AI Telemetry Tab (`src/ui/app.py`)**:
   - Added a 4th cockpit tab: "📂 Session Archives & AI Telemetry".
   - Displays all historical completed sessions with Net PnL, Win Rate %, Total Trades, and Max Drawdown in a clean table.
   - Interactive report viewer displaying executive markdown reports and generating actionable AI post-trade diagnostic prompts.
   - Direct download buttons for `.jsonl` tick logs and summary `.json`.
5. **Comprehensive Unit Tests (`tests/test_session_logger.py`)**:
   - Tested timed arming, automatic expiration, JSONL record schema integrity, and `SessionLogManager` analysis generation. Test suite expanded to 34 passing tests.

### Pros & Cons
- **Pros**:
  - Full transparency: 100% of decisions, inputs, and market states are recorded without loss.
  - Zero risk of unattended runaways: timed slots guarantee the bot disarms after the user's chosen duration.
  - Rich diagnostics ready for future AI-driven strategy tuning and hyperparameter optimization.
- **Cons & Trade-offs**:
  - Long high-frequency sessions (e.g. several hours of 1-second ticks) produce detailed multi-megabyte `.jsonl` files (mitigated by Git ignoring large `.jsonl` outputs while preserving summaries).

---

## Release `v2.1.1` — EnsembleAlpha Bugfix, Category Axis Monolith Collapse Fix & Volume Calibration

- **Date**: 2026-09-24
- **Branch**: `overhaul/tradesense-best`
- **Scope**: Signal Engine Bugfix, Plotly Category Collision Resolution, and Tick Volume Calibration

### Project State
Directly fixes two high-visibility bugs reported by the user: (1) an unhandled `NameError: name 'price' is not defined` inside `EnsembleAlpha.generate_signal()`, and (2) a Plotly category axis collapse bug where daily bars all sharing timestamp `00:00:00` collapsed into a single vertical monolith with 2 Trillion volume. All 32 unit tests now pass cleanly.

### Key Features & Fixes Added
1. **`EnsembleAlpha.generate_signal()` Price Resolution (`src/models/strategies.py`)**:
   - Explicitly defined `price = float(df["Close"].iloc[current_idx]) if not df.empty else 100.0`.
   - Restored autonomous bot signal generation and eliminated fatal Streamlit fragment crashes.
2. **Guaranteed Unique Category Axis Formatting (`src/ui/app.py`)**:
   - Built `_format_chart_time_labels()` that inspects timestamps and dynamically formats daily bars as `%b %d` and intraday bars as `%H:%M:%S`.
   - Appends collision disambiguators (`#2`, `#3`) for any identical time stamps, guaranteeing Plotly never stacks multiple candles into a single monolithic category.
3. **Decoupled Microstructure Baseline & Volume Calibration (`src/data/live_feed.py`)**:
   - In `SYNTHETIC_MICROSTRUCTURE` mode, `LiveMarketFeed` now creates a clean intraday baseline ending right at the current timestamp, rather than appending sub-minute ticks onto multi-month daily bars.
   - Scaled tick and candle volumes realistically: crypto assets (BTC, ETH, SOL) produce realistic fractional volumes (0.05 to 1.8 BTC per tick, 2 to 18 BTC per bar) instead of multi-thousand lots that blew up chart volume into trillions.
4. **Clean Stream Mode Switching**:
   - Added `LiveMarketFeed.mode` property setter that automatically re-initializes data streams when toggling between Synthetic Microstructure and Historical Replay.
5. **Regression Unit Tests (`tests/test_bot.py`)**:
   - Added `test_ensemble_alpha_no_name_error`, `test_live_feed_mode_switch`, and `test_chart_time_label_uniqueness`. Test suite expanded to 32 tests passing.

### Pros & Cons
- **Pros**:
  - Eliminates the `NameError` crash completely; the bot executes autonomous entries reliably.
  - Candlestick chart renders perfectly spaced, clean candles with crisp OHLC bars and realistic volume bars.
  - Zero collision between daily historical data and live intraday ticks.
- **Cons & Trade-offs**:
  - Switching between Historical Replay and Synthetic Microstructure re-initializes the active chart buffer to maintain time-scale consistency.

---

## Release `v2.1.0` — Institutional Execution Hardening, Fractional Sizing & Professional Candle Aggregation

- **Date**: 2026-09-24
- **Branch**: `overhaul/tradesense-best`
- **Scope**: Execution Hardening, Crypto Support, and Chart Aesthetics

### Project State
Addresses real-world edge cases in high-priced asset execution (e.g. BTC at $80k+), fixes per-tick candle fragmentation, introduces fractional order sizing (USD amount & fractional coins), adds dynamic indicator calculation on streaming bars, and enforces strict cash validation with friendly UI error handling.

### Key Features Added
1. **Fractional Sizing & Multi-Asset Support (`src/models/trader.py`)**:
   - `TradeOrder`, `Position`, and `Fill` support high-precision `float` quantities.
   - Fixed fatal integer truncation bug where trading BTC ($80k) with a $10k portfolio evaluated to `int(0)` units.
2. **Professional Forming Candle Aggregation (`src/data/live_feed.py`)**:
   - Ticks now update the active forming candle; new candles open continuously with `Open = previous Close`, eliminating disjointed floating sticks.
   - Calibrated Merton Jump-Diffusion SDE to realistic 4-bps volatility with mean-reverting anchor drift.
   - Dynamically computes EMAs (9, 20, 50), Bollinger Bands, RSI (14), and ATR on streaming bars.
3. **Smart Order Sizing & Graceful UI Validation (`src/ui/app.py`)**:
   - Added dual order sizing modes: **Capital to Commit in USD ($)** vs **Asset Units**.
   - Real-time estimated cost badge with cash validation.
   - Replaced raw Streamlit crash tracebacks with user-friendly error and toast notifications.
4. **Enhanced Candlestick Chart Aesthetics (`src/ui/app.py`)**:
   - Category x-axis mode eliminates awkward gaps between bars.
   - Plotly dark neon palette (`#00E676` bullish / `#FF1744` bearish) with live horizontal price trackers, entry price levels, and Bollinger cloud.
5. **Adaptive Strategy Consensus (`src/models/strategies.py`, `src/models/bot.py`)**:
   - `EnsembleAlpha` scales threshold dynamically when ML predictions are pending.
   - Heartbeat telemetry logs active scanning so the user sees real-time bot decision-making.

### Pros & Cons
- **Pros**:
  - Completely fixes the manual buy crash (`ValueError: Insufficient cash`).
  - Bot can now seamlessly trade any asset from penny stocks to $100k Bitcoin with fractional lots.
  - Candlestick charts look continuous, solid, and professional like Binance/TradingView.
- **Cons & Trade-offs**:
  - Forming candle aggregation groups ticks into intervals (e.g. 5 ticks), meaning a new candle appears every ~10s instead of every 2s, but the active candle updates continuously.

---

## Release `v2.0.0` — Real-Time Streaming Cockpit, Microstructure Replay & Institutional Strategy Arena

- **Date**: 2026-09-24
- **Branch**: `overhaul/tradesense-best`
- **Scope**: Complete Platform Overhaul & Real-Time Transformation

### Project State
TradeSense is transformed from a static, click-to-trigger prototype into a modern, institutional-grade live algorithmic trading portal. The UI streams live ticking market data every 2 seconds via `@st.fragment`, features an autonomous trading bot with OCO bracket orders and ratcheting trailing stops, offers a 5-strategy tournament arena, provides a simulated/real deep learning studio, and performs multi-asset whale order-flow scanning.

### Key Features Added
1. **Real-Time Live Streaming via `@st.fragment(run_every="2s")`**:
   - Zero-rerun live ticker ribbon, candlestick charts, and execution logs.
   - Eliminates page flashes and input interruptions while market data streams continuously.
2. **High-Frequency Market Microstructure Simulation (`src/data/live_feed.py`)**:
   - Merton Jump-Diffusion Geometric Brownian Motion with Poisson Whale block trade arrival.
   - Dual-mode support: Synthetic Microstructure (24/7 anytime simulation) and Historical Replay.
3. **Institutional Multi-Strategy Suite & Tournament Arena (`src/models/strategies.py`)**:
   - 5 Hedge-Fund Strategies: DeepQuantileSniper, InstitutionalWhaleFlow, VolatilityBreakout, StatisticalMeanReversion, and EnsembleAlpha.
   - 1-Click Strategy Tournament with gold/silver/bronze podium, comprehensive metrics table, and comparative head-to-head equity curve chart.
4. **Autonomous Sniper Trading Bot (`src/models/bot.py`)**:
   - Complete agent loop with ARMED/DISARMED states, automated bracket execution, dynamic ratcheting trailing stops, and tick-by-tick microstructure logging.
   - CSV export functionality for all trade records.
5. **Deep Learning & Simulated Training Studio (`src/ui/app.py`)**:
   - High-speed simulated / dummy training mode with animated epoch pinball loss curves.
   - Production GPU training (PyTorch LSTM + CUDA + XGBoost).
   - Quantile monotonicity verification (100% $P_{10} \le P_{50} \le P_{90}$).
6. **AI Agent Context Ecosystem**:
   - `AGENT.md`, `.cursorrules`, `CLAUDE.md`, and updated `ARCHITECTURE.md`.
7. **Comprehensive Unit Test Suite**:
   - 26 tests passing 100% in under 10 seconds.

### Pros & Cons
- **Pros**:
  - Completely solves the user's primary complaint: charts and feeds are genuinely live and stream continuously.
  - Realistic institutional simulation enables strategy validation even when financial markets are closed.
  - Mathematical monotonicity is strictly guaranteed, eliminating nonsensical crossing quantiles.
  - Complete context files enable any AI assistant or developer to contribute seamlessly.
- **Cons & Trade-offs**:
  - Live fragment polling introduces continuous CPU/GPU load while active; mitigated by the "Auto-Stream" toggle.
  - Synthetic microstructure uses standard Merton Jump-Diffusion parameters, which may differ from specific low-liquidity altcoins.

---

## Release `v1.1.0` — Sniper Execution Engine & Institutional Risk Management

- **Scope**: Execution Realism, Dynamic Stops & Risk Mitigation

### Project State
The platform gained an event-driven paper trading execution engine (`Gauntlet`) with realistic transaction costs and an automated execution gatekeeper (`SniperEngine`).

### Key Features Added
1. **Intra-Bar Candle Polarity Execution**:
   - Evaluates intra-bar High/Low penetrations respecting candle polarity ($Open \to Low \to High \to Close$ for bullish bars; $Open \to High \to Low \to Close$ for bearish bars).
2. **Dynamic Ratcheting Trailing Stops**:
   - Trailing stop floors automatically ratchet upwards as price rises, locking in unrealized gains.
3. **Institutional Risk-to-Reward Filtering**:
   - Rejects trade setups with Reward-to-Risk ratio $< 1.5$x or model confidence $< 70\%$.
4. **Portfolio State Persistence**:
   - Persistent portfolio tracking across application restarts.

### Pros & Cons
- **Pros**:
  - Dramatically improves backtest realism by preventing premature profit-taking or artificial loss avoidance.
  - Trailing stops significantly increase win rate and profit factors in trending regimes.
- **Cons & Trade-offs**:
  - Requires tick-level or intra-bar OHLC data; cannot be tested accurately with Close-only time series.
  - High confidence thresholds filter out 60–70% of potential signals, requiring patient execution.

---

## Release `v1.0.0` — Core Engine Stabilization & Quantile Monotonicity

- **Scope**: Core Bug Fixes, Anti-Lookahead Hardening & Neural Quantile Design

### Project State
The original codebase suffered from critical target lookahead bias (predicting `Close[t]` with `Close[t]`), crossing quantiles ($P_{10} > P_{90}$), ghost row leakage in Ichimoku indicators, and broken imports. Release `v1.0.0` stabilized the foundation.

### Key Features Added
1. **Target Lookahead Elimination**:
   - Shifted targets by -1 (`df["Close"].shift(-1)`), establishing strict zero-lookahead integrity.
2. **Softplus Monotonic Quantile Network**:
   - Reformulated LSTM heads to output softplus offsets from the median, mathematically preventing quantile crossing.
3. **Data Pipeline Hardening**:
   - Fixed Ichimoku forward projection indexing; added multi-asset watchlist scanning and VADER sentiment fallback.
4. **Health Diagnostics**:
   - Integrated GPU RTX 4070 VRAM, CUDA 12.4, and model directory verification.

### Pros & Cons
- **Pros**:
  - Eliminated dangerous illusions of profitability caused by lookahead bias.
  - Replaced broken independent quantile regressors with a mathematically unified model.
- **Cons & Trade-offs**:
  - Target forward-shifting reduces apparent raw training accuracy compared to flawed lookahead models, but reflects true out-of-sample performance.
