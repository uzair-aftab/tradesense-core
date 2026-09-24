# TradeSense Institutional System — Master AI Agent Operating Manual (`AGENT.md`)

> **Target Audience**: Autonomous Agentic Coding AIs (Google Antigravity, Anthropic Claude Code, Cursor, Cline, Devin, SWE-bench) and Quantitative Engineers.
> **Current Version**: `v2.1.1-institutional`  
> **Platform**: Windows / Python 3.12+ (Virtualenv: `c:\Python Virtual Environments\Personal Projects\TradeSense`)  
> **Repository Path**: `c:\Python Virtual Environments\Personal Projects\Trading`  
> **Hardware Acceleration**: NVIDIA GeForce RTX 4070 Laptop GPU (CUDA 12.4 / PyTorch 2.6.0)

---

## 1. System Vision & Architecture

TradeSense is an institutional-grade algorithmic trading, probabilistic machine learning, and market microstructure simulation platform. It transforms retail execution into hedge-fund caliber operations by eliminating data leakage, enforcing mathematical quantile monotonicity, simulating realistic order flow dynamics, and automating execution via an autonomous sniper bot.

### Core Architecture Pipeline
```
[Market Data Sources] (yfinance / Finnhub / SQLite Cache)
         │
         ▼
[DataLake & Feature Engine] (Ichimoku, Whale Z-scores, RSI, FinBERT/VADER)
         │
         ├─────────────────────────────────────────┐
         ▼                                         ▼
[AIEngine (Probabilistic Quantiles)]    [LiveMarketFeed & Microstructure]
  • Softplus Monotonic LSTM              • Synthetic Jump-Diffusion Whale Flow
  • Monotonic Gradient Boosted Trees     • Forming Candle Aggregation (5 ticks/bar)
  • Calibrated Stacking Meta-Learner     • Continuous Candle Opens (Open = Prev Close)
         │                               • Realistic Volume Scaling (Crypto vs Equities)
         │                                         │
         └────────────────────┬────────────────────┘
                              ▼
                 [Strategy Tournament Arena]
                   1. DeepQuantileSniper (RRR >= 1.5x, Conf >= 70%)
                   2. InstitutionalWhaleFlow (VWAP Anchoring)
                   3. VolatilityBreakout (Donchian Expansion)
                   4. StatisticalMeanReversion (2.5-Sigma Extremes)
                   5. EnsembleAlpha (Meta-Consensus Voting)
                              │
                              ▼
                 [Autonomous Sniper Bot]
                   • Live tick-by-tick evaluation
                   • Fractional position sizing (Crypto / Equities)
                   • Dynamic ratcheting trailing stops
                   • Intra-bar polarity execution
                              │
                              ▼
                 [Gauntlet Execution Simulator]
                   • Float-precision order fills & position tracking
                   • Slippage & Commission modeling
                   • Mark-to-market PnL tracking
                              │
                              ▼
                 [Streamlit Live Cockpit (@st.fragment)]
                   • Category x-axis live charts with collision-free labels
                   • USD Amount & Fractional unit order entry
                   • Graceful cash validation (no unhandled tracebacks)
```

---

## 2. Fundamental Domain Invariants (NEVER VIOLATE)

1. **Zero Lookahead Bias in Target Construction**:
   - In financial time-series forecasting, `Close[t]` cannot predict `Close[t]`!
   - Target must ALWAYS be shifted: `target = df["Close"].shift(-1)`.
   - Never compute future rolling metrics using data after bar $t$.

2. **Quantile Monotonicity Invariance ($P_{10} \le P_{50} \le P_{90}$)**:
   - Quantile crossing is a fatal mathematical defect.
   - The neural network architecture MUST output:
     $$y_{50} = \text{base}, \quad y_{10} = y_{50} - \text{softplus}(\Delta_{10}), \quad y_{90} = y_{50} + \text{softplus}(\Delta_{90})$$
   - Monotonicity rate must verify at 100.0% on every build.

3. **Fractional Position Sizing for Multi-Asset Universes**:
   - High-priced assets (e.g. Bitcoin at $80,000+) MUST support fractional quantities (`float`).
   - Never cast order quantity to `int()` without a minimum fraction fallback, as `int(10000 / 80000)` truncates to `0`.
   - Orders must be sized against available cash, supporting both USD capital allocation and fractional units.

4. **Continuous Candle Aggregation**:
   - Ticks must update the active forming candle.
   - When a bar completes, the new bar MUST open at `Open = Previous_Bar.Close` to prevent artificial disconnected gaps.

5. **Intra-Bar Candle Polarity Execution Order**:
   - Never evaluate High and Low simultaneously without respecting candle polarity:
     - **Bullish Bar** ($Close \ge Open$): Order is $Open \to Low \to High \to Close$. Evaluate stop-loss against the Low *before* evaluating take-profit against the High.
     - **Bearish Bar** ($Close < Open$): Order is $Open \to High \to Low \to Close$. Evaluate take-profit/ratchet stop against the High *before* evaluating stop-loss against the Low.

6. **Dynamic Trailing Stop Ratchet**:
   - Trailing stops can only ever ratchet *in the profitable direction*. Once increased, a trailing stop floor must NEVER decrease.

7. **Asymmetric Risk-to-Reward Filter**:
   - Sniper execution requires: $\text{Reward-to-Risk Ratio} \ge 1.5\text{x}$ and $\text{Strategy Confidence} \ge 70\%$.

8. **Non-Blocking UI Streaming**:
   - Never use full-page `st.rerun()` loops for live streaming.
   - Use `@st.fragment(run_every="2s")` so the live market updates smoothly without interfering with sidebar widgets or tab inputs.

9. **Category Axis Label Uniqueness in Plotly**:
   - In Plotly candlestick charts configured with `type="category"`, bars sharing an identical x-axis string label will collapse into a single bin, stacking OHLC bodies and summing their volume into a trillion-dollar monolith.
   - Labels must be context-aware (`%b %d` for daily bars, `%H:%M:%S` for intraday bars) and guaranteed unique via `seen` tracking (`#2`, `#3`) to prevent collapse.

10. **Signal Target/Stop Price Resolution**:
    - Any strategy or meta-ensemble generating dynamic target or stop orders must always resolve the reference price from the active dataframe (`price = float(df["Close"].iloc[current_idx])`) before evaluating fallback percentages to prevent `NameError` crashes.

---

## 3. Environment & Key Commands

### Virtual Environment
- **Path**: `C:\Python Virtual Environments\Personal Projects\TradeSense\Scripts\python.exe`
- **Shell**: PowerShell (Windows)

### Essential Commands
```powershell
# Run full unit test suite (32 tests)
& "C:\Python Virtual Environments\Personal Projects\TradeSense\Scripts\python.exe" -m pytest tests/ -v

# Run targeted test file
& "C:\Python Virtual Environments\Personal Projects\TradeSense\Scripts\python.exe" -m pytest tests/test_bot.py -v

# Launch Live Cockpit Dashboard
& "C:\Python Virtual Environments\Personal Projects\TradeSense\Scripts\python.exe" -m streamlit run src/ui/app.py

# Check GPU CUDA availability
& "C:\Python Virtual Environments\Personal Projects\TradeSense\Scripts\python.exe" -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 4. Repository Structure & Module Responsibilities

```
Trading/
├── AGENT.md                       # This file (Master AI Agent operating manual)
├── .cursorrules                   # Cursor IDE rules and formatting standards
├── CLAUDE.md                      # Anthropic Claude Code developer manual
├── ARCHITECTURE.md                # System technical architecture and mathematical proofs
├── RELEASES.md                    # Release history, version states, pros and cons
├── .env.example                   # API keys template (Finnhub, AlphaVantage)
├── .gitignore                     # Git ignore rules (protects large model checkpoints)
├── requirements.txt               # Production dependencies
├── src/
│   ├── config.py                  # Global application constants and paths
│   ├── data/
│   │   ├── cache.py               # SQLite TTL query cache
│   │   ├── data_feed.py           # DataLake: yfinance ingestion, Ichimoku, Whale Z-scores
│   │   ├── live_feed.py           # LiveMarketFeed: Poisson Jump-Diffusion & Historical Replay
│   │   └── watchlists.py          # Multi-asset taxonomy (Crypto, Tech, ETFs, Forex)
│   ├── models/
│   │   ├── ai_engine.py           # Softplus Quantile LSTM, Monotonic XGBoost, Stacking
│   │   ├── trader.py              # Gauntlet: Slippage, Intra-bar execution, Trailing stops
│   │   ├── sniper.py              # SniperEngine: Institutional RRR & Confidence filtering
│   │   ├── strategies.py          # 5 Hedge-Fund Quant Strategies & StrategyArena tournament
│   │   └── bot.py                 # AutonomousSniperBot: Live tick engine & OCO brackets
│   ├── ui/
│   │   └── app.py                 # Streamlit Live Trading Cockpit with @st.fragment
│   └── utils/
│       ├── health.py              # Diagnostics: GPU VRAM, CUDA, Net, Model check
│       └── logger.py              # Structured micro-second logging
└── tests/
    ├── test_ai_engine.py          # Quantile monotonicity, lookahead, save/reload
    ├── test_bot.py                # Autonomous bot lifecycle, bracket execution, trailing stop
    ├── test_data_feed.py          # Feature row integrity, ghost rows, whale volume
    ├── test_health.py             # Hardware health checks
    ├── test_sniper.py             # Intra-bar stop loss, take profit, RRR rejection
    └── test_trader.py             # Gauntlet paper trading, mark-to-market, position sizing
```

---

## 5. Agent Workflow & Updating Instructions

When making modifications as an agentic AI:
1. **Never break existing tests**: Run `pytest tests/ -v` before and after any changes.
2. **Update Context Files**: When features, models, or configurations are updated, immediately reflect the change in `AGENT.md`, `.cursorrules`, and `ARCHITECTURE.md`.
3. **Commit Cleanly**: Record semantic commits with clear explanations of changes, state of the project, and architectural trade-offs.
4. **Clean up**: Never leave temporary debug scripts or unreferenced files in the repository root.
