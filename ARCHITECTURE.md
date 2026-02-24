# TradeSense HFP — Architecture

> **Read this first** to understand the project structure, data flow, and design decisions.

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   Streamlit UI (app.py)                      │
│  Sidebar → Ticker/TF/Lookback → Fetch → Train → Predict    │
│  Charts: Candlestick + Confidence Cloud + Volume + Whales   │
│  Controls: Buy/Sell + Position Sizer + Equity Curve         │
├──────────────┬──────────────────┬───────────────────────────┤
│  DataLake    │   AIEngine       │   Gauntlet (Simulator)    │
│  data_feed   │   ai_engine      │   trader                  │
│  + Cache     │   LSTM + XGBoost │   Orders + Fills          │
│  + Watchlist │   + Meta Ridge   │   + Position Tracking     │
│  + Sentiment │   → P10/P50/P90  │   + Sharpe/Drawdown       │
└──────────────┴──────────────────┴───────────────────────────┘
```

## Module Map

| Module | File | Responsibility |
|--------|------|---------------|
| **DataLake** | `src/data/data_feed.py` | Fetch OHLCV (yfinance), engineer features (pandas-ta), sentiment (FinBERT/VADER), whale tracker, normalisation |
| **DataCache** | `src/data/cache.py` | SQLite TTL cache for API responses and price data |
| **Watchlists** | `src/data/watchlists.py` | Managed stock/ETF/crypto lists (CA + US + Crypto) |
| **AIEngine** | `src/models/ai_engine.py` | LSTM (PyTorch, attention, quantile heads) + XGBoost (3 quantile regressors) + Ridge meta-learner → P10/P50/P90 |
| **Gauntlet** | `src/models/trader.py` | Paper trading with slippage, commission, position sizing (fixed-risk + Kelly), backtesting |
| **Config** | `src/config.py` | Environment-based settings (API keys, engine params, paths) |
| **HealthCheck** | `src/utils/health.py` | GPU, model files, API keys, internet, memory, cache status |
| **Logger** | `src/utils/logger.py` | Shared logging with file + console output |

## Data Flow

```
User selects ticker → DataLake.fetch()
                    → DataLake.engineer_features()
                    → DataLake.normalise()
                    → AIEngine.train(features, target="Close")
                        ├─ LSTM sequences → P10/P50/P90
                        ├─ XGBoost flat → P10/P50/P90
                        └─ Ridge meta → final P10/P50/P90
                    → AIEngine.predict() → QuantilePrediction
                    → Gauntlet.submit_order() → Fill
                    → Gauntlet.mark_to_market() → Equity update
```

## Key Design Decisions

1. **Quantile Output** — Not a point estimate. Every prediction is a range (P10/P50/P90) with confidence.
2. **Stacking Ensemble** — LSTM captures temporal patterns, XGBoost captures feature correlations, Ridge blends them.
3. **GPU First** — LSTM + FinBERT on CUDA (RTX 4070). XGBoost on GPU when available.
4. **Realistic Simulation** — Gauntlet applies random slippage (0.01%–0.05%) and commission ($1 flat or 0.1%).
5. **Whale Tracker** — Custom volume anomaly detector (z-score > 2.0 on 20-bar rolling window).

## File Structure

```
Trading/
├── .agents/workflows/       # Agent runbooks (launch, develop)
├── src/
│   ├── config.py            # Central configuration
│   ├── data/
│   │   ├── data_feed.py     # DataLake (706 lines)
│   │   ├── cache.py         # SQLite TTL cache
│   │   └── watchlists.py    # Managed asset lists
│   ├── models/
│   │   ├── ai_engine.py     # LSTM+XGB+Ridge ensemble (981 lines)
│   │   └── trader.py        # Gauntlet simulator (687 lines)
│   ├── ui/
│   │   └── app.py           # Streamlit War Room
│   └── utils/
│       ├── logger.py        # Shared logging
│       └── health.py        # System diagnostics
├── data/                    # SQLite cache, paper portfolio (gitignored)
├── models/hfp/              # Saved model weights (gitignored)
├── logs/                    # Log files (gitignored)
├── requirements.txt         # Python dependencies
├── ARCHITECTURE.md          # This file
└── README.md                # Project overview
```

## Environment

- **Python**: 3.10+
- **CUDA**: 11.8 (RTX 4070)
- **OS**: Windows 11
- **Venv**: `C:\Python Virtual Environments\Personal Projects\TradeSense\`
- **Code**: `C:\Python Virtual Environments\Personal Projects\Trading\`
- **GitHub**: `uzair-aftab/tradesense` (branch: `overhaul/tradesense-best`)
