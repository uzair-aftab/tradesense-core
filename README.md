---
title: TradeSense — Institutional Quant & Wealthsimple Copier
emoji: 💎
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.53.0
app_file: src/ui/app.py
pinned: false
license: mit
---

# TradeSense v6

> **Institutional Wealthsimple Alpha & AI Execution Platform** — Combines an 8-Model PyTorch Deep Learning & Machine Learning Quant Zoo with Split Conformal Calibration, Quarter-Kelly risk sizing, Net-EV alpha scanning, and a passwordless multi-user glass portal for Wealthsimple traders.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch CUDA](https://img.shields.io/badge/PyTorch-CUDA%20Accelerated-green.svg)](https://pytorch.org/)
[![Wealthsimple Ready](https://img.shields.io/badge/Wealthsimple-TSX%20%7C%20NEO%20%7C%20US%20%7C%20Crypto-cyan.svg)](https://www.wealthsimple.com/)

---

## 🎯 What's New in v6

1. **Passwordless Multi-Account Profiles**:
   - 1-Click Glass Login Gate with dedicated profiles for **Uzair** and **Waleed** + `＋ Add Account` option.
   - Separate virtual ledgers, real Wealthsimple trade logs, and shareable URLs (`?user=uzair`, `?user=waleed`).

2. **Interactive Chart Manipulation Suite**:
   - Plotly Frosted-Glass Modebar with scroll-wheel zoom, box zoom, pan, trendline drawing, and snapshot export.
   - Inline Glass Toolbar with `➕` / `➖` zoom buttons, viewport presets (`30B`, `60B`, `120B`, `All`), forecast horizons (`3B`, `5B`, `10B`, `20B`), chart styles (`Candles`, `Heikin-Ashi`, `Line`), and technical overlays (`EMA 9/21`, `VWAP`, `Bollinger Bands`, `S&R Zones`, `Volume Bars`).
   - Dynamic Y-axis auto-scaling on visible bars.

3. **8-Model ML/DL Quant Zoo & Custom Combinations**:
   - **Attention-BiLSTM Net** (PyTorch Deep Learning Sequence Net)
   - **Dilated TCN-ResNet** (Causal Dilated Convolutions & Skip Connections)
   - **Multi-Head Transformer** (Temporal Self-Attention Transformer)
   - **XGBoost Quantile Regressor** (Gradient Boosted Decision Trees with feature importances)
   - **ExtraTrees + HistGBDT** (Randomized Quantile Forest)
   - **RBF-SVR + Bayesian Ridge** (Kernel Regression & Analytical Uncertainty)
   - **Kalman State-Space + GARCH** (Latent Drift Velocity & Volatility Clustering)
   - **NLP Catalyst + Whale Flow** (Multimodal News Sentiment & Order Flow Imbalance)
   - Presets (`Auto-Match DNA`, `Pure DL`, `Tree ML`, `All 8 Models`) or **Custom Model Selection** with `Walk-Forward`, `Ridge Stacking`, or `Uniform` blending.

4. **Profit-Maximizing Institutional Stack**:
   - **AlphaScanner**: Multi-asset Net Expected Value % ranking after Wealthsimple commission, spreads, and crypto fees.
   - **PositionRiskSizer**: Mathematically optimal Quarter-Kelly allocations with strict max risk-at-stop caps and portfolio heat guards.
   - **MarketHoursGuard**: Real-time Toronto market clock, limit-order execution shields, and Canadian TFSA/RRSP tax optimization advice.
   - **TrailingStopRatchet**: Automated Breakeven locks and Chandelier trailing stops.

5. **Verified End-to-End Wealthsimple Execution**:
   - Real HTML5 `navigator.clipboard` copy cards for Limit Entry, Take Profit, Stop Loss, and Quantity.
   - Fixed CAD proxy fee math (`BTCC.TO`, `ETHX-B.TO`, `NVDA.NE`) showing exact CAD saved.
   - Live logged Wealthsimple trades monitor with realized P&L accounting.

---

## 🚀 Quick Start

### Local Setup
```bash
# Clone the repository
git clone https://github.com/uzair-aftab/tradesense.git
cd tradesense

# Create and activate virtual environment
python -m venv venv
./venv/Scripts/Activate.ps1  # Windows PowerShell

# Install dependencies
pip install -r requirements.txt

# Run the app locally
streamlit run src/ui/app.py
```

### Free Live Shareable URL

#### Option A: 24/7 Permanent Cloud URL (Free)
1. **Streamlit Community Cloud**:
   - Push code to GitHub: `https://github.com/uzair-aftab/tradesense`
   - Go to [share.streamlit.io](https://share.streamlit.io), connect your repository, and select `src/ui/app.py`.
2. **Hugging Face Spaces (Recommended for Heavy ML/DL)**:
   - Create a free Space on [huggingface.co/new-space](https://huggingface.co/new-space) (16 GB RAM free tier).
   - Link to your GitHub repo or push remote. The included Hugging Face YAML header configures everything automatically.

#### Option B: Live Laptop GPU Streaming Tunnel
To share your local RTX 4070 GPU session with a friend in 3 seconds:
```powershell
./scripts/share_live_laptop.ps1
```
This automatically starts a secure Cloudflare Quick Tunnel and outputs a live temporary HTTPS URL (e.g. `https://xxxx.trycloudflare.com`) with zero port forwarding or accounts required!

---

## 🏛️ Project Architecture

```
tradesense/
├── .github/workflows/ci.yml       # Automated GitHub Actions CI pipeline
├── scripts/
│   └── share_live_laptop.ps1     # 1-click Cloudflare live tunnel for laptop
├── src/
│   ├── data/
│   │   ├── database.py           # SQLite WAL persistence (profiles, virtual bot, real WS trades)
│   │   ├── profiles.py           # Multi-account engine (Uzair, Waleed, +Add)
│   │   ├── ws_catalog.py         # 173+ Wealthsimple asset catalog & fee schedules
│   │   ├── data_feed.py          # DataLake OHLCV pipeline & whale volume tracker
│   │   └── market_intel.py       # VADER + FinBERT sentiment & news catalyst feed
│   ├── models/
│   │   ├── model_zoo.py          # 8-Model ML/DL Quant Zoo & Conformal Calibrator
│   │   ├── profit_engine.py      # AlphaScanner, Quarter-Kelly sizer, & trailing stop ratchet
│   │   ├── adaptive_predictor.py # Dynamic heuristic consensus engine
│   │   ├── regime.py             # Market regime classifier (Bull, Ranging, Bear)
│   │   └── virtual_bot.py        # Proof-of-edge paper trading engine
│   ├── ui/
│   │   ├── app.py                # Obsidian frosted-glass cockpit & profile gate
│   │   ├── chart_builder.py      # Plotly modebar, scroll zoom, & multi-indicator builder
│   │   └── clipboard.py          # HTML5 1-click clipboard copy cards
│   └── utils/
│       ├── health.py             # Live GPU/CPU & RAM telemetry
│       └── logger.py             # Structured logger
├── tests/
│   ├── test_overhaul_v5.py       # Core regression tests
│   └── test_overhaul_v6.py       # v6 profiles, 8 models, profit engine, & zero-stub tests
└── requirements.txt              # Cloud & CUDA ready dependencies
```

---

## ⚖️ License
MIT License. TradeSense is built for educational, quantitative research, and personal algorithmic trading purposes.
