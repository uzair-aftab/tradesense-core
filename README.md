# TradeSense v7 — Institutional Quantitative Terminal

> **Institutional Wealthsimple Alpha & Autonomous Execution Platform** — Fast, robust Slate-950 financial terminal powered by FastAPI, TradingView Lightweight Charts v4, an 8-Model PyTorch Deep Learning & Machine Learning Quant Zoo with Split Conformal CQR calibration, Quarter-Kelly position sizing, CAD proxy execution routing, and a single permanent GitHub Pages gateway.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch CUDA](https://img.shields.io/badge/PyTorch-CUDA%20Accelerated-green.svg)](https://pytorch.org/)
[![TradingView](https://img.shields.io/badge/TradingView-Lightweight%20Charts%20v4-131722.svg)](https://www.tradingview.com/HTML5-stock-forex-bitcoin-charting-library/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🌐 Single Permanent Gateway (Share With Friends)

Access the live terminal from anywhere in the world using one permanent link:
```
https://uzair-aftab.github.io/tradesense-gateway/
```
- **Zero URL clutter**: Passwordless cookie-based authentication via `tradesense_session` (HMAC-SHA256, 90-day persistence). No `?user=uzair` or `?user=waleed` query string contamination.
- **One-click account switching**: Dedicated isolated virtual portfolios for **Uzair** and **Waleed**, plus a 1-click `+ Create New Account` card.
- **Dynamic Gist Synchronization**: When the host laptop starts `start_public_server.bat`, the Cloudflare Quick Tunnel automatically updates the dedicated GitHub Gist (`950ecdcb5db925cc7d5323908c45ec27`), transparently routing the GitHub Pages gateway directly to the live server.

---

## 🚀 Key Capabilities in v7

### 1. Institutional Slate-950 Financial Terminal
- **Design Philosophy**: High-density, clutter-free institutional styling matching `pr-plan` (`#020617` / `#0f172a`, `1px solid #1e293b`, `tabular-nums` monospace numbers). Strict zero fake 3D frosted-glass clutter.
- **TradingView Lightweight Charts v4**: 60 FPS hardware-accelerated canvas with zero-latency client-side zoom controls (`+`, `−`, `Reset`), historical window buttons (`30B`, `60B`, `120B`, `All`), forecast horizons (`3B`, `5B`, `10B`, `20B`), and instant toggleable overlays (`EMA 9/21`, `VWAP`, `Bollinger Bands`, `S/R Levels`, `Forecast Cone`, `Order Lines`).
- **1-Click Copy Buttons**: Instant clipboard copying for Limit Entry, Take Profit (P90), Stop Loss (P10), and Capital Allocation to quickly paste directly into the Wealthsimple mobile or web app.

### 2. 8-Model ML/DL Quant Zoo (Zero Target Leakage)
- **Deep Learning Sequence Models**:
  - `Attention-BiLSTM Net`: Bidirectional recurrent network with temporal self-attention for sequential momentum.
  - `Dilated TCN-ResNet`: Causal dilated temporal convolutions with residual skip connections and strict left-only causal padding (`Chomp1d`).
  - `Multi-Head Transformer`: Multi-head self-attention with learned positional encodings.
- **Machine Learning & Tree Ensembles**:
  - `XGBoost Quantile Regressor`: Split Conformal pinball loss regressor with top feature attribution.
  - `ExtraTrees + HistGBDT`: Randomized decision tree forest with high overfit resistance on choppy TSX equities.
  - `RBF-SVR + Bayesian Ridge`: Radial basis support vector regression with Bayesian ridge parameter shrinkage.
- **Econometric & Multimodal Models**:
  - `Kalman State-Space + GARCH`: Latent drift velocity filtering with GARCH(1,1) dynamic volatility clustering.
  - `NLP Catalyst + Whale Flow`: FinBERT/VADER news sentiment flow coupled with order flow volume anomaly tracking.
- **Walk-Forward Validation**: Embargoed rolling validation ensuring strictly zero lookahead bias. All 8 models are evaluated on every pass (zero `"N/A"` rows).
- **Split Conformal Calibration**: Calibrated prediction intervals guaranteeing $p_{10} \le p_{50} \le p_{90}$.

### 3. Wealthsimple Friction & CAD Proxy Routing
- **Exact Fee Accounting**:
  - Raw Crypto: 2.0% one-way (4.10% round-trip hurdle).
  - Canadian TSX Stocks & ETFs: $0 commission (0.08% estimated round-trip spread).
  - US Stocks (CAD Account): 1.5% FX + 1.5% FX (3.08% round-trip hurdle).
- **Smart CAD ETF Proxy Routing**: Automatically routes crypto inquiries to zero-commission TSX spot ETFs (e.g. `BTC-CAD` -> `BTCC.TO`, `ETH-CAD` -> `ETHX-B.TO`, `SOL-CAD` -> `SOLQ.TO`), saving up to **\$40+ CAD per \$1,000 traded** and enabling 100% tax-free TFSA execution.

### 4. Autonomous Virtual Trading Bot
- **Autonomous 60-Second Background Cycle**: Evaluates open positions and monitors new entry opportunities every 60 seconds without blocking API requests.
- **Trailing Stop Ratchet**: Auto-locks breakeven after 1R gain and ratchets stops trailing P90 targets.
- **Walk-Forward Efficiency Scoring**: Tracks win rate, profit factor, and net return after exact Wealthsimple deduction over a 60-bar historical window.

---

## 🛠️ Quickstart

### Prerequisites
- Python 3.10+ (tested on Python 3.12 with PyTorch CUDA)
- GitHub CLI (`gh`) authenticated (for public gateway synchronization)

### Local Development
```powershell
# Activate virtual environment
.\TradeSense\Scripts\Activate.ps1

# Run TradeSense terminal locally on http://127.0.0.1:8010
python run.py
```

### Public Gateway Launcher
```cmd
# Double click start_public_server.bat or run from terminal:
start_public_server.bat
```
This starts:
1. TradeSense FastAPI server on `http://127.0.0.1:8010`
2. Official Cloudflare HTTPS Quick Tunnel (`bin/cloudflared.exe`)
3. Synchronizes the live tunnel URL to the dedicated GitHub Gist
4. Surfaces the single permanent URL: `https://uzair-aftab.github.io/tradesense-gateway/`

### Running Test Suite
```powershell
pytest tests/test_tradesense_v7.py -v
```

---

## 📁 Repository Architecture

```
TradeSense/
├── bin/
│   └── cloudflared.exe            # Official Cloudflare tunnel binary
├── data/
│   ├── tradesense.db              # SQLite persistent database (wal mode)
│   └── friendly_urls.json         # Dedicated Gist ID and gateway config
├── scripts/
│   └── setup_gateway.py           # GitHub Pages & Gist provisioning script
├── src/
│   ├── config.py                  # Environment & engine configuration
│   ├── data/
│   │   ├── data_feed.py           # DataLake (caching, whale tracker, features)
│   │   ├── database.py            # DatabaseManager (migrations, positions, ledger)
│   │   ├── market_intel.py        # News scraping, FinBERT/VADER sentiment
│   │   ├── profiles.py            # Multi-user profile management
│   │   └── ws_catalog.py          # Wealthsimple fee schedules & CAD proxy registry
│   ├── models/
│   │   ├── model_zoo.py           # 8-Model ML/DL Quant Zoo & Split Conformal CQR
│   │   ├── profit_engine.py       # AlphaScanner, PositionRiskSizer, TrailingStopRatchet
│   │   └── virtual_bot.py         # Autonomous paper execution & efficiency engine
│   ├── utils/
│   │   ├── health.py              # System health telemetry
│   │   └── logger.py              # Structured logging
│   └── web/
│       ├── auth_cookie.py         # HMAC-SHA256 cookie session manager
│       ├── server.py              # FastAPI server (port 8010, lifespan background bot)
│       ├── static/
│       │   ├── css/styles.css     # Institutional Slate-950 dark theme
│       │   ├── js/app.js          # TradingView Lightweight Charts v4 controller
│       │   └── img/favicon.svg    # Terminal vector mark
│       └── templates/
│           ├── login.html         # 1-Click passwordless account portal
│           └── index.html         # 3-Tab institutional terminal
├── tests/
│   └── test_tradesense_v7.py      # Full verification test suite
├── run.bat / run.py               # Local launchers
├── start_public_server.bat / .py  # Public gateway launchers
└── requirements.txt               # Pinned dependencies
```

---

## 🔒 Security & Privacy

- **Private Repository**: Main codebase is maintained in a private repository (`uzair-aftab/tradesense-core`).
- **Public Gateway**: The public GitHub Pages repository (`uzair-aftab/trade-sense`) contains only the client-side gateway redirector (`index.html`) reading the public Gist.
- **Zero Authentication Credential Leaks**: Session cookies are signed with a local HMAC-SHA256 secret.
