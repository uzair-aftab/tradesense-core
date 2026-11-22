# TradeSense v8 — Institutional Quantitative & Wealthsimple Predict Terminal

> **Institutional Wealthsimple Alpha, GPU FinBERT & Prediction Markets Platform** — Fast Slate-950 financial terminal combining an 8-Model PyTorch ML/DL Quant Zoo (Stocks, ETFs, CDRs, Crypto) with live **GPU `ProsusAI/finbert` + Finnhub Macro Intelligence** and a full **Wealthsimple Predict (Kalshi) Paper Trading Sandbox & Alpha Scanner**.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch CUDA](https://img.shields.io/badge/PyTorch-CUDA%20Accelerated-green.svg)](https://pytorch.org/)
[![FinBERT GPU](https://img.shields.io/badge/FinBERT-RTX%204070%20CUDA-purple.svg)](https://huggingface.co/ProsusAI/finbert)
[![TradingView](https://img.shields.io/badge/TradingView-Lightweight%20Charts%20v4-131722.svg)](https://www.tradingview.com/HTML5-stock-forex-bitcoin-charting-library/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🌐 Single Permanent Gateway (Share With Friends)

Access the live terminal from anywhere in the world using one permanent link:
```
https://uzair-aftab.github.io/tradesense-gateway/
```
- **Zero URL clutter**: Passwordless cookie-based authentication via `tradesense_session` (HMAC-SHA256, 90-day persistence).
- **One-click account switching**: Isolated virtual portfolios & `$500 USD` Predict Sandboxes for **Uzair** and **Waleed**, plus `+ Create New Account`.
- **Dynamic Gist Synchronization**: Launching `start_public_server.bat` starts the Cloudflare Quick Tunnel and updates the dedicated GitHub Gist (`950ecdcb5db925cc7d5323908c45ec27`) automatically.

---

## 🚀 Key Capabilities

### 1. GPU FinBERT + Finnhub Live Macro & Equities Wire (New in v8)
- **Local CUDA Transformer Scoring (`ProsusAI/finbert`)**: Runs tri-class `(positive, negative, neutral)` transformer inference in fp16 on RTX 4070 with SHA-256 headline caching and automatic VADER fallback.
- **Rate-Limited Finnhub REST v1 Client**: Pulls live market news (`general`, `forex`, `crypto`), symbol-mapped company news, and upcoming corporate earnings calendars (`/calendar/earnings`) cleanly within the free-tier 60 calls/min budget.
- **Polarity-Aware Macro Mapping**: Resolves semantic inversions on central-bank and inflation contracts (e.g., *"Hot CPI keeps Fed hawkish"* boosts **Fed Hike/Hold** contracts and lowers **Fed Cut** contracts).

### 2. Wealthsimple Predict (Kalshi) Simulator & Alpha Scanner (New in v8)
- **Live Kalshi v2 Public Feed**: Connects directly to Kalshi's unauthenticated public REST v2 (`*_dollars` / `*_fp` fixed-point schema), filtered to CIRO-permitted Canadian categories (**Economics**, **Financials**, **Climate & Weather** — zero sports/politics).
- **Macro Event Countdown Strip**: Tracks live schedules for **FOMC Fed Decisions**, **Bank of Canada**, **US CPI/PCE**, **Non-Farm Payrolls**, **GDP**, and corporate earnings.
- **Exact Fee & Micro-Stake Kelly Sizing**: Implements Kalshi's variable taker fee `ceil(100 * 0.07 * C * P * (1 - P)) / 100`, Bayesian log-odds fair probability estimation, TradingView probability charts (`0c–100c`), and `$1` / `$5` / `$10` / `$25` / `$50` micro-stake tickets.
- **Dedicated `$500 USD` Virtual Predict Sandbox**: Supplies the missing practice account for Wealthsimple Predict — supports fractional YES/NO fills, live mark-to-market early exits, `$1.00`/`$0.00` auto-settlement, optional `$5` Auto-Pilot (`>= +4%` Net EV), **AI vs. Market Brier Score calibration**, and 1-click Copy Tickets.

### 3. Institutional Slate-950 Terminal & 8-Model ML/DL Quant Zoo
- **TradingView Lightweight Charts v4**: Hardware-accelerated canvas with zero-latency zoom (`+`, `−`, `Reset`), windows (`30B–All`), horizons (`3B–20B`), and technical overlays (`EMA 9/21`, `VWAP`, `Bollinger Bands`, `S/R`, `Forecast Cone`).
- **8-Model Quant Zoo + Split Conformal CQR**: Evaluates `Attention-BiLSTM`, `Dilated TCN-ResNet`, `Multi-Head Transformer`, `XGBoost Quantile`, `ExtraTrees + HistGBDT`, `RBF-SVR + Bayesian Ridge`, `Kalman-GARCH`, and `NLP Catalyst Flow` with embargoed walk-forward validation ($p_{10} \le p_{50} \le p_{90}$).
- **Wealthsimple CAD Proxy Routing**: Automatically routes crypto signals to `$0`-commission TSX spot ETFs (`BTC-CAD` -> `BTCC.TO`, `ETH-CAD` -> `ETHX-B.TO`, `SOL-CAD` -> `SOLQ.TO`), slashing round-trip friction from 4.10% down to 0.08% inside TFSAs.

---

## 🛠️ Quickstart

### Local Development
```powershell
# Activate virtual environment
.\TradeSense\Scripts\Activate.ps1

# Run TradeSense terminal locally on http://127.0.0.1:8010
python run.py
```

### Public Gateway Launcher
```cmd
start_public_server.bat
```

### Running Complete Test Suite
```powershell
pytest tests/test_tradesense_v7.py tests/test_predictions_v8.py -v
```

---

## 📁 Repository Architecture

```
TradeSense/
├── bin/
│   └── cloudflared.exe             # Official Cloudflare tunnel binary
├── data/
│   ├── tradesense.db               # SQLite WAL database (Equity + Predict ledgers)
│   └── friendly_urls.json          # Dedicated Gist ID and gateway config
├── scripts/
│   └── setup_gateway.py            # GitHub Pages & Gist provisioning script
├── src/
│   ├── config.py                   # Env config (FINNHUB_API_KEY, optional FRED_API_KEY)
│   ├── data/
│   │   ├── data_feed.py            # DataLake (OHLCV caching, whale tracker, features)
│   │   ├── database.py             # DatabaseManager (Equity/Crypto + Predict tables)
│   │   ├── finbert_engine.py       # CUDA ProsusAI/finbert + macro polarity mapping
│   │   ├── finnhub_client.py       # Rate-limited Finnhub v1 + optional FRED hook
│   │   ├── market_intel.py         # Asset & macro topic news fusion + FinBERT batching
│   │   ├── prediction_feed.py      # Kalshi v2 fixed-point feed, orderbook & macro calendar
│   │   ├── profiles.py             # Multi-user profile management
│   │   └── ws_catalog.py           # Wealthsimple fee schedules & CAD proxy registry
│   ├── models/
│   │   ├── model_zoo.py            # 8-Model ML/DL Quant Zoo & Split Conformal CQR
│   │   ├── prediction_engine.py    # Binary Bayesian EV, Kalshi fee math & micro-Kelly
│   │   ├── prediction_bot.py       # $500 USD Predict Sandbox, Auto-Pilot & Brier auditor
│   │   ├── profit_engine.py        # AlphaScanner, PositionRiskSizer, TrailingStopRatchet
│   │   └── virtual_bot.py          # Autonomous equity/crypto paper execution engine
│   └── web/
│       ├── auth_cookie.py          # HMAC-SHA256 cookie session manager
│       ├── server.py               # FastAPI server (port 8010, 60s dual bot loop)
│       ├── static/js/
│       │   ├── app.js              # Tabs 1–3 controller
│       │   └── predictions.js      # Tab 4 Wealthsimple Predict & FinBERT controller
│       └── templates/
│           ├── login.html          # 1-Click passwordless account portal
│           └── index.html          # 4-Tab institutional financial terminal
├── tests/
│   ├── test_tradesense_v7.py       # Core ML/DL Quant Zoo & REST suite
│   └── test_predictions_v8.py      # FinBERT, Finnhub, Kalshi & Predict Sandbox suite
├── run.bat / run.py                # Local launchers
├── start_public_server.bat / .py   # Public gateway launchers
└── requirements.txt                # Pinned dependencies
```

---

## 🔒 Security & Privacy

- **Zero Unofficial Broker Scraping**: Wealthsimple has no public trading API; TradeSense respects Terms of Service via a simulated `PaperKalshiAdapter` seam + 1-click Copy Tickets.
- **Public Gateway**: `uzair-aftab/tradesense-gateway` contains only the client-side redirector (`index.html`).
- **Signed Cookies**: Sessions are signed locally with HMAC-SHA256 (`tradesense_session`).
