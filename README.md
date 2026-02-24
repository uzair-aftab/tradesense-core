# TradeSense HFP

> **High-Frequency Prediction & Paper Trading System** — An institutional-grade trading platform combining LSTM + XGBoost ensemble predictions with quantile output (P10/P50/P90), paper trading with realistic market friction, and GPU-accelerated sentiment analysis.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA](https://img.shields.io/badge/CUDA-11.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)

## 🎯 Features

- **Hybrid AI Engine** — LSTM (temporal patterns) + XGBoost (feature importance) → Ridge meta-learner → P10/P50/P90 quantile output
- **GPU Accelerated** — LSTM + FinBERT on CUDA (RTX 4070), XGBoost GPU training
- **Paper Trading** — "The Gauntlet" simulator with slippage, commission, stop-loss, take-profit
- **Whale Tracker** — Custom volume anomaly detector (z-score on rolling window)
- **Sentiment Analysis** — FinBERT (GPU) with VADER fallback, normalised to [-1, 1]
- **Data Caching** — SQLite TTL cache for API responses, eliminates redundant fetches
- **Portfolio Persistence** — JSON state saves survive app restarts

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA 11.8 support (recommended)
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/uzair-aftab/tradesense.git
cd tradesense

# Create virtual environment
python -m venv venv
./venv/Scripts/Activate.ps1  # Windows PowerShell

# Install PyTorch with CUDA support (FIRST)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Copy environment template and add your API keys
cp .env.example .env
```

### Running the App

```bash
streamlit run src/ui/app.py
```

## 📊 Watchlists

Default watchlists (5 per category, expandable):

| Category | Tickers |
|----------|---------|
| Canadian Stocks | RY.TO, TD.TO, SHOP.TO, ENB.TO, CNR.TO |
| US Stocks | AAPL, MSFT, NVDA, GOOGL, AMZN |
| ETFs | SPY, QQQ, XIU.TO, VFV.TO, ARKK |
| Crypto (CAD) | BTC-CAD, ETH-CAD, SOL-CAD, ADA-CAD, AVAX-CAD |

## 🛠️ Project Structure

```
tradesense/
├── .agents/workflows/       # Agent runbooks (launch, develop)
├── src/
│   ├── config.py            # Central configuration
│   ├── data/
│   │   ├── data_feed.py     # DataLake — OHLCV + features + sentiment
│   │   ├── cache.py         # SQLite TTL cache
│   │   └── watchlists.py    # Managed asset lists
│   ├── models/
│   │   ├── ai_engine.py     # LSTM+XGB+Ridge ensemble
│   │   └── trader.py        # Gauntlet paper trading simulator
│   ├── ui/
│   │   └── app.py           # Streamlit War Room dashboard
│   └── utils/
│       ├── logger.py        # Shared logging
│       └── health.py        # System diagnostics
├── data/                    # SQLite cache, paper portfolio (gitignored)
├── models/hfp/              # Saved model weights (gitignored)
├── logs/                    # Log files (gitignored)
├── ARCHITECTURE.md          # System architecture reference
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## 🔧 Configuration

Create a `.env` file based on `.env.example`:

```env
# API Keys (all optional - free tiers available)
FINNHUB_API_KEY=your_key_here
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret

# Engine Defaults
LSTM_EPOCHS=50
XGB_ESTIMATORS=500
STARTING_CAPITAL=10000
SENTIMENT_BACKEND=vader
```

## 📈 How It Works

1. **Fetch** — Pull OHLCV data from yfinance (cached in SQLite)
2. **Engineer** — Compute 20+ technical features (ADX, MACD, Ichimoku, ATR, BBWidth, OBV, VWAP, Whale Tracker)
3. **Sentiment** — Score headlines via FinBERT (GPU) or VADER
4. **Train** — LSTM (sequential patterns) + XGBoost (cross-sectional) → Ridge meta-blend
5. **Predict** — Quantile output: P10 (conservative), P50 (likely), P90 (aggressive)
6. **Trade** — Paper trade with realistic slippage, stop-loss, take-profit

## ⚠️ Disclaimer

This software is for educational and research purposes only. It is not financial advice. Always do your own research and consult with a qualified financial advisor before making investment decisions.

---

*Built with ❤️ in Toronto, Canada*
