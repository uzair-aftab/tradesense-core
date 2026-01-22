# TradeSense

> **Multi-Factor Trading Signal Scanner** - A sophisticated Python-based trading signal scanner that combines technical indicators, fundamental analysis, sentiment analysis (NLP/LLM), options flow, and ML predictions to generate high-confidence buy/sell signals.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-green.svg)](https://developer.nvidia.com/cuda-toolkit)

## 🎯 Features

- **Technical Analysis**: RSI, MACD, Bollinger Bands, Volume analysis, Support/Resistance
- **Fundamental Analysis**: P/E ratios, EPS beat/miss, Revenue beat/miss
- **Sentiment Analysis**: FinBERT NLP model, Reddit (WSB), StockTwits, Financial news
- **Options Flow**: Put/Call ratio analysis
- **Adaptive Thresholds**: Volatility-based dynamic threshold adjustment
- **Backtesting**: Historical performance validation with vectorbt
- **ML Predictions**: XGBoost and LSTM models for signal enhancement

## 🚀 Quick Start

### Prerequisites

- Python 3.12
- NVIDIA GPU with CUDA 12.4 support (recommended)
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/uzair-aftab/tradesense.git
cd tradesense

# Create virtual environment
py -3.12 -m venv venv
./venv/Scripts/Activate.ps1  # Windows PowerShell

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install dependencies
pip install -r requirements.txt

# Install dev dependencies (optional)
pip install -r requirements-dev.txt

# Copy environment template and add your API keys
cp .env.example .env
# Edit .env with your API keys
```

### Running the App

```bash
# Start Streamlit dashboard
streamlit run app/main.py

# Run backtesting
python -m src.backtesting.engine

# Serve documentation locally
mkdocs serve
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
├── app/                    # Streamlit UI
├── src/                    # Core source code
│   ├── data/              # Data fetchers & caching
│   ├── sentiment/         # NLP & sentiment analysis
│   ├── analysis/          # Technical & fundamental analysis
│   ├── signals/           # Signal generation
│   ├── backtesting/       # Backtest engine
│   └── ml/                # Machine learning models
├── docs/                   # MkDocs documentation
├── tests/                  # Unit & integration tests
└── data/                   # Local cache (gitignored)
```

## 📖 Documentation

Full documentation available at: [docs/](docs/)

- [Getting Started](docs/getting-started.md)
- [Configuration](docs/configuration.md)
- [API Reference](docs/api/reference.md)

## 🔧 Configuration

Create a `.env` file based on `.env.example`:

```env
# API Keys (all optional - free tiers available)
FINNHUB_API_KEY=your_key_here
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret

# Scanner Settings
SCAN_INTERVAL_SECONDS=300
BATCH_MODE=false
```

## 📈 Signal Thresholds (Default)

| Indicator | Buy Signal Condition |
|-----------|---------------------|
| RSI | ≤ 20 (oversold) |
| MACD | Bullish crossover |
| Bollinger | Price near lower band |
| Volume | > 20-day average |
| Put/Call | < 0.7 |
| P/E Ratio | Below industry average |

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. It is not financial advice. Always do your own research and consult with a qualified financial advisor before making investment decisions. Past performance does not guarantee future results.

---

*Built with ❤️ in Toronto, Canada*
