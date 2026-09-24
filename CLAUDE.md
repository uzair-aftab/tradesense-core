# TradeSense — Claude Code System Guide (`CLAUDE.md`)

This guide provides instructions and context for Anthropic Claude Code working on TradeSense.

## Project Summary
TradeSense is an institutional quantitative trading, probabilistic machine learning, and high-frequency market microstructure simulation platform.

## Environment & Commands
- **Python Executable**: `C:\Python Virtual Environments\Personal Projects\TradeSense\Scripts\python.exe`
- **Working Directory**: `c:\Python Virtual Environments\Personal Projects\Trading`
- **Run All Tests**:
  ```powershell
  & "C:\Python Virtual Environments\Personal Projects\TradeSense\Scripts\python.exe" -m pytest tests/ -v
  ```
- **Launch Live Streaming Cockpit**:
  ```powershell
  & "C:\Python Virtual Environments\Personal Projects\TradeSense\Scripts\python.exe" -m streamlit run src/ui/app.py
  ```

## Key Modules
- `src/data/data_feed.py`: `DataLake` data ingestion, Ichimoku cloud, Whale Z-score anomaly tracking, VADER/FinBERT sentiment.
- `src/data/live_feed.py`: `LiveMarketFeed` real-time streaming feed with Geometric Brownian Motion and Poisson Whale Jump-Diffusion.
- `src/models/ai_engine.py`: `AIEngine` PyTorch LSTM with Softplus monotonic quantiles + XGBoost monotonic quantile regressors.
- `src/models/strategies.py`: 5 Institutional Strategies (DeepQuantileSniper, InstitutionalWhaleFlow, VolatilityBreakout, StatisticalMeanReversion, EnsembleAlpha) + `StrategyArena` tournament runner.
- `src/models/trader.py`: `Gauntlet` institutional paper trading simulator with slippage, commissions, and intra-bar polarity stop execution.
- `src/models/sniper.py`: `SniperEngine` execution filters enforcing RRR >= 1.5x and Confidence >= 70%.
- `src/models/bot.py`: `AutonomousSniperBot` autonomous trading agent managing tick-by-tick order microstructure, OCO bracket orders, and dynamic ratcheting trailing stops.
- `src/ui/app.py`: Streamlit 1.53 Live Cockpit using `@st.fragment(run_every="2s")` for live streaming charts, Strategy Arena, Deep Learning Studio, and Market Radar.

## Coding Conventions
- Strictly enforce mathematical quantile monotonicity: $P_{10} \le P_{50} \le P_{90}$.
- Zero lookahead bias: Shift future targets (`df['Close'].shift(-1)`).
- Intra-bar candle polarity: Bullish bars ($O \to L \to H \to C$) vs Bearish bars ($O \to H \to L \to C$).
- Keep all unit tests passing 100% at all times.
- Update `AGENT.md`, `.cursorrules`, and `ARCHITECTURE.md` after any functional changes.
