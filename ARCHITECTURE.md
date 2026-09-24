# TradeSense Institutional System — Technical Architecture (`ARCHITECTURE.md`)

> **Version**: `v2.0.0-institutional`  
> **Classification**: Institutional Algorithmic Trading & Probabilistic Quant System  
> **Hardware Support**: NVIDIA GeForce RTX 4070 (CUDA 12.4 / PyTorch 2.6.0)

---

## 1. Executive System Topology

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        TradeSense Live Trading Cockpit (src/ui/app.py)                │
│                                                                                        │
│   ┌────────────────────────────────────────────────────────────────────────────────┐   │
│   │  @st.fragment(run_every="2s") Live Stream Engine                               │   │
│   │   • Live Ticker Ribbon (Flashing Price, Bid/Ask, Whale Alert, Portfolio PnL)   │   │
│   │   • Real-Time Candlestick + Quantile Cloud (P10 / P50 / P90) + Dynamic Stops   │   │
│   │   • Microstructure Execution Log (Order events, Whale spikes, Trailing stops)  │   │
│   │   • One-Click Bracket Order Cockpit & Open Positions Monitor                   │   │
│   └──────────────────────────────────────┬─────────────────────────────────────────┘   │
│                                          │                                             │
│   ┌────────────────────────┬─────────────┴───────────────┬─────────────────────────┐   │
│   │ ⚔️ Strategy Arena      │ 🧪 Deep Learning Studio      │ 📡 Market Radar         │   │
│   │  • 5 Quant Algorithms  │  • Real & Simulated Training │  • Multi-Asset Scanner  │   │
│   │  • Leaderboard Podium  │  • Pinball Loss Convergence  │  • Whale Order Anomalies│   │
│   │  • Head-to-Head Equity │  • Monotonicity Verification │  • 1-Click Cockpit Load │   │
│   └────────────────────────┴─────────────────────────────┴─────────────────────────┘   │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
       ┌───────────────────────────────────┼───────────────────────────────────┐
       ▼                                   ▼                                   ▼
┌───────────────────────┐       ┌───────────────────────┐       ┌───────────────────────┐
│ LiveMarketFeed        │       │ AutonomousSniperBot   │       │ AIEngine              │
│ (src/data/live_feed)  │       │ (src/models/bot)      │       │ (src/models/ai_engine)│
│ • Jump-Diffusion GBM  │◄──────┤ • Real-time tick loop │◄──────┤ • Softplus LSTM       │
│ • Poisson Whale Jumps │       │ • Bracket orders      │       │ • Monotonic XGBoost   │
│ • Historical Replay   │       │ • Trailing stop floor │       │ • Stacking Meta-Model │
└───────────────────────┘       └──────────┬────────────┘       └───────────────────────┘
                                           │
                                           ▼
                                ┌───────────────────────┐
                                │ Gauntlet Simulator    │
                                │ (src/models/trader)   │
                                │ • Intra-bar polarity  │
                                │ • Slippage & comms    │
                                │ • Mark-to-market PnL  │
                                └───────────────────────┘
```

---

## 2. Core Mathematical & Algorithmic Formulations

### 2.1 Target Construction & Elimination of Lookahead Bias
In classical naive implementations, predicting $Close_t$ with features computed up to bar $t$ creates severe data leakage. In TradeSense, all training targets are rigorously forward-shifted:
$$y_t = \text{Close}_{t+1}$$
All technical indicators (RSI, Ichimoku, Bollinger Bands, Whale Z-scores) computed at index $t$ utilize strictly information available at or before timestamp $t$.

### 2.2 Mathematical Quantile Monotonicity ($P_{10} \le P_{50} \le P_{90}$)
Standard independent quantile regressors frequently suffer from "quantile crossing," where $P_{10} > P_{50}$ during high volatility regimes, destroying probabilistic validity. TradeSense enforces monotonicity via a single forward-pass softplus offset parameterization:

$$\hat{y}_{50} = f_{\text{median}}(h)$$
$$\hat{y}_{10} = \hat{y}_{50} - \text{softplus}(\Delta_{10}(h))$$
$$\hat{y}_{90} = \hat{y}_{50} + \text{softplus}(\Delta_{90}(h))$$

Because $\text{softplus}(x) = \ln(1 + e^x) > 0$ for all $x \in \mathbb{R}$, this mathematically guarantees:
$$\hat{y}_{10} < \hat{y}_{50} < \hat{y}_{90} \quad \forall x$$

The models are jointly optimized using the Asymmetric Pinball (Tilted) Loss:
$$\mathcal{L}_q(y, \hat{y}) = \max(q(y - \hat{y}), (1 - q)(\hat{y} - y))$$
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{0.10}(y, \hat{y}_{10}) + \mathcal{L}_{0.50}(y, \hat{y}_{50}) + \mathcal{L}_{0.90}(y, \hat{y}_{90})$$

### 2.3 Market Microstructure & Poisson Jump-Diffusion Simulation
The `LiveMarketFeed` implements a continuous-time jump-diffusion stochastic differential equation (Merton Jump-Diffusion):

$$dS_t = \mu S_t dt + \sigma S_t dW_t + S_t dJ_t$$

Where:
- $dW_t \sim \mathcal{N}(0, dt)$ represents standard Brownian market noise.
- $dJ_t = \sum_{i=1}^{N_t} (Y_i - 1)$ represents institutional Whale block trades governed by a Poisson process $N_t$ with arrival intensity $\lambda = 0.08$.
- When $dN_t = 1$, a block trade occurs with random log-magnitude $Y_i \sim \mathcal{N}(\mu_J, \sigma_J^2)$ producing an institutional order-flow spike with volume Z-score $Z > 2.5$.

### 2.4 Realistic Intra-Bar Candle Polarity Execution
Evaluating stop-loss and take-profit levels using only closing prices overlooks intra-bar price penetrations. TradeSense simulates intra-bar trajectory by respecting candle polarity:

- **Bullish Bar ($Close \ge Open$)**:
  $$\text{Trajectory}: Open \longrightarrow Low \longrightarrow High \longrightarrow Close$$
  1. Test $Low$ against effective stop-loss.
  2. If stop not hit, ratchet trailing stop to $High$.
  3. Test $High$ against take-profit target.
  4. Settle at $Close$.

- **Bearish Bar ($Close < Open$)**:
  $$\text{Trajectory}: Open \longrightarrow High \longrightarrow Low \longrightarrow Close$$
  1. Test $High$ against take-profit target and ratchet trailing stop.
  2. If target not hit, test $Low$ against effective stop-loss.
  3. Settle at $Close$.

### 2.5 Dynamic Ratcheting Trailing Stop Algorithm
For an open long position at fill price $P_{\text{entry}}$ and trailing percentage $\tau$:
$$P_{\text{peak}}(t) = \max_{0 \le s \le t} P_s$$
$$\text{StopFloor}(t) = \max\left(\text{StopFloor}(t-1), P_{\text{peak}}(t) \cdot (1 - \tau)\right)$$

This guarantees that profit is systematically locked in as price advances, and the stop floor never retreats.

---

## 3. Institutional Multi-Strategy Suite

TradeSense provides five institutional quant strategies tested in the `StrategyArena`:

| Strategy | Engine Class | Primary Alpha Drivers | Entry Criteria |
|---|---|---|---|
| **Deep Quantile Sniper** | `DeepQuantileSniper` | Neural quantile asymmetry | Direction = BULLISH, Conf $\ge 70\%$, RRR $\ge 1.5$x |
| **Institutional Whale Flow** | `InstitutionalWhaleFlow` | Block trade volume + VWAP | Volume $Z \ge 2.0$, Close above VWAP |
| **Volatility Breakout** | `VolatilityBreakout` | Donchian expansion + ADX | Close $>$ Upper Donchian (20), ADX $> 25$ |
| **Statistical Mean Reversion** | `StatisticalMeanReversion` | Bollinger 2.5-Sigma + RSI | Close $<$ Lower Band (20, 2.5), RSI $< 30$ |
| **Ensemble Alpha** | `EnsembleAlpha` | Meta-consensus voting | Weighted consensus score $\ge 0.50$ |

---

## 4. UI Live Streaming Architecture (`@st.fragment`)

Streamlit traditionally reruns the entire Python script upon any user interaction or timer event. In TradeSense v2.0:
1. The live market ticker, candlestick chart, order book, and bot terminal are isolated inside `@st.fragment(run_every="2s")`.
2. Every 2 seconds, the fragment executes:
   - `feed.next_tick()` $\to$ updates prices and detects whale spikes.
   - `gauntlet.mark_to_market()` $\to$ updates positions and ratchets trailing stops.
   - `bot.step()` $\to$ evaluates quant strategies and executes bracket orders.
   - Redraws the Plotly candlestick chart and live telemetry ribbon.
3. The rest of the page (Sidebar, Tournament Arena, Deep Learning Studio, and Market Radar) remains responsive without disruption.

---

## 5. Testing & Verification Standard

All algorithmic changes must maintain 100% test coverage across the 26-test verification suite:
- `test_ai_engine.py`: Monotonicity, lookahead shifting, model serialization.
- `test_bot.py`: Autonomous execution, bracket orders, trailing stop ratchets.
- `test_data_feed.py`: Feature engineering integrity, ghost rows, whale volume detection.
- `test_health.py`: Hardware & GPU diagnostics.
- `test_sniper.py`: Intra-bar polarity execution, RRR filtering.
- `test_trader.py`: Gauntlet fills, mark-to-market, slippage, commission.
