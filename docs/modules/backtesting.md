# Backtesting

The backtesting module validates trading strategies against historical data.

## Overview

TradeSense uses [vectorbt](https://vectorbt.dev/) for high-performance backtesting.

## Metrics

| Metric | Target |
|--------|--------|
| Sharpe Ratio | > 1.5 |
| Win Rate | > 55% |
| Profit Factor | > 1.3 |
| Max Drawdown | < 20% |

## Usage

```python
from src.backtesting.engine import Backtester

backtester = Backtester()
results = backtester.run(
    tickers=['AAPL', 'MSFT'],
    start_date='2024-01-01',
    end_date='2025-01-01'
)

print(results.metrics)
# {
#   'sharpe_ratio': 1.8,
#   'win_rate': 0.62,
#   'profit_factor': 1.45,
#   'max_drawdown': 0.12
# }
```

## API Reference

::: src.backtesting.engine.Backtester

## Walk-Forward Validation

TradeSense supports walk-forward validation to prevent overfitting:

1. Train on first 80% of data
2. Test on remaining 20%
3. Roll forward and repeat
4. Average results across all windows
