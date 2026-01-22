# Technical Analysis

The technical analysis module provides indicators and signal generation based on price and volume data.

## Indicators

TradeSense uses [pandas-ta](https://github.com/twopirllc/pandas-ta) for technical indicators.

### Supported Indicators

| Indicator | Signal Condition | Module |
|-----------|-----------------|--------|
| RSI | ≤ 20 (oversold) | `pandas_ta.rsi()` |
| MACD | Bullish crossover | `pandas_ta.macd()` |
| Bollinger Bands | Price near lower band | `pandas_ta.bbands()` |
| Volume | > 20-day average | Custom calculation |
| Support/Resistance | Price near levels | Custom algorithm |

## Usage

```python
from src.analysis.technical import TechnicalAnalyzer
import yfinance as yf

# Get data
df = yf.download('AAPL', period='3mo')

# Analyze
analyzer = TechnicalAnalyzer(df)
analyzer.calculate_all()
signals = analyzer.generate_signals()

print(signals)
# {'rsi_signal': True, 'macd_signal': False, 'bb_signal': True, ...}
```

## API Reference

::: src.analysis.technical.TechnicalAnalyzer
