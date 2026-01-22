# Configuration

TradeSense is highly configurable through environment variables and the settings page in the UI.

## Environment Variables

All configuration is done through the `.env` file. Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

### API Keys

```env
# Finnhub - Primary data source
FINNHUB_API_KEY=your_key_here

# Reddit API - For WSB sentiment
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=TradeSense/1.0

# Marketaux - Backup news source
MARKETAUX_API_KEY=your_key_here
```

### Scanner Settings

```env
# Scan interval in seconds (default: 300 = 5 minutes)
SCAN_INTERVAL_SECONDS=300

# Enable batch mode instead of real-time
BATCH_MODE=false
```

### Signal Thresholds

```env
# RSI thresholds
RSI_OVERSOLD=20
RSI_OVERBOUGHT=80

# Put/Call ratio threshold
PUT_CALL_RATIO_THRESHOLD=0.7

# Volume confirmation multiplier
VOLUME_MULTIPLIER=1.0
```

## Watchlists

Default watchlists are defined in `src/data/watchlists.py`:

```python
STOCKS_WATCHLIST = ['RY.TO', 'TD.TO', 'SHOP.TO', 'ENB.TO', 'CNR.TO']
US_STOCKS_WATCHLIST = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN']
ETFS_WATCHLIST = ['SPY', 'QQQ', 'XIU.TO', 'VFV.TO', 'ARKK']
CRYPTO_WATCHLIST = ['BTC-CAD', 'ETH-CAD', 'SOL-CAD', 'ADA-CAD', 'AVAX-CAD']
```

### Adding/Removing Tickers

```python
from src.data.watchlists import WatchlistManager

# Add a ticker
WatchlistManager.add_ticker('stocks', 'BCE.TO')

# Remove a ticker
WatchlistManager.remove_ticker('crypto', 'ADA-CAD')

# Get all tickers
all_tickers = WatchlistManager.get_all_tickers()
```

## Adaptive Thresholds

TradeSense automatically adjusts thresholds based on:

1. **Asset Category**: Crypto gets wider RSI bands
2. **Volatility**: High ATR = wider thresholds
3. **Market Regime**: Trending vs ranging markets

### Category Adjustments

| Category | RSI Oversold | RSI Overbought | Volume Multiplier |
|----------|--------------|----------------|-------------------|
| Stocks | 20 | 80 | 1.0x |
| Crypto | 25 | 75 | 1.5x |
| ETFs | 30 | 70 | 1.0x |
| Small Cap | 25 | 75 | 2.0x |

## Logging

```env
# Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL=INFO

# Log file path (leave empty for console only)
LOG_FILE_PATH=logs/tradesense.log
```

## Database

```env
# SQLite cache database path
CACHE_DB_PATH=data/cache.db

# Cache TTL in seconds (default: 1 hour)
CACHE_TTL_SECONDS=3600
```
