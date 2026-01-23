"""
Train RL Agent

Trains the PPO agent on historical data.
"""

from src.rl.agent import trading_agent
from src.data import stock_fetcher
import pandas as pd

print("="*60)
print("RL AGENT TRAINING")
print("="*60)

ticker = "AAPL"
print(f"Fetching data for {ticker}...")
df = stock_fetcher.get_historical(ticker, period="3y") # More data for RL

if len(df) > 500:
    print(f"Training on {len(df)} candles...")
    trading_agent.train(df, timesteps=50000)
    print("✅ Training Complete!")
else:
    print("❌ Not enough data!")
