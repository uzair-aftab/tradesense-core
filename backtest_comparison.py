"""
ML vs Heuristic Strategy Backtest Comparison

Compares performance of:
1. Heuristic Strategy (RSI + MACD only)
2. ML-Enhanced Strategy (Technical + ML prediction)
"""

import pandas as pd
import numpy as np
import vectorbt as vbt
from src.data import stock_fetcher
from src.ml import FeatureEngineer, signal_model
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_heuristic_strategy(df: pd.DataFrame, rsi_threshold: int = 30) -> dict:
    """
    Run pure technical heuristic strategy (RSI + MACD).
    """
    close = df['Close']
    
    # Calculate indicators
    rsi = vbt.RSI.run(close, window=14)
    macd = vbt.MACD.run(close, fast_window=12, slow_window=26)
    
    # Entry: RSI oversold AND MACD histogram positive
    entries = (rsi.rsi < rsi_threshold) & (macd.hist > 0)
    
    # Exit: RSI overbought OR MACD histogram negative
    exits = (rsi.rsi > 70) | (macd.hist < 0)
    
    # Clean signals
    clean_entries, clean_exits = entries.vbt.signals.clean(exits)
    
    # Run portfolio
    portfolio = vbt.Portfolio.from_signals(
        close,
        clean_entries,
        clean_exits,
        fees=0.001,
        slippage=0.001,
        init_cash=10000,
        freq="1D"
    )
    
    stats = portfolio.stats()
    return {
        "total_return": stats["Total Return [%]"],
        "sharpe_ratio": stats["Sharpe Ratio"],
        "max_drawdown": stats["Max Drawdown [%]"],
        "win_rate": stats["Win Rate [%]"],
        "trades": stats["Total Trades"],
        "portfolio": portfolio
    }


def run_ml_enhanced_strategy(df: pd.DataFrame, ml_threshold: float = 0.6) -> dict:
    """
    Run ML-enhanced strategy.
    Entry: Technical signals + ML probability > threshold
    """
    close = df['Close']
    
    # Calculate technical indicators
    rsi = vbt.RSI.run(close, window=14)
    macd = vbt.MACD.run(close, fast_window=12, slow_window=26)
    
    # Generate ML predictions
    engineer = FeatureEngineer()
    features_df = engineer.calculate_features(df.copy(), include_targets=False)
    
    # ML predictions (probability of price increase)
    X = features_df[engineer.feature_columns]
    ml_probs = signal_model.predict_proba(X)
    
    # Create ML probability series aligned with close
    ml_prob_series = pd.Series(index=features_df.index, data=ml_probs)
    ml_prob_aligned = ml_prob_series.reindex(close.index).fillna(0.5)
    
    # Technical signals
    tech_entry = (rsi.rsi < 30) & (macd.hist > 0)
    tech_exit = (rsi.rsi > 70) | (macd.hist < 0)
    
    # ML-enhanced entry: Technical signal + ML bullish
    entries = tech_entry & (ml_prob_aligned > ml_threshold)
    
    # Exit: Technical exit OR ML turns bearish
    exits = tech_exit | (ml_prob_aligned < 0.4)
    
    # Clean signals
    clean_entries, clean_exits = entries.vbt.signals.clean(exits)
    
    # Run portfolio
    portfolio = vbt.Portfolio.from_signals(
        close,
        clean_entries,
        clean_exits,
        fees=0.001,
        slippage=0.001,
        init_cash=10000,
        freq="1D"
    )
    
    stats = portfolio.stats()
    return {
        "total_return": stats["Total Return [%]"],
        "sharpe_ratio": stats["Sharpe Ratio"],
        "max_drawdown": stats["Max Drawdown [%]"],
        "win_rate": stats["Win Rate [%]"],
        "trades": stats["Total Trades"],
        "portfolio": portfolio
    }


def compare_strategies(ticker: str = "AAPL", period: str = "2y"):
    """
    Compare heuristic vs ML-enhanced strategies.
    """
    print("=" * 60)
    print(f"STRATEGY COMPARISON: {ticker} ({period})")
    print("=" * 60)
    
    # Fetch data
    print(f"\nFetching {ticker} data...")
    df = stock_fetcher.get_historical(ticker, period=period)
    print(f"Data: {len(df)} rows")
    
    # Run strategies
    print("\n[1/2] Running Heuristic Strategy (RSI + MACD)...")
    heuristic_results = run_heuristic_strategy(df)
    
    print("[2/2] Running ML-Enhanced Strategy...")
    ml_results = run_ml_enhanced_strategy(df)
    
    # Compare results
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)
    
    metrics = ["total_return", "sharpe_ratio", "max_drawdown", "win_rate", "trades"]
    labels = ["Total Return (%)", "Sharpe Ratio", "Max Drawdown (%)", "Win Rate (%)", "Trades"]
    
    print(f"\n{'Metric':<20} {'Heuristic':>15} {'ML-Enhanced':>15} {'Difference':>15}")
    print("-" * 65)
    
    for metric, label in zip(metrics, labels):
        h_val = heuristic_results[metric]
        ml_val = ml_results[metric]
        diff = ml_val - h_val if metric != "max_drawdown" else h_val - ml_val
        
        # Format values
        if metric in ["sharpe_ratio"]:
            h_str = f"{h_val:.3f}"
            ml_str = f"{ml_val:.3f}"
            diff_str = f"{diff:+.3f}"
        else:
            h_str = f"{h_val:.2f}"
            ml_str = f"{ml_val:.2f}"
            diff_str = f"{diff:+.2f}"
        
        print(f"{label:<20} {h_str:>15} {ml_str:>15} {diff_str:>15}")
    
    # Verdict
    print("\n" + "=" * 60)
    if ml_results["sharpe_ratio"] > heuristic_results["sharpe_ratio"]:
        improvement = ((ml_results["sharpe_ratio"] - heuristic_results["sharpe_ratio"]) 
                       / abs(heuristic_results["sharpe_ratio"]) * 100) if heuristic_results["sharpe_ratio"] != 0 else 100
        print(f"✅ ML-Enhanced strategy OUTPERFORMS by {improvement:.1f}% Sharpe improvement")
    else:
        print("⚠️ Heuristic strategy performs better (ML may need more training data)")
    print("=" * 60)
    
    return heuristic_results, ml_results


if __name__ == "__main__":
    # Test on multiple tickers
    tickers = ["AAPL", "MSFT", "GOOGL"]
    
    for ticker in tickers:
        try:
            compare_strategies(ticker, period="1y")
            print("\n")
        except Exception as e:
            print(f"Error testing {ticker}: {e}")
