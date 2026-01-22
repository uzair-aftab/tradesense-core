
from src.backtesting import backtest_engine
import pandas as pd

def test_optimization():
    print("Testing RSI Optimization for AAPL...")
    try:
        # Create thresholds list
        thresholds = list(range(20, 45, 5))
        
        results = backtest_engine.optimize_rsi('AAPL', period='6mo', thresholds=thresholds)
        print("\nOptimization Results:")
        print(results)
        
        if not results.empty:
            # Results is a Series with index=thresholds
            best_param = results.idxmax()
            best_return = results.max()
            print(f"\nBest RSI Threshold: {best_param}")
            print(f"Return: {best_return:.2f}%")
        else:
            print("No results returned.")
            
    except Exception as e:
        print(f"Optimization failed: {e}")

if __name__ == "__main__":
    test_optimization()
