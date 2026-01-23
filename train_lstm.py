"""
Train LSTM Model on Multi-Ticker Data

Uses CUDA GPU for accelerated training.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.ml import FeatureEngineer, lstm_model
from src.data import stock_fetcher

print("="*60)
print("TRAINING LSTM MODEL WITH CUDA")
print("="*60)

# Collect data from multiple tickers
tickers = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "AMD", "META", "AMZN"]
all_features = []
fe = FeatureEngineer()

for ticker in tickers:
    try:
        # Use 1-hour data for training (60 days available)
        df = stock_fetcher.get_historical(ticker, period="1y", interval="1d")
        features_df = fe.calculate_features(df, include_targets=True, prediction_horizon=3)
        
        if not features_df.empty:
            all_features.append(features_df)
            print(f"Loaded {ticker}: {len(features_df)} samples")
    except Exception as e:
        print(f"Failed {ticker}: {e}")

# Combine all data  
combined = pd.concat(all_features, ignore_index=True)
print(f"\nTotal combined samples: {len(combined)}")

# Prepare features and targets
feature_cols = fe.feature_columns
X = combined[feature_cols].values
y = combined['target'].values

print(f"Features: {X.shape[1]}")
print(f"Class distribution: 0={sum(y==0)}, 1={sum(y==1)}")

# Train/val split (keeping time order within sequences)
split_idx = int(len(X) * 0.8)
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

print(f"\nTraining samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")

# Train LSTM
print("\n" + "="*60)
print("Starting LSTM Training...")
print("="*60)

metrics = lstm_model.train(
    X_train, y_train,
    X_val, y_val,
    epochs=50,
    batch_size=32,
    early_stopping_patience=10
)

print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)
print(f"Train Accuracy: {metrics['train_accuracy']:.2%}")
print(f"Val Accuracy:   {metrics['val_accuracy']:.2%}")
print(f"Epochs Trained: {metrics['epochs_trained']}")
print(f"\nModel saved to: {lstm_model.MODEL_PATH}")
