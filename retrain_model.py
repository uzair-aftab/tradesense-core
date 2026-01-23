"""Retrain main model with enhanced features"""
from src.ml import FeatureEngineer, signal_model
from src.data import stock_fetcher
from sklearn.model_selection import train_test_split
import pandas as pd

print("="*60)
print("RETRAINING MODEL WITH ENHANCED 39 FEATURES")
print("="*60)

# Collect data from multiple tickers
tickers = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "AMD", "META", "AMZN"]
all_features = []
fe = FeatureEngineer()

for ticker in tickers:
    try:
        df = stock_fetcher.get_historical(ticker, period="2y")
        X_ticker, y_ticker = fe.prepare_training_data(df, horizon=3)
        X_ticker['target'] = y_ticker
        all_features.append(X_ticker)
        print(f"Loaded {ticker}: {len(X_ticker)} samples")
    except Exception as e:
        print(f"Failed {ticker}: {e}")

# Combine all data
combined = pd.concat(all_features, ignore_index=True)
print(f"\nTotal combined samples: {len(combined)}")

# Split features and target
X = combined.drop(columns=['target'])
y = combined['target']
print(f"Features: {X.shape[1]}, Samples: {len(X)}")

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)

print(f"\nTraining set: {len(X_train)} samples")
print(f"Validation set: {len(X_val)} samples")
print(f"Class distribution: {y.value_counts().to_dict()}")

# Train
print("\nTraining XGBoost with 39 features...")
metrics = signal_model.train(X_train, y_train, X_val, y_val)

print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)
print(f"Train Accuracy: {metrics['train_accuracy']:.2%}")
print(f"Val Accuracy:   {metrics['val_accuracy']:.2%}")
print(f"Train Precision: {metrics['train_precision']:.2%}")
print(f"Val Precision:   {metrics['val_precision']:.2%}")

# Save model
signal_model.save()
print(f"\nModel saved to: {signal_model.model_path}")

# Feature importance
print("\nTop 10 Feature Importance:")
importance = signal_model.get_feature_importance()
print(importance.head(10).to_string(index=False))
