"""
End-to-end ML training test script.
"""
from src.data import stock_fetcher
from src.ml import FeatureEngineer, SignalModel
from sklearn.model_selection import train_test_split

def test_training():
    print("=" * 50)
    print("TradeSense ML Training Test")
    print("=" * 50)
    
    # 1. Fetch data
    print("\n[1/4] Fetching historical data for AAPL...")
    df = stock_fetcher.get_historical("AAPL", period="2y")
    print(f"  - Raw data: {len(df)} rows")
    
    # 2. Engineer features
    print("\n[2/4] Engineering features...")
    engineer = FeatureEngineer()
    X, y = engineer.prepare_training_data(df, horizon=3)
    print(f"  - Features: {X.shape[1]} columns")
    print(f"  - Samples: {len(X)} rows")
    print(f"  - Target distribution: {y.value_counts(normalize=True).to_dict()}")
    
    # 3. Train/Test Split
    print("\n[3/4] Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False  # Time-series: no shuffle
    )
    print(f"  - Train: {len(X_train)}, Val: {len(X_val)}")
    
    # 4. Train model
    print("\n[4/4] Training XGBoost model...")
    model = SignalModel(model_path="models/test_xgb.json")
    metrics = model.train(X_train, y_train, X_val, y_val)
    
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Feature importance
    print("\nTop 5 Features:")
    importance = model.get_feature_importance()
    print(importance.head())
    
    # Save
    model.save()
    print(f"\nModel saved to: {model.model_path}")

if __name__ == "__main__":
    test_training()
