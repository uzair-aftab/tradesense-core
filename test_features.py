from src.ml.feature_engineer import FeatureEngineer
from src.data import stock_fetcher
import pandas as pd

def test_feature_generation():
    print("Fetching data for AAPL...")
    df = stock_fetcher.get_historical("AAPL", period="1y")
    print(f"Raw data shape: {df.shape}")
    
    engineer = FeatureEngineer()
    
    print("\nGenerating features (Training Mode)...")
    X, y = engineer.prepare_training_data(df, horizon=3)
    
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print("\nFeature Columns Generated:")
    print(engineer.feature_columns)
    
    print("\nSample Features:")
    print(X.tail())
    
    print("\nTarget Distribution:")
    print(y.value_counts(normalize=True))
    
    # Check for NaNs
    if X.isnull().values.any():
        print("WARNING: NaNs found in X")
    else:
        print("SUCCESS: No NaNs in features")

if __name__ == "__main__":
    test_feature_generation()
