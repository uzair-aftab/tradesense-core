"""Quick test script for FeatureEngineer v2"""
import traceback
from src.ml import FeatureEngineer
from src.data import stock_fetcher

try:
    print("Fetching AAPL data...")
    df = stock_fetcher.get_historical('AAPL', period='6mo')
    print(f"Data shape: {df.shape}")
    
    print("\nCalculating features...")
    fe = FeatureEngineer()
    result = fe.calculate_features(df)
    
    print(f"\nFeature count: {fe.get_feature_count()}")
    print(f"Result shape: {result.shape}")
    print("\nFeature names:")
    for i, name in enumerate(fe.get_feature_names(), 1):
        print(f"  {i:2d}. {name}")
        
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()
