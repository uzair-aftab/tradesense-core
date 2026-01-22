"""
Feature Engineering Module

Transforms raw market data and sentiment scores into 
machine learning ready feature sets.
"""

from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
import pandas_ta as ta
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """
    Generates features for ML models.
    """
    
    def __init__(self):
        """Initialize feature engineer."""
        self.feature_columns = []  # Populated after calculation
    
    def calculate_features(
        self, 
        df: pd.DataFrame, 
        include_targets: bool = False,
        prediction_horizon: int = 3,
        profit_target: float = 0.01
    ) -> pd.DataFrame:
        """
        Calculate technical and time-based features.
        
        Args:
            df: OHLCV DataFrame
            include_targets: Whether to generate target columns (for training)
            prediction_horizon: Days ahead to predict
            profit_target: Target return (e.g., 0.01 for 1%)
            
        Returns:
            DataFrame with features
        """
        if df.empty:
            logger.warning("Empty dataframe provided to FeatureEngineer")
            return df
            
        # Avoid modifying original
        df = df.copy()
        
        # 1. basic Price Features
        df['return_1d'] = df['Close'].pct_change(1)
        df['return_3d'] = df['Close'].pct_change(3)
        df['return_5d'] = df['Close'].pct_change(5)
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # 2. Volatility
        df['volatility_5d'] = df['return_1d'].rolling(5).std()
        df['volatility_20d'] = df['return_1d'].rolling(20).std()
        
        # 3. Technical Indicators (Vectorized via pandas-ta)
        # RSI
        df['rsi'] = ta.rsi(df['Close'], length=14)
        
        # MACD
        macd = ta.macd(df['Close'])
        if macd is not None:
             df['macd'] = macd['MACD_12_26_9']
             df['macd_hist'] = macd['MACDh_12_26_9']
             df['macd_signal'] = macd['MACDs_12_26_9']
        
        # Bollinger Bands
        bb = ta.bbands(df['Close'], length=20)
        if bb is not None:
             # DEBUG: Print columns if needed, but for now let's fix the column names
             # pandas-ta default for length=20, std=2 is: BBL_20_2.0, BBM_20_2.0, BBU_20_2.0
             # My previous code used _5_ which is wrong (default is 5, but I passed 20)
             
             # Dynamic column grab based on what's returned
             cols = bb.columns.tolist()
             lower = [c for c in cols if c.startswith('BBL')][0]
             upper = [c for c in cols if c.startswith('BBU')][0]
             mid = [c for c in cols if c.startswith('BBM')][0]
             pct_b = [c for c in cols if c.startswith('BBP')][0]
             
             # Bandwidth (Volatility feature)
             df['bb_width'] = (bb[upper] - bb[lower]) / bb[mid]
             # %B (Position feature)
             df['bb_pct_b'] = bb[pct_b]
            
        # ATR (Normalized)
        atr = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        df['atr_pct'] = atr / df['Close']
        
        # Volume
        df['volume_ma_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        
        # 4. Time Features
        if isinstance(df.index, pd.DatetimeIndex):
            df['day_of_week'] = df.index.dayofweek
            df['month'] = df.index.month
            df['is_month_end'] = df.index.is_month_end.astype(int)
        
        # 5. Clean NaNs created by lagging/rolling
        # We drop initial rows that don't have enough history
        df = df.dropna()
        
        # 6. Targets (Optional - only for training)
        if include_targets:
            # Target: 1 if Close in N days > Current Close * (1 + target)
            future_close = df['Close'].shift(-prediction_horizon)
            df['target_return'] = (future_close - df['Close']) / df['Close']
            df['target'] = (df['target_return'] > profit_target).astype(int)
            
            # Remove last N rows where target is NaN
            df = df.dropna(subset=['target'])
            
        self.feature_columns = [
            c for c in df.columns 
            if c not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', 'target', 'target_return']
        ]
        
        return df

    def prepare_training_data(
        self, 
        df: pd.DataFrame, 
        horizon: int = 3
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare X and y for training.
        """
        processed_df = self.calculate_features(df, include_targets=True, prediction_horizon=horizon)
        
        X = processed_df[self.feature_columns]
        y = processed_df['target']
        
        return X, y
