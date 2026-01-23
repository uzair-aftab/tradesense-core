"""
Feature Engineering Module v2 - Enhanced

Transforms raw market data into ML-ready feature sets.
Includes 40+ technical indicators for improved predictions.
"""

from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
import pandas_ta as ta
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """
    Generates 40+ features for ML models with robust error handling.
    """
    
    def __init__(self):
        """Initialize feature engineer."""
        self.feature_columns = []
    
    def _safe_indicator(self, func, *args, **kwargs):
        """Safely calculate an indicator, returning None on error."""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.debug(f"Indicator error: {e}")
            return None
    
    def calculate_features(
        self, 
        df: pd.DataFrame, 
        include_targets: bool = False,
        prediction_horizon: int = 3,
        profit_target: float = 0.01
    ) -> pd.DataFrame:
        """
        Calculate comprehensive technical features.
        
        Args:
            df: OHLCV DataFrame
            include_targets: Generate target columns for training
            prediction_horizon: Periods ahead to predict
            profit_target: Target return threshold
            
        Returns:
            DataFrame with 40+ features
        """
        if df.empty:
            logger.warning("Empty dataframe provided to FeatureEngineer")
            return df
            
        df = df.copy()
        
        # ================================================================
        # 1. PRICE RETURNS (4 features)
        # ================================================================
        df['return_1d'] = df['Close'].pct_change(1)
        df['return_3d'] = df['Close'].pct_change(3)
        df['return_5d'] = df['Close'].pct_change(5)
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # ================================================================
        # 2. VOLATILITY (4 features)
        # ================================================================
        df['volatility_5d'] = df['return_1d'].rolling(5).std()
        df['volatility_10d'] = df['return_1d'].rolling(10).std()
        df['volatility_20d'] = df['return_1d'].rolling(20).std()
        
        # ATR (Normalized)
        atr = self._safe_indicator(ta.atr, df['High'], df['Low'], df['Close'], length=14)
        if atr is not None:
            df['atr_pct'] = atr / df['Close']
        
        # ================================================================
        # 3. TREND INDICATORS (6 features)
        # ================================================================
        
        # RSI
        rsi = self._safe_indicator(ta.rsi, df['Close'], length=14)
        if rsi is not None:
            df['rsi'] = rsi
        
        rsi_fast = self._safe_indicator(ta.rsi, df['Close'], length=7)
        if rsi_fast is not None:
            df['rsi_fast'] = rsi_fast
        
        # MACD
        macd = self._safe_indicator(ta.macd, df['Close'], fast=12, slow=26, signal=9)
        if macd is not None:
            try:
                cols = macd.columns.tolist()
                macd_col = [c for c in cols if 'MACD_' in c and 'h' not in c.lower() and 's' not in c.lower()]
                hist_col = [c for c in cols if 'MACDh' in c]
                sig_col = [c for c in cols if 'MACDs' in c]
                if macd_col: df['macd'] = macd[macd_col[0]]
                if hist_col: df['macd_hist'] = macd[hist_col[0]]
                if sig_col: df['macd_signal'] = macd[sig_col[0]]
            except Exception as e:
                logger.debug(f"MACD parse error: {e}")
        
        # ADX - Average Directional Index (Trend Strength)
        adx = self._safe_indicator(ta.adx, df['High'], df['Low'], df['Close'], length=14)
        if adx is not None:
            try:
                cols = adx.columns.tolist()
                adx_col = [c for c in cols if c.startswith('ADX')]
                dmp_col = [c for c in cols if c.startswith('DMP')]
                dmn_col = [c for c in cols if c.startswith('DMN')]
                if adx_col: df['adx'] = adx[adx_col[0]]
                if dmp_col: df['dmp'] = adx[dmp_col[0]]
                if dmn_col: df['dmn'] = adx[dmn_col[0]]
            except Exception as e:
                logger.debug(f"ADX parse error: {e}")
        
        # ================================================================
        # 4. MOMENTUM INDICATORS (8 features)
        # ================================================================
        
        # Stochastic RSI
        stoch_rsi = self._safe_indicator(ta.stochrsi, df['Close'], length=14, rsi_length=14, k=3, d=3)
        if stoch_rsi is not None:
            try:
                cols = stoch_rsi.columns.tolist()
                k_col = [c for c in cols if 'K' in c]
                d_col = [c for c in cols if 'D' in c]
                if k_col: df['stoch_rsi_k'] = stoch_rsi[k_col[0]]
                if d_col: df['stoch_rsi_d'] = stoch_rsi[d_col[0]]
            except Exception as e:
                logger.debug(f"StochRSI parse error: {e}")
        
        # Williams %R
        willr = self._safe_indicator(ta.willr, df['High'], df['Low'], df['Close'], length=14)
        if willr is not None:
            df['willr'] = willr
        
        # CCI - Commodity Channel Index
        cci = self._safe_indicator(ta.cci, df['High'], df['Low'], df['Close'], length=20)
        if cci is not None:
            df['cci'] = cci
        
        # ROC - Rate of Change
        roc_5 = self._safe_indicator(ta.roc, df['Close'], length=5)
        if roc_5 is not None:
            df['roc_5'] = roc_5
        
        roc_10 = self._safe_indicator(ta.roc, df['Close'], length=10)
        if roc_10 is not None:
            df['roc_10'] = roc_10
        
        # Ultimate Oscillator
        uo = self._safe_indicator(ta.uo, df['High'], df['Low'], df['Close'])
        if uo is not None:
            df['uo'] = uo
        
        # Momentum
        mom = self._safe_indicator(ta.mom, df['Close'], length=10)
        if mom is not None:
            df['mom_10'] = mom
        
        # ================================================================
        # 5. VOLUME INDICATORS (4 features)
        # ================================================================
        
        # Volume MA Ratio
        df['volume_ma_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        
        # OBV - On Balance Volume (normalized)
        obv = self._safe_indicator(ta.obv, df['Close'], df['Volume'])
        if obv is not None:
            df['obv_pct'] = obv.pct_change(5)
        
        # CMF - Chaikin Money Flow
        cmf = self._safe_indicator(ta.cmf, df['High'], df['Low'], df['Close'], df['Volume'], length=20)
        if cmf is not None:
            df['cmf'] = cmf
        
        # MFI - Money Flow Index
        mfi = self._safe_indicator(ta.mfi, df['High'], df['Low'], df['Close'], df['Volume'], length=14)
        if mfi is not None:
            df['mfi'] = mfi
        
        # ================================================================
        # 6. VOLATILITY BANDS (3 features)
        # ================================================================
        
        # Bollinger Bands
        bb = self._safe_indicator(ta.bbands, df['Close'], length=20, std=2)
        if bb is not None:
            try:
                cols = bb.columns.tolist()
                lower = [c for c in cols if c.startswith('BBL')]
                upper = [c for c in cols if c.startswith('BBU')]
                mid = [c for c in cols if c.startswith('BBM')]
                pct_b = [c for c in cols if c.startswith('BBP')]
                
                if lower and upper and mid:
                    df['bb_width'] = (bb[upper[0]] - bb[lower[0]]) / (bb[mid[0]] + 0.0001)
                if pct_b:
                    df['bb_pct_b'] = bb[pct_b[0]]
            except Exception as e:
                logger.debug(f"BB parse error: {e}")
        
        # ================================================================
        # 7. SUPPORT/RESISTANCE & PRICE POSITION (5 features)
        # ================================================================
        
        # Price position relative to moving averages
        sma_10 = df['Close'].rolling(10).mean()
        sma_20 = df['Close'].rolling(20).mean()
        sma_50 = df['Close'].rolling(50).mean()
        
        df['price_sma_10'] = df['Close'] / sma_10 - 1
        df['price_sma_20'] = df['Close'] / sma_20 - 1
        df['price_sma_50'] = df['Close'] / sma_50 - 1
        
        # SMA crossover signals
        df['sma_10_20_cross'] = (sma_10 > sma_20).astype(int)
        
        # Daily range position
        df['daily_range_pos'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 0.0001)
        
        # ================================================================
        # 8. TIME FEATURES (4 features)
        # ================================================================
        if isinstance(df.index, pd.DatetimeIndex):
            df['day_of_week'] = df.index.dayofweek
            df['month'] = df.index.month
            df['is_month_end'] = df.index.is_month_end.astype(int)
            df['is_month_start'] = df.index.is_month_start.astype(int)
        
        # ================================================================
        # 9. CROSS-FEATURE INTERACTIONS (2 features)
        # ================================================================
        
        # RSI + Volume confirmation
        if 'rsi' in df.columns:
            df['rsi_vol_confirm'] = df['rsi'] * df['volume_ma_ratio'] / 100
        
        # Trend strength * momentum
        if 'adx' in df.columns and 'roc_10' in df.columns:
            df['trend_momentum'] = df['adx'] * df['roc_10'] / 100
        
        # ================================================================
        # CLEANUP
        # ================================================================
        
        # Drop rows with NaN (from rolling calculations)
        df = df.dropna()
        
        # ================================================================
        # TARGETS (Optional - for training)
        # ================================================================
        if include_targets:
            future_close = df['Close'].shift(-prediction_horizon)
            df['target_return'] = (future_close - df['Close']) / df['Close']
            df['target'] = (df['target_return'] > profit_target).astype(int)
            df = df.dropna(subset=['target'])
        
        # Define feature columns
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 
                        'Stock Splits', 'target', 'target_return', 'Adj Close']
        self.feature_columns = [c for c in df.columns if c not in exclude_cols]
        
        logger.info(f"Generated {len(self.feature_columns)} features")
        return df

    def prepare_training_data(
        self, 
        df: pd.DataFrame, 
        horizon: int = 3
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare X and y for training."""
        processed_df = self.calculate_features(df, include_targets=True, prediction_horizon=horizon)
        
        X = processed_df[self.feature_columns]
        y = processed_df['target']
        
        return X, y
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return self.feature_columns.copy()
    
    def get_feature_count(self) -> int:
        """Return number of features."""
        return len(self.feature_columns)
