"""
Ensemble Model Combiner

Combines XGBoost and LSTM predictions with configurable weights.
Provides higher accuracy through model diversity.
"""

from typing import Dict, Optional
import numpy as np
import pandas as pd

from src.ml.models import signal_model
from src.ml.lstm_model import lstm_model, LSTMModel
from src.ml.transformer_model import transformer_model
from src.ml.feature_engineer import FeatureEngineer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EnsembleModel:
    """
    Combines XGBoost, LSTM, and Transformer predictions.
    
    Weighted ensemble:
    - XGBoost: Tabular features (40%)
    - LSTM: Sequential patterns (30%)
    - Transformer: Long-range attention (30%)
    """
    
    def __init__(
        self,
        xgb_weight: float = 0.4,
        lstm_weight: float = 0.3,
        transformer_weight: float = 0.3,
        sequence_length: int = 30
    ):
        self.xgb_weight = xgb_weight
        self.lstm_weight = lstm_weight
        self.transformer_weight = transformer_weight
        self.sequence_length = sequence_length
        self.feature_engineer = FeatureEngineer()
    
    def predict(
        self,
        df: pd.DataFrame,
        use_deep_learning: bool = True
    ) -> Dict:
        """
        Get ensemble prediction.
        """
        # Calculate features
        features_df = self.feature_engineer.calculate_features(df)
        
        if features_df.empty:
            return {
                "probability": 0.5,
                "signal": "HOLD",
                "confidence": 0.0,
                "components": {"xgb": 0.5, "lstm": None, "transformer": None}
            }
        
        # XGBoost prediction
        X_latest = features_df[self.feature_engineer.feature_columns].iloc[-1:]
        try:
            xgb_prob = signal_model.predict_proba(X_latest)[0]
        except Exception:
            xgb_prob = 0.5
        
        # Deep Learning predictions
        lstm_prob = 0.5
        tf_prob = 0.5
        dl_active = False
        
        if use_deep_learning and (lstm_model.is_trained or transformer_model.is_trained):
            try:
                X_full = features_df[self.feature_engineer.feature_columns].values
                if len(X_full) >= self.sequence_length:
                    X_sequence = X_full[-self.sequence_length:]
                    
                    if lstm_model.is_trained:
                        lstm_prob = lstm_model.predict_single(X_sequence)
                        dl_active = True
                        
                    if transformer_model.is_trained:
                        tf_prob = transformer_model.predict_single(X_sequence)
                        dl_active = True
            except Exception as e:
                logger.warning(f"DL prediction failed: {e}")
        
        # Normalize weights based on active models
        w_xgb = self.xgb_weight
        w_lstm = self.lstm_weight if lstm_model.is_trained and dl_active else 0
        w_tf = self.transformer_weight if transformer_model.is_trained and dl_active else 0
        
        total_weight = w_xgb + w_lstm + w_tf
        if total_weight == 0:
            return {"probability": 0.5, "signal": "HOLD", "confidence": 0.0}
            
        ensemble_prob = (
            w_xgb * xgb_prob + 
            w_lstm * lstm_prob + 
            w_tf * tf_prob
        ) / total_weight
        
        # Determine signal
        if ensemble_prob > 0.6:
            signal = "BUY"
            confidence = ensemble_prob
        elif ensemble_prob < 0.4:
            signal = "SELL"
            confidence = 1 - ensemble_prob
        else:
            signal = "HOLD"
            confidence = 0.5
        
        return {
            "probability": ensemble_prob,
            "signal": signal,
            "confidence": confidence,
            "components": {
                "xgb": xgb_prob,
                "lstm": lstm_prob if w_lstm > 0 else None,
                "transformer": tf_prob if w_tf > 0 else None
            },
            "weights": {
                "xgb": w_xgb / total_weight,
                "lstm": w_lstm / total_weight,
                "transformer": w_tf / total_weight
            }
        }
    
    def predict_proba(self, df: pd.DataFrame) -> float:
        return self.predict(df)["probability"]
    
    def set_weights(self, xgb: float, lstm: float, tf: float):
        total = xgb + lstm + tf
        self.xgb_weight = xgb / total
        self.lstm_weight = lstm / total
        self.transformer_weight = tf / total


# Singleton instance
ensemble_model = EnsembleModel()
