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
from src.ml.feature_engineer import FeatureEngineer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EnsembleModel:
    """
    Combines XGBoost (tabular) and LSTM (sequential) predictions.
    
    Weighted ensemble:
    - XGBoost: Good for feature importance, handles irregular data
    - LSTM: Good for sequential patterns, temporal dependencies
    
    Example:
        >>> ensemble = EnsembleModel()
        >>> prob = ensemble.predict(features_df)
    """
    
    def __init__(
        self,
        xgb_weight: float = 0.6,
        lstm_weight: float = 0.4,
        sequence_length: int = 30
    ):
        """
        Args:
            xgb_weight: Weight for XGBoost predictions
            lstm_weight: Weight for LSTM predictions
            sequence_length: Sequence length for LSTM
        """
        self.xgb_weight = xgb_weight
        self.lstm_weight = lstm_weight
        self.sequence_length = sequence_length
        self.feature_engineer = FeatureEngineer()
    
    def predict(
        self,
        df: pd.DataFrame,
        use_lstm: bool = True
    ) -> Dict:
        """
        Get ensemble prediction for a ticker's data.
        
        Args:
            df: OHLCV DataFrame
            use_lstm: Whether to include LSTM (if trained)
            
        Returns:
            Dict with predictions and details
        """
        # Calculate features
        features_df = self.feature_engineer.calculate_features(df)
        
        if features_df.empty:
            return {
                "probability": 0.5,
                "signal": "HOLD",
                "confidence": 0.0,
                "xgb_prob": 0.5,
                "lstm_prob": 0.5,
                "ensemble_type": "none"
            }
        
        # Get latest features for XGBoost
        X_latest = features_df[self.feature_engineer.feature_columns].iloc[-1:]
        
        # XGBoost prediction
        try:
            xgb_prob = signal_model.predict_proba(X_latest)[0]
        except Exception as e:
            logger.warning(f"XGBoost prediction failed: {e}")
            xgb_prob = 0.5
        
        # LSTM prediction
        lstm_prob = 0.5
        lstm_used = False
        
        if use_lstm and lstm_model.is_trained:
            try:
                # Get sequence for LSTM
                X_full = features_df[self.feature_engineer.feature_columns].values
                if len(X_full) >= self.sequence_length:
                    X_sequence = X_full[-self.sequence_length:]
                    lstm_prob = lstm_model.predict_single(X_sequence)
                    lstm_used = True
            except Exception as e:
                logger.warning(f"LSTM prediction failed: {e}")
        
        # Combine predictions
        if lstm_used:
            ensemble_prob = (
                self.xgb_weight * xgb_prob + 
                self.lstm_weight * lstm_prob
            )
            ensemble_type = "xgb+lstm"
        else:
            ensemble_prob = xgb_prob
            ensemble_type = "xgb_only"
        
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
            "xgb_prob": xgb_prob,
            "lstm_prob": lstm_prob if lstm_used else None,
            "lstm_used": lstm_used,
            "ensemble_type": ensemble_type,
            "weights": {
                "xgb": self.xgb_weight,
                "lstm": self.lstm_weight if lstm_used else 0
            }
        }
    
    def predict_proba(self, df: pd.DataFrame) -> float:
        """Get ensemble probability (convenience method)."""
        result = self.predict(df)
        return result["probability"]
    
    def set_weights(self, xgb_weight: float, lstm_weight: float):
        """Update ensemble weights."""
        total = xgb_weight + lstm_weight
        self.xgb_weight = xgb_weight / total
        self.lstm_weight = lstm_weight / total
        logger.info(f"Ensemble weights: XGB={self.xgb_weight:.2f}, LSTM={self.lstm_weight:.2f}")


# Singleton instance
ensemble_model = EnsembleModel()
