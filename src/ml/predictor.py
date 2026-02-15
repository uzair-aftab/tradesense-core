"""
ML Prediction Interface

Unified prediction interface for all ML/RL models.
Standardizes input/output contracts across XGBoost, LSTM, Transformer, and PPO.
"""

from typing import Dict, Optional, Any, Protocol
from dataclasses import dataclass
import pandas as pd
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PredictionResult:
    """
    Standardized prediction output across all model types.
    
    All predictors return this unified format for consistent downstream integration.
    """
    probability: float  # 0-1 probability of positive return
    signal: str  # "BUY", "SELL", or "HOLD"
    confidence: float  # 0-1 confidence in the prediction
    model_name: str  # Model identifier
    model_type: str  # "xgboost", "lstm", "transformer", "ppo", "ensemble"
    
    # Optional metadata
    raw_prediction: Optional[Any] = None
    feature_importance: Optional[Dict[str, float]] = None
    explanation: Optional[str] = None


class BasePredictor(Protocol):
    """
    Protocol defining the standard interface all predictors must implement.
    
    This ensures consistency across XGBoost, LSTM, Transformer, and PPO models.
    """
    
    def predict(self, df: pd.DataFrame) -> PredictionResult:
        """
        Generate prediction from price data.
        
        Args:
            df: OHLCV DataFrame with price history
            
        Returns:
            PredictionResult with standardized fields
        """
        ...
    
    def is_ready(self) -> bool:
        """
        Check if model is trained and ready for prediction.
        
        Returns:
            True if model can predict, False otherwise
        """
        ...


def interpret_probability(
    probability: float,
    buy_threshold: float = 0.6,
    sell_threshold: float = 0.4
) -> str:
    """
    Convert probability to trading signal.
    
    Args:
        probability: 0-1 probability of positive return
        buy_threshold: Minimum probability for BUY signal
        sell_threshold: Maximum probability for SELL signal
        
    Returns:
        "BUY", "SELL", or "HOLD"
    """
    if probability >= buy_threshold:
        return "BUY"
    elif probability <= sell_threshold:
        return "SELL"
    else:
        return "HOLD"


def calculate_confidence(
    probability: float,
    decision_boundary: float = 0.5
) -> float:
    """
    Calculate confidence based on distance from decision boundary.
    
    Args:
        probability: Model probability output
        decision_boundary: Neutral decision point
        
    Returns:
        Confidence score 0-1
    """
    distance = abs(probability - decision_boundary)
    return min(1.0, distance * 2)  # Normalize to 0-1


class UnifiedPredictor:
    """
    Unified prediction interface that wraps all model types.
    
    Provides a single entry point for predictions with automatic fallback
    and error handling across different model architectures.
    
    Example:
        >>> predictor = UnifiedPredictor()
        >>> result = predictor.predict(df, model_type="ensemble")
        >>> print(result.signal, result.probability, result.confidence)
    """
    
    def __init__(self):
        """Initialize unified predictor with lazy-loaded models."""
        self._models = {}

    def _normalize_price_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize OHLCV column names to Title case expected by feature engineering."""
        if df.empty:
            return df

        rename_map = {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
            "adj close": "Adj Close",
            "adj_close": "Adj Close",
        }

        normalized = df.copy()
        normalized.columns = [str(col) for col in normalized.columns]
        normalized = normalized.rename(
            columns={col: rename_map.get(col.lower(), col) for col in normalized.columns}
        )
        return normalized
    
    def _get_model(self, model_type: str):
        """Lazy-load model by type."""
        if model_type in self._models:
            return self._models[model_type]
        
        try:
            if model_type == "xgboost":
                from src.ml.models import signal_model
                self._models[model_type] = signal_model
            elif model_type == "lstm":
                from src.ml.lstm_model import lstm_model
                self._models[model_type] = lstm_model
            elif model_type == "transformer":
                from src.ml.transformer_model import transformer_model
                self._models[model_type] = transformer_model
            elif model_type == "ppo":
                from src.rl.agent import trading_agent
                self._models[model_type] = trading_agent
            elif model_type == "ensemble":
                from src.ml.ensemble import ensemble_model
                self._models[model_type] = ensemble_model
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            return self._models[model_type]
        except Exception as e:
            logger.error(f"Failed to load {model_type} model: {e}")
            return None
    
    def predict(
        self,
        df: pd.DataFrame,
        model_type: str = "ensemble"
    ) -> Optional[PredictionResult]:
        """
        Get prediction from specified model type.
        
        Args:
            df: OHLCV price data
            model_type: Type of model to use (xgboost, lstm, transformer, ppo, ensemble)
            
        Returns:
            PredictionResult or None if prediction fails
        """
        model = self._get_model(model_type)
        
        if model is None:
            logger.warning(f"Model type '{model_type}' not available")
            return None
        
        try:
            df = self._normalize_price_columns(df)

            if model_type == "ensemble":
                # Ensemble already returns a dict
                result_dict = model.predict(df)
                probability = result_dict.get("probability", 0.5)
                
                return PredictionResult(
                    probability=probability,
                    signal=interpret_probability(probability),
                    confidence=result_dict.get("confidence", calculate_confidence(probability)),
                    model_name=f"ensemble_{result_dict.get('active_models', 1)}",
                    model_type="ensemble",
                    raw_prediction=result_dict
                )
            
            elif model_type == "xgboost":
                from src.ml.feature_engineer import FeatureEngineer
                
                if not model.is_trained:
                    logger.warning("XGBoost model not trained")
                    return None
                
                feature_engineer = FeatureEngineer()
                features_df = feature_engineer.calculate_features(df, include_targets=False)
                
                if features_df.empty:
                    return None
                
                X_latest = features_df[feature_engineer.feature_columns].iloc[[-1]]
                probability = model.predict_proba(X_latest)[0]
                
                return PredictionResult(
                    probability=probability,
                    signal=interpret_probability(probability),
                    confidence=calculate_confidence(probability),
                    model_name="xgb_signal",
                    model_type="xgboost",
                    raw_prediction=probability
                )
            
            elif model_type in ["lstm", "transformer"]:
                if not model.is_trained:
                    logger.warning(f"{model_type.upper()} model not trained")
                    return None
                
                from src.ml.feature_engineer import FeatureEngineer
                feature_engineer = FeatureEngineer()
                features_df = feature_engineer.calculate_features(df, include_targets=False)
                
                if len(features_df) < model.config.get("sequence_length", 30):
                    logger.warning(f"Insufficient data for {model_type}")
                    return None
                
                probability = model.predict(features_df)
                
                return PredictionResult(
                    probability=probability,
                    signal=interpret_probability(probability),
                    confidence=calculate_confidence(probability),
                    model_name=f"{model_type}_predictor",
                    model_type=model_type,
                    raw_prediction=probability
                )
            
            elif model_type == "ppo":
                # PPO returns action (0, 1, 2) not probability
                # This needs environment obs construction - simplified for now
                logger.warning("PPO direct prediction not yet standardized")
                return None
            
        except Exception as e:
            logger.error(f"{model_type} prediction failed: {e}")
            return None
    
    def predict_with_fallback(
        self,
        df: pd.DataFrame,
        preferred_models: list = None
    ) -> Optional[PredictionResult]:
        """
        Try multiple models with fallback.
        
        Args:
            df: Price data
            preferred_models: Ordered list of model types to try
            
        Returns:
            First successful PredictionResult or None
        """
        if preferred_models is None:
            preferred_models = ["ensemble", "xgboost", "lstm"]
        
        for model_type in preferred_models:
            result = self.predict(df, model_type=model_type)
            if result is not None:
                logger.info(f"Prediction successful with {model_type}")
                return result
        
        logger.error("All prediction models failed")
        return None


# Singleton instance
unified_predictor = UnifiedPredictor()
