"""
Machine Learning Models

Wrappers for predictive models using XGBoost.
"""

from typing import Dict, Optional
import os
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score

from src.utils.logger import get_logger

logger = get_logger(__name__)


class SignalModel:
    """
    XGBoost classifier for trading signals.
    Predicts probability of positive return in N days.
    """
    
    def __init__(self, model_path: str = "models/xgb_model.json"):
        """
        Initialize model.
        
        Args:
            model_path: Path to save/load model
        """
        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            early_stopping_rounds=10,
            n_jobs=-1
        )
        self.model_path = model_path
        self.is_trained = False
        
        # Ensure model directory exists
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Auto-load existing model (like LSTM/Transformer)
        self.load()
    
    def train(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Dictionary of metrics
        """
        logger.info(f"Training XGBoost model on {len(X_train)} samples...")
        
        eval_set = [(X_train, y_train)]
        if X_val is not None:
            eval_set.append((X_val, y_val))
            
        self.model.fit(
            X_train, 
            y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        self.is_trained = True
        
        # Calculate metrics
        train_preds = self.model.predict(X_train)
        metrics = {
            "train_accuracy": accuracy_score(y_train, train_preds),
            "train_precision": precision_score(y_train, train_preds, zero_division=0)
        }
        
        if X_val is not None:
            val_preds = self.model.predict(X_val)
            metrics["val_accuracy"] = accuracy_score(y_val, val_preds)
            metrics["val_precision"] = precision_score(y_val, val_preds, zero_division=0)
            
        logger.info(f"Training complete. Metrics: {metrics}")
        return metrics
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of class 1 (Buy).
        
        Args:
            X: Features
            
        Returns:
            Array of probabilities
        """
        if not self.is_trained:
            # Try loading if not trained in memory
            if os.path.exists(self.model_path):
                self.load()
            else:
                logger.warning("Model not trained and no file found. Returning 0.5.")
                return np.full(len(X), 0.5)
                
        return self.model.predict_proba(X)[:, 1]
    
    def save(self):
        """Save model to disk."""
        self.model.save_model(self.model_path)
        logger.info(f"Model saved to {self.model_path}")
        
    def load(self):
        """Load model from disk."""
        if os.path.exists(self.model_path):
            self.model.load_model(self.model_path)
            self.is_trained = True
            logger.info(f"Model loaded from {self.model_path}")
        else:
            logger.error(f"Model file not found at {self.model_path}")
            
    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importance DataFrame."""
        if not self.is_trained:
            return pd.DataFrame()
            
        importance = pd.DataFrame({
            'feature': self.model.feature_names_in_,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance


# Singleton instance
signal_model = SignalModel()
