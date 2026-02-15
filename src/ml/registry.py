"""
ML Model Registry

Centralized model versioning and metadata management for all ML/RL models.
Tracks training metadata, performance metrics, and feature definitions.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import os
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ModelMetadata:
    """
    Metadata for a trained model.
    
    Tracks model configuration, training details, and performance metrics
    for reproducibility and versioning.
    """
    model_name: str
    model_type: str  # "xgboost", "lstm", "transformer", "ppo"
    version: str
    created_at: str
    trained_on_ticker: Optional[str] = None
    training_samples: int = 0
    feature_count: int = 0
    feature_names: Optional[List[str]] = None
    
    # Training configuration
    config: Dict[str, Any] = None
    
    # Performance metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc_roc: Optional[float] = None
    
    # Additional context
    training_period_start: Optional[str] = None
    training_period_end: Optional[str] = None
    notes: Optional[str] = None


class ModelRegistry:
    """
    Central registry for all ML/RL models.
    
    Manages model metadata, versioning, and provides a unified interface
    for saving/loading model configurations and performance tracking.
    
    Example:
        >>> registry = ModelRegistry()
        >>> registry.register_model(
        ...     model_name="xgb_signal",
        ...     model_type="xgboost",
        ...     version="1.0.0",
        ...     config={"n_estimators": 100, "max_depth": 5},
        ...     accuracy=0.85
        ... )
    """
    
    REGISTRY_FILE = "models/registry.json"
    
    def __init__(self):
        """Initialize model registry."""
        self.models: Dict[str, ModelMetadata] = {}
        self._ensure_models_dir()
        self.load_registry()
    
    def _ensure_models_dir(self):
        """Ensure models directory exists."""
        Path("models").mkdir(exist_ok=True)
    
    def register_model(
        self,
        model_name: str,
        model_type: str,
        version: str,
        config: Dict[str, Any],
        feature_count: int = 0,
        feature_names: Optional[List[str]] = None,
        training_samples: int = 0,
        **metrics
    ) -> ModelMetadata:
        """
        Register a new model or update existing registration.
        
        Args:
            model_name: Unique model identifier
            model_type: Type (xgboost, lstm, transformer, ppo)
            version: Semantic version string
            config: Model configuration dict
            feature_count: Number of input features
            feature_names: List of feature column names
            training_samples: Number of training samples
            **metrics: Performance metrics (accuracy, precision, etc.)
            
        Returns:
            ModelMetadata object
        """
        metadata = ModelMetadata(
            model_name=model_name,
            model_type=model_type,
            version=version,
            created_at=datetime.now().isoformat(),
            training_samples=training_samples,
            feature_count=feature_count,
            feature_names=feature_names,
            config=config,
            **metrics
        )
        
        self.models[model_name] = metadata
        self.save_registry()
        
        logger.info(
            f"Registered {model_type} model '{model_name}' v{version} "
            f"({feature_count} features, {training_samples} samples)"
        )
        
        return metadata
    
    def get_model_metadata(self, model_name: str) -> Optional[ModelMetadata]:
        """
        Get metadata for a registered model.
        
        Args:
            model_name: Model identifier
            
        Returns:
            ModelMetadata or None if not found
        """
        return self.models.get(model_name)
    
    def list_models(self, model_type: Optional[str] = None) -> List[ModelMetadata]:
        """
        List all registered models, optionally filtered by type.
        
        Args:
            model_type: Filter by model type (xgboost, lstm, etc.)
            
        Returns:
            List of ModelMetadata objects
        """
        models = list(self.models.values())
        
        if model_type:
            models = [m for m in models if m.model_type == model_type]
        
        return sorted(models, key=lambda m: m.created_at, reverse=True)
    
    def update_metrics(
        self,
        model_name: str,
        **metrics
    ) -> bool:
        """
        Update performance metrics for an existing model.
        
        Args:
            model_name: Model identifier
            **metrics: Metric key-value pairs
            
        Returns:
            True if updated, False if model not found
        """
        if model_name not in self.models:
            logger.warning(f"Model '{model_name}' not found in registry")
            return False
        
        metadata = self.models[model_name]
        
        for key, value in metrics.items():
            if hasattr(metadata, key):
                setattr(metadata, key, value)
        
        self.save_registry()
        logger.info(f"Updated metrics for '{model_name}'")
        
        return True
    
    def save_registry(self):
        """Persist registry to disk."""
        try:
            registry_data = {
                name: asdict(metadata)
                for name, metadata in self.models.items()
            }
            
            with open(self.REGISTRY_FILE, 'w') as f:
                json.dump(registry_data, f, indent=2)
            
            logger.debug(f"Saved model registry ({len(self.models)} models)")
        except Exception as e:
            logger.error(f"Failed to save model registry: {e}")
    
    def load_registry(self):
        """Load registry from disk."""
        if not os.path.exists(self.REGISTRY_FILE):
            logger.debug("No existing model registry found")
            return
        
        try:
            with open(self.REGISTRY_FILE, 'r') as f:
                registry_data = json.load(f)
            
            self.models = {
                name: ModelMetadata(**data)
                for name, data in registry_data.items()
            }
            
            logger.info(f"Loaded model registry ({len(self.models)} models)")
        except Exception as e:
            logger.error(f"Failed to load model registry: {e}")
            self.models = {}
    
    def get_best_model(
        self,
        model_type: Optional[str] = None,
        metric: str = "accuracy"
    ) -> Optional[ModelMetadata]:
        """
        Get the best performing model by a specific metric.
        
        Args:
            model_type: Filter by model type
            metric: Metric to optimize ("accuracy", "f1_score", etc.)
            
        Returns:
            ModelMetadata of best model or None
        """
        models = self.list_models(model_type=model_type)
        
        # Filter out models without the metric
        models_with_metric = [
            m for m in models
            if getattr(m, metric, None) is not None
        ]
        
        if not models_with_metric:
            return None
        
        return max(models_with_metric, key=lambda m: getattr(m, metric))


# Singleton instance
model_registry = ModelRegistry()
