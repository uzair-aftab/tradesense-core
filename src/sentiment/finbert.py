"""
TradeSense FinBERT Sentiment Analyzer

Uses the ProsusAI/finbert model for financial sentiment analysis.
Supports GPU acceleration for fast inference.
"""

from typing import List, Dict, Optional, Union
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

from src.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FinBERTAnalyzer:
    """
    Financial sentiment analyzer using FinBERT.
    
    Uses the ProsusAI/finbert model fine-tuned on financial text.
    Returns sentiment scores from -1 (negative) to +1 (positive).
    
    Example:
        >>> analyzer = FinBERTAnalyzer()
        >>> result = analyzer.analyze("Apple reports record earnings")
        >>> print(result["sentiment_score"])  # 0.85
    """
    
    MODEL_NAME = "ProsusAI/finbert"
    LABELS = ["negative", "neutral", "positive"]
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the FinBERT analyzer.
        
        Args:
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self._loaded = False
        
        logger.info(f"FinBERT analyzer initialized (device: {self.device})")
    
    def load_model(self) -> None:
        """Load the FinBERT model and tokenizer."""
        if self._loaded:
            return
        
        logger.info(f"Loading FinBERT model: {self.MODEL_NAME}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()
        
        self._loaded = True
        logger.info("FinBERT model loaded successfully")
    
    def _ensure_loaded(self) -> None:
        """Ensure model is loaded before inference."""
        if not self._loaded:
            self.load_model()
    
    def analyze(self, text: str) -> Dict:
        """
        Analyze sentiment of a single text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict with:
                - sentiment: Predicted label (positive/negative/neutral)
                - sentiment_score: Score from -1 to +1
                - confidence: Prediction confidence
                - probabilities: Dict of label probabilities
        """
        self._ensure_loaded()
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
            # Get prediction
            pred_idx = np.argmax(probs)
            sentiment = self.LABELS[pred_idx]
            confidence = float(probs[pred_idx])
            
            # Calculate sentiment score (-1 to +1)
            # negative=-1, neutral=0, positive=+1
            sentiment_score = float(probs[2] - probs[0])  # positive - negative
            
            return {
                "text": text[:100] + "..." if len(text) > 100 else text,
                "sentiment": sentiment,
                "sentiment_score": sentiment_score,
                "confidence": confidence,
                "probabilities": {
                    "negative": float(probs[0]),
                    "neutral": float(probs[1]),
                    "positive": float(probs[2]),
                },
            }
            
        except Exception as e:
            logger.error(f"Error analyzing text: {e}")
            return {
                "text": text[:100] + "..." if len(text) > 100 else text,
                "sentiment": "neutral",
                "sentiment_score": 0.0,
                "confidence": 0.0,
                "error": str(e),
            }
    
    def analyze_batch(
        self,
        texts: List[str],
        batch_size: int = 16
    ) -> List[Dict]:
        """
        Analyze sentiment of multiple texts.
        
        Args:
            texts: List of texts to analyze
            batch_size: Batch size for inference
            
        Returns:
            List of sentiment result dicts
        """
        self._ensure_loaded()
        
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                # Tokenize batch
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Inference
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
                
                # Process results
                for j, text in enumerate(batch):
                    pred_idx = np.argmax(probs[j])
                    sentiment = self.LABELS[pred_idx]
                    confidence = float(probs[j][pred_idx])
                    sentiment_score = float(probs[j][2] - probs[j][0])
                    
                    results.append({
                        "text": text[:100] + "..." if len(text) > 100 else text,
                        "sentiment": sentiment,
                        "sentiment_score": sentiment_score,
                        "confidence": confidence,
                        "probabilities": {
                            "negative": float(probs[j][0]),
                            "neutral": float(probs[j][1]),
                            "positive": float(probs[j][2]),
                        },
                    })
                    
            except Exception as e:
                logger.error(f"Error in batch analysis: {e}")
                for text in batch:
                    results.append({
                        "text": text[:100],
                        "sentiment": "neutral",
                        "sentiment_score": 0.0,
                        "confidence": 0.0,
                        "error": str(e),
                    })
        
        return results
    
    def get_aggregate_sentiment(self, texts: List[str]) -> Dict:
        """
        Get aggregate sentiment from multiple texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Dict with aggregate metrics:
                - average_score: Mean sentiment score
                - weighted_score: Confidence-weighted average
                - sentiment_distribution: Distribution of labels
        """
        if not texts:
            return {
                "average_score": 0.0,
                "weighted_score": 0.0,
                "sentiment_distribution": {"positive": 0, "neutral": 0, "negative": 0},
            }
        
        results = self.analyze_batch(texts)
        
        scores = [r["sentiment_score"] for r in results]
        confidences = [r["confidence"] for r in results]
        sentiments = [r["sentiment"] for r in results]
        
        # Calculate metrics
        avg_score = np.mean(scores)
        
        # Weighted by confidence
        if sum(confidences) > 0:
            weighted_score = np.average(scores, weights=confidences)
        else:
            weighted_score = avg_score
        
        # Distribution
        distribution = {
            "positive": sum(1 for s in sentiments if s == "positive"),
            "neutral": sum(1 for s in sentiments if s == "neutral"),
            "negative": sum(1 for s in sentiments if s == "negative"),
        }
        
        return {
            "average_score": float(avg_score),
            "weighted_score": float(weighted_score),
            "sentiment_distribution": distribution,
            "total_analyzed": len(texts),
        }


# Lazy-loaded singleton
_finbert_analyzer = None

def get_finbert_analyzer() -> FinBERTAnalyzer:
    """Get or create the FinBERT analyzer singleton."""
    global _finbert_analyzer
    if _finbert_analyzer is None:
        _finbert_analyzer = FinBERTAnalyzer()
    return _finbert_analyzer
