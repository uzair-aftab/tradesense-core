"""
FinBERT Sentiment Analyzer

Dedicated module for NLP-based financial sentiment analysis using FinBERT.
Provides standardized interface matching documentation.
"""

from typing import List, Dict, Optional, Any
import numpy as np

from src.sentiment.finbert import FinBERTAnalyzer, get_finbert_analyzer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FinBERTSentimentAnalyzer:
    """
    Wrapper for FinBERT sentiment analysis with industry-standard interface.
    
    Analyzes financial text (news titles, headlines, social posts) and returns
    sentiment scores that integrate with the trading signal pipeline.
    
    Example:
        >>> analyzer = FinBERTSentimentAnalyzer()
        >>> sentiment = analyzer.analyze_text("Apple reports record earnings")
        >>> print(sentiment["label"], sentiment["score"])
    """
    
    def __init__(self):
        """Initialize FinBERT analyzer."""
        self._analyzer = None
    
    @property
    def analyzer(self) -> FinBERTAnalyzer:
        """Lazy-load FinBERT model."""
        if self._analyzer is None:
            self._analyzer = get_finbert_analyzer()
        return self._analyzer
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of a single text.
        
        Args:
            text: Financial text to analyze
            
        Returns:
            Dict with:
                - label: "positive", "negative", or "neutral"
                - score: Confidence score 0-1
                - sentiment_value: Numeric sentiment -1 to +1
        """
        try:
            result = self.analyzer.get_sentiment(text)
            
            # Map to numeric value
            label_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
            sentiment_value = label_map.get(result.get("label", "neutral"), 0.0)
            
            return {
                "label": result.get("label", "neutral"),
                "score": result.get("score", 0.0),
                "sentiment_value": sentiment_value,
            }
        except Exception as e:
            logger.error(f"FinBERT analysis failed: {e}")
            return {
                "label": "neutral",
                "score": 0.0,
                "sentiment_value": 0.0,
                "error": str(e),
            }
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze sentiment for a batch of texts.
        
        Args:
            texts: List of financial texts
            
        Returns:
            List of sentiment dicts
        """
        return [self.analyze_text(text) for text in texts]
    
    def get_aggregate_sentiment(
        self,
        texts: List[str],
        method: str = "weighted"
    ) -> Dict[str, Any]:
        """
        Get aggregated sentiment across multiple texts.
        
        Args:
            texts: List of financial texts
            method: Aggregation method - "weighted" or "average"
            
        Returns:
            Dict with aggregated sentiment metrics
        """
        try:
            return self.analyzer.get_aggregate_sentiment(texts, method=method)
        except Exception as e:
            logger.error(f"Aggregate sentiment failed: {e}")
            return {
                "aggregate_label": "neutral",
                "weighted_score": 0.0,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
                "error": str(e),
            }
    
    def get_sentiment_signal(self, texts: List[str], threshold: float = 0.2) -> Dict[str, Any]:
        """
        Generate a trading signal from text sentiment.
        
        Args:
            texts: List of financial texts (news, social posts)
            threshold: Minimum absolute score for bullish/bearish signal
            
        Returns:
            Dict with:
                - signal: True (bullish), False (bearish), None (neutral)
                - confidence: Signal confidence 0-1
                - score: Aggregate sentiment score -1 to +1
        """
        agg = self.get_aggregate_sentiment(texts)
        score = agg.get("weighted_score", 0.0)
        
        if score > threshold:
            signal = True
            confidence = min(1.0, abs(score))
        elif score < -threshold:
            signal = False
            confidence = min(1.0, abs(score))
        else:
            signal = None
            confidence = 1 - abs(score)  # Neutral confidence
        
        return {
            "signal": signal,
            "confidence": confidence,
            "score": score,
            "text_count": len(texts),
            "positive_count": agg.get("positive_count", 0),
            "negative_count": agg.get("negative_count", 0),
            "neutral_count": agg.get("neutral_count", 0),
        }


# Singleton instance
finbert_sentiment_analyzer = FinBERTSentimentAnalyzer()
