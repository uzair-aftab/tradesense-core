"""
TradeSense Sentiment Aggregator

Combines sentiment from multiple sources into a unified score.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import numpy as np

from src.utils.logger import get_logger
from src.sentiment.finbert import get_finbert_analyzer
from src.sentiment.finbert_analyzer import finbert_sentiment_analyzer
from src.sentiment.news_fetcher import news_fetcher
from src.sentiment.reddit import reddit_sentiment
from src.sentiment.social_fetcher import social_fetcher
from src.data.cache import cache

logger = get_logger(__name__)


@dataclass
class SentimentWeights:
    """Weights for different sentiment sources."""
    news: float = 0.35
    social: float = 0.35
    finbert: float = 0.30


class SentimentAggregator:
    """
    Aggregates sentiment from multiple sources.
    
    Combines news sentiment, Reddit mentions, and FinBERT analysis
    into a unified sentiment score for trading signals.
    
    Example:
        >>> aggregator = SentimentAggregator()
        >>> sentiment = aggregator.get_sentiment("AAPL")
        >>> print(sentiment["overall_score"])  # 0.65
    """
    
    def __init__(self, weights: Optional[SentimentWeights] = None):
        """
        Initialize the aggregator.
        
        Args:
            weights: Custom weights for sentiment sources
        """
        self.weights = weights or SentimentWeights()
        self._finbert = None
    
    @property
    def finbert(self):
        """Lazy-load FinBERT analyzer."""
        if self._finbert is None:
            self._finbert = get_finbert_analyzer()
        return self._finbert
    
    def get_sentiment(
        self,
        ticker: str,
        include_news: bool = True,
        include_social: bool = True,
        include_finbert: bool = True,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Get aggregated sentiment for a ticker.
        
        Args:
            ticker: Stock symbol
            include_news: Include news sentiment
            include_social: Include social media sentiment (Reddit, Twitter, etc.)
            include_finbert: Run FinBERT on news titles
            use_cache: Use cached results
            
        Returns:
            Dict with:
                - overall_score: Combined score (-1 to +1)
                - signal: True (bullish), False (bearish), None (neutral)
                - news_sentiment: News-specific sentiment
                - social_sentiment: Social media sentiment
                - finbert_sentiment: FinBERT analysis results
                - confidence: Confidence in the signal
        """
        cache_key = f"sentiment_agg_{ticker}"
        
        if use_cache:
            cached = cache.get(cache_key)
            if cached:
                logger.debug(f"Using cached sentiment for {ticker}")
                return cached
        
        result = {
            "ticker": ticker,
            "overall_score": 0.0,
            "signal": None,
            "confidence": 0.0,
            "news_sentiment": None,
            "social_sentiment": None,
            "finbert_sentiment": None,
            "sources_used": [],
        }
        
        scores = []
        weights = []
        
        # News sentiment
        if include_news:
            news_data = self._analyze_news(ticker, include_finbert)
            result["news_sentiment"] = news_data
            if news_data.get("score") is not None:
                scores.append(news_data["score"])
                weights.append(self.weights.news)
                result["sources_used"].append("news")
        
        # Social Media sentiment (Reddit, Twitter, StockTwits)
        if include_social:
            social_data = social_fetcher.get_social_sentiment(ticker)
            result["social_sentiment"] = social_data
            if social_data.get("overall_score") is not None:
                scores.append(social_data["overall_score"])
                weights.append(self.weights.social)
                result["sources_used"].append("social")
        
        # Calculate overall score
        if scores:
            result["overall_score"] = float(np.average(scores, weights=weights))
            
            # Determine signal
            if result["overall_score"] > 0.2:
                result["signal"] = True
            elif result["overall_score"] < -0.2:
                result["signal"] = False
            else:
                result["signal"] = None
            
            # Confidence based on agreement and data availability
            result["confidence"] = self._calculate_confidence(scores, len(result["sources_used"]))
        
        # Cache for 30 minutes
        cache.set(cache_key, result, ttl=1800)
        
        logger.info(f"Sentiment for {ticker}: score={result['overall_score']:.2f}, signal={result['signal']}")
        return result
    
    def _analyze_news(
        self,
        ticker: str,
        use_finbert: bool = True
    ) -> Dict[str, Any]:
        """Analyze news sentiment for ticker."""
        try:
            # Fetch news
            articles = news_fetcher.get_news(ticker, days=3, limit=20)
            
            if not articles:
                return {"score": None, "articles": 0}
            
            # If articles have pre-analyzed sentiment, use it
            pre_analyzed = [
                a["sentiment"] for a in articles 
                if a.get("sentiment") is not None
            ]
            
            if pre_analyzed:
                pre_score = np.mean(pre_analyzed)
            else:
                pre_score = None
            
            # Run FinBERT on titles if enabled
            finbert_score = None
            if use_finbert:
                titles = [a["title"] for a in articles if a.get("title")]
                if titles:
                    finbert_result = self.finbert.get_aggregate_sentiment(titles)
                    finbert_score = finbert_result.get("weighted_score")
            
            # Combine scores
            scores = [s for s in [pre_score, finbert_score] if s is not None]
            if scores:
                final_score = np.mean(scores)
            else:
                final_score = None
            
            return {
                "score": float(final_score) if final_score is not None else None,
                "articles": len(articles),
                "finbert_score": finbert_score,
                "pre_analyzed_score": pre_score,
            }
            
        except Exception as e:
            logger.error(f"Error analyzing news for {ticker}: {e}")
            return {"score": None, "error": str(e)}
    
    def _calculate_confidence(
        self,
        scores: List[float],
        source_count: int
    ) -> float:
        """
        Calculate confidence in the sentiment signal.
        
        Args:
            scores: List of sentiment scores
            source_count: Number of sources used
            
        Returns:
            Confidence value 0-1
        """
        if not scores:
            return 0.0
        
        # Base confidence on source count
        source_factor = min(source_count / 3.0, 1.0)
        
        # Agreement factor (low std dev = high agreement)
        if len(scores) > 1:
            std_dev = np.std(scores)
            agreement_factor = max(0, 1 - std_dev)
        else:
            agreement_factor = 0.5
        
        # Signal strength factor
        strength = abs(np.mean(scores))
        strength_factor = min(strength * 2, 1.0)
        
        confidence = (source_factor * 0.3 + agreement_factor * 0.4 + strength_factor * 0.3)
        
        return float(confidence)
    
    def get_batch_sentiment(
        self,
        tickers: List[str],
        **kwargs
    ) -> Dict[str, Dict]:
        """
        Get sentiment for multiple tickers.
        
        Args:
            tickers: List of stock symbols
            **kwargs: Arguments passed to get_sentiment
            
        Returns:
            Dict mapping ticker to sentiment data
        """
        results = {}
        for ticker in tickers:
            results[ticker] = self.get_sentiment(ticker, **kwargs)
        return results
    
    def get_sentiment_signal(self, ticker: str) -> Dict[str, Any]:
        """
        Get simplified sentiment signal for trading.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Dict with:
                - signal: True/False/None
                - score: -1 to +1
                - confidence: 0 to 1
        """
        sentiment = self.get_sentiment(ticker)
        
        return {
            "ticker": ticker,
            "signal": sentiment["signal"],
            "score": sentiment["overall_score"],
            "confidence": sentiment["confidence"],
        }


# Singleton instance
sentiment_aggregator = SentimentAggregator()
