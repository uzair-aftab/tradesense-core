"""
Social Media Sentiment Fetcher

Unified interface for fetching and aggregating sentiment from social media sources
including Reddit, Twitter (X), and other platforms.
"""

from typing import Dict, List, Optional, Any
import numpy as np

from src.sentiment.reddit import reddit_sentiment
from src.utils.logger import get_logger
from src.data.cache import cache
from src.config import config

logger = get_logger(__name__)


class SocialFetcher:
    """
    Fetches and aggregates sentiment from social media platforms.
    
    Currently supports:
    - Reddit (via r/wallstreetbets, r/stocks, r/investing)
    - Future: Twitter/X, StockTwits, Discord
    
    Example:
        >>> fetcher = SocialFetcher()
        >>> sentiment = fetcher.get_social_sentiment("AAPL")
        >>> print(sentiment["overall_score"], sentiment["mention_count"])
    """
    
    def __init__(self):
        """Initialize social media fetcher."""
        pass
    
    def get_reddit_sentiment(self, ticker: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Get sentiment from Reddit.
        
        Args:
            ticker: Stock symbol
            use_cache: Use cached results
            
        Returns:
            Dict with Reddit sentiment data
        """
        try:
            cache_key = f"social:reddit:{ticker}"
            if use_cache:
                cached = cache.get(cache_key)
                if isinstance(cached, dict) and cached:
                    return cached
            
            result = reddit_sentiment.get_reddit_sentiment_signal(ticker)
            
            if result:
                cache.set(cache_key, result, ttl=1800)  # 30 min cache
            
            return result
        except Exception as e:
            logger.error(f"Reddit sentiment fetch failed for {ticker}: {e}")
            return {
                "sentiment_score": None,
                "mention_count": 0,
                "signal": None,
                "error": str(e),
            }
    
    def get_twitter_sentiment(self, ticker: str) -> Dict[str, Any]:
        """
        Get sentiment from Twitter/X.
        
        Placeholder for future Twitter API integration.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Dict with Twitter sentiment data
        """
        logger.warning("Twitter/X sentiment not yet implemented")
        return {
            "sentiment_score": None,
            "mention_count": 0,
            "signal": None,
            "source": "twitter",
            "available": False,
        }
    
    def get_stocktwits_sentiment(self, ticker: str) -> Dict[str, Any]:
        """
        Get sentiment from StockTwits.
        
        Placeholder for future StockTwits integration.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Dict with StockTwits sentiment data
        """
        logger.warning("StockTwits sentiment not yet implemented")
        return {
            "sentiment_score": None,
            "mention_count": 0,
            "signal": None,
            "source": "stocktwits",
            "available": False,
        }
    
    def get_social_sentiment(
        self,
        ticker: str,
        sources: Optional[List[str]] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Get aggregated sentiment across all social platforms.
        
        Args:
            ticker: Stock symbol
            sources: List of sources to query (default: ["reddit"])
            use_cache: Use cached results
            
        Returns:
            Dict with:
                - overall_score: Aggregated sentiment -1 to +1
                - signal: True (bullish), False (bearish), None (neutral)
                - confidence: Signal confidence 0-1
                - mention_count: Total mentions across platforms
                - by_source: Dict of individual source results
        """
        sources = sources or ["reddit"]
        
        results = {}
        scores = []
        weights = []
        total_mentions = 0
        
        if "reddit" in sources:
            reddit_data = self.get_reddit_sentiment(ticker, use_cache=use_cache)
            results["reddit"] = reddit_data
            
            if reddit_data.get("sentiment_score") is not None:
                scores.append(reddit_data["sentiment_score"])
                weights.append(min(10, reddit_data.get("mention_count", 0)))  # Cap at 10
                total_mentions += reddit_data.get("mention_count", 0)
        
        if "twitter" in sources:
            twitter_data = self.get_twitter_sentiment(ticker)
            results["twitter"] = twitter_data
            # Add to aggregation when implemented
        
        if "stocktwits" in sources:
            st_data = self.get_stocktwits_sentiment(ticker)
            results["stocktwits"] = st_data
            # Add to aggregation when implemented
        
        # Aggregate scores
        if scores:
            weight_sum = float(sum(weights))
            if weight_sum > 0:
                overall_score = float(np.average(scores, weights=weights))
            else:
                overall_score = float(np.mean(scores))
            
            # Determine signal
            if overall_score > 0.2:
                signal = True
                confidence = min(1.0, abs(overall_score))
            elif overall_score < -0.2:
                signal = False
                confidence = min(1.0, abs(overall_score))
            else:
                signal = None
                confidence = 1 - abs(overall_score)
        else:
            overall_score = 0.0
            signal = None
            confidence = 0.0
        
        return {
            "ticker": ticker,
            "overall_score": overall_score,
            "signal": signal,
            "confidence": confidence,
            "mention_count": total_mentions,
            "by_source": results,
            "sources_used": [s for s in sources if results.get(s, {}).get("sentiment_score") is not None],
        }
    
    def get_trending_tickers(
        self,
        source: str = "reddit",
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get trending tickers from social media.
        
        Args:
            source: Social platform to query
            limit: Number of trending tickers to return
            
        Returns:
            List of dicts with ticker and mention count
        """
        if source == "reddit":
            try:
                return reddit_sentiment.get_trending_tickers(limit=limit)
            except Exception as e:
                logger.error(f"Failed to get trending tickers from Reddit: {e}")
                return []
        else:
            logger.warning(f"Trending tickers not yet implemented for {source}")
            return []


# Singleton instance
social_fetcher = SocialFetcher()
