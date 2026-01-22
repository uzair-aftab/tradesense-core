"""
TradeSense Reddit Sentiment Analyzer

Fetches and analyzes Reddit sentiment using PRAW and ApeWisdom.
"""

from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import requests

from src.config import config
from src.utils.logger import get_logger
from src.data.cache import cache

logger = get_logger(__name__)


class RedditSentiment:
    """
    Reddit sentiment analyzer using PRAW and ApeWisdom API.
    
    Fetches mentions and sentiment from r/wallstreetbets,
    r/stocks, r/investing, and other trading subreddits.
    
    Example:
        >>> reddit = RedditSentiment()
        >>> mentions = reddit.get_mentions("AAPL")
        >>> trending = reddit.get_trending()
    """
    
    APEWISDOM_BASE = "https://apewisdom.io/api/v1.0"
    
    # Trading-focused subreddits
    TRADING_SUBREDDITS = [
        "wallstreetbets",
        "stocks",
        "investing",
        "options",
        "stockmarket",
    ]
    
    def __init__(
        self,
        reddit_client_id: Optional[str] = None,
        reddit_secret: Optional[str] = None
    ):
        """
        Initialize Reddit sentiment analyzer.
        
        Args:
            reddit_client_id: Reddit API client ID
            reddit_secret: Reddit API client secret
        """
        self.reddit_client_id = reddit_client_id or config.REDDIT_CLIENT_ID
        self.reddit_secret = reddit_secret or config.REDDIT_CLIENT_SECRET
        self._praw = None
    
    def _init_praw(self):
        """Initialize PRAW (Python Reddit API Wrapper)."""
        if self._praw is not None:
            return
        
        if not self.reddit_client_id or not self.reddit_secret:
            logger.warning("Reddit API credentials not configured")
            return
        
        try:
            import praw
            self._praw = praw.Reddit(
                client_id=self.reddit_client_id,
                client_secret=self.reddit_secret,
                user_agent="TradeSense/1.0 (by /u/tradesense_bot)",
            )
            logger.info("PRAW initialized successfully")
        except ImportError:
            logger.warning("PRAW not installed, using ApeWisdom only")
        except Exception as e:
            logger.error(f"Failed to initialize PRAW: {e}")
    
    def get_mentions(
        self,
        ticker: str,
        hours: int = 24,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Get mention count and sentiment for a ticker.
        
        Args:
            ticker: Stock symbol
            hours: Hours to look back
            use_cache: Whether to use cached results
            
        Returns:
            Dict with:
                - mentions: Total mention count
                - rank: Popularity rank
                - upvotes: Total upvotes
                - sentiment: Aggregated sentiment if available
        """
        cache_key = f"reddit_mentions_{ticker}_{hours}"
        
        if use_cache:
            cached = cache.get(cache_key)
            if cached:
                return cached
        
        result = {
            "ticker": ticker,
            "mentions": 0,
            "rank": None,
            "upvotes": 0,
            "sentiment": None,
            "source": "apewisdom",
        }
        
        # Try ApeWisdom API first (no auth required)
        try:
            apewisdom_data = self._fetch_apewisdom(ticker)
            if apewisdom_data:
                result.update(apewisdom_data)
        except Exception as e:
            logger.warning(f"ApeWisdom fetch failed for {ticker}: {e}")
        
        # Cache results
        cache.set(cache_key, result, ttl=1800)  # 30 min cache
        
        return result
    
    def _fetch_apewisdom(self, ticker: str) -> Optional[Dict]:
        """Fetch ticker data from ApeWisdom API."""
        # Get current trending data
        url = f"{self.APEWISDOM_BASE}/filter/all-stocks/page/1"
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Find the ticker in results
        results = data.get("results", [])
        for item in results:
            if item.get("ticker", "").upper() == ticker.upper():
                return {
                    "mentions": item.get("mentions", 0),
                    "rank": item.get("rank"),
                    "upvotes": item.get("upvotes", 0),
                    "mentions_24h_ago": item.get("mentions_24h_ago", 0),
                    "name": item.get("name", ""),
                }
        
        return None
    
    def get_trending(
        self,
        limit: int = 25,
        filter_type: str = "all-stocks"
    ) -> List[Dict]:
        """
        Get trending tickers on Reddit.
        
        Args:
            limit: Maximum number of results
            filter_type: Filter type (all-stocks, crypto, etc.)
            
        Returns:
            List of trending ticker dicts
        """
        cache_key = f"reddit_trending_{filter_type}"
        
        cached = cache.get(cache_key)
        if cached:
            return cached[:limit]
        
        try:
            url = f"{self.APEWISDOM_BASE}/filter/{filter_type}/page/1"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = data.get("results", [])[:limit]
            
            trending = []
            for item in results:
                trending.append({
                    "ticker": item.get("ticker", ""),
                    "name": item.get("name", ""),
                    "mentions": item.get("mentions", 0),
                    "rank": item.get("rank"),
                    "upvotes": item.get("upvotes", 0),
                    "mentions_24h_ago": item.get("mentions_24h_ago", 0),
                })
            
            cache.set(cache_key, trending, ttl=1800)
            return trending
            
        except Exception as e:
            logger.error(f"Failed to fetch trending: {e}")
            return []
    
    def get_subreddit_posts(
        self,
        subreddit: str,
        search_query: str,
        limit: int = 25,
        time_filter: str = "day"
    ) -> List[Dict]:
        """
        Search posts in a subreddit.
        
        Args:
            subreddit: Subreddit name
            search_query: Search query (ticker)
            limit: Maximum posts
            time_filter: Time filter (hour, day, week, month)
            
        Returns:
            List of post dicts
        """
        self._init_praw()
        
        if not self._praw:
            logger.warning("PRAW not available")
            return []
        
        try:
            sub = self._praw.subreddit(subreddit)
            posts = []
            
            for post in sub.search(search_query, time_filter=time_filter, limit=limit):
                posts.append({
                    "title": post.title,
                    "text": post.selftext[:500] if post.selftext else "",
                    "score": post.score,
                    "upvote_ratio": post.upvote_ratio,
                    "num_comments": post.num_comments,
                    "created": datetime.fromtimestamp(post.created_utc).isoformat(),
                    "url": f"https://reddit.com{post.permalink}",
                    "subreddit": subreddit,
                })
            
            return posts
            
        except Exception as e:
            logger.error(f"Failed to search r/{subreddit}: {e}")
            return []
    
    def get_reddit_sentiment_signal(self, ticker: str) -> Dict[str, Any]:
        """
        Get aggregate sentiment signal from Reddit.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Dict with:
                - signal: True if bullish, False if bearish, None if neutral
                - mentions: Mention count
                - sentiment_score: Score -1 to +1
                - trending_rank: Rank if trending
        """
        mentions_data = self.get_mentions(ticker)
        
        mentions = mentions_data.get("mentions", 0)
        mentions_24h_ago = mentions_data.get("mentions_24h_ago", 0)
        rank = mentions_data.get("rank")
        
        # Calculate momentum (growing vs declining mentions)
        if mentions_24h_ago > 0:
            momentum = (mentions - mentions_24h_ago) / mentions_24h_ago
        else:
            momentum = 0
        
        # Simple sentiment heuristic:
        # - High mentions + positive momentum = bullish
        # - Declining mentions = potential reversal
        if mentions > 100 and momentum > 0.2:
            signal = True
            sentiment_score = min(momentum, 1.0) * 0.5 + 0.25
        elif mentions > 50 and momentum > 0:
            signal = True
            sentiment_score = momentum * 0.3 + 0.1
        elif momentum < -0.3:
            signal = False
            sentiment_score = max(momentum, -1.0) * 0.5
        else:
            signal = None
            sentiment_score = 0.0
        
        return {
            "ticker": ticker,
            "signal": signal,
            "sentiment_score": sentiment_score,
            "mentions": mentions,
            "mentions_24h_ago": mentions_24h_ago,
            "momentum": momentum,
            "trending_rank": rank,
        }


# Singleton instance
reddit_sentiment = RedditSentiment()
