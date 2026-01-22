"""
TradeSense News Fetcher

Fetches financial news for sentiment analysis.
Uses free APIs: Finnhub and Marketaux.
"""

from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import requests

from src.config import config
from src.utils.logger import get_logger
from src.data.cache import cache

logger = get_logger(__name__)


class NewsFetcher:
    """
    Fetches financial news from multiple sources.
    
    Primary: Finnhub (free tier: 60 calls/min)
    Fallback: Marketaux (free tier: 100 calls/day)
    
    Example:
        >>> fetcher = NewsFetcher()
        >>> news = fetcher.get_news("AAPL", days=3)
    """
    
    FINNHUB_BASE = "https://finnhub.io/api/v1"
    MARKETAUX_BASE = "https://api.marketaux.com/v1"
    
    def __init__(
        self,
        finnhub_key: Optional[str] = None,
        marketaux_key: Optional[str] = None
    ):
        """
        Initialize news fetcher.
        
        Args:
            finnhub_key: Finnhub API key (optional, uses config)
            marketaux_key: Marketaux API key (optional, uses config)
        """
        self.finnhub_key = finnhub_key or config.FINNHUB_API_KEY
        self.marketaux_key = marketaux_key or config.MARKETAUX_API_KEY
    
    def get_news(
        self,
        ticker: str,
        days: int = 7,
        limit: int = 50,
        use_cache: bool = True
    ) -> List[Dict]:
        """
        Get news for a ticker from multiple sources.
        
        Args:
            ticker: Stock symbol
            days: Number of days to look back
            limit: Maximum number of articles
            use_cache: Whether to use cached results
            
        Returns:
            List of news article dicts with:
                - title: Article title
                - summary: Article summary/description
                - source: Source name
                - url: Article URL
                - published: Publication datetime
                - sentiment: Pre-analyzed sentiment if available
        """
        cache_key = f"news_{ticker}_{days}"
        
        if use_cache:
            cached = cache.get(cache_key)
            if cached:
                logger.debug(f"Using cached news for {ticker}")
                return cached
        
        articles = []
        
        # Try Finnhub first
        if self.finnhub_key:
            try:
                finnhub_news = self._fetch_finnhub(ticker, days)
                articles.extend(finnhub_news)
                logger.info(f"Fetched {len(finnhub_news)} articles from Finnhub for {ticker}")
            except Exception as e:
                logger.warning(f"Finnhub fetch failed for {ticker}: {e}")
        
        # Try Marketaux as fallback or supplement
        if self.marketaux_key and len(articles) < limit:
            try:
                marketaux_news = self._fetch_marketaux(ticker, days)
                # Avoid duplicates by checking URLs
                existing_urls = {a.get("url") for a in articles}
                for article in marketaux_news:
                    if article.get("url") not in existing_urls:
                        articles.append(article)
                logger.info(f"Fetched {len(marketaux_news)} articles from Marketaux for {ticker}")
            except Exception as e:
                logger.warning(f"Marketaux fetch failed for {ticker}: {e}")
        
        # Limit and sort by date
        articles = sorted(
            articles,
            key=lambda x: x.get("published", ""),
            reverse=True
        )[:limit]
        
        # Cache results
        if articles:
            cache.set(cache_key, articles, ttl=config.NEWS_CACHE_TTL)
        
        return articles
    
    def _fetch_finnhub(self, ticker: str, days: int) -> List[Dict]:
        """Fetch news from Finnhub."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        url = f"{self.FINNHUB_BASE}/company-news"
        params = {
            "symbol": ticker.replace(".TO", ""),  # Finnhub uses US symbols
            "from": start_date.strftime("%Y-%m-%d"),
            "to": end_date.strftime("%Y-%m-%d"),
            "token": self.finnhub_key,
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        articles = []
        for item in data:
            articles.append({
                "title": item.get("headline", ""),
                "summary": item.get("summary", ""),
                "source": item.get("source", "Finnhub"),
                "url": item.get("url", ""),
                "published": datetime.fromtimestamp(item.get("datetime", 0)).isoformat(),
                "category": item.get("category", ""),
                "related": item.get("related", ""),
            })
        
        return articles
    
    def _fetch_marketaux(self, ticker: str, days: int) -> List[Dict]:
        """Fetch news from Marketaux."""
        url = f"{self.MARKETAUX_BASE}/news/all"
        params = {
            "symbols": ticker,
            "filter_entities": "true",
            "language": "en",
            "published_after": (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%dT00:00"),
            "api_token": self.marketaux_key,
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        articles = []
        for item in data.get("data", []):
            # Extract pre-analyzed sentiment if available
            sentiment = None
            if "entities" in item:
                for entity in item["entities"]:
                    if entity.get("symbol") == ticker:
                        sentiment = entity.get("sentiment_score")
                        break
            
            articles.append({
                "title": item.get("title", ""),
                "summary": item.get("description", ""),
                "source": item.get("source", "Marketaux"),
                "url": item.get("url", ""),
                "published": item.get("published_at", ""),
                "sentiment": sentiment,
            })
        
        return articles
    
    def get_general_market_news(self, limit: int = 20) -> List[Dict]:
        """
        Get general market news (not ticker-specific).
        
        Args:
            limit: Maximum number of articles
            
        Returns:
            List of news article dicts
        """
        cache_key = "news_market_general"
        
        cached = cache.get(cache_key)
        if cached:
            return cached
        
        articles = []
        
        if self.finnhub_key:
            try:
                url = f"{self.FINNHUB_BASE}/news"
                params = {
                    "category": "general",
                    "token": self.finnhub_key,
                }
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                
                for item in response.json()[:limit]:
                    articles.append({
                        "title": item.get("headline", ""),
                        "summary": item.get("summary", ""),
                        "source": item.get("source", ""),
                        "url": item.get("url", ""),
                        "published": datetime.fromtimestamp(item.get("datetime", 0)).isoformat(),
                        "category": item.get("category", ""),
                    })
            except Exception as e:
                logger.warning(f"Failed to fetch general news: {e}")
        
        if articles:
            cache.set(cache_key, articles, ttl=config.NEWS_CACHE_TTL)
        
        return articles


# Singleton instance
news_fetcher = NewsFetcher()
