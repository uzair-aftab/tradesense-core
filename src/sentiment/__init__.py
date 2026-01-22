"""
TradeSense Sentiment Analysis Module

Provides sentiment analysis from multiple sources:
- FinBERT for NLP-based financial sentiment
- News fetching from Finnhub and Marketaux
- Reddit sentiment from ApeWisdom and PRAW
- Aggregated sentiment scoring
"""

from src.sentiment.finbert import FinBERTAnalyzer, get_finbert_analyzer
from src.sentiment.news_fetcher import NewsFetcher, news_fetcher
from src.sentiment.reddit import RedditSentiment, reddit_sentiment
from src.sentiment.aggregator import SentimentAggregator, SentimentWeights, sentiment_aggregator

__all__ = [
    # FinBERT
    "FinBERTAnalyzer",
    "get_finbert_analyzer",
    # News
    "NewsFetcher",
    "news_fetcher",
    # Reddit
    "RedditSentiment",
    "reddit_sentiment",
    # Aggregator
    "SentimentAggregator",
    "SentimentWeights",
    "sentiment_aggregator",
]
