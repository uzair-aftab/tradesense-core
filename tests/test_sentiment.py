"""
Unit tests for sentiment analysis module.
"""

import pytest
from unittest.mock import MagicMock, patch
from src.sentiment.aggregator import SentimentAggregator

@pytest.fixture
def mock_aggregator():
    """Create aggregator with mocked sources."""
    # Patch the INSTANCES imported in aggregator.py, not the classes
    with patch("src.sentiment.aggregator.news_fetcher") as mock_news, \
         patch("src.sentiment.aggregator.reddit_sentiment") as mock_reddit, \
         patch("src.sentiment.aggregator.get_finbert_analyzer") as mock_get_bert, \
         patch("src.sentiment.aggregator.cache") as mock_cache:
        
        # Setup mock instances
        # news_fetcher is the instance itself
        
        # for get_finbert_analyzer, it returns an analyzer instance
        mock_bert_instance = MagicMock()
        mock_get_bert.return_value = mock_bert_instance
        
        agg = SentimentAggregator()
        # Ensure lazy load uses our mock
        # We don't need to set agg.finbert manually because the property calls get_finbert_analyzer which is patched
        
        yield agg, mock_news, mock_reddit, mock_bert_instance


def test_sentiment_aggregation(mock_aggregator):
    """Test weighted score calculation."""
    agg, mock_news, mock_reddit, mock_bert = mock_aggregator
    
    # Setup return values
    mock_news.get_news.return_value = [
        {"title": "Good news", "sentiment": 0.8}
    ]
    # Check _analyze_news logic: it calls news_fetcher.get_news
    
    # We need to mock _analyze_news or mock what it calls.
    # Let's test get_sentiment calling _analyze_news
    
    # Mocking behaviors
    # We need to align with what aggregator._analyze_news expects from news_fetcher.get_news
    # It expects list of dicts or something?
    # aggregator.py: articles = news_fetcher.get_news(ticker, days=3, limit=20)
    
    mock_news.get_news.return_value = [
        {"title": "Apple is great", "sentiment": 0.9},
        {"title": "Apple stock soars", "sentiment": 0.7}
    ]
    
    mock_reddit.get_reddit_sentiment_signal.return_value = {
        "sentiment_score": 0.4,
        "post_count": 10
    }
    
    # FinBERT mock (used inside _analyze_news if enabled)
    mock_bert.get_aggregate_sentiment.return_value = {
        "weighted_score": 0.8
    }
    
    result = agg.get_sentiment("AAPL", include_finbert=True, use_cache=False)
    
    # Assertions
    assert "overall_score" in result
    assert "signal" in result
    
    # 0.8 (news avg) vs 0.4 (reddit) -> weighted avg 
    assert 0.4 <= result["overall_score"] <= 0.9


def test_empty_results(mock_aggregator):
    """Test handling of missing data."""
    agg, mock_news, mock_reddit, _ = mock_aggregator
    
    mock_news.get_news.return_value = []
    mock_reddit.get_reddit_sentiment_signal.return_value = {}
    
    result = agg.get_sentiment("UNKNOWN", use_cache=False)
    assert result["overall_score"] == 0.0
