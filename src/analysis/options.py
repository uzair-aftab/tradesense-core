"""
Options Flow Analysis Module

Analyzes options market data including put/call ratios, unusual activity,
and options-based sentiment for trading signals.
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from src.data.fetcher_options import options_fetcher
from src.utils.logger import get_logger

logger = get_logger(__name__)


class OptionsAnalyzer:
    """
    Analyzes options market data for trading signals.
    
    Uses put/call ratios, unusual options activity, and open interest
    analysis to gauge market sentiment and potential moves.
    
    Example:
        >>> analyzer = OptionsAnalyzer()
        >>> analysis = analyzer.analyze("AAPL")
        >>> print(analysis["options_score"], analysis["signal"])
    """
    
    def __init__(
        self,
        pc_bullish_threshold: float = 0.7,
        pc_bearish_threshold: float = 1.3
    ):
        """
        Initialize options analyzer.
        
        Args:
            pc_bullish_threshold: P/C ratio below this is bullish
            pc_bearish_threshold: P/C ratio above this is bearish
        """
        self.pc_bullish_threshold = pc_bullish_threshold
        self.pc_bearish_threshold = pc_bearish_threshold
    
    def analyze(
        self,
        ticker: str,
        include_unusual: bool = True
    ) -> Dict[str, Any]:
        """
        Perform comprehensive options analysis.
        
        Args:
            ticker: Stock symbol
            include_unusual: Include unusual activity analysis
            
        Returns:
            Dict with:
                - options_score: 0-1 composite score
                - signal: True (bullish), False (bearish), None (neutral)
                - pc_ratio: Put/call ratio
                - pc_sentiment: Sentiment from P/C ratio
                - unusual_activity: List of unusual options if enabled
                - reasons: List of key findings
        """
        try:
            reasons = []
            
            # 1. Put/Call Ratio Analysis
            pc_result = options_fetcher.get_put_call_ratio(ticker, method="volume")
            pc_ratio = pc_result.get("ratio")
            pc_score = 0.5  # Neutral default
            
            if pc_ratio is not None:
                pc_score = self._score_put_call_ratio(pc_ratio)
                pc_sentiment = self._interpret_pc_ratio(pc_ratio)
                
                if pc_ratio < self.pc_bullish_threshold:
                    reasons.append(f"Bullish P/C ratio ({pc_ratio:.2f})")
                elif pc_ratio > self.pc_bearish_threshold:
                    reasons.append(f"Bearish P/C ratio ({pc_ratio:.2f})")
            else:
                pc_sentiment = "No data"
            
            # 2. Unusual Activity Analysis
            unusual_activity = []
            unusual_score = 0.5
            
            if include_unusual:
                unusual_result = options_fetcher.get_unusual_activity(
                    ticker,
                    volume_threshold=2.0
                )
                unusual_activity = unusual_result.get("unusual_activity", [])
                
                if unusual_activity:
                    unusual_score = self._score_unusual_activity(unusual_activity)
                    reasons.append(f"{len(unusual_activity)} unusual options contracts")
            
            # 3. Composite Score
            options_score = (pc_score * 0.7) + (unusual_score * 0.3)
            
            # Determine signal
            if options_score >= 0.65:
                signal = True
            elif options_score <= 0.35:
                signal = False
            else:
                signal = None
            
            return {
                "ticker": ticker,
                "options_score": options_score,
                "signal": signal,
                "pc_ratio": pc_ratio,
                "pc_score": pc_score,
                "pc_sentiment": pc_sentiment,
                "unusual_activity": unusual_activity,
                "unusual_count": len(unusual_activity),
                "reasons": reasons,
            }
        except Exception as e:
            logger.error(f"Options analysis failed for {ticker}: {e}")
            return {
                "ticker": ticker,
                "options_score": 0.5,
                "signal": None,
                "reasons": [f"Analysis failed: {str(e)}"],
                "error": str(e),
            }
    
    def _score_put_call_ratio(self, ratio: float) -> float:
        """
        Convert P/C ratio to a 0-1 score.
        
        Args:
            ratio: Put/call ratio
            
        Returns:
            Score where 1.0 = very bullish, 0.0 = very bearish
        """
        if ratio <= 0.5:
            return 0.9
        elif ratio <= 0.7:
            return 0.7
        elif ratio <= 1.0:
            return 0.5
        elif ratio <= 1.3:
            return 0.3
        else:
            return 0.1
    
    def _interpret_pc_ratio(self, ratio: float) -> str:
        """
        Interpret put/call ratio as sentiment.
        
        Args:
            ratio: Put/call ratio
            
        Returns:
            Sentiment string
        """
        if ratio < 0.5:
            return "Very Bullish"
        elif ratio < 0.7:
            return "Bullish"
        elif ratio < 1.0:
            return "Neutral"
        elif ratio < 1.3:
            return "Bearish"
        else:
            return "Very Bearish"
    
    def _score_unusual_activity(self, unusual_activity: List[Dict]) -> float:
        """
        Score unusual options activity.
        
        Args:
            unusual_activity: List of unusual options contracts
            
        Returns:
            Score 0-1 based on bullish/bearish bias
        """
        if not unusual_activity:
            return 0.5
        
        call_count = sum(1 for u in unusual_activity if u.get("type") == "call")
        put_count = sum(1 for u in unusual_activity if u.get("type") == "put")
        
        total = call_count + put_count
        if total == 0:
            return 0.5
        
        # More calls = more bullish
        call_ratio = call_count / total
        
        if call_ratio >= 0.7:
            return 0.8
        elif call_ratio >= 0.6:
            return 0.65
        elif call_ratio >= 0.4:
            return 0.5
        elif call_ratio >= 0.3:
            return 0.35
        else:
            return 0.2
    
    def get_options_signal(self, ticker: str) -> Dict[str, Any]:
        """
        Get simplified options signal for trading.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Dict with signal, score, and reasons
        """
        analysis = self.analyze(ticker, include_unusual=False)
        
        return {
            "ticker": ticker,
            "signal": analysis.get("signal"),
            "score": analysis.get("options_score"),
            "pc_ratio": analysis.get("pc_ratio"),
            "reasons": analysis.get("reasons", []),
        }


# Singleton instance
options_analyzer = OptionsAnalyzer()
