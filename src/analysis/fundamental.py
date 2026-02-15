"""
Fundamental Analysis Module

Provides fundamental scoring and signal generation for stocks.
Analyzes P/E ratios, earnings, revenue growth, and financial health metrics.
"""

from typing import Dict, Optional, Any
import numpy as np

from src.data.fetcher_fundamentals import fundamentals_fetcher
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FundamentalAnalyzer:
    """
    Analyzes fundamental metrics and generates trading signals.
    
    Combines valuation, growth, profitability, and financial health
    metrics into a composite fundamental score for stock selection.
    
    Example:
        >>> analyzer = FundamentalAnalyzer()
        >>> score = analyzer.analyze("AAPL")
        >>> print(score["fundamental_score"], score["reasons"])
    """
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize fundamental analyzer.
        
        Args:
            weights: Custom weights for fundamental factors
        """
        self.weights = weights or {
            "valuation": 0.30,
            "growth": 0.30,
            "profitability": 0.25,
            "health": 0.15,
        }
    
    def analyze(self, ticker: str) -> Dict[str, Any]:
        """
        Perform comprehensive fundamental analysis.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Dict with:
                - fundamental_score: 0-1 composite score
                - signal: True (bullish), False (bearish), None (neutral)
                - valuation_score: Valuation metrics score
                - growth_score: Growth metrics score
                - profitability_score: Profitability score
                - health_score: Financial health score
                - reasons: List of key findings
        """
        try:
            fundamentals = fundamentals_fetcher.get_fundamentals(ticker)
            earnings = fundamentals_fetcher.get_earnings(ticker)
            pe_comp = fundamentals_fetcher.get_pe_vs_industry(ticker)
            
            reasons = []
            
            # 1. Valuation Score
            valuation_score = self._score_valuation(fundamentals, pe_comp, reasons)
            
            # 2. Growth Score
            growth_score = self._score_growth(fundamentals, reasons)
            
            # 3. Profitability Score
            profitability_score = self._score_profitability(fundamentals, reasons)
            
            # 4. Financial Health Score
            health_score = self._score_health(fundamentals, reasons)
            
            # 5. Earnings Beat
            eps_beat = earnings.get("eps_beat")
            if eps_beat is True:
                reasons.append("Recent earnings beat")
            elif eps_beat is False:
                reasons.append("Recent earnings miss")
            
            # Composite score
            fundamental_score = (
                valuation_score * self.weights["valuation"] +
                growth_score * self.weights["growth"] +
                profitability_score * self.weights["profitability"] +
                health_score * self.weights["health"]
            )
            
            # Earnings beat bonus
            if eps_beat:
                fundamental_score = min(1.0, fundamental_score + 0.1)
            
            # Determine signal
            if fundamental_score >= 0.65:
                signal = True
            elif fundamental_score <= 0.35:
                signal = False
            else:
                signal = None
            
            return {
                "ticker": ticker,
                "fundamental_score": fundamental_score,
                "signal": signal,
                "valuation_score": valuation_score,
                "growth_score": growth_score,
                "profitability_score": profitability_score,
                "health_score": health_score,
                "reasons": reasons,
                "details": {
                    "pe_ratio": fundamentals.get("pe_ratio"),
                    "peg_ratio": fundamentals.get("peg_ratio"),
                    "revenue_growth": fundamentals.get("revenue_growth"),
                    "earnings_growth": fundamentals.get("earnings_growth"),
                    "profit_margin": fundamentals.get("profit_margin"),
                    "roe": fundamentals.get("roe"),
                    "debt_to_equity": fundamentals.get("debt_to_equity"),
                    "eps_beat": eps_beat,
                },
            }
        except Exception as e:
            logger.error(f"Fundamental analysis failed for {ticker}: {e}")
            return {
                "ticker": ticker,
                "fundamental_score": 0.5,
                "signal": None,
                "reasons": [f"Analysis failed: {str(e)}"],
                "error": str(e),
            }
    
    def _score_valuation(
        self,
        fundamentals: Dict[str, Any],
        pe_comp: Dict[str, Any],
        reasons: List[str]
    ) -> float:
        """Score valuation metrics (P/E, PEG, P/B)."""
        score = 0.5  # Neutral default
        
        pe_ratio = fundamentals.get("pe_ratio")
        peg_ratio = fundamentals.get("peg_ratio")
        pe_below_sector = pe_comp.get("pe_below_sector")
        
        if pe_below_sector:
            score += 0.2
            reasons.append("P/E below sector average")
        
        if peg_ratio and 0 < peg_ratio < 1:
            score += 0.2
            reasons.append(f"Attractive PEG ratio ({peg_ratio:.2f})")
        elif peg_ratio and peg_ratio > 2:
            score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def _score_growth(self, fundamentals: Dict[str, Any], reasons: List[str]) -> float:
        """Score growth metrics (revenue, earnings growth)."""
        score = 0.5
        
        rev_growth = fundamentals.get("revenue_growth")
        earn_growth = fundamentals.get("earnings_growth")
        
        if rev_growth and rev_growth > 0.15:
            score += 0.3
            reasons.append(f"Strong revenue growth ({rev_growth:.1%})")
        elif rev_growth and rev_growth < 0:
            score -= 0.3
            reasons.append("Declining revenue")
        
        if earn_growth and earn_growth > 0.15:
            score += 0.2
        elif earn_growth and earn_growth < 0:
            score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def _score_profitability(
        self,
        fundamentals: Dict[str, Any],
        reasons: List[str]
    ) -> float:
        """Score profitability metrics (margins, ROE)."""
        score = 0.5
        
        profit_margin = fundamentals.get("profit_margin")
        roe = fundamentals.get("roe")
        
        if profit_margin and profit_margin > 0.20:
            score += 0.3
            reasons.append(f"High profit margin ({profit_margin:.1%})")
        elif profit_margin and profit_margin < 0.05:
            score -= 0.3
        
        if roe and roe > 0.15:
            score += 0.2
        elif roe and roe < 0:
            score -= 0.3
        
        return max(0.0, min(1.0, score))
    
    def _score_health(self, fundamentals: Dict[str, Any], reasons: List[str]) -> float:
        """Score financial health (debt, current ratio)."""
        score = 0.5
        
        debt_to_equity = fundamentals.get("debt_to_equity")
        current_ratio = fundamentals.get("current_ratio")
        
        if debt_to_equity is not None:
            if debt_to_equity < 0.5:
                score += 0.3
                reasons.append("Low debt levels")
            elif debt_to_equity > 2.0:
                score -= 0.3
                reasons.append("High debt levels")
        
        if current_ratio and current_ratio > 1.5:
            score += 0.2
        elif current_ratio and current_ratio < 1.0:
            score -= 0.2
        
        return max(0.0, min(1.0, score))


# Singleton instance
fundamental_analyzer = FundamentalAnalyzer()
