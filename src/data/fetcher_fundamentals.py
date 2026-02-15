"""
TradeSense Fundamentals Data Fetcher

Fetches fundamental data for stocks including P/E ratios,
earnings, and revenue data using yFinance.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
import yfinance as yf

from src.utils.logger import get_logger
from src.data.cache import cache
from src.config import config

logger = get_logger(__name__)


class FundamentalsFetcher:
    """
    Fetches fundamental data for stocks.
    
    Provides P/E ratios, earnings data, revenue, and other
    fundamental metrics for signal generation.
    
    Example:
        >>> fetcher = FundamentalsFetcher()
        >>> fundamentals = fetcher.get_fundamentals("AAPL")
        >>> earnings = fetcher.get_earnings("MSFT")
    """
    
    def get_fundamentals(self, ticker: str) -> Dict[str, Any]:
        """
        Get comprehensive fundamental data for a stock.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Dict with fundamental metrics:
                - pe_ratio: Price to Earnings ratio
                - forward_pe: Forward P/E
                - peg_ratio: PEG ratio
                - price_to_book: P/B ratio
                - market_cap: Market capitalization
                - enterprise_value: Enterprise value
                - profit_margin: Profit margin
                - revenue: Total revenue
                - revenue_growth: Revenue growth YoY
                - earnings_growth: Earnings growth
                - sector: Company sector
                - industry: Company industry
        """
        try:
            cache_key = f"fundamentals:core:{ticker}"
            cached = cache.get(cache_key)
            if isinstance(cached, dict) and cached.get("ticker") == ticker:
                return cached

            stock = yf.Ticker(ticker)
            info = stock.info
            
            fundamentals = {
                "ticker": ticker,
                "name": info.get("shortName", ticker),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                
                # Valuation metrics
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "peg_ratio": info.get("pegRatio"),
                "price_to_book": info.get("priceToBook"),
                "price_to_sales": info.get("priceToSalesTrailing12Months"),
                
                # Size metrics
                "market_cap": info.get("marketCap"),
                "enterprise_value": info.get("enterpriseValue"),
                
                # Profitability
                "profit_margin": info.get("profitMargins"),
                "operating_margin": info.get("operatingMargins"),
                "gross_margin": info.get("grossMargins"),
                "roe": info.get("returnOnEquity"),
                "roa": info.get("returnOnAssets"),
                
                # Growth
                "revenue": info.get("totalRevenue"),
                "revenue_growth": info.get("revenueGrowth"),
                "earnings_growth": info.get("earningsGrowth"),
                "earnings_quarterly_growth": info.get("earningsQuarterlyGrowth"),
                
                # Dividends
                "dividend_yield": info.get("dividendYield"),
                "dividend_rate": info.get("dividendRate"),
                "payout_ratio": info.get("payoutRatio"),
                
                # Financial health
                "current_ratio": info.get("currentRatio"),
                "debt_to_equity": info.get("debtToEquity"),
                "quick_ratio": info.get("quickRatio"),
                
                # Other
                "beta": info.get("beta"),
                "52_week_high": info.get("fiftyTwoWeekHigh"),
                "52_week_low": info.get("fiftyTwoWeekLow"),
                "50_day_avg": info.get("fiftyDayAverage"),
                "200_day_avg": info.get("twoHundredDayAverage"),
                
                # Analyst data
                "target_mean_price": info.get("targetMeanPrice"),
                "target_high_price": info.get("targetHighPrice"),
                "target_low_price": info.get("targetLowPrice"),
                "recommendation": info.get("recommendationKey"),
                "analyst_count": info.get("numberOfAnalystOpinions"),
            }
            
            logger.info(f"Fetched fundamentals for {ticker}")
            cache.set(cache_key, fundamentals, ttl=config.FUNDAMENTALS_CACHE_TTL)
            return fundamentals
            
        except Exception as e:
            logger.error(f"Error fetching fundamentals for {ticker}: {e}")
            return {"ticker": ticker, "error": str(e)}
    
    def get_earnings(self, ticker: str) -> Dict[str, Any]:
        """
        Get earnings data including beat/miss history.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Dict with earnings data:
                - history: DataFrame of historical earnings
                - next_date: Next earnings date
                - latest_eps_actual: Latest actual EPS
                - latest_eps_estimate: Latest EPS estimate
                - eps_beat: True if beat estimate
        """
        try:
            cache_key = f"fundamentals:earnings:{ticker}"
            cached = cache.get(cache_key)
            if isinstance(cached, dict) and cached.get("ticker") == ticker:
                return cached

            stock = yf.Ticker(ticker)
            
            # Get earnings history
            earnings_history = stock.earnings_history
            
            result = {
                "ticker": ticker,
                "earnings_history": earnings_history,
                "latest_eps_actual": None,
                "latest_eps_estimate": None,
                "eps_beat": None,
                "eps_surprise_percent": None,
            }
            
            # Parse latest earnings
            if earnings_history is not None and not earnings_history.empty:
                latest = earnings_history.iloc[-1]
                result["latest_eps_actual"] = latest.get("epsActual")
                result["latest_eps_estimate"] = latest.get("epsEstimate")
                
                if result["latest_eps_actual"] and result["latest_eps_estimate"]:
                    result["eps_beat"] = result["latest_eps_actual"] > result["latest_eps_estimate"]
                    result["eps_surprise_percent"] = (
                        (result["latest_eps_actual"] - result["latest_eps_estimate"]) 
                        / abs(result["latest_eps_estimate"]) * 100
                    )
            
            # Get calendar for next earnings
            try:
                calendar = stock.calendar
                if calendar is not None:
                    result["next_earnings_date"] = calendar.get("Earnings Date")
            except Exception:
                result["next_earnings_date"] = None
            
            logger.info(f"Fetched earnings for {ticker}")
            cache.set(cache_key, result, ttl=config.FUNDAMENTALS_CACHE_TTL)
            return result
            
        except Exception as e:
            logger.error(f"Error fetching earnings for {ticker}: {e}")
            return {"ticker": ticker, "error": str(e)}
    
    def get_revenue(self, ticker: str) -> Dict[str, Any]:
        """
        Get revenue data and estimates.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Dict with revenue data
        """
        try:
            cache_key = f"fundamentals:revenue:{ticker}"
            cached = cache.get(cache_key)
            if isinstance(cached, dict) and cached.get("ticker") == ticker:
                return cached

            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Get quarterly financials
            quarterly = stock.quarterly_financials
            
            result = {
                "ticker": ticker,
                "total_revenue": info.get("totalRevenue"),
                "revenue_per_share": info.get("revenuePerShare"),
                "revenue_growth": info.get("revenueGrowth"),
                "quarterly_revenue": None,
            }
            
            if quarterly is not None and not quarterly.empty:
                if "Total Revenue" in quarterly.index:
                    result["quarterly_revenue"] = quarterly.loc["Total Revenue"].to_dict()
            
            logger.info(f"Fetched revenue for {ticker}")
            cache.set(cache_key, result, ttl=config.FUNDAMENTALS_CACHE_TTL)
            return result
            
        except Exception as e:
            logger.error(f"Error fetching revenue for {ticker}: {e}")
            return {"ticker": ticker, "error": str(e)}
    
    def get_pe_vs_industry(self, ticker: str) -> Dict[str, Any]:
        """
        Compare P/E ratio to industry average.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Dict with P/E comparison:
                - pe_ratio: Stock's P/E
                - industry: Industry name
                - industry_pe: Industry average P/E (estimated)
                - pe_below_industry: True if P/E is below industry avg
        """
        try:
            cache_key = f"fundamentals:pe_vs_industry:{ticker}"
            cached = cache.get(cache_key)
            if isinstance(cached, dict) and cached.get("ticker") == ticker:
                return cached

            stock = yf.Ticker(ticker)
            info = stock.info
            
            pe_ratio = info.get("trailingPE")
            industry = info.get("industry", "Unknown")
            sector = info.get("sector", "Unknown")
            
            # Industry P/E estimates (simplified)
            # In production, would use sector-specific data
            SECTOR_PE_ESTIMATES = {
                "Technology": 28,
                "Financial Services": 15,
                "Healthcare": 22,
                "Consumer Cyclical": 20,
                "Consumer Defensive": 22,
                "Basic Materials": 15,
                "Energy": 12,
                "Industrials": 18,
                "Utilities": 16,
                "Real Estate": 35,
                "Communication Services": 20,
            }
            
            industry_pe = SECTOR_PE_ESTIMATES.get(sector, 20)
            
            result = {
                "ticker": ticker,
                "pe_ratio": pe_ratio,
                "sector": sector,
                "industry": industry,
                "sector_avg_pe": industry_pe,
                "pe_below_sector": pe_ratio < industry_pe if pe_ratio else None,
                "pe_difference": pe_ratio - industry_pe if pe_ratio else None,
                "pe_discount_percent": ((industry_pe - pe_ratio) / industry_pe * 100) if pe_ratio else None,
            }
            
            logger.info(f"Fetched P/E comparison for {ticker}")
            cache.set(cache_key, result, ttl=config.FUNDAMENTALS_CACHE_TTL)
            return result
            
        except Exception as e:
            logger.error(f"Error comparing P/E for {ticker}: {e}")
            return {"ticker": ticker, "error": str(e)}
    
    def get_fundamental_signals(self, ticker: str) -> Dict[str, Any]:
        """
        Get fundamental signals for trading decisions.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Dict with fundamental signals:
                - pe_signal: True if P/E below industry
                - eps_beat_signal: True if latest EPS beat
                - revenue_beat_signal: True if revenue beat
                - fundamental_score: Combined score 0-1
        """
        try:
            fundamentals = self.get_fundamentals(ticker)
            earnings = self.get_earnings(ticker)
            pe_comparison = self.get_pe_vs_industry(ticker)
            
            signals = {
                "ticker": ticker,
                "pe_signal": pe_comparison.get("pe_below_sector", False),
                "eps_beat_signal": earnings.get("eps_beat", False),
                "revenue_growth_signal": (
                    fundamentals.get("revenue_growth") and 
                    fundamentals.get("revenue_growth", 0) > 0
                ),
                "profit_margin_signal": (
                    fundamentals.get("profit_margin") and 
                    fundamentals.get("profit_margin", 0) > 0.1
                ),
            }
            
            # Calculate fundamental score (0-1)
            signal_values = [
                signals["pe_signal"],
                signals["eps_beat_signal"],
                signals["revenue_growth_signal"],
                signals["profit_margin_signal"],
            ]
            
            true_count = sum(1 for v in signal_values if v)
            signals["fundamental_score"] = true_count / len(signal_values)
            
            # Add details
            signals["details"] = {
                "pe_ratio": fundamentals.get("pe_ratio"),
                "sector_avg_pe": pe_comparison.get("sector_avg_pe"),
                "eps_beat": earnings.get("eps_beat"),
                "eps_surprise_percent": earnings.get("eps_surprise_percent"),
                "revenue_growth": fundamentals.get("revenue_growth"),
                "profit_margin": fundamentals.get("profit_margin"),
            }
            
            logger.info(f"Generated fundamental signals for {ticker}: score={signals['fundamental_score']:.2f}")
            return signals
            
        except Exception as e:
            logger.error(f"Error generating fundamental signals for {ticker}: {e}")
            return {"ticker": ticker, "error": str(e), "fundamental_score": 0}


# Singleton instance
fundamentals_fetcher = FundamentalsFetcher()
