"""
Adaptive Thresholds Module

Provides dynamic threshold adjustment based on market regime and volatility.
This module wraps the existing AdaptiveThresholds class for use in signal generation.
"""

from src.analysis.adaptive import adaptive_thresholds

# Re-export the singleton for signal generation
__all__ = ["adaptive_thresholds"]
