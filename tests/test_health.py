"""
Tests for System Health Diagnostics
"""

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pytest
from src.utils.health import HealthCheck


def test_gpu_health_check():
    """Verify GPU check returns valid status and device information."""
    gpu = HealthCheck.check_gpu()
    assert isinstance(gpu, dict)
    assert "available" in gpu
    assert "device_name" in gpu


def test_models_health_check():
    """Verify models health check returns boolean map for all model components."""
    models = HealthCheck.check_models()
    assert isinstance(models, dict)
    assert "lstm" in models
    assert "xgb_p10" in models
    assert "xgb_p50" in models
    assert "xgb_p90" in models


def test_health_summary_emoji():
    """Verify summary_emoji returns well-formatted string with status indicators."""
    summary = HealthCheck.summary_emoji()
    assert "GPU" in summary
    assert "Net" in summary
    assert "Models" in summary
