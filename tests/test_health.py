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
    """Verify models health check returns boolean map for core ML/DL libraries."""
    models = HealthCheck.check_models()
    assert isinstance(models, dict)
    assert models.get("torch") is True
    assert models.get("xgboost") is True
    assert models.get("sklearn") is True
    assert models.get("ready") is True


def test_health_run_all():
    """Verify run_all returns all expected system health sections."""
    report = HealthCheck.run_all()
    assert isinstance(report, dict)
    for key in ("gpu", "cpu", "memory", "database", "models", "api_keys", "internet"):
        assert key in report
