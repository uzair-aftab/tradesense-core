"""
System Health Check

Verifies critical system components for the HFP platform.
"""

import os
from typing import Dict, Any

from src.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HealthCheck:
    """System health diagnostics for the HFP platform."""

    @staticmethod
    def check_gpu() -> Dict[str, Any]:
        """Check GPU availability and memory."""
        try:
            import torch

            health = {
                "available": torch.cuda.is_available(),
                "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
                "memory_allocated": 0,
                "memory_reserved": 0,
            }

            if health["available"]:
                health["memory_allocated"] = f"{torch.cuda.memory_allocated(0) / 1024**2:.2f} MB"
                health["memory_reserved"] = f"{torch.cuda.memory_reserved(0) / 1024**2:.2f} MB"

            return health
        except ImportError:
            return {"available": False, "device_count": 0, "device_name": "N/A (torch not installed)"}

    @staticmethod
    def check_models() -> Dict[str, bool]:
        """Check if trained model files exist."""
        model_dir = config.MODELS_DIR
        files = {
            "lstm": model_dir / "lstm_quantile.pt",
            "xgb_p10": model_dir / "xgb_quantile" / "p10.json",
            "xgb_p50": model_dir / "xgb_quantile" / "p50.json",
            "xgb_p90": model_dir / "xgb_quantile" / "p90.json",
        }
        return {name: path.exists() for name, path in files.items()}

    @staticmethod
    def check_api_keys() -> Dict[str, bool]:
        """Check which API keys are configured."""
        return {
            "finnhub": config.has_finnhub_key(),
            "reddit": config.has_reddit_credentials(),
            "marketaux": bool(config.MARKETAUX_API_KEY),
        }

    @staticmethod
    def check_internet() -> bool:
        """Check internet connectivity."""
        try:
            import requests
            requests.get("https://google.com", timeout=3)
            return True
        except Exception:
            return False

    @staticmethod
    def check_memory() -> Dict[str, str]:
        """Check system RAM usage."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            return {
                "total": f"{mem.total / 1024**3:.1f} GB",
                "available": f"{mem.available / 1024**3:.1f} GB",
                "percent": f"{mem.percent}%",
            }
        except ImportError:
            return {"total": "N/A", "available": "N/A", "percent": "N/A"}

    @staticmethod
    def check_cache() -> Dict[str, Any]:
        """Check cache database status."""
        db_path = config.CACHE_DB_PATH
        exists = os.path.exists(db_path)
        size = os.path.getsize(db_path) if exists else 0
        return {
            "exists": exists,
            "path": db_path,
            "size": f"{size / 1024:.1f} KB" if exists else "0 KB",
        }

    @classmethod
    def run_all(cls) -> Dict[str, Any]:
        """Run all health checks and return a summary."""
        report = {
            "gpu": cls.check_gpu(),
            "models": cls.check_models(),
            "api_keys": cls.check_api_keys(),
            "internet": cls.check_internet(),
            "memory": cls.check_memory(),
            "cache": cls.check_cache(),
        }
        logger.info("Health Check completed: %s", report)
        return report

    @classmethod
    def summary_emoji(cls) -> str:
        """Return a one-line emoji health summary for the UI."""
        report = cls.run_all()
        gpu = "🟢" if report["gpu"]["available"] else "🔴"
        net = "🟢" if report["internet"] else "🔴"
        models_ready = all(report["models"].values())
        mdl = "🟢" if models_ready else "🟡"
        return f"GPU {gpu}  Net {net}  Models {mdl}"


# Singleton
health_check = HealthCheck()
