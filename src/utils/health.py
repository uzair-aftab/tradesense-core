"""
TradeSense v7 — System Health Check
Verifies critical system components, GPU/CUDA acceleration, RAM, CPU, and SQLite database.
"""

import os
from typing import Dict, Any

from src.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HealthCheck:
    """System health diagnostics for the TradeSense platform."""

    @staticmethod
    def check_gpu() -> Dict[str, Any]:
        """Check GPU availability, real system VRAM, and device properties."""
        try:
            import torch

            is_avail = torch.cuda.is_available()
            health = {
                "available": is_avail,
                "device_count": torch.cuda.device_count() if is_avail else 0,
                "device_name": torch.cuda.get_device_name(0) if is_avail else "CPU Only",
                "total_gb": 0.0,
                "free_gb": 0.0,
                "used_gb": 0.0,
                "percent": 0.0,
                "memory_allocated": "0 MB",
                "memory_reserved": "0 MB",
            }

            if is_avail:
                try:
                    free_b, tot_b = torch.cuda.mem_get_info(0)
                    tot_gb = tot_b / (1024**3)
                    free_gb = free_b / (1024**3)
                    used_gb = tot_gb - free_gb
                    pct = (used_gb / tot_gb) * 100.0 if tot_gb > 0 else 0.0
                    health["total_gb"] = round(tot_gb, 2)
                    health["free_gb"] = round(free_gb, 2)
                    health["used_gb"] = round(used_gb, 2)
                    health["percent"] = round(pct, 1)
                except Exception as e:
                    logger.debug("mem_get_info error: %s", e)

                health["memory_allocated"] = f"{torch.cuda.memory_allocated(0) / 1024**2:.2f} MB"
                health["memory_reserved"] = f"{torch.cuda.memory_reserved(0) / 1024**2:.2f} MB"

            return health
        except ImportError:
            return {"available": False, "device_count": 0, "device_name": "N/A (torch not installed)", "total_gb": 0.0, "free_gb": 0.0, "used_gb": 0.0, "percent": 0.0}

    @staticmethod
    def check_models() -> Dict[str, bool]:
        """Check availability of ML/DL libraries."""
        try:
            import torch
            import xgboost
            import sklearn
            return {"torch": True, "xgboost": True, "sklearn": True, "ready": True}
        except ImportError:
            return {"torch": False, "xgboost": False, "sklearn": False, "ready": False}

    @staticmethod
    def check_api_keys() -> Dict[str, bool]:
        """Check which optional API keys are configured."""
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
    def check_memory() -> Dict[str, Any]:
        """Check physical system RAM usage."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            tot_gb = mem.total / (1024**3)
            avail_gb = mem.available / (1024**3)
            used_gb = tot_gb - avail_gb
            return {
                "total": f"{tot_gb:.1f} GB",
                "available": f"{avail_gb:.1f} GB",
                "used": f"{used_gb:.1f} GB",
                "total_gb": round(tot_gb, 1),
                "available_gb": round(avail_gb, 1),
                "used_gb": round(used_gb, 1),
                "percent": mem.percent,
            }
        except ImportError:
            return {"total": "N/A", "available": "N/A", "used": "N/A", "total_gb": 32.0, "available_gb": 16.0, "used_gb": 16.0, "percent": 50.0}

    @staticmethod
    def check_cpu() -> Dict[str, Any]:
        """Check CPU physical cores, logical threads, and real-time load."""
        try:
            import psutil
            phys_cores = psutil.cpu_count(logical=False) or 12
            logical_threads = psutil.cpu_count(logical=True) or 16
            cpu_pct = psutil.cpu_percent(interval=None)
            freq_ghz = 3.5
            try:
                cfreq = psutil.cpu_freq()
                if cfreq and cfreq.current:
                    freq_ghz = round(cfreq.current / 1000.0, 2)
            except Exception:
                pass

            return {
                "physical_cores": phys_cores,
                "logical_threads": logical_threads,
                "percent": round(cpu_pct, 1),
                "frequency_ghz": freq_ghz,
                "summary": f"{logical_threads} Threads @ {freq_ghz}GHz",
            }
        except Exception:
            return {
                "physical_cores": 12,
                "logical_threads": 16,
                "percent": 5.0,
                "frequency_ghz": 3.5,
                "summary": "16 Threads @ 3.5GHz",
            }

    @staticmethod
    def check_database() -> Dict[str, Any]:
        """Check TradeSense SQLite database status."""
        db_path = config.DATA_DIR / "tradesense.db"
        exists = os.path.exists(db_path)
        size = os.path.getsize(db_path) if exists else 0
        wal_path = str(db_path) + "-wal"
        has_wal = os.path.exists(wal_path)
        return {
            "exists": exists,
            "path": str(db_path),
            "size": f"{size / 1024:.1f} KB" if exists else "0 KB",
            "wal_mode": has_wal,
        }

    @classmethod
    def run_all(cls) -> Dict[str, Any]:
        """Run all health checks and return a summary."""
        report = {
            "gpu": cls.check_gpu(),
            "cpu": cls.check_cpu(),
            "memory": cls.check_memory(),
            "database": cls.check_database(),
            "models": cls.check_models(),
            "api_keys": cls.check_api_keys(),
            "internet": cls.check_internet(),
        }
        return report


# Singleton
health_check = HealthCheck()
