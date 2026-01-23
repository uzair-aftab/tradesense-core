"""
System Health Check

Verifies critical system components start up correctly.
"""

import torch
import os
import requests
import psutil
from typing import Dict, Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


class HealthCheck:
    """System health diagnostics."""
    
    @staticmethod
    def check_gpu() -> Dict[str, Any]:
        """Check GPU availability and memory."""
        health = {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "memory_allocated": 0,
            "memory_reserved": 0
        }
        
        if health["available"]:
            health["memory_allocated"] = f"{torch.cuda.memory_allocated(0) / 1024**2:.2f} MB"
            health["memory_reserved"] = f"{torch.cuda.memory_reserved(0) / 1024**2:.2f} MB"
            
        return health
    
    @staticmethod
    def check_files() -> Dict[str, bool]:
        """Check if critical model files exist."""
        files = {
            "xgboost": "models/xgb_model.json",
            "lstm": "models/lstm_model.pt",
            "transformer": "models/transformer_model.pt",
            "tracked_preds": "data/tracked_predictions.json"
        }
        
        status = {}
        for name, path in files.items():
            status[name] = os.path.exists(path)
            
        return status
        
    @staticmethod
    def check_internet() -> bool:
        """Check internet connectivity."""
        try:
            requests.get("https://google.com", timeout=3)
            return True
        except Exception:
            return False
            
    @staticmethod
    def check_memory() -> Dict[str, str]:
        """Check system RAM usage."""
        mem = psutil.virtual_memory()
        return {
            "total": f"{mem.total / 1024**3:.1f} GB",
            "available": f"{mem.available / 1024**3:.1f} GB",
            "percent": f"{mem.percent}%"
        }
    
    @classmethod
    def run_all(cls) -> Dict[str, Any]:
        """Run all health checks."""
        report = {
            "gpu": cls.check_gpu(),
            "files": cls.check_files(),
            "internet": cls.check_internet(),
            "memory": cls.check_memory()
        }
        logger.info(f"Health Check: {report}")
        return report


# Health singleton
health_check = HealthCheck()
