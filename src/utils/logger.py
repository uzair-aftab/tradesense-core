"""
TradeSense Logging Utility

Provides consistent logging across all modules with
file and console output support.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "tradesense",
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up and return a configured logger.
    
    Args:
        name: Logger name (default: "tradesense")
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        format_string: Custom format string for log messages
    
    Returns:
        Configured logging.Logger instance
    
    Example:
        >>> logger = setup_logger("my_module", level="DEBUG")
        >>> logger.info("Scanner started")
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Set level
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # Default format
    if format_string is None:
        format_string = (
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )
    
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
    
    # Ensure console output handles UTF-8 on Windows
    if sys.platform == "win32":
        try:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            if hasattr(sys.stderr, "reconfigure"):
                sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "tradesense") -> logging.Logger:
    """
    Get an existing logger or create one with default settings.
    
    Args:
        name: Logger name
    
    Returns:
        logging.Logger instance
    """
    logger = logging.getLogger(name)
    
    # If logger has no handlers, set it up with defaults
    if not logger.handlers:
        from src.config import config
        return setup_logger(
            name=name,
            level=config.LOG_LEVEL,
            log_file=config.LOG_FILE_PATH if config.LOG_FILE_PATH else None
        )
    
    return logger


# Create default logger
logger = get_logger()
