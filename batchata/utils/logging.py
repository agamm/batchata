"""Logging utilities for batchata."""

import logging
import sys
from typing import Optional


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Get a configured logger for the given module name.
    
    Args:
        name: The name of the logger (typically __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Only configure if no handlers exist
    if not logger.handlers:
        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(handler)
        
        # Set level
        logger.setLevel(getattr(logging, level.upper()))
        
        # Prevent propagation to avoid duplicate logs
        logger.propagate = False
    
    return logger


def set_log_level(logger_name: Optional[str] = None, level: str = "INFO"):
    """Set logging level for a specific logger or all batchata loggers.
    
    Args:
        logger_name: Specific logger name or None for all batchata loggers
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    if logger_name:
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, level.upper()))
    else:
        # Set for all batchata loggers
        root_logger = logging.getLogger("batchata")
        root_logger.setLevel(getattr(logging, level.upper()))
        
        # Also set level for all existing child loggers
        for name, logger in logging.Logger.manager.loggerDict.items():
            if isinstance(logger, logging.Logger) and name.startswith("batchata."):
                logger.setLevel(getattr(logging, level.upper()))