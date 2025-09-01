# utils/logger.py
"""
Logging utilities for Time Series Classification project.
Provides colored console output and file logging capabilities.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

# ANSI escape sequences for colors
BLUE = "\033[94m"        # Info
GREEN = "\033[92m"       # Success
YELLOW = "\033[93m"      # Debug
ORANGE = "\033[38;5;208m" # Warning
RED = "\033[91m"         # Error
PURPLE = "\033[95m"      # Header/Section
CYAN = "\033[96m"        # Metrics
RESET = "\033[0m"
BOLD = "\033[1m"


class ColorFormatter(logging.Formatter):
    """Custom formatter to add colors based on log level."""
    
    LEVEL_COLORS = {
        logging.DEBUG: YELLOW,
        logging.INFO: BLUE,
        logging.WARNING: ORANGE,
        logging.ERROR: RED,
        logging.CRITICAL: f"{BOLD}{RED}"
    }
    
    def format(self, record):
        log_msg = super().format(record)
        color = self.LEVEL_COLORS.get(record.levelno, "")
        
        # Add special formatting for specific message types
        if hasattr(record, 'msg_type'):
            if record.msg_type == 'success':
                color = GREEN
            elif record.msg_type == 'header':
                color = f"{BOLD}{PURPLE}"
            elif record.msg_type == 'metric':
                color = CYAN
                
        return f"{color}{log_msg}{RESET}"


def setup_logger(
    name: str = "TSC",
    level: int = logging.DEBUG,
    log_dir: Optional[str] = None,
    log_to_file: bool = True
) -> logging.Logger:
    """
    Set up logger with color formatting for console and optional file output.
    
    Args:
        name: Logger name
        level: Logging level
        log_dir: Directory for log files (creates timestamped file)
        log_to_file: Whether to also log to file
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Console handler with color
    console_handler = logging.StreamHandler()
    console_formatter = ColorFormatter(
        fmt="%(asctime)s [%(levelname)-8s] %(name)s - %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (no colors)
    if log_to_file and log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = Path(log_dir) / f"tsc_{timestamp}.log"
        
        file_handler = logging.FileHandler(file_path)
        file_formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)-8s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {file_path}")
    
    logger.propagate = False
    return logger


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get or create logger for this class."""
        if not hasattr(self, '_logger'):
            self._logger = logging.getLogger(f"TSC.{self.__class__.__name__}")
        return self._logger
    
    def log_success(self, message: str):
        """Log a success message in green."""
        record = self.logger.makeRecord(
            self.logger.name, logging.INFO, "", 0, message, (), None
        )
        record.msg_type = 'success'
        self.logger.handle(record)
    
    def log_header(self, message: str, width: int = 50):
        """Log a header message with formatting."""
        border = "=" * width
        record = self.logger.makeRecord(
            self.logger.name, logging.INFO, "", 0, f"\n{border}\n{message}\n{border}", (), None
        )
        record.msg_type = 'header'
        self.logger.handle(record)
    
    def log_metric(self, metric_name: str, value: float, format_str: str = ".4f"):
        """Log a metric value with special formatting."""
        message = f"{metric_name}: {value:{format_str}}"
        record = self.logger.makeRecord(
            self.logger.name, logging.INFO, "", 0, message, (), None
        )
        record.msg_type = 'metric'
        self.logger.handle(record)