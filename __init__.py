# time_series_classification/__init__.py
"""Time Series Classification Framework."""

__version__ = "0.1.0"

# Import core components for easy access
from time_series_classification.utils.logger import setup_logger, LoggerMixin

__all__ = [
    "__version__",
    "setup_logger",
    "LoggerMixin",
]