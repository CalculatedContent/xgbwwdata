"""xgbwwdata: dataset scanning + loading utilities for XGBoost2WW experiments."""

import logging

from .config import Filters, ScanOptions
from .registry import scan_datasets, load_dataset


def enable_logging(level: int = logging.INFO) -> None:
    """Enable console logging for xgbwwdata progress messages."""
    package_logger = logging.getLogger("xgbwwdata")
    if not package_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        package_logger.addHandler(handler)
    package_logger.setLevel(level)
    package_logger.propagate = False

__all__ = ["Filters", "ScanOptions", "scan_datasets", "load_dataset", "enable_logging"]
