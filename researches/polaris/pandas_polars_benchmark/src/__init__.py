"""
Pandas vs Polars Benchmark System

Система для комплексного сравнения производительности библиотек Pandas и Polars.
"""

# Версия пакета
__version__ = "1.0.0"

from .core import Config
from .data import DataGenerator, DataLoader, DataSaver, DatasetInfo
from .utils import setup_logging, get_logger

__all__ = [
    # Core
    "Config",
    # Data
    "DataGenerator",
    "DataLoader",
    "DataSaver",
    "DatasetInfo",
    # Utils
    "setup_logging",
    "get_logger",
]
