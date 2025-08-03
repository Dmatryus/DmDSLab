"""
Модуль генерации отчетов для системы бенчмаркинга.
"""

from .data_processor import (
    DataProcessor,
    ProcessedData,
    AggregationLevel,
    MetricType
)

__all__ = [
    "DataProcessor",
    "ProcessedData", 
    "AggregationLevel",
    "MetricType"
]
