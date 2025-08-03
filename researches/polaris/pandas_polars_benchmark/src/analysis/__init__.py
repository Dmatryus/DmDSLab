"""
Модуль статистического анализа для бенчмаркинга.
"""

from .outlier_detector import (
    OutlierDetector,
    OutlierMethod,
    OutlierResult,
    compare_outlier_methods,
)

__all__ = [
    # Outlier detection
    "OutlierDetector",
    "OutlierMethod",
    "OutlierResult",
    "compare_outlier_methods",
]
