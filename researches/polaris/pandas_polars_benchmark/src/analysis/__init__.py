"""
Модуль статистического анализа для бенчмаркинга.
"""

from .outlier_detector import (
    OutlierDetector,
    OutlierMethod,
    OutlierResult,
    compare_outlier_methods,
)

from .statistics_calculator import (
    StatisticsCalculator,
    DescriptiveStats,
    NormalityTest,
    format_stats_report,
)

from .comparison_engine import (
    ComparisonEngine,
    ComparisonMetric,
    SignificanceLevel,
    ComparisonResult,
    ComparisonMatrix,
    format_comparison_report,
)

__all__ = [
    # Outlier detection
    "OutlierDetector",
    "OutlierMethod",
    "OutlierResult",
    "compare_outlier_methods",
    
    # Statistics calculation
    "StatisticsCalculator",
    "DescriptiveStats",
    "NormalityTest",
    "format_stats_report",
    
    # Comparison engine
    "ComparisonEngine",
    "ComparisonMetric",
    "SignificanceLevel",
    "ComparisonResult",
    "ComparisonMatrix",
    "format_comparison_report",
]
