"""
Модуль генерации отчетов для системы бенчмаркинга.
"""

from .data_processor import (
    DataProcessor,
    ProcessedData,
    AggregationLevel,
    MetricType
)

from .visualization_engine import (
    VisualizationEngine,
    ChartConfig,
    ChartType,
    ColorPalette
)

from .html_renderer import (
    HTMLRenderer,
    ReportConfig,
    ReportSection
)

__all__ = [
    # Data processing
    "DataProcessor",
    "ProcessedData", 
    "AggregationLevel",
    "MetricType",
    
    # Visualization
    "VisualizationEngine",
    "ChartConfig",
    "ChartType", 
    "ColorPalette",
    
    # HTML rendering
    "HTMLRenderer",
    "ReportConfig",
    "ReportSection"
]