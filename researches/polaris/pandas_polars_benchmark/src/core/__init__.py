"""
Core модуль системы бенчмаркинга.
Содержит основные компоненты для управления выполнением.
"""

from researches.polaris.pandas_polars_benchmark.src.core.config import Config, LibraryConfig
from researches.polaris.pandas_polars_benchmark.src.core.config_schema import ConfigSchema, ValidationError
from researches.polaris.pandas_polars_benchmark.src.core.checkpoint import CheckpointManager, BenchmarkState
from researches.polaris.pandas_polars_benchmark.src.core.progress import ProgressTracker, OperationTiming, create_progress_tracker
from researches.polaris.pandas_polars_benchmark.src.core.benchmark_runner import BenchmarkRunner, BenchmarkTask, create_benchmark_runner

__all__ = [
    # Configuration
    'Config',
    'LibraryConfig',
    'ConfigSchema',
    'ValidationError',
    
    # Checkpoint management
    'CheckpointManager',
    'BenchmarkState',
    
    # Progress tracking
    'ProgressTracker',
    'OperationTiming',
    'create_progress_tracker',
    
    # Main runner
    'BenchmarkRunner',
    'BenchmarkTask',
    'create_benchmark_runner',
]

# Версия модуля
__version__ = '1.0.0'