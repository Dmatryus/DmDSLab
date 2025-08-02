"""
Константы и пути для проекта бенчмарков Pandas vs Polars
"""
from pathlib import Path
from enum import Enum, auto
from typing import Final

# Базовые пути
PROJECT_ROOT: Final[Path] = Path(__file__).parent.parent.parent
SRC_ROOT: Final[Path] = PROJECT_ROOT / "src"
CONFIG_DIR: Final[Path] = PROJECT_ROOT / "config"
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
GENERATED_DATA_DIR: Final[Path] = DATA_DIR / "generated"
RESULTS_DIR: Final[Path] = PROJECT_ROOT / "results"
BENCHMARKS_DIR: Final[Path] = RESULTS_DIR / "benchmarks"
REPORTS_DIR: Final[Path] = RESULTS_DIR / "reports"
CHECKPOINTS_DIR: Final[Path] = RESULTS_DIR / "checkpoints"
LOGS_DIR: Final[Path] = PROJECT_ROOT / "logs"

# Форматы файлов
class FileFormat(Enum):
    CSV = "csv"
    PARQUET = "parquet"
    FEATHER = "feather"

# Размеры датасетов
class DatasetSize(Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    XLARGE = "xlarge"

# Формы датасетов
class DatasetShape(Enum):
    NARROW_LONG = "narrow_long"
    WIDE_SHORT = "wide_short"
    BALANCED = "balanced"
    ULTRA_WIDE = "ultra_wide"

# Типы данных
class DataTypeMix(Enum):
    NUMERIC_HEAVY = "numeric_heavy"
    STRING_HEAVY = "string_heavy"
    MIXED_BALANCED = "mixed_balanced"
    TEMPORAL_FOCUS = "temporal_focus"

# Фреймворки
class Framework(Enum):
    PANDAS_NUMPY = "pandas_numpy"
    PANDAS_PYARROW = "pandas_pyarrow"
    POLARS_EAGER = "polars_eager"
    POLARS_LAZY = "polars_lazy"

# Типы операций
class OperationType(Enum):
    READ = "read"
    FILTER = "filter"
    AGGREGATE = "aggregate"
    JOIN = "join"
    SORT = "sort"
    STRING = "string"
    COLUMN = "column"

# Уровни логирования
class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"

# Параметры по умолчанию
DEFAULT_SEED: Final[int] = 42
DEFAULT_ITERATIONS: Final[int] = 3
DEFAULT_WARMUP_ITERATIONS: Final[int] = 1
DEFAULT_CHUNK_SIZE: Final[int] = 10_000
MAX_STRING_LENGTH: Final[int] = 1000
MEMORY_PROFILING_INTERVAL: Final[float] = 0.1  # секунды

# Метаданные
METADATA_FILENAME: Final[str] = "metadata.json"
CHECKPOINT_EXTENSION: Final[str] = ".checkpoint"
REPORT_TEMPLATE: Final[str] = "benchmark_report.html"

# Версионирование
VERSION: Final[str] = "1.0.0"
SCHEMA_VERSION: Final[str] = "1.0"

# Лимиты
MAX_MEMORY_GB: Final[int] = 16
MAX_COLUMNS: Final[int] = 10_000
MAX_ROWS: Final[int] = 100_000_000
MIN_ROWS: Final[int] = 1_000

# Форматирование
TIMESTAMP_FORMAT: Final[str] = "%Y%m%d_%H%M%S"
LOG_FORMAT: Final[str] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"

# Charset для строковых данных
CHARSET_ALPHANUMERIC: Final[str] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
CHARSET_ASCII: Final[str] = "".join(chr(i) for i in range(32, 127))

# Распределения для числовых данных
class Distribution(Enum):
    UNIFORM = "uniform"
    NORMAL = "normal"
    EXPONENTIAL = "exponential"

# Статусы выполнения
class ExecutionStatus(Enum):
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    SKIPPED = auto()

# Сообщения об ошибках
ERROR_MESSAGES = {
    "CONFIG_NOT_FOUND": "Configuration file not found: {path}",
    "INVALID_CONFIG": "Invalid configuration: {error}",
    "DATA_GENERATION_FAILED": "Data generation failed: {error}",
    "BENCHMARK_FAILED": "Benchmark failed: {framework} - {operation} - {error}",
    "INSUFFICIENT_MEMORY": "Insufficient memory for operation",
    "CHECKPOINT_CORRUPTED": "Checkpoint file is corrupted: {path}",
}