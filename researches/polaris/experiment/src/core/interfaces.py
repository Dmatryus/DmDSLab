"""
Базовые интерфейсы (протоколы) для компонентов системы
"""
from abc import ABC, abstractmethod
from typing import Protocol, List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .constants import Framework, OperationType, ExecutionStatus


@dataclass
class ColumnSpec:
    """Спецификация колонки для генерации"""
    name: str
    dtype: str
    params: Dict[str, Any]


@dataclass
class DatasetMetadata:
    """Метаданные датасета"""
    filename: str
    size_bytes: int
    rows: int
    columns: int
    column_info: Dict[str, Dict[str, Any]]
    generation_params: Dict[str, Any]
    created_at: datetime


@dataclass
class Dataset:
    """Датасет с данными и метаданными"""
    data: Any  # DataFrame (pandas или polars)
    metadata: DatasetMetadata
    
    def save_csv(self, path: Path):
        """Сохранение в CSV формат"""
        pass
    
    def save_parquet(self, path: Path):
        """Сохранение в Parquet формат"""
        pass
    
    def save_feather(self, path: Path):
        """Сохранение в Feather формат"""
        pass


class IDataGenerator(Protocol):
    """Интерфейс для генератора данных"""
    
    def generate_column(self, spec: ColumnSpec) -> Any:
        """Генерация колонки по спецификации"""
        ...
    
    def calculate_row_count(self, target_size: int) -> int:
        """Расчет количества строк для достижения целевого размера"""
        ...


@dataclass
class MemoryMetrics:
    """Метрики использования памяти"""
    peak_memory_mb: float
    start_memory_mb: float
    end_memory_mb: float
    memory_profile: Optional[List[Tuple[float, float]]] = None  # [(time, memory)]


@dataclass
class TimeMetrics:
    """Метрики времени выполнения"""
    total_seconds: float
    iterations: int
    mean_seconds: float
    std_seconds: float
    min_seconds: float
    max_seconds: float


@dataclass
class OperationResult:
    """Результат выполнения операции"""
    operation_name: str
    time_metrics: TimeMetrics
    memory_metrics: MemoryMetrics
    status: ExecutionStatus
    error_message: Optional[str] = None
    result_shape: Optional[Tuple[int, int]] = None  # (rows, columns)


@dataclass
class BenchmarkResult:
    """Результат выполнения бенчмарка"""
    dataset_name: str
    framework: Framework
    operations: List[OperationResult]
    total_time: float
    started_at: datetime
    completed_at: datetime
    
    @property
    def summary(self) -> Dict[str, Any]:
        """Сводка результатов"""
        successful_ops = [op for op in self.operations if op.status == ExecutionStatus.COMPLETED]
        failed_ops = [op for op in self.operations if op.status == ExecutionStatus.FAILED]
        
        return {
            'total_operations': len(self.operations),
            'successful_operations': len(successful_ops),
            'failed_operations': len(failed_ops),
            'total_time': self.total_time,
            'avg_operation_time': sum(op.time_metrics.mean_seconds for op in successful_ops) / len(successful_ops) if successful_ops else 0,
            'peak_memory_mb': max((op.memory_metrics.peak_memory_mb for op in successful_ops), default=0)
        }


class IBenchmark(Protocol):
    """Интерфейс для бенчмарка"""
    
    def run(self, dataset: Dataset) -> BenchmarkResult:
        """Запуск бенчмарка на датасете"""
        ...
    
    def get_operations(self) -> List[str]:
        """Получение списка поддерживаемых операций"""
        ...


class IReporter(Protocol):
    """Интерфейс для генератора отчетов"""
    
    def generate_report(self, results: List[BenchmarkResult]) -> None:
        """Генерация отчета по результатам"""
        ...


@dataclass
class Operation:
    """Операция для выполнения в бенчмарке"""
    name: str
    type: OperationType
    params: Dict[str, Any]
    
    def execute(self, df: Any) -> Any:
        """Выполнение операции над DataFrame"""
        raise NotImplementedError


@dataclass
class ResourceMetrics:
    """Общие метрики ресурсов"""
    memory: MemoryMetrics
    time: TimeMetrics
    cpu_percent: Optional[float] = None
    disk_io_mb: Optional[float] = None


@dataclass
class FailedBenchmark:
    """Информация об упавшем бенчмарке"""
    dataset_name: str
    framework: str
    operation: str
    error_message: str
    traceback: str
    timestamp: datetime
    retry_count: int = 0


@dataclass
class ExperimentState:
    """Состояние эксперимента для checkpoint'ов"""
    completed_benchmarks: List[str]  # Список завершенных "dataset:framework" 
    failed_benchmarks: List[FailedBenchmark]
    current_progress: Dict[str, Any]
    started_at: datetime
    last_checkpoint: datetime
    
    def can_resume(self) -> bool:
        """Проверка возможности возобновления"""
        return bool(self.completed_benchmarks or self.failed_benchmarks)
    
    def get_pending_benchmarks(self, all_benchmarks: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Получение списка невыполненных бенчмарков"""
        completed_set = set(self.completed_benchmarks)
        return [
            (dataset, framework) 
            for dataset, framework in all_benchmarks 
            if f"{dataset}:{framework}" not in completed_set
        ]


class IMemoryTracker(Protocol):
    """Интерфейс для трекера памяти"""
    
    def start_tracking(self) -> None:
        """Начало отслеживания памяти"""
        ...
    
    def stop_tracking(self) -> MemoryMetrics:
        """Остановка отслеживания и получение метрик"""
        ...
    
    def get_current_memory(self) -> float:
        """Текущее использование памяти в МБ"""
        ...


class ITimer(Protocol):
    """Интерфейс для таймера"""
    
    def start(self) -> None:
        """Запуск таймера"""
        ...
    
    def stop(self) -> float:
        """Остановка таймера и получение времени"""
        ...
    
    def elapsed(self) -> float:
        """Прошедшее время без остановки"""
        ...


class IResultManager(Protocol):
    """Интерфейс для менеджера результатов"""
    
    def save_intermediate(self, result: OperationResult) -> None:
        """Сохранение промежуточного результата"""
        ...
    
    def save_checkpoint(self, state: ExperimentState) -> None:
        """Сохранение checkpoint'а"""
        ...
    
    def load_checkpoint(self) -> Optional[ExperimentState]:
        """Загрузка checkpoint'а"""
        ...
    
    def version_results(self) -> str:
        """Версионирование результатов с timestamp"""
        ...