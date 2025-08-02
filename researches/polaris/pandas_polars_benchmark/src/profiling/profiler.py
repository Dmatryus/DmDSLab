"""
Основной модуль для профилирования операций с поддержкой всех ОС.
"""

import os
import sys
import time
import pickle
import tempfile
import traceback
import multiprocessing as mp
from pathlib import Path
from typing import Callable, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import queue

import pandas as pd
import polars as pl
import psutil

from researches.polaris.pandas_polars_benchmark.src.utils.logging import get_logger
from researches.polaris.pandas_polars_benchmark.src.profiling.memory_tracker import (
    MemoryTracker, IsolatedMemoryTracker, MemoryStats
)
from researches.polaris.pandas_polars_benchmark.src.profiling.timer import (
    Timer, TimingResult
)


@dataclass
class ProfileResult:
    """Результат профилирования операции."""
    # Идентификация
    operation_name: str
    library: str
    backend: Optional[str] = None
    dataset_name: str = ""
    dataset_size: int = 0
    
    # Метрики времени
    execution_times: list = field(default_factory=list)
    mean_time: float = 0.0
    median_time: float = 0.0
    std_time: float = 0.0
    cv_time: float = 0.0
    min_time: float = 0.0
    max_time: float = 0.0
    
    # Метрики памяти
    peak_memory_mb: float = 0.0
    avg_memory_mb: float = 0.0
    min_memory_mb: float = 0.0
    max_memory_mb: float = 0.0
    memory_samples: int = 0
    
    # Статус выполнения
    success: bool = True
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    
    # Метаданные
    runs_count: int = 0
    warmup_runs: int = 0
    converged: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    duration_seconds: float = 0.0
    
    # Дополнительные метрики
    result_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует в словарь для сериализации."""
        return asdict(self)
    
    @classmethod
    def from_error(cls, operation_name: str, library: str, 
                   error: Exception, dataset_name: str = "") -> 'ProfileResult':
        """Создает результат с ошибкой."""
        return cls(
            operation_name=operation_name,
            library=library,
            dataset_name=dataset_name,
            success=False,
            error_message=str(error),
            error_traceback=traceback.format_exc()
        )


@dataclass 
class ProfilingConfig:
    """Конфигурация для профилирования."""
    min_runs: int = 3
    max_runs: int = 100
    target_cv: float = 0.05
    warmup_runs: int = 1
    memory_sampling_interval: float = 0.1
    timeout_seconds: int = 300
    isolate_process: bool = field(default_factory=lambda: sys.platform != 'win32')  # False на Windows
    gc_collect: bool = True
    save_intermediate: bool = True


class ProcessResult:
    """Результат выполнения в изолированном процессе."""
    def __init__(self):
        self.timing_result: Optional[TimingResult] = None
        self.memory_stats: Optional[Dict[str, Any]] = None
        self.error: Optional[Exception] = None
        self.result_info: Dict[str, Any] = {}


def _run_isolated_operation(operation_func: Callable,
                           operation_args: tuple,
                           config: ProfilingConfig,
                           result_path: str,
                           ready_event: mp.Event) -> None:
    """
    Выполняет операцию в изолированном процессе.
    
    Args:
        operation_func: Функция операции
        operation_args: Аргументы для операции
        config: Конфигурация профилирования
        result_path: Путь для сохранения результатов
        ready_event: Событие готовности процесса
    """
    try:
        # Сигнализируем о готовности процесса
        ready_event.set()
        
        # Создаем таймер
        timer = Timer(
            min_runs=config.min_runs,
            max_runs=config.max_runs,
            target_cv=config.target_cv,
            warmup_runs=config.warmup_runs,
            gc_collect=config.gc_collect
        )
        
        # Создаем обертку для операции
        def wrapper():
            return operation_func(*operation_args)
        
        # Измеряем время
        timing_result = timer.time_execution(wrapper)
        
        # Создаем результат
        result = ProcessResult()
        result.timing_result = timing_result
        
        # Дополнительная информация о результате
        try:
            # Выполняем операцию еще раз для получения результата
            op_result = operation_func(*operation_args)
            
            # Извлекаем информацию о результате
            if isinstance(op_result, (pd.DataFrame, pl.DataFrame)):
                result.result_info = {
                    'rows': len(op_result),
                    'columns': len(op_result.columns),
                    'dtypes': str(op_result.dtypes) if hasattr(op_result, 'dtypes') else None
                }
            elif isinstance(op_result, dict):
                result.result_info = {k: v for k, v in op_result.items() 
                                    if k in ['rows', 'columns', 'size', 'count']}
        except:
            pass  # Игнорируем ошибки при сборе дополнительной информации
        
        # Сохраняем результат
        with open(result_path, 'wb') as f:
            pickle.dump(result, f)
            
    except Exception as e:
        # Сохраняем ошибку
        result = ProcessResult()
        result.error = e
        
        with open(result_path, 'wb') as f:
            pickle.dump(result, f)


class Profiler:
    """Основной класс для профилирования операций с поддержкой всех ОС."""
    
    def __init__(self, config: Optional[ProfilingConfig] = None):
        """
        Инициализация профайлера.
        
        Args:
            config: Конфигурация профилирования
        """
        self.config = config or ProfilingConfig()
        self.logger = get_logger('profiler')
        
        # Проверяем возможность изоляции процессов
        self._check_isolation_support()
        
        # Директория для временных файлов
        self.temp_dir = tempfile.mkdtemp(prefix='benchmark_profiler_')
        self.logger.debug(f"Создана временная директория: {self.temp_dir}")
    
    def _check_isolation_support(self) -> None:
        """Проверяет и настраивает поддержку изоляции процессов."""
        if sys.platform == 'win32' and self.config.isolate_process:
            self.logger.warning(
                "Изоляция процессов на Windows может работать нестабильно. "
                "Автоматически отключаем для стабильности."
            )
            # Автоматически отключаем на Windows
            self.config.isolate_process = False
    
    def profile_operation(self,
                         operation_func: Callable,
                         operation_args: tuple = (),
                         operation_name: str = "",
                         library: str = "",
                         backend: Optional[str] = None,
                         dataset_name: str = "",
                         dataset_size: int = 0) -> ProfileResult:
        """
        Профилирует выполнение операции.
        
        Args:
            operation_func: Функция операции для профилирования
            operation_args: Аргументы для операции
            operation_name: Название операции
            library: Используемая библиотека
            backend: Backend (для pandas)
            dataset_name: Имя датасета
            dataset_size: Размер датасета
            
        Returns:
            ProfileResult: Результат профилирования
        """
        start_time = time.time()
        
        self.logger.info(
            f"Начало профилирования: {operation_name} | "
            f"{library}{f' ({backend})' if backend else ''} | "
            f"{dataset_name}"
        )
        
        try:
            if self.config.isolate_process:
                result = self._profile_isolated(
                    operation_func, operation_args,
                    operation_name, library, backend,
                    dataset_name, dataset_size
                )
            else:
                result = self._profile_inline(
                    operation_func, operation_args,
                    operation_name, library, backend,
                    dataset_name, dataset_size
                )
            
            # Добавляем общее время выполнения
            result.duration_seconds = time.time() - start_time
            
            # Логируем результат
            if result.success:
                self.logger.info(
                    f"Профилирование завершено: {operation_name} | "
                    f"Время: {result.mean_time:.3f}с (±{result.std_time:.3f}с) | "
                    f"Память: {result.peak_memory_mb:.1f}MB | "
                    f"Запусков: {result.runs_count}"
                )
            else:
                self.logger.error(
                    f"Ошибка профилирования: {operation_name} | "
                    f"Ошибка: {result.error_message}"
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Критическая ошибка при профилировании: {e}")
            return ProfileResult.from_error(operation_name, library, e, dataset_name)
    
    def _profile_isolated(self,
                         operation_func: Callable,
                         operation_args: tuple,
                         operation_name: str,
                         library: str,
                         backend: Optional[str],
                         dataset_name: str,
                         dataset_size: int) -> ProfileResult:
        """Профилирование в изолированном процессе."""
        # Создаем временный файл для результатов
        result_file = Path(self.temp_dir) / f"result_{os.getpid()}_{time.time()}.pkl"
        
        # События для синхронизации
        ready_event = mp.Event()
        
        # Запускаем процесс выполнения
        process = mp.Process(
            target=_run_isolated_operation,
            args=(operation_func, operation_args, self.config, 
                  str(result_file), ready_event)
        )
        process.start()
        
        # Ждем готовности процесса
        if not ready_event.wait(timeout=10):
            process.terminate()
            process.join()
            raise RuntimeError("Процесс не стал готовым за 10 секунд")
        
        # Запускаем отслеживание памяти
        memory_tracker = IsolatedMemoryTracker(self.config.memory_sampling_interval)
        memory_tracker.start_tracking(process.pid)
        
        # Ждем завершения процесса
        process.join(timeout=self.config.timeout_seconds)
        
        if process.is_alive():
            self.logger.warning(f"Таймаут операции {operation_name}, завершаем процесс")
            process.terminate()
            process.join()
            memory_tracker.stop_tracking()
            
            return ProfileResult(
                operation_name=operation_name,
                library=library,
                backend=backend,
                dataset_name=dataset_name,
                dataset_size=dataset_size,
                success=False,
                error_message=f"Timeout after {self.config.timeout_seconds}s"
            )
        
        # Останавливаем отслеживание памяти
        memory_stats = memory_tracker.stop_tracking()
        
        # Читаем результаты
        try:
            with open(result_file, 'rb') as f:
                process_result: ProcessResult = pickle.load(f)
            
            # Удаляем временный файл
            result_file.unlink()
            
            # Обрабатываем результаты
            if process_result.error:
                return ProfileResult.from_error(
                    operation_name, library, process_result.error, dataset_name
                )
            
            # Создаем итоговый результат
            timing = process_result.timing_result
            
            result = ProfileResult(
                operation_name=operation_name,
                library=library,
                backend=backend,
                dataset_name=dataset_name,
                dataset_size=dataset_size,
                # Время
                execution_times=timing.execution_times[:10],  # Сохраняем первые 10
                mean_time=timing.mean_time,
                median_time=timing.median_time,
                std_time=timing.std_dev,
                cv_time=timing.cv,
                min_time=timing.min_time,
                max_time=timing.max_time,
                # Память
                peak_memory_mb=memory_stats.get('peak_memory_mb', 0) if memory_stats else 0,
                avg_memory_mb=memory_stats.get('average_memory_mb', 0) if memory_stats else 0,
                min_memory_mb=memory_stats.get('min_memory_mb', 0) if memory_stats else 0,
                max_memory_mb=memory_stats.get('max_memory_mb', 0) if memory_stats else 0,
                memory_samples=memory_stats.get('sample_count', 0) if memory_stats else 0,
                # Статус
                success=True,
                runs_count=timing.runs_count,
                warmup_runs=timing.warmup_runs,
                converged=timing.converged,
                result_info=process_result.result_info
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Ошибка чтения результатов: {e}")
            return ProfileResult.from_error(operation_name, library, e, dataset_name)
    
    def _profile_inline(self,
                       operation_func: Callable,
                       operation_args: tuple,
                       operation_name: str,
                       library: str,
                       backend: Optional[str],
                       dataset_name: str,
                       dataset_size: int) -> ProfileResult:
        """Профилирование в текущем процессе (без изоляции)."""
        try:
            # Создаем трекер памяти
            memory_tracker = MemoryTracker(self.config.memory_sampling_interval)
            
            # Создаем таймер
            timer = Timer(
                min_runs=self.config.min_runs,
                max_runs=self.config.max_runs,
                target_cv=self.config.target_cv,
                warmup_runs=self.config.warmup_runs,
                gc_collect=self.config.gc_collect
            )
            
            # Запускаем отслеживание памяти
            memory_tracker.start_tracking()
            
            # Измеряем время
            def wrapper():
                return operation_func(*operation_args)
            
            timing_result = timer.time_execution(wrapper)
            
            # Останавливаем отслеживание памяти
            memory_stats = memory_tracker.stop_tracking()
            
            # Получаем информацию о результате
            result_info = {}
            try:
                op_result = operation_func(*operation_args)
                if isinstance(op_result, (pd.DataFrame, pl.DataFrame)):
                    result_info = {
                        'rows': len(op_result),
                        'columns': len(op_result.columns)
                    }
            except:
                pass
            
            # Создаем результат
            return ProfileResult(
                operation_name=operation_name,
                library=library,
                backend=backend,
                dataset_name=dataset_name,
                dataset_size=dataset_size,
                # Время
                execution_times=timing_result.execution_times[:10],
                mean_time=timing_result.mean_time,
                median_time=timing_result.median_time,
                std_time=timing_result.std_dev,
                cv_time=timing_result.cv,
                min_time=timing_result.min_time,
                max_time=timing_result.max_time,
                # Память
                peak_memory_mb=memory_stats.peak_memory_mb,
                avg_memory_mb=memory_stats.average_memory_mb,
                min_memory_mb=memory_stats.min_memory_mb,
                max_memory_mb=memory_stats.max_memory_mb,
                memory_samples=memory_stats.sample_count,
                # Статус
                success=True,
                runs_count=timing_result.runs_count,
                warmup_runs=timing_result.warmup_runs,
                converged=timing_result.converged,
                result_info=result_info
            )
            
        except Exception as e:
            return ProfileResult.from_error(operation_name, library, e, dataset_name)
    
    def cleanup(self) -> None:
        """Очистка временных файлов."""
        try:
            import shutil
            shutil.rmtree(self.temp_dir)
            self.logger.debug(f"Удалена временная директория: {self.temp_dir}")
        except Exception as e:
            self.logger.warning(f"Ошибка при очистке временной директории: {e}")
    
    def __enter__(self):
        """Вход в контекстный менеджер."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Выход из контекстного менеджера с очисткой."""
        self.cleanup()


# Вспомогательная функция для быстрого профилирования
def quick_profile(func: Callable,
                 *args,
                 operation_name: str = "unnamed",
                 library: str = "unknown",
                 **kwargs) -> ProfileResult:
    """
    Быстрое профилирование функции.
    
    Args:
        func: Функция для профилирования
        *args: Позиционные аргументы для функции
        operation_name: Название операции
        library: Библиотека
        **kwargs: Именованные аргументы для функции
        
    Returns:
        ProfileResult: Результат профилирования
    """
    with Profiler() as profiler:
        # Создаем обертку для передачи аргументов
        def wrapper():
            return func(*args, **kwargs)
        
        return profiler.profile_operation(
            wrapper,
            operation_name=operation_name,
            library=library
        )


# Функции для автоматического создания профайлера
def get_profiler(config: Optional[ProfilingConfig] = None) -> Profiler:
    """
    Создает профайлер с оптимальными настройками для текущей ОС.
    
    Args:
        config: Конфигурация профилирования
        
    Returns:
        Profiler: Настроенный профайлер
    """
    if config is None:
        config = ProfilingConfig()
    
    # На Windows автоматически отключаем изоляцию процессов
    if sys.platform == 'win32' and config.isolate_process:
        get_logger('profiler').info(
            "Windows обнаружена: автоматически отключаем изоляцию процессов"
        )
        config.isolate_process = False
    
    return Profiler(config)


def auto_profile(func: Callable,
                *args,
                operation_name: str = "unnamed",
                library: str = "unknown",
                config: Optional[ProfilingConfig] = None,
                **kwargs) -> ProfileResult:
    """
    Автоматически профилирует функцию с оптимальными настройками.
    
    Args:
        func: Функция для профилирования
        *args: Позиционные аргументы для функции
        operation_name: Название операции
        library: Библиотека
        config: Конфигурация профилирования
        **kwargs: Именованные аргументы для функции
        
    Returns:
        ProfileResult: Результат профилирования
    """
    with get_profiler(config) as profiler:
        # Создаем обертку для передачи аргументов
        def wrapper():
            return func(*args, **kwargs)
        
        return profiler.profile_operation(
            wrapper,
            operation_name=operation_name,
            library=library
        )
