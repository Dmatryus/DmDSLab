"""
Модуль для точного измерения времени выполнения операций с автоматическим повтором.
"""

import time
import statistics
import gc
from typing import Callable, List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field
import numpy as np

from researches.polaris.pandas_polars_benchmark.src.utils.logging import get_logger


@dataclass
class TimingResult:
    """Результат измерения времени выполнения."""
    execution_times: List[float]  # Все измерения
    mean_time: float              # Среднее время
    median_time: float            # Медианное время
    std_dev: float               # Стандартное отклонение
    cv: float                    # Коэффициент вариации
    min_time: float              # Минимальное время
    max_time: float              # Максимальное время
    runs_count: int              # Количество запусков
    warmup_runs: int             # Количество прогревочных запусков
    converged: bool              # Достигнут ли целевой CV
    
    @property
    def percentile_95(self) -> float:
        """95-й перцентиль времени выполнения."""
        return float(np.percentile(self.execution_times, 95))
    
    @property
    def iqr(self) -> float:
        """Межквартильный размах."""
        q75 = float(np.percentile(self.execution_times, 75))
        q25 = float(np.percentile(self.execution_times, 25))
        return q75 - q25
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует в словарь для сериализации."""
        return {
            'mean_time': self.mean_time,
            'median_time': self.median_time,
            'std_dev': self.std_dev,
            'cv': self.cv,
            'min_time': self.min_time,
            'max_time': self.max_time,
            'percentile_95': self.percentile_95,
            'iqr': self.iqr,
            'runs_count': self.runs_count,
            'warmup_runs': self.warmup_runs,
            'converged': self.converged,
            'execution_times': self.execution_times[:10]  # Первые 10 для экономии места
        }


class Timer:
    """Класс для измерения времени выполнения с автоматическим повтором."""
    
    def __init__(self,
                 min_runs: int = 3,
                 max_runs: int = 100,
                 target_cv: float = 0.05,
                 warmup_runs: int = 1,
                 gc_collect: bool = True):
        """
        Инициализация таймера.
        
        Args:
            min_runs: Минимальное количество запусков
            max_runs: Максимальное количество запусков
            target_cv: Целевой коэффициент вариации (например, 0.05 = 5%)
            warmup_runs: Количество прогревочных запусков
            gc_collect: Выполнять сборку мусора перед каждым замером
        """
        self.min_runs = max(1, min_runs)
        self.max_runs = max(self.min_runs, max_runs)
        self.target_cv = target_cv
        self.warmup_runs = max(0, warmup_runs)
        self.gc_collect = gc_collect
        
        self.logger = get_logger('timer')
    
    def time_execution(self, 
                      func: Callable[[], Any],
                      setup: Optional[Callable[[], None]] = None,
                      teardown: Optional[Callable[[], None]] = None) -> TimingResult:
        """
        Измеряет время выполнения функции с автоматическим повтором.
        
        Args:
            func: Функция для измерения
            setup: Функция подготовки (выполняется перед каждым запуском)
            teardown: Функция очистки (выполняется после каждого запуска)
            
        Returns:
            TimingResult: Результаты измерения
        """
        self.logger.debug(
            f"Начало измерения времени: "
            f"min_runs={self.min_runs}, max_runs={self.max_runs}, "
            f"target_cv={self.target_cv}, warmup={self.warmup_runs}"
        )
        
        # Прогревочные запуски
        self._perform_warmup(func, setup, teardown)
        
        # Основные измерения
        execution_times = []
        converged = False
        
        for run in range(self.max_runs):
            # Измеряем время одного запуска
            exec_time = self._time_single_run(func, setup, teardown)
            execution_times.append(exec_time)
            
            # Проверяем сходимость после минимального количества запусков
            if run >= self.min_runs - 1:
                cv = self._calculate_cv(execution_times)
                
                if cv <= self.target_cv:
                    converged = True
                    self.logger.debug(
                        f"Достигнут целевой CV={cv:.4f} после {len(execution_times)} запусков"
                    )
                    break
                
                # Логируем прогресс каждые 10 запусков
                if (run + 1) % 10 == 0:
                    self.logger.debug(
                        f"Запуск {run + 1}/{self.max_runs}: CV={cv:.4f} "
                        f"(целевой: {self.target_cv})"
                    )
        
        # Рассчитываем финальную статистику
        result = self._calculate_statistics(execution_times, converged)
        
        self.logger.info(
            f"Измерение завершено: {result.runs_count} запусков, "
            f"среднее время: {result.mean_time:.3f}с, CV: {result.cv:.4f}"
        )
        
        return result
    
    def _perform_warmup(self,
                       func: Callable[[], Any],
                       setup: Optional[Callable[[], None]],
                       teardown: Optional[Callable[[], None]]) -> None:
        """Выполняет прогревочные запуски."""
        if self.warmup_runs == 0:
            return
        
        self.logger.debug(f"Выполнение {self.warmup_runs} прогревочных запусков")
        
        for i in range(self.warmup_runs):
            try:
                if setup:
                    setup()
                
                func()
                
                if teardown:
                    teardown()
                    
            except Exception as e:
                self.logger.warning(f"Ошибка в прогревочном запуске {i + 1}: {e}")
    
    def _time_single_run(self,
                        func: Callable[[], Any],
                        setup: Optional[Callable[[], None]],
                        teardown: Optional[Callable[[], None]]) -> float:
        """Измеряет время одного запуска функции."""
        # Сборка мусора перед измерением
        if self.gc_collect:
            gc.collect()
            gc.disable()  # Отключаем GC во время измерения
        
        try:
            # Подготовка
            if setup:
                setup()
            
            # Измерение времени с максимальной точностью
            start_time = time.perf_counter()
            func()
            end_time = time.perf_counter()
            
            # Очистка
            if teardown:
                teardown()
            
            return end_time - start_time
            
        finally:
            if self.gc_collect:
                gc.enable()  # Включаем GC обратно
    
    def _calculate_cv(self, times: List[float]) -> float:
        """
        Рассчитывает коэффициент вариации.
        
        Args:
            times: Список времен выполнения
            
        Returns:
            float: Коэффициент вариации (CV)
        """
        if len(times) < 2:
            return float('inf')
        
        mean = statistics.mean(times)
        if mean == 0:
            return float('inf')
        
        std_dev = statistics.stdev(times)
        return std_dev / mean
    
    def _calculate_statistics(self,
                            execution_times: List[float],
                            converged: bool) -> TimingResult:
        """Рассчитывает итоговую статистику."""
        return TimingResult(
            execution_times=execution_times.copy(),
            mean_time=statistics.mean(execution_times),
            median_time=statistics.median(execution_times),
            std_dev=statistics.stdev(execution_times) if len(execution_times) > 1 else 0.0,
            cv=self._calculate_cv(execution_times),
            min_time=min(execution_times),
            max_time=max(execution_times),
            runs_count=len(execution_times),
            warmup_runs=self.warmup_runs,
            converged=converged
        )


class RepeatTimer:
    """Таймер с повторением для более точных измерений коротких операций."""
    
    def __init__(self,
                 repeat: int = 3,
                 number: int = 1000,
                 warmup: bool = True):
        """
        Инициализация таймера с повторением.
        
        Args:
            repeat: Количество повторений измерения
            number: Количество вызовов функции в одном измерении
            warmup: Выполнять прогревочный запуск
        """
        self.repeat = repeat
        self.number = number
        self.warmup = warmup
        self.logger = get_logger('repeat_timer')
    
    def timeit(self, func: Callable[[], Any]) -> Dict[str, float]:
        """
        Измеряет время выполнения функции методом timeit.
        
        Args:
            func: Функция для измерения
            
        Returns:
            Dict с результатами измерения
        """
        import timeit
        
        # Прогревочный запуск
        if self.warmup:
            self.logger.debug("Выполнение прогревочного запуска")
            func()
        
        # Измерение
        self.logger.debug(
            f"Измерение времени: repeat={self.repeat}, number={self.number}"
        )
        
        times = timeit.repeat(func, repeat=self.repeat, number=self.number)
        
        # Время на одно выполнение
        per_call_times = [t / self.number for t in times]
        
        return {
            'min_time': min(per_call_times),
            'max_time': max(per_call_times),
            'mean_time': statistics.mean(per_call_times),
            'median_time': statistics.median(per_call_times),
            'total_calls': self.repeat * self.number,
            'raw_times': per_call_times
        }


class AdaptiveTimer:
    """Адаптивный таймер, который автоматически подбирает количество повторений."""
    
    def __init__(self,
                 min_time: float = 0.2,
                 target_cv: float = 0.05):
        """
        Инициализация адаптивного таймера.
        
        Args:
            min_time: Минимальное время измерения в секундах
            target_cv: Целевой коэффициент вариации
        """
        self.min_time = min_time
        self.target_cv = target_cv
        self.logger = get_logger('adaptive_timer')
    
    def measure(self, func: Callable[[], Any]) -> TimingResult:
        """
        Адаптивно измеряет время выполнения функции.
        
        Args:
            func: Функция для измерения
            
        Returns:
            TimingResult: Результаты измерения
        """
        # Первый пробный запуск для оценки времени
        test_time = self._quick_measure(func)
        
        if test_time < 0.001:  # Очень быстрая операция
            # Используем метод timeit с множественными вызовами
            number = int(self.min_time / test_time)
            self.logger.debug(
                f"Быстрая операция ({test_time:.6f}с), "
                f"используем {number} повторений"
            )
            
            timer = RepeatTimer(repeat=5, number=number)
            result = timer.timeit(func)
            
            # Конвертируем в TimingResult
            return TimingResult(
                execution_times=result['raw_times'],
                mean_time=result['mean_time'],
                median_time=result['median_time'],
                std_dev=statistics.stdev(result['raw_times']) if len(result['raw_times']) > 1 else 0,
                cv=statistics.stdev(result['raw_times']) / result['mean_time'] if result['mean_time'] > 0 else 0,
                min_time=result['min_time'],
                max_time=result['max_time'],
                runs_count=len(result['raw_times']),
                warmup_runs=1,
                converged=True
            )
        else:
            # Обычное измерение для медленных операций
            self.logger.debug(
                f"Обычная операция ({test_time:.3f}с), "
                f"используем стандартный таймер"
            )
            
            timer = Timer(
                min_runs=3,
                max_runs=50,
                target_cv=self.target_cv
            )
            return timer.time_execution(func)
    
    def _quick_measure(self, func: Callable[[], Any]) -> float:
        """Быстрое измерение для оценки времени выполнения."""
        gc.collect()
        start = time.perf_counter()
        func()
        return time.perf_counter() - start


# Декоратор для удобного измерения времени
def measure_time(min_runs: int = 3,
                max_runs: int = 100,
                target_cv: float = 0.05):
    """
    Декоратор для измерения времени выполнения функции.
    
    Пример:
        @measure_time(min_runs=5, target_cv=0.03)
        def my_function():
            # some operation
            pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            timer = Timer(min_runs=min_runs, max_runs=max_runs, target_cv=target_cv)
            
            # Создаем функцию-обертку для передачи аргументов
            def target():
                return func(*args, **kwargs)
            
            timing_result = timer.time_execution(target)
            
            # Возвращаем результат функции и статистику времени
            result = func(*args, **kwargs)
            
            if isinstance(result, dict):
                result['timing_stats'] = timing_result.to_dict()
            
            return result
        
        return wrapper
    
    return decorator
