"""
Модуль для отслеживания потребления памяти процессом.
"""

import os
import sys
import time
import psutil
import threading
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import multiprocessing as mp
import queue

from researches.polaris.pandas_polars_benchmark.src.utils.logging import get_logger


@dataclass
class MemoryStats:
    """Статистика использования памяти."""
    peak_memory_mb: float
    average_memory_mb: float
    min_memory_mb: float
    max_memory_mb: float
    samples: List[float] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    
    @property
    def duration_seconds(self) -> float:
        """Длительность измерения в секундах."""
        if len(self.timestamps) >= 2:
            return self.timestamps[-1] - self.timestamps[0]
        return 0.0
    
    @property
    def sample_count(self) -> int:
        """Количество замеров."""
        return len(self.samples)
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует в словарь для сериализации."""
        return {
            'peak_memory_mb': self.peak_memory_mb,
            'average_memory_mb': self.average_memory_mb,
            'min_memory_mb': self.min_memory_mb,
            'max_memory_mb': self.max_memory_mb,
            'sample_count': self.sample_count,
            'duration_seconds': self.duration_seconds,
            'samples': self.samples[:10] if len(self.samples) > 10 else self.samples  # Первые 10 для экономии места
        }


class MemoryTracker:
    """Класс для отслеживания памяти в отдельном потоке."""
    
    def __init__(self, 
                 sampling_interval: float = 0.1,
                 process: Optional[psutil.Process] = None):
        """
        Инициализация трекера памяти.
        
        Args:
            sampling_interval: Интервал между замерами (в секундах)
            process: Процесс для отслеживания (если None, то текущий)
        """
        self.sampling_interval = sampling_interval
        self.process = process or psutil.Process(os.getpid())
        
        self._tracking = False
        self._thread: Optional[threading.Thread] = None
        self._samples: List[float] = []
        self._timestamps: List[float] = []
        self._start_time: Optional[float] = None
        
        self.logger = get_logger('memory_tracker')
    
    def start_tracking(self) -> None:
        """Начинает отслеживание памяти в отдельном потоке."""
        if self._tracking:
            self.logger.warning("Отслеживание уже запущено")
            return
        
        self._tracking = True
        self._samples.clear()
        self._timestamps.clear()
        self._start_time = time.time()
        
        self._thread = threading.Thread(target=self._track_memory, daemon=True)
        self._thread.start()
        
        self.logger.debug(f"Начато отслеживание памяти с интервалом {self.sampling_interval}с")
    
    def stop_tracking(self) -> MemoryStats:
        """
        Останавливает отслеживание и возвращает статистику.
        
        Returns:
            MemoryStats: Статистика использования памяти
        """
        if not self._tracking:
            self.logger.warning("Отслеживание не было запущено")
            return self._create_empty_stats()
        
        self._tracking = False
        
        # Ждем завершения потока
        if self._thread:
            self._thread.join(timeout=1.0)
        
        # Создаем статистику
        stats = self._calculate_stats()
        
        self.logger.debug(
            f"Отслеживание остановлено. "
            f"Замеров: {stats.sample_count}, "
            f"Пиковая память: {stats.peak_memory_mb:.1f} MB"
        )
        
        return stats
    
    def _track_memory(self) -> None:
        """Основной цикл отслеживания памяти (выполняется в отдельном потоке)."""
        while self._tracking:
            try:
                # Получаем информацию о памяти
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)  # Resident Set Size в MB
                
                # Сохраняем замер
                current_time = time.time()
                self._samples.append(memory_mb)
                self._timestamps.append(current_time - self._start_time)
                
                # Ждем до следующего замера
                time.sleep(self.sampling_interval)
                
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                self.logger.error(f"Ошибка при отслеживании памяти: {e}")
                break
            except Exception as e:
                self.logger.error(f"Неожиданная ошибка в потоке отслеживания: {e}")
                break
    
    def _calculate_stats(self) -> MemoryStats:
        """Рассчитывает статистику по собранным данным."""
        if not self._samples:
            return self._create_empty_stats()
        
        return MemoryStats(
            peak_memory_mb=max(self._samples),
            average_memory_mb=sum(self._samples) / len(self._samples),
            min_memory_mb=min(self._samples),
            max_memory_mb=max(self._samples),
            samples=self._samples.copy(),
            timestamps=self._timestamps.copy()
        )
    
    def _create_empty_stats(self) -> MemoryStats:
        """Создает пустую статистику."""
        return MemoryStats(
            peak_memory_mb=0.0,
            average_memory_mb=0.0,
            min_memory_mb=0.0,
            max_memory_mb=0.0
        )
    
    def get_current_memory(self) -> float:
        """
        Получает текущее использование памяти.
        
        Returns:
            float: Текущее использование памяти в MB
        """
        try:
            memory_info = self.process.memory_info()
            return memory_info.rss / (1024 * 1024)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0.0


class ProcessMemoryTracker:
    """Трекер памяти для отслеживания внешнего процесса."""
    
    def __init__(self, pid: int, sampling_interval: float = 0.1):
        """
        Инициализация трекера для внешнего процесса.
        
        Args:
            pid: ID процесса для отслеживания
            sampling_interval: Интервал между замерами
        """
        try:
            self.process = psutil.Process(pid)
            self.tracker = MemoryTracker(sampling_interval, self.process)
        except psutil.NoSuchProcess:
            raise ValueError(f"Процесс с PID {pid} не найден")
    
    def start(self) -> None:
        """Начинает отслеживание."""
        self.tracker.start_tracking()
    
    def stop(self) -> MemoryStats:
        """Останавливает отслеживание и возвращает статистику."""
        return self.tracker.stop_tracking()


def track_memory_in_subprocess(target_pid: int, 
                             sampling_interval: float,
                             result_queue: mp.Queue,
                             stop_event: mp.Event) -> None:
    """
    Функция для отслеживания памяти в отдельном процессе.
    
    Args:
        target_pid: PID процесса для отслеживания
        sampling_interval: Интервал замеров
        result_queue: Очередь для передачи результатов
        stop_event: Событие для остановки отслеживания
    """
    try:
        # Переинициализируем логгер для subprocess (Windows fix)
        import logging
        logging.basicConfig(level=logging.WARNING)
        
        process = psutil.Process(target_pid)
        samples = []
        timestamps = []
        start_time = time.time()
        
        while not stop_event.is_set():
            try:
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)
                
                samples.append(memory_mb)
                timestamps.append(time.time() - start_time)
                
                time.sleep(sampling_interval)
                
            except psutil.NoSuchProcess:
                break
            except Exception as e:
                print(f"Ошибка в subprocess tracker: {e}")
                break
        
        # Отправляем результаты
        if samples:
            stats = {
                'peak_memory_mb': max(samples),
                'average_memory_mb': sum(samples) / len(samples),
                'min_memory_mb': min(samples),
                'max_memory_mb': max(samples),
                'sample_count': len(samples)
            }
            result_queue.put(stats)
        else:
            result_queue.put(None)
            
    except Exception as e:
        result_queue.put(None)
        print(f"Критическая ошибка в subprocess tracker: {e}")


class IsolatedMemoryTracker:
    """Трекер памяти для изолированного процесса (используется с multiprocessing)."""
    
    def __init__(self, sampling_interval: float = 0.1):
        """
        Инициализация изолированного трекера.
        
        Args:
            sampling_interval: Интервал между замерами
        """
        self.sampling_interval = sampling_interval
        self.logger = get_logger('isolated_memory_tracker')
        
        self._tracker_process: Optional[mp.Process] = None
        self._result_queue: Optional[mp.Queue] = None
        self._stop_event: Optional[mp.Event] = None
        
        # Проверка поддержки multiprocessing
        if sys.platform == 'win32':
            self.logger.warning(
                "IsolatedMemoryTracker может работать нестабильно на Windows. "
                "Рекомендуется использовать обычный MemoryTracker."
            )
    
    def start_tracking(self, target_pid: int) -> None:
        """
        Начинает отслеживание процесса в изолированном процессе.
        
        Args:
            target_pid: PID процесса для отслеживания
        """
        self._result_queue = mp.Queue()
        self._stop_event = mp.Event()
        
        self._tracker_process = mp.Process(
            target=track_memory_in_subprocess,
            args=(target_pid, self.sampling_interval, self._result_queue, self._stop_event)
        )
        self._tracker_process.start()
        
        self.logger.debug(f"Запущен изолированный трекер для PID {target_pid}")
    
    def stop_tracking(self, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        """
        Останавливает отслеживание и получает результаты.
        
        Args:
            timeout: Таймаут ожидания результатов
            
        Returns:
            Dict со статистикой или None
        """
        if not self._tracker_process or not self._stop_event:
            return None
        
        # Сигнализируем об остановке
        self._stop_event.set()
        
        # Ждем завершения процесса
        self._tracker_process.join(timeout=timeout)
        
        if self._tracker_process.is_alive():
            self.logger.warning("Принудительное завершение tracker процесса")
            self._tracker_process.terminate()
            self._tracker_process.join()
        
        # Получаем результаты
        try:
            result = self._result_queue.get(timeout=1.0)
            self.logger.debug(f"Получены результаты отслеживания")
            return result
        except queue.Empty:
            self.logger.warning("Не удалось получить результаты отслеживания")
            return None
        finally:
            # Очистка ресурсов
            self._tracker_process = None
            self._result_queue = None
            self._stop_event = None


# Вспомогательная функция для удобного использования
def measure_memory(func):
    """
    Декоратор для измерения памяти при выполнении функции.
    
    Пример:
        @measure_memory
        def my_function():
            # some memory intensive operation
            pass
    """
    def wrapper(*args, **kwargs):
        tracker = MemoryTracker()
        tracker.start_tracking()
        
        try:
            result = func(*args, **kwargs)
        finally:
            stats = tracker.stop_tracking()
        
        # Добавляем статистику к результату если это словарь
        if isinstance(result, dict):
            result['memory_stats'] = stats.to_dict()
        
        return result
    
    return wrapper
